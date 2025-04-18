import { AluExp, AluOp, DType, Kernel } from "../alu";
import {
  accessorAluExp,
  accessorGlobal,
  Backend,
  BackendType,
  Executable,
  getBackend,
  Slot,
  variables,
} from "../backend";
import { ShapeTracker } from "../shape";
import {
  deepEqual,
  isPermutation,
  RecursiveArray,
  recursiveFlatten,
  repeat,
} from "../utils";
import {
  getAval,
  newMain,
  Primitive,
  ShapedArray,
  Trace,
  Tracer,
  TracerValue,
} from "./core";

const JsArray = globalThis.Array;

function gidxVar(size: number) {
  // global index
  return AluExp.special(DType.Int32, "gidx", size);
}

function ridxVar(size: number) {
  // reduction index
  return AluExp.special(DType.Int32, "ridx", size);
}

class PendingExecute {
  prepared: Executable | null = null;
  submitted = false;
  #promise: Promise<void> | null = null; // for prepare

  constructor(
    readonly kernel: Kernel,
    readonly inputs: Slot[],
    readonly outputs: Slot[],
  ) {}

  async prepare(backend: Backend) {
    if (this.prepared) return;
    if (this.#promise) {
      await this.#promise;
      return;
    }
    this.#promise = (async () => {
      this.prepared = await backend.prepare(this.kernel);
    })();
    await this.#promise;
  }

  prepareSync(backend: Backend) {
    if (this.prepared) return;
    this.prepared = backend.prepareSync(this.kernel);
  }

  submit(backend: Backend) {
    if (this.submitted) return;
    if (!this.prepared) throw new Error("Not prepared yet");
    backend.dispatch(this.prepared, this.inputs, this.outputs);
  }
}

type DTypeAndBackend = { dtype?: DType; backend?: BackendType };

/**
 * A multidimensional numeric array with data stored on CPU or GPU.
 *
 * This is the library's core data type. Equivalent to `jnp.Array` from JAX, or
 * `torch.Tensor`.
 *
 * Not to be confused with the JavaScript "Array" constructor. Avoid importing
 * this into your code's namespace if you're already using the JavaScript
 * "Array" type by name.
 */
export class Array extends Tracer {
  #dtype: DType;
  #source: AluExp | Slot;
  #st: ShapeTracker;
  #backend: Backend;
  #pendingSet: Set<PendingExecute> | null; // only if source is `Slot`

  constructor(
    source: AluExp | Slot,
    st: ShapeTracker,
    dtype: DType,
    backend: Backend,
    pending: Iterable<PendingExecute> | null = null,
  ) {
    super(baseArrayTrace);
    this.#dtype = dtype;
    this.#source = source;
    this.#st = st;
    this.#backend = backend;
    this.#pendingSet = new Set(pending);
  }

  get aval() {
    return new ShapedArray(this.#st.shape, this.#dtype);
  }

  get backend(): BackendType {
    return this.#backend.type;
  }

  get ndim(): number {
    return this.shape.length;
  }

  /** Return a simple string representation of the array's dimensions. */
  toString(): string {
    return `Array:${this.#dtype}[${this.shape.join(",")}]`;
  }

  /** Get the pending executes as a list, trimming if already submitted. */
  get #pending(): PendingExecute[] {
    if (!this.#pendingSet) return [];
    for (const p of this.#pendingSet) {
      if (p.submitted) this.#pendingSet.delete(p);
    }
    if (this.#pendingSet.size === 0) {
      this.#pendingSet = null;
      return [];
    } else {
      return [...this.#pendingSet];
    }
  }

  static zeros(shape: number[], { dtype, backend }: DTypeAndBackend = {}) {
    dtype = dtype ?? DType.Float32;
    return new Array(
      AluExp.const(dtype, 0),
      ShapeTracker.fromShape(shape),
      dtype,
      getBackend(backend),
    );
  }

  static ones(shape: number[], { dtype, backend }: DTypeAndBackend = {}) {
    dtype = dtype ?? DType.Float32;
    return new Array(
      AluExp.const(dtype, 1),
      ShapeTracker.fromShape(shape),
      dtype,
      getBackend(backend),
    );
  }

  // Movement operations
  // TODO: move to Tracer / Primitive

  #reshape(st: ShapeTracker): Array {
    return new Array(
      this.#source,
      st,
      this.#dtype,
      this.#backend,
      this.#pending,
    );
  }

  reshape(shape: number[]): Array {
    const autoIdx = shape.indexOf(-1);
    if (autoIdx !== -1) {
      const remaining = this.#st.size / shape.reduce((a, b) => a * b, -1);
      if (remaining % 1 !== 0 || remaining < 0) {
        throw new Error(
          `Invalid reshape: ${JSON.stringify(this.shape)} -> ${JSON.stringify(shape)}`,
        );
      }
      shape = shape.toSpliced(autoIdx, 1, remaining);
    }
    if (deepEqual(shape, this.shape)) {
      return this;
    }
    return this.#reshape(this.#st.reshape(shape));
  }

  flatten(): Array {
    return this.reshape([-1]);
  }
  ravel(): Array {
    return this.reshape([-1]);
  }

  #transpose(perm?: number[]): Array {
    if (perm) {
      if (!isPermutation(perm, this.ndim))
        throw new Error(`Invalid perm for transpose: ${JSON.stringify(perm)}`);
    } else if (!perm) {
      perm = this.shape.map((_, i) => i).reverse();
    }
    return this.#reshape(this.#st.permute(perm));
  }

  #unary(op: AluOp) {
    // Short circuit if the array is already AluExp.
    if (this.#source instanceof AluExp) {
      const exp = new AluExp(op, this.#dtype, [this.#source]);
      return new Array(exp, this.#st, this.#dtype, this.#backend);
    }

    const gidx = gidxVar(this.#st.size);
    const exp = new AluExp(op, this.#dtype, [
      accessorGlobal(0, this.#st, gidx),
    ]);
    const kernel = new Kernel(1, this.#st.size, exp);
    const output = this.#backend.malloc(kernel.size * 4);
    const pending = [
      ...this.#pending,
      new PendingExecute(kernel, [this.#source], [output]),
    ];
    return new Array(
      output,
      ShapeTracker.fromShape(this.shape),
      this.#dtype,
      this.#backend,
      pending,
    );
  }

  static #binaryCustom(
    name: string,
    custom: (src: AluExp[]) => AluExp,
    [lhs, rhs]: [Array, Array],
  ): Array {
    if (lhs.#dtype !== rhs.#dtype) {
      throw new Error(
        `Dtype mismatch in ${name}: ${lhs.dtype} vs ${rhs.dtype}`,
      ); // todo: dtype casting
    }
    if (lhs.#backend !== rhs.#backend) {
      throw new Error(
        `Backend mismatch in ${name}: ${lhs.#backend} vs ${rhs.#backend}`,
      );
    }
    if (!deepEqual(lhs.shape, rhs.shape)) {
      const newShape = generalBroadcast(lhs.shape, rhs.shape);
      lhs = lhs.#reshape(
        lhs.#st
          .reshape(repeat(1, newShape.length - lhs.ndim).concat(lhs.shape))
          .expand(newShape),
      );
      rhs = rhs.#reshape(
        rhs.#st
          .reshape(repeat(1, newShape.length - rhs.ndim).concat(rhs.shape))
          .expand(newShape),
      );
    }

    // Short circuit if both are already AluExp.
    if (lhs.#source instanceof AluExp && rhs.#source instanceof AluExp) {
      const exp = custom([lhs.#source, rhs.#source]);
      return new Array(exp, lhs.#st, exp.dtype, lhs.#backend);
    }

    const gidx = gidxVar(lhs.#st.size);

    const inputs: Slot[] = [];
    const src: AluExp[] = [];

    for (const ar of [lhs, rhs]) {
      if (ar.#source instanceof AluExp) {
        src.push(accessorAluExp(ar.#source, ar.#st, gidx));
      } else {
        src.push(accessorGlobal(inputs.length, ar.#st, gidx));
        inputs.push(ar.#source);
      }
    }

    const exp = custom(src);
    const kernel = new Kernel(src.length, lhs.#st.size, exp);
    const output = lhs.#backend.malloc(kernel.size * 4);
    const pending = [
      ...lhs.#pending,
      ...rhs.#pending,
      new PendingExecute(kernel, inputs, [output]),
    ];
    return new Array(
      output,
      ShapeTracker.fromShape(lhs.shape),
      exp.dtype,
      lhs.#backend,
      pending,
    );
  }

  #binary(op: AluOp, other: Array): Array {
    const custom = (src: AluExp[]) => new AluExp(op, this.#dtype, src);
    return Array.#binaryCustom(op, custom, [this, other]);
  }

  /**
   * Normalizes this array into one backed by a `Slot`.
   *
   * This mutates the array in-place, turning it into an equivalent array whose
   * source is actual, contiguous data on device.
   *
   * Calling this twice is a no-op.
   */
  #realize(): void {
    const gidx = gidxVar(this.#st.size);
    if (this.#source instanceof AluExp) {
      const exp = accessorAluExp(this.#source, this.#st, gidx);
      const kernel = new Kernel(0, this.#st.size, exp);
      const output = this.#backend.malloc(kernel.size * 4);
      const pendingItem = new PendingExecute(kernel, [], [output]);
      this.#source = output;
      this.#st = ShapeTracker.fromShape(this.shape);
      this.#pendingSet = new Set([pendingItem]);
    } else {
      // Only realize if the ShapeTracker is non-contiguous.
      if (this.#st.contiguous) return;
      const exp = accessorGlobal(0, this.#st, gidx);
      const kernel = new Kernel(1, this.#st.size, exp);
      const output = this.#backend.malloc(kernel.size * 4);
      const pendingItem = new PendingExecute(kernel, [this.#source], [output]);
      this.#source = output;
      this.#st = ShapeTracker.fromShape(this.shape);
      this.#pendingSet ??= new Set();
      this.#pendingSet.add(pendingItem);
    }
  }

  /** Realize the array and return it as data. */
  async data(): Promise<Float32Array> {
    this.#realize();
    const pending = this.#pending;
    if (pending) {
      // Compile all pending executables concurrently.
      await Promise.all(pending.map((exe) => exe.prepare(this.#backend)));
      for (const p of pending) p.submit(this.#backend);
    }
    const buf = await this.#backend.read(this.#source as Slot);
    return new Float32Array(buf);
  }

  /**
   * Realize the array and return it as data. This is a sync variant and not
   * recommended for performance reasons, as it will block rendering.
   */
  dataSync(): Float32Array {
    this.#realize();
    for (const p of this.#pending) {
      p.prepareSync(this.#backend);
      p.submit(this.#backend);
    }
    const buf = this.#backend.readSync(this.#source as Slot);
    return new Float32Array(buf);
  }

  /** Convert this array into a JavaScript object (blocking). */
  js() {
    return dataToJs(this.dataSync(), this.shape);
  }

  /** Convert this array into a JavaScript object, asynchronously. */
  async jsAsync() {
    return dataToJs(await this.data(), this.shape);
  }

  /** @private Internal plumbing method for Array / Tracer ops. */
  static _implRules(): Record<Primitive, ImplRule> {
    return {
      [Primitive.Add]([x, y]) {
        return [x.#binary(AluOp.Add, y)];
      },
      [Primitive.Mul]([x, y]) {
        return [x.#binary(AluOp.Mul, y)];
      },
      [Primitive.Neg]([x]) {
        return [zerosLike(x).#binary(AluOp.Sub, x)];
      },
      [Primitive.Sin]([x]) {
        return [x.#unary(AluOp.Sin)];
      },
      [Primitive.Cos]([x]) {
        return [x.#unary(AluOp.Cos)];
      },
      [Primitive.ReduceSum]([x], { axis }: { axis: number[] }) {
        throw new Error("unimplemented");
        // return [new Array(tf.sum(x.data, axis))];
      },
      [Primitive.Greater]([x, y]) {
        // `x > y` is equivalent to `x != y && !(x < y)`
        // TODO: handle NaN
        const custom = ([x, y]: AluExp[]) =>
          AluExp.mul(AluExp.cmpne(x, y), AluExp.cmplt(x, y).not());
        return [Array.#binaryCustom("greater", custom, [x, y])];
      },
      [Primitive.Less]([x, y]) {
        const custom = ([x, y]: AluExp[]) => AluExp.cmplt(x, y);
        return [Array.#binaryCustom("less", custom, [x, y])];
      },
      [Primitive.Transpose]([x], { perm }: { perm?: number[] }) {
        return [x.#transpose(perm)];
      },
      [Primitive.Broadcast](
        [x],
        { shape, axes }: { shape: number[]; axes: number[] },
      ) {
        let st = x.#st;
        if (axes.length > 0) {
          // First, unsqueeze each dimension of "axes".
          const unsqueezed = [...x.#st.shape];
          for (const axis of axes.toSorted()) {
            unsqueezed.splice(axis, 0, 1);
          }
          st = st.reshape(unsqueezed);
        }
        // Then, expand the data to the new shape.
        return [x.#reshape(st.expand(shape))];
      },
    };
  }
}

/** Construct an array from a single scalar constant. */
export function scalar(
  value: number | boolean,
  { dtype, backend }: DTypeAndBackend = {},
) {
  if (typeof value === "number") {
    dtype ??= DType.Float32; // default dtype for JS numbers
    if (![DType.Float32, DType.Int32].includes(dtype))
      throw new TypeError(`Mismatched dtype for scalar ${value}`);
  } else if (typeof value === "boolean") {
    dtype ??= DType.Bool;
    if (![DType.Float32, DType.Int32, DType.Bool].includes(dtype))
      throw new TypeError(`Mismatched dtype for scalar ${value}`);
  } else {
    throw new TypeError(`Invalid type for scalar ${value}`);
  }
  return new Array(
    AluExp.const(dtype, value),
    ShapeTracker.fromShape([]),
    dtype,
    getBackend(backend),
  );
}

/** Constructor for creating a new array from data. */
export function array(
  values:
    | Array
    | Float32Array
    | Int32Array
    | RecursiveArray<number>
    | RecursiveArray<boolean>,
  { shape, dtype, backend }: { shape?: number[] } & DTypeAndBackend = {},
): Array {
  if (values instanceof Array) {
    if (shape) {
      values = values.reshape(shape);
    }
    if (dtype && values.dtype !== dtype) {
      throw new Error("array astype not implemented yet");
      // values = values.astype(dtype);
    }
    return values;
  } else if (values instanceof Float32Array || values instanceof Int32Array) {
    const ar = arrayFromData(values, { dtype, backend });
    return shape ? ar.reshape(shape) : ar;
  } else {
    // Assume this is a nested array object, infer the shape.
    if (!shape) {
      shape = [];
      let cur = values;
      while (JsArray.isArray(cur)) {
        shape.push(cur.length);
        cur = cur[0];
      }
    }
    const size = shape.reduce((a, b) => a * b, 1);
    const flat = recursiveFlatten(values);
    if (flat.length !== size) {
      throw new Error(
        `Jagged shape: ${JSON.stringify(shape)} vs ${flat.length}`,
      );
    }
    if (size === 0) return Array.zeros(shape, { dtype, backend });
    if (typeof flat[0] === "boolean") {
      dtype = dtype ?? DType.Bool;
      const data = new Int32Array(flat.map((x) => (x ? 1 : 0)));
      return arrayFromData(data, { dtype, backend }).reshape(shape);
    } else {
      dtype = dtype ?? DType.Float32;
      const data = new Float32Array(flat as number[]);
      return arrayFromData(data, { dtype, backend }).reshape(shape);
    }
  }
}

function arrayFromData(
  data: Float32Array | Int32Array,
  { dtype, backend: backendType }: DTypeAndBackend = {},
): Array {
  const backend = getBackend(backendType);
  if (data instanceof Float32Array) {
    if (dtype && dtype !== DType.Float32) {
      throw new Error("Float32Array must have float32 type");
    }
    const slot = backend.malloc(data.byteLength, data.buffer);
    return new Array(
      slot,
      ShapeTracker.fromShape([data.length]),
      DType.Float32,
      backend,
    );
  } else if (data instanceof Int32Array) {
    if (dtype && dtype !== DType.Int32 && dtype !== DType.Bool) {
      throw new Error("Int32Array must have int32 or bool type");
    }
    const slot = backend.malloc(data.byteLength, data.buffer);
    return new Array(
      slot,
      ShapeTracker.fromShape([data.length]),
      dtype ?? DType.Int32,
      backend,
    );
  } else {
    throw new Error("Unsupported data type");
  }
}

function dataToJs(data: Float32Array, shape: number[]): RecursiveArray<number> {
  if (shape.length === 0) {
    return data[0];
  }
  const [first, ...rest] = shape;
  const restSize = rest.reduce((a, b) => a * b, 1);
  const ret: any[] = [];
  for (let i = 0; i < first; i++) {
    ret.push(dataToJs(data.slice(i * restSize, (i + 1) * restSize), rest));
  }
  return ret;
}

/** If x is a value, lift it into an array, otherwise leave it be. */
export function pureArray(x: TracerValue): Tracer {
  if (x instanceof Tracer) {
    return x;
  } else {
    return scalar(x);
  }
}

class EvalTrace extends Trace {
  // No boxing in Tracers needed.
  pure = (x: TracerValue) => pureArray(x);
  lift = (x: Tracer) => x;

  processPrimitive(
    primitive: Primitive,
    tracers: Array[],
    params: Record<string, any>,
  ): Tracer[] {
    return implRules[primitive](tracers, params);
  }
}

// Special bottom of the stack: must be level 0.
const baseArrayTrace = new EvalTrace(newMain(EvalTrace, null));

type ImplRule = (tracers: Array[], params: any) => Array[];
const implRules: Record<Primitive, ImplRule> = Array._implRules();

export function zerosLike(val: TracerValue): Array {
  const aval = getAval(val);
  return zeros(aval.shape, { dtype: aval.dtype });
}

export function zeros(
  shape: number[],
  { dtype, backend }: DTypeAndBackend = {},
): Array {
  return Array.zeros(shape, { dtype, backend });
}

/**
 * Create an identity matrix.
 *
 * If numCols is not provided, it defaults to numRows, i.e., a square identity
 * matrix with ones on the diagonal.
 */
export function eye(
  numRows: number,
  numCols?: number,
  { dtype, backend }: DTypeAndBackend = {},
): Array {
  numCols = numCols ?? numRows;
  dtype = dtype ?? DType.Float32;

  if (numCols > numRows) {
    // If more columns than rows, take the transpose since it's no longer a
    // simple modular arithmetic expression.
    const arr = eye(numCols, numRows, { dtype, backend });
    return arr.transpose();
  }
  if (numRows === 0) {
    return Array.zeros([0, numCols], { dtype, backend });
  }

  const exp = AluExp.cmplt(
    AluExp.mod(variables.idx, AluExp.i32(numCols + 1)),
    AluExp.i32(1),
  );
  return new Array(
    AluExp.cast(dtype, exp),
    ShapeTracker.fromShape([numRows, numCols]),
    dtype,
    getBackend(backend),
  );
}

/**
 * Implements a NumPy-style generalized broadcast rule on two array shapes.
 *
 * "When operating on two arrays, NumPy compares their shapes element-wise. It
 * starts with the trailing (i.e. rightmost) dimension and works its way left.
 * Two dimensions are compatible when:
 *   1. they are equal, or
 *   2. one of them is 1."
 *
 * Throws a TypeError if the broadcast is not possible.
 *
 * <https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules>
 */
export function generalBroadcast(a: number[], b: number[]): number[] {
  const out: number[] = [];
  let i = a.length - 1;
  let j = b.length - 1;
  for (; i >= 0 && j >= 0; i--, j--) {
    const x = a[i];
    const y = b[j];
    if (x === y) {
      out.push(x);
    } else if (x === 1) {
      out.push(y);
    } else if (y === 1) {
      out.push(x);
    } else {
      throw new TypeError(`Incompatible array broadcast shapes: ${a} vs ${b}`);
    }
  }
  for (; i >= 0; i--) {
    out.push(a[i]);
  }
  for (; j >= 0; j--) {
    out.push(b[j]);
  }
  return out.reverse();
}
