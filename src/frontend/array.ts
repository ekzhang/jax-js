import {
  accessorAluExp,
  accessorGlobal,
  AluExp,
  AluOp,
  AluVar,
  DType,
  Kernel,
  Reduction,
} from "../alu";
import { Backend, BackendType, Executable, getBackend, Slot } from "../backend";
import { ShapeTracker, unravelAlu } from "../shape";
import {
  deepEqual,
  isPermutation,
  prod,
  range,
  RecursiveArray,
  recursiveFlatten,
  rep,
} from "../utils";
import {
  CompareOp,
  getAval,
  newMain,
  Primitive,
  ShapedArray,
  Trace,
  Tracer,
  TracerValue,
  UseAfterFreeError,
} from "./core";
import { Jaxpr } from "./jaxpr";
import { jitCompile } from "./jit";

const JsArray = globalThis.Array;

/**
 * An executable operation that will be dispatched to the backend.
 *
 * This holds a reference to all input buffers used in the operation. After the
 * operation is dispatched, the references should be released.
 */
export class PendingExecute {
  prepared: Executable | null = null;
  submitted = false;
  #promise: Promise<void> | null = null; // for prepare
  #rc = 1; // since this could be held by multiple arrays, cancel when it hits 0

  constructor(
    readonly backend: Backend,
    readonly kernel: Kernel,
    readonly inputs: Slot[],
    readonly outputs: Slot[],
  ) {
    // Take a reference to all input buffers, while this execution is pending.
    // The reference is dropped after submit() or cancellation.
    for (const slot of inputs) this.backend.incRef(slot);
  }

  // Change the reference count of the PendingExecute object.
  // Used when copying the object to a new Array, or disposing an array.
  updateRc(delta: number) {
    if (this.#rc <= 0) throw new Error("internal: PendingExecute used rc<=0");
    this.#rc += delta;
    if (this.#rc <= 0 && !this.submitted) {
      // Cancel operation, release the references held to all input buffers.
      for (const slot of this.inputs) this.backend.decRef(slot);
    }
  }

  async prepare() {
    if (this.prepared) return;
    if (this.#promise) {
      await this.#promise;
      return;
    }
    this.#promise = (async () => {
      this.prepared = await this.backend.prepare(this.kernel);
    })();
    await this.#promise;
  }

  prepareSync() {
    if (this.prepared) return;
    this.prepared = this.backend.prepareSync(this.kernel);
  }

  submit() {
    if (this.submitted) return;
    if (this.#rc <= 0) throw new Error("internal: PendingExecute used rc<=0");
    if (!this.prepared) throw new Error("internal: Not prepared yet");
    this.submitted = true;
    this.backend.dispatch(this.prepared, this.inputs, this.outputs);
    for (const slot of this.inputs) this.backend.decRef(slot);
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
  static #nextId = 1001; // For unique hashing where needed.

  id: number;
  #dtype: DType;
  #source: AluExp | Slot;
  #st: ShapeTracker;
  #backend: Backend;
  #rc: number; // reference count for this specific Array object
  #pendingSet: Set<PendingExecute> | null; // only if source is `Slot`

  constructor(
    source: AluExp | Slot,
    st: ShapeTracker,
    dtype: DType,
    backend: Backend,
    pending: Iterable<PendingExecute> | null = null,
  ) {
    super(baseArrayTrace);
    this.id = Array.#nextId++;
    this.#dtype = dtype;
    this.#source = source;
    this.#st = st;
    this.#backend = backend;
    this.#rc = 1;
    this.#pendingSet = new Set(pending);

    if (!(source instanceof AluExp)) {
      backend.incRef(source); // decRef() is called when this.#rc hits 0.
    }
  }

  get aval() {
    return new ShapedArray(this.#st.shape, this.#dtype);
  }

  /** Return a simple string representation of the array's dimensions. */
  toString(): string {
    return `Array:${this.#dtype}[${this.shape.join(",")}]`;
  }

  get backend(): BackendType {
    return this.#backend.type;
  }

  #check() {
    if (this.#rc <= 0) throw new UseAfterFreeError(this);
  }

  get ref() {
    this.#check();
    this.#rc++;
    return this;
  }

  dispose() {
    this.#check();
    if (this.#rc-- === 0) {
      // Free any pending executables that haven't been submitted yet.
      for (const exe of this.#pending) exe.updateRc(-1);
      // If this has an array source, free it from the backend.
      if (typeof this.#source === "number") {
        this.#backend.decRef(this.#source);
      }
    }
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

  /**
   * Convert this array into a primitive value.
   *
   * This only works for scalars (0-dimensional arrays). It lets you get values
   * "out" of the JAX system. For instance, if `x = np.array(5)`, then you can
   * evaluate `x + 1` and `x ** 2` to get `6` and `25`, respectively.
   *
   * This method is also called for `==` equality.
   */
  [Symbol.toPrimitive](): any {
    if (this.ndim === 0) {
      return this.dataSync()[0];
    } else {
      throw new Error(
        `Cannot convert non-scalar array to primitive: ${this.toString()}`,
      );
    }
  }

  #reshape(st: ShapeTracker): Array {
    this.#check();
    const pending = this.#pending;
    for (const exe of pending) exe.updateRc(+1);
    const ar = new Array(this.#source, st, this.#dtype, this.#backend, pending);
    this.dispose(); // After constructing Array, so we don't free this.#source early.
    return ar;
  }

  /** Move axes to the rightmost dimension of the shape. */
  #moveAxesDown(axis: number[]): Array {
    this.#check();
    if (axis.length === 0) return this.reshape(this.shape.concat(1));
    const newShape: number[] = [];
    const keptAxes: number[] = [];
    const shiftedAxes: number[] = [];
    for (let i = 0; i < this.#st.shape.length; i++) {
      if (axis.includes(i)) {
        shiftedAxes.push(i);
      } else {
        keptAxes.push(i);
        newShape.push(this.#st.shape[i]);
      }
    }
    newShape.push(-1);
    return this.#transpose(keptAxes.concat(shiftedAxes)).reshape(newShape);
  }

  #transpose(perm: number[]): Array {
    this.#check();
    if (!isPermutation(perm, this.ndim))
      throw new Error(`Invalid perm for transpose: ${JSON.stringify(perm)}`);
    return this.#reshape(this.#st.permute(perm));
  }

  #unary(op: AluOp) {
    this.#check();
    // Short circuit if the array is already AluExp.
    if (this.#source instanceof AluExp) {
      const exp = new AluExp(op, this.#dtype, [this.#source]);
      return new Array(exp, this.#st, this.#dtype, this.#backend);
    }

    const indices = unravelAlu(this.#st.shape, AluVar.gidx);
    const exp = new AluExp(op, this.#dtype, [
      AluExp.globalView(this.#dtype, 0, this.#st, indices),
    ]);
    const kernel = new Kernel(1, this.#st.size, exp);
    const output = this.#backend.malloc(kernel.size * 4);
    const pending = [...this.#pending];
    for (const exe of pending) exe.updateRc(+1);
    pending.push(
      new PendingExecute(this.#backend, kernel, [this.#source], [output]),
    );

    this.dispose(); // Dispose of inputs after creating PendingExecute.
    return new Array(
      output,
      ShapeTracker.fromShape(this.shape),
      this.#dtype,
      this.#backend,
      pending,
    );
  }

  #binary(op: AluOp, other: Array): Array {
    const custom = (src: AluExp[]) => new AluExp(op, this.#dtype, src);
    return Array.#naryCustom(op, custom, [this, other]);
  }

  static #naryCustom(
    name: string,
    custom: (src: AluExp[]) => AluExp,
    arrays: Array[],
    dtypeOverride?: (DType | undefined)[],
    dtypeOutput?: DType,
  ): Array {
    const n = arrays.length;
    const backend = arrays[0].#backend;
    if (n === 0) throw new TypeError(`No inputs for ${name}`);

    for (const ar of arrays) ar.#check();

    let dtype: DType | undefined;
    for (let i = 0; i < n; i++) {
      if (dtypeOverride?.[i]) {
        if (arrays[i].#dtype !== dtypeOverride[i]) {
          throw new TypeError(
            `Wrong dtype in ${name}: expected ${dtypeOverride[i]}, got ${arrays[i].#dtype}`,
          );
        }
      } else {
        // Should match dtype of other arguments in the operation.
        if (!dtype) dtype = arrays[i].#dtype;
        else if (arrays[i].#dtype !== dtype) {
          throw new TypeError(
            `Dtype mismatch in ${name}: ${dtype} vs ${arrays[i].#dtype}`,
          );
        }
      }
      if (arrays[i].#backend !== backend) {
        throw new TypeError(
          `Backend mismatch in ${name}: ${backend.type} vs ${arrays[i].#backend.type}`,
        );
      }
    }
    dtypeOutput ??= dtype;
    if (!dtypeOutput) throw new TypeError("nary operation with no dtype");

    const newShape = arrays.map((a) => a.shape).reduce(generalBroadcast);
    arrays = arrays.map((ar) => {
      if (deepEqual(ar.shape, newShape)) return ar;
      return ar.#reshape(
        ar.#st.broadcast(newShape, range(newShape.length - ar.ndim)),
      );
    });

    // Short circuit if all are already AluExp.
    if (arrays.every((ar) => ar.#source instanceof AluExp)) {
      if (arrays.every((ar) => deepEqual(ar.#st, arrays[0].#st))) {
        // All are AluExp and have the same shape tracker.
        const exp = custom(arrays.map((ar) => ar.#source as AluExp));
        return new Array(exp, arrays[0].#st, exp.dtype, backend);
      }
      // If their shape trackers are different, we need to normalize them.
      const exp = custom(
        arrays.map((ar) => {
          const src = ar.#source as AluExp;
          if (ar.#st.contiguous) return src;
          return accessorAluExp(src, ar.#st, unravelAlu(newShape, AluVar.idx));
        }),
      );
      const st = ShapeTracker.fromShape(newShape);
      return new Array(exp, st, exp.dtype, backend);
    }

    const inputs: Slot[] = [];
    const src: AluExp[] = [];
    for (const ar of arrays) {
      const indices = unravelAlu(newShape, AluVar.gidx);
      if (ar.#source instanceof AluExp) {
        src.push(accessorAluExp(ar.#source, ar.#st, indices));
      } else {
        let gid = inputs.indexOf(ar.#source);
        if (gid === -1) {
          gid = inputs.length;
          inputs.push(ar.#source);
        }
        src.push(AluExp.globalView(ar.#dtype, gid, ar.#st, indices));
      }
    }

    const exp = custom(src);
    const kernel = new Kernel(inputs.length, arrays[0].#st.size, exp);
    const output = backend.malloc(kernel.size * 4);
    const pending = [...arrays.flatMap((ar) => ar.#pending)];
    for (const exe of pending) exe.updateRc(+1);
    pending.push(new PendingExecute(backend, kernel, inputs, [output]));

    for (const ar of arrays) ar.dispose(); // Dispose of inputs after creating PendingExecute.
    return new Array(
      output,
      ShapeTracker.fromShape(newShape),
      dtypeOutput,
      backend,
      pending,
    );
  }

  /** Reduce the last dimension of the array by an operation. */
  #reduce(op: AluOp): Array {
    this.#check();
    if (this.ndim === 0) throw new Error("Cannot reduce a scalar");
    const shape = this.shape;
    const reduction = new Reduction(this.#dtype, op, shape[shape.length - 1]);
    const newShape = shape.slice(0, -1); // first n-1 axes are in the shape
    const newSize = prod(newShape);

    const indices = [...unravelAlu(newShape, AluVar.gidx), AluVar.ridx];
    const [index, valid] = this.#st.toAluExp(indices);

    let exp: AluExp;
    const inputs: Slot[] = [];
    if (this.#source instanceof AluExp) {
      // If already AluExp, inline it into the reduction.
      exp = AluExp.where(
        valid,
        this.#source.substitute({ idx: index }),
        AluExp.f32(0),
      );
    } else {
      // Otherwise, use the global index.
      inputs.push(this.#source);
      exp = AluExp.where(
        valid,
        AluExp.globalIndex(DType.Float32, 0, index),
        AluExp.f32(0),
      );
    }

    const kernel = new Kernel(inputs.length, newSize, exp, reduction);
    const output = this.#backend.malloc(kernel.size * 4);
    const pending = [...this.#pending];
    for (const exe of pending) exe.updateRc(+1);
    pending.push(new PendingExecute(this.#backend, kernel, inputs, [output]));

    this.dispose(); // Dispose of inputs after creating PendingExecute.
    return new Array(
      output,
      ShapeTracker.fromShape(newShape),
      this.#dtype,
      this.#backend,
      pending,
    );
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
    this.#check();
    const indices = unravelAlu(this.#st.shape, AluVar.gidx);
    if (this.#source instanceof AluExp) {
      const exp = accessorAluExp(this.#source, this.#st, indices);
      const kernel = new Kernel(0, this.#st.size, exp);
      const output = this.#backend.malloc(kernel.size * 4);
      const pendingItem = new PendingExecute(
        this.#backend,
        kernel,
        [],
        [output],
      );
      this.#source = output;
      this.#st = ShapeTracker.fromShape(this.shape);
      this.#pendingSet = new Set([pendingItem]);
    } else {
      // Only realize if the ShapeTracker is non-contiguous.
      if (this.#st.contiguous) return;
      const exp = accessorGlobal(this.#dtype, 0, this.#st, indices);
      const kernel = new Kernel(1, this.#st.size, exp);
      const output = this.#backend.malloc(kernel.size * 4);
      const pendingItem = new PendingExecute(
        this.#backend,
        kernel,
        [this.#source],
        [output],
      );
      this.#source = output;
      this.#st = ShapeTracker.fromShape(this.shape);
      this.#pendingSet ??= new Set();
      this.#pendingSet.add(pendingItem);
    }
  }

  /** Realize the array and return it as data. */
  async data(): Promise<Float32Array | Int32Array> {
    this.#realize();
    const pending = this.#pending;
    if (pending) {
      // Compile all pending executables concurrently.
      await Promise.all(pending.map((exe) => exe.prepare()));
      for (const p of pending) p.submit();
    }
    const buf = await this.#backend.read(this.#source as Slot);
    this.dispose();
    return this.dtype === DType.Float32
      ? new Float32Array(buf)
      : new Int32Array(buf);
  }

  /** Wait for this array to be placed on the backend, if needed. */
  async wait(): Promise<void> {
    this.#check();
    if (this.#source instanceof AluExp) return;
    const pending = this.#pending;
    if (pending) {
      // Compile all pending executables concurrently.
      await Promise.all(pending.map((exe) => exe.prepare()));
      for (const p of pending) p.submit();
    }
    await this.#backend.read(this.#source, 0, 0);
    this.dispose();
  }

  /**
   * Realize the array and return it as data. This is a sync variant and not
   * recommended for performance reasons, as it will block rendering.
   */
  dataSync(): Float32Array | Int32Array {
    this.#realize();
    for (const p of this.#pending) {
      p.prepareSync();
      p.submit();
    }
    const buf = this.#backend.readSync(this.#source as Slot);
    this.dispose();
    return this.dtype === DType.Float32
      ? new Float32Array(buf)
      : new Int32Array(buf);
  }

  /** Convert this array into a JavaScript object (blocking). */
  js() {
    return dataToJs(this.dtype, this.dataSync(), this.shape);
  }

  /** Convert this array into a JavaScript object, asynchronously. */
  async jsAsync() {
    return dataToJs(this.dtype, await this.data(), this.shape);
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
        if (axis.length === 0) return [x];
        return [x.#moveAxesDown(axis).#reduce(AluOp.Add)];
      },
      [Primitive.Compare]([x, y], { op }: { op: CompareOp }) {
        const custom = ([x, y]: AluExp[]) => aluCompare(x, y, op);
        return [Array.#naryCustom("compare", custom, [x, y], [], DType.Bool)];
      },
      [Primitive.Where]([cond, x, y]) {
        const custom = ([cond, x, y]: AluExp[]) => AluExp.where(cond, x, y);
        return [Array.#naryCustom("where", custom, [cond, x, y], [DType.Bool])];
      },
      [Primitive.Transpose]([x], { perm }: { perm: number[] }) {
        return [x.#transpose(perm)];
      },
      [Primitive.Broadcast](
        [x],
        { shape, axis }: { shape: number[]; axis: number[] },
      ) {
        return [x.#reshape(x.#st.broadcast(shape, axis))];
      },
      [Primitive.Reshape]([x], { shape }: { shape: number[] }) {
        return [x.#reshape(x.#st.reshape(shape))];
      },
      [Primitive.Flip]([x], { axis }: { axis: number[] }) {
        const arg = rep(x.ndim, false);
        for (const ax of axis) arg[ax] = true;
        return [x.#reshape(x.#st.flip(arg))];
      },
      [Primitive.JitCall](
        args,
        { jaxpr, numConsts }: { jaxpr: Jaxpr; numConsts: number },
      ) {
        if (jaxpr.inBinders.length !== args.length) {
          throw new Error(
            `jit_call expects ${jaxpr.inBinders.length} args, got ${args.length}`,
          );
        }
        const backend = getBackend(); // TODO: Use correct backend.
        const consts = args.slice(0, numConsts);
        const tracers = args.slice(numConsts);
        const jp = jitCompile(backend, jaxpr, consts);
        const { outputs, pending } = jp.execute(
          tracers.map((x) => x._realizeSource()),
        );
        const prevPending = args.flatMap((x) => x.#pending);
        for (const exe of prevPending) exe.updateRc(+1);
        pending.splice(0, 0, ...prevPending); // Dispatch order of pending kernels is important.
        return outputs.map((source, i) => {
          return new Array(
            source,
            ShapeTracker.fromShape(jaxpr.outs[i].aval.shape),
            jaxpr.outs[i].aval.dtype,
            backend,
            pending,
          );
        });
      },
    };
  }

  // Internal methods, not public API. Do not use.
  _realizeSource() {
    this.#realize();
    return this.#source as number; // Because #realize() was called.
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
      throw new Error("array astype not implemented yet"); // TODO
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
    const size = prod(shape);
    const flat = recursiveFlatten(values);
    if (flat.length !== size) {
      throw new Error(
        `Jagged shape: ${JSON.stringify(shape)} vs ${flat.length}`,
      );
    }
    if (size === 0) return zeros(shape, { dtype, backend });
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

function dataToJs(
  dtype: DType,
  data: Float32Array | Int32Array,
  shape: number[],
): RecursiveArray<number> | RecursiveArray<boolean> {
  if (shape.length === 0) {
    return dtype === DType.Bool ? Boolean(data[0]) : data[0];
  }
  const [first, ...rest] = shape;
  const restSize = prod(rest);
  const ret: any[] = [];
  for (let i = 0; i < first; i++) {
    const subarray = data.slice(i * restSize, (i + 1) * restSize);
    ret.push(dataToJs(dtype, subarray, rest));
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

/** Return a new array of given shape and type, filled with zeros. */
export function zeros(
  shape: number[],
  { dtype, backend }: DTypeAndBackend = {},
): Array {
  return full(shape, 0, { dtype, backend });
}

/** Return a new array of given shape and type, filled with ones. */
export function ones(
  shape: number[],
  { dtype, backend }: DTypeAndBackend = {},
): Array {
  return full(shape, 1, { dtype, backend });
}

/** Return a new array of given shape and type, filled with `fill_value`. */
export function full(
  shape: number[],
  fillValue: number | boolean | Array,
  { dtype, backend }: DTypeAndBackend = {},
): Array {
  let source: AluExp;
  if (typeof fillValue === "number") {
    dtype = dtype ?? DType.Float32;
    source = AluExp.const(dtype, fillValue);
  } else if (typeof fillValue === "boolean") {
    dtype = dtype ?? DType.Bool;
    source = AluExp.const(dtype, fillValue ? 1 : 0);
  } else if (fillValue instanceof Array) {
    // TODO: Full can also take an array as a fill value. This is equivalent to
    // expanding the array.
    throw new Error("numpy.full() with array argument not implemented yet");
  } else {
    throw new TypeError(`Invalid type for full: ${fillValue}`);
  }
  return new Array(
    source,
    ShapeTracker.fromShape(shape),
    dtype ?? DType.Float32,
    getBackend(backend),
  );
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

  if (numCols < numRows) {
    // If less columns than rows, take the transpose since it's no longer a
    // simple modular arithmetic expression.
    const arr = eye(numCols, numRows, { dtype, backend });
    return arr.transpose();
  }
  if (numRows === 0) {
    return zeros([0, numCols], { dtype, backend });
  }

  const exp = AluExp.cmplt(
    AluExp.mod(AluVar.idx, AluExp.i32(numCols + 1)),
    AluExp.i32(1),
  );
  return new Array(
    AluExp.cast(dtype, exp),
    ShapeTracker.fromShape([numRows, numCols]),
    dtype,
    getBackend(backend),
  );
}

/** Return the identity array, with ones on the main diagonal. */
export function identity(
  n: number,
  { dtype, backend }: DTypeAndBackend = {},
): Array {
  return eye(n, n, { dtype, backend });
}

/** Translate a `CompareOp` into an `AluExp` on two sub-expressions. */
export function aluCompare(a: AluExp, b: AluExp, op: CompareOp): AluExp {
  switch (op) {
    case CompareOp.Greater:
      // `x > y` is equivalent to `x != y && !(x < y)`
      // TODO: handle NaN
      return AluExp.mul(AluExp.cmpne(a, b), AluExp.cmplt(a, b).not());
    case CompareOp.Less:
      return AluExp.cmplt(a, b);
    case CompareOp.Equal:
      return AluExp.cmpne(a, b).not();
    case CompareOp.NotEqual:
      return AluExp.cmpne(a, b);
    case CompareOp.GreaterEqual:
      // `x >= y` is equivalent to `!(x < y)`
      // TODO: handle NaN
      return AluExp.cmplt(a, b).not();
    case CompareOp.LessEqual:
      return AluExp.add(AluExp.cmplt(a, b), AluExp.cmpne(a, b).not());
  }
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
