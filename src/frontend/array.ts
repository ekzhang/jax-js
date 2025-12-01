import {
  accessorAluExp,
  accessorGlobal,
  AluExp,
  AluOp,
  AluVar,
  byteWidth,
  DataArray,
  DType,
  dtypedArray,
  dtypedJsArray,
  Kernel,
  Reduction,
} from "../alu";
import { Backend, Device, Executable, getBackend, Slot } from "../backend";
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
  checkConvShape,
  pool,
  poolTranspose,
  prepareConv,
} from "./convolution";
import {
  CompareOp,
  getAval,
  newMain,
  Primitive,
  PrimitiveParams,
  ShapedArray,
  Trace,
  Tracer,
  TracerValue,
  UseAfterFreeError,
} from "./core";
import { jitCompile } from "./jit";

const JsArray = globalThis.Array;

// Don't realize expression arrays smaller than this size.
const inlineArrayLimit = 128;

export type ArrayLike = Array | number | boolean;

/** Version of pureArray with fudged types. */
export const fudgeArray = pureArray as (x: ArrayLike) => Array;

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
    // Take a reference to all I/O buffers, while this execution is pending.
    // The reference is dropped after submit() or cancellation.
    for (const slot of inputs) this.backend.incRef(slot);
    for (const slot of outputs) this.backend.incRef(slot);
  }

  // Change the reference count of the PendingExecute object.
  // Used when copying the object to a new Array, or disposing an array.
  updateRc(delta: number) {
    if (this.#rc <= 0) throw new Error("internal: PendingExecute used rc<=0");
    this.#rc += delta;
    if (this.#rc <= 0 && !this.submitted) {
      // Cancel operation, release the references held to all input buffers.
      for (const slot of this.inputs) this.backend.decRef(slot);
      for (const slot of this.outputs) this.backend.decRef(slot);
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
    for (const slot of this.outputs) this.backend.decRef(slot);
  }
}

/** @inline */
export type DTypeAndDevice = { dtype?: DType; device?: Device };

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

  /**
   * @ignore
   * Constructs an array from source, shape and backend. Note that if the source
   * is a backend `Slot`, this constructor _takes ownership_ of the slot. It
   * will be freed when the array is disposed.
   */
  constructor(
    source: AluExp | Slot,
    st: ShapeTracker,
    dtype: DType,
    backend: Backend,
    {
      pending = null,
    }: {
      pending?: Iterable<PendingExecute> | null;
    } = {},
  ) {
    super(baseArrayTrace);
    this.id = Array.#nextId++;
    this.#dtype = dtype;
    this.#source = source;
    this.#st = st;
    this.#backend = backend;
    this.#rc = 1;

    this.#pendingSet = new Set(pending);
    if (this.#pendingSet.size === 0) {
      this.#pendingSet = null;
    } else if (source instanceof AluExp) {
      throw new Error("internal: AluExp source cannot have pending executes");
    }
  }

  /** @ignore */
  get aval() {
    return new ShapedArray(this.#st.shape, this.#dtype);
  }

  /** Return a simple string representation of the array's dimensions. */
  toString(): string {
    return `Array:${this.#dtype}[${this.shape.join(",")}]`;
  }

  get device(): Device {
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
    if (--this.#rc === 0) {
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
    if (typeof this.#source === "number") this.#backend.incRef(this.#source);
    const ar = new Array(this.#source, st, this.#dtype, this.#backend, {
      pending,
    });
    this.dispose(); // After constructing Array, so we don't free this.#source early.
    return ar;
  }

  /**
   * Underlying implementation of the Gather primitive. This indexes an array
   * and extracts slices based on indices in other integer arrays.
   */
  #gather(indices: Array[], axis: number[], outDim: number): Array {
    this.#check();

    if (indices.some((a) => a.#backend !== this.#backend)) {
      throw new TypeError(
        `Gather indices must have the same backend: ${this.#backend.type}`,
      );
    }
    const axisSet = new Set(axis);
    if (axisSet.size !== axis.length) {
      throw new TypeError("Gather axis must not have duplicates");
    }

    // First, broadcast each integer array in `indices`.
    indices = Array.#broadcastArrays(indices);
    const indexShape = indices[0].shape;
    const finalShape = this.shape.filter((_, i) => !axisSet.has(i));
    finalShape.splice(outDim, 0, ...indexShape);

    // Make variables for expression indices for gathered axes, and non-axis.
    const idxAll = unravelAlu(finalShape, AluVar.gidx);
    const idxNonaxis = [...idxAll];
    const idxAxis = idxNonaxis.splice(outDim, indexShape.length);

    // Then, construct a kernel expression that gathers the data.
    const inputs: Slot[] = [];
    const src: AluExp[] = [...idxNonaxis];
    for (let i = 0; i < this.shape.length; i++) {
      // insert 'null' as axis placeholder, overwritten below as src[axis[i]].
      if (axisSet.has(i)) src.splice(i, 0, null as any);
    }
    for (const [i, ar] of indices.entries()) {
      if (ar.#source instanceof AluExp) {
        src[axis[i]] = AluExp.cast(
          DType.Int32,
          accessorAluExp(ar.#source, ar.#st, idxAxis),
        );
      } else {
        let gid = inputs.indexOf(ar.#source);
        if (gid === -1) {
          gid = inputs.length;
          inputs.push(ar.#source);
        }
        src[axis[i]] = AluExp.cast(
          DType.Int32,
          AluExp.globalView(ar.#dtype, gid, ar.#st, idxAxis),
        );
      }
    }

    let exp: AluExp;
    if (this.#source instanceof AluExp) {
      // This is an AluExp, not an actual array, so turn it into an expression.
      exp = accessorAluExp(this.#source, this.#st, src);
    } else {
      let gid = inputs.indexOf(this.#source);
      if (gid === -1) {
        gid = inputs.length;
        inputs.push(this.#source);
      }
      exp = accessorGlobal(this.#dtype, gid, this.#st, src);
    }

    const kernel = new Kernel(inputs.length, prod(finalShape), exp);
    const output = this.#backend.malloc(kernel.bytes);
    const pending = [...this.#pending, ...indices.flatMap((ar) => ar.#pending)];
    for (const exe of pending) exe.updateRc(+1);
    pending.push(new PendingExecute(this.#backend, kernel, inputs, [output]));

    // Dispose of inputs after creating PendingExecute.
    this.dispose();
    for (const ar of indices) ar.dispose();

    return new Array(
      output,
      ShapeTracker.fromShape(finalShape),
      this.#dtype,
      this.#backend,
      { pending },
    );
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

  #unary(op: AluOp, dtypeOutput?: DType) {
    dtypeOutput ??= this.#dtype; // Default to current dtype unless changed.

    this.#check();
    // Short circuit if the array is already AluExp.
    if (this.#source instanceof AluExp) {
      const exp = new AluExp(op, dtypeOutput, [this.#source]);
      return new Array(exp.simplify(), this.#st, dtypeOutput, this.#backend);
    }

    const indices = unravelAlu(this.#st.shape, AluVar.gidx);
    const exp = new AluExp(op, dtypeOutput, [
      AluExp.globalView(this.#dtype, 0, this.#st, indices),
    ]);
    const kernel = new Kernel(1, this.#st.size, exp);
    const output = this.#backend.malloc(kernel.bytes);
    const pending = [...this.#pending];
    for (const exe of pending) exe.updateRc(+1);
    pending.push(
      new PendingExecute(this.#backend, kernel, [this.#source], [output]),
    );

    this.dispose(); // Dispose of inputs after creating PendingExecute.
    return new Array(
      output,
      ShapeTracker.fromShape(this.shape),
      dtypeOutput,
      this.#backend,
      { pending },
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
    {
      dtypeOverride,
      dtypeOutput,
      reduceAxis,
    }: {
      dtypeOverride?: (DType | undefined)[];
      dtypeOutput?: DType;
      reduceAxis?: boolean;
    } = {},
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

    arrays = Array.#broadcastArrays(arrays);
    const newShape = [...arrays[0].shape];

    // Short circuit if all are already AluExp.
    if (arrays.every((ar) => ar.#source instanceof AluExp) && !reduceAxis) {
      if (arrays.every((ar) => deepEqual(ar.#st, arrays[0].#st))) {
        // All are AluExp and have the same shape tracker.
        const exp = custom(arrays.map((ar) => ar.#source as AluExp));
        return new Array(exp.simplify(), arrays[0].#st, exp.dtype, backend);
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
      return new Array(exp.simplify(), st, exp.dtype, backend);
    }

    let indices: AluExp[];
    if (!reduceAxis) {
      indices = unravelAlu(newShape, AluVar.gidx);
    } else {
      const contractedShape = newShape.slice(0, -1);
      indices = [...unravelAlu(contractedShape, AluVar.gidx), AluVar.ridx];
    }

    const inputs: Slot[] = [];
    const src: AluExp[] = [];
    for (const ar of arrays) {
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
    let re: Reduction | undefined = undefined;
    if (reduceAxis) {
      const [axisSize] = newShape.splice(-1, 1); // Remove the contracted axis.
      re = new Reduction(exp.dtype, AluOp.Add, axisSize);
    }
    const kernel = new Kernel(inputs.length, prod(newShape), exp, re);
    const output = backend.malloc(kernel.bytes);
    const pending = new Set([...arrays.flatMap((ar) => ar.#pending)]);
    for (const exe of pending) exe.updateRc(+1);
    pending.add(new PendingExecute(backend, kernel, inputs, [output]));

    for (const ar of arrays) ar.dispose(); // Dispose of inputs after creating PendingExecute.
    return new Array(
      output,
      ShapeTracker.fromShape(newShape),
      dtypeOutput,
      backend,
      { pending },
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

    let exp: AluExp;
    const inputs: Slot[] = [];
    if (this.#source instanceof AluExp) {
      exp = accessorAluExp(this.#source, this.#st, indices);
    } else {
      inputs.push(this.#source);
      exp = accessorGlobal(this.#dtype, 0, this.#st, indices);
    }

    const kernel = new Kernel(inputs.length, newSize, exp, reduction);
    const output = this.#backend.malloc(kernel.bytes);
    const pending = [...this.#pending];
    for (const exe of pending) exe.updateRc(+1);
    pending.push(new PendingExecute(this.#backend, kernel, inputs, [output]));

    this.dispose(); // Dispose of inputs after creating PendingExecute.
    return new Array(
      output,
      ShapeTracker.fromShape(newShape),
      this.#dtype,
      this.#backend,
      { pending },
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
      const output = this.#backend.malloc(kernel.bytes);
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
      const output = this.#backend.malloc(kernel.bytes);
      const pendingItem = new PendingExecute(
        this.#backend,
        kernel,
        [this.#source],
        [output],
      );
      this.#backend.decRef(this.#source);
      this.#source = output;
      this.#st = ShapeTracker.fromShape(this.shape);
      this.#pendingSet ??= new Set();
      this.#pendingSet.add(pendingItem);
    }
  }

  #dataInline(): DataArray {
    this.#check();
    const exp = this.#source as AluExp;
    const ar = new Array(exp, this.#st, this.dtype, getBackend("cpu"));
    this.dispose();
    return ar.dataSync();
  }

  static #broadcastArrays(arrays: Array[]): Array[] {
    // Broadcast all arrays to the same shape.
    if (arrays.length === 0)
      throw new Error("Need at least one array to broadcast");
    if (arrays.length === 1) return arrays;
    const newShape = arrays.map((a) => a.shape).reduce(generalBroadcast);
    return arrays.map((ar) => {
      if (deepEqual(ar.shape, newShape)) return ar;
      return ar.#reshape(
        ar.#st.broadcast(newShape, range(newShape.length - ar.ndim)),
      );
    });
  }

  /** Realize the array and return it as data. */
  async data(): Promise<DataArray> {
    if (
      this.#source instanceof AluExp &&
      this.size < inlineArrayLimit &&
      this.device !== "cpu"
    ) {
      return this.#dataInline();
    }
    this.#realize();
    const pending = this.#pending;
    if (pending) {
      // Compile all pending executables concurrently.
      await Promise.all(pending.map((p) => p.prepare()));
      for (const p of pending) p.submit();
    }
    // While the array is contiguous, it might not be the whole buffer.
    const byteCount = byteWidth(this.#dtype) * this.size;
    const buf = await this.#backend.read(this.#source as Slot, 0, byteCount);
    this.dispose();
    return dtypedArray(this.dtype, buf);
  }

  /**
   * Wait for this array to finish evaluation.
   *
   * Operations and data loading in jax-js are lazy, so this function ensures
   * that pending operations are dispatched and fully executed before it
   * returns.
   *
   * If you are mapping from `data()` or `dataSync()`, it will also trigger
   * dispatch of operations as well.
   *
   * **Note:** `jax.blockUntilReady()` is a higher-level API, it calls this
   * asynchronously for multiple arrays.
   */
  async blockUntilReady(): Promise<Array> {
    this.#check();
    if (this.#source instanceof AluExp) return this;
    const pending = this.#pending;
    if (pending) {
      // Compile all pending executables concurrently.
      await Promise.all(pending.map((p) => p.prepare()));
      for (const p of pending) p.submit();
    }
    await this.#backend.read(this.#source, 0, 0);
    return this;
  }

  /**
   * Realize the array and return it as data. This is a sync variant and not
   * recommended for performance reasons, as it will block rendering.
   */
  dataSync(): DataArray {
    if (
      this.#source instanceof AluExp &&
      this.size < inlineArrayLimit &&
      this.device !== "cpu"
    ) {
      return this.#dataInline();
    }
    this.#realize();
    for (const p of this.#pending) {
      p.prepareSync();
      p.submit();
    }
    // While the array is contiguous, it might not be the whole buffer.
    const byteCount = byteWidth(this.#dtype) * this.size;
    const buf = this.#backend.readSync(this.#source as Slot, 0, byteCount);
    this.dispose();
    return dtypedArray(this.dtype, buf);
  }

  /**
   * Convert this array into a JavaScript object.
   *
   * This is a blocking operation that will compile all of the shaders and wait
   * for execution to complete, synchronously. No other JavaScript code on the
   * site will be run during shader execution.
   *
   * To avoid blocking, prefer `jsAsync()` when possible.
   */
  js() {
    return dataToJs(this.dtype, this.dataSync(), this.shape);
  }

  /** Convert this array into a JavaScript object, asynchronously. */
  async jsAsync() {
    return dataToJs(this.dtype, await this.data(), this.shape);
  }

  /**
   * Copy an element of an array to a numeric scalar and return it.
   *
   * Throws an error if the array does not have a single element. The array must
   * either be rank-0, or all dimensions of the shape are 1.
   */
  item(): number {
    if (this.size !== 1) {
      throw new Error(`item() can only be called on arrays of size 1`);
    }
    return this.dataSync()[0];
  }

  /** @private Internal plumbing method for Array / Tracer ops. */
  static _implRules(): typeof implRules {
    return {
      [Primitive.Add]([x, y]) {
        return [x.#binary(AluOp.Add, y)];
      },
      [Primitive.Mul]([x, y]) {
        return [x.#binary(AluOp.Mul, y)];
      },
      [Primitive.Idiv]([x, y]) {
        return [x.#binary(AluOp.Idiv, y)];
      },
      [Primitive.Neg]([x]) {
        return [zerosLike(x.ref).#binary(AluOp.Sub, x)];
      },
      [Primitive.Reciprocal]([x]) {
        return [x.#unary(AluOp.Reciprocal)];
      },
      [Primitive.StopGradient]([x]) {
        return [x]; // Stop gradient is a no-op, just return the input.
      },
      [Primitive.Cast]([x], { dtype }) {
        return [x.#unary(AluOp.Cast, dtype)];
      },
      [Primitive.Bitcast]([x], { dtype }) {
        if (x.dtype === DType.Bool || dtype === DType.Bool) {
          throw new TypeError("Bitcast to/from bool is not allowed");
        }
        if (x.dtype === dtype) return [x];
        if (byteWidth(x.dtype) !== byteWidth(dtype)) {
          throw new TypeError(
            `Bitcast from ${x.dtype} to ${dtype} with different byte width`,
          );
        }
        if (x.#source instanceof AluExp) {
          return [x.#unary(AluOp.Bitcast, dtype)];
        } else {
          // Just keep the same data / source, but change the dtype.
          x.#backend.incRef(x.#source);
          const pending = x.#pending;
          for (const exe of pending) exe.updateRc(+1);
          const y = new Array(x.#source, x.#st, dtype, x.#backend, { pending });
          x.dispose();
          return [y];
        }
      },
      [Primitive.RandomBits]([k0, k1], { shape, mode }) {
        const keyShape = generalBroadcast(k0.shape, k1.shape);
        if (!deepEqual(generalBroadcast(keyShape, shape), shape)) {
          throw new TypeError(
            `Keys of shapes ${k0.shape} and ${k1.shape} cannot be broadcast to shape ${shape}`,
          );
        }
        // Arrays of size >2^32 won't fit into browser memory anyway, so it's
        // okay to take lazy iota this way for counters.
        const c0 = zeros(shape, { dtype: DType.Uint32, device: k0.device });
        const c1 = arange(0, prod(shape), 1, {
          dtype: DType.Uint32,
          device: k0.device,
        }).reshape(shape);
        const custom = ([k0, k1, c0, c1]: AluExp[]) =>
          AluExp.threefry2x32(k0, k1, c0, c1, mode);
        return [Array.#naryCustom("random_bits", custom, [k0, k1, c0, c1])];
      },
      [Primitive.Sin]([x]) {
        return [x.#unary(AluOp.Sin)];
      },
      [Primitive.Cos]([x]) {
        return [x.#unary(AluOp.Cos)];
      },
      [Primitive.Asin]([x]) {
        return [x.#unary(AluOp.Asin)];
      },
      [Primitive.Atan]([x]) {
        return [x.#unary(AluOp.Atan)];
      },
      [Primitive.Exp]([x]) {
        return [x.#unary(AluOp.Exp)];
      },
      [Primitive.Log]([x]) {
        return [x.#unary(AluOp.Log)];
      },
      [Primitive.Sqrt]([x]) {
        return [x.#unary(AluOp.Sqrt)];
      },
      [Primitive.Min]([x, y]) {
        return [x.#binary(AluOp.Min, y)];
      },
      [Primitive.Max]([x, y]) {
        return [x.#binary(AluOp.Max, y)];
      },
      [Primitive.Reduce]([x], { op, axis }) {
        if (axis.length === 0) return [x];
        return [x.#moveAxesDown(axis).#reduce(op)];
      },
      [Primitive.Pool]([x], { window, strides }) {
        const st = pool(x.#st, window, strides);
        return [x.#reshape(st)];
      },
      [Primitive.PoolTranspose]([x], { inShape, window, strides }) {
        const n = inShape.length;
        let st = poolTranspose(x.#st, inShape, window, strides);
        st = st.reshape([...st.shape.slice(0, n), prod(st.shape.slice(n))]);
        return [x.#reshape(st).#reduce(AluOp.Add)];
      },
      [Primitive.Dot]([x, y]) {
        return [
          Array.#naryCustom(
            "dot",
            ([x, y]: AluExp[]) => AluExp.mul(x, y),
            [x, y],
            { reduceAxis: true },
          ),
        ];
      },
      [Primitive.Conv]([x, y], params) {
        checkConvShape(x.shape, y.shape, params);
        const [stX, stY] = prepareConv(x.#st, y.#st, params);
        return [
          Array.#naryCustom(
            "conv",
            ([x, y]: AluExp[]) => AluExp.mul(x, y),
            [x.#reshape(stX), y.#reshape(stY)],
            { reduceAxis: true },
          ),
        ];
      },
      [Primitive.Compare]([x, y], { op }) {
        const custom = ([x, y]: AluExp[]) => aluCompare(x, y, op);
        return [
          Array.#naryCustom("compare", custom, [x, y], {
            dtypeOutput: DType.Bool,
          }),
        ];
      },
      [Primitive.Where]([cond, x, y]) {
        const custom = ([cond, x, y]: AluExp[]) => AluExp.where(cond, x, y);
        return [
          Array.#naryCustom("where", custom, [cond, x, y], {
            dtypeOverride: [DType.Bool],
          }),
        ];
      },
      [Primitive.Transpose]([x], { perm }) {
        return [x.#transpose(perm)];
      },
      [Primitive.Broadcast]([x], { shape, axis }) {
        return [x.#reshape(x.#st.broadcast(shape, axis))];
      },
      [Primitive.Reshape]([x], { shape }) {
        return [x.#reshape(x.#st.reshape(shape))];
      },
      [Primitive.Flip]([x], { axis }) {
        const arg = rep(x.ndim, false);
        for (const ax of axis) arg[ax] = true;
        return [x.#reshape(x.#st.flip(arg))];
      },
      [Primitive.Shrink]([x], { slice }) {
        return [x.#reshape(x.#st.shrink(slice))];
      },
      [Primitive.Pad]([x], { width }) {
        return [x.#reshape(x.#st.pad(width))];
      },
      [Primitive.Gather]([x, ...indices], { axis, outDim }) {
        return [x.#gather(indices, axis, outDim)];
      },
      [Primitive.JitCall](args, { jaxpr, numConsts }) {
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
        for (const exe of pending) exe.updateRc(+outputs.length - 1);

        const prevPending = [...new Set(args.flatMap((x) => x.#pending))];
        for (const exe of prevPending) exe.updateRc(+outputs.length);
        pending.splice(0, 0, ...prevPending); // Dispatch order of pending kernels is important.
        args.forEach((x) => x.dispose()); // Dispose of args after dispatch.

        return outputs.map((source, i) => {
          return new Array(
            source,
            ShapeTracker.fromShape(jaxpr.outs[i].aval.shape),
            jaxpr.outs[i].aval.dtype,
            backend,
            { pending },
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
  { dtype, device }: DTypeAndDevice = {},
) {
  // TODO: This should probably be merged with numpy.full().
  if (typeof value === "number") {
    dtype ??= DType.Float32; // default dtype for JS numbers
    if (
      ![DType.Float32, DType.Float16, DType.Int32, DType.Uint32].includes(dtype)
    )
      throw new TypeError(`Mismatched dtype for scalar ${value}`);
  } else if (typeof value === "boolean") {
    dtype ??= DType.Bool;
    if (
      ![
        DType.Float32,
        DType.Float16,
        DType.Int32,
        DType.Uint32,
        DType.Bool,
      ].includes(dtype)
    )
      throw new TypeError(`Mismatched dtype for scalar ${value}`);
  } else {
    throw new TypeError(`Invalid type for scalar ${value}`);
  }
  return new Array(
    AluExp.const(dtype, value),
    ShapeTracker.fromShape([]),
    dtype,
    getBackend(device),
  );
}

/** Constructor for creating a new array from data. */
export function array(
  values:
    | Array
    | Float16Array<ArrayBuffer>
    | Float32Array<ArrayBuffer>
    | Int32Array<ArrayBuffer>
    | Uint32Array<ArrayBuffer>
    | RecursiveArray<number>
    | RecursiveArray<boolean>,
  { shape, dtype, device }: { shape?: number[] } & DTypeAndDevice = {},
): Array {
  if (values instanceof Tracer) {
    if (shape && !deepEqual(values.shape, shape)) {
      values = values.reshape(shape);
    }
    if (dtype && values.dtype !== dtype) {
      throw new Error("array astype not implemented yet"); // TODO
      // values = values.astype(dtype);
    }
    return values;
  } else if (ArrayBuffer.isView(values)) {
    return arrayFromData(values, shape ?? [values.length], {
      dtype,
      device,
    });
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
    if (size === 0) return zeros(shape, { dtype, device });
    if (typeof flat[0] === "boolean") {
      dtype = dtype ?? DType.Bool;
      const data = new Int32Array(flat.map((x) => (x ? 1 : 0)));
      return arrayFromData(data, shape, { dtype, device });
    } else {
      dtype = dtype ?? DType.Float32;
      const data = dtypedJsArray(dtype, flat as number[]);
      return arrayFromData(data, shape, { dtype, device });
    }
  }
}

function arrayFromData(
  data: DataArray,
  shape: number[],
  { dtype, device }: DTypeAndDevice = {},
): Array {
  if (data.length < inlineArrayLimit) {
    // Check if all elements are of the same value and short-circuit.
    let allEqual = true;
    for (let i = 1; i < data.length; i++) {
      if (data[i] !== data[0]) {
        allEqual = false;
        break;
      }
    }
    if (allEqual) {
      // If all elements are equal, we can use a constant expression.
      return full(shape, data[0], { dtype, device });
    }
  }

  const backend = getBackend(device);
  if (ArrayBuffer.isView(data)) {
    const buf = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    if (data instanceof Float32Array) {
      if (dtype && dtype !== DType.Float32)
        throw new Error("Float32Array must have float32 type");
      dtype ??= DType.Float32;
    } else if (data instanceof Int32Array) {
      if (dtype && dtype !== DType.Int32 && dtype !== DType.Bool)
        throw new Error("Int32Array must have int32 or bool type");
      dtype ??= DType.Int32;
    } else if (data instanceof Uint32Array) {
      if (dtype && dtype !== DType.Uint32)
        throw new Error("Uint32Array must have uint32 type");
      dtype ??= DType.Uint32;
    } else if (data instanceof Float16Array) {
      if (dtype && dtype !== DType.Float16)
        throw new Error("Float16Array must have float16 type");
      dtype ??= DType.Float16;
    } else {
      throw new Error(
        "Unsupported data array type: " + (data as any).constructor.name,
      );
    }
    const slot = backend.malloc(data.byteLength, buf);
    return new Array(slot, ShapeTracker.fromShape(shape), dtype, backend);
  } else {
    throw new Error("Unsupported data type: " + (data as any).constructor.name);
  }
}

function dataToJs(
  dtype: DType,
  data: DataArray,
  shape: number[],
): RecursiveArray<number> | RecursiveArray<boolean> | any {
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

  processPrimitive<P extends Primitive>(
    primitive: P,
    tracers: Array[],
    params: PrimitiveParams<P>,
  ): Tracer[] {
    return implRules[primitive](tracers, params);
  }
}

// Special bottom of the stack: must be level 0.
const baseArrayTrace = new EvalTrace(newMain(EvalTrace, null));

type ImplRule<P extends Primitive> = (
  tracers: Array[],
  params: PrimitiveParams<P>,
) => Array[];
const implRules: { [P in Primitive]: ImplRule<P> } = Array._implRules();

export function zerosLike(val: TracerValue, dtype?: DType): Array {
  const aval = getAval(val);
  if (val instanceof Tracer) val.dispose();
  // TODO: Use correct device.
  return zeros(aval.shape, { dtype: dtype ?? aval.dtype });
}

export function onesLike(val: TracerValue, dtype?: DType): Array {
  const aval = getAval(val);
  if (val instanceof Tracer) val.dispose();
  // TODO: Use correct device.
  return ones(aval.shape, { dtype: dtype ?? aval.dtype });
}

export function fullLike(
  val: TracerValue,
  fillValue: number | boolean | Array,
  dtype?: DType,
): Array {
  const aval = getAval(val);
  if (val instanceof Tracer) val.dispose();
  // TODO: Use correct device.
  return full(aval.shape, fillValue, { dtype: dtype ?? aval.dtype });
}

/** Return a new array of given shape and type, filled with zeros. */
export function zeros(
  shape: number[],
  { dtype, device }: DTypeAndDevice = {},
): Array {
  return full(shape, 0, { dtype, device });
}

/** Return a new array of given shape and type, filled with ones. */
export function ones(
  shape: number[],
  { dtype, device }: DTypeAndDevice = {},
): Array {
  return full(shape, 1, { dtype, device });
}

/** Return a new array of given shape and type, filled with `fill_value`. */
export function full(
  shape: number[],
  fillValue: number | boolean | Array,
  { dtype, device }: DTypeAndDevice = {},
): Array {
  let source: AluExp;
  if (typeof fillValue === "number") {
    dtype = dtype ?? DType.Float32;
    source = AluExp.const(dtype, fillValue);
  } else if (typeof fillValue === "bigint") {
    dtype = dtype ?? DType.Int32;
    source = AluExp.const(dtype, Number(fillValue));
  } else if (typeof fillValue === "boolean") {
    dtype = dtype ?? DType.Bool;
    source = AluExp.const(dtype, fillValue ? 1 : 0);
  } else if (fillValue instanceof Tracer) {
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
    getBackend(device),
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
  { dtype, device }: DTypeAndDevice = {},
): Array {
  numCols = numCols ?? numRows;
  dtype = dtype ?? DType.Float32;

  if (numCols < numRows) {
    // If less columns than rows, take the transpose since it's no longer a
    // simple modular arithmetic expression.
    const arr = eye(numCols, numRows, { dtype, device });
    return arr.transpose();
  }
  if (numRows === 0) {
    return zeros([0, numCols], { dtype, device });
  }

  const exp = AluExp.cmplt(
    AluExp.mod(AluVar.idx, AluExp.i32(numCols + 1)),
    AluExp.i32(1),
  );
  return new Array(
    AluExp.cast(dtype, exp),
    ShapeTracker.fromShape([numRows, numCols]),
    dtype,
    getBackend(device),
  );
}

/** Return the identity matrix, with ones on the main diagonal. */
export function identity(
  n: number,
  { dtype, device }: DTypeAndDevice = {},
): Array {
  return eye(n, n, { dtype, device });
}

/**
 * Return evenly spaced values within a given interval.
 *
 * This can be called with a varying number of arguments, just like the range()
 * builtin function in Python.
 *
 * - `arange(stop)` is equivalent to `arange(0, stop, 1)`.
 * - `arange(start, stop)` is equivalent to `arange(start, stop, 1)`.
 * - `arange(start, stop, step)` creates an array starting at `start`, ending
 *   before `stop`, with a step size of `step`.
 *
 * Defaults to an integer data type. This can produce unintended results when
 * using a non-integer step, so prefer linspace() in those cases.
 */
export function arange(
  start: number,
  stop?: number,
  step: number = 1,
  { dtype, device }: DTypeAndDevice = {},
) {
  dtype = dtype ?? DType.Int32; // default to int32 for arange

  if (stop === undefined) {
    stop = start;
    start = 0;
  }
  if (step === 0) {
    throw new RangeError(
      `Invalid step for arange: ${step}. Step must be non-zero.`,
    );
  }
  const size = Math.max(0, Math.ceil((stop - start) / step));
  if (size === 0) {
    return zeros([0], { dtype, device });
  }

  const exp = AluExp.add(
    AluExp.const(dtype, start),
    AluExp.mul(AluExp.cast(dtype, AluVar.idx), AluExp.const(dtype, step)),
  );
  const st = ShapeTracker.fromShape([size]);
  return new Array(exp, st, dtype, getBackend(device));
}

/**
 * Return evenly spaced numbers over a specified interval.
 *
 * Returns _num_ evenly spaced samples, calculated over the interval
 * [`start`, `stop`]. The endpoint `stop` is included in the result by default,
 * but this is controlled by the `endpoint` parameter.
 *
 * The default data type is Float32. Use arange() for integer steps.
 */
export function linspace(
  start: number,
  stop: number,
  num: number = 50,
  endpoint: boolean = true,
  { dtype, device }: DTypeAndDevice = {},
) {
  dtype = dtype ?? DType.Float32; // default to float32 for linspace

  if (num < 0 || !Number.isInteger(num)) {
    throw new RangeError(
      `Invalid num for linspace: ${num}. Must be non-negative integer.`,
    );
  } else if (num === 0) {
    return zeros([0], { dtype, device });
  } else if (num === 1) {
    return scalar(start, { dtype, device }).reshape([1]);
  } else if (start === stop) {
    return full([num], start, { dtype, device });
  }

  // Now num >= 2, there are at least 2 points.
  const delta = stop - start;
  const denom = endpoint ? num - 1 : num;
  const exp = AluExp.cast(
    dtype,
    AluExp.add(
      AluExp.f32(start),
      AluExp.mul(
        AluExp.f32(delta / denom),
        AluExp.cast(DType.Float32, AluVar.idx),
      ),
    ),
  );
  const st = ShapeTracker.fromShape([num]);
  return new Array(exp, st, dtype, getBackend(device));
}

export function aluCompare(a: AluExp, b: AluExp, op: CompareOp): AluExp {
  // Translate a `CompareOp` into an `AluExp` on two sub-expressions.
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
