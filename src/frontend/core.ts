/** @file Core library internals and interpreter stack, based on Autodidax. */

import { AluGroup, AluOp, DType, isFloatDtype, promoteTypes } from "../alu";
import { type Pair } from "../shape";
import {
  JsTreeDef,
  flatten as treeFlatten,
  unflatten as treeUnflatten,
} from "../tree";
import {
  checkAxis,
  DEBUG,
  generalBroadcast,
  isNumberPair,
  isPermutation,
  normalizeAxis,
  prod,
  range,
  rep,
} from "../utils";
import type { ConvParams } from "./convolution";
import type { Jaxpr } from "./jaxpr";

/**
 * Frontend primitive operations, which are lowered into Kernel objects before
 * being dispatched to the backend.
 *
 * Any operation between arrays can be described in these parts. This is also
 * the set of primitives that can occur in Jaxpr programs, and the level at
 * which transformations like vmap, grad, and jvp occur. They are loosely based
 * on [XLA](https://openxla.org/xla/operation_semantics).
 *
 * All n-ary operations support broadcasting, with NumPy semantics.
 */
export enum Primitive {
  Add = "add",
  Mul = "mul",
  Idiv = "idiv",
  Neg = "neg",
  Reciprocal = "reciprocal",
  StopGradient = "stop_gradient",
  Cast = "cast",
  Bitcast = "bitcast",
  RandomBits = "random_bits",
  Sin = "sin",
  Cos = "cos",
  Asin = "asin",
  Atan = "atan",
  Exp = "exp",
  Log = "log",
  Sqrt = "sqrt",
  Min = "min",
  Max = "max",
  Reduce = "reduce",
  Dot = "dot", // sum(x*y, axis=-1)
  Conv = "conv", // see lax.conv_general_dilated
  Pool = "pool",
  PoolTranspose = "pool_transpose",
  Compare = "compare",
  Where = "where",
  Transpose = "transpose",
  Broadcast = "broadcast",
  Reshape = "reshape",
  Flip = "flip",
  Shrink = "shrink",
  Pad = "pad",
  Gather = "gather",
  JitCall = "jit_call",
}

interface PrimitiveParamsImpl extends Record<Primitive, Record<string, any>> {
  [Primitive.Cast]: { dtype: DType };
  [Primitive.Bitcast]: { dtype: DType };
  [Primitive.Reduce]: { op: AluOp; axis: number[] };
  [Primitive.Conv]: ConvParams;
  [Primitive.Pool]: { window: number[]; strides: number[] };
  [Primitive.PoolTranspose]: {
    inShape: number[];
    window: number[];
    strides: number[];
  };
  [Primitive.Compare]: { op: CompareOp };
  [Primitive.Transpose]: { perm: number[] };
  [Primitive.Broadcast]: { shape: number[]; axis: number[] };
  [Primitive.RandomBits]: { shape: number[]; mode: "xor" | 0 | 1 };
  [Primitive.Reshape]: { shape: number[] };
  [Primitive.Flip]: { axis: number[] };
  [Primitive.Shrink]: { slice: Pair[] };
  [Primitive.Pad]: { width: Pair[] };
  [Primitive.Gather]: { axis: number[]; outDim: number };
  [Primitive.JitCall]: { name: string; jaxpr: Jaxpr; numConsts: number };
}

/** Type of parameters taken by each primitive. */
export type PrimitiveParams<T extends Primitive> =
  T extends keyof PrimitiveParamsImpl
    ? PrimitiveParamsImpl[T]
    : Record<string, never>;

export enum CompareOp {
  Greater = "greater",
  Less = "less",
  Equal = "equal",
  NotEqual = "not_equal",
  GreaterEqual = "greater_equal",
  LessEqual = "less_equal",
}

export function add(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Add, [x, y]);
}

export function mul(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Mul, [x, y]);
}

export function idiv(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Idiv, [x, y]);
}

export function neg(x: TracerValue) {
  return bind1(Primitive.Neg, [x]);
}

export function reciprocal(x: TracerValue) {
  return bind1(Primitive.Reciprocal, [x]);
}

export function stopGradient(x: TracerValue) {
  return bind1(Primitive.StopGradient, [x]);
}

export function cast(x: TracerValue, dtype: DType) {
  return bind1(Primitive.Cast, [x], { dtype });
}

export function bitcast(x: TracerValue, dtype: DType) {
  return bind1(Primitive.Bitcast, [x], { dtype });
}

export function randomBits(
  k0: Tracer,
  k1: Tracer,
  shape: number[],
  mode: "xor" | 0 | 1 = "xor",
) {
  return bind1(Primitive.RandomBits, [k0, k1], { shape, mode });
}

export function sin(x: TracerValue) {
  return bind1(Primitive.Sin, [x]);
}

export function cos(x: TracerValue) {
  return bind1(Primitive.Cos, [x]);
}

export function asin(x: TracerValue) {
  return bind1(Primitive.Asin, [x]);
}

export function atan(x: TracerValue) {
  return bind1(Primitive.Atan, [x]);
}

export function exp(x: TracerValue) {
  return bind1(Primitive.Exp, [x]);
}

export function log(x: TracerValue) {
  return bind1(Primitive.Log, [x]);
}

export function sqrt(x: TracerValue) {
  return bind1(Primitive.Sqrt, [x]);
}

export function min(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Min, [x, y]);
}

export function max(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Max, [x, y]);
}

/** @inline */
export type Axis = number | number[] | null;

/** @inline */
export type ReduceOpts = { keepdims?: boolean };

export function reduce(
  x: TracerValue,
  op: AluOp,
  axis: Axis = null,
  opts?: ReduceOpts,
) {
  if (!AluGroup.Reduce.has(op)) {
    throw new TypeError(`Invalid reduce operation: ${op}`);
  }
  axis = normalizeAxis(axis, ndim(x));
  const originalShape = getShape(x);
  let result = bind1(Primitive.Reduce, [x], { op, axis });
  if (opts?.keepdims) {
    result = result.reshape(
      originalShape.map((dim, i) => (axis.includes(i) ? 1 : dim)),
    );
  }
  return result;
}

export function dot(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Dot, [x, y]);
}

export function conv(x: Tracer, y: Tracer, params: Partial<ConvParams> = {}) {
  if (x.ndim !== y.ndim) {
    throw new Error(
      `conv() requires inputs with the same number of dimensions, got ${x.ndim} and ${y.ndim}`,
    );
  }
  const n = x.ndim - 2;
  if (n < 0) throw new Error("conv() requires at least 2D inputs");
  // conv shape check is delayed until interpretation.
  return bind1(Primitive.Conv, [x, y], {
    strides: params.strides ?? rep(n, 1),
    padding: params.padding ?? rep(n, [0, 0]),
    lhsDilation: params.lhsDilation ?? rep(n, 1),
    rhsDilation: params.rhsDilation ?? rep(n, 1),
  });
}

export function compare(x: TracerValue, y: TracerValue, op: CompareOp) {
  return bind1(Primitive.Compare, [x, y], { op });
}
export function greater(x: TracerValue, y: TracerValue) {
  return compare(x, y, CompareOp.Greater);
}
export function less(x: TracerValue, y: TracerValue) {
  return compare(x, y, CompareOp.Less);
}
export function equal(x: TracerValue, y: TracerValue) {
  return compare(x, y, CompareOp.Equal);
}
export function notEqual(x: TracerValue, y: TracerValue) {
  return compare(x, y, CompareOp.NotEqual);
}
export function greaterEqual(x: TracerValue, y: TracerValue) {
  return compare(x, y, CompareOp.GreaterEqual);
}
export function lessEqual(x: TracerValue, y: TracerValue) {
  return compare(x, y, CompareOp.LessEqual);
}

export function where(cond: TracerValue, x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Where, [cond, x, y]);
}

export function transpose(x: TracerValue, perm?: number[]) {
  perm = perm
    ? perm.map((a) => checkAxis(a, ndim(x)))
    : range(ndim(x)).reverse();
  if (!isPermutation(perm, ndim(x))) {
    throw new Error(
      `Invalid transpose permutation for ${ndim(x)} axes: ${JSON.stringify(perm)}`,
    );
  }
  return bind1(Primitive.Transpose, [x], { perm });
}

export function broadcast(x: TracerValue, shape: number[], axis: number[]) {
  axis = normalizeAxis(axis, shape.length);
  return bind1(Primitive.Broadcast, [x], { shape, axis });
}

export function reshape(x: TracerValue, shape: number | number[]) {
  if (typeof shape === "number") shape = [shape];
  const originalShape = getShape(x);
  const autoIdx = shape.indexOf(-1);
  if (autoIdx !== -1) {
    const remaining = prod(originalShape) / -prod(shape);
    if (!Number.isInteger(remaining) || remaining < 0) {
      throw new Error(
        `Invalid reshape: ${JSON.stringify(originalShape)} -> ${JSON.stringify(shape)}`,
      );
    }
    shape = shape.toSpliced(autoIdx, 1, remaining);
  }
  if (prod(originalShape) !== prod(shape)) {
    throw new Error(
      `Invalid reshape: ${JSON.stringify(originalShape)} -> ${JSON.stringify(shape)}`,
    );
  }
  return bind1(Primitive.Reshape, [x], { shape });
}

export function flip(x: TracerValue, axis: number[]) {
  axis = normalizeAxis(axis, ndim(x));
  return bind1(Primitive.Flip, [x], { axis });
}

export function shrink(x: TracerValue, slice: Pair[]) {
  const shape = getShape(x);
  if (!Array.isArray(slice) || !slice.every(isNumberPair)) {
    throw new Error(`Invalid shrink() type: ${JSON.stringify(slice)}`);
  }
  if (slice.length !== shape.length) {
    throw new Error(
      `Invalid shrink(): expected ${shape.length} axes, got ${slice.length}`,
    );
  }
  for (let i = 0; i < shape.length; i++) {
    const [start, end] = slice[i];
    if (start > end || start < 0 || end > shape[i]) {
      throw new Error(
        `Invalid shrink() slice for axis ${i}: [${start}, ${end}] on shape ${shape[i]}`,
      );
    }
  }
  return bind1(Primitive.Shrink, [x], { slice });
}

export function pad(x: TracerValue, width: number | Pair | Pair[]) {
  const nd = ndim(x);
  if (typeof width === "number") {
    width = [[width, width]];
  } else if (isNumberPair(width)) {
    width = [width as Pair];
  } else if (!Array.isArray(width) || !width.every(isNumberPair)) {
    throw new TypeError(`Invalid pad() type: ${JSON.stringify(width)}`);
  }
  if (width.length === 1) {
    const [w0, w1] = width[0]; // A single pair should be repeated for all axes.
    width = rep(nd, () => [w0, w1] as Pair);
  } else if (width.length !== nd) {
    throw new Error(`Invalid pad(): expected ${nd} axes, got ${width.length}`);
  }
  return bind1(Primitive.Pad, [x], { width });
}

export function gather(
  x: TracerValue,
  indices: TracerValue[],
  axis: number[], // one for each index
  outDim: number,
) {
  // Evaluate advanced indexing x[:, ...indices, :], with the index dimensions
  // starting at axis `outDim` in the output.
  if (indices.length === 0) {
    throw new Error("gather() requires at least one index");
  }
  if (!Array.isArray(axis) || axis.length !== indices.length) {
    throw new Error(
      `Invalid gather() axis: expected ${indices.length} axes, got ${JSON.stringify(axis)}`,
    );
  }
  axis = axis.map((a) => checkAxis(a, ndim(x)));
  if (new Set(axis).size !== axis.length) {
    throw new Error(
      `Invalid gather() axis: duplicate axes ${JSON.stringify(axis)}`,
    );
  }
  outDim = checkAxis(outDim, ndim(x) - axis.length + 1);
  return bind1(Primitive.Gather, [x, ...indices], { axis, outDim });
}

export function bind1<P extends Primitive>(
  prim: P,
  args: TracerValue[],
  params: PrimitiveParams<P> = {} as any,
) {
  const [results] = bind(prim, args, params);
  return results;
}

type MainTrace = {
  level: number;
  traceType: new (main: MainTrace) => Trace; // Concrete Trace subclass.
  globalData: any | null;
};

const traceStack: MainTrace[] = []; // Global trace stack, mutable
let dynamicTrace: MainTrace | null = null;

/**
 * Push an interpreter onto the trace stack. Use this like:
 * `using main = newMain(...);`
 */
export function newMain(
  traceType: any,
  globalData: any | null = null,
): Disposable & MainTrace {
  const level = traceStack.length;
  const main = { level, traceType, globalData };
  traceStack.push(main);
  return Object.assign(main, {
    [Symbol.dispose]() {
      traceStack.pop();
    },
  });
}

/**
 * Set the current dynamic trace, which stashes the current interpreter stack
 * and acts temporarily as the bottom of the stack. Use this like:
 * `using _dynamic = newDynamic(main);`
 */
export function newDynamic(main: MainTrace): Disposable {
  const prevDynamicTrace = dynamicTrace;
  dynamicTrace = main;
  return {
    [Symbol.dispose]() {
      dynamicTrace = prevDynamicTrace;
    },
  };
}

export type TracerValue = Tracer | number | boolean;

export abstract class Trace {
  constructor(readonly main: MainTrace) {}

  abstract pure(val: TracerValue): Tracer;
  abstract lift(val: Tracer): Tracer;

  abstract processPrimitive<P extends Primitive>(
    primitive: P,
    tracers: Tracer[],
    params: PrimitiveParams<P>,
  ): Tracer[];
}

/** Internal representation of an array value. */
export interface AbstractValue {
  /** Shape of the array. Must be a static tuple of non-negative dimensions. */
  shape: number[];

  /** Concrete data type of array elements. */
  dtype: DType;

  /**
   * Arrays created from JavaScript numbers (e.g., `np.array(3)`) are created as
   * _weakly typed_ unless a dtype is explicitly specified.
   *
   * Weakly typed values will automatically cast to the data type of other
   * arrays when used as an operand as an expression. This property only affects
   * how they promote in type casting; their memory layout is still determined
   * by the actual `dtype` field.
   *
   * ```ts
   * const x = np.array(3); // weakType = true, dtype = float32
   * const y = np.array([1, 2], { dtype: np.int32 }); // weakType = false, dtype = int32
   * const z = x.add(y); // z has dtype int32 because x is weakly typed
   * ```
   *
   * Weak types are present in JIT programs in their spec (e.g., Jaxpr inputs
   * and outputs can be weakly typed) form. But they're solely a frontend
   * concept. Backends are not aware of weak types.
   */
  weakType: boolean;
}

/**
 * Broadcast shapes and promote types with casting for two avals.
 *
 * This implements the weak type behavior described in `promoteTypes()`, but not
 * implemented in that function as `weakType` is not passed.
 */
export function promoteAvals(a: AbstractValue, b: AbstractValue): ShapedArray {
  const shape = generalBroadcast(a.shape, b.shape);
  const weakType = a.weakType && b.weakType;
  let dtype: DType;
  if (a.weakType === b.weakType) {
    // Both weak or both strong: use normal promotion rules.
    dtype = promoteTypes(a.dtype, b.dtype);
  } else if (a.weakType) {
    // a is weak, b is strong: use b's dtype.
    dtype = promoteTypes(b.dtype, DType.Uint32);
  } else {
    // b is weak, a is strong: use a's dtype.
    dtype = promoteTypes(a.dtype, DType.Uint32);
  }
  return new ShapedArray(shape, dtype, weakType);
}

export abstract class Tracer {
  /** @ignore */
  readonly _trace: Trace;

  constructor(trace: Trace) {
    this._trace = trace;
  }

  abstract get aval(): AbstractValue;
  abstract toString(): string;

  /**
   * Access an array by reference, incrementing the reference count.
   *
   * jax-js handles freeing arrays by using "move" semantics, like in Rust/C++.
   * Whenever you pass an array into a function, that function should consume
   * the array, and it will no longer be usable. For example, if you had:
   *
   * ```
   * const x = np.array([1, 2, 3]);
   * const y = np.add(x, x);
   * ```
   *
   * The second line does not work because the first parameter consumes `x`, and
   * then the second parameter will already have been freed / disposed.
   *
   * To fix this, you can write:
   *
   * ```
   * const y = np.add(x.ref, x);
   * ```
   *
   * Under the hood, every access to `.ref` increments the internal reference
   * count of the array. The reference count starts at 1. When it hits 0, the
   * memory behind the array is freed.
   */
  abstract get ref(): this;

  /**
   * Manually decrement the reference count of the array.
   *
   * Arrays are created with reference count 1. Whenever it is used as argument
   * to a function or other operation, it is disposed (i.e., reference count
   * decreases by 1) automatically. Whenever a `.ref` is created, the reference
   * count increases.
   *
   * You generally don't need to call this function directly since arrays are
   * automatically disposed after being passed into an operation. One common
   * exception is when writing a function and ignoring one of its arguments. In
   * that case, by convention you should dispose of that argument manually.
   *
   * ```
   * function myCustomOperation(a: np.Array, b: np.Array) {
   *   b.dispose(); // Needed to satisfy "move" rules.
   *   return a.add(1);
   * }
   * ```
   */
  abstract dispose(): void;

  /** The shape of the array. */
  get shape(): number[] {
    return this.aval.shape;
  }
  /** The total number of elements in the array. */
  get size(): number {
    return prod(this.shape);
  }
  /** The dtype of elements stored in the array. */
  get dtype(): DType {
    return this.aval.dtype;
  }
  /**
   * Whether the array is weakly typed.
   *
   * Weakly typed arrays will cast to the dtype of the other operand. See
   * `promoteTypes()` for details.
   */
  get weakType(): boolean {
    return this.aval.weakType;
  }

  /** The number of dimensions of the array. */
  get ndim(): number {
    return this.shape.length;
  }

  /** @ignore */
  fullLower(): Tracer {
    return this; // default implementation
  }

  // These types aren't technically correct since they don't account for the
  // fact that tracers can be lifted to different levels. But they simplify the
  // API visible to users.

  neg() {
    return neg(this) as this;
  }
  add(other: this | TracerValue) {
    return add(this, other) as this;
  }
  mul(other: this | TracerValue) {
    return mul(this, other) as this;
  }
  greater(other: this | TracerValue) {
    return greater(this, other) as this;
  }
  less(other: this | TracerValue) {
    return less(this, other) as this;
  }
  equal(other: this | TracerValue) {
    return equal(this, other) as this;
  }
  notEqual(other: this | TracerValue) {
    return notEqual(this, other) as this;
  }
  greaterEqual(other: this | TracerValue) {
    return greaterEqual(this, other) as this;
  }
  lessEqual(other: this | TracerValue) {
    return lessEqual(this, other) as this;
  }

  /** Sum of the elements of the array over a given axis, or axes. */
  sum(axis: Axis = null, opts?: ReduceOpts) {
    return reduce(this, AluOp.Add, axis, opts) as this;
  }

  /** Product of the array elements over a given axis. */
  prod(axis: Axis = null, opts?: ReduceOpts) {
    return reduce(this, AluOp.Mul, axis, opts) as this;
  }

  /** Compute the average of the array elements along the specified axis. */
  mean(axis: Axis = null, opts?: ReduceOpts) {
    axis = normalizeAxis(axis, this.ndim);
    const n = axis.reduce((acc, a) => acc * this.shape[a], 1);
    if (n === 0) {
      throw new Error("mean: cannot compute mean over zero-length axis");
    }
    const result = reduce(this, AluOp.Add, axis, opts);
    return result.mul(1 / n) as this;
  }

  /** Permute the dimensions of an array. Defaults to reversing the axis order. */
  transpose(perm?: number[]): this {
    return transpose(this, perm) as this;
  }

  /**
   * Give a new shape to an array without changing its data.
   *
   * One shape dimension can be -1. In this case, the value is inferred from the
   * length of the array and remaining dimensions.
   */
  reshape(shape: number | number[]): this {
    return reshape(this, shape) as this;
  }

  /** Copy the array and cast to a specified dtype. */
  astype(dtype: DType): this {
    if (this.dtype === dtype) return this; // No-op.
    return cast(this, dtype) as this;
  }

  // Below this line are composite operations built from primitives.

  /** Subtract an array from this one. */
  sub(other: this | TracerValue): this {
    return this.add(neg(other));
  }

  /** Divide an array by this one. */
  div(other: this | TracerValue): this {
    if (isFloatDtype(this.dtype)) {
      return this.mul(reciprocal(other));
    }
    return idiv(this, other) as this;
  }

  /** Return specified diagonals. See `numpy.diagonal` for full docs. */
  diagonal(offset = 0, axis1 = 0, axis2 = 1): this {
    if (!Number.isInteger(offset))
      throw new TypeError(`offset must be an integer, got ${offset}`);
    if (offset < 0) return this.diagonal(-offset, axis2, axis1);
    axis1 = checkAxis(axis1, this.ndim);
    axis2 = checkAxis(axis2, this.ndim);
    if (axis1 === axis2) throw new Error("axis1 and axis2 must not be equal");
    if (offset >= this.shape[axis2])
      throw new Error("offset exceeds axis size");

    // First, make sure that the last two axes are being taken.
    //
    // We can just move them to the end, since the behavior of diagonal() is to
    // append the new diagonal axis to the right side / end of the shape.
    let ar: Tracer = this;
    if (axis1 !== ar.ndim - 2 || axis2 !== ar.ndim - 1) {
      const perm = range(ar.ndim)
        .filter((i) => i !== axis1 && i !== axis2)
        .concat(axis1, axis2);
      ar = ar.transpose(perm);
    }

    const [n, m] = ar.shape.slice(-2);
    const diagSize = Math.min(n, m - offset);

    // Pad and reshape ar into a skewed array of shape [..., diagSize, m+1].
    ar = ar.reshape([...ar.shape.slice(0, -2), n * m]);
    const npad = diagSize * (m + 1) - n * m;
    if (npad > 0) {
      ar = pad(ar, [...rep<Pair>(ar.ndim - 1, [0, 0]), [0, npad]]);
    } else if (npad < 0) {
      ar = shrink(
        ar,
        [...ar.shape.slice(0, -1), n * m + npad].map<Pair>((x) => [0, x]),
      );
    }
    ar = ar.reshape([...ar.shape.slice(0, -1), diagSize, m + 1]);

    // Now slice the #offset element of the last axis, and this gives a diagonal.
    ar = shrink(ar, [
      ...ar.shape.slice(0, -1).map<Pair>((x) => [0, x]),
      [offset, offset + 1],
    ]).reshape(ar.shape.slice(0, -1));

    return ar as this;
  }

  /** Flatten the array without changing its data. */
  flatten(): this {
    return this.reshape(-1);
  }
  /** Flatten the array without changing its data. */
  ravel(): this {
    return this.reshape(-1);
  }

  /**
   * Iterate over the first dimension of this array, returning slices.
   *
   * This can be used to destructure arrays. For example:
   *
   * ```js
   * let x = np.array([[1, 2], [3, 4]]);
   * let [a, b] = x;
   * console.log(a.js()); // [1, 2]
   * console.log(b.js()); // [3, 4]
   * ```
   */
  *[Symbol.iterator](): IterableIterator<this> {
    if (this.ndim === 0) {
      throw new Error("Cannot iterate over a scalar array");
    }
    for (let i = 0; i < this.shape[0]; i++) {
      yield this.ref.slice(i);
    }
    this.dispose();
  }

  /**
   * Slice an array along one or more axes.
   *
   * This is the equivalent of slicing in Python, e.g. `x[1:3, 2, :, None]`. To
   * mimic this in JavaScript, we would write:
   *
   * ```js
   * x.slice([1, 3], 2, [], null);
   * ```
   *
   * The `slice` method accepts a variable number of arguments, each of which
   * can be a number, an empty array, a single-element array, a two-element
   * array, or `null`. The arguments are interpreted as follows:
   *
   * - A number `n` means to access the `n`-th element along that axis, removing
   *   that axis from the resulting shape.
   * - An empty array `[]` means to keep that axis as-is, like `:` in Python.
   * - A single-element array `[i]` means to start slicing from index `i`
   *   (inclusive) to the end of the axis, like `x[i:]`.
   * - A two-element array `[i, j]` means to slice from index `i` (inclusive)
   *   to index `j` (exclusive), like `x[i:j]`.
   * - `null` means to add a new axis at that position, like `np.newaxis`.
   *
   * Like in Python, negative indices are supported, which count from the end of
   * the axis. For example, `-1` means the last element.
   *
   * Strided slices are not yet implemented, so you cannot write `x[::2]` or
   * similar.
   *
   * Advanced indexing by integer arrays is also supported. This translates to
   * the "gather" primitive, and it allows you to access specific elements of
   * the array by integer indices stored in another array.
   */
  slice(...index: (number | [] | [number] | Pair | null | Tracer)[]): this {
    const checkBounds = (n: number, i: number): number => {
      if (i > n || i < -n)
        throw new RangeError(`Index ${i} out of bounds for axis of size ${n}`);
      return i < 0 ? n + i : i;
    };

    const hasAdvancedIdx = index.some((value) => value instanceof Tracer);
    const axesForGather: number[] = [];
    let outDim = -1; // Will be set if we have advanced indexing.
    if (hasAdvancedIdx) {
      // Find out which axes have scalar indices or advanced indices used. If
      // those axes are contiguous, we'll output advanced indexes there,
      // otherwise they get transposed to the beginning of the shape.
      //
      // Example:
      //
      // In [3]: np.zeros((10,)*4).shape
      // Out[3]: (10, 10, 10, 10)
      //
      // In [4]: np.zeros((10,)*4)[:, [1,2], [1,2], :].shape
      // Out[4]: (10, 2, 10)
      //
      // In [5]: np.zeros((10,)*4)[:, [1,2], :, [1,2]].shape
      // Out[5]: (2, 10, 10)
      //
      // In [6]: np.zeros((10,)*4)[:, [1,2], 5, [1,2]].shape
      // Out[6]: (10, 2)
      //
      // In [7]: np.zeros((10,)*4)[:, [1,2], :, 5].shape
      // Out[7]: (2, 10, 10)
      //
      // For more, see https://stackoverflow.com/questions/55829631
      const advancedAxes: number[] = [];
      let currentAxisForGather = 0;
      for (let i = 0; i < index.length; i++) {
        const value = index[i];
        if (value instanceof Tracer) {
          advancedAxes.push(i);
          axesForGather.push(currentAxisForGather++);
        } else if (typeof value === "number") {
          advancedAxes.push(i);
          // Do not increment currentAxisForGather, this is squeezed out.
        } else {
          currentAxisForGather++;
        }
      }
      if (
        advancedAxes[advancedAxes.length - 1] - advancedAxes[0] !==
        advancedAxes.length - 1
      ) {
        // If the advanced axes are not contiguous, we need to transpose them
        // to the beginning of the shape.
        outDim = 0;
      } else {
        outDim = axesForGather[0]; // OK, keep them in the same place.
      }
    }

    // 1. Calculate shape / operations for "basic" slicing.
    // 2. Squeeze out, i.e., reshape (1,) axes from scalar indices.
    // 3. Do gather if needed (hasAdvancedIdx).
    const slice: Pair[] = [];
    const basicShape: number[] = [];
    let needsReshape = false;
    let axis = 0;
    for (const value of index) {
      if (value === null) {
        // Add a new axis at this position.
        basicShape.push(1);
        needsReshape = true;
      } else if (typeof value === "number") {
        // Access the i-th element along this axis.
        if (axis >= this.shape.length) throw new RangeError("Too many indices");
        const i = checkBounds(this.shape[axis++], value);
        slice.push([i, i + 1]);
        needsReshape = true;
      } else if (Array.isArray(value)) {
        if (axis >= this.shape.length) throw new RangeError("Too many indices");
        const n = this.shape[axis++];
        if (value.length === 0) {
          // Keep this axis as-is, like `:`.
          basicShape.push(n);
          slice.push([0, n]);
        } else if (value.length === 1) {
          // Slice from index i to the end, like `x[i:]`.
          const i = checkBounds(n, value[0]);
          basicShape.push(n - i);
          slice.push([i, n]);
        } else if (value.length === 2) {
          // Slice from index i to index j, like `x[i:j]`.
          const [i, j] = value.map((v) => checkBounds(n, v));
          if (i > j) throw new RangeError(`Slice start at ${i} > end at ${j}`);
          basicShape.push(j - i);
          slice.push([i, j]);
        }
      } else if (value instanceof Tracer) {
        // Keep the full axis as-is like `:`, until the gather step.
        const n = this.shape[axis++];
        basicShape.push(n);
        slice.push([0, n]);
      } else {
        throw new TypeError(`Invalid slice argument: ${JSON.stringify(value)}`);
      }
    }
    while (axis < this.shape.length) {
      // If we didn't specify an index for this axis, keep it as-is.
      slice.push([0, this.shape[axis]]);
      basicShape.push(this.shape[axis++]);
    }
    let result = shrink(this, slice);
    result = needsReshape ? reshape(result, basicShape) : result;

    if (hasAdvancedIdx) {
      result = gather(
        result,
        index.filter((a) => a instanceof Tracer),
        axesForGather,
        outDim,
      );
    }

    return result as this;
  }
}

export function ndim(x: TracerValue) {
  if (x instanceof Tracer) {
    return x.shape.length;
  } else {
    return 0;
  }
}

export function getShape(x: TracerValue): number[] {
  return x instanceof Tracer ? x.shape : [];
}

// Note: Autodidax has a `ConcreteArray` type with "arrayAbstractionLevel" set
// to a higher value. I didn't see how this would be useful yet, so currently
// the only `AbstractValue` is a `ShapedArray` instance.
export class ShapedArray implements AbstractValue {
  constructor(
    readonly shape: number[],
    readonly dtype: DType,
    readonly weakType: boolean,
  ) {}

  static fromAval(aval: AbstractValue) {
    return new ShapedArray(aval.shape, aval.dtype, aval.weakType);
  }

  get ndim() {
    return this.shape.length;
  }

  toString() {
    return `${this.dtype}[${this.shape.join(",")}]`;
  }

  equals(other: ShapedArray) {
    return (
      this === other ||
      (this.constructor === other.constructor &&
        this.ndim === other.ndim &&
        this.shape.every((d, i) => d === other.shape[i]))
    );
  }
}

export function getAval(x: TracerValue): AbstractValue {
  if (x instanceof Tracer) {
    return x.aval;
  } else if (typeof x === "boolean" || typeof x === "number") {
    return new ShapedArray(
      [],
      typeof x === "boolean" ? DType.Bool : DType.Float32,
      typeof x === "boolean" ? false : true,
    );
  } else {
    throw new TypeError(`Unknown value: ${x}`);
  }
}

export function bind<P extends Primitive>(
  prim: P,
  args: TracerValue[],
  params: PrimitiveParams<P> = {} as any,
) {
  const topTrace = findTopTrace(args);
  const tracers = args.map((arg) => fullRaise(topTrace, arg));
  const outs = topTrace.processPrimitive(prim, tracers, params);
  if (DEBUG >= 5) {
    console.info(
      `processing rule for ${prim} on ${tracers.map((x) => x.toString())} and got ${outs.map((x) => x.toString())}`,
    );
  }
  return outs.map((out) => out.fullLower());
}

function findTopTrace(xs: TracerValue[]): Trace {
  let topMain: MainTrace = traceStack[0];
  for (const x of xs) {
    if (x instanceof Tracer && x._trace.main.level > topMain.level) {
      topMain = x._trace.main;
    }
  }
  if (dynamicTrace && dynamicTrace.level > topMain.level) {
    topMain = dynamicTrace;
  }
  return new topMain.traceType(topMain);
}

export function fullRaise(trace: Trace, val: TracerValue): Tracer {
  if (!(val instanceof Tracer)) {
    // remember to assert type(val) in jax_types
    return trace.pure(val);
  }
  const level = trace.main.level;
  if (Object.is(val._trace.main, trace.main)) {
    return val;
  } else if (val._trace.main.level < level) {
    return trace.lift(val);
  } else if (val._trace.main.level > level) {
    throw new Error(
      `Can't lift Tracer level ${val._trace.main.level} to level ${level}`,
    );
  } else {
    throw new Error(
      `Different traces at same level: ${val._trace.constructor}, ${trace.constructor}.`,
    );
  }
}

export class TreeMismatchError extends TypeError {
  constructor(where: string, left: JsTreeDef, right: JsTreeDef) {
    super(`Mismatched tree structures in ${where}: ${left} != ${right}`);
  }
}

export function flattenFun(
  f: any,
  inTree: JsTreeDef,
): [any, { value: JsTreeDef | undefined }] {
  const store: { value: JsTreeDef | undefined } = { value: undefined };
  const flatFun = (...argsFlat: any[]) => {
    const pytreeArgs = treeUnflatten(inTree, argsFlat);
    const out = f(...pytreeArgs);
    const [outFlat, outTree] = treeFlatten(out);
    store.value = outTree;
    return outFlat;
  };
  return [flatFun, store];
}

export class UseAfterFreeError extends ReferenceError {
  constructor(tracer: Tracer) {
    super(
      `Referenced tracer ${tracer.toString()} freed, please use .ref move semantics`,
    );
  }
}
