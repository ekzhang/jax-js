import { AluOp, DType, isFloatDtype } from "./alu";
import {
  arange,
  Array,
  array,
  type ArrayLike,
  eye,
  fudgeArray,
  full,
  fullLike as fullLikeUnfudged,
  identity,
  linspace,
  ones,
  onesLike as onesLikeUnfudged,
  scalar,
  zeros,
  zerosLike as zerosLikeUnfudged,
} from "./frontend/array";
import * as core from "./frontend/core";
import * as vmapModule from "./frontend/vmap";
import { checkAxis, deepEqual, prod as iprod, range, rep } from "./utils";

export {
  arange,
  Array,
  type ArrayLike,
  array,
  DType,
  eye,
  identity,
  linspace,
  scalar,
  zeros,
  ones,
  full,
};

export const float32 = DType.Float32;
export const int32 = DType.Int32;
export const uint32 = DType.Uint32;
export const bool = DType.Bool;
export const float16 = DType.Float16;

// Constants section

/** Euler's constant, `e = 2.7182818284590...` */
export const e = Math.E;

/** Euler-Mascheroni constant, `γ = 0.5772156649...` */
export const eulerGamma = 0.5772156649015329;

/** Positive infinity. */
export const inf = Number.POSITIVE_INFINITY;

/** Floating-point representation of NaN. */
export const nan = Number.NaN;

/** This is Pi, `π = 3.14159265358979...` */
export const pi = Math.PI;

// Note: These primitive wrappers have fudged types.
//
// They can take any `TracerValue` and return any `Tracer` subclass based on the
// current stack of interpreters. But we hide that away from users to mimic
// JAX's composable tracing transformations.

/** Element-wise addition, with broadcasting. */
export const add = core.add as (x: ArrayLike, y: ArrayLike) => Array;
/** Element-wise multiplication, with broadcasting. */
export const multiply = core.mul as (x: ArrayLike, y: ArrayLike) => Array;
/** Numerical negative of every element of an array. */
export const negative = core.neg as (x: ArrayLike) => Array;
/** Calculate element-wise reciprocal of the input. This is `1/x`. */
export const reciprocal = core.reciprocal as (x: ArrayLike) => Array;
/** Element-wise sine function (takes radians). */
export const sin = core.sin as (x: ArrayLike) => Array;
/** Element-wise cosine function (takes radians). */
export const cos = core.cos as (x: ArrayLike) => Array;
/** Calculate the exponential of all elements in the input array. */
export const exp = core.exp as (x: ArrayLike) => Array;
/** Calculate the natural logarithm of all elements in the input array. */
export const log = core.log as (x: ArrayLike) => Array;
/** Calculate the square root of all elements in the input array. */
export const sqrt = core.sqrt as (x: ArrayLike) => Array;
/** Return element-wise minimum of the input arrays. */
export const minimum = core.min as (x: ArrayLike, y: ArrayLike) => Array;
/** Return element-wise maximum of the input arrays. */
export const maximum = core.max as (x: ArrayLike, y: ArrayLike) => Array;
/** Compare two arrays element-wise. */
export const greater = core.greater as (x: ArrayLike, y: ArrayLike) => Array;
/** Compare two arrays element-wise. */
export const less = core.less as (x: ArrayLike, y: ArrayLike) => Array;
/** Compare two arrays element-wise. */
export const equal = core.equal as (x: ArrayLike, y: ArrayLike) => Array;
/** Compare two arrays element-wise. */
export const notEqual = core.notEqual as (x: ArrayLike, y: ArrayLike) => Array;
/** Compare two arrays element-wise. */
export const greaterEqual = core.greaterEqual as (
  x: ArrayLike,
  y: ArrayLike,
) => Array;
/** Compare two arrays element-wise. */
export const lessEqual = core.lessEqual as (
  x: ArrayLike,
  y: ArrayLike,
) => Array;
/** Element-wise ternary operator, evaluates to `x` if cond else `y`. */
export const where = core.where as (
  cond: ArrayLike,
  x: ArrayLike,
  y: ArrayLike,
) => Array;
/** Permute the dimensions of an array. Defaults to reversing the axis order. */
export const transpose = core.transpose as (
  x: ArrayLike,
  perm?: number[],
) => Array;
/**
 * Give a new shape to an array without changing its data.
 *
 * One shape dimension can be -1. In this case, the value is inferred from the
 * length of the array and remaining dimensions.
 */
export const reshape = core.reshape as (x: ArrayLike, shape: number[]) => Array;
/** Move axes of an array to new positions. Other axes retain original order. */
export const moveaxis = vmapModule.moveaxis as (
  x: ArrayLike,
  src: number,
  dst: number,
) => Array;
/**
 * Add padding (zeros) to an array.
 *
 * The `width` argument is either an integer or pair of integers, in which case
 * all axes are padded with the same width. Or if it is an array of pairs, each
 * pair specifies the padding for its corresponding axis.
 */
export const pad = core.pad as (
  x: ArrayLike,
  width: number | [number, number] | [number, number][],
) => Array;

/** Return the number of dimensions of an array. Does not consume array reference. */
export const ndim = core.ndim as (x: ArrayLike) => number;

/** Return the shape of an array. Does not consume array reference. */
export const shape = core.getShape as (x: ArrayLike) => number[];

/** Return an array of zeros with the same shape and type as a given array. */
export const zerosLike = zerosLikeUnfudged as (
  a: ArrayLike,
  dtype?: DType,
) => Array;

/** Return an array of ones with the same shape and type as a given array. */
export const onesLike = onesLikeUnfudged as (
  a: ArrayLike,
  dtype?: DType,
) => Array;

/** Return a full array with the same shape and type as a given array. */
export const fullLike = fullLikeUnfudged as (
  a: ArrayLike,
  fillValue: number | boolean | Array,
  dtype?: DType,
) => Array;

/**
 * Return the number of elements in an array, optionally along an axis.
 * Does not consume array reference.
 */
export function size(a: ArrayLike, axis?: number): number {
  const s = shape(a);
  return axis === undefined ? iprod(s) : s[axis];
}

/** Convert an array to a specified dtype. */
export function astype(a: ArrayLike, dtype: DType): Array {
  return fudgeArray(a).astype(dtype);
}

/** Sum of the elements of the array over a given axis, or axes. */
export function sum(
  a: ArrayLike,
  axis?: number | number[],
  opts?: core.ReduceOpts,
): Array {
  return core.reduce(a, AluOp.Add, axis, opts) as Array;
}

/** Product of the array elements over a given axis. */
export function prod(
  a: ArrayLike,
  axis?: number | number[],
  opts?: core.ReduceOpts,
): Array {
  return core.reduce(a, AluOp.Mul, axis, opts) as Array;
}

/** Return the minimum of array elements along a given axis. */
export function min(
  a: ArrayLike,
  axis?: number | number[],
  opts?: core.ReduceOpts,
): Array {
  return core.reduce(a, AluOp.Min, axis, opts) as Array;
}

/** Return the maximum of array elements along a given axis. */
export function max(
  a: ArrayLike,
  axis?: number | number[],
  opts?: core.ReduceOpts,
): Array {
  return core.reduce(a, AluOp.Max, axis, opts) as Array;
}

/** Compute the average of the array elements along the specified axis. */
export function mean(
  a: ArrayLike,
  axis?: number | number[],
  opts?: core.ReduceOpts,
): Array {
  return fudgeArray(a).mean(axis, opts);
}

/**
 * Returns the indices of the minimum values along an axis.
 *
 * By default, index is into the flatted array, otherwise it is along the
 * specified axis.
 */
export function argmin(
  a: ArrayLike,
  axis?: number,
  opts?: core.ReduceOpts,
): Array {
  a = fudgeArray(a);
  if (axis === undefined) {
    a = a.ravel();
    axis = 0; // Default to the first axis of the flattened array.
  } else {
    axis = checkAxis(axis, a.ndim);
  }
  const shape = a.shape;
  const isMax = equal(a, min(a.ref, axis, { keepDims: true }));
  const length = scalar(shape[axis], { dtype: int32, device: a.device });
  const idx = isMax.astype(DType.Int32).mul(
    // Index by length-i instead of i, so we can take the max and get the first i.
    arange(shape[axis], 0, -1, { dtype: int32, device: a.device }).reshape([
      shape[axis],
      ...rep(shape.length - axis - 1, 1),
    ]),
  );
  return length.sub(max(idx, axis, opts));
}

/**
 * Returns the indices of the maximum values along an axis.
 *
 * By default, index is into the flatted array, otherwise it is along the
 * specified axis.
 */
export function argmax(
  a: ArrayLike,
  axis?: number,
  opts?: core.ReduceOpts,
): Array {
  a = fudgeArray(a);
  if (axis === undefined) {
    a = a.ravel();
    axis = 0; // Default to the first axis of the flattened array.
  } else {
    axis = checkAxis(axis, a.ndim);
  }
  const shape = a.shape;
  const isMax = equal(a, max(a.ref, axis, { keepDims: true }));
  const length = scalar(shape[axis], { dtype: int32, device: a.device });
  const idx = isMax.astype(DType.Int32).mul(
    // Index by length-i instead of i, so we can take the max and get the first i.
    arange(shape[axis], 0, -1, { dtype: int32, device: a.device }).reshape([
      shape[axis],
      ...rep(shape.length - axis - 1, 1),
    ]),
  );
  return length.sub(max(idx, axis, opts));
}

/** Reverse the elements in an array along the given axes. */
export function flip(x: ArrayLike, axis?: number | number[]): Array {
  const nd = ndim(x);
  if (axis === undefined) {
    axis = range(nd);
  } else if (typeof axis === "number") {
    axis = [axis];
  }
  const seen = new Set<number>();
  for (let i = 0; i < axis.length; i++) {
    if (axis[i] >= nd || axis[i] < -nd) {
      throw new Error(
        `flip: axis ${axis[i]} out of bounds for array of ${nd} dimensions`,
      );
    }
    if (axis[i] < 0) axis[i] += nd; // convert negative to positive
    if (seen.has(axis[i])) {
      throw new Error(`flip: duplicate axis ${axis[i]} in axis list`);
    }
    seen.add(axis[i]);
  }
  return core.flip(x, axis) as Array;
}

/**
 * Join a sequence of arrays along an existing axis.
 *
 * The arrays must have the same shape, except in the dimension corresponding to
 * `axis` (the first, by default).
 *
 * No scalars can be passed to this function, as the axis is then ambiguous.
 */
export function concatenate(xs: Array[], axis: number = 0) {
  if (xs.length === 0) {
    throw new Error("Need at least one array to concatenate");
  }
  const shapes = xs.map(shape);
  axis = checkAxis(axis, shapes[0].length);
  for (let i = 1; i < shapes.length; i++) {
    if (
      shapes[i].length !== shapes[0].length ||
      !shapes[i].every((d, j) => j === axis || d === shapes[0][j])
    ) {
      throw new Error(
        `Cannot concatenate arrays with shapes ${JSON.stringify(shapes)} along axis ${axis}`,
      );
    }
  }
  const makePadAxis = (start: number, end: number): [number, number][] =>
    shapes[0].map((_, i) => (i === axis ? [start, end] : [0, 0]));
  let result = xs[0];
  for (let i = 1; i < xs.length; i++) {
    const len1 = result.shape[axis];
    const len2 = shapes[i][axis];
    // Concatenate arrays by padding with zeros and adding them together.
    result = pad(result, makePadAxis(0, len2)).add(
      pad(xs[i], makePadAxis(len1, 0)),
    );
  }
  return result;
}

/**
 * Join a sequence of arrays along a new axis.
 *
 * The `axis` parameter specifies the index of the new axis in the dimensions of
 * the result. For example, if `axis=0` it will be the first dimension and if
 * `axis=-1` it will be the last dimension.
 *
 * All shapes must have the same shape.
 */
export function stack(xs: ArrayLike[], axis: number = 0) {
  if (xs.length === 0) {
    throw new Error("Need at least one array to stack");
  }
  const shapes = xs.map((x) => shape(x));
  if (!shapes.every((s) => deepEqual(s, shapes[0]))) {
    throw new Error(
      `Cannot stack arrays with different shapes: ${JSON.stringify(shapes)}`,
    );
  }
  axis = checkAxis(axis, shapes[0].length + 1); // +1 for the new axis
  const newShape = shapes[0].toSpliced(axis, 0, 1);
  const newArrays = xs.map((x) => fudgeArray(x).reshape(newShape));
  return concatenate(newArrays, axis) as Array;
}

/**
 * Horizontally stack arrays. Inputs are promoted to rank at least 1, then
 * concatenated along axis 1 (if rank-2 or higher) or 0 (if rank-1).
 */
export function hstack(xs: ArrayLike[]): Array {
  if (xs.length === 0) {
    throw new Error("Need at least one array to hstack");
  }
  const nds = xs.map(ndim);
  if (nds.some((n) => n !== nds[0])) {
    throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
  }
  if (nds[0] === 0) {
    return stack(xs); // Rank-0 arrays become rank-1
  } else if (nds[0] === 1) {
    return concatenate(xs as Array[]); // Rank-1 arrays become rank-1
  } else {
    // Rank-2 or higher arrays are concatenated along axis 1
    return concatenate(xs as Array[], 1);
  }
}

/**
 * Vertically stack arrays. Inputs are promoted to rank at least 2, then
 * concatenated along axis 0.
 */
export function vstack(xs: ArrayLike[]): Array {
  if (xs.length === 0) {
    throw new Error("Need at least one array to vstack");
  }
  const nds = xs.map(ndim);
  if (nds.some((n) => n !== nds[0])) {
    throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
  }
  if (nds[0] === 0) {
    return stack(xs).reshape([-1, 1]); // Rank-0 arrays become rank-2
  } else if (nds[0] === 1) {
    return stack(xs); // Rank-1 arrays become rank-2
  } else {
    // Rank-2 or higher arrays are concatenated along axis 0
    return concatenate(xs as Array[]);
  }
}

/**
 * Stack arrays depth-wise. Inputs are promoted to rank at least 3, then
 * concatenated along axis 2.
 */
export function dstack(xs: ArrayLike[]): Array {
  if (xs.length === 0) {
    throw new Error("Need at least one array to dstack");
  }
  const nds = xs.map(ndim);
  if (nds.some((n) => n !== nds[0])) {
    throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
  }
  if (nds[0] === 0) {
    return stack(xs).reshape([1, 1, -1]); // Rank-0 arrays become rank-3
  } else if (nds[0] === 1) {
    const ret = stack(xs, -1); // Tricky!
    return ret.reshape([1, ...ret.shape]);
  } else if (nds[0] === 2) {
    return stack(xs, -1);
  } else {
    return concatenate(xs as Array[], 2);
  }
}

/**
 * Stack arrays column-wise. Inputs are promoted to rank at least 2, then
 * concatenated along axis 1.
 */
export function columnStack(xs: ArrayLike[]): Array {
  if (xs.length === 0) {
    throw new Error("Need at least one array to columnStack");
  }
  const nds = xs.map(ndim);
  if (nds.some((n) => n !== nds[0])) {
    throw new Error(`Cannot stack different ranks: ${JSON.stringify(nds)}`);
  }
  if (nds[0] === 0) {
    return stack(xs).reshape([1, -1]); // Rank-0 arrays become rank-2
  } else if (nds[0] === 1) {
    return stack(xs, -1); // Rank-1 arrays become rank-2
  } else {
    // Rank-2 or higher arrays are concatenated along axis 1
    return concatenate(xs as Array[], 1);
  }
}

/** Flip an array vertically (axis=0). */
export function flipud(x: ArrayLike): Array {
  return flip(x, 0);
}

/** Flip an array horizontally (axis=1). */
export function fliplr(x: ArrayLike): Array {
  return flip(x, 1);
}

// Alternate or equivalent names for functions, from numpy.
export const permuteDims = transpose;

/** Return a 1-D flattened array containing the elements of the input. */
export function ravel(a: ArrayLike): Array {
  return fudgeArray(a).ravel();
}

/**
 * Return specified diagonals.
 *
 * If a is 2D, return the diagonal of the array with the given offset. If a is
 * 3D or higher, compute diagonals along the two given axes (default: 0, 1).
 *
 * This returns a view over the existing array. The shape of the resulting array
 * is determined by removing the two axes along which the diagonal is taken,
 * then appending a new axis to the right with holding the diagonals.
 */
export function diagonal(
  a: ArrayLike,
  offset?: number,
  axis1?: number,
  axis2?: number,
): Array {
  return fudgeArray(a).diagonal(offset, axis1, axis2);
}

/**
 * Extract a diagonal or construct a diagonal array.
 *
 * If v is a 2D array, return the k-th diagonal of v (as a view). If v is a 1D
 * array, return a 2D array with v on the k-th diagonal.
 */
export function diag(v: ArrayLike, k = 0): Array {
  const a = fudgeArray(v);
  if (!Number.isInteger(k))
    throw new TypeError(`k must be an integer, got ${k}`);
  if (a.ndim === 1) {
    const n = a.shape[0];
    const ret = where(eye(n).equal(1), a.ref, zerosLike(a));
    if (k > 0) {
      return pad(ret, [
        [0, k],
        [k, 0],
      ]);
    } else if (k < 0) {
      return pad(ret, [
        [-k, 0],
        [0, -k],
      ]);
    } else {
      return ret;
    }
  } else if (a.ndim === 2) {
    return diagonal(a, k);
  } else {
    throw new TypeError("numpy.diag only supports 1D and 2D arrays");
  }
}

/** Return if two arrays are element-wise equal within a tolerance. */
export function allclose(
  actual: Parameters<typeof array>[0],
  expected: Parameters<typeof array>[0],
  options?: { rtol?: number; atol?: number },
): boolean {
  const { rtol = 1e-5, atol = 1e-7 } = options ?? {};

  const x = array(actual);
  const y = array(expected);
  if (!deepEqual(x.shape, y.shape)) {
    return false;
  }
  const xData = x.dataSync();
  const yData = y.dataSync();
  for (let i = 0; i < xData.length; i++) {
    if (Math.abs(xData[i] - yData[i]) > atol + rtol * Math.abs(yData[i])) {
      return false;
    }
  }
  return true;
}

/** Matrix product of two arrays. */
export function matmul(x: ArrayLike, y: ArrayLike) {
  if (ndim(x) === 0 || ndim(y) === 0) {
    throw new TypeError("matmul: x and y must be at least 1D");
  }
  ((x = x as Array), (y = y as Array));
  if (y.ndim === 1) {
    // Matrix-vector product
    return core.dot(x, y) as Array;
  }

  // Otherwise, we multiply x: [..., N, K] and y: [..., K, M]
  x = x.reshape(x.shape.toSpliced(-1, 0, 1)); // [..., N, 1, K]
  y = y
    .reshape(y.shape.toSpliced(-2, 0, 1))
    .transpose([
      ...range(y.shape.length - 1),
      y.shape.length,
      y.shape.length - 1,
    ]); // [..., 1, M, K]

  return core.dot(x, y) as Array;
}

/** Dot product of two arrays. */
export function dot(x: ArrayLike, y: ArrayLike): Array {
  if (ndim(x) === 0 || ndim(y) === 0) {
    // Standard, scalar multiplication
    return multiply(x, y);
  }
  ((x = x as Array), (y = y as Array));
  if (y.ndim === 1) {
    // Matrix-vector product
    return core.dot(x, y) as Array;
  }
  // Otherwise, this is the "sum product" between the last axis of x, and the
  // second-to-last axis of y. (y.ndim >= 2)
  //
  // dot(x, y)[i,j,k,m] = sum(x[i,j,:] * y[k,:,m])
  x = x.reshape(x.shape.toSpliced(-1, 0, ...rep(y.ndim - 1, 1))); // [..., N, 1, 1, ..., 1, K]
  y = y.transpose([
    ...range(y.shape.length - 2),
    y.shape.length - 1,
    y.shape.length - 2,
  ]); // [..., M, K]

  return core.dot(x, y) as Array;
}

/** Vector dot product of two arrays. */
export function vecdot(x: ArrayLike, y: ArrayLike): Array {
  return core.dot(x, y) as Array;
}

/**
 * Return the dot product of two vectors.
 *
 * Like vecdot() but flattens the arguments first into vectors.
 */
export function vdot(x: ArrayLike, y: ArrayLike): Array {
  return vecdot(ravel(x), ravel(y));
}

/**
 * Return a tuple of coordinate matrices from coordinate vectors.
 *
 * Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector
 * fields over N-D grids, given one-dimensional coordinate arrays x1, x2,…, xn.
 */
export function meshgrid(
  xs: Array[],
  { indexing }: { indexing?: "xy" | "ij" } = {},
): Array[] {
  indexing ??= "xy"; // Default for numpy is "xy"

  for (const x of xs) {
    if (x.ndim !== 1) {
      throw new TypeError(
        `meshgrid: all inputs must be 1D arrays, got ${x.ndim}D array`,
      );
    }
  }
  if (xs.length <= 1) return xs;
  if (indexing === "xy") {
    // For "xy" indexing, we just have to reverse the first two values.
    const [a, b, ...rest] = xs;
    const [rb, ra, ...rrest] = meshgrid([b, a, ...rest], { indexing: "ij" });
    return [ra, rb, ...rrest];
  }

  // Now do the actual meshgrid construction, using movement operators.
  const shape = xs.map((x) => x.shape[0]);
  return xs.map(
    (x, i) =>
      core.broadcast(x, shape, [
        ...range(i),
        ...range(i + 1, xs.length),
      ]) as Array,
  );
}

/**
 * Clip (limit) the values in an array.
 *
 * Given an interval, values outside the interval are clipped to the interval
 * edges. For example, if an interval of [0, 1] is specified, values smaller
 * than 0 become 0, and values larger than 1 become 1.
 *
 * If either bound is undefined, it is ignored.
 */
export function clip(a: ArrayLike, min?: ArrayLike, max?: ArrayLike): Array {
  a = fudgeArray(a);
  if (max !== undefined) {
    a = minimum(a, max);
  }
  if (min !== undefined) {
    a = maximum(a, min);
  }
  return a; // No clipping, just return the original array.
}

/**
 * Calculate the absolute value element-wise.
 *
 * This is the same function as `jax.numpy.abs()`.
 */
export function absolute(x: ArrayLike): Array {
  x = fudgeArray(x);
  return where(less(x.ref, 0), x.ref.mul(-1), x);
}

/** Alias of `jax.numpy.absolute()`. */
export const abs = absolute;

/** Calculate element-wise square of the input array. */
export function square(x: ArrayLike): Array {
  x = fudgeArray(x);
  return x.ref.mul(x);
}

/** Compute a trigonometric tangent of each element of input. */
export function tan(x: ArrayLike): Array {
  x = fudgeArray(x);
  return sin(x.ref).div(cos(x));
}

/** Calculates the floating-point division of x by y element-wise. */
export function trueDivide(x: ArrayLike, y: ArrayLike): Array {
  x = fudgeArray(x);
  y = fudgeArray(y);
  if (!isFloatDtype(x.dtype) || !isFloatDtype(y.dtype)) {
    // TODO: Automatically cast to float if possible?
    throw new TypeError(
      `trueDivide: x and y must be floating-point arrays, got ${x.dtype} and ${y.dtype}`,
    );
  }
  return x.div(y);
}

/** Alias of `jax.numpy.trueDivide()`. */
export const divide = trueDivide;

/** Round input to the nearest integer towards zero. */
export function trunc(x: ArrayLike): Array {
  return core.idiv(x, 1) as Array; // Integer division truncates the decimal part.
}

/** Calculate `2**p` for all p in the input array. */
export function exp2(p: ArrayLike): Array {
  return exp(multiply(p, Math.LN2));
}

/** Return the base-2 logarithm of x, element-wise. */
export function log2(x: ArrayLike): Array {
  return log(x).mul(Math.LOG2E);
}

/** Return the base-10 logarithm of x, element-wise. */
export function log10(x: ArrayLike): Array {
  return log(x).mul(Math.LOG10E);
}

/**
 * Calculate element-wise hyperbolic sine of input.
 *
 * `sinh(x) = (exp(x) - exp(-x)) / 2`
 */
export function sinh(x: ArrayLike): Array {
  const ex = exp(x);
  const emx = reciprocal(ex.ref);
  return ex.sub(emx).mul(0.5);
}

/**
 * Calculate element-wise hyperbolic cosine of input.
 *
 * `cosh(x) = (exp(x) + exp(-x)) / 2`
 */
export function cosh(x: ArrayLike): Array {
  const ex = exp(x);
  const emx = reciprocal(ex.ref);
  return ex.add(emx).mul(0.5);
}

/**
 * Calculate element-wise hyperbolic tangent of input.
 *
 * `tanh(x) = sinh(x)/cosh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
 */
export function tanh(x: ArrayLike): Array {
  x = fudgeArray(x);
  const e2x = exp(x.ref.add(x));
  return e2x.ref.sub(1).div(e2x.add(1));
}
