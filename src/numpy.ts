import { AluOp, DType, isFloatDtype, promoteTypes } from "./alu";
import {
  arange,
  Array,
  array,
  type ArrayLike,
  type DTypeAndDevice,
  eye,
  fudgeArray,
  full,
  fullLike as fullLikeUnfudged,
  identity,
  linspace,
  ones,
  onesLike as onesLikeUnfudged,
  zeros,
  zerosLike as zerosLikeUnfudged,
} from "./frontend/array";
import * as core from "./frontend/core";
import { jit } from "./frontend/jaxpr";
import * as vmapModule from "./frontend/vmap";
import {
  checkAxis,
  deepEqual,
  generalBroadcast,
  prod as iprod,
  normalizeAxis,
  range,
  rep,
} from "./utils";

export {
  arange,
  Array,
  type ArrayLike,
  array,
  DType,
  eye,
  full,
  identity,
  linspace,
  ones,
  promoteTypes,
  zeros,
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

/** @function Element-wise addition, with broadcasting. */
export const add = core.add as (x: ArrayLike, y: ArrayLike) => Array;
/** @function Element-wise multiplication, with broadcasting. */
export const multiply = core.mul as (x: ArrayLike, y: ArrayLike) => Array;
/** @function Numerical negative of every element of an array. */
export const negative = core.neg as (x: ArrayLike) => Array;
/** @function Calculate element-wise reciprocal of the input. This is `1/x`. */
export const reciprocal = core.reciprocal as (x: ArrayLike) => Array;
/** @function Element-wise sine function (takes radians). */
export const sin = core.sin as (x: ArrayLike) => Array;
/** @function Element-wise cosine function (takes radians). */
export const cos = core.cos as (x: ArrayLike) => Array;
/** @function Element-wise inverse sine function (inverse of sin). */
export const asin = core.asin as (x: ArrayLike) => Array;
/** @function Element-wise inverse tangent function (inverse of tan). */
export const atan = core.atan as (x: ArrayLike) => Array;
/** @function Calculate the exponential of all elements in the input array. */
export const exp = core.exp as (x: ArrayLike) => Array;
/** @function Calculate the natural logarithm of all elements in the input array. */
export const log = core.log as (x: ArrayLike) => Array;
/** @function Calculate the square root of all elements in the input array. */
export const sqrt = core.sqrt as (x: ArrayLike) => Array;
/** @function Return element-wise minimum of the input arrays. */
export const minimum = core.min as (x: ArrayLike, y: ArrayLike) => Array;
/** @function Return element-wise maximum of the input arrays. */
export const maximum = core.max as (x: ArrayLike, y: ArrayLike) => Array;
/** @function Compare two arrays element-wise. */
export const greater = core.greater as (x: ArrayLike, y: ArrayLike) => Array;
/** @function Compare two arrays element-wise. */
export const less = core.less as (x: ArrayLike, y: ArrayLike) => Array;
/** @function Compare two arrays element-wise. */
export const equal = core.equal as (x: ArrayLike, y: ArrayLike) => Array;
/** @function Compare two arrays element-wise. */
export const notEqual = core.notEqual as (x: ArrayLike, y: ArrayLike) => Array;
/** @function Compare two arrays element-wise. */
export const greaterEqual = core.greaterEqual as (
  x: ArrayLike,
  y: ArrayLike,
) => Array;
/** @function Compare two arrays element-wise. */
export const lessEqual = core.lessEqual as (
  x: ArrayLike,
  y: ArrayLike,
) => Array;
/** @function Element-wise ternary operator, evaluates to `x` if cond else `y`. */
export const where = core.where as (
  cond: ArrayLike,
  x: ArrayLike,
  y: ArrayLike,
) => Array;
/**
 * @function
 * Permute the dimensions of an array. Defaults to reversing the axis order.
 */
export const transpose = core.transpose as (
  x: ArrayLike,
  perm?: number[],
) => Array;
/**
 * @function
 * Give a new shape to an array without changing its data.
 *
 * One shape dimension can be -1. In this case, the value is inferred from the
 * length of the array and remaining dimensions.
 */
export const reshape = core.reshape as (x: ArrayLike, shape: number[]) => Array;
/**
 * @function
 * Move axes of an array to new positions. Other axes retain original order.
 */
export const moveaxis = vmapModule.moveaxis as (
  x: ArrayLike,
  src: number,
  dst: number,
) => Array;
/**
 * @function
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

/**
 * @function
 * Return the number of dimensions of an array. Does not consume array reference.
 */
export const ndim = core.ndim as (x: ArrayLike) => number;

/** @function Return the shape of an array. Does not consume array reference. */
export const shape = core.getShape as (x: ArrayLike) => number[];

/**
 * @function
 * Return an array of zeros with the same shape and type as a given array.
 */
export const zerosLike = zerosLikeUnfudged as (
  a: ArrayLike,
  dtype?: DType,
) => Array;

/**
 * @function
 * Return an array of ones with the same shape and type as a given array.
 */
export const onesLike = onesLikeUnfudged as (
  a: ArrayLike,
  dtype?: DType,
) => Array;

/**
 * @function
 * Return a full array with the same shape and type as a given array.
 */
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
  axis: core.Axis = null,
  opts?: core.ReduceOpts,
): Array {
  return core.reduce(a, AluOp.Add, axis, opts) as Array;
}

/** Product of the array elements over a given axis. */
export function prod(
  a: ArrayLike,
  axis: core.Axis = null,
  opts?: core.ReduceOpts,
): Array {
  return core.reduce(a, AluOp.Mul, axis, opts) as Array;
}

/** Return the minimum of array elements along a given axis. */
export function min(
  a: ArrayLike,
  axis: core.Axis = null,
  opts?: core.ReduceOpts,
): Array {
  return core.reduce(a, AluOp.Min, axis, opts) as Array;
}

/** Return the maximum of array elements along a given axis. */
export function max(
  a: ArrayLike,
  axis: core.Axis = null,
  opts?: core.ReduceOpts,
): Array {
  return core.reduce(a, AluOp.Max, axis, opts) as Array;
}

/** Compute the average of the array elements along the specified axis. */
export function mean(
  a: ArrayLike,
  axis: core.Axis = null,
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
  const isMax = equal(a, min(a.ref, axis, { keepdims: true }));
  const length = array(shape[axis], { dtype: int32, device: a.device });
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
  const isMax = equal(a, max(a.ref, axis, { keepdims: true }));
  const length = array(shape[axis], { dtype: int32, device: a.device });
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
export function flip(x: ArrayLike, axis: core.Axis = null): Array {
  const nd = ndim(x);
  axis = normalizeAxis(axis, nd);
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

/** @function Alternative name for `numpy.transpose()`. */
export const permuteDims = transpose;

/** Return a 1-D flattened array containing the elements of the input. */
export function ravel(a: ArrayLike): Array {
  return fudgeArray(a).ravel();
}

/**
 * Repeat each element of an array after themselves.
 *
 * If no axis is provided, use the flattened input array, and return a flat
 * output array.
 */
export function repeat(a: ArrayLike, repeats: number, axis?: number): Array {
  if (!Number.isInteger(repeats) || repeats < 0) {
    throw new Error(
      `repeat: repeats must be a non-negative integer, got ${repeats}`,
    );
  }
  a = fudgeArray(a);
  if (axis === undefined) {
    a = ravel(a);
    axis = 0;
  }
  axis = checkAxis(axis, a.ndim);
  if (repeats === 1) {
    return a;
  }
  const broadcastedShape = a.shape.toSpliced(axis + 1, 0, repeats);
  const finalShape = a.shape.toSpliced(axis, 1, a.shape[axis] * repeats);
  return core
    .broadcast(a, broadcastedShape, [axis + 1])
    .reshape(finalShape) as Array;
}

/**
 * Construct an array by repeating A the number of times given by reps.
 *
 * If `A` is an array of shape `(d1, d2, ..., dn)` and `reps` is a sequence of
 * integers, the resulting array will have a shape of `(reps[0] * d1,
 * reps[1] * d2, ..., reps[n] * dn)`, with `A` tiled along each dimension.
 */
export function tile(a: ArrayLike, reps: number | number[]): Array {
  a = fudgeArray(a);
  if (typeof reps === "number") reps = [reps];
  if (!reps.every((r) => Number.isInteger(r) && r >= 0)) {
    throw new Error(
      `tile: reps must be non-negative integers, got ${JSON.stringify(reps)}`,
    );
  }
  // Prepend 1s to match dimensions
  const ndiff = reps.length - a.ndim;
  if (ndiff > 0) a = a.reshape([...rep(ndiff, 1), ...a.shape]);
  if (ndiff < 0) reps = [...rep(-ndiff, 1), ...reps];
  // Build broadcasted shape by interleaving reps where > 1: [r1, d1, r2, d2, ...]
  const broadcastedShape: number[] = [];
  const broadcastAxes: number[] = [];
  for (let i = 0; i < a.ndim; i++) {
    if (reps[i] > 1) {
      broadcastedShape.push(reps[i]);
      broadcastAxes.push(broadcastedShape.length - 1);
    }
    broadcastedShape.push(a.shape[i]);
  }
  const finalShape = a.shape.map((d, i) => reps[i] * d);
  return core
    .broadcast(a, broadcastedShape, broadcastAxes)
    .reshape(finalShape) as Array;
}

/**
 * Broadcast an array to a shape, with NumPy-style broadcasing rules.
 *
 * In other words, this lets you append axes to the left, and/or expand
 * dimensions where the shape is 1.
 */
export function broadcastTo(a: ArrayLike, shape: number[]) {
  const nd = ndim(a);
  if (shape.length < nd) {
    throw new Error(
      `broadcastTo: target shape ${JSON.stringify(shape)} has fewer dimensions than input array: ${nd}`,
    );
  }
  return core.broadcast(a, shape, range(shape.length - nd)) as Array;
}

/** Broadcast input shapes to a common output shape. */
export function broadcastShapes(...shapes: number[][]): number[] {
  if (shapes.length === 0) return [];
  return shapes.reduce(generalBroadcast);
}

/** Broadcast arrays to a common shape. */
export function broadcastArrays(...arrays: ArrayLike[]): Array[] {
  const shapes = arrays.map((a) => shape(a));
  const outShape = broadcastShapes(...shapes);
  return arrays.map((a) => broadcastTo(a, outShape));
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

/**
 * Compute the inner product of two arrays.
 *
 * Unlike `jax.numpy.matmul()` or `jax.numpy.dot()`, this always performs a
 * contraction on the last axis.
 *
 * Returned array has shape `[...x.shape[:-1], ...y.shape[:-1]]`.
 */
export function inner(x: ArrayLike, y: ArrayLike): Array {
  x = reshape(x, shape(x).toSpliced(-1, 0, ...rep(ndim(y) - 1, 1)));
  return core.dot(x, y) as Array;
}

/**
 * Compute the outer product of two arrays.
 *
 * If the input arrays are not 1D, they will be flattened. Returned array will
 * be of shape `[x.size, y.size]`.
 */
export function outer(x: ArrayLike, y: ArrayLike): Array {
  x = ravel(x);
  y = ravel(y);
  return multiply(x.reshape([x.shape[0], 1]), y);
}

/** Vector dot product of two arrays along a given axis. */
export function vecdot(
  x: ArrayLike,
  y: ArrayLike,
  { axis }: { axis?: number } = {},
): Array {
  const xaxis = checkAxis(axis ?? -1, ndim(x));
  const yaxis = checkAxis(axis ?? -1, ndim(y));
  if (shape(x)[xaxis] !== shape(y)[yaxis]) {
    throw new Error(
      "vecdot: shapes " +
        `${JSON.stringify(shape(x))} and ${JSON.stringify(shape(y))} ` +
        `not aligned along axis ${axis}: ${shape(x)[xaxis]} != ${shape(y)[yaxis]}`,
    );
  }
  x = moveaxis(x, xaxis, -1);
  y = moveaxis(y, yaxis, -1);
  return core.dot(x, y) as Array;
}

/**
 * Return the dot product of two vectors.
 *
 * Like vecdot() but flattens the arguments first into vectors.
 */
export function vdot(x: ArrayLike, y: ArrayLike): Array {
  return core.dot(ravel(x), ravel(y)) as Array;
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
 * Return an array with ones on and below the diagonal and zeros elsewhere.
 *
 * If `k` is provided, it specifies the sub-diagonal on and below which the
 * array is filled with ones. `k=0` is the main diagonal, `k<0` is below it, and
 * `k>0` is above it.
 */
export function tri(
  n: number,
  m?: number,
  k: number = 0,
  { dtype, device }: DTypeAndDevice = {},
): Array {
  m ??= n;
  dtype ??= DType.Float32;
  if (!Number.isInteger(n) || n < 0) {
    throw new TypeError(`tri: n must be a non-negative integer, got ${n}`);
  }
  if (!Number.isInteger(m) || m < 0) {
    throw new TypeError(`tri: m must be a non-negative integer, got ${m}`);
  }
  if (!Number.isInteger(k)) {
    throw new TypeError(`tri: k must be an integer, got ${k}`);
  }
  const rows = arange(k, n + k, 1, { dtype: DType.Int32, device });
  const cols = arange(0, m, 1, { dtype: DType.Int32, device });
  return rows.reshape([n, 1]).greaterEqual(cols).astype(dtype);
}

/** Return the lower triangle of an array. Must be of dimension >= 2. */
export function tril(a: ArrayLike, k: number = 0): Array {
  if (ndim(a) < 2) {
    throw new TypeError(
      `tril: input array must be at least 2D, got ${ndim(a)}D`,
    );
  }
  a = fudgeArray(a);
  const [n, m] = a.shape.slice(-2);
  return where(tri(n, m, k, { dtype: bool }), a.ref, zerosLike(a)) as Array;
}

/** Return the upper triangle of an array. Must be of dimension >= 2. */
export function triu(a: ArrayLike, k: number = 0): Array {
  if (ndim(a) < 2) {
    throw new TypeError(
      `tril: input array must be at least 2D, got ${ndim(a)}D`,
    );
  }
  a = fudgeArray(a);
  const [n, m] = a.shape.slice(-2);
  return where(tri(n, m, k - 1, { dtype: bool }), zerosLike(a.ref), a) as Array;
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

/** @function Alias of `jax.numpy.absolute()`. */
export const abs = absolute;

/** Return an element-wise indication of sign of the input. */
export function sign(x: ArrayLike): Array {
  x = fudgeArray(x);
  return where(notEqual(x.ref, 0), where(less(x.ref, 0), -1, 1), 0);
}

/** Calculate element-wise square of the input array. */
export function square(x: ArrayLike): Array {
  x = fudgeArray(x);
  return x.ref.mul(x);
}

/** Element-wise tangent function (takes radians). */
export function tan(x: ArrayLike): Array {
  x = fudgeArray(x);
  return sin(x.ref).div(cos(x));
}

/** Element-wise inverse cosine function (inverse of cos). */
export function acos(x: ArrayLike): Array {
  return subtract(pi / 2, asin(x));
}

/**
 * @function
 * Return element-wise hypotenuse for the given legs of a right triangle.
 *
 * In the original NumPy/JAX implementation, this function is more numerically
 * stable than sqrt(x1**2 + x2**2). We don't currently implement those stability
 * improvements.
 */
export const hypot = jit(function hypot(x1: Array, x2: Array) {
  return sqrt(square(x1).add(square(x2)));
});

/**
 * @function
 * Element-wise arc tangent of y/x with correct quadrant.
 *
 * Returns the angle in radians between the positive x-axis and the point (x, y).
 * The result is in the range [-π, π].
 *
 * Uses numerically stable formulas:
 * - When x >= 0: atan2(y, x) = 2 * atan(y / (sqrt(x^2 + y^2) + x))
 * - When x < 0:  atan2(y, x) = 2 * atan((sqrt(x^2 + y^2) - x) / y)
 *
 * The output is ill-defined when both x and y are zero.
 */
export const atan2 = jit(function atan2(y: Array, x: Array) {
  const r = sqrt(square(x.ref).add(square(y.ref)));
  const xNeg = less(x.ref, 0);

  // Select numerator and denominator based on sign of x
  // When x >= 0: numer = y,     denom = r + x
  // When x < 0:  numer = r - x, denom = y
  const numer = where(xNeg.ref, r.ref.sub(x.ref), y.ref);
  const denom = where(xNeg, y, r.add(x));

  return atan(numer.div(denom)).mul(2);
});

/** @function Alias of `jax.numpy.acos()`. */
export const arccos = acos;
/** @function Alias of `jax.numpy.atan()`. */
export const arctan = atan;
/** @function Alias of `jax.numpy.atan2()`. */
export const arctan2 = atan2;

/** Element-wise subtraction, with broadcasting. */
export function subtract(x: ArrayLike, y: ArrayLike): Array {
  x = fudgeArray(x);
  y = fudgeArray(y);
  return x.sub(y);
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

/** @function Alias of `jax.numpy.trueDivide()`. */
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

/** Calculate `exp(x) - 1` element-wise. */
export function expm1(x: ArrayLike): Array {
  // TODO: This isn't actually higher precision than just exp(x)-1 right now.
  return exp(x).sub(1);
}

/** Calculate the natural logarithm of `1 + x` element-wise. */
export function log1p(x: ArrayLike): Array {
  // TODO: This isn't actually higher precision than just log(1+x) right now.
  return log(add(1, x));
}

/** Convert angles from degrees to radians. */
export function deg2rad(x: ArrayLike): Array {
  return multiply(x, pi / 180);
}

/** @function Alias of `jax.numpy.deg2rad()`. */
export const radians = deg2rad;

/** Convert angles from radians to degrees. */
export function rad2deg(x: ArrayLike): Array {
  return multiply(x, 180 / pi);
}

/** @function Alias of `jax.numpy.rad2deg()`. */
export const degrees = rad2deg;

/**
 * @function
 * Computes first array raised to power of second array, element-wise.
 */
export const power = jit(function power(x1: Array, x2: Array) {
  return exp(log(x1).mul(x2));
});

/** @function Alias of `jax.numpy.power()`. */
export const pow = power;

/** @function Calculate the element-wise cube root of the input array. */
export const cbrt = jit(function cbrt(x: Array) {
  // This isn't just power(x, 1/3) since we need to handle negative numbers.
  const sgn = where(less(x.ref, 0), -1, 1);
  return sgn.ref.mul(exp(log(x.mul(sgn)).mul(1 / 3)));
});

/**
 * @function
 * Calculate element-wise hyperbolic sine of input.
 *
 * `sinh(x) = (exp(x) - exp(-x)) / 2`
 */
export const sinh = jit(function sinh(x: Array) {
  const ex = exp(x);
  const emx = reciprocal(ex.ref);
  return ex.sub(emx).mul(0.5);
});

/**
 * @function
 * Calculate element-wise hyperbolic cosine of input.
 *
 * `cosh(x) = (exp(x) + exp(-x)) / 2`
 */
export const cosh = jit(function cosh(x: Array) {
  const ex = exp(x);
  const emx = reciprocal(ex.ref);
  return ex.add(emx).mul(0.5);
});

/**
 * @function
 * Calculate element-wise hyperbolic tangent of input.
 *
 * `tanh(x) = sinh(x)/cosh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))`
 */
export const tanh = jit(function tanh(x: Array) {
  // Avoid overflow for large x by taking advantage of alternate representations:
  // tanh(x) = -tanh(-x) = (1 - e^{-2x}) / (1 + e^{-2x})
  const negsgn = where(less(x.ref, 0), 1, -1);
  const en2x = exp(x.mul(negsgn.ref).mul(2));
  return en2x.ref.sub(1).div(en2x.add(1)).mul(negsgn);
});

/**
 * @function
 * Calculate element-wise inverse hyperbolic sine of input.
 *
 * `arcsinh(x) = ln(x + sqrt(x^2 + 1))`
 */
export const arcsinh = jit(function arcsinh(x: Array) {
  return log(x.ref.add(sqrt(square(x).add(1))));
});

/**
 * @function
 * Calculate element-wise inverse hyperbolic cosine of input.
 *
 * `arccosh(x) = ln(x + sqrt(x^2 - 1))`
 */
export const arccosh = jit(function arccosh(x: Array) {
  return log(x.ref.add(sqrt(square(x).sub(1))));
});

/**
 * @function
 * Calculate element-wise inverse hyperbolic tangent of input.
 *
 * `arctanh(x) = 0.5 * ln((1 + x) / (1 - x))`
 */
export const arctanh = jit(function arctanh(x: Array) {
  return log(add(1, x.ref).div(subtract(1, x))).mul(0.5);
});

/** @function Alias of `jax.numpy.arcsinh()`. */
export const asinh = arcsinh;
/** @function Alias of `jax.numpy.arccosh()`. */
export const acosh = arccosh;
/** @function Alias of `jax.numpy.arctanh()`. */
export const atanh = arctanh;

/**
 * Compute the variance of an array.
 *
 * The variance is computed for the flattened array by default, otherwise over
 * the specified axis.
 *
 * If `correction` is provided, the divisor in calculation is `N - correction`,
 * where `N` represents the number of elements (e.g., for Bessel's correction).
 */
export function var_(
  x: ArrayLike,
  axis: core.Axis = null,
  opts?: { mean?: ArrayLike; correction?: number } & core.ReduceOpts,
): Array {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  const n = axis.reduce((acc, a) => acc * x.shape[a], 1);
  if (n === 0) {
    throw new Error("var: cannot compute variance over zero-length axis");
  }
  const mu =
    opts?.mean !== undefined
      ? opts.mean
      : mean(x.ref, axis, { keepdims: true });
  return square(x.sub(mu))
    .sum(axis, { keepdims: opts?.keepdims })
    .mul(1 / (n - (opts?.correction ?? 0)));
}

/**
 * Compute the standard deviation of an array.
 *
 * The standard deviation is computed for the flattened array by default,
 * otherwise over the specified axis.
 *
 * If `correction` is provided, the divisor in calculation is `N - correction`,
 * where `N` represents the number of elements (e.g., for Bessel's correction).
 */
export function std(
  x: ArrayLike,
  axis: core.Axis = null,
  opts?: { mean?: ArrayLike; correction?: number } & core.ReduceOpts,
): Array {
  return sqrt(var_(x, axis, opts));
}
