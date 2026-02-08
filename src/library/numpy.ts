// Port of the `jax.numpy` module, typically imported as `np`.

import { AluOp, DType, isFloatDtype, promoteTypes } from "../alu";
import {
  arange,
  Array,
  array,
  type ArrayLike,
  eye,
  fudgeArray,
  full,
  fullLike as fullLikeTracer,
  identity,
  linspace,
  logspace,
  ones,
  onesLike as onesLikeTracer,
  tri,
  tril,
  triu,
  zeros,
  zerosLike as zerosLikeTracer,
} from "../frontend/array";
import * as core from "../frontend/core";
import { jit } from "../frontend/jaxpr";
import { moveaxis as moveaxisTracer } from "../frontend/vmap";
import { Pair } from "../shape";
import {
  checkAxis,
  DEBUG,
  deepEqual,
  generalBroadcast,
  prod as iprod,
  normalizeAxis,
  range,
  rep,
} from "../utils";
import * as lax from "./lax";
import { iinfo } from "./numpy";
import { finfo } from "./numpy";
import {
  computeEinsumPath,
  EinsumInput,
  parseEinsumExpression,
} from "./numpy/einsum";

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
  logspace,
  ones,
  promoteTypes,
  tri,
  tril,
  triu,
  zeros,
};

export * as fft from "./numpy-fft";
export * as linalg from "./numpy-linalg";

export const float32 = DType.Float32;
export const int32 = DType.Int32;
export const uint32 = DType.Uint32;
export const bool = DType.Bool;
export const float16 = DType.Float16;
export const float64 = DType.Float64;

export { finfo, iinfo } from "./numpy/dtype-info";

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
/** @function Round input down to the nearest integer. */
export const floor = core.floor as (x: ArrayLike) => Array;
/** @function Round input up to the nearest integer. */
export const ceil = core.ceil as (x: ArrayLike) => Array;
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
export const moveaxis = moveaxisTracer as (
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
  width: number | Pair | Pair[] | Record<number, Pair>,
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
export const zerosLike = zerosLikeTracer as (
  a: ArrayLike,
  dtype?: DType,
) => Array;

/**
 * @function
 * Return an array of ones with the same shape and type as a given array.
 */
export const onesLike = onesLikeTracer as (
  a: ArrayLike,
  dtype?: DType,
) => Array;

/**
 * @function
 * Return a full array with the same shape and type as a given array.
 */
export const fullLike = fullLikeTracer as (
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

/**
 * Test whether any array element along a given axis evaluates to True.
 *
 * Returns a boolean array with the same shape as `a` with the specified axis
 * removed. If axis is None, returns a scalar.
 */
export function any(
  a: ArrayLike,
  axis: core.Axis = null,
  opts?: core.ReduceOpts,
): Array {
  return fudgeArray(a).any(axis, opts);
}

/**
 * Test whether all array elements along a given axis evaluate to True.
 *
 * Returns a boolean array with the same shape as `a` with the specified axis
 * removed. If axis is None, returns a scalar.
 */
export function all(
  a: ArrayLike,
  axis: core.Axis = null,
  opts?: core.ReduceOpts,
): Array {
  return fudgeArray(a).all(axis, opts);
}

/** Return the peak-to-peak range along a given axis (`max - min`). */
export function ptp(
  a: ArrayLike,
  axis: core.Axis = null,
  opts?: core.ReduceOpts,
): Array {
  a = fudgeArray(a);
  return max(a.ref, axis, opts).sub(min(a, axis, opts));
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

/**
 * Cumulative sum of elements along an axis.
 *
 * Currently this function is `O(n^2)`, we'll improve this later on with a
 * two-phase parallel reduction algorithm.
 */
export function cumsum(a: ArrayLike, axis?: number): Array {
  a = fudgeArray(a);
  if (axis === undefined) {
    a = a.ravel();
    axis = 0;
  } else {
    axis = checkAxis(axis, a.ndim);
  }
  const n = a.shape[axis];
  a = moveaxis(a, axis, -1);
  a = core.broadcast(a, a.shape.concat(n), [-2]) as Array;
  return moveaxis(tril(a).sum(-1), -1, axis);
}

export { cumsum as cumulativeSum };

/** Reverse the elements in an array along the given axes. */
export function flip(x: ArrayLike, axis: core.Axis = null): Array {
  const nd = ndim(x);
  axis = normalizeAxis(axis, nd);
  return core.flip(x, axis) as Array;
}

/**
 * Split an array into multiple sub-arrays along an axis.
 *
 * @param a - The input array to split.
 * @param indicesOrSections - If an integer, it indicates the number of equal
 * sections to create along the specified axis. If a list of integers, it
 * specifies the indices at which to split the array.
 * @param axis - The axis along which to split the array. Default is 0.
 */
export function split(
  a: ArrayLike,
  indicesOrSections: number | number[],
  axis: number = 0,
): Array[] {
  a = fudgeArray(a);
  axis = checkAxis(axis, a.ndim);
  const size = a.shape[axis];
  let sizes: number[];
  if (typeof indicesOrSections === "number") {
    if (size % indicesOrSections !== 0) {
      throw new Error(
        `Array of size ${size} cannot be split into ${indicesOrSections} equal parts`,
      );
    }
    const partSize = size / indicesOrSections;
    sizes = rep(indicesOrSections, partSize);
  } else {
    const indices = indicesOrSections.map((i) => (i < 0 ? i + size : i));
    sizes = [indices[0]];
    for (let i = 1; i < indices.length; i++)
      sizes.push(indices[i] - indices[i - 1]);
    sizes.push(size - indices[indices.length - 1]);
  }
  // Split in groups of up to 8 outputs, as the transpose rule turns into a
  // Concatenate primitive that has limited input arguments.
  const results: Array[] = [];
  for (let i = 0; i < sizes.length; i += 7) {
    if (i === sizes.length) {
      results.push(a);
      break;
    } else if (i + 8 >= sizes.length) {
      results.push(...(core.split(a, axis, sizes.slice(i)) as Array[]));
      break;
    } else {
      const groupSizes = [
        ...sizes.slice(i, i + 7),
        sizes.slice(i + 7).reduce((x, y) => x + y, 0),
      ];
      const outs = core.split(a, axis, groupSizes) as Array[];
      results.push(...outs.slice(0, -1));
      a = outs[outs.length - 1];
    }
  }
  return results;
}

/**
 * Join a sequence of arrays along an existing axis.
 *
 * The arrays must have the same shape, except in the dimension corresponding to
 * `axis` (the first, by default).
 *
 * No scalars can be passed to this function, as the axis is then ambiguous.
 */
export function concatenate(xs: Array[], axis: number = 0): Array {
  if (xs.length === 0)
    throw new Error("Need at least one array to concatenate");
  const shapes = xs.map(shape);
  axis = checkAxis(axis, shapes[0].length);
  for (let i = 1; i < shapes.length; i++) {
    if (
      shapes[i].length !== shapes[0].length ||
      !shapes[i].every((d, j) => j === axis || d === shapes[0][j])
    ) {
      throw new Error(
        `Cannot concatenate arrays ${xs[0].aval} and ${xs[i].aval} along axis ${axis}`,
      );
    }
  }
  // Concatenate the arrays in groups of 8 to avoid possibly exceeding the
  // `maxArgs` of the backend.
  let result = xs[0];
  for (let i = 1; i < xs.length; i += 7) {
    const group = xs.slice(i, i + 7);
    result = core.concatenate([result, ...group], axis) as Array;
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
export function stack(xs: ArrayLike[], axis: number = 0): Array {
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

export { transpose as permuteDims };

/** Interchange two axes of an array. */
export function swapaxes(a: ArrayLike, axis1: number, axis2: number): Array {
  a = fudgeArray(a);
  axis1 = checkAxis(axis1, a.ndim);
  axis2 = checkAxis(axis2, a.ndim);
  if (axis1 === axis2) return a;
  const perm = range(a.ndim);
  perm[axis1] = axis2;
  perm[axis2] = axis1;
  return transpose(a, perm);
}

/** Transpose the last two dimensions of an array. */
export function matrixTranspose(a: ArrayLike): Array {
  if (ndim(a) < 2)
    throw new Error(`matrixTranspose: input array must be at least 2D`);
  return moveaxis(a, -1, -2);
}

/** Return a 1-D flattened array containing the elements of the input. */
export function ravel(a: ArrayLike): Array {
  return fudgeArray(a).ravel();
}

/** Remove one or more length-1 axes from an array. */
export function squeeze(a: ArrayLike, axis: core.Axis = null): Array {
  const as = shape(a);
  if (axis === null) {
    axis = range(as.length).filter((i) => as[i] === 1);
  } else if (typeof axis === "number") {
    axis = [axis];
  }
  axis = axis.map((a) => checkAxis(a, as.length));
  for (const a of axis) {
    if (as[a] !== 1) throw new Error("Cannot squeeze axis with size != 1");
  }
  const newShape = as.filter((_, i) => !axis.includes(i));
  return reshape(a, newShape);
}

/**
 * Expand the shape of an array by inserting new axes of length 1.
 *
 * @param a - Input array.
 * @param axis - Position(s) in the expanded axes where the new axis (or axes)
 *   is placed. Can be a single integer or an array of integers.
 * @returns Array with the number of dimensions increased.
 *
 * @example
 * ```ts
 * const x = np.array([1, 2]);
 * np.expandDims(x, 0); // Shape [1, 2]
 * np.expandDims(x, 1); // Shape [2, 1]
 * np.expandDims(x, [0, 2]); // Shape [1, 2, 1]
 * ```
 */
export function expandDims(a: ArrayLike, axis: number | number[]): Array {
  const as = shape(a);
  axis = typeof axis === "number" ? [axis] : axis;
  axis = normalizeAxis(axis, as.length + axis.length);

  const newShape: number[] = [];
  let srcIdx = 0;
  for (let i = 0; i < as.length + axis.length; i++) {
    if (axis.includes(i)) {
      newShape.push(1);
    } else {
      newShape.push(as[srcIdx++]);
    }
  }
  return reshape(a, newShape);
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
  if (!Number.isInteger(k)) throw new Error(`k must be an integer, got ${k}`);
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
    throw new Error("numpy.diag only supports 1D and 2D arrays");
  }
}

/** Calculate the sum of the diagonal of an array along the given axes. */
export function trace(a: ArrayLike, offset = 0, axis1 = 0, axis2 = 1): Array {
  return diagonal(a, offset, axis1, axis2).sum(-1);
}

/**
 * Return a sorted copy of an array.
 *
 * The array is sorted along a specified axis (the last by default). This may be
 * an unstable sort, and it dispatches to device-specific implementation.
 */
export function sort(a: ArrayLike, axis: number = -1): Array {
  return fudgeArray(a).sort(axis);
}

/**
 * Return indices that would sort an array. This may be an unstable sorting
 * algorithm; it need not preserve order of indices in ties.
 *
 * Returns an array of `int32` indices.
 *
 * The array is sorted along a specified axis (the last by default).
 */
export function argsort(a: ArrayLike, axis: number = -1): Array {
  return fudgeArray(a).argsort(axis);
}

/**
 * Take elements from an array along an axis.
 *
 * This is equivalent to advanced indexing with integer indices over that
 * numbered axis. By default, the flattened array is used.
 */
export function take(
  a: ArrayLike,
  indices: ArrayLike,
  axis: number | null = null,
): Array {
  if (axis === null) {
    a = ravel(a);
    axis = 0;
  }
  axis = checkAxis(axis, ndim(a));
  return core.gather(a, [indices], [axis], axis) as Array;
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
    if (isNaN(xData[i]) !== isNaN(yData[i])) {
      return false;
    }
    if (Math.abs(xData[i] - yData[i]) > atol + rtol * Math.abs(yData[i])) {
      return false;
    }
  }
  return true;
}

/** Matrix product of two arrays. */
export function matmul(x: ArrayLike, y: ArrayLike) {
  if (ndim(x) === 0 || ndim(y) === 0) {
    throw new Error("matmul: x and y must be at least 1D");
  }
  ((x = x as Array), (y = y as Array));
  if (y.ndim === 1) {
    // Matrix-vector product
    return core.dot(x, y) as Array;
  }

  // Otherwise, we multiply x: [..., N, K] and y: [..., K, M]
  const numBatchDims = Math.min(Math.max(x.ndim, 2), y.ndim) - 2;
  return lax.dot(x, y, {
    lhsContractingDims: [-1],
    rhsContractingDims: [-2],
    lhsBatchDims: range(-2 - numBatchDims, -2),
    rhsBatchDims: range(-2 - numBatchDims, -2),
  });
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
  return lax.dot(x, y, {
    lhsContractingDims: [-1],
    rhsContractingDims: [-2],
  });
}

/**
 * Compute the tensor dot product of two N-dimensional arrays.
 *
 * The behavior is determined by `axes`. If an integer `k`, sum over the last
 * `k` axes of x and the first `k` axes of y. If a tuple, then the first array
 * corresponds to the axes of x and the second to the axes of y.
 */
export function tensordot(
  x: ArrayLike,
  y: ArrayLike,
  axes: number | [number[], number[]] = 2,
): Array {
  x = fudgeArray(x);
  y = fudgeArray(y);
  if (typeof axes === "number") axes = [range(-axes, 0), range(axes)];
  return lax.dot(x, y, {
    lhsContractingDims: axes[0],
    rhsContractingDims: axes[1],
  });
}

/**
 * Einstein summation with string subscripts.
 *
 * @example
 * ```ts
 * import { numpy as np } from "@jax-js/jax";
 *
 * const a = np.ones([2, 3]);
 * const b = np.ones([3]);
 * np.einsum("ij,j", a, b); // Shape [2]
 * ```
 */
export function einsum(subscripts: string, ...operands: ArrayLike[]): Array;

/**
 * Einstein summation alternating between arrays and numeric indices.
 *
 * @example
 * ```ts
 * import { numpy as np } from "@jax-js/jax";
 *
 * const a = np.ones([2, 3]);
 * const b = np.ones([3]);
 * np.einsum(a, [0, 1], b, [1]); // Shape [2]
 * ```
 */
export function einsum(...args: (ArrayLike | number[])[]): Array;

/**
 * Einstein summation.
 *
 * This is a general API for performing tensor reductions, products,
 * transpositions, and traces using Einstein notation for referring to named
 * axes. See the docs for `numpy.einsum()` for more information.
 *
 * <https://numpy.org/doc/stable/reference/generated/numpy.einsum.html>
 *
 * The full einsum API is implemented, including implicit and explicit output
 * indices, ellipsis broadcasting, Unicode subscripts, and an optimal path
 * ordering algorithm. It lowers to one or more calls to:
 *
 * - `jax.lax.dot()`
 * - `jax.numpy.diagonal()`
 * - `jax.numpy.sum()`
 * - `jax.numpy.transpose()`
 */
export function einsum(...args: any[]): Array {
  if (args.length === 0)
    throw new Error("einsum: must provide at least one argument");
  let input: EinsumInput;
  let operands: Array[] = [];
  if (typeof args[0] === "string") {
    // Subscripts mode with remaining arguments as arrays.
    operands = args.slice(1).map(fudgeArray);
    input = parseEinsumExpression(
      args[0],
      operands.map((x) => x.shape),
    );
  } else {
    // Alternating arrays and index arrays.
    const n = args.length >> 1;
    const shapes: number[][] = [];
    const lhsIndices: number[][] = [];
    for (let i = 0; i < n; i++) {
      operands.push(fudgeArray(args[2 * i]));
      shapes.push(operands[i].shape);
      lhsIndices.push(args[2 * i + 1] as number[]);
    }
    let rhsIndex: number[];
    if (args.length % 2 === 1) {
      rhsIndex = args[2 * n] as number[];
    } else {
      // Implicit output indices: all indices that appear only once in the
      // inputs, in the order they appear.
      const indexCount: number[] = [];
      for (const i of lhsIndices.flat()) {
        indexCount[i] = (indexCount[i] ?? 0) + 1;
      }
      rhsIndex = [...indexCount.entries()]
        .filter(([_, count]) => count === 1)
        .map(([i, _]) => i);
    }
    input = { lhsIndices, rhsIndex, shapes };
  }

  const path = computeEinsumPath(input);
  if (DEBUG >= 3)
    console.info(`einsum: computed path: ${path.approximateFlops} flops`);

  // These will be mutated as we do each step of the einsum.
  const indexUsageCounts: number[] = [];
  for (const idx of [...input.lhsIndices.flat(), ...input.rhsIndex]) {
    indexUsageCounts[idx] = (indexUsageCounts[idx] ?? 0) + 1;
  }
  const indices = [...input.lhsIndices];

  // Process a tensor by reducing all indices not used in any other expressions,
  // and also taking diagonals for duplicate indices.
  const processSingleTensor = (
    ar: Array,
    index: number[],
    doNotReduce: number[] = [],
  ): [Array, number[]] => {
    index = index.slice();
    // 1. Take diagonal to remove all duplicated indices.
    diag: while (true) {
      for (let i = 0; i < index.length; i++) {
        const idx = index[i];
        const j = index.indexOf(idx, i + 1);
        if (j !== -1) {
          ar = diagonal(ar, 0, i, j);
          index.splice(j, 1);
          index.splice(i, 1);
          index.push(idx);
          continue diag;
        }
      }
      break;
    }
    // 2. Reduce all indices that are not in `doNotReduce` and have zero usage.
    for (let i = index.length - 1; i >= 0; i--) {
      const idx = index[i];
      if (indexUsageCounts[idx] === 0 && !doNotReduce.includes(idx)) {
        ar = sum(ar, i);
        index.splice(i, 1);
      }
    }
    return [ar, index];
  };

  for (const [i, j] of path.path) {
    let indexReduced: number[] = [];
    const indexGroup: number[] = [];
    for (const idx of [...indices[i], ...indices[j]]) {
      if (!indexGroup.includes(idx)) indexGroup.push(idx);
      // If the index is not in the output and isn't in any other inputs,
      // we can consider it reduced here.
      if (--indexUsageCounts[idx] === 0) indexReduced.push(idx);
    }
    const [a, aidx] = processSingleTensor(operands[i], indices[i], indices[j]);
    const [b, bidx] = processSingleTensor(operands[j], indices[j], indices[i]);
    // At this point, aidx and bidx are both unique index sets that need to be
    // in the output. We can use dot to combine them along reduction dims.
    indexReduced = indexReduced.filter((idx) => aidx.includes(idx));
    const indexBatch = aidx.filter(
      (idx) => bidx.includes(idx) && !indexReduced.includes(idx),
    );
    const result = lax.dot(a, b, {
      lhsContractingDims: indexReduced.map((idx) => aidx.indexOf(idx)),
      rhsContractingDims: indexReduced.map((idx) => bidx.indexOf(idx)),
      lhsBatchDims: indexBatch.map((idx) => aidx.indexOf(idx)),
      rhsBatchDims: indexBatch.map((idx) => bidx.indexOf(idx)),
    });
    operands.push(result);
    indices.push([
      ...indexBatch,
      ...aidx.filter((idx) => !bidx.includes(idx)),
      ...bidx.filter((idx) => !aidx.includes(idx)),
    ]);
    for (const idx of indices[indices.length - 1]) ++indexUsageCounts[idx];
  }

  // Special case: Einsum with just one operand produces an empty path, but we
  // may still need to do any reductions or traces. Process one more time.
  for (const idx of indices[operands.length - 1]) --indexUsageCounts[idx];
  const [ar, index] = processSingleTensor(
    operands[operands.length - 1],
    indices[operands.length - 1],
  );

  // At this point, `index` _must_ be some permutation of `rhsIndex`.
  const finalPerm = input.rhsIndex.map((idx) => index.indexOf(idx));
  return ar.transpose(finalPerm);
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
  return lax.dot(fudgeArray(x), fudgeArray(y), {
    lhsContractingDims: [-1],
    rhsContractingDims: [-1],
  });
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

function _convImpl(name: string, x: Array, y: Array, mode: string): Array {
  if (x.ndim !== 1 || y.ndim !== 1) {
    throw new Error(
      `${name}: both inputs must be 1D arrays, got ${x.ndim}D and ${y.ndim}D`,
    );
  }
  let flipOutput = false; // for correlate: output[k] = sum(x[i + k] * y[i]) = sum(y[i + (-k)] * x[i])
  if (x.shape[0] < y.shape[0]) {
    [x, y] = [y, x];
    if (name === "correlate") flipOutput = true;
  }
  if (name === "convolve") y = flip(y);

  let padding: lax.PaddingType;
  if (mode === "valid") padding = "VALID";
  else if (mode === "same") padding = "SAME_LOWER";
  else if (mode === "full") padding = [[y.shape[0] - 1, y.shape[0] - 1]];
  else {
    throw new Error(
      `${name}: invalid mode ${mode}, expected "full", "same", or "valid"`,
    );
  }

  const z = lax
    .conv(x.slice(null, null), y.slice(null, null), [1], padding)
    .slice(0, 0);
  return flipOutput ? flip(z) : z;
}

/** Convolution of two one-dimensional arrays. */
export function convolve(
  x: Array,
  y: Array,
  mode: "full" | "same" | "valid" = "full",
): Array {
  return _convImpl("convolve", x, y, mode);
}

/** Correlation of two one dimensional arrays. */
export function correlate(
  x: Array,
  y: Array,
  mode: "full" | "same" | "valid" = "valid",
): Array {
  return _convImpl("correlate", x, y, mode);
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
      throw new Error(
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

export { absolute as abs };

/** Return an element-wise indication of sign of the input. */
export function sign(x: ArrayLike): Array {
  x = fudgeArray(x);
  return where(notEqual(x.ref, 0), where(less(x, 0), -1, 1), 0);
}

/** @function Return element-wise positive values of the input (no-op). */
export const positive = fudgeArray;

/**
 * Return the Hamming window of size M, a taper with a weighted cosine bell.
 *
 * `w(n) = 0.54 - 0.46 * cos(2πn/(M-1))` for `0 <= n <= M-1`.
 */
export function hamming(M: number): Array {
  return cos(linspace(0, 2 * Math.PI, M))
    .mul(-0.46)
    .add(0.54);
}

/**
 * Return the Hann window of size M, a taper with a weighted cosine bell.
 *
 * `w(n) = 0.5 - 0.5 * cos(2πn/(M-1))` for `0 <= n <= M-1`.
 */
export function hann(M: number): Array {
  return cos(linspace(0, 2 * Math.PI, M))
    .mul(-0.5)
    .add(0.5);
}

/**
 * @function
 * Compute the Heaviside step function. It is defined piecewise:
 * - `heaviside(x1, x2) = 0` for `x1 < 0`,
 * - `heaviside(x1, x2) = x2` for `x1 == 0`,
 * - `heaviside(x1, x2) = 1` for `x1 > 0`.
 */
export const heaviside = jit(function heaviside(x1: Array, x2: Array) {
  return where(less(x1.ref, 0), 0, where(equal(x1, 0), x2, 1));
});

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

/**
 * @function
 * Return the normalized sinc function.
 *
 * The sinc function is defined as `sin(πx) / (πx)` for `x != 0`, and `1` for `x = 0`.
 * This is the normalized sinc function commonly used in signal processing.
 *
 * **Note:** JVP is not supported at x=0 due to discontinuous derivative. This
 * requires a custom JVP rule to handle properly (see JAX implementation).
 */
export const sinc = jit(function sinc(x: Array): Array {
  const pix = x.ref.mul(Math.PI);
  // sinc(0) = 1, otherwise sin(πx) / (πx)
  return where(equal(x, 0), 1, sin(pix.ref).div(pix));
});

/** Element-wise inverse cosine function (inverse of cos). */
export function acos(x: ArrayLike): Array {
  return subtract(pi / 2, asin(x));
}

/**
 * @function
 * Return element-wise hypotenuse for the given legs of a right triangle.
 *
 * In the original NumPy/JAX implementation, this function is more numerically
 * stable than `sqrt(x1**2 + x2**2)`. We don't currently implement those
 * stability improvements.
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

export { asin as arcsin, acos as arccos, atan as arctan, atan2 as arctan2 };

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
  if (!isFloatDtype(x.dtype) && !isFloatDtype(y.dtype)) {
    x = x.astype(DType.Float32);
    y = y.astype(DType.Float32);
  }
  return x.div(y);
}

export { trueDivide as divide };

/**
 * Return the largest integer smaller or equal to the division of the inputs.
 *
 * The result is always rounded towards negative infinity.
 *
 * For floating-point inputs, this is equivalent to `floor(x / y)`.
 * For integer inputs, we use `(x - remainder(x, y)) / y` to handle
 * negative values correctly (note: may overflow near int32 boundaries).
 *
 * @param x - Dividend array.
 * @param y - Divisor array.
 * @returns Element-wise floor division of x by y.
 */
export function floorDivide(x: ArrayLike, y: ArrayLike): Array {
  x = fudgeArray(x);
  y = fudgeArray(y);
  if (isFloatDtype(x.dtype) || isFloatDtype(y.dtype)) {
    // For floats, floor(x / y) works correctly
    return floor(trueDivide(x, y));
  }
  // For integers, use (x - remainder(x, y)) / y to round toward -infinity
  // This avoids the truncation behavior of idiv which rounds toward zero
  return subtract(x, remainder(x.ref, y.ref)).div(y) as Array;
}

/**
 * @function
 * Calculate element-wise floating-point modulo operation.
 */
export const fmod = jit(function fmod(x: Array, y: Array): Array {
  return x.ref.sub(y.ref.mul(core.idiv(x, y) as Array));
});

/**
 * @function
 * Calculate element-wise remainder of the division (matches sign of y).
 */
export const remainder = jit(function remainder(x: Array, y: Array): Array {
  // The `Mod` primitive matches the sign of x, following JS rounding rules.
  // This function must match the sign of y instead.
  return core.mod(core.mod(x, y.ref).add(y.ref), y) as Array;
});

/**
 * Return element-wise quotient and remainder simultaneously.
 *
 * Equivalent to `[floorDivide(x, y), remainder(x, y)]`.
 *
 * @param x - Dividend array.
 * @param y - Divisor array.
 * @returns Tuple of [quotient, remainder].
 */
export function divmod(x: ArrayLike, y: ArrayLike): [Array, Array] {
  const xArr = fudgeArray(x);
  const yArr = fudgeArray(y);
  // floorDivide and remainder both consume their inputs, so use .ref for the first call
  return [floorDivide(xArr.ref, yArr.ref), remainder(xArr, yArr)];
}

/** Round input to the nearest integer towards zero. */
export function trunc(x: ArrayLike): Array {
  return core.idiv(x, 1) as Array; // Integer division truncates the decimal part.
}

/**
 * Compute `x1 * 2 ** x2` as a standard multiplication and exponentiation.
 *
 * This is the inverse of `frexp()`.
 */
export function ldexp(x1: ArrayLike, x2: ArrayLike): Array {
  return multiply(x1, exp2(x2));
}

/**
 * Decompose floating-point values into mantissa and two's exponent.
 *
 * The mantissa is returned in the range `(-1, 1)` with magnitude `>= 0.5` if
 * `x != 0`, and the exponent is an integer such that
 * `x = mantissa * 2**exponent`.
 */
export function frexp(x: ArrayLike): [Array, Array] {
  x = fudgeArray(x);
  const absx = absolute(x.ref);
  const exponent = where(
    equal(x.ref, 0),
    0,
    floor(log2(absx)).add(1).astype(DType.Int32),
  );
  const mantissa = x.div(exp2(exponent.ref.astype(x.dtype)));
  return [mantissa, exponent];
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
  // TODO: This is a little bit inefficient since we need to handle negative
  // numbers to integer exponents, should eventually move it into the backend.
  const x2i = trunc(x2.ref);
  // Should be NaN if x1 < 0 and x2 is non-integer.
  const shouldBeNaN = multiply(x2.ref.notEqual(x2i.ref), x1.ref.less(0));
  // If x2 is odd integer, result sign matches x1, else it's positive.
  const resultSign = where(
    core.mod(x2i, 2).notEqual(0) as Array,
    where(x1.ref.less(0), -1, 1),
    1,
  );
  return where(
    shouldBeNaN,
    nan,
    exp(log(absolute(x1)).mul(x2)).mul(resultSign),
  );
});

export { power as pow };

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

export { arcsinh as asinh, arccosh as acosh, arctanh as atanh };

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

/** Estimate the sample covariance of a set of variables. */
export function cov(
  x: ArrayLike,
  y: ArrayLike | null = null,
  { rowvar = true }: { rowvar?: boolean } = {},
): Array {
  // x should shape (M, N) or (N,), representing N observations of M variables.
  x = fudgeArray(x);
  if (x.ndim === 1) x = x.reshape([1, x.shape[0]]);
  // optional set of additional observations, concatenated to m
  if (y !== null) {
    y = fudgeArray(y);
    if (y.ndim === 1) y = y.reshape([1, y.shape[0]]);
    x = vstack([x, y]);
  }
  if (!rowvar) x = x.transpose();
  const [_M, N] = x.shape;
  x = x.ref.sub(x.mean(1, { keepdims: true })); // Center variables
  return dot(x.ref, x.transpose()).div(N - 1); // [M, M]
}

/** Compute the Pearson correlation coefficients (in range `[-1, 1]`). */
export function corrcoef(x: ArrayLike, y?: ArrayLike): Array {
  const c = cov(x, y);
  const variances = diag(c.ref);
  const norm = sqrt(outer(variances.ref, variances));
  return c.div(norm);
}

/** Test element-wise for positive or negative infinity, return bool array. */
export function isinf(x: ArrayLike): Array {
  x = fudgeArray(x);
  return isFloatDtype(x.dtype)
    ? x.ref.equal(Infinity).add(x.equal(-Infinity))
    : fullLike(x, false);
}

/** Test element-wise for NaN (Not a Number). */
export function isnan(x: ArrayLike): Array {
  x = fudgeArray(x);
  return isFloatDtype(x.dtype) ? x.ref.notEqual(x) : fullLike(x, false);
}

/** Test element-wise for negative infinity, return bool array. */
export function isneginf(x: ArrayLike): Array {
  x = fudgeArray(x);
  return isFloatDtype(x.dtype) ? x.equal(-Infinity) : fullLike(x, false);
}

/** Test element-wise for positive infinity, return bool array. */
export function isposinf(x: ArrayLike): Array {
  x = fudgeArray(x);
  return isFloatDtype(x.dtype) ? x.equal(Infinity) : fullLike(x, false);
}

/**
 * Replace NaN and infinite entries in an array.
 *
 * By default, NaNs are replaced with `0.0`, and infinities are are substituted
 * with the corresponding maximum or minimum finite values.
 */
export function nanToNum(
  x: ArrayLike,
  {
    nan = 0.0,
    posinf = null,
    neginf = null,
  }: {
    nan?: ArrayLike;
    posinf?: ArrayLike | null;
    neginf?: ArrayLike | null;
  } = {},
): Array {
  x = fudgeArray(x);
  x = where(isnan(x.ref), nan, x);
  posinf ??= isFloatDtype(x.dtype) ? finfo(x.dtype).max : iinfo(x.dtype).max;
  neginf ??= isFloatDtype(x.dtype) ? finfo(x.dtype).min : iinfo(x.dtype).min;
  x = where(isposinf(x.ref), posinf, x);
  x = where(isneginf(x.ref), neginf, x);
  return x;
}

/**
 * @function
 * Test element-wise for finite values (not infinity or NaN).
 */
export const isfinite = jit(function isfinite(x: Array): Array {
  if (!isFloatDtype(x.dtype)) return fullLike(x, true);
  return isnan(x.ref).add(isinf(x)).notEqual(true);
});
