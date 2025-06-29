import { DType } from "./alu";
import {
  arange,
  array,
  Array,
  type ArrayLike,
  eye,
  fudgeArray,
  full,
  identity,
  linspace,
  ones,
  scalar,
  zeros,
} from "./frontend/array";
import * as core from "./frontend/core";
import { jit } from "./frontend/jaxpr";
import * as vmapModule from "./frontend/vmap";
import { deepEqual, prod, range, rep } from "./utils";

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
export const bool = DType.Bool;
export const complex64 = DType.Complex64;

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
export const sum = core.reduceSum as (
  x: ArrayLike,
  axis?: number | number[],
) => Array;
export const moveaxis = vmapModule.moveaxis as (
  x: ArrayLike,
  src: number,
  dst: number,
) => Array;

/** Return the number of dimensions of an array. */
export const ndim = core.ndim as (x: ArrayLike) => number;

/** Return the shape of an array. */
export const shape = core.getShape as (x: ArrayLike) => number[];

/** Return the number of elements in an array, optionally along an axis. */
export function size(a: ArrayLike, axis?: number): number {
  const s = shape(a);
  return axis === undefined ? prod(s) : s[axis];
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
      throw new TypeError(
        `flip: axis ${axis[i]} out of bounds for array of ${nd} dimensions`,
      );
    }
    if (axis[i] < 0) axis[i] += nd; // convert negative to positive
    if (seen.has(axis[i])) {
      throw new TypeError(`flip: duplicate axis ${axis[i]} in axis list`);
    }
    seen.add(axis[i]);
  }
  return core.flip(x, axis) as Array;
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
 * 3D or higher, compute diagonals along the two given axes.
 *
 * This returns a view over the existing array.
 */
export function diagonal(
  a: ArrayLike,
  offset?: number,
  axis1?: number,
  axis2?: number,
): Array {
  return fudgeArray(a).diagonal(offset, axis1, axis2);
}

/** Transposes a matrix or stack of matrices `x` (swap last two axes). */
export function matrixTranspose(x: ArrayLike): Array {
  const ar = fudgeArray(x);
  if (ar.ndim < 2)
    throw new TypeError("matrixTranspose only supports 2D+ arrays");
  return ar.transpose([...range(ar.ndim - 2), ar.ndim - 1, ar.ndim - 2]);
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
    const ret = where(eye(n).equal(1), a, 0);
    // TODO: pad() is unimplemented at this layer
    if (k !== 0) throw new Error("diag() for 1D arrays only for k=0");
    return ret;
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
  const { rtol = 1e-5, atol = 1e-8 } = options ?? {};

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
export const matmul = jit(function matmul(x: Array, y: Array) {
  if (x.ndim === 0 || y.ndim === 0) {
    throw new TypeError("matmul: x and y must be at least 1D");
  }
  if (y.ndim === 1) {
    // Matrix-vector product
    return x.mul(y).sum(x.ndim - 1);
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

  return x.mul(y).sum(Math.max(x.ndim, y.ndim) - 1);
});

/** Dot product of two arrays. */
export const dot = jit(function dot(x: Array, y: Array) {
  if (x.ndim === 0 || y.ndim === 0) {
    // Standard, scalar multiplication
    return multiply(x, y);
  }
  if (y.ndim === 1) {
    // Matrix-vector product
    return x.mul(y).sum(x.ndim - 1);
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

  return x.mul(y).sum(x.ndim - 1);
});

/** Vector dot product of two arrays. */
export const vecdot = jit(function vecdot(x: Array, y: Array) {
  return x.mul(y).sum(Math.max(x.ndim, y.ndim) - 1);
});

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
