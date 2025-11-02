// Common functions for neural network libraries, mirroring `jax.nn` in JAX.

import { eye, fudgeArray, ones, zeros } from "./frontend/array";
import { broadcast, shrink, stopGradient } from "./frontend/core";
import { jit } from "./frontend/jaxpr";
import {
  absolute,
  Array,
  ArrayLike,
  clip,
  DType,
  exp,
  less,
  log,
  max,
  maximum,
  negative,
  reciprocal,
  tanh,
  where,
} from "./numpy";
import { Pair } from "./shape";
import { checkAxis, range } from "./utils";

/**
 * Rectified Linear Unit (ReLU) activation function:
 * `relu(x) = max(x, 0)`.
 */
export function relu(x: ArrayLike): Array {
  return maximum(x, 0);
}

/**
 * Rectified Linear Unit 6 (ReLU6) activation function:
 * `relu6(x) = min(max(x, 0), 6)`.
 */
export function relu6(x: ArrayLike): Array {
  return clip(x, 0, 6);
}

/**
 * Sigmoid activation function, computed element-wise:
 * `sigmoid(x) = 1 / (1 + exp(-x))`.
 *
 * Reference: https://en.wikipedia.org/wiki/Sigmoid_function
 */
export function sigmoid(x: ArrayLike): Array {
  return reciprocal(exp(negative(x)).add(1));
}

/**
 * Softplus activation function:
 * `softplus(x) = log(1 + exp(x))`.
 *
 * Reference: https://en.wikipedia.org/wiki/Softplus
 */
export function softplus(x: ArrayLike): Array {
  return log(exp(x).add(1));
}

/**
 * Soft-sign activation function, computed element-wise:
 * `softsign(x) = x / (|x| + 1)`.
 */
export function softSign(x: ArrayLike): Array {
  x = fudgeArray(x);
  return x.ref.div(absolute(x).add(1));
}

/**
 * Sigmoid-weighted Linear Unit (SiLU) activation function, also known as
 * Swish, computed element-wise:
 * `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`.
 *
 * `swish()` and `silu()` are both aliases for the same function.
 *
 * Reference: https://en.wikipedia.org/wiki/Swish_function
 */
export function silu(x: ArrayLike): Array {
  x = fudgeArray(x);
  return x.ref.mul(sigmoid(x));
}

/**
 * Sigmoid-weighted Linear Unit (SiLU) activation function, also known as
 * Swish, computed element-wise:
 * `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`.
 *
 * `swish()` and `silu()` are both aliases for the same function.
 *
 * Reference: https://en.wikipedia.org/wiki/Swish_function
 */
export const swish = silu;

/**
 * Log-sigmoid activation function, computed element-wise:
 * `log_sigmoid(x) = log(sigmoid(x)) = -log(1 + exp(-x))`.
 */
export function logSigmoid(x: ArrayLike): Array {
  return negative(softplus(negative(x)));
}

/** Identity activation function. Returns the argument unmodified. */
export const identity = fudgeArray;

/** Leaky rectified linear (ReLU) activation function */
export function leakyRelu(x: ArrayLike, negativeSlope: number = 0.01): Array {
  x = fudgeArray(x);
  return where(less(x.ref, 0), x.ref.mul(negativeSlope), x);
}

/**
 * Exponential linear unit activation function.
 *
 * Computes the element-wise function:
 * `elu(x) = x > 0 ? x : alpha * (exp(x) - 1)`
 */
export function elu(x: ArrayLike, alpha: number = 1.0): Array {
  x = fudgeArray(x);
  return where(less(x.ref, 0), exp(x.ref).sub(1).mul(alpha), x);
}

/**
 * Continuously-differentiable exponential linear unit activation function.
 *
 * Computes the element-wise function:
 * `celu(x) = x > 0 ? x : alpha * (exp(x/alpha) - 1)`
 */
export function celu(x: ArrayLike, alpha: number = 1.0): Array {
  x = fudgeArray(x);
  return where(less(x.ref, 0), exp(x.ref.div(alpha)).sub(1).mul(alpha), x);
}

/**
 * Gaussion error linear unit (GELU) activation function.
 *
 * This is computed element-wise. Currently jax-js does not support the erf() or
 * gelu() functions exactly as primitives, so an approximation is used:
 * `gelu(x) ~= x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
 *
 * Reference: https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu_approx.html
 *
 * This will be improved in the future.
 */
export const gelu = jit((x: Array): Array => {
  const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
  return x.ref
    .mul(0.5)
    .mul(
      tanh(
        x.ref.mul(x.ref.mul(x).mul(0.044715).add(1)).mul(SQRT_2_OVER_PI),
      ).add(1),
    );
});

/**
 * Gated linear unit (GLU) activation function.
 *
 * Splits the `axis` dimension of the input into two halves, a and b, then
 * computes `a * sigmoid(b)`.
 */
export function glu(x: ArrayLike, axis: number = -1): Array {
  x = fudgeArray(x);
  axis = checkAxis(axis, x.ndim);
  const size = x.shape[axis];
  if (size % 2 !== 0) {
    throw new Error(
      `glu: axis ${axis} of shape (${x.shape}) does not have even length`,
    );
  }
  const slice = x.shape.map<Pair>((a) => [0, a]);
  const a = shrink(x.ref, slice.toSpliced(axis, 1, [0, size / 2])) as Array;
  const b = shrink(x, slice.toSpliced(axis, 1, [size / 2, size])) as Array;
  return a.mul(sigmoid(b));
}

/**
 * Mish activation function.
 *
 * Computes the element-wise function:
 * `mish(x) = x * tanh(softplus(x))`
 */
export function mish(x: ArrayLike): Array {
  x = fudgeArray(x);
  return x.ref.mul(tanh(softplus(x)));
}

/**
 * Softmax function. Computes the function which rescales elements to the range
 * [0, 1] such that the elements along `axis` sum to 1.
 *
 * If `axis` is not specified, it defaults to the last axis.
 *
 * Reference: https://en.wikipedia.org/wiki/Softmax_function
 */
export function softmax(x: ArrayLike, axis?: number | number[]): Array {
  x = fudgeArray(x);
  if (axis === undefined) {
    axis = x.ndim ? [x.ndim - 1] : []; // default to last axis
  } else if (typeof axis === "number") {
    axis = [axis];
  }

  if (axis.length === 0) {
    x.dispose();
    return ones(x.shape); // scalar case, return ones
  }

  const xMax = max(x.ref, axis, { keepDims: true });
  const unnormalized = exp(x.sub(stopGradient(xMax)));
  return unnormalized.ref.div(unnormalized.sum(axis, { keepDims: true }));
}

/**
 * Log-Softmax function.
 *
 * Computes the logarithm of the `softmax` function, which rescales elements to
 * the range [-infinity, 0).
 *
 * If `axis` is not specified, it defaults to the last axis.
 */
export function logSoftmax(x: ArrayLike, axis?: number | number[]): Array {
  x = fudgeArray(x);
  if (axis === undefined) {
    axis = x.ndim ? [x.ndim - 1] : []; // default to last axis
  } else if (typeof axis === "number") {
    axis = [axis];
  }

  if (axis.length === 0) {
    x.dispose();
    return zeros(x.shape); // scalar case, return log(1)
  }

  const xMax = max(x.ref, axis, { keepDims: true }); // keep dims
  const shifted = x.sub(stopGradient(xMax));
  const shiftedLogsumexp = log(exp(shifted.ref).sum(axis, { keepDims: true }));
  return shifted.sub(shiftedLogsumexp);
}

/**
 * Log-sum-exp reduction. Also a multivariate version of `softplus`.
 *
 * If no axis is specified, the reduction is performed over all elements. This
 * convention differs from `jax.nn.logSoftmax()`.
 *
 * Reference: https://en.wikipedia.org/wiki/LogSumExp
 */
export function logsumexp(x: ArrayLike, axis?: number | number[]): Array {
  x = fudgeArray(x);
  if (axis === undefined) {
    axis = range(x.ndim); // default to all axes
  } else if (typeof axis === "number") {
    axis = [axis];
  }

  if (axis.length === 0) return x;

  const xMax = stopGradient(max(x.ref, axis)) as Array;
  const xMaxDims = broadcast(xMax.ref, x.shape, axis); // keep dims
  const shifted = x.sub(xMaxDims);
  return xMax.add(log(exp(shifted).sum(axis)));
}

/**
 * One-hot encodes the given indices.
 *
 * Each index in the integer input `x` is encoded as a vector of zeros of length
 * `numClasses`, with a 1 at the index position specified by its value.
 *
 * ```js
 * import { nn, numpy as np } from '@jax-js/jax';
 *
 * nn.oneHot(np.array([1, 1, 2], { dtype: np.int32 }), 3);
 * // Output:
 * // [[0, 1, 0],
 * //  [0, 1, 0],
 * //  [0, 0, 1]]
 * ```
 */
export function oneHot(x: Array, numClasses: number): Array {
  if (x.dtype !== DType.Int32) {
    throw new TypeError(`oneHot expects integers, got ${x.dtype}`);
  }
  return eye(numClasses, undefined, { device: x.device }).slice(x);
}
