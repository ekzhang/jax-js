// Common functions for neural network libraries, mirroring `jax.nn` in JAX.

import { isFloatDtype } from "../alu";
import {
  absolute,
  Array,
  ArrayLike,
  clip,
  exp,
  less,
  log,
  max,
  maximum,
  negative,
  onesLike,
  reciprocal,
  sqrt,
  square,
  squeeze,
  tanh,
  where,
  zerosLike,
} from "./numpy";
import { eye, fudgeArray } from "../frontend/array";
import {
  type Axis,
  erfc,
  type ReduceOpts,
  shrink,
  stopGradient,
} from "../frontend/core";
import { jit } from "../frontend/jaxpr";
import { Pair } from "../shape";
import { checkAxis, normalizeAxis } from "../utils";

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
 * @function
 * Sigmoid-weighted Linear Unit (SiLU) activation function, also known as
 * Swish, computed element-wise:
 * `silu(x) = x * sigmoid(x) = x / (1 + exp(-x))`.
 *
 * `swish()` and `silu()` are both aliases for the same function.
 *
 * Reference: https://en.wikipedia.org/wiki/Swish_function
 */
export const silu = jit(function silu(x: Array) {
  return x.ref.mul(sigmoid(x));
});

/**
 * @function
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

/**
 * @function
 * Identity activation function. Returns the argument unmodified.
 */
export const identity = fudgeArray;

/** Leaky rectified linear (ReLU) activation function */
export function leakyRelu(
  x: ArrayLike,
  negativeSlope: ArrayLike = 0.01,
): Array {
  x = fudgeArray(x);
  return where(less(x.ref, 0), x.ref.mul(negativeSlope), x);
}

/**
 * Exponential linear unit activation function.
 *
 * Computes the element-wise function:
 * `elu(x) = x > 0 ? x : alpha * (exp(x) - 1)`
 */
export function elu(x: ArrayLike, alpha: ArrayLike = 1.0): Array {
  x = fudgeArray(x);
  return where(less(x.ref, 0), exp(x.ref).sub(1).mul(alpha), x);
}

/**
 * Continuously-differentiable exponential linear unit activation function.
 *
 * Computes the element-wise function:
 * `celu(x) = x > 0 ? x : alpha * (exp(x/alpha) - 1)`
 */
export function celu(x: ArrayLike, alpha: ArrayLike = 1.0): Array {
  x = fudgeArray(x);
  return where(less(x.ref, 0), exp(x.ref.div(alpha)).sub(1).mul(alpha), x);
}

/**
 * @function
 * Gaussion error linear unit (GELU) activation function.
 *
 * This is computed element-wise. There are two variants depending on whether
 * `approximate` is set (default true):
 *
 * - Approximate: `gelu(x) ~= x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
 * - Exact: `gelu(x) = x * 0.5 * erfc(-x / sqrt(2))`
 *
 * Reference: https://ml-explore.github.io/mlx/build/html/python/nn/_autosummary_functions/mlx.nn.gelu_approx.html
 */
export const gelu = jit(
  function gelu(x: Array, opts?: { approximate?: boolean }): Array {
    if (opts?.approximate ?? true) {
      const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
      return x.ref
        .mul(0.5)
        .mul(
          tanh(
            x.ref.mul(x.ref.mul(x).mul(0.044715).add(1)).mul(SQRT_2_OVER_PI),
          ).add(1),
        );
    } else {
      return x.ref.mul(0.5).mul(erfc(negative(x.ref.mul(Math.SQRT1_2))));
    }
  },
  { staticArgnums: [1] },
);

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
 * Squareplus activation function.
 *
 * Computes the element-wise function:
 * `squareplus(x) = 0.5 * (x + sqrt(x^2 + b))`
 */
export function squareplus(x: ArrayLike, b: ArrayLike = 4.0): Array {
  x = fudgeArray(x);
  return x.ref.add(sqrt(square(x).add(b))).mul(0.5);
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
export function softmax(x: ArrayLike, axis: Axis = -1): Array {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  if (axis.length === 0) {
    return onesLike(x); // scalar case, return ones
  }

  const xMax = max(x.ref, axis, { keepdims: true });
  const unnormalized = exp(x.sub(stopGradient(xMax)));
  return unnormalized.ref.div(unnormalized.sum(axis, { keepdims: true }));
}

/**
 * Log-Softmax function.
 *
 * Computes the logarithm of the `softmax` function, which rescales elements to
 * the range [-infinity, 0).
 *
 * If `axis` is not specified, it defaults to the last axis.
 */
export function logSoftmax(x: ArrayLike, axis: Axis = -1): Array {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  if (axis.length === 0) {
    return zerosLike(x); // scalar case, return log(1)
  }

  const xMax = max(x.ref, axis, { keepdims: true }); // keep dims
  const shifted = x.sub(stopGradient(xMax));
  const shiftedLogsumexp = log(exp(shifted.ref).sum(axis, { keepdims: true }));
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
export function logsumexp(
  x: ArrayLike,
  axis: Axis = null,
  opts?: ReduceOpts,
): Array {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  if (axis.length === 0) return x;

  const xMax = stopGradient(max(x.ref, axis, { keepdims: true })) as Array;
  const shifted = x.sub(xMax.ref);
  const result = xMax.add(log(exp(shifted).sum(axis, { keepdims: true })));
  return opts?.keepdims ? result : squeeze(result, axis);
}

/** Log-mean-exp reduction, like `jax.nn.logsumexp()` but subtracts `log(n)`. */
export function logmeanexp(
  x: ArrayLike,
  axis: Axis = null,
  opts?: ReduceOpts,
): Array {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  if (axis.length === 0) return x;
  const n = axis.reduce((acc, a) => acc * x.shape[a], 1);
  return logsumexp(x, axis, opts).sub(Math.log(n));
}

/**
 * Standardizes input to zero mean and unit variance.
 *
 * By default, this is computed over the last axis. You can pass in a different
 * axis, or `null` to standardize over all elements.
 *
 * Epsilon is added to denominator, it defaults to `1e-5` for stability.
 */
export function standardize(
  x: ArrayLike,
  axis: Axis = -1,
  opts: {
    mean?: ArrayLike;
    variance?: ArrayLike;
    epsilon?: ArrayLike;
  } = {},
) {
  x = fudgeArray(x);
  axis = normalizeAxis(axis, x.ndim);
  if (axis.length === 0) return x;

  const mu =
    opts.mean !== undefined
      ? fudgeArray(opts.mean)
      : x.ref.mean(axis, { keepdims: true });

  // Like JAX, we'll use the Var[X] = E[X^2] - (E[X])^2 formula for this one.
  // It's supposed to be better in the case of neural network activations.
  const sigma2 =
    opts.variance !== undefined
      ? fudgeArray(opts.variance)
      : square(x.ref).mean(axis, { keepdims: true }).sub(square(mu.ref));

  return x.sub(mu).div(sqrt(sigma2.add(opts.epsilon ?? 1e-5)));
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
  if (isFloatDtype(x.dtype)) {
    throw new TypeError(`oneHot expects integers, got ${x.dtype}`);
  }
  return eye(numClasses, undefined, { device: x.device }).slice(x);
}
