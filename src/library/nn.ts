// Common functions for neural network libraries, mirroring `jax.nn` in JAX.

import { DType, isFloatDtype } from "../alu";
import {
  absolute,
  add,
  arange,
  Array,
  ArrayLike,
  clip,
  einsum,
  exp,
  expandDims,
  expm1,
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
  tile,
  where,
  zerosLike,
} from "./numpy";
import { eye, fudgeArray, tri } from "../frontend/array";
import {
  type Axis,
  erfc,
  type ReduceOpts,
  shrink,
  stopGradient,
} from "../frontend/core";
import { jit } from "../frontend/jaxpr";
import { Pair } from "../shape";
import { checkAxis, deepEqual, normalizeAxis } from "../utils";

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
 * @function
 * Sparse plus function:
 *
 * - When `x <= -1`: `0`
 * - When `-1 < x < 1`: `(x+1)**2 / 4`
 * - When `x >= 1`: `x`
 */
export const sparsePlus = jit((x: Array): Array => {
  return where(
    x.ref.lessEqual(-1),
    0,
    where(x.ref.less(1), square(x.ref.add(1)).mul(0.25), x),
  );
});

/**
 * @function
 * Sparse sigmoid activation function.
 *
 * - When `x <= -1`: `0`
 * - When `-1 < x < 1`: `(x + 1) / 2`
 * - When `x >= 1`: `1`
 */
export const sparseSigmoid = jit((x: Array): Array => {
  return clip(x.add(1).mul(0.5), 0, 1);
});

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

export { silu as swish };

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

/** Hard sigmoid activation function: `relu6(x+3)/6`. */
export function hardSigmoid(x: ArrayLike): Array {
  return relu6(add(x, 3)).mul(1 / 6);
}

/** Hard SiLU (swish) activation function: `x * hardSigmoid(x)`. */
export function hardSilu(x: ArrayLike): Array {
  x = fudgeArray(x);
  return x.ref.mul(hardSigmoid(x));
}

export { hardSilu as hardSwish };

/** Hard tanh activation function: `clip(x, -1, 1)`. */
export function hardTanh(x: ArrayLike): Array {
  return clip(x, -1, 1);
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
 * Scaled exponential linear unit activation.
 *
 * Computes the element-wise function:
 * `selu(x) = lambda * (x > 0 ? x : alpha * (exp(x) - 1))`
 *
 * Where `alpha = 1.6732632423543772` and `lambda = 1.0507009873554805`.
 */
export const selu = jit(function selu(x: Array) {
  const alpha = 1.6732632423543772;
  const lambda = 1.0507009873554805;
  return where(x.ref.less(0), expm1(x.ref).mul(alpha), x).mul(lambda);
});

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
      return x.ref.mul(0.5).mul(erfc(negative(x.mul(Math.SQRT1_2))));
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

/**
 * Scaled dot product attention (SDPA).
 *
 * Computes `softmax((Q @ K^T) / sqrt(d) + bias) @ V`, where `Q` is the query,
 * `K` is the key, `V` is the value, and `d` is the dimensionality of each key
 * and query vector.
 *
 * Multi-query attention is applied when input `key` and `value` tensors have
 * fewer heads than `query`.
 *
 * We use the following uppercase letters to denote array shapes:
 * - `B` = batch size
 * - `S` = length of key/value sequences (source)
 * - `L` = length of query sequences
 * - `N` = number of attention heads
 * - `H` = dimensionality of each attention head
 * - `K` = number of key/value heads (for grouped-query attention)
 *
 * The batch size `B` may be omitted, which is equivalent to `B = 1`. In this
 * case it must be omitted from all inputs.
 *
 * @param query - Query array; shape `[B, L, N, H]`
 * @param key - Key array; shape `[B, S, K, H]`
 * @param value - Value array; same shape as `key`
 * @param opts.bias - Optional bias to add to the attention logits; shape
 *   `[B, N, L, S]` or broadcastable to it.
 * @param opts.mask - Optional mask to apply to the attention logits; should be
 *   a boolean array broadcastable to `[B, N, L, S]`, where `true` indicates
 *   the element should take part in attention.
 * @param opts.scale - Scaling factor override, default is `1 / sqrt(H)`.
 * @param opts.isCausal - If true, applies a casual mask.
 * @param opts.querySeqLengths - Optional sequence lengths for the queries;
 *   shape `(B,)`. Taken from the beginning of the tensor.
 * @param opts.keyValueSeqLengths - Optional sequence lengths for the keys and
 *   values; shape `(B,)`. Taken from the beginning of the tensor.
 * @param opts.localWindowSize - If specified, applies a local attention window
 *   of the given size. Can be a single number or a tuple `[left, right]`.
 *
 * @returns The result of the attention operation; shape is the same as query
 *   `[B, L, N, H]`, or `[L, N, H]` if `B` is omitted.
 */
export function dotProductAttention(
  query: ArrayLike,
  key: ArrayLike,
  value: ArrayLike,
  opts: {
    bias?: ArrayLike;
    mask?: ArrayLike;
    scale?: number;
    isCausal?: boolean;
    querySeqLengths?: ArrayLike;
    keyValueSeqLengths?: ArrayLike;
    localWindowSize?: number | [number, number];
  } = {},
): Array {
  query = fudgeArray(query);
  key = fudgeArray(key);
  value = fudgeArray(value);

  if (
    (query.ndim !== 3 && query.ndim !== 4) ||
    query.ndim !== key.ndim ||
    query.ndim !== value.ndim
  )
    throw new Error(
      `dotProductAttention: expected all tensors to have rank 3 or 4, ` +
        `got Q=${query.aval}, K=${key.aval}, V=${value.aval}`,
    );
  if (!deepEqual(key.shape, value.shape))
    throw new Error(
      `dotProductAttention: key and value shapes must match, ` +
        `got K=${key.shape}, V=${value.shape}`,
    );

  const isRank3 = query.ndim === 3;
  if (isRank3) {
    query = expandDims(query, 0);
    key = expandDims(key, 0);
    value = expandDims(value, 0);
  }

  const [B, L, N, H] = query.shape;
  if (key.shape[0] !== B || key.shape[3] !== H)
    throw new Error(
      `dotProductAttention: query and key shapes mismatch, ` +
        `got Q=${query.aval}, K=${key.aval}`,
    );

  const S = key.shape[1];
  const K = key.shape[2];

  if (N < K || (N != K && N % K !== 0))
    throw new Error(
      `dotProductAttention: number of query heads N=${N} must be ` +
        `divisible by number of key/value heads K=${K} for GQA`,
    );
  const G = N / K; // number of query groups
  key = tile(key, [1, 1, G, 1]);
  value = tile(value, [1, 1, G, 1]);

  const scale = opts.scale ?? 1 / Math.sqrt(H);
  let scores = einsum("BLNH,BSNH->BNLS", query, key).mul(scale);
  if (opts.bias !== undefined) {
    scores = scores.add(opts.bias);
  }
  if (opts.mask !== undefined) {
    scores = where(opts.mask, scores, -Infinity);
  }
  if (opts.isCausal) {
    // Causal mask: position i can only attend to positions j <= i
    // tri(L, S) creates a lower triangular boolean mask of shape [L, S]
    const causalMask = tri(L, S, 0, { dtype: DType.Bool });
    scores = where(causalMask, scores, -Infinity);
  }
  if (opts.localWindowSize !== undefined) {
    const [before, after] =
      typeof opts.localWindowSize === "number"
        ? [opts.localWindowSize, opts.localWindowSize]
        : opts.localWindowSize;
    if (
      before < 0 ||
      after < 0 ||
      !Number.isInteger(before) ||
      !Number.isInteger(after)
    ) {
      throw new Error(
        `dotProductAttention: localWindowSize values must be non-negative, ` +
          `got ${opts.localWindowSize}`,
      );
    }
    const localMask = tri(L, S, after, { dtype: DType.Bool }).mul(
      tri(L, S, -before - 1, { dtype: DType.Bool }).notEqual(true),
    );
    scores = where(localMask, scores, -Infinity);
  }
  if (opts.querySeqLengths !== undefined) {
    const sl = expandDims(opts.querySeqLengths, [-1, -2, -3]); // [B, 1, 1, 1]
    scores = where(arange(L).reshape([1, 1, L, 1]).less(sl), scores, -Infinity);
  }
  if (opts.keyValueSeqLengths !== undefined) {
    const sl = expandDims(opts.keyValueSeqLengths, [-1, -2, -3]); // [B, 1, 1, 1]
    scores = where(arange(S).reshape([1, 1, 1, S]).less(sl), scores, -Infinity);
  }
  const attn = softmax(scores, -1); // BNLS
  const out = einsum("BNLS,BSNH->BLNH", attn, value);
  return isRank3 ? out.reshape([L, N, H]) : out;
}
