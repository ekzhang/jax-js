// Port of the `jax.random` module.

import { fudgeArray } from "../frontend/array";
import * as core from "../frontend/core";
import { bitcast, randomBits } from "../frontend/core";
import { jit } from "../frontend/jaxpr";
import { checkAxis, deepEqual, generalBroadcast } from "../utils";
import { topK } from "./lax";
import {
  absolute,
  argmax,
  argsort,
  array,
  Array,
  ArrayLike,
  broadcastShapes,
  cos,
  DType,
  einsum,
  exp,
  floor,
  log,
  log1p,
  negative,
  sign,
  sqrt,
  stack,
  tan,
  where,
} from "./numpy";
import { cholesky } from "./numpy-linalg";

const JsArray = globalThis.Array;

function validateKeyShape(key: Array, scalar = false): number[] {
  if (key.ndim === 0) {
    throw new Error("Key must have at least one dimension.");
  }
  if (key.shape[key.shape.length - 1] !== 2) {
    throw new Error(
      `Invalid key shape: ${key.shape}. Expected last dimension to be 2.`,
    );
  }
  if (scalar && key.shape.length > 1) {
    throw new Error(
      `Expected a single PRNG key, but got a batch of keys with shape` +
        ` ${JSON.stringify(key.shape)} - use jax.vmap for batching.`,
    );
  }
  return key.shape.slice(0, -1);
}

function getK01(key: Array): [Array, Array] {
  const keyShape = validateKeyShape(key, true);
  let [k0, k1] = core.split(key, -1, [1, 1]) as [Array, Array];
  k0 = k0.reshape(keyShape); // Remove the last dimension of size 1
  k1 = k1.reshape(keyShape);
  return [k0, k1];
}

/** Create a pseudo-random number generator (PRNG) key from 32-bit integer seed. */
export function key(seed: ArrayLike): Array {
  seed = array(seed, { dtype: DType.Uint32 });
  if (seed.ndim !== 0) {
    throw new Error(
      `key: seed must be a scalar integer, but got shape ${seed.shape}` +
        ` - use jax.vmap for batching.`,
    );
  }
  // To match JAX, put the 32-bit seed into a 64-bit key like `[0, seed]`.
  const key = stack([0, seed]);
  // HACK: Ensure the key is realized, so it doesn't generate a bunch of kernels
  // specialized to different constant key values.
  if (key instanceof Array) key._realizeSource();
  return key;
}

/** Splits a PRNG key into `num` new keys by adding a leading axis. */
export function split(key: Array, num: number | number[] = 2): Array {
  const shape = typeof num === "number" ? [num] : num;
  for (const len of shape) {
    if (len <= 0 || !Number.isInteger(len)) {
      throw new Error(
        `Invalid split length: ${len}. Must be a positive integer.`,
      );
    }
  }

  const [k0, k1] = getK01(key);
  return stack(
    // It's inefficient to calculate the PRNG key twice, then join the halves
    // together. But this allows us to avoid refactoring AluExp to support
    // multiple outputs, while remaining consistent with JAX.
    [
      randomBits(k0.ref, k1.ref, shape, 0) as Array,
      randomBits(k0, k1, shape, 1) as Array,
    ],
    -1,
  );
}

/** Sample uniform bits in the form of unsigned integers. */
export function bits(key: Array, shape: number[] = []): Array {
  const [k0, k1] = getK01(key);
  return randomBits(k0, k1, shape) as Array;
}

/**
 * @function
 * Sample uniform random values in [minval, maxval) with given shape.
 */
export const uniform = jit(
  function uniform(
    key: Array,
    shape: number[] = [],
    { minval = 0, maxval = 1 }: { minval?: number; maxval?: number } = {},
  ): Array {
    if (minval >= maxval) {
      throw new Error(`Invalid range: [${minval}, ${maxval}).`);
    }
    // Float32 has sign bit, 8 bits of exponent, and 23 bits of mantissa.
    const mantissa = bits(key, shape).div(
      array(1 << 9, { dtype: DType.Uint32, device: key.device }),
    );
    const float12 = mantissa.add(
      array(0x3f800000, { dtype: DType.Uint32, device: key.device }),
    ); // Add 1.0 in IEEE 754, now it's a float in [1, 2).
    const rand = bitcast(float12, DType.Float32).sub(1) as Array; // [0, 1) range
    if (minval === 0 && maxval === 1) {
      return rand;
    } else {
      return rand.mul(maxval - minval).add(minval);
    }
  },
  { staticArgnums: [1, 2] },
);

// Other distributions (alphabetical order).

/**
 * @function
 * Sample points uniformly from the Euclidean unit ball in `d` dimensions.
 *
 * Only the Euclidean `p=2` case is currently supported.
 */
export const ball = jit(
  function ball(
    key: Array,
    d: number,
    { p = 2, shape = [] }: { p?: number; shape?: number[] } = {},
  ): Array {
    if (!Number.isInteger(d) || d <= 0) {
      throw new Error(`ball: dimension must be a positive integer, got ${d}`);
    }
    if (p !== 2) {
      throw new Error("ball: only the Euclidean p=2 case is supported");
    }
    const [k1, k2] = split(key, 2);
    const z = normal(k1, [...shape, d]);
    const norm = sqrt(z.ref.mul(z.ref).sum(-1, { keepdims: true }));
    const radius = exp(log(uniform(k2, [...shape, 1])).mul(1 / d));
    return z.div(norm).mul(radius);
  },
  { staticArgnums: [1, 2] },
);

/**
 * Sample Bernoulli random variables with given mean (0,1 categorical).
 *
 * Returns a random Boolean array with the specified shape. `p` can be an array
 * and must be broadcastable to `shape`.
 */
export function bernoulli(
  key: Array,
  p: ArrayLike = 0.5,
  shape: number[] = [],
): Array {
  p = fudgeArray(p);
  return uniform(key, shape).less(p);
}

/**
 * @function
 * Sample random values from categorical distributions.
 *
 * Uses the Gumbel max trick for sampling with replacement, or the Gumbel top-k
 * trick for sampling without replacement.
 *
 * Note: Sampling without replacement currently uses argsort and slices the last
 * k elements. This should be replaced with a more efficient topK implementation.
 *
 * - `key` - PRNG key
 * - `logits` - Unnormalized log probabilities of the categorical distribution(s).
 *   `softmax(logits, axis)` gives the corresponding probabilities.
 * - `axis` - Axis along which logits belong to the same categorical distribution.
 * - `shape` - Result batch shape. Must be broadcast-compatible with
 *   `logits.shape` with `axis` removed. Default is `logits.shape` with `axis` removed.
 * - `replace` - If true (default), sample with replacement. If false, sample
 *   without replacement (each category can only be selected once per batch).
 * @returns A random array with int dtype and shape given by `shape` if provided,
 *   otherwise `logits.shape` with `axis` removed.
 */
export const categorical = jit(
  function categorical(
    key: Array,
    logits: ArrayLike,
    {
      axis = -1,
      shape,
      replace = true,
    }: {
      axis?: number;
      shape?: number[];
      replace?: boolean;
    } = {},
  ): Array {
    logits = fudgeArray(logits);
    axis = checkAxis(axis, logits.ndim);
    const numCategories = logits.shape[axis];
    const batchShape = logits.shape.toSpliced(axis, 1);

    if (shape === undefined) {
      shape = batchShape;
    } else {
      if (!deepEqual(generalBroadcast(shape, batchShape), shape)) {
        throw new Error(
          `Shape ${shape} is not broadcast-compatible with batch shape ${batchShape}.`,
        );
      }
    }

    const shapePrefix = shape.slice(0, shape.length - batchShape.length);

    if (replace) {
      // Gumbel-max trick: generate noise for full output shape + categories
      const noise = gumbel(key, [...shapePrefix, ...logits.shape]);
      return argmax(noise.add(logits), axis + shapePrefix.length);
    } else {
      // Gumbel top-k trick: add noise once, use topK to get k samples
      const k = shapePrefix.reduce((a, b) => a * b, 1);

      if (k > numCategories) {
        throw new Error(
          `Number of samples without replacement (${k}) cannot exceed ` +
            `number of categories (${numCategories}).`,
        );
      }

      const noise = gumbel(key, logits.shape);
      const [values, indices] = topK(noise.add(logits), k, axis);
      values.dispose();
      return indices.reshape(shape);
    }
  },
  { staticArgnums: [2] },
);

/**
 * @function
 * Sample from a Cauchy distribution with location 0 and scale 1.
 *
 * Uses inverse transform sampling: `x = tan(π * (u - 0.5))` where u ~ Uniform(0, 1).
 */
export const cauchy = jit(
  function cauchy(key: Array, shape: number[] = []): Array {
    const u = uniform(key, shape);
    // Inverse CDF of Cauchy: tan(π * (u - 0.5))
    return tan(u.sub(0.5).mul(Math.PI));
  },
  { staticArgnums: [1] },
);

/**
 * Sample from a population with optional replacement and optional probabilities.
 *
 * This implements the common JAX-compatible cases: integer populations and
 * array populations along `axis`. Probabilities `p`, if provided, are sampled
 * via `categorical(log(p))`.
 */
export function choice(
  key: Array,
  a: number | ArrayLike,
  {
    shape = [],
    replace = true,
    p,
    axis = 0,
  }: { shape?: number[]; replace?: boolean; p?: ArrayLike; axis?: number } = {},
): Array {
  let n: number;
  let values: Array | null = null;
  if (typeof a === "number") {
    if (!Number.isInteger(a) || a < 0) {
      throw new Error(`choice: a must be a non-negative integer, got ${a}`);
    }
    n = a;
  } else {
    values = fudgeArray(a);
    axis = checkAxis(axis, values.ndim);
    n = values.shape[axis];
  }

  let indices: Array;
  if (p !== undefined) {
    indices = categorical(key, log(p), { shape, replace });
  } else if (replace) {
    indices = randint(key, { minval: 0, maxval: n, shape });
  } else {
    const k = shape.reduce((acc, x) => acc * x, 1);
    if (k > n) {
      throw new Error(
        `Number of samples without replacement (${k}) cannot exceed population size (${n}).`,
      );
    }
    indices = permutation(key, n).slice([0, k]).reshape(shape);
  }

  if (values === null) return indices;
  const index: any[] = JsArray(axis).fill([]);
  index.push(indices);
  return values.slice(...index);
}

/**
 * @function
 * Sample double-sided Maxwell random values with the provided location and scale.
 */
export const doubleSidedMaxwell = jit(
  function doubleSidedMaxwell(
    key: Array,
    loc: ArrayLike,
    scale: ArrayLike,
    shape: number[] = [],
  ): Array {
    loc = fudgeArray(loc);
    scale = fudgeArray(scale);
    const [k1, k2] = split(key, 2);
    return rademacher(k1, { shape, dtype: DType.Float32 })
      .mul(maxwell(k2, shape))
      .mul(scale)
      .add(loc);
  },
  { staticArgnums: [3] },
);

/**
 * @function
 * Sample exponential random values according to `p(x) = exp(-x)`.
 */
export const exponential = jit(
  function exponential(key: Array, shape: number[] = []): Array {
    const u = uniform(key, shape);
    return negative(log1p(negative(u))) as Array; // log(1-u) to avoid log(0)
  },
  { staticArgnums: [1] },
);

/**
 * @function
 * Sample geometric random values: the number of trials until first success.
 */
export const geometric = jit(
  function geometric(
    key: Array,
    p: ArrayLike,
    {
      shape = [],
      dtype = DType.Int32,
    }: { shape?: number[]; dtype?: DType } = {},
  ): Array {
    p = fudgeArray(p);
    return floor(log1p(negative(uniform(key, shape))).div(log1p(negative(p))))
      .add(1)
      .astype(dtype);
  },
  { staticArgnums: [2] },
);

/**
 * @function
 * Sample from a Gumbel distribution with location 0 and scale 1.
 *
 * Uses inverse transform sampling: `x = -log(-log(u))` where u ~ Uniform(0, 1).
 */
export const gumbel = jit(
  function gumbel(key: Array, shape: number[] = []): Array {
    const u = uniform(key, shape);
    // Use -log(1-u) instead of -log(u) to avoid log(0) at u=0
    // Then the formula becomes -log(-log(1-u))
    return negative(log(negative(log1p(negative(u)))));
  },
  { staticArgnums: [1] },
);

/**
 * @function
 * Sample from a Laplace distribution with location 0 and scale 1.
 *
 * Uses inverse transform sampling: the CDF is `F(x) = 0.5 + 0.5 * sign(x) * (1 - exp(-|x|))`.
 * Inverting: `x = -sign(u - 0.5) * log(1 - 2 * |u - 0.5|)`.
 */
export const laplace = jit(
  function laplace(key: Array, shape: number[] = []): Array {
    const u = uniform(key, shape);
    // u - 0.5 is in [-0.5, 0.5)
    const centered = u.sub(0.5);
    const s = sign(centered.ref);
    // |u - 0.5| ranges from 0 to 0.5, so 2*|u-0.5| ranges from 0 to 1
    // We use log1p(-(2*|centered|)) = log(1 - 2*|centered|) to avoid log(0)
    // when centered is close to ±0.5
    const absVal = absolute(centered);
    return s.mul(log1p(absVal.mul(-2)).mul(-1));
  },
  { staticArgnums: [1] },
);

/**
 * @function
 * Sample from a logistic distribution with location 0 and scale 1.
 *
 * Uses inverse transform sampling: `x = log(u) - log(1-u)`.
 */
export const logistic = jit(
  function logistic(key: Array, shape: number[] = []): Array {
    const u = uniform(key, shape);
    return log(u.ref).sub(log1p(negative(u)));
  },
  { staticArgnums: [1] },
);

/**
 * @function
 * Sample log-normal random values: `exp(sigma * normal(key, shape))`.
 */
export const lognormal = jit(
  function lognormal(
    key: Array,
    sigma: ArrayLike = 1,
    shape: number[] = [],
  ): Array {
    sigma = fudgeArray(sigma);
    return exp(normal(key, shape).mul(sigma));
  },
  { staticArgnums: [2] },
);

/**
 * @function
 * Sample Maxwell-distributed random values.
 */
export const maxwell = jit(
  function maxwell(key: Array, shape: number[] = []): Array {
    const z = normal(key, [...shape, 3]);
    return sqrt(z.ref.mul(z).sum(-1));
  },
  { staticArgnums: [1] },
);

/**
 * @function
 * Sample multivariate normal random values with given mean and covariance.
 *
 * The values are returned with the given shape, along with the final dimension
 * used to represent the n-dimensional multivariate normal factors.
 *
 * This uses Cholesky decomposition on the covariance matrix.
 *
 * - `key` - PRNG key
 * - `mean` - Mean vector of shape `[..., n]`
 * - `cov` - Covariance of shape `[..., n, n]`, must be positive-definite
 * - `shape` - Result batch shape, must be broadcastable with
 *            `mean.shape[:-1]` and `cov.shape[:-2]`
 * @returns Random samples of shape `[...shape, n]`
 */
export const multivariateNormal = jit(
  function multivariateNormal(
    key: Array,
    mean: ArrayLike,
    cov: ArrayLike,
    shape: number[] = [],
  ): Array {
    mean = fudgeArray(mean);
    cov = fudgeArray(cov);
    const n = mean.shape[mean.ndim - 1];
    if (cov.shape[cov.ndim - 1] !== n || cov.shape[cov.ndim - 2] !== n) {
      throw new Error(
        `Invalid covariance shape: ${cov.shape}. Expected last two ` +
          `dimensions to be [${n}, ${n}].`,
      );
    }
    const outputShape = broadcastShapes(
      shape,
      mean.shape.slice(0, -1),
      cov.shape.slice(0, -2),
    ).concat(n);
    const L = cholesky(cov);
    const z = normal(key, outputShape);
    return einsum("...ij,...j->...i", L, z).add(mean);
  },
  { staticArgnums: [3] },
);

/**
 * @function
 * Sample random values according to `p(x) = 1/sqrt(2pi) * exp(-x^2/2)`.
 *
 * Unlike JAX, this uses the Box-Muller transform. JAX uses the erf_inv primitive instead and
 * directly inverts the CDF, but we don't have support for that yet. Outputs will not be
 * bitwise identical to JAX.
 */
export const normal = jit(
  function normal(key: Array, shape: number[] = []): Array {
    // Box-Muller transform:
    //   z0 = sqrt(-2 * log(u1)) * cos(2pi * u2)
    //   z1 = sqrt(-2 * log(u1)) * sin(2pi * u2)
    // We only use z0 for simplicity.
    const [k1, k2] = split(key, 2);
    const u1 = uniform(k1, shape);
    const u2 = uniform(k2, shape);
    const radius = sqrt(log1p(negative(u1)).mul(-2)); // taking 1-u1 to avoid log(0)
    const theta = u2.mul(2 * Math.PI);
    return radius.mul(cos(theta)) as Array;
  },
  { staticArgnums: [1] },
);

/**
 * @function
 * Sample from a Pareto distribution with shape parameter `b` and support [1, ∞).
 */
export const pareto = jit(
  function pareto(key: Array, b: ArrayLike, shape: number[] = []): Array {
    b = fudgeArray(b);
    return exp(exponential(key, shape).div(b));
  },
  { staticArgnums: [2] },
);

/**
 * Return a random permutation of an integer range or of an array along `axis`.
 */
export function permutation(
  key: Array,
  x: number | ArrayLike,
  axis: number = 0,
): Array {
  if (typeof x === "number") {
    if (!Number.isInteger(x) || x < 0) {
      throw new Error(
        `permutation: x must be a non-negative integer, got ${x}`,
      );
    }
    return argsort(uniform(key, [x])).astype(DType.Int32);
  }

  const arr = fudgeArray(x);
  axis = checkAxis(axis, arr.ndim);
  const perm = permutation(key, arr.shape[axis]);
  const index: any[] = JsArray(axis).fill([]);
  index.push(perm);
  return arr.slice(...index);
}

/**
 * @function
 * Sample Rademacher random values, uniformly from {-1, 1}.
 */
export const rademacher = jit(
  function rademacher(
    key: Array,
    {
      shape = [],
      dtype = DType.Int32,
    }: { shape?: number[]; dtype?: DType } = {},
  ): Array {
    if (dtype === DType.Uint32 || dtype === DType.Bool) {
      throw new Error(`rademacher: unsupported dtype ${dtype}`);
    }
    const one = array(1, { dtype, device: key.device });
    const minusOne = array(-1, { dtype, device: key.device });
    return where(bernoulli(key, 0.5, shape), one, minusOne);
  },
  { staticArgnums: [1] },
);

/**
 * @function
 * Sample integer values uniformly from `[minval, maxval)`.
 *
 * This uses modulo reduction of uniform 32-bit random bits. For ranges that do
 * not divide 2^32, this introduces a very small modulo bias.
 */
export const randint = jit(
  function randint(
    key: Array,
    {
      minval,
      maxval,
      shape = [],
      dtype = DType.Int32,
    }: { minval: number; maxval: number; shape?: number[]; dtype?: DType },
  ): Array {
    if (!Number.isInteger(minval) || !Number.isInteger(maxval)) {
      throw new Error("randint: minval and maxval must be integers");
    }
    if (minval >= maxval) {
      throw new Error(`Invalid range: [${minval}, ${maxval}).`);
    }
    if (dtype !== DType.Int32 && dtype !== DType.Uint32) {
      throw new Error(`randint: dtype must be int32 or uint32, got ${dtype}`);
    }
    if (dtype === DType.Uint32 && minval < 0) {
      throw new Error("randint: uint32 dtype requires minval >= 0");
    }
    const range = maxval - minval;
    return bits(key, shape).mod(range).astype(dtype).add(minval) as Array;
  },
  { staticArgnums: [1] },
);

/**
 * @function
 * Sample Rayleigh random values with the provided scale parameter.
 */
export const rayleigh = jit(
  function rayleigh(
    key: Array,
    scale: ArrayLike = 1,
    shape: number[] = [],
  ): Array {
    scale = fudgeArray(scale);
    return sqrt(exponential(key, shape).mul(2)).mul(scale);
  },
  { staticArgnums: [2] },
);

/**
 * @function
 * Sample triangular random values on `[left, right]` with the given mode.
 */
export const triangular = jit(
  function triangular(
    key: Array,
    left: ArrayLike,
    mode: ArrayLike,
    right: ArrayLike,
    shape: number[] = [],
  ): Array {
    left = fudgeArray(left);
    mode = fudgeArray(mode);
    right = fudgeArray(right);

    const u = uniform(key, shape);
    const width = right.ref.sub(left.ref);
    const leftSpan = mode.ref.sub(left.ref);
    const rightSpan = right.ref.sub(mode);
    const cutoff = leftSpan.ref.div(width.ref);
    const cond = u.ref.less(cutoff);
    const lower = left.add(sqrt(u.ref.mul(width.ref).mul(leftSpan)));
    const upper = right.sub(sqrt(negative(u).add(1).mul(width).mul(rightSpan)));
    return where(cond, lower, upper);
  },
  { staticArgnums: [4] },
);

/**
 * @function
 * Sample Weibull minimum random values.
 *
 * Uses `scale * exponential(key) ** (1 / concentration)`.
 */
export const weibullMin = jit(
  function weibullMin(
    key: Array,
    scale: ArrayLike,
    concentration: ArrayLike,
    shape: number[] = [],
  ): Array {
    scale = fudgeArray(scale);
    concentration = fudgeArray(concentration);
    return scale.mul(exp(log(exponential(key, shape)).div(concentration)));
  },
  { staticArgnums: [3] },
);
