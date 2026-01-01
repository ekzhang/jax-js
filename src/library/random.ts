// Port of the `jax.random` module.

import {
  absolute,
  array,
  Array,
  ArrayLike,
  cos,
  DType,
  log,
  log1p,
  negative,
  sign,
  sqrt,
  stack,
  tan,
} from "./numpy";
import { fudgeArray } from "../frontend/array";
import { bitcast, randomBits } from "../frontend/core";
import { jit } from "../frontend/jaxpr";

function validateKeyShape(key: Array): number[] {
  if (key.ndim === 0) {
    throw new Error("Key must have at least one dimension.");
  }
  if (key.shape[key.shape.length - 1] !== 2) {
    throw new Error(
      `Invalid key shape: ${key.shape}. Expected last dimension to be 2.`,
    );
  }
  return key.shape.slice(0, -1);
}

/** Create a pseudo-random number generator (PRNG) key from 32-bit integer seed. */
export function key(seed: number): Array {
  seed = seed >>> 0;
  // To match JAX, put the 32-bit seed into a 64-bit key in this way.
  return array([0, seed], { dtype: DType.Uint32 });
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

  const keyShape = validateKeyShape(key);
  const k0 = key.ref.slice(...keyShape.map(() => null), 0);
  const k1 = key.slice(...keyShape.map(() => null), 1);
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
  const keyShape = validateKeyShape(key);
  return randomBits(
    key.ref.slice(...keyShape.map(() => null), 0),
    key.slice(...keyShape.map(() => null), 1),
    shape,
  ) as Array;
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
