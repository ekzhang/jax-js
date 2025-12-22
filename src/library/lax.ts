// Mirrors the `jax.lax` module in JAX.
//
// Unlike in JAX, this does not actually underpin `jax.numpy` as a more "core"
// set of operations, as they both build open the same foundations.

import { Array, ArrayLike } from "../frontend/array";
import * as core from "../frontend/core";
import { bind1, Primitive } from "../frontend/core";
import { vmap } from "../frontend/vmap";
import { checkAxis, deepEqual, prod, range, rep, zipn } from "../utils";

/**
 * Dimension numbers for general `dot()` primitive.
 *
 * Contracting dimensions act as a tensor contraction (reduction) along the
 * given axis. They must be the same size in both operands. Batch dimensions
 * are treated as vectorized, leading batch dimensions.
 *
 * The return value has a shape where the first dimensions are shared batch
 * dimensions, followed by `lhs` non-contracting dimensions, followed by
 * `rhs` non-contracting dimensions.
 */
export type DotDimensionNumbers = {
  lhsContractingDims?: number[];
  rhsContractingDims?: number[];
  lhsBatchDims?: number[];
  rhsBatchDims?: number[];
};

/**
 * General dot product/contraction operator.
 *
 * Prefer higher-level functions like `jax.numpy.dot()`, `jax.numpy.matmul()`,
 * `jax.numpy.tensordot(), and `jax.numpy.einsum()` where possible.
 */
export function dot(
  lhs: Array,
  rhs: Array,
  {
    lhsContractingDims: lc = [],
    rhsContractingDims: rc = [],
    lhsBatchDims: lb = [],
    rhsBatchDims: rb = [],
  }: DotDimensionNumbers = {},
): Array {
  // First do input validation, helps with debugging.
  if (lc.length !== rc.length) {
    throw new Error(
      `dot: contracting dims lengths mismatch, got ${JSON.stringify(lc)} and ${JSON.stringify(rc)}`,
    );
  } else if (lb.length !== rb.length) {
    throw new Error(
      `dot: batch dims lengths mismatch, got ${JSON.stringify(lb)} and ${JSON.stringify(rb)}`,
    );
  }
  lc = lc.map((a) => checkAxis(a, lhs.ndim));
  rc = rc.map((a) => checkAxis(a, rhs.ndim));
  lb = lb.map((a) => checkAxis(a, lhs.ndim));
  rb = rb.map((a) => checkAxis(a, rhs.ndim));
  if (lc.some((a) => lb.includes(a))) {
    throw new Error(
      `dot: lhs contracting dims ${JSON.stringify(lc)} ` +
        `overlap with batch dims ${JSON.stringify(lb)}`,
    );
  } else if (rc.some((a) => rb.includes(a))) {
    throw new Error(
      `dot: rhs contracting dims ${JSON.stringify(rc)} ` +
        `overlap with batch dims ${JSON.stringify(rb)}`,
    );
  }

  // Compute "free" dimensions: output shape is [...{lb/rb}, ...lf, ...rf].
  const lf = range(lhs.ndim).filter((a) => !lc.includes(a) && !lb.includes(a));
  const rf = range(rhs.ndim).filter((a) => !rc.includes(a) && !rb.includes(a));
  const lhs2 = lhs.transpose([...lb, ...lf, ...lc]);
  const rhs2 = rhs.transpose([...rb, ...rf, ...rc]);

  if (lc.length === 0) {
    // There is no contraction to perform, just do a product (not `dot`).
    return core.mul(
      lhs2.reshape([
        ...lb.map((a) => lhs.shape[a]),
        ...lf.map((a) => lhs.shape[a]),
        ...rep(rf.length, 1),
      ]),
      rhs2.reshape([
        ...rb.map((a) => rhs.shape[a]),
        ...rep(lf.length, 1),
        ...rf.map((a) => rhs.shape[a]),
      ]),
    ) as Array;
  }

  // Otherwise, we need to do a `dot` contraction.
  const dotShapeX = lc.map((a) => lhs.shape[a]);
  const dotShapeY = rc.map((a) => rhs.shape[a]);
  if (!deepEqual(dotShapeX, dotShapeY)) {
    throw new Error(
      `dot: shapes not aligned along contracting dims:` +
        ` ${JSON.stringify(dotShapeX)} != ${JSON.stringify(dotShapeY)}`,
    );
  }
  return core.dot(
    lhs2.reshape([
      ...lb.map((a) => lhs.shape[a]),
      ...lf.map((a) => lhs.shape[a]),
      ...rep(rf.length, 1),
      prod(dotShapeX),
    ]),
    rhs2.reshape([
      ...rb.map((a) => rhs.shape[a]),
      ...rep(lf.length, 1),
      ...rf.map((a) => rhs.shape[a]),
      prod(dotShapeY),
    ]),
  ) as Array;
}

export type PaddingType = "VALID" | "SAME" | "SAME_LOWER" | [number, number][];

function padtypeToPads(
  inShape: number[],
  filterShape: number[],
  strides: number[],
  dilation: number[],
  padding: string,
): [number, number][] {
  const padType = padding.toUpperCase();
  switch (padType) {
    case "VALID":
      return rep<[number, number]>(inShape.length, [0, 0]);
    case "SAME":
    case "SAME_LOWER": {
      const outShape = inShape.map((size, i) => Math.ceil(size / strides[i]));
      const padSizes = zipn(
        outShape,
        strides,
        filterShape,
        dilation,
        inShape,
      ).map(([o, s, k, d, i]) =>
        Math.max(0, (o - 1) * s + 1 + (k - 1) * d - i),
      );
      if (padType === "SAME") {
        return padSizes.map((size) => [size >> 1, size - (size >> 1)]);
      } else {
        return padSizes.map((size) => [size - (size >> 1), size >> 1]);
      }
    }
    default:
      throw new Error(`Unknown padding type: ${padType}`);
  }
}

/**
 * General n-dimensional convolution operator, with optional dilation.
 *
 * The semantics of this operation mimic the `jax.lax.conv_general_dilated`
 * function in JAX, which wraps XLA's general convolution operator.
 *
 * Grouped convolutions are not supported right now.
 */
export function convGeneralDilated(
  lhs: Array,
  rhs: Array,
  windowStrides: number[],
  padding: PaddingType,
  {
    lhsDilation,
    rhsDilation,
  }: {
    lhsDilation?: number[];
    rhsDilation?: number[];
  } = {},
): Array {
  if (lhs.ndim < 2) throw new Error("lhs must have at least 2 dimensions");
  if (rhs.ndim < 2) throw new Error("rhs must have at least 2 dimensions");
  if (typeof padding === "string") {
    if (lhsDilation?.some((d) => d !== 1)) {
      throw new Error(
        "String padding is not supported for transposed convolutions",
      );
    }
    padding = padtypeToPads(
      lhs.shape.slice(2),
      rhs.shape.slice(2),
      windowStrides,
      rhsDilation ?? rep(rhs.ndim - 2, 1),
      padding,
    );
  }
  return core.conv(lhs, rhs, {
    strides: windowStrides,
    padding,
    lhsDilation,
    rhsDilation,
  }) as Array;
}

/** Convenience wrapper around `convGeneralDilated`. */
export function convWithGeneralPadding(
  lhs: Array,
  rhs: Array,
  windowStrides: number[],
  padding: PaddingType,
  lhsDilation?: number[],
  rhsDilation?: number[],
): Array {
  return convGeneralDilated(lhs, rhs, windowStrides, padding, {
    lhsDilation,
    rhsDilation,
  });
}

/** Convenience wrapper around `convGeneralDilated`. */
export function conv(
  lhs: Array,
  rhs: Array,
  windowStrides: number[],
  padding: PaddingType,
): Array {
  return convGeneralDilated(lhs, rhs, windowStrides, padding);
}

/** Reduce a computation over padded windows. */
export function reduceWindow(
  operand: Array,
  computation: (x: Array) => Array,
  windowDimensions: number[],
  windowStrides?: number[],
): Array {
  if (operand.ndim < windowDimensions.length) {
    throw new Error(
      `Operand dimensions ${operand.ndim} < window ${windowDimensions.length}`,
    );
  }
  if (!windowStrides) windowStrides = rep(windowDimensions.length, 1);

  for (let i = 0; i < operand.ndim; i++) {
    // Vmap the computation over any pre-pooled dimensions.
    computation = vmap(computation, 0) as any;
  }
  return computation(
    bind1(Primitive.Pool, [operand], {
      window: windowDimensions,
      strides: windowStrides,
    }) as Array,
  );
}

/** The error function: `erf(x) = 2/sqrt(pi) * int[0..x] exp(-t^2) dt`. */
export function erf(x: ArrayLike): Array {
  return core.erf(x) as Array;
}

/**
 * The complementary error function: `erfc(x) = 1 - erf(x)`.
 *
 * This function is more accurate than `1 - erf(x)` for large values of `x`,
 * where `erf(x)` is very close to 1.
 */
export function erfc(x: ArrayLike): Array {
  return core.erfc(x) as Array;
}

/**
 * Stops gradient computation.
 *
 * Behaves as the identity function but prevents the flow of gradients during
 * forward or reverse-mode automatic differentiation.
 */
export function stopGradient(x: ArrayLike): Array {
  return core.stopGradient(x) as Array;
}
