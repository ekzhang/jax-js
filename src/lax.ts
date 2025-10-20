// Mirrors the `jax.lax` module in JAX.
//
// Unlike in JAX, this does not actually underpin `jax.numpy` as a more "core"
// set of operations, as they both build open the same foundations.

import { Array } from "./frontend/array";
import { bind1, conv as convPrimitive, Primitive } from "./frontend/core";
import { vmap } from "./frontend/vmap";
import { rep, zipn } from "./utils";

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
  return convPrimitive(lhs, rhs, {
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
