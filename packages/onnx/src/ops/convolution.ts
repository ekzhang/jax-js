// Convolution operations.
//
// TODO: ConvTranspose (prompt_encoder_mask_decoder)
// TODO: Resize (image_encoder, vision_encoder)
// TODO: Pad (vision_encoder)

import { lax, numpy as np } from "@jax-js/jax";

const padsMapping: Record<string, lax.PaddingType> = {
  SAME_UPPER: "SAME",
  SAME_LOWER: "SAME_LOWER",
  VALID: "VALID",
};

export function Conv(
  [x, w]: np.Array[],
  {
    auto_pad: autoPad = "NOTSET",
    dilations,
    group = 1,
    kernel_shape: _kernelShape, // inferred from weights
    pads,
    strides,
  }: {
    auto_pad?: "NOTSET" | "SAME_LOWER" | "SAME_UPPER" | "VALID";
    dilations?: number[];
    group?: number;
    kernel_shape?: number[];
    pads?: number[];
    strides?: number[];
  },
) {
  const [_batchSize, channelsIn, ...xSpatial] = x.shape;
  const [_channelsOut, channelsInGrouped, ...wSpatial] = w.shape;
  if (channelsIn !== channelsInGrouped * group) {
    throw new Error(
      `Conv: input channels ${channelsIn} must match weight channels ${channelsInGrouped} x group ${group}`,
    );
  }
  if (xSpatial.length !== wSpatial.length) {
    throw new Error(
      `Conv: input spatial dims ${xSpatial.length} must match weight spatial dims ${wSpatial.length}`,
    );
  }
  const n = xSpatial.length;
  if (group !== -1) {
    throw new Error("Conv: grouped convolution not supported yet");
  }
  const output = lax.convGeneralDilated(
    x,
    w,
    strides ?? wSpatial.map(() => 1),
    padsMapping[autoPad] ??
      pads?.slice(0, n).map((p, i) => [p, pads[i + n]]) ??
      "VALID",
    { rhsDilation: dilations },
  );
  return [output];
}

// Pad a tensor with -Infinity along spatial dimensions (for max pooling).
function padWithNegInf(
  x: np.Array,
  pads: [number, number][],
): np.Array {
  // pads is for spatial dims only, we need to add batch and channel dims
  for (let i = 0; i < pads.length; i++) {
    const [padBefore, padAfter] = pads[i];
    const axis = i + 2; // Skip batch and channel dims
    if (padBefore > 0) {
      const beforeShape = [...x.shape];
      beforeShape[axis] = padBefore;
      const before = np.full(beforeShape, -Infinity, { dtype: x.dtype });
      x = np.concatenate([before, x], axis);
    }
    if (padAfter > 0) {
      const afterShape = [...x.shape];
      afterShape[axis] = padAfter;
      const after = np.full(afterShape, -Infinity, { dtype: x.dtype });
      x = np.concatenate([x, after], axis);
    }
  }
  return x;
}

export function MaxPool(
  [x]: np.Array[],
  {
    auto_pad: autoPad = "NOTSET",
    ceil_mode: ceilMode = 0,
    dilations,
    kernel_shape: kernelShape,
    pads,
    strides,
  }: {
    auto_pad?: "NOTSET" | "SAME_LOWER" | "SAME_UPPER" | "VALID";
    ceil_mode?: number;
    dilations?: number[];
    kernel_shape: number[];
    pads?: number[];
    strides?: number[];
  },
): np.Array[] {
  if (ceilMode) {
    throw new Error("MaxPool: ceil_mode=1 is not supported");
  }
  if (dilations && dilations.some((d) => d !== 1)) {
    throw new Error("MaxPool: dilations != 1 is not supported");
  }
  const n = kernelShape.length;
  const xSpatial = x.shape.slice(2);
  if (xSpatial.length !== n) {
    throw new Error(
      `MaxPool: input spatial dims ${xSpatial.length} must match kernel dims ${n}`,
    );
  }

  // Compute explicit padding
  let explicitPads: [number, number][];
  if (autoPad !== "NOTSET") {
    const effectiveStrides = strides ?? kernelShape.map(() => 1);
    const outShape = xSpatial.map((size, i) =>
      Math.ceil(size / effectiveStrides[i]),
    );
    const padSizes = outShape.map((o, i) => {
      const s = effectiveStrides[i];
      const k = kernelShape[i];
      const inSize = xSpatial[i];
      return Math.max(0, (o - 1) * s + k - inSize);
    });
    explicitPads =
      autoPad === "SAME_UPPER"
        ? padSizes.map((size) => [size >> 1, size - (size >> 1)])
        : padSizes.map((size) => [size - (size >> 1), size >> 1]);
  } else if (pads) {
    explicitPads = pads.slice(0, n).map((p, i) => [p, pads[i + n]] as [number, number]);
  } else {
    explicitPads = kernelShape.map(() => [0, 0] as [number, number]);
  }

  // Apply padding with -Infinity if needed
  const needsPadding = explicitPads.some(([a, b]) => a > 0 || b > 0);
  const padded = needsPadding ? padWithNegInf(x, explicitPads) : x;

  const output = lax.reduceWindow(
    padded,
    np.max,
    kernelShape,
    strides ?? kernelShape.map(() => 1),
  );
  return [output];
}
