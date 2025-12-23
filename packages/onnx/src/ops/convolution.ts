// Convolution operations.
//
// TODO: ConvTranspose (prompt_encoder_mask_decoder)
// TODO: Pad (vision_encoder)

import { lax, numpy as np } from "@jax-js/jax";

const padsMapping: Record<string, lax.PaddingType> = {
  SAME_UPPER: "SAME",
  SAME_LOWER: "SAME_LOWER",
  VALID: "VALID",
};

export function Conv(
  [x, w, bias]: (np.Array | undefined)[],
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
  if (!x || !w) throw new Error("Conv: missing required inputs");
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
  let output = lax.convGeneralDilated(
    x,
    w,
    strides ?? wSpatial.map(() => 1),
    padsMapping[autoPad] ??
      pads?.slice(0, n).map((p, i) => [p, pads[i + n]]) ??
      "VALID",
    {
      rhsDilation: dilations,
      featureGroupCount: group,
    },
  );
  // Add bias if provided (reshape to [1, C, 1, 1, ...] for broadcasting)
  if (bias) {
    const biasShape = [bias.size, ...xSpatial.map(() => 1)];
    output = output.add(bias.reshape(biasShape));
  }
  return [output];
}

// Pad a tensor with -Infinity along spatial dimensions (for max pooling).
function padWithNegInf(x: np.Array, pads: [number, number][]): np.Array {
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
    explicitPads = pads
      .slice(0, n)
      .map((p, i) => [p, pads[i + n]] as [number, number]);
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

export function Resize(
  [x, _roi, scales, sizes]: np.Array[],
  {
    coordinate_transformation_mode: coordMode = "half_pixel",
    mode = "nearest",
    nearest_mode: nearestMode = "round_prefer_floor",
  }: {
    coordinate_transformation_mode?: string;
    mode?: string;
    nearest_mode?: string;
    // Ignored: cubic_coeff_a, exclude_outside, extrapolation_value,
    // keep_aspect_ratio_policy, axes
  },
): np.Array[] {
  // Only support nearest + asymmetric + floor for now
  if (mode !== "nearest") {
    throw new Error(`Resize: mode '${mode}' is not supported, only 'nearest'`);
  }
  if (coordMode !== "asymmetric") {
    throw new Error(
      `Resize: coordinate_transformation_mode '${coordMode}' is not supported, only 'asymmetric'`,
    );
  }
  if (nearestMode !== "floor") {
    throw new Error(
      `Resize: nearest_mode '${nearestMode}' is not supported, only 'floor'`,
    );
  }

  // Determine output shape from scales or sizes
  const inShape = x.shape;
  let outShape: number[];
  if (sizes && sizes.shape[0] > 0) {
    outShape = sizes.js();
  } else if (scales && scales.shape[0] > 0) {
    const scalesArr: number[] = scales.js();
    outShape = inShape.map((d, i) => Math.floor(d * scalesArr[i]));
  } else {
    throw new Error("Resize: either scales or sizes must be provided");
  }

  // For asymmetric + nearest + floor:
  // input_coord = floor(output_coord * (input_size / output_size))
  // This is equivalent to: input_coord = floor(output_coord / scale)
  //
  // We implement this by creating index arrays for each dimension and using
  // advanced indexing.
  let result = x;
  for (let axis = 0; axis < inShape.length; axis++) {
    const inSize = result.shape[axis];
    const outSize = outShape[axis];
    if (inSize === outSize) continue;

    // Create indices: floor(i * inSize / outSize) for i in 0..outSize
    const indices = np.array(
      Array.from({ length: outSize }, (_, i) =>
        Math.floor((i * inSize) / outSize),
      ),
      { dtype: np.int32 },
    );

    // Build slice args: [] for all dims except current axis
    const sliceArgs: (np.Array | [])[] = result.shape.map(() => [] as []);
    sliceArgs[axis] = indices;
    result = result.slice(...sliceArgs);
  }

  return [result];
}
