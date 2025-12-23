// Convolution operations.
//
// TODO: ConvTranspose (prompt_encoder_mask_decoder)
// TODO: MaxPool (image_encoder)
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
