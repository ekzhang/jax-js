// Utility operations, such as dtype conversion and data prep.
//
// TODO: Range (prompt_encoder_mask_decoder, vision_encoder)
// TODO: OneHot (prompt_encoder_mask_decoder)
// TODO: ScatterND (prompt_encoder_mask_decoder)

import { numpy as np } from "@jax-js/jax";
import { TensorProto } from "onnx-buf";

import { tensorToArray } from "../tensor";

export function Shape(
  [data]: np.Array[],
  { start = 0, end }: { start?: number; end?: number },
) {
  const shape = data.shape.slice(start, end);
  data.dispose();
  return [np.array(shape, { dtype: np.int32 })];
}

export function Constant(
  _: np.Array[],
  {
    value,
    value_float,
    value_floats,
    value_int,
    value_ints,
    value_string,
    value_strings,
  }: {
    value?: TensorProto;
    value_float?: number;
    value_floats?: number[];
    value_int?: number;
    value_ints?: number[];
    value_string?: Uint8Array<ArrayBuffer>;
    value_strings?: Uint8Array<ArrayBuffer>[];
  },
): [np.Array] {
  if (value !== undefined) {
    return [tensorToArray(value)];
  } else if (value_float !== undefined) {
    return [np.array(value_float)];
  } else if (value_floats !== undefined) {
    return [np.array(value_floats)];
  } else if (value_int !== undefined) {
    return [np.array(value_int, { dtype: np.int32 })];
  } else if (value_ints !== undefined) {
    return [np.array(value_ints, { dtype: np.int32 })];
  } else if (value_string !== undefined || value_strings !== undefined) {
    throw new Error("ONNX Constant string values are not supported");
  } else {
    throw new Error("ONNX Constant has no value");
  }
}

export function ConstantOfShape(
  [input]: np.Array[],
  { value }: { value?: TensorProto },
): [np.Array] {
  const shape = input.js() as number[];
  if (value !== undefined) {
    return [np.broadcastTo(tensorToArray(value), shape)];
  } else {
    return [np.zeros(shape)];
  }
}
