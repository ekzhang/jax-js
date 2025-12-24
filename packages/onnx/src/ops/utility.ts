// Utility operations, such as dtype conversion and data prep.
//
// TODO: Range (prompt_encoder_mask_decoder, vision_encoder)
// TODO: OneHot (prompt_encoder_mask_decoder)
// TODO: ScatterND (prompt_encoder_mask_decoder)

import { numpy as np } from "@jax-js/jax";
import { TensorProto } from "onnx-buf";

import {
  type Operand,
  operandToJax,
  operandToJs,
  StaticArray,
  tensorToOperand,
} from "../tensor";

export function Shape(
  [data]: Operand[],
  { start = 0, end }: { start?: number; end?: number },
): Operand[] {
  const arr = operandToJax(data);
  const shape = arr.shape.slice(start, end);
  arr.dispose();
  return [new StaticArray(shape, [shape.length], np.int32)];
}

export function Constant(
  _: Operand[],
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
): Operand[] {
  if (value !== undefined) {
    return [tensorToOperand(value)];
  } else if (value_float !== undefined) {
    return [np.array(value_float)];
  } else if (value_floats !== undefined) {
    return [np.array(value_floats)];
  } else if (value_int !== undefined) {
    return [new StaticArray([value_int], [], np.int32)];
  } else if (value_ints !== undefined) {
    return [new StaticArray(value_ints, [value_ints.length], np.int32)];
  } else if (value_string !== undefined || value_strings !== undefined) {
    throw new Error("ONNX Constant string values are not supported");
  } else {
    throw new Error("ONNX Constant has no value");
  }
}

export function ConstantOfShape(
  [input]: Operand[],
  { value }: { value?: TensorProto },
): Operand[] {
  const shape = operandToJs(input) as number[];
  if (value !== undefined) {
    const op = tensorToOperand(value);
    if (op instanceof StaticArray) {
      return [op.broadcastTo(shape)];
    } else {
      return [np.broadcastTo(op, shape)];
    }
  } else {
    return [np.zeros(shape)];
  }
}
