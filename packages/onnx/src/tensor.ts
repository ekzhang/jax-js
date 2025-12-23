// Tensor conversion utilities for ONNX to jax-js.

import { DType, numpy as np } from "@jax-js/jax";
import type { TensorProto } from "onnx-buf";
import { TensorProto_DataType } from "onnx-buf";

/** Convert an ONNX data type to a jax-js DType. */
export function onnxDtypeToJax(onnxType: TensorProto_DataType): DType {
  switch (onnxType) {
    case TensorProto_DataType.FLOAT:
      return np.float32;
    case TensorProto_DataType.INT32:
      return np.int32;
    case TensorProto_DataType.INT64: // int64 is used in shapes, we map to int32
      return np.int32;
    case TensorProto_DataType.UINT64: // uint64 may be used, we map to uint32
      return np.uint32;
    case TensorProto_DataType.FLOAT16:
      return np.float16;
    case TensorProto_DataType.DOUBLE:
      return np.float64;
    case TensorProto_DataType.BOOL:
      return np.bool;
    case TensorProto_DataType.UINT32:
      return np.uint32;
    case TensorProto_DataType.UINT8:
    case TensorProto_DataType.INT8:
    case TensorProto_DataType.UINT16:
    case TensorProto_DataType.INT16:
    default:
      throw new Error(`Unsupported ONNX dtype: ${onnxType}`);
  }
}

/** Parse raw tensor data based on ONNX data type. */
function parseRawData(
  rawData: Uint8Array<ArrayBuffer>,
  dataType: TensorProto_DataType,
):
  | Float32Array<ArrayBuffer>
  | Int32Array<ArrayBuffer>
  | Uint32Array<ArrayBuffer>
  | Float64Array<ArrayBuffer>
  | Float16Array<ArrayBuffer> {
  const buffer = rawData.buffer.slice(
    rawData.byteOffset,
    rawData.byteOffset + rawData.byteLength,
  );

  switch (dataType) {
    case TensorProto_DataType.FLOAT:
      return new Float32Array(buffer);
    case TensorProto_DataType.INT32:
      return new Int32Array(buffer);
    case TensorProto_DataType.UINT32:
      return new Uint32Array(buffer);
    case TensorProto_DataType.DOUBLE:
      return new Float64Array(buffer);
    case TensorProto_DataType.FLOAT16:
      return new Float16Array(buffer);
    case TensorProto_DataType.INT64: {
      // INT64 stored as 8 bytes per element, convert to Int32
      // Clamp to INT32 range to avoid overflow issues
      const i64 = new BigInt64Array(buffer);
      const INT32_MAX = BigInt(2147483647);
      const INT32_MIN = BigInt(-2147483648);
      return new Int32Array(
        Array.from(i64, (v) => {
          if (v > INT32_MAX) return Number(INT32_MAX);
          if (v < INT32_MIN) return Number(INT32_MIN);
          return Number(v);
        }),
      );
    }
    case TensorProto_DataType.UINT64: {
      // UINT64 stored as 8 bytes per element, convert to Uint32
      // Clamp to UINT32 range to avoid overflow issues
      const u64 = new BigUint64Array(buffer);
      const UINT32_MAX = BigInt(4294967295);
      return new Uint32Array(
        Array.from(u64, (v) =>
          v > UINT32_MAX ? Number(UINT32_MAX) : Number(v),
        ),
      );
    }
    case TensorProto_DataType.BOOL: {
      // Bool is stored as 1 byte per element
      return new Int32Array(Array.from(rawData, (v) => (v ? 1 : 0)));
    }
    case TensorProto_DataType.INT8: {
      const i8 = new Int8Array(buffer);
      return new Int32Array(i8);
    }
    case TensorProto_DataType.UINT8: {
      return new Int32Array(rawData);
    }
    case TensorProto_DataType.INT16: {
      const i16 = new Int16Array(buffer);
      return new Int32Array(i16);
    }
    case TensorProto_DataType.UINT16: {
      const u16 = new Uint16Array(buffer);
      return new Int32Array(u16);
    }
    default:
      throw new Error(`Unsupported raw data type: ${dataType}`);
  }
}

/**
 * Convert an ONNX TensorProto to a jax-js Array.
 */
export function tensorToArray(tensor: TensorProto): np.Array {
  const shape = tensor.dims.map(Number);
  const dtype = onnxDtypeToJax(tensor.dataType);

  // Determine data source and convert
  let data:
    | Float32Array<ArrayBuffer>
    | Int32Array<ArrayBuffer>
    | Uint32Array<ArrayBuffer>
    | Float64Array<ArrayBuffer>
    | Float16Array<ArrayBuffer>;

  if (tensor.rawData.length > 0) {
    // Most common: raw binary data
    data = parseRawData(tensor.rawData, tensor.dataType);
  } else if (tensor.floatData.length > 0) {
    data = Float32Array.from(tensor.floatData);
  } else if (tensor.int32Data.length > 0) {
    data = Int32Array.from(tensor.int32Data);
  } else if (tensor.doubleData.length > 0) {
    data = Float64Array.from(tensor.doubleData);
  } else if (tensor.int64Data.length > 0) {
    // We don't support int64 or uint64 natively, convert to int32/uint32.
    // Clamp to INT32 range to avoid overflow issues.
    const INT32_MAX = BigInt(2147483647);
    const INT32_MIN = BigInt(-2147483648);
    data = Int32Array.from(
      tensor.int64Data.map((v) => {
        if (v > INT32_MAX) return Number(INT32_MAX);
        if (v < INT32_MIN) return Number(INT32_MIN);
        return Number(v);
      }),
    );
  } else if (tensor.uint64Data.length > 0) {
    // Clamp to UINT32 range to avoid overflow issues.
    const UINT32_MAX = BigInt(4294967295);
    data = Uint32Array.from(
      tensor.uint64Data.map((v) =>
        v > UINT32_MAX ? Number(UINT32_MAX) : Number(v),
      ),
    );
  } else {
    // Empty tensor or scalar with no data
    if (shape.length === 0 || shape.reduce((a, b) => a * b, 1) === 0) {
      // Return empty array with correct shape
      return np.zeros(shape.length === 0 ? [] : shape, { dtype });
    }
    throw new Error(`Tensor ${tensor.name} has no data`);
  }

  return np.array(data, { shape, dtype });
}
