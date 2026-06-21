/** Supported data types for loading from Safetensors format. */
export type DType =
  | "F16" // Float16Array
  | "F32" // Float32Array
  | "F64" // Float64Array
  | "I8" // Int8Array
  | "I16" // Int16Array
  | "I32" // Int32Array
  | "I64" // BigInt64Array
  | "U8" // Uint8Array
  | "U16" // Uint16Array
  | "U32" // Uint32Array
  | "U64" // BigUint64Array
  | "BOOL"; // Represented as Uint8Array

export type Tensor = {
  dtype: DType;
  shape: number[];
  data: TensorData;
};

export type TensorData =
  | Float16Array
  | Float32Array
  | Float64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | BigInt64Array
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | BigUint64Array;

export type File = {
  tensors: { [key: string]: Tensor };
  metadata?: Record<string, string>;
  totalSize: number;
};

type TensorDataConstructor = {
  readonly BYTES_PER_ELEMENT: number;
  new (buffer: ArrayBuffer, byteOffset: number, length: number): TensorData;
};

function alignedData(
  ctor: TensorDataConstructor,
  buffer: ArrayBuffer,
  byteOffset: number,
  length: number,
): TensorData {
  if (byteOffset % ctor.BYTES_PER_ELEMENT === 0) {
    return new ctor(buffer, byteOffset, length);
  }
  // Slow path: If the array is not aligned, we need to make a copy.
  const byteLength = length * ctor.BYTES_PER_ELEMENT;
  const aligned = new Uint8Array(buffer, byteOffset, byteLength).slice();
  return new ctor(aligned.buffer, 0, length);
}

/** Load data from a safetensors file. */
export function parse(data: Uint8Array<ArrayBuffer> | ArrayBuffer): File {
  let buffer: ArrayBuffer;
  let ptr = 0;
  const len = data.byteLength;
  if (data instanceof ArrayBuffer) {
    buffer = data;
  } else {
    buffer = data.buffer;
    ptr = data.byteOffset;
  }

  if (len < 8) throw new Error("Data too short to be a valid safetensors file");
  const view = new DataView(buffer, ptr, 8);

  const headerSize = view.getBigUint64(0, true);
  if (headerSize > BigInt(len - 8)) throw new Error("Invalid header size");

  let header: any;
  try {
    header = JSON.parse(
      new TextDecoder().decode(
        new Uint8Array(buffer, ptr + 8, Number(headerSize)),
      ),
    );
  } catch (error) {
    throw new Error(`Failed to parse safetensors header as JSON: ${error}`);
  }

  const file: File = { tensors: {}, totalSize: len };
  for (const [key, value] of Object.entries(header)) {
    if (key === "__metadata__") {
      file.metadata = value as Record<string, string>;
      continue;
    }
    const { dtype, shape, data_offsets } = value as {
      dtype: DType;
      shape: number[];
      data_offsets: [number, number];
    };
    const byteOffset = ptr + Number(headerSize) + 8 + data_offsets[0];
    const byteLength = data_offsets[1] - data_offsets[0];
    let data: TensorData;
    switch (dtype) {
      case "F16":
        data = alignedData(Float16Array, buffer, byteOffset, byteLength / 2);
        break;
      case "F32":
        data = alignedData(Float32Array, buffer, byteOffset, byteLength / 4);
        break;
      case "F64":
        data = alignedData(Float64Array, buffer, byteOffset, byteLength / 8);
        break;
      case "I8":
        data = new Int8Array(buffer, byteOffset, byteLength);
        break;
      case "I16":
        data = alignedData(Int16Array, buffer, byteOffset, byteLength / 2);
        break;
      case "I32":
        data = alignedData(Int32Array, buffer, byteOffset, byteLength / 4);
        break;
      case "I64":
        data = alignedData(BigInt64Array, buffer, byteOffset, byteLength / 8);
        break;
      case "U8":
        data = new Uint8Array(buffer, byteOffset, byteLength);
        break;
      case "U16":
        data = alignedData(Uint16Array, buffer, byteOffset, byteLength / 2);
        break;
      case "U32":
        data = alignedData(Uint32Array, buffer, byteOffset, byteLength / 4);
        break;
      case "U64":
        data = alignedData(BigUint64Array, buffer, byteOffset, byteLength / 8);
        break;
      case "BOOL":
        data = new Uint8Array(buffer, byteOffset, byteLength);
        break;
      default:
        throw new Error(`Unsupported dtype: ${dtype}`);
    }
    file.tensors[key] = { dtype, shape, data };
  }
  return file;
}

/**
 * Convert a flat dictionary of tensors into a nested object.
 *
 * For example, take the case:
 *
 * ```js
 * {
 *   "layers.0.bias": b0,
 *   "layers.0.weight": w0,
 *   "layers.1.bias": b1,
 *   "layers.1.weight": w1,
 * }
 * ```
 *
 * This would become:
 *
 * ```js
 * {
 *   layers: [
 *     { bias: b0, weight: w0 },
 *     { bias: b1, weight: w1 },
 *   ],
 * }
 * ```
 */
export function toNested(tensors: { [key: string]: any }): any {
  const isNumeric = (s: string) => /^\d+$/.test(s);

  // Check if the top level should be an array
  const keys = Object.keys(tensors);
  const isTopLevelArray =
    keys.length > 0 &&
    keys.every((key) => {
      const firstPart = key.split(".")[0];
      return isNumeric(firstPart);
    });

  const result: any = isTopLevelArray ? [] : {};

  for (const [key, value] of Object.entries(tensors)) {
    if (!key) throw new Error("empty key in tensors list");

    const parts = key.split(".");
    let currentObj: any = result;
    let currentPart: string | number = parts[0];
    for (let i = 1; i < parts.length; i++) {
      const part = parts[i];
      if (isNumeric(part)) {
        if (!Object.hasOwn(currentObj, currentPart))
          currentObj[currentPart] = [];
        currentObj = currentObj[currentPart];
        currentPart = parseInt(part, 10);
      } else {
        if (!Object.hasOwn(currentObj, currentPart))
          currentObj[currentPart] = {};
        currentObj = currentObj[currentPart];
        currentPart = part;
      }
    }
    currentObj[currentPart] = value;
  }

  return result;
}

/**
 * Check if a value is a leaf node (not an object or array that should be traversed).
 * A leaf is anything that is not a plain Object or Array.
 */
function isLeaf(value: any): boolean {
  if (typeof value !== "object" || value === null) {
    return true;
  }
  return !Array.isArray(value) && value.constructor !== Object;
}

/**
 * Convert a nested object structure back into a flat dictionary of tensors.
 *
 * This is the inverse operation of `toNested`.
 *
 * For example:
 *
 * ```js
 * {
 *   layers: [
 *     { bias: b0, weight: w0 },
 *     { bias: b1, weight: w1 },
 *   ],
 * }
 * ```
 *
 * Would become:
 *
 * ```js
 * {
 *   "layers.0.bias": b0,
 *   "layers.0.weight": w0,
 *   "layers.1.bias": b1,
 *   "layers.1.weight": w1,
 * }
 * ```
 */
export function fromNested(
  nested: any,
  prefix: string = "",
): { [key: string]: any } {
  const result: { [key: string]: any } = {};

  function flatten(obj: any, currentPrefix: string) {
    if (isLeaf(obj)) {
      if (!currentPrefix)
        throw new Error("cannot have empty key in nested object");
      result[currentPrefix] = obj;
      return;
    }

    if (Array.isArray(obj)) {
      for (let i = 0; i < obj.length; i++) {
        const newPrefix = currentPrefix ? `${currentPrefix}.${i}` : `${i}`;
        flatten(obj[i], newPrefix);
      }
    } else {
      for (const [key, value] of Object.entries(obj)) {
        const newPrefix = currentPrefix ? `${currentPrefix}.${key}` : key;
        flatten(value, newPrefix);
      }
    }
  }

  flatten(nested, prefix);
  return result;
}
