// Movement operations, changing shape and indexing.
//
// TODO: Split (vision_encoder)

import { numpy as np } from "@jax-js/jax";

import {
  type Operand,
  operandToJax,
  operandToJs,
  StaticArray,
} from "../tensor";

export function Reshape(
  [data, shapeArr]: Operand[],
  { allowzero = 0 }: { allowzero?: number },
): Operand[] {
  const shape: number[] = operandToJs(shapeArr);
  if (shape.includes(0) && !allowzero) {
    // Semantics of allowzero=0 are confusing, will skip for now.
    // https://onnx.ai/onnx/operators/onnx__Reshape.html
    throw new Error(
      "Reshape with 0 in shape is not supported unless allowzero=1",
    );
  }
  if (data instanceof StaticArray) {
    if (shape.includes(-1)) {
      // Compute inferred dimension
      const knownSize = shape
        .filter((d) => d !== -1)
        .reduce((a, b) => a * b, 1);
      const inferredDim = data.data.length / knownSize;
      const finalShape = shape.map((d) => (d === -1 ? inferredDim : d));
      return [new StaticArray(data.data, finalShape, data.dtype)];
    }
    return [new StaticArray(data.data, shape, data.dtype)];
  } else {
    return [operandToJax(data).reshape(shape)];
  }
}

export function Transpose(
  inputs: Operand[],
  { perm }: { perm?: number[] },
): Operand[] {
  const [x] = inputs.map(operandToJax);
  return [np.transpose(operandToJax(x), perm)];
}

export function Flatten(
  inputs: Operand[],
  { axis = 1 }: { axis?: number },
): Operand[] {
  // Make a 2D matrix with x[:axis] and x[axis:] flattened.
  const [x] = inputs.map(operandToJax);
  if (axis <= 0) axis += x.ndim;
  const batchSize = x.shape.slice(0, axis).reduce((a, b) => a * b, 1);
  return [x.reshape([batchSize, -1])];
}

export function Expand([xOp, shape]: Operand[]): Operand[] {
  const x = operandToJax(xOp);
  const finalShape = np.broadcastShapes(x.shape, operandToJs(shape));
  return [np.broadcastTo(x, finalShape)];
}

export function Squeeze(
  [data, axes]: Operand[],
  { axes: axesBeforeOpset13 }: { axes?: number[] },
): Operand[] {
  const axis: number[] | undefined = axes
    ? operandToJs(axes)
    : (axesBeforeOpset13 ?? undefined);
  return [np.squeeze(operandToJax(data), axis)];
}

export function Unsqueeze(
  [data, axes]: Operand[],
  { axes: axesBeforeOpset13 }: { axes?: number[] },
): Operand[] {
  const axis: number[] = axes ? operandToJs(axes) : axesBeforeOpset13;
  if (!axis) {
    throw new Error("Unsqueeze requires axes");
  }
  const outputRank = data.shape.length + axis.length;
  const axisSet = new Set(axis.map((i) => (i < 0 ? outputRank + i : i)));
  const newShape = [...data.shape];
  for (const j of [...axisSet].sort()) {
    newShape.splice(j, 0, 1);
  }
  if (data instanceof StaticArray) {
    return [new StaticArray(data.data, newShape, data.dtype)];
  } else {
    return [data.reshape(newShape)];
  }
}

export function Gather(
  [dataOp, indicesOp]: Operand[],
  { axis = 0 }: { axis?: number },
): Operand[] {
  // If both are StaticArray, gather on CPU
  if (dataOp instanceof StaticArray && indicesOp instanceof StaticArray) {
    const data = dataOp;
    const indices = indicesOp;
    if (axis < 0) axis += data.shape.length;

    // For 1D data with scalar or 1D indices (common case for shape ops)
    if (data.shape.length === 1) {
      const result = new Int32Array(indices.data.length);
      for (let i = 0; i < indices.data.length; i++) {
        const idx =
          indices.data[i] < 0
            ? data.shape[0] + indices.data[i]
            : indices.data[i];
        result[i] = data.data[idx];
      }
      return [new StaticArray(result, indices.shape, data.dtype)];
    }
  }

  const data = operandToJax(dataOp);
  const indices = operandToJax(indicesOp);
  if (axis < 0) axis += data.ndim;
  const sliceArgs: (np.Array | [])[] = new Array(data.ndim).fill([]);
  sliceArgs[axis] = indices;
  return [data.slice(...sliceArgs)];
}

export function Concat(
  inputs: Operand[],
  { axis }: { axis: number },
): Operand[] {
  // If all inputs are StaticArray, concatenate on CPU
  if (inputs.every((op) => op instanceof StaticArray)) {
    const arrays = inputs as StaticArray[];
    if (axis < 0) axis += arrays[0].shape.length;

    // Calculate output shape
    const outShape = [...arrays[0].shape];
    outShape[axis] = arrays.reduce((sum, arr) => sum + arr.shape[axis], 0);

    // Concatenate data (only supports 1D arrays for simplicity - common for shapes)
    if (arrays[0].shape.length === 1 && axis === 0) {
      const totalLen = arrays.reduce((sum, arr) => sum + arr.data.length, 0);
      const result = new Int32Array(totalLen);
      let offset = 0;
      for (const arr of arrays) {
        result.set(arr.data, offset);
        offset += arr.data.length;
      }
      return [new StaticArray(result, outShape, arrays[0].dtype)];
    }
  }

  const arrays = inputs.map(operandToJax);
  return [np.concatenate(arrays, axis)];
}

export function Tile([input, repeats]: Operand[]): Operand[] {
  return [np.tile(operandToJax(input), operandToJs(repeats))];
}

/** Compute slice ranges from ONNX Slice parameters. */
function computeSliceRanges(
  shape: number[],
  startsArr: number[],
  endsArr: number[],
  axesArr: number[] | null,
  stepsArr: number[] | null,
): [number, number, number][] {
  const sliceRanges: [number, number, number][] = shape.map((d) => [0, d, 1]);
  const targetAxes = axesArr ?? startsArr.map((_, i) => i);

  for (let i = 0; i < targetAxes.length; i++) {
    let axis = targetAxes[i];
    if (axis < 0) axis += shape.length;

    const step = stepsArr ? stepsArr[i] : 1;
    if (step <= 0) {
      throw new Error("Slice with step <= 0 is not supported");
    }

    const dimSize = shape[axis];
    let start = startsArr[i];
    let end = endsArr[i];

    // Handle negative indices (but not very large values used as "to the end")
    // ONNX uses INT_MAX or very large values to mean "slice to end"
    if (start < -dimSize) start = 0;
    else if (start < 0) start = dimSize + start;

    if (end < -dimSize) end = 0;
    else if (end < 0) end = dimSize + end;

    // Clamp to valid range
    start = Math.max(0, Math.min(start, dimSize));
    end = Math.max(start, Math.min(end, dimSize));

    sliceRanges[axis] = [start, end, step];
  }
  return sliceRanges;
}

export function Slice([
  dataOp,
  starts,
  ends,
  axes,
  steps,
]: Operand[]): Operand[] {
  const startsArr: number[] = operandToJs(starts);
  const endsArr: number[] = operandToJs(ends);
  const axesArr: number[] | null = axes ? operandToJs(axes) : null;
  const stepsArr: number[] | null = steps ? operandToJs(steps) : null;

  const sliceRanges = computeSliceRanges(
    dataOp.shape,
    startsArr,
    endsArr,
    axesArr,
    stepsArr,
  );

  // Handle StaticArray for 1D case (common for shape manipulation)
  if (dataOp instanceof StaticArray && dataOp.shape.length === 1) {
    const [start, end, step] = sliceRanges[0];
    if (step === 1) {
      const result = dataOp.data.slice(start, end);
      return [new StaticArray(result, [end - start], dataOp.dtype)];
    }
    // Handle step != 1
    const len = Math.ceil((end - start) / step);
    const result = new Int32Array(len);
    for (let i = 0; i < len; i++) {
      result[i] = dataOp.data[start + i * step];
    }
    return [new StaticArray(result, [len], dataOp.dtype)];
  }

  const data = operandToJax(dataOp);

  // First pass: do basic start:end slices
  const sliceArgs: ([] | [number, number])[] = sliceRanges.map(
    ([start, end], i): [] | [number, number] =>
      start === 0 && end === data.shape[i] ? [] : [start, end],
  );
  let result = data.slice(...sliceArgs);

  // Second pass: handle steps != 1 using reshape + slice
  for (let axis = 0; axis < sliceRanges.length; axis++) {
    const [start, end, step] = sliceRanges[axis];
    if (step === 1) continue;

    const len = end - start;
    const outLen = Math.ceil(len / step);
    // Pad to make divisible by step, reshape to [outLen, step], take [:, 0]
    const padded = outLen * step;
    if (padded > len) {
      // Need to pad with zeros
      const padShape = [...result.shape];
      padShape[axis] = padded - len;
      const padding = np.zeros(padShape, { dtype: result.dtype });
      result = np.concatenate([result, padding], axis);
    }
    // Reshape to split axis into [outLen, step]
    const newShape = [
      ...result.shape.slice(0, axis),
      outLen,
      step,
      ...result.shape.slice(axis + 1),
    ];
    result = result.reshape(newShape);
    // Take index 0 on the step dimension (axis + 1)
    const selectArgs: ([] | number)[] = new Array(result.ndim).fill([]);
    selectArgs[axis + 1] = 0;
    result = result.slice(...selectArgs);
  }

  return [result];
}
