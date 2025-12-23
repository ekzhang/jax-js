// Movement operations, changing shape and indexing.
//
// TODO: Split (vision_encoder)

import { numpy as np } from "@jax-js/jax";

export function Reshape(
  [data, shapeArr]: np.Array[],
  { allowzero = 0 }: { allowzero?: number },
): np.Array[] {
  const shape = shapeArr.js();
  if (shape.includes(0) && !allowzero) {
    // Semantics of allowzero=0 are confusing, will skip for now.
    // https://onnx.ai/onnx/operators/onnx__Reshape.html
    throw new Error(
      "Reshape with 0 in shape is not supported unless allowzero=1",
    );
  }
  return [data.reshape(shape)];
}

export function Transpose(
  [x]: np.Array[],
  { perm }: { perm?: number[] },
): np.Array[] {
  return [np.transpose(x, perm)];
}

export function Flatten(
  [x]: np.Array[],
  { axis = 1 }: { axis?: number },
): np.Array[] {
  // Make a 2D matrix with x[:axis] and x[axis:] flattened.
  if (axis <= 0) axis += x.ndim;
  const batchSize = x.shape.slice(0, axis).reduce((a, b) => a * b, 1);
  return [x.reshape([batchSize, -1])];
}

export function Expand([x, shape]: np.Array[]): np.Array[] {
  const finalShape = np.broadcastShapes(x.shape, shape.js());
  return [np.broadcastTo(x, finalShape)];
}

export function Squeeze(
  [data, axes]: np.Array[],
  { axes: axesBeforeOpset13 }: { axes?: number[] },
): np.Array[] {
  const axis: number[] | undefined = axes
    ? axes.js()
    : (axesBeforeOpset13 ?? undefined);
  return [np.squeeze(data, axis)];
}

export function Unsqueeze(
  [data, axes]: np.Array[],
  { axes: axesBeforeOpset13 }: { axes?: number[] },
): np.Array[] {
  const axis: number[] = axes ? axes.js() : axesBeforeOpset13!;
  if (!axis) {
    throw new Error("Unsqueeze requires axes");
  }
  const outputRank = data.ndim + axis.length;
  const axisSet = new Set(axis.map((i) => (i < 0 ? outputRank + i : i)));
  const newShape = [...data.shape];
  for (const j of [...axisSet].sort()) {
    newShape.splice(j, 0, 1);
  }
  return [data.reshape(newShape)];
}

export function Gather(
  [data, indices]: np.Array[],
  { axis = 0 }: { axis?: number },
): np.Array[] {
  if (axis < 0) axis += data.ndim;
  const sliceArgs: (np.Array | [])[] = new Array(data.ndim).fill([]);
  sliceArgs[axis] = indices;
  return [data.slice(...sliceArgs)];
}

export function Concat(inputs: np.Array[], { axis }: { axis: number }) {
  return [np.concatenate(inputs, axis)];
}

export function Tile([input, repeats]: np.Array[]): np.Array[] {
  return [np.tile(input, repeats.js())];
}

export function Slice([
  data,
  starts,
  ends,
  axes,
  steps,
]: np.Array[]): np.Array[] {
  const startsArr: number[] = starts.js();
  const endsArr: number[] = ends.js();
  const axesArr: number[] | null = axes ? axes.js() : null;
  const stepsArr: number[] | null = steps ? steps.js() : null;

  // Build slice specification for all dimensions (default to full range)
  const sliceRanges: [number, number, number][] = data.shape.map(
    (d: number) => [0, d, 1],
  );

  const targetAxes = axesArr ?? startsArr.map((_, i) => i);
  for (let i = 0; i < targetAxes.length; i++) {
    let axis = targetAxes[i];
    if (axis < 0) axis += data.ndim;

    const step = stepsArr ? stepsArr[i] : 1;
    if (step <= 0) {
      throw new Error("Slice with step <= 0 is not supported");
    }

    const dimSize = data.shape[axis];
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
