// Movement operations, changing shape and indexing.
//
// TODO: Concat
// TODO: Split (vision_encoder)
// TODO: Slice
// TODO: Tile

import { numpy as np } from "@jax-js/jax";

export function Reshape([data, shape]: np.Array[]): np.Array[] {
  // TODO: There's an "allowzero" attribute which seems a bit confusing. I think
  // the default (allowzero=0) has ambiguous semantics, we don't implement it.
  return [data.reshape(shape.js())];
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
  if (axis === undefined) {
    // Remove all size 1 dimensions
    return [data.reshape(data.shape.filter((d) => d !== 1))];
  }
  const axisSet = new Set(axis.map((i) => (i < 0 ? data.ndim + i : i)));
  const newShape = data.shape.filter((size, i) => {
    if (size !== 1) throw new Error("Cannot squeeze dimension with size != 1");
    return !axisSet.has(i);
  });
  return [data.reshape(newShape)];
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

/*

  Concat: (inputs, { axis }) => {
    return [np.concatenate(inputs, axis)];
  },

  Split: ([x, split], { axis = 0, num_outputs }) => {
    let splitSizes: number[];
    if (split) {
      splitSizes = split.js().flat().map(Number);
    } else if (num_outputs) {
      // Equal split
      const dimSize = x.shape[axis];
      const splitSize = Math.floor(dimSize / num_outputs);
      splitSizes = Array(num_outputs).fill(splitSize);
      // Handle remainder
      const remainder = dimSize % num_outputs;
      for (let i = 0; i < remainder; i++) {
        splitSizes[i]++;
      }
    } else {
      throw new Error("Split requires either split sizes or num_outputs");
    }

    const results: np.Array[] = [];
    let offset = 0;
    for (const size of splitSizes) {
      // Build slice indices for all dimensions
      // slice() takes variadic args, each being [] (full), [start, end], or number
      const sliceArgs: ([] | [number, number])[] = x.shape.map(
        (d: number, i: number): [] | [number, number] =>
          i === axis ? [offset, offset + size] : [],
      );
      results.push(x.ref.slice(...sliceArgs));
      offset += size;
    }
    x.dispose();
    return results;
  },


  // ============================================================
  // Gather and indexing operations
  // ============================================================

  Slice: ([data, starts, ends, axes, steps]) => {
    const startsArr = starts.js().flat().map(Number);
    const endsArr = ends.js().flat().map(Number);
    const axesArr = axes ? axes.js().flat().map(Number) : null;
    const stepsArr = steps ? steps.js().flat().map(Number) : null;

    // Build slice specification for all dimensions (default to full range)
    const sliceRanges: [number, number][] = data.shape.map((d: number) => [
      0,
      d,
    ]);

    const targetAxes = axesArr || startsArr.map((_: number, i: number) => i);
    for (let i = 0; i < targetAxes.length; i++) {
      const axis =
        targetAxes[i] < 0 ? data.ndim + targetAxes[i] : targetAxes[i];
      let start = startsArr[i];
      let end = endsArr[i];
      const step = stepsArr ? stepsArr[i] : 1;

      if (step !== 1) {
        throw new Error("Slice with step != 1 not yet supported");
      }

      // Handle negative indices
      const dimSize = data.shape[axis];
      if (start < 0) start = Math.max(0, dimSize + start);
      if (end < 0) end = dimSize + end;
      // Clamp to valid range
      start = Math.max(0, Math.min(start, dimSize));
      end = Math.max(0, Math.min(end, dimSize));

      sliceRanges[axis] = [start, end];
    }

    // Convert to slice args format: [] for full dim, [start, end] for range
    const sliceArgs: ([] | [number, number])[] = sliceRanges.map(
      ([start, end], i): [] | [number, number] =>
        start === 0 && end === data.shape[i] ? [] : [start, end],
    );
    return [data.slice(...sliceArgs)];
  },
  */
