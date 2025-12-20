// Movement operations, changing shape and indexing.
//
// TODO: Transpose
// TODO: Squeeze, Unsqueeze
// TODO: Flatten
// TODO: Concat
// TODO: Split (vision_encoder)
// TODO: Gather
// TODO: Slice
// TODO: Expand
// TODO: Tile

import { numpy as np } from "@jax-js/jax";

export function Reshape([data, shape]: np.Array[]): np.Array[] {
  // TODO: There's an "allowzero" attribute which seems a bit confusing. I think
  // the default (allowzero=0) has ambiguous semantics, we don't implement it.
  return [data.reshape(shape.js())];
}

/*
  Transpose: ([x], { perm }) => {
    if (perm) {
      return [x.transpose(perm)];
    }
    // Default: reverse all axes
    return [x.transpose()];
  },

  Squeeze: ([x, axes], attrs) => {
    // ONNX opset 13+: axes is an input tensor
    // ONNX opset <13: axes is an attribute
    let axesArr: number[] | undefined;
    if (axes) {
      axesArr = axes.js().flat().map(Number);
    } else if (attrs.axes) {
      axesArr = attrs.axes;
    }

    if (axesArr && axesArr.length > 0) {
      // Remove specified axes (must be size 1)
      const newShape = x.shape.filter((_, i) => !axesArr!.includes(i));
      return [x.reshape(newShape)];
    } else {
      // Remove all axes of size 1
      const newShape = x.shape.filter((d) => d !== 1);
      return [x.reshape(newShape)];
    }
  },

  Unsqueeze: ([x, axes], attrs) => {
    // ONNX opset 13+: axes is an input tensor
    // ONNX opset <13: axes is an attribute
    let axesArr: number[];
    if (axes) {
      axesArr = axes.js().flat().map(Number);
    } else if (attrs.axes) {
      axesArr = attrs.axes;
    } else {
      throw new Error("Unsqueeze requires axes");
    }

    // Normalize negative axes and sort
    const ndim = x.ndim + axesArr.length;
    axesArr = axesArr.map((a) => (a < 0 ? ndim + a : a)).sort((a, b) => a - b);

    const shape = [...x.shape];
    for (const axis of axesArr) {
      shape.splice(axis, 0, 1);
    }
    return [x.reshape(shape)];
  },

  Flatten: ([x], { axis = 1 }) => {
    const pre = x.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const post = x.shape.slice(axis).reduce((a, b) => a * b, 1);
    return [x.reshape([pre, post])];
  },

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

  Gather: ([data, indices], { axis = 0 }) => {
    // Normalize axis
    const normalizedAxis = axis < 0 ? data.ndim + axis : axis;

    // Get indices as flat array
    const indicesArr = indices.js().flat().map(Number);

    // For now, implement simple case: gather along one axis
    // This is common for embedding lookups
    if (indices.ndim === 1 || indices.ndim === 0) {
      const gathered = indicesArr.map((idx: number) => {
        // Build slice args: [] for full dimension, [start, end] for slice
        const sliceArgs: ([] | [number, number])[] = data.ref.shape.map(
          (d: number, i: number): [] | [number, number] =>
            i === normalizedAxis ? [idx, idx + 1] : [],
        );
        return data.ref.slice(...sliceArgs);
      });
      data.dispose();

      if (gathered.length === 1) {
        // Squeeze the gathered axis if single index
        const result = gathered[0];
        if (indices.ndim === 0) {
          const newShape = result.shape.filter(
            (_: number, i: number) => i !== normalizedAxis,
          );
          return [result.reshape(newShape)];
        }
        return [result];
      }
      return [np.concatenate(gathered, normalizedAxis)];
    }

    throw new Error(
      `Gather with ${indices.ndim}D indices not yet fully supported`,
    );
  },

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
