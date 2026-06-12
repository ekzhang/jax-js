import { numpy as np } from "@jax-js/jax";

// TODO: Move this to `library/numpy.ts` after auditing for edge cases.
export function takeAlongAxis(
  data: np.Array,
  indices: np.Array,
  axis: number,
): np.Array {
  const ndim = data.ndim;
  const a = axis < 0 ? ndim + axis : axis;
  if (a < 0 || a >= ndim) {
    throw new Error(`takeAlongAxis: axis ${axis} is out of bounds`);
  }

  const dataShape = data.shape;
  const idxShape = indices.shape;
  if (indices.ndim !== ndim) {
    throw new Error(
      `takeAlongAxis: indices rank ${indices.ndim} must match data rank ${ndim}`,
    );
  }
  for (let i = 0; i < ndim; i++) {
    if (i !== a && idxShape[i] !== dataShape[i]) {
      throw new Error(
        `takeAlongAxis: indices shape ${idxShape} is incompatible with data shape ${dataShape} along axis ${a}`,
      );
    }
  }

  if (indices.dtype !== np.int32 && indices.dtype !== np.uint32) {
    indices = indices.astype(np.int32);
  }

  const sliceArgs: np.Array[] = [];
  for (let i = 0; i < ndim; i++) {
    if (i === a) {
      sliceArgs.push(indices);
      continue;
    }
    const reshapeDims = idxShape.map((_, j) => (j === i ? idxShape[i] : 1));
    sliceArgs.push(np.arange(idxShape[i]).reshape(reshapeDims));
  }

  return data.slice(...sliceArgs);
}
