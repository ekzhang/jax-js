// Reductions and matrix multiplication.
//
// TODO: ReduceMean
// TODO: ReduceSum, ReduceMax, ReduceMin, ReduceProd
// TODO: CumSum (image_encoder)

import { nn, numpy as np } from "@jax-js/jax";

export function MatMul([a, b]: np.Array[]): np.Array[] {
  return [np.matmul(a, b)];
}

export function Gemm(
  [a, b, c]: np.Array[],
  {
    alpha = 1,
    beta = 1,
    transA = 0,
    transB = 0,
  }: {
    alpha?: number;
    beta?: number;
    transA?: number;
    transB?: number;
  },
) {
  // a, b, c are all 2D
  if (transA) a = a.transpose();
  if (transB) b = b.transpose();
  let result = np.matmul(a, b);
  if (alpha !== 1) result = result.mul(alpha);
  if (c) {
    if (beta !== 0) result = result.add(c.mul(beta));
    else c.dispose();
  }
  return [result];
}

export function Softmax(
  [x]: np.Array[],
  { axis = -1 }: { axis?: number },
): np.Array[] {
  return [nn.softmax(x, axis)];
}

export function LogSoftmax(
  [x]: np.Array[],
  { axis = -1 }: { axis?: number },
): np.Array[] {
  return [nn.logSoftmax(x, axis)];
}

/*

  ReduceSum: ([x, axes], { keepdims = 1, noop_with_empty_axes = 0 }) => {
    const axesArr = axes ? axes.js().flat().map(Number) : null;
    if (axesArr?.length === 0 && noop_with_empty_axes) return [x];
    return [np.sum(x, axesArr, { keepdims: !!keepdims })];
  },
  ReduceMean: ([x, axes], { keepdims = 1 }) => {
    const axesArr = axes ? axes.js().flat().map(Number) : null;
    return [np.mean(x, axesArr, { keepdims: !!keepdims })];
  },
  ReduceMax: ([x, axes], { keepdims = 1 }) => {
    const axesArr = axes ? axes.js().flat().map(Number) : null;
    return [np.max(x, axesArr, { keepdims: !!keepdims })];
  },
  ReduceMin: ([x, axes], { keepdims = 1 }) => {
    const axesArr = axes ? axes.js().flat().map(Number) : null;
    return [np.min(x, axesArr, { keepdims: !!keepdims })];
  },
  ReduceProd: ([x, axes], { keepdims = 1 }) => {
    const axesArr = axes ? axes.js().flat().map(Number) : null;
    return [np.prod(x, axesArr, { keepdims: !!keepdims })];
  },
*/
