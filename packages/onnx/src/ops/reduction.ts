// Reductions and matrix multiplication.

import { nn, numpy as np } from "@jax-js/jax";

function wrapReduction(
  fn: (
    a: np.Array,
    axis: number[] | null,
    opts?: { keepdims?: boolean },
  ) => np.Array,
  {
    prelude,
    epilogue,
  }: {
    prelude?: (a: np.Array) => np.Array;
    epilogue?: (a: np.Array) => np.Array;
  } = {},
) {
  return (
    [x, axesInput]: np.Array[],
    {
      keepdims = 1,
      noop_with_empty_axes = 0,
      axes: axesAttr,
    }: { keepdims?: number; noop_with_empty_axes?: number; axes?: number[] },
  ) => {
    // axes can come from input tensor (opset 18+) or attribute (opset <18)
    let axis: number[] | null = axesInput ? axesInput.js() : (axesAttr ?? []);
    if (axis?.length === 0 && !noop_with_empty_axes) axis = null;
    if (prelude) x = prelude(x);
    x = fn(x, axis, { keepdims: Boolean(keepdims) });
    if (epilogue) x = epilogue(x);
    return [x];
  };
}

export const ReduceL1 = wrapReduction(np.sum, { prelude: np.abs });
export const ReduceL2 = wrapReduction(np.sum, {
  prelude: np.square,
  epilogue: np.sqrt,
});
export const ReduceLogSum = wrapReduction(np.sum, { epilogue: np.log });
export const ReduceLogSumExp = wrapReduction(nn.logsumexp);
export const ReduceMax = wrapReduction(np.max);
export const ReduceMean = wrapReduction(np.mean);
export const ReduceMin = wrapReduction(np.min);
export const ReduceProd = wrapReduction(np.prod);
export const ReduceSum = wrapReduction(np.sum);
export const ReduceSumSquare = wrapReduction(np.sum, { prelude: np.square });

export function CumSum(
  [x, axisOnnx]: np.Array[],
  { exclusive = 0, reverse = 0 }: { exclusive?: number; reverse?: number },
) {
  if (exclusive)
    throw new Error("CumSum ONNX operand does not support exclusive=true");
  if (axisOnnx.ndim !== 0)
    throw new Error("CumSum ONNX axis operand must be a scalar");
  const axis: number = axisOnnx.js();
  if (reverse) x = np.flip(x, axis);
  x = np.cumsum(x, axis);
  if (reverse) x = np.flip(x, axis);
  return [x];
}

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

export function Einsum(
  inputs: np.Array[],
  { equation }: { equation: string },
): np.Array[] {
  if (typeof equation !== "string")
    throw new Error("Einsum ONNX operand requires equation string");
  return [np.einsum(equation, ...inputs)];
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
