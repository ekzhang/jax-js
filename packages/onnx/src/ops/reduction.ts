// Reductions and matrix multiplication.

import { nn, numpy as np } from "@jax-js/jax";

import { type Operand, operandToJax, operandToJs } from "../tensor";

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
    [x, axesInput]: Operand[],
    {
      keepdims = 1,
      noop_with_empty_axes = 0,
      axes: axesAttr,
    }: { keepdims?: number; noop_with_empty_axes?: number; axes?: number[] },
  ): Operand[] => {
    // axes can come from input tensor (opset 18+) or attribute (opset <18)
    let axis: number[] | null = axesInput ? operandToJs(axesInput) : (axesAttr ?? []);
    if (axis?.length === 0 && !noop_with_empty_axes) axis = null;
    let arr = operandToJax(x);
    if (prelude) arr = prelude(arr);
    arr = fn(arr, axis, { keepdims: Boolean(keepdims) });
    if (epilogue) arr = epilogue(arr);
    return [arr];
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
  [x, axisOnnx]: Operand[],
  { exclusive = 0, reverse = 0 }: { exclusive?: number; reverse?: number },
): Operand[] {
  if (exclusive)
    throw new Error("CumSum ONNX operand does not support exclusive=true");
  const axis: number = operandToJs(axisOnnx);
  let arr = operandToJax(x);
  if (reverse) arr = np.flip(arr, axis);
  arr = np.cumsum(arr, axis);
  if (reverse) arr = np.flip(arr, axis);
  return [arr];
}

export function MatMul([a, b]: Operand[]): Operand[] {
  return [np.matmul(operandToJax(a), operandToJax(b))];
}

export function Gemm(
  [a, b, c]: Operand[],
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
): Operand[] {
  // a, b, c are all 2D
  let arrA = operandToJax(a);
  let arrB = operandToJax(b);
  if (transA) arrA = arrA.transpose();
  if (transB) arrB = arrB.transpose();
  let result = np.matmul(arrA, arrB);
  if (alpha !== 1) result = result.mul(alpha);
  if (c) {
    const arrC = operandToJax(c);
    if (beta !== 0) result = result.add(arrC.mul(beta));
    else arrC.dispose();
  }
  return [result];
}

export function Einsum(
  inputs: Operand[],
  { equation }: { equation: string },
): Operand[] {
  if (typeof equation !== "string")
    throw new Error("Einsum ONNX operand requires equation string");
  return [np.einsum(equation, ...inputs.map(operandToJax))];
}

export function Softmax(
  [x]: Operand[],
  { axis = -1 }: { axis?: number },
): Operand[] {
  return [nn.softmax(operandToJax(x), axis)];
}

export function LogSoftmax(
  [x]: Operand[],
  { axis = -1 }: { axis?: number },
): Operand[] {
  return [nn.logSoftmax(operandToJax(x), axis)];
}
