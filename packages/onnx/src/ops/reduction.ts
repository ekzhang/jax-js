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
    let axis: number[] | null = axesInput
      ? operandToJs(axesInput)
      : (axesAttr ?? []);
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

export function MeanVarianceNormalization(
  [xOp]: Operand[],
  { axes = [0, 2, 3] }: { axes?: number[] },
): Operand[] {
  const x = operandToJax(xOp);
  const mean = np.mean(x.ref, axes, { keepdims: true });
  const centered = x.sub(mean);
  const std = np
    .sqrt(
      np.mean(np.square(centered.ref), axes, {
        keepdims: true,
      }),
    )
    .add(1e-9);
  return [centered.div(std)];
}

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

export function TopK(
  [xOp, kOp]: Operand[],
  {
    axis = -1,
    largest = 1,
  }: { axis?: number; largest?: number; sorted?: number },
): Operand[] {
  const x = operandToJax(xOp);
  const kRaw = operandToJs(kOp);
  const k: number = Array.isArray(kRaw) ? kRaw[0] : kRaw;
  const normAxis = axis < 0 ? x.ndim + axis : axis;
  const size = x.shape[normAxis];
  if (k < 0 || k > size) {
    throw new Error(`TopK: k must be in the range [0, ${size}], got ${k}`);
  }

  const sortedIndices = np.argsort(x.ref, normAxis);
  const sliceArgs: ([] | [number] | [number, number])[] = new Array(
    x.ndim,
  ).fill([]);
  if (k === 0) {
    sliceArgs[normAxis] = [0, 0];
  } else if (largest) {
    sliceArgs[normAxis] = [-k];
  } else {
    sliceArgs[normAxis] = [0, k];
  }
  let indices = sortedIndices.slice(...sliceArgs);
  if (largest) {
    indices = np.flip(indices, normAxis);
  }
  const values = np.takeAlongAxis(x, indices.ref, normAxis);
  return [values, indices];
}

export function ArgMax(
  [xOp]: Operand[],
  {
    axis = 0,
    keepdims = 1,
    select_last_index = 0,
  }: { axis?: number; keepdims?: number; select_last_index?: number },
): Operand[] {
  const x = operandToJax(xOp);
  if (axis < -x.ndim || axis >= x.ndim) {
    throw new Error(
      `ArgMax: axis ${axis} is out of bounds for tensor of ndim ${x.ndim}`,
    );
  }
  if (!select_last_index) {
    return [np.argmax(x, axis, { keepdims: Boolean(keepdims) })];
  }
  const normAxis = axis < 0 ? axis + x.ndim : axis;
  const flipped = np.flip(x, normAxis);
  const idx = np.argmax(flipped, normAxis, { keepdims: Boolean(keepdims) });
  return [idx.neg().add(x.shape[normAxis] - 1)];
}

export function ArgMin(
  [xOp]: Operand[],
  {
    axis = 0,
    keepdims = 1,
    select_last_index = 0,
  }: { axis?: number; keepdims?: number; select_last_index?: number },
): Operand[] {
  const x = operandToJax(xOp);
  if (axis < -x.ndim || axis >= x.ndim) {
    throw new Error(
      `ArgMin: axis ${axis} is out of bounds for tensor of ndim ${x.ndim}`,
    );
  }
  if (!select_last_index) {
    return [np.argmin(x, axis, { keepdims: Boolean(keepdims) })];
  }
  const normAxis = axis < 0 ? axis + x.ndim : axis;
  const flipped = np.flip(x, normAxis);
  const idx = np.argmin(flipped, normAxis, { keepdims: Boolean(keepdims) });
  return [idx.neg().add(x.shape[normAxis] - 1)];
}
