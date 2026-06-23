// Linear algebra routines.

import { numpy as np } from "@jax-js/jax";

import {
  onnxDtypeToJax,
  type Operand,
  operandToJax,
  operandToJs,
} from "../tensor";

export function Det([xOp]: Operand[]): Operand[] {
  const x = operandToJax(xOp);
  return [np.linalg.det(x)];
}

export function EyeLike(
  [inputOp]: Operand[],
  { dtype, k = 0 }: { dtype?: number; k?: number } = {},
): Operand[] {
  const input = operandToJax(inputOp);
  if (input.ndim !== 2) {
    const ndim = input.ndim;
    input.dispose();
    throw new Error(`EyeLike: input must be 2D, got ${ndim}D`);
  }
  const [rows, cols] = input.shape;
  const outDtype = dtype === undefined ? input.dtype : onnxDtypeToJax(dtype);
  const device = input.device;
  input.dispose();

  if (k > 0) {
    if (k >= cols) return [np.zeros([rows, cols], { dtype: outDtype, device })];
    return [
      np.pad(np.eye(rows, cols - k, { dtype: outDtype, device }), [
        [0, 0],
        [k, 0],
      ]),
    ];
  } else if (k < 0) {
    if (-k >= rows)
      return [np.zeros([rows, cols], { dtype: outDtype, device })];
    return [
      np.pad(np.eye(rows + k, cols, { dtype: outDtype, device }), [
        [-k, 0],
        [0, 0],
      ]),
    ];
  }
  return [np.eye(rows, cols, { dtype: outDtype, device })];
}

export function LpNormalization(
  [xOp]: Operand[],
  { axis = -1, p = 2 }: { axis?: number; p?: number } = {},
): Operand[] {
  if (p !== 1 && p !== 2) {
    throw new Error(`LpNormalization: p must be 1 or 2, got ${p}`);
  }

  const x = operandToJax(xOp);
  const norm = np.linalg.vectorNorm(x.ref, { ord: p, axis, keepdims: true });
  const safeNorm = np.where(norm.ref.equal(0), np.onesLike(norm.ref), norm);
  return [x.div(safeNorm)];
}

export function Trilu(
  [xOp, kOp]: Operand[],
  { upper = 1 }: { upper?: number } = {},
): Operand[] {
  const x = operandToJax(xOp);
  let k = 0;
  if (kOp) {
    const kValue = operandToJs(kOp);
    if (typeof kValue !== "number" || !Number.isInteger(kValue)) {
      x.dispose();
      throw new Error(
        `Trilu: k must be an integer, got ${JSON.stringify(kValue)}`,
      );
    }
    k = kValue;
  }
  return [upper ? np.triu(x, k) : np.tril(x, k)];
}
