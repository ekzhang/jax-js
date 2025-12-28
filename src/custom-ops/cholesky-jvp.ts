import { AluOp } from "../alu.js";
import { array } from "../frontend/array.js";
import {
  bind1,
  broadcast,
  equal,
  greaterEqual,
  Primitive,
  reduce,
  reshape,
  triangularSolve,
  where,
  type Tracer,
} from "../frontend/core.js";

/**
 * JVP rule for Cholesky decomposition
 *
 * Reference: https://arxiv.org/pdf/1602.07527.pdf
 *
 * Given: A = L @ L^T
 * Forward: L_dot = L @ phi(L^{-1} @ dA @ L^{-T})
 *
 * where phi(X) takes lower triangular and divides diagonal by 2
 */
export function choleskyJVP(
  primals: Tracer[],
  tangents: Tracer[],
  params: { lower: boolean },
): [Tracer[], Tracer[]] {
  const [a] = primals;
  const [da] = tangents;

  // Cholesky JVP based on https://arxiv.org/pdf/1602.07527.pdf
  // L_dot = L @ phi(L^{-1} @ da @ L^{-T})
  // where phi(X) takes lower triangular and divides diagonal by 2
  const L = bind1(Primitive.CustomOp, [a], { name: "linalg.cholesky", lower: params.lower });
  const n = L.shape[0];

  // Compute L^{-1} @ da using triangular_solve(L, da, left_side=true)
  const tmp1 = triangularSolve((L as any).ref, da, {
    leftSide: true,
    lower: true,
    transposeA: false,
    unitDiagonal: false,
  });

  // Compute tmp1 @ L^{-T} using triangular_solve(L, tmp1^T, left_side=true, transpose_a=true)^T
  // Equivalently: triangular_solve(L, tmp1, left_side=false, transpose_a=true)
  const tmp2 = triangularSolve((L as any).ref, tmp1, {
    leftSide: false,
    lower: true,
    transposeA: true,
    unitDiagonal: false,
  });

  // phi function: take lower triangular, divide diagonal by 2
  // This is: tril(X) - 0.5 * diag(diag(X))
  // Implemented as: where(i > j, X, where(i == j, X/2, 0))
  const phi = phiLowerHalfDiag(tmp2, n);

  // L_dot = L @ phi - using matmul helper
  const L_dot = matmul2d((L as any).ref, phi, n);

  return [[L], [L_dot]];
}

// Helper: 2D matrix multiply using broadcast + mul + reduce
// For A (n, n) @ B (n, n) -> (n, n)
function matmul2d(a: Tracer, b: Tracer, n: number): Tracer {
  // Broadcast a to (n, n, n) by adding axis at end
  const aExp = broadcast(a, [n, n, n], [2]);
  // Broadcast b to (n, n, n) by transposing then adding axis at start
  // b: (n, n) -> b^T: (n, n) -> (n, n, n) with broadcast at axis 0
  const bT = b.transpose();
  const bExp = broadcast(bT, [n, n, n], [0]);
  // Multiply and reduce along axis 1
  const prod = aExp.mul(bExp);
  return reduce(prod, AluOp.Add, [1]);
}

// Helper: phi function for Cholesky JVP
// Takes lower triangular part and divides diagonal by 2
// phi(X) = tril(X) with diagonal multiplied by 0.5
function phiLowerHalfDiag(x: Tracer, n: number): Tracer {
  // For phi: result[i,j] = x[i,j] if i > j, x[i,j]/2 if i == j, 0 if i < j
  // phi(X) = where(lowerMask, where(diagMask, X/2, X), 0)

  // Create index arrays using array() which works at concrete level
  const indices = new Float32Array(n);
  for (let i = 0; i < n; i++) indices[i] = i;
  const idxArr = array(indices);

  // Row indices: (n,) -> (n, 1) -> broadcast to (n, n)
  const rowIdx = broadcast(reshape((idxArr as any).ref, [n, 1]), [n, n], [1]);

  // Col indices: (n,) -> (1, n) -> broadcast to (n, n)
  const colIdx = broadcast(reshape(idxArr as any, [1, n]), [n, n], [0]);

  // Lower triangular mask: row >= col
  const lowerMask = greaterEqual(rowIdx.ref, colIdx.ref);

  // Diagonal mask: row == col
  const diagMask = equal(rowIdx, colIdx);

  // phi(X) = where(lowerMask, where(diagMask, X/2, X), 0)
  const xHalf = x.ref.mul(0.5);
  const phiResult = where(lowerMask, where(diagMask, xHalf, x), 0);

  return phiResult;
}
