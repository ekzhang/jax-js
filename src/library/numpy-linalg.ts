import * as lax from "./lax";
import { triangularSolve } from "./lax-linalg";
import {
  Array,
  ArrayLike,
  broadcastTo,
  matmul,
  matrixTranspose,
  squeeze,
  take,
} from "./numpy";
import { fudgeArray } from "../frontend/array";
import { vmap } from "../frontend/vmap";
import { generalBroadcast } from "../utils";

/**
 * Compute the Cholesky decomposition of a (batched) positive-definite matrix.
 *
 * This is like `jax.lax.linalg.cholesky()`, except with an option to symmetrize
 * the input matrix, which is on by default.
 */
export function cholesky(
  a: ArrayLike,
  {
    upper = false,
    symmetrizeInput = true,
  }: {
    upper?: boolean;
    symmetrizeInput?: boolean;
  } = {},
): Array {
  a = fudgeArray(a);
  if (a.ndim < 2 || a.shape[a.ndim - 1] !== a.shape[a.ndim - 2]) {
    throw new Error(
      `cholesky: input must be at least 2D square matrix, got ${a.aval}`,
    );
  }
  if (symmetrizeInput) {
    a = a.ref.add(matrixTranspose(a)).mul(0.5);
  }
  return lax.linalg.cholesky(a, { upper });
}

export { diagonal } from "./numpy";

/**
 * Return the least-squares solution to a linear equation.
 *
 * For overdetermined systems, this finds the `x` that minimizes `norm(ax - b)`.
 * For underdetermined systems, this finds the minimum-norm solution for `x`.
 *
 * This currently uses Cholesky decomposition to solve the normal equations,
 * under the hood. The method is not as robust as QR or SVD.
 *
 * @param a coefficient matrix of shape `(M, N)`
 * @param b right-hand side of shape `(M,)` or `(M, K)`
 * @return least-squares solution of shape `(N,)` or `(N, K)`
 */
export function lstsq(a: ArrayLike, b: ArrayLike): Array {
  a = fudgeArray(a);
  b = fudgeArray(b);
  if (a.ndim !== 2)
    throw new Error(`lstsq: 'a' must be a 2D array, got ${a.aval}`);
  const [m, n] = a.shape;
  if (b.shape[0] !== m)
    throw new Error(
      `lstsq: leading dimension of 'b' must match number of rows of 'a', got ${b.aval}`,
    );
  const at = matrixTranspose(a.ref);
  if (m <= n) {
    // Underdetermined or square system: A.T @ (A @ A.T)^-1 @ B
    const aat = matmul(a, at.ref); // A @ A.T, shape (M, M)
    const l = cholesky(aat, { symmetrizeInput: false }); // L @ L.T = A @ A.T
    const lb = triangularSolve(l.ref, b, { leftSide: true, lower: true }); // L^-1 @ B
    const llb = triangularSolve(l, lb, { leftSide: true, transposeA: true }); // (A @ A.T)^-1 @ B
    return matmul(at, llb.ref); // A.T @ (A @ A.T)^-1 @ B
  } else {
    // Overdetermined system: (A.T @ A)^-1 @ A.T @ B
    const ata = matmul(at.ref, a); // A.T @ A, shape (N, N)
    const l = cholesky(ata, { symmetrizeInput: false }); // L @ L.T = A.T @ A
    const atb = matmul(at, b); // A.T @ B
    const lb = triangularSolve(l.ref, atb, { leftSide: true, lower: true }); // L^-1 @ A.T @ B
    const llb = triangularSolve(l, lb, { leftSide: true, transposeA: true }); // (A.T @ A)^-1 @ A.T @ B
    return llb;
  }
}

export { matmul } from "./numpy";
export { matrixTranspose } from "./numpy";
export { outer } from "./numpy";

/**
 * Solve a linear system of equations.
 *
 * This solves a (batched) linear system of equations `a @ x = b` for `x` given
 * `a` and `b`. If `a` is singular, this will return `nan` or `inf` values.
 *
 * @param a - Coefficient matrix of shape `(..., N, N)`.
 * @param b - Values of shape `(N,)` or `(..., N, M)`.
 * @returns Solution `x` of shape `(..., N)` or `(..., N, M)`.
 */
export function solve(a: ArrayLike, b: ArrayLike): Array {
  a = fudgeArray(a);
  b = fudgeArray(b);
  if (a.ndim < 2)
    throw new Error(`solve: a must be at least 2D, got ${a.aval}`);
  const [n, n2] = a.shape.slice(-2);
  if (n !== n2) throw new Error(`solve: a must be square, got ${a.aval}`);
  if (b.ndim === 0) throw new Error(`solve: b cannot be scalar`);
  const bIs1d = b.ndim === 1;
  if (bIs1d) {
    b = b.reshape([...b.shape, 1]); // We'll remove this at the end.
  }
  if (b.shape[b.ndim - 2] !== n) {
    throw new Error(
      `solve: leading dimension of b must match size of a, got a=${a.aval}, b=${b.aval}`,
    );
  }
  const m = b.shape[b.ndim - 1];
  const batchDims = generalBroadcast(
    a.shape.slice(0, -2),
    b.shape.slice(0, -2),
  );
  a = broadcastTo(a, [...batchDims, n, n]);
  b = broadcastTo(b, [...batchDims, n, m]);

  // Compute the LU decomposition with partial pivoting.
  const [lu, pivots, permutation] = lax.linalg.lu(a);
  pivots.dispose();

  // L @ U @ x = P @ b
  const Pb = (
    vmap((x: Array, y: Array) => take(x, y, -2), [0, 0])(
      b.reshape([-1, n, m]),
      permutation.reshape([-1, n]),
    ) as Array
  ).reshape([...batchDims, n, m]); // Janky vmap until we get `takeAlongAxis()`
  const LPb = triangularSolve(lu.ref, Pb, {
    leftSide: true,
    lower: true,
    unitDiagonal: true,
  });
  let x = triangularSolve(lu, LPb.ref, { leftSide: true, lower: false });
  if (bIs1d) {
    x = squeeze(x, -1);
  }
  return x;
}

export { tensordot } from "./numpy";
export { trace } from "./numpy";
export { vecdot } from "./numpy";
