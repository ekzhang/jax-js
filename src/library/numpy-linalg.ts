import * as lax from "./lax";
import { triangularSolve } from "./lax-linalg";
import * as np from "./numpy";
import { Array, ArrayLike, fudgeArray } from "../frontend/array";
import { generalBroadcast } from "../utils";

function checkSquare(name: string, a: Array) {
  if (a.ndim < 2 || a.shape[a.ndim - 1] !== a.shape[a.ndim - 2]) {
    throw new Error(
      `${name}: input must be at least 2D square matrix, got ${a.aval}`,
    );
  }
  return a.shape[a.ndim - 1];
}

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
  checkSquare("cholesky", a);
  if (symmetrizeInput) {
    a = a.ref.add(np.matrixTranspose(a)).mul(0.5);
  }
  return lax.linalg.cholesky(a, { upper });
}

/** Compute the determinant of a square matrix (batched). */
export function det(a: ArrayLike): Array {
  a = fudgeArray(a);
  const n = checkSquare("det", a);
  const [lu, pivots, permutation] = lax.linalg.lu(a);
  permutation.dispose();

  const parity = pivots.notEqual(np.arange(n)).astype(np.int32).sum(-1).mod(2);
  const sign = parity.mul(-2).add(1); // (-1)^parity
  const diag = lu.diagonal(0, -1, -2);
  return np.prod(diag, -1).mul(sign);
}

export { diagonal } from "./numpy";

/** Compute the inverse of a square matrix (batched). */
export function inv(a: ArrayLike): Array {
  a = fudgeArray(a);
  const n = checkSquare("inv", a);
  return solve(a, np.eye(n));
}

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
  const at = np.matrixTranspose(a.ref);
  if (m <= n) {
    // Underdetermined or square system: A.T @ (A @ A.T)^-1 @ B
    const aat = np.matmul(a, at.ref); // A @ A.T, shape (M, M)
    const l = cholesky(aat, { symmetrizeInput: false }); // L @ L.T = A @ A.T
    const lb = triangularSolve(l.ref, b, { leftSide: true, lower: true }); // L^-1 @ B
    const llb = triangularSolve(l, lb, {
      leftSide: true,
      lower: true,
      transposeA: true,
    }); // (A @ A.T)^-1 @ B
    return np.matmul(at, llb); // A.T @ (A @ A.T)^-1 @ B
  } else {
    // Overdetermined system: (A.T @ A)^-1 @ A.T @ B
    const ata = np.matmul(at.ref, a); // A.T @ A, shape (N, N)
    const l = cholesky(ata, { symmetrizeInput: false }); // L @ L.T = A.T @ A
    const atb = np.matmul(at, b); // A.T @ B
    const lb = triangularSolve(l.ref, atb, { leftSide: true, lower: true }); // L^-1 @ A.T @ B
    const llb = triangularSolve(l, lb, {
      leftSide: true,
      lower: true,
      transposeA: true,
    }); // (A.T @ A)^-1 @ A.T @ B
    return llb;
  }
}

export { matmul } from "./numpy";

/** Raise a square matrix to an integer power, via repeated squarings. */
export function matrixPower(a: ArrayLike, n: number): Array {
  if (!Number.isInteger(n))
    throw new Error(`matrixPower: exponent must be an integer, got ${n}`);
  a = fudgeArray(a);
  const m = checkSquare("matrixPower", a);
  if (n === 0) {
    a.dispose();
    return np.broadcastTo(np.eye(m), a.shape);
  }
  if (n < 0) {
    a = inv(a);
    n = -n;
  }
  let result: Array | null = null;
  let a2k = a; // a^(2^k)
  for (let k = 0; n; k++) {
    if (k > 0) a2k = np.matmul(a2k.ref, a2k);
    if (n % 2 === 1)
      result = result === null ? a2k.ref : np.matmul(result, a2k.ref);
    n = Math.floor(n / 2);
  }
  a2k.dispose();
  return result!;
}

export { matrixTranspose } from "./numpy";
export { outer } from "./numpy";

/** Return sign and natural logarithm of the determinant of `a`. */
export function slogdet(a: ArrayLike): [Array, Array] {
  a = fudgeArray(a);
  const n = checkSquare("slogdet", a);
  const [lu, pivots, permutation] = lax.linalg.lu(a);
  permutation.dispose();

  let parity = pivots.notEqual(np.arange(n)).astype(np.int32).sum(-1);
  const diag = lu.diagonal(0, -1, -2);
  parity = parity.add(diag.ref.less(0).astype(np.int32).sum(-1)).mod(2);
  const logabsdet = np.log(np.abs(diag)).sum(-1);
  const sign = parity.mul(-2).add(1); // (-1)^parity
  return [sign, logabsdet];
}

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
  const n = checkSquare("solve", a);
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
  a = np.broadcastTo(a, [...batchDims, n, n]);
  b = np.broadcastTo(b, [...batchDims, n, m]);

  // Compute the LU decomposition with partial pivoting.
  const [lu, pivots, permutation] = lax.linalg.lu(a);
  pivots.dispose();

  // L @ U @ x = P @ b
  const P = np
    .arange(n)
    .equal(permutation.reshape([...permutation.shape, 1]))
    .astype(b.dtype);
  const LPb = triangularSolve(lu.ref, np.matmul(P, b), {
    leftSide: true,
    lower: true,
    unitDiagonal: true,
  });
  let x = triangularSolve(lu, LPb, { leftSide: true, lower: false });
  if (bIs1d) {
    x = np.squeeze(x, -1);
  }
  return x;
}

export { tensordot } from "./numpy";
export { trace } from "./numpy";
export { vecdot } from "./numpy";
