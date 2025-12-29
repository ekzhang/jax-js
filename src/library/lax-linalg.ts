// Linear algebra functions, mirroring `jax.lax.linalg`.

import { moveaxis } from "./numpy";
import { Array, type ArrayLike, fudgeArray } from "../frontend/array";
import * as core from "../frontend/core";

/**
 * Compute the Cholesky decomposition of a symmetric positive-definite matrix.
 *
 * The Cholesky decomposition of a matrix `A` is:
 *
 * - A = L @ L^T  (for upper=false, default)
 * - A = U^T @ U  (for upper=true)
 *
 * where `L` is a lower-triangular matrix and `U` is an upper-triangular matrix.
 * The input matrix must be symmetric and positive-definite.
 *
 * @example
 * ```ts
 * import { lax, numpy as np } from "@jax-js/jax";
 *
 * const x = np.array([[2., 1.], [1., 2.]]);
 *
 * // Lower Cholesky factorization (default):
 * const L = lax.linalg.cholesky(x);
 * // L ≈ [[1.4142135, 0], [0.70710677, 1.2247449]]
 *
 * // Upper Cholesky factorization:
 * const U = lax.linalg.cholesky(x, { upper: true });
 * // U ≈ [[1.4142135, 0.70710677], [0, 1.2247449]]
 * ```
 */
export function cholesky(
  a: ArrayLike,
  { upper = false }: { upper?: boolean } = {},
): Array {
  const L = core.cholesky(a) as Array;
  return upper ? moveaxis(L, -2, -1) : L;
}

/**
 * Solve a triangular linear system.
 *
 * Solves `a @ x = b` (if leftSide=true) or `x @ a = b` (if leftSide=false)
 * where `a` is a triangular matrix.
 *
 * @example
 * ```ts
 * import { lax, numpy as np } from "@jax-js/jax";
 *
 * const L = np.array([[2., 0.], [1., 3.]]);
 * const b = np.array([4., 7.]).reshape([2, 1]);
 *
 * // Solve L @ x = b
 * const x = lax.linalg.triangularSolve(L, b, { leftSide: true, lower: true });
 * // x = [[2.], [5./3.]]
 * ```
 */
export function triangularSolve(
  a: ArrayLike,
  b: ArrayLike,
  {
    leftSide = false,
    lower = false,
    transposeA = false,
    unitDiagonal = false,
  }: {
    leftSide?: boolean;
    lower?: boolean;
    transposeA?: boolean;
    unitDiagonal?: boolean;
  } = {},
): Array {
  a = fudgeArray(a);
  b = fudgeArray(b);
  if (!leftSide) {
    // Transpose everything so it becomes a left-side solve.
    // Note that the `TriangularSolve` primitive automatically transposes the
    // b and x (output) values.
    transposeA = !transposeA;
  } else {
    b = moveaxis(b, -2, -1);
  }
  if (transposeA) a = moveaxis(a, -2, -1);
  let x = core.triangularSolve(a, b, { lower, unitDiagonal }) as Array;
  if (leftSide) x = moveaxis(x, -2, -1);
  return x;
}
