// Linear algebra functions, mirroring `jax.numpy.linalg` and `jax.scipy.linalg`.

import { type Array, type ArrayLike, fudgeArray } from "../frontend/array";
import * as core from "../frontend/core";
// Import custom ops to ensure they're registered
import "../custom-ops/cholesky.js";

/**
 * Compute the Cholesky decomposition of a matrix.
 *
 * The Cholesky decomposition of a matrix `A` is:
 *
 * A = L @ L^T  (for lower=true, default)
 * A = U^T @ U  (for lower=false)
 *
 * where `L` is a lower-triangular matrix and `U` is an upper-triangular matrix.
 *
 * Args:
 *   a: input array, representing a positive-definite hermitian matrix.
 *      Must have shape ``(N, N)``.
 *   lower: if true (default), compute the lower Cholesky decomposition `L`.
 *          if false, compute the upper Cholesky decomposition `U`.
 *
 * Returns:
 *   array of shape ``(N, N)`` representing the cholesky decomposition
 *   of the input.
 *
 * @example
 * ```ts
 * import { array } from "@jax-js/jax";
 * import { cholesky } from "@jax-js/jax/linalg";
 *
 * const x = array([[2., 1.], [1., 2.]]);
 *
 * // Lower Cholesky factorization (default):
 * const L = cholesky(x);
 * // L ≈ [[1.4142135, 0], [0.70710677, 1.2247449]]
 *
 * // Upper Cholesky factorization:
 * const U = cholesky(x, { lower: false });
 * // U ≈ [[1.4142135, 0.70710677], [0, 1.2247449]]
 * ```
 */
export function cholesky(
  a: ArrayLike,
  { lower = true }: { lower?: boolean } = {},
): Array {
  a = fudgeArray(a);

  // Use CustomOp primitive to dispatch to backend-specific implementation
  const result = core.bind1(core.Primitive.CustomOp, [a], {
    name: "linalg.cholesky",
    lower,
  }) as Array;

  return result;
}

/**
 * Solve a triangular linear system.
 *
 * Solves `a @ x = b` (if leftSide=true) or `x @ a = b` (if leftSide=false)
 * where `a` is a triangular matrix.
 *
 * Args:
 *   a: input array, representing a triangular matrix. Must have shape ``(N, N)``.
 *   b: input array, the right-hand side. Shape ``(N,)`` or ``(N, M)``.
 *   leftSide: if true (default), solve a @ x = b. if false, solve x @ a = b.
 *   lower: if true (default), a is lower triangular. if false, a is upper triangular.
 *   transposeA: if true, solve with the transpose of a. Default false.
 *   unitDiagonal: if true, assume diagonal elements of a are 1. Default false.
 *
 * Returns:
 *   array of same shape as b, the solution x.
 *
 * @example
 * ```ts
 * import { array } from "@jax-js/jax";
 * import { triangular_solve } from "@jax-js/jax/linalg";
 *
 * const L = array([[2., 0.], [1., 3.]]);
 * const b = array([4., 7.]);
 *
 * // Solve L @ x = b
 * const x = triangular_solve(L, b);
 * // x = [2., 5./3.]
 * ```
 */
export function triangular_solve(
  a: ArrayLike,
  b: ArrayLike,
  {
    leftSide = true,
    lower = true,
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
  return core.triangularSolve(a, b, {
    leftSide,
    lower,
    transposeA,
    unitDiagonal,
  }) as Array;
}

// Alias for scipy.linalg compatibility
export { triangular_solve as solve_triangular };

// Re-export commonly used functions from numpy for convenience
export { tril, triu } from "./numpy";
