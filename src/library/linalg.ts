// Linear algebra functions, mirroring `jax.numpy.linalg` and `jax.scipy.linalg`.

import { type Array, type ArrayLike, fudgeArray } from "../frontend/array";
import * as np from "./numpy";

/**
 * Compute the Cholesky decomposition of a matrix.
 *
 * The Cholesky decomposition of a matrix `A` is:
 *
 * A = L @ L^T  (for lower=true)
 * A = U^T @ U  (for lower=false, default)
 *
 * where `L` is a lower-triangular matrix and `U` is an upper-triangular matrix.
 *
 * Args:
 *   a: input array, representing a positive-definite hermitian matrix.
 *      Must have shape ``(N, N)``.
 *   lower: if true, compute the lower Cholesky decomposition `L`. if false
 *          (default), compute the upper Cholesky decomposition `U`.
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
 * // Lower Cholesky factorization:
 * const L = cholesky(x, { lower: true });
 * // L ≈ [[1.4142135, 0], [0.70710677, 1.2247449]]
 *
 * // Upper Cholesky factorization:
 * const U = cholesky(x); // lower=false is default
 * // U ≈ [[1.4142135, 0.70710677], [0, 1.2247449]]
 * ```
 */
export function cholesky(
  a: ArrayLike,
  { lower = false }: { lower?: boolean } = {},
): Array {
  a = fudgeArray(a);
  if (a.ndim !== 2) {
    throw new TypeError(`cholesky: input must be 2D, got ${a.ndim}D`);
  }
  if (a.shape[0] !== a.shape[1]) {
    throw new TypeError(
      `cholesky: matrix must be square, got ${a.shape[0]}x${a.shape[1]}`,
    );
  }

  // For a working version without native backend support, we compute
  // the decomposition on the CPU using dataSync()
  const data = a.dataSync();
  const n = a.shape[0];
  const result = new Float32Array(n * n);

  // Standard Cholesky-Crout algorithm (CPU version)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += result[i * n + k] * result[j * n + k];
      }

      if (i === j) {
        // Diagonal element
        const diag = data[i * n + i] - sum;
        result[i * n + i] = diag > 0 ? Math.sqrt(diag) : NaN;
      } else {
        // Off-diagonal element
        result[i * n + j] =
          (1.0 / result[j * n + j]) * (data[i * n + j] - sum);
      }
    }
  }

  // Create output array from CPU result
  const out = np.array(result as any, {
    dtype: a.dtype,
    shape: [n, n],
    device: a.device,
  });

  if (lower) {
    return out;
  } else {
    // For upper triangular, return transpose of L
    return out.transpose();
  }
}

// Re-export commonly used functions from numpy for convenience
export { tril, triu } from "./numpy";
