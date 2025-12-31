import * as lax from "./lax";
import { Array, ArrayLike, matrixTranspose } from "./numpy";
import { fudgeArray } from "../frontend/array";

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
export { matmul } from "./numpy";
export { matrixTranspose } from "./numpy";
export { outer } from "./numpy";
export { tensordot } from "./numpy";
export { trace } from "./numpy";
export { vecdot } from "./numpy";
