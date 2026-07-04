// Linear algebra functions, mirroring `jax.lax.linalg`.

import * as np from "./numpy";
import { DType, isFloatDtype } from "../alu";
import { Array, type ArrayLike, fudgeArray } from "../frontend/array";
import * as core from "../frontend/core";
import { checkSquare } from "../utils";

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
  return upper ? np.moveaxis(L, -2, -1) : L;
}

/**
 * Eigendecomposition of real symmetric matrices.
 *
 * This uses a cyclic Jacobi routine with an internal convergence check and a
 * fixed maximum number of sweeps.
 *
 * Eigenvectors are returned as columns in the first result, and eigenvalues are
 * returned in ascending order in the second result.
 */
export function eigh(
  x: ArrayLike,
  {
    lower = true,
    sortEigenvalues = true,
    symmetrizeInput = true,
  }: {
    lower?: boolean;
    sortEigenvalues?: boolean;
    symmetrizeInput?: boolean;
  } = {},
): [Array, Array] {
  x = fudgeArray(x);
  const n = checkSquare("eigh", x.shape);
  if (!isFloatDtype(x.dtype) || x.dtype === DType.Float16) {
    x = x.astype(np.float32);
  }
  if (symmetrizeInput) {
    x = x.ref.add(np.matrixTranspose(x)).mul(0.5);
  } else if (!lower) {
    x = np.matrixTranspose(x);
  }

  const batchShape = x.shape.slice(0, -2);
  let v: Array;
  [x, v] = core.jacobiEigh(x, {
    maxSweeps: Math.max(8, 2 * n), // Default maximum number of sweeps
    tolerance: x.dtype === DType.Float64 ? 1e-12 : 1e-6,
  }) as [Array, Array];

  const valuesUnsorted = np.diagonal(x, 0, -2, -1);
  if (!sortEigenvalues) return [v, valuesUnsorted];

  const order = np.argsort(valuesUnsorted.ref);
  const values = np.takeAlongAxis(valuesUnsorted, order.ref, -1);
  const vectors = np.takeAlongAxis(v, order.reshape([...batchShape, 1, n]), -1);
  return [vectors, values];
}

/**
 * Singular value decomposition of real matrices.
 *
 * This computes a thin SVD using a symmetric eigendecomposition of `A.T @ A`
 * or `A @ A.T`. It is intended as a baseline implementation; it is less
 * numerically stable than a dedicated SVD routine for ill-conditioned inputs.
 *
 * With `computeUv: true`, returns `[u, s, vh]` such that
 * `a ~= u @ diag(s) @ vh`. Only `fullMatrices: false` is supported for
 * non-square matrices.
 */
export function svd(
  a: ArrayLike,
  {
    computeUv = true,
    fullMatrices = false,
  }: {
    computeUv?: boolean;
    fullMatrices?: boolean;
  } = {},
): [Array, Array, Array] | Array {
  a = fudgeArray(a);
  if (a.ndim < 2) throw new Error(`svd: input must be at least 2D, got ${a}`);
  if (!isFloatDtype(a.dtype) || a.dtype === DType.Float16) {
    a = a.astype(np.float32);
  }

  const [m, n] = a.shape.slice(-2);
  if (fullMatrices && m !== n) {
    throw new Error(
      "svd: fullMatrices=true is only supported for square input",
    );
  }

  const batchShape = a.shape.slice(0, -2);
  const k = Math.min(m, n);

  const singularValues = (values: Array) =>
    np.sqrt(np.maximum(np.flip(values, -1), 0));
  const invSingularValues = (s: Array) =>
    np.where(np.greater(s.ref, 0), np.reciprocal(s.ref), 0);

  if (m >= n) {
    const gram = np.matmul(np.matrixTranspose(a.ref), a.ref);
    const [v, values] = eigh(gram, { symmetrizeInput: false });
    const s = singularValues(values);
    if (!computeUv) {
      v.dispose();
      return s;
    }

    const vDesc = np.flip(v, -1);
    const invS = invSingularValues(s.ref);
    const u = np.matmul(a, vDesc.ref).mul(invS.reshape([...batchShape, 1, k]));
    const vh = np.matrixTranspose(vDesc);
    return [u, s, vh];
  } else {
    const gram = np.matmul(a.ref, np.matrixTranspose(a.ref));
    const [u, values] = eigh(gram, { symmetrizeInput: false });
    const s = singularValues(values);
    if (!computeUv) {
      u.dispose();
      return s;
    }

    const uDesc = np.flip(u, -1);
    const invS = invSingularValues(s.ref);
    const vh = np
      .matmul(np.matrixTranspose(uDesc.ref), a)
      .mul(invS.reshape([...batchShape, k, 1]));
    return [uDesc, s, vh];
  }
}

/**
 * LU decomposition with partial pivoting.
 *
 * Computes the matrix decomposition: `P @ A = L @ U`, where `P` is a
 * permutation of the rows of `A`, `L` is lower-triangular with unit diagonal,
 * and `U` is upper-triangular.
 *
 * @param x - A batch of matrices with shape `[..., m, n]`.
 *
 * @returns A tuple `(lu, pivots, permutation)` where:
 * - `lu`: combined lower and upper triangular matrices.
 * - `pivots`: an array of pivot indices with shape `[..., min(m, n)]`.
 * - `permutation`: the permutation generated by pivots with shape `[..., m]`.
 *
 * @example
 * ```ts
 * import { lax, numpy as np } from "@jax-js/jax";
 *
 * const A = np.array([[4., 3.], [6., 3.]]);
 * const [lu, pivots, permutation] = lax.linalg.lu(A);
 * // lu ≈ [[6., 3.], [0.6666667, 1.0]]
 * // pivots = [1, 1]
 * // permutation = [1, 0]
 * ```
 */
export function lu(x: ArrayLike): [Array, Array, Array] {
  return core.lu(x) as [Array, Array, Array];
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
    b = np.moveaxis(b, -2, -1);
  }
  if (transposeA) {
    a = np.moveaxis(a, -2, -1);
    lower = !lower;
  }
  let x = core.triangularSolve(a, b, { lower, unitDiagonal }) as Array;
  if (leftSide) x = np.moveaxis(x, -2, -1);
  return x;
}
