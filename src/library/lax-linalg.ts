// Linear algebra functions, mirroring `jax.lax.linalg`.

import * as np from "./numpy";
import { DType, isFloatDtype } from "../alu";
import { Array, type ArrayLike, fudgeArray } from "../frontend/array";
import * as core from "../frontend/core";
import { jit } from "../frontend/jaxpr";
import { checkSquare } from "../utils";

const JsArray = globalThis.Array;

type JacobiRoundPlan = {
  p: number[];
  q: number[];
  active: boolean[];
  mate: number[];
  pairId: number[];
  sign: number[];
};

function jacobiRoundPlans(n: number): JacobiRoundPlan[] {
  if (n < 2) return [];

  const participants = JsArray.from({ length: n }, (_, i) => i);
  if (n % 2 === 1) participants.push(-1);

  const rounds: JacobiRoundPlan[] = [];
  const m = participants.length;
  const half = m / 2;
  const rotating = participants.slice();

  for (let round = 0; round < m - 1; round++) {
    const mate = JsArray.from({ length: n }, (_, i) => i);
    const pairId = JsArray(n);
    const sign = JsArray(n);
    const p: number[] = [];
    const q: number[] = [];
    const active: boolean[] = [];

    for (let k = 0; k < half; k++) {
      let a = rotating[k];
      let b = rotating[m - 1 - k];
      if (a === -1 || b === -1) {
        const i = a === -1 ? b : a;
        p.push(i);
        q.push(i);
        active.push(false);
        pairId[i] = k;
        sign[i] = 0;
      } else {
        if (a > b) [a, b] = [b, a];
        p.push(a);
        q.push(b);
        active.push(true);
        mate[a] = b;
        mate[b] = a;
        pairId[a] = k;
        pairId[b] = k;
        sign[a] = -1;
        sign[b] = 1;
      }
    }

    rounds.push({ p, q, active, mate, pairId, sign });

    const tail = rotating.pop()!;
    rotating.splice(1, 0, tail);
  }

  return rounds;
}

function jacobiSweepPlanRows(n: number): number[][] {
  return jacobiRoundPlans(n).map(({ p, q, active, pairId, mate, sign }) => [
    ...p,
    ...q,
    ...active.map(Number),
    ...pairId,
    ...mate,
    ...sign,
  ]);
}

function applyJacobiRound(
  a: Array,
  v: Array,
  p: Array,
  q: Array,
  pairId: Array,
  mate: Array,
  sign: Array,
  pairActive: Array,
): [Array, Array] {
  const n = a.shape[a.ndim - 1];
  const batchShape = a.shape.slice(0, -2);
  const batchRank = a.ndim - 2;
  const batchIndex = JsArray.from({ length: batchRank }, () => [] as []);
  const app = a.ref.slice(...batchIndex, p.ref, p.ref);
  const aqq = a.ref.slice(...batchIndex, q.ref, q.ref);
  const apq = a.ref.slice(...batchIndex, p, q);

  const active = np.logicalAnd(pairActive, np.abs(apq.ref).greater(0));
  const safeApq = np.where(active.ref, apq, 1);
  const tau = aqq.sub(app).div(safeApq.mul(2));
  const tauSign = np.where(tau.ref.greaterEqual(0), 1, -1);
  const t = tauSign.div(np.abs(tau.ref).add(np.sqrt(tau.ref.mul(tau).add(1))));
  const cRaw = np.reciprocal(np.sqrt(t.ref.mul(t.ref).add(1)));
  const sRaw = t.mul(cRaw.ref);
  const cPair = np.where(active.ref, cRaw, 1);
  const sPair = np.where(active, sRaw, 0);

  const cIdx = np.take(cPair, pairId.ref, -1);
  const sIdx = np.take(sPair, pairId, -1).mul(sign);
  const cRow = cIdx.ref.reshape([...batchShape, n, 1]);
  const cCol = cIdx.reshape([...batchShape, 1, n]);
  const sRow = sIdx.ref.reshape([...batchShape, n, 1]);
  const sCol = sIdx.reshape([...batchShape, 1, n]);

  const term1 = cRow.ref.mul(cCol.ref).mul(a.ref);
  const term2 = cRow.mul(sCol.ref).mul(np.take(a.ref, mate.ref, -1));
  const term3 = sRow.ref.mul(cCol.ref).mul(np.take(a.ref, mate.ref, -2));
  const term4 = sRow
    .mul(sCol.ref)
    .mul(np.take(np.take(a, mate.ref, -2), mate.ref, -1));
  const nextA = term1.add(term2).add(term3).add(term4);
  const nextV = v.ref.mul(cCol).add(np.take(v, mate, -1).mul(sCol));

  return [nextA, nextV];
}

const applyJacobiSweepJit = jit(function applyJacobiSweepJit(
  a: Array,
  v: Array,
  plan: Array,
): [Array, Array] {
  const n = a.shape[a.ndim - 1];
  const half = Math.ceil(n / 2);
  const splitPoints = [
    half,
    2 * half,
    3 * half,
    3 * half + n,
    3 * half + 2 * n,
  ];
  for (let round = 0; round < plan.shape[0]; round++) {
    const [p, q, active, pairId, mate, sign] = np.split(
      plan.ref.slice(round),
      splitPoints,
      -1,
    );
    [a, v] = applyJacobiRound(
      a,
      v,
      p,
      q,
      pairId,
      mate,
      sign.astype(a.dtype),
      active.astype(np.bool),
    );
  }
  return [a, v];
});

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
 * This uses a fixed-sweep cyclic Jacobi method. It does not stop early based on
 * convergence, which avoids synchronizing GPU-computed residuals back to JS.
 * Eigenvectors are returned as columns in the first result, and eigenvalues are
 * returned in ascending order in the second result.
 */
export function eigh(
  x: ArrayLike,
  {
    lower = true,
    symmetrizeInput = true,
  }: {
    lower?: boolean;
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
  } else if (lower) {
    const lowerTriangle = np.tril(x.ref);
    x = lowerTriangle.ref.add(np.matrixTranspose(np.tril(x, -1)));
  } else {
    const upperTriangle = np.triu(x.ref);
    x = upperTriangle.ref.add(np.matrixTranspose(np.triu(x, 1)));
  }

  const batchShape = x.shape.slice(0, -2);
  let v = np.broadcastTo(
    np.eye(n, undefined, { dtype: x.dtype, device: x.device }),
    x.shape,
  );
  const plan = np.array(jacobiSweepPlanRows(n), {
    dtype: np.int32,
    device: x.device,
  });
  const sweeps = Math.max(8, 2 * n);
  for (let sweep = 0; sweep < sweeps; sweep++) {
    [x, v] = applyJacobiSweepJit(x, v, plan.ref);
  }
  plan.dispose();

  const valuesUnsorted = np.diagonal(x, 0, -2, -1);
  const order = np.argsort(valuesUnsorted.ref);
  const values = np.takeAlongAxis(valuesUnsorted, order.ref, -1);
  const vectors = np.takeAlongAxis(v, order.reshape([...batchShape, 1, n]), -1);
  return [vectors, values];
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
