// Custom lowering for advanced operations that don't fit into AluExp.

import { DataArray, DType, dtypedArray } from "./alu";

/**
 * Advanced operations that don't fit into the `AluExp` compiler representation.
 *
 * Some routines like iterative matrix algorithms, FFTs, or sorting may not be
 * easy to express efficiently as a `Kernel` object. These also tend to be
 * somewhat expensive, so the benefit of kernel fusion and inlining is less
 * relevant.
 *
 * For these operations, we dispatch them as a custom operation on the backend,
 * which each backend implements in a specific way. These are listed in the
 * `Routines` enum below.
 *
 * Routines cannot be fused into other kernels and always operate on contiguous
 * arrays (default `ShapeTracker`).
 */
export class Routine {
  constructor(
    /** The name of the routine. */
    readonly name: Routines,
    /** Dtype and shape of the inputs and outputs. */
    readonly type: RoutineType,
    /** Extra parameters specific to the routine. */
    readonly params?: any,
  ) {}
}

/** One of the valid `Routine` that can be dispatched to backend. */
export enum Routines {
  /** Stable sorting algorithm along the last axis. */
  Sort = "Sort",

  /** Returns `int32` indices of the stably sorted array. */
  Argsort = "Argsort",

  /**
   * Solve a triangular system of equations.
   *
   * The first batch of inputs `A` should be of shape `[..., N, N]` and upper
   * triangular, while the second batch `B` should be of shape `[..., M, N]`.
   *
   * Solves for `X` in the equation `A @ X.T = B.T`, where `A` is the
   * triangular matrix. This is equivalent to `X = B @ A^-T`.
   */
  TriangularSolve = "TriangularSolve",

  /**
   * Cholesky decomposition of 2D positive semi-definite matrices.
   *
   * The input batch should be of shape `[..., N, N]`, and the output batch is
   * of the same shape, containing the lower-triangular matrix `L` such that
   * `A = L @ L.T`. Behavior is unspecified if A is not positive semi-definite.
   */
  Cholesky = "Cholesky",

  /**
   * LU decomposition of 2D rectangular matrices.
   *
   * The input is a batch of shape `[..., M, N]`, and the output is a tuple of
   * three arrays: `LU, Pivots, Permutation`.
   *
   * - `LU` is of shape `[..., M, N]`, containing the combined lower and upper
   *   triangular matrices. (lower triangular = implicit unit diagonal)
   * - `Pivots` is of shape `[..., min(M, N)]`, containing the row swaps.
   * - `Permutation` is of shape `[..., M]`, containing the permutation vector
   *   such that `P = eye(M).slice(Permutation)` -> `P @ A = L @ U`.
   */
  LU = "LU",
}

export interface RoutineType {
  inputShapes: number[][];
  inputDtypes: DType[];
  outputShapes: number[][];
  outputDtypes: DType[];
}

// Reference implementation of each routine in CPU is below.
//
// The remaining backends implement these routines within their own folders, to
// allow for code splitting between backends. This is for encapsulation.

export function runCpuRoutine(
  routine: Routine,
  inputs: Uint8Array<ArrayBuffer>[],
  outputs: Uint8Array<ArrayBuffer>[],
) {
  const { name, type } = routine;
  const inputAr = inputs.map((buf, i) => dtypedArray(type.inputDtypes[i], buf));
  const outputAr = outputs.map((buf, i) =>
    dtypedArray(type.outputDtypes[i], buf),
  );
  switch (name) {
    case Routines.Sort:
      return runSort(type, inputAr, outputAr);
    case Routines.Argsort:
      return runArgsort(type, inputAr, outputAr);
    case Routines.TriangularSolve:
      return runTriangularSolve(type, inputAr, outputAr, routine.params);
    case Routines.Cholesky:
      return runCholesky(type, inputAr, outputAr);
    case Routines.LU:
      return runLU(type, inputAr, outputAr);
    default:
      name satisfies never; // Exhaustiveness check
  }
}

function runSort(type: RoutineType, [x]: DataArray[], [y]: DataArray[]) {
  const xs = type.inputShapes[0];
  if (xs.length === 0) throw new Error("sort: cannot sort a scalar");
  const n = xs[xs.length - 1];
  y.set(x);
  for (let i = 0; i < y.length; i += n) {
    y.subarray(i, i + n).sort(); // In-place
  }
}

function runArgsort(type: RoutineType, [x]: DataArray[], [y, yi]: DataArray[]) {
  const xs = type.inputShapes[0];
  if (xs.length === 0) throw new Error("argsort: cannot sort a scalar");
  const n = xs[xs.length - 1];
  for (let offset = 0; offset < y.length; offset += n) {
    const ar = x.subarray(offset, offset + n);
    const out = y.subarray(offset, offset + n);
    const outi = yi.subarray(offset, offset + n);
    for (let i = 0; i < n; i++) outi[i] = i;
    outi.sort((a, b) => ar[a] - ar[b]);
    for (let i = 0; i < n; i++) out[i] = ar[outi[i]];
  }
}

function runTriangularSolve(
  type: RoutineType,
  [a, b]: DataArray[],
  [x]: DataArray[],
  { unitDiagonal }: { unitDiagonal: boolean },
) {
  const as = type.inputShapes[0];
  const bs = type.inputShapes[1];
  if (as.length < 2)
    throw new Error(`triangular_solve: a must be at least 2D, got ${as}`);
  if (bs.length < 2)
    throw new Error(`triangular_solve: b must be at least 2D, got ${bs}`);
  // Assuming that a is square, solve for a @ x.T = b.T
  const n = as[as.length - 2];
  if (n !== as[as.length - 1] || n !== bs[bs.length - 1])
    throw new Error(`triangular_solve: incompatible shapes a=${as}, b=${bs}`);
  const batch = bs[bs.length - 2];
  for (let counter = 0; counter < a.length / (n * n); counter++) {
    const a1 = a.subarray(counter * n * n, (counter + 1) * n * n);
    for (let t = 0; t < batch; t++) {
      const b1 = b.subarray(
        (counter * batch + t) * n,
        (counter * batch + t + 1) * n,
      );
      const x1 = x.subarray(
        (counter * batch + t) * n,
        (counter * batch + t + 1) * n,
      );
      // Now solve matvec a1 @ x1 = b1 for x1, where a1 is upper-triangular.
      for (let i = n - 1; i >= 0; i--) {
        let sum = b1[i];
        for (let j = i + 1; j < n; j++) {
          sum -= a1[i * n + j] * x1[j];
        }
        x1[i] = unitDiagonal ? sum : sum / a1[i * n + i];
      }
    }
  }
}

function runCholesky(type: RoutineType, [x]: DataArray[], [y]: DataArray[]) {
  const xs = type.inputShapes[0];
  if (xs.length < 2) throw new Error("cholesky: input must be at least 2D");
  const n = xs[xs.length - 2];
  const m = xs[xs.length - 1];
  if (n !== m)
    throw new Error(`cholesky: input must be square, got [${n}, ${m}]`);

  for (let offset = 0; offset < y.length; offset += n * n) {
    const ar = x.subarray(offset, offset + n * n);
    const out = y.subarray(offset, offset + n * n);
    // Cholesky-Banachiewicz algorithm: compute lower triangular L where A = L * L^T
    // https://en.wikipedia.org/wiki/Cholesky_decomposition#Computation
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = ar[i * n + j];
        for (let k = 0; k < j; k++) {
          sum -= out[i * n + k] * out[j * n + k];
        }
        out[i * n + j] = i === j ? Math.sqrt(sum) : sum / out[j * n + j];
      }
    }
  }
}

function runLU(
  type: RoutineType,
  [a]: DataArray[],
  [lu, pivots, perm]: DataArray[],
) {
  const shape = type.inputShapes[0];
  if (shape.length < 2) throw new Error("lu: input must be at least 2D");
  const m = shape[shape.length - 2]; // rows
  const n = shape[shape.length - 1]; // cols
  const r = Math.min(m, n);

  for (let offset = 0; offset < a.length; offset += m * n) {
    const ar = a.subarray(offset, offset + m * n);
    const out = lu.subarray(offset, offset + m * n);
    const batchIdx = offset / (m * n);
    const piv = pivots.subarray(batchIdx * r, (batchIdx + 1) * r);
    const p = perm.subarray(batchIdx * m, (batchIdx + 1) * m);

    out.set(ar);
    for (let i = 0; i < m; i++) p[i] = i;

    for (let j = 0; j < r; j++) {
      // Partial pivoting on column j
      let maxVal = Math.abs(out[j * n + j]);
      let maxRow = j;
      for (let i = j + 1; i < m; i++) {
        const val = Math.abs(out[i * n + j]);
        if (val > maxVal) {
          maxVal = val;
          maxRow = i;
        }
      }
      piv[j] = maxRow;
      if (maxRow !== j) {
        for (let col = 0; col < n; col++) {
          const tmp = out[j * n + col];
          out[j * n + col] = out[maxRow * n + col];
          out[maxRow * n + col] = tmp;
        }
        const tmpP = p[j];
        p[j] = p[maxRow];
        p[maxRow] = tmpP;
      }

      // Update L[j+1:,j] and U[j+1:,j+1:] matrices
      const diag = out[j * n + j];
      if (diag !== 0) {
        for (let i = j + 1; i < m; i++) {
          const factor = out[i * n + j] / diag;
          out[i * n + j] = factor; // L
          for (let col = j + 1; col < n; col++)
            out[i * n + col] -= factor * out[j * n + col];
        }
      }
    }
  }
}
