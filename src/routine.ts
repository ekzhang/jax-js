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
  /**
   * Sort along the last axis.
   *
   * This may be _unstable_ but it often doesn't matter, sorting numbers is
   * bitwise unique up to signed zeros and NaNs.
   */
  Sort = "Sort",

  /** Stable sorting, returns `int32` indices and values of the sorted array. */
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

  /**
   * Singular value decomposition of 2D matrices.
   *
   * The input is a batch of shape `[..., M, N]`. With `computeUv`, the output is
   * `U, S, Vh`; otherwise the output is only `S`.
   */
  SVD = "SVD",
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
    case Routines.SVD:
      return runSVD(type, inputAr, outputAr, routine.params);
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
    outi.sort((a, b) => {
      // Special cases: NaNs sort to end, and Infinities are equal.
      const x = ar[a];
      const y = ar[b];
      if (isNaN(x)) return isNaN(y) ? 0 : 1;
      if (isNaN(y)) return -1;
      return x === y ? 0 : x < y ? -1 : 1;
    });
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

function identityMatrix(n: number): number[] {
  const out = new Array(n * n).fill(0);
  for (let i = 0; i < n; i++) out[i * n + i] = 1;
  return out;
}

function transposeMatrix(a: number[], rows: number, cols: number): number[] {
  const out = new Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) out[j * rows + i] = a[i * cols + j];
  }
  return out;
}

function multiplyMatrix(
  a: number[],
  aRows: number,
  aCols: number,
  b: number[],
  bCols: number,
): number[] {
  const out = new Array(aRows * bCols).fill(0);
  for (let i = 0; i < aRows; i++) {
    for (let k = 0; k < aCols; k++) {
      const aik = a[i * aCols + k];
      for (let j = 0; j < bCols; j++) {
        out[i * bCols + j] += aik * b[k * bCols + j];
      }
    }
  }
  return out;
}

function symmetricJacobiEigen(
  a: number[],
  n: number,
): { values: number[]; vectors: number[] } {
  const work = a.slice();
  const vectors = identityMatrix(n);
  const maxIter = Math.max(50, 50 * n * n);

  for (let iter = 0; iter < maxIter; iter++) {
    let p = 0;
    let q = 1;
    let max = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const value = Math.abs(work[i * n + j]);
        if (value > max) {
          max = value;
          p = i;
          q = j;
        }
      }
    }

    if (n < 2) break;
    const scale = Math.max(
      1,
      Math.abs(work[p * n + p]),
      Math.abs(work[q * n + q]),
    );
    if (max <= 1e-10 * scale) break;

    const app = work[p * n + p];
    const aqq = work[q * n + q];
    const apq = work[p * n + q];
    const tau = (aqq - app) / (2 * apq);
    const t =
      Math.sign(tau || 1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
    const c = 1 / Math.sqrt(1 + t * t);
    const s = t * c;

    for (let k = 0; k < n; k++) {
      const wkp = work[k * n + p];
      const wkq = work[k * n + q];
      work[k * n + p] = c * wkp - s * wkq;
      work[k * n + q] = s * wkp + c * wkq;
    }
    for (let k = 0; k < n; k++) {
      const wpk = work[p * n + k];
      const wqk = work[q * n + k];
      work[p * n + k] = c * wpk - s * wqk;
      work[q * n + k] = s * wpk + c * wqk;
    }
    for (let k = 0; k < n; k++) {
      const vkp = vectors[k * n + p];
      const vkq = vectors[k * n + q];
      vectors[k * n + p] = c * vkp - s * vkq;
      vectors[k * n + q] = s * vkp + c * vkq;
    }
  }

  return {
    values: Array.from({ length: n }, (_, i) => work[i * n + i]),
    vectors,
  };
}

function normalizeColumns(a: number[], rows: number, cols: number) {
  for (let col = 0; col < cols; col++) {
    let norm = 0;
    for (let row = 0; row < rows; row++) {
      norm = Math.hypot(norm, a[row * cols + col]);
    }
    if (norm === 0) continue;
    for (let row = 0; row < rows; row++) a[row * cols + col] /= norm;
  }
}

function runSVD(
  type: RoutineType,
  [a]: DataArray[],
  outputs: DataArray[],
  { computeUv, fullMatrices }: { computeUv: boolean; fullMatrices: boolean },
) {
  const shape = type.inputShapes[0];
  if (shape.length < 2) throw new Error("svd: input must be at least 2D");
  const m = shape[shape.length - 2];
  const n = shape[shape.length - 1];
  const k = Math.min(m, n);
  const batches = a.length / (m * n);
  const [uOut, sOut, vhOut] = computeUv
    ? outputs
    : [undefined, outputs[0], undefined];

  for (let b = 0; b < batches; b++) {
    const mat = Array.from(a.subarray(b * m * n, (b + 1) * m * n));
    const at = transposeMatrix(mat, m, n);
    const ata = multiplyMatrix(at, n, m, mat, n);
    const eig = symmetricJacobiEigen(ata, n);
    const order = eig.values
      .map((value, index) => ({ value: Math.max(0, value), index }))
      .sort((x, y) => y.value - x.value);
    const singularValues = order
      .slice(0, k)
      .map((item) => Math.sqrt(item.value));

    for (let i = 0; i < k; i++) sOut[b * k + i] = singularValues[i];
    if (!computeUv) continue;

    const v = new Array(n * n).fill(0);
    for (let col = 0; col < n; col++) {
      const src = order[col]?.index ?? col;
      for (let row = 0; row < n; row++) {
        v[row * n + col] = eig.vectors[row * n + src];
      }
    }

    const uFull = identityMatrix(m);
    for (let col = 0; col < k; col++) {
      if (singularValues[col] <= 1e-12) continue;
      for (let row = 0; row < m; row++) {
        let sum = 0;
        for (let j = 0; j < n; j++) sum += mat[row * n + j] * v[j * n + col];
        uFull[row * m + col] = sum / singularValues[col];
      }
    }
    normalizeColumns(uFull, m, m);

    const uCols = fullMatrices ? m : k;
    for (let row = 0; row < m; row++) {
      for (let col = 0; col < uCols; col++) {
        uOut![b * m * uCols + row * uCols + col] = uFull[row * m + col];
      }
    }

    const vhRows = fullMatrices ? n : k;
    for (let row = 0; row < vhRows; row++) {
      for (let col = 0; col < n; col++) {
        vhOut![b * vhRows * n + row * n + col] = v[col * n + row];
      }
    }
  }
}
