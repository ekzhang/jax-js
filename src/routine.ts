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

  /**
   * Real eigenvalues of 2D square matrices.
   *
   * The input is a batch of shape `[..., N, N]`, and the output is a batch of
   * shape `[..., N]`. Complex eigenvalue pairs are not represented.
   */
  Eigvals = "Eigvals",
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
    case Routines.Eigvals:
      return runEigvals(type, inputAr, outputAr);
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
    const t = Math.sign(tau || 1) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
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

function completeOrthonormalColumns(
  basis: number[],
  rows: number,
  cols: number,
): number[] {
  const out = new Array(rows * rows).fill(0);
  let outCol = 0;

  const addCandidate = (candidate: number[]) => {
    for (let col = 0; col < outCol; col++) {
      let dot = 0;
      for (let row = 0; row < rows; row++)
        dot += candidate[row] * out[row * rows + col];
      for (let row = 0; row < rows; row++)
        candidate[row] -= dot * out[row * rows + col];
    }
    let norm = 0;
    for (let row = 0; row < rows; row++)
      norm = Math.hypot(norm, candidate[row]);
    if (norm <= 1e-12) return;
    for (let row = 0; row < rows; row++)
      out[row * rows + outCol] = candidate[row] / norm;
    outCol++;
  };

  for (let col = 0; col < cols && outCol < rows; col++) {
    addCandidate(
      Array.from({ length: rows }, (_, row) => basis[row * cols + col]),
    );
  }
  for (let col = 0; col < rows && outCol < rows; col++) {
    const candidate = new Array(rows).fill(0);
    candidate[col] = 1;
    addCandidate(candidate);
  }
  return out;
}

function golubReinschTallSVD(
  input: number[],
  m: number,
  n: number,
): { s: number[]; u: number[]; v: number[] } {
  const a = input.slice();
  const s = new Array(Math.min(m + 1, n)).fill(0);
  const e = new Array(n).fill(0);
  const work = new Array(m).fill(0);
  const uCols = n;
  const u = new Array(m * uCols).fill(0);
  const v = new Array(n * n).fill(0);
  const nct = Math.min(m - 1, n);
  const nrt = Math.max(0, Math.min(n - 2, m));
  const mrc = Math.max(nct, nrt);
  const get = (row: number, col: number) => a[row * n + col];
  const set = (row: number, col: number, value: number) => {
    a[row * n + col] = value;
  };

  for (let k = 0; k < mrc; k++) {
    if (k < nct) {
      s[k] = 0;
      for (let i = k; i < m; i++) s[k] = Math.hypot(s[k], get(i, k));
      if (s[k] !== 0) {
        if (get(k, k) < 0) s[k] = -s[k];
        for (let i = k; i < m; i++) set(i, k, get(i, k) / s[k]);
        set(k, k, get(k, k) + 1);
      }
      s[k] = -s[k];
    }

    for (let j = k + 1; j < n; j++) {
      if (k < nct && s[k] !== 0) {
        let t = 0;
        for (let i = k; i < m; i++) t += get(i, k) * get(i, j);
        t = -t / get(k, k);
        for (let i = k; i < m; i++) set(i, j, get(i, j) + t * get(i, k));
      }
      e[j] = get(k, j);
    }

    if (k < nct) {
      for (let i = k; i < m; i++) u[i * uCols + k] = get(i, k);
    }

    if (k < nrt) {
      e[k] = 0;
      for (let i = k + 1; i < n; i++) e[k] = Math.hypot(e[k], e[i]);
      if (e[k] !== 0) {
        if (e[k + 1] < 0) e[k] = -e[k];
        for (let i = k + 1; i < n; i++) e[i] /= e[k];
        e[k + 1] += 1;
      }
      e[k] = -e[k];
      if (k + 1 < m && e[k] !== 0) {
        for (let i = k + 1; i < m; i++) work[i] = 0;
        for (let i = k + 1; i < m; i++) {
          for (let j = k + 1; j < n; j++) work[i] += e[j] * get(i, j);
        }
        for (let j = k + 1; j < n; j++) {
          const t = -e[j] / e[k + 1];
          for (let i = k + 1; i < m; i++) set(i, j, get(i, j) + t * work[i]);
        }
      }
      for (let i = k + 1; i < n; i++) v[i * n + k] = e[i];
    }
  }

  const pInit = Math.min(n, m + 1);
  if (nct < n) s[nct] = get(nct, nct);
  if (m < pInit) s[pInit - 1] = 0;
  if (nrt + 1 < pInit) e[nrt] = get(nrt, pInit - 1);
  e[pInit - 1] = 0;

  for (let j = nct; j < uCols; j++) {
    for (let i = 0; i < m; i++) u[i * uCols + j] = 0;
    u[j * uCols + j] = 1;
  }
  for (let k = nct - 1; k >= 0; k--) {
    if (s[k] !== 0) {
      for (let j = k + 1; j < uCols; j++) {
        let t = 0;
        for (let i = k; i < m; i++) t += u[i * uCols + k] * u[i * uCols + j];
        t = -t / u[k * uCols + k];
        for (let i = k; i < m; i++) u[i * uCols + j] += t * u[i * uCols + k];
      }
      for (let i = k; i < m; i++) u[i * uCols + k] = -u[i * uCols + k];
      u[k * uCols + k] += 1;
      for (let i = 0; i < k - 1; i++) u[i * uCols + k] = 0;
    } else {
      for (let i = 0; i < m; i++) u[i * uCols + k] = 0;
      u[k * uCols + k] = 1;
    }
  }

  for (let k = n - 1; k >= 0; k--) {
    if (k < nrt && e[k] !== 0) {
      for (let j = k + 1; j < n; j++) {
        let t = 0;
        for (let i = k + 1; i < n; i++) t += v[i * n + k] * v[i * n + j];
        t = -t / v[(k + 1) * n + k];
        for (let i = k + 1; i < n; i++) v[i * n + j] += t * v[i * n + k];
      }
    }
    for (let i = 0; i < n; i++) v[i * n + k] = 0;
    v[k * n + k] = 1;
  }

  let p = pInit;
  const pp = p - 1;
  let iter = 0;
  const eps = Number.EPSILON;
  while (p > 0) {
    let k: number;
    let kase: number;
    for (k = p - 2; k >= -1; k--) {
      if (k === -1) break;
      const alpha =
        Number.MIN_VALUE + eps * (Math.abs(s[k]) + Math.abs(s[k + 1]));
      if (Math.abs(e[k]) <= alpha || Number.isNaN(e[k])) {
        e[k] = 0;
        break;
      }
    }
    if (k === p - 2) {
      kase = 4;
    } else {
      let ks: number;
      for (ks = p - 1; ks >= k; ks--) {
        if (ks === k) break;
        const t =
          (ks !== p ? Math.abs(e[ks]) : 0) +
          (ks !== k + 1 ? Math.abs(e[ks - 1]) : 0);
        if (Math.abs(s[ks]) <= eps * t) {
          s[ks] = 0;
          break;
        }
      }
      if (ks === k) kase = 3;
      else if (ks === p - 1) kase = 1;
      else {
        kase = 2;
        k = ks;
      }
    }
    k++;

    switch (kase) {
      case 1: {
        let f = e[p - 2];
        e[p - 2] = 0;
        for (let j = p - 2; j >= k; j--) {
          let t = Math.hypot(s[j], f);
          const cs = s[j] / t;
          const sn = f / t;
          s[j] = t;
          if (j !== k) {
            f = -sn * e[j - 1];
            e[j - 1] = cs * e[j - 1];
          }
          for (let i = 0; i < n; i++) {
            t = cs * v[i * n + j] + sn * v[i * n + p - 1];
            v[i * n + p - 1] = -sn * v[i * n + j] + cs * v[i * n + p - 1];
            v[i * n + j] = t;
          }
        }
        break;
      }
      case 2: {
        let f = e[k - 1];
        e[k - 1] = 0;
        for (let j = k; j < p; j++) {
          let t = Math.hypot(s[j], f);
          const cs = s[j] / t;
          const sn = f / t;
          s[j] = t;
          f = -sn * e[j];
          e[j] = cs * e[j];
          for (let i = 0; i < m; i++) {
            t = cs * u[i * uCols + j] + sn * u[i * uCols + k - 1];
            u[i * uCols + k - 1] =
              -sn * u[i * uCols + j] + cs * u[i * uCols + k - 1];
            u[i * uCols + j] = t;
          }
        }
        break;
      }
      case 3: {
        const scale = Math.max(
          Math.abs(s[p - 1]),
          Math.abs(s[p - 2]),
          Math.abs(e[p - 2]),
          Math.abs(s[k]),
          Math.abs(e[k]),
        );
        const sp = s[p - 1] / scale;
        const spm1 = s[p - 2] / scale;
        const epm1 = e[p - 2] / scale;
        const sk = s[k] / scale;
        const ek = e[k] / scale;
        const b = ((spm1 + sp) * (spm1 - sp) + epm1 * epm1) / 2;
        const c = sp * epm1 * (sp * epm1);
        let shift = 0;
        if (b !== 0 || c !== 0) {
          shift = b < 0 ? -Math.sqrt(b * b + c) : Math.sqrt(b * b + c);
          shift = c / (b + shift);
        }
        let f = (sk + sp) * (sk - sp) + shift;
        let g = sk * ek;
        for (let j = k; j < p - 1; j++) {
          let t = Math.hypot(f, g);
          if (t === 0) t = Number.MIN_VALUE;
          let cs = f / t;
          let sn = g / t;
          if (j !== k) e[j - 1] = t;
          f = cs * s[j] + sn * e[j];
          e[j] = cs * e[j] - sn * s[j];
          g = sn * s[j + 1];
          s[j + 1] = cs * s[j + 1];
          for (let i = 0; i < n; i++) {
            t = cs * v[i * n + j] + sn * v[i * n + j + 1];
            v[i * n + j + 1] = -sn * v[i * n + j] + cs * v[i * n + j + 1];
            v[i * n + j] = t;
          }
          t = Math.hypot(f, g);
          if (t === 0) t = Number.MIN_VALUE;
          cs = f / t;
          sn = g / t;
          s[j] = t;
          f = cs * e[j] + sn * s[j + 1];
          s[j + 1] = -sn * e[j] + cs * s[j + 1];
          g = sn * e[j + 1];
          e[j + 1] = cs * e[j + 1];
          if (j < m - 1) {
            for (let i = 0; i < m; i++) {
              t = cs * u[i * uCols + j] + sn * u[i * uCols + j + 1];
              u[i * uCols + j + 1] =
                -sn * u[i * uCols + j] + cs * u[i * uCols + j + 1];
              u[i * uCols + j] = t;
            }
          }
        }
        e[p - 2] = f;
        iter++;
        if (iter > 1000)
          throw new Error("svd: QR iteration failed to converge");
        break;
      }
      case 4: {
        if (s[k] <= 0) {
          s[k] = s[k] < 0 ? -s[k] : 0;
          for (let i = 0; i <= pp; i++) v[i * n + k] = -v[i * n + k];
        }
        while (k < pp) {
          if (s[k] >= s[k + 1]) break;
          let t = s[k];
          s[k] = s[k + 1];
          s[k + 1] = t;
          if (k < n - 1) {
            for (let i = 0; i < n; i++) {
              t = v[i * n + k + 1];
              v[i * n + k + 1] = v[i * n + k];
              v[i * n + k] = t;
            }
          }
          if (k < m - 1) {
            for (let i = 0; i < m; i++) {
              t = u[i * uCols + k + 1];
              u[i * uCols + k + 1] = u[i * uCols + k];
              u[i * uCols + k] = t;
            }
          }
          k++;
        }
        iter = 0;
        p--;
        break;
      }
    }
  }

  return { s: s.slice(0, n), u, v };
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
    const tall = m >= n;
    const svd = tall
      ? golubReinschTallSVD(mat, m, n)
      : golubReinschTallSVD(transposeMatrix(mat, m, n), n, m);
    const singularValues = svd.s.slice(0, k);

    for (let i = 0; i < k; i++) sOut[b * k + i] = singularValues[i] ?? 0;
    if (!computeUv) continue;

    const uBasis = tall ? svd.u : svd.v;
    const uBasisCols = tall ? n : m;
    const vBasis = tall ? svd.v : svd.u;
    const vBasisRows = n;
    const vBasisCols = tall ? n : m;
    const uFull = completeOrthonormalColumns(uBasis, m, uBasisCols);
    const vFull = completeOrthonormalColumns(vBasis, vBasisRows, vBasisCols);

    const uCols = fullMatrices ? m : k;
    for (let row = 0; row < m; row++) {
      for (let col = 0; col < uCols; col++) {
        uOut![b * m * uCols + row * uCols + col] = uFull[row * m + col];
      }
    }

    const vhRows = fullMatrices ? n : k;
    for (let row = 0; row < vhRows; row++) {
      for (let col = 0; col < n; col++) {
        vhOut![b * vhRows * n + row * n + col] = vFull[col * n + row];
      }
    }
  }
}

function hessenbergHouseholder(a: number[], n: number): number[] {
  const h = a.slice();
  for (let k = 0; k < n - 2; k++) {
    let norm = 0;
    for (let i = k + 1; i < n; i++) norm = Math.hypot(norm, h[i * n + k]);
    if (norm === 0) continue;

    const sign = h[(k + 1) * n + k] >= 0 ? 1 : -1;
    const v = new Array(n).fill(0);
    v[k + 1] = h[(k + 1) * n + k] + sign * norm;
    for (let i = k + 2; i < n; i++) v[i] = h[i * n + k];

    let betaDen = 0;
    for (let i = k + 1; i < n; i++) betaDen += v[i] * v[i];
    if (betaDen === 0) continue;
    const beta = 2 / betaDen;

    for (let j = k; j < n; j++) {
      let dot = 0;
      for (let i = k + 1; i < n; i++) dot += v[i] * h[i * n + j];
      for (let i = k + 1; i < n; i++) h[i * n + j] -= beta * v[i] * dot;
    }

    for (let i = 0; i < n; i++) {
      let dot = 0;
      for (let j = k + 1; j < n; j++) dot += h[i * n + j] * v[j];
      for (let j = k + 1; j < n; j++) h[i * n + j] -= beta * dot * v[j];
    }

    for (let i = k + 2; i < n; i++) h[i * n + k] = 0;
  }
  return h;
}

function qrDecompose(a: number[], n: number): { q: number[]; r: number[] } {
  const q = new Array(n * n).fill(0);
  const r = new Array(n * n).fill(0);
  const cols: number[][] = [];
  for (let j = 0; j < n; j++) {
    cols[j] = Array.from({ length: n }, (_, i) => a[i * n + j]);
  }

  for (let j = 0; j < n; j++) {
    const v = cols[j].slice();
    for (let i = 0; i < j; i++) {
      let rij = 0;
      for (let row = 0; row < n; row++) rij += q[row * n + i] * cols[j][row];
      r[i * n + j] = rij;
      for (let row = 0; row < n; row++) v[row] -= rij * q[row * n + i];
    }

    const norm = Math.hypot(...v);
    r[j * n + j] = norm;
    if (norm !== 0) {
      for (let row = 0; row < n; row++) q[row * n + j] = v[row] / norm;
    }
  }
  return { q, r };
}

function hasComplexTwoByTwoBlock(a: number[], n: number, i: number): boolean {
  const aa = a[i * n + i];
  const bb = a[i * n + i + 1];
  const cc = a[(i + 1) * n + i];
  const dd = a[(i + 1) * n + i + 1];
  const trace = aa + dd;
  const det = aa * dd - bb * cc;
  return trace * trace - 4 * det < 0;
}

function realQrEigenvalues(input: number[], n: number): number[] {
  if (n === 1) return [input[0]];

  const h = hessenbergHouseholder(input, n);
  const maxIter = Math.max(100, 200 * n * n);
  for (let iter = 0; iter < maxIter; iter++) {
    let done = true;
    for (let i = 1; i < n; i++) {
      const tol =
        1e-10 * (Math.abs(h[(i - 1) * n + i - 1]) + Math.abs(h[i * n + i]) + 1);
      if (Math.abs(h[i * n + i - 1]) <= tol) h[i * n + i - 1] = 0;
      else done = false;
    }
    if (done) return Array.from({ length: n }, (_, i) => h[i * n + i]);

    for (let i = 0; i < n - 1; i++) {
      if (Math.abs(h[(i + 1) * n + i]) > 1e-7) {
        if (hasComplexTwoByTwoBlock(h, n, i)) {
          throw new Error("eigvals: complex eigenvalues are not supported yet");
        }
      }
    }

    const mu = h[(n - 1) * n + n - 1];
    const shifted = h.slice();
    for (let i = 0; i < n; i++) shifted[i * n + i] -= mu;
    const { q, r } = qrDecompose(shifted, n);
    const next = multiplyMatrix(r, n, n, q, n);
    for (let i = 0; i < n; i++) next[i * n + i] += mu;
    h.splice(0, h.length, ...next);
  }

  for (let i = 0; i < n - 1; i++) {
    if (
      Math.abs(h[(i + 1) * n + i]) > 1e-7 &&
      hasComplexTwoByTwoBlock(h, n, i)
    ) {
      throw new Error("eigvals: complex eigenvalues are not supported yet");
    }
  }
  throw new Error("eigvals: QR iteration failed to converge");
}

function runEigvals(type: RoutineType, [a]: DataArray[], [out]: DataArray[]) {
  const shape = type.inputShapes[0];
  if (shape.length < 2) throw new Error("eigvals: input must be at least 2D");
  const n = shape[shape.length - 1];
  if (shape[shape.length - 2] !== n)
    throw new Error("eigvals: input must be square");

  for (let offset = 0; offset < a.length; offset += n * n) {
    const batch = offset / (n * n);
    const values = realQrEigenvalues(
      Array.from(a.subarray(offset, offset + n * n)),
      n,
    );
    for (let i = 0; i < n; i++) out[batch * n + i] = values[i];
  }
}
