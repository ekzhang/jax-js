// Custom lowering for advanced operations that don't fit into AluExp.

import { byteWidth, DataArray, DType, dtypedArray, isFloatDtype } from "./alu";

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
   * Symmetric eigendecomposition using cyclic Jacobi rotations.
   *
   * The input is a batch of real symmetric matrices with shape `[..., N, N]`;
   * only the lower triangle is read. The output is a tuple `Diagonalized,
   * Vectors`, both shape `[..., N, N]`. `Diagonalized` is approximately
   * diagonal and `Vectors` contains the accumulated eigenvectors as columns.
   * Sorting eigenpairs is handled by the frontend wrapper.
   */
  JacobiEigh = "JacobiEigh",

  /**
   * Complex FFT along the last axis for real/imaginary array pairs.
   *
   * Inputs and outputs have identical shape, and the backend receives a
   * factorization of the FFT dimension.
   */
  Fft = "Fft",
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
    case Routines.JacobiEigh:
      return runJacobiEigh(type, inputAr, outputAr, routine.params);
    case Routines.Fft:
      return runFft(type, inputAr, outputAr, routine.params);
    default:
      name satisfies never; // Exhaustiveness check
  }
}

/** JS source for running CPU routines inside a Web Worker. */
export function cpuRoutineJSForWorkers(): string {
  return `
const DType = ${JSON.stringify(DType)};
const Routines = ${JSON.stringify(Routines)};
const ${byteWidth.name} = ${byteWidth.toString()};
const ${isFloatDtype.name} = ${isFloatDtype.toString()};
const ${dtypedArray.name} = ${dtypedArray.toString()};
${runCpuRoutine.toString()}
${runSort.toString()}
${runArgsort.toString()}
${runTriangularSolve.toString()}
${runCholesky.toString()}
${runLU.toString()}
${runJacobiEigh.toString()}
${runFft.toString()}
const __minify_safe_runCpuRoutine = ${runCpuRoutine.name};
`;
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

function runJacobiEigh(
  type: RoutineType,
  [input]: DataArray[],
  [diagonalized, vectors]: DataArray[],
  { maxSweeps, tolerance }: { maxSweeps: number; tolerance: number },
) {
  const shape = type.inputShapes[0];
  if (shape.length < 2)
    throw new Error("jacobi_eigh: input must be at least 2D");
  const n = shape[shape.length - 1];
  if (n !== shape[shape.length - 2])
    throw new Error(`jacobi_eigh: input must be square, got ${shape}`);
  if (!isFloatDtype(type.inputDtypes[0]))
    throw new TypeError(`jacobi_eigh: input must be floating-point`);

  function symIndex(i: number, j: number): number {
    return i >= j ? i * n + j : j * n + i;
  }

  function maxAbsMatrix(a: DataArray): number {
    let maxAbs = 1;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        maxAbs = Math.max(maxAbs, Math.abs(a[i * n + j]));
      }
    }
    return maxAbs;
  }

  function maxAbsOffDiagonal(a: DataArray): number {
    let maxAbs = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < i; j++) {
        maxAbs = Math.max(maxAbs, Math.abs(a[i * n + j]));
      }
    }
    return maxAbs;
  }

  function applyJacobiRotation(
    a: DataArray,
    v: DataArray,
    p: number,
    q: number,
  ) {
    const pp = p * n + p;
    const qq = q * n + q;
    const pq = symIndex(p, q);
    const app = a[pp];
    const aqq = a[qq];
    const apq = a[pq];
    if (apq === 0) return;

    const tau = (aqq - app) / (2 * apq);
    const tauSign = tau >= 0 ? 1 : -1;
    const t = tauSign / (Math.abs(tau) + Math.sqrt(tau * tau + 1));
    const c = 1 / Math.sqrt(t * t + 1);
    const s = t * c;

    for (let k = 0; k < n; k++) {
      if (k === p || k === q) continue;
      const kp = symIndex(k, p);
      const kq = symIndex(k, q);
      const akp = a[kp];
      const akq = a[kq];
      const nextKp = c * akp - s * akq;
      const nextKq = s * akp + c * akq;
      a[kp] = nextKp;
      a[kq] = nextKq;
    }

    a[pp] = c * c * app - 2 * s * c * apq + s * s * aqq;
    a[qq] = s * s * app + 2 * s * c * apq + c * c * aqq;
    a[pq] = 0;

    for (let k = 0; k < n; k++) {
      const kp = k * n + p;
      const kq = k * n + q;
      const vkp = v[kp];
      const vkq = v[kq];
      v[kp] = c * vkp - s * vkq;
      v[kq] = s * vkp + c * vkq;
    }
  }

  const matrixSize = n * n;
  for (let offset = 0; offset < input.length; offset += matrixSize) {
    const x = input.subarray(offset, offset + matrixSize);
    const a = diagonalized.subarray(offset, offset + matrixSize);
    const v = vectors.subarray(offset, offset + matrixSize);
    a.fill(0);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        a[i * n + j] = x[i * n + j];
      }
    }
    v.fill(0);
    for (let i = 0; i < n; i++) v[i * n + i] = 1;

    const threshold = tolerance * maxAbsMatrix(a);
    let maxOffDiagonal = maxAbsOffDiagonal(a);
    for (
      let sweep = 0;
      sweep < maxSweeps && maxOffDiagonal > threshold;
      sweep++
    ) {
      for (let p = 0; p < n - 1; p++) {
        for (let q = p + 1; q < n; q++) {
          applyJacobiRotation(a, v, p, q);
        }
      }
      maxOffDiagonal = maxAbsOffDiagonal(a);
    }
  }
}

function runFft(
  type: RoutineType,
  [real, imag]: DataArray[],
  [outReal, outImag]: DataArray[],
  { factors, inverse }: { factors: number[]; inverse: boolean },
) {
  const shape = type.inputShapes[0];
  if (shape.length < 1) throw new Error("fft: input must be at least 1D");
  const n = shape[shape.length - 1];
  if (n < 1) throw new Error(`fft: final axis must be non-empty, got ${n}`);
  if (
    type.inputDtypes[0] !== type.inputDtypes[1] ||
    !isFloatDtype(type.inputDtypes[0])
  ) {
    throw new Error("fft: expected matching floating-point real/imag arrays");
  }

  const angleScale = (inverse ? 2 : -2) * Math.PI;
  const permutation = new Uint32Array(n);
  for (let index = 0; index < n; index++) {
    let remaining = index;
    let stride = 1;
    let reversed = 0;
    for (const factor of factors) {
      const digit = remaining % factor;
      remaining = Math.floor(remaining / factor);
      stride *= factor;
      reversed += digit * (n / stride);
    }
    permutation[index] = reversed;
  }
  const scratchReal = new Array<number>(Math.max(1, ...factors));
  const scratchImag = new Array<number>(scratchReal.length);

  for (let offset = 0; offset < real.length; offset += n) {
    for (let index = 0; index < n; index++) {
      const source = offset + permutation[index];
      outReal[offset + index] = real[source];
      outImag[offset + index] = imag[source];
    }

    let prev = 1;
    for (let stage = 0; stage < factors.length; stage++) {
      const radix = factors[stage];
      const span = prev * radix;
      const stageScale = inverse && stage === factors.length - 1 ? 1 / n : 1;
      for (let group = 0; group < n; group += span) {
        for (let j = 0; j < prev; j++) {
          for (let q = 0; q < radix; q++) {
            const idx = offset + group + j + q * prev;
            const angle = (angleScale * q * j) / span;
            const c = Math.cos(angle);
            const s = Math.sin(angle);
            const xr = outReal[idx];
            const xi = outImag[idx];
            scratchReal[q] = xr * c - xi * s;
            scratchImag[q] = xr * s + xi * c;
          }
          for (let p = 0; p < radix; p++) {
            let sumReal = 0;
            let sumImag = 0;
            for (let q = 0; q < radix; q++) {
              const angle = (angleScale * q * p) / radix;
              const c = Math.cos(angle);
              const s = Math.sin(angle);
              const xr = scratchReal[q];
              const xi = scratchImag[q];
              sumReal += xr * c - xi * s;
              sumImag += xr * s + xi * c;
            }
            const idx = offset + group + j + p * prev;
            outReal[idx] = sumReal * stageScale;
            outImag[idx] = sumImag * stageScale;
          }
        }
      }
      prev = span;
    }
  }
}
