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

  /** Solve a triangular system of questions. */
  TriangularSolve = "TriangularSolve",
  /** Cholesky decomposition of 2D positive semi-definite matrices. */
  Cholesky = "Cholesky",
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
  const inputArrays = inputs.map((buf, i) =>
    dtypedArray(type.inputDtypes[i], buf),
  );
  const outputArrays = outputs.map((buf, i) =>
    dtypedArray(type.outputDtypes[i], buf),
  );
  switch (name) {
    case Routines.Sort:
      return runSort(type, inputArrays, outputArrays);
    case Routines.Argsort:
      return runArgsort(type, inputArrays, outputArrays);
    case Routines.TriangularSolve:
      return runTriangularSolve(type, inputArrays, outputArrays);
    case Routines.Cholesky:
      return runCholesky(type, inputArrays, outputArrays);
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

function runArgsort(type: RoutineType, [x]: DataArray[], [y]: DataArray[]) {
  const xs = type.inputShapes[0];
  if (xs.length === 0) throw new Error("argsort: cannot sort a scalar");
  const n = xs[xs.length - 1];
  for (let offset = 0; offset < y.length; offset += n) {
    const ar = x.subarray(offset, offset + n);
    const out = y.subarray(offset, offset + n);
    for (let i = 0; i < n; i++) out[i] = i;
    out.sort((a, b) => ar[a] - ar[b]);
  }
}

function runTriangularSolve(
  type: RoutineType,
  [a, b]: DataArray[],
  [x]: DataArray[],
) {
  const as = type.inputShapes[0];
  const bs = type.inputShapes[1];
  if (as.length !== 2) throw new Error("triangular_solve: a must be 2D");
  if (bs.length < 1 || bs.length > 2)
    throw new Error("triangular_solve: b must be 1D or 2D");
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
