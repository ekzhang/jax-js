import { DataArray, dtypedArray } from "./alu";
import { ShapedArray } from "./frontend/core";

/**
 * Advanced operations that don't fit into the "AluExp" compiler representation.
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
    /** Shapes and types of the arguments. */
    readonly avals: ShapedArray[],
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

  /** Cholesky decomposition of 2D positive semi-definite matrices. */
  Cholesky = "Cholesky",
}

// Reference implementation of each routine in CPU is below.
//
// The remaining backends implement these routines within their own folders, to
// allow for code splitting between backends. This is encapsulation.

export function runSort([x]: DataArray[], [xs]: ShapedArray[]): DataArray[] {
  if (xs.ndim === 0) throw new Error("sort: cannot sort a scalar");
  const n = xs.shape[xs.shape.length - 1];
  const y = x.slice();
  for (let i = 0; i < y.length; i += n) {
    y.subarray(i, i + n).sort(); // In-place
  }
  return [y];
}

export function runArgsort([x]: DataArray[], [xs]: ShapedArray[]): DataArray[] {
  if (xs.ndim === 0) throw new Error("argsort: cannot sort a scalar");
  const n = xs.shape[xs.shape.length - 1];
  const y = new Int32Array(x.length);
  for (let offset = 0; offset < y.length; offset += n) {
    const ar = x.subarray(offset, offset + n);
    const out = y.subarray(offset, offset + n);
    for (let i = 0; i < n; i++) out[i] = i;
    out.sort((a, b) => ar[a] - ar[b]);
  }
  return [y];
}

export function runCholesky(
  [x]: DataArray[],
  [xs]: ShapedArray[],
): DataArray[] {
  if (xs.ndim < 2) throw new Error("cholesky: input must be at least 2D");
  const n = xs.shape[xs.shape.length - 2];
  const m = xs.shape[xs.shape.length - 1];
  if (n !== m)
    throw new Error(`cholesky: input must be square, got [${n}, ${m}]`);

  const y = dtypedArray(xs.dtype, new Uint8Array(x.byteLength));
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
  return [y];
}
