/**
 * @file Optimizations applied to kernels by different backends.
 *
 * The main optimizations (for reductions) are:
 *
 * - "Upcast": Multiple values are computed per thread, along a non-reduction
 *   dimension. If appropriate, this lowers to vector/SIMD instructions. Each
 *   thread computes a chunk of output values, which helps with cache
 *   performance (e.g., matmul tiling).
 *
 * - "Unroll": Similar to Upcast, but along a loop dimension, which translates
 *   to loop unrolling. You increment the loop index by the unroll factor. This
 *   does not use vector/SIMD instructions.
 *
 * - "Group": Multiple threads compute the same value. For example, when summing
 *   up the numbers in a vector, K threads each accumulate 1/K of the vector,
 *   stores in shared memory, and thread 0 accumulates at the end.
 *   - Regular order: 4 threads grouped as [1234123412341234]
 *   - "Top": 4 threads grouped as [1111222233334444]
 *
 * These are inspired by Tinygrad's heuristic optimizations.
 * https://github.com/tinygrad/tinygrad/blob/685d5c46df/tinygrad/codegen/heuristic.py
 */

import { accessorGlobal, AluExp, AluOp, AluVar, DType, Kernel } from "./alu";
import { ShapeTracker, unravelAlu } from "./shape";
import { DEBUG, deepEqual } from "./utils";

// gidx = (0 ... dim.local ... dim.reduce)
// ridx = (dim.reduce .[local index].
//         dim.group .[reduce loops].
//         dim.unroll .[upcast]. dim.upcast)
// uidx = (dim.upcast .[unroll]. ...)
// result[gidx + uidx] = <<< eval(kernel.exp) >>>;

export interface TuneResult {
  /** New expression with GlobalView ops and gidx/ridx lowered. */
  exp: AluExp;

  /**
   * Number of threads in a group.
   * If greater than 1, group index is available as `AluExp.special("group")`.
   */
  group?: number;

  /** Amount to upcast in reduction, set via `AluVar.unroll`. */
  unroll?: number;

  /** Amount to upcast in non-reduce dimensions, set via `AluVar.upcast`. */
  upcast?: number[];
}

/** Stores dimensions of the kernel's applied shape. Globals start at 0. */
class TuneDims {
  st: ShapeTracker; // Shape tracker for the optimizations.

  // local: number; // TODO: Split gidx -> global and local dimensions during tuning.
  reduce: number; // Reductions start here.
  group: number; // Single reduction thread, equal to reduce if no groups.
  unroll: number; // Upcast along the reduce dimension.
  upcast: number; // Upcast along output dimension.

  constructor(shape: number[]) {
    this.st = ShapeTracker.fromShape(shape);
    this.reduce = this.st.shape.length - 1;
    this.group = this.st.shape.length - 1;
    this.unroll = this.st.shape.length;
    this.upcast = this.st.shape.length;
  }
}

/** Tuning step that does not apply any optimization. */
export function tuneNullopt(kernel: Kernel): TuneResult {
  const vars: Record<string, AluExp> = {};
  vars.gidx = AluExp.special(DType.Int32, "gidx", kernel.size);
  if (kernel.reduction)
    vars.ridx = AluExp.special(DType.Int32, "ridx", kernel.reduction.size);
  return {
    exp: kernel.exp
      .rewrite((exp) => {
        if (exp.op === AluOp.GlobalView) {
          const gid: number = exp.arg[0];
          const st: ShapeTracker = exp.arg[1];
          return accessorGlobal(gid, st, exp.src);
        }
      })
      .substitute(vars)
      .simplify(),
  };
}

/** Tuning for WebGPU kernels. */
export function tuneWebgpu(kernel: Kernel): TuneResult {
  if (!kernel.reduction) return tuneNullopt(kernel);
  const { exp, reduction } = kernel;

  // 1. Check that kernel GlobalView ops have consistent src[], where the last
  //    dimension is reduction, and others are gidx.
  const globalViews = exp.collect((exp) => exp.op === AluOp.GlobalView);
  if (globalViews.length === 0) {
    if (DEBUG >= 4) console.info("Tuning: No GlobalView ops found in kernel.");
    return tuneNullopt(kernel); // TODO: Nullary kernel, write opts for this.
  }
  const shape: number[] = globalViews[0].arg[1].shape;
  const expectedSrc = [
    ...unravelAlu(shape.slice(0, -1), AluVar.gidx),
    AluVar.ridx,
  ];
  for (const gv of globalViews) {
    if (!gv.src.length || !deepEqual(gv.src, expectedSrc)) {
      if (DEBUG >= 4)
        console.info("Tuning: GlobalView src[] not consistent with reduction.");
      return tuneNullopt(kernel);
    }
  }

  // 2. Collect all shape trackers for kernel GlobalView ops.
  const sts: ShapeTracker[] = globalViews.map((gv) => gv.arg[1]);
  for (const st of sts) {
    if (!deepEqual(st.shape, shape))
      throw new Error("Invariant violation: GlobalView shape mismatch"); // sanity check
  }

  // 3. Apply heuristic optimizations based on the shape trackers.
  const dim = new TuneDims(shape);

  void reduction; // TODO: Edit the `dim` object to reflect new dims.

  // 4. Return the tuned kernel result.
  const newExp = exp.rewrite((exp) => {
    if (exp.op === AluOp.GlobalView) {
      return undefined; // TODO
    }
  });

  return { exp: newExp };
}
