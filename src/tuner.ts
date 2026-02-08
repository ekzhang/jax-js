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
import { DEBUG, deepEqual, lexCompare, prod, range, sorted } from "./utils";

export interface TuneResult {
  /** New expression with GlobalView ops and gidx/ridx lowered. */
  exp: AluExp;

  /** New reduction epilogue expression, present when `kernel.reduction` is present. */
  epilogue?: AluExp;

  /** Expression for indexing the result array, including upcast. */
  outputIdxExp: AluExp;

  /** How many total threads to dispatch in the grid. */
  threadCount: number;

  /** Sizes of various dimensions of the kernel. */
  size: {
    /**
     * Number of threads for each group.
     * If greater than 1, group index is available as `AluExp.special("group")`.
     */
    groups?: number;

    /** Number of iterations for the reduce loop, `AluExp.special("ridx")`. */
    reduce: number;

    /** Amount to upcast in reduce loop, set via `AluVar.unroll`. */
    unroll?: number;

    /** Amount to upcast in non-reduce dimensions, set via `AluVar.upcast`. */
    upcast?: number;
  };
}

/** Stores dimensions of the kernel's applied shape. Globals start at 0. */
class TuneDims {
  st: ShapeTracker; // Shape tracker including reduction axes.
  outputSt: ShapeTracker; // Shape tracker including only output axes.

  // local: number; // TODO: Split gidx -> global and local axes during tuning.
  groups: number; // Reductions start here, with groups.
  reduce: number; // Single reduction thread.
  unroll: number; // Upcast along the reduce dimension.
  upcast: number; // Upcast along output dimension.

  get end() {
    return this.st.shape.length;
  }

  constructor(shape: number[]) {
    this.st = ShapeTracker.fromShape(shape);
    this.outputSt = ShapeTracker.fromShape(shape.slice(0, -1));
    this.groups = this.st.shape.length - 1;
    this.reduce = this.st.shape.length - 1;
    this.unroll = this.st.shape.length;
    this.upcast = this.st.shape.length;
  }

  // Place the axis at the end of the shape, so it is part of each workgroup.
  applyLocal(axis: number, amount: number) {
    if (axis >= this.groups) throw new Error("Cannot localize reduction axis");
    const length = this.st.shape[axis];
    if (length % amount !== 0)
      throw new Error(`Localize by ${amount} on axis length ${length}`);

    if (length !== amount) {
      // First split it.
      (this.groups++, this.reduce++, this.unroll++, this.upcast++);
      this.st = this.st.reshape([
        ...this.st.shape.slice(0, axis),
        length / amount,
        amount,
        ...this.st.shape.slice(axis + 1),
      ]);
      this.outputSt = this.outputSt.reshape([
        ...this.outputSt.shape.slice(0, axis),
        length / amount,
        amount,
        ...this.outputSt.shape.slice(axis + 1),
      ]);
      axis++;
    }

    // Now permute axis to the end of the real axes, before groups.
    this.st = this.st.permute([
      ...range(axis),
      ...range(axis + 1, this.groups),
      axis,
      ...range(this.groups, this.st.shape.length),
    ]);
    this.outputSt = this.outputSt.permute([
      ...range(axis),
      ...range(axis + 1, this.groups),
      axis,
      ...range(this.groups, this.outputSt.shape.length),
    ]);
  }

  applyUpcast(axis: number, amount: number) {
    if (axis >= this.groups)
      throw new Error("Cannot upcast along reduction axis");
    const length = this.st.shape[axis];
    if (length % amount !== 0)
      throw new Error(`Upcast by ${amount} on axis length ${length}`);
    this.st = this.st
      .reshape([
        ...this.st.shape.slice(0, axis),
        length / amount,
        amount,
        ...this.st.shape.slice(axis + 1),
      ])
      .permute([
        ...range(axis + 1),
        ...range(axis + 2, this.st.shape.length + 1),
        axis + 1,
      ]);
    this.outputSt = this.outputSt
      .reshape([
        ...this.outputSt.shape.slice(0, axis),
        length / amount,
        amount,
        ...this.outputSt.shape.slice(axis + 1),
      ])
      .permute([
        ...range(axis + 1),
        ...range(axis + 2, this.outputSt.shape.length + 1),
        axis + 1,
      ]);
  }

  applyUnroll(axis: number, amount: number) {
    if (axis < this.groups) throw new Error("Cannot unroll non-reduce axis");
    if (axis >= this.unroll) throw new Error("Axis already unrolled");
    const length = this.st.shape[axis];
    if (length % amount !== 0)
      throw new Error(`Unroll by ${amount} on axis length ${length}`);
    // We're unrolling away the whole axis.
    if (length === amount) {
      this.st = this.st.permute([
        ...range(axis),
        ...range(axis + 1, this.upcast),
        axis,
        ...range(this.upcast, this.st.shape.length),
      ]);
      if (axis < this.reduce) this.reduce--;
      this.unroll--;
    } else {
      this.st = this.st
        .reshape([
          ...this.st.shape.slice(0, axis),
          length / amount,
          amount,
          ...this.st.shape.slice(axis + 1),
        ])
        .permute([
          ...range(axis + 1),
          ...range(axis + 2, this.upcast + 1), // Move to just before upcast
          axis + 1,
          ...range(this.upcast + 1, this.st.shape.length + 1),
        ]);
      this.upcast++;
    }
  }
}

/** Tuning step that does not apply any optimization. */
export function tuneNullopt(kernel: Kernel): TuneResult {
  const vars: Record<string, AluExp> = {};
  vars.gidx = AluExp.special(DType.Int32, "gidx", kernel.size);
  if (kernel.reduction)
    vars.ridx = AluExp.special(DType.Int32, "ridx", kernel.reduction.size);
  return {
    exp: kernel.exp.substitute(vars).rewriteGlobalViews().simplify(),
    epilogue: kernel.reduction?.epilogue
      .substitute({ gidx: vars.gidx })
      .rewriteGlobalViews()
      .simplify(),
    outputIdxExp: vars.gidx,
    threadCount: kernel.size,
    size: {
      reduce: kernel.reduction ? kernel.reduction.size : 0,
    },
  };
}

/** Tuning for WebGPU kernels. */
export function tuneWebgpu(kernel: Kernel): TuneResult {
  const { exp, reduction } = kernel;
  if (!reduction) return tuneNullopt(kernel);

  const globalIndexes = exp.collect((exp) => exp.op === AluOp.GlobalIndex);
  if (globalIndexes.length > 0) {
    if (DEBUG >= 4)
      console.info("Tuning: Found GlobalIndex ops, skipping opt.");
    return tuneNullopt(kernel);
  }

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
  ].map((e) => e.simplify());
  for (const gv of globalViews) {
    if (!gv.src.length || !deepEqual(gv.src, expectedSrc)) {
      if (DEBUG >= 4)
        console.info("Tuning: GlobalView src[] not consistent with reduction.");
      return tuneNullopt(kernel);
    }
  }
  if (shape[shape.length - 1] !== reduction.size)
    throw new Error("Invariant violation: shape doesn't match reduction size.");

  // 2. Collect all shape trackers for kernel GlobalView ops.
  const sts: ShapeTracker[] = globalViews.map((gv) => gv.arg[1]);
  for (const st of sts) {
    if (!deepEqual(st.shape, shape))
      throw new Error("Invariant violation: GlobalView shape mismatch"); // sanity check
  }

  // 3. Apply heuristic optimizations based on the shape trackers.
  const dim = new TuneDims(shape);

  // Try to do upcasts of non-reduce axes for global memory coalescing.
  // Heuristic is based on strides, and borrowed from tinygrad.
  const upcastedAxis = new Set<number>();
  while (prod(dim.st.shape.slice(0, dim.groups)) >= 1024) {
    const choices: number[][] = [];
    const composedSts = sts.map((st) => st.compose(dim.st));
    for (let axis = 0; axis < dim.groups; axis++) {
      for (const amount of [3, 4, 5]) {
        // Axis is not upcasted, divisible, and has a buffer with stride 0 on
        // that axis (mem coalescing) while not already a stride-0 upcast.
        if (
          !upcastedAxis.has(axis) &&
          dim.st.shape[axis] % amount === 0 &&
          composedSts.some(
            (st) =>
              st.lastStrides[axis] === 0 &&
              st.lastStrides.slice(dim.unroll).every((stride) => stride > 0),
          )
        ) {
          let nonzeroStrides = 0;
          let totalStrides = 0;
          for (const st of composedSts) {
            nonzeroStrides += st.lastStrides[axis] > 0 ? 1 : 0;
            totalStrides += st.lastStrides[axis];
          }
          choices.push([nonzeroStrides, totalStrides, axis, amount]);
        }
      }
    }
    if (choices.length > 0) {
      choices.sort(lexCompare);
      dim.applyUpcast(choices[0][2], choices[0][3]);
      upcastedAxis.add(choices[0][2]);
    } else {
      break;
    }
  }

  // Try to do loop unrolling on the reduce axis, with an upcast limit.
  // Skip doing this on mobile browsers, as it may reduce performance.
  if (
    !/Mobi|Android/i.test(navigator.userAgent) &&
    dim.reduce < dim.unroll &&
    (prod(dim.st.shape.slice(dim.unroll)) <= 4 ||
      (dim.unroll === dim.upcast && prod(dim.st.shape.slice(dim.upcast)) < 64))
  ) {
    // Fully unroll the reduce axis.
    const s = dim.st.shape[dim.unroll - 1];
    if (0 < s && s <= 32) {
      dim.applyUnroll(dim.reduce, s);
    } else {
      // Partially unroll the reduce axis.
      //
      // Note: Unrolling by 8 previously made this faster in January 2026, but
      // in later versions of Chrome on macOS, it seems to have regressed 40%.
      // Seems like 4 is a more stable choice at the moment.
      for (const splits of [4, 2]) {
        if (s % splits === 0) {
          dim.applyUnroll(dim.unroll - 1, splits);
          break;
        }
      }
    }
  }

  for (const ax of sorted(upcastedAxis)) {
    const s = dim.st.shape[ax];
    // TODO: These applyLocal() calls are a hack / bad heuristic, make this better.
    for (const amount of [8, 4]) {
      if (s % amount === 0) {
        dim.applyLocal(ax, amount);
        break;
      }
    }
  }

  // 4. Return the tuned kernel result.
  const indices: AluExp[] = [];
  const addIndices = (s: number[], exp: AluExp) => {
    if (s.length === 0) return;
    else if (s.length === 1) indices.push(exp);
    else indices.push(...unravelAlu(s, exp));
  };
  if (0 < dim.groups) {
    const s = dim.st.shape.slice(0, dim.groups);
    addIndices(s, AluExp.special(DType.Int32, "gidx", prod(s)));
  }
  if (dim.groups < dim.reduce) {
    const s = dim.st.shape.slice(dim.groups, dim.reduce);
    addIndices(s, AluExp.special(DType.Int32, "group", prod(s)));
  }
  if (dim.reduce <= dim.unroll) {
    const s = dim.st.shape.slice(dim.reduce, dim.unroll);
    addIndices(s, AluExp.special(DType.Int32, "ridx", prod(s)));
  }
  if (dim.unroll < dim.upcast) {
    const s = dim.st.shape.slice(dim.unroll, dim.upcast);
    addIndices(s, AluVar.unroll);
  }
  if (dim.upcast < dim.end) {
    const s = dim.st.shape.slice(dim.upcast);
    addIndices(s, AluVar.upcast);
  }

  // Substitute old values of AluVar.gidx and AluVar.ridx.
  //
  // As an optimization to `.substitute(vars).rewriteGlobalViews()`, we compose
  // dim.st with the ShapeTracker of each GlobalView op, which generates better
  // code due to shape simplifications.
  let newExp = exp.rewrite((exp) => {
    if (exp.op === AluOp.GlobalView) {
      const gid: number = exp.arg[0];
      const st: ShapeTracker = exp.arg[1];
      return accessorGlobal(exp.dtype, gid, st.compose(dim.st), indices);
    }
  });
  // Substitute any remaining gidx/ridx variables not in views.
  const [iexpr, vexpr] = dim.st.toAluExp(indices);
  if (vexpr.min !== 1) throw new Error("Invariant violation: vexpr !== true");
  newExp = newExp.substitute({
    gidx: AluExp.idiv(iexpr, AluExp.i32(reduction.size)).simplify(),
    ridx: AluExp.mod(iexpr, AluExp.i32(reduction.size)).simplify(),
  });

  const outputGidx = dim.outputSt.shape.slice(0, dim.groups);
  const outputUpcast = dim.outputSt.shape.slice(dim.groups);
  const outputIndices = [
    ...unravelAlu(
      outputGidx,
      AluExp.special(DType.Int32, "gidx", prod(outputGidx)),
    ),
    ...unravelAlu(outputUpcast, AluVar.upcast), // Needs later substitution.
  ];
  const [outputIdxExp, _] = dim.outputSt.toAluExp(outputIndices);
  const newEpilogue = reduction.epilogue.rewrite((exp) => {
    if (exp.op === AluOp.GlobalView) {
      const gid: number = exp.arg[0];
      const st: ShapeTracker = exp.arg[1];
      return accessorGlobal(
        exp.dtype,
        gid,
        st.compose(dim.outputSt),
        outputIndices,
      );
    }
  });

  // Sanity-check that reduction size looks correct.
  if (prod(dim.st.shape.slice(dim.groups, dim.upcast)) !== reduction.size) {
    throw new Error(
      `Invariant violation: reduction size ${reduction.size} does not match ` +
        `tuned dims ${JSON.stringify(dim.st.shape.slice(dim.groups, dim.upcast))}`,
    );
  }

  const size = {
    groups: prod(dim.st.shape.slice(dim.groups, dim.reduce)),
    reduce: prod(dim.st.shape.slice(dim.reduce, dim.unroll)),
    unroll: prod(dim.st.shape.slice(dim.unroll, dim.upcast)),
    upcast: prod(dim.st.shape.slice(dim.upcast)),
  };

  return {
    exp: newExp.simplify(),
    epilogue: newEpilogue.simplify(),
    outputIdxExp: outputIdxExp.simplify(),
    threadCount: (kernel.size / size.upcast) * size.groups,
    size,
  };
}
