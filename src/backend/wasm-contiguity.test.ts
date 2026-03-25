/**
 * Tests for the post-lowering contiguity check (isContiguousWrt).
 *
 * These build kernels the way the JIT does (GlobalView + unravelAlu),
 * pass them through tuneNullopt to get real simplified GlobalIndex trees,
 * then verify that isContiguousWrt correctly identifies stride-1 accesses.
 *
 * Not for committing — exploratory tests for the SIMD contiguity analysis.
 */
import { expect, suite, test } from "vitest";

import { AluExp, AluOp, AluVar, DType, Kernel, Reduction } from "../alu";
import { ShapeTracker, unravelAlu } from "../shape";
import { tuneNullopt } from "../tuner";

/**
 * Check if a flat index expression has stride 1 w.r.t. a stepping variable.
 *
 * After simplification, a contiguous access looks like:
 * - just the variable: `gidx`
 * - the variable added to terms that don't involve it: `gidx * 8 + ridx`
 */
function isContiguousWrt(exp: AluExp, varName: string): boolean {
  const isVar = (e: AluExp): boolean =>
    e.op === AluOp.Special && e.arg[0] === varName;
  const containsVar = (e: AluExp): boolean =>
    isVar(e) || e.src.some(containsVar);

  if (isVar(exp)) return true;
  if (exp.op === AluOp.Add) {
    for (let i = 0; i < 2; i++) {
      if (isContiguousWrt(exp.src[i], varName) && !containsVar(exp.src[1 - i]))
        return true;
    }
  }
  return false;
}

/** Helper: build a pointwise kernel the way the JIT does for realized inputs. */
function pointwiseKernel(
  shapes: number[][],
  buildExp: (views: AluExp[]) => AluExp,
): Kernel {
  const views = shapes.map((shape, i) => {
    const st = ShapeTracker.fromShape(shape);
    const indices = unravelAlu(shape, AluVar.gidx);
    return AluExp.globalView(DType.Float32, i, st, indices);
  });
  const size = shapes[0].reduce((a, b) => a * b, 1);
  return new Kernel(shapes.length, size, buildExp(views));
}

/** Helper: build a pointwise kernel with a non-standard ShapeTracker. */
function pointwiseKernelWithSt(
  sts: ShapeTracker[],
  buildExp: (views: AluExp[]) => AluExp,
): Kernel {
  const views = sts.map((st, i) => {
    const indices = unravelAlu(st.shape, AluVar.gidx);
    return AluExp.globalView(DType.Float32, i, st, indices);
  });
  const size = sts[0].shape.reduce((a, b) => a * b, 1);
  return new Kernel(sts.length, size, buildExp(views));
}

/** Helper: build a reduction kernel the way the JIT does. */
function reductionKernel(
  shape: number[],
  reductionSize: number,
  buildExp: (view: AluExp) => AluExp,
  reductionOp: AluOp = AluOp.Add,
): Kernel {
  const st = ShapeTracker.fromShape(shape);
  const indices = [...unravelAlu(shape.slice(0, -1), AluVar.gidx), AluVar.ridx];
  const view = AluExp.globalView(DType.Float32, 0, st, indices);
  const outputSize = shape.slice(0, -1).reduce((a, b) => a * b, 1);
  const reduction = new Reduction(DType.Float32, reductionOp, reductionSize);
  return new Kernel(1, outputSize, buildExp(view), reduction);
}

/** Collect GlobalIndex contiguity results from a tuned expression. */
function analyzeContiguity(kernel: Kernel): Map<number, boolean> {
  const tune = tuneNullopt(kernel);
  const steppingVar = kernel.reduction ? "ridx" : "gidx";
  const result = new Map<number, boolean>();
  tune.exp
    .collect((e) => e.op === AluOp.GlobalIndex)
    .forEach((gi) => {
      const [gid] = gi.arg as [number, number];
      const contig = isContiguousWrt(gi.src[0], steppingVar);
      // AND per-gid (same buffer accessed multiple times)
      result.set(gid, (result.get(gid) ?? true) && contig);
    });
  return result;
}

// ============================================================
// Pointwise kernels — contiguity w.r.t. gidx
// ============================================================

suite("pointwise contiguity (post-lowering)", () => {
  test("contiguous 1D: [16] identity load", () => {
    const kernel = pointwiseKernel([[16]], ([a]) => a);
    const result = analyzeContiguity(kernel);
    expect(result.get(0)).toBe(true);
  });

  test("contiguous 1D: [16] x + 2", () => {
    const kernel = pointwiseKernel([[16]], ([a]) =>
      AluExp.add(a, AluExp.f32(2)),
    );
    const result = analyzeContiguity(kernel);
    expect(result.get(0)).toBe(true);
  });

  test("contiguous 2D: [4,3] identity load", () => {
    const kernel = pointwiseKernel([[4, 3]], ([a]) => a);
    const result = analyzeContiguity(kernel);
    expect(result.get(0)).toBe(true);
  });

  test("contiguous 3D: [2,4,3] identity load", () => {
    const kernel = pointwiseKernel([[2, 4, 3]], ([a]) => a);
    const result = analyzeContiguity(kernel);
    expect(result.get(0)).toBe(true);
  });

  test("contiguous: two same-shape inputs x + y on [16]", () => {
    const kernel = pointwiseKernel([[16], [16]], ([a, b]) => AluExp.add(a, b));
    const result = analyzeContiguity(kernel);
    expect(result.get(0)).toBe(true);
    expect(result.get(1)).toBe(true);
  });

  test("contiguous: two same-shape 2D inputs x * y on [4,8]", () => {
    const kernel = pointwiseKernel(
      [
        [4, 8],
        [4, 8],
      ],
      ([a, b]) => AluExp.mul(a, b),
    );
    const result = analyzeContiguity(kernel);
    expect(result.get(0)).toBe(true);
    expect(result.get(1)).toBe(true);
  });

  test("contiguous: chained ops x.add(2).mul(3) on [16]", () => {
    const kernel = pointwiseKernel([[16]], ([a]) =>
      AluExp.mul(AluExp.add(a, AluExp.f32(2)), AluExp.f32(3)),
    );
    const result = analyzeContiguity(kernel);
    expect(result.get(0)).toBe(true);
  });

  test("non-contiguous: transposed [4,3] (permute [1,0])", () => {
    const st = ShapeTracker.fromShape([4, 3]).permute([1, 0]);
    // After permute, shape is [3,4] with strides [1,3] — not row-major
    const kernel = pointwiseKernelWithSt([st], ([a]) => a);
    const result = analyzeContiguity(kernel);
    expect(result.get(0)).toBe(false);
  });

  test("non-contiguous: flipped [16]", () => {
    const st = ShapeTracker.fromShape([16]).flip([true]);
    const kernel = pointwiseKernelWithSt([st], ([a]) => a);
    const result = analyzeContiguity(kernel);
    expect(result.get(0)).toBe(false);
  });

  test("non-contiguous: broadcast [1,8] → [4,8]", () => {
    const st = ShapeTracker.fromShape([1, 8]).expand([4, 8]);
    // Stride 0 on first dim — not consecutive
    const kernel = pointwiseKernelWithSt([st], ([a]) => a);
    const result = analyzeContiguity(kernel);
    expect(result.get(0)).toBe(false);
  });

  test("mixed: contiguous + non-contiguous inputs", () => {
    const st0 = ShapeTracker.fromShape([4, 3]);
    const st1 = ShapeTracker.fromShape([4, 3]).permute([1, 0]);
    // st1 after permute has shape [3,4] — but we need same output shape.
    // Actually let's use broadcast instead: one contiguous [8], one broadcast [1]→[8]
    const stContig = ShapeTracker.fromShape([8]);
    const stBroadcast = ShapeTracker.fromShape([1]).expand([8]);
    const views = [
      AluExp.globalView(
        DType.Float32,
        0,
        stContig,
        unravelAlu([8], AluVar.gidx),
      ),
      AluExp.globalView(
        DType.Float32,
        1,
        stBroadcast,
        unravelAlu([8], AluVar.gidx),
      ),
    ];
    const kernel = new Kernel(2, 8, AluExp.add(views[0], views[1]));
    const result = analyzeContiguity(kernel);
    expect(result.get(0)).toBe(true);
    expect(result.get(1)).toBe(false);
  });
});

// ============================================================
// Reduction kernels — contiguity w.r.t. ridx
// ============================================================

suite("reduction contiguity (post-lowering)", () => {
  test("contiguous: [4,8] sum axis=1 (row sum)", () => {
    const kernel = reductionKernel([4, 8], 8, (v) => v);
    const result = analyzeContiguity(kernel);
    expect(result.get(0)).toBe(true);
  });

  test("contiguous: [2,4,8] sum axis=2", () => {
    const kernel = reductionKernel([2, 4, 8], 8, (v) => v);
    const result = analyzeContiguity(kernel);
    expect(result.get(0)).toBe(true);
  });

  test("contiguous: [100] sum (1D reduction)", () => {
    // 1D sum: shape [100], output size 1, reduction size 100
    const st = ShapeTracker.fromShape([100]);
    const view = AluExp.globalView(DType.Float32, 0, st, [AluVar.ridx]);
    const reduction = new Reduction(DType.Float32, AluOp.Add, 100);
    const kernel = new Kernel(1, 1, view, reduction);
    const result = analyzeContiguity(kernel);
    expect(result.get(0)).toBe(true);
  });

  test("non-contiguous: [4,8] sum axis=0 (column sum)", () => {
    // Column sum: output shape [8], reduction over rows (axis 0).
    // Shape [8,4] with ridx indexing the first dim (stride 4).
    // Actually, let's think about how the JIT would represent this.
    // For sum(axis=0) on [4,8]: output size = 8, reduction size = 4.
    // The array is reshaped/permuted so reduction axis is last.
    // That means shape becomes [8,4] with strides [1,8] (transposed).
    const st = ShapeTracker.fromShape([4, 8]).permute([1, 0]);
    // st.shape = [8, 4], strides = [1, 8] — not consecutive
    const indices = [...unravelAlu([8], AluVar.gidx), AluVar.ridx];
    const view = AluExp.globalView(DType.Float32, 0, st, indices);
    const reduction = new Reduction(DType.Float32, AluOp.Add, 4);
    const kernel = new Kernel(1, 8, view, reduction);
    const result = analyzeContiguity(kernel);
    expect(result.get(0)).toBe(false);
  });

  test("contiguous: [4,8] max axis=1", () => {
    const kernel = reductionKernel([4, 8], 8, (v) => v, AluOp.Max);
    const result = analyzeContiguity(kernel);
    expect(result.get(0)).toBe(true);
  });
});

// ============================================================
// isContiguousWrt unit tests — direct checks on expression structure
// ============================================================

suite("isContiguousWrt direct", () => {
  const gidx = AluExp.special(DType.Int32, "gidx", 16);
  const ridx = AluExp.special(DType.Int32, "ridx", 8);

  test("bare gidx", () => {
    expect(isContiguousWrt(gidx, "gidx")).toBe(true);
  });

  test("bare ridx", () => {
    expect(isContiguousWrt(ridx, "ridx")).toBe(true);
  });

  test("gidx + constant offset", () => {
    const exp = AluExp.add(gidx, AluExp.i32(5)).simplify();
    expect(isContiguousWrt(exp, "gidx")).toBe(true);
  });

  test("gidx * stride (not contiguous)", () => {
    const exp = AluExp.mul(gidx, AluExp.i32(3)).simplify();
    expect(isContiguousWrt(exp, "gidx")).toBe(false);
  });

  test("gidx * 8 + ridx — contiguous w.r.t. ridx", () => {
    const exp = AluExp.add(AluExp.mul(gidx, AluExp.i32(8)), ridx).simplify();
    expect(isContiguousWrt(exp, "ridx")).toBe(true);
  });

  test("gidx * 8 + ridx — NOT contiguous w.r.t. gidx", () => {
    const exp = AluExp.add(AluExp.mul(gidx, AluExp.i32(8)), ridx).simplify();
    expect(isContiguousWrt(exp, "gidx")).toBe(false);
  });

  test("ridx * 2 — not contiguous (stride 2)", () => {
    const exp = AluExp.mul(ridx, AluExp.i32(2)).simplify();
    expect(isContiguousWrt(exp, "ridx")).toBe(false);
  });

  test("constant (no variable at all)", () => {
    const exp = AluExp.i32(42);
    expect(isContiguousWrt(exp, "gidx")).toBe(false);
  });

  test("wrong variable name", () => {
    expect(isContiguousWrt(gidx, "ridx")).toBe(false);
  });
});
