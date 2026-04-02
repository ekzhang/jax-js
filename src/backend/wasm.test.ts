import { expect, suite, test } from "vitest";

import {
  accessorGlobal,
  AluExp,
  AluOp,
  AluVar,
  DType,
  Kernel,
  Reduction,
} from "../alu";
import { init } from "../backend";
import { ShapeTracker, unravelAlu } from "../shape";
import { analyzeStride, isSimdEligible } from "./wasm";
import { tuneNullopt } from "../tuner";

await init();

suite("isSimdEligible", () => {
  test("contiguous pointwise f32 kernel is eligible", () => {
    const shape = ShapeTracker.fromShape([16]);
    const gidx = AluVar.gidx;
    const arg = accessorGlobal(DType.Float32, 0, shape, [gidx]);
    const kernel = new Kernel(1, 16, AluExp.add(arg, AluExp.f32(2)));
    expect(isSimdEligible(kernel.exp, kernel)).toBe(true);
  });

  test("f64 dtype is not eligible", () => {
    const shape = ShapeTracker.fromShape([16]);
    const gidx = AluVar.gidx;
    const arg = accessorGlobal(DType.Float64, 0, shape, [gidx]);
    const kernel = new Kernel(1, 16, arg);
    expect(isSimdEligible(kernel.exp, kernel)).toBe(false);
  });

  test("i32 dtype is eligible", () => {
    const shape = ShapeTracker.fromShape([16]);
    const gidx = AluVar.gidx;
    const arg = accessorGlobal(DType.Int32, 0, shape, [gidx]);
    const kernel = new Kernel(1, 16, AluExp.add(arg, AluExp.i32(2)));
    expect(isSimdEligible(kernel.exp, kernel)).toBe(true);
  });

  test("size less than 4 is not eligible", () => {
    const shape = ShapeTracker.fromShape([3]);
    const gidx = AluVar.gidx;
    const arg = accessorGlobal(DType.Float32, 0, shape, [gidx]);
    const kernel = new Kernel(1, 3, AluExp.add(arg, AluExp.f32(2)));
    expect(isSimdEligible(kernel.exp, kernel)).toBe(false);
  });

  test("f32 reduction with output size >= 4 is eligible", () => {
    const shape = ShapeTracker.fromShape([4, 8]);
    const view = AluExp.globalView(DType.Float32, 0, shape, [
      ...unravelAlu([4], AluVar.gidx),
      AluVar.ridx,
    ]);
    const reduction = new Reduction(DType.Float32, AluOp.Add, 8);
    const kernel = new Kernel(1, 4, view, reduction);
    const tune = tuneNullopt(kernel);
    expect(isSimdEligible(tune.exp, kernel)).toBe(true);
  });

  test("non-f32 reduction is not eligible", () => {
    const shape = ShapeTracker.fromShape([4, 8]);
    const view = AluExp.globalView(DType.Float64, 0, shape, [
      ...unravelAlu([4], AluVar.gidx),
      AluVar.ridx,
    ]);
    const reduction = new Reduction(DType.Float64, AluOp.Add, 8);
    const kernel = new Kernel(1, 4, view, reduction);
    const tune = tuneNullopt(kernel);
    expect(isSimdEligible(tune.exp, kernel)).toBe(false);
  });

  test("unsupported op (Sin) is not eligible", () => {
    const shape = ShapeTracker.fromShape([16]);
    const gidx = AluVar.gidx;
    const arg = accessorGlobal(DType.Float32, 0, shape, [gidx]);
    const kernel = new Kernel(1, 16, AluExp.sin(arg));
    expect(isSimdEligible(kernel.exp, kernel)).toBe(false);
  });
});

suite("analyzeStride", () => {
  const gidx = AluExp.special(DType.Int32, "gidx", 64);
  const c = (n: number) => AluExp.i32(n);

  test("Mod(gidx, 8) is contiguous with tileSize 8", () => {
    const exp = AluExp.mod(gidx, c(8));
    expect(analyzeStride(exp)).toEqual({ kind: "contiguous", tileSize: 8 });
  });

  test("Idiv(Mod(gidx, 6), 4) is gather due to fragment risk", () => {
    // tileSize=6 from Mod doesn't divide evenly by N=4. The last group
    // before the Mod wraps is only 2 wide so a SIMD group could straddle it.
    const exp = AluExp.idiv(AluExp.mod(gidx, c(6)), c(4));
    expect(analyzeStride(exp).kind).toBe("gather");
  });

  test("Mod(Mod(gidx, 5), 3) is gather due to fragment risk", () => {
    // Inner tileSize=5 doesn't divide by N=3. Same issue, Mod branch.
    const exp = AluExp.mod(AluExp.mod(gidx, c(5)), c(3));
    expect(analyzeStride(exp).kind).toBe("gather");
  });

  test("Mul(gidx, 3) is gather (stride != 1)", () => {
    // Contiguous * C produces stride C, which is not 1,  must gather.
    const exp = AluExp.mul(gidx, c(3));
    expect(analyzeStride(exp).kind).toBe("gather");
  });

  test("Mul(broadcast, C) stays broadcast", () => {
    // Broadcast * C = still broadcast (0 * C = 0). Must not downgrade to gather.
    const exp = AluExp.mul(AluExp.idiv(gidx, c(4)), c(8));
    expect(analyzeStride(exp).kind).toBe("broadcast");
  });
});
