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
import { isSimdEligible } from "./wasm";
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

  test("multiple contiguous f32 inputs is eligible", () => {
    const shape = ShapeTracker.fromShape([16]);
    const gidx = AluVar.gidx;
    const arg1 = accessorGlobal(DType.Float32, 0, shape, [gidx]);
    const arg2 = accessorGlobal(DType.Float32, 1, shape, [gidx]);
    const kernel = new Kernel(2, 16, AluExp.add(arg1, arg2));
    expect(isSimdEligible(kernel.exp, kernel)).toBe(true);
  });

  test("non-contiguous (flipped) input is eligible (gather fallback)", () => {
    const shape = ShapeTracker.fromShape([16]);
    const gidx = AluVar.gidx;
    const arg = accessorGlobal(DType.Float32, 0, shape.flip([true]), [gidx]);
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

  test("f32 reduction with size >= 4 is eligible", () => {
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

  test("f32 reduction with size < 4 is not eligible", () => {
    const shape = ShapeTracker.fromShape([4, 3]);
    const view = AluExp.globalView(DType.Float32, 0, shape, [
      ...unravelAlu([4], AluVar.gidx),
      AluVar.ridx,
    ]);
    const reduction = new Reduction(DType.Float32, AluOp.Add, 3);
    const kernel = new Kernel(1, 4, view, reduction);
    const tune = tuneNullopt(kernel);
    expect(isSimdEligible(tune.exp, kernel)).toBe(false);
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
