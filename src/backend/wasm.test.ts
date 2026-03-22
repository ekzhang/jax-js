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
import { ShapeTracker } from "../shape";
import { isSimdEligible } from "./wasm";

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

  test("non-contiguous (flipped) input is not eligible", () => {
    const shape = ShapeTracker.fromShape([16]);
    const gidx = AluVar.gidx;
    const arg = accessorGlobal(DType.Float32, 0, shape.flip([true]), [gidx]);
    const kernel = new Kernel(1, 16, AluExp.add(arg, AluExp.f32(2)));
    expect(isSimdEligible(kernel.exp, kernel)).toBe(false);
  });

  test("f64 dtype is not eligible", () => {
    const shape = ShapeTracker.fromShape([16]);
    const gidx = AluVar.gidx;
    const arg = accessorGlobal(DType.Float64, 0, shape, [gidx]);
    const kernel = new Kernel(1, 16, arg);
    expect(isSimdEligible(kernel.exp, kernel)).toBe(false);
  });

  test("i32 dtype is not eligible", () => {
    const shape = ShapeTracker.fromShape([16]);
    const gidx = AluVar.gidx;
    const arg = accessorGlobal(DType.Int32, 0, shape, [gidx]);
    const kernel = new Kernel(1, 16, AluExp.add(arg, AluExp.i32(2)));
    expect(isSimdEligible(kernel.exp, kernel)).toBe(false);
  });

  test("reduction kernel is not eligible", () => {
    const shape = ShapeTracker.fromShape([4, 4]);
    const exp = AluExp.globalView(DType.Float32, 0, shape, [
      AluVar.gidx,
      AluVar.ridx,
    ]);
    const reduction = new Reduction(DType.Float32, AluOp.Add, 4);
    const kernel = new Kernel(1, 4, exp, reduction);
    expect(isSimdEligible(kernel.exp, kernel)).toBe(false);
  });

  test("size less than 4 is not eligible", () => {
    const shape = ShapeTracker.fromShape([3]);
    const gidx = AluVar.gidx;
    const arg = accessorGlobal(DType.Float32, 0, shape, [gidx]);
    const kernel = new Kernel(1, 3, AluExp.add(arg, AluExp.f32(2)));
    expect(isSimdEligible(kernel.exp, kernel)).toBe(false);
  });
});

