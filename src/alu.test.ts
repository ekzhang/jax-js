import { expect, test } from "vitest";

import { AluExp, DType } from "./alu";

test("AluExp can be evaluated", () => {
  const e = AluExp.i32(3);
  expect(e.evaluate({})).toEqual(3);

  const e2 = AluExp.add(AluExp.i32(3), AluExp.i32(4));
  expect(e2.evaluate({})).toEqual(7);

  const e3 = AluExp.add(AluExp.i32(3), e2);
  expect(e3.evaluate({})).toEqual(10);

  const e4 = AluExp.mul(AluExp.special(DType.Int32, "idx", 10), AluExp.i32(50));
  expect(e4.evaluate({ idx: 10 })).toEqual(500);
});

test("AluExp works with ternaries", () => {
  const x = AluExp.special(DType.Int32, "x", 100);

  const e = AluExp.where(
    AluExp.cmplt(x, AluExp.i32(70)),
    AluExp.i32(0),
    AluExp.i32(1),
  );
  expect(e.dtype).toBe(DType.Int32);
  expect(e.src).toHaveLength(3);
  expect(e.src[0].dtype).toBe(DType.Bool);
  expect(e.evaluate({ x: 50 })).toEqual(0);
  expect(e.evaluate({ x: 69 })).toEqual(0);
  expect(e.evaluate({ x: 70 })).toEqual(1);
  expect(e.evaluate({ x: 80 })).toEqual(1);
});

test("AluExp handles boolean ops", () => {
  const t = AluExp.bool(true);
  const f = AluExp.bool(false);

  expect(AluExp.mul(t, t).evaluate({})).toBe(1);
  expect(AluExp.mul(t, f).evaluate({})).toBe(0);
  expect(AluExp.mul(f, f).evaluate({})).toBe(0);

  expect(AluExp.add(t, t).evaluate({})).toBe(1);
  expect(AluExp.add(t, f).evaluate({})).toBe(1);
  expect(AluExp.add(f, f).evaluate({})).toBe(0);
});

test("AluExp has .min and .max", () => {
  const e = AluExp.add(AluExp.i32(3), AluExp.i32(4));
  expect(e.min).toEqual(7);
  expect(e.max).toEqual(7);

  const e2 = AluExp.add(
    AluExp.special(DType.Int32, "x", 10),
    AluExp.special(DType.Int32, "y", 20),
  );
  expect(e2.min).toEqual(0);
  expect(e2.max).toEqual(28);
});

test("AluExp raises TypeError for unsupported dtypes", () => {
  expect(() => AluExp.sin(AluExp.bool(true))).toThrow(TypeError);
  expect(() => AluExp.cos(AluExp.bool(false))).toThrow(TypeError);
  expect(() => AluExp.reciprocal(AluExp.bool(true))).toThrow(TypeError);
});

test("AluOp.Min and AluOp.Max", () => {
  const a = AluExp.i32(3);
  const b = AluExp.i32(4);
  const minOp = AluExp.min(a, b);
  expect(minOp.evaluate({})).toBe(3);
  expect(minOp.dtype).toBe(DType.Int32);

  const maxOp = AluExp.max(a, b);
  expect(maxOp.evaluate({})).toBe(4);
  expect(maxOp.dtype).toBe(DType.Int32);

  const c = AluExp.special(DType.Int32, "c", 5);
  const minOp2 = AluExp.min(a, c);
  expect(minOp2.evaluate({ c: 2 })).toBe(2);
});

test("AluOp.Exp", () => {
  const e = AluExp.exp(AluExp.f32(3));
  expect(e.evaluate({})).toBeCloseTo(Math.E ** 3);
  expect(e.dtype).toBe(DType.Float32);

  const e2 = AluExp.exp(AluExp.f32(0.25));
  expect(e2.evaluate({})).toBeCloseTo(Math.E ** 0.25);
});

test("AluOp.Log", () => {
  const e = AluExp.log(AluExp.f32(8));
  expect(e.evaluate({})).toBeCloseTo(Math.log(8));
  expect(e.dtype).toBe(DType.Float32);

  const e2 = AluExp.log(AluExp.f32(0.25));
  expect(e2.evaluate({})).toBeCloseTo(Math.log(0.25));

  const e3 = AluExp.log(AluExp.f32(-1));
  expect(e3.evaluate({})).toBeNaN();
});

test("AluOp.Sqrt", () => {
  const e = AluExp.sqrt(AluExp.f32(16));
  expect(e.evaluate({})).toBe(4);
  expect(e.dtype).toBe(DType.Float32);
});

test("AluOp.Cast", () => {
  const e = AluExp.cast(DType.Float32, AluExp.i32(42));
  expect(e.evaluate({})).toBe(42);
  expect(e.dtype).toBe(DType.Float32);

  const e2 = AluExp.cast(DType.Int32, AluExp.f32(3.14));
  expect(e2.evaluate({})).toBe(3);
  expect(e2.dtype).toBe(DType.Int32);

  const e3 = AluExp.cast(DType.Int32, AluExp.bool(true));
  expect(e3.evaluate({})).toBe(1);
  expect(e3.dtype).toBe(DType.Int32);

  const e4 = AluExp.cast(DType.Bool, AluExp.i32(5));
  expect(e4.evaluate({})).toBe(1);
  expect(e4.dtype).toBe(DType.Bool);

  const e5 = AluExp.cast(DType.Bool, AluExp.i32(0));
  expect(e5.evaluate({})).toBe(0);
  expect(e5.dtype).toBe(DType.Bool);
});

test("AluOp.Bitcast", () => {
  // Assumes little-endian byte order, which is used in modern browsers.
  const e = AluExp.bitcast(DType.Float32, AluExp.i32(0x3f800000)); // 1.0 in IEEE 754
  expect(e.evaluate({})).toBe(1.0);
  expect(e.dtype).toBe(DType.Float32);

  // Try also bitcasting Infinity and NaN.
  const e2 = AluExp.bitcast(DType.Int32, AluExp.f32(Infinity));
  expect(e2.evaluate({})).toBe(0x7f800000); // IEEE 754 representation of Infinity
  const e3 = AluExp.bitcast(DType.Int32, AluExp.f32(NaN));
  expect(e3.evaluate({})).toBe(0x7fc00000); // IEEE 754 representation of NaN

  // Cast two's complement integers.
  const e4 = AluExp.bitcast(DType.Uint32, AluExp.i32(-1));
  expect(e4.evaluate({})).toBe(0xffffffff); // -1 in two's complement is 0xffffffff in uint32

  // Invalid bitcast types should throw.
  expect(() => AluExp.bitcast(DType.Bool, AluExp.i32(1))).toThrow(TypeError);
  expect(() => AluExp.bitcast(DType.Float32, AluExp.bool(true))).toThrow(
    TypeError,
  );
  expect(() => AluExp.bitcast(DType.Int32, AluExp.bool(false))).toThrow(
    TypeError,
  );
  expect(() => AluExp.bitcast(DType.Uint32, AluExp.bool(true))).toThrow(
    TypeError,
  );
  expect(() => AluExp.bitcast(DType.Bool, AluExp.f32(1.0))).toThrow(TypeError);
});

test("AluOp.Threefry2x32", () => {
  const k0 = AluExp.u32(0);
  const k1 = AluExp.u32(0);
  const c0 = AluExp.u32(0);
  const c1 = AluExp.u32(0);
  const exp = AluExp.threefry2x32(k0, k1, c0, c1);
  expect(exp.evaluate({})).toBe((1797259609 ^ 2579123966) >>> 0); // x0 ^ x1
});

test("AluOp.Idiv", () => {
  // Make sure that idiv uses truncating division.
  const e = AluExp.idiv(AluExp.i32(7), AluExp.i32(3));
  expect(e.evaluate({})).toBe(2);

  const e2 = AluExp.idiv(AluExp.i32(-7), AluExp.i32(3));
  expect(e2.evaluate({})).toBe(-2);
  expect(e2.min).toBe(-2);
  expect(e2.max).toBe(-2);
});

test("AluOp.Mod", () => {
  // Make sure that mod uses the sign of the numerator.
  const e = AluExp.mod(AluExp.i32(7), AluExp.i32(3));
  expect(e.evaluate({})).toBe(1);

  const e2 = AluExp.mod(AluExp.i32(-7), AluExp.i32(3));
  expect(e2.evaluate({})).toBe(-1);
  expect(e2.min).toBeLessThanOrEqual(-1);
  expect(e2.max).toBeGreaterThanOrEqual(-1);

  const e3 = AluExp.mod(AluExp.i32(7), AluExp.i32(-3));
  expect(e3.evaluate({})).toBe(1);
  expect(e3.min).toBeLessThanOrEqual(1);
  expect(e3.max).toBeGreaterThanOrEqual(1);

  // Floating point mod also works.
  const e4 = AluExp.mod(AluExp.f32(5.5), AluExp.f32(-1.7));
  expect(e4.evaluate({})).toBeCloseTo(0.4);
});
