import { defaultDevice, jit, numpy as np } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

beforeEach(() => {
  defaultDevice("cpu"); // float16 is not available on Wasm
});

suite("dtype-op edge cases", () => {
  test("uint32 subtraction underflow", () => {
    const a = np.array(3, { dtype: np.uint32 });
    const b = np.array(5, { dtype: np.uint32 });
    const c = a.sub(b);
    expect(c.dtype).toBe(np.uint32);
    expect(c.js()).toEqual(4294967294); // 2^32 - 2
  });
});

suite("dtype promotion rules", () => {
  test("promote uint32 and int32 to int32", () => {
    const a = np.array(3, { dtype: np.uint32 });
    const b = np.array(-2, { dtype: np.int32 });
    const c = a.add(b);
    expect(c.dtype).toBe(np.int32);
    expect(c.js()).toEqual(1);
  });

  test("promote int32 and float16 to float16", () => {
    const a = np.array(3, { dtype: np.int32 });
    const b = np.array(2.5, { dtype: np.float16 });
    const c = a.mul(b);
    expect(c.dtype).toBe(np.float16);
    expect(c.js()).toEqual(7.5);
  });

  test("promote uint32 and float32 to float32", () => {
    const a = np.array(4, { dtype: np.uint32 });
    const b = np.array(1.5, { dtype: np.float32 });
    const c = a.sub(b);
    expect(c.dtype).toBe(np.float32);
    expect(c.js()).toEqual(2.5);
  });

  test("promote bool and int32 to int32", () => {
    const a = np.array(true, { dtype: np.bool });
    const b = np.array(10, { dtype: np.int32 });
    const c = a.add(b);
    expect(c.dtype).toBe(np.int32);
    expect(c.js()).toEqual(11);
  });

  test("promote float16 and float32 to float32", () => {
    const a = np.array(2.5, { dtype: np.float16 });
    const b = np.array(1.5, { dtype: np.float32 });
    const c = a.div(b);
    expect(c.dtype).toBe(np.float32);
    expect(c).toBeAllclose(2.5 / 1.5);
  });
});

suite("weak types", () => {
  test("number constants are weak", () => {
    const a = np.array(5);
    expect(a.dtype).toBe(np.float32);
    expect(a.weakType).toBe(true);
    a.dispose();
    const b = np.multiply(3, 5);
    expect(b.dtype).toBe(np.float32);
    expect(b.weakType).toBe(true);
    b.dispose();
  });

  test("bool constants are not weak type", () => {
    const a = np.array(true);
    expect(a.dtype).toBe(np.bool);
    expect(a.weakType).toBe(false);
    a.dispose();
    const b = np.array([true, false]);
    expect(b.dtype).toBe(np.bool);
    expect(b.weakType).toBe(false);
    b.dispose();
  });

  test("arrays of numbers are not weak", () => {
    const a = np.array([1, 2, 3]);
    expect(a.dtype).toBe(np.float32);
    expect(a.weakType).toBe(false);
    a.dispose();
  });

  test("constant as operand is cast to int32", () => {
    const a = np.array(5, { dtype: np.int32 });
    const b = a.add(3); // 3 is a JS number constant
    expect(b.dtype).toBe(np.int32);
    expect(b.weakType).toBe(false);
    b.dispose();
  });

  test("constant as operand is cast to uint32", () => {
    const a = np.array(5, { dtype: np.uint32 });
    const b = a.add(2.8); // Should truncate to 2, which fits in uint32
    expect(b.dtype).toBe(np.uint32);
    expect(b.weakType).toBe(false);
    expect(b.js()).toEqual(7);
  });

  test("ops preserve weak float", () => {
    const a = np.array(5, { dtype: np.int32 });
    const b = a.add(np.multiply(3, 3));
    expect(b.dtype).toBe(np.int32);
    expect(b.weakType).toBe(false);
    expect(b.js()).toEqual(14);
  });

  test("weak type in jit constants", () => {
    const f = jit(() => {
      return np.sin(3);
    });
    let a = f();
    expect(a.dtype).toBe(np.float32);
    expect(a.weakType).toBe(true);
    a = a.add(np.array(2, { dtype: np.float16 }));
    expect(a.dtype).toBe(np.float16);
    expect(a.weakType).toBe(false);
    expect(a.js()).toBeCloseTo(Math.sin(3) + 2, 2);
  });

  test("weak type added in jit op", () => {
    const f = jit((x: np.Array) => x.add(3));
    for (const dtype of [np.int32, np.float32]) {
      const a = np.array(4, { dtype });
      expect(a.weakType).toBe(false);
      const b = f(a);
      expect(b.dtype).toBe(dtype);
      expect(b.weakType).toBe(false);
      expect(b.js()).toEqual(7);
    }
  });

  test("weak type preserved by jit op", () => {
    const f = jit((x: np.Array) => x.add(3));
    const a = f(5); // should be weak
    expect(a.dtype).toBe(np.float32);
    expect(a.weakType).toBe(true);

    const b = a.add(np.array(2, { dtype: np.int32 }));
    expect(b.dtype).toBe(np.int32);
    expect(b.weakType).toBe(false);
    expect(b.js()).toEqual(10);
  });
});

suite("canCast", () => {
  const allDtypes = [
    np.bool,
    np.uint32,
    np.int32,
    np.float16,
    np.float32,
    np.float64,
  ];

  test("safe casting is the default", () => {
    expect(np.canCast(np.int32, np.float64)).toBe(true);
    expect(np.canCast(np.int32, np.float32)).toBe(false);
  });

  test("bool safely casts to everything", () => {
    for (const dtype of allDtypes) {
      expect(np.canCast(np.bool, dtype, "safe")).toBe(true);
    }
  });

  test("safe casting preserves values", () => {
    // int32/uint32 do not fit in float32's 24-bit mantissa.
    expect(np.canCast(np.uint32, np.float64)).toBe(true);
    expect(np.canCast(np.uint32, np.float32)).toBe(false);
    expect(np.canCast(np.uint32, np.float16)).toBe(false);
    expect(np.canCast(np.uint32, np.int32)).toBe(false);
    expect(np.canCast(np.int32, np.uint32)).toBe(false);
    expect(np.canCast(np.float16, np.float32)).toBe(true);
    expect(np.canCast(np.float32, np.float64)).toBe(true);
    expect(np.canCast(np.float64, np.float32)).toBe(false);
    expect(np.canCast(np.float32, np.int32)).toBe(false);
    for (const dtype of allDtypes) {
      expect(np.canCast(dtype, dtype)).toBe(true);
      if (dtype !== np.bool) {
        expect(np.canCast(dtype, np.bool)).toBe(false);
      }
    }
  });

  test("no and equiv require equal dtypes", () => {
    for (const casting of ["no", "equiv"] as const) {
      expect(np.canCast(np.int32, np.int32, casting)).toBe(true);
      expect(np.canCast(np.int32, np.float64, casting)).toBe(false);
      expect(np.canCast(np.bool, np.int32, casting)).toBe(false);
    }
  });

  test("same_kind allows casts within a kind", () => {
    expect(np.canCast(np.float64, np.float16, "same_kind")).toBe(true);
    expect(np.canCast(np.uint32, np.int32, "same_kind")).toBe(true);
    expect(np.canCast(np.int32, np.uint32, "same_kind")).toBe(false);
    expect(np.canCast(np.int32, np.float16, "same_kind")).toBe(true);
    expect(np.canCast(np.float16, np.int32, "same_kind")).toBe(false);
    expect(np.canCast(np.int32, np.bool, "same_kind")).toBe(false);
    expect(np.canCast(np.bool, np.float16, "same_kind")).toBe(true);
  });

  test("unsafe allows any cast", () => {
    for (const from of allDtypes) {
      for (const to of allDtypes) {
        expect(np.canCast(from, to, "unsafe")).toBe(true);
      }
    }
  });

  test("accepts an array as the source", () => {
    const a = np.array([1, 2, 3], { dtype: np.int32 });
    expect(np.canCast(a, np.float64)).toBe(true);
    expect(np.canCast(a, np.float32)).toBe(false);
    a.dispose();
  });
});

suite("resultType", () => {
  test("promotes dtype arguments", () => {
    expect(np.resultType(np.uint32, np.int32)).toBe(np.int32);
    expect(np.resultType(np.int32, np.float16)).toBe(np.float16);
    expect(np.resultType(np.bool, np.int32)).toBe(np.int32);
    expect(np.resultType(np.float16, np.float32)).toBe(np.float32);
    expect(np.resultType(np.float32, np.float64)).toBe(np.float64);
  });

  test("single argument returns its dtype", () => {
    expect(np.resultType(np.float64)).toBe(np.float64);
    expect(np.resultType(3)).toBe(np.float32);
    expect(np.resultType(true)).toBe(np.bool);
  });

  test("weak numbers defer to strong dtypes", () => {
    const a = np.array([1, 2], { dtype: np.int32 });
    expect(np.resultType(a, 3.5)).toBe(np.int32);
    a.dispose();
    expect(np.resultType(2, np.float16)).toBe(np.float16);
    expect(np.resultType(1, 2.5)).toBe(np.float32);
  });

  test("matches promotion behavior of ops", () => {
    // Weak numbers promote bool to at least uint32, same as np.add() does.
    const a = np.array([true, false]);
    const b = a.add(2);
    expect(np.resultType(true, 2)).toBe(b.dtype);
    expect(b.dtype).toBe(np.uint32);
    b.dispose();
  });

  test("ignores shapes and does not consume references", () => {
    const a = np.array([1, 2]);
    const b = np.array([1, 2, 3], { dtype: np.float16 });
    expect(np.resultType(a, b)).toBe(np.float32);
    expect(a.js()).toEqual([1, 2]);
    expect(b.js()).toEqual([1, 2, 3]);
  });

  test("throws on invalid arguments", () => {
    expect(() => np.resultType()).toThrow(TypeError);
    expect(() => np.resultType("float99" as np.DType)).toThrow(/invalid dtype/);
  });
});
