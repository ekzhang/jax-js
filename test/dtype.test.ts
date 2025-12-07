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
