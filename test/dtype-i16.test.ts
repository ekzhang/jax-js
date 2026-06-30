// Tests for 16-bit integer data types.

import { defaultDevice, init, jit, numpy as np } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

// 16-bit integer storage is currently supported on CPU and scalar Wasm.
const devices = ["cpu", "wasm"] as const;

const devicesAvailable = await init(...devices);

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  test("create and access int16 and uint16 arrays", async () => {
    const i16 = np.array([-32769, -1, 0, 32767, 32768], {
      dtype: np.int16,
    });
    expect(i16.dtype).toBe(np.int16);
    expect(i16.shape).toEqual([5]);
    expect(await i16.ref.data()).toEqual(
      new Int16Array([32767, -1, 0, 32767, -32768]),
    );
    expect(i16.ref.dataSync()).toEqual(
      new Int16Array([32767, -1, 0, 32767, -32768]),
    );
    expect(i16.js()).toEqual([32767, -1, 0, 32767, -32768]);

    const u16 = np.array([-1, 0, 1, 65535, 65536], { dtype: np.uint16 });
    expect(u16.dtype).toBe(np.uint16);
    expect(u16.dataSync()).toEqual(new Uint16Array([65535, 0, 1, 65535, 0]));
  });

  test("accept typed arrays as int16 and uint16 inputs", () => {
    const i16 = np.array(new Int16Array([-2, -1, 0, 1]));
    expect(i16.dtype).toBe(np.int16);
    expect(i16.dataSync()).toEqual(new Int16Array([-2, -1, 0, 1]));

    const u16 = np.array(new Uint16Array([0, 1, 65535]));
    expect(u16.dtype).toBe(np.uint16);
    expect(u16.dataSync()).toEqual(new Uint16Array([0, 1, 65535]));
  });

  test("arithmetic uses 16-bit storage and 32-bit compute", () => {
    const values = Array.from({ length: 130 }, (_, i) => (i === 0 ? 32767 : i));
    const i16 = np.array(values, { dtype: np.int16 });
    const y = i16.add(np.array(1, { dtype: np.int16 }));
    expect(y.dtype).toBe(np.int16);
    const data = y.dataSync() as Int16Array;
    expect(data[0]).toBe(-32768);
    expect(data[1]).toBe(2);

    const uvalues = Array.from({ length: 130 }, (_, i) =>
      i === 0 ? 65535 : i,
    );
    const u16 = np.array(uvalues, { dtype: np.uint16 });
    const z = u16.add(np.array(1, { dtype: np.uint16 }));
    expect(z.dtype).toBe(np.uint16);
    const udata = z.dataSync() as Uint16Array;
    expect(udata[0]).toBe(0);
    expect(udata[1]).toBe(2);
  });

  test("jit reduction epilogue observes 16-bit reduction output", () => {
    const mod100 = jit((x: np.Array) => x.sum().mod(100));

    const i16 = mod100(np.ones([70_000], { dtype: np.int16 }));
    expect(i16.dtype).toBe(np.int16);
    expect(i16.js()).toBe(64);

    const u16 = mod100(np.ones([70_000], { dtype: np.uint16 }));
    expect(u16.dtype).toBe(np.uint16);
    expect(u16.js()).toBe(64);

    const prodMod100 = jit((x: np.Array) => x.prod().mod(100));

    const i16Prod = prodMod100(np.array([256, 256], { dtype: np.int16 }));
    expect(i16Prod.dtype).toBe(np.int16);
    expect(i16Prod.js()).toBe(0);

    const u16Prod = prodMod100(np.array([256, 256], { dtype: np.uint16 }));
    expect(u16Prod.dtype).toBe(np.uint16);
    expect(u16Prod.js()).toBe(0);
  });

  test("casts to int16 and uint16 clamp floats and wrap integers", () => {
    const f = np.array([-32769, -32768, 1.9, 32767, 32768], {
      dtype: np.float32,
    });
    const i16 = f.astype(np.int16);
    expect(i16.dataSync()).toEqual(
      new Int16Array([-32768, -32768, 1, 32767, 32767]),
    );

    const g = np.array([-1, 0, 1.9, 65535, 65536], { dtype: np.float32 });
    const u16 = g.astype(np.uint16);
    expect(u16.dataSync()).toEqual(new Uint16Array([0, 0, 1, 65535, 65535]));

    const ints = np.array([-1, 65536], { dtype: np.int32 });
    expect(ints.ref.astype(np.uint16).dataSync()).toEqual(
      new Uint16Array([65535, 0]),
    );
    expect(ints.astype(np.int16).dataSync()).toEqual(new Int16Array([-1, 0]));
  });

  test("promotion follows the integer type lattice", () => {
    expect(np.promoteTypes(np.int16, np.int16)).toBe(np.int16);
    expect(np.promoteTypes(np.uint16, np.uint16)).toBe(np.uint16);
    expect(np.promoteTypes(np.uint16, np.uint32)).toBe(np.uint32);
    expect(np.promoteTypes(np.uint32, np.int32)).toBe(np.int32);
    expect(np.promoteTypes(np.int16, np.uint16)).toBe(np.int32);

    const i16 = np.array([1, 2, 3], { dtype: np.int16 });
    const weak = i16.add(2);
    expect(weak.dtype).toBe(np.int16);
    expect(weak.dataSync()).toEqual(new Int16Array([3, 4, 5]));

    const u16 = np.array([1, 2, 3], { dtype: np.uint16 });
    const uweak = u16.add(2);
    expect(uweak.dtype).toBe(np.uint16);
    expect(uweak.dataSync()).toEqual(new Uint16Array([3, 4, 5]));

    const promoted = np
      .array([1, 2, 3], { dtype: np.int16 })
      .add(np.array([4, 5, 6], { dtype: np.int32 }));
    expect(promoted.dtype).toBe(np.int32);
    expect(promoted.dataSync()).toEqual(new Int32Array([5, 7, 9]));

    const signedUnsigned = np
      .array([1, 2, 3], { dtype: np.int16 })
      .add(np.array([4, 5, 6], { dtype: np.uint16 }));
    expect(signedUnsigned.dtype).toBe(np.int32);
    expect(signedUnsigned.dataSync()).toEqual(new Int32Array([5, 7, 9]));
  });

  test("jit works with int16 and uint16 arrays", () => {
    const f = jit((x: np.Array) => x.add(2).mul(3));

    const i16 = f(np.arange(0, 130, 1, { dtype: np.int16 }));
    expect(i16.dtype).toBe(np.int16);
    expect(Array.from(i16.dataSync() as Int16Array).slice(0, 5)).toEqual([
      6, 9, 12, 15, 18,
    ]);

    const u16 = f(np.arange(0, 130, 1, { dtype: np.uint16 }));
    expect(u16.dtype).toBe(np.uint16);
    expect(Array.from(u16.dataSync() as Uint16Array).slice(0, 5)).toEqual([
      6, 9, 12, 15, 18,
    ]);
  });

  test("jit normalizes uint16 arithmetic before compare and remainder", () => {
    const notEqualTwo = jit((x: np.Array) => x.add(3).notEqual(2));
    const remainder = jit((x: np.Array) => x.add(3).mod(3));

    const compare = notEqualTwo(np.array([65535], { dtype: np.uint16 }));
    expect(compare.dtype).toBe(np.bool);
    expect(compare.js()).toEqual([false]);

    const rem = remainder(np.array([65535], { dtype: np.uint16 }));
    expect(rem.dtype).toBe(np.uint16);
    expect(rem.dataSync()).toEqual(new Uint16Array([2]));
  });
});
