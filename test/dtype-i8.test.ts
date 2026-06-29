// Tests for 8-bit integer data types.

import { defaultDevice, init, jit, numpy as np } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

// 8-bit integer storage is currently supported on CPU and scalar Wasm.
const devices = ["cpu", "wasm"] as const;

const devicesAvailable = await init(...devices);

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  test("create and access int8 and uint8 arrays", async () => {
    const i8 = np.array([-129, -1, 0, 127, 128], {
      dtype: np.int8,
    });
    expect(i8.dtype).toBe(np.int8);
    expect(i8.shape).toEqual([5]);
    expect(await i8.ref.data()).toEqual(
      new Int8Array([127, -1, 0, 127, -128]),
    );
    expect(i8.ref.dataSync()).toEqual(
      new Int8Array([127, -1, 0, 127, -128]),
    );
    expect(i8.js()).toEqual([127, -1, 0, 127, -128]);

    const u8 = np.array([-1, 0, 1, 255, 256], { dtype: np.uint8 });
    expect(u8.dtype).toBe(np.uint8);
    expect(u8.dataSync()).toEqual(new Uint8Array([255, 0, 1, 255, 0]));
  });

  test("accept typed arrays as int8 and uint8 inputs", () => {
    const i8 = np.array(new Int8Array([-2, -1, 0, 1]));
    expect(i8.dtype).toBe(np.int8);
    expect(i8.dataSync()).toEqual(new Int8Array([-2, -1, 0, 1]));

    const u8 = np.array(new Uint8Array([0, 1, 255]));
    expect(u8.dtype).toBe(np.uint8);
    expect(u8.dataSync()).toEqual(new Uint8Array([0, 1, 255]));
  });

  test("arithmetic uses 8-bit storage and 32-bit compute", () => {
    const values = Array.from({ length: 130 }, (_, i) => (i === 0 ? 127 : i));
    const i8 = np.array(values, { dtype: np.int8 });
    const y = i8.add(np.array(1, { dtype: np.int8 }));
    expect(y.dtype).toBe(np.int8);
    const data = y.dataSync() as Int8Array;
    expect(data[0]).toBe(-128);
    expect(data[1]).toBe(2);

    const uvalues = Array.from({ length: 130 }, (_, i) => (i === 0 ? 255 : i));
    const u8 = np.array(uvalues, { dtype: np.uint8 });
    const z = u8.add(np.array(1, { dtype: np.uint8 }));
    expect(z.dtype).toBe(np.uint8);
    const udata = z.dataSync() as Uint8Array;
    expect(udata[0]).toBe(0);
    expect(udata[1]).toBe(2);
  });

  test("casts to int8 and uint8 clamp floats and wrap integers", () => {
    const f = np.array([-129, -128, 1.9, 127, 128], {
      dtype: np.float32,
    });
    const i8 = f.astype(np.int8);
    expect(i8.dataSync()).toEqual(new Int8Array([-128, -128, 1, 127, 127]));

    const g = np.array([-1, 0, 1.9, 255, 256], { dtype: np.float32 });
    const u8 = g.astype(np.uint8);
    expect(u8.dataSync()).toEqual(new Uint8Array([0, 0, 1, 255, 255]));

    const ints = np.array([-1, 256], { dtype: np.int32 });
    expect(ints.ref.astype(np.uint8).dataSync()).toEqual(
      new Uint8Array([255, 0]),
    );
    expect(ints.astype(np.int8).dataSync()).toEqual(new Int8Array([-1, 0]));
  });

  test("promotion follows the integer type lattice", () => {
    expect(np.promoteTypes(np.int8, np.int8)).toBe(np.int8);
    expect(np.promoteTypes(np.uint8, np.uint8)).toBe(np.uint8);
    expect(np.promoteTypes(np.uint8, np.uint16)).toBe(np.uint16);
    expect(np.promoteTypes(np.uint16, np.uint32)).toBe(np.uint32);
    expect(np.promoteTypes(np.int8, np.uint8)).toBe(np.int16);
    expect(np.promoteTypes(np.uint8, np.int16)).toBe(np.int16);
    expect(np.promoteTypes(np.uint16, np.int8)).toBe(np.int32);
    expect(np.promoteTypes(np.uint32, np.int32)).toBe(np.int32);

    const i8 = np.array([1, 2, 3], { dtype: np.int8 });
    const weak = i8.add(2);
    expect(weak.dtype).toBe(np.int8);
    expect(weak.dataSync()).toEqual(new Int8Array([3, 4, 5]));

    const u8 = np.array([1, 2, 3], { dtype: np.uint8 });
    const uweak = u8.add(2);
    expect(uweak.dtype).toBe(np.uint8);
    expect(uweak.dataSync()).toEqual(new Uint8Array([3, 4, 5]));

    const promoted = np
      .array([1, 2, 3], { dtype: np.int8 })
      .add(np.array([4, 5, 6], { dtype: np.int16 }));
    expect(promoted.dtype).toBe(np.int16);
    expect(promoted.dataSync()).toEqual(new Int16Array([5, 7, 9]));

    const signedUnsigned = np
      .array([1, 2, 3], { dtype: np.int8 })
      .add(np.array([4, 5, 6], { dtype: np.uint8 }));
    expect(signedUnsigned.dtype).toBe(np.int16);
    expect(signedUnsigned.dataSync()).toEqual(new Int16Array([5, 7, 9]));
  });

  test("jit works with int8 and uint8 arrays", () => {
    const f = jit((x: np.Array) => x.add(2).mul(3));

    const i8 = f(np.arange(0, 40, 1, { dtype: np.int8 }));
    expect(i8.dtype).toBe(np.int8);
    expect(Array.from(i8.dataSync() as Int8Array).slice(0, 5)).toEqual([
      6, 9, 12, 15, 18,
    ]);

    const u8 = f(np.arange(0, 40, 1, { dtype: np.uint8 }));
    expect(u8.dtype).toBe(np.uint8);
    expect(Array.from(u8.dataSync() as Uint8Array).slice(0, 5)).toEqual([
      6, 9, 12, 15, 18,
    ]);
  });

  test("jit normalizes uint8 arithmetic before compare and remainder", () => {
    const notEqualTwo = jit((x: np.Array) => x.add(3).notEqual(2));
    const remainder = jit((x: np.Array) => x.add(3).mod(3));

    const compare = notEqualTwo(np.array([255], { dtype: np.uint8 }));
    expect(compare.dtype).toBe(np.bool);
    expect(compare.js()).toEqual([false]);

    const rem = remainder(np.array([255], { dtype: np.uint8 }));
    expect(rem.dtype).toBe(np.uint8);
    expect(rem.dataSync()).toEqual(new Uint8Array([2]));
  });

  test("iinfo, invert, and views support int8 and uint8", () => {
    expect(np.iinfo(np.int8)).toMatchObject({ min: -128, max: 127, bits: 8 });
    expect(np.iinfo(np.uint8)).toMatchObject({ min: 0, max: 255, bits: 8 });

    expect(np.invert(np.array([0, 1], { dtype: np.uint8 })).dataSync()).toEqual(
      new Uint8Array([255, 254]),
    );
    expect(np.invert(np.array([0, 1], { dtype: np.int8 })).dataSync()).toEqual(
      new Int8Array([-1, -2]),
    );

    const viewed = np.array([-1, 127], { dtype: np.int8 }).view(np.uint8);
    expect(viewed.dataSync()).toEqual(new Uint8Array([255, 127]));
  });
});
