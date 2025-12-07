import { beforeEach, expect, suite, test } from "vitest";

import { defaultDevice, devices, init } from "../backend";
import { arange, array, eye, ones, zeros } from "./array";
import { DType } from "../alu";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);

  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  test("can construct zeros()", async () => {
    const ar = zeros([3, 3]);
    expect(ar.shape).toEqual([3, 3]);
    expect(ar.dtype).toEqual("float32");
    expect(await ar.ref.data()).toEqual(
      new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    );
    expect(await ar.ref.transpose().data()).toEqual(
      new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    );
    expect(ar.transpose().dataSync()).toEqual(
      new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
    );
  });

  test("can construct ones()", async () => {
    const ar = ones([2, 2]);
    expect(ar.shape).toEqual([2, 2]);
    expect(ar.dtype).toEqual("float32");
    expect(await ar.data()).toEqual(new Float32Array([1, 1, 1, 1]));
  });

  test("can add two arrays", async () => {
    const ar1 = ones([2, 2]);
    const ar2 = ones([2, 2]);
    const ar3 = ar1.add(ar2);
    expect(ar3.shape).toEqual([2, 2]);
    expect(ar3.dtype).toEqual("float32");
    expect(await ar3.data()).toEqual(new Float32Array([2, 2, 2, 2]));
  });

  test("can construct arrays from data", () => {
    const a = array([1, 2, 3, 4]);
    const b = array([10, 5, 2, -8.5]);
    const c = a.mul(b);
    expect(c.shape).toEqual([4]);
    expect(c.dtype).toEqual("float32");
    expect(c.ref.dataSync()).toEqual(new Float32Array([10, 10, 6, -34]));
    expect(c.reshape([2, 2]).transpose().dataSync()).toEqual(
      new Float32Array([10, 6, 10, -34]),
    );
  });

  test("common broadcasting", () => {
    // Start with arrays of shape [2, 2] and [2, 3].
    // Reshape first one to [2, 1, 2] and second one to [2, 3, 1].
    const a = array([
      [1, 22],
      [3, 9],
    ]).reshape([2, 1, 2]);
    const b = array([
      [10, 5, -2],
      [-8, 0, 3],
    ]).reshape([2, 3, 1]);

    // Multiply them together -- outer products of a[i] and b[i].
    const c = a.mul(b);
    expect(c.shape).toEqual([2, 3, 2]);
    expect(c.ref.js()).toEqual([
      [
        [10, 220],
        [5, 110],
        [-2, -44],
      ],
      [
        [-24, -72],
        [0, 0],
        [9, 27],
      ],
    ]);
  });

  test("flatten and ravel", () => {
    const a = array([
      [
        [1, 2],
        [3, 4],
      ],
    ]); // 3D
    expect(a.shape).toEqual([1, 2, 2]);
    expect(a.ref.flatten().js()).toEqual([1, 2, 3, 4]);
    expect(a.ravel().js()).toEqual([1, 2, 3, 4]);
    expect(array(3).flatten().js()).toEqual([3]);
  });

  test("can add array to itself", () => {
    const a = array([1, 2, 3]);
    // Make sure duplicate references don't trip up the backend.
    const b = a.ref.add(a.ref).add(a);
    expect(b.dataSync()).toEqual(new Float32Array([3, 6, 9]));
  });

  test("can coerce array to primitive", () => {
    const a = array(42);
    expect(a.ref).toBeCloseTo(42);

    // https://github.com/microsoft/TypeScript/issues/42218
    expect(+(a.ref as any)).toEqual(42);
    expect((a.ref as any) + 1).toEqual(43);
    expect((a as any) ** 2).toEqual(42 ** 2);
  });

  test("construct bool array", () => {
    const a = array([true, false, true]);
    expect(a.shape).toEqual([3]);
    expect(a.dtype).toEqual("bool");

    expect(a.ref.dataSync()).toEqual(new Int32Array([1, 0, 1]));
    expect(a.ref.js()).toEqual([true, false, true]);

    const b = array([1, 3, 4]);
    expect(b.ref.greater(2).js()).toEqual([false, true, true]);
    expect(b.ref.greater(2).dataSync()).toEqual(new Int32Array([0, 1, 1]));

    expect(b.ref.equal(3).js()).toEqual([false, true, false]);
    expect(b.notEqual(array([2, 3, 4])).js()).toEqual([true, false, false]);
  });

  test("comparison operators async", async () => {
    const x = array([1, 2, 3]);
    expect(await x.ref.greater(2).jsAsync()).toEqual([false, false, true]);
    expect(await x.ref.greaterEqual(2).jsAsync()).toEqual([false, true, true]);
    expect(await x.ref.less(2).jsAsync()).toEqual([true, false, false]);
    expect(await x.ref.lessEqual(2).jsAsync()).toEqual([true, true, false]);
    expect(await x.ref.equal(2).jsAsync()).toEqual([false, true, false]);
    expect(await x.ref.notEqual(2).jsAsync()).toEqual([true, false, true]);
    x.dispose();

    let ar = arange(0, 5000, 1, { dtype: DType.Float32 });
    await ar.ref.data(); // Ensure data is loaded
    ar = ar.add(1);
    const vals = (await ar.less(2500).data()) as Int32Array;
    for (let i = 0; i < vals.length; i++) {
      expect(vals[i]).toEqual(i + 1 < 2500 ? 1 : 0);
    }
  });

  test("slicing arrays", () => {
    const x = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);

    // Basic slicing and element access.
    expect(x.ref.slice(0, 0).js()).toEqual(1);
    expect(x.ref.slice(0, 2).js()).toEqual(3);
    expect(x.ref.slice(1, 2).js()).toEqual(6);
    expect(x.ref.slice(1).js()).toEqual([4, 5, 6]);
    expect(x.ref.slice().js()).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);

    // Try slicing with negative indices.
    expect(x.ref.slice(-1, -1).js()).toEqual(6);
    expect(x.ref.slice(-2, -1).js()).toEqual(3);
    expect(x.ref.slice(-1, -3).js()).toEqual(4);

    // Try adding new axes.
    expect(x.ref.slice(0, 0, null).js()).toEqual([1]);
    expect(x.ref.slice(0, null, 0).js()).toEqual([1]);
    expect(x.ref.slice(null).js()).toEqual([x.ref.js()]);

    x.dispose();
  });

  test("sum along negative axis", () => {
    const x = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(x.ref.sum(-1).js()).toEqual([6, 15]);
    expect(x.sum(-2).js()).toEqual([5, 7, 9]);
  });

  test("mean along multiple axes", () => {
    const x = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(x.ref.mean().js()).toEqual(3.5);
    expect(x.ref.mean([0, 1]).js()).toEqual(3.5);
    expect(x.ref.mean(0).js()).toEqual([2.5, 3.5, 4.5]);
    expect(x.ref.mean(1).js()).toEqual([2, 5]);
    x.dispose();
  });

  test("advanced indexing with gather", () => {
    const x = array([1, 3, 2], { dtype: DType.Int32 });
    // np.eye(5)[[1, 3, 2]]
    expect(eye(5).slice(x.ref).js()).toEqual([
      [0, 1, 0, 0, 0],
      [0, 0, 0, 1, 0],
      [0, 0, 1, 0, 0],
    ]);
    // np.eye(5)[[1, 3, 2], np.newaxis]
    expect(eye(5).slice(x.ref, null).js()).toEqual([
      [[0, 1, 0, 0, 0]],
      [[0, 0, 0, 1, 0]],
      [[0, 0, 1, 0, 0]],
    ]);
    // np.eye(5)[1:4, [1, 3, 2]]
    expect(eye(5).slice([1, 4], x).js()).toEqual([
      [1, 0, 0],
      [0, 0, 1],
      [0, 1, 0],
    ]);
  });

  // This checks to make sure that index calculations don't suddenly break for
  // large arrays, covering workgroups > 65535 in WebGPU for instance.
  if (device !== "cpu") {
    test("large array dispatch", async () => {
      const x = ones([100, 1000, 1000], { dtype: DType.Int32 }); // 100M elements
      await x.blockUntilReady();
      expect(await x.sum().jsAsync()).toEqual(100_000_000);
    });
  }

  test("iterate over an array", () => {
    const [a, b] = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(a.js()).toEqual([1, 2, 3]);
    expect(b.js()).toEqual([4, 5, 6]);

    const [[x, y], z] = array([1, 2, 3, 4]).reshape([2, 2]);
    expect(x.js()).toEqual(1);
    expect(y.js()).toEqual(2);
    expect(z.js()).toEqual([3, 4]);
  });

  test("u32 data type", () => {
    const a = array([1, 2, 3], { dtype: DType.Uint32 });
    expect(a.dtype).toBe(DType.Uint32);
    expect(a.ref.dataSync()).toEqual(new Uint32Array([1, 2, 3]));
    expect(a.ref.js()).toEqual([1, 2, 3]);

    const b = a.sub(array(2, { dtype: DType.Uint32 }));
    expect(b.dtype).toBe(DType.Uint32);
    expect(b.ref.dataSync()).toEqual(new Uint32Array([4294967295, 0, 1]));
    expect(b.js()).toEqual([4294967295, 0, 1]);
  });

  test("casting arrays", () => {
    const a = array([1, 2, 3], { dtype: DType.Int32 });
    expect(a.dtype).toBe(DType.Int32);
    expect(a.ref.dataSync()).toEqual(new Int32Array([1, 2, 3]));
    expect(a.ref.js()).toEqual([1, 2, 3]);

    const b = a.astype(DType.Float32);
    expect(b.dtype).toBe(DType.Float32);
    expect(b.ref.dataSync()).toEqual(new Float32Array([1, 2, 3]));
    expect(b.js()).toEqual([1, 2, 3]);
  });

  test("cast saturates from large f32 -> i32", () => {
    const a = array([1e20, -1e20, 1e10, -1e10, 1e5, -1e5], {
      dtype: DType.Float32,
    });
    const b = a.astype(DType.Int32);
    expect(b.js()).toEqual([
      2147483647, -2147483648, 2147483647, -2147483648, 100000, -100000,
    ]);
  });
});
