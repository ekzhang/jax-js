// Tests for the f16 data type.

import { grad, init, jit, jvp, numpy as np, setDevice } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

// f16 is currently only supported on WebGPU.
const devices = ["cpu", "webgpu"] as const;

const devicesAvailable = await init(...devices);

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    setDevice(device);
  });

  test("create and access f16 array", async () => {
    const a = np.array([1.5, 2.5, 3.5], { dtype: np.float16 });
    expect(a.dtype).toBe(np.float16);
    expect(a.shape).toEqual([3]);
    expect(await a.ref.data()).toEqual(new Float16Array([1.5, 2.5, 3.5]));
    expect(a.ref.dataSync()).toEqual(new Float16Array([1.5, 2.5, 3.5]));
    expect(a.js()).toEqual([1.5, 2.5, 3.5]);
  });

  test("jit of f16 calculation", () => {
    const f = jit((x: np.Array) => np.sum(x.ref.mul(x)));
    expect(f(np.arange(10).astype(np.float16))).toBeAllclose(285);
  });

  test("jvp of f16 calculation", () => {
    const f = (x: np.Array) => x.ref.mul(x);
    const [y, dy] = jvp(
      f,
      [np.array([1.5, 2.5], { dtype: np.float16 })],
      [np.array([1.0, 1.0], { dtype: np.float16 })],
    );
    expect(y.dtype).toBe(np.float16);
    expect(dy.dtype).toBe(np.float16);
    expect(y.ref.dataSync()).toEqual(new Float16Array([2.25, 6.25]));
    expect(dy.ref.dataSync()).toEqual(new Float16Array([3.0, 5.0]));
  });

  test("gradient of f16 calculation", () => {
    const f = (x: np.Array) => np.sum(x.ref.mul(x));
    const g = grad(f);

    const x = np.array([1.5, 2.5], { dtype: np.float16 });
    const y = g(x);
    expect(y.dtype).toBe(np.float16);
    expect(y.ref.dataSync()).toEqual(new Float16Array([3.0, 5.0]));
  });
});
