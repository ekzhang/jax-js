// Tests for the f16 data type.

import {
  defaultDevice,
  grad,
  init,
  jit,
  jvp,
  lax,
  nn,
  numpy as np,
} from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

// f16 is currently only supported on WebGPU.
const devices = ["cpu", "webgpu"] as const;

const devicesAvailable = await init(...devices);

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
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

  test("erfc() works for f16", () => {
    // nn.gelu() with approximate=false uses erfc().
    const x = np.array([-1.0, 0.0, 1.0], { dtype: np.float16 });
    const y = nn.gelu(x, { approximate: false });
    expect(y.dtype).toBe(np.float16);
    expect(y).toBeAllclose([-0.1587, 0.0, 0.8413], { rtol: 1e-3 });
  });

  test("f16 reductions are performed in f32", () => {
    // dot() / matmul() should be performed in f32 by default
    let x = np.array([100, 1, 1, 1, 1, 1, 1, 1, 1], { dtype: np.float16 });
    let y = np.dot(x.ref, x);
    expect(y.dtype).toBe(np.float16);
    expect(y.dataSync()).toEqual(new Float16Array([10008]));

    // sum() should also be performed in f32
    x = np.array([16000, 1, 1, 1, 1, 1, 1, 1, 1], { dtype: np.float16 });
    y = np.sum(x);
    expect(y.dtype).toBe(np.float16);
    expect(y.dataSync()).toEqual(new Float16Array([16008]));
  });

  test("jit ConvTranspose lowering supports f16", () => {
    const convTransposeFn = jit((x: np.Array, kernel: np.Array) =>
      lax.convGeneralDilated(x, kernel, [1], [[2, 1]], {
        lhsDilation: [2],
      }),
    );
    const x = np.array(
      [
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
      ],
      { dtype: np.float16 },
    );
    const kernel = np.array(
      [
        [
          [1, 0, -1],
          [2, 1, 0],
        ],
        [
          [0, 1, 2],
          [-1, 0, 1],
        ],
      ],
      { dtype: np.float16 },
    );

    const expected = lax.convGeneralDilated(x.ref, kernel.ref, [1], [[2, 1]], {
      lhsDilation: [2],
    });
    const actual = convTransposeFn(x, kernel);
    expect(actual.dtype).toBe(np.float16);
    expect(actual).toBeAllclose(expected, { rtol: 1e-3, atol: 1e-3 });
  });
});
