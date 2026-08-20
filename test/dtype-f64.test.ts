// Tests for the f64 data type.

import {
  defaultDevice,
  grad,
  init,
  jit,
  jvp,
  nn,
  numpy as np,
} from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

// f64 is currently only supported on WebAssembly.
const devices = ["cpu", "wasm"] as const;

const devicesAvailable = await init(...devices);

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  test("create and access f64 array", async () => {
    const a = np.array([1.5, 2.5, 3.5], { dtype: np.float64 });
    expect(a.dtype).toBe(np.float64);
    expect(a.shape).toEqual([3]);
    expect(await a.ref.data()).toEqual(new Float64Array([1.5, 2.5, 3.5]));
    expect(a.ref.dataSync()).toEqual(new Float64Array([1.5, 2.5, 3.5]));
    expect(a.js()).toEqual([1.5, 2.5, 3.5]);
  });

  test("signbit() distinguishes f64 signed zero", () => {
    const x = np.array([-1.5, -0, 0, 1.5], { dtype: np.float64 });
    expect(np.signbit(x).js()).toEqual([true, true, false, false]);
  });

  test("signbit() preserves the sign of f64 NaN", () => {
    // JS may canonicalize f64 NaNs on the CPU backend.
    if (device === "cpu") return;
    const values = new Float64Array(2);
    // Write the NaNs as raw bits, since float writes may canonicalize NaN.
    const bits = new BigUint64Array(values.buffer);
    bits[0] = 0xfff8000000000000n; // -NaN
    bits[1] = 0x7ff8000000000000n; // NaN
    const x = np.array(values);
    expect(np.signbit(x).js()).toEqual([true, false]);
  });

  test("jit of f64 calculation", () => {
    const f = jit((x: np.Array) => np.sum(x.ref.mul(x)));
    expect(f(np.arange(10).astype(np.float64))).toBeAllclose(285);
  });

  test("jvp of f64 calculation", () => {
    const f = (x: np.Array) => x.ref.mul(x);
    const [y, dy] = jvp(
      f,
      [np.array([1.5, 2.5], { dtype: np.float64 })],
      [np.array([1.0, 1.0], { dtype: np.float64 })],
    );
    expect(y.dtype).toBe(np.float64);
    expect(dy.dtype).toBe(np.float64);
    expect(y.ref.dataSync()).toEqual(new Float64Array([2.25, 6.25]));
    expect(dy.ref.dataSync()).toEqual(new Float64Array([3.0, 5.0]));
  });

  test("gradient of f64 calculation", () => {
    const f = (x: np.Array) => np.sum(x.ref.mul(x));
    const g = grad(f);

    const x = np.array([1.5, 2.5], { dtype: np.float64 });
    const y = g(x);
    expect(y.dtype).toBe(np.float64);
    expect(y.ref.dataSync()).toEqual(new Float64Array([3.0, 5.0]));
  });

  test("erfc() works for f64", () => {
    // nn.gelu() with approximate=false uses erfc().
    const x = np.array([-1.0, 0.0, 1.0], { dtype: np.float64 });
    const y = nn.gelu(x, { approximate: false });
    expect(y.dtype).toBe(np.float64);
    expect(y).toBeAllclose([-0.15865525, 0.0, 0.84134475]);
  });

  test("precision of f64 is high", async () => {
    const a = np.array([1 + 1e-15, 1 + 2e-15], { dtype: np.float64 });
    await a.blockUntilReady();
    const b: number = await a.ref.slice(1).sub(a.slice(0)).jsAsync();
    expect(b).toBeCloseTo(1e-15, 15);
  });

  test("unwrap() matches numpy at a rounding-tie boundary", () => {
    // The delta is just below -pi, which makes it wrap to +pi.
    const x = np.array([0, -3.1415926535897936], { dtype: np.float64 });
    const y = np.unwrap(x);
    expect(y.dtype).toBe(np.float64);
    expect(y.ref.dataSync()[1]).toBe(3.1415926535897927);
  });

  test("polyint() preserves f64 dtype", () => {
    const p = np.array([3, 2, 1], { dtype: np.float64 });
    const y = np.polyint(p);
    expect(y.dtype).toBe(np.float64);
    expect(y.js()).toEqual([1, 1, 1, 0]);
  });

  test("polymul() preserves f64 precision", () => {
    const a = np.array([1, 1e-12], { dtype: np.float64 });
    const b = np.array([1, 1], { dtype: np.float64 });
    const y = np.polymul(a, b);
    expect(y.dtype).toBe(np.float64);
    // The middle coefficient 1 + 1e-12 would round to 1 in f32.
    expect(y.ref.dataSync()[1]).toBe(1 + 1e-12);
  });
});
