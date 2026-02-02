// Tests for the f64 data type.

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

  test("cholesky works for f64", () => {
    // Positive definite matrix
    const A = np.array(
      [
        [4, 2],
        [2, 5],
      ],
      { dtype: np.float64 },
    );
    const L = lax.linalg.cholesky(A.ref);
    expect(L.dtype).toBe(np.float64);

    // Verify L @ L.T ≈ A
    const reconstructed = np.matmul(L.ref, np.transpose(L));
    expect(reconstructed).toBeAllclose(A, { rtol: 1e-10, atol: 1e-12 });
  });

  test("lu works for f64", () => {
    const A = np.array(
      [
        [4, 3],
        [6, 3],
      ],
      { dtype: np.float64 },
    );

    const [lu, pivots, permutation] = lax.linalg.lu(A);
    expect(lu.dtype).toBe(np.float64);
    expect(pivots.dtype).toBe(np.int32);
    expect(permutation.dtype).toBe(np.int32);

    // Basic validation: output shapes are correct
    expect(lu.shape).toEqual([2, 2]);
    expect(pivots.shape).toEqual([2]);
    expect(permutation.shape).toEqual([2]);

    // LU values should be finite numbers
    const luData = lu.dataSync();
    expect(luData.every((v) => isFinite(v))).toBe(true);

    pivots.dispose();
    permutation.dispose();
  });

  test("triangular_solve works for f64", () => {
    const L = np.array(
      [
        [2, 0],
        [1, 3],
      ],
      { dtype: np.float64 },
    );
    const b = np.array([[4], [7]], { dtype: np.float64 });

    const x = lax.linalg.triangularSolve(L.ref, b.ref, {
      leftSide: true,
      lower: true,
    });
    expect(x.dtype).toBe(np.float64);

    // Verify L @ x ≈ b
    const reconstructed = np.matmul(L, x);
    expect(reconstructed).toBeAllclose(b, { rtol: 1e-10, atol: 1e-12 });
  });
});
