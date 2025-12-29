import { defaultDevice, Device, init, lax, numpy as np } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

suite.each(["cpu", "wasm"] as Device[])("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  suite("jax.lax.linalg.cholesky()", () => {
    test("computes lower Cholesky decomposition for 2x2 matrix", () => {
      const x = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
      ]);
      const L = lax.linalg.cholesky(x.ref);

      // L should be lower triangular
      const LData = L.ref.js();
      expect(LData[0][1]).toBeCloseTo(0);
      expect(LData[1][0]).not.toBe(0);

      // Verify: L @ L^T should equal x
      const reconstructed = np.matmul(L.ref, L.transpose());
      expect(reconstructed).toBeAllclose(x);
    });

    test("computes Cholesky decomposition for 3x3 matrix", () => {
      const x = np.array([
        [4.0, 2.0, 1.0],
        [2.0, 5.0, 3.0],
        [1.0, 3.0, 6.0],
      ]);
      const L = lax.linalg.cholesky(x.ref);

      // Verify: L @ L^T should equal x
      const reconstructed = np.matmul(L.ref, L.transpose());
      expect(reconstructed).toBeAllclose(x);
    });

    test("throws on non-square matrix", () => {
      const x = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      expect(() => lax.linalg.cholesky(x).js()).toThrow();
    });

    test("throws on non-2D array", () => {
      const x = np.array([1.0, 2.0, 3.0]);
      expect(() => lax.linalg.cholesky(x).js()).toThrow();
    });
  });

  suite("jax.lax.linalg.triangularSolve()", () => {
    test("solves lower-triangular system", () => {
      // Solve L @ x = b
      const L = np.array([
        [2, 0],
        [1, 3],
      ]);
      const b = np.array([4, 7]).reshape([2, 1]);
      const x = lax.linalg.triangularSolve(L, b, {
        leftSide: true,
        lower: true,
      });
      expect(x.flatten()).toBeAllclose([2, 5 / 3]);
    });
  });
});
