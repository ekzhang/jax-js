import { linalg, numpy as np } from "@jax-js/jax";
import { expect, suite, test } from "vitest";

suite("linalg.cholesky()", () => {
  test("computes lower Cholesky decomposition for 2x2 matrix", () => {
    const x = np.array([
      [2.0, 1.0],
      [1.0, 2.0],
    ]);
    const L = linalg.cholesky(x, { lower: true });

    // L should be lower triangular
    expect(L.js()[0][1]).toBeCloseTo(0);
    expect(L.js()[1][0]).not.toBe(0);

    // Verify: L @ L^T should equal x
    const reconstructed = np.matmul(L, L.ref.transpose());
    expect(reconstructed.js()).toBeAllclose(x.js());
  });

  test("computes upper Cholesky decomposition for 2x2 matrix", () => {
    const x = np.array([
      [2.0, 1.0],
      [1.0, 2.0],
    ]);
    const U = linalg.cholesky(x); // lower=false is default

    // U should be upper triangular
    expect(U.js()[0][1]).not.toBe(0);
    expect(U.js()[1][0]).toBeCloseTo(0);

    // Verify: U^T @ U should equal x
    const reconstructed = np.matmul(U.ref.transpose(), U);
    expect(reconstructed.js()).toBeAllclose(x.js());
  });

  test("computes Cholesky decomposition for 3x3 matrix", () => {
    const x = np.array([
      [4.0, 2.0, 1.0],
      [2.0, 5.0, 3.0],
      [1.0, 3.0, 6.0],
    ]);
    const L = linalg.cholesky(x, { lower: true });

    // Verify: L @ L^T should equal x
    const reconstructed = np.matmul(L, L.ref.transpose());
    expect(reconstructed.js()).toBeAllclose(x.js());
  });

  test("throws on non-square matrix", () => {
    const x = np.array([
      [1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
    ]);
    expect(() => linalg.cholesky(x)).toThrow();
  });

  test("throws on non-2D array", () => {
    const x = np.array([1.0, 2.0, 3.0]);
    expect(() => linalg.cholesky(x)).toThrow();
  });
});
