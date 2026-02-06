import {
  defaultDevice,
  Device,
  grad,
  init,
  jvp,
  lax,
  numpy as np,
  random,
} from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();
const devicesWithLinalg: Device[] = ["cpu", "wasm", "webgpu"];

suite.each(devicesWithLinalg)("device:%s", (device) => {
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

    test("works with jvp", () => {
      const x = np.array([
        [4.0, 2.0],
        [2.0, 5.0],
      ]);
      const dx = np.array([
        [0.1, 0.05],
        [0.05, 0.1],
      ]);
      const [L, dL] = jvp(lax.linalg.cholesky, [x.ref], [dx.ref]);

      // Verify L is correct
      expect(np.matmul(L.ref, L.ref.transpose())).toBeAllclose(x.ref);

      // Verify dL by finite differences: (cholesky(x + eps*dx) - L) / eps ≈ dL
      const eps = 1e-4;
      const L2 = lax.linalg.cholesky(x.add(dx.mul(eps)));
      const dL_fd = L2.sub(L).div(eps);
      expect(dL).toBeAllclose(dL_fd, { rtol: 1e-2, atol: 2e-3 });
    });

    test("works with grad", () => {
      const x = np.array([
        [4.0, 2.0],
        [2.0, 5.0],
      ]);
      // Loss: sum of squared elements of L
      const f = (x: np.Array) => {
        x = x.ref.add(x.transpose()).mul(0.5); // Ensure symmetry
        return np.square(lax.linalg.cholesky(x)).sum();
      };
      const dx = grad(f)(x.ref);

      // Verify gradient by finite differences
      const eps = 1e-4;
      const xData = x.js() as number[][];
      const expected: number[][] = [[], []];
      for (let i = 0; i < 2; i++) {
        for (let j = 0; j < 2; j++) {
          const xp = xData.map((row) => [...row]);
          const xm = xData.map((row) => [...row]);
          xp[i][j] += eps;
          xm[i][j] -= eps;
          const fp = f(np.array(xp)).js() as number;
          const fm = f(np.array(xm)).js() as number;
          expected[i][j] = (fp - fm) / (2 * eps);
        }
      }
      expect(dx).toBeAllclose(expected, { rtol: 1e-2, atol: 1e-3 });
    });
  });

  suite("jax.lax.linalg.lu()", () => {
    test("example with partial pivoting", () => {
      const A = np.array([
        [4, 3],
        [6, 3],
      ]);
      const [lu, pivots, permutation] = lax.linalg.lu(A);
      expect(lu).toBeAllclose([
        [6, 3],
        [0.6666667, 1.0],
      ]);
      expect(pivots.js()).toEqual([1, 1]);
      expect(permutation.js()).toEqual([1, 0]);
    });

    test("P @ A = L @ U holds", () => {
      const n = 30;
      const A = random.uniform(random.key(0), [n, n]);
      const [lu, pivots, permutation] = lax.linalg.lu(A.ref);

      pivots.dispose(); // Not needed
      const P = np.eye(n).slice(permutation);
      const L = np.tril(lu.ref, -1).add(np.eye(n));
      const U = np.triu(lu);

      const PA = np.matmul(P, A);
      const LU = np.matmul(L, U);
      expect(PA).toBeAllclose(LU, { rtol: 1e-5, atol: 1e-6 });
    });

    test("works with jvp", () => {
      const A = np.array([
        [4.0, 3.0, 6.3],
        [6.0, 3.0, -2.4],
      ]);
      const dA = np.array([
        [0.1, 0.2, -0.2],
        [0.3, 0.4, -0.1],
      ]);

      const luFn = (x: np.Array) => {
        const [lu, pivots, permutation] = lax.linalg.lu(x);
        pivots.dispose();
        permutation.dispose();
        return lu;
      };
      const [lu, dlu] = jvp(luFn, [A.ref], [dA.ref]);

      // Verify dlu by finite differences
      const eps = 1e-4;
      const lu2 = lax.linalg.lu(A.add(dA.mul(eps)))[0];
      const dlu_fd = lu2.sub(lu).div(eps);
      expect(dlu).toBeAllclose(dlu_fd, { rtol: 1e-2, atol: 1e-3 });
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
      expect(x).toBeAllclose([[2], [5 / 3]]);
    });

    test("works with jvp on b", () => {
      const L = np.array([
        [2, 0],
        [1, 3],
      ]);
      const b = np.array([[4], [7]]);
      const db = np.array([[0.1], [0.2]]);

      const solve = (b: np.Array) =>
        lax.linalg.triangularSolve(L.ref, b, { leftSide: true, lower: true });
      const [x, dx] = jvp(solve, [b.ref], [db.ref]);

      // Verify x is correct
      expect(np.matmul(L.ref, x.ref)).toBeAllclose(b.ref);

      // Verify dx by finite differences
      const eps = 1e-4;
      const x2 = lax.linalg.triangularSolve(L, b.add(db.mul(eps)), {
        leftSide: true,
        lower: true,
      });
      const dx_fd = x2.sub(x).div(eps);
      expect(dx).toBeAllclose(dx_fd, { rtol: 1e-2, atol: 1e-3 });
    });

    test("works with grad on b", () => {
      const L = np.array([
        [2, 0],
        [1, 3],
      ]);
      const b = np.array([[4], [7]]);

      // Loss: sum of squared elements of solution
      const f = (b: np.Array) =>
        np
          .square(
            lax.linalg.triangularSolve(L.ref, b, {
              leftSide: true,
              lower: true,
            }),
          )
          .sum();
      const db = grad(f)(b.ref);

      // Verify gradient by finite differences
      const eps = 1e-4;
      const bData = b.js() as number[][];
      const expected: number[][] = [[], []];
      for (let i = 0; i < 2; i++) {
        const bp = bData.map((row) => [...row]);
        const bm = bData.map((row) => [...row]);
        bp[i][0] += eps;
        bm[i][0] -= eps;
        const fp = f(np.array(bp)).js() as number;
        const fm = f(np.array(bm)).js() as number;
        expected[i][0] = (fp - fm) / (2 * eps);
      }
      expect(db).toBeAllclose(expected, { rtol: 1e-2, atol: 1e-3 });
    });

    test("behavior with transposed A", () => {
      // See: https://github.com/ekzhang/jax-js/issues/73
      const L = np.array([
        [1, 1000000],
        [1, 1],
      ]);
      const b = np.array([[1], [1]]);
      const x = lax.linalg.triangularSolve(L, b, {
        leftSide: true,
        lower: true,
        transposeA: true,
      });
      expect(x).toBeAllclose([[0], [1]]);
    });

    test("right-hand side triangular solve", () => {
      // Solve x @ U = b
      const U = np.array([
        [2, 1],
        [0, 3],
      ]);
      const b = np.array([[4, 7]]);
      const x = lax.linalg.triangularSolve(U, b, {
        leftSide: false,
        lower: false,
      });
      expect(x).toBeAllclose([[2, 5 / 3]]);
    });
  });
});
