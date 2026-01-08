import {
  defaultDevice,
  Device,
  grad,
  init,
  jvp,
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

  suite("numpy.linalg.cholesky()", () => {
    test("symmetrizes input by default", () => {
      const x = np.array([
        [4.0, 2.01],
        [1.99, 5.0],
      ]);
      const L = np.linalg.cholesky(x.ref);
      const reconstructed = np.matmul(L.ref, L.transpose());
      const symmetrized = x.ref.add(x.transpose()).mul(0.5);
      expect(reconstructed).toBeAllclose(symmetrized);
    });
  });

  suite("numpy.linalg.det()", () => {
    test("computes determinant of simple matrix", () => {
      const a = np.array([
        [4.0, 7.0],
        [2.0, 6.0],
      ]);
      const detA = np.linalg.det(a.ref);
      expect(detA).toBeAllclose(10.0);
    });
  });

  suite("numpy.linalg.inv()", () => {
    test("computes inverse of simple matrix", () => {
      const a = np.array([
        [4.0, 7.0],
        [2.0, 6.0],
      ]);
      const aInv = np.linalg.inv(a.ref);
      const identity = np.matmul(a, aInv);
      expect(identity).toBeAllclose(np.eye(2));
    });

    test("computes inverse of batched matrices", () => {
      const a = random.uniform(random.key(0), [2, 3, 4, 4]);
      const aInv = np.linalg.inv(a.ref);
      const identity = np.matmul(a, aInv);
      expect(identity).toBeAllclose(np.broadcastTo(np.eye(4), [2, 3, 4, 4]), {
        atol: 1e-4,
      });
    });
  });

  suite("numpy.linalg.lstsq()", () => {
    test("solves overdetermined system (M > N)", () => {
      // 3x2 system: more equations than unknowns
      const a = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
      ]);
      const b = np.array([[1.0], [2.0], [3.0]]);
      const x = np.linalg.lstsq(a.ref, b.ref);

      // Verify solution minimizes ||Ax - b||
      // The normal equations: A^T A x = A^T b
      const atA = np.matmul(a.ref.transpose(), a.ref);
      const atb = np.matmul(a.ref.transpose(), b.ref);
      const lhs = np.matmul(atA.ref, x.ref);
      expect(lhs).toBeAllclose(atb, { rtol: 1e-4, atol: 1e-4 });
    });

    test("solves underdetermined system (M < N)", () => {
      // 2x3 system: fewer equations than unknowns
      const a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      const b = np.array([[1.0], [2.0]]);
      const x = np.linalg.lstsq(a.ref, b.ref);

      // Verify Ax = b (should be exact for underdetermined systems)
      const ax = np.matmul(a.ref, x.ref);
      expect(ax).toBeAllclose(b, { rtol: 1e-4, atol: 1e-4 });
    });

    test("solves square system exactly", () => {
      const a = np.array([
        [2.0, 1.0],
        [1.0, 3.0],
      ]);
      const b = np.array([[5.0], [7.0]]);
      const x = np.linalg.lstsq(a.ref, b.ref);

      // Verify Ax = b
      const ax = np.matmul(a.ref, x.ref);
      expect(ax).toBeAllclose(b, { rtol: 1e-4, atol: 1e-4 });
    });

    test("handles multiple right-hand sides (M > N)", () => {
      const a = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
      ]);
      const b = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
      ]);
      const x = np.linalg.lstsq(a.ref, b.ref);

      // x should have shape (2, 2)
      expect(x.shape).toEqual([2, 2]);

      // Verify normal equations for each column
      const atA = np.matmul(a.ref.transpose(), a.ref);
      const atb = np.matmul(a.ref.transpose(), b.ref);
      const lhs = np.matmul(atA.ref, x.ref);
      expect(lhs).toBeAllclose(atb, { rtol: 1e-4, atol: 1e-4 });
    });

    test("handles multiple right-hand sides (M < N)", () => {
      const a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      const b = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      const x = np.linalg.lstsq(a.ref, b.ref);

      // x should have shape (3, 2)
      expect(x.shape).toEqual([3, 2]);

      // Verify Ax = b
      const ax = np.matmul(a.ref, x.ref);
      expect(ax).toBeAllclose(b, { rtol: 1e-4, atol: 1e-4 });
    });

    test("throws on non-2D coefficient matrix", () => {
      const a = np.array([1.0, 2.0, 3.0]);
      const b = np.array([1.0, 2.0, 3.0]);
      expect(() => np.linalg.lstsq(a, b).js()).toThrow();
    });

    test("throws on mismatched dimensions", () => {
      const a = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      const b = np.array([1.0, 2.0, 3.0]); // Wrong size
      expect(() => np.linalg.lstsq(a, b).js()).toThrow();
    });

    test("works with jvp on b (underdetermined)", () => {
      const a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      const b = np.array([[1.0], [2.0]]);
      const db = np.array([[0.1], [0.1]]);

      const solve = (b: np.Array) => np.linalg.lstsq(a.ref, b);
      const [x, dx] = jvp(solve, [b.ref], [db.ref]);

      // Verify dx by finite differences
      const eps = 1e-4;
      const x2 = np.linalg.lstsq(a, b.add(db.mul(eps)));
      const dx_fd = x2.sub(x).div(eps);
      expect(dx).toBeAllclose(dx_fd, { rtol: 1e-2, atol: 1e-3 });
    });

    test("works with grad on b (underdetermined)", () => {
      const a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      const b = np.array([[1.0], [2.0]]);

      const f = (b: np.Array) => np.square(np.linalg.lstsq(a.ref, b)).sum();
      const db = grad(f)(b.ref);

      // Verify gradient by finite differences
      const eps = 1e-4;
      const bData = b.js() as number[][];
      const expected: number[][] = [];
      for (let i = 0; i < 2; i++) {
        const bp = bData.map((row) => [...row]);
        const bm = bData.map((row) => [...row]);
        bp[i][0] += eps;
        bm[i][0] -= eps;
        const fp = f(np.array(bp)).js() as number;
        const fm = f(np.array(bm)).js() as number;
        expected.push([(fp - fm) / (2 * eps)]);
      }
      expect(db).toBeAllclose(expected, { rtol: 1e-2, atol: 1e-3 });
    });
  });

  suite("numpy.linalg.slogdet()", () => {
    test("computes slogdet of simple matrix", () => {
      const a = np.array([
        [4.0, 7.0],
        [2.0, 6.0],
      ]);
      const [sign, logdet] = np.linalg.slogdet(a.ref);
      expect(sign).toBeAllclose(1);
      expect(logdet).toBeAllclose(Math.log(10));
    });
  });

  suite("numpy.linalg.solve()", () => {
    test("solves simple Ax = b", () => {
      const a = np.array([
        [3.0, 2.0],
        [1.0, 2.0],
      ]);
      const b = np.array([5.0, 4.0]);
      const x = np.linalg.solve(a, b);
      expect(x).toBeAllclose([0.5, 1.75]);
    });

    test("solves random batched AX = B", () => {
      const [k1, k2] = random.split(random.key(0), 2);
      const a = random.uniform(k1, [10, 15, 15]);
      const xTrue = random.uniform(k2, [10, 15, 5]);
      const b = np.matmul(a.ref, xTrue.ref); // B = A @ X_true
      expect(b.shape).toEqual(xTrue.shape);

      const xPred = np.linalg.solve(a, b);
      expect(xPred.shape).toEqual(xTrue.shape);
      expect(xPred).toBeAllclose(xTrue, { rtol: 1e-2, atol: 1e-4 });
    });
  });
});
