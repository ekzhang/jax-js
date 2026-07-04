import {
  defaultDevice,
  Device,
  grad,
  init,
  jvp,
  numpy as np,
  random,
  valueAndGrad,
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

  suite("numpy.linalg.eigh()", () => {
    test("sorts diagonal eigenvalues and eigenvectors", () => {
      const a = np.array([
        [3.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 2.0],
      ]);
      const [values, vectors] = np.linalg.eigh(a);

      expect(values.ref).toBeAllclose([1.0, 2.0, 3.0]);
      expect(vectors).toBeAllclose([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
      ]);
      values.dispose();
    });

    test("reconstructs a symmetric matrix", () => {
      const a = np.array([
        [4.0, 1.0, 2.0],
        [1.0, 3.0, 0.5],
        [2.0, 0.5, 5.0],
      ]);
      const [values, vectors] = np.linalg.eigh(a.ref);
      const reconstructed = np.matmul(
        vectors.ref.mul(values.reshape([1, -1])),
        vectors.ref.transpose(),
      );
      const orthogonal = np.matmul(vectors.ref.transpose(), vectors);

      expect(reconstructed).toBeAllclose(a, { rtol: 1e-3, atol: 1e-3 });
      expect(orthogonal).toBeAllclose(np.eye(3), { rtol: 1e-3, atol: 1e-3 });
    });
  });

  suite("numpy.linalg.eigvalsh()", () => {
    test("returns eigenvalues only", () => {
      const a = np.array([
        [2.0, 1.0],
        [1.0, 2.0],
      ]);
      expect(np.linalg.eigvalsh(a)).toBeAllclose([1.0, 3.0], {
        rtol: 1e-4,
        atol: 1e-4,
      });
    });

    test("supports batched matrices", () => {
      const a = np.array([
        [
          [2.0, 1.0],
          [1.0, 2.0],
        ],
        [
          [3.0, 1.0],
          [1.0, 3.0],
        ],
      ]);
      expect(np.linalg.eigvalsh(a)).toBeAllclose(
        [
          [1.0, 3.0],
          [2.0, 4.0],
        ],
        { rtol: 1e-4, atol: 1e-4 },
      );
    });

    test("only forwards symmetrizeInput option", () => {
      const a = np.array([
        [2.0, 100.0],
        [1.0, 2.0],
      ]);
      expect(
        np.linalg.eigvalsh(a, {
          symmetrizeInput: false,
          lower: false,
        } as { symmetrizeInput: boolean }),
      ).toBeAllclose([1.0, 3.0], {
        rtol: 1e-4,
        atol: 1e-4,
      });
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

    test("gradient of det is adjugate.mT", () => {
      const a = random.uniform(random.key(0), [15, 15]);
      const g = valueAndGrad(np.linalg.det);
      const [detA, da] = g(a.ref);
      const adjA = np.linalg.inv(a).mul(detA);
      expect(da).toBeAllclose(np.matrixTranspose(adjA), { rtol: 1e-3 });
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

    test("gradient of inv sum matches JAX", () => {
      const a = np.array([
        [2.0, 1.0],
        [1.0, 3.0],
      ]);
      const f = (a: np.Array) => np.linalg.inv(a).sum();
      const da = grad(f)(a);
      expect(da).toBeAllclose([
        [-0.16, -0.08],
        [-0.08, -0.04],
      ]);
    });

    test("inv preserves input dtype", () => {
      const dtype = device === "webgpu" ? np.float16 : np.float64;
      const a = np.array(
        [
          [2, 1],
          [1, 3],
        ],
        { dtype },
      );
      const aInv = np.linalg.inv(a.ref);
      expect(aInv.dtype).toBe(dtype);
      const identity = np.matmul(a, aInv);
      expect(identity).toBeAllclose(np.eye(2), { rtol: 0, atol: 1e-3 });
    });

    test("computes inverse of batched matrices", () => {
      const a = random.uniform(random.key(0), [2, 3, 4, 4]);
      const aInv = np.linalg.inv(a.ref);
      const identity = np.matmul(a, aInv);
      expect(identity).toBeAllclose(np.broadcastTo(np.eye(4), [2, 3, 4, 4]), {
        rtol: 0,
        atol: 1e-4,
      });
    });
  });

  suite("numpy.linalg.matrixPower()", () => {
    test("matrixPower(A, 0) preserves input dtype", () => {
      const dtype = device === "webgpu" ? np.float16 : np.float64;
      const a = np.array(
        [
          [2, 1],
          [1, 3],
        ],
        { dtype },
      );
      const result = np.linalg.matrixPower(a, 0);
      expect(result.dtype).toBe(dtype);
      expect(result).toBeAllclose(np.eye(2), { rtol: 0, atol: 1e-3 });
    });
  });

  suite("numpy.linalg.multiDot()", () => {
    test("multiplies a chain of matrices", () => {
      const a = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      const b = np.array([
        [5.0, 6.0, 7.0],
        [8.0, 9.0, 10.0],
      ]);
      const c = np.array([
        [11.0, 12.0],
        [13.0, 14.0],
        [15.0, 16.0],
      ]);

      const result = np.linalg.multiDot([a.ref, b.ref, c.ref]);
      const expected = np.matmul(np.matmul(a, b), c);
      expect(result).toBeAllclose(expected);
    });

    test("handles the two-array case", () => {
      const a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      const b = np.array([
        [7.0, 8.0],
        [9.0, 10.0],
        [11.0, 12.0],
      ]);

      const result = np.linalg.multiDot([a.ref, b.ref]);
      const expected = np.matmul(a, b);
      expect(result).toBeAllclose(expected);
    });

    test("treats 1D endpoints as row and column vectors", () => {
      const x = np.array([1.0, 2.0]);
      const a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      const b = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
      ]);
      const y = np.array([2.0, 3.0]);

      const result = np.linalg.multiDot([x.ref, a.ref, b.ref, y.ref]);
      expect(result.shape).toEqual([]);
      expect(result).toBeAllclose(129.0);
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

  suite("numpy.linalg.vectorNorm()", () => {
    test("L2 norm (default)", () => {
      const x = np.array([3.0, 4.0]);
      expect(np.linalg.vectorNorm(x)).toBeAllclose(5.0);
    });

    test("L1 norm", () => {
      const x = np.array([3.0, -4.0]);
      expect(np.linalg.vectorNorm(x, { ord: 1 })).toBeAllclose(7.0);
    });

    test("Linf norm", () => {
      const x = np.array([3.0, -4.0]);
      expect(np.linalg.vectorNorm(x, { ord: Infinity })).toBeAllclose(4.0);
    });

    test("negative Inf norm", () => {
      const x = np.array([3.0, -4.0]);
      expect(np.linalg.vectorNorm(x, { ord: -Infinity })).toBeAllclose(3.0);
    });

    test("L0 norm (count nonzeros)", () => {
      const x = np.array([0.0, 3.0, 0.0, -4.0, 5.0]);
      expect(np.linalg.vectorNorm(x, { ord: 0 })).toBeAllclose(3.0);
    });

    test("axis argument", () => {
      const x = np.array([
        [3.0, 4.0],
        [5.0, 12.0],
      ]);
      expect(np.linalg.vectorNorm(x, { axis: 1 })).toBeAllclose([5.0, 13.0]);
    });

    test("keepdims", () => {
      const x = np.array([
        [3.0, 4.0],
        [5.0, 12.0],
      ]);
      const norms = np.linalg.vectorNorm(x, { axis: 1, keepdims: true });
      expect(norms.shape).toEqual([2, 1]);
      expect(norms).toBeAllclose([[5.0], [13.0]]);
    });
  });

  suite("numpy.linalg.matrixNorm()", () => {
    const matrix = () =>
      np.array([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
    const batched = () => np.arange(18).astype(np.float32).reshape([2, 3, 3]);

    const expectNorm = (
      input: np.Array,
      ord: number | "fro",
      keepdims: boolean,
      expected: number | number[] | number[][] | number[][][],
      shape: number[],
    ) => {
      const norm = np.linalg.matrixNorm(input, { ord, keepdims });
      expect(norm).toBeAllclose(expected);
      expect(norm.shape).toEqual(shape);
    };

    test("Throws on array with less than 2 dimensions", () => {
      const a = np.array([1.0, 2.0, 3.0, 4.0]);
      expect(() => np.linalg.matrixNorm(a)).toThrow();
    });

    test("Throws on non-supported norms", () => {
      expect(() => np.linalg.matrixNorm(matrix(), { ord: -2 })).toThrow();
      expect(() => np.linalg.matrixNorm(matrix(), { ord: 2 })).toThrow();
      expect(() => np.linalg.matrixNorm(matrix(), { ord: "nuc" })).toThrow();
    });

    test("Frobenius norm (default)", () => {
      expectNorm(matrix(), "fro", false, 5.477225575051661, []);
    });

    test("Frobenius norm keepdims", () => {
      expectNorm(matrix(), "fro", true, [[5.477225575051661]], [1, 1]);
    });

    test("Frobenius norm batched", () => {
      expectNorm(batched(), "fro", false, [14.28285686, 39.7617907], [2]);
    });

    test("Frobenius norm batched keepdims", () => {
      expectNorm(
        batched(),
        "fro",
        true,
        [[[14.28285686]], [[39.7617907]]],
        [2, 1, 1],
      );
    });

    test("L1 norm", () => {
      expectNorm(matrix(), 1, false, 6.0, []);
    });

    test("L1 norm keepdims", () => {
      expectNorm(matrix(), 1, true, [[6.0]], [1, 1]);
    });

    test("L1 norm batched", () => {
      expectNorm(batched(), 1, false, [15.0, 42.0], [2]);
    });

    test("L1 norm batched keepdims", () => {
      expectNorm(batched(), 1, true, [[[15.0]], [[42.0]]], [2, 1, 1]);
    });

    test("L-1 norm", () => {
      expectNorm(matrix(), -1, false, 4.0, []);
    });

    test("L-1 norm keepdims", () => {
      expectNorm(matrix(), -1, true, [[4.0]], [1, 1]);
    });

    test("L-1 norm batched", () => {
      expectNorm(batched(), -1, false, [9.0, 36.0], [2]);
    });

    test("L-1 norm batched keepdims", () => {
      expectNorm(batched(), -1, true, [[[9.0]], [[36.0]]], [2, 1, 1]);
    });

    test("L-infinity norm", () => {
      expectNorm(matrix(), Infinity, false, 7.0, []);
    });

    test("L-infinity norm keepdims", () => {
      expectNorm(matrix(), Infinity, true, [[7.0]], [1, 1]);
    });

    test("L-infinity norm batched", () => {
      expectNorm(batched(), Infinity, false, [21.0, 48.0], [2]);
    });

    test("L-infinity norm batched keepdims", () => {
      expectNorm(batched(), Infinity, true, [[[21.0]], [[48.0]]], [2, 1, 1]);
    });

    test("L-negative-infinity norm", () => {
      expectNorm(matrix(), -Infinity, false, 3.0, []);
    });

    test("L-negative-infinity norm keepdims", () => {
      expectNorm(matrix(), -Infinity, true, [[3.0]], [1, 1]);
    });

    test("L-negative-infinity norm batched", () => {
      expectNorm(batched(), -Infinity, false, [3.0, 30.0], [2]);
    });

    test("L-negative-infinity norm batched keepdims", () => {
      expectNorm(batched(), -Infinity, true, [[[3.0]], [[30.0]]], [2, 1, 1]);
    });
  });
});
