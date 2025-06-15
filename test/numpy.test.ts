import {
  backendTypes,
  grad,
  init,
  jvp,
  numpy as np,
  setBackend,
} from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const backendsAvailable = await init();

suite.each(backendTypes)("backend:%s", (backend) => {
  const skipped = !backendsAvailable.includes(backend);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    setBackend(backend);
  });

  suite("jax.numpy.eye()", () => {
    test("computes a square matrix", () => {
      const x = np.eye(3);
      expect(x).toBeAllclose([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
    });

    test("computes a rectangular matrix", () => {
      const x = np.eye(2, 3);
      expect(x).toBeAllclose([
        [1, 0, 0],
        [0, 1, 0],
      ]);
    });

    test("can be multiplied", () => {
      const x = np.eye(3, 5).mul(-42);
      expect(x.ref.sum()).toBeAllclose(-126);
      expect(x).toBeAllclose([
        [-42, 0, 0, 0, 0],
        [0, -42, 0, 0, 0],
        [0, 0, -42, 0, 0],
      ]);
    });
  });

  suite("jax.numpy.diag()", () => {
    test("constructs diagonal from 1D array", () => {
      const x = np.array([1, 2, 3]);
      const y = np.diag(x);
      expect(y.js()).toEqual([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3],
      ]);
    });
  });

  suite("jax.numpy.where()", () => {
    test("computes where", () => {
      const x = np.array([1, 2, 3]);
      const y = np.array([4, 5, 6]);
      const z = np.array([true, false, true]);
      const result = np.where(z, x, y);
      expect(result.js()).toEqual([1, 5, 3]);
    });

    test("works with jvp", () => {
      const x = np.array([1, 2, 3]);
      const y = np.array([4, 5, 6]);
      const z = np.array([true, false, true]);
      const result = jvp(
        (x: np.Array, y: np.Array) => np.where(z, x, y),
        [x, y],
        [np.array([1, 1, 1]), np.zeros([3])],
      );
      expect(result[0].js()).toEqual([1, 5, 3]);
      expect(result[1].js()).toEqual([1, 0, 1]);
    });

    test("works with grad reverse-mode", () => {
      const x = np.array([1, 2, 3]);
      const y = np.array([4, 5, 6]);
      const z = np.array([true, false, true]);
      const f = ({ x, y }: { x: np.Array; y: np.Array }) =>
        np.where(z.ref, x, y).sum();
      const grads = grad(f)({ x, y });
      expect(grads.x.js()).toEqual([1, 0, 1]);
      expect(grads.y.js()).toEqual([0, 1, 0]);
      z.dispose();
    });

    test("where broadcasting", () => {
      const z = np.array([true, false, true, true]);
      expect(np.where(z, 1, 3).js()).toEqual([1, 3, 1, 1]);
      expect(np.where(false, 1, 3).js()).toEqual(3);
      expect(np.where(false, 1, np.array([10, 11])).js()).toEqual([10, 11]);
      expect(np.where(true, 7, np.array([10, 11, 12])).js()).toEqual([7, 7, 7]);
    });
  });

  suite("jax.numpy.equal()", () => {
    test("computes equal", () => {
      const x = np.array([1, 2, 3, 4]);
      const y = np.array([4, 5, 3, 4]);
      expect(np.equal(x.ref, y.ref).js()).toEqual([false, false, true, true]);
      expect(np.notEqual(x, y).js()).toEqual([true, true, false, false]);
    });

    test("does not propagate gradients", () => {
      const x = np.array([1, 2, 3]);
      const y = np.array([0, 5, 6]);
      const f = ({ x, y }: { x: np.Array; y: np.Array }) =>
        np.where(np.equal(x, y), 1, 0).sum();
      const grads = grad(f)({ x, y });
      expect(grads.x.js()).toEqual([0, 0, 0]);
      expect(grads.y.js()).toEqual([0, 0, 0]);
    });
  });

  suite("jax.numpy.transpose()", () => {
    test("transposes a 1D array (no-op)", () => {
      const x = np.array([1, 2, 3]);
      const y = np.transpose(x);
      expect(y.js()).toEqual([1, 2, 3]);
    });

    test("transposes a 2D array", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const y = np.transpose(x);
      expect(y.js()).toEqual([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);
    });

    test("composes with jvp", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const [y, dy] = jvp(
        (x: np.Array) => x.ref.transpose().mul(x.transpose()),
        [x.ref],
        [np.ones([2, 3])],
      );
      expect(y).toBeAllclose(x.ref.mul(x.ref).transpose());
      expect(dy).toBeAllclose(x.mul(2).transpose());
    });

    test("composes with grad", () => {
      const x = np.ones([3, 4]);
      const dx = grad((x: np.Array) => x.transpose().sum())(x.ref);
      expect(dx).toBeAllclose(x);
    });
  });

  suite("jax.numpy.matrixTranspose()", () => {
    test("throws TypeError on 1D array", () => {
      const x = np.zeros([20]);
      expect(() => np.matrixTranspose(x)).toThrow(TypeError);
    });

    test("transposes a stack of matrices", () => {
      const x = np.zeros([5, 60, 7]);
      expect(np.matrixTranspose(x).shape).toEqual([5, 7, 60]);
    });
  });

  suite("jax.numpy.reshape()", () => {
    test("reshapes a 1D array", () => {
      const x = np.array([1, 2, 3, 4]);
      const y = np.reshape(x, [2, -1]);
      expect(y.js()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });

    test("raises TypeError on incompatible shapes", () => {
      const x = np.array([1, 2, 3, 4]);
      expect(() => np.reshape(x, [3, 2])).toThrow(TypeError);
      expect(() => np.reshape(x, [2, 3])).toThrow(TypeError);
      expect(() => np.reshape(x, [2, 2, 2])).toThrow(TypeError);
      expect(() => np.reshape(x, [3, -1])).toThrow(TypeError);
      expect(() => np.reshape(x, [-1, -1])).toThrow(TypeError);
    });

    test("composes with jvp", () => {
      const x = np.array([1, 2, 3, 4]);
      const [y, dy] = jvp(
        (x: np.Array) => np.reshape(x, [2, 2]).sum(),
        [x],
        [np.ones([4])],
      );
      expect(y).toBeAllclose(10);
      expect(dy).toBeAllclose(4);
    });
  });

  suite("jax.numpy.flip()", () => {
    test("flips a 1D array", () => {
      const x = np.array([1, 2, 3]);
      expect(np.flip(x).js()).toEqual([3, 2, 1]);
    });

    test("flips a 2D array", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(np.flip(x.ref).js()).toEqual([
        [6, 5, 4],
        [3, 2, 1],
      ]);
      expect(np.flip(x.ref, 0).js()).toEqual([
        [4, 5, 6],
        [1, 2, 3],
      ]);
      expect(np.flip(x, 1).js()).toEqual([
        [3, 2, 1],
        [6, 5, 4],
      ]);
    });
  });

  suite("jax.numpy.matmul()", () => {
    test("acts as vector dot product", () => {
      const x = np.array([1, 2, 3, 4]);
      const y = np.array([10, 100, 1000, 1]);
      const z = np.matmul(x, y);
      expect(z.js()).toEqual(3214);
    });

    test("computes 2x2 matmul", () => {
      const x = np.array([
        [1, 2],
        [3, 4],
      ]);
      const y = np.array([
        [5, 6],
        [7, 8],
      ]);
      const z = np.matmul(x, y);
      expect(z.js()).toEqual([
        [19, 22],
        [43, 50],
      ]);
    });

    test("computes 2x3 matmul", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const y = np.array([
        [7, 8],
        [9, 10],
        [11, 12],
      ]);
      const z = np.matmul(x, y);
      expect(z.js()).toEqual([
        [58, 64],
        [139, 154],
      ]);
    });

    test("computes stacked 3x3 matmul", () => {
      const a = np.array([
        [
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ],
        [
          [10, 11, 12],
          [13, 14, 15],
          [16, 17, 18],
        ],
      ]);
      const b = np.array([
        [20, 21, 22],
        [23, 24, 25],
        [26, 27, 28],
      ]);
      const c = np.matmul(a, b);
      expect(c.shape).toEqual([2, 3, 3]);
      expect(c.js()).toEqual([
        [
          [144, 150, 156],
          [351, 366, 381],
          [558, 582, 606],
        ],
        [
          [765, 798, 831],
          [972, 1014, 1056],
          [1179, 1230, 1281],
        ],
      ]);
    });
  });

  suite("jax.numpy.dot()", () => {
    test("acts as scalar multiplication", () => {
      const z = np.dot(3, 4);
      expect(z.js()).toEqual(12);
    });

    test("computes 1D dot product", () => {
      const x = np.array([1, 2, 3]);
      const y = np.array([4, 5, 6]);
      const z = np.dot(x, y);
      expect(z.js()).toEqual(32);
    });

    test("computes 2D dot product", () => {
      const x = np.array([
        [1, 2],
        [3, 4],
      ]);
      const y = np.array([
        [5, 6],
        [7, 8],
      ]);
      const z = np.dot(x, y);
      expect(z.js()).toEqual([
        [19, 22],
        [43, 50],
      ]);
    });

    test("produces correct shape", () => {
      const x = np.zeros([2, 3, 4, 5]);
      const y = np.zeros([1, 4, 5, 6]);
      const z = np.dot(x, y);
      expect(z.shape).toEqual([2, 3, 4, 1, 4, 6]);
    });
  });

  suite("jax.numpy.meshgrid()", () => {
    test("creates xy meshgrid", () => {
      const x = np.array([1, 2, 3]);
      const y = np.array([4, 5]);
      const [X, Y] = np.meshgrid([x, y]);
      expect(X.js()).toEqual([
        [1, 2, 3],
        [1, 2, 3],
      ]);
      expect(Y.js()).toEqual([
        [4, 4, 4],
        [5, 5, 5],
      ]);
    });

    test("works with ij indexing", () => {
      const x = np.array([1, 2, 3]);
      const y = np.array([4, 5]);
      const [X, Y] = np.meshgrid([x, y], { indexing: "ij" });
      expect(X.js()).toEqual([
        [1, 1],
        [2, 2],
        [3, 3],
      ]);
      expect(Y.js()).toEqual([
        [4, 5],
        [4, 5],
        [4, 5],
      ]);
    });

    test("works with 3D arrays", () => {
      // Note: XYZ -> [Y, X, Z]
      const x = np.array([1, 2]);
      const y = np.array([3, 4, 5]);
      const z = np.array([6, 7, 8, 9]);
      const [X, Y, Z] = np.meshgrid([x, y, z]); // "xy" indexing
      expect(X.shape).toEqual([3, 2, 4]);
      expect(Y.shape).toEqual([3, 2, 4]);
      expect(Z.shape).toEqual([3, 2, 4]);
    });
  });
});
