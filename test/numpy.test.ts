import { devices, grad, init, jvp, numpy as np, setDevice } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

import { DType } from "../src/alu";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    setDevice(device);
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

  suite("jax.numpy.arange()", () => {
    test("can be called with 1 argument", () => {
      let x = np.arange(5);
      expect(x.js()).toEqual([0, 1, 2, 3, 4]);

      x = np.arange(0);
      expect(x.js()).toEqual([]);

      x = np.arange(-10);
      expect(x.js()).toEqual([]);
    });

    test("can be called with 2 arguments", () => {
      let x = np.arange(50, 60);
      expect(x.js()).toEqual([50, 51, 52, 53, 54, 55, 56, 57, 58, 59]);

      x = np.arange(-10, -5);
      expect(x.js()).toEqual([-10, -9, -8, -7, -6]);
    });

    test("can be called with 3 arguments", () => {
      let x = np.arange(0, 10, 2);
      expect(x.js()).toEqual([0, 2, 4, 6, 8]);

      x = np.arange(10, 0, -2);
      expect(x.js()).toEqual([10, 8, 6, 4, 2]);

      x = np.arange(0, -10, -2);
      expect(x.js()).toEqual([0, -2, -4, -6, -8]);
    });

    test("works with non-integer step", () => {
      // By default, it uses Int32 dtype, so this rounds down.
      let x = np.arange(0, 1, 0.2);
      expect(x.js()).toEqual([0, 0, 0, 0, 0]);

      // Explicitly set dtype to Float32.
      x = np.arange(0, 1, 0.2, { dtype: DType.Float32 });
      expect(x).toBeAllclose([0, 0.2, 0.4, 0.6, 0.8]);
    });
  });

  suite("jax.numpy.linspace()", () => {
    test("creates a linear space with 5 elements", () => {
      const x = np.linspace(0, 1, 5);
      expect(x.js()).toEqual([0, 0.25, 0.5, 0.75, 1]);
    });

    test("creates a linear space with 1-3 elements", () => {
      let x = np.linspace(0, 1, 3);
      expect(x.js()).toEqual([0, 0.5, 1]);

      x = np.linspace(0, 1, 2);
      expect(x.js()).toEqual([0, 1]);

      x = np.linspace(0, 1, 1);
      expect(x.js()).toEqual([0]);
    });

    test("defaults to 50 elements", () => {
      const x = np.linspace(0, 1);
      expect(x.shape).toEqual([50]);
      const ar = x.js() as number[];
      expect(ar[0]).toEqual(0);
      expect(ar[49]).toEqual(1);
      expect(ar[25]).toBeCloseTo(25 / 49);
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

  suite("jax.numpy.minimum()", () => {
    test("computes element-wise minimum", () => {
      const x = np.array([1, 2, 3]);
      const y = np.array([4, 2, 0]);
      const z = np.minimum(x, y);
      expect(z.js()).toEqual([1, 2, 0]);
    });

    test("works with jvp", () => {
      const x = np.array([1, 3, 3]);
      const y = np.array([4, 2, 0]);
      const [z, dz] = jvp(
        (x: np.Array, y: np.Array) => np.minimum(x, y),
        [x, y],
        [np.ones([3]), np.zeros([3])],
      );
      expect(z.js()).toEqual([1, 2, 0]);
      expect(dz.js()).toEqual([1, 0, 0]);
    });
  });

  suite("jax.numpy.maximum()", () => {
    test("computes element-wise maximum", () => {
      const x = np.array([1, 2, 3]);
      const y = np.array([4, 2, 0]);
      const z = np.maximum(x, y);
      expect(z.js()).toEqual([4, 2, 3]);
    });

    test("works with jvp", () => {
      const x = np.array([1, 1, 3]);
      const y = np.array([4, 2, 0]);
      const [z, dz] = jvp(
        (x: np.Array, y: np.Array) => np.maximum(x, y),
        [x, y],
        [np.ones([3]), np.zeros([3])],
      );
      expect(z.js()).toEqual([4, 2, 3]);
      expect(dz.js()).toEqual([0, 0, 1]);
    });
  });

  suite("jax.numpy.absolute()", () => {
    test("computes absolute value", () => {
      const x = np.array([-1, 2, -3]);
      const y = np.absolute(x.ref);
      expect(y.js()).toEqual([1, 2, 3]);

      const z = np.abs(x); // Alias for absolute
      expect(z.js()).toEqual([1, 2, 3]);
    });
  });

  suite("jax.numpy.reciprocal()", () => {
    test("computes element-wise reciprocal", () => {
      const x = np.array([1, 2, 3]);
      const y = np.reciprocal(x);
      expect(y.js()).toBeAllclose([1, 0.5, 1 / 3]);
    });

    test("works with jvp", () => {
      const x = np.array([1, 2, 3]);
      const [y, dy] = jvp(
        (x: np.Array) => np.reciprocal(x),
        [x],
        [np.ones([3])],
      );
      expect(y).toBeAllclose([1, 0.5, 1 / 3]);
      expect(dy).toBeAllclose([-1, -0.25, -1 / 9]);
    });

    test("can be used in grad", () => {
      const x = np.array([1, 2, 3]);
      const dx = grad((x: np.Array) => np.reciprocal(x).sum())(x);
      expect(dx).toBeAllclose([-1, -0.25, -1 / 9]);
    });

    test("called via Array.div() and jax.numpy.divide()", () => {
      const x = np.array([1, 2, 3]);
      const y = np.array([4, 5, 6]);
      const z = x.ref.div(y.ref);
      expect(z).toBeAllclose([0.25, 0.4, 0.5]);

      const w = np.divide(x, y);
      expect(w.js()).toBeAllclose([0.25, 0.4, 0.5]);
    });
  });

  suite("jax.numpy.exp()", () => {
    test("computes element-wise exponential", () => {
      const x = np.array([1, 2, 3]);
      const y = np.exp(x);
      expect(y.js()).toBeAllclose([Math.E, Math.E ** 2, Math.E ** 3]);
    });

    test("works with jvp", () => {
      const x = np.array([1, 2, 3]);
      const [y, dy] = jvp((x: np.Array) => np.exp(x), [x], [np.ones([3])]);
      expect(y.js()).toBeAllclose([Math.E, Math.E ** 2, Math.E ** 3]);
      expect(dy.js()).toBeAllclose([Math.E, Math.E ** 2, Math.E ** 3]);
    });

    test("can be used in grad", () => {
      const x = np.array([1, 2, 3]);
      const dx = grad((x: np.Array) => np.exp(x).sum())(x);
      expect(dx.js()).toBeAllclose([Math.E, Math.E ** 2, Math.E ** 3]);
    });

    test("exp2(10) = 1024", () => {
      const x = np.exp2(10);
      expect(x.js()).toBeCloseTo(1024);
    });

    test("exp2(0) = 1", () => {
      const x = np.exp2(0);
      expect(x.js()).toBeCloseTo(1);
    });
  });

  suite("jax.numpy.log()", () => {
    test("computes element-wise natural logarithm", () => {
      const x = np.array([1, Math.E, Math.E ** 2]);
      const y = np.log(x);
      expect(y.js()).toBeAllclose([0, 1, 2]);
    });

    test("works with jvp", () => {
      const x = np.array([1, Math.E, Math.E ** 2]);
      const [y, dy] = jvp((x: np.Array) => np.log(x), [x], [np.ones([3])]);
      expect(y.js()).toBeAllclose([0, 1, 2]);
      expect(dy.js()).toBeAllclose([1, 1 / Math.E, 1 / Math.E ** 2]);
    });

    test("can be used in grad", () => {
      const x = np.array([1, Math.E, Math.E ** 2]);
      const dx = grad((x: np.Array) => np.log(x).sum())(x);
      expect(dx.js()).toBeAllclose([1, 1 / Math.E, 1 / Math.E ** 2]);
    });

    test("log2 and log10", () => {
      const x = np.array([1, 2, 4, 8]);
      const y2 = np.log2(x.ref);
      const y10 = np.log10(x);
      expect(y2.js()).toBeAllclose([0, 1, 2, 3]);
      expect(y10.js()).toBeAllclose([
        0,
        Math.log10(2),
        Math.log10(4),
        Math.log10(8),
      ]);
    });
  });

  suite("jax.numpy.min()", () => {
    test("computes minimum of 1D array", () => {
      const x = np.array([3, 1, 4, 2]);
      const y = np.min(x);
      expect(y.js()).toEqual(1);
    });

    test("computes minimum of 2D array along axis", () => {
      const x = np.array([
        [3, 1, 4],
        [2, 5, 0],
      ]);
      const y = np.min(x, 0);
      expect(y.js()).toEqual([2, 1, 0]);
    });

    test("computes minimum of 2D array without axis", () => {
      const x = np.array([
        [3, 1, 4],
        [2, 5, 0],
      ]);
      const y = np.min(x);
      expect(y.js()).toEqual(0);
    });

    test("can have grad of min", () => {
      const x = np.array([3, 1, 4, 1]);
      const dx = grad((x: np.Array) => np.min(x))(x);
      expect(dx.js()).toEqual([0, 0.5, 0, 0.5]); // Gradient is 1 at the minimum
    });
  });

  suite("jax.numpy.max()", () => {
    test("computes maximum of 1D array", () => {
      const x = np.array([3, 1, 4, 2]);
      const y = np.max(x);
      expect(y.js()).toEqual(4);
    });

    test("computes maximum of 2D array along axis", () => {
      const x = np.array([
        [3, 1, 4],
        [2, 5, 0],
      ]);
      const y = np.max(x, 0);
      expect(y.js()).toEqual([3, 5, 4]);
    });

    test("computes maximum of 2D array without axis", () => {
      const x = np.array([
        [3, 1, 4],
        [2, 5, 0],
      ]);
      const y = np.max(x);
      expect(y.js()).toEqual(5);
    });

    test("can have grad of max", () => {
      const x = np.array([10, 3, 4, 10]);
      const dx = grad((x: np.Array) => np.max(x))(x);
      expect(dx.js()).toEqual([0.5, 0, 0, 0.5]); // Gradient is 1 at the maximum
    });
  });
});
