import {
  defaultDevice,
  devices,
  grad,
  init,
  jvp,
  numpy as np,
} from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  suite("jax.numpy.sum()", () => {
    test("can take multiple axes", () => {
      const x = np.arange(24).reshape([2, 3, 4]);
      const y = x.sum([0, 2]);
      expect(y.js()).toEqual([60, 92, 124]);
    });

    test("keepdims preserves dim of size 1", () => {
      const x = np.arange(24).reshape([2, 3, 4]);
      const y = x.sum([0, 2], { keepdims: true });
      expect(y.shape).toEqual([1, 3, 1]);
      expect(y.js()).toEqual([[[60], [92], [124]]]);
    });
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

    test("fetches diagonal of 2D array", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const y = np.diag(x.ref);
      expect(y.js()).toEqual([1, 5, 9]);
      const z = np.diag(x, 1);
      expect(z.js()).toEqual([2, 6]);
    });

    test("can construct off-diagonal", () => {
      expect(np.diag(np.array([1, 2]), 1).js()).toEqual([
        [0, 1, 0],
        [0, 0, 2],
        [0, 0, 0],
      ]);
      expect(np.diag(np.array([1, 2]), -2).js()).toEqual([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 2, 0, 0],
      ]);
    });
  });

  suite("jax.numpy.diagonal()", () => {
    test("diagonal defaults to first two axes", () => {
      const a = np.arange(4).reshape([2, 2]);
      expect(a.ref.diagonal().js()).toEqual([0, 3]);
      expect(a.ref.diagonal(1).js()).toEqual([1]);
      expect(a.diagonal(-1).js()).toEqual([2]);

      const b = np.arange(8).reshape([2, 2, 2]);
      expect(b.diagonal().js()).toEqual([
        [0, 6],
        [1, 7],
      ]);
    });

    test("can take diagonal over other axes", () => {
      const a = np.arange(12).reshape([3, 2, 2]);
      expect(a.ref.diagonal(0, 1, 2).js()).toEqual([
        [0, 3],
        [4, 7],
        [8, 11],
      ]);

      // a[:, :, 0] = [[0, 2], [4, 6], [8, 10]]
      expect(np.diagonal(a.ref, 0, 0, 1).js()).toEqual([
        [0, 6],
        [1, 7],
      ]);
      expect(np.diagonal(a.ref, 1, 0, 1).js()).toEqual([[2], [3]]);
      expect(np.diagonal(a, 1, 1, 0).js()).toEqual([
        [4, 10],
        [5, 11],
      ]);
    });

    test("gradient over diagonal sum-of-squares", () => {
      const a = np.arange(6).astype(np.float32).reshape([2, 3]);
      const f = (a: np.Array) => a.ref.mul(a).diagonal(1).sum();
      expect(grad(f)(a).js()).toEqual([
        [0, 2, 0],
        [0, 0, 10],
      ]);
    });
  });

  suite("jax.numpy.tri()", () => {
    test("computes lower-triangular matrix", () => {
      const x = np.tri(3);
      expect(x.js()).toEqual([
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1],
      ]);
    });

    test("computes rectangular lower-triangular matrix", () => {
      const x = np.tri(2, 4, 1);
      expect(x.js()).toEqual([
        [1, 1, 0, 0],
        [1, 1, 1, 0],
      ]);
    });

    test("triu works", () => {
      const x = np.arange(24).reshape([2, 3, 4]);
      const y = np.triu(x);
      expect(y.js()).toEqual([
        [
          [0, 1, 2, 3],
          [0, 5, 6, 7],
          [0, 0, 10, 11],
        ],
        [
          [12, 13, 14, 15],
          [0, 17, 18, 19],
          [0, 0, 22, 23],
        ],
      ]);
    });

    test("tril works", () => {
      const x = np.arange(24).reshape([2, 3, 4]);
      const y = np.tril(x);
      expect(y.js()).toEqual([
        [
          [0, 0, 0, 0],
          [4, 5, 0, 0],
          [8, 9, 10, 0],
        ],
        [
          [12, 0, 0, 0],
          [16, 17, 0, 0],
          [20, 21, 22, 0],
        ],
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
      x = np.arange(0, 1, 0.2, { dtype: np.float32 });
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

  suite("jax.numpy.less()", () => {
    test("computes less", () => {
      const x = np.array([1, 2, 3, 4]);
      const y = np.array([4, 5, 3, 2]);
      expect(np.less(x.ref, y.ref).js()).toEqual([true, true, false, false]);
    });
  });

  suite("jax.numpy.lessEqual()", () => {
    test("computes less", () => {
      const x = np.array([1, 2, 3, 4]);
      const y = np.array([4, 5, 3, 2]);
      expect(np.lessEqual(x.ref, y.ref).js()).toEqual([
        true,
        true,
        true,
        false,
      ]);
    });
  });

  suite("jax.numpy.greater()", () => {
    test("computes greater", () => {
      const x = np.array([1, 2, 3, 4]);
      const y = np.array([4, 5, 3, 2]);
      expect(np.greater(x.ref, y.ref).js()).toEqual([
        false,
        false,
        false,
        true,
      ]);
    });
  });

  suite("jax.numpy.greaterEqual()", () => {
    test("computes greater", () => {
      const x = np.array([1, 2, 3, 4]);
      const y = np.array([4, 5, 3, 2]);
      expect(np.greaterEqual(x.ref, y.ref).js()).toEqual([
        false,
        false,
        true,
        true,
      ]);
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

  suite("jax.numpy.reshape()", () => {
    test("reshapes a 1D array", () => {
      const x = np.array([1, 2, 3, 4]);
      const y = np.reshape(x, [2, -1]);
      expect(y.js()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });

    test("raises Error on incompatible shapes", () => {
      const x = np.array([1, 2, 3, 4]);
      expect(() => np.reshape(x, [3, 2])).toThrow(Error);
      expect(() => np.reshape(x, [2, 3])).toThrow(Error);
      expect(() => np.reshape(x, [2, 2, 2])).toThrow(Error);
      expect(() => np.reshape(x, [3, -1])).toThrow(Error);
      expect(() => np.reshape(x, [-1, -1])).toThrow(Error);
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

    if (device !== "cpu") {
      test("200-256-200 matrix product", async () => {
        const x = np
          .arange(200)
          .astype(np.float32)
          .reshape([200, 1])
          .mul(np.ones([200, 256]));
        const y = np.ones([256, 200]);
        await Promise.all([x.ref.data(), y.ref.data()]);
        const buf = await np.dot(x, y).data();
        expect(buf.length).toEqual(200 * 200);
        expect(buf[0]).toEqual(0);
        expect(buf[200]).toEqual(256);
        expect(buf[200 * 200 - 1]).toEqual(199 * 256);
      });
    }

    // This test observes a past tuning / shape tracking issue where indices
    // would be improperly calculated applying the Unroll optimization.
    test("1-784-10 matrix product", async () => {
      const x = np.arange(784).astype(np.float32).reshape([1, 784]);
      const y = np.ones([784, 10]);
      await Promise.all([x.ref.data(), y.ref.data()]);
      const buf = await np.dot(x, y).data();
      expect(buf.length).toEqual(10);
      expect(buf).toEqual(
        new Float32Array(Array.from({ length: 10 }, () => (784 * 783) / 2)),
      );
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

  suite("jax.numpy.sign()", () => {
    test("computes sign function", () => {
      const x = np.array([-10, 0, 5]);
      const y = np.sign(x);
      expect(y.js()).toEqual([-1, 0, 1]);
    });

    // TODO: Fix sign(NaN) returning 1 instead of NaN
    test.fails("works with NaN", () => {
      expect(np.sign(NaN).js()).toBeNaN();
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

    test("recip of 0 is infinity", () => {
      const x = np.reciprocal(0);
      expect(x.js()).toEqual(Infinity);

      const y = np.array(9.0).div(0);
      expect(y.js()).toEqual(Infinity);
    });
  });

  suite("jax.numpy.exp()", () => {
    test("computes element-wise exponential", () => {
      const x = np.array([-Infinity, 0, 1, 2, 3]);
      const y = np.exp(x);
      expect(y.js()).toBeAllclose([0, 1, Math.E, Math.E ** 2, Math.E ** 3]);
    });

    test("works with small and large numbers", () => {
      const x = np.array([-1000, -100, -50, -10, 0, 10, 50, 100, 1000]);
      const y = np.exp(x);
      expect(y.js()).toBeAllclose([
        0,
        3.720075976020836e-44,
        1.9287498479639178e-22,
        4.5399929762484854e-5,
        1,
        22026.465794806718,
        5.184705528587072e21,
        2.6881171418161356e43,
        Infinity,
      ]);
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

  suite("jax.numpy.sqrt()", () => {
    test("computes element-wise square root", () => {
      const x = np.array([1, 4, 9]);
      const y = np.sqrt(x);
      expect(y.js()).toBeAllclose([1, 2, 3]);
    });

    test("returns NaN for negative inputs", () => {
      const x = np.array([-1, -4, 9]);
      const y = np.sqrt(x);
      expect(y.js()).toEqual([NaN, NaN, 3.0]);
    });
  });

  suite("jax.numpy.cbrt()", () => {
    test("computes element-wise cube root", () => {
      const x = np.array([-8, -1, 0, 1, 8]);
      const y = np.cbrt(x);
      expect(y).toBeAllclose([-2, -1, 0, 1, 2]);
    });

    test("works with jvp", () => {
      const x = np.array([-8, -1, 0, 1, 8]);
      const [y, dy] = jvp(np.cbrt, [x], [np.ones([5])]);
      expect(y).toBeAllclose([-2, -1, 0, 1, 2]);
      expect(dy).toBeAllclose([1 / 12, 1 / 3, Infinity, 1 / 3, 1 / 12]);
    });
  });

  suite("jax.numpy.power()", () => {
    test("computes element-wise power", () => {
      const x = np.array([-1, 2, 3, 4]);
      const y = np.power(x, 3);
      expect(y).toBeAllclose([NaN, 8, 27, 64]);
    });

    test("multiple different exponents", () => {
      const y = np.power(3, np.array([-2, 0, 0.5, 1, 2]));
      expect(y).toBeAllclose([1 / 9, 1, Math.sqrt(3), 3, 9]);
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

  suite("jax.numpy.pad()", () => {
    test("pads an array equally", () => {
      const a = np.array([1, 2, 3]);
      const b = np.pad(a, 1);
      expect(b.js()).toEqual([0, 1, 2, 3, 0]);

      const c = np.array([
        [1, 2],
        [3, 4],
      ]);
      const d = np.pad(c, 1);
      expect(d.js()).toEqual([
        [0, 0, 0, 0],
        [0, 1, 2, 0],
        [0, 3, 4, 0],
        [0, 0, 0, 0],
      ]);
    });

    test("pads an array with uneven widths", () => {
      const a = np.array([[1]]);
      const b = np.pad(a, [
        [1, 2],
        [3, 0],
      ]);
      expect(b.js()).toEqual([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]);
    });

    test("raises TypeError on axis mismatch", () => {
      const a = np.zeros([1, 2, 3]);
      expect(() => np.pad(a, [])).toThrow(Error);
      expect(() => np.pad(a, [[0, 1]])).not.toThrow(Error);
      expect(() =>
        np.pad(a, [
          [0, 1],
          [1, 2],
        ]),
      ).toThrow(Error);
    });

    test("pad handles backprop", () => {
      const a = np.array([1, 2, 3]);
      expect(grad((x: np.Array) => np.pad(x, 1).sum())(a).js()).toEqual([
        1, 1, 1,
      ]);
    });
  });

  suite("jax.numpy.concatenate()", () => {
    // This suite also handles stack, hstack, vstack, dstack, etc.

    test("can concatenate 1D arrays", () => {
      const a = np.array([1, 2, 3]);
      const b = np.array([4, 5, 6]);
      const c = np.concatenate([a, b]);
      expect(c.js()).toEqual([1, 2, 3, 4, 5, 6]);
    });

    test("concatenation size mismatch", () => {
      const a = np.zeros([2, 3]);
      let b = np.zeros([3, 2]);
      expect(() => np.concatenate([a, b])).toThrow(Error);
      expect(() => np.concatenate([a, b], 1)).toThrow(Error);
      b = b.transpose();
      expect(() => np.concatenate([a, b]).dispose()).not.toThrow(Error);
    });

    test("stack() and variants work", () => {
      expect(np.stack([2, 3]).js()).toEqual([2, 3]);
      expect(np.stack([2, 3], -1).js()).toEqual([2, 3]);
      expect(() => np.stack([2, 3], 1)).toThrow(Error); // invalid axis
      expect(() => np.stack([2, 3], 2)).toThrow(Error); // invalid axis

      expect(np.vstack([1, 2, 3]).js()).toEqual([[1], [2], [3]]);
      expect(np.vstack([np.array([1, 2, 3]), np.ones([3])]).js()).toEqual([
        [1, 2, 3],
        [1, 1, 1],
      ]);

      expect(np.hstack([1, 2, 3]).js()).toEqual([1, 2, 3]);
      expect(np.hstack([np.array([1, 2, 3]), np.ones([3])]).js()).toEqual([
        1, 2, 3, 1, 1, 1,
      ]);

      expect(np.dstack([1, 2, 3]).js()).toEqual([[[1, 2, 3]]]);
      expect(np.dstack([np.array([1, 2, 3]), np.ones([3])]).js()).toEqual([
        [
          [1, 1],
          [2, 1],
          [3, 1],
        ],
      ]);
    });
  });

  suite("jax.numpy.argmax()", () => {
    test("finds maximum of logits", () => {
      const x = np.argmax(np.array([0.1, 0.2, 0.3, 0.2]));
      expect(x.js()).toEqual(2);
    });

    test("retrieves first index of maximum", () => {
      const x = np.argmax(
        np.array([
          [0.1, -0.2, -0.3, 0.1],
          [0, 0.1, 0.3, 0.3],
        ]),
        1,
      );
      expect(x.js()).toEqual([0, 2]);
    });

    test("runs on flattened array by default", () => {
      const x = np.argmax(
        np.array([
          [0.1, -0.2],
          [0.3, 0.1],
        ]),
      );
      expect(x.js()).toEqual(2); // Index of maximum in flattened array
    });
  });

  suite("jax.numpy.tanh()", () => {
    const vals = [-1, -0.7, 0, 0.5, 1.7, 10, 50, 100, 1000];

    test("sinh values", () => {
      for (const x of vals) {
        expect(np.sinh(x)).toBeAllclose(Math.sinh(x));
      }
    });

    test("cosh values", () => {
      for (const x of vals) {
        expect(np.cosh(x)).toBeAllclose(Math.cosh(x));
      }
    });

    test("tanh values", () => {
      for (const x of vals) {
        expect(np.tanh(x)).toBeAllclose(Math.tanh(x));
      }
      expect(np.tanh(Infinity).js()).toEqual(1);
    });
  });

  suite("jax.numpy.atan()", () => {
    test("arctan values", () => {
      const vals = [-1000, -100, -10, -1, 0, 1, 10, 100, 1000, Infinity];
      const atanvals: number[] = np.atan(np.array(vals)).js();
      for (let i = 0; i < vals.length; i++) {
        expect(atanvals[i]).toBeCloseTo(Math.atan(vals[i]), 5);
      }
    });

    test("arcsin and arccos values", () => {
      const vals = [-1, -0.7, 0, 0.5, 1];
      const asinvals: number[] = np.asin(np.array(vals)).js();
      const acosvals: number[] = np.acos(np.array(vals)).js();
      for (let i = 0; i < vals.length; i++) {
        expect(asinvals[i]).toBeCloseTo(Math.asin(vals[i]), 5);
        expect(acosvals[i]).toBeCloseTo(Math.acos(vals[i]), 5);
      }
    });

    test("grad of arctan", () => {
      const x = np.array([1, Math.sqrt(3), 0]);
      const dx = grad((x: np.Array) => np.atan(x).sum())(x);
      const expected = [0.5, 0.25, 1];
      expect(dx.js()).toBeAllclose(expected);
    });

    test("grad of arcsin", () => {
      const x = np.array([-0.5, 0, 0.5]);
      const dx = grad((x: np.Array) => np.asin(x).sum())(x);
      const expected = [2 / Math.sqrt(3), 1, 2 / Math.sqrt(3)];
      expect(dx.js()).toBeAllclose(expected);
    });
  });

  suite("jax.numpy.atan2()", () => {
    test("arctan2 values", () => {
      // Test all four quadrants and special cases with various values
      const y = [3, 5, -7, -2, 4, -6, 0, 0, 1.5, -2.5];
      const x = [4, -2, -3, 8, 0, 0, 5, -9, 1.5, -2.5];
      const result: number[] = np.atan2(np.array(y), np.array(x)).js();
      for (let i = 0; i < y.length; i++) {
        expect(result[i]).toBeCloseTo(Math.atan2(y[i], x[i]), 5);
      }
    });
  });

  suite("jax.numpy.repeat()", () => {
    test("repeats elements of 1D array", () => {
      const x = np.array([1, 2, 3]);
      const y = np.repeat(x, 2);
      expect(y.js()).toEqual([1, 1, 2, 2, 3, 3]);
    });

    test("repeats elements of 2D array along axis", () => {
      const x = np.array([
        [1, 2],
        [3, 4],
      ]);
      const y = np.repeat(x.ref, 2, 0);
      expect(y.js()).toEqual([
        [1, 2],
        [1, 2],
        [3, 4],
        [3, 4],
      ]);

      const z = np.repeat(x, 3, 1);
      expect(z.js()).toEqual([
        [1, 1, 1, 2, 2, 2],
        [3, 3, 3, 4, 4, 4],
      ]);
    });

    test("flattens input when axis is null", () => {
      const x = np.array([
        [1, 2],
        [3, 4],
      ]);
      const y = np.repeat(x, 2);
      expect(y.js()).toEqual([1, 1, 2, 2, 3, 3, 4, 4]);
    });
  });

  suite("jax.numpy.tile()", () => {
    test("tiles 1D array", () => {
      const x = np.array([1, 2, 3]);
      const y = np.tile(x, 2);
      expect(y.js()).toEqual([1, 2, 3, 1, 2, 3]);
    });

    test("tiles 2D array along multiple axes", () => {
      const x = np.array([
        [1, 2],
        [3, 4],
      ]);
      const y = np.tile(x.ref, [2, 1]);
      expect(y.js()).toEqual([
        [1, 2],
        [3, 4],
        [1, 2],
        [3, 4],
      ]);

      const z = np.tile(x, 3);
      expect(z.js()).toEqual([
        [1, 2, 1, 2, 1, 2],
        [3, 4, 3, 4, 3, 4],
      ]);
    });

    test("tiles with reps having more dimensions than array", () => {
      const x = np.array([1, 2]);
      const y = np.tile(x, [2, 2]);
      expect(y.js()).toEqual([
        [1, 2, 1, 2],
        [1, 2, 1, 2],
      ]);
    });
  });

  suite("jax.numpy.var_()", () => {
    test("computes variance", () => {
      const x = np.array([1, 2, 3, 4]);
      const y = np.var_(x);
      expect(y).toBeAllclose(1.25);
    });

    test("computes standard deviation", () => {
      const x = np.array([1, 2, 3, 4]);
      const y = np.std(x);
      expect(y).toBeAllclose(Math.sqrt(1.25));
    });
  });
});
