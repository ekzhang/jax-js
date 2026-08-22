import {
  defaultDevice,
  devices,
  grad,
  init,
  jit,
  jvp,
  numpy as np,
  vmap,
} from "@jax-js/jax";
import { beforeEach, expect, onTestFinished, suite, test } from "vitest";

import { hasStrictNumerics } from "./setup";

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

    test("is identity on empty axes", () => {
      const x = np.arange(24).reshape([2, 3, 4]);
      const y = x.ref.sum([]);
      expect(x.js()).toEqual(y.js());
    });
  });

  suite("jax.numpy.countNonzero()", () => {
    test("counts across all dimensions", () => {
      const x = np.array([
        [1, 0, -2],
        [0, 3, 0],
      ]);
      const result = np.countNonzero(x);
      expect(result.dtype).toBe(np.int32);
      expect(result.js()).toEqual(3);
    });

    test("counts along one or more axes", () => {
      const x = np.array([
        [1, 0, -2],
        [0, 3, 0],
      ]);
      expect(np.countNonzero(x.ref, 0).js()).toEqual([1, 1, 1]);
      expect(np.countNonzero(x.ref, 1).js()).toEqual([2, 1]);
      expect(np.countNonzero(x, [0, 1]).js()).toEqual(3);
    });

    test("supports keepdims", () => {
      const x = np.array([
        [1, 0, -2],
        [0, 3, 0],
      ]);
      const result = np.countNonzero(x, 1, { keepdims: true });
      expect(result.shape).toEqual([2, 1]);
      expect(result.js()).toEqual([[2], [1]]);
    });

    test("handles booleans, NaN, and empty arrays", () => {
      expect(np.countNonzero(np.array([false, true, true])).js()).toEqual(2);
      expect(np.countNonzero(np.array([0, NaN, 2])).js()).toEqual(2);
      expect(np.countNonzero(np.zeros([0, 3])).js()).toEqual(0);
    });

    test("works inside jit", () => {
      const countRows = jit((x: np.Array) =>
        np.countNonzero(x, 1, { keepdims: true }),
      );
      const result = countRows(
        np.array([
          [0, 1, 2],
          [0, 0, 3],
        ]),
      );
      expect(result.js()).toEqual([[2], [1]]);
    });
  });

  suite("jax.numpy.isscalar()", () => {
    test("returns true for JS numbers and booleans", () => {
      expect(np.isscalar(3.1)).toBe(true);
      expect(np.isscalar(2)).toBe(true);
      expect(np.isscalar(NaN)).toBe(true);
      expect(np.isscalar(true)).toBe(true);
    });

    test("treats zero-dimensional arrays as scalars", () => {
      const x = np.array(3.1);
      expect(np.isscalar(x)).toBe(true);
      x.dispose();
    });

    test("returns false for arrays with one or more dimensions", () => {
      const x = np.array([3.1]);
      expect(np.isscalar(x)).toBe(false);
      x.dispose();
      const y = np.ones([2, 3]);
      expect(np.isscalar(y)).toBe(false);
      y.dispose();
    });

    test("returns false for other JS values", () => {
      expect(np.isscalar([3.1])).toBe(false);
      expect(np.isscalar("3.1")).toBe(false);
      expect(np.isscalar(null)).toBe(false);
      expect(np.isscalar(undefined)).toBe(false);
    });

    test("does not consume the array reference", () => {
      const x = np.array(5);
      expect(np.isscalar(x)).toBe(true);
      expect(x.js()).toEqual(5);
    });

    test("works on tracers inside jit", () => {
      const f = jit((x: np.Array) => {
        expect(np.isscalar(x)).toBe(false);
        const s = x.sum();
        expect(np.isscalar(s)).toBe(true);
        return s;
      });
      expect(f(np.array([1, 2, 3])).js()).toEqual(6);
    });
  });

  suite("jax.numpy.bartlett()", () => {
    test("odd window size", () => {
      const w = np.bartlett(5);
      expect(w.dtype).toBe(np.float32);
      expect(w).toBeAllclose([0, 0.5, 1, 0.5, 0]);
    });

    test("even window size", () => {
      const w = np.bartlett(4);
      expect(w).toBeAllclose([0, 2 / 3, 2 / 3, 0]);
    });

    test("larger window matches numpy", () => {
      const w = np.bartlett(9);
      expect(w).toBeAllclose([0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0]);
    });

    test("size 0 and 1 edge cases", () => {
      expect(np.bartlett(0).js()).toEqual([]);
      expect(np.bartlett(1).js()).toEqual([1]);
      expect(np.bartlett(2).js()).toEqual([0, 0]);
    });

    test("rejects invalid window sizes", () => {
      expect(() => np.bartlett(-1)).toThrow(/non-negative integer/);
      expect(() => np.bartlett(0.5)).toThrow(/non-negative integer/);
    });

    test("works inside jit", () => {
      const f = jit(() => np.bartlett(5).sum());
      expect(f()).toBeAllclose(2);
    });
  });

  suite("jax.numpy.average()", () => {
    test("no weights is same as mean", () => {
      const x = np.array([1, 2, 3, 4]);
      expect(np.average(x).js()).toEqual(2.5);
    });

    test("with weights", () => {
      const x = np.array([1, 2, 3, 4]);
      const w = np.array([4, 3, 2, 1]);
      expect(np.average(x, null, { weights: w }).js()).toEqual(2);
    });

    test("with weights along axis", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const w = np.array([0.25, 0.5, 0.25]);
      expect(np.average(x, 1, { weights: w }).js()).toEqual([2, 5]);
    });

    test("with matching shape weights", () => {
      const x = np.array([
        [1, 2],
        [3, 4],
      ]);
      const w = np.array([
        [1, 2],
        [3, 4],
      ]);
      expect(np.average(x, 1, { weights: w })).toBeAllclose([5 / 3, 25 / 7]);
    });

    test("keepdims", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const result = np.average(x, 1, { keepdims: true });
      expect(result.shape).toEqual([2, 1]);
      expect(result.js()).toEqual([[2], [5]]);
    });
  });

  suite("jax.numpy.mean()", () => {
    test("promotes integer input to float", () => {
      // Regression test: mean() used to cast the result back to the input
      // dtype, truncating the fractional part (e.g. mean([1,2,3,4]) -> 2).
      const x = np.array([1, 2, 3, 4], { dtype: np.int32 });
      const y = np.mean(x);
      expect(y.dtype).toBe(np.float32);
      expect(y.js()).toBeCloseTo(2.5);
    });

    test("promotes boolean input to float", () => {
      const x = np.array([true, false, true], { dtype: np.bool });
      const y = np.mean(x);
      expect(y.dtype).toBe(np.float32);
      expect(y.js()).toBeCloseTo(2 / 3);
    });

    test("keeps float32 dtype", () => {
      const x = np.array([1, 2, 3, 4], { dtype: np.float32 });
      const y = np.mean(x);
      expect(y.dtype).toBe(np.float32);
      expect(y.js()).toBeCloseTo(2.5);
    });

    test("works along an axis", () => {
      const x = np.array(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        { dtype: np.int32 },
      );
      const y = np.mean(x, 1);
      expect(y.dtype).toBe(np.float32);
      expect(y.js()).toEqual([2, 5]);
    });
  });

  suite("jax.numpy.cumsum()", () => {
    test("computes cumsum along axis", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const y = np.cumsum(x.ref, 0);
      expect(y.js()).toEqual([
        [1, 2, 3],
        [5, 7, 9],
      ]);
      const z = np.cumsum(x, 1);
      expect(z.js()).toEqual([
        [1, 3, 6],
        [4, 9, 15],
      ]);
    });

    test("handles 0-dimensional scalars", () => {
      expect(np.cumsum(5).js()).toEqual([5]);
      expect(np.cumsum(5, 0).js()).toEqual([5]);
      expect(() => np.cumsum(5, 1)).toThrow("out of bounds");

      expect(np.cumulativeSum(5).js()).toEqual([5]);
      expect(np.cumulativeSum(5, { axis: 0 }).js()).toEqual([5]);
      expect(np.cumulativeSum(5, { includeInitial: true }).js()).toEqual([
        0, 5,
      ]);
    });

    test("cumulative product works", () => {
      const x = np.array([1, 2, 3, 4]);
      expect(np.cumprod(x).js()).toEqual([1, 2, 6, 24]);
    });
  });

  suite("jax.numpy.diff()", () => {
    test("computes the first difference", () => {
      const x = np.array([1, 2, 4, 7, 0]);
      expect(np.diff(x).js()).toEqual([1, 2, 3, -7]);
    });

    test("computes higher-order differences", () => {
      const x = np.array([1, 2, 4, 7, 0]);
      expect(np.diff(x.ref, 2).js()).toEqual([1, 1, -10]);
      expect(np.diff(x, 3).js()).toEqual([0, -11]);
    });

    test("returns the array unchanged for n=0, ignoring edge values", () => {
      const x = np.array([1, 5, 2]);
      expect(np.diff(x, 0, -1, { prepend: 0, append: 10 }).js()).toEqual([
        1, 5, 2,
      ]);
    });

    test("returns an empty array when n exceeds the axis size", () => {
      const x = np.arange(3);
      const y = np.diff(x, 5);
      expect(y.shape).toEqual([0]);
      expect(y.js()).toEqual([]);
    });

    test("differences along an axis", () => {
      const x = np.array([
        [1, 3, 6, 10],
        [0, 5, 6, 8],
      ]);
      expect(np.diff(x.ref).js()).toEqual([
        [2, 3, 4],
        [5, 1, 2],
      ]);
      expect(np.diff(x, 1, 0).js()).toEqual([[-1, 2, 0, -2]]);
    });

    test("supports prepend and append values", () => {
      const x = np.array([1, 2, 4, 7, 0]);
      expect(np.diff(x.ref, 1, -1, { prepend: 0, append: 10 }).js()).toEqual([
        1, 1, 2, 3, -7, 10,
      ]);
      expect(np.diff(x, 1, -1, { prepend: np.array([0, 1]) }).js()).toEqual([
        1, 0, 1, 2, 3, -7,
      ]);
    });

    test("uses notEqual for boolean arrays", () => {
      const x = np.array([true, true, false, true]);
      expect(np.diff(x).js()).toEqual([false, true, true]);
    });

    test("throws on invalid inputs", () => {
      expect(() => np.diff(5)).toThrow("at least one-dimensional");
      expect(() => np.diff(np.arange(3), -1)).toThrow("non-negative");
    });

    test("works with jit and grad", () => {
      const f = jit((x: np.Array) => np.diff(x, 2));
      expect(f(np.array([1, 2, 4, 7, 0])).js()).toEqual([1, 1, -10]);

      const x = np.array([1, 2, 4]);
      const dx = grad((x: np.Array) => np.diff(x).sum())(x);
      expect(dx.js()).toEqual([-1, 0, 1]);
    });
  });

  suite("jax.numpy.ediff1d()", () => {
    test("computes consecutive differences", () => {
      const x = np.array([1, 2, 4, 7, 0]);
      expect(np.ediff1d(x).js()).toEqual([1, 2, 3, -7]);
    });

    test("flattens the input array", () => {
      const x = np.array([
        [1, 2, 4],
        [1, 6, 24],
      ]);
      expect(np.ediff1d(x).js()).toEqual([1, 2, -3, 5, 18]);
    });

    test("prepends toBegin and appends toEnd", () => {
      const x = np.array([1, 2, 4, 7, 0]);
      const y = np.ediff1d(x, { toBegin: -99, toEnd: np.array([88, 99]) });
      expect(y.js()).toEqual([-99, 1, 2, 3, -7, 88, 99]);
    });

    test("casts toBegin and toEnd to the input dtype", () => {
      const x = np.arange(4); // int32
      const y = np.ediff1d(x, { toBegin: -99.5, toEnd: np.array([1.5, 2.5]) });
      expect(y.dtype).toBe(np.int32);
      expect(y.js()).toEqual([-99, 1, 1, 1, 1, 2]);
    });

    test("returns an empty array for scalar and single-element inputs", () => {
      expect(np.ediff1d(5).js()).toEqual([]);
      expect(np.ediff1d(np.array([5])).js()).toEqual([]);
      expect(np.ediff1d(np.zeros([0])).js()).toEqual([]);
    });

    test("handles toBegin and toEnd with an empty difference", () => {
      const y = np.ediff1d(np.array([5]), { toBegin: 1, toEnd: 2 });
      expect(y.js()).toEqual([1, 2]);
    });

    test("works inside jit", () => {
      const f = jit((x: np.Array) =>
        np.ediff1d(x, { toBegin: 0, toEnd: np.array([10]) }),
      );
      expect(f(np.array([1, 2, 4, 7])).js()).toEqual([0, 1, 2, 3, 10]);
    });

    test("works with grad", () => {
      const f = (x: np.Array) => np.ediff1d(x).sum();
      const g = grad(f)(np.array([1.0, 2.0, 4.0]));
      expect(g.js()).toEqual([-1, 0, 1]);
    });
  });

  suite("jax.numpy.trapezoid()", () => {
    test("integrates with default unit spacing", () => {
      const y = np.array([1, 2, 3]);
      const result = np.trapezoid(y);
      expect(result.dtype).toBe(np.float32);
      expect(result.js()).toEqual(4);
    });

    test("uses dx spacing", () => {
      const y = np.array([1, 2, 3]);
      expect(np.trapezoid(y, null, { dx: 2 }).js()).toEqual(8);
    });

    test("broadcasts array-valued dx along a non-final axis", () => {
      const y = np.array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const dx = np.array([[1], [2]]);
      expect(np.trapezoid(y, null, { dx, axis: 0 }).js()).toEqual([10, 13]);
    });

    test("uses 1-D sample points x", () => {
      const y = np.array([1, 2, 3]);
      const x = np.array([4, 6, 8]);
      expect(np.trapezoid(y.ref, x).js()).toEqual(8);

      const xUneven = np.array([0, 1, 3]);
      expect(np.trapezoid(y, xUneven)).toBeAllclose(6.5);
    });

    test("integrates along an axis of a 2-D array", () => {
      const y = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(np.trapezoid(y.ref, null, { axis: 1 }).js()).toEqual([4, 10]);
      expect(np.trapezoid(y.ref, null, { axis: -1 }).js()).toEqual([4, 10]);
      expect(np.trapezoid(y, null, { axis: 0 }).js()).toEqual([2.5, 3.5, 4.5]);
    });

    test("broadcasts 1-D x against a 2-D y", () => {
      const y = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const x = np.array([0, 1, 3]);
      expect(np.trapezoid(y, x)).toBeAllclose([6.5, 15.5]);
    });

    test("accepts x with the same shape as y", () => {
      const y = np.array([
        [1, 1, 1],
        [2, 2, 2],
      ]);
      const x = np.array([
        [0, 1, 2],
        [0, 2, 4],
      ]);
      expect(np.trapezoid(y, x).js()).toEqual([2, 8]);
    });

    test("broadcasts lower-rank x against y", () => {
      const y = np.ones([2, 2, 3]);
      const x = np.array([
        [0, 1, 3],
        [0, 2, 4],
      ]);
      expect(np.trapezoid(y, x).js()).toEqual([
        [3, 4],
        [3, 4],
      ]);
    });

    test("rejects incompatible x shapes", () => {
      const y = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const x = np.zeros([2, 3, 4]);
      expect(() => np.trapezoid(y, x)).toThrow();
    });

    test("promotes integers before intermediate arithmetic", () => {
      expect(
        np.trapezoid(np.array([2_000_000_000, 2_000_000_000])),
      ).toBeAllclose(2_000_000_000);

      const y = np.array([1, 1]);
      const x = np.array([-2_000_000_000, 2_000_000_000]);
      expect(np.trapezoid(y, x)).toBeAllclose(4_000_000_000);
    });

    test("returns zero for a size-1 axis", () => {
      expect(np.trapezoid(np.array([5])).js()).toEqual(0);
    });

    test("works with jit and grad", () => {
      const f = jit((y: np.Array) => np.trapezoid(y));
      expect(f(np.array([1, 2, 3])).js()).toEqual(4);

      const g = grad((y: np.Array) => np.trapezoid(y));
      expect(g(np.array([1, 2, 3])).js()).toEqual([0.5, 1, 0.5]);
    });
  });

  suite("jax.numpy.cross()", () => {
    test("2D cross product", () => {
      const a = np.array([1, 2]);
      const b = np.array([3, 4]);
      expect(np.cross(a, b).js()).toEqual(-2);
    });

    test("3D cross product", () => {
      const a = np.array([1, 2, 3]);
      const b = np.array([4, 5, 6]);
      expect(np.cross(a, b).js()).toEqual([-3, 6, -3]);
    });

    test("batched 3D cross product", () => {
      const a = np.array([
        [1, 2, 3],
        [3, 4, 3],
      ]);
      const b = np.array([
        [2, 3, 2],
        [4, 5, 6],
      ]);
      expect(np.cross(a, b).js()).toEqual([
        [-5, 4, -1],
        [9, -6, -1],
      ]);
    });

    test("cross product along axis=0", () => {
      const a = np.array([
        [1, 2, 3],
        [3, 4, 3],
      ]);
      const b = np.array([
        [2, 3, 2],
        [4, 5, 6],
      ]);
      expect(np.cross(a, b, { axis: 0 }).js()).toEqual([-2, -2, 12]);
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

  suite("jax.numpy.diagflat()", () => {
    test("constructs diagonal from 1D array", () => {
      const x = np.array([1, 2, 3]);
      expect(np.diagflat(x).js()).toEqual([
        [1, 0, 0],
        [0, 2, 0],
        [0, 0, 3],
      ]);
    });

    test("flattens 2D input before constructing diagonal", () => {
      const x = np.array([
        [1, 2],
        [3, 4],
      ]);
      expect(np.diagflat(x).js()).toEqual([
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 4],
      ]);
    });

    test("can construct off-diagonal", () => {
      expect(np.diagflat(np.array([[1, 2]]), 1).js()).toEqual([
        [0, 1, 0],
        [0, 0, 2],
        [0, 0, 0],
      ]);
      expect(np.diagflat(np.array([1, 2]), -1).js()).toEqual([
        [0, 0, 0],
        [1, 0, 0],
        [0, 2, 0],
      ]);
    });

    test("works inside jit", () => {
      const f = jit((x: np.Array) => np.diagflat(x, 1));
      const result = f(
        np.array([
          [1, 2],
          [3, 4],
        ]),
      );
      expect(result.js()).toEqual([
        [0, 1, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 3, 0],
        [0, 0, 0, 0, 4],
        [0, 0, 0, 0, 0],
      ]);
    });
  });

  suite("jax.numpy.diagIndices()", () => {
    test("returns indices for the main diagonal of a 2D array", () => {
      const [rows, cols] = np.diagIndices(3);
      expect(rows.dtype).toBe(np.int32);
      expect(cols.dtype).toBe(np.int32);
      expect(rows.js()).toEqual([0, 1, 2]);
      expect(cols.js()).toEqual([0, 1, 2]);
    });

    test("supports higher-dimensional arrays", () => {
      const indices = np.diagIndices(2, 3);
      expect(indices).toHaveLength(3);
      expect(indices[0]).toBe(indices[1]);
      expect(indices[1]).toBe(indices[2]);
      for (const index of indices) {
        expect(index.js()).toEqual([0, 1]);
      }
    });

    test("can be used to access the diagonal", () => {
      const x = np.arange(9).reshape([3, 3]);
      const [rows, cols] = np.diagIndices(3);
      expect(x.slice(rows, cols).js()).toEqual([0, 4, 8]);
    });

    test("handles n=0 and ndim=0", () => {
      const [rows, cols] = np.diagIndices(0);
      expect(rows.js()).toEqual([]);
      expect(cols.js()).toEqual([]);
      expect(np.diagIndices(3, 0)).toHaveLength(0);
    });

    test("throws on invalid arguments", () => {
      expect(() => np.diagIndices(-1)).toThrow(
        "n must be a nonnegative integer",
      );
      expect(() => np.diagIndices(3, -1)).toThrow(
        "ndim must be a nonnegative integer",
      );
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

    test("computes trace", () => {
      const x = np.arange(9).reshape([3, 3]);
      expect(np.trace(x.ref).js()).toEqual(12);
      expect(np.trace(x, 1).js()).toEqual(6);
    });
  });

  suite("jax.numpy.diagIndicesFrom()", () => {
    test("returns diagonal indices for a 2D array", () => {
      const x = np.zeros([3, 3]);
      const [rows, cols] = np.diagIndicesFrom(x);
      expect(rows.dtype).toBe(np.int32);
      expect(rows.js()).toEqual([0, 1, 2]);
      expect(cols.js()).toEqual([0, 1, 2]);
    });

    test("indexes the main diagonal of an array", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      const [rows, cols] = np.diagIndicesFrom(x.ref);
      expect(x.slice(rows, cols).js()).toEqual([1, 5, 9]);
    });

    test("returns ndim index arrays for higher-dimensional arrays", () => {
      const x = np.zeros([2, 2, 2]);
      const indices = np.diagIndicesFrom(x);
      expect(indices).toHaveLength(3);
      expect(indices[0]).toBe(indices[1]);
      expect(indices[1]).toBe(indices[2]);
      for (const idx of indices) {
        expect(idx.js()).toEqual([0, 1]);
      }
    });

    test("throws on non-square or low-dimensional arrays", () => {
      expect(() => np.diagIndicesFrom(np.zeros([3, 4]))).toThrow(
        "all dimensions of input must be equal",
      );
      expect(() => np.diagIndicesFrom(np.zeros([2, 2, 3]))).toThrow(
        "all dimensions of input must be equal",
      );
      expect(() => np.diagIndicesFrom(np.zeros([3]))).toThrow(
        "input array must be at least 2D",
      );
    });

    test("works inside jit", () => {
      const takeDiag = jit((x: np.Array) => {
        const [rows, cols] = np.diagIndicesFrom(x.ref);
        return x.slice(rows, cols);
      });
      const result = takeDiag(
        np.array([
          [1, 2],
          [3, 4],
        ]),
      );
      expect(result.js()).toEqual([1, 4]);
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

  suite("jax.numpy.logspace()", () => {
    test("creates log-spaced values with base 10", () => {
      // logspace(0, 2, 3) should give 10^0, 10^1, 10^2 = [1, 10, 100]
      const x = np.logspace(0, 2, 3);
      expect(x.js()).toBeAllclose([1, 10, 100]);
    });

    test("creates log-spaced values with base 2", () => {
      // logspace(0, 3, 4, base=2) should give 2^0, 2^1, 2^2, 2^3 = [1, 2, 4, 8]
      const x = np.logspace(0, 3, 4, true, 2);
      expect(x.js()).toBeAllclose([1, 2, 4, 8]);
    });

    test("handles endpoint=false", () => {
      // logspace(0, 2, 4, endpoint=false) should give [1, ~3.16, 10, ~31.6]
      const x = np.logspace(0, 2, 4, false);
      const result = x.js() as number[];
      expect(result[0]).toBeCloseTo(1, 5);
      expect(result[1]).toBeCloseTo(Math.pow(10, 0.5), 5);
      expect(result[2]).toBeCloseTo(10, 5);
      expect(result[3]).toBeCloseTo(Math.pow(10, 1.5), 5);
    });

    test("defaults to 50 elements with base 10", () => {
      const x = np.logspace(0, 1);
      expect(x.shape).toEqual([50]);
      const ar = x.js() as number[];
      expect(ar[0]).toBeCloseTo(1, 5); // 10^0
      expect(ar[49]).toBeCloseTo(10, 5); // 10^1
    });
  });

  suite("jax.numpy.geomspace()", () => {
    test("creates a geometric progression", () => {
      const x = np.geomspace(1, 1000, 4);
      expect(x.js()).toBeAllclose([1, 10, 100, 1000]);
    });

    test("supports non-power-of-10 endpoints", () => {
      const x = np.geomspace(1, 256, 9);
      expect(x.js()).toBeAllclose([1, 2, 4, 8, 16, 32, 64, 128, 256]);
    });

    test("supports decreasing sequences", () => {
      const x = np.geomspace(1000, 1, 4);
      expect(x.js()).toBeAllclose([1000, 100, 10, 1]);
    });

    test("supports negative sequences", () => {
      const x = np.geomspace(-1000, -1, 4);
      expect(x.js()).toBeAllclose([-1000, -100, -10, -1]);
    });

    test("handles endpoint=false", () => {
      const x = np.geomspace(1, 10000, 4, false);
      expect(x.js()).toBeAllclose([1, 10, 100, 1000]);
    });

    test("defaults to 50 elements", () => {
      const x = np.geomspace(1, 10);
      expect(x.shape).toEqual([50]);
      const ar = x.js() as number[];
      expect(ar[0]).toBeCloseTo(1, 5);
      expect(ar[49]).toBeCloseTo(10, 5);
    });

    test("supports integer output dtype", () => {
      const x = np.geomspace(1, 16, 5, true, { dtype: np.int32 });
      expect(x.dtype).toBe(np.int32);
      expect(x.js()).toEqual([1, 2, 4, 8, 16]);
    });

    test("throws on zero or mixed-sign endpoints", () => {
      expect(() => np.geomspace(0, 10, 5)).toThrow(RangeError);
      expect(() => np.geomspace(1, 0, 5)).toThrow(RangeError);
      expect(() => np.geomspace(-1, 10, 5)).toThrow(RangeError);
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

  suite("jax.numpy.select()", () => {
    // https://numpy.org/devdocs/reference/generated/numpy.select.html
    test("selects from choices with a default, using the NumPy docs example", () => {
      const x = np.arange(6);
      const condlist = [np.less(x.ref, 3), np.greater(x.ref, 3)];
      const choicelist = [np.negative(x.ref), np.square(x)];
      const result = np.select(condlist, choicelist, 42);
      expect(result.js()).toEqual([0, -1, -2, 42, 16, 25]);
    });

    test("first matching condition wins when conditions overlap", () => {
      const x = np.arange(6);
      const condlist = [np.lessEqual(x.ref, 4), np.greater(x.ref, 3)];
      const choicelist = [x.ref, np.square(x)];
      const result = np.select(condlist, choicelist, 55);
      expect(result.js()).toEqual([0, 1, 2, 3, 4, 25]);
    });

    test("default value defaults to zero", () => {
      const x = np.arange(4);
      const result = np.select([np.greater(x.ref, 1)], [x]);
      expect(result.js()).toEqual([0, 0, 2, 3]);
    });

    test("converts numeric conditions to boolean", () => {
      const result = np.select(
        [np.array([0, 1, -2]), np.array([1, 0, 0])],
        [10, 20],
        30,
      );
      expect(result.js()).toEqual([20, 10, 10]);
    });

    test("broadcasts conditions and scalar choices", () => {
      const cond = np.array([
        [true, false],
        [false, true],
      ]);
      const result = np.select(
        [cond, np.array([false, true])],
        [1, np.array([[10], [20]])],
        -1,
      );
      expect(result.js()).toEqual([
        [1, 10],
        [-1, 1],
      ]);
    });

    test("promotes dtypes across choices and default", () => {
      const x = np.arange(3);
      const c = np.array([0.5, 1.5, 2.5], { dtype: np.float32 });
      const result = np.select([np.less(x, 1)], [c], 2);
      expect(result.dtype).toBe(np.float32);
      expect(result.js()).toEqual([0.5, 2, 2]);
    });

    test("throws on mismatched or empty inputs", () => {
      expect(() => np.select([true], [])).toThrow(
        "condlist must have length equal to choicelist",
      );
      expect(() => np.select([], [])).toThrow("condlist must be non-empty");
    });

    test("works with grad reverse-mode", () => {
      const f = (x: np.Array) =>
        np
          .select(
            [np.less(x.ref, 2), np.greater(x.ref, 3)],
            [x.ref.mul(2), x.ref.mul(3)],
            0,
          )
          .sum();
      const g = grad(f)(np.array([1.0, 2.0, 3.0, 4.0]));
      expect(g.js()).toEqual([2, 0, 0, 3]);
    });

    test("works inside jit", () => {
      const f = jit((x: np.Array) =>
        np.select(
          [np.less(x.ref, 3), np.greater(x.ref, 3)],
          [x.ref, np.square(x)],
          42,
        ),
      );
      expect(f(np.arange(6)).js()).toEqual([0, 1, 2, 42, 16, 25]);
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

  suite("jax.numpy.arrayEqual()", () => {
    test("equal arrays", () => {
      const a = np.array([1, 2, 3]);
      const b = np.array([1, 2, 3]);
      expect(np.arrayEqual(a, b).js()).toBe(true);
    });

    test("unequal arrays", () => {
      const a = np.array([1, 2, 3]);
      const b = np.array([1, 2, 4]);
      expect(np.arrayEqual(a, b).js()).toBe(false);
    });

    test("different shapes", () => {
      const a = np.array([1, 2, 3]);
      const b = np.array([1, 2]);
      expect(np.arrayEqual(a, b).js()).toBe(false);
    });

    test("NaN not equal by default", () => {
      const a = np.array([1, NaN]);
      const b = np.array([1, NaN]);
      expect(np.arrayEqual(a, b).js()).toBe(false);
    });

    test("NaN equal with equalNaN", () => {
      const a = np.array([1, NaN]);
      const b = np.array([1, NaN]);
      expect(np.arrayEqual(a, b, { equalNaN: true }).js()).toBe(true);
    });
  });

  suite("jax.numpy.arrayEquiv()", () => {
    test("equal arrays", () => {
      const a = np.array([1, 2, 3]);
      const b = np.array([1, 2, 3]);
      expect(np.arrayEquiv(a, b).js()).toBe(true);
    });

    test("broadcast-compatible arrays", () => {
      const a = np.array([
        [1, 2],
        [1, 2],
      ]);
      const b = np.array([1, 2]);
      expect(np.arrayEquiv(a, b).js()).toBe(true);
    });

    test("broadcast-compatible but unequal", () => {
      const a = np.array([
        [1, 2],
        [3, 4],
      ]);
      const b = np.array([1, 2]);
      expect(np.arrayEquiv(a, b).js()).toBe(false);
    });

    test("incompatible shapes", () => {
      const a = np.array([1, 2, 3]);
      const b = np.array([1, 2]);
      expect(np.arrayEquiv(a, b).js()).toBe(false);
    });
  });

  suite("jax.numpy.isin()", () => {
    test("tests membership element-wise", () => {
      const element = np.array([
        [0, 2],
        [4, 6],
      ]);
      const testElements = np.array([1, 2, 4, 8]);
      const result = np.isin(element, testElements);
      expect(result.dtype).toBe(np.bool);
      expect(result.js()).toEqual([
        [false, true],
        [true, false],
      ]);
    });

    test("supports invert", () => {
      const element = np.array([
        [0, 2],
        [4, 6],
      ]);
      const testElements = np.array([1, 2, 4, 8]);
      expect(np.isin(element, testElements, { invert: true }).js()).toEqual([
        [true, false],
        [false, true],
      ]);
    });

    test("flattens testElements of any shape", () => {
      const element = np.array([1, 2, 3, 4]);
      const testElements = np.array([
        [1, 3],
        [5, 7],
      ]);
      expect(np.isin(element, testElements).js()).toEqual([
        true,
        false,
        true,
        false,
      ]);
    });

    test("handles scalars and empty testElements", () => {
      expect(np.isin(3, np.array([1, 2, 3])).js()).toEqual(true);
      expect(np.isin(np.array([1, 2]), np.array([])).js()).toEqual([
        false,
        false,
      ]);
      expect(
        np.isin(np.array([1, 2]), np.array([]), { invert: true }).js(),
      ).toEqual([true, true]);
    });

    test("handles empty element arrays", () => {
      const result = np.isin(np.array([]), np.array([1, 2]));
      expect(result.dtype).toBe(np.bool);
      expect(result.js()).toEqual([]);
      expect(
        np.isin(np.array([]), np.array([1, 2]), { invert: true }).js(),
      ).toEqual([]);
      const empty2d = np.isin(np.zeros([2, 0]), np.array([1]));
      expect(empty2d.shape).toEqual([2, 0]);
      expect(empty2d.js()).toEqual([[], []]);
    });

    test("promotes mixed dtypes and NaN never matches", () => {
      expect(np.isin(np.array([1, 2]), np.array([2.0, 3.5])).js()).toEqual([
        false,
        true,
      ]);
      expect(np.isin(np.array([NaN, 1]), np.array([NaN, 1])).js()).toEqual([
        false,
        true,
      ]);
    });

    test("works inside jit", () => {
      const f = jit((x: np.Array, y: np.Array) => np.isin(x, y));
      const result = f(np.array([1, 2, 3]), np.array([2, 3, 5]));
      expect(result.js()).toEqual([false, true, true]);

      const g = jit((x: np.Array, y: np.Array) =>
        np.isin(x, y, { invert: true }),
      );
      const inverted = g(np.array([1, 2, 3]), np.array([2, 3, 5]));
      expect(inverted.js()).toEqual([true, false, false]);
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

  suite("jax.numpy.swapaxes()", () => {
    test("swaps axis of an array", () => {
      const x = np.arange(12).reshape([2, 2, 3]);
      expect(np.swapaxes(x, 1, 2).js()).toEqual([
        [
          [0, 3],
          [1, 4],
          [2, 5],
        ],
        [
          [6, 9],
          [7, 10],
          [8, 11],
        ],
      ]);
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

  suite("jax.numpy.rot90()", () => {
    test("rotates a 2D array counter-clockwise by default", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(np.rot90(x).js()).toEqual([
        [3, 6],
        [2, 5],
        [1, 4],
      ]);
    });

    test("handles negative rotations", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(np.rot90(x, -1).js()).toEqual([
        [4, 1],
        [5, 2],
        [6, 3],
      ]);
    });

    test("rotates across the specified axes", () => {
      const x = np.array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      expect(np.rot90(x, 1, [1, 2]).js()).toEqual([
        [
          [2, 4],
          [1, 3],
        ],
        [
          [6, 8],
          [5, 7],
        ],
      ]);
    });

    test("requires distinct axes", () => {
      const x = np.array([
        [1, 2],
        [3, 4],
      ]);
      expect(() => np.rot90(x, 1, [0, 0])).toThrow(Error);
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

    test("jit with fused bias and relu", () => {
      const matmulWithBiasAndRelu = jit(
        (x: np.Array, w: np.Array, b: np.Array) => {
          const y = np.matmul(x, w).add(b);
          return np.maximum(y, 0);
        },
      );

      const x = np.array([
        [1, -1],
        [-1, 1],
      ]);
      const w = np.array([
        [2, 3],
        [4, 6],
      ]);
      const b = np.array([10, -10]);

      const y = matmulWithBiasAndRelu(x, w, b);
      expect(y.js()).toEqual([
        [8, 0],
        [12, 0],
      ]);
    });
  });

  suite("jax.numpy.matvec()", () => {
    test("basic matrix-vector product", () => {
      const a = np.array([
        [1, 2],
        [3, 4],
      ]);
      const v = np.array([5, 6]);
      expect(np.matvec(a, v).js()).toEqual([17, 39]);
    });

    test("batched matrix-vector product", () => {
      const a = np.array([
        [
          [1, 0],
          [0, 1],
        ],
        [
          [0, 1],
          [1, 0],
        ],
      ]);
      const v = np.array([3, 7]);
      const result = np.matvec(a, v);
      expect(result.js()).toEqual([
        [3, 7],
        [7, 3],
      ]);
    });

    test("rotation example from docs", () => {
      const a = np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1],
      ]);
      const v = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 6, 8],
      ]);
      const result = np.matvec(a, v);
      expect(result.js()).toEqual([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
        [6, 0, 8],
      ]);
    });
  });

  suite("jax.numpy.vecmat()", () => {
    test("basic vector-matrix product", () => {
      const v = np.array([1, 2]);
      const a = np.array([
        [3, 4],
        [5, 6],
      ]);
      expect(np.vecmat(v, a).js()).toEqual([13, 16]);
    });

    test("projection example from docs", () => {
      const v = np.array([0, 4, 2]);
      const a = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
      ]);
      expect(np.vecmat(v, a).js()).toEqual([0, 4, 0]);
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

    // This caught a past regression in Wasm codegen.
    test("jitted dot with bias", async () => {
      const rows = 16;
      const depth = 64;
      const cols = 4;
      const aData = new Float32Array(rows * depth);
      const wData = new Float32Array(depth * cols);
      const biasData = new Float32Array(cols);
      const expectedData = new Float32Array(rows * cols);

      for (let i = 0; i < aData.length; i++) {
        aData[i] =
          Math.sin(i * 0.17) * 0.7 + (Math.floor(i / depth) % 7) * 0.03;
      }
      for (let i = 0; i < wData.length; i++) {
        wData[i] = Math.cos(i * 0.11) * 0.2;
      }
      for (let i = 0; i < biasData.length; i++) {
        biasData[i] = (i - 2) * 0.1;
      }
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          let sum = biasData[col];
          for (let k = 0; k < depth; k++) {
            sum += aData[row * depth + k] * wData[k * cols + col];
          }
          expectedData[row * cols + col] = sum;
        }
      }

      const affine = jit((a: np.Array, w: np.Array, b: np.Array) =>
        np.dot(a, w).add(b),
      );
      const actual = affine(
        np.array(aData).reshape([rows, depth]),
        np.array(wData).reshape([depth, cols]),
        np.array(biasData),
      );
      expect(actual.shape).toEqual([rows, cols]);

      const actualData = await actual.data();
      let maxDiff = 0;
      for (let i = 0; i < expectedData.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(actualData[i] - expectedData[i]));
      }
      expect(maxDiff).toBeLessThan(1e-4);
    });
  });

  suite("jax.numpy.tensordot()", () => {
    test("2-3-4 with 3-4-5", async () => {
      const x1 = np.arange(24).reshape([2, 3, 4]);
      const x2 = np.ones([3, 4, 5]);
      let z = np.tensordot(x1.ref, x2.ref);
      expect(await z.jsAsync()).toEqual([
        [66, 66, 66, 66, 66],
        [210, 210, 210, 210, 210],
      ]);
      // Equivalent to the above as explicit sequences.
      z = np.tensordot(x1, x2, [
        [1, 2],
        [0, 1],
      ]);
      expect(await z.jsAsync()).toEqual([
        [66, 66, 66, 66, 66],
        [210, 210, 210, 210, 210],
      ]);
    });
  });

  suite("jax.numpy.einsum()", () => {
    test("basic einsum matmul", () => {
      const a = np.arange(6).reshape([2, 3]);
      const b = np.ones([3, 4]);
      const c = np.einsum("ik,kj->ij", a, b);
      expect(c.js()).toEqual([
        [3, 3, 3, 3],
        [12, 12, 12, 12],
      ]);
    });

    test("einsum one-array sums", () => {
      const a = np.arange(6).reshape([2, 3]);
      let c = np.einsum("ij->", a.ref);
      expect(c.js()).toEqual(15);

      c = np.einsum(a.ref, [0, 1], []);
      expect(c.js()).toEqual(15);

      c = np.einsum(a.ref, [0, 1], []);
      expect(c.js()).toEqual(15);

      c = np.einsum("ij->j", a.ref);
      expect(c.js()).toEqual([3, 5, 7]);

      c = np.einsum("ji->j", a.ref);
      expect(c.js()).toEqual([3, 12]);

      c = np.einsum("ii->", a.slice([0, 2], [1, 3]));
      expect(c.js()).toEqual(6);
    });

    test("einsum transposition", () => {
      const a = np.arange(6).reshape([2, 3]);
      const b = np.einsum("ji", a);
      expect(b.js()).toEqual([
        [0, 3],
        [1, 4],
        [2, 5],
      ]);
    });

    test("examples from jax docs", () => {
      // https://docs.jax.dev/en/latest/_autosummary/jax.numpy.einsum.html
      const M = np.arange(16).reshape([4, 4]);
      const x = np.arange(4);
      const y = np.array([5, 4, 3, 2]);
      onTestFinished(() => {
        M.dispose();
        x.dispose();
        y.dispose();
      });

      // Vector product
      expect(np.einsum("i,i", x.ref, y.ref).js()).toEqual(16);
      expect(np.einsum("i,i->", x.ref, y.ref).js()).toEqual(16);
      expect(np.einsum(x.ref, [0], y.ref, [0]).js()).toEqual(16);
      expect(np.einsum(x.ref, [0], y.ref, [0], []).js()).toEqual(16);

      // Matrix product
      expect(np.einsum("ij,j->i", M.ref, x.ref).js()).toEqual([14, 38, 62, 86]);
      expect(np.einsum("ij,j", M.ref, x.ref).js()).toEqual([14, 38, 62, 86]);
      expect(np.einsum(M.ref, [0, 1], x.ref, [1], [0]).js()).toEqual([
        14, 38, 62, 86,
      ]);
      expect(np.einsum(M.ref, [0, 1], x.ref, [1]).js()).toEqual([
        14, 38, 62, 86,
      ]);

      // Outer product
      const outerExpected = [
        [0, 0, 0, 0],
        [5, 4, 3, 2],
        [10, 8, 6, 4],
        [15, 12, 9, 6],
      ];
      expect(np.einsum("i,j->ij", x.ref, y.ref).js()).toEqual(outerExpected);
      expect(np.einsum("i,j", x.ref, y.ref).js()).toEqual(outerExpected);
      expect(np.einsum(x.ref, [0], y.ref, [1], [0, 1]).js()).toEqual(
        outerExpected,
      );
      expect(np.einsum(x.ref, [0], y.ref, [1]).js()).toEqual(outerExpected);

      // 1D array sum
      expect(np.einsum("i->", x.ref).js()).toEqual(6);
      expect(np.einsum(x.ref, [0], []).js()).toEqual(6);

      // Sum along an axis
      expect(np.einsum("...j->...", M.ref).js()).toEqual([6, 22, 38, 54]);

      // Matrix transpose
      const y2 = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      onTestFinished(() => y2.dispose());
      const transposeExpected = [
        [1, 4],
        [2, 5],
        [3, 6],
      ];
      expect(np.einsum("ij->ji", y2.ref).js()).toEqual(transposeExpected);
      expect(np.einsum("ji", y2.ref).js()).toEqual(transposeExpected);
      expect(np.einsum(y2.ref, [1, 0]).js()).toEqual(transposeExpected);
      expect(np.einsum(y2.ref, [0, 1], [1, 0]).js()).toEqual(transposeExpected);

      // Matrix diagonal
      expect(np.einsum("ii->i", M.ref).js()).toEqual([0, 5, 10, 15]);

      // Matrix trace
      expect(np.einsum("ii", M.ref).js()).toEqual(30);

      // Tensor products
      const tx = np.arange(30).reshape([2, 3, 5]);
      const ty = np.arange(60).reshape([3, 4, 5]);
      onTestFinished(() => {
        tx.dispose();
        ty.dispose();
      });
      const tensorExpected = [
        [3340, 3865, 4390, 4915],
        [8290, 9940, 11590, 13240],
      ];
      expect(np.einsum("ijk,jlk->il", tx.ref, ty.ref).js()).toEqual(
        tensorExpected,
      );
      expect(np.einsum("ijk,jlk", tx.ref, ty.ref).js()).toEqual(tensorExpected);
      expect(
        np.einsum(tx.ref, [0, 1, 2], ty.ref, [1, 3, 2], [0, 3]).js(),
      ).toEqual(tensorExpected);
      expect(np.einsum(tx.ref, [0, 1, 2], ty.ref, [1, 3, 2]).js()).toEqual(
        tensorExpected,
      );

      // Chained dot products
      const w = np.arange(5, 9).reshape([2, 2]);
      const cx = np.arange(6).reshape([2, 3]);
      const cy = np.arange(-2, 4).reshape([3, 2]);
      const z = np.array([
        [2, 4, 6],
        [3, 5, 7],
      ]);
      onTestFinished(() => {
        w.dispose();
        cx.dispose();
        cy.dispose();
        z.dispose();
      });
      const chainedExpected = [
        [481, 831, 1181],
        [651, 1125, 1599],
      ];
      expect(
        np.einsum("ij,jk,kl,lm->im", w.ref, cx.ref, cy.ref, z.ref).js(),
      ).toEqual(chainedExpected);
      expect(
        np
          .einsum(w.ref, [0, 1], cx.ref, [1, 2], cy.ref, [2, 3], z.ref, [3, 4])
          .js(),
      ).toEqual(chainedExpected);
    });

    test("shape tests", () => {
      const checkEinsumShapes = async (expr: string, ...shapes: number[][]) => {
        const result = np.einsum(
          expr,
          ...shapes.slice(0, -1).map((shape) => np.zeros(shape)),
        );
        expect(result.shape).toEqual(shapes[shapes.length - 1]);
        result.dispose();
      };

      // Tests without ellipsis
      checkEinsumShapes("", [], []);
      checkEinsumShapes("i,i->", [3], [3], []);
      checkEinsumShapes("ijj->i", [2, 3, 3], [2]);
      checkEinsumShapes("i,i->i", [3], [3], [3]);
      checkEinsumShapes("ij,j->i", [2, 3], [3], [2]);
      checkEinsumShapes("ij,ji", [3, 4], [4, 3], []);
      checkEinsumShapes("ij,jk", [2, 3], [3, 4], [2, 4]);
      checkEinsumShapes("ij,jk->ki", [2, 3], [3, 4], [4, 2]);
      checkEinsumShapes("abc,cde->abde", [2, 3, 4], [4, 5, 6], [2, 3, 5, 6]);
      checkEinsumShapes(
        "abcd,cdef->abef",
        [2, 3, 4, 5],
        [4, 5, 6, 7],
        [2, 3, 6, 7],
      );
      checkEinsumShapes(
        "abcd,efcd->abef",
        [2, 3, 4, 5],
        [6, 7, 4, 5],
        [2, 3, 6, 7],
      );
      checkEinsumShapes(
        "abc,bcd,efa,fab",
        [2, 3, 4],
        [3, 4, 5],
        [10, 6, 2],
        [6, 2, 3],
        [5, 10],
      );

      // Tests with ellipsis (can be in middle of indices)
      checkEinsumShapes("...", [5, 1], [5, 1]);
      checkEinsumShapes("i...", [5, 1], [1, 5]);
      checkEinsumShapes("...,...->...", [2, 3, 4], [3, 4], [2, 3, 4]);
      checkEinsumShapes("...i,i->...", [2, 3, 4], [4], [2, 3]);
      checkEinsumShapes("i,...i->...", [4], [2, 3, 4], [2, 3]);
      checkEinsumShapes("...ij,jk->...ik", [5, 2, 3], [3, 4], [5, 2, 4]);
      checkEinsumShapes(
        "...ij,...jk->...ik",
        [6, 5, 2, 3],
        [5, 3, 4],
        [6, 5, 2, 4],
      );
      checkEinsumShapes(
        "ab...cd,cd...ef->ab...ef",
        [2, 3, 4, 5, 6, 7],
        [6, 7, 8, 9],
        [2, 3, 4, 5, 8, 9],
      );

      // Tests with broadcasting dims
      checkEinsumShapes("ii->i", [3, 3], [3]);
      checkEinsumShapes("ii->i", [3, 1], [1]);
      checkEinsumShapes("i,i->i", [3], [1], [3]);
      checkEinsumShapes("i,i->i", [1], [3], [3]);
      checkEinsumShapes("ii,i->i", [3, 3], [1], [3]);
      checkEinsumShapes("ii,i->i", [1, 1], [3], [3]);
      checkEinsumShapes("ij,ij->ij", [1, 10], [5, 1], [5, 10]);
      checkEinsumShapes("...,...->...", [1, 10], [5, 1], [5, 10]);
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

  suite("jax.numpy.indices()", () => {
    test("creates dense index grid", () => {
      const grid = np.indices([2, 3]);
      expect(grid.shape).toEqual([2, 2, 3]);
      expect(grid.dtype).toBe(np.int32);
      expect(grid.js()).toEqual([
        [
          [0, 0, 0],
          [1, 1, 1],
        ],
        [
          [0, 1, 2],
          [0, 1, 2],
        ],
      ]);
    });

    test("works with a single dimension", () => {
      const grid = np.indices([4]);
      expect(grid.shape).toEqual([1, 4]);
      expect(grid.js()).toEqual([[0, 1, 2, 3]]);
    });

    test("supports sparse output", () => {
      const [row, col] = np.indices([2, 3], { sparse: true });
      expect(row.shape).toEqual([2, 1]);
      expect(col.shape).toEqual([1, 3]);
      expect(row.js()).toEqual([[0], [1]]);
      expect(col.js()).toEqual([[0, 1, 2]]);
    });

    test("supports a dynamic sparse option", () => {
      const makeIndices = (sparse: boolean) => np.indices([2, 3], { sparse });
      expect(Array.isArray(makeIndices(false))).toBe(false);
      expect(Array.isArray(makeIndices(true))).toBe(true);
    });

    test("supports dtype option", () => {
      const grid = np.indices([2, 2], { dtype: np.float32 });
      expect(grid.dtype).toBe(np.float32);
      expect(grid.js()).toEqual([
        [
          [0, 0],
          [1, 1],
        ],
        [
          [0, 1],
          [0, 1],
        ],
      ]);
    });

    test("handles empty dimensions", () => {
      const grid = np.indices([]);
      expect(grid.shape).toEqual([0]);
      expect(grid.js()).toEqual([]);
      expect(np.indices([], { sparse: true })).toEqual([]);
    });

    test("rejects invalid dimensions", () => {
      expect(() => np.indices([2, -1])).toThrow("non-negative integers");
      expect(() => np.indices([1.5])).toThrow("non-negative integers");
    });
  });

  suite("jax.numpy.fromfunction()", () => {
    test("builds a multiplication table, using the JAX docs example", () => {
      const table = np.fromfunction((i, j) => i.mul(j), [3, 6], {
        dtype: np.int32,
      });
      expect(table.dtype).toBe(np.int32);
      expect(table.js()).toEqual([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 2, 3, 4, 5],
        [0, 2, 4, 6, 8, 10],
      ]);
    });

    test("defaults to float32 indices", () => {
      const a = np.fromfunction((i, j) => i.add(j), [2, 3]);
      expect(a.dtype).toBe(np.float32);
      expect(a.js()).toEqual([
        [0, 1, 2],
        [1, 2, 3],
      ]);
    });

    test("varies the first index along the leading axis", () => {
      const a = np.fromfunction((i, j) => i.mul(10).add(j), [2, 3], {
        dtype: np.int32,
      });
      expect(a.js()).toEqual([
        [0, 1, 2],
        [10, 11, 12],
      ]);
    });

    test("works with one and three dimensions", () => {
      const a = np.fromfunction((i) => i.mul(2), [4], { dtype: np.int32 });
      expect(a.js()).toEqual([0, 2, 4, 6]);

      const b = np.fromfunction(
        (i, j, k) => i.mul(4).add(j.mul(2)).add(k),
        [2, 2, 2],
        { dtype: np.int32 },
      );
      expect(b.js()).toEqual([
        [
          [0, 1],
          [2, 3],
        ],
        [
          [4, 5],
          [6, 7],
        ],
      ]);
    });

    test("broadcasts results that ignore an index", () => {
      const a = np.fromfunction((i, _j) => i, [2, 3], { dtype: np.int32 });
      expect(a.js()).toEqual([
        [0, 0, 0],
        [1, 1, 1],
      ]);
    });

    test("non-scalar results have leading dimensions of shape", () => {
      const a = np.fromfunction((x) => x.add(1).mul(np.arange(3)), [2]);
      expect(a.shape).toEqual([2, 3]);
      expect(a.js()).toEqual([
        [0, 1, 2],
        [0, 2, 4],
      ]);
    });

    test("maps multiple results independently", () => {
      const [sum, product] = np.fromfunction(
        (i, j) => [i.ref.add(j.ref), i.mul(j)],
        [2, 3],
      );
      expect(sum.js()).toEqual([
        [0, 1, 2],
        [1, 2, 3],
      ]);
      expect(product.js()).toEqual([
        [0, 0, 0],
        [0, 1, 2],
      ]);
    });

    test("handles empty and zero-size shapes", () => {
      expect(np.fromfunction(() => 5, []).js()).toEqual(5);
      const a = np.fromfunction((i, j) => i.add(j), [0, 2]);
      expect(a.shape).toEqual([0, 2]);
    });

    test("rejects invalid shapes", () => {
      expect(() => np.fromfunction((i) => i, [-1])).toThrow(
        "non-negative integers",
      );
      expect(() => np.fromfunction((i) => i, [1.5])).toThrow(
        "non-negative integers",
      );
    });

    test("works inside jit and grad", () => {
      const f = jit(() =>
        np.fromfunction((i, j) => i.add(j), [2, 2], { dtype: np.int32 }),
      );
      expect(f().js()).toEqual([
        [0, 1],
        [1, 2],
      ]);

      // sum(x * i for i in 0..2) = 3 * x, so the gradient is 3.
      const g = grad((x: np.Array) =>
        np.fromfunction((i) => x.mul(i), [3]).sum(),
      );
      expect(g(np.array(2)).js()).toEqual(3);
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

    test("minimum of bools", () => {
      const x = np.array([true, false, true]);
      const y = np.array([false, false, true]);
      const z = np.minimum(x, y);
      expect(z.js()).toEqual([false, false, true]);
    });
  });

  suite("jax.numpy.fmin()", () => {
    test("computes element-wise minimum", () => {
      const x = np.array([1, 2, 3]);
      const y = np.array([4, 2, 0]);
      const z = np.fmin(x, y);
      expect(z.js()).toEqual([1, 2, 0]);
    });

    test("ignores NaN unless both elements are NaN", () => {
      const x = np.array([NaN, 2, NaN, -Infinity]);
      const y = np.array([5, NaN, NaN, 3]);
      const z = np.fmin(x, y);
      expect(z.js()).toEqual([5, 2, NaN, -Infinity]);
    });

    test("broadcasts inputs", () => {
      const x = np.array([
        [1, 5],
        [4, 2],
      ]);
      const z = np.fmin(x, np.array([3]));
      expect(z.js()).toEqual([
        [1, 3],
        [3, 2],
      ]);
    });

    test("works with jvp", () => {
      const x = np.array([1, 3, 3]);
      const y = np.array([4, 2, 0]);
      const [z, dz] = jvp(
        (x: np.Array, y: np.Array) => np.fmin(x, y),
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

  suite("jax.numpy.fmax()", () => {
    test("computes element-wise maximum", () => {
      const x = np.array([1, 2, 3]);
      const y = np.array([4, 2, 0]);
      const z = np.fmax(x, y);
      expect(z.js()).toEqual([4, 2, 3]);
    });

    test("broadcasts inputs", () => {
      const x = np.array([
        [1, 5],
        [4, 2],
      ]);
      const y = np.array([3, 3]);
      expect(np.fmax(x, y).js()).toEqual([
        [3, 5],
        [4, 3],
      ]);
    });

    test("ignores NaN unless both elements are NaN", () => {
      const x = np.array([1, NaN, NaN, 5]);
      const y = np.array([NaN, 3, NaN, 2]);
      const z = np.fmax(x, y);
      expect(z.js()).toEqual([1, 3, NaN, 5]);
    });

    test("handles infinities", () => {
      const x = np.array([-Infinity, Infinity, NaN]);
      const y = np.array([2, NaN, -Infinity]);
      expect(np.fmax(x, y).js()).toEqual([2, Infinity, -Infinity]);
    });

    test("works with jvp", () => {
      const x = np.array([1, 1, 3]);
      const y = np.array([4, 2, 0]);
      const [z, dz] = jvp(
        (x: np.Array, y: np.Array) => np.fmax(x, y),
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

  suite("jax.numpy.signbit()", () => {
    test("identifies negative values and signed zero", () => {
      const x = np.array([-Infinity, -3, -0, 0, 2, Infinity]);
      const result = np.signbit(x);
      expect(result.dtype).toBe(np.bool);
      expect(result.js()).toEqual([true, true, true, false, false, false]);
    });

    test("distinguishes signed zero for scalar and constant inputs", () => {
      expect(np.signbit(np.array(-0)).js()).toBe(true);
      expect(np.signbit(np.array([0, -0])).js()).toEqual([false, true]);
    });

    test("supports integer and boolean inputs", () => {
      expect(
        np.signbit(np.array([-2, 0, 3], { dtype: np.int32 })).js(),
      ).toEqual([true, false, false]);
      expect(
        np.signbit(np.array([0, 1, 0xffffffff], { dtype: np.uint32 })).js(),
      ).toEqual([false, false, false]);
      expect(np.signbit(np.array([false, true])).js()).toEqual([false, false]);
    });

    test("preserves the sign of NaN", () => {
      if (!hasStrictNumerics(device)) return;
      const bits = new Uint32Array([0xffc00000, 0x7fc00000]); // [-NaN, NaN]
      const x = np.array(new Float32Array(bits.buffer));
      expect(np.signbit(x).js()).toEqual([true, false]);
    });

    test("works with jit and vmap", () => {
      const signbitJit = jit((x: np.Array) => np.signbit(x));
      expect(signbitJit(np.array([-0, 0, -4, 4])).js()).toEqual([
        true,
        false,
        true,
        false,
      ]);

      const signbitVmap = vmap((x: np.Array) => np.signbit(x));
      expect(signbitVmap(np.array([-2, 0, 3])).js()).toEqual([
        true,
        false,
        false,
      ]);
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

  suite("jax.numpy.floorDivide()", () => {
    test("computes element-wise floor division", () => {
      const x = np.array([7, 7, -7, -7]);
      const y = np.array([3, -3, 3, -3]);
      const z = np.floorDivide(x, y);
      // floor(7/3)=2, floor(7/-3)=-3, floor(-7/3)=-3, floor(-7/-3)=2
      expect(z.js()).toEqual([2, -3, -3, 2]);
    });

    test("handles integer division that rounds toward negative infinity", () => {
      const x = np.array([5, -5, 10, -10]);
      const y = np.array([2, 2, 3, 3]);
      const z = np.floorDivide(x, y);
      // floor(5/2)=2, floor(-5/2)=-3, floor(10/3)=3, floor(-10/3)=-4
      expect(z.js()).toEqual([2, -3, 3, -4]);
    });

    test("works with scalars", () => {
      expect(np.floorDivide(7, 3).js()).toBeCloseTo(2, 5);
      expect(np.floorDivide(-7, 3).js()).toBeCloseTo(-3, 5);
    });

    test("works with int32 dtype", () => {
      const x = np.array([7, 7, -7, -7], { dtype: np.int32 });
      const y = np.array([3, -3, 3, -3], { dtype: np.int32 });
      const z = np.floorDivide(x, y);
      // Should round toward -infinity, not toward zero
      // floor(7/3)=2, floor(7/-3)=-3, floor(-7/3)=-3, floor(-7/-3)=2
      expect(z.js()).toEqual([2, -3, -3, 2]);
      expect(z.dtype).toBe(np.int32);
    });
  });

  suite("jax.numpy.fmod()", () => {
    test("computes element-wise fmod", () => {
      const x = np.array([5, 7, -9, -11]);
      const y = np.array([3, -4, 2, -3]);
      const z = np.fmod(x, y);
      expect(z.js()).toEqual([2, 3, -1, -2]);
    });

    test("gradient is correct", () => {
      const x = np.array([5, 7, -9, -11]);
      const y = np.array([3, -4, 2, -3]);
      const { x: dx, y: dy } = vmap(
        grad(({ x, y }: { x: np.Array; y: np.Array }) => np.fmod(x, y)),
      )({ x, y });
      expect(dx.js()).toEqual([1, 1, 1, 1]);
      expect(dy.js()).toEqual([
        -Math.trunc(5 / 3),
        -Math.trunc(7 / -4),
        -Math.trunc(-9 / 2),
        -Math.trunc(-11 / -3),
      ]);
    });
  });

  suite("jax.numpy.remainder()", () => {
    test("computes element-wise remainder", () => {
      const x = np.array([5, 5, -5, -5]);
      const y = np.array([3, -3, 3, -3]);
      const z = np.remainder(x, y);
      // Should follow the sign of the divisor, like Python (but unlike JS).
      expect(z.js()).toEqual([2, -1, 1, -2]);
    });

    test("remainder gradient is correct", () => {
      const x = np.array([5, 5, -5, -5]);
      const y = np.array([3, -3, 3, -3]);
      const { x: dx, y: dy } = vmap(
        grad(({ x, y }: { x: np.Array; y: np.Array }) => np.remainder(x, y)),
      )({ x, y });
      expect(dx.js()).toEqual([1, 1, 1, 1]);
      expect(dy.js()).toEqual([
        -Math.floor(5 / 3),
        -Math.floor(5 / -3),
        -Math.floor(-5 / 3),
        -Math.floor(-5 / -3),
      ]);
    });
  });

  suite("jax.numpy.divmod()", () => {
    test("returns floor division and remainder", () => {
      const x = np.array([7, 7, -7, -7]);
      const y = np.array([3, -3, 3, -3]);
      const [q, r] = np.divmod(x, y);
      // floor(7/3)=2, floor(7/-3)=-3, floor(-7/3)=-3, floor(-7/-3)=2
      expect(q.js()).toEqual([2, -3, -3, 2]);
      // remainder follows sign of divisor y
      expect(r.js()).toEqual([1, -2, 2, -1]);
    });

    test("satisfies invariant x == q*y + r", () => {
      const x = np.array([5, -5, 10, -10]);
      const y = np.array([3, 3, 4, 4]);
      const [q, r] = np.divmod(x, y.ref);
      // Verify: x == q * y + r
      const reconstructed = np.add(np.multiply(q, y), r);
      expect(reconstructed.js()).toEqual([5, -5, 10, -10]);
    });

    test("works with scalars", () => {
      const [q, r] = np.divmod(7, 3);
      expect(q.js()).toBeCloseTo(2, 5);
      expect(r.js()).toBeCloseTo(1, 5);
    });

    test("works with int32 dtype", () => {
      const x = np.array([7, -7], { dtype: np.int32 });
      const y = np.array([3, 3], { dtype: np.int32 });
      const [q, r] = np.divmod(x, y);
      expect(q.js()).toEqual([2, -3]);
      expect(r.js()).toEqual([1, 2]);
      expect(q.dtype).toBe(np.int32);
      expect(r.dtype).toBe(np.int32);
    });
  });

  suite("jax.numpy.modf()", () => {
    test("returns fractional and integral parts", () => {
      const x = np.array([3.5, -3.5, 0, 1.25]);
      const [frac, whole] = np.modf(x);
      expect(frac.js()).toBeAllclose([0.5, -0.5, 0, 0.25], { atol: 1e-6 });
      expect(whole.js()).toEqual([3, -3, 0, 1]);
    });

    test("both parts match the sign of the input", () => {
      const x = np.array([-2.75, 2.75]);
      const [frac, whole] = np.modf(x);
      expect(frac.js()).toBeAllclose([-0.75, 0.75], { atol: 1e-6 });
      expect(whole.js()).toEqual([-2, 2]);
    });

    test("satisfies invariant x == frac + whole", () => {
      const x = np.array([1.7, -4.2, 100.001, -0.5]);
      const [frac, whole] = np.modf(x.ref);
      expect(np.add(frac, whole).js()).toBeAllclose(x.js(), { atol: 1e-6 });
    });

    test("promotes integer inputs to float32", () => {
      const [frac, whole] = np.modf(np.array([5, -3], { dtype: np.int32 }));
      expect(frac.dtype).toBe(np.float32);
      expect(whole.dtype).toBe(np.float32);
      expect(frac.js()).toEqual([0, 0]);
      expect(whole.js()).toEqual([5, -3]);
    });

    test("works with scalars", () => {
      const [frac, whole] = np.modf(2.5);
      expect(frac.js()).toBeCloseTo(0.5, 5);
      expect(whole.js()).toBeCloseTo(2, 5);
    });
  });

  suite("jax.numpy.unwrap()", () => {
    test("unwraps phase jumps larger than pi, using the NumPy docs example", () => {
      const phase = np.array([
        0,
        Math.PI / 4,
        Math.PI / 2,
        (3 * Math.PI) / 4 + Math.PI,
        2 * Math.PI,
      ]);
      const result = np.unwrap(phase);
      expect(result.js()).toBeAllclose(
        [0, Math.PI / 4, Math.PI / 2, -Math.PI / 4, 0],
        { atol: 1e-5 },
      );
    });

    test("supports a custom period", () => {
      const a = np.unwrap(np.array([0, 1, 2, -1, 0]), null, -1, 4);
      expect(a.js()).toBeAllclose([0, 1, 2, 3, 4]);
      const b = np.unwrap(np.array([2, 3, 4, 5, 2, 3, 4, 5]), null, -1, 4);
      expect(b.js()).toBeAllclose([2, 3, 4, 5, 6, 7, 8, 9]);
    });

    test("larger discont preserves values", () => {
      expect(np.unwrap(np.array([0, 3.5])).js()).toBeAllclose([
        0,
        3.5 - 2 * Math.PI,
      ]);
      expect(np.unwrap(np.array([0, 3.5]), 4).js()).toBeAllclose([0, 3.5]);
    });

    test("uses the dtype-rounded period for the default discont", () => {
      // The period rounds to infinity in float32, so its half-period does too.
      const result = np.unwrap(
        np.array([0, 3e38], { dtype: np.float32 }),
        null,
        -1,
        3.5e38,
      );
      expect(result.js()).toEqual([0, Math.fround(3e38)]);
    });

    test("operates along the given axis", () => {
      const x = np.array([
        [0, 2 * Math.PI + 0.1, 4 * Math.PI + 0.2],
        [0.5, 2 * Math.PI + 0.6, 4 * Math.PI + 0.7],
      ]);
      expect(np.unwrap(x.ref).js()).toBeAllclose(
        [
          [0, 0.1, 0.2],
          [0.5, 0.6, 0.7],
        ],
        { atol: 1e-5 },
      );
      expect(np.unwrap(np.transpose(x), null, 0).js()).toBeAllclose(
        [
          [0, 0.5],
          [0.1, 0.6],
          [0.2, 0.7],
        ],
        { atol: 1e-5 },
      );
    });

    test("operates along a higher-rank interior axis", () => {
      const x = np.array([
        [
          [
            [0, 0.5],
            [2 * Math.PI + 0.1, 2 * Math.PI + 0.6],
            [4 * Math.PI + 0.2, 4 * Math.PI + 0.7],
          ],
        ],
      ]);

      expect(np.unwrap(x, null, 2).js()).toBeAllclose(
        [
          [
            [
              [0, 0.5],
              [0.1, 0.6],
              [0.2, 0.7],
            ],
          ],
        ],
        { atol: 1e-5 },
      );
    });

    test("handles half-period deltas like NumPy", () => {
      const up = np.unwrap(np.array([0, Math.PI, 2 * Math.PI]));
      expect(up.js()).toBeAllclose([0, Math.PI, 2 * Math.PI], { atol: 1e-5 });
      const down = np.unwrap(np.array([0, -Math.PI, -2 * Math.PI]));
      expect(down.js()).toBeAllclose([0, -Math.PI, -2 * Math.PI], {
        atol: 1e-5,
      });
    });

    test("promotes integers to float and handles short arrays", () => {
      const x = np.array([0, 1, 2, -1, 0], { dtype: np.int32 });
      const result = np.unwrap(x, null, -1, 4);
      expect(result.dtype).toBe(np.float32);
      expect(result.js()).toBeAllclose([0, 1, 2, 3, 4]);
      expect(np.unwrap(np.array([1.5])).js()).toBeAllclose([1.5]);
      expect(np.unwrap(np.zeros([0])).js()).toEqual([]);
    });

    test("works inside jit and grad", () => {
      const f = jit((x: np.Array) => np.unwrap(x));
      const result = f(np.array([0, Math.PI - 0.1, 2 * Math.PI + 0.5]));
      expect(result.js()).toBeAllclose([0, Math.PI - 0.1, 0.5], { atol: 1e-5 });

      // Each output is x minus a fixed constant, such as x - 0 or x - 2π,
      // so its gradient with respect to x is 1.
      const g = grad((x: np.Array) => np.unwrap(x).sum());
      expect(g(np.array([0, 3.5, 7])).js()).toEqual([1, 1, 1]);
    });
  });

  suite("jax.numpy.exp()", () => {
    test("computes element-wise exponential", () => {
      const x = np.array([-Infinity, 0, 1, 2, 3]);
      const y = np.exp(x);
      expect(y.js()).toBeAllclose([0, 1, Math.E, Math.E ** 2, Math.E ** 3]);
    });

    test("exp(-Infinity) = 0", () => {
      const x = np.exp(-Infinity);
      expect(x.js()).toEqual(0);
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

  suite("jax.numpy.expm1()", () => {
    test("remains accurate for small inputs", () => {
      const values = [
        -1, -0.1, -0.01, -1e-4, -1e-8, 0, 1e-8, 1e-4, 0.01, 0.1, 1,
      ];
      expect(np.expm1(np.array(values))).toBeAllclose(values.map(Math.expm1), {
        rtol: 2e-5,
        atol: 0,
      });
    });

    test("has the correct gradient", () => {
      const values = [-0.1, -0.001, 0, 0.001, 0.1];
      const dx = grad((x: np.Array) => np.expm1(x).sum())(np.array(values));
      expect(dx).toBeAllclose(values.map(Math.exp));
    });
  });

  suite("jax.numpy.log()", () => {
    test("computes element-wise natural logarithm", () => {
      const x = np.array([1, Math.E, Math.E ** 2]);
      const y = np.log(x);
      expect(y.js()).toBeAllclose([0, 1, 2]);
    });

    test("log(0) is -Infinity", () => {
      const x = np.log(0);
      expect(x.js()).toEqual(-Infinity);
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

  suite("jax.numpy.log1p()", () => {
    test("remains accurate for small inputs", () => {
      const values = [
        -0.9, -0.2, -0.1, -0.01, -1e-4, -1e-8, 0, 1e-8, 1e-4, 0.01, 0.1, 0.2, 1,
      ];
      expect(np.log1p(np.array(values))).toBeAllclose(values.map(Math.log1p), {
        rtol: 2e-5,
        atol: 0,
      });
    });

    test("has the correct gradient", () => {
      const values = [-0.5, -0.001, 0, 0.001, 0.5];
      const dx = grad((x: np.Array) => np.log1p(x).sum())(np.array(values));
      expect(dx).toBeAllclose(values.map((x) => 1 / (1 + x)));
    });
  });

  suite("jax.numpy.logaddexp()", () => {
    test("computes logaddexp", () => {
      const x = np.array([1, 2, 3]);
      const y = np.array([4, 5, 6]);
      const z = np.logaddexp(x, y);
      expect(z.js()).toBeAllclose([
        Math.log(Math.exp(1) + Math.exp(4)),
        Math.log(Math.exp(2) + Math.exp(5)),
        Math.log(Math.exp(3) + Math.exp(6)),
      ]);
    });

    test("avoids simple overflow", () => {
      const x = np.logaddexp2(1000, 1000);
      expect(x).toBeAllclose(1001);
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

    if (hasStrictNumerics(device)) {
      test("works with jvp", () => {
        const x = np.array([-8, -1, 0, 1, 8]);
        const [y, dy] = jvp(np.cbrt, [x], [np.ones([5])]);
        expect(y).toBeAllclose([-2, -1, 0, 1, 2]);
        expect(dy).toBeAllclose([1 / 12, 1 / 3, NaN, 1 / 3, 1 / 12], {
          equalNaN: true,
        });
      });
    }
  });

  suite("jax.numpy.power()", () => {
    test("computes element-wise power", () => {
      const x = np.array([-1, 2, 3, 4]);
      const y = np.power(x, 3);
      expect(y).toBeAllclose([-1, 8, 27, 64]);
    });

    test("multiple different exponents", () => {
      const y = np.power(3, np.array([-2, 0, 0.5, 1, 2]));
      expect(y).toBeAllclose([1 / 9, 1, Math.sqrt(3), 3, 9]);
    });

    test("works with negative numbers", () => {
      // const y = np.power(-3, np.array([-2, -1, 0, 1, 2, 3, 4, 5]));
      // expect(y).toBeAllclose([1 / 9, -1 / 3, 1, -3, 9, -27, 81, -243]);
      const z = np.power(-3, np.array([0.5, 1.5, 2.5]));
      expect(z.js()).toEqual([NaN, NaN, NaN]);
    });

    if (hasStrictNumerics(device)) {
      test("power of zero", () => {
        const y = np.power(0, np.array([-2, -1, 0, 0.5, 1, 2]));
        expect(y.js()).toEqual([Infinity, Infinity, NaN, 0, 0, 0]);
      });
    }
  });

  suite("jax.numpy.floatPower()", () => {
    test("promotes integer inputs to float", () => {
      const x = np.array([1, 2, 3, 4], { dtype: np.int32 });
      const exponent = np.array(3, { dtype: np.float32 });
      const y = np.floatPower(x, exponent);
      expect(y.dtype).toBe(np.float32);
      expect(y).toBeAllclose([1, 8, 27, 64]);
    });

    test("keeps floating-point inputs as-is", () => {
      const y = np.floatPower(np.array([1.5, 2.5]), 2);
      expect(y.dtype).toBe(np.float32);
      expect(y).toBeAllclose([2.25, 6.25]);
    });

    test("fractional exponents", () => {
      const y = np.floatPower(np.array([4, 9, 16]), 0.5);
      expect(y).toBeAllclose([2, 3, 4]);
    });

    test("negative base with non-integer exponent is NaN", () => {
      const y = np.floatPower(-3, np.array([0.5, 1.5, 2.5]));
      expect(y.js()).toEqual([NaN, NaN, NaN]);
    });

    test("works inside jit", () => {
      const f = jit((x: np.Array) => np.floatPower(x, 2));
      const y = f(np.array([1, 2, 3]));
      expect(y.dtype).toBe(np.float32);
      expect(y).toBeAllclose([1, 4, 9]);
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

    test("works with jit and a prior operation", () => {
      // See comment about `needsCleanShapePrimitives` in JIT.
      const f = jit((x: np.Array) => {
        const y = x.add(2);
        return np.pad(y, 1);
      });
      const a = np.array([1, 2, 3]);
      const b = f(a);
      expect(b.js()).toEqual([0, 3, 4, 5, 0]);
    });

    test("slices a padded lazy nested stack", () => {
      const row = (a: number, b: number) =>
        np.stack([np.array(a), np.array(b)]);
      const x = np.stack([row(1, 2), row(3, 4), row(5, 6)]);
      const y = np.pad(x.slice([0, 2]), [
        [1, 0],
        [0, 0],
      ]);
      expect(y.slice([], [1, 2]).js()).toEqual([[0], [2], [4]]);
    });

    test("pad with explicit indices", () => {
      const x = np.zeros([2, 3, 4, 5]);
      const y = np.pad(x, { 1: [0, 2], [-1]: [3, 0] });
      expect(y.shape).toEqual([2, 5, 4, 8]);
      y.dispose();
    });
  });

  suite("jax.numpy.split()", () => {
    test("splits into equal parts with integer", () => {
      const x = np.arange(6);
      const [a, b, c] = np.split(x, 3);
      expect(a.js()).toEqual([0, 1]);
      expect(b.js()).toEqual([2, 3]);
      expect(c.js()).toEqual([4, 5]);
    });

    test("splits 2D array along axis 0", () => {
      const x = np.arange(12).reshape([4, 3]);
      const [a, b] = np.split(x, 2, 0);
      expect(a.js()).toEqual([
        [0, 1, 2],
        [3, 4, 5],
      ]);
      expect(b.js()).toEqual([
        [6, 7, 8],
        [9, 10, 11],
      ]);
    });

    test("splits 2D array along axis 1", () => {
      const x = np.arange(12).reshape([3, 4]);
      const [a, b] = np.split(x, 2, 1);
      expect(a.js()).toEqual([
        [0, 1],
        [4, 5],
        [8, 9],
      ]);
      expect(b.js()).toEqual([
        [2, 3],
        [6, 7],
        [10, 11],
      ]);
    });

    test("splits at indices", () => {
      const x = np.arange(10);
      const [a, b, c] = np.split(x, [3, 7]);
      expect(a.js()).toEqual([0, 1, 2]);
      expect(b.js()).toEqual([3, 4, 5, 6]);
      expect(c.js()).toEqual([7, 8, 9]);
    });

    test("splits at indices with empty sections", () => {
      const x = np.arange(5);
      const [a, b, c, d] = np.split(x, [0, 0, 3]);
      expect(a.js()).toEqual([]);
      expect(b.js()).toEqual([]);
      expect(c.js()).toEqual([0, 1, 2]);
      expect(d.js()).toEqual([3, 4]);
    });

    test("throws on uneven split", () => {
      const x = np.arange(5);
      expect(() => np.split(x, 2)).toThrow(Error);
      expect(() => np.split(x, 3)).toThrow(Error);
    });

    test("works with negative axis", () => {
      const x = np.arange(12).reshape([3, 4]);
      const [a, b] = np.split(x, 2, -1);
      expect(a.js()).toEqual([
        [0, 1],
        [4, 5],
        [8, 9],
      ]);
      expect(b.js()).toEqual([
        [2, 3],
        [6, 7],
        [10, 11],
      ]);
    });

    test("works with grad", () => {
      const x = np.arange(6).astype(np.float32);
      const f = (x: np.Array) => {
        const [a, b] = np.split(x, 2);
        return a.sum().add(b.mul(2).sum());
      };
      const dx = grad(f)(x);
      expect(dx.js()).toEqual([1, 1, 1, 2, 2, 2]);
    });

    test("works inside jit", () => {
      const f = jit((x: np.Array) => {
        const [a, b] = np.split(x, 2);
        return a.add(b);
      });
      const x = np.arange(6);
      const y = f(x);
      expect(y.js()).toEqual([3, 5, 7]);
    });

    test("splits an array into 20 parts", () => {
      const x = np.arange(20);
      for (const [i, a] of np.split(x, 20).entries()) {
        expect(a.js()).toEqual([i]);
      }
    });
  });

  suite("jax.numpy.arraySplit()", () => {
    test("splits an array into uneven parts", () => {
      const x = np.arange(8);
      const [a, b, c] = np.arraySplit(x, 3);
      expect(a.js()).toEqual([0, 1, 2]);
      expect(b.js()).toEqual([3, 4, 5]);
      expect(c.js()).toEqual([6, 7]);
    });

    test("returns empty trailing parts when sections exceed axis size", () => {
      const x = np.arange(3);
      const parts = np.arraySplit(x, 5);
      expect(parts.map((part) => part.js())).toEqual([[0], [1], [2], [], []]);
    });

    test("splits along a negative axis", () => {
      const x = np.arange(10).reshape([2, 5]);
      const [a, b, c] = np.arraySplit(x, 3, -1);
      expect(a.js()).toEqual([
        [0, 1],
        [5, 6],
      ]);
      expect(b.js()).toEqual([
        [2, 3],
        [7, 8],
      ]);
      expect(c.js()).toEqual([[4], [9]]);
    });

    test("supports explicit split indices", () => {
      const x = np.arange(7);
      const [a, b, c] = np.arraySplit(x, [2, 5]);
      expect(a.js()).toEqual([0, 1]);
      expect(b.js()).toEqual([2, 3, 4]);
      expect(c.js()).toEqual([5, 6]);
    });

    test("supports an empty list of split indices", () => {
      const x = np.arange(5);
      const [a] = np.arraySplit(x, []);
      expect(a.js()).toEqual([0, 1, 2, 3, 4]);
    });

    test("works inside jit", () => {
      const f = jit((x: np.Array) => {
        const [a, b, c] = np.arraySplit(x, 3);
        return np.concatenate([c, b, a]);
      });
      expect(f(np.arange(8)).js()).toEqual([6, 7, 3, 4, 5, 0, 1, 2]);
    });
  });

  suite("jax.numpy.unstack()", () => {
    test("unstacks along the first axis by default", () => {
      const x = np.arange(6).reshape([3, 2]);
      const [a, b, c] = np.unstack(x);
      expect(a.js()).toEqual([0, 1]);
      expect(b.js()).toEqual([2, 3]);
      expect(c.js()).toEqual([4, 5]);
    });

    test("unstacks along a negative axis", () => {
      const x = np.arange(6).reshape([3, 2]);
      const [a, b] = np.unstack(x, -1);
      expect(a.js()).toEqual([0, 2, 4]);
      expect(b.js()).toEqual([1, 3, 5]);
    });

    test("unstacks a 1D array into scalars", () => {
      const x = np.array([5, 7, 9]);
      const parts = np.unstack(x);
      expect(parts.map((part) => part.shape)).toEqual([[], [], []]);
      expect(parts.map((part) => part.js())).toEqual([5, 7, 9]);
    });

    test("is the inverse of stack", () => {
      const x = np.arange(12).reshape([2, 3, 2]);
      const y = np.stack(np.unstack(x.ref, 1), 1);
      expect(y.js()).toEqual(x.js());
    });

    test("throws on scalar input", () => {
      expect(() => np.unstack(5)).toThrow(Error);
    });

    test("returns an empty list for an empty axis", () => {
      const x = np.zeros([0, 3]);
      expect(np.unstack(x)).toEqual([]);
    });

    test("works inside jit", () => {
      const f = jit((x: np.Array) => {
        const [a, b, c] = np.unstack(x);
        return np.stack([c, b, a]);
      });
      expect(f(np.arange(6).reshape([3, 2])).js()).toEqual([
        [4, 5],
        [2, 3],
        [0, 1],
      ]);
    });
  });

  suite("jax.numpy.dsplit()", () => {
    test("splits a 3D array along the depth axis", () => {
      const x = np.arange(16).reshape([2, 2, 4]);
      const [a, b] = np.dsplit(x, 2);
      expect(a.js()).toEqual([
        [
          [0, 1],
          [4, 5],
        ],
        [
          [8, 9],
          [12, 13],
        ],
      ]);
      expect(b.js()).toEqual([
        [
          [2, 3],
          [6, 7],
        ],
        [
          [10, 11],
          [14, 15],
        ],
      ]);
    });

    test("supports explicit split indices", () => {
      const x = np.arange(8).reshape([1, 2, 4]);
      const [a, b, c] = np.dsplit(x, [1, 3]);
      expect(a.js()).toEqual([[[0], [4]]]);
      expect(b.js()).toEqual([
        [
          [1, 2],
          [5, 6],
        ],
      ]);
      expect(c.js()).toEqual([[[3], [7]]]);
    });

    test("throws on arrays with fewer than 3 dimensions", () => {
      const x = np.arange(6).reshape([2, 3]);
      expect(() => np.dsplit(x, 3)).toThrow(
        "dsplit only works on arrays of 3 or more dimensions",
      );
    });

    test("works inside jit", () => {
      const f = jit((x: np.Array) => {
        const [a, b] = np.dsplit(x, 2);
        return np.concatenate([b, a], 2);
      });
      const y = f(np.arange(8).reshape([1, 2, 4]));
      expect(y.js()).toEqual([
        [
          [2, 3, 0, 1],
          [6, 7, 4, 5],
        ],
      ]);
    });
  });

  suite("jax.numpy.hsplit()", () => {
    test("splits a 2D array along columns", () => {
      const x = np.arange(12).reshape([3, 4]);
      const [a, b] = np.hsplit(x, 2);
      expect(a.js()).toEqual([
        [0, 1],
        [4, 5],
        [8, 9],
      ]);
      expect(b.js()).toEqual([
        [2, 3],
        [6, 7],
        [10, 11],
      ]);
    });

    test("splits a 1D array along axis 0", () => {
      const x = np.arange(6);
      const [a, b, c] = np.hsplit(x, 3);
      expect(a.js()).toEqual([0, 1]);
      expect(b.js()).toEqual([2, 3]);
      expect(c.js()).toEqual([4, 5]);
    });

    test("splits a 3D array along axis 1", () => {
      const x = np.arange(8).reshape([2, 2, 2]);
      const [a, b] = np.hsplit(x, 2);
      expect(a.js()).toEqual([[[0, 1]], [[4, 5]]]);
      expect(b.js()).toEqual([[[2, 3]], [[6, 7]]]);
    });

    test("supports explicit split indices", () => {
      const x = np.arange(12).reshape([2, 6]);
      const [a, b, c] = np.hsplit(x, [1, 4]);
      expect(a.js()).toEqual([[0], [6]]);
      expect(b.js()).toEqual([
        [1, 2, 3],
        [7, 8, 9],
      ]);
      expect(c.js()).toEqual([
        [4, 5],
        [10, 11],
      ]);
    });

    test("throws on uneven split", () => {
      const x = np.arange(10).reshape([2, 5]);
      expect(() => np.hsplit(x, 2)).toThrow(Error);
    });

    test("throws on scalar input", () => {
      const x = np.array(1);
      expect(() => np.hsplit(x, 1)).toThrow(
        "hsplit only works on arrays of 1 or more dimensions",
      );
    });

    test("works inside jit", () => {
      const f = jit((x: np.Array) => {
        const [a, b] = np.hsplit(x, 2);
        return a.add(b);
      });
      const x = np.arange(8).reshape([2, 4]);
      expect(f(x).js()).toEqual([
        [2, 4],
        [10, 12],
      ]);
    });
  });

  suite("jax.numpy.vsplit()", () => {
    test("splits a 2D array into equal parts", () => {
      const x = np.arange(12).reshape([4, 3]);
      const [a, b] = np.vsplit(x, 2);
      expect(a.js()).toEqual([
        [0, 1, 2],
        [3, 4, 5],
      ]);
      expect(b.js()).toEqual([
        [6, 7, 8],
        [9, 10, 11],
      ]);
    });

    test("splits a 1D array along axis 0", () => {
      const x = np.array([1, 2, 3, 4, 5, 6]);
      const [a, b] = np.vsplit(x, 2);
      expect(a.js()).toEqual([1, 2, 3]);
      expect(b.js()).toEqual([4, 5, 6]);
    });

    test("splits at indices", () => {
      const x = np.arange(12).reshape([4, 3]);
      const [a, b, c] = np.vsplit(x, [1, 3]);
      expect(a.js()).toEqual([[0, 1, 2]]);
      expect(b.js()).toEqual([
        [3, 4, 5],
        [6, 7, 8],
      ]);
      expect(c.js()).toEqual([[9, 10, 11]]);
    });

    test("throws on uneven split", () => {
      const x = np.arange(15).reshape([5, 3]);
      expect(() => np.vsplit(x, 2)).toThrow(Error);
    });

    test("works inside jit", () => {
      const f = jit((x: np.Array) => {
        const [a, b] = np.vsplit(x, 2);
        return a.add(b);
      });
      const x = np.arange(8).reshape([4, 2]);
      expect(f(x).js()).toEqual([
        [4, 6],
        [8, 10],
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

    test("concatenate works in jit", () => {
      const f = jit(np.concatenate);
      const c = f([np.flip(np.array([1, 2])), np.array([3, 4]), np.array([5])]);
      expect(c.js()).toEqual([2, 1, 3, 4, 5]);
    });
  });

  suite("jax.numpy.roll()", () => {
    test("rolls a 1D array", () => {
      const x = np.array([1, 2, 3, 4, 5]);
      const y = np.roll(x, 2);
      expect(y.js()).toEqual([4, 5, 1, 2, 3]);
    });

    test("rolls a 2D array with/out axis", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const y1 = np.roll(x.ref, 1);
      expect(y1.js()).toEqual([
        [6, 1, 2],
        [3, 4, 5],
      ]);
      const y2 = np.roll(x.ref, 1, 0);
      expect(y2.js()).toEqual([
        [4, 5, 6],
        [1, 2, 3],
      ]);
      const y3 = np.roll(x.ref, -1, 1);
      expect(y3.js()).toEqual([
        [2, 3, 1],
        [5, 6, 4],
      ]);
      const y4 = np.roll(x, [2, 1], [1, 0]);
      expect(y4.js()).toEqual([
        [5, 6, 4],
        [2, 3, 1],
      ]);
    });

    test("rolls with large shifts", () => {
      const x = np.array([1, 2, 3, 4, 5]);
      const y = np.roll(x, 20);
      expect(y.js()).toEqual([1, 2, 3, 4, 5]);
    });

    test("rolls a 3x3 lazy nested stack over two axes", () => {
      const row = (a: number, b: number, c: number) =>
        np.stack([np.array(a), np.array(b), np.array(c)]);
      const x = np.stack([row(1, 2, 3), row(4, 5, 6), row(7, 8, 9)]);
      const y = np.roll(np.roll(x, 1, 0), 1, 1);
      expect(y.js()).toEqual([
        [9, 7, 8],
        [3, 1, 2],
        [6, 4, 5],
      ]);
    });
  });

  suite("jax.numpy.vander()", () => {
    test("builds a Vandermonde matrix", () => {
      const x = np.array([0, 2, 3]);
      const y = np.vander(x);
      expect(y).toBeAllclose([
        [0, 0, 1],
        [4, 2, 1],
        [9, 3, 1],
      ]);
    });

    test("supports explicit column count and increasing powers", () => {
      const x = np.array([1, 2, 3]);
      const y = np.vander(x, { n: 4, increasing: true });
      expect(y).toBeAllclose([
        [1, 1, 1, 1],
        [1, 2, 4, 8],
        [1, 3, 9, 27],
      ]);
    });

    test("supports zero columns", () => {
      const x = np.array([1, 2, 3]);
      const y = np.vander(x, { n: 0 });
      expect(y.shape).toEqual([3, 0]);
      y.dispose();
    });

    test("rejects non-vector inputs", () => {
      expect(() => np.vander(np.ones([2, 2]))).toThrow(
        "vander: input must be 1D",
      );
    });
  });

  suite("jax.numpy.polyval()", () => {
    test("evaluates a polynomial at scalar and array points", () => {
      const p = np.array([3, 0, 1]); // 3x^2 + 1
      expect(np.polyval(p.ref, np.array(5)).js()).toEqual(76);
      expect(np.polyval(p, np.array([0, 1, 2]))).toBeAllclose([1, 4, 13]);
    });

    test("preserves promoted input dtype", () => {
      const y = np.polyval(
        np.array([1, 2], { dtype: np.int32 }),
        np.array([3, 4], { dtype: np.int32 }),
      );
      expect(y.dtype).toBe(np.int32);
      expect(y).toBeAllclose([5, 6]);
      const scalar = np.polyval(np.array([2, 5, 1], { dtype: np.int32 }), 3);
      expect(scalar.dtype).toBe(np.int32);
      expect(scalar.js()).toEqual(34);
      const mixed = np.polyval(
        np.array([1, 2], { dtype: np.int32 }),
        np.array([0.5, 1.5]),
      );
      expect(mixed.dtype).toBe(np.float32);
      expect(mixed).toBeAllclose([2.5, 3.5]);
    });

    test("handles empty and constant coefficients", () => {
      const y = np.polyval(np.zeros([0]), np.array([1.5, 2.5]));
      expect(y.js()).toEqual([0, 0]);
      const c = np.polyval(np.array([7]), np.ones([2, 2]));
      expect(c.js()).toEqual([
        [7, 7],
        [7, 7],
      ]);
    });

    test("supports batched coefficients", () => {
      const p = np.array([
        [1, 2],
        [0, 3],
        [4, 5],
      ]);
      expect(np.polyval(p, np.array([2, 3]))).toBeAllclose([8, 32]);

      const empty = np.polyval(np.zeros([0, 2, 1]), np.ones([3]));
      expect(empty.shape).toEqual([2, 3]);
      expect(empty.js()).toEqual([
        [0, 0, 0],
        [0, 0, 0],
      ]);
    });

    test("supports grad and jit", () => {
      const f = (x: np.Array) => np.polyval(np.array([3, 0, 1]), x);
      const dx = grad(f)(np.array(2.0));
      expect(dx.js()).toEqual(12); // d/dx (3x^2 + 1) = 6x
      const y = jit(f)(np.array(3.0));
      expect(y.js()).toEqual(28);
    });

    test("rejects scalar coefficients", () => {
      expect(() => np.polyval(np.array(1), np.array(1))).toThrow(
        "polyval: coefficients must have at least one dimension",
      );
    });
  });

  suite("jax.numpy.polyadd()", () => {
    test("adds polynomials of equal length", () => {
      const a1 = np.array([1, 2, 3]);
      const a2 = np.array([4, 5, 6]);
      expect(np.polyadd(a1, a2).js()).toEqual([5, 7, 9]);
    });

    test("pads the shorter polynomial with leading zeros", () => {
      const a1 = np.array([1, 2, 3, 4]);
      const a2 = np.array([10, 20]);
      expect(np.polyadd(a1.ref, a2.ref).js()).toEqual([1, 2, 13, 24]);
      expect(np.polyadd(a2, a1).js()).toEqual([1, 2, 13, 24]);
    });

    test("supports empty coefficient arrays", () => {
      const a1 = np.array([1, 2]);
      const a2 = np.zeros([0]);
      expect(np.polyadd(a1, a2).js()).toEqual([1, 2]);
    });

    test("promotes dtypes", () => {
      const a1 = np.array([1, 2, 3]);
      const a2 = np.array([0.5, 1.5]);
      const y = np.polyadd(a1, a2);
      expect(y.dtype).toBe(np.float32);
      expect(y).toBeAllclose([1, 2.5, 4.5]);
    });

    test("supports batched polynomial coefficients", () => {
      const a1 = np.array([[2, 3, 1]]);
      const a2 = np.array([
        [5, 7, 3],
        [8, 2, 6],
      ]);
      expect(np.polyadd(a1, a2).js()).toEqual([
        [5, 7, 3],
        [10, 5, 7],
      ]);

      const batched = np.array([
        [5, 7, 9],
        [8, 6, 4],
      ]);
      expect(np.polyadd(batched, np.array([2])).js()).toEqual([
        [5, 7, 9],
        [10, 8, 6],
      ]);
    });

    test("rejects incompatible coefficient batches", () => {
      expect(() =>
        np.polyadd(
          np.array([1, 3, 5]),
          np.array([
            [5, 7, 9],
            [8, 6, 4],
          ]),
        ),
      ).toThrow();
    });

    test("rejects scalar inputs", () => {
      expect(() => np.polyadd(np.array(1), np.ones([2]))).toThrow(
        "polyadd: both inputs must be at least 1D",
      );
    });

    test("works inside jit and grad", () => {
      const f = jit((a: np.Array, b: np.Array) => np.polyadd(a, b));
      expect(f(np.array([1, 2, 3]), np.array([4, 5])).js()).toEqual([1, 6, 8]);

      const g = (a: np.Array) =>
        np
          .polyadd(a, np.array([1, 1, 1, 1]))
          .mul(np.array([1, 2, 3, 4]))
          .sum();
      const da = grad(g)(np.array([1, 2], { dtype: np.float32 }));
      expect(da.js()).toEqual([3, 4]);
    });
  });

  suite("jax.numpy.polysub()", () => {
    test("subtracts polynomials of equal length", () => {
      const a1 = np.array([1, 2, 3]);
      const a2 = np.array([4, 5, 6]);
      expect(np.polysub(a1, a2).js()).toEqual([-3, -3, -3]);
    });

    test("pads the shorter polynomial with leading zeros", () => {
      const a1 = np.array([1, 2, 3, 4]);
      const a2 = np.array([10, 20]);
      expect(np.polysub(a1.ref, a2.ref).js()).toEqual([1, 2, -7, -16]);
      expect(np.polysub(a2, a1).js()).toEqual([-1, -2, 7, 16]);
    });

    test("supports empty coefficient arrays", () => {
      const a1 = np.array([1, 2]);
      const a2 = np.zeros([0]);
      expect(np.polysub(a1.ref, a2.ref).js()).toEqual([1, 2]);
      expect(np.polysub(a2, a1).js()).toEqual([-1, -2]);
    });

    test("promotes dtypes", () => {
      const a1 = np.array([1, 2, 3]);
      const a2 = np.array([0.5, 1.5]);
      const y = np.polysub(a1, a2);
      expect(y.dtype).toBe(np.float32);
      expect(y).toBeAllclose([1, 1.5, 1.5]);
    });

    test("supports batched polynomial coefficients", () => {
      const a1 = np.array([[2, 3, 1]]);
      const a2 = np.array([
        [5, 7, 3],
        [8, 2, 6],
      ]);
      expect(np.polysub(a1, a2).js()).toEqual([
        [-5, -7, -3],
        [-6, 1, -5],
      ]);

      const batched = np.array([
        [5, 7, 9],
        [8, 6, 4],
      ]);
      expect(np.polysub(batched, np.array([2])).js()).toEqual([
        [5, 7, 9],
        [6, 4, 2],
      ]);
    });

    test("rejects incompatible coefficient batches", () => {
      expect(() =>
        np.polysub(
          np.array([1, 3, 5]),
          np.array([
            [5, 7, 9],
            [8, 6, 4],
          ]),
        ),
      ).toThrow();
    });

    test("rejects scalar inputs", () => {
      expect(() => np.polysub(np.array(1), np.ones([2]))).toThrow(
        "polysub: both inputs must be at least 1D",
      );
    });

    test("works inside jit and grad", () => {
      const f = jit((a: np.Array, b: np.Array) => np.polysub(a, b));
      expect(f(np.array([1, 2, 3]), np.array([4, 5])).js()).toEqual([
        1, -2, -2,
      ]);

      const g = (a: np.Array) =>
        np
          .polysub(a, np.array([1, 1, 1, 1]))
          .mul(np.array([1, 2, 3, 4]))
          .sum();
      const da = grad(g)(np.array([1, 2], { dtype: np.float32 }));
      expect(da.js()).toEqual([3, 4]);
    });
  });

  suite("jax.numpy.polyder()", () => {
    test("computes the first derivative", () => {
      const p = np.array([3, 0, 1]); // 3x^2 + 1
      expect(np.polyder(p).js()).toEqual([6, 0]);
      expect(np.polyder(np.array([1, 2, 3, 4])).js()).toEqual([3, 4, 3]);
    });

    test("computes higher-order derivatives", () => {
      const p = np.array([1, 2, 3, 4]); // x^3 + 2x^2 + 3x + 4
      expect(np.polyder(p.ref, 2).js()).toEqual([6, 4]);
      expect(np.polyder(p, 3).js()).toEqual([6]);
    });

    test("promotes integer coefficients to float", () => {
      const y = np.polyder(np.array([2, 4, 6], { dtype: np.int32 }));
      expect(y.dtype).toBe(np.float32);
      expect(y.js()).toEqual([4, 4]);

      const same = np.polyder(np.array([1, 2, 3], { dtype: np.int32 }), 0);
      expect(same.dtype).toBe(np.float32);
      expect(same.js()).toEqual([1, 2, 3]);
    });

    test("returns a promoted scalar unchanged for order zero", () => {
      const y = np.polyder(np.array(3, { dtype: np.int32 }), 0);
      expect(y.dtype).toBe(np.float32);
      expect(y.js()).toBe(3);
    });

    test("returns an empty polynomial when order reaches the length", () => {
      const y = np.polyder(np.array([5]));
      expect(y.shape).toEqual([0]);
      expect(y.js()).toEqual([]);
      expect(np.polyder(np.array([1, 2]), 5).shape).toEqual([0]);
      expect(np.polyder(np.zeros([0])).shape).toEqual([0]);
    });

    test("supports batched polynomial coefficients", () => {
      const p = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
      ]);
      expect(np.polyder(p).js()).toEqual([
        [2, 20],
        [2, 20],
      ]);
    });

    test("matches the derivative of polyval", () => {
      // d/dx (2x^3 - 3x^2 + 0.5x + 1) at x = 1.5 is 5.
      const p = np.array([2, -3, 0.5, 1]);
      const dx = grad((x: np.Array) => np.polyval(p.ref, x))(np.array(1.5));
      expect(dx).toBeAllclose(5);
      expect(np.polyval(np.polyder(p), np.array(1.5))).toBeAllclose(5);
    });

    test("works inside jit and grad", () => {
      const f = jit((p: np.Array) => np.polyder(p, 2));
      expect(f(np.array([1, 2, 3, 4])).js()).toEqual([6, 4]);

      const g = (p: np.Array) =>
        np
          .polyder(p)
          .mul(np.array([1, 2]))
          .sum();
      const dp = grad(g)(np.array([1, 2, 3], { dtype: np.float32 }));
      expect(dp.js()).toEqual([2, 2, 0]);
    });

    test("rejects scalar coefficients and invalid orders", () => {
      expect(() => np.polyder(np.array(1))).toThrow(
        "polyder: coefficients must have at least one dimension",
      );
      expect(() => np.polyder(np.array([1, 2]), -1)).toThrow(
        "polyder: order of derivative must be a non-negative integer",
      );
      expect(() => np.polyder(np.array([1, 2]), 1.5)).toThrow(
        "polyder: order of derivative must be a non-negative integer",
      );
    });
  });

  suite("jax.numpy.polymul()", () => {
    test("multiplies polynomials of equal length", () => {
      const a1 = np.array([2, 1, 0]);
      const a2 = np.array([0, 5, 3]);
      expect(np.polymul(a1, a2).js()).toEqual([0, 10, 11, 3, 0]);
    });

    test("multiplies polynomials of different lengths", () => {
      const a1 = np.array([1, 2]);
      const a2 = np.array([3, 4, 5]);
      expect(np.polymul(a1.ref, a2.ref).js()).toEqual([3, 10, 13, 10]);
      expect(np.polymul(a2, a1).js()).toEqual([3, 10, 13, 10]);
    });

    test("multiplies constant polynomials", () => {
      const a1 = np.array([2]);
      const a2 = np.array([3]);
      expect(np.polymul(a1, a2).js()).toEqual([6]);
    });

    test("promotes inputs to floating point", () => {
      const a1 = np.array([1, 2, 3]);
      const a2 = np.array([4, 5, 6]);
      const y = np.polymul(a1, a2);
      expect(y.dtype).toBe(np.float32);
      expect(y).toBeAllclose([4, 13, 28, 27, 18]);
    });

    test("treats empty coefficient arrays as zero", () => {
      const a1 = np.zeros([0]);
      const a2 = np.array([1, 2]);
      expect(np.polymul(a1.ref, a2.ref).js()).toEqual([0, 0]);
      expect(np.polymul(a2, a1.ref).js()).toEqual([0, 0]);
      expect(np.polymul(a1.ref, a1).js()).toEqual([0]);
    });

    test("rejects scalar and batched inputs", () => {
      expect(() => np.polymul(np.array(1), np.ones([2]))).toThrow(
        "polymul: both inputs must be 1D arrays",
      );
      expect(() => np.polymul(np.ones([2, 2]), np.ones([2]))).toThrow(
        "polymul: both inputs must be 1D arrays",
      );
    });

    test("works inside jit and grad", () => {
      const f = jit((a: np.Array, b: np.Array) => np.polymul(a, b));
      expect(f(np.array([1, 2, 3]), np.array([4, 5])).js()).toEqual([
        4, 13, 22, 15,
      ]);

      const g = (a: np.Array) =>
        np
          .polymul(a, np.array([1, 2]))
          .mul(np.array([1, 2, 3]))
          .sum();
      const da = grad(g)(np.array([1, 2], { dtype: np.float32 }));
      expect(da.js()).toEqual([5, 8]);
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

  suite("jax.numpy.sinc()", () => {
    test("sinc(0) = 1", () => {
      expect(np.sinc(0).js()).toBeCloseTo(1, 5);
    });

    test("sinc at integer values is 0", () => {
      // sinc(n) = sin(πn) / (πn) = 0 for non-zero integers
      const x = np.array([1, 2, 3, -1, -2, -3]);
      const result: number[] = np.sinc(x).js();
      for (const val of result) {
        expect(val).toBeCloseTo(0, 5);
      }
    });

    test("sinc at 0.5", () => {
      // sinc(0.5) = sin(π/2) / (π/2) = 1 / (π/2) = 2/π
      expect(np.sinc(0.5).js()).toBeCloseTo(2 / Math.PI, 5);
    });

    test("sinc is symmetric", () => {
      const x = np.array([0.1, 0.5, 1.5, 2.5]);
      const negX = np.array([-0.1, -0.5, -1.5, -2.5]);
      expect(np.sinc(x).js()).toBeAllclose(np.sinc(negX).js());
    });

    test("sinc on array", () => {
      const x = np.array([0, 0.5, 1]);
      const expected = [1, 2 / Math.PI, 0];
      expect(np.sinc(x).js()).toBeAllclose(expected, { atol: 2e-7 });
    });
  });

  suite("jax.numpy.blackman()", () => {
    test("blackman(5) matches reference values", () => {
      const expected = [0, 0.34, 1, 0.34, 0];
      expect(np.blackman(5).js()).toBeAllclose(expected, { atol: 1e-6 });
    });

    test("blackman(10) matches reference values", () => {
      const expected = [
        -1.38777878e-17, 5.08696327e-2, 2.58000502e-1, 6.3e-1, 9.51129866e-1,
        9.51129866e-1, 6.3e-1, 2.58000502e-1, 5.08696327e-2, -1.38777878e-17,
      ];
      expect(np.blackman(10).js()).toBeAllclose(expected, { atol: 1e-6 });
    });

    test("blackman(1) returns [1]", () => {
      expect(np.blackman(1).js()).toEqual([1]);
    });

    test("blackman(0) returns an empty array", () => {
      expect(np.blackman(0).js()).toEqual([]);
    });

    test("rejects invalid window sizes", () => {
      expect(() => np.blackman(-1)).toThrow(/non-negative integer/);
      expect(() => np.blackman(0.5)).toThrow(/non-negative integer/);
    });
  });

  suite("jax.numpy.atan()", () => {
    const numDigits = hasStrictNumerics(device) ? 5 : 3;

    test("arctan values", () => {
      const vals = [-1000, -100, -10, -1, 0, 1, 10, 100, 1000, Infinity];
      const atanvals: number[] = np.atan(np.array(vals)).js();
      for (let i = 0; i < vals.length; i++) {
        expect(atanvals[i]).toBeCloseTo(Math.atan(vals[i]), numDigits);
      }
    });

    test("arcsin and arccos values", () => {
      const vals = [-1, -0.7, 0, 0.5, 1];
      const asinvals: number[] = np.asin(np.array(vals)).js();
      const acosvals: number[] = np.acos(np.array(vals)).js();
      for (let i = 0; i < vals.length; i++) {
        expect(asinvals[i]).toBeCloseTo(Math.asin(vals[i]), numDigits);
        expect(acosvals[i]).toBeCloseTo(Math.acos(vals[i]), numDigits);
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
    const numDigits = hasStrictNumerics(device) ? 5 : 3;

    test("arctan2 values", () => {
      // Test all four quadrants and special cases with various values
      const y = [3, 5, -7, -2, 4, -6, 0, 0, 1.5, -2.5];
      const x = [4, -2, -3, 8, 0, 0, 5, -9, 1.5, -2.5];
      const result: number[] = np.atan2(np.array(y), np.array(x)).js();
      for (let i = 0; i < y.length; i++) {
        expect(result[i]).toBeCloseTo(Math.atan2(y[i], x[i]), numDigits);
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

  suite("jax.numpy.cov()", () => {
    test("computes covariance matrix", () => {
      const x = np.array([
        [0, 1, 2],
        [0, 1, 2],
      ]);
      const cov1 = np.cov(x);
      expect(cov1.js()).toBeAllclose([
        [1, 1],
        [1, 1],
      ]);
    });

    test("computes covariance matrix for anti-correlated data", () => {
      const x = np.array([
        [-1, 0, 1],
        [1, 0, -1],
      ]);
      const cov2 = np.cov(x);
      expect(cov2.js()).toBeAllclose([
        [1, -1],
        [-1, 1],
      ]);
    });

    test("computes covariance matrix from separate arrays", () => {
      const x = np.array([-1, 0, 1]);
      const y = np.array([1, 0, -1]);
      const cov3 = np.cov(x, y);
      expect(cov3.js()).toBeAllclose([
        [1, -1],
        [-1, 1],
      ]);
    });
  });

  suite("jax.numpy.isnan()", () => {
    test("identify special values", () => {
      // Test isnan and related functions (isinf, isfinite, etc.)
      const x = np.array([NaN, Infinity, -Infinity, 1]);
      expect(np.isnan(x.ref).js()).toEqual([true, false, false, false]);
      expect(np.isinf(x.ref).js()).toEqual([false, true, true, false]);
      expect(np.isfinite(x.ref).js()).toEqual([false, false, false, true]);
      expect(np.isneginf(x.ref).js()).toEqual([false, false, true, false]);
      expect(np.isposinf(x.ref).js()).toEqual([false, true, false, false]);
      x.dispose();
    });
  });

  suite("jax.numpy.nanToNum()", () => {
    test("replaces NaN with 0 by default", () => {
      const x = np.array([1, NaN, 3]);
      const y = np.nanToNum(x);
      expect(y.js()).toEqual([1, 0, 3]);
    });

    test("replaces NaN with custom value", () => {
      const x = np.array([NaN, 2, NaN]);
      const y = np.nanToNum(x, { nan: 99 });
      expect(y.js()).toEqual([99, 2, 99]);
    });

    test("replaces positive infinity when specified", () => {
      const x = np.array([1, Infinity, 3]);
      const y = np.nanToNum(x, { posinf: 999 });
      expect(y.js()).toEqual([1, 999, 3]);
    });

    test("replaces negative infinity when specified", () => {
      const x = np.array([1, -Infinity, 3]);
      const y = np.nanToNum(x, { neginf: -999 });
      expect(y.js()).toEqual([1, -999, 3]);
    });

    test("sets infinity to limit values when not specified", () => {
      const x = np.array([Infinity, -Infinity]);
      const y = np.nanToNum(x);
      expect(y).toBeAllclose([3.40282347e38, -3.40282347e38]);
    });

    test("handles all special values together", () => {
      const x = np.array([NaN, Infinity, -Infinity, 42]);
      const y = np.nanToNum(x, { nan: 0, posinf: 100, neginf: -100 });
      expect(y.js()).toEqual([0, 100, -100, 42]);
    });
  });

  suite("jax.numpy.convolve()", () => {
    test("computes 1D convolution", () => {
      const x = np.array([1, 2, 3, 2, 1]);
      const y = np.array([4, 1, 2]);

      const full = np.convolve(x.ref, y.ref);
      expect(full.js()).toEqual([4, 9, 16, 15, 12, 5, 2]);

      const same = np.convolve(x.ref, y.ref, "same");
      expect(same.js()).toEqual([9, 16, 15, 12, 5]);

      const valid = np.convolve(x, y, "valid");
      expect(valid.js()).toEqual([16, 15, 12]);
    });

    test("computes 1D correlation", () => {
      const x = np.array([1, 2, 3, 2, 1]);
      const y = np.array([4, 5, 6]);

      const valid = np.correlate(x.ref, y.ref);
      expect(valid.js()).toEqual([32, 35, 28]);

      const full = np.correlate(x.ref, y.ref, "full");
      expect(full.js()).toEqual([6, 17, 32, 35, 28, 13, 4]);

      const same = np.correlate(x, y, "same");
      expect(same.js()).toEqual([17, 32, 35, 28, 13]);

      const x1 = np.array([1, 2, 3, 2, 1]);
      const y1 = np.array([4, 5, 4]);
      const corr = np.correlate(x1.ref, y1.ref, "full");
      const conv = np.convolve(x1, y1, "full");
      expect(corr.js()).toEqual([4, 13, 26, 31, 26, 13, 4]);
      expect(conv.js()).toEqual([4, 13, 26, 31, 26, 13, 4]);
    });
  });

  suite("jax.numpy.all()", () => {
    test("returns true when all elements are true", () => {
      const x = np.array([true, true, true]);
      expect(np.all(x).js()).toEqual(true);
    });

    test("returns false when any element is false", () => {
      const x = np.array([true, false, true]);
      expect(np.all(x).js()).toEqual(false);
    });

    test("works along axis", () => {
      const x = np.array([
        [true, false],
        [true, true],
      ]);
      expect(np.all(x.ref, 0).js()).toEqual([true, false]);
      expect(np.all(x, 1).js()).toEqual([false, true]);
    });

    test("works with numeric arrays (truthy values)", () => {
      const x = np.array([1, 2, 3]);
      expect(np.all(x).js()).toEqual(true);

      const y = np.array([1, 0, 3]);
      expect(np.all(y).js()).toEqual(false);
    });

    test("supports keepdims", () => {
      const x = np.array([
        [true, true],
        [true, false],
      ]);
      const result = np.all(x, 1, { keepdims: true });
      expect(result.shape).toEqual([2, 1]);
      expect(result.js()).toEqual([[true], [false]]);
    });
  });

  suite("jax.numpy.any()", () => {
    test("returns true when any element is true", () => {
      const x = np.array([false, true, false]);
      expect(np.any(x).js()).toEqual(true);
    });

    test("returns false when all elements are false", () => {
      const x = np.array([false, false, false]);
      expect(np.any(x).js()).toEqual(false);
    });

    test("works along axis", () => {
      const x = np.array([
        [false, false],
        [true, false],
      ]);
      expect(np.any(x.ref, 0).js()).toEqual([true, false]);
      expect(np.any(x, 1).js()).toEqual([false, true]);
    });

    test("works with numeric arrays (truthy values)", () => {
      const x = np.array([0, 0, 0]);
      expect(np.any(x).js()).toEqual(false);

      const y = np.array([0, 1, 0]);
      expect(np.any(y).js()).toEqual(true);
    });

    test("supports keepdims", () => {
      const x = np.array([
        [false, false],
        [true, false],
      ]);
      const result = np.any(x, 1, { keepdims: true });
      expect(result.shape).toEqual([2, 1]);
      expect(result.js()).toEqual([[false], [true]]);
    });
  });

  suite("jax.numpy.expandDims()", () => {
    test("expands dims at position 0", () => {
      const x = np.array([1, 2, 3]);
      const y = np.expandDims(x, 0);
      expect(y.shape).toEqual([1, 3]);
      expect(y.js()).toEqual([[1, 2, 3]]);
    });

    test("expands dims at position 1", () => {
      const x = np.array([1, 2, 3]);
      const y = np.expandDims(x, 1);
      expect(y.shape).toEqual([3, 1]);
      expect(y.js()).toEqual([[1], [2], [3]]);
    });

    test("expands dims with negative axis", () => {
      const x = np.array([1, 2, 3]);
      const y = np.expandDims(x, -1);
      expect(y.shape).toEqual([3, 1]);
      expect(y.js()).toEqual([[1], [2], [3]]);
    });

    test("expands multiple dims at once", () => {
      const x = np.array([1, 2]);
      const y = np.expandDims(x, [0, 2]);
      expect(y.shape).toEqual([1, 2, 1]);
      expect(y.js()).toEqual([[[1], [2]]]);
    });

    test("expands dims on 2D array", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const y = np.expandDims(x.ref, 0);
      expect(y.shape).toEqual([1, 2, 3]);

      const z = np.expandDims(x, 2);
      expect(z.shape).toEqual([2, 3, 1]);
    });

    test("throws on out of bounds axis", () => {
      const x = np.array([1, 2, 3]);
      expect(() => np.expandDims(x, 3)).toThrow(Error);
      expect(() => np.expandDims(x, -4)).toThrow(Error);
    });

    test("throws on repeated axis", () => {
      const x = np.array([1, 2, 3]);
      expect(() => np.expandDims(x, [0, 0])).toThrow(Error);
    });

    test("works with jvp", () => {
      const x = np.array([1, 2, 3]);
      const [y, dy] = jvp(
        (x: np.Array) => np.expandDims(x, 0),
        [x],
        [np.ones([3])],
      );
      expect(y.shape).toEqual([1, 3]);
      expect(dy.shape).toEqual([1, 3]);
    });

    test("works with grad", () => {
      const x = np.array([1, 2, 3]);
      const dx = grad((x: np.Array) => np.expandDims(x, 0).sum())(x);
      expect(dx.js()).toEqual([1, 1, 1]);
    });
  });

  suite("jax.numpy.applyAlongAxis()", () => {
    test("applies a scalar-valued function along an axis", () => {
      const x = np.arange(24).reshape([2, 3, 4]);
      const y = np.applyAlongAxis((x: np.Array) => x.sum(), 1, x);
      expect(y.shape).toEqual([2, 4]);
      expect(y.js()).toEqual([
        [12, 15, 18, 21],
        [48, 51, 54, 57],
      ]);
    });

    test("inserts vector-valued results at the mapped axis", () => {
      const x = np.arange(24).reshape([2, 3, 4]);
      const y = np.applyAlongAxis(
        (x: np.Array) => np.stack([x.ref.sum(), x.max()]),
        -1,
        x,
      );
      expect(y.shape).toEqual([2, 3, 2]);
      expect(y.js()).toEqual([
        [
          [6, 3],
          [22, 7],
          [38, 11],
        ],
        [
          [54, 15],
          [70, 19],
          [86, 23],
        ],
      ]);
    });

    test("supports higher-rank function results", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const y = np.applyAlongAxis(
        (x: np.Array) => np.stack([x.ref, x.add(10)]),
        1,
        x,
      );
      expect(y.shape).toEqual([2, 2, 3]);
      expect(y.js()).toEqual([
        [
          [1, 2, 3],
          [11, 12, 13],
        ],
        [
          [4, 5, 6],
          [14, 15, 16],
        ],
      ]);
    });
  });

  suite("jax.numpy.applyOverAxes()", () => {
    test("expands reduced dimensions after applying over axes", () => {
      const x = np.arange(24).reshape([2, 3, 4]);
      const y = np.applyOverAxes(
        (x: np.Array, axis: number) => x.sum(axis),
        x,
        [0, 2],
      );
      expect(y.shape).toEqual([1, 3, 1]);
      expect(y.js()).toEqual([[[60], [92], [124]]]);
    });

    test("keeps dimensions returned by the function", () => {
      const x = np.arange(24).reshape([2, 3, 4]);
      const y = np.applyOverAxes(
        (x: np.Array, axis: number) => x.sum(axis, { keepdims: true }),
        x,
        [1],
      );
      expect(y.shape).toEqual([2, 1, 4]);
      expect(y.js()).toEqual([[[12, 15, 18, 21]], [[48, 51, 54, 57]]]);
    });

    test("supports negative axes", () => {
      const x = np.arange(24).reshape([2, 3, 4]);
      const y = np.applyOverAxes(
        (x: np.Array, axis: number) => x.sum(axis),
        x,
        [-1],
      );
      expect(y.shape).toEqual([2, 3, 1]);
      expect(y.js()).toEqual([
        [[6], [22], [38]],
        [[54], [70], [86]],
      ]);
    });

    test("requires functions to preserve or reduce rank by one", () => {
      const x = np.arange(6).reshape([2, 3]);
      expect(() =>
        np.applyOverAxes((x: np.Array) => np.expandDims(x, 0), x, [0]),
      ).toThrow(Error);
    });
  });

  if (device !== "webgl") {
    suite("jax.numpy.sort()", () => {
      test("sorts 1D array", () => {
        const x = np.array([3, 1, 4, 1, 5, 9, 2, 6]);
        const y = np.sort(x);
        expect(y.js()).toEqual([1, 1, 2, 3, 4, 5, 6, 9]);
      });

      test("sorts 2D array along axis", () => {
        const x = np.array([
          [3, 1, 2],
          [6, 4, 5],
        ]);
        const y0 = np.sort(x.ref, 0);
        expect(y0.js()).toEqual([
          [3, 1, 2],
          [6, 4, 5],
        ]);
        const y1 = np.sort(x, 1);
        expect(y1.js()).toEqual([
          [1, 2, 3],
          [4, 5, 6],
        ]);
      });

      test("sorts NaN to the end", () => {
        const x = np.array([3, NaN, 1, NaN, 2]);
        const y = np.sort(x);
        expect(y.js()).toEqual([1, 2, 3, NaN, NaN]);
      });

      test("works with jvp", () => {
        const x = np.array([3, 1, 2]);
        const [y, dy] = jvp(np.sort, [x], [np.array([10, 20, 30])]);
        expect(y.js()).toEqual([1, 2, 3]);
        expect(dy.js()).toEqual([20, 30, 10]);
      });

      test("works with jvp for batched inputs", () => {
        const x = np.array([
          [3, 1, 2],
          [4, 6, 5],
        ]);
        const dx = np.array([
          [10, 20, 30],
          [40, 60, 50],
        ]);
        const [y, dy] = jvp((x) => np.sort(x, 1), [x], [dx]);
        expect(y.js()).toEqual([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        expect(dy.js()).toEqual([
          [20, 30, 10],
          [40, 50, 60],
        ]);
      });

      test("works with grad", () => {
        const x = np.array([3, 1, 4, 2]);
        const f = (x: np.Array) => np.sort(x).slice([0, 2]).sum();
        const dx = grad(f)(x);
        expect(dx.js()).toEqual([0, 1, 0, 1]);
      });

      test("works with grad for batched inputs", () => {
        const weights = np.array([
          [10, 20, 30],
          [40, 50, 60],
        ]);
        const dx = grad((x: np.Array) => np.sort(x, 1).mul(weights).sum())(
          np.array([
            [3, 1, 2],
            [4, 6, 5],
          ]),
        );
        expect(dx.js()).toEqual([
          [30, 10, 20],
          [40, 60, 50],
        ]);
      });

      test("works inside a jit function", () => {
        const x = np.array([5, 2, 8, 1]);
        const f = jit((x: np.Array) => np.sort(x));
        const y = f(x);
        expect(y.js()).toEqual([1, 2, 5, 8]);
      });

      test("works for int and bool dtypes", () => {
        for (const dtype of [np.int32, np.uint32]) {
          const x = np.array([3, 1, 4, 1, 5], { dtype });
          const y = np.sort(x);
          expect(y.js()).toEqual([1, 1, 3, 4, 5]);
          expect(y.dtype).toBe(dtype);
        }
        const x = np.array([true, false, true, false, true]);
        const y = np.sort(x);
        expect(y.js()).toEqual([false, false, true, true, true]);
        expect(y.dtype).toBe(np.bool);
      });

      test("handles zero-sized arrays", () => {
        const x = np.array([[], [], []], { dtype: np.float32 });
        const y = np.sort(x);
        expect(y.shape).toEqual([3, 0]);
        expect(y.dtype).toBe(np.float32);
      });

      test("can sort 8192 elements", async () => {
        // If the maximum workgroup size is 1024, then only 2048 elements can fit
        // into a single-workgroup sort. This test exercises multi-pass sorting in
        // global memory for GPUs.
        const x = np.linspace(0, 1, 8192);
        const y = np.sort(np.flip(x.ref));
        expect(y).toBeAllclose(x);
      });
    });

    suite("jax.numpy.argsort()", () => {
      test("argsorts 1D array", () => {
        const x = np.array([3, 1, 4, 2, 5]);
        const idx = np.argsort(x);
        expect(idx.js()).toEqual([1, 3, 0, 2, 4]);
        expect(idx.dtype).toBe("int32");
      });

      test("argsorts 2D array", () => {
        const x = np.array([
          [3, 1, 2],
          [6, 4, 5],
        ]);
        const idx = np.argsort(x, 1);
        expect(idx.js()).toEqual([
          [1, 2, 0],
          [1, 2, 0],
        ]);
      });

      test("is a stable sorting algorithm", () => {
        const x = np.array([
          3,
          1,
          1,
          NaN,
          Infinity,
          2,
          NaN,
          1,
          0,
          -0,
          Infinity,
        ]);
        const idx = np.argsort(x);
        expect(idx.js()).toEqual([8, 9, 1, 2, 7, 5, 0, 4, 10, 3, 6]);
      });

      test("produces zero gradient", () => {
        const x = np.array([3, 1, 2]);
        const f = (x: np.Array) => np.argsort(x).astype(np.float32).sum();
        const dx = grad(f)(x);
        expect(dx.js()).toEqual([0, 0, 0]);
      });

      test("can argsort 8191 elements", async () => {
        // Testing 8191 as it's not exactly a power-of-two size.
        const x = np.linspace(0, 1, 8191);
        const y = np.argsort(np.flip(x));
        const ar = y.js() as number[];
        expect(ar).toEqual(Array.from({ length: 8191 }, (_, i) => 8190 - i));
      });
    });
  }

  suite("jax.numpy.take()", () => {
    test("takes elements from 1D array", () => {
      const x = np.array([10, 20, 30, 40, 50]);
      const indices = np.array([3, 0, 4, 1]);
      const y = np.take(x, indices);
      expect(y.js()).toEqual([40, 10, 50, 20]);
    });

    test("takes elements from 2D array along axis", () => {
      const x = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
      ]);
      const indices = np.array([2, 0]);
      const y0 = np.take(x.ref, indices.ref, 0);
      expect(y0.js()).toEqual([
        [70, 80, 90],
        [10, 20, 30],
      ]);
      const y1 = np.take(x, indices, 1);
      expect(y1.js()).toEqual([
        [30, 10],
        [60, 40],
        [90, 70],
      ]);
    });

    if (device !== "webgl") {
      test("works with grad and repeated indices", () => {
        const x = np.arange(10).astype(np.float32).reshape([5, 2]);
        const indices = np.array([3, 0, 3, 1], { dtype: np.int32 });
        const dx = grad((x: np.Array) => np.take(x, indices, 0).sum())(x);
        expect(dx).toBeAllclose([
          [1, 1],
          [1, 1],
          [0, 0],
          [2, 2],
          [0, 0],
        ]);
      });

      test("works with grad inside jit", () => {
        const f = jit(
          grad((x: np.Array) =>
            np
              .take(x, np.array([3, 1, 3], { dtype: np.int32 }), 0)
              .mul(2)
              .sum(),
          ),
        );
        const dx = f(np.zeros([5, 2]));
        expect(dx).toBeAllclose([
          [0, 0],
          [2, 2],
          [0, 0],
          [4, 4],
          [0, 0],
        ]);
      });

      test("vmaps gradients with mapped indices", () => {
        const f = vmap(
          grad((x: np.Array, indices: np.Array) => np.take(x, indices).sum()),
          [0, 0],
        );
        const dx = f(
          np.ones([2, 5]),
          np.array(
            [
              [0, 0, 2],
              [1, 3, 3],
            ],
            { dtype: np.int32 },
          ),
        );
        expect(dx).toBeAllclose([
          [2, 0, 1, 0, 0],
          [0, 1, 0, 2, 0],
        ]);
      });

      test("vmaps gradients with shared indices", () => {
        const indices = np.array([0, 0, 2], { dtype: np.int32 });
        const f = vmap(
          grad((x: np.Array) => np.take(x, indices).sum()),
          0,
        );
        const dx = f(np.ones([2, 5]));
        expect(dx).toBeAllclose([
          [2, 0, 1, 0, 0],
          [2, 0, 1, 0, 0],
        ]);
      });

      if (device === "cpu" || device === "webgpu") {
        test("supports float16 gradients", () => {
          const indices = np.array([1, 1, 3], { dtype: np.int32 });
          const dx = grad((x: np.Array) =>
            np
              .take(x, indices)
              .mul(np.array([0.5, 1.25, 2], { dtype: np.float16 }))
              .sum(),
          )(np.zeros([4], { dtype: np.float16 }));
          expect(dx).toBeAllclose([0, 1.75, 0, 2]);
        });
      }
    }
  });

  suite("jax.numpy.append()", () => {
    test("flattens inputs when axis is omitted", () => {
      const x = np.array([
        [1, 2],
        [3, 4],
      ]);
      const y = np.append(x, np.array([[5, 6]]));
      expect(y.shape).toEqual([6]);
      expect(y.js()).toEqual([1, 2, 3, 4, 5, 6]);
    });

    test("appends along an axis", () => {
      const x = np.array([
        [1, 2],
        [3, 4],
      ]);
      const y0 = np.append(x.ref, np.array([[5, 6]]), 0);
      expect(y0.js()).toEqual([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);

      const y1 = np.append(x, np.array([[5], [6]]), 1);
      expect(y1.js()).toEqual([
        [1, 2, 5],
        [3, 4, 6],
      ]);
    });

    test("works with grad", () => {
      const x = np.array([1, 2, 3]);
      const dx = grad((x: np.Array) => np.append(x, np.array([4, 5])).sum())(x);
      expect(dx.js()).toEqual([1, 1, 1]);
    });
  });

  suite("jax.numpy.takeAlongAxis()", () => {
    test("takes values along columns", () => {
      const x = np.array([
        [10, 20, 30],
        [40, 50, 60],
      ]);
      const indices = np.array([
        [2, 0],
        [1, 1],
      ]);
      const y = np.takeAlongAxis(x, indices, 1);
      expect(y.js()).toEqual([
        [30, 10],
        [50, 50],
      ]);
    });

    test("takes values along rows", () => {
      const x = np.array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
      ]);
      const indices = np.array([
        [2, 0, 1],
        [0, 2, 2],
      ]);
      const y = np.takeAlongAxis(x, indices, 0);
      expect(y.js()).toEqual([
        [70, 20, 60],
        [10, 80, 90],
      ]);
    });

    test("broadcasts non-axis dimensions", () => {
      const x = np.arange(12).reshape([3, 4]);
      const indices = np.array([[0, 2]]);
      const y = np.takeAlongAxis(x, indices, 1);
      expect(y.js()).toEqual([
        [0, 2],
        [4, 6],
        [8, 10],
      ]);
    });

    test("uses the last axis by default", () => {
      const x = np.array([
        [10, 20, 30],
        [40, 50, 60],
      ]);
      const indices = np.array([[1], [0]]);
      const y = np.takeAlongAxis(x, indices);
      expect(y.js()).toEqual([[20], [40]]);
    });

    if (device !== "webgl") {
      test("works with grad and repeated indices", () => {
        const indices = np.array(
          [
            [2, 0],
            [1, 1],
          ],
          { dtype: np.int32 },
        );
        const dx = grad((x: np.Array) => np.takeAlongAxis(x, indices, 1).sum())(
          np.zeros([2, 3]),
        );
        expect(dx).toBeAllclose([
          [1, 0, 1],
          [0, 2, 0],
        ]);
      });
    }

    test("rejects rank mismatches", () => {
      expect(() => np.takeAlongAxis(np.ones([2, 3]), np.ones([2]), 1)).toThrow(
        "takeAlongAxis: input and indices must have the same rank",
      );
    });

    test("rejects unbroadcastable non-axis dimensions", () => {
      expect(() =>
        np.takeAlongAxis(np.ones([2, 3]), np.ones([4, 2]), 1),
      ).toThrow("takeAlongAxis: non-axis dimensions must broadcast");
    });
  });

  suite("jax.numpy.logicalAnd()", () => {
    test("basic logical and", () => {
      const result = np.logicalAnd(
        np.array([1, 0, 3, 0]),
        np.array([false, false, true, false]),
      );
      expect(result.js()).toEqual([false, false, true, false]);
    });

    test("float inputs", () => {
      const result = np.logicalAnd(
        np.array([1.5, 0.0, -2.0]),
        np.array([0.5, 0.0, 1.0]),
      );
      expect(result.js()).toEqual([true, false, true]);
    });
  });

  suite("jax.numpy.logicalOr()", () => {
    test("basic logical or", () => {
      const result = np.logicalOr(
        np.array([1, 0, 3, 0]),
        np.array([0, 0, 1, 0]),
      );
      expect(result.js()).toEqual([true, false, true, false]);
    });
  });

  suite("jax.numpy.logicalXor()", () => {
    test("basic logical xor", () => {
      const result = np.logicalXor(
        np.array([1, 0, 3, 0]),
        np.array([0, 0, 1, 0]),
      );
      expect(result.js()).toEqual([true, false, false, false]);
    });
  });

  suite("jax.numpy.logicalNot()", () => {
    test("basic logical not", () => {
      const result = np.logicalNot(np.array([1, 0, 3, 0]));
      expect(result.js()).toEqual([false, true, false, true]);
    });
  });

  suite("jax.numpy.bitwiseAnd()", () => {
    test("uint32 and", () => {
      const a = np.array([0xff00ff00, 0x0f0f0f0f], { dtype: np.uint32 });
      const b = np.array([0x00ff00ff, 0xf0f0f0f0], { dtype: np.uint32 });
      expect(np.bitwiseAnd(a, b).js()).toEqual([0x00000000, 0x00000000]);
    });

    test("bool and", () => {
      const result = np.bitwiseAnd(
        np.array([true, true, false, false]),
        np.array([true, false, true, false]),
      );
      expect(result.js()).toEqual([true, false, false, false]);
    });
  });

  suite("jax.numpy.bitwiseOr()", () => {
    test("uint32 or", () => {
      const a = np.array([0xff00ff00, 7], { dtype: np.uint32 });
      const b = np.array([0x00ff00ff, 3], { dtype: np.uint32 });
      expect(np.bitwiseOr(a, b).js()).toEqual([0xffffffff, 7]);
    });
  });

  suite("jax.numpy.bitwiseXor()", () => {
    test("uint32 xor", () => {
      const a = np.array([0xaaaaaaaa, 7], { dtype: np.uint32 });
      const b = np.array([0x55555555, 3], { dtype: np.uint32 });
      expect(np.bitwiseXor(a, b).js()).toEqual([0xffffffff, 4]);
    });
  });

  suite("jax.numpy.invert()", () => {
    test("uint32 invert", () => {
      const result = np.invert(np.array([0, 0xffffffff], { dtype: np.uint32 }));
      expect(result.js()).toEqual([0xffffffff, 0]);
    });

    test("bool invert", () => {
      const result = np.invert(np.array([true, false]));
      expect(result.js()).toEqual([false, true]);
    });

    test("int32 invert", () => {
      const result = np.invert(np.array([0, -3], { dtype: np.int32 }));
      expect(result.js()).toEqual([-1, 2]);
    });
  });

  suite("jax.numpy.leftShift()", () => {
    test("basic left shift", () => {
      const result = np.leftShift(
        np.array([1, 1, 1], { dtype: np.uint32 }),
        np.array([0, 8, 16], { dtype: np.uint32 }),
      );
      expect(result.js()).toEqual([1, 256, 65536]);
    });
  });

  suite("jax.numpy.rightShift()", () => {
    test("basic right shift", () => {
      const result = np.rightShift(
        np.array([256, 65536, 0xffff0000], { dtype: np.uint32 }),
        np.array([0, 1, 8], { dtype: np.uint32 }),
      );
      expect(result.js()).toEqual([256, 32768, 0x00ffff00]);
    });
  });

  suite("jax.numpy.copysign()", () => {
    test("basic copysign", () => {
      const result = np.copysign(
        np.array([1, -2, 3, -4]),
        np.array([-1, 1, -1, 1]),
      );
      expect(result.js()).toEqual([-1, 2, -3, 4]);
    });

    test("copysign with zero", () => {
      const result = np.copysign(np.array([5, -5]), np.array([0, 0]));
      expect(result.js()).toEqual([0, 0]);
    });
  });

  suite("jax.numpy.round()", () => {
    test("round to integer", () => {
      const result = np.round(np.array([1.4, 1.5, 2.5, 3.5, -0.5]), 0);
      // Banker's rounding: 1.5 -> 2, 2.5 -> 2, 3.5 -> 4, -0.5 -> 0
      expect(result.js()).toBeAllclose([1, 2, 2, 4, 0]);
    });

    test("round to decimals", () => {
      const result = np.round(np.array([1.234, 2.567, 3.891]), 2);
      expect(result.js()).toBeAllclose([1.23, 2.57, 3.89]);
    });

    test("round with negative decimals", () => {
      const result = np.round(np.array([123, 456, 789]), -2);
      expect(result.js()).toBeAllclose([100, 500, 800]);
    });
  });

  suite("jax.numpy.rint()", () => {
    test("basic rint", () => {
      const result = np.rint(np.array([1.4, 1.5, 2.5, 3.6]));
      expect(result.js()).toBeAllclose([1, 2, 2, 4]);
    });
  });
});
