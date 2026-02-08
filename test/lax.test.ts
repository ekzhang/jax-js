import { defaultDevice, devices, init, lax, numpy as np } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  const deviceHasSort = ["cpu", "wasm", "webgpu"].includes(device);

  if (deviceHasSort) {
    suite("jax.lax.topK()", () => {
      test("returns top k values and indices", () => {
        const x = np.array([3, 1, 4, 1, 5, 9, 2, 6]);
        const [values, indices] = lax.topK(x, 3);
        expect(values.js()).toEqual([9, 6, 5]);
        expect(indices.js()).toEqual([5, 7, 4]);
      });

      test("stability: equal values preserve original order", () => {
        const x = np.array([1, 3, 2, 3, 3, 0]);
        const [values, indices] = lax.topK(x, 4);
        expect(values.js()).toEqual([3, 3, 3, 2]);
        expect(indices.js()).toEqual([1, 3, 4, 2]);
      });

      test("k equals array length", () => {
        const x = np.array([5, 2, 8]);
        const [values, indices] = lax.topK(x, 3);
        expect(values.js()).toEqual([8, 5, 2]);
        expect(indices.js()).toEqual([2, 0, 1]);
      });

      test("k = 0 returns empty", () => {
        const x = np.array([1, 2, 3]);
        const [values, indices] = lax.topK(x, 0);
        expect(values.shape).toEqual([0]);
        expect(indices.shape).toEqual([0]);
      });

      test("works with 2D along either axis", () => {
        const x = np.array([
          [3, 1, 4],
          [1, 5, 9],
        ]);
        let [values, indices] = lax.topK(x.ref, 2);
        expect(values.js()).toEqual([
          [4, 3],
          [9, 5],
        ]);
        expect(indices.js()).toEqual([
          [2, 0],
          [2, 1],
        ]);

        [values, indices] = lax.topK(x.ref, 1, 0);
        expect(values.js()).toEqual([[3, 5, 9]]);
        expect(indices.js()).toEqual([[0, 1, 1]]);
      });

      test("works with floats and NaN is highest", () => {
        const x = np.array([1.5, NaN, 3.5, 2.1]);
        const [values, indices] = lax.topK(x, 2);
        expect(values.js()).toEqual([NaN, 3.5]); // this is consistent with JAX
        expect(indices.js()).toEqual([1, 2]);
      });

      test("throws for invalid k", () => {
        const x = np.array([1, 2, 3]);
        expect(() => lax.topK(x, -1)).toThrow();
        expect(() => lax.topK(x, 4)).toThrow();
      });
    });
  }
});
