import {
  defaultDevice,
  devices,
  grad,
  init,
  numpy as np,
} from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

// Tests run on all available backends: CPU, WASM, and WebGPU
suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  suite("jax.numpy.argmax() performance", () => {
    test("efficiently finds argmax in large array", () => {
      const x = np.arange(10000);
      const startTime = performance.now();
      const result = np.argmax(x);
      const endTime = performance.now();
      
      expect(result.js()).toEqual(9999);
      
      // Should complete reasonably fast (< 10ms per operation on large arrays)
      // This demonstrates the argreduce primitive is being used instead of
      // a full reduction followed by a search
      const elapsed = endTime - startTime;
      expect(elapsed).toBeLessThan(50); // Allow some margin for slower backends
    });

    test("efficiently finds argmax along axis", () => {
      const x = np.arange(100000).reshape([100, 1000]);
      const startTime = performance.now();
      const result = np.argmax(x.ref, 1);
      const endTime = performance.now();
      
      expect(result.shape).toEqual([100]);
      const resultData = result.js();
      expect(resultData[0]).toEqual(999);
      expect(resultData[99]).toEqual(999);
      
      const elapsed = endTime - startTime;
      expect(elapsed).toBeLessThan(100); // 100 reductions should be fast
    });

    test("argmax returns first index on ties", () => {
      // Verify tie-breaking behavior: should prefer earlier index
      const x = np.array([3, 5, 5, 2, 5, 1]);
      const result = np.argmax(x);
      expect(result.js()).toEqual(1); // First occurrence of maximum (5)
    });

    test("argmax works with negative values", () => {
      const x = np.array([-10, -5, -20, -5, -15]);
      const result = np.argmax(x);
      expect(result.js()).toEqual(1); // First occurrence of maximum (-5)
    });
  });

  suite("jax.numpy.argmin() performance", () => {
    test("efficiently finds argmin in large array", () => {
      const x = np.arange(10000, 0, -1); // Descending
      const startTime = performance.now();
      const result = np.argmin(x);
      const endTime = performance.now();
      
      expect(result.js()).toEqual(9999);
      
      const elapsed = endTime - startTime;
      expect(elapsed).toBeLessThan(50);
    });

    test("efficiently finds argmin along axis", () => {
      const x = np.arange(100000, 0, -1).reshape([100, 1000]);
      const startTime = performance.now();
      const result = np.argmin(x.ref, 1);
      const endTime = performance.now();
      
      expect(result.shape).toEqual([100]);
      const resultData = result.js();
      expect(resultData[0]).toEqual(999);
      expect(resultData[99]).toEqual(999);
      
      const elapsed = endTime - startTime;
      expect(elapsed).toBeLessThan(100);
    });

    test("argmin returns first index on ties", () => {
      const x = np.array([3, 1, 5, 1, 1, 2]);
      const result = np.argmin(x);
      expect(result.js()).toEqual(1); // First occurrence of minimum (1)
    });
  });

  suite("argreduce correctness", () => {
    test("argmax handles all equal values", () => {
      const x = np.ones([5]);
      const result = np.argmax(x);
      expect(result.js()).toEqual(0); // Returns first index
    });

    test("argmin handles all equal values", () => {
      const x = np.ones([5]).mul(42);
      const result = np.argmin(x);
      expect(result.js()).toEqual(0); // Returns first index
    });

    test("argmax 2D various shapes", () => {
      const x = np.array([
        [1, 2, 3],
        [6, 5, 4],
        [7, 8, 9],
      ]);
      
      expect(np.argmax(x.ref).js()).toEqual(8); // Global argmax
      expect(np.argmax(x.ref, 0).js()).toEqual([2, 2, 2]); // Column-wise
      expect(np.argmax(x, 1).js()).toEqual([2, 0, 2]); // Row-wise
    });

    test("argmin 2D various shapes", () => {
      const x = np.array([
        [9, 8, 7],
        [4, 5, 6],
        [3, 2, 1],
      ]);
      
      expect(np.argmin(x.ref).js()).toEqual(8); // Global argmin
      expect(np.argmin(x.ref, 0).js()).toEqual([2, 2, 2]); // Column-wise
      expect(np.argmin(x, 1).js()).toEqual([2, 0, 2]); // Row-wise
    });
  });
});
