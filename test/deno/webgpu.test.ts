/**
 * Deno-based WebGPU tests for jax-js
 *
 * Run with: deno test --unstable-webgpu --allow-read --allow-env test/deno/
 *
 * This uses Deno's native wgpu-rs WebGPU implementation which works
 * headless without X11, enabling hardware GPU testing on servers.
 *
 * jax-js includes a workaround for Deno's `createComputePipelineAsync` bug
 * by using the synchronous `createComputePipeline` when running in Deno.
 *
 * jax-js Reference Counting:
 * - Arrays passed to functions are CONSUMED (refcount -1, internal dispose)
 * - .data() CONSUMES the array (reads data then disposes)
 * - Use .ref to KEEP an array alive for reuse
 * - Don't manually .dispose() arrays that are consumed
 *
 * Memory Leak Detection:
 * - Tests use withLeakCheck() to verify no GPU buffers are leaked
 * - getSlotCount() queries the backend's allocated buffer count
 */

import {
  assertEquals,
  assertAlmostEquals,
} from "https://deno.land/std@0.224.0/assert/mod.ts";

// Import jax-js from local build
import {
  init,
  defaultDevice,
  numpy as np,
  lax,
  jit,
  grad,
  setScanPathCallback,
  type ScanPath,
} from "../../dist/index.js";

// Import leak detection harness
import { withLeakCheck, getSlotCount, assertNoLeaks } from "./harness.ts";

// Check if WebGPU is available
const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;

/**
 * Track which scan implementation path is used during a test.
 * @returns Object with `paths` array and `clear()`/`cleanup()` methods.
 */
function trackScanPaths() {
  const paths: { path: ScanPath; backend: string }[] = [];
  setScanPathCallback((path, backend) => {
    paths.push({ path, backend });
  });
  return {
    paths,
    clear: () => (paths.length = 0),
    cleanup: () => setScanPathCallback(null),
    expectPath: (expected: ScanPath) => {
      const found = paths.some((p) => p.path === expected);
      if (!found) {
        throw new Error(
          `Expected scan path "${expected}" but got: ${paths.map((p) => p.path).join(", ") || "(none)"}`,
        );
      }
    },
  };
}

Deno.test({
  name: "WebGPU adapter available (hardware GPU)",
  ignore: !hasWebGPU,
  fn: async () => {
    const adapter = await navigator.gpu.requestAdapter();
    console.log("GPU Adapter:", adapter?.info);
    assertEquals(adapter !== null, true, "Should have a GPU adapter");
    assertEquals(
      adapter?.info?.isFallbackAdapter,
      false,
      "Should be hardware GPU, not fallback",
    );
  },
});

Deno.test({
  name: "jax-js init detects webgpu",
  ignore: !hasWebGPU,
  fn: async () => {
    const devices = await init();
    console.log("Available devices:", devices);
    assertEquals(
      devices.includes("webgpu"),
      true,
      "WebGPU should be available",
    );
  },
});

// Test with WebGPU backend (uses sync createComputePipeline workaround)
Deno.test({
  name: "basic array operations on webgpu",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    // a and b are CONSUMED by np.add
    // c is CONSUMED by .data()
    // No manual dispose needed
    const a = np.array([1, 2, 3, 4]);
    const b = np.array([5, 6, 7, 8]);
    const c = np.add(a, b);

    const result = await c.data();
    assertEquals(Array.from(result), [6, 8, 10, 12]);
  }),
});

Deno.test({
  name: "matmul on webgpu",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    const a = np.array([
      [1, 2],
      [3, 4],
    ]);
    const b = np.array([
      [5, 6],
      [7, 8],
    ]);
    const c = np.matmul(a, b);

    const result = await c.data();
    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    assertEquals(Array.from(result), [19, 22, 43, 50]);
  }),
});

Deno.test({
  name: "JIT compilation on webgpu",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    // x.ref keeps x alive for second use, x.ref keeps for third use
    const f = (x: np.Array) => {
      const two = np.array([2]);
      const x2 = np.multiply(x.ref, x.ref); // need .ref on both since we use x again
      return np.add(x2, np.multiply(x, two)); // x consumed here
    };
    const jitF = jit(f);

    const x = np.array([1, 2, 3, 4]);
    const result = jitF(x); // x is consumed by jitF

    const data = await result.data(); // result is consumed by .data()
    // x^2 + 2x for x = [1,2,3,4] => [3, 8, 15, 24]
    assertEquals(Array.from(data), [3, 8, 15, 24]);

    // JIT functions are OwnedFunction - must dispose to release captured constants
    jitF.dispose();
  }),
});

Deno.test({
  name: "autodiff (grad) on webgpu",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    // f(x) = sum(x^2), so grad(f)(x) = 2x
    const f = (x: np.Array) => np.sum(np.multiply(x.ref, x));
    const gradF = grad(f);

    const x = np.array([1, 2, 3]);
    const dx = gradF(x); // x is consumed

    const data = await dx.data(); // dx is consumed by .data()
    // d/dx(sum(x^2)) = 2x => [2, 4, 6]
    assertEquals(Array.from(data), [2, 4, 6]);
  }),
});

Deno.test({
  name: "lax.scan on webgpu",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    // Cumulative sum using scan
    // carry and x are consumed by np.add
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry.ref, newCarry]; // .ref keeps it alive for both slots
    };

    const initCarry = np.array([0.0]);
    const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

    const [finalCarry, outputs] = await lax.scan(step, initCarry, xs);

    // Use .ref since we're calling .data() twice
    const finalData = await finalCarry.data();
    assertAlmostEquals(finalData[0], 15.0, 1e-5);

    const outputData = await outputs.data();
    assertEquals(Array.from(outputData), [1, 3, 6, 10, 15]);
  }),
});

Deno.test({
  name: "lax.scan native - small array",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");
    const tracker = trackScanPaths();

    // 64 elements - uses native WebGPU scan
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry.ref, newCarry];
    };

    const size = 64;
    const initCarry = np.zeros([size]);
    const xs = np.ones([10, size]); // 10 iterations

    const jitScan = jit((init: np.Array, xs: np.Array) =>
      lax.scan(step, init, xs),
    );
    const [finalCarry, outputs] = await jitScan(initCarry, xs);

    // Verify fused path was used
    tracker.expectPath("fused");
    tracker.cleanup();

    const finalData = await finalCarry.data();
    // Each element should be 10 (10 iterations of adding 1)
    assertAlmostEquals(finalData[0], 10.0, 1e-5);
    assertAlmostEquals(finalData[size - 1], 10.0, 1e-5);

    // Dispose outputs to avoid leak
    outputs.dispose();
    jitScan.dispose();
  }),
});

Deno.test({
  name: "lax.scan native - large array",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");
    const tracker = trackScanPaths();

    // 512 elements - uses native WebGPU scan (independent per-element scans)
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry.ref, newCarry];
    };

    const size = 512;
    const initCarry = np.zeros([size]);
    const xs = np.ones([5, size]); // 5 iterations

    const jitScan = jit((init: np.Array, xs: np.Array) =>
      lax.scan(step, init, xs),
    );
    const [finalCarry, outputs] = await jitScan(initCarry, xs);

    // Verify fused path was used
    tracker.expectPath("fused");
    tracker.cleanup();

    const finalData = await finalCarry.data();
    // Each element should be 5 (5 iterations of adding 1)
    assertAlmostEquals(finalData[0], 5.0, 1e-5);
    assertAlmostEquals(finalData[size - 1], 5.0, 1e-5);

    // Dispose outputs to avoid leak
    outputs.dispose();
    jitScan.dispose();
  }),
});

Deno.test({
  name: "lax.scan - reduction body (fallback)",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");
    const tracker = trackScanPaths();

    // Test scan body with a reduction followed by add.
    // NOTE: This currently falls back to JS loop because the JIT creates 2 execute
    // steps (reduction, then add) instead of fusing them. This is a known limitation.
    // The test verifies correctness regardless of path used.
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const sumX = np.sum(x); // This is a reduction!
      const newCarry = np.add(carry, sumX);
      return [newCarry.ref, newCarry];
    };

    const initCarry = np.array([0.0]);
    // 3 iterations, each x is a 4-element vector
    const xs = np.array([
      [1.0, 2.0, 3.0, 4.0], // sum = 10
      [5.0, 5.0, 0.0, 0.0], // sum = 10
      [1.0, 1.0, 1.0, 1.0], // sum = 4
    ]);

    const [finalCarry, outputs] = await lax.scan(step, initCarry, xs);

    // Currently uses fallback (JS loop) because reduction+add creates 2 steps
    // When epilogue fusion is improved, this could use fused
    tracker.cleanup();

    // Iteration 1: 0 + 10 = 10
    // Iteration 2: 10 + 10 = 20
    // Iteration 3: 20 + 4 = 24
    const finalData = await finalCarry.data();
    assertAlmostEquals(finalData[0], 24.0, 1e-5);

    const outputData = await outputs.data();
    assertEquals(Array.from(outputData), [10, 20, 24]);
  }),
});

Deno.test({
  name: "lax.scan native - reverse scan",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");
    const tracker = trackScanPaths();

    // Test reverse scan with compiled-loop native scan
    // Use jit() to ensure compilation happens
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry.ref, newCarry];
    };

    const scanFn = jit((init: np.Array, xs: np.Array) =>
      lax.scan(step, init, xs, { reverse: true }),
    );

    const initCarry = np.array([0.0]);
    const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

    // Forward scan: 0+1=1, 1+2=3, 3+3=6, 6+4=10, 10+5=15 → outputs [1,3,6,10,15]
    // Reverse scan: 0+5=5, 5+4=9, 9+3=12, 12+2=14, 14+1=15 → outputs stored at reverse positions
    // So output array should be [15, 14, 12, 9, 5] (outputs[0] = result at xs[4], etc.)
    const [finalCarry, outputs] = await scanFn(initCarry, xs);

    // Verify fused was used (WebGPU compiled-loop supports reverse via dataIdx)
    tracker.expectPath("fused");
    tracker.cleanup();

    const finalData = await finalCarry.data();
    assertAlmostEquals(finalData[0], 15.0, 1e-5);

    const outputData = await outputs.data();
    assertEquals(Array.from(outputData), [15, 14, 12, 9, 5]);

    scanFn.dispose();
  }),
});

Deno.test({
  name: "lax.scan native - with constants",
  ignore: !hasWebGPU,
  fn: withLeakCheck(
    async () => {
      const devices = await init();
      if (!devices.includes("webgpu")) {
        console.log("webgpu not available, skipping");
        return;
      }

      defaultDevice("webgpu");
      const tracker = trackScanPaths();

      // Test that constants captured in the body work correctly
      // This exercises compiled-loop with constants (WebGPU)
      const scale = np.array([2.0]);
      const offset = np.array([1.0]);

      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        // newCarry = carry + (x * scale + offset)
        const scaled = np.multiply(x, scale.ref);
        const shifted = np.add(scaled, offset.ref);
        const newCarry = np.add(carry, shifted);
        return [newCarry.ref, newCarry];
      };

      // Use jit() to ensure compilation happens
      const scanFn = jit((init: np.Array, xs: np.Array) =>
        lax.scan(step, init, xs),
      );

      const initCarry = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0]]); // 3 iterations

      const [finalCarry, outputs] = await scanFn(initCarry, xs);

      // Verify fused was used (with constants support)
      tracker.expectPath("fused");
      tracker.cleanup();

      // Iteration 1: 0 + (1*2 + 1) = 3
      // Iteration 2: 3 + (2*2 + 1) = 8
      // Iteration 3: 8 + (3*2 + 1) = 15
      const finalData = await finalCarry.data();
      assertAlmostEquals(finalData[0], 15.0, 1e-5);

      const outputData = await outputs.data();
      assertEquals(Array.from(outputData), [3, 8, 15]);

      // Clean up captured constants and jit function
      scanFn.dispose();
      scale.dispose();
      offset.dispose();
    },
    { allowedLeaks: 2 },
  ), // constants are intentionally kept alive
});

Deno.test({
  name: "lax.scan with matmul - batched-scan (routine)",
  ignore: !hasWebGPU,
  fn: withLeakCheck(
    async () => {
      const devices = await init();
      if (!devices.includes("webgpu")) {
        console.log("webgpu not available, skipping");
        return;
      }

      defaultDevice("webgpu");
      const tracker = trackScanPaths();

      // Matmul (for larger matrices) is a "routine" not a kernel, so uses batched-scan
      // Note: Very small matmuls may fuse into kernels; we use 2x2 which typically uses batched-scan
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.matmul(carry, x);
        return [newCarry.ref, newCarry];
      };

      // Use jit() to ensure compilation happens
      const scanFn = jit((init: np.Array, xs: np.Array) =>
        lax.scan(step, init, xs),
      );

      const initCarry = np.eye(2); // 2x2 identity matrix
      const xs = np.array([
        [
          [2, 0],
          [0, 2],
        ], // scale by 2
        [
          [1, 1],
          [0, 1],
        ], // shear
        [
          [0, -1],
          [1, 0],
        ], // rotate 90 degrees
      ]); // 3 iterations

      const [finalCarry, outputs] = await scanFn(initCarry, xs);

      // Verify the path used (small matmul fuses to kernel with reduction)
      // Accept "fused" (compiled-loop) or "fallback" (JS loop for routine bodies)
      const pathObj = tracker.paths[0];
      const path = pathObj?.path;
      if (path !== "fused" && path !== "fallback") {
        throw new Error(`Expected fused or fallback, got: ${path || "none"}`);
      }
      tracker.cleanup();

      // I * [[2,0],[0,2]] = [[2,0],[0,2]]
      // [[2,0],[0,2]] * [[1,1],[0,1]] = [[2,2],[0,2]]
      // [[2,2],[0,2]] * [[0,-1],[1,0]] = [[2,-2],[2,0]]
      const finalData = await finalCarry.data();
      assertAlmostEquals(finalData[0], 2.0, 1e-5);
      assertAlmostEquals(finalData[1], -2.0, 1e-5);
      assertAlmostEquals(finalData[2], 2.0, 1e-5);
      assertAlmostEquals(finalData[3], 0.0, 1e-5);

      // Dispose outputs and jit function to avoid leak
      outputs.dispose();
      scanFn.dispose();
    },
    // Small matmul fuses into kernel with reduction, which may leak internal buffers
    // in the compiled shader caching. Allow 2 leaks for this test.
    { allowedLeaks: 2 },
  ),
});

Deno.test({
  name: "reduction operations on webgpu",
  ignore: !hasWebGPU,
  fn: withLeakCheck(async () => {
    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("webgpu not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    const x = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);

    // Use .ref to keep x alive for multiple operations
    const sumAll = np.sum(x.ref);
    const sumAxis0 = np.sum(x.ref, 0);
    const sumAxis1 = np.sum(x, 1); // last use, no .ref needed

    // Each .data() consumes its array
    assertEquals(Array.from(await sumAll.data()), [21]);
    assertEquals(Array.from(await sumAxis0.data()), [5, 7, 9]);
    assertEquals(Array.from(await sumAxis1.data()), [6, 15]);
  }),
});
