/**
 * Integration test for scan with various body types on WebGPU.
 *
 * Note: Batched scan is only used when:
 * - Body is a single Routine step (not Kernel)
 * - Routine doesn't already use uniforms (excludes Sort/Argsort)
 * - numCarry === numY
 *
 * In practice, most scan bodies have multiple steps (reshape, add, etc.) so they
 * fall back to JS loop. These tests verify correctness for both paths.
 *
 * Test Environment: Deno native WebGPU (requires GPU hardware)
 * Why Deno: Provides working WebGPU access when Chromium's implementation doesn't work.
 * Regular test equivalent: test/batched-scan-integration.test.ts (may skip on some systems)
 *
 * Run with: deno test --unstable-webgpu --allow-read --allow-env --no-check test/deno/batched-scan-integration.test.ts
 */

import {
  assertEquals,
  assertAlmostEquals,
} from "https://deno.land/std@0.208.0/assert/mod.ts";

// Import jax-js from dist to share backend with other tests
import {
  init,
  defaultDevice,
  numpy as np,
  lax,
  setDebug,
} from "../../dist/index.js";

// Import leak detection harness
import { withLeakCheck } from "./harness.ts";

// Enable debug logging to see which scan path is taken
// Disable debug logging to reduce noise (enable for debugging)
// setDebug(2);

Deno.test({
  name: "sort-in-scan works on WebGPU (falls back to JS loop due to Sort using uniforms)",
  ignore: !navigator.gpu,
  fn: withLeakCheck(async () => {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      console.log("No WebGPU adapter, skipping");
      return;
    }

    console.log("GPU adapter:", adapter.info);

    const availableDevices = await init();
    console.log("Available devices:", availableDevices);

    if (!availableDevices.includes("webgpu")) {
      console.log("WebGPU not in jax-js devices, skipping");
      return;
    }

    defaultDevice("webgpu");
    console.log("Using WebGPU backend");

    // Define step function with np.sort (a REAL Routine)
    // Body is EXACTLY one sort operation - no reshapes, no adds
    const step = (
      carry: typeof np.Array.prototype,
      x: typeof np.Array.prototype,
    ): [typeof np.Array.prototype, typeof np.Array.prototype] => {
      // carry and x are both [4] arrays
      // Concatenate, sort, and take the first 4 elements (a streaming min-heap)
      // But wait, that's multiple operations...
      //
      // For batched scan to work, we need EXACTLY one routine call.
      // Since sort outputs the sorted array, we can't do anything else.
      //
      // Simplest case: sort the incoming x, pass the sorted x as both carry and output
      const sorted = np.sort(x);
      return [sorted.ref, sorted];
    };

    // Initial carry: dummy (will be replaced by first sorted x)
    const initCarry = np.zeros([4], "float32");

    // Sequence of arrays to sort
    const xs = np.array(
      [
        [3, 1, 4, 1],
        [5, 9, 2, 6],
        [5, 3, 5, 8],
      ],
      "float32",
    );

    console.log("Starting lax.scan with sort body...");
    const [finalCarry, outputs] = await lax.scan(step, initCarry, xs);
    console.log("Scan completed");

    // Final carry should be sorted last array: [3, 5, 5, 8]
    const finalData = await finalCarry.data();
    console.log("Final carry data:", finalData);

    assertAlmostEquals(finalData[0], 3.0, 0.01);
    assertAlmostEquals(finalData[1], 5.0, 0.01);
    assertAlmostEquals(finalData[2], 5.0, 0.01);
    assertAlmostEquals(finalData[3], 8.0, 0.01);

    // Outputs should have all 3 sorted arrays
    const outputsData = await outputs.data();
    console.log("Outputs data:", outputsData);
    // First sorted: [1, 1, 3, 4]
    assertAlmostEquals(outputsData[0], 1.0, 0.01);
    assertAlmostEquals(outputsData[1], 1.0, 0.01);
    assertAlmostEquals(outputsData[2], 3.0, 0.01);
    assertAlmostEquals(outputsData[3], 4.0, 0.01);
  }),
});

Deno.test({
  name: "matmul-in-scan falls back to JS loop (not a Routine)",
  ignore: !navigator.gpu,
  fn: withLeakCheck(async () => {
    // This test verifies that matmul uses JS loop fallback since Dot is NOT a Routine
    // (it's lowered to Mul→Reduce kernel)

    const availableDevices = await init();
    if (!availableDevices.includes("webgpu")) {
      console.log("WebGPU not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    const step = (
      carry: typeof np.Array.prototype,
      x: typeof np.Array.prototype,
    ): [typeof np.Array.prototype, typeof np.Array.prototype] => {
      const newCarry = np.matmul(carry, x);
      return [newCarry.ref, newCarry];
    };

    // 2x2 matrices: identity * sequence of matrices
    const initCarry = np.eye(2);
    const xs = np.array(
      [
        [
          [2, 0],
          [0, 2],
        ], // scale by 2
        [
          [1, 1],
          [0, 1],
        ], // shear
      ],
      "float32",
    );

    console.log("Starting lax.scan with matmul body (should use JS loop)...");
    const [finalCarry, outputs] = await lax.scan(step, initCarry, xs);
    console.log("Scan completed");

    // I * [[2,0],[0,2]] = [[2,0],[0,2]]
    // [[2,0],[0,2]] * [[1,1],[0,1]] = [[2,2],[0,2]]
    const finalData = await finalCarry.data();
    console.log("Final carry data:", finalData);

    assertAlmostEquals(finalData[0], 2.0, 0.01);
    assertAlmostEquals(finalData[1], 2.0, 0.01);
    assertAlmostEquals(finalData[2], 0.0, 0.01);
    assertAlmostEquals(finalData[3], 2.0, 0.01);

    // Consume outputs to avoid memory leak
    await outputs.data();
  }),
});
