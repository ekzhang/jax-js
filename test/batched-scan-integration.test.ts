/**
 * Integration test for scan with various body types.
 *
 * Note: Matmul/Dot is NOT a Routine - it's lowered to Mul→Reduce Kernel.
 * Batched scan only applies to Routines like TriangularSolve, Cholesky, LU.
 *
 * These tests verify that scan works correctly on WebGPU regardless of path.
 *
 * Test Environment: Vitest + Playwright (Chromium with --enable-unsafe-webgpu)
 * WebGPU Note: May not work on all systems. For hardware WebGPU testing, see test/deno/batched-scan-integration.test.ts
 */

import { describe, expect, test } from "vitest";

import { defaultDevice, init, lax, numpy as np } from "../src";

describe("batched scan integration", () => {
  test("matmul-in-scan works on WebGPU (uses JS loop, not batched scan)", async () => {
    const availableDevices = await init();

    // Check if WebGPU is available
    console.log("Available devices:", availableDevices);

    if (!availableDevices.includes("webgpu")) {
      console.log(
        "WebGPU not available, skipping batched scan integration test",
      );
      return;
    }

    defaultDevice("webgpu");
    console.log("Using WebGPU backend");

    // Define step function with matmul (a routine)
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
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
        [
          [0, -1],
          [1, 0],
        ], // rotate 90 degrees
      ],
      { dtype: np.float32 },
    );

    console.log("Starting lax.scan with matmul body...");
    const [finalCarry, outputs] = await lax.scan(step, initCarry, xs);
    console.log("Scan completed");

    // I * [[2,0],[0,2]] = [[2,0],[0,2]]
    // [[2,0],[0,2]] * [[1,1],[0,1]] = [[2,2],[0,2]]
    // [[2,2],[0,2]] * [[0,-1],[1,0]] = [[2,-2],[2,0]]
    const finalData = await finalCarry.data();
    console.log("Final carry data:", finalData);

    expect(finalData[0]).toBeCloseTo(2.0, 3);
    expect(finalData[1]).toBeCloseTo(-2.0, 3);
    expect(finalData[2]).toBeCloseTo(2.0, 3);
    expect(finalData[3]).toBeCloseTo(0.0, 3);

    // Check outputs - should have 3 stacked 2x2 matrices
    const outputsData = await outputs.data();
    console.log("Outputs shape:", outputs.shape);
    console.log("Outputs data:", outputsData);

    // First output: [[2,0],[0,2]]
    expect(outputsData[0]).toBeCloseTo(2.0, 3);
    expect(outputsData[1]).toBeCloseTo(0.0, 3);
    expect(outputsData[2]).toBeCloseTo(0.0, 3);
    expect(outputsData[3]).toBeCloseTo(2.0, 3);
  });

  test("larger matmul sequence for performance testing", async () => {
    const availableDevices = await init();

    if (!availableDevices.includes("webgpu")) {
      console.log("WebGPU not available, skipping");
      return;
    }

    defaultDevice("webgpu");

    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.matmul(carry, x);
      return [newCarry.ref, newCarry];
    };

    // 4x4 matrices, 10 iterations
    const n = 4;
    const iterations = 10;
    const initCarry = np.eye(n);

    // Create scaling matrices (scale by 1.1)
    const scaleMatrix: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < n; j++) {
        row.push(i === j ? 1.1 : 0);
      }
      scaleMatrix.push(row);
    }

    const xsList: number[][][] = [];
    for (let i = 0; i < iterations; i++) {
      xsList.push(scaleMatrix);
    }
    const xs = np.array(xsList, { dtype: np.float32 });

    console.log(`Starting scan: ${iterations} iterations of ${n}x${n} matmul`);
    const start = performance.now();
    const [finalCarry, _outputs] = await lax.scan(step, initCarry, xs);
    const elapsed = performance.now() - start;
    console.log(`Scan completed in ${elapsed.toFixed(2)}ms`);

    // Final should be identity * (1.1)^iterations along diagonal
    const expected = Math.pow(1.1, iterations);
    const finalData = await finalCarry.data();
    console.log(
      `Expected diagonal: ${expected.toFixed(4)}, got: ${finalData[0].toFixed(4)}`,
    );

    expect(finalData[0]).toBeCloseTo(expected, 2);
    expect(finalData[n + 1]).toBeCloseTo(expected, 2); // diagonal
    expect(finalData[1]).toBeCloseTo(0.0, 5); // off-diagonal
  });
});
