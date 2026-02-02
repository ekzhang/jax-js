/**
 * Manual test for batched scan with routine body (matmul).
 * This tests the uniform-based offset approach without full JIT integration.
 *
 * Test Environment: Vitest + Playwright (Chromium with --enable-unsafe-webgpu)
 * WebGPU Note: May not work on all systems. For hardware WebGPU testing, see test/deno/batched-scan.test.ts
 */

import { describe, expect, it } from "vitest";

import type { WebGPUBackend } from "../src/backend/webgpu";
import type { ScanBindingInfo } from "../src/backend/webgpu/scan-wrapper";
import { getBackend, init, lax, numpy as np } from "../src/index";

describe("batched scan manual test", () => {
  it("verifies uniform offset approach with matmul body", async () => {
    const devices = await init("webgpu");
    if (!devices.includes("webgpu")) {
      console.log("WebGPU not available, skipping");
      return;
    }

    // Simple scan with matmul: accumulate matrix products
    // carry = carry @ x for each x in xs
    const n = 4; // Matrix size
    const length = 3; // Number of iterations

    // Create test data
    // initCarry: identity matrix
    const initCarry = np.eye(n);

    // xs: stack of matrices to multiply
    // xs[0] = [[2, 0], [0, 1], ...]  (scale x by 2)
    // xs[1] = [[1, 0], [0, 2], ...]  (scale y by 2)
    // xs[2] = [[1, 1], [0, 1], ...]  (shear)
    const xsData = new Float32Array(length * n * n);

    // xs[0]: scale x by 2
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        xsData[0 * n * n + i * n + j] = i === j ? (i === 0 ? 2 : 1) : 0;
      }
    }

    // xs[1]: scale y by 2
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        xsData[1 * n * n + i * n + j] = i === j ? (i === 1 ? 2 : 1) : 0;
      }
    }

    // xs[2]: add first row to second row (shear in x)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) {
          xsData[2 * n * n + i * n + j] = 1;
        } else if (i === 0 && j === 1) {
          xsData[2 * n * n + i * n + j] = 1;
        } else {
          xsData[2 * n * n + i * n + j] = 0;
        }
      }
    }

    const xs = np.array(xsData).reshape([length, n, n]);

    // Run scan with JS loop (reference)
    const scanFn = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const result = np.matmul(carry.ref, x);
      return [result.ref, result];
    };

    const [finalCarry, ys] = await lax.scan(scanFn, initCarry.ref, xs.ref);

    // Get reference results
    const finalCarryData = await finalCarry.data();
    const _ysData = await ys.data();

    console.log(
      "Final carry (reference):",
      Array.from(finalCarryData).slice(0, 16),
    );
    console.log("ys shape:", ys.shape);

    // Compute expected result manually:
    // After iter 0: I @ xs[0] = xs[0] (scale x by 2)
    // After iter 1: xs[0] @ xs[1] = diag(2, 2, 1, 1)
    // After iter 2: diag(2,2,1,1) @ shear

    // Verify with reference computation
    const I = np.eye(n);
    const x0 = np.array(xsData.slice(0, n * n)).reshape([n, n]);
    const x1 = np.array(xsData.slice(n * n, 2 * n * n)).reshape([n, n]);
    const x2 = np.array(xsData.slice(2 * n * n, 3 * n * n)).reshape([n, n]);

    const y0 = np.matmul(I.ref, x0);
    const y1 = np.matmul(y0.ref, x1);
    const y2 = np.matmul(y1.ref, x2);

    const expectedFinal = await y2.data();
    console.log("Expected final:", Array.from(expectedFinal).slice(0, 16));

    // Verify
    for (let i = 0; i < n * n; i++) {
      expect(finalCarryData[i]).toBeCloseTo(expectedFinal[i], 5);
    }

    // Cleanup
    I.dispose();
    x0.dispose();
    x1.dispose();
    x2.dispose();
    y0.dispose();
    y1.dispose();

    console.log("✓ JS loop scan with matmul body works correctly");
  });

  it("tests prepareBatchedScan shader wrapping", async () => {
    const devices = await init("webgpu");
    if (!devices.includes("webgpu")) {
      console.log("WebGPU not available, skipping");
      return;
    }
    const backend = getBackend() as WebGPUBackend;
    if (backend.type !== "webgpu") {
      console.log("WebGPU not available, skipping");
      return;
    }

    // We need to manually construct a BatchedScanParams and test prepareBatchedScan
    // This requires creating a routine executable, which is complex.
    // For now, let's just verify the scan-wrapper functions work in isolation.

    const { wrapRoutineForScan, createAllIterationsOffsetsBuffer } =
      await import("../src/backend/webgpu/scan-wrapper");

    // Test shader wrapping with a matmul-like pattern
    const mockShader = {
      code: `
@group(0) @binding(0) var<storage, read> carry_in: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> carry_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  let col = id.y;
  let n = 4u;
  
  if (row >= n || col >= n) { return; }
  
  var sum = 0.0;
  for (var k = 0u; k < n; k = k + 1u) {
    sum = sum + carry_in[row * n + k] * x[k * n + col];
  }
  
  let idx = row * n + col;
  carry_out[idx] = sum;
  y[idx] = sum;
}
`,
      numInputs: 2,
      numOutputs: 2,
      hasUniform: false,
      passes: [{ grid: [1, 1] as [number, number] }],
    };

    const scanInfo: ScanBindingInfo = {
      numConsts: 0,
      numCarry: 1, // carry_in
      numX: 1, // x
      numY: 1, // y (carry_out is carry)
      numInputs: 2,
      numOutputs: 2,
    };

    const wrapped = wrapRoutineForScan(mockShader, scanInfo);

    console.log("Wrapped shader:");
    console.log(wrapped.code);

    expect(wrapped.hasUniform).toBe(true);
    expect(wrapped.code).toContain("struct ScanOffsets");
    expect(wrapped.code).toContain("x_offset");
    expect(wrapped.code).toContain("y_offset");
    // carry_in and carry_out should NOT have offsets
    expect(wrapped.code).not.toContain("carry_in_offset");
    expect(wrapped.code).not.toContain("carry_out_offset");

    // Verify x accesses are transformed
    expect(wrapped.code).toContain("x[x_offset +");
    expect(wrapped.code).toContain("y[y_offset +");

    // Verify carry accesses are NOT transformed
    expect(wrapped.code).toContain("carry_in[row * n + k]");
    expect(wrapped.code).toContain("carry_out[idx]");

    console.log("✓ Shader wrapping works correctly");

    // Test offset buffer creation
    const length = 3;
    const xsElemStrides = [16]; // 4x4 matrix = 16 elements
    const ysElemStrides = [16];
    const alignment = backend.getBatchedScanAlignment();

    const { buffer, alignment: actualAlignment } =
      createAllIterationsOffsetsBuffer(
        1,
        1,
        length,
        xsElemStrides,
        ysElemStrides,
        alignment,
      );

    console.log("Offset buffer size:", buffer.length);
    console.log("Alignment:", actualAlignment);

    // Verify offsets
    const view = new DataView(buffer.buffer);
    for (let i = 0; i < length; i++) {
      const xOffset = view.getUint32(i * actualAlignment, true);
      const yOffset = view.getUint32(i * actualAlignment + 4, true);
      console.log(`Iteration ${i}: x_offset=${xOffset}, y_offset=${yOffset}`);
      expect(xOffset).toBe(i * 16);
      expect(yOffset).toBe(i * 16);
    }

    console.log("✓ Offset buffer creation works correctly");
  });
});
