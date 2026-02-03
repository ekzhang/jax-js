/**
 * Test for AssemblyScript-compiled Cholesky routine.
 *
 * This verifies that the WASM compiled from AssemblyScript produces
 * correct results matching the existing implementation.
 */

import { describe, expect, it } from "vitest";

import { getRoutineModuleSync } from "../backend/wasm/generated/routines";

describe("AssemblyScript Cholesky", () => {
  it("should compute correct Cholesky decomposition for 3x3 matrix", async () => {
    // Load the compiled WASM from the generated module
    const module = getRoutineModuleSync("cholesky");

    // Create shared memory (1 page = 64KB)
    const memory = new WebAssembly.Memory({ initial: 1 });

    // Instantiate with our memory
    const instance = await WebAssembly.instantiate(module, {
      env: {
        memory,
        abort: () => {
          throw new Error("WASM abort called");
        },
      },
    });

    const exports = instance.exports as {
      cholesky_f32: (inPtr: number, outPtr: number, n: number) => void;
    };

    // Test matrix: A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
    // Expected L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
    // Such that A = L @ L^T

    const n = 3;
    const inPtr = 0;
    const outPtr = n * n * 4; // After input matrix

    const view = new Float32Array(memory.buffer);

    // Input matrix (row-major)
    view[0] = 4;
    view[1] = 12;
    view[2] = -16;
    view[3] = 12;
    view[4] = 37;
    view[5] = -43;
    view[6] = -16;
    view[7] = -43;
    view[8] = 98;

    // Call Cholesky
    exports.cholesky_f32(inPtr, outPtr, n);

    // Read output
    const outOffset = outPtr / 4;
    const L = [
      [view[outOffset + 0], view[outOffset + 1], view[outOffset + 2]],
      [view[outOffset + 3], view[outOffset + 4], view[outOffset + 5]],
      [view[outOffset + 6], view[outOffset + 7], view[outOffset + 8]],
    ];

    // Expected L (lower triangular)
    expect(L[0][0]).toBeCloseTo(2, 5);
    expect(L[0][1]).toBeCloseTo(0, 5);
    expect(L[0][2]).toBeCloseTo(0, 5);
    expect(L[1][0]).toBeCloseTo(6, 5);
    expect(L[1][1]).toBeCloseTo(1, 5);
    expect(L[1][2]).toBeCloseTo(0, 5);
    expect(L[2][0]).toBeCloseTo(-8, 5);
    expect(L[2][1]).toBeCloseTo(5, 5);
    expect(L[2][2]).toBeCloseTo(3, 5);
  });

  it("should match existing implementation for random 4x4 matrix", async () => {
    const module = getRoutineModuleSync("cholesky");
    const memory = new WebAssembly.Memory({ initial: 1 });

    const instance = await WebAssembly.instantiate(module, {
      env: {
        memory,
        abort: () => {
          throw new Error("WASM abort called");
        },
      },
    });

    const exports = instance.exports as {
      cholesky_f32: (inPtr: number, outPtr: number, n: number) => void;
    };

    // Generate a random positive-definite matrix: A = B @ B^T
    const n = 4;
    const B = new Float32Array(n * n);
    for (let i = 0; i < n * n; i++) {
      B[i] = Math.random() * 2 - 1;
    }

    // Compute A = B @ B^T
    const A = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += B[i * n + k] * B[j * n + k];
        }
        A[i * n + j] = sum;
      }
    }

    // Copy A to WASM memory
    const view = new Float32Array(memory.buffer);
    const inPtr = 0;
    const outPtr = n * n * 4;
    view.set(A, 0);

    // Call Cholesky
    exports.cholesky_f32(inPtr, outPtr, n);

    // Read L
    const outOffset = outPtr / 4;
    const L = new Float32Array(n * n);
    for (let i = 0; i < n * n; i++) {
      L[i] = view[outOffset + i];
    }

    // Verify L @ L^T ≈ A
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += L[i * n + k] * L[j * n + k];
        }
        expect(sum).toBeCloseTo(A[i * n + j], 4);
      }
    }
  });
});
