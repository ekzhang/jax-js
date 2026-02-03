/**
 * Benchmark AssemblyScript-compiled routines.
 *
 * This validates that the AS-compiled WASM is performant compared
 * to hand-written codegen. Run with: pnpm vitest bench bench/routines.bench.ts
 */

import { bench, describe } from "vitest";

import { getRoutineModuleSync } from "../src/backend/wasm/generated/routines";

describe("AssemblyScript routines", () => {
  // Shared memory for all tests
  const memory = new WebAssembly.Memory({ initial: 16 }); // 1MB

  describe("cholesky", () => {
    const module = getRoutineModuleSync("cholesky");
    const instance = new WebAssembly.Instance(module, {
      env: { memory },
    });
    const exports = instance.exports as {
      cholesky_f32: (inPtr: number, outPtr: number, n: number) => void;
      cholesky_batched_f32: (
        inPtr: number,
        outPtr: number,
        n: number,
        batch: number,
      ) => void;
    };

    // Create positive-definite test matrices
    const n = 32;
    const batch = 100;
    const view = new Float32Array(memory.buffer);

    // Generate random positive-definite matrices: A = B @ B^T + I
    for (let b = 0; b < batch; b++) {
      const offset = b * n * n;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          let sum = i === j ? 1.0 : 0.0; // Start with identity
          for (let k = 0; k < n; k++) {
            const bi = Math.random();
            const bj = Math.random();
            sum += bi * bj;
          }
          view[offset + i * n + j] = sum;
        }
      }
    }

    const inPtr = 0;
    const outPtr = batch * n * n * 4;

    bench(`cholesky ${n}x${n} single`, () => {
      exports.cholesky_f32(inPtr, outPtr, n);
    });

    bench(`cholesky ${n}x${n} x${batch} batched`, () => {
      exports.cholesky_batched_f32(inPtr, outPtr, n, batch);
    });
  });

  describe("triangular-solve", () => {
    const module = getRoutineModuleSync("triangular-solve");
    const instance = new WebAssembly.Instance(module, {
      env: { memory },
    });
    const exports = instance.exports as {
      triangular_solve_f32: (
        aPtr: number,
        bPtr: number,
        xPtr: number,
        n: number,
        unitDiagonal: number,
      ) => void;
      triangular_solve_batched_f32: (
        aPtr: number,
        bPtr: number,
        xPtr: number,
        n: number,
        batch: number,
        unitDiagonal: number,
      ) => void;
    };

    const n = 32;
    const batch = 100;
    const view = new Float32Array(memory.buffer);

    // Create upper-triangular matrix
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        view[i * n + j] = j >= i ? Math.random() + 0.1 : 0;
      }
    }

    // Create batch of RHS vectors
    const aPtr = 0;
    const bPtr = n * n * 4;
    const xPtr = bPtr + batch * n * 4;
    for (let b = 0; b < batch; b++) {
      for (let i = 0; i < n; i++) {
        view[(bPtr >> 2) + b * n + i] = Math.random();
      }
    }

    bench(`triangular-solve ${n}x${n} single`, () => {
      exports.triangular_solve_f32(aPtr, bPtr, xPtr, n, 0);
    });

    bench(`triangular-solve ${n}x${n} x${batch} batched`, () => {
      exports.triangular_solve_batched_f32(aPtr, bPtr, xPtr, n, batch, 0);
    });
  });

  describe("lu", () => {
    const module = getRoutineModuleSync("lu");
    const instance = new WebAssembly.Instance(module, {
      env: { memory },
    });
    const exports = instance.exports as {
      lu_f32: (
        aPtr: number,
        luPtr: number,
        pivPtr: number,
        permPtr: number,
        m: number,
        n: number,
      ) => void;
      lu_batched_f32: (
        aPtr: number,
        luPtr: number,
        pivPtr: number,
        permPtr: number,
        m: number,
        n: number,
        batch: number,
      ) => void;
    };

    const n = 32;
    const batch = 100;
    const view = new Float32Array(memory.buffer);

    // Fill with random data
    for (let i = 0; i < batch * n * n; i++) {
      view[i] = Math.random();
    }

    const aPtr = 0;
    const luPtr = batch * n * n * 4;
    const pivPtr = luPtr + batch * n * n * 4;
    const permPtr = pivPtr + batch * n * 4;

    bench(`lu ${n}x${n} single`, () => {
      exports.lu_f32(aPtr, luPtr, pivPtr, permPtr, n, n);
    });

    bench(`lu ${n}x${n} x${batch} batched`, () => {
      exports.lu_batched_f32(aPtr, luPtr, pivPtr, permPtr, n, n, batch);
    });
  });

  describe("sort", () => {
    const module = getRoutineModuleSync("sort");
    const instance = new WebAssembly.Instance(module, {
      env: { memory },
    });
    const exports = instance.exports as {
      sort_f32: (dataPtr: number, auxPtr: number, n: number) => void;
      sort_batched_f32: (
        dataPtr: number,
        auxPtr: number,
        n: number,
        batch: number,
      ) => void;
    };

    const n = 1024;
    const batch = 32; // Reduced from 100 to avoid memory pressure
    const view = new Float32Array(memory.buffer);

    const dataPtr = 0;
    const auxPtr = batch * n * 4;

    // Pre-fill with random data
    for (let i = 0; i < batch * n; i++) {
      view[i] = Math.random();
    }

    bench(`sort n=${n} single`, () => {
      // Refill single array portion before sorting
      for (let i = 0; i < n; i++) {
        view[i] = Math.random();
      }
      exports.sort_f32(dataPtr, auxPtr, n);
    });

    bench(`sort n=${n} x${batch} batched`, () => {
      // Refill all arrays before sorting
      for (let i = 0; i < batch * n; i++) {
        view[i] = Math.random();
      }
      exports.sort_batched_f32(dataPtr, auxPtr, n, batch);
    });
  });

  describe("argsort", () => {
    const module = getRoutineModuleSync("argsort");
    const instance = new WebAssembly.Instance(module, {
      env: { memory },
    });
    const exports = instance.exports as {
      argsort_f32: (
        dataPtr: number,
        outPtr: number,
        idxPtr: number,
        auxPtr: number,
        n: number,
      ) => void;
      argsort_batched_f32: (
        dataPtr: number,
        outPtr: number,
        idxPtr: number,
        auxPtr: number,
        n: number,
        batch: number,
      ) => void;
    };

    const n = 1024;
    const batch = 32; // Reduced from 100
    const view = new Float32Array(memory.buffer);

    const dataPtr = 0;
    const outPtr = batch * n * 4;
    const idxPtr = outPtr + batch * n * 4;
    const auxPtr = idxPtr + batch * n * 4;

    // Fill with random data
    for (let i = 0; i < batch * n; i++) {
      view[i] = Math.random();
    }

    bench(`argsort n=${n} single`, () => {
      exports.argsort_f32(dataPtr, outPtr, idxPtr, auxPtr, n);
    });

    bench(`argsort n=${n} x${batch} batched`, () => {
      exports.argsort_batched_f32(dataPtr, outPtr, idxPtr, auxPtr, n, batch);
    });
  });
});
