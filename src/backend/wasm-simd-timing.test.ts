/**
 * Timing test for WASM SIMD — not for committing.
 *
 * Run with: npx vitest run src/backend/wasm-simd-timing.test.ts
 *
 * To compare SIMD vs scalar, change `const useSimd = isSimdEligible(tune.exp, kernel)`
 * to `const useSimd = false` in wasm.ts and rerun.
 */
import { expect, suite, test } from "vitest";

import {
  accessorGlobal,
  AluExp,
  AluOp,
  AluVar,
  DType,
  Kernel,
  Reduction,
} from "../alu";
import { getBackend, init } from "../backend";
import { ShapeTracker } from "../shape";
import { isSimdEligible } from "./wasm";
import { tuneNullopt } from "../tuner";

await init();

suite("SIMD timing (backend level)", () => {
  for (const size of [
    1_000, 10_000, 100_000, 1_000_000, 4_000_000, 100_000_000,
  ]) {
    test(`backend: x.add(2) [${size.toLocaleString()}]`, async () => {
      const backend = getBackend("wasm");
      const data = new Float32Array(size);
      for (let i = 0; i < size; i++) data[i] = i;
      const a = backend.malloc(size * 4, new Uint8Array(data.buffer));
      const out = backend.malloc(size * 4);

      try {
        const shape = ShapeTracker.fromShape([size]);
        const gidx = AluVar.gidx;
        const arg = accessorGlobal(DType.Float32, 0, shape, [gidx]);
        const kernel = new Kernel(1, size, AluExp.add(arg, AluExp.f32(2)));
        console.log(`isSimdEligible: ${isSimdEligible(kernel.exp, kernel)}`);

        const exe = await backend.prepareKernel(kernel);

        // Warm up
        backend.dispatch(exe, [a], [out]);

        const N = size <= 10_000 ? 1000 : size <= 1_000_000 ? 200 : 50;
        const start = performance.now();
        for (let i = 0; i < N; i++) {
          backend.dispatch(exe, [a], [out]);
        }
        const elapsed = performance.now() - start;
        console.log(
          `[${size.toLocaleString()}] ${(elapsed / N).toFixed(4)}ms/dispatch (${N} runs)`,
        );
      } finally {
        backend.decRef(a);
        backend.decRef(out);
      }
    });
  }
});

suite("SIMD timing: contiguous vs gather reduction", () => {
  // Compare contiguous wide load (row sum) vs non-contiguous gather (column sum)
  // on the same data. Both do the same amount of arithmetic, but the memory
  // access pattern differs: contiguous reads 4 adjacent floats per v128.load,
  // gather reads 4 separate floats and packs them.

  for (const rows of [256, 1024, 4096]) {
    const cols = 256;
    const totalElements = rows * cols;

    test(`contiguous row sum [${rows}x${cols}]`, async () => {
      const backend = getBackend("wasm");
      const data = new Float32Array(totalElements);
      for (let i = 0; i < totalElements; i++) data[i] = 1;
      const a = backend.malloc(totalElements * 4, new Uint8Array(data.buffer));
      const out = backend.malloc(rows * 4);

      try {
        // Row sum: [rows, cols], reduce along cols (last axis).
        // ridx has stride 1 → contiguous → wide v128.load
        const st = ShapeTracker.fromShape([rows, cols]);
        const exp = AluExp.globalView(DType.Float32, 0, st, [
          AluVar.gidx,
          AluVar.ridx,
        ]);
        const kernel = new Kernel(
          1, rows, exp,
          new Reduction(DType.Float32, AluOp.Add, cols),
        );
        const tune = tuneNullopt(kernel);
        console.log(`row sum [${rows}x${cols}] simd=${isSimdEligible(tune.exp, kernel)}`);

        const exe = await backend.prepareKernel(kernel);
        backend.dispatch(exe, [a], [out]);

        const N = rows <= 256 ? 100 : 20;
        const start = performance.now();
        for (let i = 0; i < N; i++) {
          backend.dispatch(exe, [a], [out]);
        }
        const elapsed = performance.now() - start;
        console.log(`[row sum ${rows}x${cols}] ${(elapsed / N).toFixed(4)}ms/dispatch (${N} runs)`);
      } finally {
        backend.decRef(a);
        backend.decRef(out);
      }
    });

    test(`gather col sum [${rows}x${cols}] (permuted)`, async () => {
      const backend = getBackend("wasm");
      const data = new Float32Array(totalElements);
      for (let i = 0; i < totalElements; i++) data[i] = 1;
      const a = backend.malloc(totalElements * 4, new Uint8Array(data.buffer));
      const out = backend.malloc(cols * 4);

      try {
        // Column sum via permute: [rows, cols].permute([1,0]) gives shape [cols, rows]
        // with strides [1, cols]. ridx has stride cols → non-contiguous → gather
        const st = ShapeTracker.fromShape([rows, cols]).permute([1, 0]);
        const exp = AluExp.globalView(DType.Float32, 0, st, [
          AluVar.gidx,
          AluVar.ridx,
        ]);
        const kernel = new Kernel(
          1, cols, exp,
          new Reduction(DType.Float32, AluOp.Add, rows),
        );
        const tune = tuneNullopt(kernel);
        console.log(`col sum [${rows}x${cols}] simd=${isSimdEligible(tune.exp, kernel)}`);

        const exe = await backend.prepareKernel(kernel);
        backend.dispatch(exe, [a], [out]);

        const N = rows <= 256 ? 100 : 20;
        const start = performance.now();
        for (let i = 0; i < N; i++) {
          backend.dispatch(exe, [a], [out]);
        }
        const elapsed = performance.now() - start;
        console.log(`[col sum ${rows}x${cols}] ${(elapsed / N).toFixed(4)}ms/dispatch (${N} runs)`);
      } finally {
        backend.decRef(a);
        backend.decRef(out);
      }
    });
  }
});
