/**
 * Timing test for WASM SIMD matmul — not for committing.
 *
 * Run with:
 *   npm run build && npx vitest run test/wasm-simd-matmul-timing.test.ts
 */
import { suite, test } from "vitest";
import { defaultDevice, init, jit, numpy as np, random } from "@jax-js/jax";

const devices = await init();

suite.skipIf(!devices.includes("wasm"))("WASM SIMD matmul timing", () => {
  defaultDevice("wasm");

  for (const n of [32, 64, 128, 256]) {
    test(`matmul ${n}x${n}`, async () => {
      const a = random.uniform(random.key(0), [n, n]);
      const b = random.uniform(random.key(1), [n, n]);

      const f = jit((a: np.Array, b: np.Array) => np.matmul(a, b));

      // Warm up
      const warmup = f(a.ref, b.ref);
      await warmup.blockUntilReady();
      warmup.dispose();

      const N = n <= 64 ? 500 : 100;
      const start = performance.now();
      for (let i = 0; i < N; i++) {
        const result = f(a.ref, b.ref);
        await result.blockUntilReady();
        result.dispose();
      }
      const elapsed = performance.now() - start;
      const msPerCall = elapsed / N;
      const flops = 2 * n * n * n;
      const gflops = flops / (msPerCall / 1000) / 1e9;
      console.log(
        `[matmul ${n}x${n}] ${msPerCall.toFixed(3)}ms/call (${gflops.toFixed(2)} GFLOP/s, ${N} runs)`,
      );

      a.dispose();
      b.dispose();
    });
  }
});
