/**
 * Timing test for WASM SIMD through the full jit path — not for committing.
 *
 * Run with:
 *   npm run build && npx vitest run test/wasm-simd-jit-timing.test.ts
 *
 * To compare SIMD vs scalar, change `const useSimd = isSimdEligible(tune.exp, kernel)`
 * to `const useSimd = false` in src/backend/wasm.ts, rebuild, and rerun.
 */
import { suite, test } from "vitest";
import { defaultDevice, init, jit, numpy as np, random } from "@jax-js/jax";

const devices = await init();

suite.skipIf(!devices.includes("wasm"))("SIMD timing (full jit path)", () => {
  defaultDevice("wasm");

  const sizes = [1_000_000, 10_000_000];

  // Pointwise: x.add(2).mul(3)
  for (const size of sizes) {
    test(`pointwise: x.add(2).mul(3) [${size.toLocaleString()}]`, async () => {
      const x = random.uniform(random.key(0), [size]);
      const f = jit((x: np.Array) => x.add(2).mul(3));

      const warmup = f(x.ref);
      await warmup.blockUntilReady();
      warmup.dispose();

      const N = size <= 1_000_000 ? 100 : 20;
      const start = performance.now();
      for (let i = 0; i < N; i++) {
        const result = f(x.ref);
        await result.blockUntilReady();
        result.dispose();
      }
      const elapsed = performance.now() - start;
      console.log(
        `[pointwise ${size.toLocaleString()}] ${(elapsed / N).toFixed(4)}ms/call (${N} runs)`,
      );
      x.dispose();
    });
  }

  // Reduction: x.sum()
  for (const size of sizes) {
    test(`reduction sum: x.sum() [${size.toLocaleString()}]`, async () => {
      const x = random.uniform(random.key(0), [size]);
      const f = jit((x: np.Array) => x.sum());

      const warmup = f(x.ref);
      await warmup.blockUntilReady();
      warmup.dispose();

      const N = size <= 1_000_000 ? 100 : 20;
      const start = performance.now();
      for (let i = 0; i < N; i++) {
        const result = f(x.ref);
        await result.blockUntilReady();
        result.dispose();
      }
      const elapsed = performance.now() - start;
      console.log(
        `[sum ${size.toLocaleString()}] ${(elapsed / N).toFixed(4)}ms/call (${N} runs)`,
      );
      x.dispose();
    });
  }

  // Reduction: row sum of 2D array
  for (const size of [1000, 4000]) {
    test(`reduction row sum: [${size},${size}].sum(axis=1)`, async () => {
      const x = random.uniform(random.key(0), [size, size]);
      const f = jit((x: np.Array) => x.sum(1));

      const warmup = f(x.ref);
      await warmup.blockUntilReady();
      warmup.dispose();

      const N = 20;
      const start = performance.now();
      for (let i = 0; i < N; i++) {
        const result = f(x.ref);
        await result.blockUntilReady();
        result.dispose();
      }
      const elapsed = performance.now() - start;
      console.log(
        `[row sum ${size}x${size}] ${(elapsed / N).toFixed(4)}ms/call (${N} runs)`,
      );
      x.dispose();
    });
  }

  // Reduction: column sum (axis=0, non-contiguous ridx → gather path)
  for (const size of [100, 500, 1000, 2000, 4000]) {
    test(`reduction col sum (gather): [${size},${size}].sum(axis=0)`, async () => {
      const x = random.uniform(random.key(0), [size, size]);
      const f = jit((x: np.Array) => x.sum(0));

      const warmup = f(x.ref);
      await warmup.blockUntilReady();
      warmup.dispose();

      const N = 20;
      const start = performance.now();
      for (let i = 0; i < N; i++) {
        const result = f(x.ref);
        await result.blockUntilReady();
        result.dispose();
      }
      const elapsed = performance.now() - start;
      console.log(
        `[col sum ${size}x${size}] ${(elapsed / N).toFixed(4)}ms/call (${N} runs)`,
      );
      x.dispose();
    });
  }

  // Reduction: max
  for (const size of sizes) {
    test(`reduction max: x.max() [${size.toLocaleString()}]`, async () => {
      const x = random.uniform(random.key(0), [size]);
      const f = jit((x: np.Array) => x.max());

      const warmup = f(x.ref);
      await warmup.blockUntilReady();
      warmup.dispose();

      const N = size <= 1_000_000 ? 100 : 20;
      const start = performance.now();
      for (let i = 0; i < N; i++) {
        const result = f(x.ref);
        await result.blockUntilReady();
        result.dispose();
      }
      const elapsed = performance.now() - start;
      console.log(
        `[max ${size.toLocaleString()}] ${(elapsed / N).toFixed(4)}ms/call (${N} runs)`,
      );
      x.dispose();
    });
  }

  // Contiguous vs gather comparison:
  // Row sum (ridx = last axis, stride 1 → wide load)
  // vs column sum (ridx = first axis via transpose, stride > 1 → gather)
  for (const n of [256, 512, 1024, 2048]) {
    test(`contiguous row sum [${n},${n}].sum(axis=1)`, async () => {
      const x = random.uniform(random.key(0), [n, n]);
      const f = jit((x: np.Array) => x.sum(1));

      const warmup = f(x.ref);
      await warmup.blockUntilReady();
      warmup.dispose();

      const N = n <= 512 ? 50 : 10;
      const start = performance.now();
      for (let i = 0; i < N; i++) {
        const result = f(x.ref);
        await result.blockUntilReady();
        result.dispose();
      }
      const elapsed = performance.now() - start;
      console.log(`[row sum ${n}x${n}] ${(elapsed / N).toFixed(4)}ms/call (${N} runs)`);
      x.dispose();
    });

    test(`gather flipped row sum [${n},${n}].flip(1).sum(axis=1)`, async () => {
      const x = random.uniform(random.key(0), [n, n]);
      // Flip along axis 1 reverses each row. ridx now has stride -1,
      // so isContiguousWrt returns false → gather path.
      // But cache behavior is similar to the contiguous case since we're
      // still reading within contiguous rows, just in reverse order.
      const f = jit((x: np.Array) => np.flip(x, [1]).sum(1));

      const warmup = f(x.ref);
      await warmup.blockUntilReady();
      warmup.dispose();

      const N = n <= 512 ? 50 : 10;
      const start = performance.now();
      for (let i = 0; i < N; i++) {
        const result = f(x.ref);
        await result.blockUntilReady();
        result.dispose();
      }
      const elapsed = performance.now() - start;
      console.log(`[flipped row sum ${n}x${n}] ${(elapsed / N).toFixed(4)}ms/call (${N} runs)`);
      x.dispose();
    });
  }
});
