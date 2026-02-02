/**
 * Benchmark for lax.scan to measure JS loop overhead
 *
 * Run with: pnpm test run bench/scan.bench.ts
 */

import { beforeAll, bench, describe } from "vitest";

import { defaultDevice, init, jit, lax, numpy as np } from "../src";

describe("scan benchmarks", () => {
  beforeAll(async () => {
    await init();
  });

  describe("wasm backend", () => {
    beforeAll(() => {
      defaultDevice("wasm");
    });

    bench("cumsum 100 iterations", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry.ref];
      };
      const initCarry = np.array([0.0]);
      const xs = np.ones([100, 1]);
      const [finalCarry, _outputs] = await lax.scan(step, initCarry, xs);
      await finalCarry.data();
    });

    bench("cumsum 1000 iterations", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry.ref];
      };
      const initCarry = np.array([0.0]);
      const xs = np.ones([1000, 1]);
      const [finalCarry, _outputs] = await lax.scan(step, initCarry, xs);
      await finalCarry.data();
    });

    bench("cumsum jit step 100 iterations", async () => {
      const step = jit((carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry.ref];
      });
      const initCarry = np.array([0.0]);
      const xs = np.ones([100, 1]);
      const [finalCarry, _outputs] = await lax.scan(step, initCarry, xs);
      await finalCarry.data();
    });

    bench("cumsum jit step 1000 iterations", async () => {
      const step = jit((carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry.ref];
      });
      const initCarry = np.array([0.0]);
      const xs = np.ones([1000, 1]);
      const [finalCarry, _outputs] = await lax.scan(step, initCarry, xs);
      await finalCarry.data();
    });

    // Baseline: what does 1000 adds cost without scan overhead?
    bench("baseline 1000 sequential adds (no scan)", async () => {
      let carry = np.array([0.0]);
      for (let i = 0; i < 1000; i++) {
        carry = np.add(carry.ref, np.array([1.0]));
      }
      await carry.data();
    });

    // Compare with matmul-equivalent compute
    bench("matmul 64x64 (for compute reference)", async () => {
      const a = np.ones([64, 64]);
      const b = np.ones([64, 64]);
      const c = np.matmul(a, b);
      await c.data();
    });
  });

  describe("cpu backend", () => {
    beforeAll(() => {
      defaultDevice("cpu");
    });

    bench("cumsum 100 iterations cpu", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry.ref];
      };
      const initCarry = np.array([0.0]);
      const xs = np.ones([100, 1]);
      const [finalCarry, _outputs] = await lax.scan(step, initCarry, xs);
      await finalCarry.data();
    });

    bench("cumsum 1000 iterations cpu", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry.ref];
      };
      const initCarry = np.array([0.0]);
      const xs = np.ones([1000, 1]);
      const [finalCarry, _outputs] = await lax.scan(step, initCarry, xs);
      await finalCarry.data();
    });
  });
});
