import {
  blockUntilReady,
  defaultDevice,
  init,
  numpy as np,
  random,
} from "@jax-js/jax";
import { test } from "vitest";

const devices = await init("wasm", "webgpu");

test.skipIf(!devices.includes("webgpu"))(
  "gpu sort/argsort",
  async ({ bench }) => {
    defaultDevice("webgpu");

    const batch = 32768; // GPU supports much more parallelism.
    const size = 1024;
    const a = random.uniform(random.key(0), [batch, size]);
    await blockUntilReady(a);
    try {
      await bench.compare(
        bench("sort", async () => {
          const c = np.sort(a.ref);
          await c.blockUntilReady();
          c.dispose();
        }),
        bench("argsort", async () => {
          const c = np.argsort(a.ref);
          await c.blockUntilReady();
          c.dispose();
        }),
        { iterations: 10, warmupIterations: 5 },
      );
    } finally {
      a.dispose();
    }
  },
);

test.skipIf(!devices.includes("wasm"))(
  "cpu sort/argsort",
  async ({ bench }) => {
    defaultDevice("wasm");

    const batch = 128;
    const size = 1024;
    const a = random.uniform(random.key(0), [batch, size]);
    await blockUntilReady(a);
    try {
      await bench.compare(
        bench("sort", async () => {
          const c = np.sort(a.ref);
          await c.blockUntilReady();
          c.dispose();
        }),
        bench("argsort", async () => {
          const c = np.argsort(a.ref);
          await c.blockUntilReady();
          c.dispose();
        }),
        { iterations: 10, warmupIterations: 5 },
      );
    } finally {
      a.dispose();
    }
  },
);
