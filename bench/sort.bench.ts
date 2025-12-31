import {
  blockUntilReady,
  defaultDevice,
  init,
  numpy as np,
  random,
} from "@jax-js/jax";
import { afterAll, bench, suite } from "vitest";

const devices = await init("wasm", "webgpu");

suite.skipIf(!devices.includes("webgpu"))("gpu sort/argsort", async () => {
  defaultDevice("webgpu");

  const batch = 32768; // GPU supports much more parallelism.
  const size = 1024;
  const a = random.uniform(random.key(0), [batch, size]);
  await blockUntilReady(a);
  afterAll(() => {
    a.dispose();
  });

  bench("sort", async () => {
    const c = np.sort(a.ref);
    await c.blockUntilReady();
    c.dispose();
  });

  bench("argsort", async () => {
    const c = np.argsort(a.ref);
    await c.blockUntilReady();
    c.dispose();
  });
});

suite.skipIf(!devices.includes("wasm"))("cpu sort/argsort", async () => {
  defaultDevice("wasm");

  const batch = 128;
  const size = 1024;
  const a = random.uniform(random.key(0), [batch, size]);
  await blockUntilReady(a);
  afterAll(() => {
    a.dispose();
  });

  bench("sort", async () => {
    const c = np.sort(a.ref);
    await c.blockUntilReady();
    c.dispose();
  });

  bench("argsort", async () => {
    const c = np.argsort(a.ref);
    await c.blockUntilReady();
    c.dispose();
  });
});
