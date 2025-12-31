import {
  blockUntilReady,
  defaultDevice,
  init,
  numpy as np,
  random,
} from "@jax-js/jax";
import { afterAll, bench, suite } from "vitest";

const devices = await init("webgpu");

suite.skipIf(!devices.includes("webgpu"))("gpu matmul", async () => {
  defaultDevice("webgpu");

  const a2048 = random.uniform(random.key(0), [2048, 2048]);
  const b2048 = random.uniform(random.key(1), [2048, 2048]);
  await blockUntilReady([a2048, b2048]);
  afterAll(() => {
    a2048.dispose();
    b2048.dispose();
  });

  bench("2048x2048", async () => {
    const c = np.matmul(a2048.ref, b2048.ref);
    await c.blockUntilReady();
    c.dispose();
  });

  const a4096 = random.uniform(random.key(0), [4096, 4096]);
  const b4096 = random.uniform(random.key(1), [4096, 4096]);
  await blockUntilReady([a4096, b4096]);
  afterAll(() => {
    a4096.dispose();
    b4096.dispose();
  });

  bench("4096x4096", async () => {
    const c = np.matmul(a4096.ref, b4096.ref);
    await c.blockUntilReady();
    c.dispose();
  });
});
