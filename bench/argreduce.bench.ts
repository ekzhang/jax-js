import {
  blockUntilReady,
  defaultDevice,
  init,
  numpy as np,
  random,
} from "@jax-js/jax";
import { test } from "vitest";

const devices = await init();

// Benchmark argreduce operations on WebGPU
test.skipIf(!devices.includes("webgpu"))("gpu argreduce", async ({ bench }) => {
  defaultDevice("webgpu");

  // 1D array benchmarks
  const arr10k = random.uniform(random.key(0), [10000]);
  await blockUntilReady([arr10k]);
  // 2D array benchmarks (reduction along axis)
  const arr2d = random.uniform(random.key(1), [100, 1000]);
  await blockUntilReady([arr2d]);

  try {
    await bench("argmax 10k elements", async () => {
      const result = np.argmax(arr10k.ref);
      await result.blockUntilReady();
      result.dispose();
    }).run();

    await bench("argmin 10k elements", async () => {
      const result = np.argmin(arr10k.ref);
      await result.blockUntilReady();
      result.dispose();
    }).run();

    await bench("argmax along axis (100x1000)", async () => {
      const result = np.argmax(arr2d.ref, 1);
      await result.blockUntilReady();
      result.dispose();
    }).run();

    await bench("argmin along axis (100x1000)", async () => {
      const result = np.argmin(arr2d.ref, 1);
      await result.blockUntilReady();
      result.dispose();
    }).run();
  } finally {
    arr10k.dispose();
    arr2d.dispose();
  }
});
