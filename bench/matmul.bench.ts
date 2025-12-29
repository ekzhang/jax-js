import { defaultDevice, init, numpy as np } from "@jax-js/jax";
import { bench, suite } from "vitest";

const devices = await init();

if (devices.includes("webgpu")) {
  suite("webgpu matrix multiplication", async () => {
    defaultDevice("webgpu");

    bench("2048x2048", async () => {
      const a = np.ones([2048, 2048]);
      const b = np.full([2048, 2048], 2);
      a._realizeSource();
      b._realizeSource();
      const c = np.matmul(a, b);
      await c.blockUntilReady();
    });

    bench("4096x4096", async () => {
      const a = np.ones([4096, 4096]);
      const b = np.full([4096, 4096], 2);
      a._realizeSource();
      b._realizeSource();
      const c = np.matmul(a, b);
      await c.blockUntilReady();
    });
  });
}
