import {
  blockUntilReady,
  defaultDevice,
  init,
  lax,
  numpy as np,
  random,
} from "@jax-js/jax";
import { afterAll, bench, suite } from "vitest";

const devices = await init("webgpu");

suite.skipIf(!devices.includes("webgpu"))("webgpu cholesky", async () => {
  defaultDevice("webgpu");

  const inputs = [128, 256, 512, 1024, 2048].map((n) => {
    const a = random.normal(random.key(n), [n, n]);
    const spd = np
      .matmul(np.matrixTranspose(a.ref), a.ref)
      .add(np.eye(n).mul(n));
    a.dispose();
    return { n, spd };
  });

  await blockUntilReady(inputs.map(({ spd }) => spd));

  afterAll(() => {
    for (const { spd } of inputs) spd.dispose();
  });

  for (const { n, spd } of inputs) {
    bench(`cholesky ${n}x${n}`, async () => {
      const l = lax.linalg.cholesky(spd.ref);
      await l.blockUntilReady();
      l.dispose();
    });
  }
});
