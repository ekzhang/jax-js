import {
  blockUntilReady,
  defaultDevice,
  init,
  numpy as np,
  random,
} from "@jax-js/jax";
import { afterAll, bench, suite } from "vitest";

const devices = await init("webgpu");
const MATMUL_SIZES = [2048, 4096] as const;

suite.skipIf(!devices.includes("webgpu"))("webgpu fp32 matmul", async () => {
  defaultDevice("webgpu");

  const matrices = MATMUL_SIZES.map((n) => ({
    n,
    a: random.uniform(random.key(0), [n, n]),
    b: random.uniform(random.key(1), [n, n]),
  }));
  await blockUntilReady(matrices.flatMap(({ a, b }) => [a, b]));

  afterAll(() => {
    for (const { a, b } of matrices) {
      a.dispose();
      b.dispose();
    }
  });

  for (const { n, a, b } of matrices) {
    bench(`${n}x${n}`, async () => {
      const c = np.matmul(a.ref, b.ref);
      await c.blockUntilReady();
      c.dispose();
    });
  }
});
