import {
  blockUntilReady,
  defaultDevice,
  init,
  numpy as np,
  random,
} from "@jax-js/jax";
import { test } from "vitest";

const devices = await init("webgpu");
const MATMUL_SIZES = [2048, 4096] as const;

test.skipIf(!devices.includes("webgpu"))(
  "webgpu fp32 matmul",
  async ({ bench }) => {
    defaultDevice("webgpu");

    const matrices = MATMUL_SIZES.map((n) => ({
      n,
      a: random.uniform(random.key(0), [n, n]),
      b: random.uniform(random.key(1), [n, n]),
    }));
    await blockUntilReady(matrices.flatMap(({ a, b }) => [a, b]));

    try {
      await bench.compare(
        ...matrices.map(({ n, a, b }) =>
          bench(`${n}x${n}`, async () => {
            const c = np.matmul(a.ref, b.ref);
            await c.blockUntilReady();
            c.dispose();
          }),
        ),
        {
          iterations: 3,
          time: 250,
          warmupIterations: 1,
          warmupTime: 50,
        },
      );
    } finally {
      for (const { a, b } of matrices) {
        a.dispose();
        b.dispose();
      }
    }
  },
);
