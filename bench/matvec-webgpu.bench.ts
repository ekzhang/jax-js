import {
  blockUntilReady,
  defaultDevice,
  init,
  numpy as np,
  random,
} from "@jax-js/jax";
import { test } from "vitest";

const devices = await init("webgpu");
const MATVEC_SIZES = [512, 1024, 2048, 4096] as const;

test.skipIf(!devices.includes("webgpu"))(
  "webgpu fp32 matvec",
  async ({ bench }) => {
    defaultDevice("webgpu");

    const inputs = MATVEC_SIZES.map((n) => ({
      n,
      a: random.uniform(random.key(0), [n, n]),
      x: random.uniform(random.key(1), [n]),
    }));
    await blockUntilReady(inputs.flatMap(({ a, x }) => [a, x]));

    try {
      await bench.compare(
        ...inputs.flatMap(({ n, a, x }) => [
          bench(`${n}x${n} @ vector`, async () => {
            const y = np.matvec(a.ref, x.ref);
            await y.blockUntilReady();
            y.dispose();
          }),
          bench(`${n}x${n}.T @ vector`, async () => {
            const y = np.matvec(a.ref.transpose(), x.ref);
            await y.blockUntilReady();
            y.dispose();
          }),
        ]),
        {
          iterations: 3,
          time: 250,
          warmupIterations: 1,
          warmupTime: 50,
        },
      );
    } finally {
      for (const { a, x } of inputs) {
        a.dispose();
        x.dispose();
      }
    }
  },
);
