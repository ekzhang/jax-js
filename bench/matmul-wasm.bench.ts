import { blockUntilReady, defaultDevice, init, numpy as np } from "@jax-js/jax";
import { test } from "vitest";

const devices = await init("wasm");
const MATMUL_SIZES = [64, 128, 256, 512, 1024, 2048, 4096] as const;

function makeMatrix(n: number): np.Array {
  const data = new Float32Array(n * n);
  for (let i = 0; i < data.length; i++) data[i] = (i % 7) - 3;
  return np.array(data, { shape: [n, n], device: "wasm" });
}

test.skipIf(!devices.includes("wasm"))(
  "wasm fp32 matmul",
  async ({ bench }) => {
    defaultDevice("wasm");

    const matrices = MATMUL_SIZES.map((n) => ({
      n,
      a: makeMatrix(n),
      b: makeMatrix(n),
    }));
    await blockUntilReady(matrices);

    try {
      await bench.compare(
        ...matrices.flatMap(({ n, a, b }) => [
          bench(`${n}x${n}`, async () => {
            const c = np.matmul(a.ref, b.ref);
            await c.blockUntilReady();
            c.dispose();
          }),
          bench(`${n}x${n} @ transpose`, async () => {
            const c = np.matmul(a.ref, b.ref.transpose());
            await c.blockUntilReady();
            c.dispose();
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
      for (const { a, b } of matrices) {
        a.dispose();
        b.dispose();
      }
    }
  },
);
