import { blockUntilReady, defaultDevice, init, numpy as np } from "@jax-js/jax";
import { test } from "vitest";

const devices = await init("wasm");
const MATVEC_SIZES = [512, 1024, 2048, 4096] as const;

function makeMatrix(n: number): np.Array {
  const data = new Float32Array(n * n);
  for (let i = 0; i < data.length; i++) data[i] = (i % 7) - 3;
  return np.array(data, { shape: [n, n], device: "wasm" });
}

function makeVector(n: number): np.Array {
  const data = new Float32Array(n);
  for (let i = 0; i < data.length; i++) data[i] = (i % 5) - 2;
  return np.array(data, { shape: [n], device: "wasm" });
}

test.skipIf(!devices.includes("wasm"))(
  "wasm fp32 matvec",
  async ({ bench }) => {
    defaultDevice("wasm");

    const inputs = MATVEC_SIZES.map((n) => ({
      n,
      a: makeMatrix(n),
      x: makeVector(n),
    }));
    await blockUntilReady(inputs);

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
