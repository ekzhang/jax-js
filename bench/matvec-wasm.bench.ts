import { blockUntilReady, defaultDevice, init, numpy as np } from "@jax-js/jax";
import { afterAll, bench, suite } from "vitest";

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

suite.skipIf(!devices.includes("wasm"))("wasm fp32 matvec", async () => {
  defaultDevice("wasm");

  const inputs = MATVEC_SIZES.map((n) => ({
    n,
    a: makeMatrix(n),
    x: makeVector(n),
  }));
  await blockUntilReady(inputs);

  afterAll(() => {
    for (const { a, x } of inputs) {
      a.dispose();
      x.dispose();
    }
  });

  for (const { n, a, x } of inputs) {
    bench(`${n}x${n} @ vector`, async () => {
      const y = np.matvec(a.ref, x.ref);
      await y.blockUntilReady();
      y.dispose();
    });

    bench(`${n}x${n}.T @ vector`, async () => {
      const y = np.matvec(a.ref.transpose(), x.ref);
      await y.blockUntilReady();
      y.dispose();
    });
  }
});
