import { blockUntilReady, defaultDevice, init, numpy as np } from "@jax-js/jax";
import { test } from "vitest";

const devices = await init("wasm", "webgpu");

const cases = [
  { name: "256", shape: [256], phase: 0.0 },
  { name: "1024", shape: [1024], phase: 0.01 },
  { name: "1000", shape: [1000], phase: 0.02 },
  { name: "64x256", shape: [64, 256], phase: 0.03 },
  { name: "100k", shape: [100_000], phase: 0.04 },
  { name: "1m", shape: [1_000_000], phase: 0.05 },
] as const;

function prod(shape: readonly number[]) {
  return shape.reduce((a, b) => a * b, 1);
}

function makeData(shape: readonly number[], phase: number) {
  const data = new Float32Array(prod(shape));
  for (let i = 0; i < data.length; i++) {
    data[i] =
      Math.sin((i + 1) * (0.013 + phase)) +
      0.25 * Math.cos((i + 3) * (0.037 + phase));
  }
  return data;
}

for (const device of ["wasm", "webgpu"] as const) {
  test.skipIf(!devices.includes(device))(`${device} fft`, async ({ bench }) => {
    defaultDevice(device);

    const inputs = cases.map(({ shape, phase }) => ({
      real: np.array(makeData(shape, phase), { shape: [...shape] }),
      imag: np.array(makeData(shape, phase + 0.01), { shape: [...shape] }),
    }));
    await blockUntilReady(inputs.flatMap(({ real, imag }) => [real, imag]));

    try {
      await bench.compare(
        ...cases.map((benchmarkCase, i) =>
          bench(`fft ${benchmarkCase.name}`, async () => {
            const y = np.fft.fft({
              real: inputs[i].real.ref,
              imag: inputs[i].imag.ref,
            });
            await blockUntilReady([y.real, y.imag]);
            y.real.dispose();
            y.imag.dispose();
          }),
        ),
        {
          iterations: 3,
          warmupIterations: 1,
        },
      );
    } finally {
      for (const { real, imag } of inputs) {
        real.dispose();
        imag.dispose();
      }
    }
  });
}
