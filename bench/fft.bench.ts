import { blockUntilReady, defaultDevice, init, numpy as np } from "@jax-js/jax";
import { afterAll, bench, suite } from "vitest";

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
  suite.skipIf(!devices.includes(device))(`${device} fft`, async () => {
    defaultDevice(device);

    const inputs = cases.map(({ shape, phase }) => ({
      real: np.array(makeData(shape, phase), { shape: [...shape] }),
      imag: np.array(makeData(shape, phase + 0.01), { shape: [...shape] }),
    }));
    await blockUntilReady(inputs.flatMap(({ real, imag }) => [real, imag]));

    afterAll(() => {
      for (const { real, imag } of inputs) {
        real.dispose();
        imag.dispose();
      }
    });

    for (let i = 0; i < cases.length; i++) {
      bench(`fft ${cases[i].name}`, async () => {
        const y = np.fft.fft({
          real: inputs[i].real.ref,
          imag: inputs[i].imag.ref,
        });
        await blockUntilReady([y.real, y.imag]);
        y.real.dispose();
        y.imag.dispose();
      });
    }
  });
}
