import {
  blockUntilReady,
  defaultDevice,
  init,
  numpy as np,
  random,
} from "@jax-js/jax";
import { test } from "vitest";

const devices = await init("wasm", "webgpu");
const cases = [
  { name: "128x128", shape: [128, 128], key: 0 },
  { name: "1024x128", shape: [1024, 128], key: 1 },
  { name: "32768x256", shape: [32768, 256], key: 2 },
] as const;

for (const device of ["wasm", "webgpu"] as const) {
  test.skipIf(!devices.includes(device))(`${device} svd`, async ({ bench }) => {
    defaultDevice(device);

    const matrices = cases.map(({ shape, key }) =>
      random.normal(random.key(key), [...shape]),
    );
    await blockUntilReady(matrices);

    try {
      await bench.compare(
        ...cases.flatMap(({ name }, i) => {
          const a = matrices[i];
          return [
            bench(`svdvals ${name}`, async () => {
              const s = np.linalg.svdvals(a.ref);
              await s.blockUntilReady();
              s.dispose();
            }),
            bench(`thin svd ${name}`, async () => {
              const [u, s, vh] = np.linalg.svd(a.ref);
              await blockUntilReady([u, s, vh]);
              u.dispose();
              s.dispose();
              vh.dispose();
            }),
          ];
        }),
        {
          iterations: 3,
          time: 250,
          warmupIterations: 1,
          warmupTime: 50,
        },
      );
    } finally {
      for (const a of matrices) a.dispose();
    }
  });
}
