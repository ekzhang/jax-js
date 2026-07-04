import {
  blockUntilReady,
  defaultDevice,
  init,
  numpy as np,
  random,
} from "@jax-js/jax";
import { afterAll, bench, suite } from "vitest";

const devices = await init("wasm", "webgpu");
const cases = [
  { name: "128x128", shape: [128, 128], key: 0 },
  { name: "1024x128", shape: [1024, 128], key: 1 },
  { name: "32768x256", shape: [32768, 256], key: 2 },
] as const;

for (const device of ["wasm", "webgpu"] as const) {
  suite.skipIf(!devices.includes(device))(`${device} svd`, async () => {
    defaultDevice(device);

    const matrices = cases.map(({ shape, key }) =>
      random.normal(random.key(key), [...shape]),
    );
    await blockUntilReady(matrices);

    afterAll(() => {
      for (const a of matrices) a.dispose();
    });

    for (let i = 0; i < cases.length; i++) {
      const { name } = cases[i];
      const a = matrices[i];

      bench(`svdvals ${name}`, async () => {
        const s = np.linalg.svdvals(a.ref);
        await s.blockUntilReady();
        s.dispose();
      });

      bench(`thin svd ${name}`, async () => {
        const [u, s, vh] = np.linalg.svd(a.ref);
        await blockUntilReady([u, s, vh]);
        u.dispose();
        s.dispose();
        vh.dispose();
      });
    }
  });
}
