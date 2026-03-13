// If you stick to the CPU and Wasm backends, Node.js is fully supported.

import { defaultDevice, Device, init, jit, nn, numpy as np } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const nodeDevices: Device[] = ["cpu", "wasm"];
const devicesAvailable = await init(...nodeDevices);

suite.each(nodeDevices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  test("basic operations work", async () => {
    const x = np.array([1, 2, 3]);
    const y = x.add(1);
    expect(await y.ref.jsAsync()).toEqual([2, 3, 4]);
    expect(y.js()).toEqual([2, 3, 4]);
  });

  test("two-layer MLP JIT", async () => {
    const x = np.array([
      [1, 2],
      [3, 4],
    ]);
    const w1 = np.array([
      [1, 2],
      [3, 4],
    ]);
    const b1 = np.array([1, 2]);
    const w2 = np.array([
      [1, 2],
      [3, 4],
    ]);
    const b2 = np.array([1, 2]);

    const forward = jit(
      (x: np.Array, w1: np.Array, b1: np.Array, w2: np.Array, b2: np.Array) => {
        const h = nn.relu(np.dot(x, w1).add(b1));
        const y = nn.logSoftmax(np.dot(h, w2).add(b2));
        return y;
      },
    );

    const h = forward(x, w1, b1, w2, b2);
    expect(await h.jsAsync()).toEqual([
      [-21, 0],
      [-41, 0],
    ]);
  });
});
