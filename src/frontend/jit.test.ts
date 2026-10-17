import { expect, test, vi } from "vitest";

import { defaultDevice, getBackend, init } from "../backend";
import { jit, numpy as np } from "../index";

await init("cpu");

test("splits kernels that exceed maxArgs when an input is already black", () => {
  // WebGPU and WebGL allow 7-8 buffers per kernel. Lower the limit on the cpu
  // backend so a small program has to be split.
  defaultDevice("cpu");
  const maxArgs = vi
    .spyOn(getBackend("cpu"), "maxArgs", "get")
    .mockReturnValue(3);
  try {
    // The where() sees 4 buffers: w through its condition, and a, b, c through
    // y and sin(y). The condition is marked black first, which is not enough,
    // so y must be marked black next instead of the condition again.
    const f = jit((w: np.Array, a: np.Array, b: np.Array, c: np.Array) => {
      const y = a.add(b).add(c);
      return np.where(w.greater(0), y.ref, np.sin(y));
    });
    const out = f(
      np.array([1, -1]),
      np.array([1, 2]),
      np.array([3, 4]),
      np.array([5, 6]),
    );
    const [x0, x1] = out.js() as number[];
    expect(x0).toBeCloseTo(9);
    expect(x1).toBeCloseTo(Math.sin(12));
  } finally {
    maxArgs.mockRestore();
  }
});
