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

test("distinguishes JIT programs whose scalar literals have the same hash", async () => {
  const first = jit(() => np.array(1, { dtype: np.float64 }));
  const second = jit(() => np.array(3189052130706922, { dtype: np.float64 }));
  try {
    expect((await first().data())[0]).toBe(1);
    expect((await second().data())[0]).toBe(3189052130706922);
  } finally {
    first.dispose();
    second.dispose();
  }
});

test("distinguishes JIT programs by scalar literal dtype", async () => {
  const floats = jit(() => np.array(2, { dtype: np.float32 }));
  const integers = jit(() => np.array(2, { dtype: np.int32 }));
  try {
    expect(await floats().data()).toEqual(Float32Array.of(2));
    expect(await integers().data()).toEqual(Int32Array.of(2));
  } finally {
    floats.dispose();
    integers.dispose();
  }
});

test("computes literal products with uint32 arithmetic", async () => {
  const maximum = () => np.array(0xffffffff, { dtype: np.uint32 });
  const compiled = jit(() => maximum().mul(maximum()));
  try {
    expect(await compiled().data()).toEqual(Uint32Array.of(1));
  } finally {
    compiled.dispose();
  }
});

test("keeps the order of floating-point addition in JIT", async () => {
  const magnitude = 2 ** 53;
  const compiled = jit((x: np.Array) => x.add(magnitude).add(-magnitude));
  try {
    const input = np.array(Float64Array.of(1));
    expect(await compiled(input).data()).toEqual(Float64Array.of(0));
  } finally {
    compiled.dispose();
  }
});
