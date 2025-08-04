// Tests for convolution-related operations.

import { devices, init, jit, lax, numpy as np, setDevice } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    setDevice(device);
  });

  test("1d convolution", () => {
    const x = np.array([[[1, 2, 3, 4, 5]]]);
    const y = np.array([[[2, 0.5, -1]]]);
    const result = lax.convGeneralDilated(x.ref, y.ref, [1], "VALID");
    expect(result.js()).toEqual([[[0, 1.5, 3]]]);

    const result2 = lax.convGeneralDilated(x, y, [1], "SAME");
    expect(result2.js()).toEqual([[[-1.5, 0, 1.5, 3, 10.5]]]);
  });

  test("padding 'SAME' and 'SAME_LOWER'", () => {
    const x = np.ones([1, 1, 5]);
    const y = np.ones([1, 1, 4]);
    const resultSame = lax.convGeneralDilated(x.ref, y.ref, [1], "SAME");
    expect(resultSame.slice(0, 0).js()).toEqual([3, 4, 4, 3, 2]);
    const resultSameLower = lax.convGeneralDilated(x, y, [1], "SAME_LOWER");
    expect(resultSameLower.slice(0, 0).js()).toEqual([2, 3, 4, 4, 3]);
  });

  test("2d convolution", () => {
    const x = np
      .array([
        [3, 1, 5],
        [2, 2, 9],
      ])
      .reshape([1, 1, 2, 3]);
    const y = np
      .array([
        [1, 2],
        [3, 4],
      ])
      .reshape([1, 1, 2, 2]);
    const result = lax.convGeneralDilated(x, y, [1, 1], "VALID");
    expect(result.slice(0, 0).js()).toEqual([[19, 53]]);
  });

  test("conv works with jit", () => {
    const convFn = jit((a: np.Array, b: np.Array) =>
      lax.convGeneralDilated(a, b, [1], "SAME"),
    );
    const x = np.array([[[1, 2, 3, 4, 5]]]);
    const y = np.array([[[2, 0.5, -1]]]);
    const result = convFn(x, y);
    expect(result.js()).toEqual([[[-1.5, 0, 1.5, 3, 10.5]]]);
  });
});
