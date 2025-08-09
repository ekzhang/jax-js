// Tests for convolution-related operations.

import {
  devices,
  grad,
  init,
  jit,
  lax,
  numpy as np,
  setDevice,
} from "@jax-js/jax";
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

  test("0d convolution", () => {
    const x = np.array([
      [1, 2],
      [3, 4],
      [5, 8],
    ]);
    const y = np.array([
      [6, 4],
      [3, 2],
    ]);
    const result = lax.convGeneralDilated(x, y, [], "VALID");
    expect(result.js()).toEqual([
      [14, 7],
      [34, 17],
      [62, 31],
    ]);
  });

  test("grad of 0d convolution", () => {
    const x = np.array([
      [1, 2],
      [3, 4],
      [5, 8],
    ]);
    const y = np.array([
      [6, 4],
      [3, 2],
    ]);
    const f = (x: np.Array, y: np.Array) =>
      lax.convGeneralDilated(x, y, [], "VALID").sum();
    expect(grad(f)(x, y).js()).toEqual([
      [9, 6],
      [9, 6],
      [9, 6],
    ]);
  });

  test("grad of 1d convolution", () => {
    const f = (x: np.Array, y: np.Array) =>
      lax.convGeneralDilated(x, y, [1], "SAME").slice(0, 0, 3);
    const x = np.array([[[1, 2, 3, 4, 5, 6, 7]]]);
    const y = np.array([[[2, 0.5, -1]]]);
    const dx = grad(f)(x.ref, y.ref);
    expect(dx.slice(0, 0).js()).toEqual([0, 0, 2, 0.5, -1, 0, 0]);

    const dy = grad((y: np.Array, x: np.Array) => f(x, y))(y, x);
    expect(dy.slice(0, 0).js()).toEqual([3, 4, 5]);
  });

  test.only("grad shape test with stride 2", () => {
    const f = (x: np.Array, y: np.Array) =>
      lax.convGeneralDilated(x, y, [2, 2], "VALID").sum();
    const g = (y: np.Array, x: np.Array) =>
      lax.convGeneralDilated(x, y, [2, 2], "VALID").sum();

    for (const xDim of [1, 3, 8, 12, 15]) {
      for (const kDim of [1, 3, 4]) {
        if (xDim < kDim) continue;
        const x = np.zeros([3, 1, xDim, xDim]);
        const y = np.zeros([1, 1, kDim, kDim]);
        const dx = grad(f)(x, y);
        expect(dx.shape).toEqual(x.shape);

        const dy = grad(g)(y, x);
        expect(dy.shape).toEqual(y.shape);
      }
    }
  });
});
