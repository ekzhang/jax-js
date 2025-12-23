// Tests for convolution-related operations.

import {
  defaultDevice,
  devices,
  grad,
  init,
  jit,
  lax,
  numpy as np,
} from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
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

  test("grad shape test with stride 2", () => {
    const f = (x: np.Array, y: np.Array) =>
      lax.convGeneralDilated(x, y, [2, 2], "VALID").sum();
    const g = (y: np.Array, x: np.Array) =>
      lax.convGeneralDilated(x, y, [2, 2], "VALID").sum();

    for (const xDim of [1, 3, 8, 12, 15]) {
      for (const kDim of [1, 3, 4]) {
        if (xDim < kDim) continue;
        const x = np.zeros([3, 1, xDim, xDim]);
        const y = np.zeros([1, 1, kDim, kDim]);
        const dx = grad(f)(x.ref, y.ref);
        expect(dx.shape).toEqual(x.shape);

        const dy = grad(g)(y, x);
        expect(dy.shape).toEqual(y.shape);
      }
    }
  });

  test("max-pooling and min-pooling", () => {
    const x = np.array([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
      [9, 10, 11, 12],
    ]);
    const result = lax.reduceWindow(x.ref, np.max, [2, 2], [1, 2]);
    expect(result.js()).toEqual([
      [6, 8],
      [10, 12],
    ]);

    const resultMin = lax.reduceWindow(x, np.min, [2, 2], [1, 2]);
    expect(resultMin.js()).toEqual([
      [1, 3],
      [5, 7],
    ]);
  });

  test("grad of max-pool 2d", () => {
    const x = np.array([
      [1, 5, 3, 4],
      [1, 2, 3, 4],
    ]);
    const maxPool2x2Sum = (x: np.Array) =>
      lax.reduceWindow(x, np.max, [2, 2], [2, 2]).sum();

    expect(maxPool2x2Sum(x.ref).js()).toEqual(9); // 5 + 4
    expect(grad(maxPool2x2Sum)(x).js()).toEqual([
      [0, 1, 0, 0.5],
      [0, 0, 0, 0.5],
    ]);
  });

  test("grouped convolution shape", () => {
    // Test with 2 groups: input has 4 channels, output has 6 channels
    // Each group: 2 input channels -> 3 output channels
    const x = np.zeros([2, 4, 8, 8]); // [N, C_in, H, W]
    const y = np.zeros([6, 2, 3, 3]); // [C_out, C_in/G, kH, kW]
    const result = lax.convGeneralDilated(x, y, [1, 1], "VALID", {
      featureGroupCount: 2,
    });
    expect(result.shape).toEqual([2, 6, 6, 6]);

    // Test with 4 groups (depthwise-like): 4 channels, each convolved separately
    const x2 = np.zeros([1, 4, 5, 5]);
    const y2 = np.zeros([8, 1, 3, 3]); // 2 output channels per group
    const result2 = lax.convGeneralDilated(x2, y2, [1, 1], "SAME", {
      featureGroupCount: 4,
    });
    expect(result2.shape).toEqual([1, 8, 5, 5]);
  });

  test("grouped convolution values", () => {
    // 2 groups, each doing independent 1d convolution
    // Group 1: channel 0 with kernel 0
    // Group 2: channel 1 with kernel 1
    const x = np
      .array([
        [[1, 2, 3, 4]], // channel 0
        [[5, 6, 7, 8]], // channel 1
      ])
      .reshape([1, 2, 4]); // [N=1, C_in=2, W=4]

    const y = np
      .array([
        [[1, 0, -1]], // kernel for group 0 -> out channel 0
        [[1, 1, 1]], // kernel for group 1 -> out channel 1
      ])
      .reshape([2, 1, 3]); // [C_out=2, C_in/G=1, kW=3]

    const result = lax.convGeneralDilated(x, y, [1], "VALID", {
      featureGroupCount: 2,
    });
    // Group 0: [1,2,3,4] conv [1,0,-1] = [1-3, 2-4] = [-2, -2]
    // Group 1: [5,6,7,8] conv [1,1,1] = [5+6+7, 6+7+8] = [18, 21]
    expect(result.shape).toEqual([1, 2, 2]);
    expect(result.js()).toEqual([
      [
        [-2, -2],
        [18, 21],
      ],
    ]);
  });

  test("grad of depthwise conv1d", () => {
    // Depthwise conv: each input channel convolved with its own kernel
    // 3 input channels, 3 output channels (1 per group)
    const f = (x: np.Array, y: np.Array) =>
      lax.convGeneralDilated(x, y, [1], "VALID", { featureGroupCount: 3 });

    const x = np
      .array([
        [[1, 2, 3, 4, 5]], // channel 0
        [[2, 3, 4, 5, 6]], // channel 1
        [[3, 4, 5, 6, 7]], // channel 2
      ])
      .reshape([1, 3, 5]); // [N=1, C=3, W=5]

    const y = np
      .array([
        [[1, -1]], // kernel for channel 0
        [[1, 0]], // kernel for channel 1
        [[0, 1]], // kernel for channel 2
      ])
      .reshape([3, 1, 2]); // [C_out=3, C_in/G=1, kW=2]

    // Forward pass check
    const result = f(x.ref, y.ref);
    expect(result.shape).toEqual([1, 3, 4]);
    // Channel 0: [1-2, 2-3, 3-4, 4-5] = [-1, -1, -1, -1]
    // Channel 1: [2, 3, 4, 5]
    // Channel 2: [4, 5, 6, 7]
    expect(result.js()).toEqual([
      [
        [-1, -1, -1, -1],
        [2, 3, 4, 5],
        [4, 5, 6, 7],
      ],
    ]);

    // Gradient w.r.t. input
    const sumF = (x: np.Array, y: np.Array) => f(x, y).sum();
    const dx = grad(sumF)(x.ref, y.ref);
    expect(dx.shape).toEqual([1, 3, 5]);

    // Gradient w.r.t. kernel
    const dy = grad((y: np.Array, x: np.Array) => sumF(x, y))(y, x);
    expect(dy.shape).toEqual([3, 1, 2]);
    // dy[0] = sum of x[0] windows = [1+2+3+4, 2+3+4+5] = [10, 14]
    // dy[1] = sum of x[1] windows = [2+3+4+5, 3+4+5+6] = [14, 18]
    // dy[2] = sum of x[2] windows = [3+4+5+6, 4+5+6+7] = [18, 22]
    expect(dy.js()).toEqual([[[10, 14]], [[14, 18]], [[18, 22]]]);
  });
});
