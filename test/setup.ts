import { type Device, numpy as np } from "@jax-js/jax";
import { expect } from "vitest";

expect.extend({
  toBeAllclose(
    actual: np.ArrayLike,
    expected: np.ArrayLike,
    options: { rtol?: number; atol?: number } = {},
  ) {
    const { isNot } = this;
    const actualArray = np.array(actual);
    const expectedArray = np.array(expected);
    return {
      pass: np.allclose(actualArray.ref, expectedArray.ref, options),
      message: () => `expected array to be${isNot ? " not" : ""} allclose`,
      actual: actualArray.js(),
      expected: expectedArray.js(),
    };
  },
  toBeWithinRange(actual: number, min: number, max: number) {
    const { isNot } = this;
    const pass = actual >= min && actual <= max;
    return {
      pass,
      message: () =>
        `expected ${actual} to be${isNot ? " not" : ""} within range [${min}, ${max}]`,
      actual,
      expected: `[${min}, ${max}]`,
    };
  },
});

/**
 * Some devices have numerics edge cases around handling of NaN/Infinity values
 * and overflow, we skip certain tests for them.
 */
export function hasStrictNumerics(device: Device): boolean {
  return device !== "webgl";
}
