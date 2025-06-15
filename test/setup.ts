import { numpy as np } from "@jax-js/jax";
import { expect } from "vitest";

expect.extend({
  toBeAllclose(actual: np.ArrayLike, expected: np.ArrayLike) {
    const { isNot } = this;
    const actualArray = np.array(actual);
    const expectedArray = np.array(expected);
    return {
      pass: np.allclose(actualArray.ref, expectedArray.ref),
      message: () => `expected array to be${isNot ? " not" : ""} allclose`,
      actual: actualArray.js(),
      expected: expectedArray.js(),
    };
  },
});
