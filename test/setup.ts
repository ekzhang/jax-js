import { expect } from "vitest";
import { numpy as np } from "jax-js";

expect.extend({
  toBeClose(received: np.ArrayLike, expected: np.ArrayLike) {
    const { isNot } = this;
    return {
      pass: np.allclose(received, expected),
      message: () => `expected array to be${isNot ? " not" : ""} allclose`,
      actual: np.array(received).js(),
      expected: np.array(expected).js(),
    };
  },
});
