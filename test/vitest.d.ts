import "vitest";
import { numpy as np } from "jax-js";

interface CustomMatchers<R = unknown> {
  toBeClose: (expected: np.ArrayLike) => R;
}

declare module "vitest" {
  interface Assertion<T = any> extends CustomMatchers<T> {}
  interface AsymmetricMatchersContaining extends CustomMatchers {}
}
