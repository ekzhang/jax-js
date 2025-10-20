import "vitest";
import { numpy as np } from "@jax-js/jax";

interface CustomMatchers<R = unknown> {
  toBeAllclose: (
    expected: Parameters<typeof np.array>[0],
    options: { rtol?: number; atol?: number } = {},
  ) => R;
  toBeWithinRange: (min: number, max: number) => R;
}

declare module "vitest" {
  interface Assertion<T = any> extends CustomMatchers<T> {}
  interface AsymmetricMatchersContaining extends CustomMatchers {}
}
