import { numpy as np } from "@jax-js/jax";
import { clipByGlobalNorm } from "@jax-js/optax";
import { expect, test } from "vitest";

test("clipByGlobalNorm: no clipping when norm is below threshold", () => {
  // global norm = sqrt(3^2 + 4^2) = 5.0, maxNorm = 10.0 > 5.0
  const params = [np.zeros([1]), np.zeros([1])];
  const updates = [np.array([3.0]), np.array([4.0])];

  const clipper = clipByGlobalNorm(10.0);
  const state = clipper.init(params);
  const [clipped] = clipper.update(updates, state);

  expect(clipped[0]).toBeAllclose([3.0]);
  expect(clipped[1]).toBeAllclose([4.0]);
});

test("clipByGlobalNorm: clips when norm exceeds threshold", () => {
  // global norm = 5.0, maxNorm = 2.5, scale = 2.5 / 5.0 = 0.5
  const params = [np.zeros([1]), np.zeros([1])];
  const updates = [np.array([3.0]), np.array([4.0])];

  const clipper = clipByGlobalNorm(2.5);
  const state = clipper.init(params);
  const [clipped] = clipper.update(updates, state);

  expect(clipped[0]).toBeAllclose([1.5]);
  expect(clipped[1]).toBeAllclose([2.0]);
});

test("clipByGlobalNorm: handles multi-dimensional gradients", () => {
  // global norm = sqrt(3^2 + 4^2 + 5^2 + 12^2) = sqrt(194) ≈ 13.93
  const params = [np.zeros([1, 2]), np.zeros([1, 2])];
  const updates = [np.array([[3.0, 4.0]]), np.array([[5.0, 12.0]])];

  const clipper = clipByGlobalNorm(5.0);
  const state = clipper.init(params);
  const [clipped] = clipper.update(updates, state);

  expect(clipped[0]).toBeAllclose([[1.08, 1.44]], { atol: 0.01 });
  expect(clipped[1]).toBeAllclose([[1.8, 4.31]], { atol: 0.01 });
});

test("clipByGlobalNorm: zero gradients", () => {
  const params = [np.zeros([1]), np.zeros([1])];
  const updates = [np.array([0.0]), np.array([0.0])];

  const clipper = clipByGlobalNorm(1.0);
  const state = clipper.init(params);
  const [clipped] = clipper.update(updates, state);

  expect(clipped[0]).toBeAllclose([0.0]);
  expect(clipped[1]).toBeAllclose([0.0]);
});
