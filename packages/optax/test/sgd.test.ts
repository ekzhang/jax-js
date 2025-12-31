import { grad, numpy as np } from "@jax-js/jax";
import { applyUpdates, sgd, squaredError } from "@jax-js/optax";
import { expect, test } from "vitest";

test("stochastic gradient descent", () => {
  let params = np.array([1.0, 2.0, 3.0]);

  const solver = sgd(0.11);
  let optState = solver.init(params.ref);
  let updates: np.Array;

  const f = (x: np.Array) => squaredError(x, np.ones([3])).sum();
  const paramsGrad = grad(f)(params.ref);
  [updates, optState] = solver.update(paramsGrad, optState);
  params = applyUpdates(params, updates);

  expect(params).toBeAllclose([1.0, 1.78, 2.56]);
});

test("sgd with momentum", () => {
  let params = np.array([1.0, 2.0, 3.0]);

  const solver = sgd(0.1, { momentum: 0.9 });
  let optState = solver.init(params.ref);
  let updates: np.Array;

  const f = (x: np.Array) => squaredError(x, np.ones([3])).sum();

  // First update
  let paramsGrad = grad(f)(params.ref);
  [updates, optState] = solver.update(paramsGrad, optState);
  params = applyUpdates(params, updates);

  // Second update
  paramsGrad = grad(f)(params.ref);
  [updates, optState] = solver.update(paramsGrad, optState);
  params = applyUpdates(params, updates);

  expect(params.shape).toEqual([3]);
  expect(params.dtype).toEqual(np.float32);
});

test("sgd with nesterov momentum", () => {
  let params = np.array([1.0, 2.0, 3.0]);

  const solver = sgd(0.1, { momentum: 0.9, nesterov: true });
  let optState = solver.init(params.ref);
  let updates: np.Array;

  const f = (x: np.Array) => squaredError(x, np.ones([3])).sum();

  // First update
  let paramsGrad = grad(f)(params.ref);
  [updates, optState] = solver.update(paramsGrad, optState);
  params = applyUpdates(params, updates);

  // Second update
  paramsGrad = grad(f)(params.ref);
  [updates, optState] = solver.update(paramsGrad, optState);
  params = applyUpdates(params, updates);

  expect(params.shape).toEqual([3]);
  expect(params.dtype).toEqual(np.float32);
});
