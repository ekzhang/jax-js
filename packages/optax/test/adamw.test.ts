import { grad, jit, JsTree, numpy as np, tree } from "@jax-js/jax";
import { adamw, applyUpdates, squaredError } from "@jax-js/optax";
import { expect, test } from "vitest";

test("adamw optimizer", () => {
  let params = np.array([1.0, 2.0, 3.0]);

  const solver = adamw(0.001);
  let optState = solver.init(params.ref);
  let updates: np.Array;

  const f = (x: np.Array) => squaredError(x, np.ones([3])).sum();
  const paramsGrad = grad(f)(params.ref);
  [updates, optState] = solver.update(paramsGrad, optState, params.ref);
  params = applyUpdates(params, updates);

  expect(params.shape).toEqual([3]);
  expect(params.dtype).toEqual(np.float32);
});

test("adamw with custom weight decay", () => {
  let params = np.array([1.0, 2.0, 3.0]);

  const solver = adamw(0.001, { weightDecay: 0.01 });
  let optState = solver.init(params.ref);
  let updates: np.Array;

  const f = (x: np.Array) => squaredError(x, np.ones([3])).sum();
  const paramsGrad = grad(f)(params.ref);
  [updates, optState] = solver.update(paramsGrad, optState, params.ref);
  params = applyUpdates(params, updates);

  expect(params.shape).toEqual([3]);
  expect(params.dtype).toEqual(np.float32);
});

test("adamw with nesterov", () => {
  let params = np.array([1.0, 2.0, 3.0]);

  const solver = adamw(0.001, { nesterov: true, weightDecay: 0.005 });
  let optState = solver.init(params.ref);
  let updates: np.Array;

  const f = (x: np.Array) => squaredError(x, np.ones([3])).sum();
  const paramsGrad = grad(f)(params.ref);
  [updates, optState] = solver.update(paramsGrad, optState, params.ref);
  params = applyUpdates(params, updates);

  expect(params.shape).toEqual([3]);
  expect(params.dtype).toEqual(np.float32);
});

test("adamw with callable mask", () => {
  let params = np.array([1.0, 2.0, 3.0]);

  // Mask function that returns a mask tree - only apply decay to first element
  const maskFn = (updates: JsTree<np.Array>): JsTree<np.Array> => {
    return tree.map((u: np.Array) => {
      u.dispose();
      return np.array([1.0, 0.0, 0.0]);
    }, updates);
  };

  const solver = adamw(0.001, { weightDecay: 0.01, mask: maskFn });
  let optState = solver.init(params.ref);
  let updates: np.Array;

  const f = (x: np.Array) => squaredError(x, np.ones([3])).sum();
  const paramsGrad = grad(f)(params.ref);
  [updates, optState] = solver.update(paramsGrad, optState, params.ref);
  params = applyUpdates(params, updates);

  expect(params.shape).toEqual([3]);
  expect(params.dtype).toEqual(np.float32);
});

test("adamw steps can be jit-compiled and match eager", () => {
  const f = (x: np.Array) => squaredError(x, np.ones([3])).sum();
  const solver = adamw(0.1);

  let eager = np.array([1.0, 2.0, 3.0]);
  let eagerState = solver.init(eager.ref);
  let jitted = np.array([1.0, 2.0, 3.0]);
  let jittedState = solver.init(jitted.ref);

  // bias correction must stay in-graph: a count.item() read would throw on
  // the tracer and make this step impossible to compile
  const step = jit((p: np.Array, s: JsTree<np.Array>) => {
    const [updates, next] = solver.update(grad(f)(p.ref), s, p.ref);
    return [applyUpdates(p, updates), next] as [np.Array, JsTree<np.Array>];
  });

  for (let i = 0; i < 3; i++) {
    const [updates, next] = solver.update(
      grad(f)(eager.ref),
      eagerState,
      eager.ref,
    );
    eager = applyUpdates(eager, updates);
    eagerState = next;
    [jitted, jittedState] = step(jitted, jittedState);
  }

  // early adam steps divide by near-zero bias denominators, so eager and
  // fused float32 orderings legitimately differ in the low digits
  expect(jitted).toBeAllclose(eager, { rtol: 1e-2 });
});
