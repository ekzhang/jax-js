import { JsTree, numpy as np, tree } from "@jax-js/jax";

/** The internal state of the optimizer, varies in shape. */
export type OptState = JsTree<np.Array>;

/**
 * A pair of pure functions implementing a gradient transformation.
 *
 * Optimizers are implemented with this interface. They do not contain any
 * internal state. The "optimizer state" `OptState` is initialized, then passed
 * into each update call, which returns a new state.
 *
 * Gradients are transformed during the update call, and they should have the
 * same PyTree shape as the parameters.
 */
export interface GradientTransformation {
  init<Params extends JsTree<np.Array>>(params: Params): OptState;
  update<Params extends JsTree<np.Array>>(
    updates: Params,
    state: OptState,
    params?: Params,
  ): [Params, OptState];
}

/**
 * A schedule consuming a scalar uint32 step count and returning a scalar array.
 *
 * Use array operations rather than JavaScript arithmetic so schedules work
 * inside `jit`. Cast the count to a floating-point dtype for fractional math.
 * Like other array functions, a schedule consumes its input reference and
 * returns an owned reference. Retain captured arrays with `.ref` when needed.
 *
 * @inline
 */
export type Schedule = (count: np.Array) => np.Array;

/** @inline */
export type ScalarOrSchedule = number | Schedule;

/** Simplest possible state for a transformation. */
export function initEmptyState(params: JsTree<np.Array>) {
  tree.dispose(params);
  return [];
}

/** Stateless identity transformation that leaves input gradients untouched. */
export function identity(): GradientTransformation {
  return {
    init: initEmptyState,
    update<Params extends JsTree<np.Array>>(
      updates: Params,
      state: OptState,
      params?: Params,
    ): [Params, OptState] {
      tree.dispose(params);
      return [updates, state];
    },
  };
}

/** Stateless transformation that maps input gradients to zero. */
export function setToZero(): GradientTransformation {
  return {
    init: initEmptyState,
    update<Params extends JsTree<np.Array>>(
      updates: Params,
      state: OptState,
      params?: Params,
    ): [Params, OptState] {
      tree.dispose(params);
      const zeros = tree.map((g: np.Array) => np.zerosLike(g), updates);
      return [zeros, state];
    },
  };
}

/**
 * Applies an update to the corresponding parameters.
 *
 * This function is provided for convenience, as it just adds the updates to the
 * parameters directly. You can get updates from a `GradientTransformation`.
 */
export function applyUpdates<Params extends JsTree<np.Array>>(
  params: Params,
  updates: Params,
): Params {
  return tree.map((p: np.Array, u: np.Array) => p.add(u), params, updates);
}
