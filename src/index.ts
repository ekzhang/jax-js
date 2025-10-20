import { Device, devices, init, setDevice } from "./backend";
import * as jaxprModule from "./frontend/jaxpr";
import * as jvpModule from "./frontend/jvp";
import * as linearizeModule from "./frontend/linearize";
import * as vmapModule from "./frontend/vmap";
import * as lax from "./lax";
import * as nn from "./nn";
import * as numpy from "./numpy";
import { Array, ArrayLike, DType } from "./numpy";
import * as random from "./random";
import * as tree from "./tree";
import type { JsTree, JsTreeDef, MapJsTree } from "./tree";

import "./polyfills";

export {
  init,
  type Device,
  devices,
  DType,
  type JsTree,
  type JsTreeDef,
  lax,
  nn,
  numpy,
  random,
  setDevice,
  tree,
};

// Assert that a function's arguments are a subtype of the given type.
/** @inline */
type WithArgsSubtype<F extends (args: any[]) => any, T> =
  Parameters<F> extends T ? F : never;

/** Compute the forward-mode Jacobian-vector product for a function. */
export const jvp = jvpModule.jvp as <
  F extends (...args: any[]) => JsTree<Array>,
>(
  f: WithArgsSubtype<F, JsTree<ArrayLike>>,
  primals: MapJsTree<Parameters<F>, Array, ArrayLike>,
  tangents: MapJsTree<Parameters<F>, Array, ArrayLike>,
) => [ReturnType<F>, ReturnType<F>];

/** Vectorize an operation on a batched axis for one or more inputs. */
export const vmap = vmapModule.vmap as <
  F extends (...args: any[]) => JsTree<Array>,
>(
  f: WithArgsSubtype<F, JsTree<ArrayLike>>,
  inAxes?: number | MapJsTree<Parameters<F>, ArrayLike, number | null>,
) => (...args: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>;

/** Compute the Jacobian evaluated column-by-column by forward-mode AD. */
export const jacfwd = vmapModule.jacfwd as <F extends (x: Array) => Array>(
  f: F,
) => (...args: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>;

/** Construct a Jaxpr by dynamically tracing a function with example inputs. */
export const makeJaxpr = jaxprModule.makeJaxpr as unknown as <
  F extends (...args: any[]) => JsTree<Array>,
>(
  f: WithArgsSubtype<F, JsTree<ArrayLike>>,
) => (...args: Parameters<F>) => {
  jaxpr: jaxprModule.Jaxpr;
  consts: Array[];
  treedef: JsTreeDef;
};

/**
 * Mark a function for automatic JIT compilation, with operator fusion.
 *
 * The function will be compiled the first time it is called with a set of
 * argument shapes.
 */
export const jit = jaxprModule.jit as <
  F extends (...args: any[]) => JsTree<Array>,
>(
  f: WithArgsSubtype<F, JsTree<ArrayLike>>,
) => (...args: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>;

/**
 * Produce a local linear approximation to a function at a point using jvp() and
 * partial evaluation.
 */
export const linearize = linearizeModule.linearize as <
  F extends (...args: any[]) => JsTree<Array>,
>(
  f: WithArgsSubtype<F, JsTree<ArrayLike>>,
  ...primals: MapJsTree<Parameters<F>, Array, ArrayLike>
) => [
  ReturnType<F>,
  (...tangents: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>,
];

/** Calculate the reverse-mode vector-Jacobian product for a function. */
export const vjp = linearizeModule.vjp as <
  F extends (...args: any[]) => JsTree<Array>,
>(
  f: WithArgsSubtype<F, JsTree<ArrayLike>>,
  ...primals: MapJsTree<Parameters<F>, Array, ArrayLike>
) => [
  ReturnType<F>,
  (
    cotangents: MapJsTree<ReturnType<F>, Array, ArrayLike>,
  ) => MapJsTree<Parameters<F>, ArrayLike, Array>,
];

/**
 * Compute the gradient of a scalar-valued function `f` with respect to its
 * first argument.
 */
export const grad = linearizeModule.grad as <
  F extends (...args: any[]) => JsTree<Array>,
>(
  f: WithArgsSubtype<F, JsTree<ArrayLike>>,
) => (
  ...primals: MapJsTree<Parameters<F>, Array, ArrayLike>
) => MapJsTree<Parameters<F>[0], ArrayLike, Array>;

/** Create a function that evaluates both `f` and the gradient of `f`. */
export const valueAndGrad = linearizeModule.valueAndGrad as <
  F extends (...args: any[]) => JsTree<Array>,
>(
  f: WithArgsSubtype<F, JsTree<ArrayLike>>,
) => (
  ...primals: MapJsTree<Parameters<F>, Array, ArrayLike>
) => [ReturnType<F>, MapJsTree<Parameters<F>[0], ArrayLike, Array>];

/** Compute the Jacobian evaluated row-by-row by reverse-mode AD. */
export const jacrev = linearizeModule.jacrev as typeof jacfwd;

/** Compute the Jacobian with reverse-mode AD. Alias for `jacrev()`. */
export const jacobian = jacrev;
