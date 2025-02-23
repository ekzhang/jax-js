import * as core from "./core";
import * as numpy from "./numpy";
import { Array, ArrayLike } from "./numpy";
import * as tree from "./tree";
import type { JsTree, JsTreeDef } from "./tree";

export { numpy, tree };

// Convert a subtype of JsTree<A> into a JsTree<B>, with the same structure.
type MapJsTree<T, A, B> = T extends A
  ? B
  : T extends globalThis.Array<infer U>
    ? MapJsTree<U, A, B>[]
    : { [K in keyof T]: MapJsTree<T[K], A, B> };

// Assert that a function's arguments are a subtype of the given type.
type WithArgsSubtype<F extends (args: any[]) => any, T> =
  Parameters<F> extends T ? F : never;

/** Compute the forward-mode Jacobian-vector product for a function. */
export const jvp = core.jvp as <F extends (...args: any[]) => JsTree<Array>>(
  f: WithArgsSubtype<F, JsTree<ArrayLike>>,
  primals: MapJsTree<Parameters<F>, Array, ArrayLike>,
  tangents: MapJsTree<Parameters<F>, Array, ArrayLike>,
) => [ReturnType<F>, ReturnType<F>];

/** Vectorize an operation on a batched axis for one or more inputs. */
export const vmap = core.vmap as <F extends (...args: any[]) => JsTree<Array>>(
  f: WithArgsSubtype<F, JsTree<ArrayLike>>,
  inAxes: MapJsTree<Parameters<F>, Array, number | null>,
) => F;

/** Compute the Jacobian evaluated column-by-column by forward-mode AD. */
export const jacfwd = core.jacfwd as <F extends (x: Array) => Array>(
  f: F,
  x: Array,
) => F;

/** Construct a Jaxpr by dynamically tracing a function with example inputs. */
export const makeJaxpr = core.makeJaxpr as unknown as <
  F extends (...args: any[]) => JsTree<Array>,
>(
  f: WithArgsSubtype<F, JsTree<ArrayLike>>,
) => (...args: Parameters<F>) => {
  jaxpr: core.Jaxpr;
  consts: Array[];
  treedef: JsTreeDef;
};

/**
 * Produce a local linear approximation to a function at a point using jvp() and
 * partial evaluation.
 */
export const linearize = core.linearize as <
  F extends (...args: any[]) => JsTree<Array>,
>(
  f: WithArgsSubtype<F, JsTree<ArrayLike>>,
  ...primals: MapJsTree<Parameters<F>, Array, ArrayLike>
) => [
  ReturnType<F>,
  (...tangents: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>,
];

/** Calculate the reverse-mode vector-Jacobian product for a function. */
export const vjp = core.vjp as <F extends (...args: any[]) => JsTree<Array>>(
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
export const grad = core.grad as <F extends (...args: any[]) => JsTree<Array>>(
  f: WithArgsSubtype<F, JsTree<ArrayLike>>,
) => (
  ...primals: MapJsTree<Parameters<F>, Array, ArrayLike>
) => MapJsTree<Parameters<F>[0], ArrayLike, Array>;
