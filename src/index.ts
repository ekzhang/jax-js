import { DType } from "./alu";
import { defaultDevice, Device, devices, getBackend, init } from "./backend";
import { Array, ArrayLike } from "./frontend/array";
import * as jaxprModule from "./frontend/jaxpr";
import { ClosedJaxpr, Jaxpr, OwnedFunction } from "./frontend/jaxpr";
import * as jvpModule from "./frontend/jvp";
import * as linearizeModule from "./frontend/linearize";
import * as vmapModule from "./frontend/vmap";
import * as lax from "./library/lax";
import * as nn from "./library/nn";
import * as numpy from "./library/numpy";
import * as random from "./library/random";
import * as scipySpecial from "./library/scipy-special";
import * as tree from "./tree";
import type { JsTree, JsTreeDef, MapJsTree } from "./tree";
import { setDebug } from "./utils";

import "./polyfills";

export {
  init,
  Array,
  ClosedJaxpr,
  defaultDevice,
  type Device,
  devices,
  DType,
  Jaxpr,
  type JsTree,
  type JsTreeDef,
  lax,
  nn,
  numpy,
  type OwnedFunction,
  random,
  setDebug,
  scipySpecial,
  tree,
};

/**
 * @function
 * Compute the forward-mode Jacobian-vector product for a function.
 */
export const jvp = jvpModule.jvp as <
  F extends (...args: any[]) => JsTree<Array>,
>(
  f: F,
  primals: MapJsTree<Parameters<F>, Array, ArrayLike>,
  tangents: MapJsTree<Parameters<F>, Array, ArrayLike>,
) => [ReturnType<F>, ReturnType<F>];

/**
 * @function
 * Vectorize an operation on a batched axis for one or more inputs.
 */
export const vmap = vmapModule.vmap as <
  F extends (...args: any[]) => JsTree<Array>,
>(
  f: F,
  inAxes?: number | (number | null | JsTree<number | null>)[],
) => (...args: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>;

/**
 * @function
 * Compute the Jacobian evaluated column-by-column by forward-mode AD.
 */
export const jacfwd = vmapModule.jacfwd as <F extends (x: Array) => Array>(
  f: F,
) => (...args: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>;

/**
 * @function
 * Construct a Jaxpr by dynamically tracing a function with example inputs.
 */
export const makeJaxpr = jaxprModule.makeJaxpr as unknown as <
  F extends (...args: any[]) => JsTree<Array>,
>(
  f: F,
) => (...args: Parameters<F>) => {
  jaxpr: ClosedJaxpr;
  treedef: JsTreeDef;
};

/**
 * @function
 * Mark a function for automatic JIT compilation, with operator fusion.
 *
 * The function will be compiled the first time it is called with a set of
 * argument shapes.
 *
 * You can call `.dispose()` on the returned, JIT-compiled function after all
 * calls to free memory associated with array constants.
 *
 * **Options:**
 * - `staticArgnums`: An array of argument indices to treat as static
 *   (compile-time constant). These arguments must be hashable, won't be traced,
 *   and different values will trigger recompilation.
 * - `device`: The device to place the computation on. If not specified, the
 *   computation will be placed on the default device.
 */
export const jit = jaxprModule.jit as <
  F extends (...args: any[]) => JsTree<Array>,
>(
  f: F,
  opts?: jaxprModule.JitOpts,
) => OwnedFunction<
  (...args: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>
>;

/**
 * @function
 * Produce a local linear approximation to a function at a point using jvp() and
 * partial evaluation.
 */
export const linearize = linearizeModule.linearize as <
  F extends (...args: any[]) => JsTree<Array>,
>(
  f: F,
  ...primals: MapJsTree<Parameters<F>, Array, ArrayLike>
) => [
  ReturnType<F>,
  (...tangents: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>,
];

/**
 * @function
 * Calculate the reverse-mode vector-Jacobian product for a function.
 *
 * When `{ hasAux: true }` is passed as the second argument, the function `f` is
 * expected to return a `[output, aux]` tuple, and `vjp` returns `[output, vjpFn, aux]`.
 *
 * @example
 * ```ts
 * // Without hasAux
 * const [y, vjpFn] = vjp(f, x);
 *
 * // With hasAux
 * const [y, vjpFn, aux] = vjp(f, { hasAux: true }, x);
 * ```
 */
// vjp overloads - discriminate on opts parameter
export function vjp<F extends (...args: any[]) => JsTree<Array>>(
  f: F,
  ...primals: MapJsTree<Parameters<F>, Array, ArrayLike>
): [
  ReturnType<F>,
  OwnedFunction<
    (
      cotangents: MapJsTree<ReturnType<F>, Array, ArrayLike>,
    ) => MapJsTree<Parameters<F>, ArrayLike, Array>
  >,
];
export function vjp<
  F extends (...args: any[]) => [JsTree<Array>, JsTree<Array>],
>(
  f: F,
  opts: { hasAux: true },
  ...primals: MapJsTree<Parameters<F>, Array, ArrayLike>
): [
  ReturnType<F>[0],
  OwnedFunction<
    (
      cotangents: MapJsTree<ReturnType<F>[0], Array, ArrayLike>,
    ) => MapJsTree<Parameters<F>, ArrayLike, Array>
  >,
  ReturnType<F>[1],
];
export function vjp(f: any, ...args: any[]): any {
  return linearizeModule.vjp(f, ...args);
}

/**
 * @function
 * Compute the gradient of a scalar-valued function `f` with respect to its
 * first argument.
 *
 * When `{ hasAux: true }` is passed, the function `f` is expected to return a
 * `[scalar, aux]` tuple, and the returned function returns `[gradient, aux]`.
 *
 * @example
 * ```ts
 * // Without hasAux
 * const gradient = grad(f)(x);
 *
 * // With hasAux
 * const [gradient, aux] = grad(f, { hasAux: true })(x);
 * ```
 */
// grad overloads - discriminate on opts parameter
export function grad<F extends (...args: any[]) => JsTree<Array>>(
  f: F,
): (
  ...primals: MapJsTree<Parameters<F>, Array, ArrayLike>
) => MapJsTree<Parameters<F>[0], ArrayLike, Array>;
export function grad<
  F extends (...args: any[]) => [JsTree<Array>, JsTree<Array>],
>(
  f: F,
  opts: { hasAux: true },
): (
  ...primals: MapJsTree<Parameters<F>, Array, ArrayLike>
) => [MapJsTree<Parameters<F>[0], ArrayLike, Array>, ReturnType<F>[1]];
export function grad(f: any, opts?: any): any {
  return linearizeModule.grad(f, opts);
}

/**
 * @function
 * Create a function that evaluates both `f` and the gradient of `f`.
 *
 * When `{ hasAux: true }` is passed, the function `f` is expected to return a
 * `[scalar, aux]` tuple, and the returned function returns `[value, gradient, aux]`.
 *
 * @example
 * ```ts
 * // Without hasAux
 * const [value, gradient] = valueAndGrad(f)(x);
 *
 * // With hasAux
 * const [value, gradient, aux] = valueAndGrad(f, { hasAux: true })(x);
 * ```
 */
// valueAndGrad overloads - discriminate on opts parameter
export function valueAndGrad<F extends (...args: any[]) => JsTree<Array>>(
  f: F,
): (
  ...primals: MapJsTree<Parameters<F>, Array, ArrayLike>
) => [ReturnType<F>, MapJsTree<Parameters<F>[0], ArrayLike, Array>];
export function valueAndGrad<
  F extends (...args: any[]) => [JsTree<Array>, JsTree<Array>],
>(
  f: F,
  opts: { hasAux: true },
): (
  ...primals: MapJsTree<Parameters<F>, Array, ArrayLike>
) => [
  ReturnType<F>[0],
  MapJsTree<Parameters<F>[0], ArrayLike, Array>,
  ReturnType<F>[1],
];
export function valueAndGrad(f: any, opts?: any): any {
  return linearizeModule.valueAndGrad(f, opts);
}

/**
 * @function
 * Compute the Jacobian evaluated row-by-row by reverse-mode AD.
 */
export const jacrev = linearizeModule.jacrev as typeof jacfwd;

export { jacrev as jacobian };

/**
 * Wait until all `Array` leaves are ready by calling `Array.blockUntilReady()`.
 *
 * This can be used to wait for the results of an intermediate computation to
 * finish. It's recommended to call this regularly in an iterative computation
 * to avoid queueing up too many pending operations.
 *
 * Does not consume reference to the arrays.
 */
export async function blockUntilReady<T extends JsTree<any>>(x: T): Promise<T> {
  const promises: Promise<Array>[] = [];
  for (const leaf of tree.leaves(x)) {
    if (leaf instanceof Array) {
      promises.push(leaf.blockUntilReady());
    }
  }
  await Promise.all(promises);
  return x;
}

/**
 * Transfer `x` to `device`.
 *
 * `x` may be a nested container of arrays or scalars. The resulting structure
 * is committed to the device.
 *
 * If `device` is not specified, this function behaves as identity if the input
 * is already an `Array`, otherwise it places the scalar uncommitted on the
 * default device.
 */
export async function devicePut<T extends JsTree<any>>(
  x: T,
  device?: Device,
): Promise<MapJsTree<T, number | boolean, Array>> {
  const [xflat, structure] = tree.flatten(x);
  const yflat = await Promise.all(
    xflat.map((leaf) => {
      if (leaf instanceof Array) {
        return device ? leaf._put(getBackend(device)) : Promise.resolve(leaf);
      } else {
        return Promise.resolve(numpy.array(leaf as any, { device }));
      }
    }),
  );
  return tree.unflatten(structure, yflat) as any;
}
