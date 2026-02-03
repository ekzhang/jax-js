// lax.scan primitive implementation for jax-js
//
// Scans a function over leading axis of input arrays while carrying state.
// This enables expressing recurrences (like Kalman filters) in a functional style.
//
// This version uses the Scan primitive for efficient execution.

import * as numpy from "./numpy";
import type { Array } from "../frontend/array";
import { bind, getAval, Primitive, ShapedArray } from "../frontend/core";
import { makeJaxpr } from "../frontend/jaxpr";
import * as tree from "../tree";
import type { JsTree } from "../tree";
import type { ScanPath } from "../utils";

/**
 * Options for {@link scan}.
 */
export interface ScanOptions {
  /**
   * Number of iterations. If not provided, inferred from the leading axis of `xs`.
   * Required when `xs` is empty or null.
   */
  length?: number;

  /**
   * If `true`, process `xs` in reverse order (from last to first element).
   * The output `ys` will also be in reverse order.
   * @default false
   */
  reverse?: boolean;

  /**
   * Require a specific scan implementation path. If the JIT cannot use the
   * required path, it throws an error instead of falling back.
   *
   * This is primarily useful for testing to ensure optimized code paths are used.
   *
   * @example
   * ```ts
   * // Require the fused (native) scan path
   * lax.scan(f, init, xs, { requirePath: "fused" });
   *
   * // Allow either fused or fallback
   * lax.scan(f, init, xs, { requirePath: ["fused", "fallback"] });
   * ```
   */
  requirePath?: ScanPath | ScanPath[];
}

/**
 * Scan a function over leading array axes while carrying along state.
 *
 * Think of `scan` as a functional `reduce` that also returns all intermediate
 * results. It iterates over the leading axis of `xs`, threading a "carry" value
 * through each step and collecting outputs.
 *
 * ## Type Signature
 *
 * ```ts
 * scan(f, init, xs) → [finalCarry, ys]
 * scan(f, init, null, { length }) → [finalCarry, ys]  // carry-only scan
 *
 * // Where:
 * // f: (carry: C, x: X | null) => [C, Y]  -- step function
 * // init: C                               -- initial carry
 * // xs: X[] | null                        -- input array or null for carry-only
 * // finalCarry: C                         -- carry after last iteration
 * // ys: Y[]                               -- stacked outputs from each iteration
 * ```
 *
 * ## Semantics
 *
 * The semantics are roughly equivalent to this JavaScript:
 * ```ts
 * function scan(f, init, xs) {
 *   let carry = init;
 *   const ys = [];
 *   for (const x of xs) {
 *     const [newCarry, y] = f(carry, x);
 *     carry = newCarry;
 *     ys.push(y);
 *   }
 *   return [carry, np.stack(ys)];
 * }
 * ```
 *
 * Unlike a plain JavaScript loop:
 * - Both `xs` and `ys` can be arbitrary pytrees (nested objects/arrays)
 * - The scan is compiled to efficient native code (WASM/WebGPU)
 * - Supports autodiff: `grad(f)` works through scan
 * - The carry shape/dtype must be fixed across all iterations
 *
 * ## Reference Counting Contract
 *
 * **Inputs (consumed):**
 * - `init` and `xs` are consumed by scan (refcount decremented)
 * - Use `.ref` if you need to keep inputs alive: `scan(f, init.ref, xs.ref)`
 *
 * **Body function:**
 * - `carry` and `x` are **borrowed** — do NOT dispose them
 * - Return **new** arrays for `newCarry` and `y`
 * - For passthrough (same array in both), use `.ref`: `[result.ref, result]`
 *
 * **Outputs (caller owns):**
 * - `finalCarry` and `ys` are owned by caller — dispose when done
 *
 * @param f - Step function `(carry, x) => [newCarry, y]` where:
 *   - `carry` is the current state (same structure as `init`)
 *   - `x` is a slice of `xs` along axis 0, or `null` if `xs` is null
 *   - `newCarry` is the updated state (same structure/shape as `carry`)
 *   - `y` is the output for this iteration
 * @param init - Initial carry value. Can be a single array or a pytree of arrays.
 * @param xs - Input sequence to scan over, or `null` for carry-only scans.
 *   When an array/pytree, the leading axis is the scan dimension.
 *   When `null`, you must provide `{ length }` in options.
 * @param options - Scan options or legacy `length` number
 * @returns `[finalCarry, ys]` where:
 *   - `finalCarry` has the same structure as `init`
 *   - `ys` has the same structure as `y` from `f`, with each leaf having
 *     an additional leading axis of size `length`
 *
 * @example Cumulative sum
 * ```ts
 * import { lax, numpy as np } from '@jax-js/jax';
 *
 * const step = (carry, x) => {
 *   const sum = np.add(carry, x);
 *   return [sum, sum.ref];  // .ref: sum used in both outputs
 * };
 *
 * const init = np.array([0.0]);
 * const xs = np.array([[1], [2], [3], [4], [5]]);
 * const [final, sums] = await lax.scan(step, init, xs);
 *
 * console.log(await final.data());  // [15]
 * console.log(await sums.data());   // [[1], [3], [6], [10], [15]]
 *
 * final.dispose();
 * sums.dispose();
 * ```
 *
 * @example Factorial via scan
 * ```ts
 * // Compute n! for n = 1..5
 * const step = (carry, x) => {
 *   const next = np.multiply(carry, x);
 *   return [next, next.ref];
 * };
 *
 * const init = np.array([1]);
 * const xs = np.array([[1], [2], [3], [4], [5]]);
 * const [final, factorials] = await lax.scan(step, init, xs);
 * // factorials = [[1], [2], [6], [24], [120]]
 * ```
 *
 * @example Pytree carry (multiple state variables)
 * ```ts
 * // Track both sum and count
 * const step = (carry, x) => {
 *   const newSum = np.add(carry.sum, x);
 *   const newCount = np.add(carry.count, np.array([1]));
 *   return [
 *     { sum: newSum.ref, count: newCount.ref },
 *     { sum: newSum, count: newCount }
 *   ];
 * };
 *
 * const init = { sum: np.array([0]), count: np.array([0]) };
 * const xs = np.array([[10], [20], [30]]);
 * const [final, history] = await lax.scan(step, init, xs);
 * // final.sum = [60], final.count = [3]
 * ```
 *
 * @example Reverse scan
 * ```ts
 * // Process sequence from end to beginning
 * const [final, ys] = await lax.scan(step, init, xs, { reverse: true });
 * ```
 *
 * @example Carry-only scan (xs=null)
 * ```ts
 * // Generate a sequence without allocating input arrays.
 * // Useful for RNG, counters, Fibonacci, or any state-only iteration.
 * const step = (carry, _x) => {
 *   const next = np.add(carry.ref, np.array([1.0]));
 *   return [next, carry];  // output is old carry value
 * };
 *
 * const init = np.array([0.0]);
 * // Must provide length when xs is null
 * const [final, ys] = await lax.scan(step, init, null, { length: 5 });
 *
 * console.log(await ys.data());  // [0, 1, 2, 3, 4]
 * console.log(await final.data());  // [5]
 * ```
 *
 * @example jit(scan) - Compile the entire scan loop
 * ```ts
 * import { jit, lax, numpy as np } from '@jax-js/jax';
 *
 * // Wrap scan in jit to compile the entire loop into optimized native code.
 * // This is the most common and efficient pattern for production use.
 * const step = (carry, x) => {
 *   const newCarry = np.add(carry, x);
 *   return [newCarry, newCarry.ref];
 * };
 *
 * const scanFn = jit((init, xs) => lax.scan(step, init, xs));
 *
 * const init = np.array([0.0]);
 * const xs = np.array([[1.0], [2.0], [3.0]]);
 * const [final, ys] = await scanFn(init, xs);
 *
 * console.log(await final.data());  // [6]
 * scanFn.dispose();  // Free compiled program
 * ```
 *
 * @example scan(jit(body)) - JIT-compile only the step function
 * ```ts
 * import { jit, lax, numpy as np } from '@jax-js/jax';
 *
 * // JIT-compile just the step function. Each iteration calls compiled code,
 * // but the loop itself runs in JavaScript. Useful when step is expensive
 * // but you want to inspect intermediate values or the scan body is dynamic.
 * const step = jit((carry, x) => {
 *   const newCarry = np.add(carry, x);
 *   return [newCarry, newCarry.ref];
 * });
 *
 * const init = np.array([0.0]);
 * const xs = np.array([[1.0], [2.0], [3.0]]);
 * const [final, ys] = await lax.scan(step, init, xs);
 *
 * console.log(await final.data());  // [6]
 * step.dispose();  // Free compiled step function
 * ```
 *
 * @example With grad for differentiation
 * ```ts
 * import { grad, lax, numpy as np } from '@jax-js/jax';
 *
 * const loss = (init, xs) => {
 *   const [final, ys] = lax.scan(step, init, xs);
 *   final.dispose();
 *   return np.sum(ys);
 * };
 *
 * const gradLoss = grad(loss);
 * const [dInit, dXs] = await gradLoss(init, xs);
 * ```
 *
 * @example Carry-only scan (no input xs)
 * ```ts
 * // Generate sequence without input arrays (saves memory)
 * const step = (carry, _) => {
 *   const next = np.add(carry, np.array([1]));
 *   return [next, carry.ref];
 * };
 *
 * const init = np.array([0]);
 * // Must provide length when xs is null
 * const [final, ys] = await lax.scan(step, init, null, { length: 5 });
 * // ys = [[0], [1], [2], [3], [4]], final = [5]
 * ```
 *
 * @see {@link https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html | JAX lax.scan}
 */
export function scan<
  Carry extends JsTree<Array>,
  X extends JsTree<Array> | null | undefined,
  Y extends JsTree<Array>,
>(
  f: (carry: Carry, x: X) => [Carry, Y],
  init: Carry,
  xs: X,
  options?: number | ScanOptions,
): [Carry, Y] {
  // Handle legacy length-only argument
  const opts: ScanOptions =
    typeof options === "number" ? { length: options } : (options ?? {});
  const { length: lengthOpt, reverse = false, requirePath } = opts;

  // Handle xs=null case (carry-only scan with no input arrays)
  const xsIsNull = xs === null || xs === undefined;

  // Flatten inputs for primitive call
  const [initFlat, initTreedef] = tree.flatten(init);
  const [xsFlat, xsTreedef] = xsIsNull ? [[], null] : tree.flatten(xs);

  // Determine scan length from input
  const n = lengthOpt ?? (xsFlat.length > 0 ? xsFlat[0].shape[0] : 0);

  if (n === 0) {
    throw new Error(
      xsIsNull
        ? "scan: length option is required when xs is null"
        : "scan: cannot determine length from empty inputs",
    );
  }

  // Get abstract values for carry and x_slice (xs with leading dim removed)
  const carryAvals = initFlat.map((arr) => ShapedArray.fromAval(getAval(arr)));
  const xSliceAvals = xsFlat.map((arr) => {
    const aval = getAval(arr);
    // Remove leading dimension for x_slice
    return new ShapedArray(aval.shape.slice(1), aval.dtype, aval.weakType);
  });

  // Create a wrapper function that takes flat arrays and returns flat arrays
  const flatF = (
    carryFlat: Array[],
    xSliceFlat: Array[],
  ): [Array[], Array[]] => {
    const carry = tree.unflatten(initTreedef, carryFlat) as Carry;
    // For xs=null, pass null/undefined to body function (matches JAX behavior)
    const xSlice = xsIsNull
      ? xs
      : (tree.unflatten(xsTreedef!, xSliceFlat) as X);
    const [newCarry, y] = f(carry, xSlice as X);
    const [newCarryFlat] = tree.flatten(newCarry);
    const [yFlat, yTreedef] = tree.flatten(y);
    // Store yTreedef for later reconstruction
    (flatF as any)._yTreedef = yTreedef;
    return [newCarryFlat, yFlat];
  };

  // Trace the function to get jaxpr
  // makeJaxpr expects a function that takes individual arrays as arguments
  const traceFn = (...args: Array[]): Array[] => {
    const numCarry = carryAvals.length;
    const carryFlat = args.slice(0, numCarry);
    const xSliceFlat = args.slice(numCarry);
    const [newCarryFlat, yFlat] = flatF(carryFlat, xSliceFlat);
    return [...newCarryFlat, ...yFlat];
  };

  // Build abstract values for tracing
  const traceAvals = [...carryAvals, ...xSliceAvals];

  // Trace to get jaxpr
  const { jaxpr: closedJaxpr, treedef: _outTreedef } = makeJaxpr(traceFn)(
    ...traceAvals,
  );
  const jaxpr = closedJaxpr.jaxpr;
  const consts = closedJaxpr.consts;

  // Call the Scan primitive
  // Args: [...consts, ...initCarry, ...xs]
  const scanArgs = [
    ...consts.map((c) => c.ref),
    ...initFlat.map((arr) => arr.ref),
    ...xsFlat.map((arr) => arr.ref),
  ];

  const numCarry = initFlat.length;
  const numConsts = consts.length;

  const results = bind(Primitive.Scan, scanArgs, {
    jaxpr,
    numCarry,
    numConsts,
    length: n,
    reverse,
    requirePath,
  });

  // Dispose original inputs
  initFlat.forEach((arr) => arr.dispose());
  xsFlat.forEach((arr) => arr.dispose());
  closedJaxpr.dispose();

  // Split results into carry and ys
  const carryOut = results.slice(0, numCarry) as Array[];
  const ysFlat = results.slice(numCarry) as Array[];

  // Reconstruct pytrees
  const finalCarry = tree.unflatten(initTreedef, carryOut) as Carry;
  const yTreedef = (flatF as any)._yTreedef;
  const ys = tree.unflatten(yTreedef, ysFlat) as Y;

  return [finalCarry, ys];
}

/**
 * Stack a list of pytrees along a new leading axis.
 *
 * Each pytree in the list must have the same structure (same keys, same nesting).
 * The corresponding leaves are stacked using {@link numpy.stack}.
 *
 * This is useful for manually accumulating scan-like results when you need
 * more control than {@link scan} provides.
 *
 * @param trees - Array of pytrees to stack. All must have identical structure.
 * @returns A single pytree with the same structure, where each leaf is the
 *   stack of corresponding leaves from input trees (new axis at position 0).
 * @throws If `trees` is empty or pytrees have mismatched structures.
 *
 * @example Single arrays
 * ```ts
 * const a = np.array([1, 2]);
 * const b = np.array([3, 4]);
 * const c = np.array([5, 6]);
 * const stacked = stackPyTree([a, b, c]);
 * // stacked.shape = [3, 2], values = [[1,2], [3,4], [5,6]]
 * ```
 *
 * @example Pytrees (objects)
 * ```ts
 * const trees = [
 *   { x: np.array([1]), y: np.array([2]) },
 *   { x: np.array([3]), y: np.array([4]) },
 * ];
 * const stacked = stackPyTree(trees);
 * // stacked.x.shape = [2, 1], stacked.y.shape = [2, 1]
 * ```
 */
export function stackPyTree<T extends JsTree<Array>>(trees: T[]): T {
  if (trees.length === 0) {
    throw new Error("stackPyTree: empty list");
  }

  const [firstLeaves, treedef] = tree.flatten(trees[0]);
  const allLeaves = trees.map((t) => tree.leaves(t));

  // Number of leaves per tree
  const numLeaves = firstLeaves.length;

  // Stack each leaf position across all trees
  const stackedLeaves: Array[] = [];
  for (let leafIdx = 0; leafIdx < numLeaves; leafIdx++) {
    const toStack = allLeaves.map((leaves) => leaves[leafIdx].ref);
    // Use np.stack to combine along new axis 0
    const stacked = numpy.stack(toStack, 0);
    stackedLeaves.push(stacked);
  }

  // Reconstruct pytree with stacked leaves
  return tree.unflatten(treedef, stackedLeaves) as T;
}
