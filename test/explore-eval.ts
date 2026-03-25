/**
 * Exploring eager evaluation (no transformations active).
 *
 * The call chain for x.add(2):
 *
 *   Tracer.add(2)
 *     → add(this, 2)                      core.ts free function
 *     → bind1(Primitive.Add, [x, 2])
 *     → bind(Primitive.Add, [x, 2], {})   THE central dispatch
 *         findTopTrace([x, 2])            → EvalTrace (level 0, bottom of stack)
 *         fullRaise(trace, x)             → x is already an Array, returned as-is
 *         fullRaise(trace, 2)             → trace.pure(2), wraps number into Array
 *         trace.processPrimitive(Add, [x_arr, two_arr], {})
 *           → EvalTrace just calls implRules[Primitive.Add]
 *           → which calls x_arr.#binary(AluOp.Add, two_arr)
 *           → builds a kernel expression and runs it
 *         result.fullLower()              → no-op, returns the concrete Array
 *
 * No expression tree. No Jaxpr. Each op eagerly runs a kernel.
 */

import { numpy as np } from "@jax-js/jax";

// Each .add() / .mul() goes through bind() → EvalTrace → kernel execution.
// The result is a concrete Array with real data at every step.
const x = np.array([1, 2, 3]);
const y = x.add(2);
const z = y.mul(3);

console.log("x     =", x.js()); // [1, 2, 3]
console.log("x + 2 =", y.js()); // [3, 4, 5]
console.log("(x+2)*3 =", z.js()); // [9, 12, 15]
