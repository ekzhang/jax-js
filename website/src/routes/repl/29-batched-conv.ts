import { lax, numpy as np } from "@jax-js/jax";

// The leading dimension is the batch, so one conv handles all examples.
const x = np.arange(0, 3 * 16, 1, { dtype: np.float32 }).reshape([3, 1, 16]);
const kernel = np.array([1, 0, -1]).reshape([1, 1, 3]);

const y = lax.conv(x, kernel, [1], "SAME");

console.log("batched conv shape =", y.shape);
console.log("first example =", await y.slice(0, 0, []).jsAsync());
