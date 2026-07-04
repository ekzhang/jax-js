import { lax, numpy as np } from "@jax-js/jax";

// reduceWindow() implements pooling over local windows.
const x = np.arange(0, 64, 1, { dtype: np.float32 }).reshape([8, 8]);
const pooled = lax.reduceWindow(x.ref, np.max, [2, 2], [2, 2]);

console.log("input shape =", x.shape);
console.log("pooled shape =", pooled.shape);
console.log(await pooled.jsAsync());
