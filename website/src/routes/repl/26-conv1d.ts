import { lax, numpy as np } from "@jax-js/jax";

// 1D convolution uses [batch, channels, length] layout.
const x = np.linspace(0, 1, 32).reshape([1, 1, 32]);
const kernel = np.array([0.25, 0.5, 0.25]).reshape([1, 1, 3]);

const y = lax.conv(x, kernel, [1], "SAME").reshape([32]);

console.log("smoothed first 10 =", await y.slice([0, 10]).jsAsync());
