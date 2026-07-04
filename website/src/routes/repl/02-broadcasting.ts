import { numpy as np } from "@jax-js/jax";

// Broadcasting combines [4, 1] and [1, 5] without materializing repeats first.
const rows = np.arange(4).reshape([4, 1]);
const cols = np.arange(5).reshape([1, 5]);
const grid = rows.mul(10).add(cols);

// Reductions work over any axis.
console.log("shape =", grid.shape);
console.log(await grid.ref.jsAsync());
console.log("row means =", await grid.mean(1).jsAsync());
