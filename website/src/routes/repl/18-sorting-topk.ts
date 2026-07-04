import { lax, numpy as np } from "@jax-js/jax";

// topK returns values and original indices in descending order.
const scores = np.array([3, 1, 4, 1, 5, 9, 2, 6]);
const [values, indices] = lax.topK(scores.ref, 3);

// argsort gives the full stable ordering.
console.log("top values =", await values.jsAsync());
console.log("top indices =", await indices.jsAsync());
console.log("argsort =", await np.argsort(scores).jsAsync());
