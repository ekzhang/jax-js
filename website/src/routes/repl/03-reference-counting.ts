import { numpy as np } from "@jax-js/jax";

// jax-js arrays use move semantics. Pass x.ref when the caller should keep x.
function vectorNorm(x: np.Array) {
  const squared = x.ref.mul(x);
  return np.sqrt(squared.sum());
}

const x = np.array([3, 4, 12]);
// The original x is still available because vectorNorm received a reference.
console.log("x is reused because vectorNorm receives x.ref");
console.log("norm =", await vectorNorm(x.ref).jsAsync());
console.log("x + 1 =", await x.add(1).jsAsync());
