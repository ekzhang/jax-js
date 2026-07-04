import { grad, numpy as np } from "@jax-js/jax";

// Linear algebra routines participate in autodiff.
const f = (A: np.Array) => np.linalg.inv(A).sum();
const A = np.array([
  [2.0, 0.5],
  [0.5, 1.5],
]);

// Compare the matrix inverse with the gradient of a scalar function of it.
console.log("inv(A) =", await np.linalg.inv(A.ref).jsAsync());
console.log("grad sum(inv(A)) =", await grad(f)(A).jsAsync());
