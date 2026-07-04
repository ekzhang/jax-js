import { lax, numpy as np } from "@jax-js/jax";

// Solve a small linear system and inspect lower-level LU output.
const A = np.array([
  [4, 7],
  [2, 6],
]);
const b = np.array([[1], [0]]);

console.log("solve Ax=b =", await np.linalg.solve(A.ref, b).jsAsync());
console.log("det(A) =", await np.linalg.det(A.ref).jsAsync());

// lax.linalg.lu exposes packed LU factors and the row permutation.
const [lu, pivots, permutation] = lax.linalg.lu(A);
console.log("LU packed =", await lu.jsAsync());
console.log("pivots =", await pivots.jsAsync());
console.log("permutation =", await permutation.jsAsync());
