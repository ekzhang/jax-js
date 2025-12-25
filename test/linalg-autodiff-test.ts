// Test script for autodiff with cholesky and triangular_solve
// Run with: bunx tsx test/linalg-autodiff-test.ts

import { linalg, numpy as np, grad } from "../src/index.ts";

console.log("Testing autodiff for cholesky and triangular_solve...\n");

// Test 1: triangular_solve basic
console.log("Test 1: triangular_solve basic");
{
  const L = np.array([[2., 0.], [1., 3.]]);
  const b = np.array([4., 10.]);
  const x = linalg.triangular_solve(L.ref, b.ref, { lower: true });
  console.log("L:", L.ref.js());
  console.log("b:", b.ref.js());
  console.log("x = solve(L, b):", x.ref.js());

  // Verify: L @ x should equal b
  const Lx = np.matmul(L, x);
  console.log("L @ x (should equal b):", Lx.js());
}

// Test 2: cholesky with solve
console.log("\nTest 2: cholesky + triangular_solve");
{
  const A = np.array([[4., 2.], [2., 5.]]);
  const cholL = linalg.cholesky(A.ref, { lower: true });
  console.log("A:", A.js());
  console.log("cholesky(A):", cholL.ref.js());

  // Solve A @ x = b using Cholesky:
  const b2 = np.array([6., 7.]);
  const y = linalg.triangular_solve(cholL.ref, b2.ref, { lower: true });
  const xSol = linalg.triangular_solve(cholL.ref, y.ref, { lower: true, transposeA: true });
  console.log("Solve A @ x = b:");
  console.log("  b:", b2.ref.js());
  console.log("  x:", xSol.ref.js());

  // Verify A @ x = b - create fresh copy of A
  const A2 = np.array([[4., 2.], [2., 5.]]);
  const Ax = np.matmul(A2, xSol);
  console.log("  A @ x (should equal b):", Ax.js());
}

// Test 3: try gradient through triangular_solve
console.log("\nTest 3: gradient through triangular_solve");
try {
  // f(b) = sum(triangular_solve(L, b))
  const gradFn = grad((b_in: any) => {
    const L3 = np.array([[2., 0.], [1., 3.]]);
    const x_out = linalg.triangular_solve(L3, b_in, { lower: true });
    return x_out.sum();
  });
  
  const b3 = np.array([1., 1.]);
  const grad_b = gradFn(b3);
  console.log("grad of sum(solve(L, b)) w.r.t. b:", grad_b.js());
  console.log("(This is L^{-T} @ ones)");
} catch (e: any) {
  console.log("Gradient computation note:", e.message.slice(0, 150) + "...");
}

console.log("\n✓ Tests completed!");
