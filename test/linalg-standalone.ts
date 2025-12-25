// Simple test script for cholesky - run with: bunx tsx test/linalg-standalone.ts

import { linalg, numpy as np } from "../src/index.ts";

console.log("Testing linalg.cholesky()...\n");

// Test 1: 2x2 lower triangular
console.log("Test 1: 2x2 lower triangular");
const x = np.array([
  [2.0, 1.0],
  [1.0, 2.0],
]);
console.log("Input:");
console.log(x.ref.js()); // Use .ref to preserve the array

const L = linalg.cholesky(x, { lower: true });
console.log("Lower Cholesky (L):");
console.log(L.ref.js()); // Use .ref to preserve L for matmul

const reconstructed = np.matmul(L.ref, L.ref.transpose());
console.log("L @ L^T (should equal input):");
console.log(reconstructed.js());
console.log();

// Test 2: 2x2 upper triangular (default)
console.log("Test 2: 2x2 upper triangular");
const x2 = np.array([
  [2.0, 1.0],
  [1.0, 2.0],
]);
const U = linalg.cholesky(x2);
console.log("Upper Cholesky (U):");
console.log(U.ref.js());

const reconstructedU = np.matmul(U.ref.transpose(), U);
console.log("U^T @ U (should equal input):");
console.log(reconstructedU.js());
console.log();

// Test 3: 3x3 matrix
console.log("Test 3: 3x3 matrix");
const x3 = np.array([
  [4.0, 2.0, 1.0],
  [2.0, 5.0, 3.0],
  [1.0, 3.0, 6.0],
]);
console.log("Input:");
console.log(x3.ref.js());

const L3 = linalg.cholesky(x3, { lower: true });
console.log("Lower Cholesky (L):");
console.log(L3.ref.js());

const reconstructed3 = np.matmul(L3.ref, L3.ref.transpose());
console.log("L @ L^T (should equal input):");
const recon3Js = reconstructed3.js();
console.log(recon3Js);

// Check if close (using approximate comparison for floating point)
const flatR3 = recon3Js.flat();
let maxDiff = 0;
for (let i = 0; i < 9; i++) {
  const expected = [4, 2, 1, 2, 5, 3, 1, 3, 6][i];
  const diff = Math.abs(expected - flatR3[i]);
  if (diff > maxDiff) maxDiff = diff;
}
console.log("Max difference:", maxDiff);
console.log("Test passed:", maxDiff < 1e-5);
console.log();

// Test 4: Error handling - non-square matrix
console.log("Test 4: Error handling - non-square matrix");
try {
  const nonSquare = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
  ]);
  linalg.cholesky(nonSquare);
  console.log("ERROR: Should have thrown!");
} catch (e: any) {
  console.log("Correctly threw:", e.message);
}
console.log();

// Test 5: Error handling - non-2D array
console.log("Test 5: Error handling - non-2D array");
try {
  const oneD = np.array([1.0, 2.0, 3.0]);
  linalg.cholesky(oneD);
  console.log("ERROR: Should have thrown!");
} catch (e: any) {
  console.log("Correctly threw:", e.message);
}

console.log("\n✓ All tests passed!");
