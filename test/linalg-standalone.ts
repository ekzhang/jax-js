// Simple test script for cholesky - run with: bunx tsx test/linalg-standalone.ts

import { linalg, numpy as np } from "../src/index.ts";

function verifyReconstruction(
  inputJs: any[][],
  resultRef: any,
  isLower: boolean = false,
  label: string = ""
) {
  let reconstructed;
  if (isLower) {
    reconstructed = np.matmul(resultRef.ref, resultRef.ref.transpose());
  } else {
    reconstructed = np.matmul(resultRef.ref.transpose(), resultRef.ref);
  }

  const reconJs = reconstructed.js();
  const flatInput = inputJs.flat();
  const flatRecon = reconJs.flat();

  let maxDiff = 0;
  for (let i = 0; i < flatInput.length; i++) {
    const diff = Math.abs(flatInput[i] - flatRecon[i]);
    if (diff > maxDiff) maxDiff = diff;
  }

  console.log(`${label} Reconstruction:`);
  console.log(reconJs);
  console.log("Max difference:", maxDiff);
  console.log("Test passed:", maxDiff < 1e-5);
  console.log();
  return maxDiff < 1e-5;
}

console.log("Testing linalg.cholesky()...\n");

// Test 1: 2x2 lower triangular
console.log("Test 1: 2x2 lower triangular");
const x = np.array([
  [2.0, 1.0],
  [1.0, 2.0],
]);
const xJs = x.ref.js(); // Keep JS version for verification
console.log("Input:");
console.log(xJs);

const L = linalg.cholesky(x.ref, { lower: true });
console.log("Lower Cholesky (L):");
console.log(L.ref.js());
verifyReconstruction(xJs, L, true, "Test 1");

// Test 2: 2x2 upper triangular (default)
console.log("Test 2: 2x2 upper triangular");
const x2 = np.array([
  [2.0, 1.0],
  [1.0, 2.0],
]);
const x2Js = x2.ref.js();
const U = linalg.cholesky(x2.ref);
console.log("Upper Cholesky (U):");
console.log(U.ref.js());
verifyReconstruction(x2Js, U, false, "Test 2");

// Test 3: 3x3 matrix
console.log("Test 3: 3x3 matrix");
const x3 = np.array([
  [4.0, 2.0, 1.0],
  [2.0, 5.0, 3.0],
  [1.0, 3.0, 6.0],
]);
const x3Js = x3.ref.js();
console.log("Input:");
console.log(x3Js);

const L3 = linalg.cholesky(x3.ref, { lower: true });
console.log("Lower Cholesky (L):");
console.log(L3.ref.js());
verifyReconstruction(x3Js, L3, true, "Test 3");

// Test 4: Error handling - non-square matrix
console.log("Test 4: Error handling - non-square matrix");
try {
  const nonSquare = np.array([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
  ]);
  linalg.cholesky(nonSquare.ref);
  console.log("ERROR: Should have thrown!");
} catch (e: any) {
  console.log("Correctly threw:", e.message);
}
console.log();

// Test 5: Error handling - non-2D array
console.log("Test 5: Error handling - non-2D array");
try {
  const oneD = np.array([1.0, 2.0, 3.0]);
  linalg.cholesky(oneD.ref);
  console.log("ERROR: Should have thrown!");
} catch (e: any) {
  console.log("Correctly threw:", e.message);
}