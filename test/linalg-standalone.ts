// Simple test script for cholesky - run with: bunx tsx test/linalg-standalone.ts

import { linalg, numpy as np } from "../src/index.ts";

function verifyReconstruction(
  inputJs: any[][],
  resultRef: any,
  isLower: boolean = false,
  label: string = "",
  verbose: boolean = true
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

  if (verbose) {
    console.log(`${label} Reconstruction:`);
    console.log(reconJs);
  }
  console.log(`${label} Max difference:`, maxDiff);
  console.log(`${label} Test passed:`, maxDiff < 1e-5);
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
console.log();

// Helper function to generate a positive definite matrix of size n x n
function generatePositiveDefinite(n: number): number[][] {
  // Create a diagonally dominant matrix which is guaranteed positive definite
  const matrix: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) {
        // Diagonal: make it large enough for diagonal dominance
        row.push(n + 1.0);
      } else {
        // Off-diagonal: use a pattern based on distance from diagonal
        row.push(1.0 / (1 + Math.abs(i - j)));
      }
    }
    matrix.push(row);
  }
  return matrix;
}

// Test 6: 8x8 matrix
console.log("Test 6: 8x8 matrix");
const x8Data = generatePositiveDefinite(8);
const x8 = np.array(x8Data);
const x8Js = x8.ref.js();
console.log("Input (8x8 positive definite matrix)");

const L8 = linalg.cholesky(x8.ref, { lower: true });
console.log("Lower Cholesky computed");
verifyReconstruction(x8Js, L8, true, "Test 6 (8x8)", false);

// Test 7: 16x16 matrix
console.log("Test 7: 16x16 matrix");
const x16Data = generatePositiveDefinite(16);
const x16 = np.array(x16Data);
const x16Js = x16.ref.js();
console.log("Input (16x16 positive definite matrix)");

const L16 = linalg.cholesky(x16.ref, { lower: true });
console.log("Lower Cholesky computed");
verifyReconstruction(x16Js, L16, true, "Test 7 (16x16)", false);

// Test 8: 32x32 matrix
console.log("Test 8: 32x32 matrix");
const x32Data = generatePositiveDefinite(32);
const x32 = np.array(x32Data);
const x32Js = x32.ref.js();
console.log("Input (32x32 positive definite matrix)");

const L32 = linalg.cholesky(x32.ref, { lower: true });
console.log("Lower Cholesky computed");
verifyReconstruction(x32Js, L32, true, "Test 8 (32x32)", false);