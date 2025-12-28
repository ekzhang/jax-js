/**
 * Benchmark Cholesky decomposition performance
 * Compares Pure JS vs optimized JAX-JS implementation
 */

import { linalg, numpy as np, init, blockUntilReady } from "../src/index";

// Generate a diagonally dominant positive definite matrix
function generatePositiveDefinite(n: number): number[][] {
  const matrix: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) {
        row.push(n + 1.0);
      } else {
        row.push(1.0 / (1 + Math.abs(i - j)));
      }
    }
    matrix.push(row);
  }
  return matrix;
}

// Pure JS Cholesky for baseline
function choleskyPureJS(A: number[][]): number[][] {
  const n = A.length;
  const L: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += L[i][k] * L[j][k];
      }
      L[i][j] =
        i === j
          ? Math.sqrt(Math.max(A[i][i] - sum, 1e-10))
          : (A[i][j] - sum) / L[j][j];
    }
  }
  return L;
}

async function benchmark(n: number, numRuns: number = 10) {
  console.log(`\n========== Benchmarking ${n}x${n} matrix ==========`);

  const pdMatrix = generatePositiveDefinite(n);
  const flops = (1 / 3) * n * n * n;

  // Warm-up and Pure JS benchmark
  console.log("\n[Pure JS]");
  const jsTimes: number[] = [];
  for (let i = 0; i < numRuns; i++) {
    const start = performance.now();
    const L = choleskyPureJS(pdMatrix);
    const end = performance.now();
    const time = (end - start) / 1000;
    if (i > 0) jsTimes.push(time); // Skip first run
  }
  const jsAvg = jsTimes.reduce((a, b) => a + b, 0) / jsTimes.length;
  const jsGflops = flops / 1e9 / jsAvg;
  console.log(`  Avg time: ${(jsAvg * 1000).toFixed(3)}ms`);
  console.log(`  GFLOPs: ${jsGflops.toFixed(3)}`);

  // CPU backend
  console.log("\n[JAX-JS CPU]");
  await init("cpu");
  const cpuTimes: number[] = [];
  for (let i = 0; i < numRuns; i++) {
    const A = np.array(pdMatrix, { device: "cpu" });
    A.ref; // Keep alive
    const start = performance.now();
    const L = linalg.cholesky(A.ref);
    L.ref; // Keep alive
    await blockUntilReady(L.ref); // Force computation
    const end = performance.now();
    const time = (end - start) / 1000;
    if (i > 0) cpuTimes.push(time);
    L.dispose();
    A.dispose();
  }
  const cpuAvg = cpuTimes.reduce((a, b) => a + b, 0) / cpuTimes.length;
  const cpuGflops = flops / 1e9 / cpuAvg;
  const cpuSpeedup = jsAvg / cpuAvg;
  console.log(`  Avg time: ${(cpuAvg * 1000).toFixed(3)}ms`);
  console.log(`  GFLOPs: ${cpuGflops.toFixed(3)}`);
  console.log(
    `  Speedup vs Pure JS: ${cpuSpeedup.toFixed(2)}x ${cpuSpeedup > 1 ? "FASTER ✓" : "SLOWER ✗"}`,
  );

  // WASM backend
  console.log("\n[JAX-JS WASM]");
  await init("wasm");
  const wasmTimes: number[] = [];
  for (let i = 0; i < numRuns; i++) {
    const A = np.array(pdMatrix, { device: "wasm" });
    A.ref; // Keep alive
    const start = performance.now();
    const L = linalg.cholesky(A.ref);
    L.ref; // Keep alive
    await blockUntilReady(L.ref);
    const end = performance.now();
    const time = (end - start) / 1000;
    if (i > 0) wasmTimes.push(time);
    L.dispose();
    A.dispose();
  }
  const wasmAvg = wasmTimes.reduce((a, b) => a + b, 0) / wasmTimes.length;
  const wasmGflops = flops / 1e9 / wasmAvg;
  const wasmSpeedup = jsAvg / wasmAvg;
  console.log(`  Avg time: ${(wasmAvg * 1000).toFixed(3)}ms`);
  console.log(`  GFLOPs: ${wasmGflops.toFixed(3)}`);
  console.log(
    `  Speedup vs Pure JS: ${wasmSpeedup.toFixed(2)}x ${wasmSpeedup > 1 ? "FASTER ✓" : "SLOWER ✗"}`,
  );
}

async function main() {
  console.log("Cholesky Decomposition Performance Benchmark");
  console.log("=============================================");

  await benchmark(32, 10);
  await benchmark(64, 10);
  await benchmark(128, 5);
  await benchmark(256, 3);

  console.log("\n========== Benchmark Complete ==========\n");
}

main().catch(console.error);
