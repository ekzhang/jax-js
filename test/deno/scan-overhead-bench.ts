/**
 * Benchmark scan iteration overhead with a simple body.
 * Run with: deno run --unstable-webgpu --allow-read --allow-env --no-check test/deno/scan-overhead-bench.ts
 */

const jax = await import("../../src/index.ts");
const { init, defaultDevice, numpy, lax, setDebug } = jax;
const np = numpy;

// Check WebGPU availability
if (!navigator.gpu) {
  console.log("WebGPU not available");
  Deno.exit(1);
}

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
  console.log("No WebGPU adapter");
  Deno.exit(1);
}

const availableDevices = await init();
if (!availableDevices.includes("webgpu")) {
  console.log("WebGPU not in jax-js devices");
  Deno.exit(1);
}

defaultDevice("webgpu");
console.log("Using WebGPU backend");
console.log("GPU:", adapter.info);
console.log();

// Simple cumulative sum with additional ops (simulating Kalman-like complexity)
const complexStep = (
  carry: typeof np.Array.prototype,
  x: typeof np.Array.prototype,
): [typeof np.Array.prototype, typeof np.Array.prototype] => {
  // Simulate multiple ops per iteration:
  // 1. Add x to carry
  const sum1 = np.add(carry.ref, x.ref);
  // 2. Scale by 0.999 (decay)
  const sum2 = np.multiply(sum1, 0.999);
  // 3. Add a small offset
  const sum3 = np.add(sum2, 0.001);
  // 4. Another multiply
  const result = np.multiply(sum3.ref, 1.0);

  return [result.ref, result];
};

console.log("=== Scan overhead benchmark ===\n");
console.log("Body: 4 ops (add, mul, add, mul) per iteration\n");

// Size of arrays (small to isolate overhead)
const arraySize = 16;
const initCarry = np.zeros([arraySize], "float32");

// Benchmark different iteration counts
const results: { n: number; ms: number }[] = [];

for (const n of [10, 50, 100, 200, 500, 1000]) {
  // Create xs
  const xsData = [];
  for (let i = 0; i < n; i++) {
    xsData.push(Array(arraySize).fill(0.01));
  }
  const xs = np.array(xsData, "float32");

  // Warmup
  if (n === 10) {
    const [c, _] = await lax.scan(complexStep, initCarry.ref, xs.ref);
    await c.data();
  }

  const start = performance.now();
  const [finalCarry, outputs] = await lax.scan(
    complexStep,
    initCarry.ref,
    xs.ref,
  );
  // Wait for completion
  await finalCarry.data();
  const elapsed = performance.now() - start;

  results.push({ n, ms: elapsed });
  console.log(
    `n=${n.toString().padStart(4)}: ${elapsed.toFixed(1).padStart(7)}ms (${(elapsed / n).toFixed(3)}ms/iter)`,
  );
}

// Compute overhead by linear regression
// If overhead dominates: time ≈ a + b*n, where a is fixed cost, b is per-iter overhead
console.log("\n=== Analysis ===\n");

const n1 = results[0].n;
const t1 = results[0].ms;
const n2 = results[results.length - 1].n;
const t2 = results[results.length - 1].ms;

const perIterMs = (t2 - t1) / (n2 - n1);
const fixedMs = t1 - perIterMs * n1;

console.log(
  `Linear fit: time = ${fixedMs.toFixed(1)}ms + ${perIterMs.toFixed(3)}ms × n`,
);
console.log(`Estimated per-iteration overhead: ${perIterMs.toFixed(3)}ms`);
console.log(`Estimated fixed cost: ${fixedMs.toFixed(1)}ms`);

// For Kalman with 100 observations
const kalmanIters = 100;
const estimatedKalman = fixedMs + perIterMs * kalmanIters;
console.log(
  `\nProjected time for n=${kalmanIters}: ${estimatedKalman.toFixed(1)}ms`,
);

// If we had batched dispatch (1 roundtrip total):
const batchedOverhead = 0.2; // ~200µs for single submit
console.log(
  `With batched scan (theoretical): ${batchedOverhead.toFixed(1)}ms + compute time`,
);
console.log(
  `Potential speedup: ~${(estimatedKalman / batchedOverhead).toFixed(0)}×`,
);

console.log("\n=== Done ===");
