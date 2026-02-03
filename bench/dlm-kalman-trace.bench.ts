/**
 * Full Kalman filter forward step benchmark.
 * Tests the WebGPU buffer limit issue and constant packing solution.
 *
 * Run with: deno run --unstable-webgpu --allow-read --allow-env --no-check test/deno/dlm-kalman-trace.ts
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

console.log("GPU:", adapter.info.description);
console.log("Limits:", {
  maxStorageBuffersPerShaderStage:
    adapter.limits?.maxStorageBuffersPerShaderStage,
  maxBindGroups: adapter.limits?.maxBindGroups,
});

const availableDevices = await init();
if (!availableDevices.includes("webgpu")) {
  console.log("WebGPU not in jax-js devices");
  Deno.exit(1);
}

defaultDevice("webgpu");
setDebug(0); // Disable debug output for performance
console.log("Using WebGPU backend\n");

// Full Kalman filter forward step (from dlm-js)
// System matrices for local linear trend: state = [level, slope]
const F = np.array([[1.0, 0.0]], "float32"); // [1, 2] observation matrix
const G = np.array(
  [
    [1.0, 1.0],
    [0.0, 1.0],
  ],
  "float32",
); // [2, 2] transition matrix
const W = np.array(
  [
    [0.01, 0.0],
    [0.0, 0.01],
  ],
  "float32",
); // [2, 2] state noise

// Full Kalman filter forward step
const forwardStep = (
  carry: { x: typeof np.Array.prototype; C: typeof np.Array.prototype },
  inp: { y: typeof np.Array.prototype; V2: typeof np.Array.prototype },
): [
  { x: typeof np.Array.prototype; C: typeof np.Array.prototype },
  {
    x_pred: typeof np.Array.prototype;
    C_pred: typeof np.Array.prototype;
    v: typeof np.Array.prototype;
    Cp: typeof np.Array.prototype;
  },
] => {
  const { x: xi, C: Ci } = carry;
  const { y: yi, V2: V2i } = inp;

  // Innovation: v = y - F·x
  const v = np.subtract(yi.ref, np.matmul(F.ref, xi.ref));

  // Innovation covariance: Cp = F·C·F' + V²
  const Cp = np.add(np.einsum("ij,jk,lk->il", F.ref, Ci.ref, F.ref), V2i.ref);

  // Kalman gain: K = G·C·F' / Cp
  const GCFt = np.einsum("ij,jk,lk->il", G.ref, Ci.ref, F.ref);
  const K = np.divide(GCFt.ref, Cp.ref);

  // L = G - K·F
  const L = np.subtract(G.ref, np.matmul(K.ref, F.ref));

  // Next state prediction: x_next = G·x + K·v
  const x_next = np.add(np.matmul(G.ref, xi.ref), np.matmul(K.ref, v.ref));

  // Next covariance: C_next = G·C·L' + W
  const C_next = np.add(np.einsum("ij,jk,lk->il", G.ref, Ci.ref, L.ref), W.ref);

  const output = {
    x_pred: xi.ref,
    C_pred: Ci.ref,
    v: v.ref,
    Cp: Cp.ref,
  };

  // Dispose intermediates
  GCFt.dispose();
  L.dispose();
  K.dispose();
  v.dispose();
  Cp.dispose();

  return [{ x: x_next, C: C_next }, output];
};

console.log("=== Full Kalman Filter Scan Test ===\n");

// Initial state
const x0 = np.array([[100.0], [0.0]], "float32"); // initial state [level, slope]
const C0 = np.array(
  [
    [1000.0, 0.0],
    [0.0, 1000.0],
  ],
  "float32",
); // initial covariance

// Create test data: varying observation counts
for (const n of [10, 50, 100]) {
  const yData = [];
  const vData = [];
  for (let i = 0; i < n; i++) {
    yData.push([[100 + i * 2 + Math.sin(i) * 5]]); // observations with trend + noise
    vData.push([[100]]); // observation variance
  }
  const y_arr = np.array(yData, "float32");
  const V2_arr = np.array(vData, "float32");

  console.log(`n=${n.toString().padStart(3)}: `);

  try {
    const start = performance.now();
    const [finalCarry, outputs] = await lax.scan(
      forwardStep,
      { x: x0.ref, C: C0.ref },
      { y: y_arr, V2: V2_arr },
    );
    const elapsed = performance.now() - start;

    // Consume outputs
    await finalCarry.x.data();
    await outputs.v.data();

    console.log(
      `       ${elapsed.toFixed(1).padStart(7)}ms (${(elapsed / n).toFixed(2)}ms/iter)`,
    );
  } catch (e) {
    console.log(`       ERROR: ${e.message.slice(0, 50)}...`);
  }
}

console.log("\n=== Done ===");
