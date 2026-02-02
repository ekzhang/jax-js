/**
 * Debug script to understand scan body compilation.
 * Run with: deno run --unstable-webgpu --allow-read --allow-env --no-check test/deno/debug-scan-body.ts
 */

// Import jax-js
const jax = await import("../../src/index.ts");
const { init, defaultDevice, numpy: np, lax, setDebug, makeJaxpr } = jax;

// IMPORTANT: setDebug MUST be called before any jit compilation happens!
setDebug(1);
console.log("DEBUG level set to 1");

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
console.log("Using WebGPU backend\n");

// Simple triangularSolve step function (no existing uniforms)
const step = (
  carry: typeof np.Array.prototype,
  b: typeof np.Array.prototype,
): [typeof np.Array.prototype, typeof np.Array.prototype] => {
  // Lower triangular matrix L - simplified: identity + lower off-diagonal
  // [[1, 0], [0.5, 1]]
  const L = np.array(
    [
      [1, 0],
      [0.5, 1],
    ],
    "float32",
  );
  // Solve L @ x = b for x
  const x = lax.linalg.triangularSolve(L, b.reshape([2, 1]), {
    leftSide: true,
    lower: true,
  });
  const xFlat = x.reshape([2]);
  return [xFlat.ref, xFlat];
};

console.log("\n=== Running triangularSolve scan (no JIT) ===");
const initCarry1 = np.zeros([2], "float32");
const xs1 = np.array(
  [
    [2, 3], // L @ x = [2, 3] => x = [2, 2] (since 1*2=2, 0.5*2+1*2=3)
    [1, 1.5], // L @ x = [1, 1.5] => x = [1, 1]
  ],
  "float32",
);

const [finalCarry1, outputs1] = await lax.scan(step, initCarry1, xs1);
console.log("Final carry (no JIT):", await finalCarry1.data());
// Expected: last solution = [1, 1]

// Now wrap scan in JIT
const { jit } = jax;

console.log("\n=== Running triangularSolve scan (with JIT on step fn) ===");
// JIT the step function, not the whole scan
const jittedStep = jit(
  (
    carry: typeof np.Array.prototype,
    b: typeof np.Array.prototype,
  ): [typeof np.Array.prototype, typeof np.Array.prototype] => {
    const L = np.array(
      [
        [1, 0],
        [0.5, 1],
      ],
      "float32",
    );
    const x = lax.linalg.triangularSolve(L, b.reshape([2, 1]), {
      leftSide: true,
      lower: true,
    });
    const xFlat = x.reshape([2]);
    return [xFlat.ref, xFlat];
  },
);

const initCarry2 = np.zeros([2], "float32");
const xs2 = np.array(
  [
    [2, 3],
    [1, 1.5],
  ],
  "float32",
);

const [finalCarry2, outputs2] = await lax.scan(jittedStep, initCarry2, xs2);
console.log("Final carry (with JIT):", await finalCarry2.data());

console.log("\n=== Results ===");
console.log("Both should be the same");
