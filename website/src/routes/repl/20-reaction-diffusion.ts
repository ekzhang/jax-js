import { jit, lax, numpy as np } from "@jax-js/jax";

// Gray-Scott reaction-diffusion: two fields diffuse and react on a grid.
const n = 128;
const kernel = np
  .array([
    [0.05, 0.2, 0.05],
    [0.2, -1.0, 0.2],
    [0.05, 0.2, 0.05],
  ])
  .reshape([1, 1, 3, 3]);

// A 2D Laplacian is just a small convolution.
function laplace(x: np.Array) {
  return lax
    .convGeneralDilated(x.reshape([1, 1, n, n]), kernel.ref, [1, 1], "SAME")
    .reshape([n, n]);
}

// jit() compiles one simulation step, then the loop reuses the compiled step.
const step = jit((u: np.Array, v: np.Array) => {
  const feed = 0.055;
  const kill = 0.062;
  const uvv = u.ref.mul(v.ref).mul(v.ref);
  const du = laplace(u.ref)
    .mul(1.0)
    .sub(uvv.ref)
    .add(np.ones([n, n]).sub(u.ref).mul(feed));
  const dv = laplace(v.ref)
    .mul(0.5)
    .add(uvv)
    .sub(v.ref.mul(feed + kill));
  return [u.add(du.mul(1.0)), v.add(dv.mul(1.0))];
});

// Seed the second chemical in a small square patch.
const uData = new Float32Array(n * n).fill(1);
const vData = new Float32Array(n * n);
for (let y = 54; y < 74; y++) {
  for (let x = 54; x < 74; x++) vData[y * n + x] = 1;
}

let u = np.array(uData, { shape: [n, n] });
let v = np.array(vData, { shape: [n, n] });

// Show the pattern as it forms over time.
const snapshots: np.Array[] = [];
for (let i = 0; i <= 120; i++) {
  if (i % 30 === 0) snapshots.push(v.ref);
  [u, v] = step(u, v);
}

// The REPL has a displayImage() builtin for 2D or RGB arrays.
await displayImage(
  np.concatenate(
    snapshots.map((x) => np.clip(x.mul(4), 0, 1)),
    1,
  ),
);
