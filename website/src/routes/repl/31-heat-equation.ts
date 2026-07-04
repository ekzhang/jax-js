import { jit, lax, numpy as np } from "@jax-js/jax";

// Diffuse heat by repeatedly applying a Laplacian update.
const n = 96;
const kernel = np
  .array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0],
  ])
  .reshape([1, 1, 3, 3]);

const step = jit((u: np.Array) => {
  const lap = lax
    .conv(u.ref.reshape([1, 1, n, n]), kernel.ref, [1, 1], "SAME")
    .reshape([n, n]);
  return u.add(lap.mul(0.12));
});

const data = new Float32Array(n * n);
for (let y = 38; y < 58; y++) {
  for (let x = 38; x < 58; x++) data[y * n + x] = 1;
}

let u = np.array(data, { shape: [n, n] });
const snapshots: np.Array[] = [];
for (let i = 0; i <= 120; i++) {
  if (i % 30 === 0) snapshots.push(u.ref);
  u = step(u);
}

await displayImage(
  np.concatenate(
    snapshots.map((x) => np.clip(x.mul(2), 0, 1)),
    1,
  ),
);
