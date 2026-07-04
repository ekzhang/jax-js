import { numpy as np, random } from "@jax-js/jax";

// Build 4D observations from a 2D latent source plus small noise.
const samples = 512;
const latent = random.normal(random.key(0), [samples, 2]);
const mixing = np.array([
  [2.2, 0.2, 0.8, -1.1],
  [-0.5, 1.6, 0.3, 0.7],
]);
const noise = random.normal(random.key(1), [samples, 4]).mul(0.08);
const x = np.dot(latent, mixing).add(noise);
const centered = x.ref.sub(np.mean(x.ref, 0));

// PCA can be read directly from the singular values of centered data.
const s = np.linalg.svd(centered, { computeUv: false });
const variance = np.square(s.ref).div(samples - 1);
const explained = variance.ref.div(variance.sum());

console.log("singular values =", await s.jsAsync());
console.log("explained variance =", await explained.jsAsync());
