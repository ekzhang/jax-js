import { numpy as np, random } from "@jax-js/jax";

// Cholesky turns a covariance matrix into a linear transform of iid normals.
const mean = np.array([1, -1]);
const covariance = np.array([
  [1.0, 0.8],
  [0.8, 1.5],
]);
const L = np.linalg.cholesky(covariance.ref);
const z = random.normal(random.key(0), [5000, 2]);
const samples = np.dot(z, L.transpose()).add(mean);

// The sample statistics should approach the requested mean and covariance.
console.log("sample mean =", await samples.ref.mean(0).jsAsync());
console.log("sample covariance =", await np.cov(samples.transpose()).jsAsync());
