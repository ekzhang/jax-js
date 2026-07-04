import { numpy as np, random } from "@jax-js/jax";

// random.multivariateNormal uses Cholesky under the hood.
const mean = np.array([1, -2]);
const covariance = np.array([
  [1.0, 0.6],
  [0.6, 2.0],
]);

const samples = random.multivariateNormal(
  random.key(0),
  mean.ref,
  covariance.ref,
  [4000],
);

console.log("sample mean =", await samples.ref.mean(0).jsAsync());
console.log("sample covariance =", await np.cov(samples.transpose()).jsAsync());
