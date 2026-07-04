import { numpy as np, random, vmap } from "@jax-js/jax";

// vmap() batches a random function over a batch of keys.
const keys = random.split(random.key(0), 8);
const sampleMean = (key: np.Array) => random.normal(key, [4]).mean();

const means = vmap(sampleMean)(keys);
console.log("eight sample means =", await means.jsAsync());
