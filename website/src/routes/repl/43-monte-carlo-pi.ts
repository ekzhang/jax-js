import { numpy as np, random } from "@jax-js/jax";

// Estimate pi by sampling points in a square and counting those in the unit disk.
const points = random.uniform(random.key(0), [100_000, 2], {
  minval: -1,
  maxval: 1,
});
const radiusSquared = np.square(points).sum(1);
const estimate = radiusSquared.less(1).astype(np.float32).mean().mul(4);

console.log("pi estimate =", await estimate.jsAsync());
