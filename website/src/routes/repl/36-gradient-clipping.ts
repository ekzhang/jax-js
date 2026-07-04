import { numpy as np, tree } from "@jax-js/jax";
import { clipByGlobalNorm, treeNorm } from "@jax-js/optax";

// Clip a nested gradient tree to a fixed global norm.
const grads = {
  w: np.array([30, -40, 15]),
  b: np.array([10]),
};

const clipping = clipByGlobalNorm(5);
const state = clipping.init(tree.ref(grads));
const rawNorm = treeNorm(tree.ref(grads));
const [clipped] = clipping.update(grads, state);

console.log("raw norm =", await rawNorm.jsAsync());
console.log("clipped norm =", await treeNorm(clipped).jsAsync());
