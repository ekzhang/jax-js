import { nn, numpy as np } from "@jax-js/jax";

// Causal attention only lets position i attend to positions <= i.
const q = np.ones([4, 1, 2]);
const k = np.ones([4, 1, 2]);
const v = np.arange(0, 8, 1, { dtype: np.float32 }).reshape([4, 1, 2]);

const causal = nn.dotProductAttention(q.ref, k.ref, v.ref, { isCausal: true });
const full = nn.dotProductAttention(q, k, v);

console.log("causal output =", await causal.reshape([4, 2]).jsAsync());
console.log("full output =", await full.reshape([4, 2]).jsAsync());
