import { nn, numpy as np, random } from "@jax-js/jax";

// Scaled dot-product attention over a batch, sequence, heads, and channels.
const [kq, kk, kv] = random.split(random.key(0), 3);
const q = random.normal(kq, [1, 4, 2, 8]);
const k = random.normal(kk, [1, 4, 2, 8]);
const v = random.normal(kv, [1, 4, 2, 8]);

const out = nn.dotProductAttention(q, k, v);

console.log("output shape =", out.shape);
console.log(
  "first token, first head =",
  await out.slice(0, 0, 0, []).jsAsync(),
);
