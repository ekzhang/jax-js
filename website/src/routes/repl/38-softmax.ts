import { nn, numpy as np } from "@jax-js/jax";

// Cross-entropy from logits is log-softmax plus a one-hot target.
const logits = np.array([
  [1.0, 0.5, 2.0],
  [0.2, 2.4, -0.1],
]);
const labels = np.array([2, 1], { dtype: np.int32 });

const probs = nn.softmax(logits.ref, -1);
const targets = nn.oneHot(labels, 3);
const loss = nn.logSoftmax(logits, -1).mul(targets).sum(1).mean().neg();

console.log("probabilities =", await probs.jsAsync());
console.log("mean cross entropy =", await loss.jsAsync());
