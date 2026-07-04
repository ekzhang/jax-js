import { grad, jit, nn, numpy as np, random } from "@jax-js/jax";
import { applyUpdates, sgd } from "@jax-js/optax";

// Synthetic binary classification from a hidden linear separator.
const trueW = np.array([2.0, -1.0, 0.5, -1.5]);
const x = random.uniform(random.key(0), [500, 4], { minval: -1, maxval: 1 });
const y = np.dot(x.ref, trueW).greater(0).astype(np.float32);

// Logistic loss from logits: softplus(logit) - y * logit.
const loss = jit((w: np.Array) => {
  const logits = np.dot(x.ref, w);
  return nn.softplus(logits.ref).sub(y.ref.mul(logits)).mean();
});
const lossGrad = grad(loss);

let w = np.zeros([4]);
const opt = sgd(0.4);
let state = opt.init(w.ref);

// Plain SGD is enough for this convex problem.
for (let step = 0; step < 80; step++) {
  const [updates, nextState] = opt.update(lossGrad(w.ref), state);
  state = nextState;
  w = applyUpdates(w, updates);
}

const pred = np.dot(x, w.ref).greater(0).astype(np.float32);
const accuracy = np.equal(pred, y).astype(np.float32).mean();
console.log("learned weights =", await w.jsAsync());
console.log("accuracy =", await accuracy.jsAsync());
