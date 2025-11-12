import { grad, jit, nn, numpy as np, random } from "@jax-js/jax";
import { applyUpdates, sgd } from "@jax-js/optax";

// Logistic regression on a sample dataset.
//   > Classifier: y = sigmoid(X @ w)
//   > Loss: Binary cross-entropy,
//           loss(w) = y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)

const wTrue = np.array([2.0, -1.0, 0.5, -1.5]);

const key = random.key(0);
const X = random.uniform(key, [500, 4], { minval: -1, maxval: 1 });
const y = np.dot(X.ref, wTrue.ref).greater(0).astype(np.float32);

// Define loss function (binary cross-entropy).
const lossFn = jit((w: np.Array) => {
  const logits = np.dot(X.ref, w);
  const yPred = nn.sigmoid(logits);
  const loss = np
    .add(
      y.ref.mul(np.log(yPred.ref)),
      np.subtract(1, y.ref).mul(np.log(np.subtract(1, yPred))),
    )
    .mean()
    .neg();
  return loss;
});

const lossGrad = grad(lossFn);

// Training loop.
const steps = 100;
const solver = sgd(0.2);

let w = np.zerosLike(wTrue.ref);
let optState = solver.init(w.ref.ref);
let updates: typeof w;

for (let step = 0; step < steps; step++) {
  const grads = lossGrad(w.ref);
  [updates, optState] = solver.update(grads, optState);
  w = applyUpdates(w, updates);
  if (step % 10 === 9) {
    await w.wait();
    const loss = await lossFn(w.ref).jsAsync();
    console.log(`Step ${step + 1}: loss = ${loss}`);
  }
}

// Output learned weights.
console.log("Learned weights:", await w.jsAsync());
