import { jit, nn, numpy as np, random, tree, valueAndGrad } from "@jax-js/jax";
import { adam, applyUpdates } from "@jax-js/optax";

// A tiny two-layer classifier on two curved classes.
type Params = {
  w1: np.Array;
  b1: np.Array;
  w2: np.Array;
  b2: np.Array;
};

const points: number[][] = [];
const labels: number[] = [];
for (let i = 0; i < 128; i++) {
  const t = (i / 64) * Math.PI;
  const cls = i >= 64 ? 1 : 0;
  const r = cls ? 1.2 : 0.7;
  points.push([
    r * Math.cos(t) + (cls ? 0.25 : -0.25),
    r * Math.sin(t) * (cls ? -1 : 1),
  ]);
  labels.push(cls);
}

// Move the dataset and parameters into jax-js arrays.
const x = np.array(points);
const y = np.array(labels);
const [k1, k2] = random.split(random.key(0), 2);

let params: Params = {
  w1: random.normal(k1, [2, 16]).mul(0.7),
  b1: np.zeros([16]),
  w2: random.normal(k2, [16, 1]).mul(0.7),
  b2: np.zeros([1]),
};

// Compile the forward pass; the training loop can reuse it every step.
const predict = jit((params: Params, x: np.Array) => {
  const h = np.tanh(np.dot(x, params.w1).add(params.b1));
  return np.dot(h, params.w2).add(params.b2).reshape([-1]);
});

// Binary cross-entropy from logits: softplus(logit) - y * logit.
const loss = (params: Params, x: np.Array, y: np.Array) => {
  const logits = predict(tree.ref(params), x);
  return nn.softplus(logits.ref).sub(y.mul(logits)).mean();
};

const valueGrad = valueAndGrad(loss);
const solver = adam(0.04);
let optState = solver.init(tree.ref(params));

// Optimize with Optax-style updates.
for (let step = 0; step < 80; step++) {
  const [value, grads] = valueGrad(tree.ref(params), x.ref, y.ref);
  const [updates, newOptState] = solver.update(
    grads,
    optState,
    tree.ref(params),
  );
  optState = newOptState;
  params = applyUpdates(params, updates);
  if (step % 20 === 0) console.log(step, await value.jsAsync());
}

// Evaluate the learned classifier on the training points.
const logits = predict(tree.ref(params), x.ref);
const accuracy = np
  .equal(logits.greater(0).astype(np.float32), y)
  .astype(np.float32)
  .mean();
console.log("accuracy =", await accuracy.jsAsync());
