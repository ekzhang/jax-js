import { grad, nn, numpy as np, tree } from "@jax-js/jax";

type Params = { w: np.Array; b: np.Array };

// Parameters can be plain JavaScript trees of arrays.
const predict = (params: Params, x: np.Array) =>
  np.dot(x, params.w).add(params.b);
const loss = (params: Params, x: np.Array, y: np.Array) =>
  nn.logSoftmax(predict(params, x)).mul(y).sum().neg();

const params = {
  w: np.array([
    [0.2, -0.1],
    [0.4, 0.3],
  ]),
  b: np.zeros([2]),
};
const x = np.array([[1, 2]]);
const y = np.array([[0, 1]]);

// grad() returns a tree with the same structure as the parameter tree.
const grads = grad(loss)(tree.ref(params), x, y);
console.log("dw =", await grads.w.jsAsync());
console.log("db =", await grads.b.jsAsync());
