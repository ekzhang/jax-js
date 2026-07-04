import { nn, numpy as np } from "@jax-js/jax";

// A tiny hand-written MLP: 2 inputs -> 4 hidden units -> 3 classes.
const params = {
  w1: np.array([
    [1.2, -0.8, 0.4, 0.1],
    [-0.2, 0.9, 0.6, -1.0],
  ]),
  b1: np.array([0.1, -0.1, 0.0, 0.2]),
  w2: np.array([
    [1.0, -0.4, 0.2],
    [-0.6, 1.1, 0.1],
    [0.2, 0.3, 0.9],
    [0.4, -0.1, 0.8],
  ]),
  b2: np.array([0.0, 0.1, -0.1]),
};

const x = np.array([
  [1.0, 0.2],
  [-0.3, 1.2],
  [-1.0, -0.8],
]);

// ReLU hidden features followed by softmax class probabilities.
const hidden = nn.relu(np.dot(x, params.w1).add(params.b1));
const logits = np.dot(hidden, params.w2).add(params.b2);
const probs = nn.softmax(logits, -1);

console.log("class probabilities =", await probs.jsAsync());
