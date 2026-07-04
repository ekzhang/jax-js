import { nn, numpy as np } from "@jax-js/jax";

// Common neural-network activations are elementwise array functions.
const x = np.linspace(-4, 4, 9);
const table = np.stack(
  [x.ref, nn.relu(x.ref), nn.sigmoid(x.ref), np.tanh(x)],
  1,
);

console.log("[x, relu, sigmoid, tanh] =");
console.log(await table.jsAsync());
