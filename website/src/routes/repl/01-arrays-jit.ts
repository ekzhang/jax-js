import { jit, numpy as np } from "@jax-js/jax";

// Start with a vector on the current device.
const x = np.linspace(-3, 3, 12);

// jit() traces this function once, then runs the fused operation on later calls.
const fused = jit((x: np.Array) => {
  const radius = np.sqrt(np.square(x.ref).add(1));
  return np.sin(x).mul(radius);
});

// Use jsAsync() when you want to bring an array back to JavaScript.
console.log("x =", await x.ref.jsAsync());
console.log("fused result =", await fused(x).jsAsync());
