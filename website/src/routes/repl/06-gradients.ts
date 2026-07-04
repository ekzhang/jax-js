import { grad, numpy as np, valueAndGrad } from "@jax-js/jax";

// valueAndGrad() returns the loss and its gradient from one traced function.
const loss = (x: np.Array) => {
  const residual = np.sin(x.ref).sub(0.5);
  return residual.ref.mul(residual).mean();
};

const x = np.linspace(-2, 2, 8);
const [value, gradient] = valueAndGrad(loss)(x.ref);

console.log("loss =", await value.jsAsync());
console.log("gradient =", await gradient.jsAsync());

// grad() also works on scalar JavaScript inputs.
const scalar = (x: np.Array) => x.ref.mul(x.ref).mul(x.ref).add(x);
console.log("d/dx (x^3 + x) at 5 =", await grad(scalar)(5).jsAsync());
