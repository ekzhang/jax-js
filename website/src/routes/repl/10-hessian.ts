import { hessian, numpy as np } from "@jax-js/jax";

// hessian() is grad-of-grad for scalar-output functions.
const f = (x: np.Array) => {
  const [x0, x1, x2] = x;
  return x0.ref.mul(x1.ref).add(x1.mul(x2)).add(np.square(x0).mul(0.1));
};

// The result is a dense matrix of second partial derivatives.
const H = hessian(f)(np.array([1, 2, 3]));
console.log(await H.jsAsync());
