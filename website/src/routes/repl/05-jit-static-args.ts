import { jit, numpy as np } from "@jax-js/jax";

// staticArgnums means degree is a compile-time constant.
const polynomial = jit(
  (x: np.Array, degree: number) => {
    let y = np.ones(x.shape);
    for (let i = 0; i < degree; i++) y = y.mul(x.ref);
    x.dispose();
    return y;
  },
  { staticArgnums: [1] },
);

const x = np.array([1, 2, 3, 4]);
// Different static values compile different specialized functions.
console.log("x^3 =", await polynomial(x.ref, 3).jsAsync());
console.log("x^5 =", await polynomial(x, 5).jsAsync());
