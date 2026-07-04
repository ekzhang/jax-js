import { grad, numpy as np, vmap } from "@jax-js/jax";

// vmap() lifts a scalar function over a batch axis.
const f = (x: np.Array) => np.sin(x.ref).mul(x);
const xs = np.linspace(-3, 3, 9);

// Transformations compose: vmap(grad(f)) gives per-example derivatives.
console.log("f(xs) =", await vmap(f)(xs.ref).jsAsync());
console.log("df(xs) =", await vmap(grad(f))(xs).jsAsync());
