import { grad, numpy as np } from "@jax-js/jax";

const f = (x: np.Array) => np.sqrt(x.ref.mul(x).sum());
const df = grad(f);

const x = np.array([1, 2, 3, 4]);
console.log(f(x.ref).js());
console.log(df(x).js());
