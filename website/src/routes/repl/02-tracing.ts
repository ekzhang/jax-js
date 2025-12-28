import { jvp, makeJaxpr, numpy as np } from "@jax-js/jax";

// f(x) = (x + 2) * x
const f = (x: np.Array) => x.ref.add(2).mul(x);

// fdot(x) = 2 * x + 2
const fdot = (x: np.Array) => jvp(f, [x], [np.array(1)])[1];

console.log(makeJaxpr(f)(np.array(2)).jaxpr.toString());

const { jaxpr } = makeJaxpr(fdot)(np.array(2));
console.log(jaxpr.toString());
