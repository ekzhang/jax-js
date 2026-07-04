import { grad, makeJaxpr, numpy as np } from "@jax-js/jax";

// makeJaxpr() shows the primitive program produced by tracing.
const f = (x: np.Array) => {
  const y = x.ref.add(2);
  return x.ref.mul(x).add(y).sum();
};

// Compare the forward computation with the traced gradient.
console.log(makeJaxpr(f)(np.ones([2, 3])).jaxpr.toString());
console.log(makeJaxpr(grad(f))(np.ones([2, 3])).jaxpr.toString());
