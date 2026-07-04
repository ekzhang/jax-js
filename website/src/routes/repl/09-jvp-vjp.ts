import { jvp, numpy as np, vjp } from "@jax-js/jax";

// This vector-valued function has two outputs: [x^2, sin(x)].
const f = (x: np.Array) => np.stack([x.ref.mul(x.ref), np.sin(x)]);

// jvp() pushes a tangent forward through the function.
const [y, dy] = jvp(f, [np.array(2)], [np.array(1)]);
console.log("f(2) =", await y.jsAsync());
console.log("JVP in direction 1 =", await dy.jsAsync());

// vjp() pulls a cotangent backward through the function.
const [out, pullback] = vjp(f, [np.array(2)]);
const [dx] = pullback(np.array([1, 1]));
console.log("VJP cotangent [1, 1] =", await dx.jsAsync());
out.dispose();
pullback.dispose();
