import { numpy as np } from "@jax-js/jax";

// matmul broadcasts leading dimensions.
const a = np.arange(24).reshape([2, 3, 4]);
const b = np.arange(20).reshape([4, 5]);

// einsum spells out the same contraction explicitly.
const viaMatmul = np.matmul(a.ref, b.ref);
const viaEinsum = np.einsum("...ik,kj->...ij", a, b);

console.log("shape =", viaMatmul.shape);
console.log("same =", np.allclose(viaMatmul, viaEinsum));
