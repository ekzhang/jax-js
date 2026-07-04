import { numpy as np } from "@jax-js/jax";

// Fit a quadratic curve by solving min ||X c - y||_2.
const x = np.linspace(-2, 2, 40);
const y = x.ref
  .mul(x.ref)
  .mul(0.7)
  .sub(x.ref.mul(1.5))
  .add(0.25)
  .reshape([-1, 1]);
const design = np.stack([np.square(x.ref), x, np.ones(x.shape)], 1);
const coeffs = np.linalg.lstsq(design, y);

// The recovered coefficients should be close to [0.7, -1.5, 0.25].
console.log("fit y = ax^2 + bx + c");
console.log("[a, b, c] =", await coeffs.reshape([-1]).jsAsync());
