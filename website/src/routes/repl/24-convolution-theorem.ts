import { numpy as np } from "@jax-js/jax";

// Circular convolution can be computed by multiplying FFTs.
const a = np.array([1, 2, 1, 0, 0, 0, 0, 0]);
const b = np.array([1, -1, 0.5, 0, 0, 0, 0, 0]);

const fa = np.fft.fft({ real: a, imag: np.zeros([8]) });
const fb = np.fft.fft({ real: b, imag: np.zeros([8]) });

// Complex multiply: (ar + i ai) * (br + i bi).
const real = fa.real.ref.mul(fb.real.ref).sub(fa.imag.ref.mul(fb.imag.ref));
const imag = fa.real.mul(fb.imag).add(fa.imag.mul(fb.real));
const convolved = np.fft.ifft({ real, imag }).real;

console.log("circular convolution =", await convolved.jsAsync());
