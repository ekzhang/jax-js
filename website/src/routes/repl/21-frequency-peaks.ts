import { lax, numpy as np } from "@jax-js/jax";

// Build a signal with two known tones.
const n = 1024;
const t = np.arange(0, n, 1, { dtype: np.float32 }).div(n);
const signal = np
  .sin(t.ref.mul(2 * Math.PI * 17))
  .add(np.sin(t.mul(2 * Math.PI * 91)).mul(0.5));

// The largest FFT magnitudes recover the active frequency bins.
const spectrum = np.fft.rfft(signal);
const magnitude = np.sqrt(
  np.square(spectrum.real).add(np.square(spectrum.imag)),
);
const [power, bins] = lax.topK(magnitude.slice([1]), 4);

console.log("peak bins =", await bins.add(1).jsAsync());
console.log("peak magnitudes =", await power.jsAsync());
