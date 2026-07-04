import { numpy as np } from "@jax-js/jax";

// Low-pass filtering keeps slow structure and removes fast oscillations.
const n = 512;
const t = np.arange(0, n, 1, { dtype: np.float32 }).div(n);
const clean = np.sin(t.ref.mul(2 * Math.PI * 5));
const noisy = clean.ref.add(np.sin(t.mul(2 * Math.PI * 96)).mul(0.4));

const spectrum = np.fft.rfft(noisy.ref);
const keep = np.fft.rfftfreq(n).less(0.06).astype(np.float32);
const filtered = np.fft.irfft({
  real: spectrum.real.mul(keep.ref),
  imag: spectrum.imag.mul(keep),
});

// Mean squared error drops after removing the high-frequency component.
const before = np.square(noisy.sub(clean.ref)).mean();
const after = np.square(filtered.sub(clean)).mean();
console.log("MSE before =", await before.jsAsync());
console.log("MSE after =", await after.jsAsync());
