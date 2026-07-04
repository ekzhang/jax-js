import { numpy as np } from "@jax-js/jax";

// A clean 7 Hz signal with an added high-frequency tone.
const n = 512;
const t = np.arange(0, n, 1, { dtype: np.float32 }).div(n);
const clean = np.sin(t.ref.mul(2 * Math.PI * 7));
const noisy = clean.ref.add(np.sin(t.mul(2 * Math.PI * 80)).mul(0.35));

// Drop all frequency bins above the cutoff, then transform back.
const spectrum = np.fft.rfft(noisy.ref);
const freqs = np.fft.rfftfreq(n);
const keep = freqs.less(0.08).astype(np.float32);
const filtered = np.fft.irfft({
  real: spectrum.real.mul(keep.ref),
  imag: spectrum.imag.mul(keep),
});

// Inspect the first few samples before and after filtering.
console.log("noisy first 12 =", await noisy.slice([0, 12]).jsAsync());
console.log("filtered first 12 =", await filtered.slice([0, 12]).jsAsync());
