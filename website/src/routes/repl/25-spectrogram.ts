import { numpy as np } from "@jax-js/jax";

// A chirp sweeps upward in frequency over time.
const n = 1024;
const t = np.arange(0, n, 1, { dtype: np.float32 }).div(n);
const chirp = np.sin(t.ref.mul(t.ref).mul(2 * Math.PI * 180));
const tone = np.sin(t.mul(2 * Math.PI * 24)).mul(0.35);
const signal = chirp.add(tone);

// Short-time FFT: slice overlapping windows and stack their spectra.
const win = 128;
const hop = 32;
const window = np.sin(
  np.arange(0, win, 1, { dtype: np.float32 }).div(win).mul(Math.PI),
);
const frames: np.Array[] = [];
for (let start = 0; start + win <= n; start += hop) {
  const frame = signal.ref.slice([start, start + win]).mul(window.ref);
  const spec = np.fft.rfft(frame);
  frames.push(
    np.log(np.sqrt(np.square(spec.real).add(np.square(spec.imag))).add(1)),
  );
}

// Frequency is vertical, time is horizontal.
const image = np.stack(frames).transpose();
await displayImage(image.ref.div(image.max()));
