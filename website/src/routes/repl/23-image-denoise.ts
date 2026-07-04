import { numpy as np } from "@jax-js/jax";

// Make a small image with smooth content plus high-frequency texture.
const height = 128;
const width = 128;
const cleanData = new Float32Array(height * width);
const noisyData = new Float32Array(height * width);
for (let y = 0; y < height; y++) {
  for (let x = 0; x < width; x++) {
    const dx = (x - width / 2) / width;
    const dy = (y - height / 2) / height;
    const clean = Math.exp(-18 * (dx * dx + dy * dy));
    const noise = 0.18 * Math.sin(0.8 * x + 1.7 * y);
    cleanData[y * width + x] = clean;
    noisyData[y * width + x] = Math.max(0, Math.min(1, clean + noise));
  }
}

const clean = np.array(cleanData, { shape: [height, width] });
const noisy = np.array(noisyData, { shape: [height, width] });

// Mask high radial frequencies in the real 2D FFT.
const maskData = new Float32Array(height * (width / 2 + 1));
for (let y = 0; y < height; y++) {
  const fy = y <= height / 2 ? y / height : (height - y) / height;
  for (let x = 0; x <= width / 2; x++) {
    const fx = x / width;
    maskData[y * (width / 2 + 1) + x] = Math.hypot(fx, fy) < 0.12 ? 1 : 0;
  }
}

const spectrum = np.fft.rfft2(noisy.ref);
const mask = np.array(maskData, { shape: [height, width / 2 + 1] });
const denoised = np.fft.irfft2({
  real: spectrum.real.mul(mask.ref),
  imag: spectrum.imag.mul(mask),
});

// Left to right: clean, noisy, frequency-filtered.
await displayImage(np.concatenate([clean.ref, noisy.ref, denoised], 1));
