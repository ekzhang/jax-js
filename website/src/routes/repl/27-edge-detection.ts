import { lax, numpy as np } from "@jax-js/jax";

// A Sobel filter is a 2D convolution with horizontal and vertical kernels.
const n = 64;
const pixels = new Float32Array(n * n);
for (let y = 0; y < n; y++) {
  for (let x = 0; x < n; x++) {
    pixels[y * n + x] = x > 18 && x < 46 && y > 14 && y < 50 ? 1 : 0;
  }
}

const image = np.array(pixels, { shape: [n, n] });
const sobel = np.array([
  [
    [
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1],
    ],
  ],
  [
    [
      [-1, -2, -1],
      [0, 0, 0],
      [1, 2, 1],
    ],
  ],
]);

const edges = lax.conv(image.ref.reshape([1, 1, n, n]), sobel, [1, 1], "SAME");
const gx = edges.ref.slice(0, 0, [], []);
const gy = edges.slice(0, 1, [], []);
const magnitude = np.sqrt(np.square(gx).add(np.square(gy)));

// Left: input. Right: detected edges.
await displayImage(
  np.concatenate([image, magnitude.ref.div(magnitude.max())], 1),
);
