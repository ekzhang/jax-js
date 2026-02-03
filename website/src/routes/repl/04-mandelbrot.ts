import { jit, numpy as np } from "@jax-js/jax";

// This example draws a Mandelbrot fractal using array operations and jit() for each step.
// The computation is already fully vectorized: each iteration operates on all 750x600 pixels
// simultaneously, making it GPU-friendly. The jit() compilation fuses operations into
// efficient kernels, and WebGPU backends process all pixels in parallel.
const width = 750;
const height = 600;

function mandelbrotIteration(
  A: np.Array,
  B: np.Array,
  V: np.Array,
  X: np.Array,
  Y: np.Array,
) {
  const Asq = A.ref.mul(A.ref);
  const Bsq = B.ref.mul(B.ref);
  V = V.add(Asq.ref.add(Bsq.ref).less(100).astype(np.float32));
  const A2 = np.clip(Asq.sub(Bsq).add(X), -50, 50);
  const B2 = np.clip(A.mul(B).mul(2).add(Y), -50, 50);
  return [A2, B2, V];
}

function calculateMandelbrot(iters: number) {
  const x = np.linspace(-2, 0.5, width);
  const y = np.linspace(-1, 1, height);

  const [X, Y] = np.meshgrid([x, y]);

  const f = jit(mandelbrotIteration);

  let A = np.zeros(X.shape);
  let B = np.zeros(Y.shape);
  let V = np.zeros(X.shape);
  for (let i = 0; i < iters; i++) {
    [A, B, V] = f(A, B, V, X.ref, Y.ref);
  }
  X.dispose();
  Y.dispose();
  A.dispose();
  B.dispose();

  return V;
}

const ar = calculateMandelbrot(100);
const image = np.subtract(1, ar.div(100));

// The REPL has a displayImage() builtin for drawing image pixels.
await displayImage(image);
