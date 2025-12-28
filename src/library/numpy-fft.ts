// Port of the `jax.numpy.fft` module, Fast Fourier Transform.

import { arange, Array, concatenate, cos, sin } from "./numpy";
import { isFloatDtype } from "../alu";
import { jit } from "../frontend/jaxpr";
import { checkAxis, deepEqual, invertPermutation, range, rep } from "../utils";

/**
 * A pair of arrays representing real and imaginary part `a + bj`. Both arrays
 * must have the same shape.
 */
export type ComplexPair = {
  real: Array;
  imag: Array;
};

function checkPairInput(name: string, a: ComplexPair) {
  const fullName = `jax.numpy.fft.${name}`;
  if (!deepEqual(a.real.shape, a.imag.shape)) {
    throw new Error(
      `${fullName}: real and imaginary parts must have the same shape, got ${JSON.stringify(a.real.shape)} and ${JSON.stringify(a.imag.shape)}`,
    );
  }
  if (a.real.dtype !== a.imag.dtype) {
    throw new Error(
      `${fullName}: real and imaginary parts must have the same dtype, got ${a.real.dtype} and ${a.imag.dtype}`,
    );
  }
  if (!isFloatDtype(a.real.dtype)) {
    throw new Error(
      `${fullName}: input must have a float dtype, got ${a.real.dtype}`,
    );
  }
}

function checkPowerOfTwo(name: string, n: number) {
  if ((n & (n - 1)) !== 0) {
    throw new Error(
      `jax.numpy.fft.${name}: size must be a power of two, got ${n}`,
    );
  }
}

const fftUpdate = jit(
  function fftUpdate(i: number, { real, imag }: ComplexPair): ComplexPair {
    const half = 2 ** i;

    real = real.reshape([-1, 2 * half]);
    imag = imag.reshape([-1, 2 * half]);

    const k = arange(0, half, 1, { dtype: real.dtype });
    const theta = k.mul(-Math.PI / half);
    const wr = cos(theta.ref);
    const wi = sin(theta);

    const ur = real.ref.slice([], [0, half]);
    const ui = imag.ref.slice([], [0, half]);
    const vr = real.slice([], [half, 2 * half]);
    const vi = imag.slice([], [half, 2 * half]);

    // t = w * v
    const tr = vr.ref.mul(wr.ref).sub(vi.ref.mul(wi.ref));
    const ti = vr.mul(wi).add(vi.mul(wr));

    // store [u + t, u - t]
    return {
      real: concatenate([ur.ref.add(tr.ref), ur.sub(tr)], -1),
      imag: concatenate([ui.ref.add(ti.ref), ui.sub(ti)], -1),
    };
  },
  { staticArgnums: [0] },
);

/**
 * Compute a one-dimensional discrete Fourier transform.
 *
 * Currently, the size of the axis must be a power of two.
 */
export function fft(a: ComplexPair, axis: number = -1): ComplexPair {
  checkPairInput("fft", a);
  let { real, imag } = a;
  axis = checkAxis(axis, real.ndim);
  const n = real.shape[axis];
  checkPowerOfTwo("fft", n);
  const logN = Math.log2(n);

  // If axis is not at the end, move it to the end
  let perm: number[] | null = null;
  if (axis !== real.ndim - 1) {
    perm = range(real.ndim);
    perm.splice(axis, 1);
    perm.push(axis);
    real = real.transpose(perm);
    imag = imag.transpose(perm);
  }

  // Cooley-Tukey FFT (radix-2)
  const originalShape = real.shape;
  real = real
    .reshape([-1, ...rep(logN, 2)])
    .transpose([0, ...range(1, logN + 1).reverse()])
    .flatten();
  imag = imag
    .reshape([-1, ...rep(logN, 2)])
    .transpose([0, ...range(1, logN + 1).reverse()])
    .flatten();

  // Hack: If you don't do it, the arrays might be lazy and grow exponentially.
  for (let i = 0; i < logN; i++) {
    ({ real, imag } = fftUpdate(i, { real, imag }));
  }
  real = real.reshape(originalShape);
  imag = imag.reshape(originalShape);

  // If axis was moved, move it back
  if (perm !== null) {
    real = real.transpose(invertPermutation(perm));
    imag = imag.transpose(invertPermutation(perm));
  }
  return { real, imag };
}

/**
 * Compute a one-dimensional inverse discrete Fourier transform.
 *
 * Currently, the size of the axis must be a power of two.
 */
export function ifft(a: ComplexPair, axis: number = -1): ComplexPair {
  checkPairInput("ifft", a);
  let { real, imag } = a;
  axis = checkAxis(axis, real.ndim);
  const n = real.shape[axis];
  checkPowerOfTwo("ifft", n);

  // ifft(a) = 1/n * conj(fft(conj(a)))
  imag = imag.mul(-1);
  const result = fft({ real, imag }, axis);
  return {
    real: result.real.div(n),
    imag: result.imag.mul(-1).div(n),
  };
}
