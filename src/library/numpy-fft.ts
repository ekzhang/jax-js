// Port of the `jax.numpy.fft` module, Fast Fourier Transform.

import {
  arange,
  array,
  Array,
  concatenate,
  cos,
  DType,
  flip,
  roll,
  sin,
  zerosLike,
} from "./numpy";
import { isFloatDtype } from "../alu";
import { jit } from "../frontend/jaxpr";
import {
  checkAxis,
  deepEqual,
  invertPermutation,
  normalizeAxis,
  range,
  rep,
} from "../utils";

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
  if (!Number.isInteger(n) || n < 1 || 2 ** Math.floor(Math.log2(n)) !== n) {
    throw new Error(
      `jax.numpy.fft.${name}: size must be a power of two, got ${n}`,
    );
  }
}

function checkRealInput(name: string, a: Array) {
  if (!isFloatDtype(a.dtype)) {
    throw new Error(
      `jax.numpy.fft.${name}: input must have a float dtype, got ${a.dtype}`,
    );
  }
}

function checkFrequencyArgs(name: string, n: number, d: number) {
  if (!Number.isInteger(n) || n < 1) {
    throw new Error(
      `jax.numpy.fft.${name}: n must be a positive integer, got ${n}`,
    );
  }
  if (!Number.isFinite(d) || d === 0) {
    throw new Error(
      `jax.numpy.fft.${name}: d must be a finite non-zero number, got ${d}`,
    );
  }
}

// TODO: Replace this with `lax.sliceInDim` once added.
function sliceAlongAxis(
  a: Array,
  axis: number,
  start: number,
  end: number,
): Array {
  const index: (number | [] | [number] | [number, number] | null)[] =
    globalThis.Array.from({ length: a.ndim }, () => []);
  index[axis] = [start, end];
  return a.slice(...index);
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

function transformN(
  name: string,
  a: ComplexPair,
  axes: number[] | null,
  transform: (a: ComplexPair, axis: number) => ComplexPair,
): ComplexPair {
  checkPairInput(name, a);
  const normalizedAxes = normalizeAxis(axes, a.real.ndim, false);
  let result = a;
  for (const axis of normalizedAxes) {
    result = transform(result, axis);
  }
  return result;
}

/**
 * Compute an N-dimensional discrete Fourier transform.
 *
 * Currently, every transformed axis must have a power-of-two size.
 */
export function fftn(
  a: ComplexPair,
  axes: number[] | null = null,
): ComplexPair {
  return transformN("fftn", a, axes, fft);
}

/**
 * Compute a two-dimensional discrete Fourier transform.
 *
 * Currently, every transformed axis must have a power-of-two size.
 */
export function fft2(a: ComplexPair, axes: number[] = [-2, -1]): ComplexPair {
  return fftn(a, axes);
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

/**
 * Compute an N-dimensional inverse discrete Fourier transform.
 *
 * Currently, every transformed axis must have a power-of-two size.
 */
export function ifftn(
  a: ComplexPair,
  axes: number[] | null = null,
): ComplexPair {
  return transformN("ifftn", a, axes, ifft);
}

/**
 * Compute a two-dimensional inverse discrete Fourier transform.
 *
 * Currently, every transformed axis must have a power-of-two size.
 */
export function ifft2(a: ComplexPair, axes: number[] = [-2, -1]): ComplexPair {
  return ifftn(a, axes);
}

/**
 * Compute a one-dimensional FFT of real input.
 *
 * The output stores only the non-negative frequency terms. Currently, the size
 * of the axis must be a power of two.
 */
export function rfft(a: Array, axis: number = -1): ComplexPair {
  checkRealInput("rfft", a);
  axis = checkAxis(axis, a.ndim);
  const n = a.shape[axis];
  checkPowerOfTwo("rfft", n);

  const result = fft({ real: a, imag: zerosLike(a.ref) }, axis);
  const stop = Math.floor(n / 2) + 1;
  return {
    real: sliceAlongAxis(result.real, axis, 0, stop),
    imag: sliceAlongAxis(result.imag, axis, 0, stop),
  };
}

/**
 * Compute the inverse of `rfft`.
 *
 * The real output length is inferred as `2 * (m - 1)`, where `m` is the packed
 * spectrum length. Currently, the inferred length must be a power of two.
 */
export function irfft(a: ComplexPair, axis: number = -1): Array {
  checkPairInput("irfft", a);
  const { real, imag } = a;
  axis = checkAxis(axis, real.ndim);
  const m = real.shape[axis];
  if (m < 2) {
    throw new Error(
      `jax.numpy.fft.irfft: packed input length must be at least 2, got ${m}`,
    );
  }
  const n = 2 * (m - 1);
  checkPowerOfTwo("irfft", n);

  const mirroredReal = flip(sliceAlongAxis(real.ref, axis, 1, m - 1), axis);
  const mirroredImag = flip(sliceAlongAxis(imag.ref, axis, 1, m - 1), axis).mul(
    -1,
  );
  const result = ifft(
    {
      real: concatenate([real, mirroredReal], axis),
      imag: concatenate([imag, mirroredImag], axis),
    },
    axis,
  );
  result.imag.dispose();
  return result.real;
}

/**
 * Compute an N-dimensional FFT of real input.
 *
 * The final transformed axis stores only the non-negative frequency terms.
 * Currently, every transformed axis must have a power-of-two size.
 */
export function rfftn(a: Array, axes: number[] | null = null): ComplexPair {
  checkRealInput("rfftn", a);
  const normalizedAxes = normalizeAxis(axes, a.ndim, false);
  if (normalizedAxes.length === 0) {
    return { real: a, imag: zerosLike(a.ref) };
  }

  const realAxis = normalizedAxes[normalizedAxes.length - 1];
  let result = rfft(a, realAxis);
  for (const axis of normalizedAxes.slice(0, -1)) {
    result = fft(result, axis);
  }
  return result;
}

/**
 * Compute a two-dimensional FFT of real input.
 *
 * The final transformed axis stores only the non-negative frequency terms.
 * Currently, every transformed axis must have a power-of-two size.
 */
export function rfft2(a: Array, axes: number[] = [-2, -1]): ComplexPair {
  return rfftn(a, axes);
}

/**
 * Compute the inverse of `rfftn`.
 *
 * The real output length for the final transformed axis is inferred as
 * `2 * (m - 1)`. Currently, every transformed axis must have a power-of-two
 * size.
 */
export function irfftn(a: ComplexPair, axes: number[] | null = null): Array {
  checkPairInput("irfftn", a);
  const normalizedAxes = normalizeAxis(axes, a.real.ndim, false);
  if (normalizedAxes.length === 0) {
    a.imag.dispose();
    return a.real;
  }

  const realAxis = normalizedAxes[normalizedAxes.length - 1];
  let result = a;
  for (const axis of normalizedAxes.slice(0, -1)) {
    result = ifft(result, axis);
  }
  return irfft(result, realAxis);
}

/**
 * Compute the inverse of `rfft2`.
 *
 * Currently, every transformed axis must have a power-of-two size.
 */
export function irfft2(a: ComplexPair, axes: number[] = [-2, -1]): Array {
  return irfftn(a, axes);
}

/**
 * Compute the FFT of a Hermitian-symmetric signal.
 *
 * The real output length is inferred as `2 * (m - 1)`, where `m` is the packed
 * Hermitian input length. Currently, the inferred length must be a power of two.
 */
export function hfft(a: ComplexPair, axis: number = -1): Array {
  checkPairInput("hfft", a);
  axis = checkAxis(axis, a.real.ndim);
  const n = 2 * (a.real.shape[axis] - 1);
  checkPowerOfTwo("hfft", n);
  return irfft({ real: a.real, imag: a.imag.mul(-1) }, axis).mul(n);
}

/**
 * Compute the inverse of `hfft`.
 *
 * Currently, the size of the axis must be a power of two.
 */
export function ihfft(a: Array, axis: number = -1): ComplexPair {
  checkRealInput("ihfft", a);
  axis = checkAxis(axis, a.ndim);
  const n = a.shape[axis];
  checkPowerOfTwo("ihfft", n);

  const result = rfft(a, axis);
  return {
    real: result.real.div(n),
    imag: result.imag.mul(-1).div(n),
  };
}

/** Return sample frequencies for a discrete Fourier transform. */
export function fftfreq(n: number, d: number = 1.0): Array {
  checkFrequencyArgs("fftfreq", n, d);
  const scale = 1 / (n * d);
  const positiveEnd = Math.floor((n - 1) / 2) + 1;
  const values: number[] = [];
  for (let i = 0; i < positiveEnd; i++) values.push(i * scale);
  for (let i = -Math.floor(n / 2); i < 0; i++) values.push(i * scale);
  return array(values, { dtype: DType.Float32 });
}

/** Return sample frequencies for a real-input discrete Fourier transform. */
export function rfftfreq(n: number, d: number = 1.0): Array {
  checkFrequencyArgs("rfftfreq", n, d);
  const scale = 1 / (n * d);
  const values: number[] = [];
  for (let i = 0; i <= Math.floor(n / 2); i++) values.push(i * scale);
  return array(values, { dtype: DType.Float32 });
}

/** Shift the zero-frequency component to the center of the spectrum. */
export function fftshift(
  a: Array,
  axes: number | number[] | null = null,
): Array {
  const normalizedAxes = normalizeAxis(axes, a.ndim, false);
  const shifts = normalizedAxes.map((axis) => Math.floor(a.shape[axis] / 2));
  return roll(a, shifts, normalizedAxes);
}

/** Undo `fftshift`. */
export function ifftshift(
  a: Array,
  axes: number | number[] | null = null,
): Array {
  const normalizedAxes = normalizeAxis(axes, a.ndim, false);
  const shifts = normalizedAxes.map((axis) => -Math.floor(a.shape[axis] / 2));
  return roll(a, shifts, normalizedAxes);
}
