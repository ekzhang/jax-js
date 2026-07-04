import { defaultDevice, devices, init, numpy as np } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();
const fftDevices = devices.filter((device) => device !== "webgl");

function dft(
  real: number[],
  imag: number[],
): { real: number[]; imag: number[] } {
  const n = real.length;
  const outReal: number[] = [];
  const outImag: number[] = [];
  for (let k = 0; k < n; k++) {
    let sumReal = 0;
    let sumImag = 0;
    for (let t = 0; t < n; t++) {
      const angle = (-2 * Math.PI * t * k) / n;
      const c = Math.cos(angle);
      const s = Math.sin(angle);
      sumReal += real[t] * c - imag[t] * s;
      sumImag += real[t] * s + imag[t] * c;
    }
    outReal.push(sumReal);
    outImag.push(sumImag);
  }
  return { real: outReal, imag: outImag };
}

const fftCases: [string, number[], number[]][] = [
  ["simple real input", [0, 1, 2, 3], [0, 0, 0, 0]],
  ["impulse signal", [1, 0, 0, 0], [0, 0, 0, 0]],
  ["constant signal", [3, 3, 3, 3], [0, 0, 0, 0]],
  ["length-one input", [3], [-2]],
  [
    "complex input (length 8)",
    [1, 3, 0, -2, 5, 1, 2, -1],
    [2, -1, 4, 1, 0, -3, 2, 1],
  ],
  ["mixed-radix length 6", [1, 3, 0, -2, 5, 1], [2, -1, 4, 1, 0, -3]],
  ["prime length 5", [4, -1, 2, 0, 3], [0, 2, -3, 1, 5]],
];

suite.each(fftDevices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  suite("jax.numpy.fft.fft()", () => {
    test.each(fftCases)("computes FFT of %s", (_name, real, imag) => {
      const expected = dft(real, imag);
      const result = np.fft.fft({
        real: np.array(real),
        imag: np.array(imag),
      });

      expect(result.real).toBeAllclose(expected.real, { atol: 1e-4 });
      expect(result.imag).toBeAllclose(expected.imag, { atol: 1e-4 });
    });

    test("FFT along a non-default axis", () => {
      // 2x4 matrix, FFT along axis=0 (length 2)
      const real = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const imag = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]);
      const result = np.fft.fft({ real, imag }, 0);

      expect(result.real).toBeAllclose([
        [6, 8, 10, 12],
        [-4, -4, -4, -4],
      ]);
      expect(result.imag).toBeAllclose([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]);
    });
  });

  suite("jax.numpy.fft.ifft()", () => {
    test("computes IFFT of a simple complex input", () => {
      const real = np.array([6, -2, -2, -2]);
      const imag = np.array([0, 2, 0, -2]);
      const result = np.fft.ifft({ real, imag });

      expect(result.real).toBeAllclose([0, 1, 2, 3]);
      expect(result.imag).toBeAllclose([0, 0, 0, 0]);
    });

    test.each([
      ["power-of-two", [1, 2, 3, 4, 5, 6, 7, 8], [-5, 9, 0, 3, -1, 4, 2, 8]],
      ["mixed-radix", [1, 2, 3, 4, 5, 6], [-5, 9, 0, 3, -1, 4]],
    ])(
      "FFT followed by IFFT returns original %s input",
      (_name, realData, imagData) => {
        const real = np.array(realData);
        const imag = np.array(imagData);
        const fftResult = np.fft.fft({ real: real.ref, imag: imag.ref });
        const result = np.fft.ifft(fftResult);

        expect(result.real).toBeAllclose(real, { atol: 1e-5 });
        expect(result.imag).toBeAllclose(imag, { atol: 1e-5 });
      },
    );

    test("IFFT followed by FFT returns original input", () => {
      const real = np.array([2, -1, 4, 0]);
      const imag = np.array([1, 3, -2, 5]);
      const ifftResult = np.fft.ifft({ real: real.ref, imag: imag.ref });
      const fftResult = np.fft.fft(ifftResult);

      expect(fftResult.real).toBeAllclose(real, { atol: 1e-5 });
      expect(fftResult.imag).toBeAllclose(imag, { atol: 1e-5 });
    });
  });

  suite("jax.numpy.fft.fftn()", () => {
    test("computes FFTN over all axes", () => {
      const real = np.array([
        [1, 2],
        [3, 4],
      ]);
      const imag = np.zeros([2, 2]);
      const result = np.fft.fftn({ real, imag });

      expect(result.real).toBeAllclose([
        [10, -2],
        [-4, 0],
      ]);
      expect(result.imag).toBeAllclose([
        [0, 0],
        [0, 0],
      ]);
    });

    test("FFTN followed by IFFTN returns original input", () => {
      const real = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const imag = np.array([
        [1, -1, 2, -2],
        [3, -3, 4, -4],
      ]);
      const fftResult = np.fft.fft2({ real: real.ref, imag: imag.ref });
      const result = np.fft.ifft2(fftResult);

      expect(result.real).toBeAllclose(real, { atol: 1e-5 });
      expect(result.imag).toBeAllclose(imag, { atol: 1e-5 });
    });
  });

  suite("jax.numpy.fft.rfft()", () => {
    test("computes RFFT of a simple real input", () => {
      const x = np.array([0, 1, 2, 3]);
      const result = np.fft.rfft(x);

      expect(result.real).toBeAllclose([6, -2, -2], { atol: 1e-5 });
      expect(result.imag).toBeAllclose([0, 2, 0], { atol: 1e-5 });
    });

    test("RFFT followed by IRFFT returns original input", () => {
      const x = np.array([0, 1, 2, 3, 4, 5, 6, 7]);
      const spectrum = np.fft.rfft(x.ref);
      const result = np.fft.irfft(spectrum);

      expect(result).toBeAllclose(x, { atol: 1e-5 });
    });

    test("computes RFFT2 with packed final axis", () => {
      const x = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const result = np.fft.rfft2(x);

      expect(result.real).toBeAllclose(
        [
          [36, -4, -4],
          [-16, 0, 0],
        ],
        { atol: 1e-5 },
      );
      expect(result.imag).toBeAllclose(
        [
          [0, 4, 0],
          [0, 0, 0],
        ],
        { atol: 1e-5 },
      );
    });

    test("RFFTN followed by IRFFTN returns original input", () => {
      const x = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const spectrum = np.fft.rfftn(x.ref);
      const result = np.fft.irfftn(spectrum);

      expect(result).toBeAllclose(x, { atol: 1e-5 });
    });

    test("computes HFFT and IHFFT", () => {
      const real = np.array([1, 2, 4]);
      const imag = np.array([0, 3, 0]);
      const hfftResult = np.fft.hfft({ real, imag });
      expect(hfftResult).toBeAllclose([9, 3, 1, -9], {
        atol: 1e-5,
      });
      const ihfftResult = np.fft.ihfft(np.array([9, 3, 1, -9]));

      expect(ihfftResult.real).toBeAllclose([1, 2, 4], { atol: 1e-5 });
      expect(ihfftResult.imag).toBeAllclose([0, 3, 0], { atol: 1e-5 });
    });
  });

  suite("jax.numpy.fftfreq()", () => {
    test("computes FFT sample frequencies", () => {
      expect(np.fft.fftfreq(4)).toBeAllclose([0, 0.25, -0.5, -0.25]);
      expect(np.fft.fftfreq(5, 0.5)).toBeAllclose([0, 0.4, 0.8, -0.8, -0.4]);
    });

    test("computes RFFT sample frequencies", () => {
      expect(np.fft.rfftfreq(4)).toBeAllclose([0, 0.25, 0.5]);
      expect(np.fft.rfftfreq(5, 0.5)).toBeAllclose([0, 0.4, 0.8]);
    });

    test("shifts FFT spectra", async () => {
      expect(await np.fft.fftshift(np.array([0, 1, 2, 3])).jsAsync()).toEqual([
        2, 3, 0, 1,
      ]);
      expect(
        await np.fft.fftshift(np.array([0, 1, 2, 3, 4])).jsAsync(),
      ).toEqual([3, 4, 0, 1, 2]);
      expect(
        await np.fft.ifftshift(np.array([0, 1, 2, 3, 4])).jsAsync(),
      ).toEqual([2, 3, 4, 0, 1]);
    });
  });
});
