import { defaultDevice, devices, init, numpy as np } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  suite("jax.numpy.fft.fft()", () => {
    test("computes FFT of a simple real input", async () => {
      const real = np.array([0, 1, 2, 3]);
      const imag = np.array([0, 0, 0, 0]);
      const result = np.fft.fft({ real, imag });
      expect(await result.real.jsAsync()).toBeAllclose([6, -2, -2, -2]);
      expect(await result.imag.jsAsync()).toBeAllclose([0, 2, 0, -2]);
    });

    test("FFT of an impulse (delta) signal", async () => {
      // FFT of [1, 0, 0, 0] should be all ones (flat spectrum)
      const real = np.array([1, 0, 0, 0]);
      const imag = np.array([0, 0, 0, 0]);
      const result = np.fft.fft({ real, imag });
      expect(await result.real.jsAsync()).toBeAllclose([1, 1, 1, 1]);
      expect(await result.imag.jsAsync()).toBeAllclose([0, 0, 0, 0]);
    });

    test("FFT of a constant signal", async () => {
      // FFT of [c, c, c, c] should be [4c, 0, 0, 0]
      const real = np.array([3, 3, 3, 3]);
      const imag = np.array([0, 0, 0, 0]);
      const result = np.fft.fft({ real, imag });
      expect(await result.real.jsAsync()).toBeAllclose([12, 0, 0, 0]);
      expect(await result.imag.jsAsync()).toBeAllclose([0, 0, 0, 0]);
    });

    test("FFT with complex input (length 8)", async () => {
      const real = np.array([1, 3, 0, -2, 5, 1, 2, -1]);
      const imag = np.array([2, -1, 4, 1, 0, -3, 2, 1]);
      const result = np.fft.fft({ real, imag });
      expect(await result.real.jsAsync()).toBeAllclose(
        [9, 1.5355339, -2, -6.7071068, 7, -5.5355339, 10, -5.2928932],
        { atol: 1e-4 },
      );
      expect(await result.imag.jsAsync()).toBeAllclose(
        [6, 4.7071068, -11, -2.1213203, 10, 3.2928932, 3, 2.1213203],
        { atol: 1e-4 },
      );
    });

    test("FFT along a non-default axis", async () => {
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
      // FFT of length-2 along axis 0: [a+b, a-b]
      expect(await result.real.jsAsync()).toBeAllclose([
        [6, 8, 10, 12],
        [-4, -4, -4, -4],
      ]);
      expect(await result.imag.jsAsync()).toBeAllclose([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ]);
    });
  });

  suite("jax.numpy.fft.ifft()", () => {
    test("computes IFFT of a simple complex input", async () => {
      const real = np.array([6, -2, -2, -2]);
      const imag = np.array([0, 2, 0, -2]);
      const result = np.fft.ifft({ real, imag });
      expect(await result.real.jsAsync()).toBeAllclose([0, 1, 2, 3]);
      expect(await result.imag.jsAsync()).toBeAllclose([0, 0, 0, 0]);
    });

    test("FFT followed by IFFT returns original input", async () => {
      const real = np.array([1, 2, 3, 4, 5, 6, 7, 8]);
      const imag = np.array([-5, 9, 0, 3, -1, 4, 2, 8]);
      const fftResult = np.fft.fft({ real: real.ref, imag: imag.ref });
      const ifftResult = np.fft.ifft(fftResult);
      expect(await ifftResult.real.jsAsync()).toBeAllclose(real, {
        atol: 1e-5,
      });
      expect(await ifftResult.imag.jsAsync()).toBeAllclose(imag, {
        atol: 1e-5,
      });
    });

    test("IFFT followed by FFT returns original input", async () => {
      const real = np.array([2, -1, 4, 0]);
      const imag = np.array([1, 3, -2, 5]);
      const ifftResult = np.fft.ifft({ real: real.ref, imag: imag.ref });
      const fftResult = np.fft.fft(ifftResult);
      expect(await fftResult.real.jsAsync()).toBeAllclose(real, {
        atol: 1e-5,
      });
      expect(await fftResult.imag.jsAsync()).toBeAllclose(imag, {
        atol: 1e-5,
      });
    });
  });

  suite("jax.numpy.fft.fftn()", () => {
    test("computes FFTN over all axes", async () => {
      const real = np.array([
        [1, 2],
        [3, 4],
      ]);
      const imag = np.zeros([2, 2]);
      const result = np.fft.fftn({ real, imag });

      expect(await result.real.jsAsync()).toBeAllclose([
        [10, -2],
        [-4, 0],
      ]);
      expect(await result.imag.jsAsync()).toBeAllclose([
        [0, 0],
        [0, 0],
      ]);
    });

    test("FFTN followed by IFFTN returns original input", async () => {
      const real = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const imag = np.array([
        [1, -1, 2, -2],
        [3, -3, 4, -4],
      ]);
      const fftResult = np.fft.fft2({ real: real.ref, imag: imag.ref });
      const ifftResult = np.fft.ifft2(fftResult);

      expect(await ifftResult.real.jsAsync()).toBeAllclose(real, {
        atol: 1e-5,
      });
      expect(await ifftResult.imag.jsAsync()).toBeAllclose(imag, {
        atol: 1e-5,
      });
    });
  });

  suite("jax.numpy.fft.rfft()", () => {
    test("computes RFFT of a simple real input", async () => {
      const x = np.array([0, 1, 2, 3]);
      const result = np.fft.rfft(x);

      expect(await result.real.jsAsync()).toBeAllclose([6, -2, -2], {
        atol: 1e-5,
      });
      expect(await result.imag.jsAsync()).toBeAllclose([0, 2, 0], {
        atol: 1e-5,
      });
    });

    test("RFFT followed by IRFFT returns original input", async () => {
      const x = np.array([0, 1, 2, 3, 4, 5, 6, 7]);
      const spectrum = np.fft.rfft(x.ref);
      const result = np.fft.irfft(spectrum);

      expect(await result.jsAsync()).toBeAllclose(x, { atol: 1e-5 });
    });

    test("computes RFFT2 with packed final axis", async () => {
      const x = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const result = np.fft.rfft2(x);

      expect(await result.real.jsAsync()).toBeAllclose(
        [
          [36, -4, -4],
          [-16, 0, 0],
        ],
        { atol: 1e-5 },
      );
      expect(await result.imag.jsAsync()).toBeAllclose(
        [
          [0, 4, 0],
          [0, 0, 0],
        ],
        { atol: 1e-5 },
      );
    });

    test("RFFTN followed by IRFFTN returns original input", async () => {
      const x = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ]);
      const spectrum = np.fft.rfftn(x.ref);
      const result = np.fft.irfftn(spectrum);

      expect(await result.jsAsync()).toBeAllclose(x, { atol: 1e-5 });
    });

    test("computes HFFT and IHFFT", async () => {
      const real = np.array([1, 2, 4]);
      const imag = np.array([0, 3, 0]);
      const hfftResult = np.fft.hfft({ real, imag });
      expect(await hfftResult.jsAsync()).toBeAllclose([9, 3, 1, -9], {
        atol: 1e-5,
      });

      const ihfftResult = np.fft.ihfft(np.array([9, 3, 1, -9]));
      expect(await ihfftResult.real.jsAsync()).toBeAllclose([1, 2, 4], {
        atol: 1e-5,
      });
      expect(await ihfftResult.imag.jsAsync()).toBeAllclose([0, 3, 0], {
        atol: 1e-5,
      });
    });
  });

  suite("jax.numpy.fftfreq()", () => {
    test("computes FFT sample frequencies", async () => {
      expect(await np.fft.fftfreq(4).jsAsync()).toBeAllclose([
        0, 0.25, -0.5, -0.25,
      ]);
      expect(await np.fft.fftfreq(5, 0.5).jsAsync()).toBeAllclose([
        0, 0.4, 0.8, -0.8, -0.4,
      ]);
    });

    test("computes RFFT sample frequencies", async () => {
      expect(await np.fft.rfftfreq(4).jsAsync()).toBeAllclose([0, 0.25, 0.5]);
      expect(await np.fft.rfftfreq(5, 0.5).jsAsync()).toBeAllclose([
        0, 0.4, 0.8,
      ]);
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
