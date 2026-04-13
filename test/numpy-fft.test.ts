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
});
