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
  });
});
