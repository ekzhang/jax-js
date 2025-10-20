import { devices, grad, init, nn, numpy as np, setDevice } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    setDevice(device);
  });

  suite("jax.nn.relu()", () => {
    test("should compute ReLU", () => {
      const x = np.array([-1, 0, 1, 2]);
      const y = nn.relu(x);
      expect(y.js()).toEqual([0, 0, 1, 2]);
    });

    test("should compute ReLU gradient", () => {
      const x = np.array([-10, -1, 1, 2]);
      const gradFn = grad((x: np.Array) => nn.relu(x).sum());
      const gx = gradFn(x);
      expect(gx.js()).toEqual([0, 0, 1, 1]);
    });
  });

  suite("jax.nn.sigmoid()", () => {
    test("should compute sigmoid", () => {
      const x = np.array([-1, 0, 1, 2]);
      const y = nn.sigmoid(x);
      expect(y).toBeAllclose([0.26894142, 0.5, 0.73105858, 0.88079708]);
    });

    test("should compute sigmoid gradient", () => {
      const x = np.array([-10, -1, 1, 2]);
      const gradFn = grad((x: np.Array) => nn.sigmoid(x).sum());
      const gx = gradFn(x);
      expect(gx).toBeAllclose([0.0000454, 0.19661193, 0.19661193, 0.10499359]);
    });
  });

  suite("jax.nn.softSign()", () => {
    test("should compute softsign", () => {
      const x = np.array([-1, 0, 1, 2]);
      const y = nn.softSign(x);
      expect(y).toBeAllclose([-0.5, 0, 0.5, 2 / 3]);
    });

    test("should compute softsign gradient", () => {
      const x = np.array([-10, -1, 1, 2]);
      const gradFn = grad((x: np.Array) => nn.softSign(x).sum());
      const gx = gradFn(x);
      expect(gx).toBeAllclose([1 / 121, 1 / 4, 1 / 4, 1 / 9]);
    });
  });

  suite("jax.nn.leakyRelu()", () => {
    test("works for positive and negative values", () => {
      const x = np.array([-100, -10, -1, 0, 1, 5, 10]);
      const y = nn.leakyRelu(x);
      expect(y).toBeAllclose([-1, -0.1, -0.01, 0, 1, 5, 10]);
    });

    test("takes in alpha as second param", () => {
      const x = np.array([-100, -10, -1, 0, 1, 5, 10]);
      const y = nn.leakyRelu(x, 0.2);
      expect(y).toBeAllclose([-20, -2, -0.2, 0, 1, 5, 10]);
    });

    test("has correct gradient", () => {
      const x = np.array([-10, -1, 1, 2]);
      const gradFn = grad((x: np.Array) => nn.leakyRelu(x).sum());
      const gx = gradFn(x);
      expect(gx).toBeAllclose([0.01, 0.01, 1, 1]);
    });
  });

  suite("jax.nn.gelu()", () => {
    test("estimates gelu for various values", () => {
      // computed from torch.nn.functional.gelu()
      const geluValues = [
        [-10, -1, -0.3, 0, 0.1, 0.5, 1, 3, 5],
        [-0, -0.158655, -0.114627, 0, 0.053983, 0.345731, 0.841345, 2.99595, 5],
      ];
      const x = np.array(geluValues[0]);
      const y = nn.gelu(x);
      expect(y).toBeAllclose(geluValues[1], { atol: 0.02 });
    });
  });

  suite("jax.nn.softmax()", () => {
    test("should compute softmax", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const y = nn.softmax(x);
      expect(y).toBeAllclose([
        [0.09003057, 0.24472848, 0.66524094],
        [0.09003057, 0.24472848, 0.66524094],
      ]);
    });

    test("should compute softmax over 2 axes", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const y = nn.softmax(x, [0, 1]);
      expect(y).toBeAllclose([
        [0.00426978, 0.01160646, 0.03154963],
        [0.08576079, 0.23312202, 0.6336913],
      ]);
    });

    test("should work with no axes", () => {
      expect(nn.softmax(np.zeros([])).js()).toEqual(1);
      expect(nn.softmax(np.array([1, 2, 3]), []).js()).toEqual([1, 1, 1]);
      expect(nn.softmax(np.zeros([0])).js()).toEqual([]);
    });

    test("sum should be constant", () => {
      const x = np.array([-10, -1, 1, 2]);
      const gradFn = grad((x: np.Array) => nn.softmax(x).sum());
      const gx = gradFn(x);
      expect(gx).toBeAllclose([0, 0, 0, 0]);
    });

    test("should compute softmax gradient", () => {
      const x = np.array([-10, -1, 1, 2]);
      const gradFn = grad((x: np.Array) =>
        nn
          .softmax(x)
          .mul(np.array([0, 0, 1, 0])) // Select one element
          .sum(),
      );
      const gx = gradFn(x);
      expect(gx).toBeAllclose([
        -1.1246564e-6, -9.1131842e-3, 1.9215752e-1, -1.8304321e-1,
      ]);
    });

    test("is consistent with logSoftmax", () => {
      const x = np.array([-10, -1, 1, 2]);
      const softmax = nn.softmax(x.ref);
      const logSoftmax = nn.logSoftmax(x);
      expect(np.log(softmax)).toBeAllclose(logSoftmax);
    });
  });

  suite("jax.nn.logsumexp()", () => {
    test("computes logsumexp correctly", () => {
      const x = np.array([-10, -1, 1, 2]);
      const y = nn.logsumexp(x);
      expect(y.js()).toBeCloseTo(2.3490167);

      const z = nn.logsumexp(
        np.array([
          [1, 2, 3, 4],
          [5, 6, 7, 8],
        ]),
      );
      expect(z.js()).toBeCloseTo(8.45834);
    });
  });

  suite("jax.nn.oneHot()", () => {
    test("does basic one-hot encoding", () => {
      const x = np.array([1, 1, 2], { dtype: np.int32 });
      const y = nn.oneHot(x, 3);
      expect(y.js()).toEqual([
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
    });

    test("takes multi-dimensional input", () => {
      const x = np.array(
        [
          [0, 1],
          [2, 1],
        ],
        { dtype: np.int32 },
      );
      const y = nn.oneHot(x, 3);
      expect(y.js()).toEqual([
        [
          [1, 0, 0],
          [0, 1, 0],
        ],
        [
          [0, 0, 1],
          [0, 1, 0],
        ],
      ]);
    });
  });
});
