import {
  defaultDevice,
  devices,
  grad,
  init,
  nn,
  numpy as np,
} from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
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
    // computed from torch.nn.functional.gelu()
    const geluValues = [
      [-10, -1, -0.3, 0, 0.1, 0.5, 3, 5, 500],
      [-0, -0.158655, -0.114627, 0, 0.053983, 0.345731, 2.99595, 5, 500],
    ];

    test("estimates gelu for various values", () => {
      const x = np.array(geluValues[0]);
      const y = nn.gelu(x);
      expect(y).toBeAllclose(geluValues[1], { atol: 1e-3 });
    });

    test("exact gelu for various values", () => {
      const x = np.array(geluValues[0]);
      const y = nn.gelu(x, { approximate: false });
      expect(y).toBeAllclose(geluValues[1], { rtol: 1e-5, atol: 1e-7 });
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
      expect(nn.softmax(np.zeros([]), null).js()).toEqual(1);
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

  suite("jax.nn.standardize()", () => {
    test("standardizes over last axis by default", () => {
      const x = np.array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const y = nn.standardize(x);
      expect(y).toBeAllclose([
        [-1.22474487, 0, 1.22474487],
        [-1.22474487, 0, 1.22474487],
      ]);
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

    test("works for uint32", () => {
      const x = np.array([0, 2, 1], { dtype: np.uint32 });
      const y = nn.oneHot(x, 3);
      expect(y.js()).toEqual([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
      ]);
    });
  });

  suite("jax.nn.dotProductAttention()", () => {
    test("basic attention with rank 4 tensors", () => {
      // Shape: [B=1, L=2, N=1, H=4] - 2 query positions, 1 head, 4-dim vectors
      const query = np.array([
        [[[1, 0, 0, 0]], [[0, 1, 0, 0]]], // B=1, L=2, N=1, H=4
      ]);
      const key = np.array([
        [[[1, 0, 0, 0]], [[0, 1, 0, 0]]], // B=1, S=2, K=1, H=4
      ]);
      const value = np.array([
        [[[1, 2, 3, 4]], [[5, 6, 7, 8]]], // B=1, S=2, K=1, H=4
      ]);

      const out = nn.dotProductAttention(query, key, value);
      expect(out.shape).toEqual([1, 2, 1, 4]);
      // First query [1,0,0,0] attends more to first key [1,0,0,0]
      // Second query [0,1,0,0] attends more to second key [0,1,0,0]
      // With scale = 1/sqrt(4) = 0.5, dot products are 0.5 and 0
      // softmax([0.5, 0]) ≈ [0.622, 0.378]
      // softmax([0, 0.5]) ≈ [0.378, 0.622]
      expect(out.slice(0, 0, 0)).toBeAllclose(
        [
          0.62245935 * 1 + 0.37754068 * 5,
          0.62245935 * 2 + 0.37754068 * 6,
          0.62245935 * 3 + 0.37754068 * 7,
          0.62245935 * 4 + 0.37754068 * 8,
        ],
        { atol: 1e-5 },
      );
    });

    test("basic attention with rank 3 tensors (no batch)", () => {
      // Shape: [L=2, N=1, H=4]
      const query = np.array([[[1, 0, 0, 0]], [[0, 1, 0, 0]]]);
      const key = np.array([[[1, 0, 0, 0]], [[0, 1, 0, 0]]]);
      const value = np.array([[[1, 2, 3, 4]], [[5, 6, 7, 8]]]);

      const out = nn.dotProductAttention(query, key, value);
      expect(out.shape).toEqual([2, 1, 4]);
    });

    test("attention with custom scale", () => {
      // Shape: [B=1, L=2, N=1, H=2]
      const query = np.array([[[[1, 0]], [[0, 1]]]]);
      const key = np.array([[[[1, 0]], [[0, 1]]]]);
      const value = np.array([[[[1, 0]], [[0, 1]]]]);

      // With scale=1.0, dot products are 1 and 0
      // softmax([1, 0]) ≈ [0.731, 0.269]
      const out = nn.dotProductAttention(query, key, value, { scale: 1.0 });
      expect(out.shape).toEqual([1, 2, 1, 2]);
      expect(out.slice(0, 0, 0)).toBeAllclose([0.7310586, 0.26894143], {
        atol: 1e-5,
      });
    });

    test("attention with bias", () => {
      // Shape: [B=1, L=2, N=1, H=2]
      const query = np.array([[[[1, 0]], [[0, 1]]]]);
      const key = np.array([[[[1, 0]], [[0, 1]]]]);
      const value = np.array([[[[1, 0]], [[0, 1]]]]);

      // Bias shape: [B=1, N=1, L=2, S=2]
      // Add large negative bias to block second key for first query
      const bias = np.array([[[[0, -1000], [0, 0]]]]);

      const out = nn.dotProductAttention(query, key, value, { bias });
      expect(out.shape).toEqual([1, 2, 1, 2]);
      // First query should only attend to first key (value [1, 0])
      expect(out.slice(0, 0, 0)).toBeAllclose([1, 0], { atol: 1e-3 });
    });

    test("multi-head attention", () => {
      // B=1, L=2, N=2 (heads), H=2
      const query = np.array([
        [
          [
            [1, 0],
            [0, 1],
          ],
          [
            [1, 0],
            [0, 1],
          ],
        ],
      ]);
      const key = np.array([
        [
          [
            [1, 0],
            [0, 1],
          ],
          [
            [1, 0],
            [0, 1],
          ],
        ],
      ]);
      const value = np.array([
        [
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ],
      ]);

      const out = nn.dotProductAttention(query, key, value);
      expect(out.shape).toEqual([1, 2, 2, 2]);
    });

    test("grouped-query attention (GQA)", () => {
      // Q: B=1, L=2, N=4 (query heads), H=2
      // K/V: B=1, S=2, K=2 (key/value heads), H=2
      // Each pair of query heads shares one K/V head
      const query = np.ones([1, 2, 4, 2]);
      const key = np.ones([1, 2, 2, 2]);
      const value = np.ones([1, 2, 2, 2]);

      const out = nn.dotProductAttention(query, key, value);
      expect(out.shape).toEqual([1, 2, 4, 2]);
    });

    test("multi-query attention (MQA)", () => {
      // Q: B=1, L=2, N=4 (query heads), H=2
      // K/V: B=1, S=2, K=1 (single key/value head), H=2
      // All query heads share the same K/V head
      const query = np.ones([1, 2, 4, 2]);
      const key = np.ones([1, 2, 1, 2]);
      const value = np.ones([1, 2, 1, 2]);

      const out = nn.dotProductAttention(query, key, value);
      expect(out.shape).toEqual([1, 2, 4, 2]);
    });

    test("attention is differentiable", () => {
      // Shape: [B=1, L=2, N=1, H=2]
      const query = np.array([[[[1, 0]], [[0, 1]]]]);
      const key = np.array([[[[1, 0]], [[0, 1]]]]);
      const value = np.array([[[[1, 2]], [[3, 4]]]]);

      const gradFn = grad((q: np.Array) =>
        nn.dotProductAttention(q, key, value).sum(),
      );
      const gq = gradFn(query);
      expect(gq.shape).toEqual(query.shape);
      // Gradient should be non-zero
      expect(np.abs(gq).sum().js()).toBeGreaterThan(0);
    });

    test("throws on rank mismatch", () => {
      const query = np.ones([2, 2, 4]); // rank 3
      const key = np.ones([1, 2, 2, 4]); // rank 4
      const value = np.ones([1, 2, 2, 4]);

      expect(() => nn.dotProductAttention(query, key, value)).toThrow();
    });

    test("throws on key/value shape mismatch", () => {
      const query = np.ones([1, 2, 2, 4]);
      const key = np.ones([1, 2, 2, 4]);
      const value = np.ones([1, 3, 2, 4]); // different S dimension

      expect(() => nn.dotProductAttention(query, key, value)).toThrow();
    });
  });
});
