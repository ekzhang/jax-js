/**
 * Tests for lax.scan implementation
 */

import {
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
  suite,
  test,
} from "vitest";

import {
  defaultDevice,
  devices,
  grad,
  init,
  jit,
  jvp,
  lax,
  numpy as np,
  setScanPathCallback,
  vmap,
} from "../src";

const devicesAvailable = await init();

describe("lax.scan", () => {
  beforeAll(async () => {
    const devices = await init();
    if (devices.includes("cpu")) {
      defaultDevice("cpu");
    }
  });

  describe("stackPyTree", () => {
    it("stacks list of pytrees", async () => {
      const tree1 = { a: np.array([1.0]), b: np.array([2.0]) };
      const tree2 = { a: np.array([3.0]), b: np.array([4.0]) };
      const tree3 = { a: np.array([5.0]), b: np.array([6.0]) };

      const stacked = lax.stackPyTree([tree1, tree2, tree3]);

      const aData = await (stacked as { a: np.Array; b: np.Array }).a.data();
      const bData = await (stacked as { a: np.Array; b: np.Array }).b.data();

      expect(Array.from(aData)).toEqual([1, 3, 5]);
      expect(Array.from(bData)).toEqual([2, 4, 6]);
    });
  });

  describe("scan basic", () => {
    it("computes cumulative sum", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry.ref];
      };

      const init = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

      const [finalCarry, outputs] = await lax.scan(step, init, xs);

      const finalData = await finalCarry.data();
      expect(finalData[0]).toBeCloseTo(15.0);

      const outputData = await outputs.data();
      expect(Array.from(outputData)).toEqual([1, 3, 6, 10, 15]);
    });

    it("computes factorial-like recurrence", async () => {
      // x(t) = x(t-1) * t
      const step = (carry: np.Array, t: np.Array): [np.Array, np.Array] => {
        const newCarry = np.multiply(carry, t);
        return [newCarry, newCarry.ref];
      };

      const init = np.array([1.0]);
      const ts = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

      const [final, outputs] = await lax.scan(step, init, ts);

      const finalData = await final.data();
      expect(finalData[0]).toBeCloseTo(120.0); // 5!

      const outputData = await outputs.data();
      expect(Array.from(outputData)).toEqual([1, 2, 6, 24, 120]);
    });
  });

  describe("scan with pytree carry", () => {
    it("tracks two values simultaneously", async () => {
      type Carry = { sum: np.Array; count: np.Array };

      const step = (carry: Carry, x: np.Array): [Carry, np.Array] => {
        const newSum = np.add(carry.sum, x);
        const newCount = np.add(carry.count, np.array([1.0]));
        return [
          { sum: newSum, count: newCount },
          np.divide(newSum.ref, newCount.ref),
        ];
      };

      const init = { sum: np.array([0.0]), count: np.array([0.0]) };
      const xs = np.array([[2.0], [4.0], [6.0], [8.0]]);

      const [final, runningMeans] = await lax.scan(step, init, xs);

      const sumData = await final.sum.data();
      const countData = await final.count.data();

      expect(sumData[0]).toBeCloseTo(20.0);
      expect(countData[0]).toBeCloseTo(4.0);

      // Running means: 2/1, 6/2, 12/3, 20/4 = 2, 3, 4, 5
      const meanData = await runningMeans.data();
      expect(Array.from(meanData)).toEqual([2, 3, 4, 5]);
    });
  });

  describe("scan with pytree inputs", () => {
    it("handles pytree xs", async () => {
      type X = { a: np.Array; b: np.Array };
      type Carry = np.Array;
      type Y = np.Array;

      const step = (carry: Carry, x: X): [Carry, Y] => {
        const sum = np.add(x.a, x.b);
        const newCarry = np.add(carry, sum);
        return [newCarry, newCarry.ref];
      };

      const init = np.array([0.0]);
      const xs = {
        a: np.array([[1.0], [2.0], [3.0]]),
        b: np.array([[10.0], [20.0], [30.0]]),
      };

      const [final, _outputs] = await lax.scan(step, init, xs);

      // (1+10) + (2+20) + (3+30) = 11 + 22 + 33 = 66
      const finalData = await final.data();
      expect(finalData[0]).toBeCloseTo(66.0);
    });
  });
});

/**
 * Multi-backend scan tests - runs on all available devices including WebGPU
 */
suite.each(devices)("lax.scan device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  test("cumulative sum", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry.ref];
    };

    const initCarry = np.array([0.0]);
    const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

    const [finalCarry, outputs] = await lax.scan(step, initCarry, xs);

    const finalData = await finalCarry.data();
    expect(finalData[0]).toBeCloseTo(15.0);

    const outputData = await outputs.data();
    expect(Array.from(outputData)).toEqual([1, 3, 6, 10, 15]);
  });

  test("jit + scan", async () => {
    // Note: lax.scan is async, so we test jit on the step function itself
    const step = jit((carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry.ref];
    });

    const initCarry = np.array([0.0]);
    const xs = np.array([[1.0], [2.0], [3.0]]);

    const [finalCarry, outputs] = await lax.scan(step, initCarry, xs);

    const finalData = await finalCarry.data();
    expect(finalData[0]).toBeCloseTo(6.0);

    const outputData = await outputs.data();
    expect(Array.from(outputData)).toEqual([1, 3, 6]);
  });

  test("pytree carry with multiple arrays", async () => {
    type Carry = { sum: np.Array; product: np.Array };

    const step = (carry: Carry, x: np.Array): [Carry, np.Array] => {
      const newSum = np.add(carry.sum.ref, x.ref);
      const newProduct = np.multiply(carry.product, x);
      return [{ sum: newSum, product: newProduct }, newSum.ref];
    };

    const initCarry = { sum: np.array([0.0]), product: np.array([1.0]) };
    const xs = np.array([[2.0], [3.0], [4.0]]);

    const [final, outputs] = await lax.scan(step, initCarry, xs);

    const sumData = await final.sum.data();
    const productData = await final.product.data();

    expect(sumData[0]).toBeCloseTo(9.0); // 2+3+4
    expect(productData[0]).toBeCloseTo(24.0); // 2*3*4

    const outputData = await outputs.data();
    expect(Array.from(outputData)).toEqual([2, 5, 9]);
  });

  test("larger iteration count", async () => {
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const newCarry = np.add(carry, x);
      return [newCarry, newCarry.ref];
    };

    const n = 100;
    const initCarry = np.array([0.0]);
    const xs = np.ones([n, 1]);

    const [finalCarry, _outputs] = await lax.scan(step, initCarry, xs);

    const finalData = await finalCarry.data();
    expect(finalData[0]).toBeCloseTo(n);
  });

  test("elementwise ops in scan body", async () => {
    // More complex body: carry = tanh(carry + x * 0.1)
    const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
      const scaled = np.multiply(x, np.array([0.1]));
      const added = np.add(carry, scaled);
      const newCarry = np.tanh(added);
      return [newCarry, newCarry.ref];
    };

    const initCarry = np.array([0.0]);
    const xs = np.array([[1.0], [1.0], [1.0], [1.0], [1.0]]);

    const [finalCarry, _outputs] = await lax.scan(step, initCarry, xs);

    // Should converge towards tanh saturation
    const finalData = await finalCarry.data();
    expect(finalData[0]).toBeGreaterThan(0);
    expect(finalData[0]).toBeLessThan(1);
  });

  describe("compiled-loop scan", () => {
    test("small array", async () => {
      // Small carry array (64 elements) - uses compiled-loop on WebGPU/WASM
      // This test verifies fusion works for elementwise bodies
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry.ref];
      };

      const size = 64;
      const initCarry = np.zeros([size]);
      const xs = np.ones([10, size]); // 10 iterations

      // requirePath: "fused" ensures this doesn't silently regress to fallback
      const [finalCarry, _outputs] = await lax.scan(step, initCarry, xs, {
        requirePath: "fused",
      });

      const finalData = await finalCarry.data();
      // Each element should be 10 (10 iterations of adding 1)
      expect(finalData[0]).toBeCloseTo(10.0);
      expect(finalData[size - 1]).toBeCloseTo(10.0);
    });

    test("large array", async () => {
      // Large carry array (512 elements) - uses compiled-loop on WebGPU/WASM
      // For elementwise bodies, each element's scan is independent so any size works
      // This test verifies fusion works for larger arrays
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry.ref];
      };

      const size = 512;
      const initCarry = np.zeros([size]);
      const xs = np.ones([5, size]); // 5 iterations

      // requirePath: "fused" ensures this doesn't silently regress to fallback
      const [finalCarry, _outputs] = await lax.scan(step, initCarry, xs, {
        requirePath: "fused",
      });

      const finalData = await finalCarry.data();
      // Each element should be 5 (5 iterations of adding 1)
      expect(finalData[0]).toBeCloseTo(5.0);
      expect(finalData[size - 1]).toBeCloseTo(5.0);
    });

    test("with constants", async () => {
      // Test that constants captured in the body work correctly
      // This exercises compiled-loop with constants (WASM/WebGPU)
      // A fallback would indicate constant handling is broken
      const scale = np.array([2.0]);
      const offset = np.array([1.0]);

      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        // newCarry = carry + (x * scale + offset)
        const scaled = np.multiply(x, scale.ref);
        const shifted = np.add(scaled, offset.ref);
        const newCarry = np.add(carry, shifted);
        return [newCarry, newCarry.ref];
      };

      const initCarry = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0]]); // 3 iterations

      // requirePath: "fused" ensures constant handling works in fused path
      const [finalCarry, outputs] = await lax.scan(step, initCarry, xs, {
        requirePath: "fused",
      });

      // Iteration 1: 0 + (1*2 + 1) = 3
      // Iteration 2: 3 + (2*2 + 1) = 8
      // Iteration 3: 8 + (3*2 + 1) = 15
      const finalData = await finalCarry.data();
      expect(finalData[0]).toBeCloseTo(15.0);

      const outputData = await outputs.data();
      expect(Array.from(outputData)).toEqual([3, 8, 15]);

      // Clean up captured constants
      scale.dispose();
      offset.dispose();
    });

    test("with reduction (dot product accumulation)", async () => {
      // Test scan body with a reduction (sum) - correctness test only
      // NOTE: This currently falls back to JS loop because the JIT creates 2 execute
      // steps (reduction, then add) instead of fusing them. This is a known limitation.
      // When epilogue fusion is improved, this could use fused path.
      // We do NOT use requirePath here because this is testing correctness, not fusion.

      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        // sum the elements of x, then add to carry
        const sumX = np.sum(x); // This is a reduction!
        const newCarry = np.add(carry, sumX);
        return [newCarry, newCarry.ref];
      };

      const initCarry = np.array([0.0]);
      // 3 iterations, each x is a 4-element vector
      const xs = np.array([
        [1.0, 2.0, 3.0, 4.0], // sum = 10
        [5.0, 5.0, 0.0, 0.0], // sum = 10
        [1.0, 1.0, 1.0, 1.0], // sum = 4
      ]);

      const [finalCarry, outputs] = await lax.scan(step, initCarry, xs);

      // Iteration 1: 0 + 10 = 10
      // Iteration 2: 10 + 10 = 20
      // Iteration 3: 20 + 4 = 24
      const finalData = await finalCarry.data();
      expect(finalData[0]).toBeCloseTo(24.0);

      const outputData = await outputs.data();
      expect(Array.from(outputData)).toEqual([10, 20, 24]);
    });
  });

  describe("compiled-body scan", () => {
    test("matmul in body (routine)", async () => {
      // Matmul is a "routine" (not an elementwise kernel), so compiled-loop is ineligible
      // This tests compiled-body scan (JS loop or pre-encoded dispatches)
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        // carry: [2, 2], x: [2, 2] -> matmul produces [2, 2]
        const newCarry = np.matmul(carry, x);
        return [newCarry.ref, newCarry];
      };

      // 2x2 matrices
      const initCarry = np.eye(2); // identity matrix
      const xs = np.array([
        [
          [2, 0],
          [0, 2],
        ], // scale by 2
        [
          [1, 1],
          [0, 1],
        ], // shear
        [
          [0, -1],
          [1, 0],
        ], // rotate 90 degrees
      ]); // 3 iterations of 2x2 matrices

      const [finalCarry, _outputs] = await lax.scan(step, initCarry, xs);

      // I * [[2,0],[0,2]] = [[2,0],[0,2]]
      // [[2,0],[0,2]] * [[1,1],[0,1]] = [[2,2],[0,2]]
      // [[2,2],[0,2]] * [[0,-1],[1,0]] = [[2,-2],[2,0]]
      const finalData = await finalCarry.data();
      expect(finalData[0]).toBeCloseTo(2.0);
      expect(finalData[1]).toBeCloseTo(-2.0);
      expect(finalData[2]).toBeCloseTo(2.0);
      expect(finalData[3]).toBeCloseTo(0.0);
    });

    test("matmul in body with reverse", async () => {
      // Matmul routine with reverse=true
      // This verifies compiled-body scan handles reverse correctly
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.matmul(carry, x);
        return [newCarry.ref, newCarry];
      };

      const initCarry = np.eye(2); // identity matrix
      const xs = np.array([
        [
          [2, 0],
          [0, 2],
        ], // scale by 2   (index 0, processed last)
        [
          [1, 1],
          [0, 1],
        ], // shear        (index 1, processed second)
        [
          [0, -1],
          [1, 0],
        ], // rotate 90    (index 2, processed first)
      ]);

      const [finalCarry, _outputs] = await lax.scan(step, initCarry, xs, {
        reverse: true,
      });

      // Processing order: xs[2], xs[1], xs[0]
      // I * [[0,-1],[1,0]] = [[0,-1],[1,0]]
      // [[0,-1],[1,0]] * [[1,1],[0,1]] = [[0,-1],[1,1]]
      // [[0,-1],[1,1]] * [[2,0],[0,2]] = [[0,-2],[2,2]]
      const finalData = await finalCarry.data();
      expect(finalData[0]).toBeCloseTo(0.0);
      expect(finalData[1]).toBeCloseTo(-2.0);
      expect(finalData[2]).toBeCloseTo(2.0);
      expect(finalData[3]).toBeCloseTo(2.0);
    });

    test("complex composite body (Kalman-like)", async () => {
      // Tests multiple matmul + arithmetic ops in a single scan step.
      // This pattern is common in Kalman filters and state-space models.
      // Covers: matmul, add, subtract, multiply in one body (many ops per step).

      // Simplified 2D state-space model:
      // state: [2, 1], cov: [2, 2], obs: [1, 1]
      // F: state transition [2, 2]
      // H: observation matrix [1, 2]

      const F = np.array([
        [1, 0.1],
        [0, 1],
      ]); // State transition
      const H = np.array([[1, 0]]); // Observation (extracts first element)
      const Q = np.array([
        [0.01, 0],
        [0, 0.01],
      ]); // Process noise
      const R = np.array([[0.1]]); // Observation noise

      type Carry = { state: np.Array; cov: np.Array };
      type X = { obs: np.Array };
      type Y = { pred: np.Array; innovation: np.Array };

      const step = (carry: Carry, x: X): [Carry, Y] => {
        const { state, cov } = carry;
        const { obs } = x;

        // Predict step: state_pred = F @ state
        const statePred = np.matmul(F.ref, state.ref);

        // Predicted covariance: cov_pred = F @ cov @ F.T + Q
        const FCov = np.matmul(F.ref, cov.ref);
        const covPred = np.add(np.matmul(FCov, F.ref.transpose()), Q.ref);

        // Innovation: y = obs - H @ state_pred
        const innovation = np.subtract(
          obs.ref,
          np.matmul(H.ref, statePred.ref),
        );

        // Kalman gain (simplified): K = cov_pred @ H.T @ inv(S)
        // For simplicity, we use a scalar approximation: K = cov_pred @ H.T * scale
        const covH = np.matmul(covPred.ref, H.ref.transpose());
        const scale = np.array([[0.5]]); // Simplified gain scale
        const K = np.multiply(covH, scale);

        // Update: state_new = state_pred + K @ innovation
        const stateNew = np.add(
          statePred.ref,
          np.matmul(K.ref, innovation.ref),
        );

        // Covariance update (simplified): cov_new = cov_pred * 0.9
        const covNew = np.multiply(covPred.ref, np.array([[0.9]]));

        return [
          { state: stateNew, cov: covNew },
          { pred: statePred, innovation },
        ];
      };

      // Initial state and covariance
      const initState = np.array([[0], [0]]);
      const initCov = np.array([
        [1, 0],
        [0, 1],
      ]);

      // Observations: 5 time steps
      const observations = np.array([[[1]], [[2]], [[2.5]], [[3]], [[3.5]]]);

      const [finalCarry, outputs] = await lax.scan(
        step,
        { state: initState, cov: initCov },
        { obs: observations },
      );

      // Verify outputs have correct shapes
      expect(finalCarry.state.shape).toEqual([2, 1]);
      expect(finalCarry.cov.shape).toEqual([2, 2]);
      expect(outputs.pred.shape).toEqual([5, 2, 1]);
      expect(outputs.innovation.shape).toEqual([5, 1, 1]);

      // Check that state has been updated (not all zeros)
      const finalStateData = await finalCarry.state.ref.data();
      expect(Math.abs(finalStateData[0])).toBeGreaterThan(0.1);

      // Cleanup
      F.dispose();
      H.dispose();
      Q.dispose();
      R.dispose();
      finalCarry.state.dispose();
      finalCarry.cov.dispose();
      outputs.pred.dispose();
      outputs.innovation.dispose();
    });
  });

  describe("routine in scan body (native scan path)", () => {
    // Tests for scan with routine ops in body (Cholesky, Sort)
    // These routines are called via WASM imports from the native scan loop

    test.skipIf(!devicesAvailable.includes("wasm"))(
      "cholesky in body",
      async () => {
        defaultDevice("wasm");

        // Create simple 2x2 positive definite matrices
        const xs = np.array(
          [
            [
              [4, 2],
              [2, 4],
            ], // PD matrix 1
            [
              [9, 3],
              [3, 9],
            ], // PD matrix 2
            [
              [16, 4],
              [4, 16],
            ], // PD matrix 3
          ],
          { dtype: np.float32 },
        ); // 3 iterations of 2x2 matrices

        const initCarry = np.array([0], { dtype: np.float32 }); // Dummy carry

        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const L = lax.linalg.cholesky(x);
          return [carry, L];
        };

        // JIT should use native routine scan (fused path)
        const jitScan = jit((matrices: np.Array) => {
          return lax.scan(step, initCarry.ref, matrices, {
            requirePath: "fused",
          });
        });

        const [finalCarry, outputs] = jitScan(xs.ref);
        finalCarry.dispose();

        const outputData = await outputs.data();

        // Verify Cholesky factorizations
        // Matrix 1: [[4, 2], [2, 4]] -> L = [[2, 0], [1, sqrt(3)]]
        expect(outputData[0]).toBeCloseTo(2.0);
        expect(outputData[1]).toBeCloseTo(0.0);
        expect(outputData[2]).toBeCloseTo(1.0);
        expect(outputData[3]).toBeCloseTo(Math.sqrt(3));

        // Matrix 2: [[9, 3], [3, 9]] -> L = [[3, 0], [1, sqrt(8)]]
        expect(outputData[4]).toBeCloseTo(3.0);
        expect(outputData[5]).toBeCloseTo(0.0);
        expect(outputData[6]).toBeCloseTo(1.0);
        expect(outputData[7]).toBeCloseTo(Math.sqrt(8));

        // Cleanup - dispose jit first (releases captured constants), then arrays
        jitScan.dispose();
        xs.dispose();
        initCarry.dispose();
      },
    );

    test.skipIf(!devicesAvailable.includes("wasm"))(
      "cholesky with reverse",
      async () => {
        defaultDevice("wasm");

        const xs = np.array(
          [
            [
              [4, 2],
              [2, 4],
            ],
            [
              [9, 3],
              [3, 9],
            ],
          ],
          { dtype: np.float32 },
        );

        const initCarry = np.array([0], { dtype: np.float32 });

        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const L = lax.linalg.cholesky(x);
          return [carry, L];
        };

        const jitScanRev = jit((matrices: np.Array) => {
          return lax.scan(step, initCarry.ref, matrices, {
            reverse: true,
            requirePath: "fused",
          });
        });

        const [finalCarry, outputs] = jitScanRev(xs.ref);
        finalCarry.dispose();

        const outputData = await outputs.data();

        // With reverse=true, outputs are still indexed [0, 1, ...]
        // but processing order is [1, 0]
        // Output 0 comes from input 0, output 1 from input 1
        expect(outputData[0]).toBeCloseTo(2.0); // sqrt(4)
        expect(outputData[4]).toBeCloseTo(3.0); // sqrt(9)

        // Cleanup - dispose jit first (releases captured constants), then arrays
        jitScanRev.dispose();
        xs.dispose();
        initCarry.dispose();
      },
    );

    test.skipIf(!devicesAvailable.includes("wasm"))(
      "mixed kernel + routine body",
      async () => {
        // Test scan body that mixes elementwise Kernel ops with Routine (Cholesky)
        // Body: x -> scale by 2 -> cholesky
        // This tests the unified general scan path handling both Kernels and Routines
        defaultDevice("wasm");

        // 2x2 positive definite matrices (scaled by 2 inside scan should still be PD)
        const xs = np.array(
          [
            [
              [1, 0],
              [0, 1],
            ], // Identity matrix
            [
              [2, 1],
              [1, 2],
            ], // PD matrix
          ],
          { dtype: np.float32 },
        );

        const initCarry = np.zeros([1], { dtype: np.float32 });

        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          // Kernel op: scale matrix by 2
          const scaled = np.multiply(x, np.array([[2]], { dtype: np.float32 }));
          // Routine: Cholesky factorization
          const L = lax.linalg.cholesky(scaled);
          return [carry, L];
        };

        const jitScan = jit((matrices: np.Array) => {
          // Uses native scan with both Kernel (multiply) and Routine (cholesky)
          return lax.scan(step, initCarry.ref, matrices, {
            requirePath: "fused",
          });
        });

        const [finalCarry, outputs] = jitScan(xs.ref);
        finalCarry.dispose();

        const outputData = await outputs.data();

        // Matrix 1: 2*I = [[2,0],[0,2]], chol = [[sqrt(2),0],[0,sqrt(2)]]
        expect(outputData[0]).toBeCloseTo(Math.sqrt(2), 4); // L[0,0]
        expect(outputData[1]).toBeCloseTo(0, 4); // L[0,1]
        expect(outputData[2]).toBeCloseTo(0, 4); // L[1,0]
        expect(outputData[3]).toBeCloseTo(Math.sqrt(2), 4); // L[1,1]

        // Matrix 2: 2*[[2,1],[1,2]] = [[4,2],[2,4]], chol = [[2,0],[1,sqrt(3)]]
        expect(outputData[4]).toBeCloseTo(2, 4); // L[0,0]
        expect(outputData[5]).toBeCloseTo(0, 4); // L[0,1]
        expect(outputData[6]).toBeCloseTo(1, 4); // L[1,0]
        expect(outputData[7]).toBeCloseTo(Math.sqrt(3), 4); // L[1,1]

        jitScan.dispose();
        xs.dispose();
        initCarry.dispose();
      },
    );

    test.skipIf(!devicesAvailable.includes("wasm"))(
      "sort in body with passthrough carry",
      async () => {
        // Test scan with Sort routine and passthrough carry pattern
        // This tests: passthrough carry handling + Sort routine in unified path
        defaultDevice("wasm");

        const xs = np.array(
          [
            [4, 3, 1, 2],
            [8, 5, 9, 6],
            [3, 5, 5, 8],
          ],
          { dtype: np.float32 },
        ); // 3 iterations of 4-element arrays

        const initCarry = np.zeros([1], { dtype: np.float32 });

        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const sorted = np.sort(x);
          return [carry, sorted]; // Passthrough carry pattern
        };

        const jitScan = jit((matrices: np.Array) => {
          // Uses native scan with Sort routine (requires aux buffer)
          return lax.scan(step, initCarry.ref, matrices, {
            requirePath: "fused",
          });
        });

        const [finalCarry, outputs] = jitScan(xs.ref);
        finalCarry.dispose();

        const outputData = await outputs.data();

        // Each row should be sorted
        // [4, 3, 1, 2] -> [1, 2, 3, 4]
        expect(outputData[0]).toBeCloseTo(1);
        expect(outputData[1]).toBeCloseTo(2);
        expect(outputData[2]).toBeCloseTo(3);
        expect(outputData[3]).toBeCloseTo(4);

        // [8, 5, 9, 6] -> [5, 6, 8, 9]
        expect(outputData[4]).toBeCloseTo(5);
        expect(outputData[5]).toBeCloseTo(6);
        expect(outputData[6]).toBeCloseTo(8);
        expect(outputData[7]).toBeCloseTo(9);

        // [3, 5, 5, 8] -> [3, 5, 5, 8]
        expect(outputData[8]).toBeCloseTo(3);
        expect(outputData[9]).toBeCloseTo(5);
        expect(outputData[10]).toBeCloseTo(5);
        expect(outputData[11]).toBeCloseTo(8);

        jitScan.dispose();
        xs.dispose();
        initCarry.dispose();
      },
    );

    test.skipIf(!devicesAvailable.includes("wasm"))(
      "triangular_solve in body",
      async () => {
        // Test scan with TriangularSolve routine
        // Solves A @ X = B for X where A is upper-triangular
        // Using leftSide: true for left-side solve (A @ X = B)
        defaultDevice("wasm");

        // Upper-triangular matrices A (3 iterations of 2x2)
        const A = np.array(
          [
            [
              [2, 1],
              [0, 3],
            ], // Upper-triangular
            [
              [1, 2],
              [0, 4],
            ],
            [
              [3, 1],
              [0, 2],
            ],
          ],
          { dtype: np.float32 },
        );

        // B vectors [3, 2, 1] - note: leftSide: true requires [n, 1] shaped B
        const B = np.array(
          [
            [[5], [6]], // B[0]: solve A[0] @ X = B[0]
            [[3], [8]], // B[1]
            [[7], [4]], // B[2]
          ],
          { dtype: np.float32 },
        );

        const initCarry = np.zeros([1], { dtype: np.float32 });

        const step = (carry: np.Array, x: np.Array[]): [np.Array, np.Array] => {
          const [a, b] = x;
          // triangular_solve with leftSide: true solves A @ X = B
          const X = lax.linalg.triangularSolve(a, b, {
            lower: false,
            leftSide: true,
          });
          return [carry, X];
        };

        const jitScan = jit((inputs: np.Array[]) => {
          return lax.scan(step, initCarry.ref, inputs, {
            requirePath: "fused",
          });
        });

        const [finalCarry, outputs] = jitScan([A.ref, B.ref]);
        finalCarry.dispose();

        const outputData = await outputs.data();

        // Verify solutions by checking A @ X = B
        // For A[0] = [[2,1],[0,3]], B[0] = [[5],[6]]
        // Back-substitution: X[1] = 6/3 = 2, X[0] = (5 - 1*2)/2 = 1.5
        expect(outputData[0]).toBeCloseTo(1.5, 4); // X[0,0]
        expect(outputData[1]).toBeCloseTo(2.0, 4); // X[0,1]

        jitScan.dispose();
        A.dispose();
        B.dispose();
        initCarry.dispose();
      },
    );

    test.skipIf(!devicesAvailable.includes("wasm"))("LU in body", async () => {
      // Test scan with LU decomposition routine
      // LU returns: (lu, pivots, permutation)
      defaultDevice("wasm");

      // Square matrices for LU decomposition
      const xs = np.array(
        [
          [
            [4, 3],
            [6, 3],
          ],
          [
            [1, 2],
            [3, 4],
          ],
        ],
        { dtype: np.float32 },
      );

      const initCarry = np.zeros([1], { dtype: np.float32 });

      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array[]] => {
        const [lu, pivots, perm] = lax.linalg.lu(x);
        // Return LU matrix as output (piv and perm would need separate outputs)
        return [carry, [lu, pivots, perm]];
      };

      const jitScan = jit((matrices: np.Array) => {
        return lax.scan(step, initCarry.ref, matrices, {
          requirePath: "fused",
        });
      });

      const [finalCarry, outputs] = jitScan(xs.ref);
      finalCarry.dispose();

      // outputs is a pytree with [lu, pivots, perm] for each iteration
      const [luOut, _pivotsOut, _permOut] = outputs as unknown as np.Array[];
      const luData = await luOut.data();

      // Check that LU decomposition is valid
      // For matrix [[4,3],[6,3]], expect pivoting to swap rows
      // After pivoting: [[6,3],[4,3]] -> L=[[1,0],[2/3,1]], U=[[6,3],[0,1]]
      expect(luData[0]).toBeCloseTo(6, 4); // U[0,0]
      expect(luData[1]).toBeCloseTo(3, 4); // U[0,1]
      // L[1,0] stored in lu[1,0], U[1,1] stored in lu[1,1]

      jitScan.dispose();
      xs.dispose();
      initCarry.dispose();
    });

    test.skipIf(!devicesAvailable.includes("wasm"))(
      "argsort in body",
      async () => {
        // Test scan with Argsort routine
        // np.argsort only returns indices, so this test verifies argsort works in scan
        defaultDevice("wasm");

        const xs = np.array(
          [
            [4, 1, 3, 2],
            [9, 5, 7, 6],
          ],
          { dtype: np.float32 },
        );

        // Init carry should be int32 (indices type) with zeros
        const initCarry = np.zeros([4]);

        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          // np.argsort returns only indices
          const indices = np.argsort(x);
          return [indices, indices.ref];
        };

        const jitScan = jit((arrays: np.Array) => {
          return lax.scan(step, initCarry.ref, arrays, {
            requirePath: "fused",
          });
        });

        const [finalCarry, outputs] = jitScan(xs.ref);
        finalCarry.dispose();

        const indicesData = await outputs.data();

        // [4, 1, 3, 2] -> indices [1, 3, 2, 0]
        expect(indicesData[0]).toBe(1);
        expect(indicesData[1]).toBe(3);
        expect(indicesData[2]).toBe(2);
        expect(indicesData[3]).toBe(0);

        // [9, 5, 7, 6] -> indices [1, 3, 2, 0]
        expect(indicesData[4]).toBe(1);
        expect(indicesData[5]).toBe(3);
        expect(indicesData[6]).toBe(2);
        expect(indicesData[7]).toBe(0);

        jitScan.dispose();
        xs.dispose();
        initCarry.dispose();
      },
    );
  });

  describe("reverse scan", () => {
    test("basic", async () => {
      // Reverse scan processes xs in reverse order: xs[2], xs[1], xs[0]
      // but outputs are still aligned to xs indices
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry.ref];
      };

      const initCarry = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0]]);

      const [finalCarry, outputs] = await lax.scan(step, initCarry, xs, {
        reverse: true,
      });

      // Processing order: xs[2]=3, xs[1]=2, xs[0]=1
      // Carry states: 0 → 3 → 5 → 6
      // Final carry = 6 (same as forward, since addition is commutative)
      const finalData = await finalCarry.data();
      expect(finalData[0]).toBeCloseTo(6.0);

      // Outputs are aligned to xs indices, but computed in reverse
      // outputs[2] = 3 (first iteration, carry was 0)
      // outputs[1] = 5 (second iteration, carry was 3)
      // outputs[0] = 6 (third iteration, carry was 5)
      const outputData = await outputs.data();
      expect(Array.from(outputData)).toEqual([6, 5, 3]);
    });
  });
});

describe("scan autodiff", () => {
  beforeAll(async () => {
    const devices = await init();
    if (devices.includes("cpu")) {
      defaultDevice("cpu");
    }
  });

  describe("JVP (forward-mode)", () => {
    it("computes jvp of cumulative sum", async () => {
      // f(xs) = cumsum(xs) via scan
      // For xs = [1, 2, 3], cumsum = [1, 3, 6], final_carry = 6
      // d/dxs[0] cumsum = [1, 1, 1]  (every subsequent sum includes xs[0])
      // d/dxs[1] cumsum = [0, 1, 1]
      // d/dxs[2] cumsum = [0, 0, 1]
      // So tangent wrt all-ones: [1, 1, 1] + [0, 1, 1] + [0, 0, 1] = [1, 2, 3]
      // And for final carry: 1+1+1 = 3

      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, _outputs] = lax.scan(step, init, xs);
        // Return just the final carry as a scalar to simplify
        return finalCarry;
      };

      const xs = np.array([[1.0], [2.0], [3.0]]);
      const xs_dot = np.ones([3, 1]); // tangent is all 1s

      const [primal, tangent] = jvp(cumsumScan, [xs], [xs_dot]);

      // Primal: cumsum final = 1 + 2 + 3 = 6
      expect(await primal.data()).toEqual(new Float32Array([6]));

      // Tangent: d(final)/d(xs) with all-ones = 1+1+1 = 3
      expect(await tangent.data()).toEqual(new Float32Array([3]));
    });

    it("computes jvp of cumulative product", async () => {
      // f(xs) = cumprod(xs) via scan
      // For xs = [2, 3, 4], cumprod = [2, 6, 24], final_carry = 24
      // d/dx[0] final = 3*4 = 12
      // d/dx[1] final = 2*4 = 8
      // d/dx[2] final = 2*3 = 6
      // With tangent = [1, 1, 1]: 12 + 8 + 6 = 26

      const cumprodScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.multiply(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.ones([1]);
        const [finalCarry, _outputs] = lax.scan(step, init, xs);
        return finalCarry;
      };

      const xs = np.array([[2.0], [3.0], [4.0]]);
      const xs_dot = np.ones([3, 1]);

      const [primal, tangent] = jvp(cumprodScan, [xs], [xs_dot]);

      // Primal: cumprod final = 2 * 3 * 4 = 24
      expect(await primal.data()).toEqual(new Float32Array([24]));

      // Tangent: 12 + 8 + 6 = 26
      expect(await tangent.data()).toEqual(new Float32Array([26]));
    });

    it("jvp respects different tangent values", async () => {
      // Same cumsum but with tangent = [1, 0, 0] - only perturb first element
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, init, xs);
        return finalCarry;
      };

      const xs = np.array([[1.0], [2.0], [3.0]]);

      // Perturb only first element
      const xs_dot1 = np.array([[1.0], [0.0], [0.0]]);
      const [p1, t1] = jvp(cumsumScan, [xs.ref], [xs_dot1]);
      expect(await t1.data()).toEqual(new Float32Array([1])); // d(final)/d(xs[0]) = 1
      p1.dispose();

      // Perturb only second element
      const xs_dot2 = np.array([[0.0], [1.0], [0.0]]);
      const [p2, t2] = jvp(cumsumScan, [xs.ref], [xs_dot2]);
      expect(await t2.data()).toEqual(new Float32Array([1])); // d(final)/d(xs[1]) = 1
      p2.dispose();

      // Perturb only third element
      const xs_dot3 = np.array([[0.0], [0.0], [1.0]]);
      const [p3, t3] = jvp(cumsumScan, [xs], [xs_dot3]);
      expect(await t3.data()).toEqual(new Float32Array([1])); // d(final)/d(xs[2]) = 1
      p3.dispose();
    });

    it("jvp works with reverse scan", async () => {
      // Reverse cumsum: processes xs[2], xs[1], xs[0] but carry logic is same
      const revCumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, init, xs, { reverse: true });
        return finalCarry;
      };

      const xs = np.array([[1.0], [2.0], [3.0]]);
      const xs_dot = np.ones([3, 1]); // Perturb all elements

      // Final carry = 1+2+3 = 6 (same as forward)
      // Each input contributes 1 to final, so tangent = 3
      const [primal, tangent] = jvp(revCumsumScan, [xs], [xs_dot]);

      expect(await primal.data()).toEqual(new Float32Array([6]));
      expect(await tangent.data()).toEqual(new Float32Array([3]));
    });

    it("jvp respects perturbation order in reverse scan", async () => {
      const revCumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, init, xs, { reverse: true });
        return finalCarry;
      };

      const xs = np.array([[1.0], [2.0], [3.0]]);

      // Perturb only first element
      const xs_dot1 = np.array([[1.0], [0.0], [0.0]]);
      const [p1, t1] = jvp(revCumsumScan, [xs.ref], [xs_dot1]);
      // In reverse scan, xs[0] is processed last, but it still contributes 1 to final carry
      expect(await t1.data()).toEqual(new Float32Array([1]));
      p1.dispose();

      // Perturb only last element
      const xs_dot3 = np.array([[0.0], [0.0], [1.0]]);
      const [p3, t3] = jvp(revCumsumScan, [xs], [xs_dot3]);
      // In reverse scan, xs[2] is processed first, but it still contributes 1 to final carry
      expect(await t3.data()).toEqual(new Float32Array([1]));
      p3.dispose();
    });
  });

  describe("VJP (reverse-mode)", () => {
    it("computes gradient through scan (sum of final carry)", async () => {
      // VJP/grad through scan now works!
      // This test verifies gradients flow correctly.
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, init, xs);
        return finalCarry.sum(); // Return scalar for grad
      };

      const xs = np.array([[1.0], [2.0], [3.0]]);

      // Gradient of sum(finalCarry) w.r.t. xs
      // finalCarry = xs[0] + xs[1] + xs[2], so gradient = [1, 1, 1]
      const dxs = grad(cumsumScan)(xs);
      expect(dxs.shape).toEqual([3, 1]);
      expect(await dxs.data()).toEqual(new Float32Array([1, 1, 1]));
    });

    it("computes gradient through scan (sum of all cumsum values)", async () => {
      // Loss = sum of all cumsum values
      // cumsum = [xs[0], xs[0]+xs[1], xs[0]+xs[1]+xs[2]] = [1, 3, 6]
      // loss = 1 + 3 + 6 = 10
      // gradient = [3, 2, 1] (xs[0] contributes 3 times, xs[1] twice, xs[2] once)
      const sumOfCumsum = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [_, ys] = lax.scan(step, init, xs);
        return ys.sum();
      };

      const xs = np.array([[1.0], [2.0], [3.0]]);
      const dxs = grad(sumOfCumsum)(xs);
      expect(dxs.shape).toEqual([3, 1]);
      expect(await dxs.data()).toEqual(new Float32Array([3, 2, 1]));
    });

    it("computes gradient through reverse scan", async () => {
      // Reverse cumsum: processes xs[2], xs[1], xs[0]
      // Loss = sum(finalCarry) = sum over all xs
      // Since sum is commutative, gradient should still be [1, 1, 1]
      const reverseCumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, init, xs, { reverse: true });
        return finalCarry.sum();
      };

      const xs = np.array([[1.0], [2.0], [3.0]]);
      const dxs = grad(reverseCumsumScan)(xs);
      expect(dxs.shape).toEqual([3, 1]);
      expect(await dxs.data()).toEqual(new Float32Array([1, 1, 1]));
    });
  });

  describe("makeJaxpr of scan with JVP", () => {
    it("traces scan jvp correctly", async () => {
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, init, xs);
        return finalCarry;
      };

      // We can at least verify that jvp produces consistent results
      const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);
      const xs_dot = np.ones([5, 1]);

      const [primal, tangent] = jvp(cumsumScan, [xs], [xs_dot]);

      // Primal: 1+2+3+4+5 = 15
      expect(await primal.data()).toEqual(new Float32Array([15]));

      // Tangent: 5 (each input contributes 1 to the final sum)
      expect(await tangent.data()).toEqual(new Float32Array([5]));
    });
  });

  describe("vmap", () => {
    it("vmaps cumulative sum over batch dimension", async () => {
      // vmap a cumsum scan over a batch of input sequences
      // Each batch element runs an independent scan
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, outputs] = lax.scan(step, init, xs);
        outputs.dispose(); // Only return final carry for simplicity
        return finalCarry;
      };

      // Batch of 3 sequences, each with 4 timesteps, 1 feature
      // Sequence 1: [1, 2, 3, 4] -> cumsum = 10
      // Sequence 2: [2, 4, 6, 8] -> cumsum = 20
      // Sequence 3: [1, 1, 1, 1] -> cumsum = 4
      const xs = np.array([
        [[1.0], [2.0], [3.0], [4.0]], // batch 0
        [[2.0], [4.0], [6.0], [8.0]], // batch 1
        [[1.0], [1.0], [1.0], [1.0]], // batch 2
      ]); // shape: [3, 4, 1]

      const batchedCumsum = vmap(cumsumScan);
      const result = batchedCumsum(xs);

      expect(result.shape).toEqual([3, 1]);
      const data = await result.data();
      expect(data[0]).toBeCloseTo(10.0);
      expect(data[1]).toBeCloseTo(20.0);
      expect(data[2]).toBeCloseTo(4.0);
    });

    it("vmaps scan returning both carry and outputs", async () => {
      // vmap scan that returns both final carry and all outputs
      const cumsumWithOutputs = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, outputs] = lax.scan(step, init, xs);
        return { carry: finalCarry, outputs };
      };

      const xs = np.array([
        [[1.0], [2.0], [3.0]], // batch 0: cumsum = [1, 3, 6]
        [[1.0], [1.0], [1.0]], // batch 1: cumsum = [1, 2, 3]
      ]); // shape: [2, 3, 1]

      const batchedScan = vmap(cumsumWithOutputs);
      const result = batchedScan(xs) as { carry: np.Array; outputs: np.Array };

      expect(result.carry.shape).toEqual([2, 1]);
      expect(result.outputs.shape).toEqual([2, 3, 1]);

      const carryData = await result.carry.data();
      expect(carryData[0]).toBeCloseTo(6.0); // 1+2+3
      expect(carryData[1]).toBeCloseTo(3.0); // 1+1+1

      const outputData = await result.outputs.data();
      // batch 0: [1, 3, 6]
      expect(outputData[0]).toBeCloseTo(1.0);
      expect(outputData[1]).toBeCloseTo(3.0);
      expect(outputData[2]).toBeCloseTo(6.0);
      // batch 1: [1, 2, 3]
      expect(outputData[3]).toBeCloseTo(1.0);
      expect(outputData[4]).toBeCloseTo(2.0);
      expect(outputData[5]).toBeCloseTo(3.0);
    });

    it("vmaps scan with multiply body", async () => {
      // Cumulative product scan
      const cumprodScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.multiply(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.ones([1]);
        const [finalCarry, _] = lax.scan(step, init, xs);
        return finalCarry;
      };

      // Batch of 2 sequences
      // Sequence 1: [2, 3, 4] -> cumprod = 24
      // Sequence 2: [1, 2, 3] -> cumprod = 6
      const xs = np.array([
        [[2.0], [3.0], [4.0]],
        [[1.0], [2.0], [3.0]],
      ]);

      const batchedCumprod = vmap(cumprodScan);
      const result = batchedCumprod(xs);

      expect(result.shape).toEqual([2, 1]);
      const data = await result.data();
      expect(data[0]).toBeCloseTo(24.0);
      expect(data[1]).toBeCloseTo(6.0);
    });

    it("jit(vmap(scan)) works correctly", async () => {
      // JIT compile a vmapped scan
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, init, xs);
        return finalCarry;
      };

      const xs = np.array([
        [[1.0], [2.0], [3.0], [4.0]], // batch 0: sum = 10
        [[2.0], [4.0], [6.0], [8.0]], // batch 1: sum = 20
        [[1.0], [1.0], [1.0], [1.0]], // batch 2: sum = 4
      ]);

      const jittedBatchedCumsum = jit(vmap(cumsumScan));
      const result = jittedBatchedCumsum(xs);

      expect(result.shape).toEqual([3, 1]);
      const data = await result.data();
      expect(data[0]).toBeCloseTo(10.0);
      expect(data[1]).toBeCloseTo(20.0);
      expect(data[2]).toBeCloseTo(4.0);

      jittedBatchedCumsum.dispose();
    });

    it("jit(vmap(scan)) with outputs works correctly", async () => {
      // JIT compile vmapped scan returning both carry and outputs
      const cumsumWithOutputs = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        return lax.scan(step, init, xs);
      };

      const xs = np.array([
        [[1.0], [2.0], [3.0]], // batch 0: cumsum = [1, 3, 6]
        [[1.0], [1.0], [1.0]], // batch 1: cumsum = [1, 2, 3]
      ]);

      const jittedBatchedScan = jit(vmap(cumsumWithOutputs));
      const [carry, outputs] = jittedBatchedScan(xs) as [np.Array, np.Array];

      expect(carry.shape).toEqual([2, 1]);
      expect(outputs.shape).toEqual([2, 3, 1]);

      const carryData = await carry.data();
      expect(carryData[0]).toBeCloseTo(6.0);
      expect(carryData[1]).toBeCloseTo(3.0);

      const outputData = await outputs.data();
      expect(outputData[0]).toBeCloseTo(1.0);
      expect(outputData[1]).toBeCloseTo(3.0);
      expect(outputData[2]).toBeCloseTo(6.0);
      expect(outputData[3]).toBeCloseTo(1.0);
      expect(outputData[4]).toBeCloseTo(2.0);
      expect(outputData[5]).toBeCloseTo(3.0);

      jittedBatchedScan.dispose();
    });

    it("vmap(jit(scan)) works correctly", async () => {
      // vmap over a JIT-compiled scan
      const cumsumScan = jit((xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, init, xs);
        return finalCarry;
      });

      const xs = np.array([
        [[1.0], [2.0], [3.0]], // batch 0: sum = 6
        [[2.0], [2.0], [2.0]], // batch 1: sum = 6
      ]);

      const batchedJittedCumsum = vmap(cumsumScan);
      const result = batchedJittedCumsum(xs);

      expect(result.shape).toEqual([2, 1]);
      const data = await result.data();
      expect(data[0]).toBeCloseTo(6.0);
      expect(data[1]).toBeCloseTo(6.0);

      cumsumScan.dispose();
    });
  });

  describe("scan over views (sliced/transposed xs)", () => {
    it("scan over sliced xs", async () => {
      // Create a larger array and slice it to use as xs
      const full = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]);
      // Slice to get [2.0], [3.0], [4.0] (indices 2:5)
      const xs = full.slice([2, 5]);

      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry.ref];
      };
      const init = np.array([0.0]);

      const [finalCarry, outputs] = await lax.scan(step, init, xs);

      const finalData = await finalCarry.data();
      expect(finalData[0]).toBeCloseTo(9.0); // 2 + 3 + 4

      const outputData = await outputs.data();
      expect(Array.from(outputData)).toEqual([2, 5, 9]); // cumsum of [2,3,4]
    });

    it("scan over transposed xs", async () => {
      // Create [3, 2] array and transpose to [2, 3] where axis 0 is scan axis
      // Original: [[1,2], [3,4], [5,6]] shape [3, 2]
      // Transposed: [[1,3,5], [2,4,6]] shape [2, 3] - scan over 2 iterations
      const original = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
      ]);
      const xs = np.transpose(original); // shape [2, 3]

      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        // carry shape [3], x shape [3]
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry.ref];
      };
      const init = np.zeros([3]);

      const [finalCarry, outputs] = await lax.scan(step, init, xs);

      // iter 0: [0,0,0] + [1,3,5] = [1,3,5]
      // iter 1: [1,3,5] + [2,4,6] = [3,7,11]
      const finalData = await finalCarry.data();
      expect(Array.from(finalData).map((x) => Math.round(x))).toEqual([
        3, 7, 11,
      ]);

      const outputData = await outputs.data();
      // outputs shape [2, 3]: [[1,3,5], [3,7,11]]
      expect(Array.from(outputData).map((x) => Math.round(x))).toEqual([
        1, 3, 5, 3, 7, 11,
      ]);
    });

    it("jit(scan) over sliced xs", async () => {
      const full = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]]);
      const xs = full.ref.slice([1, 4]); // [1.0], [2.0], [3.0]

      const f = jit(() => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.array([0.0]);
        return lax.scan(step, init, xs);
      });

      const [finalCarry, outputs] = f();

      const finalData = await finalCarry.data();
      expect(finalData[0]).toBeCloseTo(6.0); // 1 + 2 + 3

      const outputData = await outputs.data();
      expect(Array.from(outputData)).toEqual([1, 3, 6]);

      full.dispose();
      f.dispose();
    });

    it("scan over reshaped xs", async () => {
      // Create [2, 3, 2] and reshape to [2, 6] - tests if reshape view works with scan
      const original = np.array([
        [
          [1.0, 2.0],
          [3.0, 4.0],
          [5.0, 6.0],
        ],
        [
          [7.0, 8.0],
          [9.0, 10.0],
          [11.0, 12.0],
        ],
      ]); // shape [2, 3, 2]

      const xs = np.reshape(original, [2, 6]); // shape [2, 6]

      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        // x shape [6], sum it
        const xSum = np.sum(x);
        const newCarry = np.add(carry, xSum);
        return [newCarry, newCarry.ref];
      };
      const init = np.array([0.0]);

      const [finalCarry, outputs] = await lax.scan(step, init, xs);

      // iter 0: sum([1,2,3,4,5,6]) = 21
      // iter 1: 21 + sum([7,8,9,10,11,12]) = 21 + 57 = 78
      const finalData = await finalCarry.data();
      expect(finalData[0]).toBeCloseTo(78.0);

      const outputData = await outputs.data();
      expect(outputData[0]).toBeCloseTo(21.0);
      expect(outputData[1]).toBeCloseTo(78.0);
    });
  });

  describe("scan with routines", () => {
    it("scan with Cholesky in body", async () => {
      // Cumulative Cholesky: repeatedly apply Cholesky to carry (idempotent for identity-like matrices)
      // This tests that Routines work within scan's JS loop
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        // x is a scalar multiplier to modify the carry
        // Create a positive definite matrix from carry
        const scaled = np.multiply(carry.ref, x);
        const L = lax.linalg.cholesky(scaled);
        // Reconstruct: L @ L^T to keep positive definite
        const reconstructed = np.matmul(L.ref, L.transpose());
        return [reconstructed, L];
      };

      // Start with a simple positive definite 2x2 matrix
      const init = np.array([
        [4.0, 2.0],
        [2.0, 5.0],
      ]);
      // Scalars that keep it positive definite
      const xs = np.array([[1.0], [1.0], [1.0]]);

      const [finalCarry, outputs] = await lax.scan(step, init, xs);

      // Verify outputs are valid Cholesky decompositions
      const outputData = await outputs.data();
      expect(outputData.length).toBe(3 * 4); // 3 iterations, 2x2 matrices

      const finalData = await finalCarry.data();
      expect(finalData.length).toBe(4); // 2x2 matrix
      // Final carry should be positive definite
      expect(finalData[0]).toBeGreaterThan(0);
      expect(finalData[3]).toBeGreaterThan(0);
    });

    it("jit + scan with Cholesky", async () => {
      const f = jit(() => {
        const step = (carry: np.Array, _x: np.Array): [np.Array, np.Array] => {
          const L = lax.linalg.cholesky(carry);
          // Reconstruct to keep positive definite
          const reconstructed = np.matmul(L.ref, L.transpose());
          return [reconstructed, L];
        };

        const init = np.array([
          [4.0, 2.0],
          [2.0, 5.0],
        ]);
        // Dummy xs just to drive iterations
        const xs = np.array([[1.0], [1.0]]);

        return lax.scan(step, init, xs);
      });

      const [finalCarry, outputs] = f();

      const finalData = await finalCarry.data();
      expect(finalData.length).toBe(4);
      expect(finalData[0]).toBeGreaterThan(0);

      const outputData = await outputs.data();
      expect(outputData.length).toBe(2 * 4); // 2 iterations, 2x2 matrices

      f.dispose();
    });

    it("grad through jit(scan) with mixed kernel+routine body", async () => {
      // Test gradient flow through a scan with both Kernel (multiply) and Routine (cholesky)
      // This verifies autodiff works with the unified general scan path
      defaultDevice("wasm");

      // Loss = sum of Cholesky outputs from scaled positive definite matrices
      // For a PD matrix A, cholesky(s*A) = sqrt(s) * cholesky(A) for scalar s > 0
      // So d(sum(chol(s*A)))/ds should be (1/2sqrt(s)) * sum(chol(A))
      const loss = (scale: np.Array) => {
        const xs = np.array(
          [
            [
              [4, 0],
              [0, 4],
            ], // 4*I -> chol = [[2,0],[0,2]]
          ],
          { dtype: np.float32 },
        );

        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const scaled = np.multiply(x, scale.ref);
          const L = lax.linalg.cholesky(scaled);
          return [carry, L];
        };

        const initCarry = np.zeros([1], { dtype: np.float32 });
        const [finalCarry, outputs] = lax.scan(step, initCarry, xs);
        finalCarry.dispose();
        return np.sum(outputs);
      };

      // scale = 1.0: chol(4*I) = [[2,0],[0,2]], sum = 4
      // d/ds at s=1: chol(s*4*I) = sqrt(s)*[[2,0],[0,2]]
      // sum = 4*sqrt(s), d/ds = 4 * 0.5 * s^(-0.5) = 2/sqrt(s) = 2 at s=1
      const scale = np.array([1.0], { dtype: np.float32 });

      // First verify forward pass
      const fwdResult = loss(scale.ref);
      const fwdData = await fwdResult.data();
      expect(fwdData[0]).toBeCloseTo(4.0, 4); // 2+0+0+2 = 4

      // Now test gradient
      const gradFn = grad(loss);
      const dScale = gradFn(scale);

      const gradData = await dScale.data();
      // d(sum)/ds = 2 at s=1 (from chain rule through sqrt(s))
      expect(gradData[0]).toBeCloseTo(2.0, 3);
    });
  });

  describe("transform sandwiches", () => {
    // Tests for compositions of transformations with scan.
    // These verify that jit, grad, and vmap compose correctly in various orders.

    it("jit(grad(scan)) computes gradient through JIT-compiled grad", async () => {
      // Compile the gradient computation itself
      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, init, xs);
        return finalCarry.sum();
      };

      // JIT compile the gradient function
      const jitGrad = jit(grad(cumsumScan));

      const xs = np.array([[1.0], [2.0], [3.0]]);

      // First call (traces and compiles)
      const dxs1 = jitGrad(xs.ref);
      expect(dxs1.shape).toEqual([3, 1]);
      expect(await dxs1.data()).toEqual(new Float32Array([1, 1, 1]));

      // Second call (uses cached compilation)
      const xs2 = np.array([[4.0], [5.0], [6.0]]);
      const dxs2 = jitGrad(xs2);
      expect(await dxs2.data()).toEqual(new Float32Array([1, 1, 1]));

      jitGrad.dispose();
    });

    it("grad with jit inside scan body computes gradient correctly", async () => {
      // Test gradient through a function where jit is used inside
      // (grad(jit(f)) directly doesn't work - this tests the supported pattern)
      const cumsumWithJitBody = (xs: np.Array) => {
        // JIT-compile just the body computation
        const jitAdd = jit((a: np.Array, b: np.Array) => np.add(a, b));

        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = jitAdd(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, init, xs);
        const result = finalCarry.sum();
        jitAdd.dispose();
        return result;
      };

      const xs = np.array([[1.0], [2.0], [3.0]]);

      // Take gradient - jit inside works because grad traces through it
      const dxs = grad(cumsumWithJitBody)(xs);
      expect(dxs.shape).toEqual([3, 1]);
      expect(await dxs.data()).toEqual(new Float32Array([1, 1, 1]));
    });

    it("vmap(scan) vs scan(vmap(body)) equivalence", async () => {
      // vmap(scan) applies scan independently to each batch element
      // scan(vmap(body)) applies a batched body at each timestep
      // For element-wise operations these should produce equivalent results

      // Approach 1: vmap(scan) - independent scans over batch
      const singleScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, ys] = lax.scan(step, init, xs);
        return [finalCarry, ys] as [np.Array, np.Array];
      };

      const batchedXs = np.array([
        [[1.0], [2.0], [3.0]], // batch 0: cumsum = [1, 3, 6], final = 6
        [[2.0], [2.0], [2.0]], // batch 1: cumsum = [2, 4, 6], final = 6
      ]);

      // vmap over first axis
      const vmappedScan = vmap(singleScan);
      const [vmapCarries, vmapYs] = vmappedScan(batchedXs.ref);

      expect(vmapCarries.shape).toEqual([2, 1]);
      expect(vmapYs.shape).toEqual([2, 3, 1]);
      const vmapCarryData = await vmapCarries.data();
      expect(vmapCarryData[0]).toBeCloseTo(6.0);
      expect(vmapCarryData[1]).toBeCloseTo(6.0);
      const vmapYsData = await vmapYs.data();
      // batch 0: [1, 3, 6], batch 1: [2, 4, 6]
      expect(vmapYsData[0]).toBeCloseTo(1.0);
      expect(vmapYsData[1]).toBeCloseTo(3.0);
      expect(vmapYsData[2]).toBeCloseTo(6.0);
      expect(vmapYsData[3]).toBeCloseTo(2.0);
      expect(vmapYsData[4]).toBeCloseTo(4.0);
      expect(vmapYsData[5]).toBeCloseTo(6.0);

      // Approach 2: scan(vmap(body)) - batched body at each timestep
      // Transpose xs to be [time, batch, features]
      const transposedXs = batchedXs.transpose([1, 0, 2]); // [3, 2, 1]

      const vmapStep = vmap(
        (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        },
      );

      const batchInit = np.zeros([2, 1]); // batch of init carries
      const batchStep = (
        carry: np.Array,
        x: np.Array,
      ): [np.Array, np.Array] => {
        return vmapStep(carry, x);
      };

      const [scanVmapCarry, scanVmapYs] = lax.scan(
        batchStep,
        batchInit,
        transposedXs,
      );

      expect(scanVmapCarry.shape).toEqual([2, 1]);
      const scanVmapCarryData = await scanVmapCarry.data();
      expect(scanVmapCarryData[0]).toBeCloseTo(6.0);
      expect(scanVmapCarryData[1]).toBeCloseTo(6.0);

      // Transpose outputs back for comparison: [3, 2, 1] -> [2, 3, 1]
      const scanVmapYsTransposed = scanVmapYs.ref.transpose([1, 0, 2]);
      expect(scanVmapYsTransposed.shape).toEqual([2, 3, 1]);
      const scanVmapYsData = await scanVmapYsTransposed.data();
      expect(scanVmapYsData[0]).toBeCloseTo(1.0);
      expect(scanVmapYsData[1]).toBeCloseTo(3.0);
      expect(scanVmapYsData[2]).toBeCloseTo(6.0);
      expect(scanVmapYsData[3]).toBeCloseTo(2.0);
      expect(scanVmapYsData[4]).toBeCloseTo(4.0);
      expect(scanVmapYsData[5]).toBeCloseTo(6.0);
    });

    it("jit(vmap(scan)) works on WASM backend", async () => {
      // Explicitly test jit(vmap(scan)) on WASM (vs the existing test that may use CPU)
      defaultDevice("wasm");

      const cumsumScan = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, ys] = lax.scan(step, init, xs);
        return [finalCarry, ys] as [np.Array, np.Array];
      };

      const jitVmapScan = jit(vmap(cumsumScan));

      const batchedXs = np.array([
        [[1.0], [2.0], [3.0]], // batch 0
        [[2.0], [2.0], [2.0]], // batch 1
        [[1.0], [1.0], [1.0]], // batch 2
      ]);

      const [carries, ys] = jitVmapScan(batchedXs);

      expect(carries.shape).toEqual([3, 1]);
      expect(ys.shape).toEqual([3, 3, 1]);

      const carryData = await carries.data();
      expect(carryData[0]).toBeCloseTo(6.0);
      expect(carryData[1]).toBeCloseTo(6.0);
      expect(carryData[2]).toBeCloseTo(3.0);

      jitVmapScan.dispose();
    });

    it.skipIf(!devicesAvailable.includes("webgpu"))(
      "jit(vmap(scan)) works on WebGPU backend",
      async () => {
        // Explicitly test jit(vmap(scan)) on WebGPU
        defaultDevice("webgpu");

        const cumsumScan = (xs: np.Array) => {
          const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
            const newCarry = np.add(carry.ref, x);
            return [newCarry, newCarry.ref];
          };
          const init = np.zeros([1]);
          const [finalCarry, ys] = lax.scan(step, init, xs);
          return [finalCarry, ys] as [np.Array, np.Array];
        };

        const jitVmapScan = jit(vmap(cumsumScan));

        const batchedXs = np.array([
          [[1.0], [2.0], [3.0]], // batch 0
          [[2.0], [2.0], [2.0]], // batch 1
          [[1.0], [1.0], [1.0]], // batch 2
        ]);

        const [carries, ys] = jitVmapScan(batchedXs);

        expect(carries.shape).toEqual([3, 1]);
        expect(ys.shape).toEqual([3, 3, 1]);

        const carryData = await carries.data();
        expect(carryData[0]).toBeCloseTo(6.0);
        expect(carryData[1]).toBeCloseTo(6.0);
        expect(carryData[2]).toBeCloseTo(3.0);

        jitVmapScan.dispose();
      },
    );

    it("grad(vmap(scan)) computes batched gradients", async () => {
      // Gradient of a batched scan (sum of all final carries)
      const batchedCumsum = vmap((xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, init, xs);
        return finalCarry;
      });

      // Wrapper that returns a scalar
      const loss = (batchedXs: np.Array) => {
        const carries = batchedCumsum(batchedXs);
        return np.sum(carries);
      };

      const batchedXs = np.array([
        [[1.0], [2.0], [3.0]], // batch 0: final = 6
        [[2.0], [2.0], [2.0]], // batch 1: final = 6
      ]);

      // Each xs element contributes 1 to each final carry (since it's a sum)
      // Gradient should be all ones
      const dxs = grad(loss)(batchedXs);
      expect(dxs.shape).toEqual([2, 3, 1]);
      const gradData = await dxs.data();
      for (let i = 0; i < 6; i++) {
        expect(gradData[i]).toBeCloseTo(1.0);
      }
    });

    it("vmap(grad(scan)) computes gradient per batch element", async () => {
      // vmap of gradient (different from grad of vmap)
      const cumsumWithSum = (xs: np.Array) => {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const newCarry = np.add(carry.ref, x);
          return [newCarry, newCarry.ref];
        };
        const init = np.zeros([1]);
        const [finalCarry, _] = lax.scan(step, init, xs);
        return finalCarry.sum();
      };

      const gradOfCumsum = grad(cumsumWithSum);
      const vmapGrad = vmap(gradOfCumsum);

      const batchedXs = np.array([
        [[1.0], [2.0], [3.0]], // batch 0
        [[2.0], [2.0], [2.0]], // batch 1
      ]);

      const dxs = vmapGrad(batchedXs);
      expect(dxs.shape).toEqual([2, 3, 1]);
      const gradData = await dxs.data();
      // Each batch gets gradient [1, 1, 1]
      for (let i = 0; i < 6; i++) {
        expect(gradData[i]).toBeCloseTo(1.0);
      }
    });
  });

  describe("requirePath option", () => {
    it("throws when requirePath is not satisfied", async () => {
      // Use CPU backend where scans always use fallback path
      defaultDevice("cpu");

      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry.ref];
      };

      const init = np.array([0.0]);
      const xs = np.array([[1.0], [2.0]]);

      // Requiring fused on CPU should throw (CPU always uses fallback)
      const f = jit(() => lax.scan(step, init, xs, { requirePath: "fused" }));

      expect(() => f()).toThrow(/requirePath/);

      f.dispose();
      // Reset to wasm for subsequent tests
      defaultDevice("wasm");
    });

    it("succeeds when requirePath matches actual path", async () => {
      // Simple cumsum body that should compile to fused
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry.ref];
      };

      const init = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0]]);

      // On wasm/webgpu, this should use fused; on cpu it's fallback
      // Use array to allow either fused or fallback
      const f = jit(() =>
        lax.scan(step, init, xs, {
          requirePath: ["fused", "fallback"],
        }),
      );

      const [carry, ys] = f();

      const carryData = await carry.data();
      expect(carryData[0]).toBeCloseTo(6.0);

      const ysData = await ys.data();
      expect(Array.from(ysData)).toEqual([1, 3, 6]);

      f.dispose();
    });

    it("allows array of paths", async () => {
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x);
        return [newCarry, newCarry.ref];
      };

      const init = np.array([0.0]);
      const xs = np.array([[1.0], [2.0]]);

      // Allow multiple paths (now just fused or fallback)
      const f = jit(() =>
        lax.scan(step, init, xs, {
          requirePath: ["fused", "fallback"],
        }),
      );

      const [carry] = f();
      const carryData = await carry.data();
      expect(carryData[0]).toBeCloseTo(3.0);

      f.dispose();
    });
  });

  /**
   * ============================================================================
   * KNOWN LIMITATIONS - Tests that verify documented missing features
   * ============================================================================
   *
   * These tests verify that KNOWN LIMITATIONS still exist as documented in
   * .github/copilot-instructions.md. They PASS when the limitation exists.
   *
   * If a test FAILS, it means the limitation has been FIXED! 🎉
   * When that happens:
   * 1. Update .github/copilot-instructions.md to remove/update the limitation
   * 2. Convert this test to a normal test using requirePath to verify the fix
   * 3. Celebrate! 🎊
   *
   * See: .github/copilot-instructions.md section "Note on compiled-body (routine bodies)"
   * ============================================================================
   */
  describe("KNOWN LIMITATIONS (pass = limitation exists, fail = limitation fixed)", () => {
    it("WebGPU: Cholesky in scan body uses fallback instead of fused", async () => {
      const availableDevices = await init();
      if (!availableDevices.includes("webgpu")) {
        // Skip on non-WebGPU environments
        return;
      }

      defaultDevice("webgpu");

      let capturedPath: string | null = null;
      setScanPathCallback((path) => {
        capturedPath = path;
      });

      try {
        const step = (carry: np.Array, _x: np.Array): [np.Array, np.Array] => {
          const L = lax.linalg.cholesky(carry);
          const reconstructed = np.matmul(L.ref, L.transpose());
          return [reconstructed, L];
        };

        const initCarry = np.array([
          [4.0, 2.0],
          [2.0, 5.0],
        ]);
        const xs = np.array([[1.0], [1.0], [1.0]]);

        const f = jit(() => lax.scan(step, initCarry, xs));
        const [carry, ys] = f();
        await carry.data();
        await ys.data();
        f.dispose();

        // This test PASSES if the limitation still exists (uses fallback)
        // If this FAILS, the limitation was fixed - update docs!
        expect(
          capturedPath,
          "🎉 LIMITATION FIXED! Cholesky now uses fused. " +
            "Please update .github/copilot-instructions.md and convert this to a normal test.",
        ).not.toBe("fused");
      } finally {
        setScanPathCallback(null);
      }
    });

    it("WebGPU: TriangularSolve in scan body uses fallback instead of fused", async () => {
      const availableDevices = await init();
      if (!availableDevices.includes("webgpu")) {
        return;
      }

      defaultDevice("webgpu");

      let capturedPath: string | null = null;
      setScanPathCallback((path) => {
        capturedPath = path;
      });

      try {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const L = np.array([
            [2.0, 0.0],
            [1.0, 3.0],
          ]);
          const result = lax.linalg.triangularSolve(L, x, { lower: true });
          const newCarry = np.add(carry, result);
          return [newCarry, result];
        };

        const initCarry = np.array([
          [0.0, 0.0],
          [0.0, 0.0],
        ]);
        const xs = np.array([
          [
            [2.0, 1.0],
            [5.0, 2.0],
          ],
          [
            [4.0, 3.0],
            [7.0, 4.0],
          ],
        ]);

        const f = jit(() => lax.scan(step, initCarry, xs));
        const [carry, ys] = f();
        await carry.data();
        await ys.data();
        f.dispose();

        expect(
          capturedPath,
          "🎉 LIMITATION FIXED! TriangularSolve now uses fused. " +
            "Please update .github/copilot-instructions.md and convert this to a normal test.",
        ).not.toBe("fused");
      } finally {
        setScanPathCallback(null);
      }
    });

    it("WebGPU: LU in scan body uses fallback instead of fused", async () => {
      const availableDevices = await init();
      if (!availableDevices.includes("webgpu")) {
        return;
      }

      defaultDevice("webgpu");

      let capturedPath: string | null = null;
      setScanPathCallback((path) => {
        capturedPath = path;
      });

      try {
        const step = (carry: np.Array, _x: np.Array): [np.Array, np.Array] => {
          const [lu, pivots] = lax.linalg.lu(carry);
          return [lu, pivots];
        };

        const initCarry = np.array([
          [4.0, 3.0],
          [6.0, 3.0],
        ]);
        const xs = np.array([[1.0], [1.0]]);

        const f = jit(() => lax.scan(step, initCarry, xs));
        const [carry, ys] = f();
        await carry.data();
        await ys.data();
        f.dispose();

        expect(
          capturedPath,
          "🎉 LIMITATION FIXED! LU now uses fused. " +
            "Please update .github/copilot-instructions.md and convert this to a normal test.",
        ).not.toBe("fused");
      } finally {
        setScanPathCallback(null);
      }
    });

    it("WebGPU: Sort in scan body uses fallback (uniforms conflict)", async () => {
      // Sort/Argsort use uniforms internally which conflict with fused scan's offset uniforms
      const availableDevices = await init();
      if (!availableDevices.includes("webgpu")) {
        return;
      }

      defaultDevice("webgpu");

      let capturedPath: string | null = null;
      setScanPathCallback((path) => {
        capturedPath = path;
      });

      try {
        const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
          const sorted = np.sort(x);
          const newCarry = np.add(carry, sorted);
          return [newCarry, sorted];
        };

        const initCarry = np.array([0.0, 0.0, 0.0]);
        const xs = np.array([
          [3.0, 1.0, 2.0],
          [6.0, 4.0, 5.0],
        ]);

        const f = jit(() => lax.scan(step, initCarry, xs));
        const [carry, ys] = f();
        await carry.data();
        await ys.data();
        f.dispose();

        // Sort uses uniforms internally, so fused is not possible
        expect(
          capturedPath,
          "🎉 LIMITATION FIXED! Sort now uses fused (uniforms conflict resolved). " +
            "Please update .github/copilot-instructions.md and convert this to a normal test.",
        ).not.toBe("fused");
      } finally {
        setScanPathCallback(null);
      }
    });

    it("WebGPU: numCarry ≠ numY uses fallback instead of fused", async () => {
      // WebGPU compiled-loop requires numCarry === numY
      // When they differ, falls back to JS loop
      const availableDevices = await init();
      if (!availableDevices.includes("webgpu")) {
        return;
      }

      defaultDevice("webgpu");

      let capturedPath: string | null = null;
      setScanPathCallback((path) => {
        capturedPath = path;
      });

      try {
        // Body with 1 carry but 2 outputs (numCarry=1, numY=2)
        const step = (
          carry: np.Array,
          x: np.Array,
        ): [np.Array, [np.Array, np.Array]] => {
          const newCarry = np.add(carry, x);
          const y1 = newCarry.ref;
          const y2 = np.multiply(newCarry.ref, np.array([2.0]));
          return [newCarry, [y1, y2]];
        };

        const initCarry = np.array([0.0]);
        const xs = np.array([[1.0], [2.0], [3.0]]);

        const f = jit(() => lax.scan(step, initCarry, xs));
        const [carry, [ys1, ys2]] = f() as [np.Array, [np.Array, np.Array]];
        await carry.data();
        await ys1.data();
        await ys2.data();
        f.dispose();

        // WebGPU doesn't support numCarry ≠ numY in fused
        // (WASM's general scan does, but WebGPU falls back)
        expect(
          capturedPath,
          "🎉 LIMITATION FIXED! WebGPU now supports numCarry ≠ numY in fused. " +
            "Please update .github/copilot-instructions.md and convert this to a normal test.",
        ).toBe("fallback");
      } finally {
        setScanPathCallback(null);
      }
    });
  });

  describe("ownership edge cases", () => {
    // These tests specifically target the scanRunner refcount logic for edge cases
    // that could cause leaks or double-frees.

    it("duplicate-slot output: return [x.ref, x]", async () => {
      // Body returns the same array in both carry and y positions
      // Tests: incRef handling for duplicate slots in bodyOuts
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const result = np.add(carry, x);
        return [result.ref, result]; // Same slot in both positions
      };

      const initCarry = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0]]);

      const [finalCarry, ys] = (await lax.scan(
        step,
        initCarry.ref,
        xs.ref,
      )) as [np.Array, np.Array];

      expect(await finalCarry.data()).toEqual(new Float32Array([6.0]));
      expect(await ys.data()).toEqual(new Float32Array([1.0, 3.0, 6.0]));
      initCarry.dispose();
      xs.dispose();
    });

    it("carry passthrough: return [carry.ref, y]", async () => {
      // Carry is unchanged, only y is new
      // Tests: carry passthrough detection
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const y = np.add(carry.ref, x);
        return [carry, y]; // Carry unchanged, y is computed
      };

      const initCarry = np.array([10.0]);
      const xs = np.array([[1.0], [2.0], [3.0]]);

      const [finalCarry, ys] = (await lax.scan(
        step,
        initCarry.ref,
        xs.ref,
      )) as [np.Array, np.Array];

      // Carry stays 10, ys = [10+1, 10+2, 10+3]
      expect(await finalCarry.data()).toEqual(new Float32Array([10.0]));
      expect(await ys.data()).toEqual(new Float32Array([11.0, 12.0, 13.0]));
      initCarry.dispose();
      xs.dispose();
    });

    it("xs passthrough: return [newCarry, x]", async () => {
      // Y output is the xs slice itself
      // Tests: xs-passthrough detection and copy
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x.ref); // Use x.ref since we use x again
        return [newCarry, x]; // Y is the input x
      };

      const initCarry = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0]]);

      const [finalCarry, ys] = (await lax.scan(
        step,
        initCarry.ref,
        xs.ref,
      )) as [np.Array, np.Array];

      expect(await finalCarry.data()).toEqual(new Float32Array([6.0]));
      // ys should be a copy of xs
      expect(await ys.data()).toEqual(new Float32Array([1.0, 2.0, 3.0]));
      initCarry.dispose();
      xs.dispose();
    });

    it("jit(scan) with duplicate-slot output", async () => {
      // Same as above but through JIT
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const result = np.add(carry, x);
        return [result.ref, result];
      };

      const initCarry = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0]]);

      const f = jit(() => lax.scan(step, initCarry, xs));
      const [finalCarry, ys] = f() as [np.Array, np.Array];

      expect(await finalCarry.data()).toEqual(new Float32Array([6.0]));
      expect(await ys.data()).toEqual(new Float32Array([1.0, 3.0, 6.0]));
      f.dispose();
    });

    it("jit(scan) with xs passthrough, reverse=true", async () => {
      // Reverse scan with xs passthrough
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const newCarry = np.add(carry, x.ref); // Use x.ref since we use x again
        return [newCarry, x];
      };

      const initCarry = np.array([0.0]);
      const xs = np.array([[1.0], [2.0], [3.0]]);

      const f = jit(() => lax.scan(step, initCarry, xs, { reverse: true }));
      const [finalCarry, ys] = f() as [np.Array, np.Array];

      // Reverse: processes xs[2], xs[1], xs[0] → carry = 3+2+1 = 6
      expect(await finalCarry.data()).toEqual(new Float32Array([6.0]));
      // ys[0] = xs[2] (first iteration output), ys[1] = xs[1], ys[2] = xs[0]
      // But the stacking happens in iteration order, so ys = [xs[2], xs[1], xs[0]]
      // Wait, actually JAX semantics: ys are stacked in their original iteration index
      // So ys[i] = output at iteration i, regardless of reverse
      // For reverse=true: iter 0 processes xs[2], iter 1 processes xs[1], etc.
      // So ys = [3, 2, 1] if we stack outputs in iteration order
      // But our implementation might stack in xs order... let me check the actual output
      // The actual output is [1, 2, 3] meaning ys maintains xs order, which is correct!
      expect(await ys.data()).toEqual(new Float32Array([1.0, 2.0, 3.0]));
      f.dispose();
    });

    it("jit(scan) with carry passthrough and multiple iterations", async () => {
      // More iterations to stress the carry passthrough path
      const step = (carry: np.Array, x: np.Array): [np.Array, np.Array] => {
        const y = np.multiply(carry.ref, x);
        return [carry, y];
      };

      const initCarry = np.array([2.0]);
      const xs = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

      const f = jit(() => lax.scan(step, initCarry, xs));
      const [finalCarry, ys] = f() as [np.Array, np.Array];

      // Carry stays 2, ys = [2*1, 2*2, 2*3, 2*4, 2*5]
      expect(await finalCarry.data()).toEqual(new Float32Array([2.0]));
      expect(await ys.data()).toEqual(
        new Float32Array([2.0, 4.0, 6.0, 8.0, 10.0]),
      );
      f.dispose();
    });
  });
});
