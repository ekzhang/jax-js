import { beforeAll, describe, expect, it } from "vitest";

import { defaultDevice, init, jit, lax, numpy as np } from "../src";

/**
 * Test case extracted from dlm-js to reproduce jit(scan) issue.
 *
 * The pattern:
 * 1. jit(scan) with passthrough outputs (carry inputs returned as Y outputs)
 * 2. flip the stacked outputs
 * 3. read data from flipped arrays
 */
describe("jit(scan) with passthrough and flip", () => {
  beforeAll(async () => {
    await init();
  });

  it("should handle passthrough Y outputs followed by flip", async () => {
    defaultDevice("wasm");

    // Simplified Kalman-like forward step
    // Carry: { x, C } - state and covariance
    // Input: { y } - observation
    // Output: { x_pred, C_pred, v } - predictions and innovation
    type Carry = { x: np.Array; C: np.Array };
    type X = { y: np.Array };
    type Y = { x_pred: np.Array; C_pred: np.Array; v: np.Array };

    const forwardStep = (carry: Carry, inp: X): [Carry, Y] => {
      const { x: xi, C: Ci } = carry;
      const { y: yi } = inp;

      // Innovation: v = y - x (simplified)
      const v = np.subtract(yi.ref, xi.ref);

      // Next state: x_next = x + 0.5 * v (simplified Kalman update)
      const x_next = np.add(xi.ref, np.multiply(np.array([0.5]), v.ref));

      // Covariance update (simplified): C_next = C * 0.9
      const C_next = np.multiply(Ci.ref, np.array([0.9]));

      // Output includes passthrough of carry inputs
      const output: Y = {
        x_pred: xi.ref, // passthrough from carry
        C_pred: Ci.ref, // passthrough from carry
        v: v.ref,
      };

      return [{ x: x_next, C: C_next }, output];
    };

    // Initial state
    const x0 = np.array([0.0]);
    const C0 = np.array([1.0]);

    // Observations
    const ys = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

    // Run jit(scan)
    const [finalCarry, outputs] = await jit(
      (x0: np.Array, C0: np.Array, ys: np.Array) =>
        lax.scan(forwardStep, { x: x0, C: C0 }, { y: ys }),
    )(x0, C0, ys);

    // Check outputs are valid
    expect(outputs.x_pred.shape).toEqual([5, 1]);
    expect(outputs.C_pred.shape).toEqual([5, 1]);
    expect(outputs.v.shape).toEqual([5, 1]);

    // Read stacked output data
    const x_pred_data = await outputs.x_pred.ref.data();
    console.log("x_pred_data:", x_pred_data);
    expect(x_pred_data[0]).toBeCloseTo(0.0, 5); // First prediction is initial x

    // Flip the outputs
    const x_pred_rev = np.flip(outputs.x_pred.ref, 0);
    const C_pred_rev = np.flip(outputs.C_pred.ref, 0);

    // Read flipped data
    const x_pred_rev_data = await x_pred_rev.ref.data();
    console.log("x_pred_rev_data:", x_pred_rev_data);

    // Last value in forward order should be first in reversed
    expect(x_pred_rev_data).not.toContain(NaN);
    expect(x_pred_rev_data[0]).not.toBeNull();

    // Cleanup
    finalCarry.x.dispose();
    finalCarry.C.dispose();
    outputs.x_pred.dispose();
    outputs.C_pred.dispose();
    outputs.v.dispose();
    x_pred_rev.dispose();
    C_pred_rev.dispose();
  });

  it("should handle non-jit scan with same pattern (baseline)", async () => {
    defaultDevice("wasm");

    type Carry = { x: np.Array; C: np.Array };
    type X = { y: np.Array };
    type Y = { x_pred: np.Array; C_pred: np.Array; v: np.Array };

    const forwardStep = (carry: Carry, inp: X): [Carry, Y] => {
      const { x: xi, C: Ci } = carry;
      const { y: yi } = inp;

      const v = np.subtract(yi.ref, xi.ref);
      const x_next = np.add(xi.ref, np.multiply(np.array([0.5]), v.ref));
      const C_next = np.multiply(Ci.ref, np.array([0.9]));

      const output: Y = {
        x_pred: xi.ref,
        C_pred: Ci.ref,
        v: v.ref,
      };

      return [{ x: x_next, C: C_next }, output];
    };

    const x0 = np.array([0.0]);
    const C0 = np.array([1.0]);
    const ys = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]]);

    // Run non-jit scan
    const [finalCarry, outputs] = lax.scan(
      forwardStep,
      { x: x0, C: C0 },
      { y: ys },
    );

    // Check outputs are valid
    expect(outputs.x_pred.shape).toEqual([5, 1]);

    // Read stacked output data
    const x_pred_data = await outputs.x_pred.ref.data();
    console.log("non-jit x_pred_data:", x_pred_data);
    expect(x_pred_data[0]).toBeCloseTo(0.0, 5);

    // Flip the outputs
    const x_pred_rev = np.flip(outputs.x_pred.ref, 0);

    // Read flipped data
    const x_pred_rev_data = await x_pred_rev.ref.data();
    console.log("non-jit x_pred_rev_data:", x_pred_rev_data);

    expect(x_pred_rev_data).not.toContain(NaN);

    // Cleanup
    finalCarry.x.dispose();
    finalCarry.C.dispose();
    outputs.x_pred.dispose();
    outputs.C_pred.dispose();
    outputs.v.dispose();
    x_pred_rev.dispose();
  });

  it("should match dlm-js two-pass pattern with slicing", async () => {
    defaultDevice("wasm");

    // This test more closely matches dlm-js: forward scan -> flip -> backward scan -> flip -> slice
    type ForwardCarry = { x: np.Array; C: np.Array };
    type ForwardX = { y: np.Array };
    type ForwardY = { x_pred: np.Array; C_pred: np.Array; v: np.Array };

    const forwardStep = (
      carry: ForwardCarry,
      inp: ForwardX,
    ): [ForwardCarry, ForwardY] => {
      const { x: xi, C: Ci } = carry;
      const { y: yi } = inp;

      const v = np.subtract(yi.ref, xi.ref);
      const x_next = np.add(
        xi.ref,
        np.multiply(np.array([[0.5], [0.1]]), v.ref),
      );
      const C_next = np.multiply(
        Ci.ref,
        np.array([
          [0.9, 0],
          [0, 0.9],
        ]),
      );

      // Passthrough pattern - carry inputs returned as Y outputs
      return [
        { x: x_next, C: C_next },
        { x_pred: xi.ref, C_pred: Ci.ref, v: v.ref },
      ];
    };

    // 2x1 state, 2x2 covariance
    const x0 = np.array([[0.0], [0.0]]);
    const C0 = np.array([
      [1.0, 0.0],
      [0.0, 1.0],
    ]);
    const ys = np.array([
      [[1.0], [0.5]],
      [[2.0], [1.0]],
      [[3.0], [1.5]],
    ]); // 3 observations

    // Run jit(scan) for forward pass
    const [finalCarry, forwardOutputs] = await jit(
      (x0: np.Array, C0: np.Array, ys: np.Array) =>
        lax.scan(forwardStep, { x: x0, C: C0 }, { y: ys }),
    )(x0, C0, ys);

    console.log("Forward outputs shapes:", {
      x_pred: forwardOutputs.x_pred.shape,
      C_pred: forwardOutputs.C_pred.shape,
      v: forwardOutputs.v.shape,
    });

    // Flip outputs
    const x_pred_rev = np.flip(forwardOutputs.x_pred.ref, 0);
    const C_pred_rev = np.flip(forwardOutputs.C_pred.ref, 0);
    const v_rev = np.flip(forwardOutputs.v.ref, 0);

    // Read flipped data
    const x_pred_rev_data = await x_pred_rev.ref.data();
    console.log("x_pred_rev_data:", x_pred_rev_data);

    // Slice individual timesteps from stacked outputs (like dlm-js does)
    const n = 3;
    const x_pred_slices: np.Array[] = [];
    for (let i = 0; i < n; i++) {
      const sliced = forwardOutputs.x_pred.ref.slice(i);
      const reshaped = np.reshape(sliced, [2, 1]);
      x_pred_slices.push(reshaped);
    }

    // Read data from slices
    for (let i = 0; i < n; i++) {
      const data = await x_pred_slices[i].ref.data();
      console.log(`x_pred_slice[${i}]:`, data);
      expect(data[0]).not.toBeNull();
      expect(data).not.toContain(NaN);
    }

    // Cleanup
    finalCarry.x.dispose();
    finalCarry.C.dispose();
    forwardOutputs.x_pred.dispose();
    forwardOutputs.C_pred.dispose();
    forwardOutputs.v.dispose();
    x_pred_rev.dispose();
    C_pred_rev.dispose();
    v_rev.dispose();
    x_pred_slices.forEach((s) => s.dispose());
  });

  it("should match dlm-js full pattern: two calls returning sliced arrays", async () => {
    defaultDevice("wasm");

    // This mimics dlmFit calling dlmSmo twice
    // Key: dlmSmo slices the stacked outputs, disposes the stacked, and returns the slices
    type ForwardCarry = { x: np.Array; C: np.Array };
    type ForwardX = { y: np.Array };
    type ForwardY = { x_pred: np.Array };

    const forwardStep = (
      carry: ForwardCarry,
      inp: ForwardX,
    ): [ForwardCarry, ForwardY] => {
      const { x: xi, C: Ci } = carry;
      const { y: yi } = inp;

      const v = np.subtract(yi.ref, xi.ref);
      const x_next = np.add(
        xi.ref,
        np.multiply(np.array([[0.5], [0.1]]), v.ref),
      );
      const C_next = np.multiply(
        Ci.ref,
        np.array([
          [0.9, 0],
          [0, 0.9],
        ]),
      );

      // Passthrough: return carry input as Y output
      return [{ x: x_next, C: C_next }, { x_pred: xi.ref }];
    };

    // Function that mimics dlmSmo - returns sliced arrays after disposing stacked
    const runSmoother = async (
      x0: np.Array,
      C0: np.Array,
      ys: np.Array,
      useJit: boolean,
    ) => {
      const n = ys.shape[0];

      const [finalCarry, forwardOutputs] = useJit
        ? await jit((x0: np.Array, C0: np.Array, ys: np.Array) =>
            lax.scan(forwardStep, { x: x0, C: C0 }, { y: ys }),
          )(x0, C0, ys)
        : lax.scan(forwardStep, { x: x0, C: C0 }, { y: ys });

      // Dispose carry (like dlm-js does)
      finalCarry.x.dispose();
      finalCarry.C.dispose();

      // Slice stacked outputs into per-timestep arrays (like dlm-js does)
      const x_pred_stacked = forwardOutputs.x_pred;
      const x_pred: np.Array[] = [];
      for (let i = 0; i < n; i++) {
        const sliced = x_pred_stacked.ref.slice(i);
        const reshaped = np.reshape(sliced, [2, 1]);
        x_pred.push(reshaped);
      }

      // Dispose stacked array AFTER slicing (like dlm-js line 729)
      x_pred_stacked.dispose();

      return { xf: x_pred };
    };

    // Initial state
    const x0 = np.array([[0.0], [0.0]]);
    const C0 = np.array([
      [1.0, 0.0],
      [0.0, 1.0],
    ]);
    const ys = np.array([
      [[1.0], [0.5]],
      [[2.0], [1.0]],
      [[3.0], [1.5]],
    ]);

    // Pass 1
    console.log("Running Pass 1 with jit...");
    const out1 = await runSmoother(x0.ref, C0.ref, ys.ref, true);

    // Extract data from Pass 1 (like dlmFit lines 846-847)
    const x0_new = await out1.xf[0].ref.data();
    console.log("Pass 1 xf[0]:", x0_new);
    expect(x0_new[0]).not.toBeNull();
    expect(x0_new).not.toContain(NaN);

    // Dispose Pass 1 arrays
    out1.xf.forEach((arr) => arr.dispose());

    // Pass 2 with updated initial state
    console.log("Running Pass 2 with jit...");
    const x0_updated = np.array([[x0_new[0]], [x0_new[1]]]);
    const out2 = await runSmoother(x0_updated, C0.ref, ys.ref, true);

    // Extract data from Pass 2 (like dlmFit lines 877-879)
    const xf = [new Float32Array(3), new Float32Array(3)];
    for (let i = 0; i < 3; i++) {
      const xfi = await out2.xf[i].ref.data();
      console.log(`Pass 2 xf[${i}]:`, xfi);
      xf[0][i] = xfi[0];
      xf[1][i] = xfi[1];
      out2.xf[i].dispose();
    }

    // Verify no null values
    console.log("Final xf[0]:", xf[0]);
    console.log("Final xf[1]:", xf[1]);
    for (let i = 0; i < 3; i++) {
      expect(xf[0][i]).not.toBeNull();
      expect(xf[1][i]).not.toBeNull();
    }

    // Cleanup
    x0.dispose();
    C0.dispose();
    ys.dispose();
  });
});
