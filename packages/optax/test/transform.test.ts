import { numpy as np } from "@jax-js/jax";
import {
  addDecayedWeights,
  scaleByLearningRate,
  scaleBySchedule,
  trace,
} from "@jax-js/optax";
import { expect, test } from "vitest";

test("scaleByLearningRate function", () => {
  const params = np.array([1.0, 2.0, 3.0]);
  const updates = np.array([0.1, 0.2, 0.3]);

  const transform = scaleByLearningRate(0.1);
  const state = transform.init(params.ref);

  const [newUpdates, _newState] = transform.update(
    updates.ref,
    state,
    params.ref,
  );

  // Should scale by -0.1 (negative learning rate)
  expect(newUpdates).toBeAllclose([-0.01, -0.02, -0.03]);
});

test("addDecayedWeights function with scalar", () => {
  const params = np.array([1.0, 2.0, 3.0]);
  const updates = np.array([0.1, 0.2, 0.3]);

  const transform = addDecayedWeights({ weightDecay: 0.01 });
  const state = transform.init(params.ref);

  const [newUpdates, _newState] = transform.update(
    updates.ref,
    state,
    params.ref,
  );

  // Should add weight decay: [0.1 + 0.01*1, 0.2 + 0.01*2, 0.3 + 0.01*3]
  expect(newUpdates).toBeAllclose([0.11, 0.22, 0.33]);
});

test("addDecayedWeights function with schedule", () => {
  const params = np.array([1.0, 2.0, 3.0]);
  const updates = np.array([0.1, 0.2, 0.3]);

  // Weight decay schedule that increases: 0.01, 0.02, 0.03, ...
  const weightDecaySchedule = (step: number) => 0.01 * (step + 1);
  const transform = addDecayedWeights({ weightDecay: weightDecaySchedule });
  const state = transform.init(params.ref);

  // First update (step 0): weight_decay = 0.01
  let [newUpdates, newState] = transform.update(updates.ref, state, params.ref);
  expect(newUpdates).toBeAllclose([0.11, 0.22, 0.33]);

  // Second update (step 1): weight_decay = 0.02
  [newUpdates, newState] = transform.update(updates.ref, newState, params.ref);
  expect(newUpdates).toBeAllclose([0.12, 0.24, 0.36]);
});

test("addDecayedWeights function with mask", () => {
  const params = np.array([1.0, 2.0, 3.0]);
  const updates = np.array([0.1, 0.2, 0.3]);
  // Mask: apply weight decay to first and third params only
  const mask = np.array([1.0, 0.0, 1.0]);

  const transform = addDecayedWeights({ weightDecay: 0.01, mask });
  const state = transform.init(params.ref);

  const [newUpdates, _newState] = transform.update(
    updates.ref,
    state,
    params.ref,
  );

  // Should add weight decay only where mask = 1: [0.1 + 0.01*1, 0.2 + 0, 0.3 + 0.01*3]
  expect(newUpdates).toBeAllclose([0.11, 0.2, 0.33]);
});

test("addDecayedWeights throws error when params is undefined", () => {
  const updates = np.array([0.1, 0.2, 0.3]);

  const transform = addDecayedWeights({ weightDecay: 0.01 });
  const state = transform.init(np.array([1.0, 2.0, 3.0]));

  expect(() => {
    transform.update(updates.ref, state, undefined);
  }).toThrow("addDecayedWeights requires params to be provided");
});

test("scaleBySchedule function with dynamic learning rate", () => {
  const params = np.array([1.0, 2.0, 3.0]);
  const updates = np.array([0.1, 0.2, 0.3]);

  // Learning rate starts at 1.0 and decreases by 10% each step
  const schedule = (step: number) => Math.pow(0.9, step);
  const transform = scaleBySchedule(schedule);
  const state = transform.init(params.ref);

  // First update (step 0)
  let [newUpdates, newState] = transform.update(updates.ref, state, params.ref);
  expect(newUpdates).toBeAllclose([0.1, 0.2, 0.3]); // 1.0 * updates

  // Second update (step 1)
  [newUpdates, newState] = transform.update(updates.ref, newState, params.ref);
  expect(newUpdates).toBeAllclose([0.09, 0.18, 0.27]); // 0.9 * updates
});

test("trace transformation", () => {
  const params = np.array([1.0, 2.0, 3.0]);
  const updates = np.array([0.1, 0.2, 0.3]);

  const transform = trace({ decay: 0.9 });
  const state = transform.init(params.ref);

  // First update: trace should be equal to updates
  let [newUpdates, newState] = transform.update(updates.ref, state, params.ref);
  expect(newUpdates).toBeAllclose([0.1, 0.2, 0.3]);

  // Second update: trace = updates + 0.9 * prev_trace
  [newUpdates, newState] = transform.update(updates.ref, newState, params.ref);
  expect(newUpdates).toBeAllclose([0.19, 0.38, 0.57]); // 0.1 + 0.9*0.1, etc.
});

test("trace transformation with nesterov", () => {
  const params = np.array([1.0, 2.0, 3.0]);
  const updates = np.array([0.1, 0.2, 0.3]);

  const transform = trace({ decay: 0.9, nesterov: true });
  const state = transform.init(params.ref);

  // First update with nesterov
  let [newUpdates, newState] = transform.update(updates.ref, state, params.ref);
  expect(newUpdates).toBeAllclose([0.19, 0.38, 0.57]); // g + decay * (g + decay * 0)

  // Second update with nesterov
  [newUpdates, newState] = transform.update(updates.ref, newState, params.ref);
  expect(newUpdates.shape).toEqual([3]);
  expect(newUpdates.dtype).toEqual(np.float32);
});
