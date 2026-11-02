import { jit, numpy as np, tree } from "@jax-js/jax";
import {
  adam,
  adamw,
  addDecayedWeights,
  applyUpdates,
  type OptState,
  scaleByLearningRate,
  scaleBySchedule,
  type Schedule,
} from "@jax-js/optax";
import { expect, test } from "vitest";

// Start at zero to catch accidentally freezing the schedule during tracing.
type Params = { a: np.Array; b: np.Array };

const linear: Schedule = (count) => count.astype(np.float32).mul(0.1);
const piecewise: Schedule = (count) => np.where(count.less(2), 0.0, 0.2);

test.each([false, true])(
  "scaleBySchedule updates scalar count (jit=%s)",
  (compile) => {
    const transform = scaleBySchedule(linear);
    const apply = (updates: Params, state: OptState) =>
      transform.update(updates, state);
    const update = compile ? jit(apply) : apply;
    let state = transform.init([]);
    for (let i = 0; i < 4; i++) {
      const oldCount = (state as { count: np.Array }).count;
      const [updates, next] = update(
        { a: np.array([2.0, 3.0]), b: np.array(-4.0) },
        state,
      );
      state = next;
      expect(updates.a).toBeAllclose([0.2 * i, 0.3 * i]);
      expect(updates.b).toBeAllclose(-0.4 * i);
      expect(oldCount.refCount).toBe(0);
      const { count } = state as { count: np.Array };
      expect(count.dtype).toBe(np.uint32);
      expect(count.shape).toEqual([]);
      expect(count.ref.item()).toBe(i + 1);
    }
    tree.dispose(state);
  },
);

test.each([false, true])(
  "scheduled learning-rate sign (flipSign=%s)",
  (flipSign) => {
    const transform = scaleByLearningRate(
      (count) => count.astype(np.float32).add(1).mul(0.1),
      flipSign,
    );
    const update = jit((updates: np.Array, state: OptState) =>
      transform.update(updates, state),
    );
    let state = transform.init([]);
    for (let i = 0; i < 3; i++) {
      const [updates, next] = update(np.array(2.0), state);
      state = next;
      expect(updates).toBeAllclose((flipSign ? -1 : 1) * 0.2 * (i + 1));
    }
    tree.dispose(state);
  },
);

test.each([false, true])(
  "scheduled weight decay supports masks (mask=%s)",
  (masked) => {
    const transform = addDecayedWeights({
      weightDecay: piecewise,
      mask: masked
        ? (params) => tree.map((p: np.Array) => p.mul(0).add(0.5), params)
        : null,
    });
    const update = jit((updates: Params, state: OptState, params: Params) =>
      transform.update(updates, state, params),
    );
    let state = transform.init([]);
    for (let i = 0; i < 4; i++) {
      const [updates, next] = update(
        { a: np.array([1.0, 1.0]), b: np.array(1.0) },
        state,
        { a: np.array([2.0, 3.0]), b: np.array(4.0) },
      );
      state = next;
      const decay = i < 2 ? 0 : masked ? 0.1 : 0.2;
      expect(updates.a).toBeAllclose([1 + 2 * decay, 1 + 3 * decay]);
      expect(updates.b).toBeAllclose(1 + 4 * decay);
      const { count } = state as { count: np.Array };
      expect(count.dtype).toBe(np.uint32);
      expect(count.ref.item()).toBe(i + 1);
    }
    tree.dispose(state);
  },
);

test.each(["adam", "adamw"])(
  "%s with schedules matches eager over multiple steps",
  (name) => {
    // Binary-exact moment coefficients isolate schedules from bias-correction rounding.
    const solver =
      name === "adam"
        ? adam(linear, { b1: 0.5, b2: 0.5 })
        : adamw(linear, { b1: 0.5, b2: 0.5, weightDecay: piecewise });
    const initial = () => ({ a: np.array([1.0, 2.0]), b: np.array(3.0) });
    const step = (params: ReturnType<typeof initial>, state: OptState) => {
      const grads = tree.map(
        (p: np.Array) => p.mul(0).add(1),
        tree.ref(params),
      );
      const [updates, next] = solver.update(grads, state, tree.ref(params));
      return [applyUpdates(params, updates), next] as [
        ReturnType<typeof initial>,
        OptState,
      ];
    };
    const compiled = jit(step);
    let eager = initial();
    let eagerState = solver.init(tree.ref(eager));
    let jitted = initial();
    let jittedState = solver.init(tree.ref(jitted));
    for (let i = 0; i < 4; i++) {
      [eager, eagerState] = step(eager, eagerState);
      [jitted, jittedState] = compiled(jitted, jittedState);
      expect(jitted.a.ref).toBeAllclose(eager.a.ref, { rtol: 1e-5 });
      expect(jitted.b.ref).toBeAllclose(eager.b.ref, { rtol: 1e-5 });
      if (i === 0) {
        expect(jitted.a.ref).toBeAllclose([1, 2]);
      }
    }
    tree.dispose([eager, eagerState, jitted, jittedState]);
  },
);

test.each([
  scaleBySchedule,
  (schedule: Schedule) => addDecayedWeights({ weightDecay: schedule }),
])(
  "scheduled scalar references are released for empty trees",
  (makeTransform) => {
    let scalar: np.Array;
    const transform = makeTransform(
      (count) => (scalar = count.astype(np.float32)),
    );
    const [updates, state] = transform.update([], transform.init([]), []);
    expect(updates).toEqual([]);
    expect(scalar!.refCount).toBe(0);
    tree.dispose(state);
  },
);

test.each([
  scaleBySchedule,
  (schedule: Schedule) => addDecayedWeights({ weightDecay: schedule }),
])("schedules must return a scalar", (makeTransform) => {
  const transform = makeTransform((count) => count.reshape([1]));
  const state = transform.init([]);
  expect(() => transform.update([], state, [])).toThrow(
    "Schedule must return a scalar array",
  );
  tree.dispose(state);
});

test("exponential schedules use floating-point arithmetic under jit", () => {
  const transform = scaleBySchedule((step) =>
    np.power(0.9, step.astype(np.float32)),
  );
  const update = jit((updates: np.Array, state: OptState) =>
    transform.update(updates, state),
  );
  let state = transform.init([]);
  for (let i = 0; i < 4; i++) {
    const [updates, next] = update(np.array(2.0), state);
    state = next;
    expect(updates).toBeAllclose(2 * Math.pow(0.9, i));
  }
  tree.dispose(state);
});

test("static zero weight decay retains its shortcut", () => {
  const transform = addDecayedWeights({ weightDecay: 0 });
  const [updates, state] = transform.update(
    np.array(1.0),
    transform.init([]),
    np.array(Infinity),
  );
  expect(updates).toBeAllclose(1);
  tree.dispose(state);
});
