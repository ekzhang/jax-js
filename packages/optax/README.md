# @jax-js/optax

This is a port of [Optax](https://github.com/google-deepmind/optax) to jax-js, for gradient
processing and optimization. It includes implementations of common algorithms like Adam.

## Example

```ts
import { adam } from "@jax-js/optax";

let params = np.array([1.0, 2.0, 3.0]);

const solver = adam(1e-3);
let optState = solver.init(params.ref);
let updates: np.Array;

const f = (x: np.Array) => squaredError(x, np.ones([3])).sum();

for (let i = 0; i < 100; i++) {
  const paramsGrad = grad(f)(params.ref);
  [updates, optState] = solver.update(paramsGrad, optState);
  params = applyUpdates(params, updates);
}
```

## Schedules

Learning-rate and weight-decay schedules consume a scalar `uint32` array containing the zero-based
step count and return a scalar array. Using array operations keeps schedules compatible with `jit`:

```ts
import { numpy as np } from "@jax-js/jax";
import { adamw, type Schedule } from "@jax-js/optax";

const learningRate: Schedule = (step) => np.power(0.9, step.astype(np.float32)).mul(1e-3);
const weightDecay: Schedule = (step) => np.where(step.less(100), 0.0, 0.01);
const solver = adamw(learningRate, { weightDecay });
```

This replaces the previous `(step: number) => number` schedule API. Migrate JavaScript arithmetic,
`Math` functions, and conditionals to array operations (for example, `np.power` and `np.where`).
Cast the integer count to a floating-point dtype before fractional arithmetic. Constant numeric
learning rates and weight decay remain unchanged.

Schedules follow the usual array ownership rules: consume the input count, and return an owned
scalar reference. Use `.ref` when reusing the count or returning a captured array, and dispose the
count if the schedule ignores it. Scheduled zero weight decay is evaluated as multiplication, rather
than taking the static numeric-zero shortcut; nonfinite parameters therefore follow normal array
arithmetic (for example, zero times infinity is NaN).
