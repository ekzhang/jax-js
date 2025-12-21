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
