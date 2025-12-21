# @jax-js/onnx

ONNX model loader for jax-js.

## Usage

Fetch a model from external path and load it.

```ts
import { ONNXModel } from "@jax-js/onnx";
import { numpy as np } from "@jax-js/jax";

const modelBytes = await fetch("./model.onnx").then((r) => r.bytes());
const model = new ONNXModel(modelBytes);

const input = np.ones([1, 3, 224, 224]);
const { output } = model.run({ input });
```

Loaded models are ordinary functions and can be differentiated through. Use JIT when possible for
best performance.

```ts
import { grad, jit } from "@jax-js/jax";

const run = jit(model.run);
const runGrad = grad((input: np.Array) => {
  const { output } = run({ input });
  return computeLoss(output).mean();
});

const dx = runGrad(input);
```

After you're done, you can free the model weights.

```ts
model.dispose();
```
