<h1 align="center">jax-js: JAX in pure JavaScript</h1>

<p align="center"><strong>
  <a href="https://jax-js.com">Website</a> |
  <a href="https://jax-js.com/docs/">API Reference</a> |
  <a href="./FEATURES.md">Compatibility Table</a> |
  <a href="https://discord.gg/BW6YsCd4Tf">Discord</a>
</strong></p>

**jax-js** is a machine learning framework for the browser. It aims to bring
[JAX](https://jax.dev)-style, high-performance CPU and GPU kernels to JavaScript, so you can run
numerical applications on the web.

```bash
npm i @jax-js/jax
```

Under the hood, it translates array operations into a compiler representation, then synthesizes
kernels in WebAssembly and WebGPU.

The library is written from scratch, with zero external dependencies. It maintains close API
compatibility with NumPy/JAX. Since everything runs client-side, jax-js is likely the most portable
GPU ML framework, since it runs anywhere a browser can run.

## Quickstart

```js
import { numpy as np } from "@jax-js/jax";

// Array operations, compatible with JAX/NumPy.
const x = np.array([1, 2, 3]);
const y = x.mul(4); // [4, 8, 12]
```

### Web usage (CDN)

In vanilla JavaScript (without a bundler), just import from a module script tag. This is the easiest
way to get started on a blank HTML page.

```html
<script type="module">
  import { numpy as np } from "https://esm.sh/@jax-js/jax";
</script>
```

## Examples

Demos on the jax-js website:

- [Training neural networks on MNIST](https://jax-js.com/mnist)
- [Voice cloning: Kyutai Pocket TTS](https://jax-js.com/tts)
- [CLIP embeddings for books in-browser](https://jax-js.com/mobileclip)
- [Object detection: DETR ResNet-50 (ONNX)](https://jax-js.com/detr-resnet-50)
- [In-browser REPL](https://jax-js.com/repl)
- [Matmul benchmark](https://jax-js.com/bench/matmul)
- [Conv2d benchmark](https://jax-js.com/bench/conv2d)
- [Mandelbrot set](https://jax-js.com/mandelbrot)

And some cool things made by the community:

- [**tanh.xyz**: Interactive ML visualizations](https://tanh.xyz/)

## Feature comparison

Here's a quick, high-level comparison with other popular web ML runtimes:

| Feature                         | jax-js     | TensorFlow.js   | onnxruntime-web    |
| ------------------------------- | ---------- | --------------- | ------------------ |
| **Overview**                    |            |                 |                    |
| API style                       | JAX/NumPy  | TensorFlow-like | Static ONNX graphs |
| Latest release                  | 2026       | ⚠️ 2024         | 2026               |
| Speed                           | Fastest    | Fast            | Fastest            |
| Bundle size (gzip)              | 80 KB      | 269 KB          | 90 KB + 24 MB Wasm |
| **Autodiff & JIT**              |            |                 |                    |
| Gradients                       | ✅         | ✅              | ❌                 |
| Jacobian and Hessian            | ✅         | ❌              | ❌                 |
| `jvp()` forward differentiation | ✅         | ❌              | ❌                 |
| `jit()` kernel fusion           | ✅         | ❌              | ❌                 |
| `vmap()` auto-vectorization     | ✅         | ❌              | ❌                 |
| Graph capture                   | ✅         | ❌              | ✅                 |
| **Backends & Data**             |            |                 |                    |
| WebGPU backend                  | ✅         | 🟡 Preview      | ✅                 |
| WebGL backend                   | ✅         | ✅              | ✅                 |
| Wasm (CPU) backend              | ✅         | ✅              | ✅                 |
| Eager array API                 | ✅         | ✅              | ❌                 |
| Run ONNX models                 | 🟡 Partial | ❌              | ✅                 |
| Read safetensors                | ✅         | ❌              | ❌                 |
| Float64                         | ✅         | ❌              | ❌                 |
| Float32                         | ✅         | ✅              | ✅                 |
| Float16                         | ✅         | ❌              | ✅                 |
| BFloat16                        | ❌         | ❌              | ❌                 |
| Packed Uint8                    | ❌         | ❌              | 🟡 Partial         |
| Mixed precision                 | ✅         | ❌              | ✅                 |
| Mixed devices                   | ✅         | ❌              | ❌                 |
| **Ops & Numerics**              |            |                 |                    |
| Arithmetic functions            | ✅         | ✅              | ✅                 |
| Matrix multiplication           | ✅         | ✅              | ✅                 |
| General einsum                  | ✅         | 🟡 Partial      | 🟡 Partial         |
| Sorting                         | ✅         | ❌              | ❌                 |
| Activation functions            | ✅         | ✅              | ✅                 |
| NaN/Inf numerics                | ✅         | ✅              | ✅                 |
| Basic convolutions              | ✅         | ✅              | ✅                 |
| n-d convolutions                | ✅         | ❌              | ✅                 |
| Strided/dilated convolution     | ✅         | ✅              | ✅                 |
| Cholesky, Lstsq                 | ✅         | ❌              | ❌                 |
| LU, Solve, Determinant          | ✅         | ❌              | ❌                 |
| SVD                             | ❌         | ❌              | ❌                 |
| FFT                             | ✅         | ✅              | ✅                 |
| Basic RNG (Uniform, Normal)     | ✅         | ✅              | ✅                 |
| Advanced RNG                    | ✅         | ❌              | ❌                 |

## Tutorial

Programming in `jax-js` looks [very similar to JAX](https://docs.jax.dev/en/latest/jax-101.html),
just in JavaScript.

### Arrays

Create an array with `np.array()`:

```ts
import { numpy as np } from "@jax-js/jax";

const ar = np.array([1, 2, 3]);
```

By default, this is a float32 array, but you can specify a different dtype:

```ts
const ar = np.array([1, 2, 3], { dtype: np.int32 });
```

For more efficient construction, create an array from a JS `TypedArray` buffer:

```ts
const buf = new Float32Array([10, 20, 30, 100, 200, 300]);
const ar = np.array(buf).reshape([2, 3]);
```

Once you're done with it, you can unwrap a `jax.Array` back into JavaScript. This will also apply
any pending operations or lazy updates:

```ts
// 1) Returns a possibly nested JavaScript array.
ar.js();
await ar.jsAsync(); // Faster, non-blocking

// 2) Returns a flat TypedArray data buffer.
ar.dataSync();
await ar.data(); // Fastest, non-blocking
```

Arrays can have mathematical operations applied to them. For example:

```ts
import { numpy as np, scipySpecial as special } from "@jax-js/jax";

const x = np.arange(100).astype(np.float32); // array of integers [0..99]

const y1 = x.ref.add(x.ref); // x + x
const y2 = np.sin(x.ref); // sin(x)
const y3 = np.tanh(x.ref).mul(5); // 5 * tanh(x)
const y4 = special.erfc(x.ref); // erfc(x)
```

Notice that in the above code, we used `x.ref`. This is because of the memory model, jax-js uses
reference-counted _ownership_ to track when the memory of an Array can be freed. More on this below.

### Reference counting

Big Arrays take up a lot of memory. Python ML libraries override the `__del__()` method to free
memory, but JavaScript has no such API for running object destructors
([cf.](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/FinalizationRegistry)).
This means that you have to track references manually. jax-js tries to make this as ergonomic as
possible, so you don't accidentally leak memory in a loop.

Every `jax.Array` has a reference count. This satisfies the following rules:

- Whenever you create an Array, its reference count starts at `1`.
- When an Array's reference count reaches `0`, it is freed and can no longer be used.
- Given an Array `a`:
  - Accessing `a.ref` returns `a` and changes its reference count by `+1`.
  - Passing `a` into any function as argument changes its reference count by `-1`.
  - Calling `a.dispose()` also changes its reference count by `-1`.

What this means is that all functions in jax-js must _take ownership_ of their arguments as
references. Whenever you would like to pass an Array as argument, you can pass it directly to
dispose of it, or use `.ref` if you'd like to use it again later.

**You must follow these rules on your own functions as well!** All combinators like `jvp`, `grad`,
`jit` assume that you are following these conventions on how arguments are passed, and they will
respect them as well.

```ts
// Bad: Uses `x` twice, decrementing its reference count twice.
function foo_bad(x: np.Array, y: np.Array) {
  return x.add(x.mul(y));
}

// Good: The first usage of `x` is `x.ref`, adding +1 to refcount.
function foo_good(x: np.Array, y: np.Array) {
  return x.ref.add(x.mul(y));
}
```

Here's another example:

```ts
// Bad: Doesn't consume `x` in the `if`-branch.
function bar_bad(x: np.Array, skip: boolean) {
  if (skip) return np.zeros(x.shape);
  return x;
}

// Good: Consumes `x` the one time in each branch.
function bar_good(x: np.Array, skip: boolean) {
  if (skip) {
    const ret = np.zeros(x.shape);
    x.dispose();
    return ret;
  }
  return x;
}
```

You can assume that every function in jax-js takes ownership properly, except with a couple of very
rare exceptions that are documented.

### grad(), vmap() and jit()

JAX's signature composable transformations are also supported in jax-js. Here is a simple example of
using `grad` and `vmap` to compute the derivaive of a function:

```ts
import { numpy as np, grad, vmap } from "@jax-js/jax";

const x = np.linspace(-10, 10, 1000);

const y1 = vmap(grad(np.sin))(x.ref); // d/dx sin(x) = cos(x)
const y2 = np.cos(x);

np.allclose(y1, y2); // => true
```

The `jit` function is especially useful when doing long sequences of primitives on GPU, since it
fuses operations together into a single kernel dispatch. This
[improves memory bandwidth usage](https://substack.com/home/post/p-163548742) on hardware
accelerators, which is the bottleneck on GPU rather than raw FLOPs. For instance:

```ts
export const hypot = jit(function hypot(x1: np.Array, x2: np.Array) {
  return np.sqrt(np.square(x1).add(np.square(x2)));
});
```

Without JIT, the `hypot()` function would require four kernel dispatches: two multiplies, one add,
and one sqrt. JIT fuses these together into a single kernel that does it all at once.

All functional transformations can take typed `JsTree` of inputs and outputs. These are similar to
[JAX's pytrees](https://docs.jax.dev/en/latest/pytrees.html), and it's basically just a structure of
nested JavaScript objects and arrays. For instance:

```ts
import { grad, numpy as np } from "@jax-js/jax";

type Params = {
  foo: np.Array;
  bar: np.Array[];
};

function getSums(p: Params) {
  const fooSum = p.foo.sum();
  const barSum = p.bar.map((x) => x.sum()).reduce(np.add);
  return fooSum.add(barSum);
}

grad(getSums)({
  foo: np.array([1, 2, 3]),
  bar: [np.array([10]), np.array([11, 12])],
});
// => { foo: [1, 1, 1], bar: [[1], [1, 1]] }
```

Note that you need to use `type` alias syntax rather than `interface` to define fine-grained
`JsTree` types.

### Devices

Similar to JAX, jax-js has a concept of "devices" which are a backend that stores Arrays in memory
and determines how to execute compiled operations on them.

There are currently 4 devices in jax-js:

- `cpu`: Slow, interpreted JS, only meant for debugging.
- `wasm`: [WebAssembly](https://webassembly.org/), currently single-threaded and blocking.
- `webgpu`: [WebGPU](https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API), available on
  [supported browsers](https://caniuse.com/webgpu) (Chrome, Firefox, Safari, iOS).
- `webgl`: [WebGL2](https://developer.mozilla.org/en-US/docs/Web/API/WebGL2RenderingContext), via
  fragment shaders. This is an older graphics API that runs on almost all browsers, but it is much
  slower than WebGPU. It's offered on a best-effort basis and not as well-supported.

**We recommend `webgpu` for best performance, especially when running neural networks.** The default
device is `wasm`, but you can change this at startup time:

```ts
import { defaultDevice, init } from "@jax-js/jax";

const devices = await init(); // Starts all available backends.

if (devices.includes("webgpu")) {
  defaultDevice("webgpu");
} else {
  console.warn("WebGPU is not supported, falling back to Wasm.");
}
```

You can also place individual arrays on specific devices:

```ts
import { devicePut, numpy as np } from "@jax-js/jax";

const ar = np.array([1, 2, 3]); // Starts with device="wasm"
await devicePut(ar, "webgpu"); // Now device="webgpu"
```

### Helper libraries

There are other libraries in the `@jax-js` namespace that can work with jax-js, or be used in a
self-contained way in other projects.

- [**`@jax-js/loaders`**](packages/loaders) can load tensors from various formats like Safetensors,
  includes a fast and compliant implementation of BPE, and caches HTTP requests for large assets
  like model weights in OPFS.
- [**`@jax-js/onnx`**](packages/onnx) is a model loader from the [ONNX](https://onnx.ai/) format
  into native jax-js functions.
- [**`@jax-js/optax`**](packages/optax) provides implementations of optimizers like Adam and SGD.

### Performance

The WebGPU runtime includes an ML compiler with tile-aware optimizations, tuned for indiidual
browsers. Also, this library uniquely has the `jit()` feature that fuses operations together and
records an execution graph. jax-js achieves **over 7000 GFLOP/s** for matrix multiplication on an
Apple M4 Max chip ([try it](https://jax-js.com/bench/matmul)).

For that example, it's significantly faster than both
[TensorFlow.js](https://github.com/tensorflow/tfjs) and
[ONNX Runtime Web](https://www.npmjs.com/package/onnxruntime-web), which both use handwritten
libraries of custom kernels.

It's still early though. There's a lot of low-hanging fruit to continue optimizing the library, as
well as unique optimizations such as FlashAttention variants.

### API Reference

That's all for this short tutorial. Please see the generated
[API reference](https://jax-js.com/docs) for detailed documentation.

## Development

_The following technical details are for contributing to jax-js and modifying its internals._

This repository is managed by [`pnpm`](https://pnpm.io/). You can compile and build all packages in
watch mode with:

```bash
pnpm install
pnpm run build:watch
```

The `pnpm install` command automatically sets up Git hooks via
[Husky](https://typicode.github.io/husky/). Pre-commit hooks will run ESLint and Prettier on staged
files to ensure code quality.

You can also run linting and formatting manually:

```bash
pnpm lint          # Run ESLint
pnpm format        # Format all files with Prettier
pnpm format:check  # Check formatting without writing
pnpm check         # Run TypeScript type checking
```

Then you can run tests in a headless browser using [Vitest](https://vitest.dev/).

```bash
pnpm exec playwright install
pnpm test
```

We are currently on an older version of Playwright that supports using WebGPU in headless mode;
newer versions skip the WebGPU tests.

To start a Vite dev server running the website, demos and REPL:

```bash
pnpm -C website dev
```

## Future work / help wanted

Contributions are welcomed! Some fruitful areas to look into:

- Adding support for more JAX functions and operations, see [compatibility table](./FEATURES.md).
- Improving performance of the WebGPU and Wasm runtimes, generating better kernels, and using SIMD
  and multithreading. (Even single-threaded Wasm could be ~20x faster.)
- Adding support for `jax.profiling`, in particular the start and end trace functions. We should be
  able to generate `traceEvents` from backends (especially on GPU, with precise timestamp queries)
  to help with model performance debugging.
- Helping the JIT compiler to fuse operations in more cases, like `tanh` branches.
- Making a fast transformer inference engine, comparing against onnxruntime-web.

You may join our [Discord server](https://discord.gg/BW6YsCd4Tf) and chat with the community.
