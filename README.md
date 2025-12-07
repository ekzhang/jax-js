# jax-js: JAX in pure JavaScript

[Website](https://www.ekzhang.com/jax-js/) | [API Reference](https://www.ekzhang.com/jax-js/docs/)

**jax-js** is a machine learning framework for the browser. It aims to bring JAX-style,
high-performance CPU and GPU kernels to JavaScript, so you can run numerical applications on the
web.

```bash
npm i @jax-js/jax
```

Under the hood, it translates array operations into a compiler representation, then synthesizes
kernels in WebAssembly and WebGPU.

The library is written from scratch, with zero external dependencies. It maintains close API
compatibility with NumPy/JAX. Since everything runs client-side, jax-js is likely the most portable
GPU ML framework, since it runs anywhere a browser can run.

## Quickstart

You can use `jax-js` as an array API, just like NumPy.

```js
import { numpy as np } from "@jax-js/jax";

// Array operations, compatible with NumPy.
const x = np.array([1, 2, 3]);
const y = x.mul(4); // [4, 8, 12]
```

It also lets you take derivatives with `grad` like in JAX (as well as `vmap`, `jit`).

```js
import { grad, numpy as np } from "@jax-js/jax";

// Calculate derivatives with reverse-mode AD.
const norm = (a) => a.ref.mul(a).sum();

const x = np.array([1, 2, 3]);
const xnorm = norm(x.ref); // 1^2 + 2^2 + 3^2 = 14
const xgrad = grad(norm)(x); // [2, 4, 6]
```

The default backend runs on CPU, but on [supported browsers](https://caniuse.com/webgpu) including
Chrome and iOS Safari, you can switch to GPU for better performance.

```js
import { defaultDevice, init, numpy as np } from "@jax-js/jax";

// Initialize the GPU backend.
await init("webgpu");

// Change the default backend to GPU.
defaultDevice("webgpu");

const x = np.ones([4096, 4096]);
const y = np.dot(x.ref, x); // JIT-compiled into a matrix multiplication kernel
```

Most common JAX APIs are supported. See the [compatibility table](./FEATURES.md) for a full
breakdown of what features are available.

### Web usage (CDN)

If you want to use `jax-js` in vanilla JavaScript (without a bundler), just import from a module
script tag. This is the easiest way to get started on a blank HTML page.

```html
<script type="module">
  import { numpy as np } from "https://esm.sh/@jax-js/jax";
</script>
```

### Performance

We haven't spent a ton of time optimizing yet, but performance is generally pretty good. `jit` is
very helpful for fusing operations together, and it's a feature only available on the web in jax-js.
The default kernel-tuning heuristics get about 3000 GFLOP/s for matrix multiplication on an M4 Pro
chip ([try it](https://www.ekzhang.com/jax-js/bench/matmul)).

For that example, it's around the same GFLOP/s as
[TensorFlow.js](https://github.com/tensorflow/tfjs) and
[ONNX Runtime Web](https://www.npmjs.com/package/onnxruntime-web), which both use handwritten
libraries of custom kernels (versus jax-js, which generates kernels with an ML compiler).

## Examples

If you make something cool with jax-js, don't be a stranger! We can feature it here.

- [In-browser REPL](https://www.ekzhang.com/jax-js/repl)
- [Interactive MNIST training](https://www.ekzhang.com/jax-js/mnist)
- [Matmul benchmark](https://www.ekzhang.com/jax-js/bench/matmul)
- [Conv2d benchmark](https://www.ekzhang.com/jax-js/bench/conv2d)
- [Mandelbrot set](https://www.ekzhang.com/jax-js/mandelbrot)

## Development

_The following technical details are for contributing to jax-js and modifying its internals._

This repository is managed by [`pnpm`](https://pnpm.io/). You can compile and build all packages in
watch mode with:

```bash
pnpm install
pnpm run build:watch
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

Contributions are welcomed in the following areas:

- Adding support for more JAX functions and operations, see [compatibility table](./FEATURES.md).
- Improving performance of the WebGPU and Wasm runtimes, generating better kernels, and using SIMD
  and multithreading.
- Helping the JIT compiler to fuse operations in more cases.
- Adding WebGL runtime for older browsers that don't support WebGPU.
- Making a fast transformer inference engine, comparing against onnxruntime-web.
- Ergonomics and API improvements.

## Next on Eric's mind

- Finish CLIP inference demo and associated features (depthwise convolution, vmap of gather, etc.)
- Performance
  - Improve perf of MobileCLIP neural network
    - Add fused epilogue to JIT
    - Fix fusion of activation functions with branches like tanh
    - Reduce kernel overhead of constants / inline expressions
  - How many threads to create per workgroup, depends on hardware

## Milestones

- [x] It works!
- [x] Demos: Browser REPL / editor
- [x] First custom kernel
- [x] Custom WebGPU backend, removing tfjs dependency
  - [x] Low-level operations
  - [x] Create `class Array {}` wrappers
  - [x] Reduction operations
- [ ] Kernel tuning (see `tuner.ts`)
  - [x] "Upcast" optimizations (compute a tile per thread, e.g., matmul)
  - [x] "Unroll" optimizations (multiple loop iters per thread, e.g., matmul)
  - [ ] "Group" optimizations (multiple threads per value, e.g., matvec)
  - [ ] Blocks respect local dimensions
- [x] Other dtypes like int32 and bool
- [x] `jit()` support via Jaxprs and kernel fusion
- [x] We figure out the `dispose()` / refcount / linear types stuff
  - [x] `dispose()` for saved "const" tracers in Jaxprs
  - [x] Garbage collection for JIT programs
  - [x] Debug grad-grad-jit test producing a UseAfterFreeError
- [ ] Demos: Navier-Stokes, neural networks, statistics
- [x] Features for neural networks
  - [x] Convolution
  - [x] Random and initializers
  - [x] Optimizers (optax package?)
- [x] Wasm backend (needs malloc)
  - [x] Better memory allocation that frees buffers
  - [ ] SIMD support for Wasm backend
  - [ ] Async / multithreading Wasm support
- [ ] Full support of weak types and committed devices
  - [x] High-level ops have automatic type promotion
  - [x] Weak types - [ref](https://docs.jax.dev/en/latest/type_promotion.html#weak-types)
  - [ ] Committed devices -
        [ref](https://docs.jax.dev/en/latest/sharded-computation.html#sharded-data-placement)
  - [ ] Device switching with `device_put()` between webgpu/cpu/wasm
- [x] numpy/jax API compatibility table
