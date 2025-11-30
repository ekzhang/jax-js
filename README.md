# jax-js: JAX in pure JavaScript

[Website](https://www.ekzhang.com/jax-js/) | [API Reference](https://www.ekzhang.com/jax-js/docs/)

This is a machine learning framework for the browser. It aims to bring JAX-style, high-performance
CPU and GPU kernels to JavaScript, so you can run numerical applications on the web.

```bash
npm i @jax-js/jax
```

Under the hood, it translates array operations into a compiler representation, then synthesizes
kernels in WebAssembly and WebGPU.

## Quickstart

You can use `jax-js` as an array API, just like NumPy.

```js
import { numpy as np } from "@jax-js/jax";

// Array operations, compatible with NumPy.
const x = np.array([1, 2, 3]);
const y = x.mul(4); // [4, 8, 12]
```

It also lets you take derivatives like in JAX.

```js
import { grad, numpy as np } from "@jax-js/jax";

// Calculate derivatives with reverse-mode AD.
const norm = (a) => a.ref.mul(a).sum();

const x = np.array([1, 2, 3]);
const xnorm = norm(x.ref); // 1^2 + 2^2 + 3^2 = 14
const xgrad = grad(norm)(x); // [2, 4, 6]
```

The default backend runs on CPU, but on [supported browsers](https://caniuse.com/webgpu),
you can switch to GPU for better performance.

```js
import { defaultDevice, numpy as np } from "@jax-js/jax";

// Change the default backend to GPU.
defaultDevice("webgpu");

const x = np.ones([4096, 4096]);
const y = np.dot(x.ref, x); // JIT-compiled into a matrix multiplication kernel
```

Most common JAX APIs are supported. See the [compatibility table](./FEATURES.md) for a full
breakdown of what features are available.

## Development

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

_We are currently on an older version of Playwright that supports using WebGPU in headless mode;
newer versions seem to skip the WebGPU tests._

To start a Vite dev server running the website, demos and REPL:

```bash
pnpm -C website dev
```

## Next on Eric's mind

- Finish CLIP inference demo and associated features (depthwise convolution, vmap of gather, etc.)
- Performance
  - Improve perf of MNIST neural network
    - Optimize conv2d further (maybe blocks -> local dims?)
    - Add fused epilogue to JIT
    - Reduce kernel overhead of constants / inline expressions
  - Investigate why jax-js Matmul is 2x slower on Safari TP than unroll kernel
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
  - [ ] High-level ops have automatic type promotion
  - [ ] Weak types - [ref](https://docs.jax.dev/en/latest/type_promotion.html#weak-types)
  - [ ] Committed devices - [ref](https://docs.jax.dev/en/latest/sharded-computation.html#sharded-data-placement)
  - [ ] Device switching with `device_put()` between webgpu/cpu/wasm
- [x] numpy/jax API compatibility table

## Future work / help wanted

Contributions are welcomed in the following areas:

- Adding support for more JAX functions and operations, see [compatibility table](./FEATURES.md).
- Improving performance of the WebGPU and Wasm runtimes, generating better kernels, using SIMD and multithreading.
- Adding WebGL runtime for older browsers that don't support WebGPU.
- Making a fast transformer inference engine, comparing against onnxruntime-web.
- Ergonomics and API improvements.
