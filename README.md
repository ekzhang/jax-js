# jax-js: JAX in pure JavaScript

[Website](https://www.ekzhang.com/jax-js/)

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
you can switch to GPU for maximum performance.

```js
import { numpy as np, setDevice } from "@jax-js/jax";

// Change the default backend to GPU.
setDevice("webgpu");

const x = np.ones([4096, 4096]);
const y = np.dot(x.ref, x); // JIT-compiled into a matrix multiplication kernel
```

## Development

Under construction.

```bash
pnpm install
pnpm run build:watch

# Run tests
pnpm exec playwright install
pnpm test
```

## Next on Eric's mind

- Fix jit-of-grad returning very incorrect result
- Probably add static_argnums to jit() so that clip and some nn functions have jit added
- Improve perf of MNIST neural network
  - Optimize conv2d further (maybe blocks -> local dims?)
  - Add fused reductions to JIT
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
  - [ ] `dispose()` for saved "const" tracers in Jaxprs
  - [ ] Garbage collection for JIT programs
  - [ ] Memory scheduling, buffer allocation (can be tricky)
- [ ] Demos: Navier-Stokes, neural networks, statistics
- [x] Features for neural networks
  - [x] Convolution
  - [x] Random and initializers
  - [x] Optimizers (optax package?)
- [x] Wasm backend (needs malloc)
  - [ ] Better memory allocation that frees buffers
  - [ ] SIMD support for Wasm backend
  - [ ] Async / multithreading Wasm support
- [ ] Device switching with `.to()` between webgpu/cpu/wasm
- [ ] numpy/jax API compatibility table
- [ ] Import tfjs models
