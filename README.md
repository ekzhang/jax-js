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
import { numpy as np, setBackend } from "@jax-js/jax";

// Change the default backend to GPU.
setBackend("webgpu");

const x = np.ones([4096, 4096]);
const y = np.dot(x.ref, x); // JIT-compiled into a matrix multiplication kernel
```

## Development

Under construction.

```bash
npm install
npm run build:watch
npm test
```

## Next on Eric's mind

- Stuff to do for mandelbrot: arange, linspace, maybe comparisons?
- How many threads to create per workgroup, depends on hardware
  - Need to break up kernel dispatches if workgroup count exceeds 65536
- Think about two-stage `cumsum()`
- Disposal by enforcing a `.ref` getter? This isn't included in console.log, but it's included in `{...spread}` syntax, hopefully not used often with arrays.
  ```js
  let f = (x) => x.ref.mul(x.ref).mul(x);
  let df = grad(f); // d/dx (x^3) = 3x^2
  expect(df(4)).toBeAllclose(48);
  expect(df(5)).toBeAllclose(75);
  ```
- Frontend transformations need to match backend type for pureArray() and zeros() calls
- Need to break up operations if jit stitches more than 16 inputs

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
  - [ ] Garbage collection for JIT programs, maybe needs to be moved off-device
  - [ ] Memory scheduling, buffer allocation (can be tricky)
- [ ] Demos: Navier-Stokes, neural networks, statistics
- [ ] Wasm backend (needs malloc)
  - [ ] SIMD support for Wasm backend
- [ ] Device switching with `.to()` between webgpu/cpu/wasm
- [ ] numpy/jax API compatibility table
- [ ] Import tfjs models
