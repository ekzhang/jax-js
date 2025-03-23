# jax-js

Under construction.

```bash
npm install
npm run build:watch
npm test
```

## Next on Eric's mind

- Reduce verbosity of, one line per operation
- Sync reading of array data
- "Array" interface to replace tfjs (`await` to get data)

## Milestones

- [x] It works!
- [x] Demos: Browser REPL / editor
- [x] First custom kernel
- [ ] Custom WebGPU backend, removing tfjs dependency
  - [ ] Low-level operations
  - [ ] Create `class Array {}` wrappers
  - [ ] Reduction operations
  - [ ] "Group" optimizations
  - [ ] "Unroll" optimizations
  - [ ] "Upcast" optimizations (i.e., `vec4<f32>`)
- [ ] We figure out the `dispose()` / refcount / linear types stuff
- [ ] Device switching with `.to()` between webgl/webgpu/cpu/wasm
- [ ] Demos: Navier-Stokes, neural networks, statistics
- [ ] `jit()` support via Jaxprs and kernel fusion
- [ ] Other dtypes like int32 and bool
- [ ] numpy/jax API compatibility table
- [ ] Import tfjs models
