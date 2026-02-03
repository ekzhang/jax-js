# GitHub Copilot instructions for jax-js

These notes help AI coding agents be immediately productive and document the scan feature
development for PR review. The document has two parts:

1. **Repository Overview** — General jax-js knowledge for any development work
2. **Scan Feature Development** — Detailed tracking of the `lax.scan` implementation for PR review

---

# Part 1: Repository Overview

## Architecture overview

- **Core library** (`@jax-js/jax`, root `src/`): array API, autodiff (`grad`, `jvp`, `vjp`), JIT
  compilation, device placement.
  - Frontend modules in `src/frontend/`: `array.ts` (Array class), `jit.ts` (kernel fusion),
    `jvp.ts`/`linearize.ts` (forward/reverse AD), `vmap.ts` (vectorization), `convolution.ts`.
  - Library namespaces in `src/library/`: `numpy.ts`, `lax.ts`, `nn.ts`, `random.ts`,
    `scipy-special.ts`, `numpy-linalg.ts`, `numpy-fft.ts`.
- **Backends** (`src/backend/`): `cpu.ts` (debug only), `wasm.ts` + `wasm/`, `webgl.ts` + `webgl/`,
  `webgpu.ts` + `webgpu/` (ML compiler & shader codegen).
- **Aux packages**: `packages/loaders` (safetensors, OPFS cache, BPE tokenizers), `packages/onnx`
  (ONNX model → native ops), `packages/optax` (optimizers).
- **Website & demos**: `website/` — live examples that double as integration tests.

## Developer workflows

```bash
pnpm install                       # requires pnpm ≥ 10
pnpm run build                     # tsdown → dist/*.js, dist/*.d.ts
pnpm run build:watch               # watch mode
pnpm exec playwright install       # one-time: install Chromium for WebGPU tests
pnpm test                          # Vitest + Playwright (browser + node)
pnpm run test:all                  # Vitest + Deno WebGPU (auto-fallback)
pnpm run test:deno                 # Deno WebGPU tests only (headless GPU)
pnpm test test/conv.test.ts        # single file
pnpm run check                     # tsc type-check
pnpm run lint && pnpm run format   # ESLint + Prettier
pnpm -C website dev                # local dev server for demos
```

### Pre-commit CI checks

**Before any commit**, run the full CI validation:

```bash
pnpm build:routines                # Rebuild AssemblyScript routines (if src/routines/ changed)
pnpm build                         # Build the project
pnpm check                         # TypeScript type checking
pnpm lint --max-warnings 0         # ESLint with zero warnings tolerance
pnpm exec playwright install chromium-headless-shell  # (if not already installed)
pnpm test                          # Run all tests
pnpm format:check                  # Verify Prettier formatting (or `pnpm format` to fix)
```

These match the checks in `.github/workflows/ci.yaml`.

### Temporary files

Use `tmp/` in the project root for temporary/scratch files. This directory is gitignored and allows
file operations without manual approval in VS Code.

### Debug logging

**IMPORTANT:** Do NOT use environment variables like `DEBUG=1`. Use the runtime function:

```typescript
import { setDebug } from "@jax-js/jax";
setDebug(1); // Enable debug logging BEFORE any jit compilation
```

| Level | Output                                    |
| ----- | ----------------------------------------- |
| 0     | No debug output (default)                 |
| 1     | JIT compile logs, scan path selection     |
| 2     | Shader code (WGSL/WASM), detailed tracing |
| 3     | Expressions and metadata                  |
| 4     | JIT programs, tuning details              |
| 5     | Most verbose operation traces             |

### WebGPU testing on headless servers

For GPU tests on a headless server, Chrome requires specific flags:

```bash
google-chrome --headless=new --use-angle=vulkan --enable-features=Vulkan \
  --disable-vulkan-surface --enable-unsafe-webgpu --disable-software-rasterizer
```

**Alternative: Deno for headless hardware WebGPU** — Deno's WebGPU (based on wgpu-rs) works headless
without X11:

```bash
curl -fsSL https://deno.land/install.sh | sh
deno eval --unstable-webgpu 'const a = await navigator.gpu.requestAdapter(); console.log(a?.info)'
pnpm run test:deno
```

### GPU device permissions (Linux)

```bash
sudo usermod -a -G render,video $USER  # then log out/in
```

## Reference-counting & ownership (critical)

Function arguments are **consumed by default** (refcount −1). Reuse an array by accessing `.ref`
(refcount +1). The `.data()` method also **consumes the array** after reading.

```ts
// BAD: consumes x twice
function foo(x) {
  return x.add(x.mul(y));
}

// GOOD: .ref keeps x alive for the second use
function foo(x) {
  return x.ref.add(x.mul(y));
}

// .data() consumes - don't dispose after!
const result = await arr.data(); // arr is now consumed
```

Canonical examples: `test/refcount.test.ts`, `test/conv.test.ts`, `test/deno/webgpu.test.ts`.

### Memory lifecycle

1. **Array creation** — `np.array(...)` allocates a backend `Slot` with refcount = 1.
2. **`.ref` accessor** — increments the Array object's `#rc`; same underlying Slot.
3. **Function call** — passing an Array decrements `#rc` by 1 (ownership transfer).
4. **`.data()` / `.dataSync()`** — reads buffer, then calls `dispose()` internally.
5. **`.dispose()`** — decrements `#rc`; when 0, calls `backend.decRef(slot)`.
6. **Pending ops** — `PendingExecute` holds refs on Slots until `submit()`.

### Backend memory comparison

| Aspect        | Wasm (`src/backend/wasm.ts`)              | WebGPU (`src/backend/webgpu.ts`)                      |
| ------------- | ----------------------------------------- | ----------------------------------------------------- |
| Allocation    | `WasmAllocator` over `WebAssembly.Memory` | `device.createBuffer()` with `GPUBufferUsage.STORAGE` |
| Slot tracking | `Map<Slot, {ptr, size, ref}>`             | `Map<Slot, {buffer, size, ref}>`                      |
| Sync read     | Direct memory view                        | `SyncReader` with staging buffer + `mapAsync`         |
| Dispatch      | Instantiate Wasm module, call exported fn | `commandEncoder.dispatchWorkgroups()`, queue submit   |

### Autodiff and ownership

> ⚠️ **Critical difference from Python JAX:** Letting `vjpFn` go out of scope will **NOT** free GPU
> memory. You **MUST** call `.dispose()` explicitly.

```ts
const [y, vjpFn] = vjp(f, [x]);
const dx = vjpFn(dy);
vjpFn.dispose(); // free captured forward-pass intermediates

const jitF = jit((x) => np.multiply(x, np.array([2])));
const result = jitF(x);
jitF.dispose(); // free captured constants
```

## Exports & public API

All public symbols must be exported from `src/index.ts`. Key exports:

- Transformations: `jit`, `grad`, `valueAndGrad`, `jvp`, `vjp`, `vmap`, `jacfwd`, `jacrev`,
  `hessian`, `linearize`, `makeJaxpr`
- Device control: `init`, `defaultDevice`, `devicePut`, `blockUntilReady`, `devices`, `getBackend`
- Namespaces: `numpy`, `lax`, `nn`, `random`, `scipySpecial`, `tree`
- Testing utilities: `setScanPathCallback`, `ScanPath` (type)

## Extending the codebase

| Area             | Key files                                          | Notes                                                          |
| ---------------- | -------------------------------------------------- | -------------------------------------------------------------- |
| New numpy/lax op | `src/library/{numpy,lax}.ts`                       | Follow existing function signatures; add to exports if public. |
| Backend kernel   | `src/backend/webgpu/builtins.ts`, shader templates | Mirror existing patterns; test on Chromium via Playwright.     |
| Loader/tokenizer | `packages/loaders/src/`                            | See `safetensors.ts`, `tokenizers.ts`.                         |
| ONNX op          | `packages/onnx/src/ops.ts`                         | Implement lowering; wire in `index.ts`.                        |

## JIT compiler internals

The JIT system lives in `src/frontend/jit.ts` and `src/frontend/jaxpr.ts`.

**Pipeline:**

1. **Tracing** – `makeJaxpr(f)` traces a function to produce a `Jaxpr` (IR in ANF form)
2. **Simplification** – `jaxpr.flatten().simplify()` canonicalizes the graph
3. **Graph splitting** – `splitGraphDataflow()` identifies "black nodes" vs fusable ops
4. **Kernel fusion** – Consecutive elementwise ops merge into a single `Kernel`
5. **Compilation** – `jitCompile(backend, jaxpr)` emits a `JitProgram` (list of `JitStep`s)
6. **Execution** – `JitProgram.execute(slots)` runs steps, managing memory lifetime

**Key types:**

| Type                              | File         | Role                                       |
| --------------------------------- | ------------ | ------------------------------------------ |
| `Jaxpr`, `JaxprEqn`, `Var`, `Lit` | `jaxpr.ts`   | IR nodes and bindings                      |
| `JitProgram`, `JitStep`           | `jit.ts`     | Compiled program + step types              |
| `Kernel`                          | `alu.ts`     | Fused elementwise kernel descriptor        |
| `Routine`                         | `routine.ts` | Backend-specific op (sort, cholesky, etc.) |

**Adding a new primitive:**

1. Declare in `Primitive` enum (`src/frontend/core.ts`)
2. Add tracing rule in `implRules` / `jvpRules` / `transposeRules`
3. If fusable elementwise, add ALU lowering in `jit.ts`
4. If needs dedicated kernel, register in `routinePrimitives` and implement in `src/backend/*`

## Common pitfalls

- Forgetting `.ref` → double-consume → `ReferenceError` in tests
- Exporting a symbol from library but not `src/index.ts` → missing from published types
- Changing WebGPU shaders without browser tests → silent breakage
- **CPU backend GlobalView detection**: Collect both `AluOp.GlobalIndex` AND `AluOp.GlobalView`
  nodes when finding used input buffers
- **JIT pending ops before scan**: Flush pending ops before scan step execution

## Known flaky tests

- **LU JVP finite-differences** (`test/lax-linalg.test.ts`): Occasionally fails with precision
  errors at the edge of f32 machine epsilon. Not a bug — inherent to finite-difference verification.

## Commit checklist

**Before every commit**, AI agents MUST:

1. Run pre-commit CI checks (see above)
2. Update documentation when adding new features or APIs
3. Add/adjust tests exercising `.ref` and `.dispose()` for new behavior
4. Export new public symbols from `src/index.ts`
5. Update `FEATURES.md` for user-visible changes

## Documentation files

| File                              | Purpose                                    | When to update                 |
| --------------------------------- | ------------------------------------------ | ------------------------------ |
| `README.md`                       | Main project intro, tutorial               | Major features, API changes    |
| `FEATURES.md`                     | JAX/NumPy API compatibility table          | New supported functions        |
| `.github/copilot-instructions.md` | AI agent onboarding, scan feature tracking | New patterns, scan development |
| `packages/*/README.md`            | Package-specific docs                      | Package feature changes        |

## Where to start reading

- Entry & exports: `src/index.ts`
- Memory model: `test/refcount.test.ts`
- Backends: `src/backend/webgpu/`, `src/backend/wasm/`
- Demos: `website/src/routes/repl/`, `website/src/routes/mobileclip/`
- Deno WebGPU tests: `test/deno/webgpu.test.ts` — headless hardware GPU testing
- Scan tests: `test/lax-scan.test.ts` — comprehensive scan suite (~2000 lines)

---

# Part 2: Scan Feature Development

This section documents the `lax.scan` implementation for PR review. It explains design choices,
tracks test coverage, and describes the implementation architecture.

## Overview & Motivation

`lax.scan` applies a function over the leading axis of arrays, threading carry state — essential for
RNNs, Kalman filters, cumulative operations, and other sequential computations.

**Signature:**

```ts
const [finalCarry, stackedOutputs] = await lax.scan(f, initCarry, xs, options);
// f: (carry, x) => [newCarry, y]
```

**Options:**

- `length?: number` — Number of iterations (inferred from xs if not provided)
- `reverse?: boolean` — Process xs in reverse order (default: false)
- `requirePath?: ScanPath | ScanPath[]` — Enforce scan path; throws if path doesn't match

**Scan paths (`ScanPath` type):**

- `"fused"` — Loop runs in native code (`native-scan` or `batched-scan` step)
- `"fallback"` — JS loop with per-iteration dispatch (`scan` step)

Use `requirePath: "fused"` in tests to ensure native compilation doesn't regress.

**Use cases:**

- Cumulative sum/product
- RNN/LSTM forward pass
- Kalman filter (forward and backward passes)
- Any sequential state machine

**Key files:**

| File                                 | Role                                                  |
| ------------------------------------ | ----------------------------------------------------- |
| `src/library/lax-scan.ts`            | Public API                                            |
| `src/frontend/core.ts`               | `Primitive.Scan` enum + params type                   |
| `src/frontend/jaxpr.ts`              | Abstract eval rule                                    |
| `src/frontend/array.ts`              | Scan impl rule with `scanRunner` callback             |
| `src/frontend/jit.ts`                | JIT step types: `scan`, `native-scan`, `batched-scan` |
| `src/frontend/linearize.ts`          | JVP + transpose rules for autodiff                    |
| `src/frontend/vmap.ts`               | Scan vmap rule (batches independent scans)            |
| `src/backend/wasm.ts`                | Compiled-loop scan codegen                            |
| `src/backend/webgpu.ts`              | Compiled-loop scan + compiled-body scan               |
| `src/backend/webgpu/scan-wrapper.ts` | WGSL shader transformer for uniform offsets           |

---

## Feature Status by Backend

### CPU Backend

The CPU backend uses JavaScript-interpreted evaluation. It serves as the reference implementation
for correctness testing.

| Feature / Test             | Status                           | Notes                |
| -------------------------- | -------------------------------- | -------------------- |
| `scan basic`               | [✅ Pass](test/lax-scan.test.ts) |                      |
| `scan with pytree carry`   | [✅ Pass](test/lax-scan.test.ts) |                      |
| `reverse scan`             | [✅ Pass](test/lax-scan.test.ts) |                      |
| `jit + scan`               | [✅ Pass](test/lax-scan.test.ts) |                      |
| `JVP (forward-mode)`       | [✅ Pass](test/lax-scan.test.ts) |                      |
| `VJP (reverse-mode)`       | [✅ Pass](test/lax-scan.test.ts) |                      |
| `vmap`                     | [✅ Pass](test/lax-scan.test.ts) |                      |
| `vmap` > `jit(vmap(scan))` | [✅ Pass](test/lax-scan.test.ts) |                      |
| `vmap` > `vmap(jit(scan))` | [✅ Pass](test/lax-scan.test.ts) |                      |
| `scan over views`          | [✅ Pass](test/lax-scan.test.ts) | sliced/transposed xs |

### WASM Backend

The WASM backend supports **compiled-loop scan**: the entire scan loop is compiled into a
WebAssembly module, eliminating JS/WASM boundary overhead per iteration.

| Feature / Test                          | Status                           | Notes                        |
| --------------------------------------- | -------------------------------- | ---------------------------- |
| `scan basic`                            | [✅ Pass](test/lax-scan.test.ts) |                              |
| `scan with pytree carry`                | [✅ Pass](test/lax-scan.test.ts) |                              |
| `reverse scan`                          | [✅ Pass](test/lax-scan.test.ts) |                              |
| `jit + scan`                            | [✅ Pass](test/lax-scan.test.ts) |                              |
| `JVP (forward-mode)`                    | [✅ Pass](test/lax-scan.test.ts) |                              |
| `VJP (reverse-mode)`                    | [✅ Pass](test/lax-scan.test.ts) |                              |
| `vmap`                                  | [✅ Pass](test/lax-scan.test.ts) |                              |
| `vmap` > `jit(vmap(scan))`              | [✅ Pass](test/lax-scan.test.ts) |                              |
| `vmap` > `vmap(jit(scan))`              | [✅ Pass](test/lax-scan.test.ts) |                              |
| `scan over views`                       | [✅ Pass](test/lax-scan.test.ts) | sliced/transposed xs         |
| `compiled-loop scan`                    | [✅ Pass](test/lax-scan.test.ts) |                              |
| `compiled-loop scan` > `with constants` | [✅ Pass](test/lax-scan.test.ts) |                              |
| Compiled-loop `reverse=true`            | ✅ Pass                          | all variants support reverse |
| `compiled-body scan`                    | [✅ Pass](test/lax-scan.test.ts) |                              |
| `routine in scan body`                  | [✅ Pass](test/lax-scan.test.ts) | uses native scan via imports |
| `grad` through `scan` with routines     | [✅ Pass](test/lax-scan.test.ts) | works via native path        |

**Performance benchmarks:**

- Compiled-body (JS loop, Kalman filter): ~308 iter/sec
- Compiled-loop (general, Kalman filter): ~1.5M iter/sec

**Scan vs jit(loop) overhead:**

| Matrix Size | Overhead | Notes                                 |
| ----------- | -------- | ------------------------------------- |
| 16×16       | **-73%** | Scan FASTER (fewer JS→WASM crossings) |
| 32×32       | **-21%** | Scan FASTER                           |
| 64×64       | +28%     | Loop faster                           |
| 128×128     | +51%     | Loop faster                           |

Crossover point: ~48×48 matrices.

### WebGPU Backend

The WebGPU backend keeps data on GPU between iterations. Supports **compiled-loop scan** for
elementwise kernels and **multi-kernel scan** for bodies with multiple independent kernels.

| Feature / Test                          | Status                           | Notes                                    |
| --------------------------------------- | -------------------------------- | ---------------------------------------- |
| `scan basic`                            | [✅ Pass](test/lax-scan.test.ts) | uses compiled-loop on WebGPU             |
| `scan with pytree carry`                | [✅ Pass](test/lax-scan.test.ts) |                                          |
| `reverse scan`                          | [✅ Pass](test/lax-scan.test.ts) | uses compiled-loop with dataIdx          |
| `jit + scan`                            | [✅ Pass](test/lax-scan.test.ts) |                                          |
| `JVP (forward-mode)`                    | [✅ Pass](test/lax-scan.test.ts) |                                          |
| `VJP (reverse-mode)`                    | [✅ Pass](test/lax-scan.test.ts) |                                          |
| `vmap`                                  | [✅ Pass](test/lax-scan.test.ts) |                                          |
| `vmap` > `jit(vmap(scan))`              | [✅ Pass](test/lax-scan.test.ts) |                                          |
| `vmap` > `vmap(jit(scan))`              | [✅ Pass](test/lax-scan.test.ts) |                                          |
| `scan over views`                       | [✅ Pass](test/lax-scan.test.ts) | sliced/transposed xs                     |
| `compiled-loop scan`                    | [✅ Pass](test/lax-scan.test.ts) | kernel gids reindexed to scan layout     |
| `compiled-loop scan` > `with reduction` | [✅ Pass](test/lax-scan.test.ts) | e.g., `carry += sum(x)` or matmul        |
| `compiled-loop scan` > `with reverse`   | [✅ Pass](test/lax-scan.test.ts) | uses dataIdx like WASM                   |
| `compiled-loop scan` > `with constants` | [✅ Pass](test/lax-scan.test.ts) | captured constants bound as storage      |
| `multi-kernel scan`                     | ✅ Pass                          | derives output mapping from body outputs |
| `compiled-body scan` (routine bodies)   | ⚠️ Fallback                      | uses JS loop (see Known Limitations)     |

**Note on numCarry ≠ numY:** WebGPU compiled-loop requires `numCarry === numY`. When they differ,
WebGPU falls back to JS loop. WASM's general scan handles this case.

**Tested on:** NVIDIA RTX 4070 Ti SUPER via Deno WebGPU (headless, no X11)

### WebGL Backend

The WebGL backend has **no native scan support**. All scans use the JS fallback path, which executes
the body program per iteration. This works correctly but lacks the compiled-loop optimization.

| Feature / Test | Status      | Notes                                         |
| -------------- | ----------- | --------------------------------------------- |
| `scan basic`   | ⚠️ Untested | Uses fallback path; requires browser with GPU |
| `jit + scan`   | ⚠️ Untested | Uses fallback path                            |

**Note:** WebGL tests exist in `test/lax-scan.test.ts` but are **untested in CI** because:

- Deno doesn't provide WebGL (only WebGPU)
- Playwright's headless Chromium doesn't expose WebGL in the test environment
- The dev system lacks a display for headed browser testing

The fallback `scanRunner` is backend-agnostic and tested with CPU/WASM/WebGPU, so WebGL should work
identically. To verify manually, run website demos in a WebGL-capable browser.

---

## Design Choices & Rationales

### Why compiled-loop vs compiled-body?

| Approach          | How it works                                  | When used                                        |
| ----------------- | --------------------------------------------- | ------------------------------------------------ |
| **Compiled-loop** | Entire scan loop in native code (WASM/shader) | Elementwise kernels (WASM+WebGPU), WASM routines |
| **Compiled-body** | Pre-encode dispatches, JS drives iteration    | WebGPU routines (falls back, binding issue)      |
| **Fallback**      | JS loop calling body program per iteration    | WebGPU routines, unsupported patterns            |

**Rationale:** Compiled-loop is preferred because:

1. Eliminates JS↔native boundary per iteration (~5000× speedup for WASM)
2. Enables compiler optimizations across iterations
3. Single WASM module instantiation vs N calls

Compiled-body was designed for WebGPU routines that can't be inlined into a shader, but buffer
binding issues make it currently impractical. We use fallback instead.

### Why AssemblyScript for WASM routines?

**Problem:** Hand-writing WASM bytecode is error-prone and unmaintainable.

**Solution:** AssemblyScript — a TypeScript-like language that compiles to WASM.

**Benefits:**

- Readable, maintainable source code
- TypeScript-like syntax familiar to contributors
- Small output (~1KB per routine)
- Easy to debug via source inspection

**Alternative considered:** Emscripten (C/C++) — rejected due to larger runtime overhead and less
familiar syntax for JS developers.

### Why 3 routine implementations (CPU/WASM/WebGPU)?

| Backend    | Implementation          | Location                         | Algorithm Style            |
| ---------- | ----------------------- | -------------------------------- | -------------------------- |
| **CPU**    | JavaScript (TypedArray) | `src/routine.ts`                 | Sequential (for debugging) |
| **WASM**   | AssemblyScript → WASM   | `src/routines/*.ts`              | Sequential (optimized)     |
| **WebGPU** | Hand-written WGSL       | `src/backend/webgpu/routines.ts` | Parallel (GPU-optimized)   |

**Why 3 routine implementations (CPU/WASM/WebGPU)?**

1. **CPU backend assumes WASM unavailable** — exists for environments without WebAssembly
2. **WebGPU uses different algorithms** — GPU parallelism requires fundamentally different
   approaches:
   - Sort: Bitonic sort (parallel) vs merge sort (sequential)
   - Cholesky: Column-parallel Cholesky-Crout vs row-by-row Cholesky-Banachiewicz

**Future unification options:**

- JS → AS codegen: Auto-generate AssemblyScript from JS (unifies CPU + WASM)
- Jaxpr-based routines: Express as traced functions (blocked by missing `scatter`/`dynamic_slice`)

### Why WASM imports for routines in scan?

**Problem:** How to call routines (Cholesky, Sort, etc.) from inside a compiled scan loop?

**Options considered:**

1. **Duplicate routine code in scan module** — Rejected: code bloat, maintenance burden
2. **Call out to JS** — Rejected: defeats purpose of compiled loop
3. **WASM imports** — Chosen: scan module imports pre-compiled routine functions

**Implementation:**

```typescript
// At codegen time:
cg.importFunction("routines", "cholesky_f32", [Type.i32, Type.i32, Type.i32]);

// At instantiation:
dispatchNativeScanGeneral(module, {
  routines: { cholesky_f32: routineInstance.exports.cholesky_f32 },
});
```

**Result:** Full native performance (~1.5M iter/sec) with zero code duplication.

---

## API Contract

### Scan reference contract

This contract applies to both `lax.scan()` and `jit(() => lax.scan(...))()`:

**Inputs — consumed:**

```ts
const [carry, ys] = lax.scan(f, init, xs);
// init and xs are consumed (refcount -1)
// Use .ref if you need them after: lax.scan(f, init.ref, xs.ref)
```

**Body function — borrowed references:**

```ts
const f = (carry, x) => {
  // carry and x are BORROWED — do NOT dispose them
  // Return NEW arrays for newCarry and y
  const result = np.add(carry.ref, x.ref); // .ref because we use them
  return [result, result.ref]; // .ref for dual-use (passthrough pattern)
};
```

**Outputs — caller owns:**

```ts
const [finalCarry, stackedYs] = lax.scan(f, init, xs);
// Caller owns these — dispose when done:
finalCarry.dispose();
stackedYs.dispose();
```

**Common patterns:**

| Pattern      | Code                                   | Notes                |
| ------------ | -------------------------------------- | -------------------- |
| Simple body  | `return [newCarry, y]`                 | Two distinct arrays  |
| Passthrough  | `return [newCarry.ref, newCarry]`      | Same array in both   |
| Pytree carry | `return [{ a: a.ref, b }, { out: a }]` | Mix of refs          |
| Keep inputs  | `scan(f, init.ref, xs.ref)`            | Don't consume inputs |

---

## Implementation Architecture

### Execution flow

```
lax.scan(f, init, xs, { reverse })
  → Trace f → bodyJaxpr (once)
  → Primitive.Scan(jaxpr, numCarry, numConsts, length, reverse)
  → scanRunner (JS loop or native):
      for i in 0..length:
        idx = reverse ? (length-1-i) : i
        xSlice = xs[idx] via ShapeTracker (zero-copy view)
        [carry, y] = bodyProgram.execute(carry, xSlice)
      return [carry, stack(ys)]
```

**Argument layout:**

```
Primitive args:   [...consts, ...initCarry, ...xs]
Body jaxpr input: [...consts, ...carry, ...x_slice]
```

### JIT step types

| ScanPath   | Meaning                              | Internal Step Types           |
| ---------- | ------------------------------------ | ----------------------------- |
| `fused`    | Loop runs in native code (fast path) | `native-scan`, `batched-scan` |
| `fallback` | JS loop with per-iteration dispatch  | `scan`                        |

### Terminology glossary

The documentation uses descriptive terms that map to code constructs:

| Doc Term          | Code Step Type | Backend      | Description                                  |
| ----------------- | -------------- | ------------ | -------------------------------------------- |
| **compiled-loop** | `native-scan`  | WASM, WebGPU | Entire scan loop compiled to native code     |
| **compiled-loop** | `batched-scan` | WebGPU       | Multi-kernel scan in GPU shader              |
| **compiled-body** | (fallback)     | WebGPU       | Pre-encoded dispatches, JS loop (deprecated) |
| **fallback**      | `scan`         | All          | JS loop calling body program per iteration   |

Note: `batched-scan` is named for its ability to handle multiple kernels/buffers, not for batching
iterations. Both `native-scan` and `batched-scan` implement the "fused" scan path.

### Native scan routing

The `tryPrepareNativeScan()` dispatcher routes to backend-specific implementations:

- **WebGPU kernel-only** → `tryPrepareWebGPUNativeScan()` → uses `prepareNativeScan()` or
  `prepareNativeScanMulti()`
- **WASM (kernels + routines)** → `tryPrepareWasmNativeScan()` → uses `prepareNativeScanGeneral()`

### Compiled-loop eligibility

**WASM compiled-loop:**

- All body steps are Kernels or supported Routines
- Constants allowed, reductions allowed
- Any `numCarry`/`numY` combination
- Y outputs can be: carry passthrough, xs passthrough, or internal buffer
- Supported routines: Cholesky, Sort, TriangularSolve, LU, Argsort

**WebGPU compiled-loop (single kernel):**

- Single Kernel body (elementwise with or without reductions)
- Single carry/output (`numCarry === 1`, `numY === 1`)
- Constants supported, reverse supported

**WebGPU multi-kernel scan:**

- Multiple independent Kernels
- `numCarry === numY`, total buffers ≤ 8

### WASM compiled-loop details

All WASM scan variants use `codegenNativeScanGeneral`:

1. Allocate WASM locals: iteration counter, data index, element indices
2. Allocate internal temporary buffers for intermediate results
3. Copy initCarry to carryOut (working buffer)
4. Generate outer loop (iter = 0..length):
   - Compute dataIdx (reverse-aware)
   - For each step: evaluate kernel or call imported routine
   - Copy Y outputs from source (carry passthrough, xs passthrough, or internal buffer)
   - Copy carry outputs for next iteration
5. Free internal buffers, return WebAssembly.Module

**Y output sources (`YOutputSource` type):**

| Type             | Source                              | Use case                         |
| ---------------- | ----------------------------------- | -------------------------------- |
| `passthrough`    | Copy from carry input               | `return [newC, oldC.ref]`        |
| `xs-passthrough` | Copy from xs slice at current iter  | `return [newC, x.ref]`           |
| `internal`       | Copy from internal buffer (compute) | `return [newC, someComputation]` |

**Carry output sources (`CarryOutputSource` type):**

| Type          | Source                    | Use case                                 |
| ------------- | ------------------------- | ---------------------------------------- |
| `passthrough` | Copy from carry input     | `return [oldC.ref, y]` (carry unchanged) |
| `internal`    | Copy from internal buffer | `return [computation, y]`                |

### WebGPU compiled-loop details

Shader codegen in `nativeScanShaderSource()`:

```wgsl
for (var iter: u32 = 0; iter < length; iter++) {
  var acc: f32 = 0.0;  // reduction identity
  for (var ridx: u32 = 0; ridx < reductionSize; ridx++) {
    acc = acc + /* expression using ridx */;
  }
  carry[gidx] = /* epilogue using acc */;
  ys[iter * carrySize + gidx] = carry[gidx];
}
```

Key insight: Thread `i` only reads/writes `carry[i]` and `xs[:,i]`. No `workgroupBarrier()` needed.

### WebGPU compiled-body details (broken)

For routine bodies, the approach uses pre-encoded dispatches with uniform-based offsets.

**Why uniform-based (not buffer offsets):**

- `minStorageBufferOffsetAlignment` is 256 bytes on most GPUs
- Typical strides (e.g., 120 bytes) fail alignment requirements
- Solution: Bind entire buffers, pass offset as uniform variable

The `wrapRoutineForScan` function transforms routine shaders to add offset uniforms, but incorrectly
maps bindings — see "Fix Compiled-Body Binding Detection" in Known Limitations.

### Critical implementation patterns

**Pending ops flush:** Scan execution requires flushing pending ops before scan step:

```ts
case "scan": {
  for (const p of pending) { p.prepareSync(); p.submit(); }
  pending.length = 0;
}
```

**IncRef for duplicate slots:** When body outputs contain duplicate slots (passthrough):

```ts
const seenSlots = new Set<Slot>();
const outArrays = bodyOuts.map((slot) => {
  if (seenSlots.has(slot)) backend.incRef(slot);
  else seenSlots.add(slot);
  return new Array({ source: slot, ... });
});
```

---

## Autodiff Support

### JVP (Forward-mode AD)

JVP tracing produces a doubled scan: primals + tangents flow together:

- Body becomes `(carryP, carryT, xP, xT) → (newCarryP, newCarryT, yP, yT)`
- Single scan executes both primal and tangent computation

### VJP/Grad (Reverse-mode AD)

Uses the JVP-transpose pattern for control flow:

```
grad(f)(xs)
  → vjp(f, [xs])
  → linearizeFlatUtil(f, primals)
  → partialEvalFlat (JVP'd scan with doubled args)
  → transpose(jvpResult, cotangents)
  → Scan transpose rule: iterate backward, transpose each step
```

**Key insights:**

1. Forward pass stores all intermediate carries in `allCarries`
2. Backward pass iterates from `length-1` to `0`
3. `evalJaxprTransposed` propagates "known" status for residuals

### Vmap (Vectorized Scan)

Each batch element runs an independent scan:

1. Move batch dims: consts/carry → axis 0, xs → axis 1
2. Create vmapped body jaxpr with batch at axis 0
3. Run single scan over batched arrays
4. Move ys batch from axis 1 to axis 0

**Compositions work:** `jit(vmap(scan))` and `vmap(jit(scan))`

**Transform sandwiches tested:** The `test/lax-scan.test.ts` "transform sandwiches" suite verifies
additional compositions: `jit(grad(scan))`, `grad(vmap(scan))`, `vmap(grad(scan))`, and `vmap(scan)`
vs `scan(vmap(body))` equivalence. Note: `grad(jit(scan))` is not supported — use `jit` inside the
grad-wrapped function instead.

---

## Routine System

### Implementation status

| Routine             | Status         | Source                             | Notes                                          |
| ------------------- | -------------- | ---------------------------------- | ---------------------------------------------- |
| **Cholesky**        | ✅ Implemented | `src/routines/cholesky.ts`         | f32/f64, single/batched, 740 bytes compiled    |
| **TriangularSolve** | ✅ Implemented | `src/routines/triangular-solve.ts` | Upper/lower triangular, unit/non-unit diagonal |
| **LU**              | ✅ Implemented | `src/routines/lu.ts`               | Partial pivoting, 1,308 bytes compiled         |
| **Sort**            | ✅ Implemented | `src/routines/sort.ts`             | Bottom-up merge sort, NaN-aware, 930 bytes     |
| **Argsort**         | ✅ Implemented | `src/routines/argsort.ts`          | Stable merge sort on indices, 1,215 bytes      |

**Build command:** `pnpm build:routines` compiles `src/routines/*.ts` →
`src/backend/wasm/generated/routines.ts`

### Adding a new routine (checklist)

| Step | File                        | What to add                                                |
| ---- | --------------------------- | ---------------------------------------------------------- |
| 1    | `src/routines/<name>.ts`    | AssemblyScript implementation                              |
| 2    | `src/routine.ts`            | Add to `Routines` enum                                     |
| 3    | `src/frontend/core.ts`      | Add to `routinePrimitives` map                             |
| 4    | `src/backend/wasm.ts`       | Add to `routineModuleNames`, add dispatch case             |
| 5    | `src/frontend/jit.ts`       | Add to `supportedRoutines` in `tryPrepareWasmNativeScan()` |
| 6    | `src/backend/wasm.ts`       | Add codegen case in `codegenNativeScanGeneral()`           |
| opt  | `src/routine.ts`            | Add CPU fallback in `runCpuRoutine()`                      |
| opt  | `src/frontend/jvp.ts`       | Add JVP rule if autodiff needed                            |
| opt  | `src/frontend/linearize.ts` | Add transpose rule if grad needed                          |

### AssemblyScript patterns

```typescript
export function example_f32(inPtr: usize, outPtr: usize, n: i32): void {
  const elemSize: i32 = 4;
  for (let i: i32 = 0; i < n; i++) {
    const val: f32 = load<f32>(inPtr + <usize>(i * elemSize));
    store<f32>(outPtr + <usize>(i * elemSize), val * 2.0);
  }
}
```

**Key differences from TypeScript:**

- Use `usize` for pointers, `i32` for integers, `f32`/`f64` for floats
- Use `load<T>(ptr)` and `store<T>(ptr, val)` for memory access
- Cast offsets: `<usize>(i * elemSize)`
- Use `sqrt<f32>(x)` not `Math.sqrt(x)`

### Autodiff of routines (example: Cholesky)

Routines remain **opaque primitives** — the Jaxpr just contains `cholesky a`. The internal algorithm
is NOT traced.

The JVP rule defines the derivative **in terms of other primitives**:

```typescript
[Primitive.Cholesky]([a], [da]) {
  const L = cholesky(a.ref);
  da = da.ref.add(mT(da)).mul(0.5);  // Symmetrize
  const W = triangularSolve(L.ref, da, { lower: true });
  const ST = triangularSolve(L.ref, mT(W), { lower: true });
  const dL = batchMatmulT(L.ref, triu(ST, 1).add(triu(ST)).mul(0.5));
  return [[L], [dL]];
}
```

The gradient is computed by:

1. **JVP tracing** → produces a Jaxpr containing `cholesky`, `triangular_solve`, matmul, etc.
2. **Transpose** → walks the JVP Jaxpr and transposes each primitive

The result (`grad(sum(cholesky))`) produces a **fully expanded Jaxpr** with ~30 operations. The
derivative of `cholesky` requires `triangular_solve` — both are Routines that dispatch to native
WASM. The gradient computation expresses the math in terms of these primitives.

---

## Known Limitations & Future Work

### Current limitations

| Limitation                       | Workaround                | Backend |
| -------------------------------- | ------------------------- | ------- |
| Scan body multi-output fusion    | Uses separate kernels     | All     |
| WebGPU routine bodies (compiled) | Uses fallback (JS loop)   | WebGPU  |
| `numCarry ≠ numY` on WebGPU      | Falls back to JS loop     | WebGPU  |
| `grad(scan)` memory O(N)         | None (stores all carries) | All     |

**Resolved:** WebGPU kernel-only bodies now use native scan via `tryPrepareWebGPUNativeScan()`.

### Future work

| Priority | Feature                              | Notes                                              |
| -------- | ------------------------------------ | -------------------------------------------------- |
| High     | Scan body multi-output kernel fusion | Reuse existing `jitCompile` fusion for body        |
| High     | Sqrt(N) checkpointing for grad(scan) | Reduce memory from O(N) to O(√N) with 2× recompute |
| Medium   | Fix WebGPU compiled-body binding     | `wrapRoutineForScan` arg mapping is incorrect      |
| Low      | `lax.scatter` / `dynamic_slice`      | Would enable Jaxpr-based routines                  |

### Scan Body Multi-Output Kernel Fusion

Scan bodies with multiple carry outputs (e.g., Mandelbrot with A, B, V arrays) currently compile to
**separate kernels for each output**, while regular `jit()` fuses them into a single multi-output
kernel.

**Example:** Mandelbrot body with 3 carry outputs:

- Regular `jit(f)`: **1 kernel** with 3 outputs
- Scan body: **6 kernels** (each carry + intermediates)

**Impact:** `jit(scan(vectorized_body))` is ~25-30% slower than JS-loop + `jit(body)` for
purely-elementwise multi-output bodies.

**Fix approach:** Reuse existing kernel fusion from `jitCompile()` for scan body compilation. The
`splitGraphDataflow()` algorithm already supports multi-output fusion — the issue is that scan body
compilation runs each output through separate kernel generation.

**Testing note:** This limitation should be tested on each backend separately:

- WASM: `npx tsx` or `pnpm test`
- WebGPU: Deno (`pnpm run test:deno`)

See test: `test/lax-scan.test.ts` → "KNOWN LIMITATIONS" → "Scan body multi-output fusion"

### Fix Compiled-Body Binding Detection

The `wrapRoutineForScan` function in `scan-wrapper.ts` incorrectly assumes routine shader bindings
correspond to scan body args (consts, carry, xs). In practice:

- Scan body has args: `[carry, x]`
- Routine is called with: `[x]` only
- Routine shader bindings: 1 input (x), 1 output (L)

The wrapper tries to find xs bindings starting at index `numConsts + numCarry` but the routine only
has 1 input at index 0. Fix would require tracking the mapping from body args to routine args
through the JIT compilation.

---

## Test Coverage Summary

### Test files

| File                                         | Purpose                            |
| -------------------------------------------- | ---------------------------------- |
| `test/lax-scan.test.ts`                      | Main scan test suite (~2000 lines) |
| `test/jit-scan-dlm.test.ts`                  | Kalman filter integration tests    |
| `test/deno/webgpu.test.ts`                   | Headless WebGPU tests via Deno     |
| `test/deno/batched-scan.test.ts`             | Batched scan integration           |
| `test/deno/batched-scan-integration.test.ts` | Multi-kernel WebGPU scan           |

### Deno WebGPU test guidelines

**Critical: Avoid creating multiple `GPUDevice` instances**

- **Always reuse jax-js's WebGPU device** instead of calling `navigator.gpu.requestAdapter()` +
  `adapter.requestDevice()`.
- Creating a second `GPUDevice` can destabilize Deno's WebGPU runtime and cause flakiness, memory
  leaks, or segfaults across test files.
- Use the `getJaxJsWebGPUDevice()` helper pattern (see `test/deno/batched-scan.test.ts`) to access
  the backend's device.
- **Never call `device.destroy()`** on the shared backend device — let the backend manage its
  lifecycle.

**Import from `dist/` not `src/`**

- Deno tests MUST import from `../../dist/index.js` to share backend instances across test modules.
- Mixed `src/` vs `dist/` imports create separate module graphs with separate backend instances,
  causing leak detection failures.

**Buffer cleanup**

- Track all created `GPUBuffer`s in an array: `const createdBuffers: GPUBuffer[] = []`.
- Destroy them in `finally` blocks: `for (const b of createdBuffers) b.destroy()`.
- Call `await device.queue.onSubmittedWorkDone()` before destroying buffers to ensure GPU work is
  complete.

**Module parallelism**

- Deno test runner supports parallel module execution via `--parallel` flag or `DENO_JOBS`
  environment variable.
- Current `test:deno` script uses `DENO_JOBS=1` to run test files sequentially, reducing GPU-related
  flakiness.

### Memory leak detection (Deno)

```ts
import { withLeakCheck, getSlotCount, assertNoLeaks } from "./harness.ts";

Deno.test({
  name: "my test",
  fn: withLeakCheck(async () => {
    const result = await someComputation();
    await result.data();
    jitF.dispose();
  }),
});
```

### Scan path tracking

```ts
import { setScanPathCallback, type ScanPath } from "../../dist/index.js";

function trackScanPaths() {
  const paths: { path: ScanPath; backend: string }[] = [];
  setScanPathCallback((path, backend) => paths.push({ path, backend }));
  return {
    paths,
    cleanup: () => setScanPathCallback(null),
    expectPath: (expected: ScanPath) => {
      if (!paths.some((p) => p.path === expected)) {
        throw new Error(`Expected ${expected}, got: ${paths.map((p) => p.path)}`);
      }
    },
  };
}
```

**Why this matters:** Tests named "compiled-loop scan" may silently fall back to JS loop. Use
`requirePath: "fused"` to ensure tests fail if fusion regresses.

### Test coverage by category

| Category                      | Backend | Path     | Purpose                  |
| ----------------------------- | ------- | -------- | ------------------------ |
| `scan basic`                  | CPU     | fallback | Core correctness         |
| `compiled-loop scan`          | WASM    | fused    | Verify fusion works      |
| `compiled-loop > reduction`   | WASM    | fallback | Known limitation         |
| `compiled-body scan > matmul` | WASM    | fallback | Routine bodies           |
| `KNOWN LIMITATIONS`           | WebGPU  | fallback | Verify graceful fallback |

Tests under "KNOWN LIMITATIONS" PASS when the limitation exists. If you fix a limitation, the test
will FAIL with instructions to update this documentation.
