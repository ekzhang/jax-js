# GitHub Copilot instructions for jax-js

These notes help AI coding agents be immediately productive: what to read, how to run the project,
and non-obvious conventions.

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

**Before any commit**, run the full CI validation to catch issues early:

```bash
pnpm build                         # Build the project
pnpm check                         # TypeScript type checking
pnpm lint --max-warnings 0         # ESLint with zero warnings tolerance
pnpm exec playwright install chromium-headless-shell  # (if not already installed)
pnpm test                          # Run all tests
pnpm format:check                  # Verify Prettier formatting (or `pnpm format` to fix)
```

These match the checks in `.github/workflows/ci.yaml` (`build-test` and `format` jobs).

### Temporary files

Use `tmp/` in the project root for temporary/scratch files instead of `/tmp`. This directory is
gitignored and allows file operations without manual approval in VS Code. Create it if needed:
`mkdir -p tmp`.

### Debug logging

**IMPORTANT:** Do NOT use environment variables like `DEBUG=1` to enable debug logging. jax-js uses
a runtime function instead:

```typescript
import { setDebug } from "@jax-js/jax"; // or "../dist/index.js" for local dev

setDebug(1); // Enable debug logging BEFORE any jit compilation
```

**Debug levels:**

| Level | Output                                    |
| ----- | ----------------------------------------- |
| 0     | No debug output (default)                 |
| 1     | JIT compile logs, scan path selection     |
| 2     | Shader code (WGSL/WASM), detailed tracing |
| 3     | Expressions and metadata                  |
| 4     | JIT programs, tuning details              |
| 5     | Most verbose operation traces             |

**Example debug script:**

```typescript
import { init, jit, lax, numpy as np, setDebug, defaultDevice } from "../dist/index.js";

await init();
setDebug(2); // Must be set BEFORE jit() calls to see shader code
defaultDevice("webgpu");

const scanFn = jit((init, xs) => lax.scan((c, x) => [np.add(c, x), c], init, xs));
const [carry, ys] = await scanFn(np.array([0]), np.array([[1], [2], [3]]));
```

### WebGPU testing on headless servers

For GPU tests on a headless dev server (no display), Chrome requires specific flags:

```bash
google-chrome --headless=new --use-angle=vulkan --enable-features=Vulkan \
  --disable-vulkan-surface --enable-unsafe-webgpu --disable-software-rasterizer
```

**Prerequisites:**

- NVIDIA Vulkan ICD: `sudo apt install libnvidia-gl-<version>` (e.g., `libnvidia-gl-565`)
- Verify Vulkan: `vulkaninfo --summary` should show your GPU

**Known limitation:** Chrome's headless mode may not expose `navigator.gpu` on some systems even
with correct Vulkan setup. In that case, WebGPU tests will be skipped; cpu and wasm backends still
run. To test WebGPU, use a machine with a display or run headed Chrome via xvfb:

```bash
sudo apt-get install -y xvfb
xvfb-run -a pnpm test run
```

**Alternative: Deno for headless hardware WebGPU** Deno's WebGPU implementation (based on wgpu-rs)
works headless without X11:

```bash
# Install Deno
curl -fsSL https://deno.land/install.sh | sh

# Verify hardware GPU access (no display required)
deno eval --unstable-webgpu 'const a = await navigator.gpu.requestAdapter(); console.log(a?.info)'
# Should print: { description: "NVIDIA GeForce RTX 4070 Ti SUPER", ... }

# Run Deno-based tests
pnpm run test:deno                # runs test/deno/*.test.ts
```

**Deno WebGPU workaround:** Deno's `createComputePipelineAsync` has a WebIDL binding bug where the
`compute` field is not recognized. jax-js automatically uses the synchronous `createComputePipeline`
when running in Deno, enabling full WebGPU support on hardware GPUs headlessly.

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
// arr.dispose(); // WRONG - would double-free
```

Canonical examples: `test/refcount.test.ts`, `test/conv.test.ts`, `test/deno/webgpu.test.ts`.

### Memory lifecycle in detail

1. **Array creation** — `np.array(...)` allocates a backend `Slot` (buffer) with refcount = 1.
2. **`.ref` accessor** — increments the _Array object's_ `#rc`; same underlying Slot.
3. **Function call** — passing an Array decrements `#rc` by 1 (ownership transfer).
4. **`.data()` / `.dataSync()`** — reads the buffer, then calls `dispose()` internally (consumes the
   array).
5. **`.dispose()`** — decrements `#rc`; when it hits 0:
   - Cancels any pending `PendingExecute` not yet submitted.
   - Calls `backend.decRef(slot)` → frees GPU/Wasm memory when slot refcount = 0.
6. **Pending ops** — `PendingExecute` holds refs on input/output Slots until `submit()` to prevent
   premature free.

### Backend memory (Wasm vs WebGPU)

| Aspect        | Wasm (`src/backend/wasm.ts`)              | WebGPU (`src/backend/webgpu.ts`)                      |
| ------------- | ----------------------------------------- | ----------------------------------------------------- |
| Allocation    | `WasmAllocator` over `WebAssembly.Memory` | `device.createBuffer()` with `GPUBufferUsage.STORAGE` |
| Slot tracking | `Map<Slot, {ptr, size, ref}>`             | `Map<Slot, {buffer, size, ref}>`                      |
| Sync read     | Direct memory view                        | `SyncReader` with staging buffer + `mapAsync`         |
| Dispatch      | Instantiate Wasm module, call exported fn | `commandEncoder.dispatchWorkgroups()`, queue submit   |

- **cpu** backend (`src/backend/cpu.ts`) is JS-interpreted, no native buffers — for debugging only.
- **webgl** backend uses textures as storage; less optimized than WebGPU.

### Autodiff (grad/vjp) and ownership

- `grad(f)` internally calls `vjp(f, primals)`, producing a _linear_ backward function.
- The backward function **captures** references to intermediate arrays from the forward pass (stored
  in a `ClosedJaxpr`).
- Calling `dispose()` on the returned `OwnedFunction` (e.g., `vjpFn.dispose()`) releases those
  intermediates.
- **Pattern**: always call `.dispose()` on `jit`/`linearize`/`vjp` results when finished, or memory
  leaks on GPU.

> ⚠️ **Critical difference from Python JAX:** Unlike Python JAX where garbage collection
> automatically frees memory, letting `vjpFn` go out of scope in jax-js will **NOT** free GPU
> memory. You **MUST** call `.dispose()` explicitly or you will leak VRAM.

```ts
const [y, vjpFn] = vjp(f, [x]);
const dx = vjpFn(dy);
vjpFn.dispose(); // free captured forward-pass intermediates

// jit() also returns OwnedFunction:
const jitF = jit((x) => {
  const two = np.array([2]); // captured as constant
  return np.multiply(x, two);
});
const result = jitF(x);
jitF.dispose(); // free captured constants
```

## Exports & public API

All public symbols must be exported from `src/index.ts`. Key exports:

- Transformations: `jit`, `grad`, `valueAndGrad`, `jvp`, `vjp`, `vmap`, `jacfwd`, `jacrev`,
  `hessian`, `linearize`, `makeJaxpr`
- Device control: `init`, `defaultDevice`, `devicePut`, `blockUntilReady`, `devices`, `getBackend`
- Namespaces: `numpy`, `lax`, `nn`, `random`, `scipySpecial`, `tree`
  - Convention: users often alias `import { numpy as np }` for brevity
- Testing utilities: `setScanPathCallback`, `ScanPath` (type) — for verifying scan implementation
  paths

## Extending the codebase

| Area             | Key files                                          | Notes                                                          |
| ---------------- | -------------------------------------------------- | -------------------------------------------------------------- |
| New numpy/lax op | `src/library/{numpy,lax}.ts`                       | Follow existing function signatures; add to exports if public. |
| Backend kernel   | `src/backend/webgpu/builtins.ts`, shader templates | Mirror existing patterns; test on Chromium via Playwright.     |
| Loader/tokenizer | `packages/loaders/src/`                            | See `safetensors.ts`, `tokenizers.ts`.                         |
| ONNX op          | `packages/onnx/src/ops.ts`                         | Implement lowering; wire in `index.ts`.                        |

## Testing

- Tests run in a **headless Chromium** (Playwright) to cover WebGPU.
- Always exercise `.ref` / `.dispose()` semantics in new tests.
- Use `website/src/routes/repl/*` and `website/src/routes/mobileclip/*` as integration smoke tests.

### Memory leak detection (Deno tests)

The Deno test harness includes memory leak detection via `test/deno/harness.ts`:

```ts
import { withLeakCheck, getSlotCount, assertNoLeaks } from "./harness.ts";

// Wrap test function with automatic leak checking
Deno.test({
  name: "my test",
  fn: withLeakCheck(async () => {
    // test code - all arrays should be consumed/disposed by end
    const result = await someComputation();
    await result.data(); // consumes result
    jitF.dispose(); // release jit-captured constants
  }),
});

// Or manual checking
Deno.test("my test", async () => {
  const before = getSlotCount();
  // ... test code ...
  assertNoLeaks(before, "test name");
});
```

**Key functions:**

- `getSlotCount()` — query current backend slot count (via `backend.slotCount()`)
- `assertNoLeaks(baseline, name)` — throw if slots leaked since baseline
- `withLeakCheck(fn)` — wrapper that checks for leaks after test completion

### Scan path tracking (test verification)

Use `setScanPathCallback` to verify tests actually exercise the expected scan implementation path:

```ts
import { setScanPathCallback, type ScanPath } from "../../dist/index.js";

function trackScanPaths() {
  const paths: { path: ScanPath; backend: string }[] = [];
  setScanPathCallback((path, backend) => {
    paths.push({ path, backend });
  });
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

// Usage in test:
const tracker = trackScanPaths();
const scanFn = jit((init, xs) => lax.scan(step, init, xs));
const [carry, ys] = await scanFn(initCarry, xs);
tracker.expectPath("fused"); // Verify fused path was used
tracker.cleanup();
```

**ScanPath values:** `"fused"` (loop runs in native code) or `"fallback"` (JS loop).

**Why this matters:** Tests named "compiled-loop scan" may silently fall back to JS loop if
eligibility conditions aren't met. Use `requirePath: "fused"` to ensure tests fail if fusion
regresses.

**Test coverage:**

| Test Category                                  | Backend | Path     | Purpose                               |
| ---------------------------------------------- | ------- | -------- | ------------------------------------- |
| `scan basic` (cumsum, factorial, pytree)       | CPU     | fallback | Core correctness via JS loop          |
| `compiled-loop scan` (small, large, constants) | WASM    | fused    | Verify fusion works                   |
| `compiled-loop > reduction`                    | WASM    | fallback | Known limitation (2 steps don't fuse) |
| `compiled-body scan > matmul`                  | WASM    | fallback | Correctness for routine bodies        |
| Deno `reduction body (fallback)`               | WebGPU  | fallback | Explicit fallback correctness         |
| `KNOWN LIMITATIONS` (Cholesky, etc.)           | WebGPU  | fallback | Verify routines fall back gracefully  |

### GPU device permissions (Linux)

To run WebGPU tests on Linux, the user must have access to GPU render devices:

```bash
# Check current groups
id

# Add user to render and video groups (requires sudo)
sudo usermod -a -G render,video $USER

# Log out and back in for changes to take effect
# Or use `sg render -c 'command'` to run a single command with the new group
```

The render devices (`/dev/dri/renderD*`) are owned by the `render` group.

## Common pitfalls

- Forgetting `.ref` → double-consume → `ReferenceError` in tests.
- Exporting a symbol from a library file but not from `src/index.ts` → missing from published types.
- Changing WebGPU shaders without running browser tests → silent breakage.
- **CPU backend GlobalView detection**: When iterating over kernel expressions to find used input
  buffers, collect both `AluOp.GlobalIndex` AND `AluOp.GlobalView` nodes. GlobalView is used for
  lazy reshape/transpose operations. Missing this causes zeros to be returned instead of actual
  data.
- **JIT pending ops before scan**: The scan step reads synchronously from input slots. Any preceding
  kernels (like Transpose) create `PendingExecute` objects that must be submitted before the scan.
  Flush pending ops before scan step execution.

## Known flaky tests

- **LU JVP finite-differences** (`test/lax-linalg.test.ts`, all backends): The `lax.linalg.lu > JVP`
  test occasionally fails with numerical precision errors (e.g., expected 0.30040 vs actual
  0.30000). This is at the edge of f32 machine epsilon for finite-difference gradients, not a bug in
  the implementation. The test uses `rtol: 2e-2, atol: 2e-3` but some matrix configurations push
  values just beyond tolerance. Affects CPU, WASM, and WebGPU equally since the issue is inherent to
  f32 precision in the finite-difference verification, not the LU algorithm itself.

## JIT compiler internals

The JIT system lives in `src/frontend/jit.ts` and `src/frontend/jaxpr.ts`.

**Pipeline:**

1. **Tracing** – `makeJaxpr(f)` traces a function to produce a `Jaxpr` (JAX expression), an IR of
   primitives in ANF form.
2. **Simplification** – `jaxpr.flatten().simplify()` canonicalizes the graph.
3. **Graph splitting** – `splitGraphDataflow()` identifies "black nodes" (ops that must dispatch
   immediately, e.g. routines/reductions) vs fusable elementwise ops.
4. **Kernel fusion** – Consecutive elementwise ALU ops are merged into a single `Kernel` via lazy
   `AluExp` composition.
5. **Compilation** – `jitCompile(backend, jaxpr)` emits a `JitProgram` — a list of `JitStep`s
   (malloc, execute, free, scan).
6. **Execution** – `JitProgram.execute(slots)` runs steps on the backend, managing memory lifetime.

**Key types:**

| Type                              | File         | Role                                     |
| --------------------------------- | ------------ | ---------------------------------------- |
| `Jaxpr`, `JaxprEqn`, `Var`, `Lit` | `jaxpr.ts`   | IR nodes and bindings                    |
| `JitProgram`, `JitStep`           | `jit.ts`     | Compiled program + step types            |
| `Kernel`                          | `alu.ts`     | Fused elementwise kernel descriptor      |
| `Routine`                         | `routine.ts` | Backend-specific op (matmul, conv, etc.) |

**Adding a new primitive:**

1. Declare in `Primitive` enum (`src/frontend/core.ts`).
2. Add tracing rule in `implRules` / `jvpRules` / `transposeRules` as needed.
3. If it's fusable elementwise, add an ALU lowering in `jit.ts` switch.
4. If it needs a dedicated kernel (routine), register in `routinePrimitives` and implement in
   `src/backend/*`.

## Commit checklist

**Before every commit**, AI agents MUST:

1. **Run pre-commit CI checks** (see "Pre-commit CI checks" section above):

   ```bash
   pnpm build && pnpm check && pnpm lint --max-warnings 0 && pnpm test && pnpm format:check
   ```

   Fix any failures before committing. Use `pnpm format` to auto-fix formatting issues.

2. **Update documentation** when adding new features or APIs:
   - New public API → Add to exports in `src/index.ts`, add TSDoc comments
   - New architecture/pattern → Update `.github/copilot-instructions.md`
   - New JAX/NumPy function → Update `FEATURES.md` compatibility table
   - See "Documentation files" section for full guidelines

**Code-specific checks:**

3. Add/adjust tests exercising `.ref` and `.dispose()` for new behavior.
4. Export new public symbols from `src/index.ts`; run `pnpm check`.
5. Update `FEATURES.md` and website examples for user-visible changes.
6. For backend work, add a micro-benchmark in `bench/` where relevant.

---

# Scan Primitive (`lax.scan`)

`lax.scan` applies a function over the leading axis of arrays, threading carry state — essential for
RNNs, Kalman filters, cumulative operations, and other sequential computations.

## Overview

**Signature:**

```ts
const [finalCarry, stackedOutputs] = await lax.scan(f, initCarry, xs, options);
// f: (carry, x) => [newCarry, y]
```

**Options:**

- `length?: number` — Number of iterations (inferred from xs if not provided)
- `reverse?: boolean` — Process xs in reverse order (default: false)
- `requirePath?: ScanPath | ScanPath[]` — Enforce scan path; throws if path doesn't match

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
| `src/backend/wasm.ts`                | Compiled-loop scan codegen (single/multi/general)     |
| `src/backend/webgpu.ts`              | Compiled-loop scan + compiled-body scan for routines  |
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
WebAssembly module, eliminating JS/WASM boundary overhead per iteration. The unified **general
scan** path handles both Kernels and Routines in the same scan body, with Cholesky and Sort routines
embedded directly into the WASM module.

| Feature / Test                                  | Status                           | Notes                           |
| ----------------------------------------------- | -------------------------------- | ------------------------------- |
| `scan basic`                                    | [✅ Pass](test/lax-scan.test.ts) |                                 |
| `scan with pytree carry`                        | [✅ Pass](test/lax-scan.test.ts) |                                 |
| `reverse scan`                                  | [✅ Pass](test/lax-scan.test.ts) |                                 |
| `jit + scan`                                    | [✅ Pass](test/lax-scan.test.ts) |                                 |
| `JVP (forward-mode)`                            | [✅ Pass](test/lax-scan.test.ts) |                                 |
| `VJP (reverse-mode)`                            | [✅ Pass](test/lax-scan.test.ts) |                                 |
| `vmap`                                          | [✅ Pass](test/lax-scan.test.ts) |                                 |
| `vmap` > `jit(vmap(scan))`                      | [✅ Pass](test/lax-scan.test.ts) |                                 |
| `vmap` > `vmap(jit(scan))`                      | [✅ Pass](test/lax-scan.test.ts) |                                 |
| `scan over views`                               | [✅ Pass](test/lax-scan.test.ts) | sliced/transposed xs            |
| `compiled-loop scan`                            | [✅ Pass](test/lax-scan.test.ts) |                                 |
| `compiled-loop scan` > `with constants`         | [✅ Pass](test/lax-scan.test.ts) |                                 |
| Compiled-loop `reverse=true`                    | ✅ Pass                          | all variants support reverse    |
| `compiled-body scan`                            | [✅ Pass](test/lax-scan.test.ts) |                                 |
| `native routine scan` > `cholesky`              | [✅ Pass](test/lax-scan.test.ts) | via unified general scan path   |
| `native routine scan` > `cholesky with reverse` | [✅ Pass](test/lax-scan.test.ts) |                                 |
| `native routine scan` > `sort`                  | [✅ Pass](test/lax-scan.test.ts) | passthrough carry pattern       |
| `native routine scan` > `mixed kernel+routine`  | [✅ Pass](test/lax-scan.test.ts) | multiply → cholesky in one body |
| `grad` through `scan` with routines             | [✅ Pass](test/lax-scan.test.ts) | mixed kernel+routine body       |

**Performance benchmarks:**

- Compiled-body (JS loop, Kalman filter): ~308 iter/sec
- Compiled-loop (general, Kalman filter): ~1.5M iter/sec
- Native routine scan (Cholesky, 500 iterations, 4×4): 960× faster than fallback

**Scan vs jit(loop) overhead:**

For reduction-heavy bodies like matmul, `jit(scan)` has overhead compared to an equivalent unrolled
`jit(loop)`:

| Matrix Size | Overhead | Notes                                 |
| ----------- | -------- | ------------------------------------- |
| 16×16       | **-73%** | Scan FASTER (fewer JS→WASM crossings) |
| 32×32       | **-21%** | Scan FASTER                           |
| 64×64       | +28%     | Loop faster                           |
| 128×128     | +51%     | Loop faster                           |

**Crossover point:** ~48×48 matrices. For smaller matrices, scan wins due to single WASM call. For
larger matrices, the inner-loop overhead accumulates. An xs pointer precomputation optimization
(hoisting `dataIdx * stride` out of the reduction loop) reduced overhead from 35% to 28% for 64×64.

### WebGPU Backend

The WebGPU backend keeps data on GPU between iterations. Supports **compiled-loop scan** for
elementwise kernels (with or without reductions), **multi-kernel scan** for bodies with multiple
independent kernels (e.g., Kalman filter with 2 carries), and **compiled-body scan** for routines
(pre-encoded dispatches, one per iteration).

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
| `compiled-loop scan`                    | [✅ Pass](test/lax-scan.test.ts) | elementwise kernels, single carry/output |
| `compiled-loop scan` > `with reduction` | [✅ Pass](test/lax-scan.test.ts) | e.g., `carry += sum(x)` or matmul        |
| `compiled-loop scan` > `with reverse`   | [✅ Pass](test/lax-scan.test.ts) | uses dataIdx like WASM                   |
| `compiled-loop scan` > `with constants` | [✅ Pass](test/lax-scan.test.ts) | captured constants bound as storage      |
| `multi-kernel scan`                     | ✅ Pass                          | multiple carries, up to 8 total buffers  |
| `compiled-body scan` (routine bodies)   | ⚠️ Broken                        | see note below                           |

**Multi-kernel scan:** See
[Compiled-Loop vs Compiled-Body Eligibility](#compiled-loop-vs-compiled-body-eligibility) for
detailed requirements.

**Note on compiled-body (routine bodies):** Currently broken for all routines — see
[Future Work > Fix Compiled-Body Binding Detection](#fix-compiled-body-binding-detection) for
details.

**Note on numCarry ≠ numY:** WebGPU compiled-loop requires `numCarry === numY`. When they differ,
WebGPU falls back to JS loop. WASM's general scan handles this case.

The "compiled-body scan" tests in `lax-scan.test.ts` actually use compiled-loop (matmul fuses to
kernel). True routine bodies fall back to `fallback` (JS loop).

These limitations are verified by tests in [test/lax-scan.test.ts](test/lax-scan.test.ts) under
"KNOWN LIMITATIONS". Those tests PASS when the limitation exists. If you fix a limitation, the test
will FAIL with instructions to update this documentation.

**Tested on:** NVIDIA RTX 4070 Ti SUPER via Deno WebGPU (headless, no X11)

---

## WASM Routine Implementation Status

Native WASM implementations of routines for maximum performance. These replace the JS
`runCpuRoutine` fallback.

| Routine             | WASM Status    | Test                                          | Notes                                                                  |
| ------------------- | -------------- | --------------------------------------------- | ---------------------------------------------------------------------- |
| **Cholesky**        | ✅ Implemented | [lax-linalg.test.ts](test/lax-linalg.test.ts) | `src/backend/wasm/cholesky.ts`, ~37M matrices/sec (4×4)                |
| **TriangularSolve** | ✅ Implemented | [lax-linalg.test.ts](test/lax-linalg.test.ts) | `src/backend/wasm/triangular-solve.ts`, forward/back-substitution      |
| **LU**              | ✅ Implemented | [lax-linalg.test.ts](test/lax-linalg.test.ts) | `src/backend/wasm/lu.ts`, Gaussian elim + partial pivoting             |
| **Sort**            | ✅ Implemented | [numpy.test.ts](test/numpy.test.ts)           | `src/backend/wasm/sort.ts`, bottom-up merge sort O(n log n), NaN-aware |
| **Argsort**         | ✅ Implemented | [numpy.test.ts](test/numpy.test.ts)           | `src/backend/wasm/argsort.ts`, stable merge sort on indices            |

**Implementation pattern:** See `src/backend/wasm/cholesky.ts` for reference. Each routine:

1. Defines a `wasm_<routine>(cg, ft)` function that generates WASM bytecode using wasmblr
   - `cg` is the CodeGenerator, `ft` is the float type (`cg.f32` or `cg.f64`)
2. Wraps in `create<Routine>Module()` with batching loop
3. Caches module/instance in `WasmBackend` for reuse
4. Dispatches via `#dispatch<Routine>Wasm()` method

---

## Scan Reference Contract

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

### Execution Flow

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

### JIT Step Types

The JIT compiler emits different step types internally, but the `ScanPath` values reported via
`setScanPathCallback` are simplified to just two values:

| ScanPath   | Meaning                              | Internal Step Types           |
| ---------- | ------------------------------------ | ----------------------------- |
| `fused`    | Loop runs in native code (fast path) | `native-scan`, `batched-scan` |
| `fallback` | JS loop with per-iteration dispatch  | `scan`                        |

**Internal details (for debugging only):**

- WASM: single/multi/general kernel scans, routine scans (e.g., Cholesky)
- WebGPU: compiled-loop (elementwise kernels), batched-scan (pre-encoded routine dispatches)

### Scan Kernel Tuning (WASM & WebGPU)

Compiled-loop scans explicitly use `tuneNullopt` for body kernels instead of backend-specific tuners
(like `tuneWebgpu`). `tuneNullopt` ensures the body is compiled as a pure, inlineable elementwise
expression safe for embedding.

### Compiled-Loop vs Compiled-Body Eligibility

**WASM compiled-loop** (any of single/multi/general):

- All body steps are Kernels (not Routines)
- Constants allowed
- Reductions allowed (matmul = Mul→Sum)
- Any `numCarry`/`numY` combination (general handles `numCarry ≠ numY`)
- **Routine support**: Cholesky and Sort routines can be embedded in the scan loop
- Passthrough carry patterns supported (when carry is returned unchanged)

**WASM general scan with routines:**

The unified general scan path supports both Kernels and Routines in the same scan body. This means
complex scans mixing elementwise ops with routines (e.g., `multiply → cholesky → matmul`) can be
compiled into a single WASM module.

- Supported routines: Cholesky, Sort
- Files: `tryPrepareNativeScanGeneral()` in `jit.ts`, `codegenNativeScanGeneral()` in `wasm.ts`
- Routine functions (`wasm_cholesky`, `wasm_merge_sort`) are embedded in the generated WASM module

**WebGPU compiled-loop (single kernel):**

- Single Kernel body (elementwise with or without reductions)
- Single carry/output (`numCarry === 1`, `numY === 1`)
- Constants supported (bound as read-only storage buffers)
- Reverse scan supported (uses `dataIdx` for iteration-dependent indexing)
- Any size (multi-workgroup dispatch)

**WebGPU multi-kernel scan:**

- Multiple independent Kernels (each writes to its own carry buffer)
- `numCarry === numY` (carry outputs match Y outputs)
- Total buffers ≤ 8: consts + initCarry + xs + carry + ys
- Supports constants, reductions, and reverse scan
- Uses conditional execution per kernel (`if (gidx < kernel_size)`)

**WebGPU compiled-body (pre-encoded dispatches):**

- Single Routine body (TriangularSolve, Cholesky, LU)
- Routine must not use uniforms (excludes Sort/Argsort)
- `numCarry === numY`

**What qualifies as Kernel vs Routine:**

- **Kernel**: Fused elementwise ops + reductions (Add, Mul, Sum, Max, Matmul via Mul→Sum)
- **Routine**: Special algorithms needing dedicated shaders (TriangularSolve, Cholesky, LU, Sort,
  Argsort)

---

## Autodiff Support

### JVP (Forward-mode AD)

`jvp(f, primals, tangents)` works for functions containing scan.

**How it works:**

- JVP tracing produces a doubled scan: primals + tangents flow together
- Body becomes `(carryP, carryT, xP, xT) → (newCarryP, newCarryT, yP, yT)`
- Single scan executes both primal and tangent computation

### VJP/Grad (Reverse-mode AD)

`grad(f)` works for functions containing scan, using the JVP-transpose pattern.

> **Note:** The JVP-transpose pattern is used specifically for control flow like `scan`. Basic ops
> (add, mul, sin, etc.) have hand-written, optimized VJP rules in `src/frontend/linearize.ts` for
> performance.

**Architecture:**

```
grad(f)(xs)
  → vjp(f, [xs])
  → linearizeFlatUtil(f, primals)
  → partialEvalFlat (JVP'd scan with doubled args)
  → transpose(jvpResult, cotangents)
  → Scan transpose rule: iterate backward, transpose each step
```

**Transpose rule key insights:**

1. Extract `tangentBody` from JVP body (only tangent outputs, not all 4)
2. Forward pass stores all intermediate carries in `allCarries`
3. Backward pass iterates from `length-1` to `0`, calling `evalJaxprTransposed`
4. Use `actualUndefMask` (based on `instanceof UndefPrimal`) not JVP structure mask
5. `evalJaxprTransposed` propagates "known" status through equations (primal vs tangent)
6. Intermediate variables from primal computations are computed lazily as residuals

**Transpose with complex JVP rules (e.g., Cholesky):**

When the scan body includes routines with complex JVP rules (like Cholesky which uses triangular
solves and matmuls internally), the JVP'd body jaxpr contains both primal and tangent computations.
During transpose:

- `evalJaxprTransposed` does a forward pass to identify which intermediate variables are "known"
  (computed from only primal inputs)
- Known intermediates are computed on-demand via `getOrComputePrimal()` to serve as residuals
- Unknown intermediates (depending on tangent inputs) get transposed normally

### Vmap (Vectorized Scan)

`vmap(f)` works for functions containing scan. Each batch element runs an independent scan.

**How it works:**

1. Move batch dims: consts/carry → axis 0, xs → axis 1 (axis 0 is scan length)
2. Create vmapped body jaxpr with batch at axis 0 for all inputs
3. Run single scan over batched arrays
4. Move ys batch from axis 1 to axis 0 for output

**Compositions work:**

- `jit(vmap(scan))` — JIT compile a vmapped scan
- `vmap(jit(scan))` — vmap over a JIT-compiled scan

---

## WASM Compiled-Loop Scan Details

### Single-Kernel Compiled-Loop

Eligible when body compiles to exactly one Kernel with `numCarry === numY`.

**Codegen (`codegenNativeScan`):**

1. Allocate WASM locals: iteration counter, element index
2. Generate outer loop (i = 0..length):
   - Inner loop (elem = 0..size): evaluate kernel expression, write to carry
   - Copy carry → stacked ys at offset
3. Return WebAssembly.Module

**Key types:**

- `NativeScanParams`: length, carrySizes, xsStrides, bodyKernel
- `translateExpWithScanContext()`: scan-aware expression codegen

### Multi-Kernel Compiled-Loop

When body has multiple independent Kernels (e.g., Kalman filter with 2 matmuls).

**Eligibility:**

- All steps are Kernels (not Routines)
- Each kernel writes exactly one carry output
- Number of kernels = numCarry
- `numCarry === numY`

**Codegen (`codegenNativeScanMulti`):**

- Outer loop evaluates all kernels in sequence per iteration
- Each kernel has its own inner loop over output elements

### General Compiled-Loop

When `numCarry ≠ numY` (e.g., Kalman filter: 2 carry values, 5 outputs).

**Handles:**

- Different carry and Y counts
- Data dependencies between steps (reads from internal buffers)
- Passthrough Y outputs (carry values passed through unchanged)

**Codegen (`codegenNativeScanGeneral`):**

- Allocates internal temporary buffers for intermediate results
- Tracks Y output sources: passthrough (from carry) or internal buffer
- Frees internal buffers after WASM call

**Critical: Reduction epilogue reindexing** Must reindex both `kernel.exp` AND `kernel.reduction`
when extracting kernels. Operations like `y - matmul(F, x)` have epilogues reading from multiple
inputs.

### Reverse Scan Support

All WASM compiled-loop variants support `reverse=true`:

- Added `dataIdx` local: `dataIdx = reverse ? (length-1-iter) : iter`
- Use `dataIdx` for xs/ys offset computation
- Keep `iter` for loop counter (always forward)

---

## WebGPU Compiled-Loop Scan Details

WebGPU compiled-loop scan is wired into the JIT pipeline and supports elementwise kernels with
reductions. The shader codegen lives in `nativeScanShaderSource()` in `webgpu.ts`.

### Elementwise Compiled-Loop

For elementwise bodies where each element's scan is independent:

```
Element 0: carry[0] → f(carry[0], xs[0,0]) → f(..., xs[1,0]) → ...
Element 1: carry[1] → f(carry[1], xs[0,1]) → f(..., xs[1,1]) → ...
```

**Key insight:** Thread `i` only reads/writes `carry[i]` and `xs[:,i]`. No `workgroupBarrier()`
needed.

**Reduction support:** For kernels with reductions (e.g., `carry += sum(x)`), the shader generates
an inner `ridx` loop over the reduction dimension:

```wgsl
// Generated structure for reduction scan
for (var iter: u32 = 0; iter < length; iter++) {
  var acc: f32 = 0.0;  // identity for Add
  for (var ridx: u32 = 0; ridx < reductionSize; ridx++) {
    acc = acc + /* expression using ridx */;
  }
  carry[gidx] = /* epilogue using acc */;
  ys[iter * carrySize + gidx] = carry[gidx];
}
```

**Multi-workgroup support:** The shader uses `calculateGrid(Math.ceil(kernelSize / 256))` to
dispatch multiple workgroups for arrays larger than 256 elements.

### Compiled-Body Scan (Routine Bodies)

For routine bodies (TriangularSolve, Cholesky, LU), uses pre-encoded dispatches with uniform-based
offsets.

**Why uniform-based (not buffer offsets):**

- `minStorageBufferOffsetAlignment` is 256 bytes on most GPUs
- Typical strides (e.g., 120 bytes) fail alignment requirements
- Solution: Bind entire buffers, pass offset as uniform variable

**Architecture:**

```
prepareCompiledBodyScan():
  1. Wrap shader with uniform offset variables
  2. Create uniform buffer with all iteration offsets (aligned)

dispatchCompiledBodyScan():
  1. Ping-pong buffers for carry state
  2. Bind full buffers (group 0)
  3. Dynamic uniform offsets for iterations (group 1)
  4. Encode all iterations in single command buffer
```

**Critical:** Must create explicit `GPUBindGroupLayout` with `hasDynamicOffset: true`. Layout "auto"
doesn't work for dynamic offsets.

---

## Critical Implementation Patterns

### Pending Ops Flush

Scan execution requires flushing pending ops at multiple points to ensure data is ready:

**Before scan step execution** — flush JIT-accumulated pending ops:

```ts
case "scan": {
  for (const p of pending) {
    p.prepareSync();
    p.submit();
  }
  pending.length = 0;
  // ... then run scanRunner
}
```

Without this, preceding kernels (like Transpose in `vmap(scan)`) haven't written their outputs yet.

**Before body execution** — flush xSlice pending ops:

```ts
for (const x of xSlice) {
  for (const exe of x.#pending) {
    exe.prepareSync();
    exe.submit();
  }
}
```

**After each iteration** — flush body pending ops:

```ts
for (const exe of pending) {
  exe.prepareSync();
  exe.submit();
}
```

### IncRef for Duplicate Slots

When body outputs contain duplicate slots (passthrough pattern), increment refcount to prevent
double-free:

```ts
const seenSlots = new Set<Slot>();
const outArrays = bodyOuts.map((slot) => {
  if (seenSlots.has(slot)) {
    backend.incRef(slot);  // Prevent double-free
  } else {
    seenSlots.add(slot);
  }
  return new Array({ source: slot, ... });
});
```

### Passthrough Y Output Lifecycle

When body returns carry as Y output (passthrough pattern):

```ts
const oldCarrySlots = new Set(carry.map((c) => c._realizeSource()));
for (const y of ySlice) {
  const slot = y._realizeSource();
  if (oldCarrySlots.has(slot)) backend.incRef(slot);
}
carry.forEach((c) => c.dispose());
```

---

## Future Work

| Priority | Feature                                    | Notes                                                   |
| -------- | ------------------------------------------ | ------------------------------------------------------- |
| High     | Sqrt(N) checkpointing for grad(scan)       | Reduce memory from O(N) to O(√N) with 2× recompute      |
| Medium   | Fix WebGPU compiled-body (routine) binding | `wrapRoutineForScan` incorrectly maps routine→scan args |

### Fix Compiled-Body Binding Detection

The `wrapRoutineForScan` function in `scan-wrapper.ts` incorrectly assumes routine shader bindings
correspond to scan body args (consts, carry, xs). In practice:

- Scan body has args: `[carry, x]`
- Routine is called with: `[x]` only
- Routine shader bindings: 1 input (x), 1 output (L)

The wrapper tries to find xs bindings starting at index `numConsts + numCarry` but the routine only
has 1 input at index 0. Fix would require tracking the mapping from body args to routine args
through the JIT compilation.

### More WASM Routines in Native Scan

The unified general scan path supports Routines with these requirements:

1. Supported in `tryPrepareNativeScanGeneral` (currently: Cholesky, Sort)
2. Has a WASM codegen function (`wasm_cholesky`, `wasm_merge_sort`)
3. Function embedded in `codegenNativeScanGeneral` via step handler

**How to add a routine:**

1. Add routine name check in `tryPrepareNativeScanGeneral` (jit.ts)
2. Implement `wasm_<routine>(cg, ft)` function in `wasm.ts`
3. Add case in `codegenNativeScanGeneral` step loop for the routine

**Routine eligibility:**

| Routine         | Status         | Notes                                            |
| --------------- | -------------- | ------------------------------------------------ |
| Cholesky        | ✅ Implemented | `wasm_cholesky(cg, ft)` embedded in general scan |
| Sort            | ✅ Implemented | `wasm_merge_sort(cg, ft)` + aux buffer support   |
| Argsort         | ❌ Not yet     | Would need 2 output buffers (values + indices)   |
| TriangularSolve | ❌ Not yet     | Would need 2 inputs (A constant + x variable)    |
| LU              | ❌ Not yet     | Would need 3 outputs (L, U, pivots)              |

**Key insight:** Routines are **black-box primitives** in Jaxpr. The autodiff rules are defined in
terms of other primitives (e.g., Cholesky's JVP calls TriangularSolve), so the backend
implementation is completely decoupled.

### Alternative: Expressing routines with lax.scan

Some routines could potentially be rewritten using `lax.scan` + elementwise primitives, making them
fully traceable and automatically compilable through the JIT pipeline. This would enable autodiff
without explicit JVP rules.

**Blocked by missing primitives:**

| Primitive                | JAX name                   | Purpose                                 | Status             |
| ------------------------ | -------------------------- | --------------------------------------- | ------------------ |
| **Scatter**              | `lax.scatter`              | Write at indices: `L[i,j] = value`      | ❌ Not implemented |
| **Dynamic slice**        | `lax.dynamic_slice`        | Slice with runtime bounds: `x[i:i+k]`   | ❌ Not implemented |
| **Dynamic update slice** | `lax.dynamic_update_slice` | Update at runtime: `x.at[i:i+k].set(v)` | ❌ Not implemented |

**Routine feasibility with scan:**

| Routine         | Feasibility | Notes                                             |
| --------------- | ----------- | ------------------------------------------------- |
| TriangularSolve | ⚠️ Possible | Could build result vector incrementally with scan |
| Cholesky        | ⚠️ Possible | Row-by-row construction might work                |
| LU              | ❌ Blocked  | Needs row swaps (scatter) for pivoting            |
| Sort/Argsort    | ❌ Blocked  | Needs scatter for element swaps                   |

### Autodiff of Routines (Cholesky example)

**Q: Does Cholesky get translated to a Jaxpr?**

Yes, but it remains an **opaque primitive** — the Jaxpr just contains `cholesky a`:

```
{ lambda a:float32[2,2] .
  let b:float32[2,2] = cholesky a
  in ( b ) }
```

The internal algorithm (nested loops) is NOT traced. Cholesky is a black-box primitive.

**Q: How does grad(cholesky) work then?**

The JVP rule for Cholesky ([jvp.ts#L336](src/frontend/jvp.ts#L336)) is defined **in terms of other
primitives**:

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

The result (`grad(sum(cholesky))`) produces a **fully expanded Jaxpr** with ~30 operations:

- `cholesky` (forward pass, kept for residuals)
- `triangular_solve` (called twice in the backward pass)
- `flip`, `transpose`, `mul`, `add`, `where`, `reduce`, etc.

**Key insight:** The derivative of `cholesky` requires `triangular_solve`. Both are Routines that
dispatch to native WASM. The gradient computation doesn't avoid the Routine — it just expresses the
math in terms of these primitives.

---

## Where to start reading

- Entry & exports: `src/index.ts`
- Memory model: README.md § "Reference counting", `test/refcount.test.ts`
- Backends: `src/backend/webgpu/`, `src/backend/wasm/`
- Demos: `website/src/routes/repl/`, `website/src/routes/mobileclip/`
- Deno WebGPU tests: `test/deno/webgpu.test.ts` — headless hardware GPU testing
- Scan tests: `test/lax-scan.test.ts`, `test/deno/` (compiled-body integration tests)
  - Includes: basic, pytree, reverse, jvp, vjp/grad, vmap, jit+vmap compositions, view inputs
    (sliced/transposed/reshaped xs)

---

## Documentation files

The project has several documentation files that should be kept in sync with code changes:

### User-facing documentation

| File                         | Purpose                                          | When to update                                                |
| ---------------------------- | ------------------------------------------------ | ------------------------------------------------------------- |
| `README.md`                  | Main project intro, feature comparison, tutorial | New major features, API changes, getting started improvements |
| `FEATURES.md`                | JAX/NumPy API compatibility table (~700 lines)   | When adding/changing supported functions, new ops             |
| `packages/loaders/README.md` | @jax-js/loaders package docs                     | New loader features, API changes                              |
| `packages/onnx/README.md`    | @jax-js/onnx package docs                        | New ONNX ops, usage changes                                   |
| `packages/optax/README.md`   | @jax-js/optax package docs                       | New optimizers, API changes                                   |

### Developer documentation

| File                              | Purpose                                     | When to update                                 |
| --------------------------------- | ------------------------------------------- | ---------------------------------------------- |
| `.github/copilot-instructions.md` | AI agent onboarding, architecture, patterns | New subsystems, patterns, conventions, gotchas |

### Generated documentation

- **API Reference** (`pnpm run docs`): TypeDoc generates API docs from TSDoc comments in source code
- Output goes to `docs/` directory (gitignored)
- Published to https://jax-js.com/docs/
- Config: `typedoc.ts`

### Documentation update guidelines

1. **New public API**: Add TSDoc comments in source → auto-generates API docs
2. **New JAX/NumPy function**: Update `FEATURES.md` compatibility table (🟢/🟡/🟠/🔴 status)
3. **New package feature**: Update the relevant `packages/*/README.md`
4. **New architecture/pattern**: Update `.github/copilot-instructions.md`
5. **Major feature**: Update `README.md` feature table and possibly add tutorial section
6. **Breaking change**: Update all affected docs + add migration notes to README

### FEATURES.md legend

The compatibility table uses these status markers:

- 🟢 = supported (~45%)
- 🟡 = supported with API limitations (~2%)
- 🟠 = not supported, easy to add <1 day (~35%)
- 🔴 = not supported (~18%)
- ⚪️ = not applicable, will not support (see notes)
