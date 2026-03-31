import {
  AluExp,
  AluGroup,
  AluOp,
  byteWidth,
  DType,
  isFloatDtype,
  Kernel,
} from "../alu";
import {
  Backend,
  Device,
  Executable,
  Slot,
  SlotError,
  UnsupportedOpError,
} from "../backend";
import { Routine, runCpuRoutine } from "../routine";
import { emitTrace, isTracing, traceSourceInfo } from "../tracing";
import { tuneNullopt } from "../tuner";
import {
  DEBUG,
  FpHash,
  mapSetUnion,
  rep,
  runWithCache,
  runWithCacheAsync,
} from "../utils";
import { WasmAllocator } from "./wasm/allocator";
import {
  wasm_asin,
  wasm_atan,
  wasm_cos,
  wasm_erf,
  wasm_erfc,
  wasm_exp,
  wasm_log,
  wasm_sin,
  wasm_threefry2x32,
} from "./wasm/builtins";
import {
  createWorkerPool,
  hasSharedArrayBuffer,
  WasmWorkerPool,
} from "./wasm/parallel";
import { CodeGenerator } from "./wasm/wasmblr";

/**
 * SIMD version of translateExp: emits v128 (f32x4 or i32x4) instructions instead of scalar.
 * stepping: which loop variable steps by 4 — "gidx" for pointwise, "ridx" for reduction inner loop.
 * contiguous: map from GlobalIndex nodes to whether they can use wide v128.load (true) or need 4 scalar loads (false).
 */
function translateExpSimd(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  exp: AluExp,
  ctx: Record<string, number>,
  stepping: "gidx" | "ridx",
  contiguous: Map<AluExp, boolean>,
): void {
  const references = new Map<AluExp, number>();
  const seen = new Set<AluExp>();
  const countReferences = (exp: AluExp) => {
    references.set(exp, (references.get(exp) ?? 0) + 1);
    if (!seen.has(exp)) {
      seen.add(exp);
      for (const src of exp.src) countReferences(src);
    }
  };

  const expContext = new Map<AluExp, number>();
  const gen = (exp: AluExp) => {
    if (expContext.has(exp)) return cg.local.get(expContext.get(exp)!);
    const { op, src, arg, dtype } = exp;
    const isInt = dtype === DType.Int32 || dtype === DType.Uint32 || dtype === DType.Bool;
    const isSigned = dtype === DType.Int32;

    if (op === AluOp.Add) {
      gen(src[0]);
      gen(src[1]);
      if (isInt) cg.i32x4.add();
      else cg.f32x4.add();
    } else if (op === AluOp.Sub) {
      gen(src[0]);
      gen(src[1]);
      if (isInt) cg.i32x4.sub();
      else cg.f32x4.sub();
    } else if (op === AluOp.Mul) {
      gen(src[0]);
      gen(src[1]);
      if (isInt) cg.i32x4.mul();
      else cg.f32x4.mul();
    } else if (op === AluOp.Min) {
      gen(src[0]);
      gen(src[1]);
      if (isInt) {
        if (isSigned) cg.i32x4.min_s();
        else cg.i32x4.min_u();
      } else cg.f32x4.min();
    } else if (op === AluOp.Max) {
      gen(src[0]);
      gen(src[1]);
      if (isInt) {
        if (isSigned) cg.i32x4.max_s();
        else cg.i32x4.max_u();
      } else cg.f32x4.max();
    } else if (op === AluOp.Sqrt) {
      gen(src[0]);
      cg.f32x4.sqrt();
    } else if (op === AluOp.Floor) {
      gen(src[0]);
      cg.f32x4.floor();
    } else if (op === AluOp.Ceil) {
      gen(src[0]);
      cg.f32x4.ceil();
    } else if (op === AluOp.Const) {
      if (isInt) {
        cg.i32.const(arg as number);
        cg.i32x4.splat();
      } else {
        cg.f32.const(arg as number);
        cg.f32x4.splat();
      }
    } else if (op === AluOp.Cast) {
      gen(src[0]);
      const dtype0 = src[0].dtype;
      const src0IsInt = dtype0 === DType.Int32 || dtype0 === DType.Uint32 || dtype0 === DType.Bool;
      if (isInt && !src0IsInt) {
        // f32 → i32/u32
        if (isSigned) cg.i32x4.trunc_sat_f32x4_s();
        else cg.i32x4.trunc_sat_f32x4_u();
      } else if (!isInt && src0IsInt) {
        // i32/bool → f32 (bool uses signed to match scalar path)
        if (dtype0 === DType.Int32 || dtype0 === DType.Bool)
          cg.f32x4.convert_i32x4_s();
        else cg.f32x4.convert_i32x4_u();
      }
      // i32 ↔ u32: no-op (same bit representation)
    } else if (op === AluOp.Cmplt) {
      gen(src[0]);
      gen(src[1]);
      const srcDtype = src[0].dtype;
      if (srcDtype === DType.Float32) cg.f32x4.lt();
      else if (srcDtype === DType.Int32) cg.i32x4.lt_s();
      else if (srcDtype === DType.Uint32) cg.i32x4.lt_u();
      // SIMD comparisons produce 0xFFFFFFFF per lane; normalize to 0/1.
      cg.i32.const(1);
      cg.i32x4.splat();
      cg.v128.and();
    } else if (op === AluOp.Cmpne) {
      gen(src[0]);
      gen(src[1]);
      const srcDtype = src[0].dtype;
      if (srcDtype === DType.Float32) cg.f32x4.ne();
      else cg.i32x4.ne();
      // SIMD comparisons produce 0xFFFFFFFF per lane; normalize to 0/1.
      cg.i32.const(1);
      cg.i32x4.splat();
      cg.v128.and();
    } else if (op === AluOp.Where) {
      gen(src[1]); // true value
      gen(src[2]); // false value
      // Normalize condition to all-ones/all-zeros bitmask.
      gen(src[0]);
      cg.i32.const(0);
      cg.i32x4.splat();
      cg.i32x4.ne();
      cg.v128.bitselect();
    } else if (op === AluOp.Variable || op === AluOp.Special) {
      // These are scalar context variables (gidx, ridx, acc, etc.). They only
      // appear inside GlobalIndex's index subtree, which we handle below via
      // scalar translateExp. If we reach here outside of GlobalIndex, the
      // eligibility check missed something.
      throw new Error(`translateExpSimd: unexpected ${op}(${arg})`);
    } else if (op === AluOp.GlobalIndex) {
      const [gid, len] = arg as [number, number];
      const indexSubtree = src[0];
      const isContig = contiguous.get(exp) ?? false;

      if (isContig) {
        // Wide load: evaluate index subtree once (scalar) to get starting
        // address, then v128.load 4 consecutive elements.
        // If index is out-of-bounds, clamp to len-4 to prevent WASM traps.
        translateExp(cg, funcs, indexSubtree, ctx);
        {
          const maxIdx = Math.max(len - 4, 0);
          const wideIdx = cg.local.declare(cg.i32);
          cg.local.set(wideIdx);
          cg.local.get(wideIdx); // val_true = index
          cg.i32.const(maxIdx); // val_false = maxIdx
          cg.local.get(wideIdx);
          cg.i32.const(maxIdx);
          cg.i32.lt_u(); // condition: index < maxIdx
          cg.select();
        }

        cg.i32.const(byteWidth(dtype));
        cg.i32.mul();
        cg.local.get(gid); // base pointer
        cg.i32.add();
        if (isInt) cg.i32x4.load(4);
        else cg.f32x4.load(4);
      } else {
        // Gather: evaluate index subtree 4 times with stepping_var+0,+1,+2,+3,
        // do 4 scalar loads, pack into v128.
        const steppingLocal = ctx[stepping];
        const origValue = cg.local.declare(cg.i32);
        cg.local.get(steppingLocal);
        cg.local.set(origValue);

        // Start with zeros, replace each lane
        if (isInt) {
          cg.i32.const(0);
          cg.i32x4.splat();
        } else {
          cg.f32.const(0);
          cg.f32x4.splat();
        }
        const vec = cg.local.declare(isInt ? cg.i32x4 : cg.f32x4);
        cg.local.set(vec);

        const idx = cg.local.declare(cg.i32);
        const scalarVal = cg.local.declare(isInt ? cg.i32 : cg.f32);

        for (let lane = 0; lane < 4; lane++) {
          // Set stepping var to original + lane
          cg.local.get(origValue);
          if (lane > 0) {
            cg.i32.const(lane);
            cg.i32.add();
          }
          cg.local.set(steppingLocal);

          // Evaluate index subtree to get flat element index, with OOB clamping.
          // Same bounds check as scalar translateExp's GlobalIndex handler:
          // if index >= len, use 0 instead (prevents WASM memory traps).
          translateExp(cg, funcs, indexSubtree, ctx);
          cg.local.tee(idx);
          cg.i32.const(0);
          (cg.local.get(idx), cg.i32.const(len), cg.i32.lt_u());
          cg.select();

          cg.i32.const(byteWidth(dtype));
          cg.i32.mul();
          cg.local.get(gid); // base pointer
          cg.i32.add();
          if (isInt) cg.i32.load(2);
          else cg.f32.load(2);

          // Pack into v128: replace_lane expects [v128, scalar] on stack
          cg.local.set(scalarVal);
          cg.local.get(vec);
          cg.local.get(scalarVal);
          if (isInt) cg.i32x4.replace_lane(lane);
          else cg.f32x4.replace_lane(lane);
          cg.local.set(vec);
        }

        // Restore original stepping var value
        cg.local.get(origValue);
        cg.local.set(steppingLocal);

        // Push the gathered v128 onto the stack
        cg.local.get(vec);
      }
    } else {
      throw new Error(`translateExpSimd: unsupported op ${op}`);
    }

    // CSE: if this node is used more than once, store in a local
    if ((references.get(exp) ?? 0) > 1) {
      const local = cg.local.declare(isInt ? cg.i32x4 : cg.f32x4);
      cg.local.tee(local);
      expContext.set(exp, local);
    }
  };

  countReferences(exp);
  gen(exp);
}

/**
 * Check if a flat index expression has stride 1 w.r.t. a stepping variable.
 *
 * After lowering GlobalView to GlobalIndex and simplifying, a contiguous
 * access's flat index is either just the stepping variable (`gidx` or `ridx`)
 * or the stepping variable added to terms that don't involve it (`gidx*8 + ridx`).
 *
 * Relies on the simplifier folding div-mod identities like
 * `(x/A)*A + x%A => x`, so the flat index reduces to the bare variable
 * when the access is truly stride-1.
 */
function isContiguousWrt(exp: AluExp, varName: string): boolean {
  const isVar = (e: AluExp): boolean =>
    e.op === AluOp.Special && e.arg[0] === varName;
  const containsVar = (e: AluExp): boolean =>
    isVar(e) || e.src.some(containsVar);

  if (isVar(exp)) return true;
  if (exp.op === AluOp.Add) {
    for (let i = 0; i < 2; i++) {
      if (isContiguousWrt(exp.src[i], varName) && !containsVar(exp.src[1 - i]))
        return true;
    }
  }
  return false;
}

/** Ops that have direct SIMD (f32x4) instruction variants. */
const simdF32Ops = new Set([
  AluOp.Add,
  AluOp.Sub,
  AluOp.Mul,
  AluOp.Floor,
  AluOp.Ceil,
  AluOp.Min,
  AluOp.Max,
  AluOp.Sqrt,
  AluOp.Cast,
  AluOp.Where,
  AluOp.Const,
  AluOp.GlobalIndex,
]);

/** Ops that have direct SIMD (i32x4) instruction variants. */
const simdI32Ops = new Set([
  AluOp.Add,
  AluOp.Sub,
  AluOp.Mul,
  AluOp.Min,
  AluOp.Max,
  AluOp.Cast,
  AluOp.Where,
  AluOp.Const,
  AluOp.GlobalIndex,
]);

/** Ops that produce Bool (i32x4 bitmask) in SIMD. */
const simdBoolOps = new Set([
  AluOp.Cmplt,
  AluOp.Cmpne,
  AluOp.Const,
  AluOp.GlobalIndex,
]);

/**
 * Check if a kernel is eligible for SIMD codegen.
 *
 * A kernel qualifies when:
 * - For pointwise: size >= 4
 * - For reductions: reduction.size >= 4, dtype is f32/i32/u32, and the
 *   reduction op has a SIMD variant for that dtype
 * - All nodes have a supported dtype (f32, i32, u32, bool) with SIMD variants
 */
export function isSimdEligible(tunedExp: AluExp, kernel: Kernel): boolean {
  if (kernel.reduction) {
    if (kernel.reduction.size < 4) return false;
    if (!simdSupportedOpsForDtype(kernel.reduction.dtype)?.has(kernel.reduction.op)) return false;
  } else {
    if (kernel.size < 4) return false;
  }

  const check = (exp: AluExp, visited: Set<AluExp>): boolean => {
    if (visited.has(exp)) return true;
    visited.add(exp);

    const supportedOps = simdSupportedOpsForDtype(exp.dtype);
    if (!supportedOps || !supportedOps.has(exp.op)) return false;

    // GlobalIndex: skip the index subtree. It is evaluated scalarly
    // (via translateExp), either once for contiguous wide loads or
    // four times with lane offsets for the gather fallback.
    if (exp.op === AluOp.GlobalIndex) return true;

    // Recurse into children.
    for (const child of exp.src) {
      if (!check(child, visited)) return false;
    }
    return true;
  };

  return check(tunedExp, new Set());
}

function simdSupportedOpsForDtype(dtype: DType): Set<AluOp> | null {
  if (dtype === DType.Float32) return simdF32Ops;
  if (dtype === DType.Int32 || dtype === DType.Uint32) return simdI32Ops;
  if (dtype === DType.Bool) return simdBoolOps;
  return null;
}

interface WasmBuffer {
  ptr: number;
  size: number;
  ref: number;
}

interface WasmProgram {
  module: WebAssembly.Module;
  parallel: boolean;
}

const moduleCache = new Map<string, WebAssembly.Module>();

/** Backend that compiles into WebAssembly bytecode for immediate execution. */
export class WasmBackend implements Backend {
  readonly type: Device = "wasm";
  readonly maxArgs = 64; // Arbitrary choice

  #memory: WebAssembly.Memory;
  #nextSlot: number;
  #allocator: WasmAllocator;
  #buffers: Map<Slot, WasmBuffer>;
  #workerPool: WasmWorkerPool | null;
  #pendingWork: Map<Slot, bigint> = new Map();

  constructor() {
    this.#memory = hasSharedArrayBuffer()
      ? new WebAssembly.Memory({ initial: 0, maximum: 65536, shared: true })
      : new WebAssembly.Memory({ initial: 0 });
    this.#allocator = new WasmAllocator(this.#memory);
    this.#nextSlot = 1;
    this.#buffers = new Map();
    this.#workerPool = createWorkerPool(this.#memory);
  }

  malloc(size: number, initialData?: Uint8Array): Slot {
    const ptr = this.#allocator.malloc(size);

    if (initialData) {
      if (initialData.byteLength !== size)
        throw new Error("initialData size does not match buffer size");
      new Uint8Array(this.#memory.buffer, ptr, size).set(initialData);
    }

    const slot = this.#nextSlot++;
    this.#buffers.set(slot, { ptr, size, ref: 1 });
    return slot;
  }

  incRef(slot: Slot): void {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref++;
  }

  decRef(slot: Slot): void {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref--;
    if (buffer.ref === 0) {
      this.#allocator.free(buffer.ptr);
      this.#buffers.delete(slot);
    }
  }

  async read(
    slot: Slot,
    start?: number,
    count?: number,
  ): Promise<Uint8Array<ArrayBuffer>> {
    const epoch = this.#pendingWork.get(slot);
    if (epoch) await this.#workerPool!.waitForEpoch(epoch);
    return this.#readData(slot, start, count);
  }

  readSync(
    slot: Slot,
    start?: number,
    count?: number,
  ): Uint8Array<ArrayBuffer> {
    const epoch = this.#pendingWork.get(slot);
    if (epoch && this.#workerPool!.epoch < epoch)
      throw new Error("cannot read synchronously from a slot with async work");
    return this.#readData(slot, start, count);
  }

  #readData(
    slot: Slot,
    start?: number,
    count?: number,
  ): Uint8Array<ArrayBuffer> {
    const buffer = this.#getBuffer(slot);
    if (start === undefined) start = 0;
    if (count === undefined) count = buffer.byteLength - start;
    if (buffer.buffer instanceof SharedArrayBuffer) {
      // For SharedArrayBuffer, we need to copy the data to ArrayBuffer.
      return new Uint8Array(buffer.slice(start, start + count));
    } else {
      return buffer.slice(start, start + count);
    }
  }

  async prepareKernel(kernel: Kernel): Promise<Executable<WasmProgram>> {
    const kernelHash = FpHash.hash(kernel);
    const module = await runWithCacheAsync(
      moduleCache,
      kernelHash.toString(),
      () => WebAssembly.compile(codegenWasm(kernel)),
    );
    return new Executable(kernel, {
      module,
      parallel: this.#workerPool !== null,
    });
  }

  prepareKernelSync(kernel: Kernel): Executable<WasmProgram> {
    const kernelHash = FpHash.hash(kernel);
    const module = runWithCache(
      moduleCache,
      kernelHash.toString(),
      () => new WebAssembly.Module(codegenWasm(kernel)),
    );
    return new Executable(kernel, {
      module,
      parallel: false,
    });
  }

  async prepareRoutine(routine: Routine): Promise<Executable<WasmProgram>> {
    return this.prepareRoutineSync(routine);
  }

  prepareRoutineSync(routine: Routine): Executable<WasmProgram> {
    // Currently, Wasm routines fall back to the CPU reference implementation
    // implementation. We may optimize this in the future.
    return new Executable(routine, { module: undefined!, parallel: false });
  }

  dispatch(
    exe: Executable<WasmProgram>,
    inputs: Slot[],
    outputs: Slot[],
  ): void {
    const tracing = isTracing();
    const start = tracing ? performance.now() : 0;

    if (exe.source instanceof Routine) {
      runCpuRoutine(
        exe.source,
        inputs.map((slot) => this.#getBuffer(slot)),
        outputs.map((slot) => this.#getBuffer(slot)),
      );
    } else {
      const ptrs = [...inputs, ...outputs].map(
        (slot) => this.#buffers.get(slot)!.ptr,
      );
      if (exe.data.parallel && this.#workerPool) {
        const epoch = this.#workerPool.dispatch(
          exe.data.module,
          ptrs,
          exe.source.size,
        );
        for (const slot of outputs) this.#pendingWork.set(slot, epoch);
      } else {
        if (
          inputs.some((slot) => {
            const epoch = this.#pendingWork.get(slot);
            return epoch && this.#workerPool!.epoch < epoch;
          })
        ) {
          throw new Error(
            "cannot dispatch synchronously with pending async work",
          );
        }
        const instance = new WebAssembly.Instance(exe.data.module, {
          env: { memory: this.#memory },
        });
        const func = instance.exports.kernel as (...args: number[]) => void;
        func(...ptrs, 0, exe.source.size);
      }
    }

    if (tracing) {
      const info = traceSourceInfo(exe.source);
      emitTrace("wasm", info, start, performance.now());
    }
  }

  #getBuffer(slot: Slot): Uint8Array<ArrayBuffer> {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    return new Uint8Array(this.#memory.buffer, buffer.ptr, buffer.size);
  }
}

function codegenWasm(kernel: Kernel): Uint8Array<ArrayBuffer> {
  const tune = tuneNullopt(kernel);
  const re = kernel.reduction;

  if (DEBUG >= 3) {
    console.info(`kernel.exp: ${kernel.exp}\ntune.exp: ${tune.exp}`);
  }

  // Use tune.exp (not kernel.exp) since GlobalView nodes have been rewritten
  // to GlobalIndex by this point, matching what translateExpSimd will codegen.
  const useSimd = isSimdEligible(tune.exp, kernel);

  // Analyze per-node contiguity on the simplified tree (only needed for SIMD).
  // For pointwise SIMD, gidx steps by 4, so we check stride-1 w.r.t. gidx.
  // For reduction SIMD, ridx steps by 4, so we check stride-1 w.r.t. ridx.
  const bufferContiguity = new Map<AluExp, boolean>();
  if (useSimd) {
    const steppingVar = re ? "ridx" : "gidx";
    tune.exp
      .collect((e) => e.op === AluOp.GlobalIndex)
      .forEach((gi) => {
        bufferContiguity.set(gi, isContiguousWrt(gi.src[0], steppingVar));
      });
  }

  const cg = new CodeGenerator();
  cg.memory.import("env", "memory");
  if (hasSharedArrayBuffer()) {
    cg.memory.pages(0, 65536).shared(true);
  }

  const distinctOps = mapSetUnion(
    tune.exp.distinctOps(),
    tune.epilogue?.distinctOps(),
  );
  const funcs: Record<string, number> = {};
  if (distinctOps.has(AluOp.Sin)) funcs.sin = wasm_sin(cg);
  if (distinctOps.has(AluOp.Cos)) funcs.cos = wasm_cos(cg);
  if (distinctOps.has(AluOp.Asin)) funcs.asin = wasm_asin(cg);
  if (distinctOps.has(AluOp.Atan)) funcs.atan = wasm_atan(cg);
  if (
    distinctOps.has(AluOp.Exp) ||
    distinctOps.has(AluOp.Erf) ||
    distinctOps.has(AluOp.Erfc)
  )
    funcs.exp = wasm_exp(cg);
  if (distinctOps.has(AluOp.Log)) funcs.log = wasm_log(cg);
  if (distinctOps.has(AluOp.Erf)) funcs.erf = wasm_erf(cg, funcs.exp);
  if (distinctOps.has(AluOp.Erfc)) funcs.erfc = wasm_erfc(cg, funcs.exp);
  if (distinctOps.has(AluOp.Threefry2x32))
    funcs.threefry2x32 = wasm_threefry2x32(cg);

  // Params: arg0, ..., argN-1, output, begin, end
  const paramBegin = kernel.nargs + 1;
  const paramEnd = kernel.nargs + 2;
  const kernelFunc = cg.function(rep(kernel.nargs + 3, cg.i32), [], () => {
    const gidx = cg.local.declare(cg.i32);
    cg.local.get(paramBegin);
    cg.local.set(gidx);

    if (useSimd && !re) {
      // Pointwise SIMD loop: process 4 output elements per iteration.
      // Compute simdEnd at runtime: begin + ((end - begin) & ~3)
      const simdEnd = cg.local.declare(cg.i32);
      cg.local.get(paramEnd);
      cg.local.get(paramBegin);
      cg.i32.sub();
      cg.i32.const(~3);
      cg.i32.and();
      cg.local.get(paramBegin);
      cg.i32.add();
      cg.local.set(simdEnd);

      cg.loop(cg.void);
      {
        // if (gidx >= simdEnd) break;
        cg.block(cg.void);
        cg.local.get(gidx);
        cg.local.get(simdEnd);
        cg.i32.ge_u();
        cg.br_if(0);

        // Output address for v128.store
        cg.local.get(kernel.nargs);
        cg.local.get(gidx);
        cg.i32.const(byteWidth(kernel.dtype));
        cg.i32.mul();
        cg.i32.add();

        // Evaluate expression tree in SIMD mode, pushes v128.
        translateExpSimd(
          cg,
          funcs,
          tune.exp,
          { gidx },
          "gidx",
          bufferContiguity,
        );

        // Store 4 results at once.
        cg.v128.store(4);

        // gidx += 4
        cg.local.get(gidx);
        cg.i32.const(4);
        cg.i32.add();
        cg.local.set(gidx);

        cg.br(1);
        cg.end();
      }
      cg.end();
      // gidx is now simdEnd, fall through to scalar tail.
    }

    // Scalar loop (full loop for non-SIMD, tail loop for SIMD).
    cg.loop(cg.void);
    {
      // if (gidx >= end) break;
      cg.block(cg.void);
      cg.local.get(gidx);
      cg.local.get(paramEnd);
      cg.i32.ge_u();
      cg.br_if(0);

      // Push memory index of output onto stack (will be used at end).
      cg.local.get(kernel.nargs); // output buffer is last argument
      cg.local.get(gidx);
      cg.i32.const(byteWidth(kernel.dtype));
      cg.i32.mul();
      cg.i32.add();

      if (re && useSimd) {
        // SIMD reduction: vectorize the inner ridx loop, stepping by 4.
        // Note: accumulating 4 partial results then horizontal-reducing
        // reorders operations vs. the scalar path, so floating-point
        // results may differ slightly due to reassociation.
        const simdReSize = re.size & ~3;
        const ridx = cg.local.declare(cg.i32);
        const reIsInt = re.dtype === DType.Int32 || re.dtype === DType.Uint32;
        const reIsSigned = re.dtype === DType.Int32;

        // v128 accumulator initialized with identity values.
        const vecAcc = cg.local.declare(reIsInt ? cg.i32x4 : cg.f32x4);
        if (reIsInt) {
          cg.i32.const(re.identity);
          cg.i32x4.splat();
        } else {
          cg.f32.const(re.identity);
          cg.f32x4.splat();
        }
        cg.local.set(vecAcc);

        // SIMD inner loop: ridx steps by 4.
        cg.i32.const(0);
        cg.local.set(ridx);
        cg.loop(cg.void);
        {
          cg.block(cg.void);
          cg.local.get(ridx);
          cg.i32.const(simdReSize);
          cg.i32.ge_u();
          cg.br_if(0);

          // Evaluate expression tree in SIMD mode (stepping ridx by 4).
          translateExpSimd(
            cg,
            funcs,
            tune.exp,
            { gidx, ridx },
            "ridx",
            bufferContiguity,
          );

          // vecAcc = op(vecAcc, values)
          cg.local.get(vecAcc);
          if (reIsInt) {
            if (re.op === AluOp.Add) cg.i32x4.add();
            else if (re.op === AluOp.Mul) cg.i32x4.mul();
            else if (re.op === AluOp.Min) {
              if (reIsSigned) cg.i32x4.min_s();
              else cg.i32x4.min_u();
            } else if (re.op === AluOp.Max) {
              if (reIsSigned) cg.i32x4.max_s();
              else cg.i32x4.max_u();
            } else throw new Error(`invalid SIMD reduction op: ${re.op}`);
          } else {
            if (re.op === AluOp.Add) cg.f32x4.add();
            else if (re.op === AluOp.Mul) cg.f32x4.mul();
            else if (re.op === AluOp.Min) cg.f32x4.min();
            else if (re.op === AluOp.Max) cg.f32x4.max();
            else throw new Error(`invalid SIMD reduction op: ${re.op}`);
          }
          cg.local.set(vecAcc);

          // ridx += 4
          cg.local.get(ridx);
          cg.i32.const(4);
          cg.i32.add();
          cg.local.set(ridx);

          cg.br(1);
          cg.end();
        }
        cg.end();

        // Horizontal reduce: collapse 4 lanes into 1 scalar.
        const acc = cg.local.declare(reIsInt ? cg.i32 : cg.f32);
        cg.local.get(vecAcc);
        if (reIsInt) cg.i32x4.extract_lane(0);
        else cg.f32x4.extract_lane(0);
        cg.local.set(acc);
        for (let lane = 1; lane < 4; lane++) {
          cg.local.get(vecAcc);
          if (reIsInt) cg.i32x4.extract_lane(lane);
          else cg.f32x4.extract_lane(lane);
          cg.local.get(acc);
          if (reIsInt) {
            if (re.op === AluOp.Add) cg.i32.add();
            else if (re.op === AluOp.Mul) cg.i32.mul();
            else if (re.op === AluOp.Min) {
              // Wasm has no i32.min, emulate with select.
              const tmp = cg.local.declare(cg.i32);
              cg.local.set(tmp);
              cg.local.tee(acc);
              cg.local.get(tmp);
              cg.local.get(acc);
              cg.local.get(tmp);
              if (reIsSigned) cg.i32.lt_s();
              else cg.i32.lt_u();
              cg.select();
            } else if (re.op === AluOp.Max) {
              const tmp = cg.local.declare(cg.i32);
              cg.local.set(tmp);
              cg.local.tee(acc);
              cg.local.get(tmp);
              cg.local.get(acc);
              cg.local.get(tmp);
              if (reIsSigned) cg.i32.gt_s();
              else cg.i32.gt_u();
              cg.select();
            }
          } else {
            if (re.op === AluOp.Add) cg.f32.add();
            else if (re.op === AluOp.Mul) cg.f32.mul();
            else if (re.op === AluOp.Min) cg.f32.min();
            else if (re.op === AluOp.Max) cg.f32.max();
          }
          cg.local.set(acc);
        }

        // Scalar tail: handle remaining ridx values.
        // ridx is already at simdReSize from the SIMD loop.
        cg.loop(cg.void);
        {
          cg.block(cg.void);
          cg.local.get(ridx);
          cg.i32.const(re.size);
          cg.i32.ge_u();
          cg.br_if(0);

          translateExp(cg, funcs, tune.exp, { gidx, ridx });
          cg.local.get(acc);
          if (reIsInt) {
            if (re.op === AluOp.Add) cg.i32.add();
            else if (re.op === AluOp.Mul) cg.i32.mul();
            else if (re.op === AluOp.Min) {
              const tmp = cg.local.declare(cg.i32);
              cg.local.set(tmp);
              cg.local.tee(acc);
              cg.local.get(tmp);
              cg.local.get(acc);
              cg.local.get(tmp);
              if (reIsSigned) cg.i32.lt_s();
              else cg.i32.lt_u();
              cg.select();
            } else if (re.op === AluOp.Max) {
              const tmp = cg.local.declare(cg.i32);
              cg.local.set(tmp);
              cg.local.tee(acc);
              cg.local.get(tmp);
              cg.local.get(acc);
              cg.local.get(tmp);
              if (reIsSigned) cg.i32.gt_s();
              else cg.i32.gt_u();
              cg.select();
            }
          } else {
            if (re.op === AluOp.Add) cg.f32.add();
            else if (re.op === AluOp.Mul) cg.f32.mul();
            else if (re.op === AluOp.Min) cg.f32.min();
            else if (re.op === AluOp.Max) cg.f32.max();
          }
          cg.local.set(acc);

          cg.local.get(ridx);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(ridx);

          cg.br(1);
          cg.end();
        }
        cg.end();

        // Apply epilogue with scalar accumulator.
        translateExp(cg, funcs, tune.epilogue!, { acc, gidx });
      } else if (re) {
        // Scalar reduction (non-SIMD eligible).
        const acc = cg.local.declare(dty(cg, null, kernel.exp.dtype));
        dty(cg, null, kernel.exp.dtype).const(re.identity);
        cg.local.set(acc);

        const ridx = cg.local.declare(cg.i32);
        cg.i32.const(0);
        cg.local.set(ridx);
        cg.loop(cg.void);
        {
          // if (ridx >= reduction.size) break;
          cg.block(cg.void);
          cg.local.get(ridx);
          cg.i32.const(re.size);
          cg.i32.ge_u();
          cg.br_if(0);

          // Translate tune.exp to expression and push onto stack.
          translateExp(cg, funcs, tune.exp, { gidx, ridx });

          // acc = reduction.evaluate(acc, exp)
          if (re.op === AluOp.Add) {
            cg.local.get(acc);
            if (re.dtype === DType.Bool) cg.i32.or();
            else dty(cg, re.op, re.dtype).add();
          } else if (re.op === AluOp.Mul) {
            cg.local.get(acc);
            if (re.dtype === DType.Bool) cg.i32.and();
            else dty(cg, re.op, re.dtype).mul();
          } else if (re.op === AluOp.Min || re.op === AluOp.Max) {
            if (isFloatDtype(re.dtype)) {
              cg.local.get(acc);
              if (re.op === AluOp.Min) dtyF(cg, re.op, re.dtype).min();
              else dtyF(cg, re.op, re.dtype).max();
            } else if (
              [DType.Int32, DType.Uint32, DType.Bool].includes(re.dtype)
            ) {
              // Wasm has no i32.min/max, so emulate with select.
              const local = cg.local.declare(cg.i32);
              cg.local.tee(local);
              cg.local.get(acc);
              cg.local.get(local);
              cg.local.get(acc);
              if (re.op === AluOp.Min) {
                if (re.dtype === DType.Int32) cg.i32.lt_s();
                else cg.i32.lt_u();
              } else {
                if (re.dtype === DType.Int32) cg.i32.gt_s();
                else cg.i32.gt_u();
              }
              cg.select();
            } else
              throw new Error(`invalid reduction min/max over ${re.dtype}`);
          } else throw new Error(`invalid wasm reduction op: ${re.op}`);
          cg.local.set(acc);

          // ridx++
          cg.local.get(ridx);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(ridx);

          cg.br(1); // continue ridx loop
          cg.end();
        }
        cg.end();

        translateExp(cg, funcs, tune.epilogue!, { acc, gidx });
      } else {
        // Translate tune.exp to expression and push onto stack.
        translateExp(cg, funcs, tune.exp, { gidx });
      }

      // Store value into output buffer.
      dty(cg, null, kernel.dtype).store(Math.log2(byteWidth(kernel.dtype)));

      // gidx++
      cg.local.get(gidx);
      cg.i32.const(1);
      cg.i32.add();
      cg.local.set(gidx);

      cg.br(1); // continue gidx loop
      cg.end();
    }
    cg.end();
  });
  cg.export(kernelFunc, "kernel");

  return cg.finish();
}

function translateExp(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  exp: AluExp,
  ctx: Record<string, number>,
) {
  const references = new Map<AluExp, number>();
  const seen = new Set<AluExp>();
  const countReferences = (exp: AluExp) => {
    references.set(exp, (references.get(exp) ?? 0) + 1);
    if (!seen.has(exp)) {
      seen.add(exp);
      for (const src of exp.src) countReferences(src);
    }
  };

  const expContext = new Map<AluExp, number>();
  const gen = (exp: AluExp) => {
    if (expContext.has(exp)) return cg.local.get(expContext.get(exp)!);
    const { op, src, dtype, arg } = exp;

    // Some of these cases early `return` to force-inline them (no local.set).
    if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
      gen(src[0]);
      gen(src[1]);
      if (op === AluOp.Add) {
        if (dtype === DType.Bool) cg.i32.or();
        else dty(cg, op, dtype).add();
      } else if (op === AluOp.Sub) {
        dty(cg, op, dtype).sub();
      } else if (op === AluOp.Mul) {
        if (dtype === DType.Bool) cg.i32.and();
        else dty(cg, op, dtype).mul();
      } else if (op === AluOp.Idiv) {
        if (isFloatDtype(dtype)) {
          dtyF(cg, op, dtype).div();
          dtyF(cg, op, dtype).trunc();
        } else if (dtype === DType.Uint32) cg.i32.div_u();
        else if (dtype === DType.Int32) cg.i32.div_s();
        else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.Mod) {
        if (isFloatDtype(dtype)) {
          // Emulate a % b = a - trunc(a/b)*b
          const dt = dtyF(cg, op, dtype);
          const a = cg.local.declare(dt);
          const b = cg.local.declare(dt);
          cg.local.set(b);
          cg.local.tee(a); // stack: a
          cg.local.get(a);
          cg.local.get(b);
          dt.div();
          dt.trunc(); // stack: a, trunc(a/b)
          cg.local.get(b);
          dt.mul(); // stack: a, trunc(a/b)*b
          dt.sub();
        } else if (dtype === DType.Uint32) cg.i32.rem_u();
        else if (dtype === DType.Int32) cg.i32.rem_s();
        else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.Min || op === AluOp.Max) {
        if (isFloatDtype(dtype)) {
          if (op === AluOp.Min) dtyF(cg, op, dtype).min();
          else dtyF(cg, op, dtype).max();
        } else if (
          dtype === DType.Int32 ||
          dtype === DType.Uint32 ||
          dtype === DType.Bool
        ) {
          // Wasm has no i32.min, so emulate with select.
          const a = cg.local.declare(cg.i32);
          const b = cg.local.declare(cg.i32);
          cg.local.set(b);
          cg.local.tee(a);
          cg.local.get(b);
          cg.local.get(a);
          cg.local.get(b);
          if (dtype === DType.Int32) {
            if (op === AluOp.Min) cg.i32.lt_s();
            else cg.i32.gt_s();
          } else {
            if (op === AluOp.Min) cg.i32.lt_u();
            else cg.i32.gt_u();
          }
          cg.select();
        } else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.BitCombine) {
        if (arg === "and") cg.i32.and();
        else if (arg === "or") cg.i32.or();
        else cg.i32.xor();
      } else if (op === AluOp.BitShift) {
        if (arg === "shl") cg.i32.shl();
        else cg.i32.shr_u();
      } else if (op === AluOp.Cmplt) {
        const srcDtype = src[0].dtype;
        if (isFloatDtype(srcDtype)) dtyF(cg, op, srcDtype).lt();
        else if (srcDtype === DType.Int32) cg.i32.lt_s();
        else if (srcDtype === DType.Uint32) cg.i32.lt_u();
        else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.Cmpne) dty(cg, op, src[0].dtype).ne();
      else throw new UnsupportedOpError(op, dtype, "wasm");
    } else if (AluGroup.Unary.has(op)) {
      // TODO: Our intrinsics are only implemented in f32 precision currently,
      // so we cast to f32 first for other floating-point inputs.
      const callFuncF32 = (func: number): void => {
        if (dtype !== DType.Float32) {
          if (dtype === DType.Float64) cg.f32.demote_f64();
          else throw new UnsupportedOpError(op, dtype, "wasm");
        }
        cg.call(func);
        if (dtype === DType.Float64) cg.f64.promote_f32();
      };
      if (op === AluOp.Sin) (gen(src[0]), callFuncF32(funcs.sin));
      else if (op === AluOp.Cos) (gen(src[0]), callFuncF32(funcs.cos));
      else if (op === AluOp.Asin) (gen(src[0]), callFuncF32(funcs.asin));
      else if (op === AluOp.Atan) (gen(src[0]), callFuncF32(funcs.atan));
      else if (op === AluOp.Exp) (gen(src[0]), callFuncF32(funcs.exp));
      else if (op === AluOp.Log) (gen(src[0]), callFuncF32(funcs.log));
      else if (op === AluOp.Erf) (gen(src[0]), callFuncF32(funcs.erf));
      else if (op === AluOp.Erfc) (gen(src[0]), callFuncF32(funcs.erfc));
      else if (op === AluOp.Sqrt) (gen(src[0]), dtyF(cg, op, dtype).sqrt());
      else if (op === AluOp.Reciprocal) {
        const dt = dtyF(cg, op, dtype);
        (dt.const(1), gen(src[0]), dt.div());
      } else if (op === AluOp.Floor) (gen(src[0]), dtyF(cg, op, dtype).floor());
      else if (op === AluOp.Ceil) (gen(src[0]), dtyF(cg, op, dtype).ceil());
      else if (op === AluOp.Cast) {
        gen(src[0]);
        const dtype0 = src[0].dtype;
        const i32repr =
          dtype0 === DType.Int32 ||
          dtype0 === DType.Uint32 ||
          dtype0 === DType.Bool;
        if (dtype === DType.Int32) {
          if (dtype0 === DType.Float32) cg.i32.trunc_sat_f32_s();
          else if (dtype0 === DType.Float64) cg.i32.trunc_sat_f64_s();
          else if (i32repr) void 0;
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else if (dtype === DType.Uint32) {
          if (dtype0 === DType.Float32) cg.i32.trunc_sat_f32_u();
          else if (dtype0 === DType.Float64) cg.i32.trunc_sat_f64_u();
          else if (i32repr) void 0;
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else if (dtype === DType.Float32) {
          if (dtype0 === DType.Float32) void 0;
          else if (dtype0 === DType.Float64) cg.f32.demote_f64();
          else if (dtype0 === DType.Int32 || dtype0 === DType.Bool)
            cg.f32.convert_i32_s();
          else if (dtype0 === DType.Uint32) cg.f32.convert_i32_u();
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else if (dtype === DType.Float64) {
          if (dtype0 === DType.Float32) cg.f64.promote_f32();
          else if (dtype0 === DType.Float64) void 0;
          else if (dtype0 === DType.Int32 || dtype0 === DType.Bool)
            cg.f64.convert_i32_s();
          else if (dtype0 === DType.Uint32) cg.f64.convert_i32_u();
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else if (dtype === DType.Bool) {
          if (dtype0 === DType.Bool) void 0;
          else if (i32repr) (cg.i32.const(0), cg.i32.ne());
          else if (dtype0 === DType.Float32) (cg.f32.const(0), cg.f32.ne());
          else if (dtype0 === DType.Float64) (cg.f64.const(0), cg.f64.ne());
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.Bitcast) {
        gen(src[0]);
        const dtype0 = src[0].dtype;
        if (dtype !== dtype0) {
          const i32repr = dtype0 === DType.Int32 || dtype0 === DType.Uint32;
          if (dtype === DType.Int32 || dtype === DType.Uint32) {
            if (dtype0 === DType.Float32) cg.i32.reinterpret_f32();
            else if (i32repr) void 0;
            else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
          } else if (dtype === DType.Float32) {
            if (i32repr) cg.f32.reinterpret_i32();
            else if (dtype0 === DType.Float32) void 0;
            else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
          } else throw new UnsupportedOpError(op, dtype, "wasm");
        }
      } else throw new UnsupportedOpError(op, dtype, "wasm");
    } else if (op === AluOp.Where) {
      gen(src[1]); // t
      gen(src[2]); // f
      gen(src[0]); // cond
      cg.select();
    } else if (op === AluOp.Threefry2x32) {
      for (let i = 0; i < 4; i++) gen(src[i]);
      cg.call(funcs.threefry2x32);
      if (arg === "xor") cg.i32.xor();
      else if (arg === 0) cg.drop();
      else if (arg === 1) {
        const local = cg.local.declare(cg.i32);
        cg.local.set(local);
        cg.drop();
        cg.local.get(local);
      } else throw new UnsupportedOpError(op, dtype, "wasm", arg);
    } else if (op === AluOp.Const) {
      return dty(cg, op, dtype).const(arg as number);
    } else if (op === AluOp.Special) {
      return cg.local.get(ctx[arg[0] as string]);
    } else if (op === AluOp.Variable) {
      return cg.local.get(ctx[arg as string]);
    } else if (op === AluOp.GlobalIndex) {
      const [gid, len] = arg as [number, number];
      gen(src[0]);

      // If value is out-of-bounds, just set it to be zero.
      // This extra bounds-check is needed in Wasm because otherwise we will get
      // out-of-bounds memory access traps. WebGPU just silently returns 0.
      const local = cg.local.declare(cg.i32);
      cg.local.tee(local);
      cg.i32.const(0);
      (cg.local.get(local), cg.i32.const(len), cg.i32.lt_u());
      cg.select();

      cg.i32.const(byteWidth(dtype));
      cg.i32.mul();
      cg.local.get(gid); // base offset of array
      cg.i32.add();
      dty(cg, op, dtype).load(Math.log2(byteWidth(dtype)));
    } else throw new UnsupportedOpError(op, dtype, "wasm");

    if ((references.get(exp) ?? 0) > 1) {
      const local = cg.local.declare(dty(cg, op, dtype));
      cg.local.tee(local);
      expContext.set(exp, local);
    }
  };

  countReferences(exp);
  gen(exp);
}

function dty(cg: CodeGenerator, op: AluOp | null, dtype: DType) {
  switch (dtype) {
    case DType.Float32:
      return cg.f32;
    case DType.Float64:
      return cg.f64;
    case DType.Int32:
    case DType.Uint32:
    case DType.Bool:
      return cg.i32;
    default:
      throw new UnsupportedOpError(op, dtype, "wasm");
  }
}

function dtyF(
  cg: CodeGenerator,
  op: AluOp | null,
  dtype: DType,
): CodeGenerator["f32" | "f64"] {
  switch (dtype) {
    case DType.Float32:
      return cg.f32;
    case DType.Float64:
      return cg.f64;
    default:
      throw new UnsupportedOpError(op, dtype, "wasm");
  }
}
