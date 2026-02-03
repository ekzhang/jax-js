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
import { Routine, Routines, runCpuRoutine } from "../routine";
import { tuneNullopt } from "../tuner";
import { DEBUG, FpHash, mapSetUnion, rep, runWithCache } from "../utils";
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
import { getRoutineModuleSync } from "./wasm/generated/routines";
import { CodeGenerator } from "./wasm/wasmblr";

interface WasmBuffer {
  ptr: number;
  size: number;
  ref: number;
}

interface WasmProgram {
  module: WebAssembly.Module;
}

/** Describes a step in a general scan body (handles data dependencies). */
export interface GeneralScanStep {
  /** The source: either a Kernel (elementwise) or a Routine (special algorithm). */
  source: Kernel | Routine;
  /**
   * Input slot indices. Can reference:
   * - [0, numConsts): constant slots
   * - [numConsts, numConsts+numCarry): carry input slots
   * - [numConsts+numCarry, numConsts+numCarry+numX): xs slots
   * - [numConsts+numCarry+numX, ...): internal buffers from previous steps
   */
  inputSlots: number[];
  /** Which internal buffer this step writes to (index into internalSizes). */
  outputInternalIdx: number;
  /** For multi-output routines: indices of all internal buffers written. */
  outputInternalIndices?: number[];
  /** For routine steps: info needed to call the routine. */
  routineCallInfo?: {
    /** Index into routineInfos array (which routine to call). */
    routineInfoIdx: number;
    /** Static params to pass (e.g., matrix dimension n). */
    staticParams: number[];
  };
}

/** Describes the source for a Y output. */
export interface YOutputSource {
  /** 'passthrough' copies a carry input, 'xs-passthrough' copies from xs input, 'internal' copies from internal buffer. */
  type: "passthrough" | "xs-passthrough" | "internal";
  /** For passthrough: index into carry inputs. */
  carryIdx?: number;
  /** For xs-passthrough: index into xs inputs. */
  xsIdx?: number;
  /** For internal: index into internal buffers. */
  internalIdx?: number;
}

/** Parameters for general native scan execution (handles numCarry !== numY). */
/** Info about a routine used in a scan body, for WASM import. */
export interface ScanRoutineInfo {
  /** The routine enum value (e.g., Routines.Cholesky). */
  routine: Routines;
  /** The WASM export name to call (e.g., "cholesky_f32"). */
  exportName: string;
  /** Number of i32 parameters the routine takes. */
  numParams: number;
}

export interface NativeScanGeneralParams {
  /** Number of scan iterations (length of xs along axis 0). */
  length: number;
  /** Number of constant arrays (passed to body but unchanged). */
  numConsts: number;
  /** Sizes of each constant buffer in bytes. */
  constSizes: number[];
  /** Number of carry arrays. */
  numCarry: number;
  /** Sizes of each carry buffer in bytes. */
  carrySizes: number[];
  /** Number of x inputs per iteration. */
  numX: number;
  /** Strides (in bytes) along axis 0 for each xs input. */
  xsStrides: number[];
  /** Number of y outputs per iteration. */
  numY: number;
  /** Strides (in bytes) along axis 0 for each stacked y output. */
  ysStrides: number[];
  /** Sizes of internal buffers in bytes. */
  internalSizes: number[];
  /** The sequence of steps to execute each iteration (Kernels and/or Routines). */
  steps: GeneralScanStep[];
  /**
   * Maps each carry output to its source (passthrough from carry input or internal buffer).
   */
  carryOutSources: CarryOutputSource[];
  /** Maps each Y output to its source. */
  yOutputSources: YOutputSource[];
  /** Whether to scan in reverse order. */
  reverse?: boolean;
  /** Total size of auxiliary buffers needed for routines (e.g., Sort merge buffer). */
  auxBufferSize?: number;
  /** Element size for routines (4 for f32, 8 for f64). */
  elementSize?: 4 | 8;
  /** Routines used in the scan body that need to be imported. */
  routineInfos?: ScanRoutineInfo[];
}

/** Source for a carry output in general native scan. */
export interface CarryOutputSource {
  type: "passthrough" | "internal";
  /** For passthrough: which carry input this comes from. */
  carryIdx?: number;
  /** For internal: which internal buffer this comes from. */
  internalIdx?: number;
}

const moduleCache = new Map<string, WebAssembly.Module>();

/** Map routine enum to AS module name. */
const routineModuleNames: Record<Routines, string> = {
  [Routines.Cholesky]: "cholesky",
  [Routines.TriangularSolve]: "triangular-solve",
  [Routines.LU]: "lu",
  [Routines.Sort]: "sort",
  [Routines.Argsort]: "argsort",
};

/** Backend that compiles into WebAssembly bytecode for immediate execution. */
export class WasmBackend implements Backend {
  readonly type: Device = "wasm";
  readonly maxArgs = 64; // Arbitrary choice

  #memory: WebAssembly.Memory;
  #nextSlot: number;
  #allocator: WasmAllocator;
  #buffers: Map<Slot, WasmBuffer>;
  /** Cache WebAssembly instances keyed by module for reuse in dispatch. */
  #instanceCache: WeakMap<WebAssembly.Module, WebAssembly.Instance>;
  /** Cached WASM routine instances, keyed by "routine:elementSize". */
  #routineInstances: Map<string, WebAssembly.Instance> = new Map();

  constructor() {
    this.#memory = new WebAssembly.Memory({ initial: 0 });
    this.#allocator = new WasmAllocator(this.#memory);
    this.#nextSlot = 1;
    this.#buffers = new Map();
    this.#instanceCache = new WeakMap();
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

  slotCount(): number {
    return this.#buffers.size;
  }

  async read(
    slot: Slot,
    start?: number,
    count?: number,
  ): Promise<Uint8Array<ArrayBuffer>> {
    return this.readSync(slot, start, count);
  }

  readSync(
    slot: Slot,
    start?: number,
    count?: number,
  ): Uint8Array<ArrayBuffer> {
    const buffer = this.#getBuffer(slot);
    if (start === undefined) start = 0;
    if (count === undefined) count = buffer.byteLength - start;
    return buffer.slice(start, start + count);
  }

  async prepareKernel(kernel: Kernel): Promise<Executable<WasmProgram>> {
    return this.prepareKernelSync(kernel);
  }

  prepareKernelSync(kernel: Kernel): Executable<WasmProgram> {
    const kernelHash = FpHash.hash(kernel);
    const module = runWithCache(moduleCache, kernelHash.toString(), () => {
      const bytes = codegenWasm(kernel);
      return new WebAssembly.Module(bytes);
    });
    return new Executable(kernel, { module });
  }

  async prepareRoutine(routine: Routine): Promise<Executable<WasmProgram>> {
    return this.prepareRoutineSync(routine);
  }

  prepareRoutineSync(routine: Routine): Executable<WasmProgram> {
    // Currently, Wasm routines fall back to the CPU reference implementation
    // implementation. We may optimize this in the future.
    return new Executable(routine, undefined as any);
  }

  dispatch(
    exe: Executable<WasmProgram>,
    inputs: Slot[],
    outputs: Slot[],
  ): void {
    if (exe.source instanceof Routine) {
      const routine = exe.source;
      // Determine element size from dtype (f32=4, f64=8)
      const dtype = routine.type.inputDtypes[0];
      const isF32 = dtype === DType.Float32;
      const isF64 = dtype === DType.Float64;
      if (isF32 || isF64) {
        const elementSize: 4 | 8 = isF32 ? 4 : 8;
        switch (routine.name) {
          case Routines.Cholesky:
            return this.#dispatchCholesky(
              routine,
              inputs,
              outputs,
              elementSize,
            );
          case Routines.TriangularSolve:
            return this.#dispatchTriangularSolve(
              routine,
              inputs,
              outputs,
              elementSize,
            );
          case Routines.LU:
            return this.#dispatchLU(routine, inputs, outputs, elementSize);
          case Routines.Sort:
            return this.#dispatchSort(routine, inputs, outputs, elementSize);
          case Routines.Argsort:
            return this.#dispatchArgsort(routine, inputs, outputs, elementSize);
        }
      }
      // Fall back to CPU for non-float or unimplemented routines
      return runCpuRoutine(
        routine,
        inputs.map((slot) => this.#getBuffer(slot)),
        outputs.map((slot) => this.#getBuffer(slot)),
      );
    }

    // Reuse cached instance if available (334x faster than creating new)
    let instance = this.#instanceCache.get(exe.data.module);
    if (!instance) {
      instance = new WebAssembly.Instance(exe.data.module, {
        env: { memory: this.#memory },
      });
      this.#instanceCache.set(exe.data.module, instance);
    }
    const func = instance.exports.kernel as (...args: number[]) => void;
    const ptrs = [...inputs, ...outputs].map(
      (slot) => this.#buffers.get(slot)!.ptr,
    );
    func(...ptrs);
  }

  /** Get or create a WASM instance for a routine. */
  #getRoutineInstance(name: Routines): WebAssembly.Instance {
    const moduleName = routineModuleNames[name];
    let instance = this.#routineInstances.get(moduleName);
    if (!instance) {
      const module = getRoutineModuleSync(moduleName);
      instance = new WebAssembly.Instance(module, {
        env: { memory: this.#memory },
      });
      this.#routineInstances.set(moduleName, instance);
    }
    return instance;
  }

  #dispatchCholesky(
    routine: Routine,
    inputs: Slot[],
    outputs: Slot[],
    elementSize: 4 | 8,
  ): void {
    const instance = this.#getRoutineInstance(Routines.Cholesky);
    const exportName =
      elementSize === 4 ? "cholesky_batched_f32" : "cholesky_batched_f64";
    const func = instance.exports[exportName] as (
      i: number,
      o: number,
      n: number,
      b: number,
    ) => void;
    const shape = routine.type.inputShapes[0];
    func(
      this.#buffers.get(inputs[0])!.ptr,
      this.#buffers.get(outputs[0])!.ptr,
      shape[shape.length - 1],
      shape.slice(0, -2).reduce((a, b) => a * b, 1),
    );
  }

  #dispatchTriangularSolve(
    routine: Routine,
    inputs: Slot[],
    outputs: Slot[],
    elementSize: 4 | 8,
  ): void {
    const instance = this.#getRoutineInstance(Routines.TriangularSolve);
    const exportName =
      elementSize === 4
        ? "triangular_solve_batched_f32"
        : "triangular_solve_batched_f64";
    const func = instance.exports[exportName] as (
      a: number,
      b: number,
      x: number,
      n: number,
      batchRows: number,
      numBatches: number,
      unitDiagonal: number,
      lower: number,
    ) => void;
    const aShape = routine.type.inputShapes[0];
    const bShape = routine.type.inputShapes[1];
    const n = aShape[aShape.length - 1];
    const batchRows = bShape[bShape.length - 2]; // number of rows in B
    const numBatches = aShape.slice(0, -2).reduce((a, b) => a * b, 1);
    func(
      this.#buffers.get(inputs[0])!.ptr,
      this.#buffers.get(inputs[1])!.ptr,
      this.#buffers.get(outputs[0])!.ptr,
      n,
      batchRows,
      numBatches,
      routine.params?.unitDiagonal ? 1 : 0,
      routine.params?.lower ? 1 : 0,
    );
  }

  #dispatchLU(
    routine: Routine,
    inputs: Slot[],
    outputs: Slot[],
    elementSize: 4 | 8,
  ): void {
    const instance = this.#getRoutineInstance(Routines.LU);
    const exportName = elementSize === 4 ? "lu_batched_f32" : "lu_batched_f64";
    const func = instance.exports[exportName] as (
      a: number,
      lu: number,
      piv: number,
      perm: number,
      m: number,
      n: number,
      batchSize: number,
    ) => void;
    const shape = routine.type.inputShapes[0];
    func(
      this.#buffers.get(inputs[0])!.ptr,
      this.#buffers.get(outputs[0])!.ptr,
      this.#buffers.get(outputs[1])!.ptr,
      this.#buffers.get(outputs[2])!.ptr,
      shape[shape.length - 2],
      shape[shape.length - 1],
      shape.slice(0, -2).reduce((a, b) => a * b, 1),
    );
  }

  #dispatchSort(
    routine: Routine,
    inputs: Slot[],
    outputs: Slot[],
    elementSize: 4 | 8,
  ): void {
    const shape = routine.type.inputShapes[0];
    const n = shape[shape.length - 1];
    const batchSize = shape.slice(0, -1).reduce((a, b) => a * b, 1);
    const totalSize = n * batchSize * elementSize;

    // Copy input to output (AS sort is in-place)
    const inBuf = this.#buffers.get(inputs[0])!;
    const outBuf = this.#buffers.get(outputs[0])!;
    new Uint8Array(this.#memory.buffer, outBuf.ptr, totalSize).set(
      new Uint8Array(this.#memory.buffer, inBuf.ptr, totalSize),
    );

    // Allocate auxiliary buffer
    const auxPtr = this.#allocator.malloc(n * elementSize);

    // Call in-place sort on output buffer
    const instance = this.#getRoutineInstance(Routines.Sort);
    const exportName =
      elementSize === 4 ? "sort_batched_f32" : "sort_batched_f64";
    const func = instance.exports[exportName] as (
      data: number,
      aux: number,
      n: number,
      batchSize: number,
    ) => void;
    func(outBuf.ptr, auxPtr, n, batchSize);

    this.#allocator.free(auxPtr);
  }

  #dispatchArgsort(
    routine: Routine,
    inputs: Slot[],
    outputs: Slot[],
    elementSize: 4 | 8,
  ): void {
    const shape = routine.type.inputShapes[0];
    const n = shape[shape.length - 1];
    const batchSize = shape.slice(0, -1).reduce((a, b) => a * b, 1);

    // Allocate auxiliary buffers (aux uses 4 bytes for indices)
    const auxPtr = this.#allocator.malloc(n * 4);

    const instance = this.#getRoutineInstance(Routines.Argsort);
    const exportName =
      elementSize === 4 ? "argsort_batched_f32" : "argsort_batched_f64";
    const func = instance.exports[exportName] as (
      data: number,
      out: number,
      idx: number,
      aux: number,
      n: number,
      batchSize: number,
    ) => void;
    func(
      this.#buffers.get(inputs[0])!.ptr,
      this.#buffers.get(outputs[0])!.ptr,
      this.#buffers.get(outputs[1])!.ptr,
      auxPtr,
      n,
      batchSize,
    );

    this.#allocator.free(auxPtr);
  }

  /**
   * Prepare a native scan operation for efficient execution.
   * This is the unified implementation that handles all scan body types:
   * - Single kernel bodies (like cumsum)
   * - Multiple independent kernels (like Kalman filter with 2 matmuls)
   * - Bodies with data dependencies between steps
   * - Bodies where numCarry !== numY
   * - Routine steps (Cholesky, Sort) mixed with Kernel steps
   *
   * Note: This compiles a new WASM module for each distinct set of scan parameters.
   * However, `jit()` caching at the higher level protects against recompilation:
   * the JitProgram (including the native-scan step) is cached after the first trace,
   * so repeated calls to the same jit'd function reuse the compiled module.
   */
  prepareNativeScanGeneral(
    params: NativeScanGeneralParams,
  ): Executable<WasmProgram> | null {
    if (params.steps.length === 0) return null;

    try {
      const bytes = codegenNativeScanGeneral(params);
      const module = new WebAssembly.Module(bytes);
      // Create a synthetic executable for typing purposes
      // Find the first Kernel to use as the source (or create a dummy one)
      let firstKernel: Kernel | null = null;
      for (const step of params.steps) {
        if (step.source instanceof Kernel) {
          firstKernel = step.source;
          break;
        }
      }
      if (!firstKernel) {
        // All steps are Routines - create a minimal dummy kernel for the Executable type
        // This is a hack but the Executable is just used to hold the module
        firstKernel = new Kernel(
          0,
          0,
          AluExp.const(DType.Float32, 0),
          undefined,
        );
      }
      const syntheticKernel = new Kernel(
        firstKernel.nargs,
        firstKernel.size,
        firstKernel.exp,
        firstKernel.reduction,
      );
      return new Executable(syntheticKernel, { module });
    } catch (e) {
      if (DEBUG >= 1) {
        console.warn("General native scan codegen failed:", e);
      }
      return null;
    }
  }

  /**
   * Dispatch a general native scan operation.
   * Allocates temporary internal buffers, runs the scan, and frees them.
   *
   * @param exe - Compiled WASM executable for the scan
   * @param params - General scan parameters (includes internalSizes, auxBufferSize)
   * @param consts - Constant buffer slots
   * @param initCarry - Initial carry buffer slots
   * @param xs - Input xs buffer slots
   * @param carryOut - Output carry buffer slots
   * @param ysStacked - Output stacked ys buffer slots
   */
  dispatchNativeScanGeneral(
    exe: Executable<WasmProgram>,
    params: NativeScanGeneralParams,
    consts: Slot[],
    initCarry: Slot[],
    xs: Slot[],
    carryOut: Slot[],
    ysStacked: Slot[],
  ): void {
    // Allocate internal buffers
    const internalSlots: Slot[] = [];
    for (const size of params.internalSizes) {
      const slot = this.malloc(size);
      internalSlots.push(slot);
    }

    // Allocate aux buffer if needed (for routines like Sort)
    let auxPtr = 0;
    if (params.auxBufferSize && params.auxBufferSize > 0) {
      auxPtr = this.#allocator.malloc(params.auxBufferSize);
    }

    try {
      // Get or create WASM instance
      let instance = this.#instanceCache.get(exe.data.module);
      if (!instance) {
        // Build imports: env.memory + any routine functions
        const imports: WebAssembly.Imports = {
          env: { memory: this.#memory },
        };

        // Add routine function imports if needed
        if (params.routineInfos && params.routineInfos.length > 0) {
          const routineImports: Record<string, WebAssembly.ExportValue> = {};
          for (const info of params.routineInfos) {
            const routineInstance = this.#getRoutineInstance(info.routine);
            routineImports[info.exportName] = routineInstance.exports[
              info.exportName
            ] as WebAssembly.ExportValue;
          }
          imports.routines = routineImports;
        }

        instance = new WebAssembly.Instance(exe.data.module, imports);
        this.#instanceCache.set(exe.data.module, instance);
      }
      const func = instance.exports.scan as (...args: number[]) => void;

      // Build pointer array: [consts, initCarry, xs, carryOut, ysStacked, internals, aux?]
      const ptrs = [
        ...consts.map((slot) => this.#buffers.get(slot)!.ptr),
        ...initCarry.map((slot) => this.#buffers.get(slot)!.ptr),
        ...xs.map((slot) => this.#buffers.get(slot)!.ptr),
        ...carryOut.map((slot) => this.#buffers.get(slot)!.ptr),
        ...ysStacked.map((slot) => this.#buffers.get(slot)!.ptr),
        ...internalSlots.map((slot) => this.#buffers.get(slot)!.ptr),
      ];
      if (auxPtr) ptrs.push(auxPtr);

      func(...ptrs);
    } finally {
      // Free internal buffers
      for (const slot of internalSlots) {
        this.decRef(slot);
      }
      // Free aux buffer
      if (auxPtr) {
        this.#allocator.free(auxPtr);
      }
    }
  }

  #getBuffer(slot: Slot): Uint8Array<ArrayBuffer> {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    return new Uint8Array(this.#memory.buffer, buffer.ptr, buffer.size);
  }
}

// ============================================================================
// Shared WASM codegen helpers
// ============================================================================

/**
 * Import all required math helper functions based on the ops used.
 * Accepts either a Set<AluOp> or Map<AluOp, Set<DType>> (from distinctOps()).
 * Returns a record mapping op names to WASM function indices.
 */
function importWasmHelperFuncs(
  cg: CodeGenerator,
  ops: Set<AluOp> | Map<AluOp, Set<DType>>,
): Record<string, number> {
  const funcs: Record<string, number> = {};
  const hasOp = (op: AluOp) => (ops instanceof Map ? ops.has(op) : ops.has(op));
  if (hasOp(AluOp.Sin)) funcs.sin = wasm_sin(cg);
  if (hasOp(AluOp.Cos)) funcs.cos = wasm_cos(cg);
  if (hasOp(AluOp.Asin)) funcs.asin = wasm_asin(cg);
  if (hasOp(AluOp.Atan)) funcs.atan = wasm_atan(cg);
  if (hasOp(AluOp.Exp) || hasOp(AluOp.Erf) || hasOp(AluOp.Erfc))
    funcs.exp = wasm_exp(cg);
  if (hasOp(AluOp.Log)) funcs.log = wasm_log(cg);
  if (hasOp(AluOp.Erf)) funcs.erf = wasm_erf(cg, funcs.exp);
  if (hasOp(AluOp.Erfc)) funcs.erfc = wasm_erfc(cg, funcs.exp);
  if (hasOp(AluOp.Threefry2x32)) funcs.threefry2x32 = wasm_threefry2x32(cg);
  return funcs;
}

/**
 * Generate reduction accumulator update code.
 * Assumes the expression result is on the stack.
 * Leaves the accumulated value in the acc local.
 */
function codegenReductionAccumulate(
  cg: CodeGenerator,
  re: { op: AluOp; dtype: DType; size: number; identity: number },
  acc: number,
): void {
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
    } else if ([DType.Int32, DType.Uint32, DType.Bool].includes(re.dtype)) {
      // WASM has no i32.min/max, so emulate with select
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
    } else {
      throw new Error(`invalid reduction min/max over ${re.dtype}`);
    }
  } else {
    throw new Error(`invalid wasm reduction op: ${re.op}`);
  }
  cg.local.set(acc);
}

function codegenWasm(kernel: Kernel): Uint8Array<ArrayBuffer> {
  const tune = tuneNullopt(kernel);
  const re = kernel.reduction;

  if (DEBUG >= 3) {
    console.info(`kernel.exp: ${kernel.exp}\ntune.exp: ${tune.exp}`);
  }

  const cg = new CodeGenerator();
  cg.memory.import("env", "memory");

  const distinctOps = mapSetUnion(
    tune.exp.distinctOps(),
    tune.epilogue?.distinctOps(),
  );
  const funcs = importWasmHelperFuncs(cg, distinctOps);

  const kernelFunc = cg.function(rep(kernel.nargs + 1, cg.i32), [], () => {
    const gidx = cg.local.declare(cg.i32);
    cg.loop(cg.void);
    {
      // if (gidx >= size) break;
      cg.block(cg.void);
      cg.local.get(gidx);
      cg.i32.const(kernel.size);
      cg.i32.ge_u();
      cg.br_if(0);

      // Push memory index of output onto stack (will be used at end).
      cg.local.get(kernel.nargs); // output buffer is last argument
      cg.local.get(gidx);
      cg.i32.const(byteWidth(kernel.dtype));
      cg.i32.mul();
      cg.i32.add();

      if (re) {
        // If reduction, define accumulator and inner ridx loop.
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
          codegenReductionAccumulate(cg, re, acc);

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

/**
 * Generate WASM bytecode for a general native scan.
 * Handles scan bodies with data dependencies and numCarry !== numY.
 *
 * Memory layout for function arguments:
 * [...consts (numConsts), ...carryIn (numCarry), ...xs (numX),
 *  ...carryOut (numCarry), ...ysStacked (numY), ...internals (numInternal)]
 */
function codegenNativeScanGeneral(
  params: NativeScanGeneralParams,
): Uint8Array<ArrayBuffer> {
  const {
    length,
    numConsts,
    constSizes,
    numCarry,
    carrySizes,
    numX,
    xsStrides,
    numY,
    ysStrides,
    internalSizes,
    steps,
    carryOutSources,
    yOutputSources,
    reverse,
    routineInfos,
  } = params;

  const numInternal = internalSizes.length;

  if (DEBUG >= 2) {
    console.log("codegenNativeScanGeneral params:", {
      length,
      numConsts,
      numCarry,
      numX,
      numY,
      numInternal,
      numSteps: steps.length,
      reverse,
      routineInfos: routineInfos?.map((r) => r.exportName),
    });
  }

  const cg = new CodeGenerator();
  cg.memory.import("env", "memory");

  // Import routine functions from the "routines" module
  // These will be provided at instantiation time from the pre-compiled AS modules
  const routineFuncIndices: number[] = [];
  if (routineInfos) {
    for (const info of routineInfos) {
      const funcIdx = cg.importFunction(
        "routines",
        info.exportName,
        rep(info.numParams, cg.i32),
        [],
      );
      routineFuncIndices.push(funcIdx);
    }
  }

  // Collect all helper functions needed by kernels
  const allOps = new Set<AluOp>();
  for (const step of steps) {
    if (step.source instanceof Kernel) {
      const tune = tuneNullopt(step.source);
      for (const op of tune.exp.distinctOps().keys()) allOps.add(op);
      if (tune.epilogue) {
        for (const op of tune.epilogue.distinctOps().keys()) allOps.add(op);
      }
    }
  }

  const funcs = importWasmHelperFuncs(cg, allOps);

  // Function arguments:
  // [...consts (numConsts), ...carryIn (numCarry), ...xs (numX),
  //  ...carryOut (numCarry), ...ysStacked (numY), ...internals (numInternal), aux?]
  const needsAux = (params.auxBufferSize ?? 0) > 0;
  const numArgs =
    numConsts +
    numCarry +
    numX +
    numCarry +
    numY +
    numInternal +
    (needsAux ? 1 : 0);
  const auxArgIdx = needsAux
    ? numConsts + numCarry + numX + numCarry + numY + numInternal
    : -1;

  const scanFunc = cg.function(rep(numArgs, cg.i32), [], () => {
    // Local variables
    const iter = cg.local.declare(cg.i32); // scan iteration counter
    const gidx = cg.local.declare(cg.i32); // body kernel loop index
    const copyIdx = cg.local.declare(cg.i32); // for memory copy loops
    const dataIdx = cg.local.declare(cg.i32); // data index for xs/ys (differs from iter when reverse)

    // Argument indices
    const constsBase = 0;
    const carryInBase = numConsts;
    const xsBase = numConsts + numCarry;
    const carryOutBase = numConsts + numCarry + numX;
    const ysStackedBase = numConsts + numCarry + numX + numCarry;
    const internalsBase = numConsts + numCarry + numX + numCarry + numY;

    // Step 1: Copy carryIn to carryOut (working buffer)
    for (let c = 0; c < numCarry; c++) {
      const size = carrySizes[c];
      cg.i32.const(0);
      cg.local.set(copyIdx);
      cg.loop(cg.void);
      {
        cg.block(cg.void);
        cg.local.get(copyIdx);
        cg.i32.const(size);
        cg.i32.ge_u();
        cg.br_if(0);

        // carryOut[c][copyIdx] = carryIn[c][copyIdx]
        cg.local.get(carryOutBase + c);
        cg.local.get(copyIdx);
        cg.i32.add();
        cg.local.get(carryInBase + c);
        cg.local.get(copyIdx);
        cg.i32.add();
        cg.i32.load8_u(0);
        cg.i32.store8(0);

        cg.local.get(copyIdx);
        cg.i32.const(1);
        cg.i32.add();
        cg.local.set(copyIdx);
        cg.br(1);
        cg.end();
      }
      cg.end();
    }

    // Step 2: Main scan loop
    cg.i32.const(0);
    cg.local.set(iter);
    cg.loop(cg.void);
    {
      cg.block(cg.void);
      cg.local.get(iter);
      cg.i32.const(length);
      cg.i32.ge_u();
      cg.br_if(0);

      // Compute dataIdx = reverse ? (length - 1 - iter) : iter
      if (reverse) {
        cg.i32.const(length - 1);
        cg.local.get(iter);
        cg.i32.sub();
        cg.local.set(dataIdx);
      } else {
        cg.local.get(iter);
        cg.local.set(dataIdx);
      }

      // Step 2a: Execute each step (Kernel or Routine), writing to internal buffers
      for (let stepIdx = 0; stepIdx < steps.length; stepIdx++) {
        const step = steps[stepIdx];
        const internalIdx = step.outputInternalIdx;

        if (step.source instanceof Routine) {
          // Routine step: call the imported routine function
          const callInfo = step.routineCallInfo!;
          const funcIdx = routineFuncIndices[callInfo.routineInfoIdx];
          const routineType = routineInfos![callInfo.routineInfoIdx].routine;

          // Helper to push a slot pointer onto the stack
          const pushSlotPtr = (slotIdx: number) => {
            if (slotIdx < numConsts) {
              cg.local.get(constsBase + slotIdx);
            } else if (slotIdx < numConsts + numCarry) {
              cg.local.get(carryOutBase + (slotIdx - numConsts));
            } else if (slotIdx < numConsts + numCarry + numX) {
              // xs input: base + dataIdx * stride
              const xIdx = slotIdx - numConsts - numCarry;
              cg.local.get(xsBase + xIdx);
              cg.local.get(dataIdx);
              cg.i32.const(xsStrides[xIdx]);
              cg.i32.mul();
              cg.i32.add();
            } else {
              // Internal buffer
              const intIdx = slotIdx - numConsts - numCarry - numX;
              cg.local.get(internalsBase + intIdx);
            }
          };

          if (routineType === Routines.Cholesky) {
            // cholesky_f32(inPtr, outPtr, n)
            pushSlotPtr(step.inputSlots[0]); // inPtr
            cg.local.get(internalsBase + internalIdx); // outPtr
            for (const param of callInfo.staticParams) {
              cg.i32.const(param); // n
            }
          } else if (routineType === Routines.Sort) {
            // sort_f32(dataPtr, auxPtr, n) - in-place, needs aux buffer
            // First, copy input to output (internal buffer), then sort in place
            // For now, push: internal buffer as dataPtr, aux buffer, n
            // The input needs to be copied to internal buffer first

            // Copy input to internal buffer first
            const inputSlotIdx = step.inputSlots[0];
            const sortSize = callInfo.staticParams[0]; // n (number of elements)
            const elemSize = params.elementSize ?? 4;
            const copySize = sortSize * elemSize;

            // Copy loop: internal[i] = input[i]
            cg.i32.const(0);
            cg.local.set(copyIdx);
            cg.loop(cg.void);
            {
              cg.block(cg.void);
              cg.local.get(copyIdx);
              cg.i32.const(copySize);
              cg.i32.ge_u();
              cg.br_if(0);

              // internal[copyIdx] = input[copyIdx]
              cg.local.get(internalsBase + internalIdx);
              cg.local.get(copyIdx);
              cg.i32.add();

              // Get input pointer and add offset
              pushSlotPtr(inputSlotIdx);
              cg.local.get(copyIdx);
              cg.i32.add();
              cg.i32.load8_u(0);

              cg.i32.store8(0);

              cg.local.get(copyIdx);
              cg.i32.const(1);
              cg.i32.add();
              cg.local.set(copyIdx);
              cg.br(1);
              cg.end();
            }
            cg.end();

            // Now call sort with internal buffer as dataPtr
            cg.local.get(internalsBase + internalIdx); // dataPtr (in-place)
            cg.local.get(auxArgIdx); // auxPtr
            cg.i32.const(sortSize); // n
          } else if (routineType === Routines.TriangularSolve) {
            // triangular_solve_batched_f32(aPtr, bPtr, xPtr, n, batchRows, numBatches, unitDiag, lower)
            pushSlotPtr(step.inputSlots[0]); // aPtr
            pushSlotPtr(step.inputSlots[1]); // bPtr
            cg.local.get(internalsBase + internalIdx); // xPtr (output)
            for (const param of callInfo.staticParams) {
              cg.i32.const(param); // n, batchRows, numBatches, unitDiag, lower
            }
          } else if (routineType === Routines.LU) {
            // lu_f32(aPtr, luPtr, pivPtr, permPtr, m, n)
            const outIndices = step.outputInternalIndices!;
            pushSlotPtr(step.inputSlots[0]); // aPtr
            cg.local.get(internalsBase + outIndices[0]); // luPtr (first output)
            cg.local.get(internalsBase + outIndices[1]); // pivPtr (second output)
            cg.local.get(internalsBase + outIndices[2]); // permPtr (third output)
            for (const param of callInfo.staticParams) {
              cg.i32.const(param); // m, n
            }
          } else if (routineType === Routines.Argsort) {
            // argsort_f32(dataPtr, outPtr, idxPtr, auxPtr, n)
            const outIndices = step.outputInternalIndices!;
            const sortSize = callInfo.staticParams[0]; // n
            pushSlotPtr(step.inputSlots[0]); // dataPtr (input)
            cg.local.get(internalsBase + outIndices[0]); // outPtr (sorted values)
            cg.local.get(internalsBase + outIndices[1]); // idxPtr (indices)
            cg.local.get(auxArgIdx); // auxPtr
            cg.i32.const(sortSize); // n
          } else {
            // Generic fallback (shouldn't happen for supported routines)
            pushSlotPtr(step.inputSlots[0]);
            cg.local.get(internalsBase + internalIdx);
            for (const param of callInfo.staticParams) {
              cg.i32.const(param);
            }
          }

          // Call the routine
          cg.call(funcIdx);
        } else {
          // Kernel step: existing codegen
          const kernel = step.source;
          const tune = tuneNullopt(kernel);
          const re = kernel.reduction;

          // Inner loop over kernel output elements
          cg.i32.const(0);
          cg.local.set(gidx);
          cg.loop(cg.void);
          {
            cg.block(cg.void);
            cg.local.get(gidx);
            cg.i32.const(kernel.size);
            cg.i32.ge_u();
            cg.br_if(0);

            // Compute output address: internals[internalIdx] + gidx * elementSize
            cg.local.get(internalsBase + internalIdx);
            cg.local.get(gidx);
            cg.i32.const(byteWidth(kernel.dtype));
            cg.i32.mul();
            cg.i32.add();

            // Context for translateExpWithGeneralScanContext
            const scanCtx: GeneralScanContext = {
              gidx,
              iter,
              dataIdx,
              ridx: -1,
              constsBase,
              constSizes,
              numConsts,
              xsBase,
              xsStrides,
              carryBase: carryOutBase, // Read from carryOut (updated each iter)
              carrySizes,
              numCarry,
              internalsBase,
              internalSizes,
              numInternal,
              numInputs: numConsts + numCarry + numX,
            };

            if (re) {
              // Reduction: define accumulator and inner ridx loop
              const acc = cg.local.declare(dty(cg, null, kernel.exp.dtype));
              dty(cg, null, kernel.exp.dtype).const(re.identity);
              cg.local.set(acc);

              const ridx = cg.local.declare(cg.i32);
              cg.i32.const(0);
              cg.local.set(ridx);
              scanCtx.ridx = ridx;

              cg.loop(cg.void);
              {
                cg.block(cg.void);
                cg.local.get(ridx);
                cg.i32.const(re.size);
                cg.i32.ge_u();
                cg.br_if(0);

                translateExpWithGeneralScanContext(
                  cg,
                  funcs,
                  tune.exp,
                  scanCtx,
                );

                // acc = reduction.evaluate(acc, exp)
                codegenReductionAccumulate(cg, re, acc);

                cg.local.get(ridx);
                cg.i32.const(1);
                cg.i32.add();
                cg.local.set(ridx);

                cg.br(1);
                cg.end();
              }
              cg.end();

              // Apply epilogue
              translateExpWithGeneralScanContext(cg, funcs, tune.epilogue!, {
                ...scanCtx,
                acc,
              });
            } else {
              translateExpWithGeneralScanContext(cg, funcs, tune.exp, scanCtx);
            }

            // Store result to internal buffer (address already on stack)
            dty(cg, null, kernel.dtype).store(
              Math.log2(byteWidth(kernel.dtype)),
            );

            // gidx++
            cg.local.get(gidx);
            cg.i32.const(1);
            cg.i32.add();
            cg.local.set(gidx);
            cg.br(1);
            cg.end();
          }
          cg.end();
        }
      }

      // Step 2b: Copy Y outputs to ysStacked at iteration offset
      // NOTE: Must run BEFORE carry update (2c) so passthrough reads OLD carry values
      for (let y = 0; y < numY; y++) {
        const source = yOutputSources[y];
        const yStride = ysStrides[y];

        // Determine source pointer and size
        let size: number;
        let isXsPassthrough = false;
        let srcArgIdx: number = 0;
        let xsPassthroughIdx: number = 0;

        if (source.type === "passthrough") {
          // Read from carryOut (has carry values entering this iteration)
          srcArgIdx = carryOutBase + source.carryIdx!;
          size = carrySizes[source.carryIdx!];
        } else if (source.type === "xs-passthrough") {
          // Read from xs at current iteration offset
          isXsPassthrough = true;
          xsPassthroughIdx = source.xsIdx!;
          size = xsStrides[xsPassthroughIdx];
        } else {
          // Read from internal buffer (computed in step 2a)
          srcArgIdx = internalsBase + source.internalIdx!;
          size = internalSizes[source.internalIdx!];
        }

        // Copy loop: ysStacked[y][dataIdx * yStride + i] = src[i] for i in 0..size
        cg.i32.const(0);
        cg.local.set(copyIdx);
        cg.loop(cg.void);
        {
          cg.block(cg.void);
          cg.local.get(copyIdx);
          cg.i32.const(size);
          cg.i32.ge_u();
          cg.br_if(0);

          // ysStacked[y] + dataIdx * yStride + copyIdx
          cg.local.get(ysStackedBase + y);
          cg.local.get(dataIdx);
          cg.i32.const(yStride);
          cg.i32.mul();
          cg.i32.add();
          cg.local.get(copyIdx);
          cg.i32.add();

          // src[copyIdx] - different computation for xs-passthrough
          if (isXsPassthrough) {
            // xs[xsIdx] + dataIdx * xsStrides[xsIdx] + copyIdx
            cg.local.get(xsBase + xsPassthroughIdx);
            cg.local.get(dataIdx);
            cg.i32.const(xsStrides[xsPassthroughIdx]);
            cg.i32.mul();
            cg.i32.add();
            cg.local.get(copyIdx);
            cg.i32.add();
          } else {
            cg.local.get(srcArgIdx);
            cg.local.get(copyIdx);
            cg.i32.add();
          }
          cg.i32.load8_u(0);

          // store
          cg.i32.store8(0);

          cg.local.get(copyIdx);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(copyIdx);
          cg.br(1);
          cg.end();
        }
        cg.end();
      }

      // Step 2c: Copy carry outputs from internal buffers (or passthrough) to carryOut (for next iteration)
      for (let c = 0; c < numCarry; c++) {
        const source = carryOutSources[c];
        const size = carrySizes[c];

        // For passthrough, the carry output is the carry input (no copy needed if same buffer)
        // But we're double-buffering: carryIn and carryOut are different, so we need to copy
        const srcLocal =
          source.type === "passthrough"
            ? carryInBase + source.carryIdx!
            : internalsBase + source.internalIdx!;

        cg.i32.const(0);
        cg.local.set(copyIdx);
        cg.loop(cg.void);
        {
          cg.block(cg.void);
          cg.local.get(copyIdx);
          cg.i32.const(size);
          cg.i32.ge_u();
          cg.br_if(0);

          // carryOut[c][copyIdx] = src[copyIdx]
          cg.local.get(carryOutBase + c);
          cg.local.get(copyIdx);
          cg.i32.add();
          cg.local.get(srcLocal);
          cg.local.get(copyIdx);
          cg.i32.add();
          cg.i32.load8_u(0);
          cg.i32.store8(0);

          cg.local.get(copyIdx);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(copyIdx);
          cg.br(1);
          cg.end();
        }
        cg.end();
      }

      // iter++
      cg.local.get(iter);
      cg.i32.const(1);
      cg.i32.add();
      cg.local.set(iter);

      cg.br(1);
      cg.end();
    }
    cg.end();
  });

  cg.export(scanFunc, "scan");
  return cg.finish();
}

/** Context for general scan expression translation. */
interface GeneralScanContext {
  gidx: number;
  iter: number;
  dataIdx: number; // Iteration index for data access (differs from iter when reverse=true)
  ridx: number;
  acc?: number;
  constsBase: number;
  constSizes: number[];
  numConsts: number;
  xsBase: number;
  xsStrides: number[];
  carryBase: number;
  carrySizes: number[];
  numCarry: number;
  internalsBase: number;
  internalSizes: number[];
  numInternal: number;
  /** Total number of jaxpr inputs (consts + carry + xs). */
  numInputs: number;
}

/**
 * Translate an AluExp to WASM code within a general scan context.
 * Supports reading from internal buffers (for data dependencies between steps).
 */
function translateExpWithGeneralScanContext(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  exp: AluExp,
  ctx: GeneralScanContext,
) {
  const references = new Map<AluExp, number>();
  const seen = new Set<AluExp>();
  const countReferences = (e: AluExp) => {
    references.set(e, (references.get(e) ?? 0) + 1);
    if (!seen.has(e)) {
      seen.add(e);
      for (const src of e.src) countReferences(src);
    }
  };

  const expContext = new Map<AluExp, number>();
  const gen = (e: AluExp): void => {
    if (expContext.has(e)) {
      cg.local.get(expContext.get(e)!);
      return;
    }
    const { op, src, dtype, arg } = e;

    // Handle GlobalIndex: load from appropriate buffer based on slot classification
    if (op === AluOp.GlobalIndex) {
      // arg = [gid, len], src = [indexExp]
      // Note: gid is already reindexed to be the jaxpr slot (const/carry/xs/internal index)
      const gid = arg[0] as number;
      const bw = byteWidth(dtype);

      // gid is the jaxpr slot after reindexing:
      // [0, numConsts): constant
      // [numConsts, numConsts+numCarry): carry
      // [numConsts+numCarry, numInputs): xs
      // [numInputs, numInputs+numInternal): internal buffer

      if (gid < ctx.numConsts) {
        // Constant input (no iteration offset)
        cg.local.get(ctx.constsBase + gid);
      } else if (gid < ctx.numConsts + ctx.numCarry) {
        // Carry input (read from carryOut which has current carry values)
        const carryIdx = gid - ctx.numConsts;
        cg.local.get(ctx.carryBase + carryIdx);
      } else if (gid < ctx.numInputs) {
        // X input with iteration offset (use dataIdx for reverse support)
        const xIdx = gid - ctx.numConsts - ctx.numCarry;
        cg.local.get(ctx.xsBase + xIdx);
        cg.local.get(ctx.dataIdx);
        cg.i32.const(ctx.xsStrides[xIdx]);
        cg.i32.mul();
        cg.i32.add();
      } else {
        // Internal buffer (result from previous step)
        const internalIdx = gid - ctx.numInputs;
        cg.local.get(ctx.internalsBase + internalIdx);
      }

      // Add element index offset
      gen(src[0]); // This gives the element index
      cg.i32.const(bw);
      cg.i32.mul();
      cg.i32.add();

      // Load the value
      dty(cg, op, dtype).load(Math.log2(bw));
    } else if (op === AluOp.Add) {
      gen(src[0]);
      gen(src[1]);
      if (dtype === DType.Bool) cg.i32.or();
      else dty(cg, op, dtype).add();
    } else if (op === AluOp.Sub) {
      gen(src[0]);
      gen(src[1]);
      dty(cg, op, dtype).sub();
    } else if (op === AluOp.Mul) {
      gen(src[0]);
      gen(src[1]);
      if (dtype === DType.Bool) cg.i32.and();
      else dty(cg, op, dtype).mul();
    } else if (op === AluOp.Idiv) {
      gen(src[0]);
      gen(src[1]);
      if (dtype === DType.Int32) cg.i32.div_s();
      else cg.i32.div_u();
    } else if (op === AluOp.Mod) {
      gen(src[0]);
      gen(src[1]);
      if (dtype === DType.Int32) cg.i32.rem_s();
      else cg.i32.rem_u();
    } else if (op === AluOp.Min) {
      gen(src[0]);
      gen(src[1]);
      if (isFloatDtype(dtype)) dtyF(cg, op, dtype).min();
      else throw new Error("integer min in scan not yet supported");
    } else if (op === AluOp.Max) {
      gen(src[0]);
      gen(src[1]);
      if (isFloatDtype(dtype)) dtyF(cg, op, dtype).max();
      else throw new Error("integer max in scan not yet supported");
    } else if (op === AluOp.Const) {
      dty(cg, op, dtype).const(arg);
    } else if (op === AluOp.Special && arg[0] === "gidx") {
      cg.local.get(ctx.gidx);
    } else if (op === AluOp.Special && arg[0] === "ridx") {
      if (ctx.ridx < 0)
        throw new Error("ridx used but not in reduction context");
      cg.local.get(ctx.ridx);
    } else if (op === AluOp.Variable && arg === "gidx") {
      cg.local.get(ctx.gidx);
    } else if (op === AluOp.Variable && arg === "ridx") {
      if (ctx.ridx < 0)
        throw new Error("ridx used but not in reduction context");
      cg.local.get(ctx.ridx);
    } else if (op === AluOp.Variable && arg === "acc") {
      if (ctx.acc === undefined)
        throw new Error("acc used but not in epilogue context");
      cg.local.get(ctx.acc);
    } else if (op === AluOp.Reciprocal) {
      dtyF(cg, op, dtype).const(1.0);
      gen(src[0]);
      dtyF(cg, op, dtype).div();
    } else if (op === AluOp.Sqrt) {
      gen(src[0]);
      dtyF(cg, op, dtype).sqrt();
    } else if (op === AluOp.Cmplt) {
      gen(src[0]);
      gen(src[1]);
      const srcDtype = src[0].dtype;
      if (isFloatDtype(srcDtype)) dtyF(cg, op, srcDtype).lt();
      else if (srcDtype === DType.Int32) cg.i32.lt_s();
      else cg.i32.lt_u();
    } else if (op === AluOp.Cmpne) {
      gen(src[0]);
      gen(src[1]);
      dty(cg, op, src[0].dtype).ne();
    } else if (op === AluOp.Where) {
      gen(src[1]); // true value
      gen(src[2]); // false value
      gen(src[0]); // condition
      cg.select();
    } else if (op === AluOp.Cast) {
      gen(src[0]);
      const srcDtype = src[0].dtype;
      // Handle common casts
      if (srcDtype === dtype) {
        // no-op
      } else if (srcDtype === DType.Float32 && dtype === DType.Int32) {
        cg.i32.trunc_sat_f32_s();
      } else if (srcDtype === DType.Int32 && dtype === DType.Float32) {
        cg.f32.convert_i32_s();
      } else if (srcDtype === DType.Float32 && dtype === DType.Float64) {
        cg.f64.promote_f32();
      } else if (srcDtype === DType.Float64 && dtype === DType.Float32) {
        cg.f32.demote_f64();
      } else if (srcDtype === DType.Uint32 && dtype === DType.Float32) {
        cg.f32.convert_i32_u();
      } else if (srcDtype === DType.Float32 && dtype === DType.Uint32) {
        cg.i32.trunc_sat_f32_u();
      } else {
        throw new Error(
          `Cast ${srcDtype} -> ${dtype} in scan not yet supported`,
        );
      }
    } else {
      throw new Error(
        `translateExpWithGeneralScanContext: unsupported op ${op}`,
      );
    }

    if ((references.get(e) ?? 0) > 1) {
      const local = cg.local.declare(dty(cg, op, dtype));
      cg.local.tee(local);
      expContext.set(e, local);
    }
  };

  countReferences(exp);
  gen(exp);
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
