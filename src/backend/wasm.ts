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
import { createArgsortModule } from "./wasm/argsort";
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
import { createCholeskyModule, wasm_cholesky } from "./wasm/cholesky";
import { createLUModule } from "./wasm/lu";
import { createSortModule, wasm_merge_sort } from "./wasm/sort";
import { createTriangularSolveModule } from "./wasm/triangular-solve";
import { CodeGenerator } from "./wasm/wasmblr";

interface WasmBuffer {
  ptr: number;
  size: number;
  ref: number;
}

interface WasmProgram {
  module: WebAssembly.Module;
}

/** Parameters for native scan execution. */
export interface NativeScanParams {
  /** Number of scan iterations (length of xs along axis 0). */
  length: number;
  /** Number of constant arrays (passed to body but unchanged). */
  numConsts: number;
  /** Sizes of each constant buffer in bytes. */
  constSizes: number[];
  /** Sizes of each carry buffer in bytes. */
  carrySizes: number[];
  /** Strides (in bytes) along axis 0 for each xs input. */
  xsStrides: number[];
  /** Strides (in bytes) along axis 0 for each stacked y output. */
  ysStrides: number[];
  /** The body kernel to execute each iteration. */
  bodyKernel: Kernel;
  /** Number of carry arrays. */
  numCarry: number;
  /** Whether to scan in reverse order. */
  reverse?: boolean;
}

/** Describes a single step in a multi-kernel scan body. */
export interface NativeScanStep {
  /** The kernel to execute. */
  kernel: Kernel;
  /**
   * Input mapping: indices into [consts, carry, xs] flattened.
   * For a step, these are the indices of inputs it reads from.
   */
  inputs: number[];
  /**
   * Output mapping: which carry/y slot this kernel writes to.
   * For now, simplified: one output per kernel, maps to carry index.
   */
  outputCarryIdx: number;
  /** Size of output in bytes. */
  outputSize: number;
}

/** Parameters for multi-kernel native scan execution. */
export interface NativeScanMultiParams {
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
  /** The sequence of kernels to execute each iteration. */
  steps: NativeScanStep[];
  /** Whether to scan in reverse order. */
  reverse?: boolean;
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
}

/** Describes the source for a Y output. */
export interface YOutputSource {
  /** 'passthrough' copies a carry input, 'internal' copies from internal buffer. */
  type: "passthrough" | "internal";
  /** For passthrough: index into carry inputs. */
  carryIdx?: number;
  /** For internal: index into internal buffers. */
  internalIdx?: number;
}

/** Parameters for general native scan execution (handles numCarry !== numY). */
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

/**
 * Key for routine module cache: combines routine name and element size.
 * Format: "routineName:elementSize" (e.g., "Cholesky:4" or "Cholesky:8")
 */
function routineModuleKey(name: Routines, elementSize: 4 | 8): string {
  return `${name}:${elementSize}`;
}

/** Cached WASM modules for routines (lazy-initialized), keyed by routine+dtype. */
const routineModules: Map<string, WebAssembly.Module> = new Map();

/** Module creators for each WASM-accelerated routine (now accept elementSize). */
const routineModuleCreators: Partial<
  Record<Routines, (elementSize: 4 | 8) => WebAssembly.Module>
> = {
  [Routines.Cholesky]: createCholeskyModule,
  [Routines.TriangularSolve]: createTriangularSolveModule,
  [Routines.LU]: createLUModule,
  [Routines.Sort]: createSortModule,
  [Routines.Argsort]: createArgsortModule,
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

  /** Get or create a WASM instance for a routine with specific element size. */
  #getRoutineFunc<T>(
    name: Routines,
    exportName: string,
    elementSize: 4 | 8 = 4,
  ): T {
    const key = routineModuleKey(name, elementSize);
    let instance = this.#routineInstances.get(key);
    if (!instance) {
      let module = routineModules.get(key);
      if (!module) {
        const creator = routineModuleCreators[name];
        if (!creator) throw new Error(`No WASM module for routine: ${name}`);
        module = creator(elementSize);
        routineModules.set(key, module);
      }
      instance = new WebAssembly.Instance(module, {
        env: { memory: this.#memory },
      });
      this.#routineInstances.set(key, instance);
    }
    return instance.exports[exportName] as T;
  }

  #dispatchCholesky(
    routine: Routine,
    inputs: Slot[],
    outputs: Slot[],
    elementSize: 4 | 8,
  ): void {
    const func = this.#getRoutineFunc<
      (i: number, o: number, n: number, b: number) => void
    >(Routines.Cholesky, "cholesky", elementSize);
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
    const func = this.#getRoutineFunc<
      (
        a: number,
        b: number,
        x: number,
        n: number,
        rows: number,
        batches: number,
        unit: number,
        lower: number,
      ) => void
    >(Routines.TriangularSolve, "triangularSolve", elementSize);
    const aShape = routine.type.inputShapes[0];
    const bShape = routine.type.inputShapes[1];
    func(
      this.#buffers.get(inputs[0])!.ptr,
      this.#buffers.get(inputs[1])!.ptr,
      this.#buffers.get(outputs[0])!.ptr,
      aShape[aShape.length - 1],
      bShape[bShape.length - 2],
      aShape.slice(0, -2).reduce((a, b) => a * b, 1),
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
    const func = this.#getRoutineFunc<
      (
        i: number,
        lu: number,
        piv: number,
        perm: number,
        m: number,
        n: number,
        b: number,
      ) => void
    >(Routines.LU, "lu", elementSize);
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
    const func = this.#getRoutineFunc<
      (i: number, o: number, aux: number, n: number, b: number) => void
    >(Routines.Sort, "sort", elementSize);
    const shape = routine.type.inputShapes[0];
    const n = shape[shape.length - 1];
    const auxPtr = this.#allocator.malloc(n * elementSize);
    func(
      this.#buffers.get(inputs[0])!.ptr,
      this.#buffers.get(outputs[0])!.ptr,
      auxPtr,
      n,
      shape.slice(0, -1).reduce((a, b) => a * b, 1),
    );
    this.#allocator.free(auxPtr);
  }

  #dispatchArgsort(
    routine: Routine,
    inputs: Slot[],
    outputs: Slot[],
    elementSize: 4 | 8,
  ): void {
    const func = this.#getRoutineFunc<
      (
        i: number,
        oD: number,
        oI: number,
        aD: number,
        aI: number,
        n: number,
        b: number,
      ) => void
    >(Routines.Argsort, "argsort", elementSize);
    const shape = routine.type.inputShapes[0];
    const n = shape[shape.length - 1];
    // auxData uses elementSize, auxIdx uses 4 (always i32)
    const auxDataPtr = this.#allocator.malloc(n * elementSize);
    const auxIdxPtr = this.#allocator.malloc(n * 4);
    func(
      this.#buffers.get(inputs[0])!.ptr,
      this.#buffers.get(outputs[0])!.ptr,
      this.#buffers.get(outputs[1])!.ptr,
      auxDataPtr,
      auxIdxPtr,
      n,
      shape.slice(0, -1).reduce((a, b) => a * b, 1),
    );
    this.#allocator.free(auxDataPtr);
    this.#allocator.free(auxIdxPtr);
  }

  /**
   * Prepare a native scan operation for efficient execution.
   * Returns null if the scan cannot be natively executed (e.g., has routines).
   */
  prepareNativeScan(params: NativeScanParams): Executable<WasmProgram> | null {
    // For now, only support scans with a single body kernel (no routines)
    const { bodyKernel } = params;
    if (!bodyKernel) return null;

    try {
      const bytes = codegenNativeScan(params);
      const module = new WebAssembly.Module(bytes);
      // Use a synthetic Kernel as the source for the Executable
      const syntheticKernel = new Kernel(
        bodyKernel.nargs,
        bodyKernel.size,
        bodyKernel.exp,
        bodyKernel.reduction,
      );
      return new Executable(syntheticKernel, { module });
    } catch (e) {
      if (DEBUG >= 2) {
        console.warn("Native scan codegen failed:", e);
      }
      return null;
    }
  }

  /**
   * Dispatch a native scan operation.
   * @param exe - The prepared native scan executable
   * @param consts - Constant buffer slots (unchanged across iterations)
   * @param initCarry - Initial carry buffer slots
   * @param xs - Input xs buffer slots
   * @param carryOut - Output carry buffer slots
   * @param ysStacked - Output stacked ys buffer slots
   */
  dispatchNativeScan(
    exe: Executable<WasmProgram>,
    consts: Slot[],
    initCarry: Slot[],
    xs: Slot[],
    carryOut: Slot[],
    ysStacked: Slot[],
  ): void {
    let instance = this.#instanceCache.get(exe.data.module);
    if (!instance) {
      instance = new WebAssembly.Instance(exe.data.module, {
        env: { memory: this.#memory },
      });
      this.#instanceCache.set(exe.data.module, instance);
    }
    const func = instance.exports.scan as (...args: number[]) => void;
    const ptrs = [
      ...consts.map((slot) => this.#buffers.get(slot)!.ptr),
      ...initCarry.map((slot) => this.#buffers.get(slot)!.ptr),
      ...xs.map((slot) => this.#buffers.get(slot)!.ptr),
      ...carryOut.map((slot) => this.#buffers.get(slot)!.ptr),
      ...ysStacked.map((slot) => this.#buffers.get(slot)!.ptr),
    ];
    func(...ptrs);
  }

  /**
   * Prepare a multi-kernel native scan operation for efficient execution.
   * Handles scan bodies with multiple independent kernels (e.g., 2 matmuls).
   */
  prepareNativeScanMulti(
    params: NativeScanMultiParams,
  ): Executable<WasmProgram> | null {
    if (params.steps.length === 0) return null;

    try {
      const bytes = codegenNativeScanMulti(params);
      const module = new WebAssembly.Module(bytes);
      // Use the first kernel as the source for the Executable (just for typing)
      const firstKernel = params.steps[0].kernel;
      const syntheticKernel = new Kernel(
        firstKernel.nargs,
        firstKernel.size,
        firstKernel.exp,
        firstKernel.reduction,
      );
      return new Executable(syntheticKernel, { module });
    } catch (e) {
      if (DEBUG >= 2) {
        console.warn("Multi-kernel native scan codegen failed:", e);
      }
      return null;
    }
  }

  /**
   * Prepare a general native scan operation for efficient execution.
   * Handles scan bodies with data dependencies and numCarry !== numY.
   * Also supports Routine steps (Cholesky, Sort) mixed with Kernel steps.
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
        instance = new WebAssembly.Instance(exe.data.module, {
          env: { memory: this.#memory },
        });
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

/**
 * Generate WASM bytecode for a native scan operation.
 *
 * The generated function takes pointers to:
 *   [...initCarry, ...xs, ...carryOut, ...ysStacked]
 *
 * Memory layout:
 *   - initCarry[i]: buffer of size carrySizes[i]
 *   - xs[i]: buffer of size length * xsStrides[i]
 *   - carryOut[i]: buffer of size carrySizes[i]
 *   - ysStacked[i]: buffer of size length * ysStrides[i]
 *
 * Algorithm:
 *   1. Copy initCarry to working carry buffers (use carryOut as working buffer)
 *   2. For iter = 0..length:
 *        - Compute x_ptr = xs[i] + iter * xsStrides[i]
 *        - Compute y_ptr = ysStacked[i] + iter * ysStrides[i]
 *        - Execute body kernel reading from carry and x, writing to y
 *        - Copy appropriate outputs to carry for next iteration
 *   3. Final carry is already in carryOut
 */
function codegenNativeScan(params: NativeScanParams): Uint8Array<ArrayBuffer> {
  const {
    length,
    numConsts,
    constSizes,
    carrySizes,
    xsStrides,
    ysStrides,
    bodyKernel,
    numCarry,
    reverse,
  } = params;

  const tune = tuneNullopt(bodyKernel);
  const re = bodyKernel.reduction;

  if (DEBUG >= 2) {
    console.log("codegenNativeScan params:", {
      length,
      numConsts,
      numCarry,
      constSizes,
      carrySizes,
      xsStrides,
      ysStrides,
      bodyKernelNargs: bodyKernel.nargs,
      bodyKernelSize: bodyKernel.size,
      hasReduction: !!re,
      reductionSize: re?.size,
      reductionOp: re?.op,
      reverse,
    });
  }

  const cg = new CodeGenerator();
  cg.memory.import("env", "memory");

  // Import helper functions needed by the body kernel (include epilogue ops)
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

  // Function arguments:
  // [...consts (numConsts), ...initCarry (numCarry), ...xs (numX), ...carryOut (numCarry), ...ysStacked (numY)]
  const numX = xsStrides.length;
  const numY = ysStrides.length;
  const numArgs = numConsts + numCarry + numX + numCarry + numY;

  const scanFunc = cg.function(rep(numArgs, cg.i32), [], () => {
    // Local variables
    const iter = cg.local.declare(cg.i32); // scan iteration counter
    const gidx = cg.local.declare(cg.i32); // body kernel loop index
    const dataIdx = cg.local.declare(cg.i32); // data index for xs/ys (differs from iter when reverse)

    // Argument indices
    const constsBase = 0;
    const initCarryBase = numConsts;
    const xsBase = numConsts + numCarry;
    const carryOutBase = numConsts + numCarry + numX;
    const ysStackedBase = numConsts + numCarry + numX + numCarry;

    // Step 1: Copy initCarry to carryOut (working buffer)
    for (let c = 0; c < numCarry; c++) {
      const size = carrySizes[c];
      // Simple byte-by-byte copy (could optimize with memory.copy if available)
      const copyIdx = cg.local.declare(cg.i32);
      cg.i32.const(0);
      cg.local.set(copyIdx);
      cg.loop(cg.void);
      {
        cg.block(cg.void);
        cg.local.get(copyIdx);
        cg.i32.const(size);
        cg.i32.ge_u();
        cg.br_if(0);

        // carryOut[c][copyIdx] = initCarry[c][copyIdx]
        cg.local.get(carryOutBase + c); // dest base
        cg.local.get(copyIdx);
        cg.i32.add();
        cg.local.get(initCarryBase + c); // src base
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

      // Precompute xs pointers: xsPtrs[i] = xsBase[i] + dataIdx * xsStrides[i]
      // This hoists the offset calculation out of the inner reduction loop.
      const xsPtrs: number[] = [];
      for (let xi = 0; xi < numX; xi++) {
        const xsPtr = cg.local.declare(cg.i32);
        cg.local.get(xsBase + xi); // base pointer
        cg.local.get(dataIdx);
        cg.i32.const(xsStrides[xi]);
        cg.i32.mul();
        cg.i32.add();
        cg.local.set(xsPtr);
        xsPtrs.push(xsPtr);
      }

      // Also precompute ys output pointer
      const ysOutPtr = cg.local.declare(cg.i32);
      cg.local.get(ysStackedBase); // ysStacked[0] base ptr
      cg.local.get(dataIdx);
      cg.i32.const(ysStrides[0]);
      cg.i32.mul();
      cg.i32.add();
      cg.local.set(ysOutPtr);

      // Inner loop over kernel output elements
      cg.i32.const(0);
      cg.local.set(gidx);
      cg.loop(cg.void);
      {
        cg.block(cg.void);
        cg.local.get(gidx);
        cg.i32.const(bodyKernel.size);
        cg.i32.ge_u();
        cg.br_if(0);

        // Compute output address: ysOutPtr + gidx * elementSize
        cg.local.get(ysOutPtr);
        cg.local.get(gidx);
        cg.i32.const(byteWidth(bodyKernel.dtype));
        cg.i32.mul();
        cg.i32.add();

        // Context for translateExpWithScanContext
        const scanCtx = {
          gidx,
          iter,
          dataIdx,
          ridx: -1, // Will be set if reduction
          constsBase,
          constSizes,
          numConsts,
          xsBase,
          xsStrides,
          xsPtrs, // Precomputed xs pointers (optimization)
          carryBase: carryOutBase, // read from carryOut as working buffer
          carrySizes,
          numCarry,
        };

        if (re) {
          // Reduction: define accumulator and inner ridx loop
          const acc = cg.local.declare(dty(cg, null, bodyKernel.exp.dtype));
          dty(cg, null, bodyKernel.exp.dtype).const(re.identity);
          cg.local.set(acc);

          const ridx = cg.local.declare(cg.i32);
          cg.i32.const(0);
          cg.local.set(ridx);
          scanCtx.ridx = ridx;

          cg.loop(cg.void);
          {
            // if (ridx >= reduction.size) break;
            cg.block(cg.void);
            cg.local.get(ridx);
            cg.i32.const(re.size);
            cg.i32.ge_u();
            cg.br_if(0);

            // Translate tune.exp and push onto stack
            translateExpWithScanContext(cg, funcs, tune.exp, scanCtx);

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
                // Wasm has no i32.min/max, so emulate with select
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

            // ridx++
            cg.local.get(ridx);
            cg.i32.const(1);
            cg.i32.add();
            cg.local.set(ridx);

            cg.br(1); // continue ridx loop
            cg.end();
          }
          cg.end();

          // Apply epilogue: uses acc and gidx
          translateExpWithScanContext(cg, funcs, tune.epilogue!, {
            ...scanCtx,
            acc,
          });
        } else {
          // No reduction: just translate the expression
          translateExpWithScanContext(cg, funcs, tune.exp, scanCtx);
        }

        // Store result to ysStacked (address already on stack from above)
        dty(cg, null, bodyKernel.dtype).store(
          Math.log2(byteWidth(bodyKernel.dtype)),
        );

        // Also update carry for next iteration: carryOut[0] + gidx * elementSize
        cg.local.get(carryOutBase);
        cg.local.get(gidx);
        cg.i32.const(byteWidth(bodyKernel.dtype));
        cg.i32.mul();
        cg.i32.add();

        // Re-compute the value for carry update (same logic as above)
        if (re) {
          const acc2 = cg.local.declare(dty(cg, null, bodyKernel.exp.dtype));
          dty(cg, null, bodyKernel.exp.dtype).const(re.identity);
          cg.local.set(acc2);

          const ridx2 = cg.local.declare(cg.i32);
          cg.i32.const(0);
          cg.local.set(ridx2);
          const scanCtx2 = { ...scanCtx, ridx: ridx2 };

          cg.loop(cg.void);
          {
            cg.block(cg.void);
            cg.local.get(ridx2);
            cg.i32.const(re.size);
            cg.i32.ge_u();
            cg.br_if(0);

            translateExpWithScanContext(cg, funcs, tune.exp, scanCtx2);

            if (re.op === AluOp.Add) {
              cg.local.get(acc2);
              if (re.dtype === DType.Bool) cg.i32.or();
              else dty(cg, re.op, re.dtype).add();
            } else if (re.op === AluOp.Mul) {
              cg.local.get(acc2);
              if (re.dtype === DType.Bool) cg.i32.and();
              else dty(cg, re.op, re.dtype).mul();
            } else if (re.op === AluOp.Min || re.op === AluOp.Max) {
              if (isFloatDtype(re.dtype)) {
                cg.local.get(acc2);
                if (re.op === AluOp.Min) dtyF(cg, re.op, re.dtype).min();
                else dtyF(cg, re.op, re.dtype).max();
              } else if (
                [DType.Int32, DType.Uint32, DType.Bool].includes(re.dtype)
              ) {
                const local = cg.local.declare(cg.i32);
                cg.local.tee(local);
                cg.local.get(acc2);
                cg.local.get(local);
                cg.local.get(acc2);
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
            cg.local.set(acc2);

            cg.local.get(ridx2);
            cg.i32.const(1);
            cg.i32.add();
            cg.local.set(ridx2);

            cg.br(1);
            cg.end();
          }
          cg.end();

          translateExpWithScanContext(cg, funcs, tune.epilogue!, {
            ...scanCtx2,
            acc: acc2,
          });
        } else {
          translateExpWithScanContext(cg, funcs, tune.exp, scanCtx);
        }

        dty(cg, null, bodyKernel.dtype).store(
          Math.log2(byteWidth(bodyKernel.dtype)),
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

/**
 * Generate WASM bytecode for multi-kernel native scan.
 * Each iteration executes multiple kernels, each writing to its own carry buffer.
 */
function codegenNativeScanMulti(
  params: NativeScanMultiParams,
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
    steps,
    reverse,
  } = params;

  if (DEBUG >= 2) {
    console.log("codegenNativeScanMulti params:", {
      length,
      numConsts,
      numCarry,
      numX,
      numY,
      numSteps: steps.length,
      reverse,
    });
  }

  const cg = new CodeGenerator();
  cg.memory.import("env", "memory");

  // Collect all helper functions needed by all kernels
  const allOps = new Set<AluOp>();
  for (const step of steps) {
    const tune = tuneNullopt(step.kernel);
    for (const op of tune.exp.distinctOps().keys()) allOps.add(op);
    if (tune.epilogue) {
      for (const op of tune.epilogue.distinctOps().keys()) allOps.add(op);
    }
  }

  const funcs: Record<string, number> = {};
  if (allOps.has(AluOp.Sin)) funcs.sin = wasm_sin(cg);
  if (allOps.has(AluOp.Cos)) funcs.cos = wasm_cos(cg);
  if (allOps.has(AluOp.Asin)) funcs.asin = wasm_asin(cg);
  if (allOps.has(AluOp.Atan)) funcs.atan = wasm_atan(cg);
  if (allOps.has(AluOp.Exp) || allOps.has(AluOp.Erf) || allOps.has(AluOp.Erfc))
    funcs.exp = wasm_exp(cg);
  if (allOps.has(AluOp.Log)) funcs.log = wasm_log(cg);
  if (allOps.has(AluOp.Erf)) funcs.erf = wasm_erf(cg, funcs.exp);
  if (allOps.has(AluOp.Erfc)) funcs.erfc = wasm_erfc(cg, funcs.exp);
  if (allOps.has(AluOp.Threefry2x32))
    funcs.threefry2x32 = wasm_threefry2x32(cg);

  // Function arguments:
  // [...consts (numConsts), ...initCarry (numCarry), ...xs (numX), ...carryOut (numCarry), ...ysStacked (numY)]
  const numArgs = numConsts + numCarry + numX + numCarry + numY;

  const scanFunc = cg.function(rep(numArgs, cg.i32), [], () => {
    // Local variables
    const iter = cg.local.declare(cg.i32); // scan iteration counter
    const gidx = cg.local.declare(cg.i32); // body kernel loop index
    const dataIdx = cg.local.declare(cg.i32); // data index for xs/ys (differs from iter when reverse)

    // Argument indices
    const constsBase = 0;
    const initCarryBase = numConsts;
    const xsBase = numConsts + numCarry;
    const carryOutBase = numConsts + numCarry + numX;
    const ysStackedBase = numConsts + numCarry + numX + numCarry;

    // Step 1: Copy initCarry to carryOut (working buffer)
    for (let c = 0; c < numCarry; c++) {
      const size = carrySizes[c];
      const copyIdx = cg.local.declare(cg.i32);
      cg.i32.const(0);
      cg.local.set(copyIdx);
      cg.loop(cg.void);
      {
        cg.block(cg.void);
        cg.local.get(copyIdx);
        cg.i32.const(size);
        cg.i32.ge_u();
        cg.br_if(0);

        // carryOut[c][copyIdx] = initCarry[c][copyIdx]
        cg.local.get(carryOutBase + c);
        cg.local.get(copyIdx);
        cg.i32.add();
        cg.local.get(initCarryBase + c);
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

      // Execute each kernel step
      for (const step of steps) {
        const kernel = step.kernel;
        const carryIdx = step.outputCarryIdx;
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

          // Compute output address: ysStacked[carryIdx] + dataIdx * ysStrides[carryIdx] + gidx * elementSize
          cg.local.get(ysStackedBase + carryIdx);
          cg.local.get(dataIdx);
          cg.i32.const(ysStrides[carryIdx]);
          cg.i32.mul();
          cg.i32.add();
          cg.local.get(gidx);
          cg.i32.const(byteWidth(kernel.dtype));
          cg.i32.mul();
          cg.i32.add();

          // Context for translateExpWithScanContext
          const scanCtx = {
            gidx,
            iter,
            dataIdx,
            ridx: -1,
            constsBase,
            constSizes,
            numConsts,
            xsBase,
            xsStrides,
            carryBase: carryOutBase,
            carrySizes,
            numCarry,
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

              translateExpWithScanContext(cg, funcs, tune.exp, scanCtx);

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
                }
              } else {
                throw new Error(`invalid wasm reduction op: ${re.op}`);
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

            // Apply epilogue
            translateExpWithScanContext(cg, funcs, tune.epilogue!, {
              ...scanCtx,
              acc,
            });
          } else {
            translateExpWithScanContext(cg, funcs, tune.exp, scanCtx);
          }

          // Store result to ysStacked (address already on stack)
          dty(cg, null, kernel.dtype).store(Math.log2(byteWidth(kernel.dtype)));

          // Also update carry for next iteration
          cg.local.get(carryOutBase + carryIdx);
          cg.local.get(gidx);
          cg.i32.const(byteWidth(kernel.dtype));
          cg.i32.mul();
          cg.i32.add();

          // Re-compute the value for carry update (same logic as above)
          if (re) {
            const acc2 = cg.local.declare(dty(cg, null, kernel.exp.dtype));
            dty(cg, null, kernel.exp.dtype).const(re.identity);
            cg.local.set(acc2);

            const ridx2 = cg.local.declare(cg.i32);
            cg.i32.const(0);
            cg.local.set(ridx2);
            const scanCtx2 = { ...scanCtx, ridx: ridx2 };

            cg.loop(cg.void);
            {
              cg.block(cg.void);
              cg.local.get(ridx2);
              cg.i32.const(re.size);
              cg.i32.ge_u();
              cg.br_if(0);

              translateExpWithScanContext(cg, funcs, tune.exp, scanCtx2);

              if (re.op === AluOp.Add) {
                cg.local.get(acc2);
                if (re.dtype === DType.Bool) cg.i32.or();
                else dty(cg, re.op, re.dtype).add();
              } else if (re.op === AluOp.Mul) {
                cg.local.get(acc2);
                if (re.dtype === DType.Bool) cg.i32.and();
                else dty(cg, re.op, re.dtype).mul();
              } else if (re.op === AluOp.Min || re.op === AluOp.Max) {
                if (isFloatDtype(re.dtype)) {
                  cg.local.get(acc2);
                  if (re.op === AluOp.Min) dtyF(cg, re.op, re.dtype).min();
                  else dtyF(cg, re.op, re.dtype).max();
                } else if (
                  [DType.Int32, DType.Uint32, DType.Bool].includes(re.dtype)
                ) {
                  const local = cg.local.declare(cg.i32);
                  cg.local.tee(local);
                  cg.local.get(acc2);
                  cg.local.get(local);
                  cg.local.get(acc2);
                  if (re.op === AluOp.Min) {
                    if (re.dtype === DType.Int32) cg.i32.lt_s();
                    else cg.i32.lt_u();
                  } else {
                    if (re.dtype === DType.Int32) cg.i32.gt_s();
                    else cg.i32.gt_u();
                  }
                  cg.select();
                }
              }
              cg.local.set(acc2);

              cg.local.get(ridx2);
              cg.i32.const(1);
              cg.i32.add();
              cg.local.set(ridx2);

              cg.br(1);
              cg.end();
            }
            cg.end();

            translateExpWithScanContext(cg, funcs, tune.epilogue!, {
              ...scanCtx2,
              acc: acc2,
            });
          } else {
            translateExpWithScanContext(cg, funcs, tune.exp, scanCtx);
          }

          dty(cg, null, kernel.dtype).store(Math.log2(byteWidth(kernel.dtype)));

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
    });
  }

  const cg = new CodeGenerator();
  cg.memory.import("env", "memory");

  // Determine element size from first kernel or routine
  const elementSize = params.elementSize ?? 4;
  const ft = elementSize === 4 ? cg.f32 : cg.f64;

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

  const funcs: Record<string, number> = {};
  if (allOps.has(AluOp.Sin)) funcs.sin = wasm_sin(cg);
  if (allOps.has(AluOp.Cos)) funcs.cos = wasm_cos(cg);
  if (allOps.has(AluOp.Asin)) funcs.asin = wasm_asin(cg);
  if (allOps.has(AluOp.Atan)) funcs.atan = wasm_atan(cg);
  if (allOps.has(AluOp.Exp) || allOps.has(AluOp.Erf) || allOps.has(AluOp.Erfc))
    funcs.exp = wasm_exp(cg);
  if (allOps.has(AluOp.Log)) funcs.log = wasm_log(cg);
  if (allOps.has(AluOp.Erf)) funcs.erf = wasm_erf(cg, funcs.exp);
  if (allOps.has(AluOp.Erfc)) funcs.erfc = wasm_erfc(cg, funcs.exp);
  if (allOps.has(AluOp.Threefry2x32))
    funcs.threefry2x32 = wasm_threefry2x32(cg);

  // Add routine functions for any routine steps
  const routineFuncs: Map<Routines, number> = new Map();
  for (const step of steps) {
    if (step.source instanceof Routine) {
      const routineName = step.source.name as Routines;
      if (!routineFuncs.has(routineName)) {
        if (routineName === Routines.Cholesky) {
          routineFuncs.set(routineName, wasm_cholesky(cg, ft));
        } else if (routineName === Routines.Sort) {
          routineFuncs.set(routineName, wasm_merge_sort(cg, ft));
        }
      }
    }
  }

  // Check if we need aux buffer (for Sort)
  const needsAuxBuffer = params.auxBufferSize && params.auxBufferSize > 0;

  // Function arguments:
  // [...consts (numConsts), ...carryIn (numCarry), ...xs (numX),
  //  ...carryOut (numCarry), ...ysStacked (numY), ...internals (numInternal), aux?]
  const numArgs =
    numConsts +
    numCarry +
    numX +
    numCarry +
    numY +
    numInternal +
    (needsAuxBuffer ? 1 : 0);
  const auxArgIdx = needsAuxBuffer
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

      // Step 2a: Execute each step, writing to internal buffers
      for (let stepIdx = 0; stepIdx < steps.length; stepIdx++) {
        const step = steps[stepIdx];
        const source = step.source;
        const internalIdx = step.outputInternalIdx;

        if (source instanceof Routine) {
          // Routine step: call the routine function
          const routineName = source.name as Routines;
          const routineFn = routineFuncs.get(routineName);
          if (routineFn === undefined) {
            throw new Error(
              `Routine ${source.name} not supported in native scan`,
            );
          }

          // Get input pointer from inputSlots[0]
          // The input comes from either a jaxpr input or an internal buffer
          const inputSlot = step.inputSlots[0];
          if (inputSlot < numConsts) {
            // Constant input
            cg.local.get(constsBase + inputSlot);
          } else if (inputSlot < numConsts + numCarry) {
            // Carry input (read from carryOut which has current values)
            cg.local.get(carryOutBase + (inputSlot - numConsts));
          } else if (inputSlot < numConsts + numCarry + numX) {
            // xs input - need to add dataIdx * stride offset
            const xIdx = inputSlot - numConsts - numCarry;
            cg.local.get(xsBase + xIdx);
            cg.local.get(dataIdx);
            cg.i32.const(xsStrides[xIdx]);
            cg.i32.mul();
            cg.i32.add();
          } else {
            // Internal buffer from previous step
            const prevInternalIdx = inputSlot - numConsts - numCarry - numX;
            cg.local.get(internalsBase + prevInternalIdx);
          }

          // Get output pointer (internal buffer)
          cg.local.get(internalsBase + internalIdx);

          if (routineName === Routines.Cholesky) {
            // Cholesky: (inPtr, outPtr, n)
            const inputShape = source.type.inputShapes[0];
            const n = inputShape[inputShape.length - 1];
            cg.i32.const(n);
            cg.call(routineFn);
          } else if (routineName === Routines.Sort) {
            // Sort: copy input to output, then sort in place
            // First copy input to output
            const inputShape = source.type.inputShapes[0];
            const n = inputShape[inputShape.length - 1];

            // Save pointers to locals for copy loop
            const inPtrLocal = cg.local.declare(cg.i32);
            const outPtrLocal = cg.local.declare(cg.i32);
            cg.local.set(outPtrLocal); // outPtr is on stack
            cg.local.set(inPtrLocal); // inPtr is on stack

            // Copy loop: out[i] = in[i] for i in 0..n
            cg.i32.const(0);
            cg.local.set(copyIdx);
            cg.loop(cg.void);
            {
              cg.block(cg.void);
              cg.local.get(copyIdx);
              cg.i32.const(n);
              cg.i32.ge_u();
              cg.br_if(0);

              // out[copyIdx] = in[copyIdx]
              cg.local.get(outPtrLocal);
              cg.local.get(copyIdx);
              cg.i32.const(elementSize);
              cg.i32.mul();
              cg.i32.add();

              cg.local.get(inPtrLocal);
              cg.local.get(copyIdx);
              cg.i32.const(elementSize);
              cg.i32.mul();
              cg.i32.add();
              ft.load(0, 0);

              ft.store(0, 0);

              cg.local.get(copyIdx);
              cg.i32.const(1);
              cg.i32.add();
              cg.local.set(copyIdx);

              cg.br(1);
              cg.end();
            }
            cg.end();

            // Now sort in place: sort(outPtr, auxPtr, n)
            cg.local.get(outPtrLocal);
            cg.local.get(auxArgIdx); // aux buffer
            cg.i32.const(n);
            cg.call(routineFn);
          }
        } else {
          // Kernel step: original logic
          const kernel = source;
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
                  }
                } else {
                  throw new Error(`invalid wasm reduction op: ${re.op}`);
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
        } // end else (Kernel step)
      }

      // Step 2b: Copy Y outputs to ysStacked at iteration offset
      // NOTE: Must run BEFORE carry update (2c) so passthrough reads OLD carry values
      for (let y = 0; y < numY; y++) {
        const source = yOutputSources[y];
        const yStride = ysStrides[y];

        // Determine source pointer and size
        let srcArgIdx: number;
        let size: number;
        if (source.type === "passthrough") {
          // Read from carryOut (has carry values entering this iteration)
          srcArgIdx = carryOutBase + source.carryIdx!;
          size = carrySizes[source.carryIdx!];
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

          // src[copyIdx]
          cg.local.get(srcArgIdx);
          cg.local.get(copyIdx);
          cg.i32.add();
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

/**
 * Translate an AluExp to WASM code within a scan context.
 * This differs from translateExp by handling GlobalView with iteration-dependent offsets
 * and supporting ridx/acc for reductions.
 */
function translateExpWithScanContext(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  exp: AluExp,
  ctx: {
    gidx: number;
    iter: number;
    dataIdx: number; // Iteration index for data access (differs from iter when reverse=true)
    ridx?: number; // Reduction index variable (if reduction)
    acc?: number; // Accumulator variable (for epilogue)
    constsBase: number;
    constSizes: number[];
    numConsts: number;
    xsBase: number;
    xsStrides: number[];
    xsPtrs?: number[]; // Optional precomputed xs pointers (optimization)
    carryBase: number;
    carrySizes: number[];
    numCarry: number;
  },
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

    // Handle GlobalView specially for scan context
    // Body jaxpr input layout: [consts..., carry..., xs...]
    // gid 0..numConsts-1 → constants (no iteration offset)
    // gid numConsts..numConsts+numCarry-1 → carry
    // gid numConsts+numCarry.. → xs (with iteration offset)
    // Note: GlobalView is converted to GlobalIndex by tuneNullopt's rewriteGlobalViews()

    // Handle other ops - subset supported for MVP
    if (op === AluOp.Add) {
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
      if (ctx.ridx === undefined)
        throw new Error("ridx used but not in reduction context");
      cg.local.get(ctx.ridx);
    } else if (op === AluOp.Variable && arg === "gidx") {
      cg.local.get(ctx.gidx);
    } else if (op === AluOp.Variable && arg === "ridx") {
      if (ctx.ridx === undefined)
        throw new Error("ridx used but not in reduction context");
      cg.local.get(ctx.ridx);
    } else if (op === AluOp.Variable && arg === "acc") {
      if (ctx.acc === undefined)
        throw new Error("acc used but not in epilogue context");
      cg.local.get(ctx.acc);
    } else if (op === AluOp.GlobalIndex) {
      // GlobalIndex: arg = [gid, len], src = [bufidx]
      // Load from buffer at gidx position
      // gid follows same layout as GlobalView: [consts..., carry..., xs...]
      const gid = arg[0] as number;
      const bw = byteWidth(dtype);

      if (gid < ctx.numConsts) {
        // Constant input (no iteration offset)
        cg.local.get(ctx.constsBase + gid);
      } else if (gid < ctx.numConsts + ctx.numCarry) {
        // Carry input
        const carryIdx = gid - ctx.numConsts;
        cg.local.get(ctx.carryBase + carryIdx);
      } else {
        // X input with iteration offset
        const xIdx = gid - ctx.numConsts - ctx.numCarry;
        // Use precomputed pointer if available (optimization)
        if (ctx.xsPtrs && ctx.xsPtrs[xIdx] !== undefined) {
          cg.local.get(ctx.xsPtrs[xIdx]);
        } else {
          // Fallback: compute offset in inner loop (slower)
          cg.local.get(ctx.xsBase + xIdx);
          cg.local.get(ctx.dataIdx);
          cg.i32.const(ctx.xsStrides[xIdx]);
          cg.i32.mul();
          cg.i32.add();
        }
      }

      // Add gidx * bytewidth offset
      gen(src[0]); // This gives the element index
      cg.i32.const(bw);
      cg.i32.mul();
      cg.i32.add();

      // Load the value
      dty(cg, op, dtype).load(Math.log2(bw));
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
      } else {
        throw new Error(
          `Cast ${srcDtype} -> ${dtype} in scan not yet supported`,
        );
      }
    } else {
      throw new Error(`translateExpWithScanContext: unsupported op ${op}`);
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
