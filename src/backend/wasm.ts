import { Kernel } from "../alu";
import { Backend, Device, Executable, Slot, SlotError } from "../backend";
import { Routine, runCpuRoutine } from "../routine";
import { emitTrace, isTracing, traceSourceInfo } from "../tracing";
import { FpHash, runWithCache, runWithCacheAsync } from "../utils";
import { WasmAllocator } from "./wasm/allocator";
import { codegenWasm } from "./wasm/codegen";
import {
  createWorkerPool,
  hasSharedArrayBuffer,
  WasmWorkerPool,
} from "./wasm/parallel";

interface WasmBuffer {
  ptr: number;
  size: number;
  ref: number;
}

interface WasmExecutableData {
  program: WasmProgram;
  sync: boolean;
}

interface WasmProgram {
  module: WebAssembly.Module;
  workSize: number;
  chunkAlignment: number;
  minWorkPerWorker: number;
}

const compiledProgramCache = new Map<string, WasmProgram>();

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
      this.#pendingWork.delete(slot);
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
    if (hasSharedArrayBuffer() && buffer.buffer instanceof SharedArrayBuffer) {
      // For SharedArrayBuffer, we need to copy the data to ArrayBuffer.
      return new Uint8Array(buffer.slice(start, start + count));
    } else {
      return buffer.slice(start, start + count);
    }
  }

  async prepareKernel(kernel: Kernel): Promise<Executable<WasmExecutableData>> {
    const kernelHash = FpHash.hash(kernel);
    const hashKey = kernelHash.toString();
    const program = await runWithCacheAsync(
      compiledProgramCache,
      hashKey,
      async () => {
        const { bytes, ...metadata } = codegenWasm(kernel);
        const module = await WebAssembly.compile(bytes);
        return { module, ...metadata };
      },
    );
    return new Executable(kernel, { program, sync: false });
  }

  prepareKernelSync(kernel: Kernel): Executable<WasmExecutableData> {
    const kernelHash = FpHash.hash(kernel);
    const hashKey = kernelHash.toString();
    const compiled = runWithCache(compiledProgramCache, hashKey, () => {
      const { bytes, ...metadata } = codegenWasm(kernel);
      const module = new WebAssembly.Module(bytes);
      return { module, ...metadata };
    });
    return new Executable(kernel, { program: compiled, sync: true });
  }

  async prepareRoutine(
    routine: Routine,
  ): Promise<Executable<WasmExecutableData>> {
    return this.prepareRoutineSync(routine);
  }

  prepareRoutineSync(routine: Routine): Executable<WasmExecutableData> {
    // Currently, Wasm routines fall back to the CPU reference implementation
    // implementation. We may optimize this in the future.
    return new Executable(routine, {
      program: undefined!,
      sync: true,
    });
  }

  dispatch(
    exe: Executable<WasmExecutableData>,
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
      const { program, sync } = exe.data;
      const ptrs = [...inputs, ...outputs].map(
        (slot) => this.#buffers.get(slot)!.ptr,
      );
      if (this.#workerPool && !sync) {
        const retainedSlots = [...inputs, ...outputs];
        for (const slot of retainedSlots) this.incRef(slot);
        const epoch = this.#workerPool.dispatch(
          program.module,
          ptrs,
          program.workSize,
          program.chunkAlignment,
          program.minWorkPerWorker,
        );
        for (const slot of outputs) this.#pendingWork.set(slot, epoch);
        this.#workerPool.waitForEpoch(epoch).then(() => {
          for (const slot of outputs) {
            if (this.#pendingWork.get(slot) === epoch) {
              this.#pendingWork.delete(slot);
            }
          }
          for (const slot of retainedSlots) this.decRef(slot);
        });
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
        const instance = new WebAssembly.Instance(program.module, {
          env: { memory: this.#memory },
        });
        const func = instance.exports.kernel as (...args: number[]) => void;
        func(...ptrs, 0, program.workSize);
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
