import { AluOp, dtypedArray, Kernel } from "../alu";
import { Backend, Device, Executable, Slot, SlotError } from "../backend";
import { tuneNullopt } from "../tuner";

interface WasmBuffer {
  ptr: number;
  size: number;
  ref: number;
}

/** Backend that compiles into WebAssembly bytecode for immediate execution. */
export class WasmBackend implements Backend {
  readonly type: Device = "wasm";
  readonly maxArgs = Infinity;

  #memory: WebAssembly.Memory;
  #nextSlot: number;
  #headPtr: number; // first free byte in memory
  #buffers: Map<Slot, WasmBuffer>;

  constructor() {
    // 4 GiB = max memory32 size
    // https://spidermonkey.dev/blog/2025/01/15/is-memory64-actually-worth-using.html
    this.#memory = new WebAssembly.Memory({ initial: 65536 });
    this.#nextSlot = 1;
    this.#headPtr = 0;
    this.#buffers = new Map();
  }

  malloc(size: number, initialData?: ArrayBuffer): Slot {
    const ptr = this.#headPtr;
    if (initialData) {
      if (initialData.byteLength !== size) {
        throw new Error("initialData size does not match buffer size");
      }
      new Uint8Array(this.#memory.buffer, ptr, size).set(
        new Uint8Array(initialData),
      );
    }

    const slot = this.#nextSlot++;
    this.#buffers.set(slot, { ptr, size, ref: 1 });
    this.#headPtr += size;
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
      this.#buffers.delete(slot);
    }
  }

  async read(slot: Slot, start?: number, count?: number): Promise<ArrayBuffer> {
    return this.readSync(slot, start, count);
  }

  readSync(slot: Slot, start?: number, count?: number): ArrayBuffer {
    const buffer = this.#getBuffer(slot);
    if (start === undefined) start = 0;
    if (count === undefined) count = buffer.byteLength - start;
    return buffer.slice(start, start + count);
  }

  async prepare(kernel: Kernel): Promise<Executable<void>> {
    return this.prepareSync(kernel);
  }

  prepareSync(kernel: Kernel): Executable<void> {
    return new Executable(kernel, undefined);
  }

  dispatch(
    { kernel }: Executable<void>,
    inputs: Slot[],
    outputs: Slot[],
  ): void {
    const { exp } = tuneNullopt(kernel);
    const inputBuffers = inputs.map((slot) => this.#getBuffer(slot));
    const outputBuffers = outputs.map((slot) => this.#getBuffer(slot));

    const usedArgs = new Map(
      exp
        .collect((exp) => exp.op === AluOp.GlobalIndex)
        .map((exp) => [exp.arg as number, exp.dtype]),
    );

    const inputArrays = inputBuffers.map((buf, i) => {
      const dtype = usedArgs.get(i);
      if (!dtype) return null!; // This arg is unused, so we just blank it out.
      return dtypedArray(dtype, buf);
    });
    const outputArray = dtypedArray(kernel.dtype, outputBuffers[0]);

    const globals = (gid: number, bufidx: number) => {
      if (gid < 0 || gid >= inputArrays.length)
        throw new Error("gid out of bounds: " + gid);
      if (bufidx < 0 || bufidx >= inputArrays[gid].length)
        throw new Error("bufidx out of bounds: " + bufidx);
      return inputArrays[gid][bufidx];
    };
    if (!kernel.reduction) {
      for (let i = 0; i < kernel.size; i++) {
        outputArray[i] = exp.evaluate({ gidx: i }, globals);
      }
    } else {
      for (let i = 0; i < kernel.size; i++) {
        let acc = kernel.reduction.identity;
        for (let j = 0; j < kernel.reduction.size; j++) {
          const item = exp.evaluate({ gidx: i, ridx: j }, globals);
          acc = kernel.reduction.evaluate(acc, item);
        }
        outputArray[i] = kernel.reduction.fusion.evaluate({ acc });
      }
    }
  }

  #getBuffer(slot: Slot): ArrayBuffer {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    return buffer.buffer;
  }
}
