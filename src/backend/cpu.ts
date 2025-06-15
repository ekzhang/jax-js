import { AluOp, DType, Kernel } from "../alu";
import { Backend, BackendType, Executable, Slot, SlotError } from "../backend";
import { tuneNullopt } from "../tuner";

/** Most basic implementation of `Backend` for testing. */
export class CPUBackend implements Backend {
  type: BackendType = "cpu";

  #buffers: Map<Slot, { ref: number; buffer: ArrayBuffer }>;
  #nextSlot: number;

  constructor() {
    this.#buffers = new Map();
    this.#nextSlot = 1;
  }

  malloc(size: number, initialData?: ArrayBuffer): Slot {
    const buffer = new ArrayBuffer(size);
    if (initialData) {
      if (initialData.byteLength !== size) {
        throw new Error("initialData size does not match buffer size");
      }
      new Uint8Array(buffer).set(new Uint8Array(initialData));
    }

    const slot = this.#nextSlot++;
    this.#buffers.set(slot, { buffer, ref: 1 });
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
      return dtype === DType.Float32
        ? new Float32Array(buf)
        : new Int32Array(buf);
    });
    const outputArray =
      exp.dtype === DType.Float32
        ? new Float32Array(outputBuffers[0])
        : new Int32Array(outputBuffers[0]);

    const globals = (gidx: number, bufidx: number) => inputArrays[gidx][bufidx];
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
