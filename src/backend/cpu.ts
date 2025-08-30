import { AluOp, dtypedArray, Kernel } from "../alu";
import { Backend, Device, Executable, Slot, SlotError } from "../backend";
import { tuneNullopt } from "../tuner";

/** Most basic implementation of `Backend` for testing. */
export class CpuBackend implements Backend {
  readonly type: Device = "cpu";
  readonly maxArgs = Infinity;

  #buffers: Map<Slot, { ref: number; buffer: Uint8Array }>;
  #nextSlot: number;

  constructor() {
    this.#buffers = new Map();
    this.#nextSlot = 1;
  }

  malloc(size: number, initialData?: Uint8Array): Slot {
    const buffer = new Uint8Array(size);
    if (initialData) {
      if (initialData.byteLength !== size) {
        throw new Error("initialData size does not match buffer size");
      }
      buffer.set(initialData);
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

  async read(slot: Slot, start?: number, count?: number): Promise<Uint8Array> {
    return this.readSync(slot, start, count);
  }

  readSync(slot: Slot, start?: number, count?: number): Uint8Array {
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
        .map((exp) => [exp.arg[0] as number, exp.dtype]),
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

  #getBuffer(slot: Slot): Uint8Array {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    return buffer.buffer;
  }
}
