import { AluExp } from "../alu";
import { Backend, Slot, SlotError } from "../backend";

/** Most basic implementation of `Backend` for testing. */
export class CPUBackend implements Backend {
  readonly buffers: Map<Slot, { ref: number; buffer: ArrayBuffer }>;
  nextSlot: number;

  constructor() {
    this.buffers = new Map();
    this.nextSlot = 1;
  }

  malloc(size: number, initialData?: ArrayBuffer): Slot {
    const buffer = new ArrayBuffer(size);
    if (initialData) {
      if (initialData.byteLength !== size) {
        throw new Error("initialData size does not match buffer size");
      }
      new Uint8Array(buffer).set(new Uint8Array(initialData));
    }

    const slot = this.nextSlot++;
    this.buffers.set(slot, { buffer, ref: 1 });
    return slot;
  }

  incRef(slot: Slot): void {
    const buffer = this.buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref++;
  }

  decRef(slot: Slot): void {
    const buffer = this.buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref--;
    if (buffer.ref === 0) {
      this.buffers.delete(slot);
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

  async execute(exp: AluExp, inputs: Slot[], outputs: Slot[]): Promise<void> {
    return this.executeSync(exp, inputs, outputs);
  }

  executeSync(exp: AluExp, inputs: Slot[], outputs: Slot[]): void {
    exp = exp.simplify();
    const inputBuffers = inputs.map((slot) => this.#getBuffer(slot));
    const outputBuffers = outputs.map((slot) => this.#getBuffer(slot));

    const inputArrays = inputBuffers.map((buf) => new Float32Array(buf));
    const outputArray = new Float32Array(outputBuffers[0]);

    const globals = (gidx: number, bufidx: number) => inputArrays[gidx][bufidx];
    for (let i = 0; i < outputArray.length; i++) {
      outputArray[i] = exp.evaluate({ gidx: i }, globals);
    }
  }

  #getBuffer(slot: Slot): ArrayBuffer {
    const buffer = this.buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    return buffer.buffer;
  }
}
