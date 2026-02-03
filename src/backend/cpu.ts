import { AluOp, DType, dtypedArray, Kernel, MultiKernel } from "../alu";
import { Backend, Device, Executable, Slot, SlotError } from "../backend";
import { Routine, runCpuRoutine } from "../routine";
import { tuneNullopt } from "../tuner";

/** Most basic implementation of `Backend` for testing. */
export class CpuBackend implements Backend {
  readonly type: Device = "cpu";
  readonly maxArgs = Infinity;

  #buffers: Map<Slot, { ref: number; buffer: Uint8Array<ArrayBuffer> }>;
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

  async prepareKernel(kernel: Kernel): Promise<Executable<void>> {
    return this.prepareKernelSync(kernel);
  }

  prepareKernelSync(kernel: Kernel): Executable<void> {
    return new Executable(kernel, undefined);
  }

  async prepareMultiKernel(
    multiKernel: MultiKernel,
  ): Promise<Executable<void>> {
    return this.prepareMultiKernelSync(multiKernel);
  }

  prepareMultiKernelSync(multiKernel: MultiKernel): Executable<void> {
    return new Executable(multiKernel, undefined);
  }

  async prepareRoutine(routine: Routine): Promise<Executable> {
    return this.prepareRoutineSync(routine);
  }

  prepareRoutineSync(routine: Routine): Executable {
    return new Executable(routine, undefined);
  }

  dispatch(exe: Executable<void>, inputs: Slot[], outputs: Slot[]): void {
    if (exe.source instanceof Routine) {
      return runCpuRoutine(
        exe.source,
        inputs.map((slot) => this.#getBuffer(slot)),
        outputs.map((slot) => this.#getBuffer(slot)),
      );
    }

    if (exe.source instanceof MultiKernel) {
      return this.#dispatchMultiKernel(exe.source, inputs, outputs);
    }

    const kernel = exe.source as Kernel;
    const { exp, epilogue } = tuneNullopt(kernel);
    const inputBuffers = inputs.map((slot) => this.#getBuffer(slot));
    const outputBuffers = outputs.map((slot) => this.#getBuffer(slot));

    // Collect both GlobalIndex and GlobalView nodes to determine which input buffers are used.
    // GlobalView is used for lazy reshape/transpose operations within JIT.
    const usedArgs = new Map(
      [
        ...exp.collect(
          (exp) => exp.op === AluOp.GlobalIndex || exp.op === AluOp.GlobalView,
        ),
        ...(epilogue
          ? epilogue.collect(
              (exp) =>
                exp.op === AluOp.GlobalIndex || exp.op === AluOp.GlobalView,
            )
          : []),
      ].map((exp) => [exp.arg[0] as number, exp.dtype]),
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
        outputArray[i] = epilogue!.evaluate({ acc, gidx: i }, globals);
      }
    }
  }

  #dispatchMultiKernel(
    multiKernel: MultiKernel,
    inputs: Slot[],
    outputs: Slot[],
  ): void {
    const inputBuffers = inputs.map((slot) => this.#getBuffer(slot));
    const size = multiKernel.outputs[0].size;

    // Collect all used args from all output expressions
    const allUsedArgs = new Map<number, DType>();
    for (const out of multiKernel.outputs) {
      const usedArgs = out.exp.collect(
        (exp) => exp.op === AluOp.GlobalIndex || exp.op === AluOp.GlobalView,
      );
      for (const exp of usedArgs) {
        allUsedArgs.set(exp.arg[0] as number, exp.dtype);
      }
    }

    const inputArrays = inputBuffers.map((buf, i) => {
      const dtype = allUsedArgs.get(i);
      if (!dtype) return null!;
      return dtypedArray(dtype, buf);
    });

    const outputArrays = outputs.map((slot, i) =>
      dtypedArray(multiKernel.outputs[i].dtype, this.#getBuffer(slot)),
    );

    const globals = (gid: number, bufidx: number) => {
      if (gid < 0 || gid >= inputArrays.length)
        throw new Error("gid out of bounds: " + gid);
      if (bufidx < 0 || bufidx >= inputArrays[gid].length)
        throw new Error("bufidx out of bounds: " + bufidx);
      return inputArrays[gid][bufidx];
    };

    // Multi-output kernels have no reduction support
    for (let i = 0; i < size; i++) {
      for (let outIdx = 0; outIdx < multiKernel.outputs.length; outIdx++) {
        const out = multiKernel.outputs[outIdx];
        outputArrays[outIdx][i] = out.exp.evaluate({ gidx: i }, globals);
      }
    }
  }

  #getBuffer(slot: Slot): Uint8Array<ArrayBuffer> {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    return buffer.buffer;
  }
}
