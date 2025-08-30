import {
  AluExp,
  AluOp,
  byteWidth,
  DType,
  dtypedArray,
  isFloatDtype,
  Kernel,
} from "../alu";
import { Backend, Device, Executable, Slot, SlotError } from "../backend";
import { tuneNullopt } from "../tuner";
import { rep } from "../utils";
import { CodeGenerator } from "./wasm/wasmblr";

interface WasmBuffer {
  ptr: number;
  size: number;
  ref: number;
}

/** Backend that compiles into WebAssembly bytecode for immediate execution. */
export class WasmBackend implements Backend {
  readonly type: Device = "wasm";
  readonly maxArgs = 64; // Arbitrary choice

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

  malloc(size: number, initialData?: Uint8Array): Slot {
    // TODO: Currently, we just have a bump allocator and never free memory.
    const ptr = this.#headPtr;
    if (initialData) {
      if (initialData.byteLength !== size)
        throw new Error("initialData size does not match buffer size");
      new Uint8Array(this.#memory.buffer, ptr, size).set(initialData);
    }

    const slot = this.#nextSlot++;
    this.#buffers.set(slot, { ptr, size, ref: 1 });
    this.#headPtr += Math.ceil(size / 64) * 64; // align to 64 bytes, like Arrow
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

  #getBuffer(slot: Slot): Uint8Array {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    return new Uint8Array(this.#memory.buffer, buffer.ptr, buffer.size);
  }
}

function compileWasm(kernel: Kernel): Uint8Array {
  const cg = new CodeGenerator();

  cg.memory.import("env", "memory");

  const kernelFunc = cg.function(rep(kernel.nargs + 1, cg.i32), [], () => {
    if (kernel.reduction) {
      throw new Error("TODO: Reductions on wasm backend not implemented yet");
    }
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

      // Translate kernel.exp to expression and push onto stack.
      translateExp(cg, kernel.exp, { gidx });

      // Store value into output buffer.
      dty(cg, kernel.dtype).store(Math.log2(byteWidth(kernel.dtype)));

      cg.br(1); // continue loop
      cg.end();
    }
    cg.end();
  });
  cg.export(kernelFunc, "kernel");

  return cg.finish();
}

function translateExp(
  cg: CodeGenerator,
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
  countReferences(exp);

  const expContext = new Map<AluExp, number>();
  const gen = (exp: AluExp) => {
    if (expContext.has(exp)) return cg.local.get(expContext.get(exp)!);
    const { op, src, dtype, arg } = exp;

    // Some of these cases early `return` to force-inline them (no local.set).
    if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
      if (op === AluOp.Add) {
        (gen(src[0]), gen(src[1]));
        if (dtype === DType.Bool) cg.i32.or();
        else dty(cg, dtype).add();
      } else if (op === AluOp.Sub) {
        (gen(src[0]), gen(src[1]));
        dty(cg, dtype).sub();
      } else if (op === AluOp.Mul) {
        (gen(src[0]), gen(src[1]));
        if (dtype === DType.Bool) cg.i32.and();
        else dty(cg, dtype).mul();
      } else if (op === AluOp.Idiv) {
        if (dtype === DType.Float32) {
          (gen(src[0]), gen(src[1]));
          cg.f32.div();
          cg.f32.floor();
        } else if (
          dtype === DType.Uint32 ||
          (dtype === DType.Int32 && src[0].min >= 0 && src[1].min >= 0)
        ) {
          (gen(src[0]), gen(src[1]));
          cg.i32.div_u();
        } else if (dtype === DType.Int32) {
          gen(src[0]);
          cg.f32.convert_i32_s();
          gen(src[1]);
          cg.f32.convert_i32_s();
          cg.f32.div();
          cg.f32.floor();
          cg.i32.trunc_f32_s();
        } else {
          throw new Error("Unsupported dtype for Idiv in Wasm: " + dtype);
        }
      } else if (op === AluOp.Mod) {
      }
    }
  };
}

function dty(cg: CodeGenerator, dtype: DType) {
  switch (dtype) {
    case DType.Float32:
      return cg.f32;
    case DType.Int32:
    case DType.Uint32:
    case DType.Bool:
      return cg.i32;
    default:
      throw new Error(`Unsupported dtype in wasm: ${dtype}`);
  }
}
