import { AluExp, AluGroup, AluOp, byteWidth, DType, Kernel } from "../alu";
import {
  Backend,
  Device,
  Executable,
  Slot,
  SlotError,
  UnsupportedOpError,
} from "../backend";
import { tuneNullopt } from "../tuner";
import { DEBUG, rep, union } from "../utils";
import {
  wasm_cos,
  wasm_exp,
  wasm_log,
  wasm_sin,
  wasm_threefry2x32,
} from "./wasm/builtins";
import { CodeGenerator } from "./wasm/wasmblr";

interface WasmBuffer {
  ptr: number;
  size: number;
  ref: number;
}

interface WasmProgram {
  module: WebAssembly.Module;
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
    this.#memory = new WebAssembly.Memory({ initial: 0 });
    this.#nextSlot = 1;
    this.#headPtr = 0;
    this.#buffers = new Map();
  }

  malloc(size: number, initialData?: Uint8Array): Slot {
    // TODO: Currently, we just have a bump allocator and never free memory.
    const ptr = this.#headPtr;
    if (ptr + size > this.#memory.buffer.byteLength) {
      this.#memory.grow(
        ((ptr + size + 65535) >> 16) - (this.#memory.buffer.byteLength >> 16),
      );
    }
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

  async prepare(kernel: Kernel): Promise<Executable<WasmProgram>> {
    return this.prepareSync(kernel);
  }

  prepareSync(kernel: Kernel): Executable<WasmProgram> {
    const bytes = codegenWasm(kernel);
    const module = new WebAssembly.Module(bytes);
    return new Executable(kernel, { module });
  }

  dispatch(
    exe: Executable<WasmProgram>,
    inputs: Slot[],
    outputs: Slot[],
  ): void {
    const instance = new WebAssembly.Instance(exe.data.module, {
      env: { memory: this.#memory },
    });
    const func = instance.exports.kernel as (...args: number[]) => void;
    const ptrs = [...inputs, ...outputs].map(
      (slot) => this.#buffers.get(slot)!.ptr,
    );
    func(...ptrs);
  }

  #getBuffer(slot: Slot): Uint8Array {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    return new Uint8Array(this.#memory.buffer, buffer.ptr, buffer.size);
  }
}

function codegenWasm(kernel: Kernel): Uint8Array {
  const tune = tuneNullopt(kernel);
  const re = kernel.reduction;

  if (DEBUG >= 3) {
    console.info(`kernel.exp: ${kernel.exp}\ntune.exp: ${tune.exp}`);
  }

  const cg = new CodeGenerator();
  cg.memory.import("env", "memory");

  const distinctOps = union(tune.exp.distinctOps(), re?.fusion.distinctOps());
  const funcs: Record<string, number> = {};
  if (distinctOps.has(AluOp.Sin)) funcs.sin = wasm_sin(cg);
  if (distinctOps.has(AluOp.Cos)) funcs.cos = wasm_cos(cg);
  if (distinctOps.has(AluOp.Exp)) funcs.exp = wasm_exp(cg);
  if (distinctOps.has(AluOp.Log)) funcs.log = wasm_log(cg);
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
            if (re.dtype === DType.Float32) {
              cg.local.get(acc);
              if (re.op === AluOp.Min) cg.f32.min();
              else cg.f32.max();
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

        translateExp(cg, funcs, kernel.reduction.fusion, { acc });
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
        if (dtype === DType.Float32) (cg.f32.div(), cg.f32.trunc());
        else if (dtype === DType.Uint32) cg.i32.div_u();
        else if (dtype === DType.Int32) cg.i32.div_s();
        else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.Mod) {
        if (dtype === DType.Float32) {
          // Emulate a % b = a - trunc(a/b)*b
          const a = cg.local.declare(cg.f32);
          const b = cg.local.declare(cg.f32);
          cg.local.set(b);
          cg.local.tee(a); // stack: a
          cg.local.get(a);
          cg.local.get(b);
          cg.f32.div();
          cg.f32.trunc(); // stack: a, trunc(a/b)
          cg.local.get(b);
          cg.f32.mul(); // stack: a, trunc(a/b)*b
          cg.f32.sub();
        } else if (dtype === DType.Uint32) cg.i32.rem_u();
        else if (dtype === DType.Int32) cg.i32.rem_s();
        else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.Min || op === AluOp.Max) {
        if (dtype === DType.Float32) {
          if (op === AluOp.Min) cg.f32.min();
          else cg.f32.max();
        } else if (dtype === DType.Int32 || dtype === DType.Uint32) {
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
        if (srcDtype === DType.Float32) cg.f32.lt();
        else if (srcDtype === DType.Int32) cg.i32.lt_s();
        else if (srcDtype === DType.Uint32) cg.i32.lt_u();
        else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.Cmpne) dty(cg, op, src[0].dtype).ne();
      else throw new UnsupportedOpError(op, dtype, "wasm");
    } else if (AluGroup.Unary.has(op)) {
      if (op === AluOp.Sin) (gen(src[0]), cg.call(funcs.sin));
      else if (op === AluOp.Cos) (gen(src[0]), cg.call(funcs.cos));
      else if (op === AluOp.Exp) (gen(src[0]), cg.call(funcs.exp));
      else if (op === AluOp.Log) (gen(src[0]), cg.call(funcs.log));
      else if (op === AluOp.Sqrt) (gen(src[0]), cg.f32.sqrt());
      else if (op === AluOp.Reciprocal)
        (cg.f32.const(1), gen(src[0]), cg.f32.div());
      else if (op === AluOp.Cast) {
        gen(src[0]);
        const dtype0 = src[0].dtype;
        const i32repr =
          dtype0 === DType.Int32 ||
          dtype0 === DType.Uint32 ||
          dtype0 === DType.Bool;
        if (dtype === DType.Int32) {
          if (dtype0 === DType.Float32) cg.i32.trunc_f32_s();
          else if (i32repr) void 0;
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else if (dtype === DType.Uint32) {
          if (dtype0 === DType.Float32) cg.i32.trunc_f32_u();
          else if (i32repr) void 0;
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else if (dtype === DType.Float32) {
          if (dtype0 === DType.Float32) void 0;
          else if (dtype0 === DType.Int32 || dtype0 === DType.Bool)
            cg.f32.convert_i32_s();
          else if (dtype0 === DType.Uint32) cg.f32.convert_i32_u();
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else if (dtype === DType.Bool) {
          if (dtype0 === DType.Bool) void 0;
          else if (i32repr) (cg.i32.const(0), cg.i32.ne());
          else if (dtype0 === DType.Float32) (cg.f32.const(0), cg.f32.ne());
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.Bitcast) {
        gen(src[0]);
        const dtype0 = src[0].dtype;
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
    case DType.Int32:
    case DType.Uint32:
    case DType.Bool:
      return cg.i32;
    default:
      throw new UnsupportedOpError(op, dtype, "wasm");
  }
}
