/**
 * @file Minimalist WebAssembly assembler. This allows you to emit WebAssembly
 * bytecode directly from the browser.
 *
 * Self-contained port of https://github.com/bwasti/wasmblr to TypeScript.
 * Functions and variables in this module are written in `snake_case` to match
 * the names of WebAssembly operations.
 *
 * Reference: https://pengowray.github.io/wasm-ops/.
 */

const magic_module_header = [0x00, 0x61, 0x73, 0x6d];
const module_version = [0x01, 0x00, 0x00, 0x00];

function assert(condition: boolean, message?: string): asserts condition {
  if (!condition) {
    throw new Error(message || "Assertion failed");
  }
}

// From LLVM
function encode_signed(n: number): number[] {
  const out: number[] = [];
  let more = true;
  while (more) {
    let byte = n & 0x7f;
    n >>= 7;
    if ((n === 0 && (byte & 0x40) === 0) || (n === -1 && (byte & 0x40) !== 0)) {
      more = false;
    } else {
      byte |= 0x80;
    }
    out.push(byte);
  }
  return out;
}

function encode_unsigned(n: number): number[] {
  const out: number[] = [];
  do {
    let byte = n & 0x7f;
    n = n >>> 7;
    if (n !== 0) {
      byte |= 0x80;
    }
    out.push(byte);
  } while (n !== 0);
  return out;
}

function encode_string(s: string): number[] {
  const bytes = new TextEncoder().encode(s);
  return [bytes.length, ...bytes];
}

function concat(out: number[], inp: number[]): void {
  out.push(...inp);
}

class Function_ {
  input_types: Type[];
  output_types: Type[];
  body: () => void;
  locals: Type[] = [];
  constructor(input_types: Type[], output_types: Type[], body?: () => void) {
    this.input_types = input_types;
    this.output_types = output_types;
    this.body = body || (() => {});
  }
  emit() {
    this.locals = [];
    this.body();
  }
}

class Memory {
  min = 0;
  max = 0;
  is_shared = false;
  a_string = "";
  b_string = "";

  constructor(readonly cg: CodeGenerator) {}

  declare(min: number, max: number = 0): this {
    assert(this.min === 0 && this.max === 0);
    this.min = min;
    this.max = max;
    return this;
  }

  export_(a: string): this {
    assert(!this.is_import && !this.is_export, "already set");
    this.a_string = a;
    return this;
  }

  shared(make_shared: boolean): this {
    this.is_shared = make_shared;
    return this;
  }

  import_(a: string, b: string): this {
    assert(!this.is_import && !this.is_export, "already set");
    this.a_string = a;
    this.b_string = b;
    return this;
  }

  size() {
    this.cg.emit(0x3f);
    this.cg.emit(0x00);
  }

  grow() {
    this.cg.emit(0x40);
    this.cg.emit(0x00);
  }

  get is_import(): boolean {
    return this.a_string.length > 0 && this.b_string.length > 0;
  }
  get is_export(): boolean {
    return this.a_string.length > 0 && this.b_string.length === 0;
  }
}

////////////////////////////////////////
// CodeGenerator class
////////////////////////////////////////

// I32, F32, V128, Void
interface Type {
  typeId: number;
}

/** Public API of WebAssembly assembler. */
export class CodeGenerator {
  local: Local;
  i32: I32;
  f32: F32;
  v128: V128;
  memory: Memory;
  void_ = { typeId: 0x40 };

  functions: Function_[] = [];
  exported_functions = new Map<number, string>();
  cur_function: Function_ | null = null;
  cur_bytes: number[] = [];
  type_stack: Type[] = [];

  constructor() {
    this.local = new Local(this);
    this.i32 = new I32(this);
    this.f32 = new F32(this);
    this.v128 = new V128(this);
    this.memory = new Memory(this);
  }

  // Control and branching instructions
  nop() {
    this.emit(0x01);
  }
  block(type: number) {
    this.emit(0x02);
    this.emit(type);
  }
  loop(type: number) {
    this.emit(0x03);
    this.emit(type);
  }
  if_(type: number) {
    assert(this.pop() === this.i32, "if_: expected i32");
    this.emit(0x04);
    this.emit(type);
  }
  else_() {
    this.emit(0x05);
  }
  br(labelidx: number) {
    this.emit(0x0c);
    this.emit(encode_unsigned(labelidx));
  }
  br_if(labelidx: number) {
    assert(this.pop() === this.i32, "br_if: expected i32");
    this.emit(0x0d);
    this.emit(encode_unsigned(labelidx));
  }
  end() {
    this.emit(0x0b);
  }
  call(fn_idx: number) {
    assert(fn_idx < this.functions.length, "function index does not exist");
    this.emit(0x10);
    this.emit(encode_unsigned(fn_idx));
  }

  // Export a function.
  export_(fn: number, name: string) {
    this.exported_functions.set(fn, name);
  }

  // Declare a new function; returns its index.
  function(
    input_types: Type[],
    output_types: Type[],
    body: () => void,
  ): number {
    const idx = this.functions.length;
    this.functions.push(new Function_(input_types, output_types, body));
    return idx;
  }

  // --- Implementation helpers

  declare_local(type: Type): number {
    assert(this.cur_function !== null, "No current function");
    const idx =
      this.cur_function.locals.length + this.cur_function.input_types.length;
    this.cur_function.locals.push(type);
    return idx;
  }

  input_types(): Type[] {
    assert(this.cur_function !== null, "No current function");
    return this.cur_function.input_types;
  }

  locals(): Type[] {
    assert(this.cur_function !== null, "No current function");
    return this.cur_function.locals;
  }

  push(type: Type) {
    this.type_stack.push(type);
  }
  pop(): Type {
    assert(this.type_stack.length > 0, "popping empty stack");
    return this.type_stack.pop()!;
  }

  emit(bytes: number | number[]) {
    if (typeof bytes === "number") this.cur_bytes.push(bytes);
    else this.cur_bytes.push(...bytes);
  }

  // Emit the complete module as an array of bytes.
  finish(): Uint8Array {
    this.cur_bytes = [];
    let emitted_bytes: number[] = [];
    concat(emitted_bytes, magic_module_header);
    concat(emitted_bytes, module_version);

    // Type section
    let type_section_bytes: number[] = [];
    concat(type_section_bytes, encode_unsigned(this.functions.length));
    for (const f of this.functions) {
      type_section_bytes.push(0x60);
      concat(type_section_bytes, encode_unsigned(f.input_types.length));
      for (const t of f.input_types) {
        type_section_bytes.push(t.typeId);
      }
      concat(type_section_bytes, encode_unsigned(f.output_types.length));
      for (const t of f.output_types) {
        type_section_bytes.push(t.typeId);
      }
    }
    emitted_bytes.push(0x01);
    concat(emitted_bytes, encode_unsigned(type_section_bytes.length));
    concat(emitted_bytes, type_section_bytes);

    // Import section (for memory import)
    let import_section_bytes: number[] = [];
    if (this.memory.is_import) {
      // one import
      concat(import_section_bytes, encode_unsigned(1));
      concat(import_section_bytes, encode_string(this.memory.a_string));
      concat(import_section_bytes, encode_string(this.memory.b_string));
      import_section_bytes.push(0x02); // memory flag
      if (this.memory.min && this.memory.max) {
        if (this.memory.is_shared) {
          import_section_bytes.push(0x03);
        } else {
          import_section_bytes.push(0x01);
        }
        concat(import_section_bytes, encode_unsigned(this.memory.min));
        concat(import_section_bytes, encode_unsigned(this.memory.max));
      } else {
        assert(!this.memory.is_shared, "shared memory must have a max size");
        concat(import_section_bytes, encode_unsigned(this.memory.min));
      }
      emitted_bytes.push(0x02);
      concat(emitted_bytes, encode_unsigned(import_section_bytes.length));
      concat(emitted_bytes, import_section_bytes);
    }

    // Function section
    let function_section_bytes: number[] = [];
    concat(function_section_bytes, encode_unsigned(this.functions.length));
    for (let i = 0; i < this.functions.length; i++) {
      concat(function_section_bytes, encode_unsigned(i));
    }
    emitted_bytes.push(0x03);
    concat(emitted_bytes, encode_unsigned(function_section_bytes.length));
    concat(emitted_bytes, function_section_bytes);

    // Memory section (if defined locally)
    let memory_section_bytes: number[] = [];
    if (!this.memory.is_import && (this.memory.min || this.memory.max)) {
      memory_section_bytes.push(0x01); // always one memory
      if (this.memory.min && this.memory.max) {
        if (this.memory.is_shared) {
          memory_section_bytes.push(0x03);
        } else {
          memory_section_bytes.push(0x01);
        }
        concat(memory_section_bytes, encode_unsigned(this.memory.min));
        concat(memory_section_bytes, encode_unsigned(this.memory.max));
      } else {
        assert(!this.memory.is_shared, "shared memory must have a max size");
        memory_section_bytes.push(0x00);
        concat(memory_section_bytes, encode_unsigned(this.memory.min));
      }
      emitted_bytes.push(0x05);
      concat(emitted_bytes, encode_unsigned(memory_section_bytes.length));
      concat(emitted_bytes, memory_section_bytes);
    }

    // Export section
    let export_section_bytes: number[] = [];
    const num_exports =
      this.exported_functions.size + (this.memory.is_export ? 1 : 0);
    concat(export_section_bytes, encode_unsigned(num_exports));
    if (this.memory.is_export) {
      concat(export_section_bytes, encode_string(this.memory.a_string));
      export_section_bytes.push(0x02);
      export_section_bytes.push(0x00); // one memory at index 0
    }
    for (const [key, name] of this.exported_functions.entries()) {
      concat(export_section_bytes, encode_string(name));
      export_section_bytes.push(0x00);
      concat(export_section_bytes, encode_unsigned(key));
    }
    emitted_bytes.push(0x07);
    concat(emitted_bytes, encode_unsigned(export_section_bytes.length));
    concat(emitted_bytes, export_section_bytes);

    // Code section
    let code_section_bytes: number[] = [];
    concat(code_section_bytes, encode_unsigned(this.functions.length));
    for (const f of this.functions) {
      this.cur_function = f;
      this.cur_bytes = [];
      f.emit();
      this.end();
      const body_bytes = [...this.cur_bytes];
      this.cur_bytes = [];
      // Header: local declarations
      concat(this.cur_bytes, encode_unsigned(f.locals.length));
      for (const l of f.locals) {
        this.emit(0x01);
        this.emit(l.typeId);
      }
      const header_bytes = [...this.cur_bytes];
      const fn_size = header_bytes.length + body_bytes.length;
      concat(code_section_bytes, encode_unsigned(fn_size));
      concat(code_section_bytes, header_bytes);
      concat(code_section_bytes, body_bytes);
    }
    this.cur_function = null;

    emitted_bytes.push(0x0a);
    concat(emitted_bytes, encode_unsigned(code_section_bytes.length));
    concat(emitted_bytes, code_section_bytes);

    return new Uint8Array(emitted_bytes);
  }
}

////////////////////////////////////////
// Local variables
////////////////////////////////////////

class Local {
  constructor(readonly cg: CodeGenerator) {}

  // Mimic operator()(type)
  declare(type: Type): number {
    return this.cg.declare_local(type);
  }
  get(idx: number) {
    const input_types = this.cg.input_types();
    if (idx < input_types.length) {
      this.cg.push(input_types[idx]);
    } else {
      this.cg.push(this.cg.locals()[idx - input_types.length]);
    }
    this.cg.emit(0x20);
    this.cg.emit(encode_unsigned(idx));
  }
  set(idx: number) {
    const t = this.cg.pop();
    const input_types = this.cg.input_types();
    const expected_type =
      idx < input_types.length
        ? input_types[idx]
        : this.cg.locals()[idx - input_types.length];
    assert(expected_type === t, "can't set local to this value (wrong type)");
    this.cg.emit(0x21);
    this.cg.emit(encode_unsigned(idx));
  }
  tee(idx: number) {
    const t = this.cg.pop();
    const input_types = this.cg.input_types();
    const expected_type =
      idx < input_types.length
        ? input_types[idx]
        : this.cg.locals()[idx - input_types.length];
    assert(expected_type === t, "can't tee local to this value (wrong type)");
    this.cg.emit(0x22);
    this.cg.emit(encode_unsigned(idx));
    this.cg.push(expected_type);
  }
}

function UNARY_OP(
  op: string,
  opcode: number,
  in_type: string,
  out_type: string,
) {
  return function (this: any) {
    const t = this.cg.pop();
    assert(
      t === this.cg[in_type],
      `invalid type for ${op} (${in_type} -> ${out_type})`,
    );
    this.cg.emit(opcode);
    this.cg.push(this.cg[out_type]);
  };
}

function BINARY_OP(
  op: string,
  opcode: number,
  type_a: string,
  type_b: string,
  out_type: string,
) {
  return function (this: any) {
    const b = this.cg.pop();
    const a = this.cg.pop();
    assert(
      a === this.cg[type_a] && b === this.cg[type_b],
      `invalid type for ${op} (${type_a}, ${type_b} -> ${out_type})`,
    );
    this.cg.emit(opcode);
    this.cg.push(this.cg[out_type]);
  };
}

function LOAD_OP(op: string, opcode: number, out_type: string) {
  return function (this: any, alignment: number = 1, offset: number = 0) {
    const idx_type = this.cg.pop();
    assert(idx_type === this.cg.i32, `invalid type for ${op}`);
    this.cg.emit(opcode);
    this.cg.emit(encode_unsigned(alignment));
    this.cg.emit(encode_unsigned(offset));
    this.cg.push(this.cg[out_type]);
  };
}

function STORE_OP(op: string, opcode: number) {
  return function (this: any, alignment: number = 1, offset: number = 0) {
    const val_type = this.cg.pop();
    const idx_type = this.cg.pop();
    assert(idx_type === this.cg.i32, `invalid type for ${op}`);
    this.cg.emit(opcode);
    this.cg.emit(encode_unsigned(alignment));
    this.cg.emit(encode_unsigned(offset));
  };
}

////////////////////////////////////////
// I32 class
////////////////////////////////////////

class I32 {
  constructor(readonly cg: CodeGenerator) {}
  get typeId(): number {
    return 0x7f;
  }

  const_(i: number) {
    this.cg.emit(0x41);
    this.cg.emit(encode_signed(i));
    this.cg.push(this);
  }
  clz = UNARY_OP("clz", 0x67, "i32", "i32");
  ctz = UNARY_OP("ctz", 0x68, "i32", "i32");
  popcnt = UNARY_OP("popcnt", 0x69, "i32", "i32");
  lt_s = BINARY_OP("lt_s", 0x48, "i32", "i32", "i32");
  lt_u = BINARY_OP("lt_u", 0x49, "i32", "i32", "i32");
  gt_s = BINARY_OP("gt_s", 0x4a, "i32", "i32", "i32");
  gt_u = BINARY_OP("gt_u", 0x4b, "i32", "i32", "i32");
  le_s = BINARY_OP("le_s", 0x4c, "i32", "i32", "i32");
  le_u = BINARY_OP("le_u", 0x4d, "i32", "i32", "i32");
  ge_s = BINARY_OP("ge_s", 0x4e, "i32", "i32", "i32");
  ge_u = BINARY_OP("ge_u", 0x4f, "i32", "i32", "i32");
  add = BINARY_OP("add", 0x6a, "i32", "i32", "i32");
  sub = BINARY_OP("sub", 0x6b, "i32", "i32", "i32");
  mul = BINARY_OP("mul", 0x6c, "i32", "i32", "i32");
  div_s = BINARY_OP("div_s", 0x6d, "i32", "i32", "i32");
  div_u = BINARY_OP("div_u", 0x6e, "i32", "i32", "i32");
  rem_s = BINARY_OP("rem_s", 0x6f, "i32", "i32", "i32");
  rem_u = BINARY_OP("rem_u", 0x70, "i32", "i32", "i32");
  and = BINARY_OP("and", 0x71, "i32", "i32", "i32");
  or = BINARY_OP("or", 0x72, "i32", "i32", "i32");
  xor = BINARY_OP("xor", 0x73, "i32", "i32", "i32");
  shl = BINARY_OP("shl", 0x74, "i32", "i32", "i32");
  shr_s = BINARY_OP("shr_s", 0x75, "i32", "i32", "i32");
  shr_u = BINARY_OP("shr_u", 0x76, "i32", "i32", "i32");
  rotl = BINARY_OP("rotl", 0x77, "i32", "i32", "i32");
  rotr = BINARY_OP("rotr", 0x78, "i32", "i32", "i32");
  eqz = BINARY_OP("eqz", 0x45, "i32", "i32", "i32");
  eq = BINARY_OP("eq", 0x46, "i32", "i32", "i32");
  ne = BINARY_OP("ne", 0x47, "i32", "i32", "i32");
  load = LOAD_OP("load", 0x28, "i32");
  load8_s = LOAD_OP("load8_s", 0x2c, "i32");
  load8_u = LOAD_OP("load8_u", 0x2d, "i32");
  load16_s = LOAD_OP("load16_s", 0x2e, "i32");
  load16_u = LOAD_OP("load16_u", 0x2f, "i32");
  store = STORE_OP("store", 0x36);
  store8 = STORE_OP("store8", 0x3a);
  store16 = STORE_OP("store16", 0x3b);
}

////////////////////////////////////////
// F32 class
////////////////////////////////////////

class F32 {
  constructor(readonly cg: CodeGenerator) {}
  get typeId(): number {
    return 0x7d;
  }

  const_(f: number) {
    this.cg.emit(0x43);
    const buffer = new ArrayBuffer(4);
    new DataView(buffer).setFloat32(0, f, true);
    const bytes = new Uint8Array(buffer);
    for (let i = 0; i < 4; i++) {
      this.cg.emit(bytes[i]);
    }
    this.cg.push(this);
  }

  eq = BINARY_OP("eq", 0x5b, "f32", "f32", "i32");
  ne = BINARY_OP("ne", 0x5c, "f32", "f32", "i32");
  lt = BINARY_OP("lt", 0x5d, "f32", "f32", "i32");
  gt = BINARY_OP("gt", 0x5e, "f32", "f32", "i32");
  le = BINARY_OP("le", 0x5f, "f32", "f32", "i32");
  ge = BINARY_OP("ge", 0x60, "f32", "f32", "i32");
  abs = UNARY_OP("abs", 0x8b, "f32", "f32");
  neg = UNARY_OP("neg", 0x8c, "f32", "f32");
  ceil = UNARY_OP("ceil", 0x8d, "f32", "f32");
  floor = UNARY_OP("floor", 0x8e, "f32", "f32");
  trunc = UNARY_OP("trunc", 0x8f, "f32", "f32");
  nearest = UNARY_OP("nearest", 0x90, "f32", "f32");
  sqrt = UNARY_OP("sqrt", 0x91, "f32", "f32");
  add = BINARY_OP("add", 0x92, "f32", "f32", "f32");
  sub = BINARY_OP("sub", 0x93, "f32", "f32", "f32");
  mul = BINARY_OP("mul", 0x94, "f32", "f32", "f32");
  div = BINARY_OP("div", 0x95, "f32", "f32", "f32");
  min = BINARY_OP("min", 0x96, "f32", "f32", "f32");
  max = BINARY_OP("max", 0x97, "f32", "f32", "f32");
  copysign = BINARY_OP("copysign", 0x98, "f32", "f32", "f32");
  load = LOAD_OP("load", 0x2a, "f32");
  store = STORE_OP("store", 0x38);
}

function VECTOR_OP(
  op: string,
  vopcode: number,
  in_types: string[],
  out_type: string,
) {
  return function (this: any) {
    for (const in_type of in_types) {
      const actual_type = this.cg.pop();
      assert(
        actual_type === this.cg[in_type],
        `invalid type for ${op} (${in_types} -> ${out_type})`,
      );
    }
    this.cg.emit(0xfd);
    this.cg.emit(encode_unsigned(vopcode));
    this.cg.push(this.cg[out_type]);
  };
}

// Like VECTOR_OP but also takes a lane.
function VECTOR_OPL(
  op: string,
  vopcode: number,
  in_types: string[],
  out_type: string,
) {
  return function (this: any, lane: number) {
    for (const in_type of in_types) {
      const actual_type = this.cg.pop();
      assert(
        actual_type === this.cg[in_type],
        `invalid type for ${op} (${in_types} -> ${out_type})`,
      );
    }
    this.cg.emit(0xfd);
    this.cg.emit(encode_unsigned(vopcode));
    this.cg.emit(lane); // 1 byte
    this.cg.push(this.cg[out_type]);
  };
}

function VECTOR_LOAD_OP(op: string, vopcode: number) {
  return function (this: any, alignment: number = 1, offset: number = 0) {
    const idx_type = this.cg.pop();
    assert(idx_type === this.cg.i32, `invalid type for ${op}`);
    this.cg.emit(0xfd);
    this.cg.emit(encode_unsigned(vopcode));
    this.cg.emit(encode_unsigned(alignment));
    this.cg.emit(encode_unsigned(offset));
    this.cg.push(this.cg.v128);
  };
}

// prettier-ignore
class V128 {
  constructor(readonly cg: CodeGenerator) {}
  get typeId(): number {
    return 0x7b;
  }

  load = VECTOR_LOAD_OP("load", 0x00);
  load32x2_s = VECTOR_LOAD_OP("load32x2_s", 0x05);
  load32x2_u = VECTOR_LOAD_OP("load32x2_u", 0x06);
  load32_splat = VECTOR_LOAD_OP("load32_splat", 0x09);
  load32_zero = VECTOR_LOAD_OP("load32_zero", 0x5c);

  store(alignment: number = 1, offset: number = 0) {
    const val_type = this.cg.pop();
    assert(val_type === this.cg.v128, `invalid type for store`);
    const idx_type = this.cg.pop();
    assert(idx_type === this.cg.i32, `invalid type for store`);
    this.cg.emit(0xfd);
    this.cg.emit(encode_unsigned(0x0b));
    this.cg.emit(encode_unsigned(alignment));
    this.cg.emit(encode_unsigned(offset));
  }

  i32x4_splat = VECTOR_OP("i32x4_splat", 0x11, ["i32"], "v128");
  f32x4_splat = VECTOR_OP("f32x4_splat", 0x13, ["f32"], "v128");
  i32x4_extract_lane = VECTOR_OPL("i32x4_extract_lane", 0x1b, ["v128"], "i32");
  i32x4_replace_lane = VECTOR_OPL("i32x4_replace_lane", 0x1c, ["i32", "v128"], "v128");
  f32x4_extract_lane = VECTOR_OPL("f32x4_extract_lane", 0x1f, ["v128"], "f32");
  f32x4_replace_lane = VECTOR_OPL("f32x4_replace_lane", 0x20, ["f32", "v128"], "v128");

  i32x4_eq = VECTOR_OP("i32x4_eq", 0x37, ["v128", "v128"], "v128");
  i32x4_ne = VECTOR_OP("i32x4_ne", 0x38, ["v128", "v128"], "v128");
  i32x4_lt_s = VECTOR_OP("i32x4_lt_s", 0x39, ["v128", "v128"], "v128");
  i32x4_lt_u = VECTOR_OP("i32x4_lt_u", 0x3a, ["v128", "v128"], "v128");
  i32x4_gt_s = VECTOR_OP("i32x4_gt_s", 0x3b, ["v128", "v128"], "v128");
  i32x4_gt_u = VECTOR_OP("i32x4_gt_u", 0x3c, ["v128", "v128"], "v128");
  i32x4_le_s = VECTOR_OP("i32x4_le_s", 0x3d, ["v128", "v128"], "v128");
  i32x4_le_u = VECTOR_OP("i32x4_le_u", 0x3e, ["v128", "v128"], "v128");
  i32x4_ge_s = VECTOR_OP("i32x4_ge_s", 0x3f, ["v128", "v128"], "v128");
  i32x4_ge_u = VECTOR_OP("i32x4_ge_u", 0x40, ["v128", "v128"], "v128");

  f32x4_eq = VECTOR_OP("f32x4_eq", 0x41, ["v128", "v128"], "v128");
  f32x4_ne = VECTOR_OP("f32x4_ne", 0x42, ["v128", "v128"], "v128");
  f32x4_lt = VECTOR_OP("f32x4_lt", 0x43, ["v128", "v128"], "v128");
  f32x4_gt = VECTOR_OP("f32x4_gt", 0x44, ["v128", "v128"], "v128");
  f32x4_le = VECTOR_OP("f32x4_le", 0x45, ["v128", "v128"], "v128");
  f32x4_ge = VECTOR_OP("f32x4_ge", 0x46, ["v128", "v128"], "v128");

  not = VECTOR_OP("not", 0x4d, ["v128"], "v128");
  and = VECTOR_OP("and", 0x4e, ["v128", "v128"], "v128");
  andnot = VECTOR_OP("andnot", 0x4f, ["v128", "v128"], "v128");
  or = VECTOR_OP("or", 0x50, ["v128", "v128"], "v128");
  xor = VECTOR_OP("xor", 0x51, ["v128", "v128"], "v128");
  bitselect = VECTOR_OP("bitselect", 0x52, ["v128", "v128", "v128"], "v128");
  any_true = VECTOR_OP("any_true", 0x53, ["v128"], "i32");

  f32x4_ceil = VECTOR_OP("f32x4_ceil", 0x67, ["v128"], "v128");
  f32x4_floor = VECTOR_OP("f32x4_floor", 0x68, ["v128"], "v128");
  f32x4_trunc = VECTOR_OP("f32x4_trunc", 0x69, ["v128"], "v128");
  f32x4_nearest = VECTOR_OP("f32x4_nearest", 0x6a, ["v128"], "v128");

  i32x4_abs = VECTOR_OP("i32x4_abs", 0xa0, ["v128"], "v128");
  i32x4_neg = VECTOR_OP("i32x4_neg", 0xa1, ["v128"], "v128");
  i32x4_all_true = VECTOR_OP("i32x4_all_true", 0xa3, ["v128"], "i32");
  i32x4_bitmask = VECTOR_OP("i32x4_bitmask", 0xa4, ["v128"], "i32");
  i32x4_shl = VECTOR_OP("i32x4_shl", 0xab, ["v128", "i32"], "v128");
  i32x4_shr_s = VECTOR_OP("i32x4_shr_s", 0xac, ["v128", "i32"], "v128");
  i32x4_shr_u = VECTOR_OP("i32x4_shr_u", 0xad, ["v128", "i32"], "v128");
  i32x4_add = VECTOR_OP("i32x4_add", 0xae, ["v128", "v128"], "v128");
  i32x4_sub = VECTOR_OP("i32x4_sub", 0xb1, ["v128", "v128"], "v128");
  i32x4_mul = VECTOR_OP("i32x4_mul", 0xb5, ["v128", "v128"], "v128");
  i32x4_min_s = VECTOR_OP("i32x4_min_s", 0xb6, ["v128", "v128"], "v128");
  i32x4_min_u = VECTOR_OP("i32x4_min_u", 0xb7, ["v128", "v128"], "v128");
  i32x4_max_s = VECTOR_OP("i32x4_max_s", 0xb8, ["v128", "v128"], "v128");
  i32x4_max_u = VECTOR_OP("i32x4_max_u", 0xb9, ["v128", "v128"], "v128");

  f32x4_abs = VECTOR_OP("f32x4_abs", 0xe0, ["v128"], "v128");
  f32x4_neg = VECTOR_OP("f32x4_neg", 0xe1, ["v128"], "v128");
  f32x4_sqrt = VECTOR_OP("f32x4_sqrt", 0xe3, ["v128"], "v128");
  f32x4_add = VECTOR_OP("f32x4_add", 0xe4, ["v128", "v128"], "v128");
  f32x4_sub = VECTOR_OP("f32x4_sub", 0xe5, ["v128", "v128"], "v128");
  f32x4_mul = VECTOR_OP("f32x4_mul", 0xe6, ["v128", "v128"], "v128");
  f32x4_div = VECTOR_OP("f32x4_div", 0xe7, ["v128", "v128"], "v128");
  f32x4_min = VECTOR_OP("f32x4_min", 0xe8, ["v128", "v128"], "v128");
  f32x4_max = VECTOR_OP("f32x4_max", 0xe9, ["v128", "v128"], "v128");
  f32x4_pmin = VECTOR_OP("f32x4_pmin", 0xea, ["v128", "v128"], "v128");
  f32x4_pmax = VECTOR_OP("f32x4_pmax", 0xeb, ["v128", "v128"], "v128");
}
