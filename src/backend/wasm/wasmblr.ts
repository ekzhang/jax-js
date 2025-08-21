/**
 * @file Minimalist WebAssembly assembler. This allows you to emit WebAssembly
 * bytecode directly from the browser.
 *
 * Self-contained port of https://github.com/bwasti/wasmblr to TypeScript.
 * Some operation names in this module are written in `snake_case` to match
 * their names in the Wasm specification.
 *
 * Reference: https://pengowray.github.io/wasm-ops/.
 */

const magicModuleHeader = [0x00, 0x61, 0x73, 0x6d];
const moduleVersion = [0x01, 0x00, 0x00, 0x00];

function assert(condition: boolean, message?: string): asserts condition {
  if (!condition) {
    throw new Error(message || "Assertion failed");
  }
}

// From LLVM
function encodeSigned(n: number): number[] {
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

function encodeUnsigned(n: number): number[] {
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

function encodeString(s: string): number[] {
  const bytes = new TextEncoder().encode(s);
  return [bytes.length, ...bytes];
}

function encodeBlocktype(type: Type[]): number[] {
  assert(type.length > 0, "blocktype must have at least one type");
  if (type.length === 1) return [type[0].typeId];
  // multi-value block: https://github.com/WebAssembly/multi-value
  return [
    0x60, // function type
    ...encodeUnsigned(0),
    ...encodeUnsigned(type.length),
    ...type.map((t) => t.typeId),
  ];
}

function concat(out: number[], inp: number[]): void {
  out.push(...inp);
}

class Function_ {
  inputTypes: Type[];
  outputTypes: Type[];
  body: () => void;
  locals: Type[] = [];
  constructor(inputTypes: Type[], outputTypes: Type[], body?: () => void) {
    this.inputTypes = inputTypes;
    this.outputTypes = outputTypes;
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
  isShared = false;
  aString = "";
  bString = "";

  constructor(readonly cg: CodeGenerator) {}

  /** Declare the size of the memory. Each page is 64 KiB. */
  pages(min: number, max: number = 0): this {
    assert(this.min === 0 && this.max === 0);
    this.min = min;
    this.max = max;
    return this;
  }

  export(a: string): this {
    assert(!this.isImport && !this.isExport, "already set");
    this.aString = a;
    return this;
  }

  shared(isShared: boolean): this {
    this.isShared = isShared;
    return this;
  }

  import(a: string, b: string): this {
    assert(!this.isImport && !this.isExport, "already set");
    this.aString = a;
    this.bString = b;
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

  get isImport(): boolean {
    return this.aString.length > 0 && this.bString.length > 0;
  }
  get isExport(): boolean {
    return this.aString.length > 0 && this.bString.length === 0;
  }
}

////////////////////////////////////////
// CodeGenerator class
////////////////////////////////////////

// I32, F32, V128, Void, I32x4, F32x4
interface Type {
  typeId: number;
}

/** Public API of WebAssembly assembler. */
export class CodeGenerator {
  local: Local;
  i32: I32;
  f32: F32;
  v128: V128;
  i32x4: I32x4;
  f32x4: F32x4;
  memory: Memory;
  void = { typeId: 0x40 };

  #functions: Function_[] = [];
  #exportedFunctions = new Map<number, string>();
  #curFunction: Function_ | null = null;
  #curBytes: number[] = [];
  #typeStack: Type[] = [];

  constructor() {
    this.local = new Local(this);
    this.i32 = new I32(this);
    this.f32 = new F32(this);
    this.v128 = new V128(this);
    this.i32x4 = new I32x4(this);
    this.f32x4 = new F32x4(this);
    this.memory = new Memory(this);
  }

  // Control and branching instructions
  nop() {
    this.emit(0x01);
  }
  block(...type: Type[]) {
    this.emit(0x02);
    this.emit(encodeBlocktype(type));
  }
  loop(...type: Type[]) {
    this.emit(0x03);
    this.emit(encodeBlocktype(type));
  }
  if(...type: Type[]) {
    assert(this.pop().typeId === this.i32.typeId, "if_: expected i32");
    this.emit(0x04);
    this.emit(encodeBlocktype(type));
  }
  else() {
    this.emit(0x05);
  }
  /** End a block (`block`, `if`/`else`, `loop`, or function). */
  end() {
    this.emit(0x0b);
  }
  /** Branch to a block a certain depth outward on the stack. */
  br(depth: number) {
    this.emit(0x0c);
    this.emit(encodeUnsigned(depth));
  }
  /** Conditional branch to a block a certain depth outward on the stack. */
  br_if(depth: number) {
    assert(this.pop().typeId === this.i32.typeId, "br_if: expected i32");
    this.emit(0x0d);
    this.emit(encodeUnsigned(depth));
  }
  /** Return from a function, branching out of the outermost block. */
  return() {
    this.emit(0x0f);
  }
  /** Call a function with the given ID. */
  call(fn: number) {
    assert(fn < this.#functions.length, "function index does not exist");
    this.emit(0x10);
    this.emit(encodeUnsigned(fn));
  }
  /** Throw away an operand on the stack. */
  drop() {
    this.pop();
    this.emit(0x1a);
  }
  /** Select one of the first two operands based on the third operand (i32)'s value. */
  select() {
    assert(
      this.pop().typeId === this.i32.typeId,
      "select: expected i32 condition",
    );
    // condition ? a : b
    const [b, a] = [this.pop(), this.pop()];
    assert(
      a.typeId === b.typeId,
      "select: expected same type for both operands",
    );
    this.emit(0x1b);
  }

  /** Export a function. */
  export(fn: number, name: string) {
    this.#exportedFunctions.set(fn, name);
  }

  /** Declare a new function; returns its index. */
  function(inputTypes: Type[], outputTypes: Type[], body: () => void): number {
    const idx = this.#functions.length;
    this.#functions.push(new Function_(inputTypes, outputTypes, body));
    return idx;
  }

  // --- Implementation helpers

  declareLocal(type: Type): number {
    assert(this.#curFunction !== null, "No current function");
    const idx =
      this.#curFunction.locals.length + this.#curFunction.inputTypes.length;
    this.#curFunction.locals.push(type);
    return idx;
  }

  inputTypes(): Type[] {
    assert(this.#curFunction !== null, "No current function");
    return this.#curFunction.inputTypes;
  }

  locals(): Type[] {
    assert(this.#curFunction !== null, "No current function");
    return this.#curFunction.locals;
  }

  push(type: Type) {
    this.#typeStack.push(type);
  }
  pop(): Type {
    assert(this.#typeStack.length > 0, "popping empty stack");
    return this.#typeStack.pop()!;
  }

  emit(bytes: number | number[]) {
    if (typeof bytes === "number") this.#curBytes.push(bytes);
    else this.#curBytes.push(...bytes);
  }

  // Emit the complete module as an array of bytes.
  finish(): Uint8Array {
    this.#curBytes = [];
    const emittedBytes: number[] = [];
    concat(emittedBytes, magicModuleHeader);
    concat(emittedBytes, moduleVersion);

    // Type section
    const typeSectionBytes: number[] = [];
    concat(typeSectionBytes, encodeUnsigned(this.#functions.length));
    for (const f of this.#functions) {
      typeSectionBytes.push(0x60);
      concat(typeSectionBytes, encodeUnsigned(f.inputTypes.length));
      for (const t of f.inputTypes) {
        typeSectionBytes.push(t.typeId);
      }
      concat(typeSectionBytes, encodeUnsigned(f.outputTypes.length));
      for (const t of f.outputTypes) {
        typeSectionBytes.push(t.typeId);
      }
    }
    emittedBytes.push(0x01);
    concat(emittedBytes, encodeUnsigned(typeSectionBytes.length));
    concat(emittedBytes, typeSectionBytes);

    // Import section (for memory import)
    const importSectionBytes: number[] = [];
    if (this.memory.isImport) {
      // one import
      concat(importSectionBytes, encodeUnsigned(1));
      concat(importSectionBytes, encodeString(this.memory.aString));
      concat(importSectionBytes, encodeString(this.memory.bString));
      importSectionBytes.push(0x02); // memory flag
      if (this.memory.min && this.memory.max) {
        if (this.memory.isShared) {
          importSectionBytes.push(0x03);
        } else {
          importSectionBytes.push(0x01);
        }
        concat(importSectionBytes, encodeUnsigned(this.memory.min));
        concat(importSectionBytes, encodeUnsigned(this.memory.max));
      } else {
        assert(!this.memory.isShared, "shared memory must have a max size");
        concat(importSectionBytes, encodeUnsigned(this.memory.min));
      }
      emittedBytes.push(0x02);
      concat(emittedBytes, encodeUnsigned(importSectionBytes.length));
      concat(emittedBytes, importSectionBytes);
    }

    // Function section
    const functionSectionBytes: number[] = [];
    concat(functionSectionBytes, encodeUnsigned(this.#functions.length));
    for (let i = 0; i < this.#functions.length; i++) {
      concat(functionSectionBytes, encodeUnsigned(i));
    }
    emittedBytes.push(0x03);
    concat(emittedBytes, encodeUnsigned(functionSectionBytes.length));
    concat(emittedBytes, functionSectionBytes);

    // Memory section (if defined locally)
    const memorySectionBytes: number[] = [];
    if (!this.memory.isImport && (this.memory.min || this.memory.max)) {
      memorySectionBytes.push(0x01); // always one memory
      if (this.memory.min && this.memory.max) {
        if (this.memory.isShared) {
          memorySectionBytes.push(0x03);
        } else {
          memorySectionBytes.push(0x01);
        }
        concat(memorySectionBytes, encodeUnsigned(this.memory.min));
        concat(memorySectionBytes, encodeUnsigned(this.memory.max));
      } else {
        assert(!this.memory.isShared, "shared memory must have a max size");
        memorySectionBytes.push(0x00);
        concat(memorySectionBytes, encodeUnsigned(this.memory.min));
      }
      emittedBytes.push(0x05);
      concat(emittedBytes, encodeUnsigned(memorySectionBytes.length));
      concat(emittedBytes, memorySectionBytes);
    }

    // Export section
    const exportSectionBytes: number[] = [];
    const numExports =
      this.#exportedFunctions.size + (this.memory.isExport ? 1 : 0);
    concat(exportSectionBytes, encodeUnsigned(numExports));
    if (this.memory.isExport) {
      concat(exportSectionBytes, encodeString(this.memory.aString));
      exportSectionBytes.push(0x02);
      exportSectionBytes.push(0x00); // one memory at index 0
    }
    for (const [key, name] of this.#exportedFunctions.entries()) {
      concat(exportSectionBytes, encodeString(name));
      exportSectionBytes.push(0x00);
      concat(exportSectionBytes, encodeUnsigned(key));
    }
    emittedBytes.push(0x07);
    concat(emittedBytes, encodeUnsigned(exportSectionBytes.length));
    concat(emittedBytes, exportSectionBytes);

    // Code section
    const codeSectionBytes: number[] = [];
    concat(codeSectionBytes, encodeUnsigned(this.#functions.length));
    for (const f of this.#functions) {
      this.#curFunction = f;
      this.#curBytes = [];
      f.emit();
      this.end();
      const bodyBytes = [...this.#curBytes];
      this.#curBytes = [];
      // Header: local declarations
      concat(this.#curBytes, encodeUnsigned(f.locals.length));
      for (const l of f.locals) {
        this.emit(0x01);
        this.emit(l.typeId);
      }
      const headerBytes = [...this.#curBytes];
      const fnSize = headerBytes.length + bodyBytes.length;
      concat(codeSectionBytes, encodeUnsigned(fnSize));
      concat(codeSectionBytes, headerBytes);
      concat(codeSectionBytes, bodyBytes);
    }
    this.#curFunction = null;

    emittedBytes.push(0x0a);
    concat(emittedBytes, encodeUnsigned(codeSectionBytes.length));
    concat(emittedBytes, codeSectionBytes);

    return new Uint8Array(emittedBytes);
  }
}

////////////////////////////////////////
// Local variables
////////////////////////////////////////

class Local {
  constructor(readonly cg: CodeGenerator) {}

  // Mimic operator()(type)
  declare(type: Type): number {
    return this.cg.declareLocal(type);
  }
  get(idx: number) {
    const inputTypes = this.cg.inputTypes();
    if (idx < inputTypes.length) {
      this.cg.push(inputTypes[idx]);
    } else {
      this.cg.push(this.cg.locals()[idx - inputTypes.length]);
    }
    this.cg.emit(0x20);
    this.cg.emit(encodeUnsigned(idx));
  }
  set(idx: number) {
    const t = this.cg.pop();
    const inputTypes = this.cg.inputTypes();
    const expectedType =
      idx < inputTypes.length
        ? inputTypes[idx]
        : this.cg.locals()[idx - inputTypes.length];
    assert(
      expectedType.typeId === t.typeId,
      "can't set local to this value (wrong type)",
    );
    this.cg.emit(0x21);
    this.cg.emit(encodeUnsigned(idx));
  }
  tee(idx: number) {
    const t = this.cg.pop();
    const inputTypes = this.cg.inputTypes();
    const expectedType =
      idx < inputTypes.length
        ? inputTypes[idx]
        : this.cg.locals()[idx - inputTypes.length];
    assert(
      expectedType.typeId === t.typeId,
      "can't tee local to this value (wrong type)",
    );
    this.cg.emit(0x22);
    this.cg.emit(encodeUnsigned(idx));
    this.cg.push(expectedType);
  }
}

type TypeSpec = "i32" | "f32";

function UNARY_OP(
  op: string,
  opcode: number,
  inType: TypeSpec,
  outType: TypeSpec,
) {
  return function (this: { cg: CodeGenerator }) {
    const t = this.cg.pop();
    assert(
      t.typeId === this.cg[inType].typeId,
      `invalid type for ${op} (${inType} -> ${outType})`,
    );
    this.cg.emit(opcode);
    this.cg.push(this.cg[outType]);
  };
}

function BINARY_OP(
  op: string,
  opcode: number,
  typeA: TypeSpec,
  typeB: TypeSpec,
  outType: TypeSpec,
) {
  return function (this: { cg: CodeGenerator }) {
    const b = this.cg.pop();
    const a = this.cg.pop();
    assert(
      a.typeId === this.cg[typeA].typeId && b.typeId === this.cg[typeB].typeId,
      `invalid type for ${op} (${typeA}, ${typeB} -> ${outType})`,
    );
    this.cg.emit(opcode);
    this.cg.push(this.cg[outType]);
  };
}

function LOAD_OP(op: string, opcode: number, outType: TypeSpec) {
  return function (
    this: { cg: CodeGenerator },
    align: number = 0,
    offset: number = 0,
  ) {
    const idxType = this.cg.pop();
    assert(idxType.typeId === this.cg.i32.typeId, `invalid type for ${op}`);
    this.cg.emit(opcode);
    this.cg.emit(encodeUnsigned(align));
    this.cg.emit(encodeUnsigned(offset));
    this.cg.push(this.cg[outType]);
  };
}

function STORE_OP(op: string, opcode: number, inType: TypeSpec) {
  return function (
    this: { cg: CodeGenerator },
    align: number = 0,
    offset: number = 0,
  ) {
    const valType = this.cg.pop();
    const idxType = this.cg.pop();
    assert(
      valType.typeId === this.cg[inType].typeId,
      `invalid value type for ${op} (${inType})`,
    );
    assert(idxType.typeId === this.cg.i32.typeId, `invalid type for ${op}`);
    this.cg.emit(opcode);
    this.cg.emit(encodeUnsigned(align));
    this.cg.emit(encodeUnsigned(offset));
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

  const(i: number) {
    this.cg.emit(0x41);
    this.cg.emit(encodeSigned(i));
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
  trunc_f32_s = UNARY_OP("trunc_f32_s", 0xa8, "f32", "i32");
  trunc_f32_u = UNARY_OP("trunc_f32_u", 0xa9, "f32", "i32");
  load = LOAD_OP("load", 0x28, "i32");
  load8_s = LOAD_OP("load8_s", 0x2c, "i32");
  load8_u = LOAD_OP("load8_u", 0x2d, "i32");
  load16_s = LOAD_OP("load16_s", 0x2e, "i32");
  load16_u = LOAD_OP("load16_u", 0x2f, "i32");
  store = STORE_OP("store", 0x36, "i32");
  store8 = STORE_OP("store8", 0x3a, "i32");
  store16 = STORE_OP("store16", 0x3b, "i32");
}

////////////////////////////////////////
// F32 class
////////////////////////////////////////

class F32 {
  constructor(readonly cg: CodeGenerator) {}
  get typeId(): number {
    return 0x7d;
  }

  const(f: number) {
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
  convert_i32_s = UNARY_OP("convert_i32_s", 0xb2, "i32", "f32");
  convert_i32_u = UNARY_OP("convert_i32_u", 0xb3, "i32", "f32");
  load = LOAD_OP("load", 0x2a, "f32");
  store = STORE_OP("store", 0x38, "f32");
}

////////////////////////////////////////
// Vector types (SIMD)
////////////////////////////////////////

type VectorTypeSpec = TypeSpec | "v128";

function VECTOR_OP(
  op: string,
  vopcode: number,
  inTypes: VectorTypeSpec[],
  outType: VectorTypeSpec,
) {
  return function (this: { cg: CodeGenerator }) {
    for (const inType of inTypes.toReversed()) {
      const actualType = this.cg.pop();
      assert(
        actualType.typeId === this.cg[inType].typeId,
        `invalid type for ${op} (${inTypes.join(", ")} -> ${outType})`,
      );
    }
    this.cg.emit(0xfd);
    this.cg.emit(encodeUnsigned(vopcode));
    this.cg.push(this.cg[outType]);
  };
}

// Like VECTOR_OP but also takes a lane.
function VECTOR_OPL(
  op: string,
  vopcode: number,
  inTypes: VectorTypeSpec[],
  outType: VectorTypeSpec,
) {
  return function (this: { cg: CodeGenerator }, lane: number) {
    for (const inType of inTypes.toReversed()) {
      const actualType = this.cg.pop();
      assert(
        actualType.typeId === this.cg[inType].typeId,
        `invalid type for ${op} (${inTypes} -> ${outType})`,
      );
    }
    this.cg.emit(0xfd);
    this.cg.emit(encodeUnsigned(vopcode));
    this.cg.emit(lane); // 1 byte
    this.cg.push(this.cg[outType]);
  };
}

function VECTOR_LOAD_OP(op: string, vopcode: number) {
  return function (this: any, align: number = 0, offset: number = 0) {
    const idxType = this.cg.pop();
    assert(idxType.typeId === this.cg.i32.typeId, `invalid type for ${op}`);
    this.cg.emit(0xfd);
    this.cg.emit(encodeUnsigned(vopcode));
    this.cg.emit(encodeUnsigned(align));
    this.cg.emit(encodeUnsigned(offset));
    this.cg.push(this.cg.v128);
  };
}

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

  store(align: number = 0, offset: number = 0) {
    const valType = this.cg.pop();
    assert(valType.typeId === this.cg.v128.typeId, `invalid type for store`);
    const idxType = this.cg.pop();
    assert(idxType.typeId === this.cg.i32.typeId, `invalid type for store`);
    this.cg.emit(0xfd);
    this.cg.emit(encodeUnsigned(0x0b));
    this.cg.emit(encodeUnsigned(align));
    this.cg.emit(encodeUnsigned(offset));
  }

  not = VECTOR_OP("not", 0x4d, ["v128"], "v128");
  and = VECTOR_OP("and", 0x4e, ["v128", "v128"], "v128");
  andnot = VECTOR_OP("andnot", 0x4f, ["v128", "v128"], "v128");
  or = VECTOR_OP("or", 0x50, ["v128", "v128"], "v128");
  xor = VECTOR_OP("xor", 0x51, ["v128", "v128"], "v128");
  bitselect = VECTOR_OP("bitselect", 0x52, ["v128", "v128", "v128"], "v128");
  any_true = VECTOR_OP("any_true", 0x53, ["v128"], "i32");
}

class I32x4 extends V128 {
  splat = VECTOR_OP("splat", 0x11, ["i32"], "v128");
  extract_lane = VECTOR_OPL("extract_lane", 0x1b, ["v128"], "i32");
  replace_lane = VECTOR_OPL("replace_lane", 0x1c, ["v128", "i32"], "v128");

  eq = VECTOR_OP("eq", 0x37, ["v128", "v128"], "v128");
  ne = VECTOR_OP("ne", 0x38, ["v128", "v128"], "v128");
  lt_s = VECTOR_OP("lt_s", 0x39, ["v128", "v128"], "v128");
  lt_u = VECTOR_OP("lt_u", 0x3a, ["v128", "v128"], "v128");
  gt_s = VECTOR_OP("gt_s", 0x3b, ["v128", "v128"], "v128");
  gt_u = VECTOR_OP("gt_u", 0x3c, ["v128", "v128"], "v128");
  le_s = VECTOR_OP("le_s", 0x3d, ["v128", "v128"], "v128");
  le_u = VECTOR_OP("le_u", 0x3e, ["v128", "v128"], "v128");
  ge_s = VECTOR_OP("ge_s", 0x3f, ["v128", "v128"], "v128");
  ge_u = VECTOR_OP("ge_u", 0x40, ["v128", "v128"], "v128");

  abs = VECTOR_OP("abs", 0xa0, ["v128"], "v128");
  neg = VECTOR_OP("neg", 0xa1, ["v128"], "v128");
  all_true = VECTOR_OP("all_true", 0xa3, ["v128"], "i32");
  bitmask = VECTOR_OP("bitmask", 0xa4, ["v128"], "i32");
  shl = VECTOR_OP("shl", 0xab, ["v128", "i32"], "v128");
  shr_s = VECTOR_OP("shr_s", 0xac, ["v128", "i32"], "v128");
  shr_u = VECTOR_OP("shr_u", 0xad, ["v128", "i32"], "v128");
  add = VECTOR_OP("add", 0xae, ["v128", "v128"], "v128");
  sub = VECTOR_OP("sub", 0xb1, ["v128", "v128"], "v128");
  mul = VECTOR_OP("mul", 0xb5, ["v128", "v128"], "v128");
  min_s = VECTOR_OP("min_s", 0xb6, ["v128", "v128"], "v128");
  min_u = VECTOR_OP("min_u", 0xb7, ["v128", "v128"], "v128");
  max_s = VECTOR_OP("max_s", 0xb8, ["v128", "v128"], "v128");
  max_u = VECTOR_OP("max_u", 0xb9, ["v128", "v128"], "v128");
}

class F32x4 extends V128 {
  splat = VECTOR_OP("splat", 0x13, ["f32"], "v128");
  extract_lane = VECTOR_OPL("extract_lane", 0x1f, ["v128"], "f32");
  replace_lane = VECTOR_OPL("replace_lane", 0x20, ["v128", "f32"], "v128");

  eq = VECTOR_OP("eq", 0x41, ["v128", "v128"], "v128");
  ne = VECTOR_OP("ne", 0x42, ["v128", "v128"], "v128");
  lt = VECTOR_OP("lt", 0x43, ["v128", "v128"], "v128");
  gt = VECTOR_OP("gt", 0x44, ["v128", "v128"], "v128");
  le = VECTOR_OP("le", 0x45, ["v128", "v128"], "v128");
  ge = VECTOR_OP("ge", 0x46, ["v128", "v128"], "v128");

  ceil = VECTOR_OP("ceil", 0x67, ["v128"], "v128");
  floor = VECTOR_OP("floor", 0x68, ["v128"], "v128");
  trunc = VECTOR_OP("trunc", 0x69, ["v128"], "v128");
  nearest = VECTOR_OP("nearest", 0x6a, ["v128"], "v128");

  abs = VECTOR_OP("abs", 0xe0, ["v128"], "v128");
  neg = VECTOR_OP("neg", 0xe1, ["v128"], "v128");
  sqrt = VECTOR_OP("sqrt", 0xe3, ["v128"], "v128");
  add = VECTOR_OP("add", 0xe4, ["v128", "v128"], "v128");
  sub = VECTOR_OP("sub", 0xe5, ["v128", "v128"], "v128");
  mul = VECTOR_OP("mul", 0xe6, ["v128", "v128"], "v128");
  div = VECTOR_OP("div", 0xe7, ["v128", "v128"], "v128");
  min = VECTOR_OP("min", 0xe8, ["v128", "v128"], "v128");
  max = VECTOR_OP("max", 0xe9, ["v128", "v128"], "v128");
  pmin = VECTOR_OP("pmin", 0xea, ["v128", "v128"], "v128");
  pmax = VECTOR_OP("pmax", 0xeb, ["v128", "v128"], "v128");
}
