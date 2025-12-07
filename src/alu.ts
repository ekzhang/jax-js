import { PPrint } from "./pprint";
import { ShapeTracker } from "./shape";
import { clamp, FpHash, FpHashable, gcd, strip1 } from "./utils";

/** A numerical data type for array contents. */
export enum DType {
  Float32 = "float32",
  Int32 = "int32",
  Uint32 = "uint32",
  Bool = "bool",
  Float16 = "float16",
}

/** @inline */
export type DataArray =
  | Float32Array<ArrayBuffer>
  | Int32Array<ArrayBuffer>
  | Uint32Array<ArrayBuffer>
  | Float16Array<ArrayBuffer>;

export const byteWidth = (dtype: DType): number => {
  switch (dtype) {
    case DType.Float32:
    case DType.Int32:
    case DType.Uint32:
    case DType.Bool:
      return 4;
    case DType.Float16:
      return 2;
    default:
      throw new TypeError(`Unknown dtype: ${dtype}`);
  }
};

export const isFloatDtype = (dtype: DType) =>
  dtype === DType.Float32 || dtype === DType.Float16;

/**
 * Promote two dtypes to their join according to the type lattice.
 *
 * When performing operations between arrays of different types, we need to
 * promote both operands to a common type that can represent values from both
 * input types. This follows JAX's type promotion rules.
 *
 * **Type lattice:**
 * ```text
 * bool -> uint32 -> int32 -> float16 -> float32
 *  weakType --^
 * ```
 *
 * `weakType` represents weakly typed arrays. These are created for JS numbers,
 * which default to float32 but "weak" so they cast to the dtype of any array
 * they are first combined with, except `bool`.
 *
 * **Examples:**
 * - `promoteTypes(bool, int32) → int32`
 * - `promoteTypes(uint32, int32) → int32`
 * - `promoteTypes(int32, float16) → float16`
 * - `promoteTypes(float16, float32) → float32`
 * - `promoteTypes(uint32, float32) → float32`
 */
export function promoteTypes(dtype1: DType, dtype2: DType): DType {
  if (dtype1 === dtype2) return dtype1;

  // Define the promotion order in a linear chain (higher number = later in chain)
  const rank: Record<DType, number> = {
    [DType.Bool]: 0,
    [DType.Uint32]: 1,
    [DType.Int32]: 2,
    [DType.Float16]: 3,
    [DType.Float32]: 4,
  };

  // Take the type that appears later in the chain
  return rank[dtype1] > rank[dtype2] ? dtype1 : dtype2;
}

export function dtypedArray(
  dtype: DType,
  data: Uint8Array<ArrayBuffer>,
): DataArray {
  const { buffer, byteLength, byteOffset } = data;
  const length = byteLength / byteWidth(dtype);
  switch (dtype) {
    case DType.Float32:
      return new Float32Array(buffer, byteOffset, length);
    case DType.Int32:
    case DType.Bool: // Booleans are stored as 0/1 in int32.
      return new Int32Array(buffer, byteOffset, length);
    case DType.Uint32:
      return new Uint32Array(buffer, byteOffset, length);
    case DType.Float16:
      return new Float16Array(buffer, byteOffset, length);
    default:
      throw new Error(`Unimplemented dtype: ${dtype}`);
  }
}

export function dtypedJsArray(dtype: DType, data: number[]): DataArray {
  switch (dtype) {
    case DType.Float32:
      return new Float32Array(data);
    case DType.Int32:
    case DType.Bool: // Booleans are stored as 0/1 in int32.
      return new Int32Array(data);
    case DType.Uint32:
      return new Uint32Array(data);
    case DType.Float16:
      return new Float16Array(data);
    default:
      throw new Error(`Unimplemented dtype: ${dtype}`);
  }
}

/**
 * Mathematical expression on scalar values.
 *
 * This is similiar to and based on tinygrad's UOp class, but it's more specific
 * to just math on scalars. We're doing this to avoid the complexity of a full
 * graph rewrite engine.
 */
export class AluExp implements FpHashable {
  #hash?: bigint;
  #simplified?: AluExp;
  #range?: [number, number];

  constructor(
    readonly op: AluOp,
    readonly dtype: DType,
    readonly src: AluExp[],
    readonly arg: any = undefined,
  ) {
    if (AluGroup.RequiredFloat.has(op) && !isFloatDtype(dtype))
      throw new TypeError(`Unsupported dtype for ${op}: ${dtype}`);
    if (
      op === AluOp.Bitcast &&
      (dtype === DType.Bool ||
        src[0].dtype === DType.Bool ||
        byteWidth(dtype) !== byteWidth(src[0].dtype))
    )
      throw new TypeError(`Bitcast from ${src[0].dtype} -> ${dtype}`);
    if (
      op === AluOp.Threefry2x32 &&
      (dtype !== DType.Uint32 || src.some((x) => x.dtype !== DType.Uint32))
    )
      throw new TypeError("Threefry2x32 requires uint32 types");
  }

  static add(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Add, a.dtype, [a, b]);
  }
  static sub(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Sub, a.dtype, [a, b]);
  }
  static mul(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Mul, a.dtype, [a, b]);
  }
  static idiv(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Idiv, a.dtype, [a, b]);
  }
  static mod(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Mod, a.dtype, [a, b]);
  }
  static min(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Min, a.dtype, [a, b]);
  }
  static max(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Max, a.dtype, [a, b]);
  }
  static sin(a: AluExp): AluExp {
    return new AluExp(AluOp.Sin, a.dtype, [a]);
  }
  static cos(a: AluExp): AluExp {
    return new AluExp(AluOp.Cos, a.dtype, [a]);
  }
  static asin(a: AluExp): AluExp {
    return new AluExp(AluOp.Asin, a.dtype, [a]);
  }
  static atan(a: AluExp): AluExp {
    return new AluExp(AluOp.Atan, a.dtype, [a]);
  }
  static exp(a: AluExp): AluExp {
    return new AluExp(AluOp.Exp, a.dtype, [a]);
  }
  static log(a: AluExp): AluExp {
    return new AluExp(AluOp.Log, a.dtype, [a]);
  }
  static sqrt(a: AluExp): AluExp {
    return new AluExp(AluOp.Sqrt, a.dtype, [a]);
  }
  static reciprocal(a: AluExp): AluExp {
    return new AluExp(AluOp.Reciprocal, a.dtype, [a]);
  }
  static cast(dtype: DType, a: AluExp): AluExp {
    if (a.dtype === dtype) return a;
    return new AluExp(AluOp.Cast, dtype, [a]);
  }
  static bitcast(dtype: DType, a: AluExp): AluExp {
    if (a.dtype === dtype) return a;
    return new AluExp(AluOp.Bitcast, dtype, [a]);
  }
  static threefry2x32(
    k0: AluExp,
    k1: AluExp,
    c0: AluExp,
    c1: AluExp,
    mode: "xor" | 0 | 1 = "xor",
  ): AluExp {
    return new AluExp(AluOp.Threefry2x32, DType.Uint32, [k0, k1, c0, c1], mode);
  }
  static cmplt(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Cmplt, DType.Bool, [a, b]);
  }
  static cmpne(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Cmpne, DType.Bool, [a, b]);
  }
  static where(cond: AluExp, a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Where, a.dtype, [cond, a, b]);
  }
  static const(dtype: DType, value: any): AluExp {
    if (dtype === DType.Bool) {
      value = Number(Boolean(value));
    } else if (dtype === DType.Int32) {
      value = Math.trunc(value) | 0;
    } else if (dtype === DType.Uint32) {
      value = Math.trunc(value) >>> 0;
    }
    if (typeof value !== "number") {
      throw new TypeError(
        `Expected a number for constant, got ${typeof value}: ${value}`,
      );
    }
    return new AluExp(AluOp.Const, dtype, [], value);
  }
  static special(dtype: DType, name: string, n: number): AluExp {
    return new AluExp(AluOp.Special, dtype, [], [name, n]);
  }
  static variable(dtype: DType, name: string): AluExp {
    return new AluExp(AluOp.Variable, dtype, [], name);
  }
  static globalIndex(
    dtype: DType,
    gid: number,
    len: number,
    bufidx: AluExp,
  ): AluExp {
    return new AluExp(AluOp.GlobalIndex, dtype, [bufidx], [gid, len]);
  }
  static globalView(
    dtype: DType,
    gid: number,
    st: ShapeTracker,
    indices: AluExp[],
  ): AluExp {
    return new AluExp(AluOp.GlobalView, dtype, indices, [gid, st]);
  }

  static f32(value: number): AluExp {
    return AluExp.const(DType.Float32, value);
  }
  static i32(value: number): AluExp {
    return AluExp.const(DType.Int32, value);
  }
  static u32(value: number): AluExp {
    return AluExp.const(DType.Uint32, value);
  }
  static bool(value: boolean): AluExp {
    return AluExp.const(DType.Bool, Number(value));
  }
  static f16(value: number): AluExp {
    return AluExp.const(DType.Float16, value);
  }

  not(): AluExp {
    if (this.dtype !== DType.Bool) {
      throw new Error("not() can only be called on boolean expressions");
    }
    return AluExp.cmpne(this, AluExp.const(DType.Bool, true));
  }

  /** Compute a reasonable expression hash with low collision rate. */
  getHash(): bigint {
    if (this.#hash !== undefined) return this.#hash;
    const hasher = new FpHash();
    hasher.update(this.op);
    hasher.update(this.dtype);
    hasher.update(JSON.stringify(this.arg));
    hasher.update(this.src.length);
    for (const s of this.src) hasher.update(s);
    this.#hash = hasher.value;
    return this.#hash;
  }

  hash(state: FpHash): void {
    state.update(this.getHash());
  }

  /** Substitute variables in this AluExp to values. */
  substitute(variables: Record<string, AluExp>): AluExp {
    return this.rewrite((exp) => {
      if (exp.op === AluOp.Variable && Object.hasOwn(variables, exp.arg)) {
        if (exp.dtype !== variables[exp.arg].dtype) {
          throw new Error(
            `Type mismatch: ${exp.dtype} vs ${variables[exp.arg].dtype}`,
          );
        }
        return variables[exp.arg];
      }
    });
  }

  /** Reindex gid values in this expression as needed. */
  reindexGids(gidMap: Map<number, number>): AluExp {
    return this.rewrite((exp) => {
      if (exp.op === AluOp.GlobalIndex) {
        const [gid, len] = exp.arg as [number, number];
        const newGid = gidMap.get(gid);
        if (newGid !== undefined && newGid !== gid) {
          return AluExp.globalIndex(exp.dtype, newGid, len, exp.src[0]);
        }
      } else if (exp.op === AluOp.GlobalView) {
        const gid = exp.arg[0] as number;
        const newGid = gidMap.get(gid);
        if (newGid !== undefined && newGid !== gid) {
          return AluExp.globalView(exp.dtype, newGid, exp.arg[1], exp.src);
        }
      }
    });
  }

  #computeRange(): [number, number] {
    if (this.#range !== undefined) return this.#range;

    const src = this.src;
    const minMax4 = (f: (a: number, b: number) => number) => {
      const [r1, r2] = [src[0].#computeRange(), src[1].#computeRange()];
      const values = [
        f(r1[0], r2[0]),
        f(r1[0], r2[1]),
        f(r1[1], r2[0]),
        f(r1[1], r2[1]),
      ];
      return [Math.min(...values), Math.max(...values)] as [number, number];
    };

    let ret: [number, number];
    switch (this.op) {
      case AluOp.Add:
        ret = [src[0].min + src[1].min, src[0].max + src[1].max];
        break;
      case AluOp.Sub:
        ret = [src[0].min - src[1].max, src[0].max - src[1].min];
        break;
      case AluOp.Mul: {
        ret = minMax4((a, b) => a * b);
        break;
      }
      case AluOp.Idiv: {
        ret = minMax4((a, b) => Math.trunc(a / b));
        break;
      }
      case AluOp.Mod: {
        // Mod is a bit tricky since it can be negative. Two behaviors:
        //
        // - C-style: In JS and WGSL, mod depends on the sign of the dividend.
        //   Matches int division that truncates to zero.
        // - Python-style: But in NumPy, JAX, and Python, it's based on the sign
        //   of the divisor (% or the np.remainder function). Matches floor
        //   division that rounds towards -inf.
        //
        // We're going to use the C-style behavior since it's more common in the
        // web world and outside of Python. This is a deviation from JAX!
        let divisorRange = src[1].#computeRange();
        if (divisorRange[0] <= 0 && divisorRange[1] >= 0) {
          divisorRange = [0, Math.max(-divisorRange[0], divisorRange[1])];
        }
        if (divisorRange[1] < 0) {
          divisorRange = [-divisorRange[1], -divisorRange[0]];
        }
        const maxDivisor = isFloatDtype(this.dtype)
          ? divisorRange[1]
          : divisorRange[1] - 1;
        ret = [
          clamp(src[0].min, -maxDivisor, 0),
          clamp(src[0].max, 0, maxDivisor),
        ];
        break;
      }
      case AluOp.Min:
        ret = [
          Math.min(src[0].min, src[1].min),
          Math.min(src[0].max, src[1].max),
        ];
        break;
      case AluOp.Max:
        ret = [
          Math.max(src[0].min, src[1].min),
          Math.max(src[0].max, src[1].max),
        ];
        break;

      case AluOp.Sin:
        ret = [-1, 1];
        break;
      case AluOp.Cos:
        ret = [-1, 1];
        break;
      case AluOp.Asin:
        ret = [-Math.PI / 2, Math.PI / 2];
        break;
      case AluOp.Atan:
        ret = [-Math.PI / 2, Math.PI / 2];
        break;
      case AluOp.Exp:
        ret = [Math.exp(src[0].min), Math.exp(src[0].max)];
        break;
      case AluOp.Log:
        ret = [Math.log(src[0].min), Math.log(src[0].max)];
        break;
      case AluOp.Sqrt:
        ret = [Math.sqrt(src[0].min), Math.sqrt(src[0].max)];
        break;
      case AluOp.Reciprocal:
        if (src[0].min <= 0 && src[0].max >= 0) return [-Infinity, Infinity];
        ret = [1 / src[0].max, 1 / src[0].min];
        break;
      case AluOp.Cast: {
        // Casts change the dtype.
        const wasFloat = isFloatDtype(src[0].dtype);
        const bounded =
          Number.isFinite(src[0].min) && Number.isFinite(src[0].max);
        if (this.dtype === DType.Bool) {
          const canBeZero = src[0].min <= 0 && src[0].max >= 0;
          const mustBeZero = src[0].min === 0 && src[0].max === 0;
          ret = mustBeZero ? [0, 0] : canBeZero ? [0, 1] : [1, 1];
        } else if (this.dtype === DType.Int32) {
          const a = wasFloat
            ? clamp(src[0].min, -2147483648, 2147483647) | 0
            : src[0].min | 0;
          const b = wasFloat
            ? clamp(src[0].max, -2147483648, 2147483647) | 0
            : src[0].max | 0;
          ret = bounded && a <= b ? [a, b] : [-Infinity, Infinity];
        } else if (this.dtype === DType.Uint32) {
          const a = wasFloat
            ? clamp(src[0].min, 0, 4294967295) >>> 0
            : src[0].min >>> 0;
          const b = wasFloat
            ? clamp(src[0].max, 0, 4294967295) >>> 0
            : src[0].max >>> 0;
          ret = bounded && a <= b ? [a, b] : [0, Infinity];
        } else {
          ret = [src[0].min, src[0].max];
        }
        break;
      }

      case AluOp.Cmplt:
        ret = [0, 1];
        break;
      case AluOp.Cmpne:
        ret = [0, 1];
        break;
      case AluOp.Where:
        ret = [
          Math.min(src[1].min, src[2].min),
          Math.max(src[1].max, src[2].max),
        ];
        break;

      case AluOp.Const:
        ret = [this.arg, this.arg];
        break;
      case AluOp.Special:
        ret = [0, this.arg[1] - 1];
        break;

      default:
        ret = [-Infinity, Infinity];
    }
    if (isNaN(ret[0]) || isNaN(ret[1])) {
      ret = [-Infinity, Infinity];
    }
    if (this.dtype === DType.Bool) {
      ret[0] = clamp(ret[0], 0, 1);
      ret[1] = clamp(ret[1], 0, 1);
    }
    if (this.dtype === DType.Uint32) {
      ret[0] = Math.max(0, ret[0]);
    }
    this.#range = ret;
    return ret;
  }
  get min(): number {
    return this.#computeRange()[0];
  }
  get max(): number {
    return this.#computeRange()[1];
  }

  /** Largest known integer that divides self. */
  constFactor(): number {
    if (this.op === AluOp.Const) return Math.abs(this.arg);
    if (this.op === AluOp.Add)
      return gcd(this.src[0].constFactor(), this.src[1].constFactor());
    if (this.op === AluOp.Mul) {
      if (this.src[0].op === AluOp.Const) return Math.abs(this.src[0].arg);
      if (this.src[1].op === AluOp.Const) return Math.abs(this.src[1].arg);
    }
    return 1;
  }
  /**
   * Checks if divisible by an integer v and returns the quotient if it is, or
   * `null` if it's not divisible.
   */
  divides(v: number): AluExp | null {
    if (v === 1) return this;
    if (this.op === AluOp.Const && this.arg % v === 0)
      return AluExp.const(this.dtype, this.arg / v);
    if (this.op === AluOp.Add) {
      const a = this.src[0].divides(v);
      if (a !== null) {
        const b = this.src[1].divides(v);
        if (b !== null) return AluExp.add(a, b);
      }
    }
    if (this.op === AluOp.Mul) {
      const a = this.src[0].divides(v);
      if (a !== null) return AluExp.mul(a, this.src[1]);
      const b = this.src[1].divides(v);
      if (b !== null) return AluExp.mul(this.src[0], b);
    }
    return null;
  }

  #isConstInt(): boolean {
    return (
      this.op === AluOp.Const &&
      (this.dtype === DType.Int32 || this.dtype === DType.Uint32)
    );
  }

  /**
   * Get all expressions by deeply matching an operation.
   *
   * For example: `((2+(3*5))+4).splitOp(+) -> [2,(3*5),4]`.
   */
  *splitOp(sep: AluOp): IterableIterator<AluExp> {
    if (this.op === sep) {
      for (const src of this.src) {
        yield* src.splitOp(sep);
      }
    } else yield this;
  }

  /**
   * Simplify the expression by replacing any known patterns and deduping
   * identical subexpressions.
   */
  simplify(cache: Map<bigint, AluExp> = new Map()): AluExp {
    // Cache this to avoid recomputing if it's called twice.
    if (this.#simplified !== undefined) return this.#simplified;

    // Extra help: `cache` can be used across multiple calls.
    const hash = this.getHash();
    const prevCachedValue = cache.get(hash);
    if (prevCachedValue !== undefined) {
      return (this.#simplified = prevCachedValue);
    }
    const simplified = this.#simplifyInner(cache);
    const simplifiedHash = simplified.getHash();
    const prevSimplified = cache.get(simplifiedHash);
    if (prevSimplified !== undefined) {
      cache.set(hash, prevSimplified);
      this.#simplified = prevSimplified;
      return prevSimplified;
    } else {
      cache.set(hash, simplified);
      cache.set(simplifiedHash, simplified);
      this.#simplified = simplified;
      return simplified;
    }
  }

  #simplifyInner(cache: Map<bigint, AluExp>): AluExp {
    const src = this.src.map((x) => x.simplify(cache));
    const { op } = this;

    // Constant folding.
    if (src.every((x) => x.op === AluOp.Const) && !AluGroup.Variable.has(op)) {
      const newExp = new AluExp(op, this.dtype, src, this.arg);
      return AluExp.const(this.dtype, newExp.evaluate({}));
    }

    // Replacing empty ranges with constants, if non-constant.
    if (op !== AluOp.Const && this.min === this.max) {
      return AluExp.const(this.dtype, this.min);
    }

    // Folding with one item being a no-op constant.
    if (AluGroup.Binary.has(op)) {
      for (let i = 0; i < 2; i++) {
        if (src[i].op !== AluOp.Const) continue;
        const x = src[i].arg;
        if (op === AluOp.Add && x === 0) return src[1 - i];
        if (op === AluOp.Sub && i === 1 && x === 0) return src[1 - i];
        if (op === AluOp.Mul && x === 1) return src[1 - i];
        if (op === AluOp.Mul && x === 0) return AluExp.const(this.dtype, 0);
        if (op === AluOp.Idiv && i === 1 && x === 1) return src[1 - i];
      }
    }

    // x + (-1 * y) => x - y
    // x - (-1 * y) => x + y
    if ((op === AluOp.Add || op === AluOp.Sub) && src[1].op === AluOp.Mul) {
      const [a, b] = src[1].src;
      const opNeg = op === AluOp.Add ? AluOp.Sub : AluOp.Add;
      if (a.op === AluOp.Const && a.arg === -1) {
        return new AluExp(opNeg, this.dtype, [src[0], b]);
      } else if (b.op === AluOp.Const && b.arg === -1) {
        return new AluExp(opNeg, this.dtype, [src[0], a]);
      }
    }

    // Where(cond, 1, 0) => Cast(ty, cond)
    if (
      op === AluOp.Where &&
      src.slice(1).every((s, i) => s.op === AluOp.Const && s.arg === 1 - i)
    ) {
      return AluExp.cast(this.dtype, src[0]);
    }

    // No-op comparisons.
    if (op === AluOp.Cmplt) {
      if (src[0].min >= src[1].max) return AluExp.const(DType.Bool, false);
      if (src[0].max < src[1].min) return AluExp.const(DType.Bool, true);
    }
    if (op === AluOp.Cmpne) {
      if (src[0].max < src[1].min || src[0].min > src[1].max)
        return AluExp.const(DType.Bool, true);
    }

    // Select statement.
    if (op === AluOp.Where) {
      if (src[0].max === 0) return src[2];
      if (src[0].min === 1) return src[1];
    }

    // Shape tracking ops (can be made more general).
    // x % C => x
    if (
      op === AluOp.Mod &&
      src[1].op === AluOp.Const &&
      src[0].min >= 0 &&
      src[0].max < src[1].arg
    ) {
      return src[0];
    }
    // (x % A) % B => x % min(A, B), if A|B or B|A
    if (
      op === AluOp.Mod &&
      src[0].op === AluOp.Mod &&
      src[1].#isConstInt() &&
      src[0].src[1].#isConstInt()
    ) {
      const A: number = src[0].src[1].arg;
      const B: number = src[1].arg;
      if (A > 0 && B > 0 && (A % B === 0 || B % A === 0)) {
        return AluExp.mod(
          src[0].src[0],
          AluExp.const(this.dtype, Math.min(A, B)),
        ).simplify();
      }
    }
    // (...) * A + (...) % A
    if (
      op === AluOp.Add &&
      src[0].op === AluOp.Mul &&
      src[0].src[1].#isConstInt() &&
      src[1].op === AluOp.Mod &&
      src[1].src[1].#isConstInt() &&
      src[0].src[1].arg === src[1].src[1].arg
    ) {
      const [mul, mod] = src;
      const check = (exp: AluExp) => {
        return (
          exp.op === AluOp.Idiv &&
          exp.src[1].#isConstInt() &&
          exp.src[1].arg === mod.src[1].arg &&
          exp.src[0] === mod.src[0]
        );
      };
      // (x/A) * A + x % A => x
      if (check(mul.src[0])) return mod.src[0];
      // (x/A % B) * A + x % A => x % (A*B)
      if (mul.src[0].op === AluOp.Mod) {
        const [x, y] = mul.src[0].src;
        if (check(x)) {
          return AluExp.mod(mod.src[0], AluExp.mul(mod.src[1], y)).simplify(
            cache,
          );
        }
      }
    }
    if (op === AluOp.Idiv && src[1].#isConstInt()) {
      const [numer, denom] = src;
      const B: number = denom.arg;
      for (let i = 0; i < 2; i++) {
        // (x * A) / B => x * (A / B)
        if (numer.op === AluOp.Mul && numer.src[i].#isConstInt()) {
          const A: number = numer.src[i].arg;
          if (A % B === 0) {
            let ret = numer.src[1 - i]; // x
            if (A / B !== 1)
              ret = AluExp.mul(ret, AluExp.const(ret.dtype, A / B));
            return ret.simplify(cache);
          }
        }
        // (x * A + C) / B => x * (A / B) + trunc(C / B)
        for (let j = 0; j < 2; j++) {
          if (
            numer.op === AluOp.Add &&
            numer.src[j].op === AluOp.Mul &&
            numer.src[j].src[i].#isConstInt()
          ) {
            const A: number = numer.src[j].src[i].arg;
            if (A % B === 0) {
              let ret = numer.src[j].src[1 - i]; // x
              if (A / B !== 1)
                ret = AluExp.mul(ret, AluExp.const(ret.dtype, A / B));
              ret = AluExp.add(
                ret,
                AluExp.idiv(numer.src[1 - j], AluExp.const(ret.dtype, B)),
              );
              return ret.simplify(cache);
            }
          }
        }
      }
    }
    if (
      op === AluOp.Mod &&
      src[1].#isConstInt() &&
      src[1].arg > 0 &&
      src[0].min >= 0
    ) {
      const [numer, denom] = src;
      const B: number = denom.arg;
      for (let i = 0; i < 2; i++) {
        if (numer.op === AluOp.Add) {
          // (x + A) % B => x % B; when x>=0, A%B === 0
          if (numer.src[i].#isConstInt()) {
            const A: number = numer.src[i].arg;
            const x = numer.src[1 - i]; // x
            if (A % B === 0 && x.min >= 0) {
              return AluExp.mod(x, denom).simplify(cache);
            }
          }
          for (let j = 0; j < 2; j++) {
            // (x + A * y) * B => x % B, when x>=0, A%B === 0
            if (
              numer.src[i].op === AluOp.Mul &&
              numer.src[i].src[j].#isConstInt()
            ) {
              const A: number = numer.src[i].src[j].arg;
              const x = numer.src[1 - i]; // x
              if (A % B === 0 && x.min >= 0) {
                return AluExp.mod(x, denom).simplify(cache);
              }
            }
          }
        } else if (numer.op === AluOp.Mul) {
          if (numer.src[i].#isConstInt()) {
            const A: number = numer.src[i].arg;
            // (x * A) % B => 0, when A%B === 0
            if (A % B === 0) {
              return AluExp.const(this.dtype, 0);
            }
            // (x * A) % B => x % B, when A%B === 1
            if (A % B === 1) {
              return AluExp.mod(numer.src[1 - i], denom).simplify(cache);
            }
          }
        }
      }
    }

    // Heuristic: Bias toward moving consts outward and to the right.
    // - A + x -> x + A
    // - x + (y + A) -> (x + y) + A
    // - (x + A) + y -> (x + y) + A
    // - (x + A) + B -> x + (A + B)
    const commOps = [AluOp.Add, AluOp.Mul, AluOp.Max, AluOp.Min];
    if (commOps.includes(op)) {
      const p = (a: AluExp, b: AluExp) => new AluExp(op, this.dtype, [a, b]);
      if (src[0].op === AluOp.Const) {
        // A + x -> x + A
        return p(src[1], src[0]).simplify(cache);
      }
      if (src[0].op === op && src[0].src[1].op === AluOp.Const) {
        if (src[1].op === AluOp.Const) {
          // (x + A) + B -> x + (A + B)
          return p(src[0].src[0], p(src[0].src[1], src[1])).simplify(cache);
        } else {
          // (x + A) + y -> (x + y) + A
          return p(p(src[0].src[0], src[1]), src[0].src[1]).simplify(cache);
        }
      }
      if (src[1].op === op && src[1].src[1].op === AluOp.Const) {
        // x + (y + A) -> (x + y) + A
        return p(p(src[0], src[1].src[0]), src[1].src[1]).simplify(cache);
      }
    }

    // Deep rules that match iteratively, based on tinygrad's symbolic.py.
    //
    // These are needed to simplify expressions like pool() in conv2d. Otherwise
    // the resulting expressions are complex and slow.

    if (op === AluOp.Mod || (op === AluOp.Idiv && src[1].#isConstInt())) {
      const [x, y] = src;

      // divide_by_gcd: https://github.com/tinygrad/tinygrad/blob/d1224a7/tinygrad/uop/symbolic.py#L190
      {
        const factors: number[] = [];
        const terms: AluExp[] = [];
        for (const u of x.splitOp(AluOp.Add)) {
          const factor = u.constFactor();
          factors.push(factor);
          terms.push(u.divides(factor)!);
        }
        const g = gcd(y.arg, ...factors);
        if (g !== 1) {
          let ret = new AluExp(op, this.dtype, [
            factors
              .map((f, i) =>
                AluExp.mul(AluExp.const(terms[i].dtype, f / g), terms[i]),
              )
              .reduceRight((a, x) => AluExp.add(x, a)),
            AluExp.const(y.dtype, y.arg / g),
          ]);
          if (op === AluOp.Mod)
            ret = AluExp.mul(ret, AluExp.const(this.dtype, g));
          return ret.simplify(cache);
        }
      }

      // simplify_remainder: https://github.com/tinygrad/tinygrad/blob/d1224a7/tinygrad/uop/symbolic.py#L208
      if (y.arg > 0) {
        let [xNoConst, constVal] = [x, 0];
        if (x.op === AluOp.Add && x.src[1].op === AluOp.Const) {
          [xNoConst, constVal] = [x.src[0], x.src[1].arg];
        }

        const terms: AluExp[] = [];
        const factors: number[] = [];
        for (const u of xNoConst.splitOp(AluOp.Add)) {
          const f = u.constFactor();
          const divided = u.divides(f);
          terms.push(divided ?? u);
          factors.push(divided ? f : 1);
        }

        const quotients = factors.map((f) => Math.floor(f / y.arg));
        const remainders = factors.map((f) => f % y.arg);
        const gcdVal = remainders.reduce((g, r) => gcd(g, r), y.arg);

        if (
          constVal % y.arg !== constVal ||
          gcdVal !== 1 ||
          remainders.some(
            (r, i) => r === 0 || (r !== factors[i] && op === AluOp.Mod),
          )
        ) {
          let quo = AluExp.const(x.dtype, Math.floor(constVal / y.arg));
          let rem = AluExp.const(
            x.dtype,
            Math.floor((constVal % y.arg) / gcdVal),
          );

          for (let i = 0; i < terms.length; i++) {
            if (op === AluOp.Idiv && remainders[i] !== 0) {
              rem = AluExp.add(
                rem,
                AluExp.mul(
                  AluExp.const(x.dtype, Math.floor(factors[i] / gcdVal)),
                  terms[i],
                ),
              );
            } else {
              rem = AluExp.add(
                rem,
                AluExp.mul(
                  AluExp.const(x.dtype, Math.floor(remainders[i] / gcdVal)),
                  terms[i],
                ),
              );
              quo = AluExp.add(
                quo,
                AluExp.mul(AluExp.const(x.dtype, quotients[i]), terms[i]),
              );
            }
          }

          if (
            !((x.min < 0 || rem.min < 0) && remainders.some((r) => r !== 0))
          ) {
            if (op === AluOp.Mod) {
              return AluExp.add(
                AluExp.mul(
                  AluExp.const(x.dtype, gcdVal),
                  AluExp.mod(
                    rem,
                    AluExp.const(x.dtype, Math.floor(y.arg / gcdVal)),
                  ),
                ),
                AluExp.const(x.dtype, constVal % gcdVal),
              ).simplify(cache);
            } else {
              return AluExp.add(
                AluExp.idiv(
                  rem,
                  AluExp.const(x.dtype, Math.floor(y.arg / gcdVal)),
                ),
                quo,
              ).simplify(cache);
            }
          }
        }
      }
    }

    // If any src was simplified, should construct a new expression.
    const newExp = src.every((s, i) => s === this.src[i])
      ? this
      : new AluExp(op, this.dtype, src, this.arg);

    return newExp;
  }

  /** Resolve this to a value, or `undefined` if not possible. */
  resolve(): any | undefined {
    const x = this.simplify();
    if (x.op === AluOp.Const) return x.arg;
    return undefined;
  }

  /**
   * Evaluate the expression on CPU, returning the result.
   *
   * Typically you would compile the AluExp as a representation to a lower-level
   * language. This is just to define the semantics and help debug.
   *
   * Note that the representation of Bool is as a number (0 or 1) here.
   */
  evaluate(
    context: Record<string, any>,
    globals?: (gid: number, bufidx: number) => any,
  ): number {
    if (AluGroup.Binary.has(this.op) || AluGroup.Compare.has(this.op)) {
      const x = this.src[0].evaluate(context, globals);
      const y = this.src[1].evaluate(context, globals);
      switch (this.op) {
        case AluOp.Add:
          return this.dtype === DType.Bool ? Number(x || y) : x + y;
        case AluOp.Sub:
          return x - y;
        case AluOp.Mul:
          return this.dtype === DType.Bool ? Number(x && y) : x * y;
        case AluOp.Idiv:
          return Math.trunc(x / y); // Consistent with signed Mod.
        case AluOp.Mod:
          return x % y;
        case AluOp.Min:
          return Math.min(x, y);
        case AluOp.Max:
          return Math.max(x, y);
        case AluOp.Cmplt:
          return Number(x < y);
        case AluOp.Cmpne:
          return Number(x != y);
        default:
          throw new Error(`Missing implemementation for ${this.op}`);
      }
    }

    if (AluGroup.Unary.has(this.op)) {
      const x = this.src[0].evaluate(context, globals);
      switch (this.op) {
        case AluOp.Sin:
          return Math.sin(x);
        case AluOp.Cos:
          return Math.cos(x);
        case AluOp.Asin:
          return Math.asin(x);
        case AluOp.Atan:
          return Math.atan(x);
        case AluOp.Exp:
          return Math.exp(x);
        case AluOp.Log:
          return Math.log(x);
        case AluOp.Sqrt:
          return Math.sqrt(x);
        case AluOp.Reciprocal:
          return 1 / x;
        case AluOp.Cast: {
          const wasFloat = isFloatDtype(this.src[0].dtype);
          if (this.dtype === DType.Int32)
            return (wasFloat ? clamp(x, -2147483648, 2147483647) : x) | 0;
          else if (this.dtype === DType.Uint32)
            return (wasFloat ? clamp(x, 0, 4294967295) : x) >>> 0;
          else if (isFloatDtype(this.dtype)) return x;
          else if (this.dtype === DType.Bool) return Number(Boolean(x));
          else throw new Error(`Unsupported cast to ${this.dtype}`);
        }
        case AluOp.Bitcast: {
          const buf = new ArrayBuffer(byteWidth(this.dtype));
          const view = new DataView(buf);
          // Populate data in the byte view (all browsers use little-endian).
          const fromType = this.src[0].dtype;
          if (fromType === DType.Float32) view.setFloat32(0, x, true);
          else if (fromType === DType.Int32) view.setInt32(0, x, true);
          else if (fromType === DType.Uint32) view.setUint32(0, x, true);
          else if (fromType === DType.Float16) view.setFloat16(0, x, true);
          else throw new Error(`Unsupported bitcast from ${fromType}`);
          // Read the data in the target dtype.
          if (this.dtype === DType.Float32) return view.getFloat32(0, true);
          else if (this.dtype === DType.Int32) return view.getInt32(0, true);
          else if (this.dtype === DType.Uint32) return view.getUint32(0, true);
          else if (this.dtype === DType.Float16)
            return view.getFloat16(0, true);
          else throw new Error(`Unsupported bitcast to ${this.dtype}`);
        }
        default:
          throw new Error(`Missing implemementation for ${this.op}`);
      }
    }

    switch (this.op) {
      case AluOp.Where:
        return this.src[0].evaluate(context, globals)
          ? this.src[1].evaluate(context, globals)
          : this.src[2].evaluate(context, globals);
      case AluOp.Threefry2x32: {
        const [k0, k1, c0, c1] = this.src.map((x) =>
          x.evaluate(context, globals),
        );
        const [x0, x1] = threefry2x32(k0, k1, c0, c1);
        if (this.arg === "xor") return (x0 ^ x1) >>> 0;
        else if (this.arg === 0) return x0;
        else if (this.arg === 1) return x1;
        else throw new Error(`Invalid Threefry2x32 mode: ${this.arg}`);
      }
      case AluOp.Const:
        return this.arg;
      case AluOp.Special: {
        const x = context[this.arg[0]];
        if (x === undefined) throw new Error(`Missing special: ${this.arg[0]}`);
        return x;
      }
      case AluOp.Variable: {
        const x = context[this.arg];
        if (x === undefined) throw new Error(`Missing variable: ${this.arg}`);
        return x;
      }
      case AluOp.GlobalIndex: {
        if (!globals) throw new Error("Missing globals function");
        const gid: number = this.arg[0];
        const bufidx = this.src[0].evaluate(context, globals);
        return globals(gid, bufidx);
      }
      case AluOp.GlobalView: {
        // Note: This branch is very slow. It should be lowered before evaluation.
        if (!globals) throw new Error("Missing globals function");
        const gid: number = this.arg[0];
        const st: ShapeTracker = this.arg[1];
        const [iexpr, vexpr] = st.toAluExp(this.src);
        if (vexpr.evaluate(context, globals)) {
          const bufidx = iexpr.evaluate(context, globals);
          return globals(gid, bufidx);
        } else {
          return 0;
        }
      }
      default:
        throw new Error(`Missing implemementation for ${this.op}`);
    }
  }

  /** Get this expression in debug format as a string. */
  toString(): string {
    const BIN_SYM: Partial<Record<AluOp, string>> = {
      [AluOp.Add]: "+",
      [AluOp.Sub]: "-",
      [AluOp.Mul]: "*",
      [AluOp.Idiv]: "/",
      [AluOp.Mod]: "%",
    };
    const CMP_SYM: Partial<Record<AluOp, string>> = {
      [AluOp.Cmplt]: "<",
      [AluOp.Cmpne]: "!=",
    };
    const UNARY_SYM: Partial<Record<AluOp, string>> = {
      [AluOp.Reciprocal]: "1/",
    };

    return this.fold<string>((node, parts) => {
      switch (node.op) {
        case AluOp.Const:
          return (
            "" + (node.dtype === DType.Bool ? Boolean(node.arg) : node.arg)
          );

        case AluOp.Variable:
          return `$${node.arg}:${node.dtype}`;

        case AluOp.Special: {
          const [name, n] = node.arg as [string, number];
          return `#${name}{${n}}`;
        }

        case AluOp.GlobalIndex:
          return `G_${node.arg[0]}<${node.dtype}>[${strip1(parts[0])}]`;

        case AluOp.GlobalView: {
          const [gid, st] = node.arg as [number, ShapeTracker];
          const shape = st.shape.join(",");
          const lastStrides = st.lastStrides.join(",");
          const cont = st.contiguous ? "c" : "nc";
          return `GV_${gid}<${node.dtype}>{${shape}:${lastStrides}:${cont}}[${parts.map(strip1).join(", ")}]`;
        }
      }

      /* binary ops with pretty symbols ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­ */
      if (BIN_SYM[node.op]) {
        return `(${parts[0]} ${BIN_SYM[node.op]!} ${parts[1]})`;
      }
      if (CMP_SYM[node.op]) {
        return `(${parts[0]} ${CMP_SYM[node.op]!} ${parts[1]})`;
      }

      /* unary ops with pretty names ­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­­ */
      if (UNARY_SYM[node.op]) {
        return `${UNARY_SYM[node.op]!}${parts[0]}`;
      }
      if (node.op === AluOp.Cast) {
        return `Cast<${node.dtype}>(${strip1(parts[0])})`;
      }
      if (node.op === AluOp.Bitcast) {
        return `Bitcast<${node.dtype}>(${strip1(parts[0])})`;
      }

      return `${node.op}(${parts.map(strip1).join(", ")})`;
    });
  }

  /** Generic fold() operation with a reducer over the expression tree. */
  fold<T = void>(reducer: (exp: AluExp, mappedSrc: T[]) => T): T {
    const visited = new Map<AluExp, T>();
    const recurse = (exp: AluExp): T => {
      if (visited.has(exp)) return visited.get(exp)!;
      const mappedSrc = exp.src.map((s) => recurse(s));
      const result = reducer(exp, mappedSrc);
      visited.set(exp, result);
      return result;
    };
    return recurse(this);
  }

  /** Check if any expression in the tree satisfies a predicate. */
  some(predicate: (exp: AluExp) => boolean): boolean {
    const visited = new Set<AluExp>();
    const recurse = (exp: AluExp): boolean => {
      if (visited.has(exp)) return false;
      if (predicate(exp)) return true;
      visited.add(exp);
      return exp.src.some(recurse);
    };
    return recurse(this);
  }

  /** Rewrite the expression recursively using a visitor. */
  rewrite(visitor: (exp: AluExp) => AluExp | undefined | null): AluExp {
    return this.fold<AluExp>((exp, newSrc) => {
      if (
        newSrc.length === exp.src.length &&
        newSrc.every((s, i) => s === exp.src[i])
      ) {
        return visitor(exp) ?? exp;
      } else {
        // If the source changed, we need to create a new expression.
        const newExp = new AluExp(exp.op, exp.dtype, newSrc, exp.arg);
        return visitor(newExp) ?? newExp;
      }
    });
  }

  /** Collect all nodes that satisfy a predicate. */
  collect(predicate: (exp: AluExp) => boolean): AluExp[] {
    const result: AluExp[] = [];
    this.fold((exp) => {
      if (predicate(exp)) result.push(exp);
    });
    return result;
  }

  /** Produce a list of all distinct AluOp in this expression. */
  distinctOps(): Set<AluOp> {
    const ops = new Set<AluOp>();
    this.fold((exp) => {
      ops.add(exp.op);
    });
    return ops;
  }

  /** Rewrite GlobalView operations to GlobalIndex operations. */
  rewriteGlobalViews(): AluExp {
    return this.rewrite((exp) => {
      if (exp.op === AluOp.GlobalView) {
        const [gid, st] = exp.arg as [number, ShapeTracker];
        return accessorGlobal(exp.dtype, gid, st, exp.src);
      }
    });
  }
}

/** Symbolic form for each mathematical operation. */
export enum AluOp {
  Add = "Add",
  Sub = "Sub",
  Mul = "Mul",
  Idiv = "Idiv",
  Mod = "Mod",
  Min = "Min",
  Max = "Max",

  Sin = "Sin",
  Cos = "Cos",
  Asin = "Asin",
  Atan = "Atan",
  Exp = "Exp",
  Log = "Log",
  Sqrt = "Sqrt",
  Reciprocal = "Reciprocal",
  Cast = "Cast",
  Bitcast = "Bitcast",

  Cmplt = "Cmplt",
  Cmpne = "Cmpne",
  Where = "Where", // Ternary operator: `cond ? a : b`

  Threefry2x32 = "Threefry2x32", // PRNG operation, arg = 'xor' | 0 | 1

  // Const is a literal constant, while GlobalIndex takes data from an array
  // buffer. Special and Variable are distinguished since the former is for
  // indices like the global invocation, while the latter is a value.
  Const = "Const", // arg = value
  Special = "Special", // arg = [variable, n]
  Variable = "Variable", // arg = variable
  GlobalIndex = "GlobalIndex", // arg = [gid, len]; src = [bufidx]
  GlobalView = "GlobalView", // arg = [gid, ShapeTracker], src = [indices...]
}

export const AluGroup = {
  Binary: new Set([
    AluOp.Add,
    AluOp.Sub,
    AluOp.Mul,
    AluOp.Idiv,
    AluOp.Mod,
    AluOp.Min,
    AluOp.Max,
  ]),
  Unary: new Set([
    AluOp.Sin,
    AluOp.Cos,
    AluOp.Asin,
    AluOp.Atan,
    AluOp.Exp,
    AluOp.Log,
    AluOp.Sqrt,
    AluOp.Reciprocal,
    AluOp.Cast,
    AluOp.Bitcast,
  ]),
  Compare: new Set([AluOp.Cmplt, AluOp.Cmpne]),
  Variable: new Set([
    AluOp.Special,
    AluOp.Variable,
    AluOp.GlobalIndex,
    AluOp.GlobalView,
  ]),
  Reduce: new Set([AluOp.Add, AluOp.Mul, AluOp.Min, AluOp.Max]),
  RequiredFloat: new Set([
    AluOp.Sin,
    AluOp.Cos,
    AluOp.Asin,
    AluOp.Atan,
    AluOp.Exp,
    AluOp.Log,
    AluOp.Sqrt,
    AluOp.Reciprocal,
  ]),
};

/** Common variables that can be substituted in expressions. */
export const AluVar = {
  gidx: AluExp.variable(DType.Int32, "gidx"), // global index
  ridx: AluExp.variable(DType.Int32, "ridx"), // reduction index
  acc: (dtype: DType) => AluExp.variable(dtype, "acc"), // accumulator
  idx: AluExp.variable(DType.Int32, "idx"), // virtual "array index"

  unroll: AluExp.variable(DType.Int32, "unroll"), // unroll index, inside loop
  upcast: AluExp.variable(DType.Int32, "upcast"), // upcast index, inside loop
};

/**
 * Description of a kernel to be compiled.
 *
 * Each of these can be processed by a backend into some lower-level
 * representation. It consists of one or more fused operations, optionally
 * indexing into a buffer.
 */
export class Kernel implements FpHashable {
  constructor(
    /** Number of global arguments / arrays. */
    readonly nargs: number,
    /** Size of the result array in element count. */
    readonly size: number,
    /** Expression to be evaluated. */
    readonly exp: AluExp,
    /** Optional reduction to be performed. */
    readonly reduction?: Reduction,
  ) {
    this.exp = exp.simplify();
  }

  hash(state: FpHash): void {
    state
      .update(this.nargs)
      .update(this.size)
      .update(this.exp)
      .update(this.reduction);
  }

  pprint(): PPrint {
    let details = PPrint.pp(`exp = ${this.exp}`);
    details = details.concat(PPrint.pp(`size = ${this.size}`));
    if (this.reduction) {
      details = details.concat(PPrint.pp(`reduction = ${this.reduction}`));
    }
    return PPrint.pp("{ ").stack(details).stack(PPrint.pp(" }"));
  }

  toString(): string {
    return this.pprint().toString();
  }

  /** The dtype of the values output by this kernel. */
  get dtype(): DType {
    if (this.reduction) {
      return this.reduction.epilogue.dtype;
    } else {
      return this.exp.dtype;
    }
  }

  /** The number of bytes in the output array when evaluating this kernel. */
  get bytes(): number {
    return this.size * byteWidth(this.dtype);
  }
}

/**
 * Description of a reduction.
 *
 * The strategy of jax-js backends is to either handle a standard operation that
 * is dispatched in a vectorized way over an array, or to reduce over one axis
 * of some computation. This is a description of the reduction.
 *
 * Reduction only supports a few operations, and only over one axis. Users can
 * always `flatten()` the array before reducing if needed.
 *
 * The backend is responsible for implementing the reduction in a way that
 * minimizes the number of global memory loads, for efficiency. This involves
 * passing through some optimization strategy. But optimizations are not coded
 * at this level since they depend on GPU, versus CPU or Wasm.
 */
export class Reduction implements FpHashable {
  constructor(
    /** Data type of the values being reduced over. */
    readonly dtype: DType,
    /** Operation to perform. Only ops in `AluGroup.Reduce` are supported. */
    readonly op: AluOp,
    /** Size of the reduction axis. */
    readonly size: number,
    /** Follow-up expression defined with the "acc" variable, defaults to identity. */
    readonly epilogue: AluExp = AluVar.acc(dtype), // TODO: Currently not used except in tests.
  ) {
    if (!AluGroup.Reduce.has(op)) {
      throw new TypeError(`Unsupported reduction: ${op}`);
    }
    this.epilogue = epilogue.simplify();
  }

  hash(state: FpHash): void {
    state
      .update(this.dtype)
      .update(this.op)
      .update(this.size)
      .update(this.epilogue);
  }

  toString(): string {
    return `${this.op}{${this.size}} -> ${this.epilogue}`;
  }

  /** Get the identity for this reduction operation. */
  get identity(): any {
    if (this.dtype === DType.Bool) {
      return this.op === AluOp.Add || this.op === AluOp.Max ? 0 : 1;
    } else if (this.dtype === DType.Int32) {
      if (this.op === AluOp.Add) return 0;
      else if (this.op === AluOp.Mul) return 1;
      else if (this.op === AluOp.Min) return -1 >>> 1;
      else if (this.op === AluOp.Max) return 1 << 31;
    } else if (this.dtype === DType.Uint32) {
      if (this.op === AluOp.Add) return 0;
      else if (this.op === AluOp.Mul) return 1;
      else if (this.op === AluOp.Min) return -1 >>> 0;
      else if (this.op === AluOp.Max) return 0;
    } else if (isFloatDtype(this.dtype)) {
      if (this.op === AluOp.Add) return 0;
      else if (this.op === AluOp.Mul) return 1;
      else if (this.op === AluOp.Min) return Infinity;
      else if (this.op === AluOp.Max) return -Infinity;
    }
    throw new TypeError(`Unsupported reduction: ${this.op} ${this.dtype}`);
  }

  /** Evaluate this operation on CPU. */
  evaluate(...values: any) {
    if (this.dtype === DType.Bool) {
      if (this.op === AluOp.Add || this.op === AluOp.Max) {
        return values.reduce((a: boolean, b: boolean) => a || b, true);
      } else if (this.op === AluOp.Mul || this.op === AluOp.Min) {
        return values.reduce((a: boolean, b: boolean) => a && b, true);
      }
    } else if (this.dtype === DType.Int32) {
      if (this.op === AluOp.Add) {
        return values.reduce((a: number, b: number) => (a + b) | 0, 0);
      } else if (this.op === AluOp.Mul) {
        return values.reduce((a: number, b: number) => (a * b) | 0, 1);
      } else if (this.op === AluOp.Min) {
        return values.reduce(
          (a: number, b: number) => Math.min(a, b),
          -1 >>> 1,
        );
      } else if (this.op === AluOp.Max) {
        return values.reduce((a: number, b: number) => Math.max(a, b), 1 << 31);
      }
    } else if (this.dtype === DType.Uint32) {
      if (this.op === AluOp.Add) {
        return values.reduce((a: number, b: number) => (a + b) >>> 0, 0);
      } else if (this.op === AluOp.Mul) {
        return values.reduce((a: number, b: number) => (a * b) >>> 0, 1);
      } else if (this.op === AluOp.Min) {
        return values.reduce(
          (a: number, b: number) => Math.min(a, b),
          -1 >>> 0,
        );
      } else if (this.op === AluOp.Max) {
        return values.reduce((a: number, b: number) => Math.max(a, b), 0);
      }
    } else if (isFloatDtype(this.dtype)) {
      if (this.op === AluOp.Add) {
        return values.reduce((a: number, b: number) => a + b, 0);
      } else if (this.op === AluOp.Mul) {
        return values.reduce((a: number, b: number) => a * b, 1);
      } else if (this.op === AluOp.Min) {
        return values.reduce(
          (a: number, b: number) => Math.min(a, b),
          Infinity,
        );
      } else if (this.op === AluOp.Max) {
        return values.reduce(
          (a: number, b: number) => Math.max(a, b),
          -Infinity,
        );
      }
    }
    throw new TypeError(`Unsupported reduction: ${this.op} ${this.dtype}`);
  }
}

/** Expression for accessing `indices` in input array with the given shape. */
export function accessorGlobal(
  dtype: DType,
  gid: number,
  st: ShapeTracker,
  indices: AluExp[],
): AluExp {
  const [index, valid] = st.toAluExp(indices);
  const [, len] = st.views[0].dataRange();
  return AluExp.where(
    valid,
    AluExp.globalIndex(dtype, gid, len, index),
    AluExp.const(dtype, 0),
  );
}

/** Expression for accessing `indices` in an array recipe with variable "idx". */
export function accessorAluExp(
  exp: AluExp,
  st: ShapeTracker,
  indices: AluExp[],
): AluExp {
  const [index, valid] = st.toAluExp(indices);
  return AluExp.where(
    valid,
    exp.substitute({ idx: index }),
    AluExp.const(exp.dtype, 0),
  );
}

// Threefry 2x32, 20 rounds (NR = 20)
// Reference: https://github.com/jax-ml/jax/blob/jax-v0.6.2/jax/_src/prng.py#L869
function threefry2x32(k0: number, k1: number, c0: number, c1: number) {
  const rotl32 = (x: number, r: number) => ((x << r) | (x >>> (32 - r))) >>> 0;

  const ks0 = k0 >>> 0;
  const ks1 = k1 >>> 0;
  const ks2 = (ks0 ^ ks1 ^ 0x1bd11bda) >>> 0;

  let x0 = (c0 + ks0) >>> 0;
  let x1 = (c1 + ks1) >>> 0;

  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 13) ^ x0));
  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 15) ^ x0));
  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 26) ^ x0));
  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 6) ^ x0));
  x0 = (x0 + ks1) >>> 0;
  x1 = (x1 + ks2 + 1) >>> 0;

  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 17) ^ x0));
  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 29) ^ x0));
  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 16) ^ x0));
  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 24) ^ x0));
  x0 = (x0 + ks2) >>> 0;
  x1 = (x1 + ks0 + 2) >>> 0;

  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 13) ^ x0));
  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 15) ^ x0));
  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 26) ^ x0));
  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 6) ^ x0));
  x0 = (x0 + ks0) >>> 0;
  x1 = (x1 + ks1 + 3) >>> 0;

  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 17) ^ x0));
  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 29) ^ x0));
  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 16) ^ x0));
  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 24) ^ x0));
  x0 = (x0 + ks1) >>> 0;
  x1 = (x1 + ks2 + 4) >>> 0;

  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 13) ^ x0));
  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 15) ^ x0));
  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 26) ^ x0));
  ((x0 = (x0 + x1) >>> 0), (x1 = rotl32(x1, 6) ^ x0));
  x0 = (x0 + ks2) >>> 0;
  x1 = (x1 + ks0 + 5) >>> 0;

  return [x0, x1];
}
