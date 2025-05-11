import { FpHash } from "./hash";
import { ShapeTracker } from "./shape";
import { clamp } from "./utils";

export enum DType {
  Float32 = "float32",
  Int32 = "int32",
  Bool = "bool",
  Complex64 = "complex64", // unimplemented
}

const isFloatDtype = (dtype: DType) =>
  dtype === DType.Float32 || dtype === DType.Complex64;

/**
 * Mathemtical expression on scalar values.
 *
 * This is similiar to and based on tinygrad's UOp class, but it's more specific
 * to just math on scalars. We're doing this to avoid the complexity of a full
 * graph rewrite engine.
 */
export class AluExp {
  #hash?: bigint;
  #simplified?: AluExp;
  #range?: [number, number];

  constructor(
    readonly op: AluOp,
    readonly dtype: DType,
    readonly src: AluExp[],
    readonly arg: any = undefined,
  ) {}

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
  static min(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Min, a.dtype, [a, b]);
  }
  static max(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Max, a.dtype, [a, b]);
  }
  static mod(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Mod, a.dtype, [a, b]);
  }
  static cast(dtype: DType, a: AluExp): AluExp {
    if (a.dtype === dtype) return a;
    return new AluExp(AluOp.Cast, dtype, [a]);
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
    return new AluExp(AluOp.Const, dtype, [], value);
  }
  static special(dtype: DType, name: string, n: number): AluExp {
    return new AluExp(AluOp.Special, dtype, [], [name, n]);
  }
  static variable(dtype: DType, name: string): AluExp {
    return new AluExp(AluOp.Variable, dtype, [], name);
  }
  static globalIndex(dtype: DType, gid: number, bufidx: AluExp): AluExp {
    return new AluExp(AluOp.GlobalIndex, dtype, [bufidx], gid);
  }
  static globalView(
    dtype: DType,
    gid: number,
    st: ShapeTracker,
    indices: AluExp[],
  ): AluExp {
    return new AluExp(AluOp.GlobalView, dtype, indices, [gid, st]);
  }

  static i32(value: number): AluExp {
    return AluExp.const(DType.Int32, value);
  }
  static f32(value: number): AluExp {
    return AluExp.const(DType.Float32, value);
  }
  static bool(value: boolean): AluExp {
    return AluExp.const(DType.Bool, value);
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
    hasher.update(this.op, this.dtype, JSON.stringify(this.arg));
    hasher.update(BigInt(this.src.length));
    for (const s of this.src) {
      hasher.update(s.getHash());
    }
    this.#hash = hasher.value;
    return this.#hash;
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
        ret = minMax4((a, b) => Math.floor(a / b));
        break;
      }
      case AluOp.Mod: {
        // Mod is a bit tricky since it can be negative. Two behaviors:
        //
        // - C-style: In JS and WGSL, mod depends on the sign of the dividend.
        // - Python-style: But in NumPy, JAX, and Python, it's based on the sign
        //   of the divisor (% or the np.remainder function).
        //
        // We're going to use the C-style behavior since it's more common in the
        // web world and outside of Python. This is a deviation from JAX!
        let divisorRange = src[1].#computeRange();
        if (divisorRange[0] <= 0 && divisorRange[1] >= 0) {
          divisorRange = [0, Math.max(-divisorRange[0], divisorRange[1])];
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
        ret = [Math.sin(src[0].min), Math.sin(src[0].max)];
        break;
      case AluOp.Cos:
        ret = [Math.cos(src[0].min), Math.cos(src[0].max)];
        break;
      case AluOp.Cast:
        // Casts change the dtype.
        if (this.dtype === DType.Bool) {
          const canBeZero = src[0].min <= 0 && src[0].max >= 0;
          const mustBeZero = src[0].min === 0 && src[0].max === 0;
          ret = mustBeZero ? [0, 0] : canBeZero ? [0, 1] : [1, 1];
        } else if (this.dtype === DType.Int32) {
          ret = [Math.trunc(src[0].min), Math.trunc(src[0].max)];
        } else {
          ret = [src[0].min, src[0].max];
        }
        break;

      case AluOp.Cmplt:
        ret = [0, 1];
        break;
      case AluOp.Cmpne:
        ret = [0, 1];
        break;
      case AluOp.Where:
        ret = [0, 1];
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
    this.#range = ret;
    return ret;
  }
  get min(): number {
    return this.#computeRange()[0];
  }
  get max(): number {
    return this.#computeRange()[1];
  }

  #isConstInt(): boolean {
    return this.op === AluOp.Const && this.dtype === DType.Int32;
  }

  /**
   * Simplify the expression by replacing any known patterns and deduping
   * identical subexpressions.
   */
  simplify(cache: Map<bigint, AluExp> = new Map()): AluExp {
    // Cache this to avoid recomputing if it's called twice.
    if (this.#simplified !== undefined) return this.#simplified;

    // Extra help: `cache` can be used cache across multiple calls.
    const hash = this.getHash();
    if (cache.has(hash)) {
      return (this.#simplified = cache.get(hash)!);
    }
    const simplified = this.#simplifyInner(cache);
    const simplifiedHash = simplified.getHash();
    if (cache.has(simplifiedHash)) {
      const prevSimplified = cache.get(simplifiedHash)!;
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
      const B = denom.arg;
      for (let i = 0; i < 2; i++) {
        // (x * A) / B => x * (A / B)
        if (numer.op === AluOp.Mul && numer.src[i].#isConstInt()) {
          const A = numer.src[i].arg;
          if (A % B === 0) {
            let ret = numer.src[1 - i]; // x
            if (A / B !== 1) ret = AluExp.mul(ret, AluExp.i32(A / B));
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
            const A = numer.src[j].src[i].arg;
            if (A % B === 0) {
              let ret = numer.src[j].src[1 - i]; // x
              if (A / B !== 1) ret = AluExp.mul(ret, AluExp.i32(A / B));
              ret = AluExp.add(ret, AluExp.idiv(numer.src[1 - j], B));
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
      const B = denom.arg;
      for (let i = 0; i < 2; i++) {
        // (x + A) % B => x % B + (A % B); when x+A>=0, B>0
        if (numer.op === AluOp.Add && numer.src[i].#isConstInt()) {
          const A = numer.src[i].arg;
          let ret = numer.src[1 - i]; // x
          if (A % B !== 0) ret = AluExp.add(ret, AluExp.i32(A % B));
          return ret.simplify(cache);
        }
      }
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
   */
  evaluate(
    context: Record<string, any>,
    globals?: (gid: number, bufidx: number) => any,
  ): any {
    if (AluGroup.Binary.has(this.op) || AluGroup.Compare.has(this.op)) {
      const x = this.src[0].evaluate(context, globals);
      const y = this.src[1].evaluate(context, globals);
      switch (this.op) {
        case AluOp.Add:
          return this.dtype === DType.Bool ? x || y : x + y;
        case AluOp.Sub:
          return x - y;
        case AluOp.Mul:
          return this.dtype === DType.Bool ? x && y : x * y;
        case AluOp.Idiv:
          return Math.trunc(x / y); // Consistent with signed Mod.
        case AluOp.Mod:
          return x % y;
        case AluOp.Min:
          return Math.min(x, y);
        case AluOp.Max:
          return Math.max(x, y);
        case AluOp.Cmplt:
          return x < y;
        case AluOp.Cmpne:
          return x != y;
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
        case AluOp.Cast:
          if (this.dtype === DType.Int32) return Math.floor(Number(x));
          else if (this.dtype === DType.Float32) return Number(x);
          else if (this.dtype === DType.Bool) return Boolean(x);
          else throw new Error(`Unsupported cast to ${this.dtype}`);
        default:
          throw new Error(`Missing implemementation for ${this.op}`);
      }
    }

    switch (this.op) {
      case AluOp.Where:
        return this.src[0].evaluate(context, globals)
          ? this.src[1].evaluate(context, globals)
          : this.src[2].evaluate(context, globals);
      case AluOp.Const:
        return this.arg;
      case AluOp.Special:
        return context[this.arg[0]];
      case AluOp.Variable:
        return context[this.arg];
      case AluOp.GlobalIndex: {
        if (!globals) throw new Error("Missing globals function");
        const gid: number = this.arg;
        const bufidx = this.src[0].evaluate(context, globals);
        return globals(gid, bufidx);
      }
      case AluOp.GlobalView: {
        // Note: This branch is very slow.
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
      [AluOp.Sin]: "sin",
      [AluOp.Cos]: "cos",
    };

    return this.fold<string>((node, parts) => {
      switch (node.op) {
        case AluOp.Const:
          return "" + node.arg;

        case AluOp.Variable:
          return `$${node.arg}:${node.dtype}`;

        case AluOp.Special: {
          const [name, n] = node.arg as [string, number];
          return `#${name}{${n}}`;
        }

        case AluOp.GlobalIndex:
          return `G_${node.arg}<${node.dtype}>[${parts[0]}]`;

        case AluOp.GlobalView: {
          const [gid, st] = node.arg as [number, ShapeTracker];
          const shape = st.shape.join(",");
          const cont = st.contiguous ? "c" : "nc";
          return `GV_${gid}<${node.dtype}>{${shape}${cont ? "" : "*"}}[${parts.join(", ")}]`;
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
        return `${UNARY_SYM[node.op]!}(${parts[0]})`;
      }
      if (node.op === AluOp.Cast) {
        return `Cast<${node.dtype}>(${parts[0]})`;
      }

      return `${node.op}(${parts.join(", ")})`;
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
  Cast = "Cast",

  Cmplt = "Cmplt",
  Cmpne = "Cmpne",
  Where = "Where",

  // Const is a literal constant, while GlobalIndex takes data from an array
  // buffer. Special and Variable are distinguished since the former is for
  // indices like the global invocation, while the latter is a value.
  Const = "Const", // arg = value
  Special = "Special", // arg = [variable, n]
  Variable = "Variable", // arg = variable
  GlobalIndex = "GlobalIndex", // arg = gid; src = [bufidx]
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
  Unary: new Set([AluOp.Sin, AluOp.Cos, AluOp.Cast]),
  Compare: new Set([AluOp.Cmplt, AluOp.Cmpne]),
  Variable: new Set([AluOp.Special, AluOp.Variable, AluOp.GlobalIndex]),
  Reduce: new Set([AluOp.Add, AluOp.Mul, AluOp.Min, AluOp.Max]),
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
export class Kernel {
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

  /** Unique string representation of this object, usable as a map key. */
  toString() {
    JSON.stringify(this, (_key, value) => {
      // Make sure that we sort the keys alphabetically for determinism.
      if (value instanceof Object && !(value instanceof Array)) {
        return Object.fromEntries(
          Object.entries(value).sort(([a], [b]) => a.localeCompare(b)),
        );
      }
      return value;
    });
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
export class Reduction {
  constructor(
    /** Data type of the values being reduced over. */
    readonly dtype: DType,
    /** Operation to perform. Only ops in `AluGroup.Reduce` are supported. */
    readonly op: AluOp,
    /** Size of the reduction axis. */
    readonly size: number,
    /** Follow-up expression defined with the "acc" variable, defaults to identity. */
    readonly fusion: AluExp = AluVar.acc(dtype),
  ) {
    if (!AluGroup.Reduce.has(op)) {
      throw new TypeError(`Unsupported reduction: ${op}`);
    }
  }

  /** Get the identity for this reduction operation. */
  get identity(): any {
    if (this.dtype === DType.Bool) {
      return this.op === AluOp.Add || this.op === AluOp.Max ? false : true;
    } else if (this.dtype === DType.Int32) {
      if (this.op === AluOp.Add) return 0;
      else if (this.op === AluOp.Mul) return 1;
      else if (this.op === AluOp.Min) return -1 >>> 1;
      else if (this.op === AluOp.Max) return 1 << 31;
    } else if (this.dtype === DType.Float32) {
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
    } else if (this.dtype === DType.Float32) {
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
  gid: number,
  st: ShapeTracker,
  indices: AluExp[],
): AluExp {
  const [index, valid] = st.toAluExp(indices);
  return AluExp.where(
    valid,
    AluExp.globalIndex(DType.Float32, gid, index),
    AluExp.f32(0),
  );
}

/** Expression for accessing `indices` in an array recipe with variable "idx". */
export function accessorAluExp(
  exp: AluExp,
  st: ShapeTracker,
  indices: AluExp[],
): AluExp {
  const [index, valid] = st.toAluExp(indices);
  return AluExp.where(valid, exp.substitute({ idx: index }), AluExp.f32(0));
}
