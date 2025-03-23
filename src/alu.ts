export enum DType {
  Float32 = "float32",
  Int32 = "int32",
  Bool = "bool",
  Complex64 = "complex64",
}

/**
 * Mathemtical expression on scalar values.
 *
 * This is similiar to and based on tinygrad's UOp class, but it's more specific
 * to just math on scalars. We're doing this to avoid the complexity of a full
 * graph rewrite engine.
 */
export class AluExp {
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
  static mod(a: AluExp, b: AluExp): AluExp {
    return new AluExp(AluOp.Mod, a.dtype, [a, b]);
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
  static globalIndex(dtype: DType, gid: number, bufidx: AluExp): AluExp {
    return new AluExp(AluOp.GlobalIndex, dtype, [bufidx], gid);
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

  /** Simplify the expression by replacing any known patterns. */
  simplify(): AluExp {
    const src = this.src.map((x) => x.simplify());
    const { op } = this;

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

    // Select statement.
    if (op === AluOp.Where) {
      if (src[0].op === AluOp.Const) return src[src[0].arg ? 1 : 2];
    }

    // If any src was simplified, should construct a new expression.
    const newExp = src.every((s, i) => s === this.src[i])
      ? this
      : new AluExp(op, this.dtype, src, this.arg);

    // Constant folding.
    if (src.every((x) => x.op === AluOp.Const) && !AluGroup.Variable.has(op)) {
      return AluExp.const(this.dtype, newExp.evaluate({}));
    }

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
          return Math.floor(x / y);
        case AluOp.Mod:
          return x % y;
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
      case AluOp.GlobalIndex:
        if (!globals) throw new Error("Missing globals function");
        const gid: number = this.arg;
        const bufidx = this.src[0].evaluate(context, globals);
        return globals(gid, bufidx);
      default:
        throw new Error(`Missing implemementation for ${this.op}`);
    }
  }
}

/** Symbolic form for each mathematical operation. */
export enum AluOp {
  Add,
  Sub,
  Mul,
  Idiv,
  Mod,
  Sin,
  Cos,
  Cmplt,
  Cmpne,
  Where,
  Const, // arg = value
  Special, // arg = [variable, n]
  GlobalIndex, // arg = gid; src = [bufidx]
}

export const AluGroup = {
  Binary: new Set([AluOp.Add, AluOp.Sub, AluOp.Mul, AluOp.Idiv, AluOp.Mod]),
  Unary: new Set([AluOp.Sin, AluOp.Cos]),
  Compare: new Set([AluOp.Cmplt, AluOp.Cmpne]),
  Variable: new Set([AluOp.Special, AluOp.GlobalIndex]),
};
