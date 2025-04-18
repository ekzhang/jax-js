import { DType } from "../alu";
import { PPrint } from "../pprint";
import { flatten as treeFlatten, JsTreeDef } from "../tree";
import { range, unzip2, zip } from "../utils";
import { Array, generalBroadcast, pureArray } from "./array";
import {
  bind,
  flattenFun,
  fullRaise,
  getAval,
  ndim,
  newDynamic,
  newMain,
  Primitive,
  ShapedArray,
  Trace,
  Tracer,
  TracerValue,
} from "./core";

/** Variable in a Jaxpr expression. */
export class Var {
  static nextId = 1; // For debugging, since JavaScript has no id() function like Python.

  static resetIdCounter() {
    Var.nextId = 1;
  }

  readonly id: number;
  readonly aval: ShapedArray;

  constructor(aval: ShapedArray) {
    this.id = Var.nextId++;
    this.aval = aval;
  }

  get name() {
    return `v_${this.id}`;
  }

  toString(): string {
    return `${this.name}:${this.aval.strShort()}`;
  }
}

/** Literal in a Jaxpr expression. */
export class Lit {
  readonly val: Array;
  readonly aval: ShapedArray;

  constructor(val: Array | number | boolean) {
    this.aval = ShapedArray.fromAval(getAval(val));
    const ar = pureArray(val);
    if (ndim(ar) !== 0 || !(ar instanceof Array)) {
      throw new TypeError("Lit only supports scalar Array values");
    }
    this.val = ar;
  }

  get value(): number | boolean {
    return this.val.dataSync()[0];
  }
}

export type Atom = Var | Lit;

function atomIsLit(atom: Atom, literal?: number | boolean) {
  return (
    atom instanceof Lit && (literal === undefined || atom.value === literal)
  );
}

/** A single statement / binding in a Jaxpr, in ANF form. */
export class JaxprEqn {
  constructor(
    readonly primitive: Primitive,
    readonly inputs: Atom[],
    readonly params: Record<string, any>,
    readonly outBinders: Var[],
  ) {}

  pprint(usedVars?: Set<Var>): PPrint {
    const lhs = PPrint.pp(
      this.outBinders
        .map((v) => (!usedVars || usedVars.has(v) ? v : "_"))
        .join(" "),
    );
    let rhs = PPrint.pp(this.primitive);
    // pprint params
    const paramsList = Object.entries(this.params).map(([k, v]) =>
      PPrint.pp(`${k}=${v}`),
    );
    if (paramsList.length > 0) {
      rhs = rhs
        .stack(PPrint.pp(" [ "))
        .stack(PPrint.prototype.concat(...paramsList))
        .stack(PPrint.pp(" ] "));
    } else {
      rhs = rhs.stack(PPrint.pp(" "));
    }
    // pprint inputs (vars and literals)
    rhs = rhs.stack(
      PPrint.pp(
        this.inputs
          .map((x) => (x instanceof Var ? x.name : JSON.stringify(x.val.js())))
          .join(" "),
      ),
    );
    return lhs.stack(PPrint.pp(" = ")).stack(rhs);
  }

  toString(): string {
    return this.pprint().toString();
  }
}

/** Typed intermediate representation for traced computations. */
export class Jaxpr {
  constructor(
    readonly inBinders: Var[],
    readonly eqns: JaxprEqn[],
    readonly outs: Atom[],
  ) {}

  pprint(): PPrint {
    const usedVars = new Set<Var>(
      [...this.outs, ...this.eqns.flatMap((eqn) => eqn.inputs)].filter(
        (x) => x instanceof Var,
      ),
    );
    const inBinders = this.inBinders.map((v) => v.toString()).join(", ");
    const eqns = PPrint.prototype.concat(
      ...this.eqns.map((e) => e.pprint(usedVars)),
    );
    const outs = this.outs
      .map((x) => (x instanceof Var ? x.name : x.val.js()))
      .join(", ");
    return PPrint.pp(`{ lambda ${inBinders} .`).concat(
      (this.eqns.length
        ? PPrint.pp("let ")
            .stack(eqns)
            .concat(PPrint.pp(`in ( ${outs} ) }`))
        : PPrint.pp(`( ${outs} ) }`)
      ).indent(2),
    );
  }

  toString(): string {
    return this.pprint().toString();
  }

  /**
   * Produce a simplified Jaxpr with basic optimizations applied.
   *  - Trim away unused variables.
   *  - Fold away *1, *0, or +0 operations against literals.
   */
  simplify(): Jaxpr {
    const context = new Map<Var, Atom>();
    const newEqns: JaxprEqn[] = [];
    for (const e of this.eqns) {
      const inputs = e.inputs.map((x) =>
        x instanceof Var ? (context.get(x) ?? x) : x,
      );
      const eqn = new JaxprEqn(e.primitive, inputs, e.params, e.outBinders);

      if (eqn.primitive === Primitive.Add) {
        const [a, b] = inputs;
        const c = eqn.outBinders[0];
        if (atomIsLit(a, 0)) {
          context.set(c, b);
        } else if (atomIsLit(b, 0)) {
          context.set(c, a);
        } else if (atomIsLit(a) && atomIsLit(b)) {
          context.set(c, new Lit((a as any).value + (b as any).value));
        } else {
          newEqns.push(eqn);
        }
      } else if (eqn.primitive === Primitive.Mul) {
        const [a, b] = inputs;
        const c = eqn.outBinders[0];
        // TODO: handle *0 once we have shaped zero arrays
        if (atomIsLit(a, 1)) {
          context.set(c, b);
        } else if (atomIsLit(b, 1)) {
          context.set(c, a);
        } else if (atomIsLit(a) && atomIsLit(b)) {
          context.set(c, new Lit((a as any).value * (b as any).value));
        } else {
          newEqns.push(eqn);
        }
      } else {
        newEqns.push(eqn);
      }
    }

    const outs = this.outs.map((x) =>
      x instanceof Var ? (context.get(x) ?? x) : x,
    );

    // Skip unused output variables
    const usedVars = new Set<Var>(outs.filter((x) => x instanceof Var));
    const liveEqns: JaxprEqn[] = [];
    for (let i = newEqns.length - 1; i >= 0; i--) {
      const eqn = newEqns[i];
      if (eqn.outBinders.some((v) => usedVars.has(v))) {
        liveEqns.push(eqn);
        for (const v of eqn.inputs) {
          if (v instanceof Var) {
            usedVars.add(v);
          }
        }
      }
    }

    return new Jaxpr(this.inBinders, liveEqns.reverse(), outs);
  }
}

export class JaxprType {
  constructor(
    readonly inTypes: ShapedArray[],
    readonly outTypes: ShapedArray[],
  ) {}

  toString(): string {
    const inTypes = this.inTypes.map((aval) => aval.strShort()).join(", ");
    const outTypes = this.outTypes.map((aval) => aval.strShort()).join(", ");
    return `(${inTypes}) -> (${outTypes})`;
  }
}

export function typecheckJaxpr(jaxpr: Jaxpr): JaxprType {
  const env = new Set<Var>();

  for (const v of jaxpr.inBinders) {
    if (env.has(v)) {
      throw new TypeError(`Duplicate variable binding: ${v}`);
    }
    env.add(v);
  }

  for (const eqn of jaxpr.eqns) {
    const inTypes = eqn.inputs.map((x) => typecheckAtom(env, x));
    const outTypes = abstractEvalRules[eqn.primitive](inTypes, eqn.params);
    for (const [outBinder, outType] of zip(eqn.outBinders, outTypes)) {
      if (!outType.equals(outBinder.aval)) {
        throw new TypeError(
          `Output binder type mismatch in ${eqn.primitive}: ${outBinder} vs ${outType}`,
        );
      }
      if (env.has(outBinder)) {
        throw new TypeError(`Duplicate variable binding: ${outBinder}`);
      }
      env.add(outBinder);
    }
  }

  const inTypes = jaxpr.inBinders.map((v) => v.aval);
  const outTypes = jaxpr.outs.map((x) => typecheckAtom(env, x));
  return new JaxprType(inTypes, outTypes);
}

function typecheckAtom(env: Set<Var>, x: Atom): ShapedArray {
  if (x instanceof Var) {
    if (!env.has(x)) {
      throw new Error(`Unknown variable: ${x}`);
    }
    return x.aval;
  } else if (x instanceof Lit) {
    return x.aval;
  } else {
    throw new TypeError(`Invalid atom type: ${x}`);
  }
}

/** Evaluate a Jaxpr on an array of inputs. */
export function evalJaxpr(jaxpr: Jaxpr, args: Tracer[]): Tracer[] {
  const env = new Map<Var, Tracer>();

  const read = (x: Atom) => (x instanceof Var ? env.get(x)! : x.val);
  const write = (v: Var, val: Tracer) => {
    if (env.has(v)) throw new Error(`Variable already bound: ${v}`);
    env.set(v, val);
  };

  for (const [v, arg] of zip(jaxpr.inBinders, args)) write(v, arg);
  for (const eqn of jaxpr.eqns) {
    const inVals = eqn.inputs.map(read);
    const outVals = bind(eqn.primitive, inVals, eqn.params);
    for (const [v, val] of zip(eqn.outBinders, outVals)) write(v, val);
  }
  return jaxpr.outs.map(read);
}

/** Tracer that records its operations to dynamically construct a Jaxpr. */
class JaxprTracer extends Tracer {
  constructor(
    trace: Trace,
    readonly aval: ShapedArray,
  ) {
    super(trace);
  }

  toString(): string {
    return `JaxprTracer(${this.aval.strShort()})`;
  }
}

/** Analogous to the 'DynamicJaxprTrace' class in JAX. */
class JaxprTrace extends Trace {
  /** Register a Jaxpr argument with a given shape and return the tracer. */
  newArg(aval: ShapedArray): JaxprTracer {
    aval = ShapedArray.fromAval(aval);
    const tracer = this.builder.newTracer(this, aval);
    this.builder.addVar(tracer);
    return tracer;
  }

  /** Register a constant / literal in this Jaxpr. */
  getOrMakeConstTracer(val: TracerValue): JaxprTracer {
    let tracer = this.builder.constTracers.get(val);
    if (tracer === undefined) {
      tracer = this.builder.newTracer(this, ShapedArray.fromAval(getAval(val)));
      this.builder.addConst(tracer, pureArray(val));
    }
    return tracer;
  }
  pure = this.getOrMakeConstTracer;
  lift = this.getOrMakeConstTracer;

  processPrimitive(
    primitive: Primitive,
    tracers: JaxprTracer[],
    params: Record<string, any>,
  ): JaxprTracer[] {
    const avalsIn = tracers.map((t) => t.aval);
    const avalsOut = abstractEvalRules[primitive](avalsIn, params);
    const outTracers = avalsOut.map((aval) =>
      this.builder.newTracer(this, aval),
    );
    this.builder.addEqn(
      new JaxprEqn(
        primitive,
        tracers.map((t) => this.builder.getVar(t)),
        params,
        outTracers.map((t) => this.builder.addVar(t)),
      ),
    );
    return outTracers;
  }

  get builder(): JaxprBuilder {
    return this.main.globalData;
  }
}

/** Incrementally constructs a Jaxpr. */
class JaxprBuilder {
  eqns: JaxprEqn[] = [];
  tracerToVar: Map<JaxprTracer, Var> = new Map();
  constTracers: Map<TracerValue, JaxprTracer> = new Map(); // already-seen value -> tracer
  constVals: Map<Var, Tracer> = new Map(); // var -> const value
  tracers: JaxprTracer[] = [];

  newTracer(trace: JaxprTrace, aval: ShapedArray): JaxprTracer {
    const tracer = new JaxprTracer(trace, aval);
    this.tracers.push(tracer);
    return tracer;
  }

  addEqn(eqn: JaxprEqn) {
    this.eqns.push(eqn);
  }

  addVar(tracer: JaxprTracer): Var {
    if (this.tracerToVar.has(tracer)) {
      throw new Error(`Tracer was added as variable twice: ${tracer}`);
    }
    const v = new Var(tracer.aval);
    this.tracerToVar.set(tracer, v);
    return v;
  }

  getVar(tracer: JaxprTracer): Var {
    const v = this.tracerToVar.get(tracer);
    if (v === undefined) {
      throw new Error(`Could not find variable for tracer: ${tracer}`);
    }
    return v;
  }

  addConst(tracer: JaxprTracer, val: Tracer) {
    const v = this.addVar(tracer);
    this.constTracers.set(val, tracer);
    this.constVals.set(v, val);
    return v;
  }

  build(
    inTracers: JaxprTracer[],
    outTracers: JaxprTracer[],
  ): { jaxpr: Jaxpr; consts: Tracer[] } {
    // Initially, concatenate the constants as the first few inputs.
    let [constVars, consts] = unzip2(this.constVals.entries());
    const t2v = this.getVar.bind(this); // Maps tracer to value.
    const inBinders = [...constVars, ...inTracers.map(t2v)];
    const outVars = outTracers.map(t2v);
    let jaxpr = new Jaxpr(inBinders, this.eqns, outVars);

    // Inline any scalar constants as Lit and remove from the input list.
    typecheckJaxpr(jaxpr);
    [jaxpr, consts] = _inlineLiterals(jaxpr, consts);
    return { jaxpr, consts };
  }
}

function _inlineLiterals(jaxpr: Jaxpr, consts: Tracer[]): [Jaxpr, Tracer[]] {
  const literals = new Map<Atom, Lit>();
  const constBinders: Var[] = [];
  const newConsts: Tracer[] = [];

  for (let i = 0; i < consts.length; i++) {
    if (ndim(consts[i]) === 0 && consts[i] instanceof Array) {
      literals.set(jaxpr.inBinders[i], new Lit(consts[i] as any));
    } else {
      constBinders.push(jaxpr.inBinders[i]);
      newConsts.push(consts[i]);
    }
  }

  const newEqns: JaxprEqn[] = jaxpr.eqns.map(
    (eqn) =>
      new JaxprEqn(
        eqn.primitive,
        eqn.inputs.map((x) => literals.get(x) ?? x),
        eqn.params,
        eqn.outBinders,
      ),
  );
  const newOuts = jaxpr.outs.map((x) => literals.get(x) ?? x);
  const newJaxpr = new Jaxpr(
    [...constBinders, ...jaxpr.inBinders.slice(consts.length)],
    newEqns,
    newOuts,
  );
  typecheckJaxpr(newJaxpr); // Double-check for sanity.
  return [newJaxpr, newConsts];
}

type AbstractEvalRule = (shapes: ShapedArray[], params: any) => ShapedArray[];

function binopAbstractEval([x, y]: ShapedArray[]) {
  if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray)) {
    throw new TypeError("binopAbstractEval expects ShapedArray inputs");
  }
  if (x.dtype !== y.dtype) {
    // TODO: Relax this restriction on dtype equality, or add automatic casts.
    throw new TypeError(`Mismatched dtypes: ${x.dtype} vs ${y.dtype}`);
  }
  return [new ShapedArray(generalBroadcast(x.shape, y.shape), x.dtype)];
}

function compareAbstractEval([x, y]: ShapedArray[]) {
  if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray)) {
    throw new TypeError("binopAbstractEval expects ShapedArray inputs");
  }
  if (x.dtype !== y.dtype) {
    // TODO: Relax this restriction on dtype equality, or add automatic casts.
    throw new TypeError(`Mismatched dtypes: ${x.dtype} vs ${y.dtype}`);
  }
  return [new ShapedArray(generalBroadcast(x.shape, y.shape), DType.Bool)];
}

function vectorizedUnopAbstractEval([x]: ShapedArray[]) {
  return [ShapedArray.fromAval(x)];
}

export const abstractEvalRules: Record<Primitive, AbstractEvalRule> = {
  [Primitive.Add]: binopAbstractEval,
  [Primitive.Mul]: binopAbstractEval,
  [Primitive.Neg]: vectorizedUnopAbstractEval,
  [Primitive.Sin]: vectorizedUnopAbstractEval,
  [Primitive.Cos]: vectorizedUnopAbstractEval,
  [Primitive.ReduceSum]([x], { axis }: { axis: number[] }) {
    const axisSet = new Set(axis);
    const newShape = x.shape.filter((_, i) => !axisSet.has(i));
    return [new ShapedArray(newShape, x.dtype)];
  },
  [Primitive.Greater]: compareAbstractEval,
  [Primitive.Less]: compareAbstractEval,
  [Primitive.Transpose]([x], { perm }: { perm?: number[] }) {
    if (perm === undefined) {
      perm = range(x.shape.length).reverse();
    }
    return [
      new ShapedArray(
        perm.map((i) => x.shape[i]),
        x.dtype,
      ),
    ];
  },
  [Primitive.Broadcast](
    [x],
    { shape, axes }: { shape: number[]; axes: number[] },
  ) {
    return [new ShapedArray(shape, x.dtype)];
  },
};

export function makeJaxpr(
  f: (...args: any[]) => any,
): (...argsIn: any) => { jaxpr: Jaxpr; consts: Tracer[]; treedef: JsTreeDef } {
  return (...argsIn) => {
    const [avalsIn, inTree] = treeFlatten(argsIn);
    const [fFlat, outTree] = flattenFun(f, inTree);

    Var.resetIdCounter(); // Reset the counter for each new Jaxpr trace.
    const builder = new JaxprBuilder();
    using main = newMain(JaxprTrace, builder);
    using _dynamic = newDynamic(main);

    const trace = new JaxprTrace(main);
    const tracersIn = avalsIn.map((aval) =>
      trace.newArg(typeof aval === "object" ? aval : pureArray(aval)),
    );
    const outs = fFlat(...tracersIn);
    const tracersOut = outs.map(
      (out: Tracer) => fullRaise(trace, out) as JaxprTracer,
    );
    const { jaxpr, consts } = builder.build(tracersIn, tracersOut);

    if (outTree.value === undefined) {
      throw new Error("outTree was not set in makeJaxpr");
    }
    return { jaxpr: jaxpr.simplify(), consts, treedef: outTree.value };
  };
}
