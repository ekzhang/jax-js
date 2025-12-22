import { byteWidth, DType } from "../alu";
import { PPrint } from "../pprint";
import { type Pair } from "../shape";
import {
  JsTreeDef,
  MapJsTree,
  flatten as treeFlatten,
  unflatten as treeUnflatten,
} from "../tree";
import {
  deepEqual,
  FpHash,
  FpHashable,
  generalBroadcast,
  runWithCache,
  unzip2,
  zip,
} from "../utils";
import { array, Array, ArrayLike, pureArray } from "./array";
import { checkConvShape, checkPoolShape } from "./convolution";
import {
  AbstractValue,
  bind,
  flattenFun,
  fullRaise,
  getAval,
  ndim,
  newDynamic,
  newMain,
  Primitive,
  PrimitiveParams,
  promoteAvals,
  ShapedArray,
  Trace,
  Tracer,
  TracerValue,
} from "./core";

/**
 * Function callback with an associated dispose() method.
 *
 * The dispose() method should be called to clean up any tracer resources needed
 * by the function after the last time it is called.
 */
// eslint-disable-next-line @typescript-eslint/no-unsafe-function-type
export type OwnedFunction<F extends Function> = F & { dispose: () => void };

/** Variable in a Jaxpr expression. */
export class Var {
  static #nextId = 1; // For debugging, since JavaScript has no id() function like Python.

  readonly id: number;
  readonly aval: ShapedArray;

  constructor(aval: ShapedArray) {
    this.id = Var.#nextId++;
    this.aval = aval;
  }

  toString(): string {
    return `Var(${this.id}):${this.aval.toString()}`;
  }
}

/** Literal in a Jaxpr expression. Currently, only scalars are supported. */
export class Lit {
  readonly value: number;
  readonly aval: ShapedArray;

  get dtype(): DType {
    return this.aval.dtype;
  }

  constructor(aval: AbstractValue, value: number) {
    if (aval.shape.length !== 0) {
      throw new Error(`internal: Lit must be a scalar`);
    }
    this.value = value;
    this.aval = ShapedArray.fromAval(aval);
  }
}

export type Atom = Var | Lit;

function atomIsLit(
  atom: Atom,
  literal?: number | boolean,
): atom is Lit & boolean {
  return (
    atom instanceof Lit && (literal === undefined || atom.value === literal)
  );
}

class VarPrinter {
  names: Map<Var, string> = new Map();
  #next = "a";

  // a, b, c, ..., z, aa, ab, ..., az, ba, bb, ...
  #advance() {
    const ret = this.#next;
    let lastNonz = this.#next.length - 1;
    while (lastNonz >= 0 && this.#next[lastNonz] === "z") {
      lastNonz--;
    }
    if (lastNonz < 0) {
      this.#next = "a".repeat(this.#next.length + 1);
    } else {
      let result = this.#next.slice(0, lastNonz);
      result += String.fromCharCode(this.#next.charCodeAt(lastNonz) + 1);
      result += "a".repeat(this.#next.length - 1 - lastNonz);
      this.#next = result;
    }
    return ret;
  }

  name(v: Var): string {
    if (this.names.has(v)) {
      return this.names.get(v)!;
    }
    const name = this.#advance();
    this.names.set(v, name);
    return name;
  }

  nameType(v: Var): string {
    return `${this.name(v)}:${v.aval.toString()}`;
  }
}

/** A single statement / binding in a Jaxpr, in ANF form. */
export class JaxprEqn {
  constructor(
    readonly primitive: Primitive,
    readonly inputs: Atom[],
    readonly params: Record<string, any>,
    readonly outBinders: Var[],
  ) {}

  pprint(usedVars?: Set<Var>, vp = new VarPrinter()): PPrint {
    const lhs = PPrint.pp(
      this.outBinders
        .map((v) => (!usedVars || usedVars.has(v) ? vp.nameType(v) : "_"))
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
          .map((x) => (x instanceof Var ? vp.name(x) : String(x.value)))
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
export class Jaxpr implements FpHashable {
  #hash?: bigint;

  constructor(
    readonly inBinders: Var[],
    readonly eqns: JaxprEqn[],
    readonly outs: Atom[],
  ) {}

  pprint(): PPrint {
    const vp = new VarPrinter();
    const usedVars = new Set<Var>(
      [...this.outs, ...this.eqns.flatMap((eqn) => eqn.inputs)].filter(
        (x) => x instanceof Var,
      ),
    );
    const inBinders = this.inBinders.map((v) => vp.nameType(v)).join(", ");
    const eqns = PPrint.prototype.concat(
      ...this.eqns.map((e) => e.pprint(usedVars, vp)),
    );
    const outs = this.outs
      .map((x) => (x instanceof Var ? vp.name(x) : x.value))
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
   * Gets a hash of this Jaxpr.
   *
   * Var identity is not considered in the hash, so two Jaxprs with the same
   * order of assignments and operators but different variable IDs will resolve
   * to the same hash (and toString representation).
   */
  getHash(): bigint {
    if (this.#hash !== undefined) return this.#hash;
    const hasher = new FpHash();
    const varIds = new Map<Var, bigint>();
    const vi = (v: Var) => {
      if (varIds.has(v)) return varIds.get(v)!;
      const id = varIds.size + 1; // Start from 1, why not?
      varIds.set(v, FpHash.hash(id, v.aval.dtype, ...v.aval.shape));
      return id;
    };
    hasher.update(this.inBinders.length);
    for (const x of this.inBinders) hasher.update(vi(x));
    hasher.update(this.eqns.length);
    for (const eqn of this.eqns) {
      hasher.update(eqn.primitive);
      hasher.update(eqn.inputs.length);
      for (const x of eqn.inputs)
        hasher.update(x instanceof Var ? vi(x) : x.value);
      hasher.update(JSON.stringify(eqn.params));
      hasher.update(eqn.outBinders.length);
      for (const x of eqn.outBinders) hasher.update(vi(x));
    }
    hasher.update(this.outs.length);
    for (const x of this.outs)
      hasher.update(x instanceof Var ? vi(x) : x.value);
    return (this.#hash = hasher.value);
  }

  hash(state: FpHash): void {
    state.update(this.getHash());
  }

  /**
   * Produce a simplified Jaxpr with basic optimizations applied.
   *  - Trim away unused variables.
   *  - Fold away *1, *0, or +0 operations against literals.
   *  - Remove no-op movement operations.
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
          context.set(
            c,
            new Lit(
              promoteAvals(a.aval, b.aval),
              a.dtype === DType.Bool
                ? Math.min(a.value + b.value, 1) // Special case: Bool ||
                : a.value + b.value,
            ),
          );
        } else {
          newEqns.push(eqn);
        }
      } else if (eqn.primitive === Primitive.Neg) {
        const [a] = inputs;
        const c = eqn.outBinders[0];
        if (atomIsLit(a)) {
          context.set(c, new Lit(a.aval, -a.value));
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
          context.set(
            c,
            new Lit(promoteAvals(a.aval, b.aval), a.value * b.value),
          );
        } else {
          newEqns.push(eqn);
        }
      } else if (eqn.primitive === Primitive.Idiv) {
        const [a, b] = inputs;
        const c = eqn.outBinders[0];
        if (atomIsLit(b, 1)) {
          context.set(c, a);
        } else {
          newEqns.push(eqn);
        }
      } else if (
        ((eqn.primitive === Primitive.Broadcast ||
          eqn.primitive === Primitive.Reshape) &&
          deepEqual(eqn.params.shape, eqn.inputs[0].aval.shape)) ||
        (eqn.primitive === Primitive.Transpose &&
          (eqn.params.perm as number[]).every((p, i) => p === i)) ||
        (eqn.primitive === Primitive.Flip && eqn.params.axis.length === 0) ||
        (eqn.primitive === Primitive.Shrink &&
          (eqn.params.slice as Pair[]).every(
            ([s, e], i) => s === 0 && e === eqn.inputs[0].aval.shape[i],
          )) ||
        (eqn.primitive === Primitive.Pad &&
          (eqn.params.width as Pair[]).every(
            ([w0, w1]) => w0 === 0 && w1 === 0,
          ))
      ) {
        // No-op movement operation, just pass through the input.
        context.set(eqn.outBinders[0], eqn.inputs[0]);
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

  /** Flattens nested JitCall in a Jaxpr. Useful for handling jit-of-jit. */
  flatten(): Jaxpr {
    if (!this.eqns.some((eqn) => eqn.primitive === Primitive.JitCall)) {
      // Fast path: no JitCall to flatten.
      return this;
    }
    // Otherwise, we need to flatten this Jaxpr.
    const newEqns: JaxprEqn[] = [];
    const varMap = new Map<Var, Atom>(); // outBinders from JitCall are replaced with new values
    const varMapF = (x: Atom) => (x instanceof Var ? (varMap.get(x) ?? x) : x);
    for (const eqn of this.eqns) {
      if (eqn.primitive === Primitive.JitCall) {
        // First, flatten the Jaxpr recursively.
        const jaxpr = (eqn.params.jaxpr as Jaxpr).flatten();
        // Make a mapping of this Jaxpr's variables to translated values.
        const translation = new Map<Var, Atom>();
        const translationF = (x: Atom) =>
          x instanceof Var ? translation.get(x)! : x;
        for (const [v, x] of zip(jaxpr.inBinders, eqn.inputs)) {
          translation.set(v, varMapF(x));
        }
        for (const ieqn of jaxpr.eqns) {
          const inputs = ieqn.inputs.map(translationF);
          const outBinders: Var[] = [];
          for (const v of ieqn.outBinders) {
            const u = new Var(v.aval);
            outBinders.push(u);
            translation.set(v, u);
          }
          newEqns.push(
            new JaxprEqn(ieqn.primitive, inputs, ieqn.params, outBinders),
          );
        }
        // Add the outputs to the mapping.
        for (const [v, x] of zip(eqn.outBinders, jaxpr.outs)) {
          varMap.set(v, translationF(x));
        }
      } else {
        if (eqn.inputs.some((x) => x instanceof Var && varMap.has(x))) {
          // Replace any input variables if needed.
          newEqns.push(
            new JaxprEqn(
              eqn.primitive,
              eqn.inputs.map(varMapF),
              eqn.params,
              eqn.outBinders,
            ),
          );
        } else {
          newEqns.push(eqn);
        }
      }
    }
    // Replace the output variables if needed.
    const newOuts = this.outs.map(varMapF);
    return new Jaxpr(this.inBinders, newEqns, newOuts);
  }
}

export class JaxprType {
  constructor(
    readonly inTypes: ShapedArray[],
    readonly outTypes: ShapedArray[],
  ) {}

  toString(): string {
    const inTypes = this.inTypes.map((aval) => aval.toString()).join(", ");
    const outTypes = this.outTypes.map((aval) => aval.toString()).join(", ");
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
    const rule = abstractEvalRules[eqn.primitive];
    const outTypes = rule(inTypes, eqn.params as any);
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

  // Number of usages of each variable, in an eqn or the output.
  // Needed for reference tracking / move semantics.
  const usageCount = new Map<Var, number>();
  for (const x of jaxpr.eqns.flatMap((eqn) => eqn.inputs).concat(jaxpr.outs)) {
    if (x instanceof Var) usageCount.set(x, (usageCount.get(x) ?? 0) + 1);
  }

  const remainingRefs = new Map<Var, number>();

  const read = (x: Atom) => {
    if (x instanceof Var) {
      remainingRefs.set(x, (remainingRefs.get(x) ?? 0) - 1);
      return env.get(x)!;
    } else {
      return array(x.value, { dtype: x.dtype });
    }
  };

  const write = (v: Var, val: Tracer) => {
    if (env.has(v)) throw new Error(`Variable already bound: ${v}`);
    let refCount = usageCount.get(v) ?? 0;
    if (refCount) {
      env.set(v, val);
      remainingRefs.set(v, refCount);
      while (refCount-- > 1) val.ref;
    } else {
      val.dispose(); // If variable not used, dispose immediately.
    }
  };

  try {
    for (const [v, arg] of zip(jaxpr.inBinders, args)) write(v, arg);
    for (const eqn of jaxpr.eqns) {
      const inVals = eqn.inputs.map(read);
      const outVals = bind(eqn.primitive, inVals, eqn.params);
      for (const [v, val] of zip(eqn.outBinders, outVals)) write(v, val);
    }
    return jaxpr.outs.map(read);
  } catch (error) {
    // Clean up any remaining references on error, to avoid leaking memory.
    for (let [v, refCount] of remainingRefs.entries()) {
      if (refCount > 0) {
        const tracer = env.get(v)!;
        while (refCount--) tracer.dispose();
      }
    }
    throw error;
  }
}

/** Convert a Jaxpr to a callable function by evaluating it. */
export function jaxprAsFun(jaxpr: Jaxpr): (...args: Tracer[]) => Tracer[] {
  return (...args: Tracer[]) => evalJaxpr(jaxpr, args);
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
    return `JaxprTracer(${this.aval.toString()})`;
  }

  // JaxprTracer does not hold any resources, no need to be reference counted.
  get ref() {
    return this;
  }
  dispose() {}
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
      this.builder.addConst(
        tracer,
        val instanceof Tracer ? val.ref : array(val),
      );
    }
    return tracer;
  }
  pure = this.getOrMakeConstTracer;
  lift = this.getOrMakeConstTracer;

  processPrimitive<P extends Primitive>(
    primitive: P,
    tracers: JaxprTracer[],
    params: PrimitiveParams<P>,
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
      const ar = consts[i] as Array;
      literals.set(jaxpr.inBinders[i], new Lit(ar.aval, ar.dataSync()[0]));
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

type AbstractEvalRule<P extends Primitive> = (
  avals: ShapedArray[],
  params: PrimitiveParams<P>,
) => ShapedArray[];

function binopAbstractEval([x, y]: ShapedArray[]) {
  if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray)) {
    throw new TypeError("binopAbstractEval expects ShapedArray inputs");
  }
  return [promoteAvals(x, y)];
}

function compareAbstractEval([x, y]: ShapedArray[]) {
  if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray)) {
    throw new TypeError("compareAbstractEval expects ShapedArray inputs");
  }
  const aval = promoteAvals(x, y); // Make sure they can be typecast for comparison.
  return [new ShapedArray(aval.shape, DType.Bool, false)];
}

function vectorizedUnopAbstractEval([x]: ShapedArray[]) {
  return [ShapedArray.fromAval(x)];
}

export const abstractEvalRules: { [P in Primitive]: AbstractEvalRule<P> } = {
  [Primitive.Add]: binopAbstractEval,
  [Primitive.Mul]: binopAbstractEval,
  [Primitive.Idiv]: binopAbstractEval,
  [Primitive.Mod]: binopAbstractEval,
  [Primitive.Neg]: vectorizedUnopAbstractEval,
  [Primitive.Reciprocal]: vectorizedUnopAbstractEval,
  [Primitive.Floor]: vectorizedUnopAbstractEval,
  [Primitive.Ceil]: vectorizedUnopAbstractEval,
  [Primitive.StopGradient]: vectorizedUnopAbstractEval,
  [Primitive.Cast]([x]: ShapedArray[], { dtype }) {
    return [new ShapedArray(x.shape, dtype, false)];
  },
  [Primitive.Bitcast]([x]: ShapedArray[], { dtype }) {
    if (x.dtype === DType.Bool || dtype === DType.Bool) {
      throw new TypeError("Bitcast to/from bool is not allowed");
    }
    if (byteWidth(x.dtype) !== byteWidth(dtype)) {
      throw new TypeError(
        `Bitcast from ${x.dtype} to ${dtype} with different byte width`,
      );
    }
    return [new ShapedArray(x.shape, dtype, false)];
  },
  [Primitive.RandomBits]([k0, k1]: ShapedArray[], { shape }) {
    if (k0.dtype !== DType.Uint32 || k1.dtype !== DType.Uint32) {
      throw new TypeError(
        `RandomBits requires uint32 keys, got ${k0.dtype} and ${k1.dtype}`,
      );
    }
    const keyShape = generalBroadcast(k0.shape, k1.shape);
    if (!deepEqual(generalBroadcast(keyShape, shape), shape)) {
      throw new TypeError(
        `Keys of shapes ${k0.shape} and ${k1.shape} cannot be broadcast to shape ${shape}`,
      );
    }
    return [new ShapedArray(shape, DType.Uint32, false)];
  },
  [Primitive.Sin]: vectorizedUnopAbstractEval,
  [Primitive.Cos]: vectorizedUnopAbstractEval,
  [Primitive.Asin]: vectorizedUnopAbstractEval,
  [Primitive.Atan]: vectorizedUnopAbstractEval,
  [Primitive.Exp]: vectorizedUnopAbstractEval,
  [Primitive.Log]: vectorizedUnopAbstractEval,
  [Primitive.Erf]: vectorizedUnopAbstractEval,
  [Primitive.Erfc]: vectorizedUnopAbstractEval,
  [Primitive.Sqrt]: vectorizedUnopAbstractEval,
  [Primitive.Min]: binopAbstractEval,
  [Primitive.Max]: binopAbstractEval,
  [Primitive.Reduce]([x], { axis }) {
    const axisSet = new Set(axis);
    const newShape = x.shape.filter((_, i) => !axisSet.has(i));
    return [new ShapedArray(newShape, x.dtype, x.weakType)];
  },
  [Primitive.Pool]([x], { window, strides }) {
    const shape = checkPoolShape(x.shape, window, strides);
    return [new ShapedArray(shape, x.dtype, x.weakType)];
  },
  [Primitive.PoolTranspose]([x], { inShape, window, strides }) {
    const shape = checkPoolShape(inShape, window, strides);
    if (!deepEqual(shape, x.shape)) {
      throw new TypeError(
        `PoolTranspose shape mismatch: expected ${JSON.stringify(shape)}, got ${JSON.stringify(x.shape)}`,
      );
    }
    return [new ShapedArray(inShape, x.dtype, x.weakType)];
  },
  [Primitive.Dot]([x, y]) {
    if (x.ndim === 0 && y.ndim === 0)
      throw new TypeError("Dot requires at least 1D inputs");
    const { shape, dtype, weakType } = promoteAvals(x, y);
    shape.splice(-1, 1); // Remove the contracted dimension.
    return [new ShapedArray(shape, dtype, weakType)];
  },
  [Primitive.Conv]([lhs, rhs], params) {
    const { dtype, weakType } = promoteAvals(
      new ShapedArray([], lhs.dtype, lhs.weakType),
      new ShapedArray([], rhs.dtype, rhs.weakType),
    );
    const shape = checkConvShape(lhs.shape, rhs.shape, params);
    return [new ShapedArray(shape, dtype, weakType)];
  },
  [Primitive.Compare]: compareAbstractEval,
  [Primitive.Where]([cond, x, y]) {
    if (cond.dtype !== DType.Bool)
      throw new TypeError(`Condition must be boolean, got ${cond.dtype}`);
    const xy = promoteAvals(x, y);
    const shape = generalBroadcast(cond.shape, xy.shape);
    return [new ShapedArray(shape, xy.dtype, xy.weakType)];
  },
  [Primitive.Transpose]([x], { perm }) {
    return [
      new ShapedArray(
        perm.map((i) => x.shape[i]),
        x.dtype,
        x.weakType,
      ),
    ];
  },
  [Primitive.Broadcast]([x], { shape }) {
    return [new ShapedArray(shape, x.dtype, x.weakType)];
  },
  [Primitive.Reshape]([x], { shape }) {
    return [new ShapedArray(shape, x.dtype, x.weakType)];
  },
  [Primitive.Flip]([x], _) {
    return [ShapedArray.fromAval(x)];
  },
  [Primitive.Shrink]([x], { slice }) {
    const newShape = slice.map((s) => s[1] - s[0]);
    return [new ShapedArray(newShape, x.dtype, x.weakType)];
  },
  [Primitive.Pad]([x], { width }) {
    const newShape = x.shape.map((dim, i) => dim + width[i][0] + width[i][1]);
    return [new ShapedArray(newShape, x.dtype, x.weakType)];
  },
  [Primitive.Gather]([x, ...indices], { axis, outDim }) {
    for (const a of indices)
      if (a.dtype !== DType.Int32 && a.dtype !== DType.Uint32)
        throw new TypeError(
          `Gather indices must be Int32 or Uint32, got ${a.dtype}`,
        );
    if (axis.length !== indices.length)
      throw new TypeError(`Gather: ${axis} axes but ${indices.length} indices`);
    if (indices.length === 0)
      throw new TypeError("Gather must have 1+ indices with same shape");
    if (axis.some((a) => a < 0 || a >= x.shape.length))
      throw new TypeError("Gather axis out of bounds");
    if (outDim < 0 || outDim > x.shape.length - axis.length)
      throw new TypeError("Gather outDim out of bounds");
    const axisSet = new Set(axis);
    if (axisSet.size !== axis.length)
      throw new TypeError("Gather axes are not unique");
    const gatherShape = indices.reduce<number[]>(
      (shape, a) => generalBroadcast(shape, a.shape),
      [],
    );
    const newShape = x.shape.filter((_, i) => !axisSet.has(i));
    newShape.splice(outDim, 0, ...gatherShape);
    return [new ShapedArray(newShape, x.dtype, x.weakType)];
  },
  [Primitive.JitCall](args, { jaxpr }) {
    const { inTypes, outTypes } = typecheckJaxpr(jaxpr);
    if (args.length !== inTypes.length) {
      throw new TypeError(
        `jit_call expected ${inTypes.length} arguments, got ${args.length}`,
      );
    }
    for (let i = 0; i < inTypes.length; i++) {
      if (!args[i].equals(inTypes[i])) {
        throw new TypeError(
          `jit_call argument ${i} has type ${args[i]}, expected ${inTypes[i]}`,
        );
      }
    }
    return outTypes;
  },
};

function splitIdx(values: any[], argnums: Set<number>): [any[], any[]] {
  const a: any[] = [];
  const b: any[] = [];
  for (let i = 0; i < values.length; i++) {
    if (argnums.has(i)) a.push(values[i]);
    else b.push(values[i]);
  }
  return [a, b];
}

function joinIdx(n: number, a: any[], b: any[], argnums: Set<number>): any[] {
  const result: any[] = [];
  let ai = 0;
  let bi = 0;
  for (let i = 0; i < n; i++) {
    if (argnums.has(i)) result.push(a[ai++]);
    else result.push(b[bi++]);
  }
  return result;
}

/** @inline */
export type JitOpts = {
  staticArgnums?: number[];
};

export function makeJaxpr(
  f: (...args: any[]) => any,
  opts?: JitOpts,
): (...argsIn: any) => { jaxpr: Jaxpr; consts: Tracer[]; treedef: JsTreeDef } {
  return (...argsIn) => {
    const staticArgnums = new Set(opts?.staticArgnums ?? []);
    const [staticArgs, shapedArgs] = splitIdx(argsIn, staticArgnums);

    const [avalsIn, inTree] = treeFlatten(shapedArgs);
    const [fFlat, outTree] = flattenFun((...dynamicArgs: any[]) => {
      return f(
        ...joinIdx(argsIn.length, staticArgs, dynamicArgs, staticArgnums),
      );
    }, inTree);

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

export function jit<F extends (...args: any[]) => any>(
  f: F,
  opts?: JitOpts,
): OwnedFunction<
  (...args: MapJsTree<Parameters<F>, Array, ArrayLike>) => ReturnType<F>
> {
  const cache = new Map<string, ReturnType<ReturnType<typeof makeJaxpr>>>();
  const staticArgnums = new Set(opts?.staticArgnums ?? []);

  const result = ((...args) => {
    const [staticArgs, dynamicArgs] = splitIdx(args, staticArgnums);

    const [argsFlat, inTree] = treeFlatten(dynamicArgs);
    const avalsInFlat = argsFlat.map((x) => ShapedArray.fromAval(getAval(x)));
    const avalsIn = treeUnflatten(inTree, avalsInFlat) as any[];

    const jaxprArgs = joinIdx(args.length, staticArgs, avalsIn, staticArgnums);
    const {
      jaxpr,
      consts,
      treedef: outTree,
    } = runWithCache(cache, jaxprArgs, () => makeJaxpr(f, opts)(...jaxprArgs));

    const outs = bind(
      Primitive.JitCall,
      [...consts.map((c) => c.ref), ...argsFlat],
      {
        name: f.name || "closure",
        jaxpr,
        numConsts: consts.length,
      },
    );
    return treeUnflatten(outTree, outs);
  }) as OwnedFunction<F>;

  result.dispose = () => {
    for (const { consts } of cache.values()) {
      for (const c of consts) {
        c.dispose();
      }
    }
  };

  return result;
}
