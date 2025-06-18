/** @file Implementations of vjp() and partial evaluation. */

import { DType } from "../alu";
import { flatten as treeFlatten, unflatten as treeUnflatten } from "../tree";
import { invertPermutation, partitionList, toposort, unzip2 } from "../utils";
import { pureArray, scalar, zeros } from "./array";
import {
  AbstractValue,
  add,
  bind,
  broadcast,
  flattenFun,
  flip,
  fullRaise,
  mul,
  ndim,
  neg,
  newMain,
  Primitive,
  reduceSum,
  reshape,
  ShapedArray,
  Trace,
  Tracer,
  TracerValue,
  transpose,
  TreeMismatchError,
  UseAfterFreeError,
  where,
} from "./core";
import {
  abstractEvalRules,
  evalJaxpr,
  Jaxpr,
  JaxprEqn,
  Lit,
  makeJaxpr,
  typecheckJaxpr,
  Var,
} from "./jaxpr";
import { jvp } from "./jvp";

/** Array value that can either be known or unknown. */
class PartialVal {
  constructor(
    readonly val: Tracer | null,
    readonly aval: ShapedArray,
  ) {}

  static known(val: Tracer): PartialVal {
    return new PartialVal(val, ShapedArray.fromAval(val.aval));
  }

  static unknown(aval: AbstractValue): PartialVal {
    return new PartialVal(null, ShapedArray.fromAval(aval));
  }

  get isKnown(): boolean {
    return this.val !== null;
  }

  toString(): string {
    return this.val ? this.val.toString() : this.aval.strShort();
  }
}

function partialEvalFlat(
  f: (...args: any[]) => any,
  pvalsIn: PartialVal[],
): { jaxpr: Jaxpr; pvalsOut: PartialVal[]; consts: Tracer[] } {
  const main = newMain(PartialEvalTrace);
  const trace = new PartialEvalTrace(main);
  const tracersIn = pvalsIn.map((pval) => trace.newArg(pval));
  const unknownTracersIn = tracersIn
    .filter((t) => !t.pval.isKnown)
    .map((t) => t.ref);

  const outs = f(...tracersIn);
  const tracersOut: PartialEvalTracer[] = outs.map((out: TracerValue) =>
    fullRaise(trace, out),
  );

  const pvalsOut = tracersOut.map((t) => t.pval); // Ownership either transferred here, or in the next line.
  const unknownTracersOut = tracersOut.filter((t) => !t.pval.isKnown);
  const { jaxpr, consts } = partialEvalGraphToJaxpr(
    unknownTracersIn,
    unknownTracersOut,
  );
  return { jaxpr, pvalsOut, consts };
}

/**
 * Helper function with shared Jaxpr logic between linearize and vjp.
 *
 * Internally, vjp() looks very similar to linearize() but returns a function
 * evaluating the "transposed" linearized Jaxpr, pulling back cotangents instead
 * of pushing forward tangents.
 */
function linearizeFlatUtil(
  f: (...args: any[]) => any,
  primalsIn: Tracer[],
): { primalsOut: Tracer[]; jaxpr: Jaxpr; consts: Tracer[] } {
  const pvalsIn = [
    ...primalsIn.map(PartialVal.known),
    ...primalsIn.map((t) => PartialVal.unknown(t.aval)),
  ];
  const fJvp = (...x: Tracer[]) => {
    // Args contain both primals and tangents, concatenated.
    const k = x.length / 2;
    const [primalsOut, tangentsOut] = jvp(f, x.slice(0, k), x.slice(k, 2 * k));
    return [...primalsOut, ...tangentsOut];
  };
  const { jaxpr, pvalsOut, consts } = partialEvalFlat(fJvp, pvalsIn);
  const primalPvals = pvalsOut.slice(0, pvalsOut.length / 2);
  if (!primalPvals.every((pval) => pval.isKnown)) {
    throw new TypeError(
      "Not all primal values are known after partial evaluation",
    );
  }
  const primalsOut = primalPvals.map((pval) => pval.val!);
  return { primalsOut, jaxpr, consts };
}

function linearizeFlat(
  f: (...args: any[]) => any,
  primalsIn: Tracer[],
): [Tracer[], (...args: any[]) => any] {
  const { primalsOut, jaxpr, consts } = linearizeFlatUtil(f, primalsIn);
  const fLin = (...tangents: Tracer[]) =>
    evalJaxpr(jaxpr, [...consts.map((c) => c.ref), ...tangents]);
  return [primalsOut, fLin];
}

export function linearize(
  f: (...primals: any[]) => any,
  ...primalsIn: any[]
): [any, (...tangents: any[]) => any] {
  const [primalsInFlat, inTree] = treeFlatten(primalsIn);
  const [fFlat, outTree] = flattenFun(f, inTree);
  const [primalsOutFlat, fLinFlat] = linearizeFlat(
    fFlat,
    primalsInFlat.map(pureArray),
  );
  if (outTree.value === undefined) {
    throw new Error("outTree was not set in linearize");
  }
  const primalsOut = treeUnflatten(outTree.value, primalsOutFlat);
  const fLin = (...tangentsIn: any[]) => {
    const [tangentsInFlat, inTree2] = treeFlatten(tangentsIn);
    if (!inTree.equals(inTree2)) {
      throw new TreeMismatchError("linearize", inTree, inTree2);
    }
    const tangentsOutFlat = fLinFlat(...tangentsInFlat.map(pureArray));
    return treeUnflatten(outTree.value!, tangentsOutFlat);
  };
  return [primalsOut, fLin];
}

// Used in PartialEvalTracer to track recipes for "unknown" partial vals.
type JaxprRecipe =
  | {
      type: "LambdaBinding";
    }
  | {
      // Note: Not really a constant, actually just a "known" value translated
      // into unknown for abstract evaluation rules.
      type: "Const";
      val: Tracer; // holds reference
    }
  | {
      type: "JaxprEqn";
      prim: Primitive;
      tracersIn: PartialEvalTracer[]; // holds reference
      params: Record<string, any>;
      avalsOut: ShapedArray[];
      tracerRefsOut: WeakRef<PartialEvalTracer>[];
    };

class PartialEvalTracer extends Tracer {
  #rc: number; // PartialEvalTracer reference count, used to free references.

  // Note: Either pval is known and recipe is null, or pval is unknown and
  // recipe describes how to compute the value.
  constructor(
    trace: Trace,
    readonly pval: PartialVal,
    readonly recipe: JaxprRecipe | null,
  ) {
    super(trace);
    this.#rc = 1;
  }

  get aval(): AbstractValue {
    return this.pval.aval;
  }

  toString(): string {
    if (!this.recipe) {
      return `PartialEvalTracer(${this.pval.toString()})`;
    } else {
      return `PartialEvalTracer<${this.recipe.type}>(${this.pval.toString()})`;
    }
  }

  get ref() {
    this.#rc++;
    return this;
  }
  dispose() {
    if (this.#rc === 0) {
      throw new UseAfterFreeError(this);
    }
    if (--this.#rc === 0) {
      // Clear reference to the recipe and pval, if needed.
      if (this.pval.isKnown) {
        this.pval.val!.dispose();
      } else if (this.recipe) {
        if (this.recipe.type === "Const") {
          this.recipe.val.dispose();
        } else if (this.recipe.type === "JaxprEqn") {
          this.recipe.tracersIn.forEach((t) => t.dispose());
        }
      }
    }
  }

  fullLower(): Tracer {
    if (this.pval.isKnown) {
      const val = this.pval.val!.ref;
      this.dispose();
      return val;
    }
    return this;
  }
}

class PartialEvalTrace extends Trace {
  newArg(pval: PartialVal) {
    if (pval.isKnown) return new PartialEvalTracer(this, pval, null);
    return new PartialEvalTracer(this, pval, { type: "LambdaBinding" });
  }

  pure(val: TracerValue): Tracer {
    return new PartialEvalTracer(this, PartialVal.known(pureArray(val)), null);
  }
  lift = this.pure;

  instantiateConst(tracer: PartialEvalTracer) {
    if (!tracer.pval.isKnown) {
      return tracer;
    } else {
      // Translate known value into unknown "Const" recipe for abstract eval.
      const pval = PartialVal.unknown(ShapedArray.fromAval(tracer.aval));
      const val = tracer.pval.val!.ref;
      tracer.dispose();
      return new PartialEvalTracer(this, pval, { type: "Const", val });
    }
  }

  processPrimitive(
    primitive: Primitive,
    tracers: PartialEvalTracer[],
    params: Record<string, any>,
  ): Tracer[] {
    if (tracers.every((t) => t.pval.isKnown)) {
      return bind(
        primitive,
        tracers.map((t) => t.fullLower()),
        params,
      );
    }
    if (primitive === Primitive.JitCall) {
      // Special case, needs its own PartialEvalTrace handling because unlike
      // other primtiives, JitCall can have subexpressions that are known while
      // other outputs are unknown.
      return this.#partialEvalJaxpr(params.jaxpr, params.numConsts, tracers);
    }
    const tracersIn = tracers.map((t) => this.instantiateConst(t));
    const avalsIn = tracersIn.map((t) => t.pval.aval);
    const avalsOut = abstractEvalRules[primitive](avalsIn, params);
    const recipe: JaxprRecipe = {
      type: "JaxprEqn",
      prim: primitive,
      tracersIn,
      params,
      avalsOut,
      tracerRefsOut: [], // Populated later on
    };
    const tracersOut = avalsOut.map((aval, i) => {
      if (i > 0) {
        // Make sure we increment reference count for each tracer in the recipe,
        // since they belong to multiple PartialEvalTracers.
        tracersIn.forEach((t) => t.ref);
      }
      return new PartialEvalTracer(this, PartialVal.unknown(aval), recipe);
    });
    recipe.tracerRefsOut = tracersOut.map((t) => new WeakRef(t));
    return tracersOut;
  }

  /**
   * Evaluate a Jaxpr on a set of PartialEvalTracers, computing as many known
   * values as possible (with JIT) and forwarding the unknown ones.
   *
   * Used when encountering a JitCall rule during the trace.
   */
  #partialEvalJaxpr(
    jaxpr: Jaxpr,
    numConsts: number,
    tracers: PartialEvalTracer[],
  ): Tracer[] {
    void numConsts; // Unused
    jaxpr = jaxpr.flatten(); // Otherwise, we don't partially evaluate nested Jaxprs well.

    const inUnknowns = tracers.map((t) => !t.pval.isKnown);
    const { jaxpr1, jaxpr2, outUnknowns, numRes } = partialEvalJaxpr(
      jaxpr,
      inUnknowns,
    );

    const [knownTracers, unknownTracers] = partitionList(inUnknowns, tracers);

    const outs1Res = bind(
      Primitive.JitCall,
      knownTracers.map((t) => t.ref.fullLower()),
      { jaxpr: jaxpr1, numConsts: 0 },
    );
    const outs1 = outs1Res.slice(0, jaxpr1.outs.length - numRes);
    const res = outs1Res.slice(jaxpr1.outs.length - numRes);

    const resTracers = res.map((x) =>
      this.instantiateConst(fullRaise(this, x) as PartialEvalTracer),
    );
    const recipe: JaxprRecipe = {
      type: "JaxprEqn",
      prim: Primitive.JitCall,
      tracersIn: resTracers.concat(unknownTracers),
      params: { jaxpr: jaxpr2, numConsts: 0 },
      avalsOut: jaxpr2.outs.map((x) => x.aval),
      tracerRefsOut: [], // populated later
    };
    const outs2 = jaxpr2.outs.map(
      (x) => new PartialEvalTracer(this, PartialVal.unknown(x.aval), recipe),
    );
    recipe.tracerRefsOut = outs2.map((t) => new WeakRef(t));

    // Stitch the known and unknown output tracers together, both with JitCall.
    let i = 0;
    let j = 0;
    return outUnknowns.map((unk) => (unk ? outs2[j++] : outs1[i++]));
  }
}

/** Partially evaluate a Jaxpr, returning an immediate and residual Jaxpr. */
function partialEvalJaxpr(
  jaxpr: Jaxpr,
  inUnknowns: boolean[],
  instantiate?: boolean[],
): { jaxpr1: Jaxpr; jaxpr2: Jaxpr; outUnknowns: boolean[]; numRes: number } {
  jaxpr = jaxpr.flatten(); // Otherwise, we don't partially evaluate nested Jaxprs well.

  const knownIns = jaxpr.inBinders.filter((_, i) => !inUnknowns[i]);
  const knownVars = new Set(knownIns); // Var that we can evaluate immediately.
  const residuals = new Set<Var>(); // Vars to evaluate in eqns1, and pass to eqns2 (subset of knownVars).

  const eqns1: JaxprEqn[] = [];
  const eqns2: JaxprEqn[] = [];
  for (const eqn of jaxpr.eqns) {
    if (eqn.primitive === Primitive.JitCall) {
      throw new TypeError("partialEvalJaxpr requires flattened Jaxpr");
    }
    const hasUnknowns = eqn.inputs.some(
      (x) => x instanceof Var && !knownVars.has(x),
    );
    if (hasUnknowns) {
      for (const x of eqn.inputs) {
        if (x instanceof Var && knownVars.has(x)) {
          residuals.add(x);
        }
      }
      eqns2.push(eqn);
    } else {
      eqns1.push(eqn);
      for (const v of eqn.outBinders) {
        knownVars.add(v);
      }
    }
  }
  const outUnknowns = jaxpr.outs.map(
    (x) => x instanceof Var && !knownVars.has(x),
  );
  // If instantiate is provided, move selected outputs into residuals.
  if (instantiate !== undefined) {
    for (let i = 0; i < jaxpr.outs.length; i++) {
      const x = jaxpr.outs[i];
      if (instantiate[i] && !outUnknowns[i] && x instanceof Var) {
        residuals.add(x);
        outUnknowns[i] = true; // Mark as unknown.
      }
    }
  }

  const residualsL = Array.from(residuals);
  const [ins1, ins2] = partitionList(inUnknowns, jaxpr.inBinders);
  const [outs1, outs2] = partitionList(outUnknowns, jaxpr.outs);
  const jaxpr1 = new Jaxpr(ins1, eqns1, outs1.concat(residualsL));
  const jaxpr2 = new Jaxpr(residualsL.concat(ins2), eqns2, outs2);
  return { jaxpr1, jaxpr2, outUnknowns, numRes: residualsL.length };
}

/**
 * Convert the graph representation of a partial eval to a standard Jaxpr.
 * Also called `tracers_to_jaxpr()` in JAX.
 */
function partialEvalGraphToJaxpr(
  tracersIn: PartialEvalTracer[],
  tracersOut: PartialEvalTracer[],
): { jaxpr: Jaxpr; consts: Tracer[] } {
  const tracerToVar = new Map<PartialEvalTracer, Var>();
  const constToVar = new Map<Tracer, Var>();
  const processedEqns = new Set<JaxprRecipe>(); // Avoid translating the same equation multiple times.
  const eqns: JaxprEqn[] = [];

  for (const t of tracersIn) {
    tracerToVar.set(t, new Var(ShapedArray.fromAval(t.aval)));
  }

  for (const t of toposort(tracersOut, (t) =>
    t.recipe?.type === "JaxprEqn" ? t.recipe.tracersIn : [],
  )) {
    if (!t.recipe) {
      throw new TypeError("Tracer is missing a recipe, cannot construct Jaxpr");
    }
    if (t.recipe.type === "LambdaBinding") {
      // Check that the binding is in the input list.
      if (!tracersIn.includes(t)) {
        throw new TypeError("LambdaBinding tracer not in input list");
      }
    } else if (t.recipe.type === "Const") {
      const val = t.recipe.val;
      let binder = constToVar.get(val);
      if (!binder) {
        binder = new Var(ShapedArray.fromAval(val.aval));
        constToVar.set(val, binder);
      }
      tracerToVar.set(t, binder);
    } else if (t.recipe.type === "JaxprEqn") {
      if (!processedEqns.has(t.recipe)) {
        processedEqns.add(t.recipe);
        const tracersIn = t.recipe.tracersIn.map((t) => tracerToVar.get(t)!);
        const outBinders = t.recipe.avalsOut.map((aval) => new Var(aval));
        for (let i = 0; i < outBinders.length; i++) {
          const tracerOut = t.recipe.tracerRefsOut[i].deref();
          if (tracerOut) {
            tracerToVar.set(tracerOut, outBinders[i]);
          }
        }
        eqns.push(
          new JaxprEqn(t.recipe.prim, tracersIn, t.recipe.params, outBinders),
        );
      }
    }
  }

  const [consts, constvars] = unzip2(constToVar.entries());
  const inBinders = [
    ...constvars,
    ...tracersIn.map((t) => tracerToVar.get(t)!),
  ];
  const outVars = tracersOut.map((t) => tracerToVar.get(t)!);
  const jaxpr = new Jaxpr(inBinders, eqns, outVars);
  typecheckJaxpr(jaxpr); // sanity check

  // Fix up reference counts.
  for (const t of consts) t.ref;
  for (const t of tracersIn) t.dispose();
  for (const t of tracersOut) t.dispose();

  return { jaxpr: jaxpr.simplify(), consts };
}

// implementation of vjp and grad

/** Marker type for pullback, used by transpose rules. */
class UndefPrimal {
  readonly aval: ShapedArray;

  constructor(aval: AbstractValue) {
    this.aval = ShapedArray.fromAval(aval);
  }
}

/**
 * Evaluate the backward pass over a linearized Jaxpr (pullback of cotangents).
 *
 * Will raise a TypeError if the provided Jaxpr is not a linear function of its,
 * inputs, as general expressions cannot be transposed.
 */
function evalJaxprTransposed(
  jaxpr: Jaxpr,
  args: (Tracer | UndefPrimal)[],
  cotangents: Tracer[],
): Tracer[] {
  const knownPrimals = new Map<Var, Tracer>();
  for (let i = 0; i < jaxpr.inBinders.length; i++) {
    if (!(args[i] instanceof UndefPrimal)) {
      knownPrimals.set(jaxpr.inBinders[i], args[i] as Tracer);
    }
  }

  const ctStore = new Map<Var, Tracer>();

  const readCotangent = (v: Var) => {
    const ct = ctStore.get(v);
    if (ct) {
      // We should read a cotangent at most once, as an out binder.
      ctStore.delete(v);
      return ct;
    } else {
      return zeros(v.aval.shape, { dtype: v.aval.dtype });
    }
  };

  const writeCotangent = (v: Var, ct: Tracer | null) => {
    if (ct !== null) {
      const oldCt = ctStore.get(v);
      // May need to accumulate cotangents if used in multiple JaxprEqns.
      if (oldCt) ctStore.set(v, add(oldCt, ct));
      else ctStore.set(v, ct);
    }
  };

  for (let i = 0; i < jaxpr.outs.length; i++) {
    const v = jaxpr.outs[i];
    if (v instanceof Var) writeCotangent(v, cotangents[i]);
  }

  for (let i = jaxpr.eqns.length - 1; i >= 0; i--) {
    const eqn = jaxpr.eqns[i];
    // Inputs are primalsIn and cotangentsOut, outputs are cotangentsIn. We're
    // using the known primal values to _pull back_ cotangents for unknown
    // values. Tricky!
    const primalsIn = eqn.inputs.map((v) =>
      v instanceof Lit
        ? scalar(v.value, { dtype: v.dtype }) // TODO: Use correct backend
        : knownPrimals.has(v)
          ? knownPrimals.get(v)!.ref
          : new UndefPrimal(v.aval),
    );
    const cotangentsOut = eqn.outBinders.map(readCotangent);
    const rule = transposeRules[eqn.primitive];
    if (!rule) {
      throw new TypeError(`Backward pass not implemented for ${eqn.primitive}`);
    }
    const cotangentsIn = rule(cotangentsOut, primalsIn, eqn.params);
    for (let j = 0; j < eqn.inputs.length; j++) {
      const v = eqn.inputs[j];
      if (v instanceof Var && !knownPrimals.has(v)) {
        writeCotangent(v, cotangentsIn[j]);
      } else if (cotangentsIn[j] !== null) {
        throw new Error("invariant violation: cotangent should be null");
      }
    }
  }
  for (const t of knownPrimals.values()) t.dispose();

  const results: Tracer[] = [];
  for (let i = 0; i < jaxpr.inBinders.length; i++) {
    if (args[i] instanceof UndefPrimal) {
      results.push(readCotangent(jaxpr.inBinders[i]));
    }
  }
  return results;
}

class NonlinearError extends TypeError {
  constructor(primitive: Primitive) {
    super(`Nonlinear operation in backward pass for ${primitive}`);
  }
}

type TransposeRule = (
  cotangents: Tracer[],
  primals: (Tracer | UndefPrimal)[],
  params: any,
) => (Tracer | null)[];

// You need a transpose rule for a primitive p if:
//  - p is used in jvpRules, while computing a tangent (not primal)
//  - in this use, at least one argument to p is a tangent
//
// This computes a backward pass, so it pulls back cotangents to the inputs of p
// that are UndefPrimal (i.e., tangents that weren't sent forward).
const transposeRules: Partial<Record<Primitive, TransposeRule>> = {
  [Primitive.Mul]([ct], [x, y]) {
    // BUG: Doesn't handle broadcasting.
    if (x instanceof UndefPrimal === y instanceof UndefPrimal)
      throw new NonlinearError(Primitive.Mul);
    return [
      x instanceof UndefPrimal ? mul(ct, y as Tracer) : null,
      y instanceof UndefPrimal ? mul(x as Tracer, ct) : null,
    ];
  },
  [Primitive.Neg]([ct], [x]) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Neg);
    return [neg(ct)];
  },
  [Primitive.Add]([ct], [x, y]) {
    // BUG: Doesn't handle broadcasting.
    if (!(x instanceof UndefPrimal || y instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Add);
    if (x instanceof UndefPrimal && y instanceof UndefPrimal)
      return [ct.ref, ct];
    return x instanceof UndefPrimal
      ? ((y as Tracer).dispose(), [ct, null])
      : ((x as Tracer).dispose(), [null, ct]);
  },
  [Primitive.ReduceSum]([ct], [x], { axis }: { axis: number[] }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.ReduceSum);
    return [broadcast(ct, x.aval.shape, axis)];
  },
  // BUG: Doesn't handle broadcasting.
  [Primitive.Where]([ct], [cond, x, y]) {
    // Cotangent should be zero where cond doesn't apply.
    const cts: (Tracer | null)[] = [null, null, null];
    if (cond instanceof UndefPrimal) throw new NonlinearError(Primitive.Where);
    if (x instanceof UndefPrimal) {
      // TODO: Use correct backend.
      const zerosX = zeros(x.aval.shape, { dtype: x.aval.dtype });
      cts[1] = where(cond.ref, ct.ref, zerosX);
    } else {
      x.dispose();
    }
    if (y instanceof UndefPrimal) {
      // TODO: Use correct backend.
      const zerosY = zeros(x.aval.shape, { dtype: x.aval.dtype });
      cts[2] = where(cond.ref, zerosY, ct.ref);
    } else {
      y.dispose();
    }
    ct.dispose();
    cond.dispose();
    return cts;
  },
  [Primitive.Transpose]([ct], [x], { perm }: { perm: number[] }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Transpose);
    return [transpose(ct, invertPermutation(perm))];
  },
  [Primitive.Broadcast]([ct], [x], { axis }: { axis: number[] }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Broadcast);
    return [reduceSum(ct, axis)];
  },
  [Primitive.Reshape]([ct], [x], _: { shape: number[] }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Reshape);
    return [reshape(ct, x.aval.shape)];
  },
  [Primitive.Flip]([ct], [x], { axis }: { axis: number[] }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Flip);
    return [flip(ct, axis)];
  },
  [Primitive.JitCall](cts, args, { jaxpr }: { jaxpr: Jaxpr }) {
    // We need this one because the jvp() rule for JitCall generates a JitCall
    // with the transformed Jaxpr. So grad-of-jit will result in a transposed
    // JitCall, which we need to handle.
    const undefPrimals = args.map((x) => x instanceof UndefPrimal);
    const { newJaxpr, newConsts } = transposeJaxpr(jaxpr, undefPrimals);
    const residuals = args.filter((x, i) => !undefPrimals[i]) as Tracer[];
    const outs = bind(Primitive.JitCall, [...newConsts, ...residuals, ...cts], {
      jaxpr: newJaxpr,
      numConsts: newConsts.length,
    });
    // Now pull cotangents back to the corresponding UndefPrimal inputs.
    let i = 0;
    return undefPrimals.map((isUndef) => (isUndef ? outs[i++] : null));
  },
};

const transposeJaxprCache = new Map<
  Jaxpr,
  Map<string, ReturnType<typeof transposeJaxpr>>
>();

function transposeJaxpr(
  jaxpr: Jaxpr,
  undefPrimals: boolean[],
): { newJaxpr: Jaxpr; newConsts: Tracer[] } {
  const cacheKey = JSON.stringify(undefPrimals); // deterministic
  const prevResult = transposeJaxprCache.get(jaxpr)?.get(cacheKey);
  if (prevResult) return prevResult;

  // This handles grad-of-jit or transpose-of-jit. To do this, it needs to
  // evaluate the Jaxpr transposed and then retrace it. See the comment in
  // jvpJaxpr() to explain more about what's going on here.
  const { inTypes, outTypes } = typecheckJaxpr(jaxpr);

  // Need to remove the UndefPrimals from the input types, as they are not
  // inputs to the Jaxpr while tracing.
  const forwardInTypes = inTypes.filter((_, i) => !undefPrimals[i]);
  const { jaxpr: newJaxpr, consts: newConsts } = makeJaxpr(
    (forwardIn: Tracer[], cotangents: Tracer[]) => {
      const args: (Tracer | UndefPrimal)[] = [];
      let forwardInIdx = 0; // index in forwardIn
      for (let i = 0; i < undefPrimals.length; i++) {
        if (undefPrimals[i]) args.push(new UndefPrimal(inTypes[i]));
        else args.push(forwardIn[forwardInIdx++]);
      }
      return evalJaxprTransposed(jaxpr, args, cotangents);
    },
  )(forwardInTypes, outTypes);
  typecheckJaxpr(newJaxpr); // sanity check
  const result = { newJaxpr, newConsts };

  if (!transposeJaxprCache.has(jaxpr))
    transposeJaxprCache.set(jaxpr, new Map());
  transposeJaxprCache.get(jaxpr)!.set(cacheKey, result);
  return result;
}

function vjpFlat(
  f: (...x: Tracer[]) => Tracer[],
  primalsIn: Tracer[],
): [Tracer[], (...cotangents: Tracer[]) => Tracer[]] {
  const { primalsOut, jaxpr, consts } = linearizeFlatUtil(f, primalsIn);
  const transposeInputs = [
    ...consts.map((c) => c.ref),
    // Explcitly list which arguments should be transposed.
    ...primalsIn.map((t) => new UndefPrimal(t.aval)),
  ];
  // Pullback cotangents to the UndefPrimal transpose inputs.
  const fVjp = (...cotangents: Tracer[]) =>
    evalJaxprTransposed(jaxpr, transposeInputs, cotangents);
  return [primalsOut, fVjp];
}

export function vjp(
  f: (...primals: any) => any,
  ...primalsIn: any
): [any, (...cotangents: any) => any] {
  const [primalsInFlat, inTree] = treeFlatten(primalsIn);
  const [fFlat, outTree] = flattenFun(f, inTree);
  const [primalsOutFlat, fVjpFlat] = vjpFlat(
    fFlat,
    primalsInFlat.map(pureArray),
  );
  if (outTree.value === undefined) {
    throw new Error("outTree was not set in vjp");
  }
  const primalsOut = treeUnflatten(outTree.value, primalsOutFlat);

  // "cotangentsOut" because pullback
  const fVjp = (cotangentsOut: any) => {
    const [cotangentsOutFlat, outTree2] = treeFlatten(cotangentsOut);
    if (!outTree.value!.equals(outTree2)) {
      throw new TreeMismatchError("vjp", outTree.value!, outTree2);
    }
    const cotangentsInFlat = fVjpFlat(...cotangentsOutFlat.map(pureArray));
    return treeUnflatten(inTree, cotangentsInFlat);
  };

  return [primalsOut, fVjp];
}

export function grad(f: (...primals: any) => Tracer) {
  return (...x: any) => {
    const [y, fVjp] = vjp(f, ...x);
    if (!(y instanceof Tracer) || ndim(y) !== 0) {
      throw new TypeError("grad requires a scalar output");
    }
    if (y.dtype !== DType.Float32) {
      throw new TypeError("grad currently only supports float32");
    }
    // JAX convention, differentiate with respect to the first argument.
    return fVjp(pureArray(1))[0];
  };
}
