/** @file Implementations of vjp() and partial evaluation. */

import { AluOp, isFloatDtype } from "../alu";
import {
  dispose as treeDispose,
  flatten as treeFlatten,
  map as treeMap,
  unflatten as treeUnflatten,
} from "../tree";
import {
  checkInts,
  DEBUG,
  deepEqual,
  generalBroadcast,
  invertPermutation,
  partitionList,
  range,
  toposort,
  unzip2,
} from "../utils";
import { array, eye, onesLike, pureArray, zeros } from "./array";
import {
  AbstractValue,
  add,
  bind,
  broadcast,
  concatenate,
  conv,
  flattenFun,
  flattenFunWithAux,
  flip,
  fullRaise,
  mul,
  ndim,
  neg,
  newMain,
  pad,
  Primitive,
  PrimitiveParams,
  reduce,
  reshape,
  ShapedArray,
  shrink,
  split,
  stopGradient,
  Trace,
  Tracer,
  TracerValue,
  transpose,
  TreeMismatchError,
  triangularSolve,
  UseAfterFreeError,
  where,
} from "./core";
import {
  abstractEvalRules,
  ClosedJaxpr,
  evalJaxpr,
  Jaxpr,
  JaxprEqn,
  Lit,
  makeJaxpr,
  OwnedFunction,
  typecheckJaxpr,
  Var,
} from "./jaxpr";
import { jvp, lowerAux } from "./jvp";
import { jacfwd, moveaxis, vmap } from "./vmap";

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
    return this.val ? this.val.toString() : this.aval.toString();
  }
}

function partialEvalFlat(
  f: (...args: any[]) => any,
  pvalsIn: PartialVal[],
): { jaxpr: ClosedJaxpr; pvalsOut: PartialVal[] } {
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
  const jaxpr = partialEvalGraphToJaxpr(unknownTracersIn, unknownTracersOut);
  return { jaxpr, pvalsOut };
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
): { primalsOut: Tracer[]; jaxpr: ClosedJaxpr } {
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
  const { jaxpr, pvalsOut } = partialEvalFlat(fJvp, pvalsIn);
  const primalPvals = pvalsOut.slice(0, pvalsOut.length / 2);
  if (!primalPvals.every((pval) => pval.isKnown)) {
    throw new Error("Not all primal values are known after partial evaluation");
  }
  const primalsOut = primalPvals.map((pval) => pval.val!);
  return { primalsOut, jaxpr };
}

function linearizeFlat(
  f: (...args: any[]) => any,
  primalsIn: Tracer[],
): [Tracer[], (...args: Tracer[]) => Tracer[], () => void] {
  const { primalsOut, jaxpr } = linearizeFlatUtil(f, primalsIn);
  const fLin = (...tangents: Tracer[]) =>
    evalJaxpr(jaxpr.jaxpr, [...jaxpr.consts.map((c) => c.ref), ...tangents]);
  const dispose = () => jaxpr.dispose();
  return [primalsOut, fLin, dispose];
}

export function linearize(
  f: (...primals: any[]) => any,
  primalsIn: any[],
  { hasAux = false } = {},
): [any, OwnedFunction<(...tangents: any[]) => any>, any?] {
  const [primalsInFlat, inTree] = treeFlatten(primalsIn);
  let fFlat, outTree, aux;
  if (hasAux) {
    [fFlat, outTree, aux] = flattenFunWithAux(f, inTree);
  } else {
    [fFlat, outTree] = flattenFun(f, inTree);
  }
  const [primalsOutFlat, fLinFlat, dispose] = linearizeFlat(
    fFlat,
    primalsInFlat.map(pureArray),
  );
  if (outTree.value === undefined) {
    throw new Error("outTree was not set in linearize");
  }
  const primalsOut = treeUnflatten(outTree.value, primalsOutFlat);
  const fLin = ((...tangentsIn: any[]) => {
    const [tangentsInFlat, inTree2] = treeFlatten(tangentsIn);
    if (!inTree.equals(inTree2)) {
      throw new TreeMismatchError("linearize", inTree, inTree2);
    }
    const tangentsOutFlat = fLinFlat(...tangentsInFlat.map(pureArray));
    return treeUnflatten(outTree.value!, tangentsOutFlat);
  }) as OwnedFunction<(...tangents: any[]) => any>;
  fLin.dispose = dispose;
  if (hasAux) {
    return [primalsOut, fLin, lowerAux(aux!.value)];
  }
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
    if (this.#rc <= 0) {
      throw new UseAfterFreeError(this);
    }
    this.#rc++;
    return this;
  }
  dispose() {
    if (this.#rc <= 0) {
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

  processPrimitive<P extends Primitive>(
    primitive: P,
    tracers: PartialEvalTracer[],
    params: PrimitiveParams<P>,
  ): Tracer[] {
    if (tracers.every((t) => t.pval.isKnown)) {
      return bind(
        primitive,
        tracers.map((t) => t.fullLower()),
        params,
      );
    }
    if (primitive === Primitive.Jit) {
      // Special case, needs its own PartialEvalTrace handling because unlike
      // other primtiives, Jit can have subexpressions that are known while
      // other outputs are unknown.
      const { name, jaxpr, numConsts } =
        params as PrimitiveParams<Primitive.Jit>;
      return this.#partialEvalJaxpr(name, jaxpr, numConsts, tracers);
    }
    if (primitive === Primitive.Scan) {
      // Special case for JVP'd scan: primal outputs depend only on primal inputs
      return this.#partialEvalScan(
        params as PrimitiveParams<Primitive.Scan>,
        tracers,
      );
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
   * Partial eval for Scan primitive.
   *
   * When scan is encountered during partial evaluation (e.g., inside JVP for VJP):
   * - If all inputs are known, just run the scan
   * - If this is a JVP'd scan (doubled carry/xs), we can split primal (known)
   *   from tangent (unknown) outputs
   * - Otherwise, mark all outputs as unknown
   */
  #partialEvalScan(
    params: PrimitiveParams<Primitive.Scan>,
    tracers: PartialEvalTracer[],
  ): Tracer[] {
    const { numConsts: _numConsts, numCarry } = params;

    // Determine which tracers are known/unknown
    const isKnown = tracers.map((t) => t.pval.isKnown);

    // Check if any inputs are unknown
    const hasUnknown = isKnown.some((k) => !k);

    if (!hasUnknown) {
      // All inputs known, just run the scan
      const inputs = tracers.map((t) => t.fullLower());
      return bind(Primitive.Scan, inputs, params);
    }

    // Get abstract values for all outputs
    const avalsIn = tracers.map((t) => t.pval.aval);
    const avalsOut = abstractEvalRules[Primitive.Scan](avalsIn, params);
    const numY = avalsOut.length - numCarry;

    // Check if this looks like a JVP'd scan (even numCarry and numY).
    // This heuristic is safe within linearize/transpose because:
    // 1. JVP always doubles carries and outputs (primal + tangent)
    // 2. This code only runs during autodiff, not user-written scans
    // 3. A user scan with even counts would still work, just skip JVP optimizations
    const isJvpScan = numCarry % 2 === 0 && numY % 2 === 0;

    if (!isJvpScan) {
      // Not a JVP scan, mark all outputs as unknown
      const tracersIn = tracers.map((t) => this.instantiateConst(t));
      const recipe: JaxprRecipe = {
        type: "JaxprEqn",
        prim: Primitive.Scan,
        tracersIn,
        params,
        avalsOut,
        tracerRefsOut: [],
      };
      const tracersOut = avalsOut.map((aval, i) => {
        if (i > 0) tracersIn.forEach((t) => t.ref);
        return new PartialEvalTracer(this, PartialVal.unknown(aval), recipe);
      });
      recipe.tracerRefsOut = tracersOut.map((t) => new WeakRef(t));
      return tracersOut;
    }

    // This is a JVP'd scan. We need to:
    // 1. Run primal-only computation to get known outputs
    // 2. Create a residual jaxpr for tangent computation

    const numPrimalCarry = numCarry / 2;
    const numPrimalY = numY / 2;

    // Run primal-only computation using known inputs + zeros for tangent
    const fullInputs = tracers.map((t) => {
      if (t.pval.isKnown) {
        return (t.pval.val as Tracer).ref;
      } else {
        return zeros(t.pval.aval.shape, { dtype: t.pval.aval.dtype });
      }
    });

    const fullOuts = bind(Primitive.Scan, fullInputs, params);

    // Create tracersIn for the residual jaxpr
    // All inputs become either consts or lambda bindings
    const tracersIn = tracers.map((t) => this.instantiateConst(t));

    // Build recipe for the full scan - avalsOut must match actual scan outputs
    const recipe: JaxprRecipe = {
      type: "JaxprEqn",
      prim: Primitive.Scan,
      tracersIn,
      params,
      avalsOut,
      tracerRefsOut: [],
    };

    // Build output tracers
    // We'll populate tracerRefsOut to match the output indices
    const tracersOut: PartialEvalTracer[] = [];

    // Primal carry outputs (first numPrimalCarry) are known
    for (let i = 0; i < numPrimalCarry; i++) {
      tracersOut.push(
        new PartialEvalTracer(this, PartialVal.known(fullOuts[i]), null),
      );
    }

    // Tangent carry outputs are unknown
    let isFirstUnknown = true;
    for (let i = numPrimalCarry; i < numCarry; i++) {
      fullOuts[i].dispose();
      if (!isFirstUnknown) tracersIn.forEach((t) => t.ref);
      isFirstUnknown = false;
      tracersOut.push(
        new PartialEvalTracer(this, PartialVal.unknown(avalsOut[i]), recipe),
      );
    }

    // Primal Y outputs are known
    for (let i = 0; i < numPrimalY; i++) {
      tracersOut.push(
        new PartialEvalTracer(
          this,
          PartialVal.known(fullOuts[numCarry + i]),
          null,
        ),
      );
    }

    // Tangent Y outputs are unknown
    for (let i = numPrimalY; i < numY; i++) {
      fullOuts[numCarry + i].dispose();
      tracersIn.forEach((t) => t.ref);
      tracersOut.push(
        new PartialEvalTracer(
          this,
          PartialVal.unknown(avalsOut[numCarry + i]),
          recipe,
        ),
      );
    }

    // tracerRefsOut must have an entry for each position in avalsOut
    // Known positions get a "null" ref (or we skip them in processing)
    recipe.tracerRefsOut = tracersOut.map((t) =>
      t.pval.isKnown ? (null as any) : new WeakRef(t),
    );

    return tracersOut;
  }

  /**
   * Evaluate a Jaxpr on a set of PartialEvalTracers, computing as many known
   * values as possible (with JIT) and forwarding the unknown ones.
   *
   * Used when encountering a Jit rule during the trace.
   */
  #partialEvalJaxpr(
    name: string,
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
      Primitive.Jit,
      knownTracers.map((t) => t.ref.fullLower()),
      { name: `${name}_peval`, jaxpr: jaxpr1, numConsts: 0 },
    );
    const outs1 = outs1Res.slice(0, jaxpr1.outs.length - numRes);
    const res = outs1Res.slice(jaxpr1.outs.length - numRes);

    const resTracers = res.map((x) =>
      this.instantiateConst(fullRaise(this, x) as PartialEvalTracer),
    );
    const recipe: JaxprRecipe = {
      type: "JaxprEqn",
      prim: Primitive.Jit,
      tracersIn: resTracers.concat(unknownTracers),
      params: { name: `${name}_resid`, jaxpr: jaxpr2, numConsts: 0 },
      avalsOut: jaxpr2.outs.map((x) => x.aval),
      tracerRefsOut: [], // populated later
    };
    const outs2 = jaxpr2.outs.map((x, i) => {
      if (i > 0) {
        // Make sure we increment reference count for each tracer in the recipe,
        // since they belong to multiple PartialEvalTracers.
        recipe.tracersIn.forEach((t) => t.ref);
      }
      return new PartialEvalTracer(this, PartialVal.unknown(x.aval), recipe);
    });
    recipe.tracerRefsOut = outs2.map((t) => new WeakRef(t));

    // Stitch the known and unknown output tracers together, both with Jit.
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
    if (eqn.primitive === Primitive.Jit) {
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
): ClosedJaxpr {
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
          const ref = t.recipe.tracerRefsOut[i];
          // ref can be null for known outputs in partial-eval of JVP'd scan
          const tracerOut = ref?.deref?.();
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
  let jaxpr = new Jaxpr(inBinders, eqns, outVars);
  typecheckJaxpr(jaxpr); // sanity check

  // Fix up reference counts.
  for (const t of consts) t.ref;
  for (const t of tracersIn) t.dispose();
  for (const t of tracersOut) t.dispose();

  jaxpr = jaxpr.simplify();
  if (DEBUG >= 5) {
    console.info("jaxpr from partial evaluation:\n" + jaxpr.toString());
  }

  return new ClosedJaxpr(jaxpr, consts);
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
 * Helper to get or compute a primal (known) variable's value during transpose.
 * For intermediate variables that are known (computed from only known inputs),
 * we need to evaluate the equations that produce them.
 */
function getOrComputePrimal(
  jaxpr: Jaxpr,
  knownVars: Set<Var>,
  knownPrimals: Map<Var, Tracer>,
  v: Var,
): Tracer {
  // If we already have the value, return a ref
  if (knownPrimals.has(v)) {
    return knownPrimals.get(v)!.ref;
  }

  // Find the equation that produces this variable
  const eqn = jaxpr.eqns.find((eq) => eq.outBinders.some((out) => out === v));
  if (!eqn) {
    throw new Error(
      `Internal error: could not find equation producing variable`,
    );
  }

  // Recursively get values for inputs
  const inputVals = eqn.inputs.map((inp) =>
    inp instanceof Lit
      ? array(inp.value, { dtype: inp.dtype })
      : getOrComputePrimal(jaxpr, knownVars, knownPrimals, inp),
  );

  // Evaluate this equation
  const results = bind(eqn.primitive, inputVals, eqn.params as any);

  // Store all output values
  for (let i = 0; i < eqn.outBinders.length; i++) {
    knownPrimals.set(eqn.outBinders[i], results[i]);
  }

  // Return the requested value
  const result = knownPrimals.get(v);
  if (!result) {
    throw new Error(`Internal error: variable not produced by equation`);
  }
  return result.ref;
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
  // Track which variables are known (primal) vs unknown (tangent).
  // A variable is known if ALL its inputs are known (primal values propagate).
  // A variable is unknown if ANY of its inputs are unknown (tangent dependency).
  const knownVars = new Set<Var>();
  for (let i = 0; i < jaxpr.inBinders.length; i++) {
    if (!(args[i] instanceof UndefPrimal)) {
      knownVars.add(jaxpr.inBinders[i]);
    }
  }

  // Forward pass: propagate "known" status through equations
  for (const eqn of jaxpr.eqns) {
    const allInputsKnown = eqn.inputs.every(
      (v) => v instanceof Lit || knownVars.has(v),
    );
    if (allInputsKnown) {
      // All inputs are known → all outputs are known (primal computation)
      for (const outVar of eqn.outBinders) {
        knownVars.add(outVar);
      }
    }
  }

  // Now collect actual Tracer values for known input variables
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

    // Check if all inputs are known (using our forward-propagated knownVars)
    const allInputsKnown = eqn.inputs.every(
      (v) => v instanceof Lit || knownVars.has(v),
    );

    if (allInputsKnown) {
      // Skip equations where all inputs are known (residual equations).
      // These don't depend on unknowns and don't contribute to the linear function.
      continue;
    }

    // For equations with mixed inputs, we need to get residual values for known inputs
    // and mark unknown inputs as UndefPrimal
    const primalsIn = eqn.inputs.map((v) =>
      v instanceof Lit
        ? array(v.value, { dtype: v.dtype })
        : knownVars.has(v)
          ? getOrComputePrimal(jaxpr, knownVars, knownPrimals, v)
          : new UndefPrimal(v.aval),
    );

    const cotangentsOut = eqn.outBinders.map(readCotangent);
    const rule = transposeRules[eqn.primitive];
    if (!rule) {
      throw new TypeError(`Backward pass not implemented for ${eqn.primitive}`);
    }
    const cotangentsIn = rule(cotangentsOut, primalsIn, eqn.params as any);
    for (let j = 0; j < eqn.inputs.length; j++) {
      const v = eqn.inputs[j];
      if (v instanceof Var && !knownVars.has(v)) {
        writeCotangent(v, cotangentsIn[j]);
      } else if (cotangentsIn[j] !== null) {
        throw new Error("internal: cotangent should be null");
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

/**
 * Inverse operation of `generalBroadcast()` for backpropagation.
 *
 * `x` has the shape of the result of an operation that was broadcasted with
 * `target` (it's a cotangent during backprop). Returns a tracer with rank and
 * shape equal to `target`.
 */
function unbroadcast(x: Tracer, target: UndefPrimal): Tracer {
  const shape = target.aval.shape;

  // 1. Remove extra dimensions from x, if any.
  //    x can either have rank == target.ndim (fine!), or rank > target.ndim.
  //    In the latter case, we need to trim off extra dimensions on the left.
  const extraDims = x.ndim > shape.length ? range(x.ndim - shape.length) : [];
  if (x.ndim < shape.length) {
    throw new Error(
      `unbroadcast: x.ndim (${x.shape}) < target.ndim (${shape})`,
    );
  }

  // 2. Reduce (but keep) dimensions of x that are 1 in target.
  const unsqueeze: number[] = [];
  const keptReduceDims: number[] = [];
  for (let i = 0; i < shape.length; i++) {
    // i is indexed according to target.
    const indexFromEnd = shape.length - i; // >= 1
    const indexInX = x.ndim - indexFromEnd;
    const xLen = x.shape[indexInX];
    if (xLen > 1 && shape[i] === 1) {
      unsqueeze.push(i);
      keptReduceDims.push(indexInX);
    } else if (shape[i] !== xLen) {
      throw new Error("internal: unbroadcast shape mismatch");
    }
  }

  const reductionDims = [...extraDims, ...keptReduceDims];
  if (reductionDims.length === 0) return x;
  let result = x.sum(reductionDims);
  if (!deepEqual(result.shape, shape)) {
    result = broadcast(result, shape, unsqueeze); // keep dims selectively
  }
  return result;
}

class NonlinearError extends TypeError {
  constructor(primitive: Primitive) {
    super(`Nonlinear operation in backward pass for ${primitive}`);
  }
}

type TransposeRule<P extends Primitive> = (
  cotangents: Tracer[],
  primals: (Tracer | UndefPrimal)[],
  params: PrimitiveParams<P>,
) => (Tracer | null)[];

// You need a transpose rule for a primitive p if:
//  - p is used in jvpRules, while computing a tangent (not primal)
//  - in this use, at least one argument to p is a tangent
//
// This computes a backward pass, so it pulls back cotangents to the inputs of p
// that are UndefPrimal (i.e., tangents that weren't sent forward).
const transposeRules: Partial<{ [P in Primitive]: TransposeRule<P> }> = {
  [Primitive.Mul]([ct], [x, y]) {
    // TODO: For transpose rules on operations that have type promotion rules,
    // make sure the gradient is cast back to the correct dtype.
    if (x instanceof UndefPrimal === y instanceof UndefPrimal)
      throw new NonlinearError(Primitive.Mul);
    return [
      x instanceof UndefPrimal ? unbroadcast(mul(ct, y as Tracer), x) : null,
      y instanceof UndefPrimal ? unbroadcast(mul(x as Tracer, ct), y) : null,
    ];
  },
  [Primitive.Neg]([ct], [x]) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Neg);
    return [neg(ct)];
  },
  [Primitive.Add]([ct], [x, y]) {
    if (!(x instanceof UndefPrimal || y instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Add);
    if (x instanceof UndefPrimal && y instanceof UndefPrimal)
      return [unbroadcast(ct.ref, x), unbroadcast(ct, y)];
    return x instanceof UndefPrimal
      ? ((y as Tracer).dispose(), [unbroadcast(ct, x), null])
      : ((x as Tracer).dispose(), [null, unbroadcast(ct, y as UndefPrimal)]);
  },
  [Primitive.Reduce]([ct], [x], { op, axis }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Reduce);
    if (op === AluOp.Add) {
      return [broadcast(ct, x.aval.shape, axis)];
    } else {
      // Forward-mode jvp of product does not involve any products.
      // The same applies to min/max as non-additive reductions.
      throw new NonlinearError(Primitive.Reduce);
    }
  },
  [Primitive.Pool]([ct], [x], { window, strides }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Pool);
    return bind(Primitive.PoolTranspose, [ct], {
      inShape: x.aval.shape,
      window,
      strides,
    });
  },
  [Primitive.PoolTranspose]([ct], [x], { window, strides }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.PoolTranspose);
    return bind(Primitive.Pool, [ct], { window, strides });
  },
  [Primitive.Dot]([ct], [x, y]) {
    if (x instanceof UndefPrimal === y instanceof UndefPrimal)
      throw new NonlinearError(Primitive.Dot);
    const axisSize = generalBroadcast(x.aval.shape, y.aval.shape).slice(-1)[0];
    ct = broadcast(ct, ct.shape.concat(axisSize), [-1]); // Undo the contraction.
    return [
      x instanceof UndefPrimal ? unbroadcast(mul(ct, y as Tracer), x) : null,
      y instanceof UndefPrimal ? unbroadcast(mul(x as Tracer, ct), y) : null,
    ];
  },
  [Primitive.Conv]([ct], [lhs, rhs], params) {
    if (lhs instanceof UndefPrimal === rhs instanceof UndefPrimal)
      throw new NonlinearError(Primitive.Conv);
    // See rules for transposing a convolution in `convolution.ts`.
    const v = params.vmapDims;
    // Permutation to swap batch/channel dims (axes v and v+1), keeping vmapDims first.
    const rev01 = [...range(v), v + 1, v, ...range(v + 2, ct.ndim)];
    if (lhs instanceof UndefPrimal) {
      // Transpose to LHS (activations).
      let kernel = rhs as Tracer;
      kernel = transpose(kernel, rev01); // Reverse in <-> out channels.
      kernel = flip(kernel, range(v + 2, kernel.ndim)); // Flip spatial dimensions.
      const result = conv(ct, kernel, {
        vmapDims: v,
        strides: params.lhsDilation,
        // Reference: _conv_general_vjp_lhs_padding()
        padding: params.padding.map<[number, number]>(([pl, _pr], i) => {
          // dilated kernel_size in this dimension
          const dilatedKernel =
            (kernel.shape[i + v + 2] - 1) * params.rhsDilation[i] + 1;
          const dilatedCt = (ct.shape[i + v + 2] - 1) * params.strides[i] + 1;
          const padBefore = dilatedKernel - 1 - pl;
          // Cannot calculate `padAfter = dilatedKernel - 1 - pr` because strides
          // may not produce an equal dilated kernel, for instance, 6 / stride 2
          // produces [X_X_X_], but dilating the cotangents recovers [Y_Y_Y].
          //
          // Instead, we set it to make the output shape (before strides) match
          // with dilatedLhs, currently it's less than dilatedLhs.
          const dilatedLhs =
            (lhs.aval.shape[i + v + 2] - 1) * params.lhsDilation[i] + 1;
          const padAfter =
            dilatedLhs + dilatedKernel - 1 - dilatedCt - padBefore;
          return [padBefore, padAfter];
        }),
        lhsDilation: params.strides,
        rhsDilation: params.rhsDilation,
      });
      return [result, null];
    } else {
      // Transpose to RHS (filter).
      const newLhs = transpose(lhs as Tracer, rev01); // Reverse batch <-> in channels.
      const newRhs = transpose(ct, rev01); // Reverse batch <-> out channels.
      let result = conv(newLhs, newRhs, {
        vmapDims: v,
        strides: params.rhsDilation,
        // Reference: _conv_general_vjp_rhs_padding()
        padding: params.padding.map<[number, number]>(([pl, _pr], i) => {
          const dilatedLhs =
            (lhs.aval.shape[i + v + 2] - 1) * params.lhsDilation[i] + 1;
          const dilatedKernel =
            (rhs.aval.shape[i + v + 2] - 1) * params.rhsDilation[i] + 1;
          const dilatedCt = (ct.shape[i + v + 2] - 1) * params.strides[i] + 1;
          const padFromLhs = dilatedCt - dilatedLhs;
          const padFromRhs = dilatedKernel - pl - 1;
          return [pl, padFromLhs + padFromRhs];
        }),
        lhsDilation: params.lhsDilation,
        rhsDilation: params.strides,
      });
      result = transpose(result, rev01); // Reverse in <-> out channels.
      return [null, result];
    }
  },
  [Primitive.Where]([ct], [cond, x, y]) {
    // Cotangent should be zero where cond doesn't apply.
    const cts: (Tracer | null)[] = [null, null, null];
    if (cond instanceof UndefPrimal) throw new NonlinearError(Primitive.Where);
    if (x instanceof UndefPrimal) {
      cts[1] = unbroadcast(where(cond.ref, ct.ref, 0), x);
    } else {
      x.dispose();
    }
    if (y instanceof UndefPrimal) {
      cts[2] = unbroadcast(where(cond.ref, 0, ct.ref), y);
    } else {
      y.dispose();
    }
    ct.dispose();
    cond.dispose();
    return cts;
  },
  [Primitive.Concatenate]([ct], inputs, { axis }) {
    // The backprop of concatenate is split.
    if (inputs.some((x) => !(x instanceof UndefPrimal)))
      throw new NonlinearError(Primitive.Concatenate);
    const sizes = inputs.map((x) => x.aval.shape[axis]);
    return split(ct, axis, sizes);
  },
  [Primitive.Split](cts, [x], { axis }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Split);
    return [concatenate(cts, axis)];
  },
  [Primitive.Gather]([ct], [x, ...indices], { axis, outDim }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Gather);
    if (indices.some((i) => i instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Gather);
    void [ct, axis, outDim];
    throw new Error(
      "Gather transpose rule is not yet implemented, requires complex Scatter sum operation",
    );
  },
  [Primitive.Transpose]([ct], [x], { perm }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Transpose);
    return [transpose(ct, invertPermutation(perm))];
  },
  [Primitive.Broadcast]([ct], [x], { axis }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Broadcast);
    return [reduce(ct, AluOp.Add, axis)];
  },
  [Primitive.Reshape]([ct], [x], _) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Reshape);
    return [reshape(ct, x.aval.shape)];
  },
  [Primitive.Flip]([ct], [x], { axis }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Flip);
    return [flip(ct, axis)];
  },
  [Primitive.Shrink]([ct], [x], { slice }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Shrink);
    const width = slice.map(
      ([s, e], i) => [s, x.aval.shape[i] - e] as [number, number],
    );
    return [pad(ct, width)];
  },
  [Primitive.Pad]([ct], [x], { width }) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Pad);
    const slice = width.map(
      ([s, _e], i) => [s, s + x.aval.shape[i]] as [number, number],
    );
    return [shrink(ct, slice)];
  },
  [Primitive.TriangularSolve]([ct], [a, b], { unitDiagonal }) {
    if (a instanceof UndefPrimal || !(b instanceof UndefPrimal))
      throw new NonlinearError(Primitive.TriangularSolve);
    // The adjoint of solving a @ x.T = b.T for x, when differentiating w.r.t. b:
    //   If forward is: x.T = a^{-1} @ b.T
    //   Then adjoint is: ct_b.T = a^{-T} @ ct_x.T, so we just transpose A
    // Note: The primitive always operates on upper triangular a, so a^T is lower.
    const ctB = triangularSolve(moveaxis(a, -2, -1), ct, {
      lower: true,
      unitDiagonal,
    });
    return [null, ctB];
  },
  [Primitive.Jit](cts, args, { name, jaxpr }) {
    // We need this one because the jvp() rule for Jit generates a Jit
    // with the transformed Jaxpr. So grad-of-jit will result in a transposed
    // Jit, which we need to handle.
    const undefPrimals = args.map((x) => x instanceof UndefPrimal);
    const newJaxpr = transposeJaxpr(jaxpr, undefPrimals);
    const residuals = args.filter((x, i) => !undefPrimals[i]) as Tracer[];
    const outs = bind(
      Primitive.Jit,
      [...newJaxpr.consts.map((c) => c.ref), ...residuals, ...cts],
      {
        name: `${name}_t`,
        jaxpr: newJaxpr.jaxpr,
        numConsts: newJaxpr.consts.length,
      },
    );
    // Now pull cotangents back to the corresponding UndefPrimal inputs.
    let i = 0;
    return undefPrimals.map((isUndef) => (isUndef ? outs[i++] : null));
  },
  [Primitive.Scan](cts, args, { jaxpr, numCarry, numConsts, length, reverse }) {
    // Scan transpose rule with "store all" approach.
    //
    // Forward scan: (consts, init_carry, xs) -> (final_carry, ys)
    // body: (consts, carry, x) -> (new_carry, y)
    //
    // The transpose runs the body transposed in reverse order, accumulating
    // cotangents. This requires O(n) memory to store intermediate carries.
    //
    // For a JVP-transformed scan, the layout is:
    // args: [jvpConsts..., constsP..., constsT..., carryP..., carryT..., xsP..., xsT...]
    // cts:  [ct_carryP..., ct_carryT..., ct_ysP..., ct_ysT...]
    //
    // The primal parts are residuals (Tracer), tangent parts are UndefPrimal.
    // We compute cotangents for the tangent (UndefPrimal) inputs.
    //
    // IMPORTANT: For JVP'd scan, tangent inputs (carryT, xsT) might be Tracers
    // if they were known zeros. We detect JVP structure and treat the second
    // half of carry/xs as tangent even if they're Tracers.

    const numX = args.length - numConsts - numCarry;
    const numY = cts.length - numCarry;

    // Detect JVP structure: even numCarry, numX, and numY.
    // Safe heuristic: this code only runs during transpose of JVP'd functions,
    // where scan inputs are always doubled (primal + tangent pairs).
    const isJvpScan = numCarry % 2 === 0 && numY % 2 === 0 && numX % 2 === 0;

    // For JVP'd scan, identify which positions are primal vs tangent
    // Primal: first half of carry, first half of xs
    // Tangent: second half of carry, second half of xs
    const numPrimalCarry = isJvpScan ? numCarry / 2 : 0;
    const numPrimalX = isJvpScan ? numX / 2 : 0;
    const numPrimalY = isJvpScan ? numY / 2 : 0;

    // Identify which args are effectively UndefPrimal (need cotangents)
    // For JVP scan: tangent positions are UndefPrimal even if they're Tracers
    const undefMask = args.map((x, i) => {
      if (x instanceof UndefPrimal) return true;
      if (!isJvpScan) return false;

      // JVP structure: second half of each group is tangent
      if (i < numConsts) {
        // Consts: assume all are primal (wrapper consts)
        return false;
      } else if (i < numConsts + numCarry) {
        // Carry: second half is tangent
        const carryIdx = i - numConsts;
        return carryIdx >= numPrimalCarry;
      } else {
        // Xs: second half is tangent
        const xIdx = i - numConsts - numCarry;
        return xIdx >= numPrimalX;
      }
    });

    // The body jaxpr takes: [consts..., carry..., x...] (per iteration, unstacked x)
    // The number of const/carry/x inputs to body is:
    const bodyNumConsts = numConsts;
    const bodyNumCarry = numCarry;
    const _bodyNumX = numX;
    const _bodyNumY = numY;

    // Build transposed body jaxpr
    // Body input layout: [consts..., carry..., x...]
    // We need to know which body inputs correspond to UndefPrimal
    const bodyUndefPrimals: boolean[] = [];
    for (let i = 0; i < jaxpr.inBinders.length; i++) {
      if (i < bodyNumConsts) {
        bodyUndefPrimals.push(undefMask[i]);
      } else if (i < bodyNumConsts + bodyNumCarry) {
        // Carry inputs - these come from previous iteration or init
        // For JVP'd scan, half are primal (residual), half are tangent (undef)
        bodyUndefPrimals.push(undefMask[numConsts + (i - bodyNumConsts)]);
      } else {
        // X inputs - sliced from stacked xs
        bodyUndefPrimals.push(
          undefMask[numConsts + numCarry + (i - bodyNumConsts - bodyNumCarry)],
        );
      }
    }

    // Get residual (primal) values from scan args
    const constArgs = args.slice(0, numConsts);
    const carryArgs = args.slice(numConsts, numConsts + numCarry);
    const xsArgs = args.slice(numConsts + numCarry);

    // Split into primal and tangent parts
    const constResiduals = constArgs.filter(
      (_, i) => !undefMask[i],
    ) as Tracer[];
    const carryResiduals = carryArgs.filter(
      (_, i) => !undefMask[numConsts + i],
    ) as Tracer[];
    const xsResiduals = xsArgs.filter(
      (_, i) => !undefMask[numConsts + numCarry + i],
    ) as Tracer[];

    // Identify which carry positions are primals vs tangents
    const carryIsPrimal = carryArgs.map((_, i) => !undefMask[numConsts + i]);
    const xsIsPrimal = xsArgs.map(
      (_, i) => !undefMask[numConsts + numCarry + i],
    );

    // For JVP'd scan, we already computed numPrimalCarry etc above
    // For non-JVP scan, count from the mask
    const actualNumPrimalCarry = isJvpScan
      ? numPrimalCarry
      : carryIsPrimal.filter((x) => x).length;
    const _actualNumPrimalX = isJvpScan
      ? numPrimalX
      : xsIsPrimal.filter((x) => x).length;

    if (actualNumPrimalCarry === 0 || carryResiduals.length === 0) {
      throw new Error(
        "Scan transpose: no carry residuals available. grad() through scan " +
          "requires primal carry values to be available as residuals.",
      );
    }

    // Step 1: Re-run forward pass to get all intermediate primal carries
    // We need carries[0..length] where carries[0] = init, carries[i] = carry after iteration i-1
    //
    // For this, we run the primal-only part of the body scan
    // The primal body takes [constsP, carryP, xP] and returns [carryP_out, yP]

    // Extract primal body from JVP'd body
    // JVP body: inputs [constsP, carryP, xP, constsT, carryT, xT]
    //           outputs [carryP, yP, carryT, yT]
    // We need just the primal computation

    // Create a jaxpr that extracts just the primal forward pass
    const _primalBodyNumConsts = constResiduals.length;
    const _primalBodyNumCarry = numPrimalCarry;
    const _primalBodyNumX = numPrimalX;

    // Build the primal-only forward jaxpr by tracing just the primal outputs
    // Actually, for a JVP'd jaxpr, the first half of outputs are primal
    const halfOutputs = jaxpr.outs.length / 2;
    const _primalOutCount = Math.floor(halfOutputs);

    // Create forward body that only computes primals
    const forwardInTypes = jaxpr.inBinders
      .filter((_, i) => !bodyUndefPrimals[i])
      .map((v) => v.aval);

    // Build jaxpr for primal-only forward
    const { jaxpr: primalForwardJaxpr } = makeJaxpr(
      (...primalInputs: Tracer[]): Tracer[] => {
        // Build full inputs with zeros for tangent slots
        const fullInputs: Tracer[] = [];
        let primalIdx = 0;
        for (let i = 0; i < jaxpr.inBinders.length; i++) {
          if (bodyUndefPrimals[i]) {
            // Tangent slot: use zeros
            const aval = jaxpr.inBinders[i].aval;
            fullInputs.push(zeros(aval.shape, { dtype: aval.dtype }));
          } else {
            // Primal slot: use input
            fullInputs.push(primalInputs[primalIdx++].ref);
          }
        }
        // Eval original body
        const outs = evalJaxpr(jaxpr, fullInputs);
        // Return only primal outputs (first half of each group)
        const primalCarryOuts = outs.slice(0, numPrimalCarry);
        const primalYOuts = outs.slice(
          numCarry,
          numCarry + Math.floor(numY / 2),
        );
        // Dispose tangent outputs
        for (let i = numPrimalCarry; i < numCarry; i++) outs[i].dispose();
        for (let i = numCarry + Math.floor(numY / 2); i < outs.length; i++)
          outs[i].dispose();
        return [...primalCarryOuts, ...primalYOuts];
      },
    )(...forwardInTypes);

    // Run primal forward scan to get all intermediate carries.
    // We need to collect carries at each iteration for the backward pass.
    //
    // TODO: This stores all N carries, using O(N) memory. For long scans, consider
    // implementing binomial checkpointing to reduce memory to O(√N) with 2× recompute.
    // See: Griewank & Walther, "Algorithm 799: Revolve"
    const allCarries: Tracer[][] = [];
    let currentCarry = carryResiduals.map((c) => c.ref);
    allCarries.push(currentCarry.map((c) => c.ref)); // carries[0] = init

    // For each iteration, run forward body and save carry
    for (let iter = 0; iter < length; iter++) {
      const dataIdx = reverse ? length - 1 - iter : iter;

      // Slice xs for this iteration
      const xSlices: Tracer[] = [];
      for (const xs of xsResiduals) {
        // xs has shape [length, ...]
        const slice = shrink(xs.ref, [
          [dataIdx, dataIdx + 1],
          ...xs.shape
            .slice(1)
            .map((_, i) => [0, xs.shape[i + 1]] as [number, number]),
        ]);
        xSlices.push(reshape(slice, xs.shape.slice(1)));
      }

      // Build forward body inputs: [constsP, carryP, xP]
      const forwardInputs = [
        ...constResiduals.map((c) => c.ref),
        ...currentCarry.map((c) => c.ref),
        ...xSlices,
      ];

      // Run forward body
      const forwardOuts = evalJaxpr(primalForwardJaxpr.jaxpr, [
        ...primalForwardJaxpr.consts.map((c) => c.ref),
        ...forwardInputs,
      ]);

      // Extract new carry (first numPrimalCarry outputs)
      const newCarry = forwardOuts.slice(0, numPrimalCarry);
      // Dispose y outputs (we don't need them for backward)
      for (let i = numPrimalCarry; i < forwardOuts.length; i++) {
        forwardOuts[i].dispose();
      }

      // Dispose old carry
      for (const c of currentCarry) c.dispose();

      currentCarry = newCarry;
      allCarries.push(currentCarry.map((c) => c.ref));
    }

    // Dispose final carry (we have a ref in allCarries)
    for (const c of currentCarry) c.dispose();

    // Step 2: Create a tangent-only body for transposition
    //
    // The JVP'd body computes BOTH primal and tangent outputs:
    //   inputs: [constsP, carryP, xP, constsT, carryT, xT]
    //   outputs: [newCarryP, yP, newCarryT, yT]
    //
    // But we can't transpose this directly because the primal computation
    // (e.g., add(carryP, xP)) has no tangent inputs → NonlinearError.
    //
    // Instead, we need a tangent-only body that takes:
    //   inputs: [constsP (residuals), carryP (residuals), xP (residuals), constsT, carryT, xT]
    // And computes only tangent outputs:
    //   outputs: [newCarryT, yT]
    //
    // For transposition, the tangent inputs (constsT, carryT, xT) become UndefPrimal.

    // Build tangent-only body jaxpr
    // The body takes [constsP, carryP, xP, constsT, carryT, xT]
    // and returns [newCarryT, yT] only (tangent outputs)
    const numTangentConsts = numConsts - constResiduals.length;
    const numTangentCarry = numCarry - numPrimalCarry;
    const numTangentX = numX - numPrimalX;
    const _numTangentY = numY - numPrimalY;

    // Create a jaxpr that extracts just the tangent outputs from the full body
    const tangentBodyInAvals = [
      ...jaxpr.inBinders
        .filter((_, i) => !bodyUndefPrimals[i])
        .map((v) => v.aval), // primal inputs (residuals)
      ...jaxpr.inBinders
        .filter((_, i) => bodyUndefPrimals[i])
        .map((v) => v.aval), // tangent inputs
    ];

    const { jaxpr: tangentBody } = makeJaxpr(
      (...tangentBodyArgs: Tracer[]): Tracer[] => {
        // Split inputs into primal residuals and tangent inputs
        const numPrimalInputs = jaxpr.inBinders.filter(
          (_, i) => !bodyUndefPrimals[i],
        ).length;
        const primalResiduals = tangentBodyArgs.slice(0, numPrimalInputs);
        const tangentInputs = tangentBodyArgs.slice(numPrimalInputs);

        // Build full body inputs in original order
        const fullInputs: Tracer[] = [];
        let primalIdx = 0;
        let tangentIdx = 0;
        for (let i = 0; i < jaxpr.inBinders.length; i++) {
          if (bodyUndefPrimals[i]) {
            fullInputs.push(tangentInputs[tangentIdx++].ref);
          } else {
            fullInputs.push(primalResiduals[primalIdx++].ref);
          }
        }

        // Evaluate full body
        const fullOuts = evalJaxpr(jaxpr, fullInputs);

        // Return only tangent outputs
        // For JVP'd body, outputs are [carryP..., carryT..., yP..., yT...]
        // Wait, the actual output layout from the wrapper is [carryP, carryT, yP, yT]
        // So tangent outputs are positions [numPrimalCarry..numCarry) and [numCarry+numPrimalY..)
        const tangentOuts: Tracer[] = [];
        for (let i = numPrimalCarry; i < numCarry; i++) {
          tangentOuts.push(fullOuts[i]);
        }
        for (let i = numCarry + numPrimalY; i < fullOuts.length; i++) {
          tangentOuts.push(fullOuts[i]);
        }

        // Dispose primal outputs
        for (let i = 0; i < numPrimalCarry; i++) fullOuts[i].dispose();
        for (let i = numCarry; i < numCarry + numPrimalY; i++)
          fullOuts[i].dispose();

        return tangentOuts;
      },
    )(...tangentBodyInAvals);

    // Now transpose the tangent-only body
    // All tangent inputs are UndefPrimal
    const tangentBodyUndefPrimals = [
      ...Array(
        tangentBody.jaxpr.inBinders.length -
          (numTangentConsts + numTangentCarry + numTangentX),
      ).fill(false), // primal residuals
      ...Array(numTangentConsts + numTangentCarry + numTangentX).fill(true), // tangent inputs
    ];

    const transposedBody = transposeJaxpr(
      tangentBody.jaxpr,
      tangentBodyUndefPrimals,
    );

    // Step 3: Run backward pass in reverse
    // Split cotangents: [ct_carry..., ct_ys...]
    const ctCarryAll = cts.slice(0, numCarry);
    const ctYsAll = cts.slice(numCarry);

    // For JVP'd scan, cotangents are [ct_carryP, ct_carryT, ct_ysP, ct_ysT]
    // We only care about ct_carryT and ct_ysT (cotangents for tangent outputs)
    // Actually we receive cotangents for ALL outputs, but the primal ones should
    // be zero if the function is truly linear in tangents

    // Initialize running cotangent for carry
    // For the tangent carry, the "initial" cotangent is the cotangent of final carry
    let ctCarryRunning = ctCarryAll.slice(numPrimalCarry).map((c) => c.ref);
    // The primal carry cotangents should be zero (dispose them)
    for (let i = 0; i < numPrimalCarry; i++) ctCarryAll[i].dispose();

    // Accumulate cotangents for xs and consts
    const ctXsAccum: Tracer[][] = [];
    for (let i = 0; i < numTangentX; i++) {
      ctXsAccum.push([]);
    }

    let ctConstsAccum: Tracer[] | null = null;

    // Run backward iterations in reverse order
    for (let iter = length - 1; iter >= 0; iter--) {
      const dataIdx = reverse ? length - 1 - iter : iter;

      // Get primal carry for this iteration from allCarries
      const primalCarry = allCarries[iter];

      // Slice primal xs for this iteration
      const xSlices: Tracer[] = [];
      for (const xs of xsResiduals) {
        const slice = shrink(xs.ref, [
          [dataIdx, dataIdx + 1],
          ...xs.shape
            .slice(1)
            .map((_, i) => [0, xs.shape[i + 1]] as [number, number]),
        ]);
        xSlices.push(reshape(slice, xs.shape.slice(1)));
      }

      // Slice cotangent of y for this iteration
      const ctYSlices: Tracer[] = [];
      const _numTangentY = ctYsAll.length - Math.floor(numY / 2);
      for (let i = Math.floor(numY / 2); i < ctYsAll.length; i++) {
        const ctY = ctYsAll[i];
        const slice = shrink(ctY.ref, [
          [dataIdx, dataIdx + 1],
          ...ctY.shape
            .slice(1)
            .map((_, j) => [0, ctY.shape[j + 1]] as [number, number]),
        ]);
        ctYSlices.push(reshape(slice, ctY.shape.slice(1)));
      }
      // Dispose primal Y cotangent slices (they should be zero)
      // Actually we should skip these entirely - don't slice them

      // Build transposed body inputs
      // Transposed body takes: [residuals..., cotangents_out...]
      // residuals = [constsP, carryP, xP] (primals that were known)
      // cotangents_out = [ct_carry_out, ct_y] (for tangent outputs)

      // The transposed body jaxpr expects:
      // inputs: [forwardIn (primals), cotangents]
      // where forwardIn = residuals, cotangents = [ct_carryT, ct_yT]

      // Build cotangents for tangentBody outputs
      // tangentBody outputs are ONLY the tangent outputs: [carryT..., yT...]
      // So we pass: [ctCarryRunning..., ctYSlices...]
      const bodyOutCotangents: Tracer[] = [];
      // Tangent carry cotangents
      bodyOutCotangents.push(...ctCarryRunning.map((c) => c.ref));
      // Tangent Y cotangents
      bodyOutCotangents.push(...ctYSlices);

      // Run transposed body
      // transposedBody.jaxpr expects: [transposedBody.consts, forwardIn (primals), cotangents]
      const transposedInputs = [
        ...transposedBody.consts.map((c) => c.ref),
        ...constResiduals.map((c) => c.ref),
        ...primalCarry.map((c) => c.ref),
        ...xSlices,
        ...bodyOutCotangents,
      ];

      const transposedOuts = evalJaxpr(transposedBody.jaxpr, transposedInputs);

      // Transposed body outputs: cotangents for UndefPrimal inputs
      // Layout: [ct_constsT, ct_carryT_in, ct_xT]
      let outIdx = 0;

      // Cotangents for tangent consts
      const ctConstsIter: Tracer[] = [];
      for (let i = 0; i < numTangentConsts; i++) {
        ctConstsIter.push(transposedOuts[outIdx++]);
      }

      // Cotangents for tangent carry input (this becomes next iteration's running ct)
      const ctCarryNew: Tracer[] = [];
      const numTangentCarry = numCarry - numPrimalCarry;
      for (let i = 0; i < numTangentCarry; i++) {
        ctCarryNew.push(transposedOuts[outIdx++]);
      }

      // Cotangents for tangent x
      const ctXIter: Tracer[] = [];
      for (let i = 0; i < numTangentX; i++) {
        ctXIter.push(transposedOuts[outIdx++]);
      }

      // Accumulate const cotangents
      if (ctConstsAccum === null) {
        ctConstsAccum = ctConstsIter;
      } else {
        ctConstsAccum = ctConstsAccum.map((ct, i) => add(ct, ctConstsIter[i]));
      }

      // Store x cotangents (will stack later)
      for (let i = 0; i < numTangentX; i++) {
        ctXsAccum[i].push(ctXIter[i]);
      }

      // Update running carry cotangent
      for (const c of ctCarryRunning) c.dispose();
      ctCarryRunning = ctCarryNew;

      // Dispose primal carry refs from allCarries[iter]
      for (const c of primalCarry) c.dispose();
      // Note: xSlices were consumed by evalJaxpr, no need to dispose
      // Note: bodyOutCotangents were consumed by evalJaxpr, no need to dispose
    }

    // Dispose the last allCarries entry
    for (const c of allCarries[length]) c.dispose();

    // Dispose remaining cotangents
    for (let i = Math.floor(numY / 2); i < ctYsAll.length; i++)
      ctYsAll[i].dispose();
    for (let i = 0; i < Math.floor(numY / 2); i++) ctYsAll[i].dispose();

    // Stack x cotangents
    const ctXsStacked: Tracer[] = [];
    for (let i = 0; i < numTangentX; i++) {
      // ctXsAccum[i] has length elements in reverse order, need to reverse
      const reversed = ctXsAccum[i].reverse();
      if (reverse) reversed.reverse(); // For reverse scan, un-reverse
      // Stack along axis 0 by first expanding dims then concatenating
      const expanded = reversed.map((ct) =>
        broadcast(ct, [1, ...ct.shape], [0]),
      );
      const stacked = concatenate(expanded, 0);
      ctXsStacked.push(stacked);
    }

    // Build output cotangents
    // Return order matches UndefPrimal positions in args
    // Important: only return non-null cotangents for actual UndefPrimal inputs,
    // not just based on JVP structure. The evalJaxprTransposed expects this.
    const actualUndefMask = args.map((x) => x instanceof UndefPrimal);

    const result: (Tracer | null)[] = [];
    let ctConstIdx = 0;
    let ctCarryIdx = 0;
    let ctXIdx = 0;

    for (let i = 0; i < args.length; i++) {
      // First, advance the appropriate index based on undefMask (what we computed)
      const isJvpTangent = undefMask[i];

      if (!actualUndefMask[i]) {
        // This arg is a known primal (Tracer), return null
        // But we still need to advance the index if it was a JVP tangent
        if (isJvpTangent) {
          if (i < numConsts) {
            ctConstsAccum![ctConstIdx++].dispose();
          } else if (i < numConsts + numCarry) {
            ctCarryRunning[ctCarryIdx++].dispose();
          } else {
            ctXsStacked[ctXIdx++].dispose();
          }
        }
        result.push(null);
      } else if (i < numConsts) {
        // Tangent const that is UndefPrimal
        result.push(ctConstsAccum![ctConstIdx++]);
      } else if (i < numConsts + numCarry) {
        // Tangent carry that is UndefPrimal
        result.push(ctCarryRunning[ctCarryIdx++]);
      } else {
        // Tangent xs that is UndefPrimal
        result.push(ctXsStacked[ctXIdx++]);
      }
    }

    // Cleanup
    primalForwardJaxpr.dispose();
    transposedBody.dispose();
    // Dispose original residuals (they were .ref'd multiple times)
    for (const c of constResiduals) c.dispose();
    for (const c of carryResiduals) c.dispose();
    for (const c of xsResiduals) c.dispose();

    return result;
  },
};

const transposeJaxprCache = new Map<Jaxpr, Map<string, ClosedJaxpr>>();

function transposeJaxpr(jaxpr: Jaxpr, undefPrimals: boolean[]): ClosedJaxpr {
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
  const { jaxpr: newJaxpr } = makeJaxpr(
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
  typecheckJaxpr(newJaxpr.jaxpr); // sanity check

  if (!transposeJaxprCache.has(jaxpr))
    transposeJaxprCache.set(jaxpr, new Map());
  transposeJaxprCache.get(jaxpr)!.set(cacheKey, newJaxpr);
  return newJaxpr;
}

function vjpFlat(
  f: (...x: Tracer[]) => Tracer[],
  primalsIn: Tracer[],
): [Tracer[], (...cotangents: Tracer[]) => Tracer[], () => void] {
  const { primalsOut, jaxpr } = linearizeFlatUtil(f, primalsIn);
  // Pullback cotangents to the UndefPrimal transpose inputs.
  const fVjp = (...cotangents: Tracer[]) => {
    const transposeInputs = [
      ...jaxpr.consts.map((c) => c.ref),
      // Explcitly list which arguments should be transposed.
      ...primalsIn.map((t) => new UndefPrimal(t.aval)),
    ];
    return evalJaxprTransposed(jaxpr.jaxpr, transposeInputs, cotangents);
  };
  const dispose = () => jaxpr.dispose();
  return [primalsOut, fVjp, dispose];
}

export function vjp(
  f: (...primals: any) => any,
  primalsIn: any[],
  { hasAux = false } = {},
): [any, OwnedFunction<(...cotangents: any) => any>, any?] {
  const [primalsInFlat, inTree] = treeFlatten(primalsIn);
  let fFlat, outTree, aux;
  if (hasAux) {
    [fFlat, outTree, aux] = flattenFunWithAux(f, inTree);
  } else {
    [fFlat, outTree] = flattenFun(f, inTree);
  }
  const [primalsOutFlat, fVjpFlat, dispose] = vjpFlat(
    fFlat,
    primalsInFlat.map(pureArray),
  );
  if (outTree.value === undefined) {
    throw new Error("outTree was not set in vjp");
  }
  const primalsOut = treeUnflatten(outTree.value, primalsOutFlat);

  // "cotangentsOut" because pullback
  const fVjp = ((cotangentsOut: any) => {
    const [cotangentsOutFlat, outTree2] = treeFlatten(cotangentsOut);
    if (!outTree.value!.equals(outTree2)) {
      throw new TreeMismatchError("vjp", outTree.value!, outTree2);
    }
    const cotangentsInFlat = fVjpFlat(...cotangentsOutFlat.map(pureArray));
    return treeUnflatten(inTree, cotangentsInFlat);
  }) as OwnedFunction<(...cotangents: any) => any>;
  fVjp.dispose = dispose;

  if (hasAux) {
    return [primalsOut, fVjp, lowerAux(aux!.value)];
  }
  return [primalsOut, fVjp];
}

/** @inline */
export type GradOpts = {
  /**
   * Integer or sequence of integers. Specifies which positional argument(s) to
   * differentiate with respect to.
   *
   * Defaults to `0` (the first argument).
   */
  argnums?: number | number[];

  /**
   * The input function returns a pair of `[out, aux]` including an auxiliary
   * value. This `aux` is not differentiated, but is returned alongside the
   * gradient when evaluating the function.
   */
  hasAux?: boolean;
};

export function grad(f: (...primals: any) => Tracer, opts?: GradOpts) {
  const valueAndGradFn = valueAndGrad(f, opts);
  return (...x: any) => {
    if (opts?.hasAux) {
      const [[y, aux], dx] = valueAndGradFn(...x);
      y.dispose();
      return [dx, aux];
    } else {
      const [y, dx] = valueAndGradFn(...x);
      y.dispose();
      return dx;
    }
  };
}

export function valueAndGrad(f: (...primals: any) => Tracer, opts?: GradOpts) {
  const argnums = opts?.argnums ?? 0; // By default, differentiate w.r.t. first arg.
  const hasAux = opts?.hasAux ?? false;
  checkInts(argnums);
  const argnumsSet = new Set(typeof argnums === "number" ? [argnums] : argnums);
  return (...x: any) => {
    if (x.length === 0) {
      throw new Error("grad requires at least one argument to differentiate");
    }
    // Differentiate only with respect to the argnums.
    for (let i = 0; i < x.length; i++) {
      if (!argnumsSet.has(i)) x[i] = treeMap(stopGradient, x[i]);
    }
    const [y, fVjp, aux] = vjp(f, x, { hasAux });
    if (!(y instanceof Tracer) || ndim(y) !== 0) {
      throw new TypeError("grad requires a scalar output");
    }
    if (!isFloatDtype(y.dtype)) {
      throw new TypeError("grad only supports floating-point dtypes");
    }
    const cts = fVjp(onesLike(y.ref)); // backprop from scalar 1
    fVjp.dispose();
    for (let i = 0; i < cts.length; i++) {
      if (!argnumsSet.has(i)) treeDispose(cts[i]);
    }
    const grads =
      typeof argnums === "number" ? cts[argnums] : argnums.map((i) => cts[i]);
    return hasAux ? [[y, aux], grads] : [y, grads];
  };
}

// See also: jacfwd()
export function jacrev(f: any) {
  return function jacobianReverse(x: Tracer) {
    if (x.shape.length !== 1) {
      throw new TypeError("jacrev only supports 1D inputs");
    }
    const [size] = x.shape;
    const pullback = (ct: Tracer) => {
      const [y, fVjp] = vjp(f, [x]);
      y.dispose();
      const [ret] = fVjp(ct);
      fVjp.dispose();
      return ret;
    };
    return vmap(pullback, [1])(eye(size, undefined, { dtype: x.dtype }));
  };
}

// See also: jacfwd()
export function hessian(f: any) {
  return jacfwd(grad(f));
}
