// Handle Jit operations by translating Jaxprs into dispatched Kernels.

import {
  AluExp,
  AluOp,
  AluVar,
  byteWidth,
  DType,
  Kernel,
  Reduction,
} from "../alu";
import { Backend, Slot } from "../backend";
import { PPrint } from "../pprint";
import { Routine, Routines } from "../routine";
import { ShapeTracker, unravelAlu } from "../shape";
import {
  DEBUG,
  deepEqual,
  FpHash,
  generalBroadcast,
  prod,
  range,
  rep,
} from "../utils";
import { aluCompare, PendingExecute } from "./array";
import { pool, poolTranspose, prepareConv } from "./convolution";
import { Primitive, PrimitiveParams, promoteAvals, ShapedArray } from "./core";
import { Jaxpr, Lit, Var } from "./jaxpr";

// These primitives are handled via `Routine` instead of `Kernel` and are not
// compatible with operator fusion.
const routinePrimitives = new Map([
  [Primitive.Sort, Routines.Sort],
  [Primitive.Argsort, Routines.Argsort],
  [Primitive.TriangularSolve, Routines.TriangularSolve],
  [Primitive.Cholesky, Routines.Cholesky],
]);

export type JitId = number;

export type JitStep =
  | {
      type: "execute";
      source: Kernel | Routine;
      inputs: JitId[]; // mapped to backend Slot
      outputs: JitId[]; // mapped to backend Slot
    }
  | {
      type: "malloc";
      size: number;
      output: JitId;
    }
  | {
      type: "incref";
      input: JitId;
    }
  | {
      type: "free";
      input: JitId;
    };

/** Result of compiling a Jaxpr. Can be evaluated on a series of inputs. */
export class JitProgram {
  constructor(
    readonly backend: Backend,
    readonly steps: JitStep[],
    readonly inputs: JitId[],
    readonly outputs: JitId[],
  ) {}

  pprint(): PPrint {
    const steps: PPrint[] = this.steps.map((step) => {
      switch (step.type) {
        case "execute": {
          const inputsNice = step.inputs
            .map((id, i) => `${i}: %${id}`)
            .join(", ");
          const outputsNice = step.outputs.map((id) => `%${id}`).join(", ");
          const executeText = `execute (${inputsNice}) -> ${outputsNice}`;
          if (step.source instanceof Kernel) {
            return PPrint.pp(`${executeText}, kernel`).concat(
              step.source.pprint().indent(2),
            );
          } else if (step.source instanceof Routine) {
            return PPrint.pp(`${executeText}, routine ${step.source.name}`);
          } else {
            step.source satisfies never; // static check
            return PPrint.pp(executeText);
          }
        }
        case "malloc":
          return PPrint.pp(`%${step.output} = malloc <${step.size} bytes>`);
        case "incref":
          return PPrint.pp(`incref ${step.input}`);
        case "free":
          return PPrint.pp(`free ${step.input}`);
      }
    });
    const display = PPrint.prototype.concat(
      PPrint.pp(`device = ${this.backend.type}`),
      PPrint.pp("inputs = [" + this.inputs.join(", ") + "]"),
      PPrint.pp("outputs = [" + this.outputs.join(", ") + "]"),
      PPrint.pp("steps ="),
      PPrint.prototype.concat(...steps).indent(2),
    );
    return PPrint.pp("{ ").stack(display.stack(PPrint.pp(" }")));
  }

  toString(): string {
    return this.pprint().toString();
  }

  /** Execute the JitProgram with the given inputs. */
  execute(inputs: Slot[]): { outputs: Slot[]; pending: PendingExecute[] } {
    const scope = new Map<JitId, Slot>();
    if (inputs.length !== this.inputs.length) {
      throw new TypeError(
        `Expected ${this.inputs.length} inputs, got ${inputs.length}`,
      );
    }
    for (const [i, id] of this.inputs.entries()) {
      scope.set(id, inputs[i]);
    }
    const pending: PendingExecute[] = [];
    for (const step of this.steps) {
      switch (step.type) {
        case "execute": {
          const inputs = step.inputs.map((id) => scope.get(id)!);
          const outputs = step.outputs.map((id) => scope.get(id)!);
          if (
            inputs.some((s) => s === undefined) ||
            outputs.some((s) => s === undefined)
          ) {
            throw new Error(`internal: JitProgram scope undefined`);
          }
          pending.push(
            new PendingExecute(this.backend, step.source, inputs, outputs),
          );
          break;
        }
        case "malloc": {
          const slot = this.backend.malloc(step.size);
          scope.set(step.output, slot);
          break;
        }
        case "incref": {
          const slot = scope.get(step.input)!;
          this.backend.incRef(slot);
          break;
        }
        case "free": {
          const slot = scope.get(step.input)!;
          this.backend.decRef(slot);
          scope.delete(step.input);
          break;
        }
        default:
          step satisfies never;
      }
    }
    return {
      outputs: this.outputs.map((id) => scope.get(id)!),
      pending,
    };
  }
}

class JitProgramBuilder {
  backend: Backend;
  #nextId: number;
  steps: JitStep[];

  constructor(backend: Backend, nargs: number) {
    this.backend = backend;
    this.#nextId = nargs;
    this.steps = [];
  }

  pushLit(lit: Lit): JitId {
    const kernel = new Kernel(
      0,
      lit.aval.size,
      AluExp.const(lit.dtype, lit.value),
    );
    return this.pushKernel(kernel, []);
  }

  pushBuffer(size: number): JitId {
    const id = this.#nextId++;
    this.steps.push({
      type: "malloc",
      size,
      output: id,
    });
    return id;
  }

  pushKernel(kernel: Kernel, inputs: JitId[]): JitId {
    const id = this.pushBuffer(kernel.bytes);
    this.steps.push({
      type: "execute",
      source: kernel,
      inputs,
      outputs: [id],
    });
    return id;
  }

  pushRoutine(routine: Routine, inputs: JitId[], outputs: JitId[]): void {
    this.steps.push({
      type: "execute",
      source: routine,
      inputs,
      outputs,
    });
  }

  pushIncref(id: JitId): void {
    this.steps.push({
      type: "incref",
      input: id,
    });
  }

  insertFreeSteps(outputIds: JitId[]): void {
    // Only free malloc'd ids that are not used in the output.
    //
    // Intermediates are allowed to be freed independently, since they are
    // guaranteed to be unused elsewhere after the JitProgram is executed.
    // Meanwhile, inputs and consts are owned / freed elsewhere.
    const ids = this.steps
      .filter((s) => s.type === "malloc")
      .map((s) => s.output);
    for (const id of ids) {
      // Find the last usage of this id.
      if (outputIds.includes(id)) continue;
      const lastUsage = this.steps.findLastIndex(
        (s) =>
          (s.type === "execute" &&
            (s.outputs.includes(id) || s.inputs.includes(id))) ||
          (s.type === "malloc" && s.output === id),
      )!;
      this.steps.splice(lastUsage + 1, 0, {
        type: "free",
        input: id,
      });
    }
  }

  pushFree(id: JitId): void {
    // Should be paired with the output of pushKernel() when last used.
    this.steps.push({
      type: "free",
      input: id,
    });
  }
}

type JitValue =
  | { type: "imm"; arg: JitId } // Immediate
  | { type: "exp"; exp: AluExp; args: JitId[] } // Expression, lazily fused
  | { type: "red"; exp: AluExp; reduction: Reduction; args: JitId[] }; // Reduction + epilogue

const jitCompileCache = new Map<string, JitProgram>();

export function jitCompile(backend: Backend, jaxpr: Jaxpr): JitProgram {
  const cacheKey = backend.type + "," + FpHash.hash(jaxpr);

  const cached = jitCompileCache.get(cacheKey);
  if (cached) return cached;

  if (DEBUG >= 1) {
    console.info("=========== JIT Compile ===========\n" + jaxpr.toString());
  }

  jaxpr = jaxpr.flatten().simplify();
  const nargs = jaxpr.inBinders.length;
  const builder = new JitProgramBuilder(backend, nargs);

  const blackNodes = splitGraphDataflow(backend, jaxpr);

  // Initialize jaxpr inBinders.
  const ctx = new Map<Var, JitValue>();
  for (let i = 0; i < nargs; i++) {
    const v = jaxpr.inBinders[i];
    ctx.set(v, { type: "imm", arg: i }); // JitId i = input #i
  }

  // Now run each primitive through a set of rules, mirroring implRules.
  for (let i = 0; i < jaxpr.eqns.length; i++) {
    const eqn = jaxpr.eqns[i];

    // If this is a routine, construct and dispatch the routine.
    if (routinePrimitives.has(eqn.primitive)) {
      // The rest of the code collaborates to make sure that all inputs to a
      // routine are "imm" (black node, dispatched) and so is itself.
      const routine = new Routine(
        routinePrimitives.get(eqn.primitive)!,
        {
          inputShapes: eqn.inputs.map((x) => x.aval.shape),
          inputDtypes: eqn.inputs.map((x) => x.aval.dtype),
          outputShapes: eqn.outBinders.map((x) => x.aval.shape),
          outputDtypes: eqn.outBinders.map((x) => x.aval.dtype),
        },
        eqn.params as any,
      );
      const inputs: JitId[] = [];
      for (const input of eqn.inputs) {
        if (input instanceof Var) {
          const jv = ctx.get(input)!;
          if (jv.type !== "imm") {
            throw new Error(
              `jit: routine primitive ${eqn.primitive} input is not imm`,
            );
          }
          inputs.push(jv.arg);
        } else if (input instanceof Lit) {
          inputs.push(builder.pushLit(input));
        }
      }
      const outputs: JitId[] = [];
      for (const outVar of eqn.outBinders) {
        const outId = builder.pushBuffer(
          outVar.aval.size * byteWidth(outVar.aval.dtype),
        );
        outputs.push(outId);
        ctx.set(outVar, { type: "imm", arg: outId });
      }
      builder.pushRoutine(routine, inputs, outputs);
      continue;
    }

    // Transform each input into an AluExp to start, and normalize any arguments
    // as needed.
    const inputExps: AluExp[] = []; // len(inputs)
    const inputAvals: ShapedArray[] = []; // len(inputs)
    const inputArgs: JitId[] = [];

    let inputReduction: (JitValue & { type: "red" }) | null = null;

    // May need to reindex gids to match order, returns array of new gids.
    const addArgs = (args: JitId[]): number[] => {
      const newGids: number[] = [];
      for (const jitId of args) {
        let newGid = inputArgs.indexOf(jitId);
        if (newGid === -1) {
          newGid = inputArgs.length;
          inputArgs.push(jitId);
        }
        newGids.push(newGid);
      }
      return newGids;
    };

    for (const input of eqn.inputs) {
      if (input instanceof Var) {
        const jv = ctx.get(input)!;
        if (jv.type === "exp") {
          const newGids = addArgs(jv.args);
          inputExps.push(jv.exp.reindexGids(newGids));
        } else if (jv.type === "imm") {
          const [gid] = addArgs([jv.arg]);
          const st = ShapeTracker.fromShape(input.aval.shape); // "imm" is realized
          const indices = unravelAlu(st.shape, AluVar.gidx);
          inputExps.push(AluExp.globalView(input.aval.dtype, gid, st, indices));
        } else if (jv.type === "red") {
          // Special case: We are consuming a 'red' JitValue, so we must be in the
          // fused epilogue of a reduction.
          if (inputReduction)
            throw new Error("jit: unexpected, multiple red inputs");
          const newGids = addArgs(jv.args);
          inputExps.push(jv.reduction.epilogue.reindexGids(newGids));
          inputReduction = jv;
        } else {
          jv satisfies never; // static check
        }
        inputAvals.push(input.aval);
      } else if (input instanceof Lit) {
        inputExps.push(AluExp.const(input.dtype, input.value));
        inputAvals.push(input.aval);
      } else {
        throw new TypeError(`Unexpected input in Jaxpr: ${input}`);
      }
    }

    // Produce a new expression and/or reduction for the operation based on the
    // jit() implementation of the primitive.
    const rule = jitRules[eqn.primitive];
    if (!rule)
      throw new TypeError(`JIT not implemented for primitive ${eqn.primitive}`);

    let exp: AluExp;
    let reduction: Reduction | undefined;

    if (inputReduction) {
      // Special case: we are in the fused epilogue of a reduction.
      const jv = inputReduction;
      const newEpilogue = rule(inputExps, inputAvals, eqn.params as any).exp;
      exp = jv.exp.reindexGids(addArgs(jv.args));
      reduction = new Reduction(
        jv.reduction.dtype,
        jv.reduction.op,
        jv.reduction.size,
        newEpilogue,
      );
    } else {
      const ruleOutput = rule(inputExps, inputAvals, eqn.params as any);
      exp = ruleOutput.exp;
      reduction = ruleOutput.reduction;
    }

    // Then dispatch the kernel, if it is a "black" node as determined from
    // dataflow analysis above.
    const outVar = eqn.outBinders[0];
    if (blackNodes.has(outVar)) {
      const nargs = inputArgs.length;
      const size = outVar.aval.size;
      const kernel = new Kernel(nargs, size, exp, reduction);
      const outId = builder.pushKernel(kernel, inputArgs);
      ctx.set(outVar, { type: "imm", arg: outId });
    } else if (reduction) {
      // Reduction but not black, means it will have an epilogue.
      ctx.set(outVar, {
        type: "red",
        exp: exp,
        reduction: reduction,
        args: inputArgs,
      });
    } else {
      // Otherwise, fuse the kernel into the next expression.
      ctx.set(outVar, { type: "exp", exp: exp, args: inputArgs });
    }
  }

  // Finally, loop through the outputs.
  const outputIds: JitId[] = [];
  for (const out of jaxpr.outs) {
    if (out instanceof Var) {
      const jitValue = ctx.get(out)!;
      if (jitValue.type !== "imm")
        throw new Error("internal: Expected imm, since outs are black nodes");
      outputIds.push(jitValue.arg);
    } else if (out instanceof Lit) {
      outputIds.push(builder.pushLit(out));
    } else {
      out satisfies never; // static check
    }
  }

  // Each output should have its own backend reference. If an output slot is
  // returned twice, or if an input is returned directly, insert "incref" steps
  // to balance the books.
  const outputNeedsRef = new Set<JitId>(range(nargs)); // inputs
  for (const outputId of outputIds) {
    if (outputNeedsRef.has(outputId)) {
      builder.pushIncref(outputId);
    } else {
      // If this output is seen again, increment its ref at that point.
      outputNeedsRef.add(outputId);
    }
  }

  // Emit free steps after last usage of any intermediates.
  builder.insertFreeSteps(outputIds);

  const jp = new JitProgram(backend, builder.steps, range(0, nargs), outputIds);
  if (DEBUG >= 4) console.info(jp.toString());
  jitCompileCache.set(cacheKey, jp);
  return jp;
}

/**
 * Rule for fusing the operation into a JIT expression to the backend.
 *
 * This takes in the expressions of the `src[]` inputs and produces a subsequent
 * expression, as well as optionally a reduction. The expressions use
 * AluVar.gidx (output index) and AluVar.ridx (reduction index).
 */
type JitRule<P extends Primitive> = (
  exps: AluExp[],
  avals: ShapedArray[],
  params: PrimitiveParams<P>,
) => {
  exp: AluExp;
  reduction?: Reduction;
};

function reshapeViews(
  exp: AluExp,
  mapping: (st: ShapeTracker) => ShapeTracker | undefined,
  reduceAxis: boolean = false,
): AluExp {
  return exp.rewrite((exp) => {
    if (exp.op === AluOp.GlobalView) {
      const [gid, st]: [number, ShapeTracker] = exp.arg;
      const newSt = mapping(st);
      if (newSt) {
        const indices = reduceAxis
          ? unravelAlu(newSt.shape.slice(0, -1), AluVar.gidx).concat(
              AluVar.ridx,
            )
          : unravelAlu(newSt.shape, AluVar.gidx);
        return AluExp.globalView(exp.dtype, gid, newSt, indices);
      }
    } else if (exp.op === AluOp.GlobalIndex) {
      throw new Error("internal: reshapeViews() called with GlobalIndex op");
    }
  });
}

// JIT handler for a broadcasted operation on at least 1 input.
function broadcastedJit<P extends Primitive>(
  fn: (exps: AluExp[], params: PrimitiveParams<P>) => AluExp,
  opts?: { skipCastIdx?: number[] },
): JitRule<P> {
  return (exps, avals, params) => {
    let { shape: newShape, dtype: newDtype } = avals.reduce(promoteAvals);

    const skipCastIdx = opts?.skipCastIdx ?? [];
    if (skipCastIdx.length) {
      // Skip casting some indices to a shared dtype.
      newDtype = avals
        .filter((_, i) => !skipCastIdx.includes(i))
        .reduce(promoteAvals).dtype;
    }

    // Perform a broadcast on each of the input expressions.
    //
    // Only GlobalView is affected. GlobalIndex is not used here, and neither is
    // AluVar.idx, since those are realized before jit().
    exps = exps.map((exp, i) => {
      exp = reshapeViews(exp, (st) => {
        if (!deepEqual(st.shape, newShape))
          return st.broadcast(
            newShape,
            range(newShape.length - st.shape.length),
          );
      });
      if (exp.dtype !== newDtype && !skipCastIdx.includes(i)) {
        exp = AluExp.cast(newDtype, exp);
      }
      return exp;
    });

    // Then, we can call the function to produce a new expression.
    return { exp: fn(exps, params) };
  };
}

// Simpler JIT handler, equivalent to broadcastedJit for unary ops.
function unopJit<P extends Primitive>(
  fn: (exp: AluExp, params: PrimitiveParams<P>) => AluExp,
): JitRule<P> {
  return ([a], [_as], params) => {
    return { exp: fn(a, params) };
  };
}

function reshapeJit<P extends Primitive>(
  fn: (st: ShapeTracker, params: PrimitiveParams<P>) => ShapeTracker,
): JitRule<P> {
  return ([a], [_as], params) => {
    return { exp: reshapeViews(a, (st) => fn(st, params)) };
  };
}

function routineNoJit<P extends Primitive>(): JitRule<P> {
  return () => {
    throw new Error("jit: rule is not implemented for routines");
  };
}

const jitRules: { [P in Primitive]: JitRule<P> } = {
  [Primitive.Add]: broadcastedJit(([a, b]) => AluExp.add(a, b)),
  [Primitive.Mul]: broadcastedJit(([a, b]) => AluExp.mul(a, b)),
  [Primitive.Idiv]: broadcastedJit(([a, b]) => AluExp.idiv(a, b)),
  [Primitive.Mod]: broadcastedJit(([a, b]) => AluExp.mod(a, b)),
  [Primitive.Min]: broadcastedJit(([a, b]) => AluExp.min(a, b)),
  [Primitive.Max]: broadcastedJit(([a, b]) => AluExp.max(a, b)),
  [Primitive.Neg]: unopJit((a) => AluExp.sub(AluExp.const(a.dtype, 0), a)),
  [Primitive.Reciprocal]: unopJit(AluExp.reciprocal),
  [Primitive.Floor]: unopJit(AluExp.floor),
  [Primitive.Ceil]: unopJit(AluExp.ceil),
  [Primitive.StopGradient]: unopJit((a) => a), // No-op, just return the input.
  [Primitive.Cast]: unopJit((a, { dtype }) => AluExp.cast(dtype, a)),
  [Primitive.Bitcast]: unopJit((a, { dtype }) => AluExp.bitcast(dtype, a)),
  [Primitive.Sin]: unopJit(AluExp.sin),
  [Primitive.Cos]: unopJit(AluExp.cos),
  [Primitive.Asin]: unopJit(AluExp.asin),
  [Primitive.Atan]: unopJit(AluExp.atan),
  [Primitive.Exp]: unopJit(AluExp.exp),
  [Primitive.Log]: unopJit(AluExp.log),
  [Primitive.Erf]: unopJit(AluExp.erf),
  [Primitive.Erfc]: unopJit(AluExp.erfc),
  [Primitive.Sqrt]: unopJit(AluExp.sqrt),
  [Primitive.Reduce]([a], [as], { op, axis }) {
    const keptAxes: number[] = [];
    const shiftedAxes: number[] = [];
    const newShape: number[] = [];
    for (let i = 0; i < as.shape.length; i++) {
      if (axis.includes(i)) shiftedAxes.push(i);
      else {
        keptAxes.push(i);
        newShape.push(as.shape[i]);
      }
    }
    const reductionSize = prod(shiftedAxes.map((ax) => as.shape[ax]));
    newShape.push(reductionSize);

    const perm = keptAxes.concat(shiftedAxes);
    a = reshapeViews(a, (st) => st.permute(perm).reshape(newShape), true);
    const reduction = new Reduction(a.dtype, op, reductionSize);
    return { exp: a, reduction };
  },
  [Primitive.Pool]: reshapeJit((st, { window, strides }) =>
    pool(st, window, strides),
  ),
  [Primitive.PoolTranspose]([a], [as], { inShape, window, strides }) {
    let stX = poolTranspose(
      ShapeTracker.fromShape(as.shape),
      inShape,
      window,
      strides,
    );
    stX = stX.reshape([...inShape, prod(stX.shape.slice(inShape.length))]); // Combine all reduce axes.
    a = reshapeViews(a, (st) => st.compose(stX), true);
    const reduction = new Reduction(
      a.dtype,
      AluOp.Add,
      stX.shape[stX.shape.length - 1],
    );
    return { exp: a, reduction };
  },
  [Primitive.Dot]([a, b], [as, bs]) {
    // Dot is just Mul->Reduce in sequence.
    const k1 = jitRules[Primitive.Mul]([a, b], [as, bs], {});
    const c = k1.exp;
    const cs = promoteAvals(as, bs);
    return jitRules[Primitive.Reduce]([c], [cs], {
      op: AluOp.Add,
      axis: [cs.ndim - 1],
    });
  },
  [Primitive.Conv]([a, b], [as, bs], params) {
    const [stX, stY] = prepareConv(
      ShapeTracker.fromShape(as.shape),
      ShapeTracker.fromShape(bs.shape),
      params,
    );
    a = reshapeViews(a, (st) => st.compose(stX));
    b = reshapeViews(b, (st) => st.compose(stY));
    as = new ShapedArray(stX.shape, as.dtype, as.weakType);
    bs = new ShapedArray(stY.shape, bs.dtype, bs.weakType);
    return jitRules[Primitive.Dot]([a, b], [as, bs], {});
  },
  [Primitive.Compare]: broadcastedJit(([a, b], { op }) => aluCompare(a, b, op)),
  [Primitive.Where]: broadcastedJit(
    ([cond, a, b]) => AluExp.where(cond, a, b),
    { skipCastIdx: [0] },
  ),
  [Primitive.Concatenate](exps, avals, { axis }) {
    throw new Error("TODO: JIT for Concatenate");
  },
  [Primitive.Split]([a], [as], { axis, sizes }) {
    throw new Error("TODO: JIT for Split");
  },
  [Primitive.RandomBits]: (keys, keyShapes, { shape, mode }) => {
    const mapping = (st: ShapeTracker): ShapeTracker | undefined => {
      if (!deepEqual(st.shape, shape))
        return st.broadcast(shape, range(shape.length - st.shape.length));
    };
    const k0 = reshapeViews(keys[0], mapping);
    const k1 = reshapeViews(keys[1], mapping);
    const c0 = AluExp.u32(0);
    const c1 = AluExp.cast(DType.Uint32, AluVar.gidx);
    const exp = AluExp.threefry2x32(k0, k1, c0, c1, mode);
    return { exp };
  },
  [Primitive.Gather](
    [x, ...indices],
    [xs, ...indicesShapes],
    { axis, outDim },
  ) {
    const axisSet = new Set(axis);

    // First, broadcast each integer array in `indices`.
    const indexShape = indicesShapes
      .map((c) => c.shape)
      .reduce(generalBroadcast);
    const finalShape = xs.shape.filter((_, i) => !axisSet.has(i));
    finalShape.splice(outDim, 0, ...indexShape);

    // Make variables for expression indices for gathered axes, and non-axis.
    const idxAll = unravelAlu(finalShape, AluVar.gidx);
    const idxNonaxis = [...idxAll];
    const _idxAxis = idxNonaxis.splice(outDim, indexShape.length);

    // Then, construct a kernel expression that gathers the data.
    const src: AluExp[] = [...idxNonaxis];
    for (let i = 0; i < xs.shape.length; i++) {
      // insert 'null' as axis placeholder, overwritten below as src[axis[i]].
      if (axisSet.has(i)) src.splice(i, 0, null as any);
    }

    for (const [i, iexp] of indices.entries()) {
      // Index iexp by the idxAxis variables, after broadcasting (via GlobalView).
      // [ ... | outDim | ... <iexp> | outDim + indexShape.length | ... ]
      src[axis[i]] = AluExp.cast(
        DType.Int32,
        reshapeViews(iexp, (st) =>
          st.broadcast(finalShape, [
            // Broadcast indices (aligned to the right), plus leading before outDim.
            ...range(outDim + indexShape.length - st.shape.length),
            // Indices to the right of outDim.
            ...range(outDim + indexShape.length, finalShape.length),
          ]),
        ),
      );
    }

    // Finally, index into "x" by replacing its gidx with a flat accessor into
    // the gathered indices.
    const [index, valid] = ShapeTracker.fromShape(xs.shape).toAluExp(src);
    if (!valid.resolve())
      throw new Error("internal: expected full validity mask in Gather");
    return { exp: x.substitute({ gidx: index }) };
  },
  [Primitive.Transpose]: reshapeJit((st, { perm }) => st.permute(perm)),
  [Primitive.Broadcast]: reshapeJit((st, { shape, axis }) =>
    st.broadcast(shape, axis),
  ),
  [Primitive.Reshape]: reshapeJit((st, { shape }) => st.reshape(shape)),
  [Primitive.Flip]: reshapeJit((st, { axis }) => {
    const arg = rep(st.shape.length, false);
    for (const ax of axis) arg[ax] = true;
    return st.flip(arg);
  }),
  [Primitive.Shrink]: reshapeJit((st, { slice }) => st.shrink(slice)),
  [Primitive.Pad]: reshapeJit((st, { width }) => st.pad(width)),
  [Primitive.Sort]: routineNoJit(),
  [Primitive.Argsort]: routineNoJit(),
  [Primitive.TriangularSolve]: routineNoJit(),
  [Primitive.Cholesky]: routineNoJit(),
  [Primitive.Jit]() {
    throw new Error(
      "internal: Jit should have been flattened before JIT compilation",
    );
  },
};

/** Determines how to split the Jaxpr into kernels via dataflow analysis. */
function splitGraphDataflow(backend: Backend, jaxpr: Jaxpr): Set<Var> {
  const varToDefn = new Map<Var, number>(); // Var -> eqn index of definition
  const varToUsages: Map<Var, number[]> = new Map(); // Var -> eqn indices of usages
  for (let i = 0; i < jaxpr.eqns.length; i++) {
    const eqn = jaxpr.eqns[i];
    for (const v of eqn.outBinders) {
      if (v instanceof Var) varToDefn.set(v, i);
    }
    for (const input of eqn.inputs) {
      if (input instanceof Var) {
        const usages = varToUsages.get(input);
        if (usages) usages.push(i);
        else varToUsages.set(input, [i]);
      }
    }
  }

  // Calculate reduction epilogues.
  //
  // A reduction can be fused with one or more operations that use its output,
  // which are either 1) unary or 2) binary ops with a literal, or an array not
  // larger than it.
  //
  // We also need to make sure we don't fuse two reductions together.
  const reducePrimitives = [
    Primitive.Reduce,
    Primitive.Dot,
    Primitive.Conv,
    Primitive.PoolTranspose,
  ];
  const reductionEpilogueEqns = new Set<number>();
  const reductionEndpointEqns = new Set<number>();
  for (let i = 0; i < jaxpr.eqns.length; i++) {
    const eqn = jaxpr.eqns[i];
    if (reducePrimitives.includes(eqn.primitive)) {
      let head = i;
      while (true) {
        reductionEpilogueEqns.add(head);

        // Try moving outVar forward through the graph.
        const outVar = jaxpr.eqns[head].outBinders[0];
        const usages = varToUsages.get(outVar) ?? [];
        if (jaxpr.outs.includes(outVar) || usages.length !== 1) break;

        // Next is already fused into a reduction epilogue, can't fuse again.
        if (reductionEpilogueEqns.has(usages[0])) break;

        const nextEqn = jaxpr.eqns[usages[0]];
        switch (nextEqn.primitive) {
          // We can always fuse unary operations.
          case Primitive.Neg:
          case Primitive.Reciprocal:
          case Primitive.Floor:
          case Primitive.Ceil:
          case Primitive.StopGradient:
          case Primitive.Cast:
          case Primitive.Bitcast:
          case Primitive.Sin:
          case Primitive.Cos:
          case Primitive.Asin:
          case Primitive.Atan:
          case Primitive.Exp:
          case Primitive.Log:
          case Primitive.Erf:
          case Primitive.Erfc:
          case Primitive.Sqrt:
            head = usages[0];
            continue;

          // We can fuse binary operations with a literal, or with an array such
          // that the array doesn't lead to broadcasting thus recomputation.
          case Primitive.Add:
          case Primitive.Mul:
          case Primitive.Idiv:
          case Primitive.Mod:
          case Primitive.Min:
          case Primitive.Max: {
            const otherInput = nextEqn.inputs.find((v) => v !== outVar)!;
            if (
              otherInput instanceof Lit ||
              deepEqual(
                generalBroadcast(otherInput.aval.shape, outVar.aval.shape),
                outVar.aval.shape,
              )
            ) {
              head = usages[0];
              continue;
            }
            break;
          }
        }
        break; // Can't move forward anymore.
      }
      reductionEndpointEqns.add(head);
    }
  }

  // Move backwards through the program and compute "black" endpoints.
  //
  // Black nodes are the endpoints where we dispatch a kernel to the backend
  // rather than producing intermediates. This includes:
  //
  // - Kernel outputs
  // - Reductions, except when fused with epilogue
  // - Gather/RandomBits operations (violates rule that kernels must have
  //   homogeneous GlobalView indices)
  // - Inputs to Pad operations, which need clean inputs
  //
  // Also, mark a node black if there are at least two black nodes that can be
  // reached from it, while only going through non-black nodes.
  //
  // TODO: Don't do the above for 'simple' nodes: reshape, cast, etc.
  const blackNodes = new Set<Var>();
  const p1NextBlack = new Map<Var, Var>();
  for (const v of jaxpr.outs) {
    if (v instanceof Var) {
      blackNodes.add(v);
      p1NextBlack.set(v, v);
    }
  }
  const heterogeneousViewPrimitives = [
    // These primitives generate heterogeneous GlobalView outputs, there are
    // multiple views in the expression with different indexing.
    Primitive.RandomBits,
    Primitive.Gather,
  ];
  const needsCleanShapePrimitives = [
    // If Pad is applied to a non-clean input, the reshaped padding would apply
    // to the view _inside_ of the expression. Imagine `GlobalView(...)+1`: if
    // you reshape each view, it adds zeros into the inner expression, so the
    // effect is to pad the intermediate with 1s instead of 0s!
    Primitive.Pad,
  ];
  for (let i = jaxpr.eqns.length - 1; i >= 0; i--) {
    const eqn = jaxpr.eqns[i];
    if (
      reductionEndpointEqns.has(i) ||
      heterogeneousViewPrimitives.includes(eqn.primitive) ||
      routinePrimitives.has(eqn.primitive) ||
      eqn.outBinders.some((v) => blackNodes.has(v))
    ) {
      for (const v of eqn.outBinders) {
        blackNodes.add(v);
        p1NextBlack.set(v, v);
      }
      continue;
    }
    const reach = new Set<Var>();
    let needsCleanOutput = false;
    outer: for (const v of eqn.outBinders) {
      for (const j of varToUsages.get(v) ?? []) {
        if (
          needsCleanShapePrimitives.includes(jaxpr.eqns[j].primitive) ||
          routinePrimitives.has(jaxpr.eqns[j].primitive)
        ) {
          needsCleanOutput = true;
          break outer;
        }
        for (const o of jaxpr.eqns[j].outBinders) {
          const u = p1NextBlack.get(o);
          if (u) reach.add(u);
        }
      }
    }
    if (reach.size > 1 || needsCleanOutput) {
      for (const v of eqn.outBinders) {
        blackNodes.add(v);
        p1NextBlack.set(v, v);
      }
    } else if (reach.size === 1) {
      const b = reach.values().next().value!;
      for (const v of eqn.outBinders) p1NextBlack.set(v, b);
    }
  }

  // Also, mark nodes black if the maximum number of arguments per kernel is
  // exceeded (i.e., maxComputeBuffersPerShaderStage for WebGPU). This needs to
  // be done in a second forward pass over the equations list.
  const p2Deps = new Map<Var, Set<Var>>(); // -> members are Var (black) or inBinders.
  for (const v of jaxpr.inBinders) {
    p2Deps.set(v, new Set([v])); // Each input is a dependency of itself.
  }
  let p2idx = 0;
  while (p2idx < jaxpr.eqns.length) {
    const eqn = jaxpr.eqns[p2idx++];
    const deps: Set<Var>[] = [];
    for (const input of eqn.inputs) {
      if (input instanceof Var) {
        if (blackNodes.has(input)) deps.push(new Set([input]));
        else deps.push(p2Deps.get(input)!);
      } else {
        deps.push(new Set());
      }
    }
    const depCounter = new Map<Var, number>(); // includes counts
    for (const depSet of deps) {
      for (const dep of depSet) {
        depCounter.set(dep, (depCounter.get(dep) ?? 0) + 1);
      }
    }
    if (depCounter.size > backend.maxArgs) {
      // We have too many dependencies, so we need to backtrack and mark one of
      // the inputs as black. By heuristic, we'll mark the one with the most
      // unique dependencies.
      let maxUniqueDeps = 0;
      let assocInput = -1;
      for (let i = 0; i < eqn.inputs.length; i++) {
        const input = eqn.inputs[i];
        if (input instanceof Var && varToDefn.has(input)) {
          let uniqueDeps = 0;
          for (const dep of deps[i]) {
            if (depCounter.get(dep) === 1) uniqueDeps++;
          }
          if (uniqueDeps > maxUniqueDeps) {
            maxUniqueDeps = uniqueDeps;
            assocInput = i;
          }
        }
      }
      if (assocInput === -1) {
        throw new Error(
          `internal: maxArgs, no input found to mark as black in Jaxpr equation ${eqn}`,
        );
      }
      const assocVar = eqn.inputs[assocInput] as Var;
      p2idx = varToDefn.get(assocVar)!; // backtrack to that equation
      for (const out of jaxpr.eqns[p2idx++].outBinders) {
        blackNodes.add(out);
      }
    } else {
      const s = new Set(depCounter.keys());
      for (const out of eqn.outBinders) p2Deps.set(out, s);
    }
  }

  return blackNodes;
}
