// Handle JitCall operations by translating Jaxprs into dispatched Kernels.

import { AluExp, AluOp, AluVar, DType, Kernel, Reduction } from "../alu";
import { Backend, Slot } from "../backend";
import { PPrint } from "../pprint";
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
import { aluCompare, Array, PendingExecute } from "./array";
import { pool, poolTranspose, prepareConv } from "./convolution";
import { Primitive, PrimitiveParams, promoteAvals, ShapedArray } from "./core";
import { Jaxpr, Lit, Var } from "./jaxpr";

export type JitId = number;

export type JitStep =
  | {
      type: "execute";
      kernel: Kernel;
      inputs: JitId[]; // mapped to backend Slot
      outputs: JitId[]; // mapped to backend Slot
    }
  | {
      type: "const";
      slot: Slot; // must avoid being GC'd for the lifetime of JitProgram
      output: JitId;
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
          return PPrint.pp(
            `execute (${inputsNice}) -> ${outputsNice}, kernel`,
          ).concat(step.kernel.pprint().indent(2));
        }
        case "const":
          return PPrint.pp(`%${step.output} = const <Slot ${step.slot}>`);
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
            new PendingExecute(this.backend, step.kernel, inputs, outputs),
          );
          break;
        }
        case "const":
          scope.set(step.output, step.slot);
          break;
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

  pushConst(slot: Slot): JitId {
    const id = this.#nextId++;
    this.steps.push({
      type: "const",
      slot,
      output: id,
    });
    return id;
  }

  pushLit(lit: Lit): JitId {
    const kernel = new Kernel(
      0,
      prod(lit.aval.shape),
      AluExp.const(lit.dtype, lit.value),
    );
    return this.pushKernel(kernel, []);
  }

  pushKernel(kernel: Kernel, inputs: JitId[]): JitId {
    const id = this.#nextId++;
    this.steps.push({
      type: "malloc",
      size: kernel.bytes,
      output: id,
    });
    this.steps.push({
      type: "execute",
      kernel,
      inputs,
      outputs: [id],
    });
    return id;
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
  | { type: "exp"; exp: AluExp; args: JitId[] }; // Expression, lazily fused

const jitCompileCache = new Map<string, JitProgram>();

export function jitCompile(
  backend: Backend,
  jaxpr: Jaxpr,
  consts: Array[],
): JitProgram {
  if (jaxpr.inBinders.length < consts.length) {
    throw new TypeError(
      `Jaxpr has ${jaxpr.inBinders.length} inputs, but ${consts.length} consts were provided`,
    );
  }
  for (let i = 0; i < consts.length; i++) {
    if (consts[i].device !== backend.type) {
      throw new TypeError(
        `Const ${i} has device ${consts[i].device}, but expected ${backend.type}`,
      );
    }
  }

  const cacheKey =
    backend.type + FpHash.hash(jaxpr, ...consts.map((c) => c.id));

  const cached = jitCompileCache.get(cacheKey);
  if (cached) return cached;

  if (DEBUG >= 1) {
    console.info("=========== JIT Compile ===========\n" + jaxpr.toString());
  }

  jaxpr = jaxpr.flatten().simplify();
  const nargs = jaxpr.inBinders.length - consts.length;
  const builder = new JitProgramBuilder(backend, nargs);

  const blackNodes = splitGraphDataflow(backend, jaxpr);

  // Initialize jaxpr inBinders.
  const ctx = new Map<Var, JitValue>();
  for (let i = 0; i < consts.length; i++) {
    const v = jaxpr.inBinders[i];
    const slot = consts[i]._realizeSource();
    ctx.set(v, { type: "imm", arg: builder.pushConst(slot) });
  }
  for (let i = 0; i < nargs; i++) {
    const v = jaxpr.inBinders[consts.length + i];
    ctx.set(v, { type: "imm", arg: i }); // JitId i = input #i
  }

  // Now run each primitive through a set of rules, mirroring implRules.
  for (let i = 0; i < jaxpr.eqns.length; i++) {
    const eqn = jaxpr.eqns[i];

    // Transform each input into an AluExp to start, and normalize any arguments
    // as needed.
    const inputExps: AluExp[] = []; // len(inputs)
    const inputAvals: ShapedArray[] = []; // len(inputs)
    const inputArgs: JitId[] = [];
    for (const input of eqn.inputs) {
      if (input instanceof Var) {
        const jitValue = ctx.get(input)!;
        if (jitValue.type === "exp") {
          // May need to reorder args, tracked by this map.
          const gidMap = new Map<number, number>();
          for (const [gid, jitId] of jitValue.args.entries()) {
            let newGid = inputArgs.indexOf(jitId);
            if (newGid === -1) {
              newGid = inputArgs.length;
              inputArgs.push(jitId);
            }
            gidMap.set(gid, newGid);
          }
          inputExps.push(jitValue.exp.reindexGids(gidMap));
        } else if (jitValue.type === "imm") {
          let gid = inputArgs.indexOf(jitValue.arg);
          if (gid === -1) {
            gid = inputArgs.length;
            inputArgs.push(jitValue.arg);
          }
          const st = ShapeTracker.fromShape(input.aval.shape); // "imm" is realized
          const indices = unravelAlu(st.shape, AluVar.gidx);
          inputExps.push(AluExp.globalView(input.aval.dtype, gid, st, indices));
        } else {
          jitValue satisfies never; // static check
        }
        inputAvals.push(input.aval);
      } else if (input instanceof Lit) {
        inputExps.push(AluExp.const(input.dtype, input.value));
        inputAvals.push(input.aval);
      } else {
        throw new TypeError(`Unexpected input in Jaxpr: ${input}`);
      }
    }

    // Produce a new kernel for the operation based on the jit() implementation
    // of the primitive. This kernel may not be actually dispatched.
    const nargs = inputArgs.length;
    const rule = jitRules[eqn.primitive];
    if (!rule)
      throw new TypeError(`JIT not implemented for primitive ${eqn.primitive}`);
    const kernel = rule(nargs, inputExps, inputAvals, eqn.params as any);

    // Then dispatch the kernel, if it is a "black" node as determined from
    // dataflow analysis above.
    const outVar = eqn.outBinders[0];
    if (kernel.reduction || blackNodes.has(outVar)) {
      const outId = builder.pushKernel(kernel, inputArgs);
      ctx.set(outVar, { type: "imm", arg: outId });
    } else {
      // Otherwise, fuse the kernel into the next expression.
      ctx.set(outVar, { type: "exp", exp: kernel.exp, args: inputArgs });
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
  // returned twice, or if an input/const is returned directly, insert "incref"
  // steps to balance the books.
  const outputNeedsRef = new Set<JitId>([
    ...range(nargs), // inputs
    ...builder.steps.filter((s) => s.type === "const").map((s) => s.output),
  ]);
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
 * This takes in the expressions of the `src[]` inputs and produces a Kernel
 * object with a new expression, as well as a size and reduction. The expression
 * uses AluVar.gidx (output index) and AluVar.ridx (reduction index).
 *
 * Some ops trigger a dispatch, others can produce intermediates if:
 *
 * - No GlobalIndex expressions are present, and all GlobalView expressions take
 *   a plain AluVar.gidx unravelled as indices.
 * - No reductions are present. (may be changed to support epilogue)
 */
type JitRule<P extends Primitive> = (
  nargs: number,
  exps: AluExp[],
  avals: ShapedArray[],
  params: PrimitiveParams<P>,
) => Kernel;

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
): JitRule<P> {
  return (nargs, exps, avals, params) => {
    const newShape = avals.map((aval) => aval.shape).reduce(generalBroadcast);

    // Perform a broadcast on each of the input expressions.
    //
    // Only GlobalView is affected. GlobalIndex is not used here, and neither is
    // AluVar.idx, since those are realized before jit().
    exps = exps.map((exp) =>
      reshapeViews(exp, (st) => {
        if (!deepEqual(st.shape, newShape))
          return st.broadcast(
            newShape,
            range(newShape.length - st.shape.length),
          );
      }),
    );

    // Then, we can call the function to produce a new kernel.
    const exp = fn(exps, params);
    return new Kernel(nargs, prod(newShape), exp);
  };
}

// Simpler JIT handler, equivalent to broadcastedJit for unary ops.
function unopJit<P extends Primitive>(
  fn: (exp: AluExp, params: PrimitiveParams<P>) => AluExp,
): JitRule<P> {
  return (nargs, [a], [as], params) => {
    return new Kernel(nargs, prod(as.shape), fn(a, params));
  };
}

function reshapeJit<P extends Primitive>(
  fn: (st: ShapeTracker, params: PrimitiveParams<P>) => ShapeTracker,
): JitRule<P> {
  return (nargs, [a], [as], params) => {
    a = reshapeViews(a, (st) => fn(st, params));
    const newShape = fn(ShapeTracker.fromShape(as.shape), params).shape;
    return new Kernel(nargs, prod(newShape), a);
  };
}

const jitRules: { [P in Primitive]: JitRule<P> } = {
  [Primitive.Add]: broadcastedJit(([a, b]) => AluExp.add(a, b)),
  [Primitive.Mul]: broadcastedJit(([a, b]) => AluExp.mul(a, b)),
  [Primitive.Idiv]: broadcastedJit(([a, b]) => AluExp.idiv(a, b)),
  [Primitive.Neg]: unopJit((a) => AluExp.sub(AluExp.const(a.dtype, 0), a)),
  [Primitive.Reciprocal]: unopJit(AluExp.reciprocal),
  [Primitive.StopGradient]: unopJit((a) => a), // No-op, just return the input.
  [Primitive.Cast]: unopJit((a, { dtype }) => AluExp.cast(dtype, a)),
  [Primitive.Bitcast]: unopJit((a, { dtype }) => AluExp.bitcast(dtype, a)),
  [Primitive.RandomBits]: (nargs, keys, keyShapes, { shape, mode }) => {
    const mapping = (st: ShapeTracker) => {
      if (!deepEqual(st.shape, shape))
        return st.broadcast(shape, range(shape.length - st.shape.length));
    };
    const k0 = reshapeViews(keys[0], mapping);
    const k1 = reshapeViews(keys[1], mapping);
    const c0 = AluExp.u32(0);
    const c1 = AluExp.cast(DType.Uint32, AluVar.gidx);
    const exp = AluExp.threefry2x32(k0, k1, c0, c1, mode);
    return new Kernel(nargs, prod(shape), exp);
  },
  [Primitive.Sin]: unopJit(AluExp.sin),
  [Primitive.Cos]: unopJit(AluExp.cos),
  [Primitive.Asin]: unopJit(AluExp.asin),
  [Primitive.Atan]: unopJit(AluExp.atan),
  [Primitive.Exp]: unopJit(AluExp.exp),
  [Primitive.Log]: unopJit(AluExp.log),
  [Primitive.Sqrt]: unopJit(AluExp.sqrt),
  [Primitive.Min]: broadcastedJit(([a, b]) => AluExp.min(a, b)),
  [Primitive.Max]: broadcastedJit(([a, b]) => AluExp.max(a, b)),
  [Primitive.Reduce](nargs, [a], [as], { op, axis }) {
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
    const size = prod(newShape);
    const reductionSize = prod(shiftedAxes.map((ax) => as.shape[ax]));
    newShape.push(reductionSize);

    const perm = keptAxes.concat(shiftedAxes);
    a = reshapeViews(a, (st) => st.permute(perm).reshape(newShape), true);
    const reduction = new Reduction(a.dtype, op, reductionSize);
    return new Kernel(nargs, size, a, reduction);
  },
  [Primitive.Pool]: reshapeJit((st, { window, strides }) =>
    pool(st, window, strides),
  ),
  [Primitive.PoolTranspose](nargs, [a], [as], { inShape, window, strides }) {
    let stX = poolTranspose(
      ShapeTracker.fromShape(as.shape),
      inShape,
      window,
      strides,
    );
    const size = prod(inShape);
    stX = stX.reshape([...inShape, prod(stX.shape.slice(inShape.length))]); // Combine all reduce axes.
    a = reshapeViews(a, (st) => st.compose(stX), true);
    const reduction = new Reduction(
      a.dtype,
      AluOp.Add,
      stX.shape[stX.shape.length - 1],
    );
    return new Kernel(nargs, size, a, reduction);
  },
  [Primitive.Dot](nargs, [a, b], [as, bs]) {
    // Dot is just Mul->Reduce in sequence.
    const k1 = jitRules[Primitive.Mul](nargs, [a, b], [as, bs], {});
    const c = k1.exp;
    const cs = promoteAvals(as, bs);
    return jitRules[Primitive.Reduce](nargs, [c], [cs], {
      op: AluOp.Add,
      axis: [cs.ndim - 1],
    });
  },
  [Primitive.Conv](nargs, [a, b], [as, bs], params) {
    const [stX, stY] = prepareConv(
      ShapeTracker.fromShape(as.shape),
      ShapeTracker.fromShape(bs.shape),
      params,
    );
    a = reshapeViews(a, (st) => st.compose(stX));
    b = reshapeViews(b, (st) => st.compose(stY));
    as = new ShapedArray(stX.shape, as.dtype, as.weakType);
    bs = new ShapedArray(stY.shape, bs.dtype, bs.weakType);
    return jitRules[Primitive.Dot](nargs, [a, b], [as, bs], {});
  },
  [Primitive.Compare]: broadcastedJit(([a, b], { op }) => aluCompare(a, b, op)),
  [Primitive.Where]: broadcastedJit(([cond, a, b]) => AluExp.where(cond, a, b)),
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
  [Primitive.Gather](
    nargs,
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
    return new Kernel(nargs, prod(finalShape), x.substitute({ gidx: index }));
  },
  [Primitive.JitCall]() {
    throw new Error(
      "internal: JitCall should have been flattened before JIT compilation",
    );
  },
};

/** Determines how to split the Jaxpr into kernels via dataflow analysis. */
function splitGraphDataflow(backend: Backend, jaxpr: Jaxpr): Set<Var> {
  // Calculate the equation where each intermediate variable was defined.
  const varToEqn = new Map<Var, number>();
  for (let i = 0; i < jaxpr.eqns.length; i++) {
    const eqn = jaxpr.eqns[i];
    for (const v of eqn.outBinders) {
      if (v instanceof Var) varToEqn.set(v, i);
    }
  }

  // Move backwards through the program and compute "black" endpoints.
  //
  // Black nodes are the endpoints where we dispatch a kernel to the backend
  // rather than producing intermediates. This includes:
  //
  // - Kernel outputs
  // - Reductions (intermediates cannot have reductions)
  // - Gather/RandomBits operations (violates rule that kernels must have
  //   homogeneous GlobalView indices)
  //
  // Also, mark a node black if there are at least two black nodes that can be
  // reached from it, while only going through non-black nodes.
  //
  // TODO: Don't do the above for 'simple' nodes: reshape, cast, etc.
  //
  // TODO: Reductions can have epilogues.
  const blackNodes = new Set<Var>();
  const p1NextBlack = new Map<Var, Var>();
  for (const v of jaxpr.outs) {
    if (v instanceof Var) {
      blackNodes.add(v);
      p1NextBlack.set(v, v);
    }
  }
  const reducePrimitives = [
    Primitive.Reduce,
    Primitive.Dot,
    Primitive.Conv,
    Primitive.PoolTranspose,
  ];
  const heterogeneousViewPrimitives = [Primitive.Gather, Primitive.RandomBits];
  for (let i = jaxpr.eqns.length - 1; i >= 0; i--) {
    const eqn = jaxpr.eqns[i];
    if (
      reducePrimitives.includes(eqn.primitive) ||
      heterogeneousViewPrimitives.includes(eqn.primitive) ||
      eqn.outBinders.some((v) => blackNodes.has(v))
    ) {
      for (const v of eqn.outBinders) {
        blackNodes.add(v);
        p1NextBlack.set(v, v);
      }
      continue;
    }
    const reach = new Set<Var>();
    for (let j = i + 1; j < jaxpr.eqns.length; j++) {
      for (const v of jaxpr.eqns[j].inputs) {
        if (v instanceof Var && eqn.outBinders.includes(v)) {
          for (const o of jaxpr.eqns[j].outBinders) {
            const u = p1NextBlack.get(o);
            if (u) reach.add(u);
          }
        }
      }
    }
    if (reach.size === 1) {
      const b = reach.values().next().value!;
      for (const v of eqn.outBinders) p1NextBlack.set(v, b);
    } else if (reach.size > 1) {
      for (const v of eqn.outBinders) {
        blackNodes.add(v);
        p1NextBlack.set(v, v);
      }
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
    if (eqn.outBinders.some((v) => blackNodes.has(v))) {
      continue; // Already black, no need to check inputs.
    }
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
        if (input instanceof Var && varToEqn.has(input)) {
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
      p2idx = varToEqn.get(assocVar)!; // backtrack to that equation
      for (const out of jaxpr.eqns[p2idx].outBinders) {
        blackNodes.add(out);
      }
    } else {
      const s = new Set(depCounter.keys());
      for (const out of eqn.outBinders) p2Deps.set(out, s);
    }
  }

  return blackNodes;
}
