// Handle JitCall operations by translating Jaxprs into dispatched Kernels.

import { AluExp, AluOp, AluVar, Kernel, Reduction } from "../alu";
import { Backend, Slot } from "../backend";
import { ShapeTracker, unravelAlu } from "../shape";
import { DEBUG, deepEqual, FpHash, prod, range, rep } from "../utils";
import { aluCompare, Array, generalBroadcast, PendingExecute } from "./array";
import { Primitive, PrimitiveParams, ShapedArray } from "./core";
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
          const slot = this.backend.malloc(4 * step.size);
          scope.set(step.output, slot);
          break;
        }
        case "free": {
          const slot = scope.get(step.input)!;
          this.backend.decRef(slot);
          scope.delete(step.input);
          break;
        }
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
      size: kernel.size,
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

  insertFreeSteps(outputIds: JitId[]): void {
    // Only free malloc'd ids that are not used in the output.
    //
    // Intermediates are allowed to be freed independently, since they are
    // guaranteed to be unused elsewhere after the JitProgram is executed.
    // Meanwhile, inputs and consts need higher-level reference counting on the
    // Array objects, so we don't free them here. We update it at the end.
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
          const st = ShapeTracker.fromShape(input.aval.shape);
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

  // Emit free steps after last usage of any intermediates.
  builder.insertFreeSteps(outputIds);

  const jp = new JitProgram(backend, builder.steps, range(0, nargs), outputIds);
  jitCompileCache.set(cacheKey, jp);
  return jp;
}

type JitRule<P extends Primitive> = (
  nargs: number,
  exps: AluExp[],
  avals: ShapedArray[],
  params: PrimitiveParams<P>,
) => Kernel;

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
      exp.rewrite((exp) => {
        if (exp.op === AluOp.GlobalView) {
          let [gid, st]: [number, ShapeTracker] = exp.arg;
          if (!deepEqual(st.shape, newShape)) {
            st = st.broadcast(
              newShape,
              range(newShape.length - st.shape.length),
            );
            const indices = unravelAlu(st.shape, AluVar.gidx);
            return AluExp.globalView(exp.dtype, gid, st, indices);
          }
        }
      }),
    );

    // Then, we can call the function to produce a new kernel.
    const exp = fn(exps, params);
    return new Kernel(nargs, prod(newShape), exp);
  };
}

function reshapeJit<P extends Primitive>(
  fn: (st: ShapeTracker, params: PrimitiveParams<P>) => ShapeTracker,
): JitRule<P> {
  return (nargs, [a], [as], params) => {
    a = a.rewrite((exp) => {
      if (exp.op === AluOp.GlobalView) {
        const [gid, st]: [number, ShapeTracker] = exp.arg;
        const newSt = fn(st, params);
        const indices = unravelAlu(newSt.shape, AluVar.gidx);
        return AluExp.globalView(exp.dtype, gid, newSt, indices);
      }
    });
    return new Kernel(nargs, prod(as.shape), a);
  };
}

const jitRules: { [P in Primitive]: JitRule<P> } = {
  [Primitive.Add]: broadcastedJit(([a, b]) => AluExp.add(a, b)),
  [Primitive.Mul]: broadcastedJit(([a, b]) => AluExp.mul(a, b)),
  [Primitive.Idiv]: broadcastedJit(([a, b]) => AluExp.idiv(a, b)),
  [Primitive.Neg]: broadcastedJit(([a]) =>
    AluExp.sub(AluExp.const(a.dtype, 0), a),
  ),
  [Primitive.Reciprocal]: broadcastedJit(([a]) => AluExp.reciprocal(a)),
  [Primitive.StopGradient]: broadcastedJit(([a]) => a), // No-op, just return the input.
  [Primitive.Sin]: broadcastedJit(([a]) => AluExp.sin(a)),
  [Primitive.Cos]: broadcastedJit(([a]) => AluExp.cos(a)),
  [Primitive.Exp]: broadcastedJit(([a]) => AluExp.exp(a)),
  [Primitive.Log]: broadcastedJit(([a]) => AluExp.log(a)),
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

    a = a.rewrite((exp) => {
      if (exp.op === AluOp.GlobalView) {
        const [gid, st]: [number, ShapeTracker] = exp.arg;
        const newSt = st
          .permute(keptAxes.concat(shiftedAxes))
          .reshape(newShape);
        const indices = unravelAlu(newShape.slice(0, -1), AluVar.gidx);
        indices.push(AluVar.ridx);
        return AluExp.globalView(exp.dtype, gid, newSt, indices);
      }
    });

    const reduction = new Reduction(a.dtype, op, reductionSize);
    return new Kernel(nargs, size, a, reduction);
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
  [Primitive.Gather]() {
    // TODO: Implement Gather in JIT.
    throw new Error("Gather is not implemented in JIT yet");
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
  // Black nodes are the endpoints of a fused expression, where we dispatch a
  // kernel to the backend. The outputs are marked black, as well as any
  // reductions.
  //
  // Also, mark a node black if there are at least two black nodes that can be
  // reached from it, while only going through non-black nodes.
  const blackNodes = new Set<Var>();
  const p1NextBlack = new Map<Var, Var>();
  for (const v of jaxpr.outs) {
    if (v instanceof Var) {
      blackNodes.add(v);
      p1NextBlack.set(v, v);
    }
  }
  for (let i = jaxpr.eqns.length - 1; i >= 0; i--) {
    const eqn = jaxpr.eqns[i];
    if (
      eqn.primitive === Primitive.Reduce ||
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
