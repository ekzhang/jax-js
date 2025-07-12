import { AluOp } from "../alu";
import {
  JsTree,
  flatten as treeFlatten,
  unflatten as treeUnflatten,
} from "../tree";
import { unzip2, zip } from "../utils";
import { pureArray, zerosLike } from "./array";
import {
  AbstractValue,
  bind,
  broadcast,
  compare,
  cos,
  exp,
  flattenFun,
  flip,
  fullRaise,
  gather,
  idiv,
  less,
  log,
  max,
  min,
  neg,
  newMain,
  notEqual,
  pad,
  Primitive,
  PrimitiveParams,
  reciprocal,
  reduce,
  reshape,
  shrink,
  sin,
  Trace,
  Tracer,
  TracerValue,
  transpose,
  TreeMismatchError,
  where,
} from "./core";
import { Jaxpr, jaxprAsFun, makeJaxpr } from "./jaxpr";

class JVPTracer extends Tracer {
  constructor(
    trace: Trace,
    readonly primal: Tracer,
    readonly tangent: Tracer,
  ) {
    super(trace);
  }

  get aval(): AbstractValue {
    return this.primal.aval;
  }

  toString(): string {
    return `JVPTracer(${this.primal.toString()}, ${this.tangent.toString()})`;
  }

  get ref() {
    this.primal.ref, this.tangent.ref;
    return this;
  }
  dispose() {
    this.primal.dispose();
    this.tangent.dispose();
  }
}

class JVPTrace extends Trace {
  pure(val: TracerValue) {
    return this.lift(pureArray(val));
  }

  lift(val: Tracer): Tracer {
    return new JVPTracer(this, val, zerosLike(val));
  }

  processPrimitive<P extends Primitive>(
    primitive: P,
    tracers: JVPTracer[],
    params: PrimitiveParams<P>,
  ): JVPTracer[] {
    const [primalsIn, tangentsIn] = unzip2(
      tracers.map((x) => [x.primal, x.tangent]),
    );
    const jvpRule: JvpRule<P> | undefined = jvpRules[primitive];
    if (jvpRule === undefined) {
      throw new Error(`No JVP rule for: ${primitive}`);
    }
    const [primalsOut, tangentsOut] = jvpRule(primalsIn, tangentsIn, params);
    return zip(primalsOut, tangentsOut).map(
      ([x, t]) => new JVPTracer(this, x, t),
    );
  }
}

type JvpRule<P extends Primitive> = (
  primals: Tracer[],
  tangents: Tracer[],
  params: PrimitiveParams<P>,
) => [Tracer[], Tracer[]];

const jvpRules: { [P in Primitive]: JvpRule<P> } = {
  [Primitive.Add]([x, y], [dx, dy]) {
    return [[x.add(y)], [dx.add(dy)]];
  },
  [Primitive.Mul]([x, y], [dx, dy]) {
    return [[x.ref.mul(y.ref)], [x.mul(dy).add(dx.mul(y))]];
  },
  [Primitive.Idiv]([x, y], [dx, dy]) {
    dx.dispose(), dy.dispose();
    const z = idiv(x, y);
    const dz = zerosLike(z);
    return [[z], [dz]];
  },
  [Primitive.Neg]([x], [dx]) {
    return [[x.neg()], [dx.neg()]];
  },
  [Primitive.Reciprocal]([x], [dx]) {
    // d(1/x) = -x^-2 * dx
    const xRecip = reciprocal(x.ref);
    return [[xRecip.ref], [neg(xRecip.ref.mul(xRecip)).mul(dx)]];
  },
  [Primitive.StopGradient]([x], [dx]) {
    dx.dispose();
    const zeroGradient = zerosLike(x); // Stop gradient flows for x.
    return [[x], [zeroGradient]];
  },
  [Primitive.Sin]([x], [dx]) {
    return [[sin(x.ref)], [cos(x).mul(dx)]];
  },
  [Primitive.Cos]([x], [dx]) {
    return [[cos(x.ref)], [neg(sin(x)).mul(dx)]];
  },
  [Primitive.Exp]([x], [dx]) {
    // d(exp(x)) = exp(x) * dx
    const z = exp(x);
    return [[z.ref], [z.mul(dx)]];
  },
  [Primitive.Log]([x], [dx]) {
    // d(log(x)) = 1/x * dx
    return [[log(x.ref)], [reciprocal(x).mul(dx)]];
  },
  [Primitive.Min]([x, y], [dx, dy]) {
    return [[min(x.ref, y.ref)], [where(less(y, x), dy, dx)]];
  },
  [Primitive.Max]([x, y], [dx, dy]) {
    return [[max(x.ref, y.ref)], [where(less(x, y), dy, dx)]];
  },
  [Primitive.Reduce]([x], [dx], { op, axis }) {
    if (op === AluOp.Add) {
      return [[reduce(x, op, axis)], [reduce(dx, op, axis)]];
    } else if (op === AluOp.Mul) {
      // Multivariate product rule: (abc)'/abc = a'/a + b'/b + c'/c
      const primal = reduce(x.ref, op, axis);
      const tangent = broadcast(primal.ref, x.shape, axis)
        .mul(reciprocal(x))
        .mul(dx)
        .sum(axis);
      return [[primal], [tangent]];
    } else if (op === AluOp.Min || op === AluOp.Max) {
      const primal = reduce(x.ref, op, axis);
      // (min(x))' = average(where(x != min(x), inf, x'))
      //
      // We take average here to match the behavior of JAX. If there are
      // multiple minima, it's not well-defined which one to take as the tangent
      // vector (sharp discontinuity), so we average over all of them.
      const notMin = notEqual(x, broadcast(primal.ref, x.shape, axis));
      const minCount = where(notMin.ref, 0.0, 1.0).sum(axis);
      const tangent = where(notMin, op === AluOp.Min ? 0 : 0, dx)
        .sum(axis)
        .mul(reciprocal(minCount));
      return [[primal], [tangent]];
    } else {
      throw new Error(`JVP rule not implemented for reduce op: ${op}`);
    }
  },
  [Primitive.Compare]([x, y], tangents, { op }) {
    for (const t of tangents) t.dispose();
    const primal = compare(x, y, op);
    return [[primal], [zerosLike(primal)]];
  },
  [Primitive.Where]([cond, x, y], [dcond, dx, dy]) {
    dcond.dispose();
    return [[where(cond.ref, x, y)], [where(cond, dx, dy)]];
  },
  [Primitive.Transpose]([x], [dx], { perm }) {
    return [[transpose(x, perm)], [transpose(dx, perm)]];
  },
  [Primitive.Broadcast]([x], [dx], { shape, axis }) {
    return [[broadcast(x, shape, axis)], [broadcast(dx, shape, axis)]];
  },
  [Primitive.Reshape]([x], [dx], { shape }) {
    return [[reshape(x, shape)], [reshape(dx, shape)]];
  },
  [Primitive.Flip]([x], [dx], { axis }) {
    return [[flip(x, axis)], [flip(dx, axis)]];
  },
  [Primitive.Shrink]([x], [dx], { slice }) {
    return [[shrink(x, slice)], [shrink(dx, slice)]];
  },
  [Primitive.Pad]([x], [dx], { width }) {
    return [[pad(x, width)], [pad(dx, width)]];
  },
  [Primitive.Gather]([x, ...indices], [dx, ..._], { axis, outDim }) {
    // d(gather(x, indices)) = gather(dx, indices).
    // Note: We ignore the tangents for indices, since they are not differentiable.
    const indicesRef = indices.map((t) => t.ref);
    return [
      [gather(x, indices, axis, outDim)],
      [gather(dx, indicesRef, axis, outDim)],
    ];
  },
  [Primitive.JitCall](primals, tangents, { jaxpr }) {
    const { newJaxpr, newConsts } = jvpJaxpr(jaxpr);
    const outs = bind(
      Primitive.JitCall,
      [...newConsts.map((c) => c.ref), ...primals, ...tangents],
      {
        jaxpr: newJaxpr,
        numConsts: newConsts.length,
      },
    );
    const n = outs.length / 2;
    if (!Number.isInteger(n))
      throw new Error("internal: JVP Jaxpr output length is not even");
    const [primalsOut, tangentsOut] = [outs.slice(0, n), outs.slice(n)];
    return [primalsOut, tangentsOut];
  },
};

const jvpJaxprCache = new Map<Jaxpr, ReturnType<typeof jvpJaxpr>>();

function jvpJaxpr(jaxpr: Jaxpr): { newJaxpr: Jaxpr; newConsts: Tracer[] } {
  if (jvpJaxprCache.has(jaxpr)) {
    return jvpJaxprCache.get(jaxpr)!;
  }

  // Note: Following the implementation in Autodidax, consts in the Jaxpr become
  // real inputs after JVP transformation, since they are part of the primals
  // and the JVP rule takes in [primals, tangents] as a pair.
  //
  // This is also why we can ignore `numConsts` in the JVP rule. Anyway, this
  // only happens in jvp-of-jit cases, where you understandably have to
  // sacrifice some performance versus wrapping jit() outside.
  const inAvals = jaxpr.inBinders.map((v) => v.aval);
  const { jaxpr: newJaxpr, consts: newConsts } = makeJaxpr(
    (primals: Tracer[], tangents: Tracer[]) =>
      jvpFlat(jaxprAsFun(jaxpr), primals, tangents),
  )(inAvals, inAvals);
  const result = { newJaxpr, newConsts };

  jvpJaxprCache.set(jaxpr, result);
  return result;
}

function jvpFlat(
  f: (...x: Tracer[]) => TracerValue[],
  primals: TracerValue[],
  tangents: TracerValue[],
): [Tracer[], Tracer[]] {
  using main = newMain(JVPTrace);
  const trace = new JVPTrace(main);
  const tracersIn = zip(primals, tangents).map(
    ([x, t]) => new JVPTracer(trace, pureArray(x), pureArray(t)),
  );
  const outs = f(...tracersIn);
  const tracersOut = outs.map((out) => fullRaise(trace, out) as JVPTracer);
  return unzip2(tracersOut.map((t) => [t.primal, t.tangent]));
}

export function jvp<F extends (...x: any[]) => any>(
  f: F,
  primals: JsTree<TracerValue>[],
  tangents: JsTree<TracerValue>[],
): [ReturnType<F>, ReturnType<F>] {
  const [primalsFlat, inTree] = treeFlatten(primals);
  const [tangentsFlat, inTree2] = treeFlatten(tangents);
  if (!inTree.equals(inTree2)) {
    throw new TreeMismatchError("jvp", inTree, inTree2);
  }

  const [flatFun, outTree] = flattenFun(f, inTree);

  const [primalsOutFlat, tangentsOutFlat] = jvpFlat(
    flatFun,
    primalsFlat,
    tangentsFlat,
  );
  if (outTree.value === undefined) {
    throw new Error("outTree was not set in jvp");
  }
  const primalsOut = treeUnflatten(outTree.value, primalsOutFlat);
  const tangentsOut = treeUnflatten(outTree.value, tangentsOutFlat);
  return [primalsOut as any, tangentsOut as any];
}
