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
  CompareOp,
  cos,
  flattenFun,
  flip,
  fullRaise,
  neg,
  newMain,
  Primitive,
  reduceSum,
  reshape,
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

  processPrimitive(
    primitive: Primitive,
    tracers: JVPTracer[],
    params: Record<string, any>,
  ): JVPTracer[] {
    const [primalsIn, tangentsIn] = unzip2(
      tracers.map((x) => [x.primal, x.tangent]),
    );
    const jvpRule: JvpRule | undefined = jvpRules[primitive];
    if (jvpRule === undefined) {
      throw new Error(`No JVP rule for: ${primitive}`);
    }
    const [primalsOut, tangentsOut] = jvpRule(primalsIn, tangentsIn, params);
    return zip(primalsOut, tangentsOut).map(
      ([x, t]) => new JVPTracer(this, x, t),
    );
  }
}

type JvpRule = (
  primals: Tracer[],
  tangents: Tracer[],
  params: any,
) => [Tracer[], Tracer[]];

const jvpRules: Record<Primitive, JvpRule> = {
  [Primitive.Add]([x, y], [dx, dy]) {
    return [[x.add(y)], [dx.add(dy)]];
  },
  [Primitive.Mul]([x, y], [dx, dy]) {
    return [[x.ref.mul(y.ref)], [x.mul(dy).add(dx.mul(y))]];
  },
  [Primitive.Neg]([x], [dx]) {
    return [[x.neg()], [dx.neg()]];
  },
  [Primitive.Sin]([x], [dx]) {
    return [[sin(x.ref)], [cos(x).mul(dx)]];
  },
  [Primitive.Cos]([x], [dx]) {
    return [[cos(x.ref)], [neg(sin(x)).mul(dx)]];
  },
  [Primitive.ReduceSum]([x], [dx], { axis }: { axis: number[] }) {
    return [[reduceSum(x, axis)], [reduceSum(dx, axis)]];
  },
  [Primitive.Compare]([x, y], tangents, { op }: { op: CompareOp }) {
    for (const t of tangents) t.dispose();
    const primal = compare(x, y, op);
    return [[primal], [zerosLike(primal)]];
  },
  [Primitive.Where]([cond, x, y], [dcond, dx, dy]) {
    dcond.dispose();
    return [[where(cond.ref, x, y)], [where(cond, dx, dy)]];
  },
  [Primitive.Transpose]([x], [dx], { perm }: { perm: number[] }) {
    return [[transpose(x, perm)], [transpose(dx, perm)]];
  },
  [Primitive.Broadcast](
    [x],
    [dx],
    { shape, axis }: { shape: number[]; axis: number[] },
  ) {
    return [[broadcast(x, shape, axis)], [broadcast(dx, shape, axis)]];
  },
  [Primitive.Reshape]([x], [dx], { shape }: { shape: number[] }) {
    return [[reshape(x, shape)], [reshape(dx, shape)]];
  },
  [Primitive.Flip]([x], [dx], { axis }: { axis: number[] }) {
    return [[flip(x, axis)], [flip(dx, axis)]];
  },
  [Primitive.JitCall](primals, tangents, { jaxpr }: { jaxpr: Jaxpr }) {
    const { newJaxpr, newConsts } = jvpJaxpr(jaxpr);
    const outs = bind(
      Primitive.JitCall,
      [...newConsts, ...primals, ...tangents],
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
