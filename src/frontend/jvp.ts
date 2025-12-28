import { AluOp, isFloatDtype } from "../alu";
import {
  JsTree,
  flatten as treeFlatten,
  unflatten as treeUnflatten,
} from "../tree";
import { unzip2, zip } from "../utils";
import { pureArray, zerosLike } from "./array";
import {
  AbstractValue,
  asin,
  atan,
  bind,
  bind1,
  bitcast,
  broadcast,
  cast,
  cos,
  erf,
  erfc,
  exp,
  flattenFun,
  fullRaise,
  gather,
  idiv,
  less,
  log,
  max,
  min,
  mod,
  neg,
  newMain,
  notEqual,
  Primitive,
  PrimitiveParams,
  reciprocal,
  reduce,
  sin,
  sqrt,
  Trace,
  Tracer,
  TracerValue,
  TreeMismatchError,
  where,
} from "./core";
import { ClosedJaxpr, Jaxpr, jaxprAsFun, makeJaxpr } from "./jaxpr";

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
    (this.primal.ref, this.tangent.ref);
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
    return new JVPTracer(this, val, zerosLike(val.ref));
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

/** Rule that applies the same operation to primals and tangents. */
function linearTangentsJvp<P extends Primitive>(primitive: P): JvpRule<P> {
  return (primals, tangents, params) => {
    const ys = bind(primitive, primals, params);
    const dys = bind(primitive, tangents, params);
    return [ys, dys];
  };
}

/** Rule for product of gradients in bilinear operations. */
function bilinearTangentsJvp<P extends Primitive>(primitive: P): JvpRule<P> {
  return ([x, y], [dx, dy], params) => {
    const primal = bind1(primitive, [x.ref, y.ref], params);
    const tangent = bind1(primitive, [x, dy], params).add(
      bind1(primitive, [dx, y], params),
    ); // (xy)' = xy' + x'y
    return [[primal], [tangent]];
  };
}

/** Rule that zeros out any tangents. */
function zeroTangentsJvp<P extends Primitive>(primitive: P): JvpRule<P> {
  return (primals, tangents, params) => {
    for (const t of tangents) t.dispose();
    const ys = bind(primitive, primals, params);
    return [ys, ys.map((y) => zerosLike(y.ref))];
  };
}

const jvpRules: { [P in Primitive]: JvpRule<P> } = {
  [Primitive.Add]: linearTangentsJvp(Primitive.Add),
  [Primitive.Mul]: bilinearTangentsJvp(Primitive.Mul),
  [Primitive.Idiv]: zeroTangentsJvp(Primitive.Idiv),
  [Primitive.Mod]([x, y], [dx, dy]) {
    // x % y = x - y * trunc(x / y)
    // d(x % y) = dx - dy * trunc(x / y)
    if (!isFloatDtype(x.dtype) && !isFloatDtype(y.dtype)) {
      dx.dispose();
      dy.dispose();
      return [
        [x.ref, y.ref],
        [zerosLike(x), zerosLike(y)],
      ];
    }
    const q = idiv(x.ref, y.ref);
    return [[mod(x, y)], [dx.sub(dy.mul(q))]];
  },
  [Primitive.Min]([x, y], [dx, dy]) {
    return [[min(x.ref, y.ref)], [where(less(y, x), dy, dx)]];
  },
  [Primitive.Max]([x, y], [dx, dy]) {
    return [[max(x.ref, y.ref)], [where(less(x, y), dy, dx)]];
  },
  [Primitive.Neg]: linearTangentsJvp(Primitive.Neg),
  [Primitive.Reciprocal]([x], [dx]) {
    // d(1/x) = -x^-2 * dx
    const xRecip = reciprocal(x.ref);
    return [[xRecip.ref], [neg(xRecip.ref.mul(xRecip)).mul(dx)]];
  },
  [Primitive.Floor]: zeroTangentsJvp(Primitive.Floor),
  [Primitive.Ceil]: zeroTangentsJvp(Primitive.Ceil),
  [Primitive.StopGradient]: zeroTangentsJvp(Primitive.StopGradient),
  [Primitive.Cast]([x], [dx], { dtype }) {
    if (x.dtype === dtype) return [[x], [dx]]; // No-op if dtype is the same.
    // If floating-point, cast to the new dtype. Otherwise discard the tangent.
    if (isFloatDtype(dtype) && isFloatDtype(x.dtype)) {
      return [[cast(x, dtype)], [cast(dx, dtype)]];
    } else {
      dx.dispose();
      return [[cast(x.ref, dtype)], [zerosLike(x)]];
    }
  },
  [Primitive.Bitcast]([x], [dx], { dtype }) {
    if (x.dtype === dtype) return [[x], [dx]]; // No-op if dtype is the same.
    dx.dispose(); // Non-differentiable operation.
    return [[bitcast(x.ref, dtype)], [zerosLike(x)]];
  },
  [Primitive.Sin]([x], [dx]) {
    return [[sin(x.ref)], [cos(x).mul(dx)]];
  },
  [Primitive.Cos]([x], [dx]) {
    return [[cos(x.ref)], [neg(sin(x)).mul(dx)]];
  },
  [Primitive.Asin]([x], [dx]) {
    // d(asin(x)) = 1/sqrt(1 - x^2) * dx
    const denom = sqrt(reciprocal(cast(1, x.dtype).sub(x.ref.mul(x.ref))));
    return [[asin(x)], [denom.mul(dx)]];
  },
  [Primitive.Atan]([x], [dx]) {
    // d(atan(x)) = 1/(1 + x^2) * dx
    const denom = cast(1, x.dtype).add(x.ref.mul(x.ref));
    return [[atan(x)], [dx.div(denom)]];
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
  [Primitive.Erf]([x], [dx]) {
    // d(erf(x)) = 2/sqrt(pi) * exp(-x^2) * dx
    const coeff = 2 / Math.sqrt(Math.PI);
    const expTerm = exp(neg(x.ref.mul(x.ref)));
    return [[erf(x)], [expTerm.mul(coeff).mul(dx)]];
  },
  [Primitive.Erfc]([x], [dx]) {
    // d(erfc(x)) = -2/sqrt(pi) * exp(-x^2) * dx
    const coeff = -2 / Math.sqrt(Math.PI);
    const expTerm = exp(neg(x.ref.mul(x.ref)));
    return [[erfc(x)], [expTerm.mul(coeff).mul(dx)]];
  },
  [Primitive.Sqrt]([x], [dx]) {
    // d(sqrt(x)) = 1/(2*sqrt(x)) * dx
    const z = sqrt(x);
    return [[z.ref], [reciprocal(z.mul(2)).mul(dx)]];
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
      const tangent = where(notMin, 0.0, dx).sum(axis).div(minCount);
      return [[primal], [tangent]];
    } else {
      throw new Error(`JVP rule not implemented for reduce op: ${op}`);
    }
  },
  [Primitive.Pool]: linearTangentsJvp(Primitive.Pool),
  [Primitive.PoolTranspose]: linearTangentsJvp(Primitive.PoolTranspose),
  [Primitive.Dot]: bilinearTangentsJvp(Primitive.Dot),
  [Primitive.Conv]: bilinearTangentsJvp(Primitive.Conv),
  [Primitive.Compare]: zeroTangentsJvp(Primitive.Compare),
  [Primitive.Where]([cond, x, y], [dcond, dx, dy]) {
    dcond.dispose();
    return [[where(cond.ref, x, y)], [where(cond, dx, dy)]];
  },
  [Primitive.RandomBits]: zeroTangentsJvp(Primitive.RandomBits),
  [Primitive.Gather]([x, ...indices], [dx, ..._], { axis, outDim }) {
    // d(gather(x, indices)) = gather(dx, indices).
    // Note: We ignore the tangents for indices, since they are not differentiable.
    const indicesRef = indices.map((t) => t.ref);
    return [
      [gather(x, indices, axis, outDim)],
      [gather(dx, indicesRef, axis, outDim)],
    ];
  },
  [Primitive.Transpose]: linearTangentsJvp(Primitive.Transpose),
  [Primitive.Broadcast]: linearTangentsJvp(Primitive.Broadcast),
  [Primitive.Reshape]: linearTangentsJvp(Primitive.Reshape),
  [Primitive.Flip]: linearTangentsJvp(Primitive.Flip),
  [Primitive.Shrink]: linearTangentsJvp(Primitive.Shrink),
  [Primitive.Pad]: linearTangentsJvp(Primitive.Pad),
  [Primitive.Jit](primals, tangents, { name, jaxpr }) {
    const newJaxpr = jvpJaxpr(jaxpr);
    const outs = bind(
      Primitive.Jit,
      [...newJaxpr.consts.map((c) => c.ref), ...primals, ...tangents],
      {
        name: `${name}_jvp`,
        jaxpr: newJaxpr.jaxpr,
        numConsts: newJaxpr.consts.length,
      },
    );
    const n = outs.length / 2;
    if (!Number.isInteger(n))
      throw new Error("internal: JVP Jaxpr output length is not even");
    const [primalsOut, tangentsOut] = [outs.slice(0, n), outs.slice(n)];
    return [primalsOut, tangentsOut];
  },
};

const jvpJaxprCache = new Map<Jaxpr, ClosedJaxpr>();

function jvpJaxpr(jaxpr: Jaxpr): ClosedJaxpr {
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
  const { jaxpr: newJaxpr } = makeJaxpr(
    (primals: Tracer[], tangents: Tracer[]) =>
      jvpFlat(jaxprAsFun(jaxpr), primals, tangents),
  )(inAvals, inAvals);

  jvpJaxprCache.set(jaxpr, newJaxpr);
  return newJaxpr;
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
