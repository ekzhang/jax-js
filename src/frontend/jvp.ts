import { AluOp, isFloatDtype } from "../alu";
import { Pair } from "../shape";
import {
  JsTree,
  flatten as treeFlatten,
  map as treeMap,
  unflatten as treeUnflatten,
} from "../tree";
import { checkAxis, unzip2, zip } from "../utils";
import { arange, eye, pureArray, tril, triu, zerosLike } from "./array";
import {
  AbstractValue,
  argsort,
  asin,
  atan,
  bind,
  bind1,
  bitcast,
  broadcast,
  cast,
  cholesky,
  cos,
  currentTraceLevel,
  dot,
  erf,
  erfc,
  exp,
  flattenFun,
  flattenFunWithAux,
  fullRaise,
  gather,
  idiv,
  less,
  log,
  lu,
  max,
  min,
  mod,
  neg,
  newMain,
  notEqual,
  pad,
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
  triangularSolve,
  where,
} from "./core";
import { ClosedJaxpr, Jaxpr, jaxprAsFun, makeJaxpr } from "./jaxpr";
import { moveaxis } from "./vmap";

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

/** Compute `a @ b.T`, batched to last two axes. */
function batchMatmulT(a: Tracer, b: Tracer): Tracer {
  return dot(
    a.reshape(a.shape.toSpliced(-1, 0, 1)),
    b.reshape(b.shape.toSpliced(-2, 0, 1)),
  );
}
/** Batch matrix transpose. */
function mT(a: Tracer): Tracer {
  return moveaxis(a, -2, -1);
}
function sliceAxis(a: Tracer, axis: number, p: Pair): Tracer {
  const slices = Array(a.shape.length).fill([]);
  slices[checkAxis(axis, a.ndim)] = p;
  return a.slice(...slices);
}
function padAxis(a: Tracer, axis: number, p: Pair): Tracer {
  const pads = Array(a.shape.length).fill([0, 0]);
  pads[checkAxis(axis, a.ndim)] = p;
  return pad(a, pads);
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
  [Primitive.Concatenate]: linearTangentsJvp(Primitive.Concatenate),
  [Primitive.Split]: linearTangentsJvp(Primitive.Split),
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
  [Primitive.Sort]([x], [dx]) {
    // Propagate both primals and derivatives along the sorted order.
    const [y, idx] = argsort(x);
    return [[y], [gather(dx, [idx], [-1], -1)]];
  },
  [Primitive.Argsort]([x], [dx]) {
    const [y, idx] = argsort(x);
    return [
      [y, idx.ref],
      [gather(dx, [idx.ref], [-1], -1), zerosLike(idx)],
    ];
  },
  [Primitive.TriangularSolve]([a, b], [da, db], { unitDiagonal }) {
    // A @ X.T = B.T  =>  dA @ X.T + A @ dX.T = dB.T
    // So: A @ dX.T = dB.T - dA @ X.T
    // Therefore: dX.T = A^-1 @ (dB.T - dA @ X.T)
    const x = triangularSolve(a.ref, b, { unitDiagonal }); // (A^-1 @ B.T).T
    const dax = batchMatmulT(da, x.ref); // dA @ X.T
    const rhsT = db.sub(mT(dax)); // (dB.T - dA @ X.T).T
    const dx = triangularSolve(a, rhsT, { unitDiagonal });
    return [[x], [dx]];
  },
  [Primitive.Cholesky]([a], [da]) {
    // If L = cholesky(A), so that A = L @ L^T, then
    // dL = L @ tril(S - 0.5 * diag(S)),
    //   where S = L^{-1} @ dA @ L^{-T}
    const L = cholesky(a.ref);
    da = da.ref.add(mT(da)).mul(0.5); // Symmetrize dA for grad
    const W = triangularSolve(L.ref, da, { lower: true }); // (L^-1 @ dA.T).T = dA @ L^-T
    const ST = triangularSolve(L.ref, mT(W), { lower: true });
    const dL = batchMatmulT(
      L.ref,
      triu(ST.ref as any, 1)
        .add(triu(ST as any))
        .mul(0.5),
    );
    return [[L], [dL]];
  },
  [Primitive.LU]([a], [da]) {
    // https://github.com/jax-ml/jax/blob/jax-v0.8.2/jax/_src/lax/linalg.py#L1484
    const [luMatrix, pivots, permutation] = lu(a);
    const [m, n] = a.shape.slice(-2);
    const k = Math.min(m, n);
    // Extract full L: lower triangular with unit diagonal, shape [..., m, m]
    const luSliceL = sliceAxis(luMatrix.ref, -1, [0, k]);
    const lLower = tril(luSliceL as any, -1);
    const lPadded = m > k ? padAxis(lLower, -1, [0, m - k]) : lLower;
    const L = lPadded.add(eye(m));
    // Extract full U: upper triangular, shape [..., n, n]
    // U = triu(lu[:k, :]) padded to [..., n, n] + eye for remaining rows
    const luSliceU = sliceAxis(luMatrix.ref, -2, [0, k]);
    const uUpper = triu(luSliceU as any);
    const uPadded = n > k ? padAxis(uUpper, -2, [0, n - k]) : uUpper;
    const uEye =
      n > k
        ? padAxis(padAxis(eye(n - k), -1, [k, 0]), -2, [k, 0])
        : zerosLike(uPadded.ref);
    const U = uPadded.add(uEye);
    // Apply permutation to da: P @ da (reorder rows)
    const P = permutation.ref
      .reshape([...permutation.shape, 1])
      .equal(arange(m))
      .astype(da.dtype);
    const pda = batchMatmulT(P, mT(da));
    // Solve L @ la = P @ da for la (la = L^{-1} @ P @ da)
    const la = mT(
      triangularSolve(L.ref, mT(pda), {
        lower: true,
        unitDiagonal: true,
      }),
    );
    // Solve lau @ U = la for lau (lau = la @ U^{-1})
    const lau = triangularSolve(mT(U.ref), la, { lower: true });
    const lDot = batchMatmulT(L, mT(tril(lau.ref as any, -1))); // L' = L @ tril(lau)
    const uDot = batchMatmulT(triu(lau as any), mT(U)); // U' = triu(lau) @ U
    return [
      [luMatrix, pivots, permutation],
      [lDot.add(uDot), zerosLike(pivots.ref), zerosLike(permutation.ref)],
    ];
  },
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
  [Primitive.Scan](
    primals,
    tangents,
    { jaxpr, numCarry, numConsts, length, reverse },
  ) {
    // JVP of scan: run a combined scan that processes both primals and tangents.
    //
    // Original scan:
    //   body: (consts, carry, x) -> (new_carry, y)
    //   scan: (consts, init_carry, xs) -> (final_carry, ys)
    //
    // JVP body from jvpJaxpr expects inputs as: [all primals..., all tangents...]
    //   i.e., [consts, carry, x, consts_dot, carry_dot, x_dot]
    // And outputs: [primal_outs..., tangent_outs...]
    //   i.e., [new_carry, y, new_carry_dot, y_dot]
    //
    // But scan feeds body as: [consts..., carry..., x...]
    // So for JVP scan with doubled carry/xs, body receives:
    //   [constsP, constsT, carryP, carryT, xP, xT]  (scan order)
    //
    // We need to reorder to match jvpJaxpr expectations:
    //   [constsP, carryP, xP, constsT, carryT, xT]  (jvp order)
    //
    // Similarly for outputs, jvpJaxpr produces:
    //   [new_carryP, yP, new_carryT, yT]  (jvp order)
    // But scan expects:
    //   [new_carryP, new_carryT, yP, yT]  (scan order, carry then ys)
    //
    // Actually wait - the scan output layout is [carry..., y...] so:
    // For JVP scan with numCarry*2 carries and numY*2 outputs:
    //   scan produces [carryP, carryT, yP, yT] which is correct for interleaved!
    // But jvpJaxpr body returns [carryP, yP, carryT, yT].
    //
    // So we need a wrapper that:
    //   1. Reorders scan inputs [constsP, constsT, carryP, carryT, xP, xT]
    //      to jvp order [constsP, carryP, xP, constsT, carryT, xT]
    //   2. Calls jvpBody
    //   3. Reorders jvp outputs [carryP, yP, carryT, yT]
    //      to scan order [carryP, carryT, yP, yT]

    const numX = primals.length - numConsts - numCarry;
    const numY = jaxpr.outs.length - numCarry;

    // Transform the body jaxpr to compute JVP
    const jvpBody = jvpJaxpr(jaxpr);

    // The JVP transformation may capture additional constants (e.g., triu masks for Cholesky).
    // jvpBody.jaxpr.inBinders = [jvpConsts..., primals..., tangents...]
    //   where primals = [constsP, carryP, xP] and tangents = [constsT, carryT, xT]
    // jvpBody.consts = the actual values for jvpConsts
    //
    // The wrapper should:
    // 1. Take only the doubled body inputs (not JVP consts)
    // 2. Capture jvpBody.consts as constants via closure
    // 3. Reorder from scan order to JVP order

    const numJvpConsts = jvpBody.consts.length;
    const numBodyInputs = numConsts + numCarry + numX;

    // Wrapper takes: [constsP..., constsT..., carryP..., carryT..., xP..., xT...]
    // Reorders to JVP order: [constsP..., carryP..., xP..., constsT..., carryT..., xT...]
    // Calls jvpBody with: [jvpConsts..., constsP..., carryP..., xP..., constsT..., carryT..., xT...]
    // Gets: [new_carryP..., yP..., new_carryT..., yT...]
    // Reorders to: [new_carryP..., new_carryT..., yP..., yT...]

    // Get the body input avals in JVP order (primals then tangents)
    const jvpOrderAvals = jvpBody.jaxpr.inBinders
      .slice(numJvpConsts)
      .map((v) => v.aval);
    // jvpOrderAvals = [constsP..., carryP..., xP..., constsT..., carryT..., xT...]

    // Reorder to scan order: [constsP..., constsT..., carryP..., carryT..., xP..., xT...]
    const constsP_avals = jvpOrderAvals.slice(0, numConsts);
    const carryP_avals = jvpOrderAvals.slice(numConsts, numConsts + numCarry);
    const xP_avals = jvpOrderAvals.slice(numConsts + numCarry, numBodyInputs);
    const constsT_avals = jvpOrderAvals.slice(
      numBodyInputs,
      numBodyInputs + numConsts,
    );
    const carryT_avals = jvpOrderAvals.slice(
      numBodyInputs + numConsts,
      numBodyInputs + numConsts + numCarry,
    );
    const xT_avals = jvpOrderAvals.slice(numBodyInputs + numConsts + numCarry);

    const wrapperInAvals = [
      ...constsP_avals,
      ...constsT_avals,
      ...carryP_avals,
      ...carryT_avals,
      ...xP_avals,
      ...xT_avals,
    ];

    const { jaxpr: wrapperJaxpr } = makeJaxpr(
      (...scanOrderArgs: Tracer[]): Tracer[] => {
        // scanOrderArgs layout: [constsP, constsT, carryP, carryT, xP, xT]
        // (no JVP consts - those are captured via closure)

        const constsP_in = scanOrderArgs.slice(0, numConsts);
        const constsT_in = scanOrderArgs.slice(numConsts, numConsts * 2);
        const carryP_in = scanOrderArgs.slice(
          numConsts * 2,
          numConsts * 2 + numCarry,
        );
        const carryT_in = scanOrderArgs.slice(
          numConsts * 2 + numCarry,
          numConsts * 2 + numCarry * 2,
        );
        const xP_in = scanOrderArgs.slice(
          numConsts * 2 + numCarry * 2,
          numConsts * 2 + numCarry * 2 + numX,
        );
        const xT_in = scanOrderArgs.slice(numConsts * 2 + numCarry * 2 + numX);

        // Reorder to jvp order: [constsP, carryP, xP, constsT, carryT, xT]
        const jvpOrderArgs = [
          ...constsP_in.map((x) => x.ref),
          ...carryP_in.map((x) => x.ref),
          ...xP_in.map((x) => x.ref),
          ...constsT_in.map((x) => x.ref),
          ...carryT_in.map((x) => x.ref),
          ...xT_in.map((x) => x.ref),
        ];

        // Call the jvpBody jaxpr with jvpConsts (captured) first, then reordered body args
        const jvpOutputs = bind(
          Primitive.Jit,
          [...jvpBody.consts.map((c) => c.ref), ...jvpOrderArgs],
          {
            jaxpr: jvpBody.jaxpr,
            numConsts: numJvpConsts,
            name: "jvp_body",
          },
        );

        // jvpOutputs layout: [carryP..., yP..., carryT..., yT...]
        // Reorder to scan output order: [carryP..., carryT..., yP..., yT...]
        const carryP_out = jvpOutputs.slice(0, numCarry);
        const yP_out = jvpOutputs.slice(numCarry, numCarry + numY);
        const carryT_out = jvpOutputs.slice(
          numCarry + numY,
          numCarry * 2 + numY,
        );
        const yT_out = jvpOutputs.slice(numCarry * 2 + numY);

        return [...carryP_out, ...carryT_out, ...yP_out, ...yT_out];
      },
    )(...wrapperInAvals);

    // Original args: consts (numConsts), carry (numCarry), xs (numX)
    const constsP = primals.slice(0, numConsts);
    const carryP = primals.slice(numConsts, numConsts + numCarry);
    const xsP = primals.slice(numConsts + numCarry);

    const constsT = tangents.slice(0, numConsts);
    const carryT = tangents.slice(numConsts, numConsts + numCarry);
    const xsT = tangents.slice(numConsts + numCarry);

    // Build scan args in scan order:
    // [wrapperConsts..., constsP, constsT, carryP, carryT, xsP, xsT]
    // The jvpBody.consts are captured by the wrapper jaxpr (in wrapperJaxpr.consts)
    const scanArgsJvp = [
      ...wrapperJaxpr.consts.map((c) => c.ref),
      ...constsP.map((c) => c.ref),
      ...constsT.map((c) => c.ref),
      ...carryP.map((c) => c.ref),
      ...carryT.map((c) => c.ref),
      ...xsP.map((x) => x.ref),
      ...xsT.map((x) => x.ref),
    ];

    const results = bind(Primitive.Scan, scanArgsJvp, {
      jaxpr: wrapperJaxpr.jaxpr,
      numCarry: numCarry * 2,
      numConsts: wrapperJaxpr.consts.length + numConsts * 2,
      length,
      reverse,
    });

    // Dispose the wrapper jaxpr (not cached)
    // Note: jvpBody is cached via jvpJaxprCache, so we don't dispose it
    wrapperJaxpr.dispose();

    // Results layout from wrapper: [carryP..., carryT..., yP..., yT...]
    const carryOutP = results.slice(0, numCarry);
    const carryOutT = results.slice(numCarry, numCarry * 2);
    const ysP = results.slice(numCarry * 2, numCarry * 2 + numY);
    const ysT = results.slice(numCarry * 2 + numY);

    const primalsOut = [...carryOutP, ...ysP];
    const tangentsOut = [...carryOutT, ...ysT];

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
  { hasAux = false } = {},
): [any, any, any?] {
  const [primalsFlat, inTree] = treeFlatten(primals);
  const [tangentsFlat, inTree2] = treeFlatten(tangents);
  if (!inTree.equals(inTree2)) {
    throw new TreeMismatchError("jvp", inTree, inTree2);
  }

  let flatFun, outTree, aux;
  if (hasAux) {
    [flatFun, outTree, aux] = flattenFunWithAux(f, inTree);
  } else {
    [flatFun, outTree] = flattenFun(f, inTree);
  }

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

  if (hasAux) {
    return [primalsOut, tangentsOut, lowerAux(aux!.value)];
  }
  return [primalsOut, tangentsOut];
}

/** Lowering for auxiliary data returned in `hasAux: true` methods. */
export function lowerAux(aux: any): any {
  const level = currentTraceLevel();

  return treeMap((x: Tracer) => {
    if (x instanceof Tracer) {
      while (x._trace.main.level > level) {
        if (x instanceof JVPTracer) {
          x.tangent.dispose();
          x = x.primal;
        } else {
          const y = x.fullLower();
          if (y._trace.main.level >= x._trace.main.level)
            throw new Error("internal: lowerAux did not reduce trace level");
          x = y;
        }
      }
    }
    return x;
  }, aux);
}
