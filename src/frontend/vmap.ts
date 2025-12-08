import { assertNonNull, checkAxis, range, rep, unzip2, zip } from "../utils";
import { eye, pureArray } from "./array";
import {
  AbstractValue,
  add,
  asin,
  atan,
  bind,
  bitcast,
  broadcast,
  cast,
  compare,
  cos,
  dot,
  erf,
  erfc,
  exp,
  flattenFun,
  flip,
  fullRaise,
  idiv,
  log,
  max,
  min,
  mul,
  ndim,
  neg,
  newMain,
  pad,
  Primitive,
  PrimitiveParams,
  reciprocal,
  reduce,
  reshape,
  ShapedArray,
  shrink,
  sin,
  sqrt,
  stopGradient,
  Trace,
  Tracer,
  TracerValue,
  transpose,
  TreeMismatchError,
  where,
} from "./core";
import {
  JsTree,
  JsTreeDef,
  flatten as treeFlatten,
  unflatten as treeUnflatten,
} from "../tree";
import { Jaxpr, jaxprAsFun, makeJaxpr } from "./jaxpr";
import { jvp } from "./jvp";

function mappedAval(batchDim: number, aval: AbstractValue) {
  const shape = [...aval.shape];
  shape.splice(batchDim, 1); // Remove the batch dimension.
  return new ShapedArray(shape, aval.dtype, aval.weakType);
}

/** Move one axis to a different index. */
export function moveaxis(x: TracerValue, src: number, dst: number) {
  const t = pureArray(x);
  src = checkAxis(src, t.ndim);
  dst = checkAxis(dst, t.ndim);
  if (src === dst) return t;
  const perm = range(t.ndim);
  perm.splice(src, 1);
  perm.splice(dst, 0, src);
  return transpose(t, perm);
}

function moveBatchAxis(
  axisSize: number,
  src: number | null,
  dst: number,
  x: Tracer,
) {
  if (src === null) {
    // not_mapped
    const targetShape = [...x.shape];
    targetShape.splice(dst, 0, axisSize);
    return broadcast(x, targetShape, [dst]);
  } else if (src === dst) {
    return x;
  } else {
    return moveaxis(x, src, dst);
  }
}

class BatchTracer extends Tracer {
  constructor(
    trace: Trace,
    readonly val: Tracer,
    readonly batchDim: number | null,
  ) {
    super(trace);
  }

  get aval(): AbstractValue {
    if (this.batchDim === null) {
      return this.val.aval;
    } else {
      return mappedAval(this.batchDim, this.val.aval);
    }
  }

  toString(): string {
    return `BatchTracer(${this.val.toString()}, ${this.batchDim})`;
  }

  get ref() {
    this.val.ref;
    return this;
  }
  dispose() {
    this.val.dispose();
  }

  fullLower(): Tracer {
    if (this.batchDim === null) {
      return this.val.fullLower();
    } else {
      return this;
    }
  }
}

class BatchTrace extends Trace {
  pure(val: TracerValue) {
    return this.lift(pureArray(val));
  }

  lift(val: Tracer): Tracer {
    return new BatchTracer(this, val, null);
  }

  processPrimitive<P extends Primitive>(
    primitive: P,
    tracers: BatchTracer[],
    params: PrimitiveParams<P>,
  ): BatchTracer[] {
    const [valsIn, bdimsIn] = unzip2(tracers.map((t) => [t.val, t.batchDim]));
    const vmapRule = vmapRules[primitive];
    if (vmapRule === undefined) {
      throw new Error(`No vmap rule for: ${primitive}`);
    }
    if (bdimsIn.every((d) => d === null)) {
      // This should not usually happen because `fullLower()` would unwrap the
      // BatchTracer before getting here. However, I'm not sure about this in
      // edge cases, so it's better to just be safe.
      const valOuts = bind(primitive, valsIn, params);
      return valOuts.map((x) => new BatchTracer(this, x, null));
    }
    const [valOuts, bdimOuts] = vmapRule(
      this.axisSize,
      valsIn,
      bdimsIn,
      params,
    );
    return zip(valOuts, bdimOuts).map(
      ([x, bd]) => new BatchTracer(this, x, bd),
    );
  }

  get axisSize(): number {
    return this.main.globalData;
  }
}

// Apply a primitive to batched arguments with built-in broadcasting rules.
//
// This defines "how" a primitive should be vectorized over batch dimensions.
// The caller is guaranteed to pass at least one of `dims` as non-null.
type VmapRule<P extends Primitive> = (
  axisSize: number,
  args: Tracer[],
  dims: (number | null)[],
  params: PrimitiveParams<P>,
) => [Tracer[], (number | null)[]];

/**
 * Process a primitive with built-in broadcasting.
 *
 * Reference: https://github.com/jax-ml/jax/blob/jax-v0.8.1/jax/_src/interpreters/batching.py#L1029
 */
function broadcastBatcher(op: (...x: Tracer[]) => Tracer): VmapRule<Primitive> {
  return (axisSize, args, dims) => {
    if (args.length === 0) {
      throw new Error("Empty list in broadcastBatcher");
    }
    const nd = Math.max(...args.map(ndim));
    const firstIdx = dims.findIndex((d) => d !== null);
    const firstBdim = dims[firstIdx]! - args[firstIdx].ndim; // e.g., -1 if last dim
    if (
      // If only agreeing batch dims, or scalars, just call the primitive.
      zip(args, dims).every(
        ([x, d]) =>
          ndim(x) < -firstBdim || (d !== null && d - x.ndim === firstBdim),
      )
    ) {
      return [[op(...args)], [nd + firstBdim]];
    }

    // Move the batch axes to the front. If needed, expand arrays so that all
    // inputs have the same number of dimensions.
    args = args.map((x, i) => {
      if (ndim(x) === 0) return x;
      x = moveBatchAxis(axisSize, dims[i], 0, x);
      if (x.ndim < nd)
        x = x.reshape([
          x.shape[0],
          ...rep(nd - x.ndim, 1),
          ...x.shape.slice(1),
        ]);
      return x;
    });
    return [[op(...args)], [0]];
  };
}

function unopBatcher<P extends Primitive>(
  op: (x: Tracer, params: PrimitiveParams<P>) => Tracer,
): VmapRule<P> {
  return (axisSize, [x], [xBdim], params) => {
    return [[op(x, params)], [xBdim]];
  };
}

const vmapRules: Partial<{ [P in Primitive]: VmapRule<P> }> = {
  [Primitive.Add]: broadcastBatcher(add),
  [Primitive.Mul]: broadcastBatcher(mul),
  [Primitive.Idiv]: broadcastBatcher(idiv),
  [Primitive.Neg]: unopBatcher(neg),
  [Primitive.Reciprocal]: unopBatcher(reciprocal),
  [Primitive.StopGradient]: unopBatcher(stopGradient),
  [Primitive.Cast]: unopBatcher((x, { dtype }) => cast(x, dtype)),
  [Primitive.Bitcast]: unopBatcher((x, { dtype }) => bitcast(x, dtype)),
  // TODO: random_bits
  [Primitive.Sin]: unopBatcher(sin),
  [Primitive.Cos]: unopBatcher(cos),
  [Primitive.Asin]: unopBatcher(asin),
  [Primitive.Atan]: unopBatcher(atan),
  [Primitive.Exp]: unopBatcher(exp),
  [Primitive.Log]: unopBatcher(log),
  [Primitive.Erf]: unopBatcher(erf),
  [Primitive.Erfc]: unopBatcher(erfc),
  [Primitive.Sqrt]: unopBatcher(sqrt),
  [Primitive.Min]: broadcastBatcher(min),
  [Primitive.Max]: broadcastBatcher(max),
  [Primitive.Reduce](axisSize, [x], [xBdim], { op, axis }) {
    assertNonNull(xBdim);
    const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
    const outBdim = xBdim - axis.filter((ax) => ax < xBdim).length;
    return [[reduce(x, op, newAxis)], [outBdim]];
  },
  [Primitive.Dot](axisSize, [x, y], [xBdim, yBdim]) {
    // Move both the batch axes to the second-to-last position.
    x = moveBatchAxis(axisSize, xBdim, x.ndim - (xBdim === null ? 1 : 2), x);
    y = moveBatchAxis(axisSize, yBdim, y.ndim - (yBdim === null ? 1 : 2), y);
    const z = dot(x, y);
    return [[z], [z.ndim - 1]]; // The batch axis is now at the end.
  },
  // TODO: conv, pool, pool_transpose
  [Primitive.Compare](axisSize, args, dims, { op }) {
    return broadcastBatcher((x, y) => compare(x, y, op))(
      axisSize,
      args,
      dims,
      {},
    );
  },
  [Primitive.Where]: broadcastBatcher(where),
  [Primitive.Transpose](axisSize, [x], [xBdim], { perm }) {
    assertNonNull(xBdim);
    const newPerm = perm.map((p) => p + (xBdim <= p ? 1 : 0));
    newPerm.splice(xBdim, 0, xBdim); // Keep the batch dim in place.
    return [[transpose(x, newPerm)], [xBdim]];
  },
  [Primitive.Broadcast](axisSize, [x], [xBdim], { shape, axis }) {
    assertNonNull(xBdim);
    const newShape = shape.toSpliced(xBdim, 0, axisSize);
    const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
    return [[broadcast(x, newShape, newAxis)], [xBdim]];
  },
  [Primitive.Reshape](axisSize, [x], [xBdim], { shape }) {
    // Move xBdim to the front, so reshape can have contiguous axes.
    x = moveBatchAxis(axisSize, xBdim, 0, x);
    return [[reshape(x, [axisSize, ...shape])], [0]];
  },
  [Primitive.Flip](axisSize, [x], [xBdim], { axis }) {
    assertNonNull(xBdim);
    const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
    return [[flip(x, newAxis)], [xBdim]];
  },
  [Primitive.Shrink](axisSize, [x], [xBdim], { slice }) {
    assertNonNull(xBdim);
    const newSlice = slice.toSpliced(xBdim, 0, [0, axisSize]);
    return [[shrink(x, newSlice)], [xBdim]];
  },
  [Primitive.Pad](axisSize, [x], [xBdim], { width }) {
    assertNonNull(xBdim);
    const newWidth = width.toSpliced(xBdim, 0, [0, 0]);
    return [[pad(x, newWidth)], [xBdim]];
  },
  // TODO: gather
  [Primitive.JitCall](axisSize, args, dims, { name, jaxpr }) {
    const { newJaxpr, newConsts } = vmapJaxpr(jaxpr, axisSize, dims);
    const outs = bind(
      Primitive.JitCall,
      [...newConsts.map((c) => c.ref), ...args],
      {
        name: `${name}_vmap`,
        jaxpr: newJaxpr,
        numConsts: newConsts.length,
      },
    );
    return [outs, rep(outs.length, 0)];
  },
};

const vmapJaxprCache = new Map<
  Jaxpr,
  Map<string, ReturnType<typeof vmapJaxpr>>
>();

function vmapJaxpr(
  jaxpr: Jaxpr,
  axisSize: number,
  dims: (number | null)[],
): { newJaxpr: Jaxpr; newConsts: Tracer[] } {
  const cacheKey = JSON.stringify([axisSize, dims]); // deterministic
  const prevResult = vmapJaxprCache.get(jaxpr)?.get(cacheKey);
  if (prevResult) return prevResult;

  // Consts in the Jaxpr become real inputs after vmap transformation, which is
  // why we ignore numConsts.
  //
  // See the comment in jvpJaxpr() to explain more about what's going on here.
  // This is handling vmap-of-jit, which is a bit tricky. We need to turn the
  // Jaxpr back into a function and retrace it.
  const inAvals = jaxpr.inBinders.map((v, i) => {
    if (dims[i] === null) return v.aval;
    const shape = [...v.aval.shape];
    shape.splice(dims[i], 0, axisSize); // Insert the mapped axis into the shape.
    return new ShapedArray(shape, v.aval.dtype, v.aval.weakType);
  });
  const { jaxpr: newJaxpr, consts: newConsts } = makeJaxpr((args: Tracer[]) =>
    vmapFlat(jaxprAsFun(jaxpr), dims, args),
  )(inAvals);
  const result = { newJaxpr, newConsts };

  if (!vmapJaxprCache.has(jaxpr)) vmapJaxprCache.set(jaxpr, new Map());
  vmapJaxprCache.get(jaxpr)!.set(cacheKey, result);
  return result;
}

function vmapFlat(
  f: (...x: Tracer[]) => TracerValue[],
  inAxes: (number | null)[],
  args: TracerValue[],
): Tracer[] {
  let axisSize: number | undefined = undefined;
  for (let i = 0; i < args.length; i++) {
    if (inAxes[i] !== null) {
      const arg = args[i];
      if (!(arg instanceof Tracer)) {
        throw new TypeError("vmap requires Tracer argument for mapped axes");
      }
      const size = arg.shape[inAxes[i]!];
      if (axisSize === undefined) {
        axisSize = size;
      } else if (axisSize !== size) {
        throw new TypeError(
          "vmap requires all mapped axes to have the same size",
        );
      }
    }
  }
  if (axisSize === undefined) {
    throw new TypeError("vmap requires at least one mapped axis");
  }

  let valsOut: Tracer[], bdimsOut: (number | null)[];
  {
    using main = newMain(BatchTrace, axisSize);
    const trace = new BatchTrace(main);
    const tracersIn = args.map((x, i) =>
      inAxes[i] === null
        ? pureArray(x)
        : new BatchTracer(trace, pureArray(x), inAxes[i]),
    );
    const outs = f(...tracersIn);
    const tracersOut = outs.map((out) => fullRaise(trace, out) as BatchTracer);
    [valsOut, bdimsOut] = unzip2(tracersOut.map((t) => [t.val, t.batchDim]));
  }
  return zip(valsOut, bdimsOut).map(([valOut, bdim]) =>
    moveBatchAxis(axisSize, bdim, 0, valOut),
  ); // outs_transposed
}

export function vmap(
  f: (...x: any[]) => JsTree<TracerValue>,
  inAxes: number | JsTree<number | null>[] = 0,
): (...x: JsTree<TracerValue>[]) => JsTree<Tracer> {
  return (...args: any[]) => {
    const [argsFlat, inTree] = treeFlatten(args);
    let inAxesFlat: (number | null)[];
    if (typeof inAxes === "number") {
      // If mapping over a single axis, just use it for all inputs.
      inAxesFlat = rep(argsFlat.length, inAxes);
    } else {
      let inTree2: JsTreeDef;
      [inAxesFlat, inTree2] = treeFlatten(inAxes);
      if (!inTree.equals(inTree2)) {
        throw new TreeMismatchError("vmap", inTree, inTree2);
      }
    }
    const [fFlat, outTree] = flattenFun(f, inTree);
    const outsFlat = vmapFlat(fFlat, inAxesFlat, argsFlat);
    if (outTree.value === undefined) {
      throw new Error("outTree was not set in vmap");
    }
    return treeUnflatten(outTree.value, outsFlat);
  };
}

// See also: jacrev()
export function jacfwd(f: any) {
  return function jacobianForward(x: Tracer) {
    if (x.shape.length !== 1) {
      throw new TypeError("jacfwd only supports 1D inputs");
    }
    const [size] = x.shape;
    const pushfwd = (v: Tracer) => jvp(f, [x], [v])[1];
    return vmap(pushfwd, [0])(eye(size, undefined, { dtype: x.dtype }));
  };
}
