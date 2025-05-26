import { deepEqual, range, rep, unzip2, zip } from "../utils";
import { eye, pureArray } from "./array";
import {
  AbstractValue,
  add,
  bind,
  broadcast,
  compare,
  CompareOp,
  cos,
  flattenFun,
  fullRaise,
  mul,
  ndim,
  neg,
  newMain,
  Primitive,
  reduceSum,
  ShapedArray,
  sin,
  Trace,
  Tracer,
  TracerValue,
  transpose,
  TreeMismatchError,
} from "./core";
import { flatten as treeFlatten, unflatten as treeUnflatten } from "../tree";
import { Jaxpr, jaxprAsFun, makeJaxpr } from "./jaxpr";
import { jvp } from "./jvp";

function mappedAval(batchDim: number, aval: AbstractValue) {
  const shape = [...aval.shape];
  shape.splice(batchDim, 1);
  return new ShapedArray(shape, aval.dtype);
}

/** Move one axis to a different index. */
export function moveaxis(x: TracerValue, src: number, dst: number) {
  const t = pureArray(x);
  const perm = range(t.shape.length);
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

  processPrimitive(
    primitive: Primitive,
    tracers: BatchTracer[],
    params: Record<string, any>,
  ): BatchTracer[] {
    const [valsIn, bdimsIn] = unzip2(tracers.map((t) => [t.val, t.batchDim]));
    const vmapRule = vmapRules[primitive];
    if (vmapRule === undefined) {
      throw new Error(`No vmap rule for: ${primitive}`);
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

type VmapRule = (
  axisSize: number,
  args: Tracer[],
  dims: (number | null)[],
  params: any,
) => [Tracer[], (number | null)[]];

function handleScalarBroadcasting(nd: number, x: Tracer, d: number | null) {
  if (d === null || nd === ndim(x)) {
    return x;
  } else {
    const axis = range(ndim(x), nd);
    const shape = [...x.shape, ...axis.map(() => 1)];
    return broadcast(x, shape, axis);
  }
}

/** Process a primitive with built-in broadcasting. */
function broadcastBatcher(op: (...x: Tracer[]) => Tracer) {
  return (
    axisSize: number,
    args: Tracer[],
    dims: (number | null)[],
  ): ReturnType<VmapRule> => {
    if (args.length === 0) {
      throw new Error("Empty list in broadcastBatcher");
    }

    const idx = dims.findIndex((d) => d !== null);
    if (idx === -1) {
      // No-op case: no mapped indices, just pass it down to the parent tracer.
      return [[op(...args)], [null]];
    }
    if (
      // If only agreeing batch dims, or scalars, just call the primitive.
      zip(args, dims).every(
        ([x, d]) =>
          ndim(x) === 0 ||
          (deepEqual(x.shape, args[idx].shape) && d === dims[idx]),
      )
    ) {
      return [[op(...args)], [dims[idx]]];
    }

    args = args.map((x, i) =>
      ndim(x) > 0 ? moveBatchAxis(axisSize, dims[i], 0, x) : x,
    );
    // Now the batch axis has been added to the front. Handle special-case of
    // scalar broadcasting, since unmapped axes may have a singleton axis
    // inserted and then rely on the built-in broadcasting of the primitive.
    const nd = Math.max(...args.map(ndim));
    args = args.map((x, i) => handleScalarBroadcasting(nd, x, dims[i]));
    return [[op(...args)], [0]];
  };
}

function vectorizedUnopBatchingRule(op: (x: Tracer) => Tracer) {
  return (
    axisSize: number,
    [x]: Tracer[],
    [xBdim]: (number | null)[],
  ): ReturnType<VmapRule> => {
    return [[op(x)], [xBdim]];
  };
}

const vmapRules: Partial<Record<Primitive, VmapRule>> = {
  [Primitive.Add]: broadcastBatcher(add),
  [Primitive.Mul]: broadcastBatcher(mul),
  [Primitive.Neg]: vectorizedUnopBatchingRule(neg),
  [Primitive.Sin]: vectorizedUnopBatchingRule(sin),
  [Primitive.Cos]: vectorizedUnopBatchingRule(cos),
  [Primitive.ReduceSum](axisSize, [x], [xBdim], { axis }: { axis: number[] }) {
    if (xBdim === null) {
      return [[reduceSum(x, axis)], [null]];
    }
    const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
    const outBdim = xBdim - axis.filter((ax) => ax < xBdim).length;
    return [[reduceSum(x, newAxis)], [outBdim]];
  },
  [Primitive.Compare](axisSize, args, dims, { op }: { op: CompareOp }) {
    return broadcastBatcher((x, y) => compare(x, y, op))(axisSize, args, dims);
  },
  // TODO: where, transpose, broadcast, reshape, flip
  [Primitive.JitCall](axisSize, args, dims, { jaxpr }: { jaxpr: Jaxpr }) {
    const { newJaxpr, newConsts } = vmapJaxpr(jaxpr, axisSize, dims);
    const outs = bind(Primitive.JitCall, [...newConsts, ...args], {
      jaxpr: newJaxpr,
      numConsts: newConsts.length,
    });
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
    return new ShapedArray(shape, v.aval.dtype);
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
  f: (...x: any[]) => any,
  inAxes: any[],
): (...x: any[]) => any {
  return (...args: any[]) => {
    const [argsFlat, inTree] = treeFlatten(args);
    const [inAxesFlat, inTree2] = treeFlatten(inAxes);
    if (!inTree.equals(inTree2)) {
      throw new TreeMismatchError("vmap", inTree, inTree2);
    }
    const [fFlat, outTree] = flattenFun(f, inTree);
    const outsFlat = vmapFlat(fFlat, inAxesFlat, argsFlat);
    if (outTree.value === undefined) {
      throw new Error("outTree was not set in vmap");
    }
    return treeUnflatten(outTree.value, outsFlat);
  };
}

export function jacfwd(f: any, x: Tracer) {
  if (x.shape.length !== 1) {
    throw new TypeError("jacfwd only supports 1D inputs");
  }
  const [size] = x.shape;
  const pushfwd = (v: Tracer) => jvp(f, [x], [v])[1];
  return vmap(pushfwd, [0])(eye(size));
}
