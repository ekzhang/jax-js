import { assertNonNull, checkAxis, range, rep, unzip2, zip } from "../utils";
import { arange, eye, pureArray } from "./array";
import {
  AbstractValue,
  bind,
  bind1,
  broadcast,
  concatenate,
  conv,
  dot,
  flattenFun,
  flip,
  fullRaise,
  gather,
  ndim,
  newMain,
  pad,
  Primitive,
  PrimitiveParams,
  randomBits,
  reduce,
  reshape,
  ShapedArray,
  shrink,
  split,
  Trace,
  Tracer,
  TracerValue,
  transpose,
  TreeMismatchError,
} from "./core";
import {
  JsTree,
  flatten as treeFlatten,
  unflatten as treeUnflatten,
} from "../tree";
import { ClosedJaxpr, Jaxpr, jaxprAsFun, makeJaxpr } from "./jaxpr";
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
    if (valOuts.length !== bdimOuts.length) {
      throw new Error(
        `vmap rule for ${primitive} returned mismatched lengths: ` +
          `${valOuts.length} vs ${bdimOuts.length}`,
      );
    }
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
function broadcastBatcher<P extends Primitive>(prim: P): VmapRule<P> {
  return (axisSize, args, dims, params) => {
    if (args.length === 0) {
      throw new Error("Empty list in broadcastBatcher");
    }
    // Determine the output ndim after broadcasting, including batch.
    const nd = Math.max(
      ...args.map((x, i) => ndim(x) + (dims[i] === null ? 1 : 0)),
    );

    const firstIdx = dims.findIndex((d) => d !== null);
    const firstBdim = dims[firstIdx]! - args[firstIdx].ndim; // e.g., -1 if last dim
    if (
      // If only agreeing batch dims, or scalars, just call the primitive.
      zip(args, dims).every(
        ([x, d]) =>
          (d === null && ndim(x) < -firstBdim) ||
          (d !== null && d - x.ndim === firstBdim),
      )
    ) {
      return [[bind1(prim, args, params)], [nd + firstBdim]];
    }

    // Move the batch axes to the front. If needed, expand arrays so that all
    // inputs have the same number of dimensions.
    args = args.map((x, i) => {
      if (dims[i] === null) return x;
      x = moveBatchAxis(axisSize, dims[i], 0, x);
      if (x.ndim < nd)
        x = x.reshape([
          x.shape[0],
          ...rep(nd - x.ndim, 1),
          ...x.shape.slice(1),
        ]);
      return x;
    });
    return [[bind1(prim, args, params)], [0]];
  };
}

function unopBatcher<P extends Primitive>(prim: P): VmapRule<P> {
  return (axisSize, [x], [xBdim], params) => {
    return [[bind1(prim, [x], params)], [xBdim]];
  };
}

function lastDimsBatcher<P extends Primitive>(
  prim: P,
  inputDims: number,
  numOutputs: number = 1,
): VmapRule<P> {
  return (axisSize, [x], [xBdim], params) => {
    assertNonNull(xBdim);
    if (xBdim < x.ndim - inputDims) {
      return [bind(prim, [x], params), rep(numOutputs, xBdim)];
    }
    x = moveBatchAxis(axisSize, xBdim, 0, x);
    return [bind(prim, [x], params), rep(numOutputs, 0)];
  };
}

const vmapRules: Partial<{ [P in Primitive]: VmapRule<P> }> = {
  [Primitive.Add]: broadcastBatcher(Primitive.Add),
  [Primitive.Mul]: broadcastBatcher(Primitive.Mul),
  [Primitive.Idiv]: broadcastBatcher(Primitive.Idiv),
  [Primitive.Mod]: broadcastBatcher(Primitive.Mod),
  [Primitive.Min]: broadcastBatcher(Primitive.Min),
  [Primitive.Max]: broadcastBatcher(Primitive.Max),
  [Primitive.Neg]: unopBatcher(Primitive.Neg),
  [Primitive.Reciprocal]: unopBatcher(Primitive.Reciprocal),
  [Primitive.Floor]: unopBatcher(Primitive.Floor),
  [Primitive.Ceil]: unopBatcher(Primitive.Ceil),
  [Primitive.StopGradient]: unopBatcher(Primitive.StopGradient),
  [Primitive.Cast]: unopBatcher(Primitive.Cast),
  [Primitive.Bitcast]: unopBatcher(Primitive.Bitcast),
  [Primitive.Sin]: unopBatcher(Primitive.Sin),
  [Primitive.Cos]: unopBatcher(Primitive.Cos),
  [Primitive.Asin]: unopBatcher(Primitive.Asin),
  [Primitive.Atan]: unopBatcher(Primitive.Atan),
  [Primitive.Exp]: unopBatcher(Primitive.Exp),
  [Primitive.Log]: unopBatcher(Primitive.Log),
  [Primitive.Erf]: unopBatcher(Primitive.Erf),
  [Primitive.Erfc]: unopBatcher(Primitive.Erfc),
  [Primitive.Sqrt]: unopBatcher(Primitive.Sqrt),
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
  [Primitive.Conv](axisSize, [x, y], [xBdim, yBdim], params) {
    // Move batch axes to the front, then increment params.vmapDims.
    x = moveBatchAxis(axisSize, xBdim, 0, x);
    y = moveBatchAxis(axisSize, yBdim, 0, y);
    const z = conv(x, y, { ...params, vmapDims: params.vmapDims + 1 });
    return [[z], [0]];
  },
  // TODO: pool, pool_transpose
  [Primitive.Compare]: broadcastBatcher(Primitive.Compare),
  [Primitive.Where]: broadcastBatcher(Primitive.Where),
  [Primitive.Concatenate](axisSize, xs, xBdims, { axis }) {
    const minBdim = Math.min(...xBdims.filter((d) => d !== null));
    xs = xs.map((x, i) => moveBatchAxis(axisSize, xBdims[i], minBdim, x));
    const newAxis = axis + (minBdim <= axis ? 1 : 0);
    return [[concatenate(xs, newAxis)], [minBdim]];
  },
  [Primitive.Split](axisSize, [x], [xBdim], { axis, sizes }) {
    assertNonNull(xBdim);
    const newAxis = axis + (xBdim <= axis ? 1 : 0);
    const outs = split(x, newAxis, sizes);
    return [outs, rep(outs.length, xBdim)];
  },
  [Primitive.RandomBits](axisSize, [k0, k1], [bdim0, bdim1], { shape, mode }) {
    k0 = moveBatchAxis(axisSize, bdim0, 0, k0);
    k1 = moveBatchAxis(axisSize, bdim1, 0, k1);
    return [[randomBits(k0, k1, [axisSize, ...shape], mode)], [0]];
  },
  [Primitive.Gather](
    axisSize,
    [x, ...indices],
    [xBdim, ...indicesBdim],
    { axis, outDim },
  ) {
    if (indicesBdim.every((d) => d === null)) {
      // If none of the indices are mapped, this is an ordinary Gather on larger
      // x array. Just recalculate axis numbers.
      assertNonNull(xBdim);
      const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
      let newBdim = xBdim - axis.filter((ax) => ax < xBdim).length;
      let newOutDim = outDim;
      if (newOutDim < newBdim) newBdim += axis.length;
      else newOutDim += 1;
      return [[gather(x, indices, newAxis, newOutDim)], [newBdim]];
    }
    // If indices are mapped, move those mapped axes to front.
    const nd = Math.max(
      ...indices.map((m, i) => ndim(m) + (indicesBdim[i] === null ? 1 : 0)),
    );
    indices = indices.map((m, i) => {
      if (indicesBdim[i] === null) return m;
      m = moveBatchAxis(axisSize, indicesBdim[i], 0, m);
      if (m.ndim < nd)
        m = m.reshape([
          m.shape[0],
          ...rep(nd - m.ndim, 1),
          ...m.shape.slice(1),
        ]);
      return m;
    });
    // Now there are two cases. If x is not mapped, dispatch directly.
    if (xBdim === null) {
      return [[gather(x, indices, axis, outDim)], [outDim]];
    } else {
      // Otherwise, we need a new `arange(axisSize)` index.
      // For simplicity, let's also move x's batch axis to the front.
      x = moveBatchAxis(axisSize, xBdim, 0, x);
      const newAxis = [0, ...axis.map((ax) => ax + 1)];
      const extraBatchIndex = arange(axisSize).reshape([-1, ...rep(nd - 1, 1)]);
      indices.splice(0, 0, extraBatchIndex);
      return [[gather(x, indices, newAxis, outDim)], [outDim]];
    }
  },
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
  [Primitive.Sort]: lastDimsBatcher(Primitive.Sort, 1),
  [Primitive.Argsort]: lastDimsBatcher(Primitive.Argsort, 1, 2),
  [Primitive.TriangularSolve](
    axisSize,
    [a, b],
    [aBdim, bBdim],
    { unitDiagonal },
  ) {
    if (aBdim === null) {
      // If only vmapping over b, we can just call TriangularSolve directly.
      b = moveBatchAxis(axisSize, bBdim, -3, b);
      const [s, m, n] = b.shape.slice(-3);
      b = b.reshape([...b.shape.slice(0, -3), s * m, n]);
      let x = bind1(Primitive.TriangularSolve, [a, b], { unitDiagonal });
      x = x.reshape([...b.shape.slice(0, -2), s, m, n]);
      return [[x], [x.ndim - 3]];
    }
    a = moveBatchAxis(axisSize, aBdim, 0, a);
    b = moveBatchAxis(axisSize, bBdim, 0, b);
    const x = bind1(Primitive.TriangularSolve, [a, b], { unitDiagonal });
    return [[x], [0]];
  },
  [Primitive.Cholesky]: lastDimsBatcher(Primitive.Cholesky, 2),
  [Primitive.LU]: lastDimsBatcher(Primitive.LU, 2, 3),
  [Primitive.Jit](axisSize, args, dims, { name, jaxpr }) {
    const newJaxpr = vmapJaxpr(jaxpr, axisSize, dims);
    const outs = bind(
      Primitive.Jit,
      [...newJaxpr.consts.map((c) => c.ref), ...args],
      {
        name: `${name}_vmap`,
        jaxpr: newJaxpr.jaxpr,
        numConsts: newJaxpr.consts.length,
      },
    );
    return [outs, rep(outs.length, 0)];
  },
  [Primitive.Scan](
    axisSize,
    args,
    dims,
    { jaxpr, numCarry, numConsts, length, reverse },
  ) {
    // vmap of scan: batch over independent scans
    //
    // Scan args layout: [...consts, ...initCarry, ...xs]
    // Body takes: [...consts, ...carry, ...x_slice] -> [...new_carry, ...y]
    //
    // For vmap, we run independent scans for each batch element.
    // Move all batch dimensions to position 0, then create a vmapped body.
    //
    // Key insight: the scan iterates over axis 0 of xs, but after vmap
    // the batch is also at axis 0. So we need to swap: batch becomes axis 1
    // for xs (after the scan length axis), or we move batch to axis 0 and
    // let xs have shape [batch, length, ...].
    //
    // Actually, the cleanest approach: move batch to axis 0 for carry/consts,
    // and axis 1 for xs (since axis 0 is the scan length). Then the body
    // needs to be vmapped to handle batch at axis 0 for carry/consts and
    // axis 0 for x_slice (since we slice off the length dimension).

    const numX = args.length - numConsts - numCarry;
    const numY = jaxpr.outs.length - numCarry;

    // Split args
    const consts = args.slice(0, numConsts);
    const initCarry = args.slice(numConsts, numConsts + numCarry);
    const xs = args.slice(numConsts + numCarry);

    const constDims = dims.slice(0, numConsts);
    const carryDims = dims.slice(numConsts, numConsts + numCarry);
    const xsDims = dims.slice(numConsts + numCarry);

    // Move batch dims to consistent positions:
    // - consts: batch at axis 0
    // - carry: batch at axis 0
    // - xs: batch at axis 1 (axis 0 is scan length)
    const movedConsts = consts.map((c, i) =>
      moveBatchAxis(axisSize, constDims[i], 0, c),
    );
    const movedCarry = initCarry.map((c, i) =>
      moveBatchAxis(axisSize, carryDims[i], 0, c),
    );
    // For xs, we need to move batch to axis 1 (after the length axis)
    // But first, if xsDims[i] is 0 (at length position), we need to handle carefully
    const movedXs = xs.map((x, i) => {
      if (xsDims[i] === null) {
        // Not mapped - broadcast batch dim at axis 1
        const newShape = [x.shape[0], axisSize, ...x.shape.slice(1)];
        return broadcast(x, newShape, [1]);
      } else if (xsDims[i] === 0) {
        // Batch at axis 0, but we need it at axis 1 (after length)
        // xs originally has shape [batch, length, ...], need [length, batch, ...]
        return moveaxis(x, 0, 1);
      } else {
        // Batch at some other axis - move to axis 1
        return moveBatchAxis(axisSize, xsDims[i], 1, x);
      }
    });

    // The body jaxpr takes: [consts, carry, x_slice]
    // After vmap, all these will have batch at axis 0:
    // - consts: [batch, ...const_shape]
    // - carry: [batch, ...carry_shape]
    // - x_slice: [batch, ...x_shape] (we sliced off the length dim from xs)
    //
    // Create the body dims for vmapJaxpr
    const bodyDims: (number | null)[] = [
      ...rep(numConsts, 0), // consts have batch at axis 0
      ...rep(numCarry, 0), // carry has batch at axis 0
      ...rep(numX, 0), // x_slice has batch at axis 0 (from xs axis 1)
    ];

    // Create vmapped body jaxpr
    const vmappedBody = vmapJaxpr(jaxpr, axisSize, bodyDims);

    // Build scan args with moved arrays
    const scanArgs = [
      ...vmappedBody.consts.map((c) => c.ref),
      ...movedConsts,
      ...movedCarry,
      ...movedXs,
    ];

    // Run the scan
    const results = bind(Primitive.Scan, scanArgs, {
      jaxpr: vmappedBody.jaxpr,
      numCarry,
      numConsts: vmappedBody.consts.length,
      length,
      reverse,
    });

    // The vmapped body outputs have batch at axis 0
    // Results layout: [carry_out..., ys...]
    // carry_out has shape [batch, ...carry_shape]
    // ys has shape [length, batch, ...y_shape]
    //
    // We want to return batch at axis 0, so:
    // - carry: already at axis 0
    // - ys: need to move batch from axis 1 to axis 0
    const carryOut = results.slice(0, numCarry);
    const ysOut = results.slice(numCarry);

    const movedYs = ysOut.map((y) => moveaxis(y, 1, 0));

    return [[...carryOut, ...movedYs], rep(numCarry + numY, 0)];
  },
};

const vmapJaxprCache = new Map<Jaxpr, Map<string, ClosedJaxpr>>();

function vmapJaxpr(
  jaxpr: Jaxpr,
  axisSize: number,
  dims: (number | null)[],
): ClosedJaxpr {
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
  const { jaxpr: newJaxpr } = makeJaxpr((args: Tracer[]) =>
    vmapFlat(jaxprAsFun(jaxpr), dims, args),
  )(inAvals);

  if (!vmapJaxprCache.has(jaxpr)) vmapJaxprCache.set(jaxpr, new Map());
  vmapJaxprCache.get(jaxpr)!.set(cacheKey, newJaxpr);
  return newJaxpr;
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
    let inAxesFlat: (number | null)[] = [];
    if (typeof inAxes === "number") {
      // If mapping over a single axis, just use it for all inputs.
      inAxesFlat = rep(argsFlat.length, inAxes);
    } else {
      // Allow either `null | number` (or undefined), or a tree structure
      // matching each input.
      for (let i = 0; i < args.length; i++) {
        if (inAxes[i] == null) {
          inAxesFlat.push(...rep(inTree.childTreedefs[i].size, null));
        } else if (typeof inAxes[i] === "number") {
          inAxesFlat.push(
            ...rep(inTree.childTreedefs[i].size, inAxes[i] as number),
          );
        } else {
          // Must be a tree structure.
          const [axesFlat, axesTreeDef] = treeFlatten(inAxes[i]);
          if (!inTree.childTreedefs[i].equals(axesTreeDef)) {
            throw new TreeMismatchError(
              "vmap",
              inTree.childTreedefs[i],
              axesTreeDef,
            );
          }
          inAxesFlat.push(...axesFlat);
        }
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
