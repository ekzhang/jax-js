/** @file Core library internals and interpreter stack, based on Autodidax. */

import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops";
import "@tensorflow/tfjs-core/dist/register_all_gradients";
import "@tensorflow/tfjs-backend-cpu";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-webgpu";
import { deepEqual, range, toposort, unzip2, zip } from "./utils";
import {
  JsTreeDef,
  flatten as treeFlatten,
  unflatten as treeUnflatten,
} from "./tree";
import { PPrint } from "./pprint";

tf.setBackend("cpu"); // TODO: support multiple devices, move arrays between devices

export enum DType {
  Float32 = "float32",
  Int32 = "int32",
  Bool = "bool",
  Complex64 = "complex64",
}

export enum Primitive {
  Add = "add",
  Mul = "mul",
  Neg = "neg",
  Sin = "sin",
  Cos = "cos",
  ReduceSum = "reduce_sum",
  Greater = "greater",
  Less = "less",
  Transpose = "transpose",
  Broadcast = "broadcast",
}

export function add(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Add, [x, y]);
}

export function mul(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Mul, [x, y]);
}

export function neg(x: TracerValue) {
  return bind1(Primitive.Neg, [x]);
}

export function sin(x: TracerValue) {
  return bind1(Primitive.Sin, [x]);
}

export function cos(x: TracerValue) {
  return bind1(Primitive.Cos, [x]);
}

export function greater(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Greater, [x, y]);
}

export function less(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Less, [x, y]);
}

export function transpose(x: TracerValue, perm?: number[]) {
  return bind1(Primitive.Transpose, [x], { perm });
}

export function broadcast(x: TracerValue, shape: number[], axes: number[]) {
  return bind1(Primitive.Broadcast, [x], { shape, axes });
}

export function reduceSum(x: TracerValue, axis?: number | number[]) {
  if (axis === null) {
    if (x instanceof Tracer) {
      axis = [...JsArray(x.shape.length).keys()];
    } else {
      axis = [];
    }
  }
  if (typeof axis === "number") {
    axis = [axis];
  }
  return bind1(Primitive.ReduceSum, [x], { axis });
}

function bind1(
  prim: Primitive,
  args: TracerValue[],
  params: Record<string, any> = {},
) {
  const [results] = bind(prim, args, params);
  return results;
}

type MainTrace = {
  level: number;
  traceType: new (main: MainTrace) => Trace; // Concrete Trace subclass.
  globalData: any | null;
};

let traceStack: MainTrace[] = [];
let dynamicTrace: MainTrace | null = null;

/**
 * Push an interpreter onto the trace stack. Use this like:
 * `using main = newMain(...);`
 */
function newMain(
  traceType: any,
  globalData: any | null = null,
): Disposable & MainTrace {
  const level = traceStack.length;
  const main = { level, traceType, globalData };
  traceStack.push(main);
  return Object.assign(main, {
    [Symbol.dispose]() {
      traceStack.pop();
    },
  });
}

/**
 * Set the current dynamic trace, which stashes the current interpreter stack
 * and acts temporarily as the bottom of the stack. Use this like:
 * `using _dynamic = newDynamic(main);`
 */
function newDynamic(main: MainTrace): Disposable {
  const prevDynamicTrace = dynamicTrace;
  dynamicTrace = main;
  return {
    [Symbol.dispose]() {
      dynamicTrace = prevDynamicTrace;
    },
  };
}

type TracerValue = Tracer | number | boolean;

abstract class Trace {
  constructor(readonly main: MainTrace) {}

  abstract pure(val: TracerValue): Tracer;
  abstract lift(val: Tracer): Tracer;

  abstract processPrimitive(
    primitive: Primitive,
    tracers: Tracer[],
    params: Record<string, any>,
  ): Tracer[];
}

interface AbstractValue {
  shape: number[];
  dtype: DType;

  _neg: (x: Tracer) => Tracer;
  _add: (x: Tracer, y: Tracer) => Tracer;
  _mul: (x: Tracer, y: Tracer) => Tracer;
  _gt: (x: Tracer, y: Tracer) => Tracer;
  _lt: (x: Tracer, y: Tracer) => Tracer;
}

abstract class Tracer {
  readonly _trace: Trace;

  constructor(trace: Trace) {
    this._trace = trace;
  }

  abstract get aval(): AbstractValue;
  abstract toString(): string;

  get shape() {
    return this.aval.shape;
  }

  fullLower(): Tracer {
    return this; // default implementation
  }

  // These types aren't technically correct since they don't account for the
  // fact that tracers can be lifted to different levels. But they simplify the
  // API visible to users.
  neg() {
    return this.aval._neg(this) as this;
  }
  add(other: this | TracerValue) {
    return this.aval._add(this, pureArray(other)) as this;
  }
  mul(other: this | TracerValue) {
    return this.aval._mul(this, pureArray(other)) as this;
  }
  gt(other: this | TracerValue) {
    return this.aval._gt(this, pureArray(other)) as this;
  }
  lt(other: this | TracerValue) {
    return this.aval._lt(this, pureArray(other)) as this;
  }
}

export function ndim(x: TracerValue) {
  if (x instanceof Tracer) {
    return x.shape.length;
  } else {
    return 0;
  }
}

const JsArray = globalThis.Array;

class ShapedArray implements AbstractValue {
  readonly arrayAbstractionLevel: number = 1;

  constructor(
    readonly shape: number[],
    readonly dtype: DType,
  ) {}

  static fromAval(aval: AbstractValue) {
    return new ShapedArray(aval.shape, aval.dtype);
  }

  get ndim() {
    return this.shape.length;
  }

  // See note about primitive wrappers with fudged types.
  _neg = neg as any;
  _add = add as any;
  _mul = mul as any;
  _gt = greater as any;
  _lt = less as any;

  strShort() {
    return `${this.dtype}[${this.shape.join(",")}]`;
  }

  equals(other: ShapedArray) {
    return (
      this === other ||
      (this.constructor === other.constructor &&
        this.ndim === other.ndim &&
        this.shape.every((d, i) => d === other.shape[i]))
    );
  }
}

class ConcreteArray extends ShapedArray {
  readonly arrayAbstractionLevel: number = 2;

  constructor(readonly val: tf.Tensor) {
    super(val.shape, val.dtype as any);
  }
}

/**
 * Equivalent to `jnp.Array` from JAX, a tensor type.
 *
 * Not to be confused with the JavaScript "Array" constructor. Avoid importing
 * this into your code's namespace if you're already using the JavaScript
 * "Array" type by name.
 */
export class Array extends Tracer {
  readonly dtype: DType;

  constructor(readonly data: tf.Tensor) {
    super(baseArrayTrace);
    if (Object.values(DType).includes(data.dtype as any)) {
      this.dtype = data.dtype as DType;
    } else {
      throw new TypeError(`Unsupported dtype: ${data.dtype}`);
    }
  }

  get aval(): AbstractValue {
    return new ConcreteArray(this.data);
  }

  /** Return a simple string representation of the array's dimensions. */
  toString(): string {
    return `Array:${this.dtype}[${this.data.shape.join(",")}]`;
  }

  /** Convert this array into a JavaScript object (blocking). */
  js() {
    return this.data.arraySync();
  }

  /** Convert this array into a JavaScript object, asynchronously. */
  async jsAsync() {
    return await this.data.array();
  }
}

/** If x is a value, lift it into an array, otherwise leave it be. */
function pureArray(x: TracerValue): Tracer {
  if (x instanceof Tracer) {
    return x;
  } else {
    return new Array(tf.scalar(x));
  }
}

function getAval(x: TracerValue): AbstractValue {
  if (x instanceof Tracer) {
    return x.aval;
  } else if (typeof x === "boolean" || typeof x === "number") {
    return new ConcreteArray(tf.scalar(x));
  } else {
    throw new TypeError(`Unknown value: ${x}`);
  }
}

function bind(
  prim: Primitive,
  args: TracerValue[],
  params: Record<string, any> = {},
) {
  const topTrace = findTopTrace(args);
  const tracers = args.map((arg) => fullRaise(topTrace, arg));
  const outs = topTrace.processPrimitive(prim, tracers, params);
  // console.info(`processing rule for ${prim} on ${tracers} and got ${outs}`);
  return outs.map((out) => out.fullLower());
}

function findTopTrace(xs: TracerValue[]): Trace {
  let topMain: MainTrace = traceStack[0];
  for (const x of xs) {
    if (x instanceof Tracer && x._trace.main.level > topMain.level) {
      topMain = x._trace.main;
    }
  }
  if (dynamicTrace && dynamicTrace.level > topMain.level) {
    topMain = dynamicTrace;
  }
  return new topMain.traceType(topMain);
}

function fullRaise(trace: Trace, val: TracerValue): Tracer {
  if (!(val instanceof Tracer)) {
    // remember to assert type(val) in jax_types
    return trace.pure(val);
  }
  const level = trace.main.level;
  if (Object.is(val._trace.main, trace.main)) {
    return val;
  } else if (val._trace.main.level < level) {
    return trace.lift(val);
  } else if (val._trace.main.level > level) {
    throw new Error(
      `Can't lift Tracer level ${val._trace.main.level} to level ${level}`,
    );
  } else {
    throw new Error(`Different traces at same level: ${val._trace}, ${trace}.`);
  }
}

class EvalTrace extends Trace {
  // No boxing in Tracers needed.
  pure = (x: TracerValue) => pureArray(x);
  lift = (x: Tracer) => x;

  processPrimitive(
    primitive: Primitive,
    tracers: Array[],
    params: Record<string, any>,
  ): Tracer[] {
    return implRules[primitive](tracers, params);
  }
}

// Special bottom of the stack.
traceStack.push({ level: 0, traceType: EvalTrace, globalData: null });
const baseArrayTrace = new EvalTrace(traceStack[0]);

type ImplRule = (tracers: Array[], params: any) => Array[];

const implRules: Record<Primitive, ImplRule> = {
  [Primitive.Add]([x, y]) {
    return [new Array(tf.add(x.data, y.data))];
  },
  [Primitive.Mul]([x, y]) {
    return [new Array(tf.mul(x.data, y.data))];
  },
  [Primitive.Neg]([x]) {
    return [new Array(tf.neg(x.data))];
  },
  [Primitive.Sin]([x]) {
    return [new Array(tf.sin(x.data))];
  },
  [Primitive.Cos]([x]) {
    return [new Array(tf.cos(x.data))];
  },
  [Primitive.ReduceSum]([x], { axis }: { axis: number[] }) {
    return [new Array(tf.sum(x.data, axis))];
  },
  [Primitive.Greater]([x, y]) {
    return [new Array(tf.greater(x.data, y.data))];
  },
  [Primitive.Less]([x, y]) {
    return [new Array(tf.less(x.data, y.data))];
  },
  [Primitive.Transpose]([x], { perm }: { perm?: number[] }) {
    return [new Array(tf.transpose(x.data, perm))];
  },
  [Primitive.Broadcast](
    [x],
    { shape, axes }: { shape: number[]; axes: number[] },
  ) {
    let data = x.data;
    for (const axis of axes.toSorted()) {
      data = tf.expandDims(data, axis);
    }
    return [new Array(tf.broadcastTo(data, shape))];
  },
};

function zerosLike(val: TracerValue): Array {
  const aval = getAval(val);
  return zeros(aval.shape, aval.dtype);
}

function zeros(shape: number[], dtype: DType): Array {
  return new Array(tf.zeros(shape, dtype));
}

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
    return `JVPTracer(${this.primal}, ${this.tangent})`;
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
  primal: Tracer[],
  tangents: Tracer[],
  params: any,
) => [Tracer[], Tracer[]];

const jvpRules: Partial<Record<Primitive, JvpRule>> = {
  [Primitive.Add]([x, y], [dx, dy]) {
    return [[x.add(y)], [dx.add(dy)]];
  },
  [Primitive.Mul]([x, y], [dx, dy]) {
    return [[x.mul(y)], [x.mul(dy).add(dx.mul(y))]];
  },
  [Primitive.Neg]([x], [dx]) {
    return [[x.neg()], [dx.neg()]];
  },
  [Primitive.Sin]([x], [dx]) {
    return [[sin(x)], [cos(x).mul(dx)]];
  },
  [Primitive.Cos]([x], [dx]) {
    return [[cos(x)], [neg(sin(x)).mul(dx)]];
  },
  [Primitive.ReduceSum]([x], [dx], { axis }: { axis: number[] }) {
    return [[reduceSum(x, axis)], [reduceSum(dx, axis)]];
  },
  [Primitive.Greater]([x, y], _tangents) {
    const outPrimal = greater(x, y);
    return [[outPrimal], [zerosLike(outPrimal)]];
  },
  [Primitive.Less]([x, y], _tangents) {
    const outPrimal = less(x, y);
    return [[outPrimal], [zerosLike(outPrimal)]];
  },
  // TODO: transpose, broadcast
};

function jvpFlat(
  f: (...x: TracerValue[]) => TracerValue[],
  primals: TracerValue[],
  tangents: TracerValue[],
): [Tracer[], Tracer[]] {
  using main = newMain(JVPTrace);
  // console.info("creating new jvp main", traceStack);
  const trace = new JVPTrace(main);
  const tracersIn = zip(primals, tangents).map(
    ([x, t]) => new JVPTracer(trace, pureArray(x), pureArray(t)),
  );
  const outs = f(...tracersIn);
  const tracersOut = outs.map((out) => fullRaise(trace, out) as JVPTracer);
  return unzip2(tracersOut.map((t) => [t.primal, t.tangent]));
}

export function jvp(
  f: (...x: any[]) => any,
  primals: any[],
  tangents: any[],
): [any, any] {
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
  return [primalsOut, tangentsOut];
}

class TreeMismatchError extends TypeError {
  constructor(where: string, left: JsTreeDef, right: JsTreeDef) {
    super(`Mismatched tree structures in ${where}: ${left} != ${right}`);
  }
}

function flattenFun(
  f: any,
  inTree: JsTreeDef,
): [any, { value: JsTreeDef | undefined }] {
  const store: { value: JsTreeDef | undefined } = { value: undefined };
  const flatFun = (...argsFlat: any[]) => {
    const pytreeArgs = treeUnflatten(inTree, argsFlat);
    const out = f(...pytreeArgs);
    const [outFlat, outTree] = treeFlatten(out);
    store.value = outTree;
    return outFlat;
  };
  return [flatFun, store];
}

// vmap() implementation begins

function mappedAval(batchDim: number, aval: AbstractValue) {
  const shape = [...aval.shape];
  shape.splice(batchDim, 1);
  return new ShapedArray(shape, aval.dtype);
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

/** Move one axis to a different index. */
export function moveaxis(x: TracerValue, src: number, dst: number) {
  const t = pureArray(x);
  const perm = [...JsArray(t.shape.length).keys()];
  perm.splice(src, 1);
  perm.splice(dst, 0, src);
  return transpose(t, perm);
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
    return `BatchTracer(${this.val}, ${this.batchDim})`;
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
  valsIn: Tracer[],
  dimsIn: (number | null)[],
  params: any,
) => [Tracer[], (number | null)[]];

function handleScalarBroadcasting(nd: number, x: Tracer, d: number | null) {
  if (d === null || nd === ndim(x)) {
    return x;
  } else {
    const axes = range(ndim(x), nd);
    const shape = [...x.shape, ...axes.map(() => 1)];
    return broadcast(x, shape, axes);
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
  [Primitive.ReduceSum](
    axisSize: number,
    [x]: Tracer[],
    [xBdim]: (number | null)[],
    { axis }: { axis: number[] },
  ): ReturnType<VmapRule> {
    if (xBdim === null) {
      return [[reduceSum(x, axis)], [null]];
    }
    const newAxis = axis.map((ax) => ax + (xBdim <= ax ? 1 : 0));
    const outBdim = xBdim - axis.filter((ax) => ax < xBdim).length;
    return [[reduceSum(x, newAxis)], [outBdim]];
  },
};

function vmapFlat(
  f: (...x: TracerValue[]) => TracerValue[],
  inAxes: number[],
  args: TracerValue[],
): Tracer[] {
  let axisSize: number | undefined = undefined;
  for (let i = 0; i < args.length; i++) {
    if (inAxes[i] !== null) {
      const arg = args[i];
      if (!(arg instanceof Tracer)) {
        throw new TypeError("vmap requires Tracer argument for mapped axes");
      }
      const size = arg.shape[inAxes[i]];
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
    // console.info("creating new vmap main", traceStack);
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
  return vmap(pushfwd, [0])(new Array(tf.eye(size)));
}

/** Variable in a Jaxpr expression. */
export class Var {
  static nextId = 1; // For debugging, since JavaScript has no id() function like Python.

  static resetIdCounter() {
    Var.nextId = 1;
  }

  readonly id: number;
  readonly aval: ShapedArray;

  constructor(aval: ShapedArray) {
    this.id = Var.nextId++;
    this.aval = aval;
  }

  get name() {
    return `v_${this.id}`;
  }

  toString(): string {
    return `${this.name}:${this.aval.strShort()}`;
  }
}

/** Literal in a Jaxpr expression. */
export class Lit {
  readonly val: Array;
  readonly aval: ShapedArray;

  constructor(val: Array | number | boolean) {
    this.aval = ShapedArray.fromAval(getAval(val));
    const ar = pureArray(val);
    if (ndim(ar) !== 0 || !(ar instanceof Array)) {
      throw new TypeError("Lit only supports scalar Array values");
    }
    this.val = ar;
  }

  get value(): number | boolean {
    return this.val.data.dataSync()[0];
  }
}

export type Atom = Var | Lit;

function atomIsLit(atom: Atom, literal?: number | boolean) {
  return (
    atom instanceof Lit && (literal === undefined || atom.value === literal)
  );
}

/** A single statement / binding in a Jaxpr, in ANF form. */
export class JaxprEqn {
  constructor(
    readonly primitive: Primitive,
    readonly inputs: Atom[],
    readonly params: Record<string, any>,
    readonly outBinders: Var[],
  ) {}

  pprint(usedVars?: Set<Var>): PPrint {
    const lhs = PPrint.pp(
      this.outBinders
        .map((v) => (!usedVars || usedVars.has(v) ? v : "_"))
        .join(" "),
    );
    let rhs = PPrint.pp(this.primitive);
    // pprint params
    const paramsList = Object.entries(this.params).map(([k, v]) =>
      PPrint.pp(`${k}=${v}`),
    );
    if (paramsList.length > 0) {
      rhs = rhs
        .stack(PPrint.pp(" [ "))
        .stack(PPrint.prototype.concat(...paramsList))
        .stack(PPrint.pp(" ] "));
    } else {
      rhs = rhs.stack(PPrint.pp(" "));
    }
    // pprint inputs (vars and literals)
    rhs = rhs.stack(
      PPrint.pp(
        this.inputs
          .map((x) => (x instanceof Var ? x.name : JSON.stringify(x.val.js())))
          .join(" "),
      ),
    );
    return lhs.stack(PPrint.pp(" = ")).stack(rhs);
  }

  toString(): string {
    return this.pprint().toString();
  }
}

/** Typed intermediate representation for traced computations. */
export class Jaxpr {
  constructor(
    readonly inBinders: Var[],
    readonly eqns: JaxprEqn[],
    readonly outs: Atom[],
  ) {}

  pprint(): PPrint {
    const usedVars = new Set<Var>(
      [...this.outs, ...this.eqns.flatMap((eqn) => eqn.inputs)].filter(
        (x) => x instanceof Var,
      ),
    );
    const inBinders = this.inBinders.map((v) => v.toString()).join(", ");
    const eqns = PPrint.prototype.concat(
      ...this.eqns.map((e) => e.pprint(usedVars)),
    );
    const outs = this.outs
      .map((x) => (x instanceof Var ? x.name : x.val.js()))
      .join(", ");
    return PPrint.pp(`{ lambda ${inBinders} .`).concat(
      PPrint.pp("let ")
        .stack(eqns)
        .concat(PPrint.pp(`in ( ${outs} ) }`))
        .indent(2),
    );
  }

  toString(): string {
    return this.pprint().toString();
  }

  /**
   * Produce a simplified Jaxpr with basic optimizations applied.
   *  - Trim away unused variables.
   *  - Fold away *1, *0, or +0 operations against literals.
   */
  simplify(): Jaxpr {
    const context = new Map<Var, Atom>();
    const newEqns: JaxprEqn[] = [];
    for (const e of this.eqns) {
      const inputs = e.inputs.map((x) =>
        x instanceof Var ? (context.get(x) ?? x) : x,
      );
      const eqn = new JaxprEqn(e.primitive, inputs, e.params, e.outBinders);

      if (eqn.primitive === Primitive.Add) {
        const [a, b] = inputs;
        const c = eqn.outBinders[0];
        if (atomIsLit(a, 0)) {
          context.set(c, b);
        } else if (atomIsLit(b, 0)) {
          context.set(c, a);
        } else if (atomIsLit(a) && atomIsLit(b)) {
          context.set(c, new Lit((a as any).value + (b as any).value));
        } else {
          newEqns.push(eqn);
        }
      } else if (eqn.primitive === Primitive.Mul) {
        const [a, b] = inputs;
        const c = eqn.outBinders[0];
        // TODO: handle *0 once we have shaped zero arrays
        if (atomIsLit(a, 1)) {
          context.set(c, b);
        } else if (atomIsLit(b, 1)) {
          context.set(c, a);
        } else if (atomIsLit(a) && atomIsLit(b)) {
          context.set(c, new Lit((a as any).value * (b as any).value));
        } else {
          newEqns.push(eqn);
        }
      } else {
        newEqns.push(eqn);
      }
    }

    const outs = this.outs.map((x) =>
      x instanceof Var ? (context.get(x) ?? x) : x,
    );

    // Skip unused output variables
    const usedVars = new Set<Var>(outs.filter((x) => x instanceof Var));
    const liveEqns: JaxprEqn[] = [];
    for (let i = newEqns.length - 1; i >= 0; i--) {
      const eqn = newEqns[i];
      if (eqn.outBinders.some((v) => usedVars.has(v))) {
        liveEqns.push(eqn);
        for (const v of eqn.inputs) {
          if (v instanceof Var) {
            usedVars.add(v);
          }
        }
      }
    }

    return new Jaxpr(this.inBinders, liveEqns.reverse(), outs);
  }
}

export class JaxprType {
  constructor(
    readonly inTypes: ShapedArray[],
    readonly outTypes: ShapedArray[],
  ) {}

  toString(): string {
    const inTypes = this.inTypes.map((aval) => aval.strShort()).join(", ");
    const outTypes = this.outTypes.map((aval) => aval.strShort()).join(", ");
    return `(${inTypes}) -> (${outTypes})`;
  }
}

function typecheckJaxpr(jaxpr: Jaxpr): JaxprType {
  const env = new Set<Var>();

  for (const v of jaxpr.inBinders) {
    if (env.has(v)) {
      throw new TypeError(`Duplicate variable binding: ${v}`);
    }
    env.add(v);
  }

  for (const eqn of jaxpr.eqns) {
    const inTypes = eqn.inputs.map((x) => typecheckAtom(env, x));
    const outTypes = abstractEvalRules[eqn.primitive](inTypes, eqn.params);
    for (const [outBinder, outType] of zip(eqn.outBinders, outTypes)) {
      if (!outType.equals(outBinder.aval)) {
        throw new TypeError(
          `Output binder type mismatch in ${eqn.primitive}: ${outBinder} vs ${outType}`,
        );
      }
      if (env.has(outBinder)) {
        throw new TypeError(`Duplicate variable binding: ${outBinder}`);
      }
      env.add(outBinder);
    }
  }

  const inTypes = jaxpr.inBinders.map((v) => v.aval);
  const outTypes = jaxpr.outs.map((x) => typecheckAtom(env, x));
  return new JaxprType(inTypes, outTypes);
}

function typecheckAtom(env: Set<Var>, x: Atom): ShapedArray {
  if (x instanceof Var) {
    if (!env.has(x)) {
      throw new Error(`Unknown variable: ${x}`);
    }
    return x.aval;
  } else if (x instanceof Lit) {
    return x.aval;
  } else {
    throw new TypeError(`Invalid atom type: ${x}`);
  }
}

/** Evaluate a Jaxpr on an array of inputs. */
function evalJaxpr(jaxpr: Jaxpr, args: Tracer[]): Tracer[] {
  const env = new Map<Var, Tracer>();

  const read = (x: Atom) => (x instanceof Var ? env.get(x)! : x.val);
  const write = (v: Var, val: Tracer) => {
    if (env.has(v)) throw new Error(`Variable already bound: ${v}`);
    env.set(v, val);
  };

  for (const [v, arg] of zip(jaxpr.inBinders, args)) write(v, arg);
  for (const eqn of jaxpr.eqns) {
    const inVals = eqn.inputs.map(read);
    const outVals = bind(eqn.primitive, inVals, eqn.params);
    for (const [v, val] of zip(eqn.outBinders, outVals)) write(v, val);
  }
  return jaxpr.outs.map(read);
}

function jaxprAsFun(jaxpr: Jaxpr) {
  return (...args: Tracer[]) => evalJaxpr(jaxpr, args);
}

/** Tracer that records its operations to dynamically construct a Jaxpr. */
class JaxprTracer extends Tracer {
  constructor(
    trace: Trace,
    readonly aval: ShapedArray,
  ) {
    super(trace);
  }

  toString(): string {
    return `JaxprTracer(${this.aval.strShort()})`;
  }
}

/** Analogous to the 'DynamicJaxprTrace' class in JAX. */
class JaxprTrace extends Trace {
  /** Register a Jaxpr argument with a given shape and return the tracer. */
  newArg(aval: ShapedArray): JaxprTracer {
    aval = ShapedArray.fromAval(aval);
    const tracer = this.builder.newTracer(this, aval);
    this.builder.addVar(tracer);
    return tracer;
  }

  /** Register a constant / literal in this Jaxpr. */
  getOrMakeConstTracer(val: TracerValue): JaxprTracer {
    let tracer = this.builder.constTracers.get(val);
    if (tracer === undefined) {
      tracer = this.builder.newTracer(this, ShapedArray.fromAval(getAval(val)));
      this.builder.addConst(tracer, pureArray(val));
    }
    return tracer;
  }
  pure = this.getOrMakeConstTracer;
  lift = this.getOrMakeConstTracer;

  processPrimitive(
    primitive: Primitive,
    tracers: JaxprTracer[],
    params: Record<string, any>,
  ): JaxprTracer[] {
    const avalsIn = tracers.map((t) => t.aval);
    const avalsOut = abstractEvalRules[primitive](avalsIn, params);
    const outTracers = avalsOut.map((aval) =>
      this.builder.newTracer(this, aval),
    );
    this.builder.addEqn(
      new JaxprEqn(
        primitive,
        tracers.map((t) => this.builder.getVar(t)),
        params,
        outTracers.map((t) => this.builder.addVar(t)),
      ),
    );
    return outTracers;
  }

  get builder(): JaxprBuilder {
    return this.main.globalData;
  }
}

/** Incrementally constructs a Jaxpr. */
class JaxprBuilder {
  eqns: JaxprEqn[] = [];
  tracerToVar: Map<JaxprTracer, Var> = new Map();
  constTracers: Map<TracerValue, JaxprTracer> = new Map(); // already-seen value -> tracer
  constVals: Map<Var, Tracer> = new Map(); // var -> const value
  tracers: JaxprTracer[] = [];

  newTracer(trace: JaxprTrace, aval: ShapedArray): JaxprTracer {
    const tracer = new JaxprTracer(trace, aval);
    this.tracers.push(tracer);
    return tracer;
  }

  addEqn(eqn: JaxprEqn) {
    this.eqns.push(eqn);
  }

  addVar(tracer: JaxprTracer): Var {
    if (this.tracerToVar.has(tracer)) {
      throw new Error(`Tracer was added as variable twice: ${tracer}`);
    }
    const v = new Var(tracer.aval);
    this.tracerToVar.set(tracer, v);
    return v;
  }

  getVar(tracer: JaxprTracer): Var {
    const v = this.tracerToVar.get(tracer);
    if (v === undefined) {
      throw new Error(`Could not find variable for tracer: ${tracer}`);
    }
    return v;
  }

  addConst(tracer: JaxprTracer, val: Tracer) {
    const v = this.addVar(tracer);
    this.constTracers.set(val, tracer);
    this.constVals.set(v, val);
    return v;
  }

  build(
    inTracers: JaxprTracer[],
    outTracers: JaxprTracer[],
  ): { jaxpr: Jaxpr; consts: Tracer[] } {
    // Initially, concatenate the constants as the first few inputs.
    let [constVars, consts] = unzip2(this.constVals.entries());
    const t2v = this.getVar.bind(this); // Maps tracer to value.
    const inBinders = [...constVars, ...inTracers.map(t2v)];
    const outVars = outTracers.map(t2v);
    let jaxpr = new Jaxpr(inBinders, this.eqns, outVars);

    // Inline any scalar constants as Lit and remove from the input list.
    typecheckJaxpr(jaxpr);
    [jaxpr, consts] = _inlineLiterals(jaxpr, consts);
    return { jaxpr, consts };
  }
}

function _inlineLiterals(jaxpr: Jaxpr, consts: Tracer[]): [Jaxpr, Tracer[]] {
  const literals = new Map<Atom, Lit>();
  const constBinders: Var[] = [];
  const newConsts: Tracer[] = [];

  for (let i = 0; i < consts.length; i++) {
    if (ndim(consts[i]) === 0 && consts[i] instanceof Array) {
      literals.set(jaxpr.inBinders[i], new Lit(consts[i] as any));
    } else {
      constBinders.push(jaxpr.inBinders[i]);
      newConsts.push(consts[i]);
    }
  }

  const newEqns: JaxprEqn[] = jaxpr.eqns.map(
    (eqn) =>
      new JaxprEqn(
        eqn.primitive,
        eqn.inputs.map((x) => literals.get(x) ?? x),
        eqn.params,
        eqn.outBinders,
      ),
  );
  const newOuts = jaxpr.outs.map((x) => literals.get(x) ?? x);
  const newJaxpr = new Jaxpr(
    [...constBinders, ...jaxpr.inBinders.slice(consts.length)],
    newEqns,
    newOuts,
  );
  typecheckJaxpr(newJaxpr); // Double-check for sanity.
  return [newJaxpr, newConsts];
}

type AbstractEvalRule = (shapes: ShapedArray[], params: any) => ShapedArray[];

/**
 * Implements a NumPy-style generalized broadcast rule on two array shapes.
 *
 * "When operating on two arrays, NumPy compares their shapes element-wise. It
 * starts with the trailing (i.e. rightmost) dimension and works its way left.
 * Two dimensions are compatible when:
 *   1. they are equal, or
 *   2. one of them is 1."
 *
 * Throws a TypeError if the broadcast is not possible.
 *
 * <https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules>
 */
function generalBroadcast(a: number[], b: number[]): number[] {
  const out: number[] = [];
  let i = a.length - 1;
  let j = b.length - 1;
  for (; i >= 0 && j >= 0; i--, j--) {
    const x = a[i];
    const y = b[j];
    if (x === y) {
      out.push(x);
    } else if (x === 1) {
      out.push(y);
    } else if (y === 1) {
      out.push(x);
    } else {
      throw new TypeError(`Incompatible array broadcast shapes: ${a} vs ${b}`);
    }
  }
  for (; i >= 0; i--) {
    out.push(a[i]);
  }
  for (; j >= 0; j--) {
    out.push(b[j]);
  }
  return out.reverse();
}

function binopAbstractEval([x, y]: ShapedArray[]) {
  if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray)) {
    throw new TypeError("binopAbstractEval expects ShapedArray inputs");
  }
  if (x.dtype !== y.dtype) {
    // TODO: Relax this restriction on dtype equality, or add automatic casts.
    throw new TypeError(`Mismatched dtypes: ${x.dtype} vs ${y.dtype}`);
  }
  return [new ShapedArray(generalBroadcast(x.shape, y.shape), x.dtype)];
}

function compareAbstractEval([x, y]: ShapedArray[]) {
  if (!(x instanceof ShapedArray) || !(y instanceof ShapedArray)) {
    throw new TypeError("binopAbstractEval expects ShapedArray inputs");
  }
  if (x.dtype !== y.dtype) {
    // TODO: Relax this restriction on dtype equality, or add automatic casts.
    throw new TypeError(`Mismatched dtypes: ${x.dtype} vs ${y.dtype}`);
  }
  return [new ShapedArray(generalBroadcast(x.shape, y.shape), DType.Bool)];
}

function vectorizedUnopAbstractEval([x]: ShapedArray[]) {
  return [ShapedArray.fromAval(x)];
}

const abstractEvalRules: Record<Primitive, AbstractEvalRule> = {
  [Primitive.Add]: binopAbstractEval,
  [Primitive.Mul]: binopAbstractEval,
  [Primitive.Neg]: vectorizedUnopAbstractEval,
  [Primitive.Sin]: vectorizedUnopAbstractEval,
  [Primitive.Cos]: vectorizedUnopAbstractEval,
  [Primitive.ReduceSum]([x], { axis }: { axis: number[] }) {
    const axisSet = new Set(axis);
    const newShape = x.shape.filter((_, i) => !axisSet.has(i));
    return [new ShapedArray(newShape, x.dtype)];
  },
  [Primitive.Greater]: compareAbstractEval,
  [Primitive.Less]: compareAbstractEval,
  [Primitive.Transpose]([x], { perm }: { perm?: number[] }) {
    if (perm === undefined) {
      perm = [...JsArray(x.shape.length).keys()].reverse();
    }
    return [
      new ShapedArray(
        perm.map((i) => x.shape[i]),
        x.dtype,
      ),
    ];
  },
  [Primitive.Broadcast](
    [x],
    { shape, axes }: { shape: number[]; axes: number[] },
  ) {
    return [new ShapedArray(shape, x.dtype)];
  },
};

export function makeJaxpr(
  f: (...args: any[]) => any,
): (...argsIn: any) => { jaxpr: Jaxpr; consts: Tracer[]; treedef: JsTreeDef } {
  return (...argsIn) => {
    const [avalsIn, inTree] = treeFlatten(argsIn);
    const [fFlat, outTree] = flattenFun(f, inTree);

    Var.resetIdCounter(); // Reset the counter for each new Jaxpr trace.
    const builder = new JaxprBuilder();
    using main = newMain(JaxprTrace, builder);
    using _dynamic = newDynamic(main);

    const trace = new JaxprTrace(main);
    const tracersIn = avalsIn.map((aval) =>
      trace.newArg(typeof aval === "object" ? aval : pureArray(aval)),
    );
    const outs = fFlat(...tracersIn);
    const tracersOut = outs.map(
      (out: Tracer) => fullRaise(trace, out) as JaxprTracer,
    );
    const { jaxpr, consts } = builder.build(tracersIn, tracersOut);

    if (outTree.value === undefined) {
      throw new Error("outTree was not set in makeJaxpr");
    }
    return { jaxpr: jaxpr.simplify(), consts, treedef: outTree.value };
  };
}

// linearize, partial evaluation

/** Array value that can either be known or unknown. */
class PartialVal {
  constructor(
    readonly val: Tracer | null,
    readonly aval: ShapedArray,
  ) {}

  static known(val: Tracer): PartialVal {
    return new PartialVal(val, ShapedArray.fromAval(val.aval));
  }

  static unknown(aval: AbstractValue): PartialVal {
    return new PartialVal(null, ShapedArray.fromAval(aval));
  }

  get isKnown(): boolean {
    return this.val !== null;
  }

  toString(): string {
    return this.val ? this.val.toString() : this.aval.strShort();
  }
}

function partialEvalFlat(
  f: (...args: any[]) => any,
  pvalsIn: PartialVal[],
): { jaxpr: Jaxpr; pvalsOut: PartialVal[]; consts: Tracer[] } {
  const main = newMain(PartialEvalTrace);
  const trace = new PartialEvalTrace(main);
  const tracersIn = pvalsIn.map((pval) => trace.newArg(pval));
  const outs = f(...tracersIn);
  const tracersOut: PartialEvalTracer[] = outs.map((out: TracerValue) =>
    fullRaise(trace, out),
  );
  const pvalsOut = tracersOut.map((t) => t.pval);
  const unknownTracersIn = tracersIn.filter((t) => !t.pval.isKnown);
  const unknownTracersOut = tracersOut.filter((t) => !t.pval.isKnown);
  const { jaxpr, consts } = partialEvalGraphToJaxpr(
    unknownTracersIn,
    unknownTracersOut,
  );
  return { jaxpr, pvalsOut, consts };
}

/**
 * Helper function with shared Jaxpr logic between linearize and vjp.
 *
 * Internally, vjp() looks very similar to linearize() but returns a function
 * evaluating the "transposed" linearized Jaxpr, pulling back cotangents instead
 * of pushing forward tangents.
 */
function linearizeFlatUtil(
  f: (...args: any[]) => any,
  primalsIn: Tracer[],
): { primalsOut: Tracer[]; jaxpr: Jaxpr; consts: Tracer[] } {
  const pvalsIn = [
    ...primalsIn.map(PartialVal.known),
    ...primalsIn.map((t) => PartialVal.unknown(t.aval)),
  ];
  const fJvp = (...x: Tracer[]) => {
    // Args contain both primals and tangents, concatenated.
    const k = x.length / 2;
    const [primalsOut, tangentsOut] = jvp(f, x.slice(0, k), x.slice(k, 2 * k));
    return [...primalsOut, ...tangentsOut];
  };
  const { jaxpr, pvalsOut, consts } = partialEvalFlat(fJvp, pvalsIn);
  const primalPvals = pvalsOut.slice(0, pvalsOut.length / 2);
  if (!primalPvals.every((pval) => pval.isKnown)) {
    throw new TypeError(
      "Not all primal values are known after partial evaluation",
    );
  }
  const primalsOut = primalPvals.map((pval) => pval.val!);
  return { primalsOut, jaxpr, consts };
}

function linearizeFlat(
  f: (...args: any[]) => any,
  primalsIn: Tracer[],
): [Tracer[], (...args: any[]) => any] {
  const { primalsOut, jaxpr, consts } = linearizeFlatUtil(f, primalsIn);
  const fLin = (...tangents: Tracer[]) =>
    evalJaxpr(jaxpr, [...consts, ...tangents]);
  return [primalsOut, fLin];
}

export function linearize(
  f: (...primals: any[]) => any,
  ...primalsIn: any[]
): [any, (...tangents: any[]) => any] {
  const [primalsInFlat, inTree] = treeFlatten(primalsIn);
  const [fFlat, outTree] = flattenFun(f, inTree);
  const [primalsOutFlat, fLinFlat] = linearizeFlat(
    fFlat,
    primalsInFlat.map(pureArray),
  );
  if (outTree.value === undefined) {
    throw new Error("outTree was not set in linearize");
  }
  const primalsOut = treeUnflatten(outTree.value, primalsOutFlat);
  const fLin = (...tangentsIn: any[]) => {
    const [tangentsInFlat, inTree2] = treeFlatten(tangentsIn);
    if (!inTree.equals(inTree2)) {
      throw new TreeMismatchError("linearize", inTree, inTree2);
    }
    const tangentsOutFlat = fLinFlat(...tangentsInFlat.map(pureArray));
    return treeUnflatten(outTree.value!, tangentsOutFlat);
  };
  return [primalsOut, fLin];
}

// Used in PartialEvalTracer to track recipes for "unknown" partial vals.
type JaxprRecipe =
  | {
      type: "LambdaBinding";
    }
  | {
      // Note: Not really a constant, actually just a "known" value translated
      // into unknown for abstract evaluation rules.
      type: "Const";
      val: Tracer;
    }
  | {
      type: "JaxprEqn";
      prim: Primitive;
      tracersIn: PartialEvalTracer[];
      params: Record<string, any>;
      avalsOut: ShapedArray[];
      tracerRefsOut: WeakRef<PartialEvalTracer>[];
    };

class PartialEvalTracer extends Tracer {
  constructor(
    trace: Trace,
    readonly pval: PartialVal,
    readonly recipe: JaxprRecipe | null,
  ) {
    super(trace);
  }

  get aval(): AbstractValue {
    return this.pval.aval;
  }

  fullLower(): Tracer {
    if (this.pval.isKnown) {
      return this.pval.val!;
    }
    return this;
  }

  toString(): string {
    if (!this.recipe) {
      return `PartialEvalTracer(${this.pval})`;
    } else {
      return `PartialEvalTracer<${this.recipe.type}>(${this.pval})`;
    }
  }
}

class PartialEvalTrace extends Trace {
  newArg(pval: PartialVal) {
    return new PartialEvalTracer(this, pval, { type: "LambdaBinding" });
  }

  pure(val: TracerValue): Tracer {
    return new PartialEvalTracer(this, PartialVal.known(pureArray(val)), null);
  }
  lift = this.pure;

  instantiateConst(tracer: PartialEvalTracer) {
    if (!tracer.pval.isKnown) {
      return tracer;
    } else {
      // Translate known value into unknown "Const" recipe for abstract eval.
      const pval = PartialVal.unknown(ShapedArray.fromAval(tracer.aval));
      return new PartialEvalTracer(this, pval, {
        type: "Const",
        val: tracer.pval.val!,
      });
    }
  }

  processPrimitive(
    primitive: Primitive,
    tracers: PartialEvalTracer[],
    params: Record<string, any>,
  ): Tracer[] {
    if (tracers.every((t) => t.pval.isKnown)) {
      return bind(
        primitive,
        tracers.map((t) => t.fullLower()),
        params,
      );
    }
    const tracersIn = tracers.map((t) => this.instantiateConst(t));
    const avalsIn = tracersIn.map((t) => t.pval.aval);
    const avalsOut = abstractEvalRules[primitive](avalsIn, params);
    const recipe: JaxprRecipe = {
      type: "JaxprEqn",
      prim: primitive,
      tracersIn,
      params,
      avalsOut,
      tracerRefsOut: [], // Populated later on
    };
    const tracersOut = avalsOut.map(
      (aval) => new PartialEvalTracer(this, PartialVal.unknown(aval), recipe),
    );
    recipe.tracerRefsOut = tracersOut.map((t) => new WeakRef(t));
    return tracersOut;
  }
}

/**
 * Convert the graph representation of a partial eval to a standard Jaxpr.
 * Also called `tracers_to_jaxpr()` in JAX.
 */
function partialEvalGraphToJaxpr(
  tracersIn: PartialEvalTracer[],
  tracersOut: PartialEvalTracer[],
): { jaxpr: Jaxpr; consts: Tracer[] } {
  const tracerToVar = new Map<PartialEvalTracer, Var>();
  const constidToVar = new Map<Tracer, Var>();
  const constvarToVal = new Map<Var, Tracer>();
  const processedEqns = new Set<JaxprRecipe>(); // Avoid translating the same equation multiple times.
  const eqns: JaxprEqn[] = [];

  for (const t of tracersIn) {
    tracerToVar.set(t, new Var(ShapedArray.fromAval(t.aval)));
  }

  for (const t of toposort(tracersOut, (t) =>
    t.recipe?.type === "JaxprEqn" ? t.recipe.tracersIn : [],
  )) {
    if (!t.recipe) {
      throw new TypeError("Tracer is missing a recipe, cannot construct Jaxpr");
    }
    if (t.recipe.type === "LambdaBinding") {
      // Check that the binding is in the input list.
      if (!tracersIn.includes(t)) {
        throw new TypeError("LambdaBinding tracer not in input list");
      }
    } else if (t.recipe.type === "Const") {
      const val = t.recipe.val;
      let binder = constidToVar.get(val);
      if (!binder) {
        binder = new Var(ShapedArray.fromAval(val.aval));
        constidToVar.set(val, binder);
        constvarToVal.set(binder, val);
      }
      tracerToVar.set(t, binder);
    } else if (t.recipe.type === "JaxprEqn") {
      if (!processedEqns.has(t.recipe)) {
        processedEqns.add(t.recipe);
        const tracersIn = t.recipe.tracersIn.map((t) => tracerToVar.get(t)!);
        const outBinders = t.recipe.avalsOut.map((aval) => new Var(aval));
        for (let i = 0; i < outBinders.length; i++) {
          const tracerOut = t.recipe.tracerRefsOut[i].deref();
          if (tracerOut) {
            tracerToVar.set(tracerOut, outBinders[i]);
          }
        }
        eqns.push(
          new JaxprEqn(t.recipe.prim, tracersIn, t.recipe.params, outBinders),
        );
      }
    }
  }

  const [constvars, consts] = unzip2(constvarToVal.entries());
  const inBinders = [
    ...constvars,
    ...tracersIn.map((t) => tracerToVar.get(t)!),
  ];
  const outVars = tracersOut.map((t) => tracerToVar.get(t)!);
  const jaxpr = new Jaxpr(inBinders, eqns, outVars);
  typecheckJaxpr(jaxpr); // sanity check
  return { jaxpr: jaxpr.simplify(), consts };
}

// implementation of vjp and grad

/** Marker type for pullback, used by transpose rules. */
class UndefPrimal {
  readonly aval: ShapedArray;

  constructor(aval: AbstractValue) {
    this.aval = ShapedArray.fromAval(aval);
  }
}

/**
 * Evaluate the backward pass over a linearized Jaxpr (pullback of cotangents).
 *
 * Will raise a TypeError if the provided Jaxpr is not a linear function of its,
 * inputs, as general expressions cannot be transposed.
 */
function evalJaxprTransposed(
  jaxpr: Jaxpr,
  args: (Tracer | UndefPrimal)[],
  cotangents: Tracer[],
): Tracer[] {
  const knownPrimals = new Map<Var, Tracer>();
  for (let i = 0; i < jaxpr.inBinders.length; i++) {
    if (!(args[i] instanceof UndefPrimal)) {
      knownPrimals.set(jaxpr.inBinders[i], args[i] as Tracer);
    }
  }

  const ctStore = new Map<Var, Tracer>();

  const readCotangent = (v: Var) => {
    const ct = ctStore.get(v);
    if (ct) {
      // We should read a cotangent at most once, as an out binder.
      ctStore.delete(v);
      return ct;
    } else {
      return zeros(v.aval.shape, v.aval.dtype);
    }
  };

  const writeCotangent = (v: Var, ct: Tracer | null) => {
    if (ct !== null) {
      const oldCt = ctStore.get(v);
      // May need to accumulate cotangents if used in multiple JaxprEqns.
      if (oldCt) ctStore.set(v, add(oldCt, ct));
      else ctStore.set(v, ct);
    }
  };

  for (let i = 0; i < jaxpr.outs.length; i++) {
    const v = jaxpr.outs[i];
    if (v instanceof Var) writeCotangent(v, cotangents[i]);
  }

  for (let i = jaxpr.eqns.length - 1; i >= 0; i--) {
    const eqn = jaxpr.eqns[i];
    // Inputs are primalsIn and cotangentsOut, outputs are cotangentsIn. We're
    // using the known primal values to _pull back_  cotangents for unknown
    // values. Tricky!
    const primalsIn = eqn.inputs.map((v) =>
      v instanceof Lit
        ? v.val
        : (knownPrimals.get(v) ?? new UndefPrimal(v.aval)),
    );
    const cotangentsOut = eqn.outBinders.map(readCotangent);
    const rule = transposeRules[eqn.primitive];
    if (!rule) {
      throw new TypeError(`Backward pass not implemented for ${eqn.primitive}`);
    }
    const cotangentsIn = rule(cotangentsOut, primalsIn, eqn.params);
    for (let j = 0; j < jaxpr.inBinders.length; j++) {
      const v = eqn.inputs[j];
      if (v instanceof Var && !knownPrimals.has(v)) {
        writeCotangent(v, cotangentsIn[j]);
      }
    }
  }

  const results: Tracer[] = [];
  for (let i = 0; i < jaxpr.inBinders.length; i++) {
    if (args[i] instanceof UndefPrimal) {
      results.push(readCotangent(jaxpr.inBinders[i]));
    }
  }
  return results;
}

class NonlinearError extends TypeError {
  constructor(primitive: Primitive) {
    super(`Nonlinear operation in backward pass for ${primitive}`);
  }
}

type TransposeRule = (
  cotangents: Tracer[],
  primals: (Tracer | UndefPrimal)[],
  params: any,
) => (Tracer | null)[];

const transposeRules: Partial<Record<Primitive, TransposeRule>> = {
  [Primitive.Mul]([ct], [x, y]) {
    if (x instanceof UndefPrimal === y instanceof UndefPrimal)
      throw new NonlinearError(Primitive.Mul);
    return x instanceof UndefPrimal
      ? [mul(ct, y as Tracer), null]
      : [null, mul(x as Tracer, ct)];
  },
  [Primitive.Neg]([ct], [x]) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Neg);
    return [neg(ct)];
  },
  [Primitive.Add]([ct], [x, y]) {
    if (!(x instanceof UndefPrimal || y instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Add);
    return [ct, ct];
  },
  [Primitive.ReduceSum]([ct], [x], { axis }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.ReduceSum);
    return [broadcast(ct, x.aval.shape, axis)];
  },
  // TODO: transpose, broadcast
};

function vjpFlat(
  f: (...x: Tracer[]) => Tracer[],
  primalsIn: Tracer[],
): [Tracer[], (...cotangents: Tracer[]) => Tracer[]] {
  const { primalsOut, jaxpr, consts } = linearizeFlatUtil(f, primalsIn);
  const transposeInputs = [
    ...consts,
    // Explcitly list which arguments should be transposed.
    ...primalsIn.map((t) => new UndefPrimal(t.aval)),
  ];
  // Pullback cotangents to the UndefPrimal transpose inputs.
  const fVjp = (...cotangents: Tracer[]) =>
    evalJaxprTransposed(jaxpr, transposeInputs, cotangents);
  return [primalsOut, fVjp];
}

export function vjp(
  f: (...primals: any) => any,
  ...primalsIn: any
): [any, (...cotangents: any) => any] {
  const [primalsInFlat, inTree] = treeFlatten(primalsIn);
  const [fFlat, outTree] = flattenFun(f, inTree);
  const [primalsOutFlat, fVjpFlat] = vjpFlat(
    fFlat,
    primalsInFlat.map(pureArray),
  );
  if (outTree.value === undefined) {
    throw new Error("outTree was not set in vjp");
  }
  const primalsOut = treeUnflatten(outTree.value, primalsOutFlat);

  // "cotangentsOut" because pullback
  const fVjp = (cotangentsOut: any) => {
    const [cotangentsOutFlat, outTree2] = treeFlatten(cotangentsOut);
    if (!outTree.value!.equals(outTree2)) {
      throw new TreeMismatchError("vjp", outTree.value!, outTree2);
    }
    const cotangentsInFlat = fVjpFlat(...cotangentsOutFlat.map(pureArray));
    return treeUnflatten(inTree, cotangentsInFlat);
  };

  return [primalsOut, fVjp];
}

export function grad(f: (...primals: any) => Tracer) {
  return (...x: any) => {
    const [y, fVjp] = vjp(f, ...x);
    if (!(y instanceof Tracer) || ndim(y) !== 0) {
      throw new TypeError("grad requires a scalar output");
    }
    if (y.aval.dtype !== DType.Float32) {
      throw new TypeError("grad currently only supports float32");
    }
    // JAX convention, differentiate with respect to the first argument.
    return fVjp(pureArray(1))[0];
  };
}
