/** @file Core library internals and interpreter stack, based on Autodidax. */

import { DType } from "../alu";
import { DEBUG, range } from "../utils";
import {
  JsTreeDef,
  flatten as treeFlatten,
  unflatten as treeUnflatten,
} from "../tree";

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
      axis = range(x.shape.length);
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
export function newMain(
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
export function newDynamic(main: MainTrace): Disposable {
  const prevDynamicTrace = dynamicTrace;
  dynamicTrace = main;
  return {
    [Symbol.dispose]() {
      dynamicTrace = prevDynamicTrace;
    },
  };
}

export type TracerValue = Tracer | number | boolean;

export abstract class Trace {
  constructor(readonly main: MainTrace) {}

  abstract pure(val: TracerValue): Tracer;
  abstract lift(val: Tracer): Tracer;

  abstract processPrimitive(
    primitive: Primitive,
    tracers: Tracer[],
    params: Record<string, any>,
  ): Tracer[];
}

export interface AbstractValue {
  shape: number[];
  dtype: DType;
}

export abstract class Tracer {
  readonly _trace: Trace;

  constructor(trace: Trace) {
    this._trace = trace;
  }

  abstract get aval(): AbstractValue;
  abstract toString(): string;

  get shape() {
    return this.aval.shape;
  }
  get dtype() {
    return this.aval.dtype;
  }

  fullLower(): Tracer {
    return this; // default implementation
  }

  // These types aren't technically correct since they don't account for the
  // fact that tracers can be lifted to different levels. But they simplify the
  // API visible to users.
  neg() {
    return neg(this) as this;
  }
  add(other: this | TracerValue) {
    return add(this, other) as this;
  }
  mul(other: this | TracerValue) {
    return mul(this, other) as this;
  }
  gt(other: this | TracerValue) {
    return greater(this, other) as this;
  }
  lt(other: this | TracerValue) {
    return less(this, other) as this;
  }
  transpose(perm?: number[]): this {
    return transpose(this, perm) as this;
  }
}

export function ndim(x: TracerValue) {
  if (x instanceof Tracer) {
    return x.shape.length;
  } else {
    return 0;
  }
}

// Note: Autodidax has a `ConcreteArray` type with "arrayAbstractionLevel" set
// to a higher value. I didn't see how this would be useful yet, so currently
// the only `AbstractValue` is a `ShapedArray` instance.
export class ShapedArray implements AbstractValue {
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

export function getAval(x: TracerValue): AbstractValue {
  if (x instanceof Tracer) {
    return x.aval;
  } else if (typeof x === "boolean" || typeof x === "number") {
    return new ShapedArray(
      [],
      typeof x === "boolean" ? DType.Bool : DType.Float32,
    );
  } else {
    throw new TypeError(`Unknown value: ${x}`);
  }
}

export function bind(
  prim: Primitive,
  args: TracerValue[],
  params: Record<string, any> = {},
) {
  const topTrace = findTopTrace(args);
  const tracers = args.map((arg) => fullRaise(topTrace, arg));
  const outs = topTrace.processPrimitive(prim, tracers, params);
  if (DEBUG >= 4) {
    console.info(`processing rule for ${prim} on ${tracers} and got ${outs}`);
  }
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

export function fullRaise(trace: Trace, val: TracerValue): Tracer {
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

export class TreeMismatchError extends TypeError {
  constructor(where: string, left: JsTreeDef, right: JsTreeDef) {
    super(`Mismatched tree structures in ${where}: ${left} != ${right}`);
  }
}

export function flattenFun(
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
