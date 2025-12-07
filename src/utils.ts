/** @file Generic programming utilities with no dependencies on library code. */

export let DEBUG: number = 0;

/**
 * Set the debug level for verbose logging.
 *
 * 1. JIT compile logs
 * 2. Shader code
 * 3. Expressions and metadata
 * 4. JIT programs, tuning details
 * 5. Most verbose operation traces
 *
 * This is an experimental API and may change in behavior. Do not rely on this
 * in production.
 */
export function setDebug(level: number) {
  DEBUG = level;
}

export function unzip2<T, U>(pairs: Iterable<[T, U]>): [T[], U[]] {
  const lst1: T[] = [];
  const lst2: U[] = [];
  for (const [x, y] of pairs) {
    lst1.push(x);
    lst2.push(y);
  }
  return [lst1, lst2];
}

export function zip<T, U>(xs: T[], ys: U[]): [T, U][] {
  return xs.map((x, i) => [x, ys[i]]);
}

export function zipn<T>(...arrays: T[][]): T[][] {
  const minLength = Math.min(...arrays.map((x) => x.length));
  return Array.from({ length: minLength }, (_, i) =>
    arrays.map((arr) => arr[i]),
  );
}

export function rep<T>(
  length: number,
  value: T,
): (T extends (...args: any) => infer R ? R : T)[] {
  if (value instanceof Function) {
    return new Array(length).fill(0).map((_, i) => value(i));
  }
  return new Array(length).fill(value);
}

export function prod(arr: number[]): number {
  return arr.reduce((acc, x) => acc * x, 1);
}

export function gcd(...values: number[]): number {
  let a = 0;
  for (let b of values) {
    while (b !== 0) [a, b] = [b, a % b];
  }
  return Math.abs(a);
}

/** Shorthand for integer division, like in Python. */
export function intdiv(a: number, b: number): number {
  return Math.floor(a / b);
}

/** Clamp `x` to the range `[min, max]`. */
export function clamp(x: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, x));
}

/** Check if two objects are deep equal. */
export function deepEqual(a: any, b: any): boolean {
  if (a === b) {
    return true;
  }
  if (typeof a !== "object" || typeof b !== "object") {
    return false;
  }
  if (a === null || b === null) {
    return false;
  }
  if (Object.keys(a).length !== Object.keys(b).length) {
    return false;
  }
  for (const key of Object.keys(a)) {
    if (!deepEqual(a[key], b[key])) {
      return false;
    }
  }
  return true;
}

export function union<T>(...sets: (Iterable<T> | null | undefined)[]): Set<T> {
  const result = new Set<T>();
  for (const s of sets) {
    if (s) {
      for (const x of s) result.add(x);
    }
  }
  return result;
}

/** Splits the list based on a condition, `false` first then `true`. */
export function partitionList<T>(which: boolean[], array: T[]): [T[], T[]] {
  const falseList: T[] = [];
  const trueList: T[] = [];
  for (let i = 0; i < which.length; i++) {
    if (which[i]) {
      trueList.push(array[i]);
    } else {
      falseList.push(array[i]);
    }
  }
  return [falseList, trueList];
}

/** Compare two arrays of numbers lexicographically. */
export function lexCompare(a: number[], b: number[]): number {
  const minLength = Math.min(a.length, b.length);
  for (let i = 0; i < minLength; i++) {
    if (a[i] < b[i]) return -1;
    if (a[i] > b[i]) return 1;
  }
  return a.length - b.length;
}

/** Check if an object is a number pair, i.e., a tuple of two numbers. */
export function isNumberPair(x: unknown): x is [number, number] {
  return (
    Array.isArray(x) &&
    x.length === 2 &&
    typeof x[0] === "number" &&
    typeof x[1] === "number"
  );
}

/** Check an axis against number of dimensions, and resolve negative axes. */
export function checkAxis(axis: number, ndim: number): number {
  if (axis < -ndim || axis >= ndim) {
    throw new Error(
      `Axis ${axis} out of bounds for array of dimension ${ndim}`,
    );
  }
  return axis < 0 ? axis + ndim : axis;
}

/** Normalize common axis argument for functions, defaulting to all axes. */
export function normalizeAxis(
  axis: number | number[] | null,
  ndim: number,
): number[] {
  if (axis === null) {
    return range(ndim); // default to all axes
  } else if (typeof axis === "number") {
    return [checkAxis(axis, ndim)];
  } else {
    const seen = new Set<number>();
    for (const a of axis) {
      const ca = checkAxis(a, ndim);
      if (seen.has(ca))
        throw new Error(`Duplicate axis ${ca} passed to function`);
      seen.add(ca);
    }
    return [...seen].sort();
  }
}

export function range(
  start: number,
  stop?: number,
  step: number = 1,
): number[] {
  if (stop === undefined) {
    stop = start;
    start = 0;
  }
  const result = [];
  for (let i = start; i < stop; i += step) {
    result.push(i);
  }
  return result;
}

export function isPermutation(axis: number[], n: number): boolean {
  if (axis.length !== n) return false;
  const seen = new Set<number>();
  for (const x of axis) {
    if (x < 0 || x >= n) return false;
    seen.add(x);
  }
  return seen.size === n;
}

export function invertPermutation(axis: number[]): number[] {
  const n = axis.length;
  if (!isPermutation(axis, n))
    throw new Error("invertPermutation: axis is not a permutation");
  const result = new Array(n);
  for (let i = 0; i < n; i++) {
    result[axis[i]] = i;
  }
  return result;
}

/** Topologically sort a DAG, given terminal nodes and an ancestor function. */
export function toposort<T>(terminals: T[], parents: (node: T) => T[]): T[] {
  const childCounts: Map<T, number> = new Map();

  // First iteartion counts the number of children for each node.
  const stack = [...new Set(terminals)];
  while (true) {
    const node = stack.pop();
    if (!node) break;
    for (const parent of parents(node)) {
      if (childCounts.has(parent)) {
        childCounts.set(parent, childCounts.get(parent)! + 1);
      } else {
        childCounts.set(parent, 1);
        stack.push(parent);
      }
    }
  }
  for (const node of terminals) {
    childCounts.set(node, childCounts.get(node)! - 1);
  }

  // Second iteration produces a reverse topological order.
  const order: T[] = [];
  const frontier = terminals.filter((n) => !childCounts.get(n));
  while (true) {
    const node = frontier.pop();
    if (!node) break;
    order.push(node);
    for (const parent of parents(node)) {
      const c = childCounts.get(parent)! - 1;
      childCounts.set(parent, c);
      if (c == 0) {
        frontier.push(parent);
      }
    }
  }

  return order.reverse();
}

/**
 * Returns the largest power of 2 less than or equal to `max`.
 *
 * If `hint` is nonzero, it will not return a number greater than the first
 * power of 2 that is greater than or equal to `hint`.
 */
export function findPow2(hint: number, max: number): number {
  if (max < 1) {
    throw new Error("max must be a positive integer");
  }
  let ret = 1;
  while (ret < hint && 2 * ret <= max) {
    ret *= 2;
  }
  return ret;
}

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
export function generalBroadcast(a: number[], b: number[]): number[] {
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

/** @inline */
export type RecursiveArray<T> = T | RecursiveArray<T>[];

export function recursiveFlatten<T>(ar: RecursiveArray<T>): T[] {
  if (!Array.isArray(ar)) return [ar];
  return (ar as any).flat(Infinity); // Escape infinite type depth
}

/** Strip an outermost pair of nested parentheses from an expression, if any. */
export function strip1(str: string): string {
  if (str[0] === "(" && str[str.length - 1] === ")") {
    return str.slice(1, -1);
  }
  return str;
}

export interface FpHashable {
  hash(state: FpHash): void;
}

const _stagingbuf = new DataView(new ArrayBuffer(8));

/**
 * Polynomial hashes modulo p are good at avoiding collisions in expectation.
 * Probability-wise, it's good enough to be used for something like
 * deduplicating seen compiler expressions, although it's not adversarial.
 *
 * See https://en.wikipedia.org/wiki/Lagrange%27s_theorem_(number_theory)
 */
export class FpHash {
  value: bigint = 8773157n;

  #update(x: bigint) {
    // These primes were arbitrarily chosen, should be at least 10^9.
    const base = 873192869n;
    const modulus = 3189051996290219n; // Less than 2^53-1, for convenience.

    this.value = (this.value * base + x) % modulus;
  }

  update(
    x: string | boolean | number | bigint | null | undefined | FpHashable,
  ): this {
    if (typeof x === "string") {
      this.#update(BigInt(x.length));
      for (let i = 0; i < x.length; i++) {
        this.#update(BigInt(199 + x.charCodeAt(i)));
      }
    } else if (typeof x === "number") {
      if (Number.isInteger(x)) {
        this.#update(68265653n ^ BigInt(x));
      } else {
        _stagingbuf.setFloat64(0, x, true);
        this.#update(_stagingbuf.getBigUint64(0, true));
      }
    } else if (typeof x === "boolean") {
      this.#update(x ? 69069841n : 63640693n);
    } else if (typeof x === "bigint") {
      // When combining the output of another hash, must be nonlinear to avoid
      // obvious collisions.
      this.#update(x ^ 71657401n);
    } else if (x === null) {
      this.#update(37832657n);
    } else if (x === undefined) {
      this.#update(18145117n);
    } else {
      // If the object has a hash method, call it.
      x.hash(this);
    }
    return this;
  }

  static hash(
    ...values: (
      | string
      | boolean
      | number
      | bigint
      | null
      | undefined
      | FpHashable
    )[]
  ): bigint {
    const h = new FpHash();
    for (const x of values) h.update(x);
    return h.value;
  }
}

/** Run a function while caching it inline inside a `Map`. */
export function runWithCache<K, V>(
  cache: Map<K, V>,
  key: K,
  thunk: () => V,
): V {
  if (cache.has(key)) {
    return cache.get(key)!;
  } else {
    const value = thunk();
    cache.set(key, value);
    return value;
  }
}
