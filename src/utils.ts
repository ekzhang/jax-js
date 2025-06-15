/** @file Generic programming utilities with no dependencies on library code. */

export const DEBUG: number = 3;

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

/** Shorthand for integer division, like in Python. */
export function idiv(a: number, b: number): number {
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
export function toposort<T>(terminals: T[], parents: (node: T) => T[]) {
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

  update(...values: (string | boolean | bigint | null | undefined)[]): this {
    for (const x of values) {
      if (typeof x === "string") {
        for (const c of x) this.#update(BigInt(199 + c.charCodeAt(0)));
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
      }
    }
    return this;
  }

  static hash(
    ...values: (string | boolean | bigint | null | undefined)[]
  ): bigint {
    return new FpHash().update(...values).value;
  }
}
