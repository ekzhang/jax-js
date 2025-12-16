// Utilities for working with tree-like container data structures ("pytrees").

import type { Array } from "./numpy";
import { deepEqual, unzip2 } from "./utils";

const JsArray = globalThis.Array;

export enum NodeType {
  Array = "Array",
  Object = "Object",
  Leaf = "Leaf",
}

/** Analog to the JAX "pytree" object, but for JavaScript. */
export type JsTree<T> = T | JsTree<T>[] | { [key: string]: JsTree<T> };

type Same<X, Y> =
  (<T>() => T extends X ? 1 : 2) extends <T>() => T extends Y ? 1 : 2
    ? true
    : false;

type MappedJsTree<T, A, B> = T extends A
  ? B
  : T extends Array // Special case: Do not recurse into np.Array
    ? T
    : T extends globalThis.Array<infer U>
      ? number extends T["length"]
        ? MapJsTree<U, A, B>[] // plain array
        : { [K in keyof T]: MapJsTree<T[K], A, B> } // tuple: map each slot, keep tuple shape
      : { [K in keyof T]: MapJsTree<T[K], A, B> }; // object: map each slot, keep object shape

/** @ignore Convert a subtype of JsTree<A> into a JsTree<B>, with the same structure. */
export type MapJsTree<T, A, B> =
  Same<A, B> extends true ? T : MappedJsTree<T, A, B>;

/** Represents the structure of a JsTree. */
export class JsTreeDef {
  static leaf = new JsTreeDef(NodeType.Leaf, null, []);

  constructor(
    readonly nodeType: NodeType,
    readonly nodeMetadata: any, // Must be comparable with deepEqual.
    readonly childTreedefs: JsTreeDef[],
  ) {}

  /** Returns a string representation of this tree definition. */
  toString(root = true): string {
    if (root) {
      return "JsTreeDef(" + this.toString(false) + ")";
    }
    switch (this.nodeType) {
      case NodeType.Leaf:
        return "*";
      case NodeType.Array:
        return `[${this.childTreedefs.map((x) => x.toString(false)).join(", ")}]`;
      case NodeType.Object: {
        const parts: string[] = [];
        for (let i = 0; i < this.childTreedefs.length; i++) {
          parts.push(
            `${quoteObjectKey(this.nodeMetadata[i])}: ${this.childTreedefs[i].toString(false)}`,
          );
        }
        return `{${parts.join(", ")}}`;
      }
    }
  }

  /** Compare this tree definition with another. */
  equals(other: JsTreeDef): boolean {
    return (
      this.nodeType === other.nodeType &&
      deepEqual(this.nodeMetadata, other.nodeMetadata) &&
      this.childTreedefs.length === other.childTreedefs.length &&
      this.childTreedefs.every((x, i) => x.equals(other.childTreedefs[i]))
    );
  }
}

function quoteObjectKey(key: string): string {
  // Check if the key is a valid JavaScript identifier
  if (/^[a-zA-Z_$][a-zA-Z0-9_$]*$/.test(key)) {
    return key; // No need to quote
  }
  return JSON.stringify(key);
}

/** Flatten a structured object, returning the tree definition. */
export function flatten<T>(tree: JsTree<T>): [T[], JsTreeDef] {
  const leaves: T[] = [];
  const treedef = _flatten(tree, leaves);
  return [leaves, treedef];
}

function _flatten<T>(tree: JsTree<T>, leaves: T[]): JsTreeDef {
  if (JsArray.isArray(tree)) {
    const childTrees = tree.map((c) => _flatten(c, leaves));
    return new JsTreeDef(NodeType.Array, null, childTrees);
  } else if (
    typeof tree === "object" &&
    tree !== null &&
    tree.constructor === Object // Needed to avoid treating Array as an object.
  ) {
    const [keys, values] = unzip2(Object.entries(tree));
    const childTrees = values.map((c) => _flatten(c, leaves));
    return new JsTreeDef(NodeType.Object, keys, childTrees);
  } else {
    leaves.push(tree as T);
    return JsTreeDef.leaf;
  }
}

/** Get the leaves of a tree. */
export function leaves<T>(tree: JsTree<T>): T[] {
  return flatten<T>(tree)[0];
}

/** Get the treedef for a tree. */
export function structure<T>(tree: JsTree<T>): JsTreeDef {
  return flatten<T>(tree)[1];
}

/** Reconstruct a structured object from the flattened representation. */
export function unflatten<T>(
  treedef: JsTreeDef,
  leaves: Iterable<T>,
): JsTree<T> {
  return _unflatten(treedef, leaves[Symbol.iterator]());
}

function _unflatten<T>(treedef: JsTreeDef, leaves: Iterator<T>): JsTree<T> {
  switch (treedef.nodeType) {
    case NodeType.Leaf: {
      const { value, done } = leaves.next();
      if (done) {
        throw new TypeError("Ran out of leaves while unflattening JsTree");
      }
      return value;
    }
    case NodeType.Array:
      return treedef.childTreedefs.map((c) => _unflatten(c, leaves));
    case NodeType.Object: {
      const obj: any = {};
      for (let i = 0; i < treedef.childTreedefs.length; i++) {
        obj[treedef.nodeMetadata[i]] = _unflatten(
          treedef.childTreedefs[i],
          leaves,
        );
      }
      return obj;
    }
  }
}

/** Maps a multi-input function over pytree args to produce a new pytree. */
export function map<T, U, Tree extends JsTree<T>>(
  fn: (...args: T[]) => U,
  tree: Tree,
  ...rest: Tree[]
): MapJsTree<Tree, T, U> {
  const [leaves, treedef] = flatten<T>(tree);
  const restLeaves = rest.map((x) => flatten<T>(x)[0]);
  const resultLeaves: U[] = [];
  for (let i = 0; i < leaves.length; i++) {
    resultLeaves.push(fn(leaves[i], ...restLeaves.map((x) => x[i])));
  }
  return unflatten(treedef, resultLeaves) as MapJsTree<Tree, T, U>;
}

/** Take a reference of every array in a tree. */
export function ref<Tree extends JsTree<Array>>(tree: Tree): Tree {
  return map((x: Array) => x.ref, tree) as unknown as Tree;
}

/** Dispose every array in a tree. */
export function dispose<Tree extends JsTree<Array>>(
  tree: Tree | null | undefined,
): void {
  if (tree) {
    map((x: Array) => x.dispose(), tree);
  }
}
