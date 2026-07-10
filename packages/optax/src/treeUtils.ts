import { DType, JsTree, numpy as np, tree } from "@jax-js/jax";

export function treeZerosLike(
  tr: JsTree<np.Array>,
  dtype?: DType,
): JsTree<np.Array> {
  return tree.map((x: np.Array) => np.zerosLike(x, { dtype }), tr);
}

export function treeOnesLike(
  tr: JsTree<np.Array>,
  dtype?: DType,
): JsTree<np.Array> {
  return tree.map((x: np.Array) => np.onesLike(x, { dtype }), tr);
}

function ipow(a: np.Array, order: number) {
  if (!Number.isInteger(order) || order <= 0) {
    throw new Error("Order must be a positive integer");
  }
  let result = a.ref;
  for (let i = 1; i < order; i++) {
    result = result.mul(a.ref);
  }
  a.dispose();
  return result;
}

export function treeUpdateMoment(
  updates: JsTree<np.Array>,
  moments: JsTree<np.Array>,
  decay: number,
  order: number,
): JsTree<np.Array> {
  return tree.map(
    (g: np.Array, t: np.Array) =>
      ipow(g, order)
        .mul(1 - decay)
        .add(t.mul(decay)),
    updates,
    moments,
  );
}

/** Performs bias correction, dividing by 1-decay^count. */
export function treeBiasCorrection(
  moments: JsTree<np.Array>,
  decay: number,
  count: np.Array,
): JsTree<np.Array> {
  // Computed in-graph rather than via count.item(): a data read throws on
  // tracers, which made every optimizer built on bias correction (adam,
  // adamw, ...) impossible to wrap in jit().
  const correction = np.reciprocal(
    np.subtract(1.0, np.power(decay, count.astype(np.float32))),
  );
  const result = tree.map((t: np.Array) => t.mul(correction.ref), moments);
  correction.dispose();
  return result;
}

/** Sum all elements across all arrays in a pytree. */
export function treeSum(tr: JsTree<np.Array>): np.Array {
  const [leaves] = tree.flatten(tr);
  return leaves.reduce((total, leaf) => total.add(np.sum(leaf)), np.array(0.0));
}

/** Max of all elements across all arrays in a pytree. */
export function treeMax(tr: JsTree<np.Array>): np.Array {
  const [leaves] = tree.flatten(tr);
  return leaves.reduce(
    (maxVal, leaf) => np.maximum(maxVal, np.max(leaf)),
    np.array(-Infinity),
  );
}

export type NormOrd = 1 | 2 | "inf" | "infinity" | number | null;

/** Compute the vector norm of the given ord of a pytree. */
export function treeNorm(
  tr: JsTree<np.Array>,
  ord: NormOrd = null,
  squared = false,
): np.Array {
  if (ord === null || ord === 2) {
    const squaredTree = tree.map(np.square, tr);
    const sqNorm = treeSum(squaredTree);
    return squared ? sqNorm : np.sqrt(sqNorm);
  } else if (ord === 1) {
    const absTree = tree.map(np.abs, tr);
    const result = treeSum(absTree);
    return squared ? np.square(result) : result;
  } else if (ord === "inf" || ord === "infinity" || ord === Infinity) {
    const absTree = tree.map(np.abs, tr);
    const result = treeMax(absTree);
    return squared ? np.square(result) : result;
  } else {
    tree.dispose(tr);
    throw new Error(`Unsupported ord: ${ord}`);
  }
}
