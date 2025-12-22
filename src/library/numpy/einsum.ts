import { range } from "../../utils";

const bprod = (...xs: bigint[]) => xs.reduce((acc, x) => acc * x, 1n);
const bmax = (...xs: bigint[]) => xs.reduce((max, x) => (x > max ? x : max));
const uniq = <T>(arr: T[]): T[] => Array.from(new Set(arr));

const EINSUM_COMPONENT_RE = /\p{ID_Start}|\.\.\./gu;

/**
 * Explicit form of an input to `einsum()`.
 *
 * This is stored as an object. For example, if the arguments are such as
 * `np.einsum(x, [0, 1], y, [1], [0])`, then the corresponding input is:
 *
 * ```
 * {
 *   shapes: [x.shape, y.shape],
 *   lhsIndices: [[0, 1], [1]],
 *   rhsIndex: [0],
 * }
 * ```
 */
export interface EinsumInput {
  shapes: bigint[][];
  lhsIndices: number[][];
  rhsIndex: number[];
}

/** Parse an Einstein notation string into a runnable `EinsumInput`. */
export function parseEinsumExpression(
  expr: string,
  shapes: bigint[][],
): EinsumInput {
  const idents = [
    ...expr
      .split("->")[0]
      .matchAll(EINSUM_COMPONENT_RE)
      .map((m) => m[0])
      .filter((c) => c !== "..."),
  ];
  if (!expr.includes("->")) {
    // Implicit-form einsum expression. Identifiers used exactly once are the
    // input are returned in Unicode order in the output.
    const counts = new Map<string, number>();
    for (const c of idents) {
      counts.set(c, (counts.get(c) ?? 0) + 1);
    }
    const outputIndices = Array.from(counts.entries())
      .filter(([, count]) => count === 1)
      .map(([char]) => char)
      .sort();
    if (expr.includes("...")) outputIndices.splice(0, 0, "...");
    expr += "->" + outputIndices.join("");
  }

  const identToIndex = new Map<string, number>(
    uniq(idents)
      .sort()
      .map((c, i) => [c, i]),
  );
  const componentsToIndices = (components: string[]) =>
    components.flatMap((c) =>
      c === "..."
        ? range(identToIndex.size, identToIndex.size + ellipsisRank!)
        : identToIndex.get(c)!,
    );

  let ellipsisRank: number | null = null; // Number of dimensions that "..." represents

  const [lhs, rhs] = expr.split("->");
  const lhsIndices: number[][] = [];
  for (const [i, part] of lhs.split(",").entries()) {
    const shape = shapes[i];
    const components = [...part.matchAll(EINSUM_COMPONENT_RE)].map((m) => m[0]);
    const ellipsisIndex = components.indexOf("...");
    if (ellipsisIndex !== -1) {
      if (components.lastIndexOf("...") !== ellipsisIndex)
        throw new Error(
          "Multiple ellipses in one einsum operand is not allowed",
        );
      const numExplicit = components.length - 1;
      if (shape.length < numExplicit)
        throw new Error(
          `Einsum operand ${i} has shape ${JSON.stringify(shape)} but indexed with "${part}"`,
        );
      const thisEllipsisRank = shape.length - numExplicit;
      if (ellipsisRank === null) ellipsisRank = thisEllipsisRank;
      else if (ellipsisRank !== thisEllipsisRank)
        throw new Error(
          `Einsum operand ${i} has shape ${JSON.stringify(shape)} but indexed with "${part}", inconsistent with previous ellipsis of rank ${ellipsisRank}`,
        );
    }
    lhsIndices.push(componentsToIndices(components));
  }
  const rhsIndex = componentsToIndices(
    [...rhs.matchAll(EINSUM_COMPONENT_RE)].map((m) => m[0]),
  );
  return { shapes, lhsIndices, rhsIndex };
}

export class EinsumPath {
  /** Parsed and normalized input for the einsum. */
  readonly input: EinsumInput;

  /** Mapping of each index number to its size in the shape array. */
  readonly sizeMap: Map<number, bigint>;

  /**
   * A list of tensor contractions.
   *
   * This is ordered by operation order. Each entry corresponds to a single
   * elementwise product and/or inner contraction between two tensors, and it
   * contains the indices of the tensors to be contracted.
   *
   * The indices of input tensors are [0..n), and each intermediate from the
   * path at index i produces a new tensor at index n + i at the end
   * (opt_einsum internally calls this "SSA form").
   *
   * Invariants:
   * - Each group in the path consists of two tensors.
   * - For n input tensors, there are n-1 groups in the path.
   * - Every tensor must be in the path exactly once, except the final output.
   *
   * @example
   * Given einsum for `(A, B, C)`, this path corresponds to `(A, B)` and then
   * `(AB, C)`.
   * ```
   * [[0, 1], [3, 2]]
   * ```
   */
  readonly path: [number, number][];

  constructor(
    input: EinsumInput,
    sizeMap: Map<number, bigint>,
    path: [number, number][],
  ) {
    this.input = input;
    this.sizeMap = sizeMap;
    this.path = path;
  }

  /** Shape of the final output tensor. */
  get outputShape(): bigint[] {
    return this.input.rhsIndex.map((i) => this.sizeMap.get(i)!);
  }

  /** Estimate the number of FLOPs to execute this einsum path. */
  get approximateFlops(): bigint {
    return approximatePathFlops(this.input, this.sizeMap, this.path);
  }
}

function approximatePathFlops(
  input: EinsumInput,
  sizeMap: Map<number, bigint>,
  path: [number, number][],
): bigint {
  if (path.length == 0) {
    // Special case: 0-length path returned if there's only one input tensor.
    // This is the case if we take the trace or transpose.
    const [indices] = input.lhsIndices;
    return bprod(...uniq(indices).map((i) => sizeMap.get(i)!));
  }
  const indexUsageCounts: number[] = [];
  for (const idx of [...input.lhsIndices.flat(), ...input.rhsIndex]) {
    indexUsageCounts[idx] = (indexUsageCounts[idx] ?? 0) + 1;
  }
  const indices = [...input.lhsIndices];
  let totalFlops = 0n;
  for (const tensorGroup of path) {
    const indexReduced: number[] = [];
    const indexGroup: number[] = [];
    for (const tensorIdx of tensorGroup) {
      for (const idx of indices[tensorIdx]) {
        if (!indexGroup.includes(idx)) indexGroup.push(idx);
        // If the index is not in the output and isn't in any other inputs,
        // we can consider it reduced here.
        if (--indexUsageCounts[idx] === 0) indexReduced.push(idx);
      }
    }
    totalFlops += approximateCountFlops(
      indexGroup,
      indexReduced.length > 0,
      tensorGroup.length,
      sizeMap,
    );
    indices.push(indexGroup.filter((x) => !indexReduced.includes(x)).sort());
  }
  return totalFlops;
}

function approximateCountFlops(
  indexGroup: number[],
  hasReduction: boolean,
  numTerms: number,
  sizeMap: Map<number, bigint>,
): bigint {
  const elements = bprod(...indexGroup.map((i) => sizeMap.get(i)!));
  const flopsPerLoopIteration =
    BigInt(numTerms) - 1n + (hasReduction ? 1n : 0n);
  return bmax(elements * flopsPerLoopIteration, 1n);
}

/** Compute size for each index in the einsum expression, also validates input. */
function computeSizeMap({
  shapes,
  lhsIndices,
  rhsIndex,
}: EinsumInput): Map<number, bigint> {
  if (shapes.length === 0) {
    throw new Error("Einsum must have at least one input tensor");
  }
  if (lhsIndices.length !== shapes.length) {
    throw new Error(
      `Mismatched number of lhs operands (${lhsIndices.length}) and shapes (${shapes.length})`,
    );
  }
  for (let i = 0; i < shapes.length; i++) {
    if (lhsIndices[i].length !== shapes[i].length) {
      throw new Error(
        `Mismatched number of indices (${lhsIndices[i].length}) and shape (${JSON.stringify(shapes[i])}) for operand ${i}`,
      );
    }
  }
  const rhsIndexSet = new Set<number>();
  for (const idx of rhsIndex) {
    if (rhsIndexSet.has(idx)) {
      throw new Error(`Repeated index ${idx} in einsum output`);
    }
    rhsIndexSet.add(idx);
  }

  const sizeMap = new Map<number, bigint>();
  for (let i = 0; i < shapes.length; i++) {
    const shape = shapes[i];
    const lhsIndex = lhsIndices[i];
    for (let j = 0; j < lhsIndex.length; j++) {
      const idx = lhsIndex[j];
      const dim = shape[j];
      const existing = sizeMap.get(idx);
      if (existing === undefined) {
        sizeMap.set(idx, dim);
      } else if (existing !== dim) {
        throw new Error(
          `Inconsistent size for index ${idx} in einsum: ${existing} vs ${dim}`,
        );
      }
    }
  }

  // Additional input validation (just in case).
  for (const [idx, size] of sizeMap) {
    if (!Number.isInteger(idx) || idx < 0) {
      throw new Error(
        `Invalid index ${idx} in einsum expression, must be non-negative integer`,
      );
    } else if (size < 0) {
      throw new Error(
        `Invalid size ${size} for index ${idx} in einsum expression, must be non-negative`,
      );
    }
  }
  for (const idx of rhsIndex) {
    if (!sizeMap.has(idx)) {
      throw new Error(`Output index ${idx} not present in einsum inputs`);
    }
  }

  return sizeMap;
}

/** @inline */
export type ComputePathMethod = "naive" | "optimal";

export function computePath(
  input: EinsumInput,
  method?: ComputePathMethod,
): EinsumPath {
  const sizeMap = computeSizeMap(input);
  if (input.shapes.length === 1) {
    // Trivial case, just return the empty path.
    return new EinsumPath(input, sizeMap, []);
  }
  if (!method) {
    method = input.shapes.length <= 4 ? "optimal" : "naive";
  }
  switch (method) {
    case "naive":
      return computePathNaive(input, sizeMap);
    case "optimal":
      return computePathOptimal(input, sizeMap);
    default:
      throw new Error(`Unknown computePath method: ${method}`);
  }
}

function computePathNaive(input: EinsumInput, sizeMap: Map<number, bigint>) {
  const n = input.shapes.length;
  const path: [number, number][] = [];
  let lastTensorIndex = 0;
  for (let i = 1; i < n; i++) {
    path.push([lastTensorIndex, i]);
    lastTensorIndex = n + i - 1;
  }
  return new EinsumPath(input, sizeMap, path);
}

function computePathOptimal(input: EinsumInput, sizeMap: Map<number, bigint>) {
  const n = input.shapes.length;
  let bestPath: [number, number][] | null = null;
  let bestFlops: bigint | null = null;
  for (const path of allPaths(range(n), n)) {
    const flops = approximatePathFlops(input, sizeMap, path);
    if (bestFlops === null || flops < bestFlops) {
      bestPath = path;
      bestFlops = flops;
    }
  }
  return new EinsumPath(input, sizeMap, bestPath!);
}

// Note: This is slow, I think it scales with O((n!)^2).
function* allPaths(
  tensors: number[],
  next: number,
): IterableIterator<[number, number][]> {
  // Must be at least two tensors, base case.
  if (tensors.length === 2) {
    yield [[tensors[0], tensors[1]]];
    return;
  }
  for (let i = 0; i < tensors.length; i++) {
    for (let j = i + 1; j < tensors.length; j++) {
      const pair: [number, number] = [tensors[i], tensors[j]];
      const remaining = tensors.filter((t) => t !== pair[0] && t !== pair[1]);
      const newTensors = [...remaining, next];
      for (const subpath of allPaths(newTensors, next + 1)) {
        yield [pair, ...subpath];
      }
    }
  }
}
