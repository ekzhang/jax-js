import { range, runWithCache } from "../../utils";

const bprod = (...xs: number[]) => xs.reduce((acc, x) => acc * BigInt(x), 1n);
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
  shapes: number[][];
  lhsIndices: number[][];
  rhsIndex: number[];
}

const einsumParseCache = new Map<string, EinsumInput>();

/** Parse an Einstein notation string into a runnable `EinsumInput`. */
export function parseEinsumExpression(
  expr: string,
  shapes: number[][],
): EinsumInput {
  return runWithCache(einsumParseCache, [expr, shapes], () => {
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
    const componentsToIndices = (components: string[], rank?: number) =>
      components.flatMap((c) => {
        if (c === "...") {
          // Full rank if `(components.length - 1) + ellipsisRank === rank`
          const start =
            rank !== undefined
              ? components.length - 1 + ellipsisRank - rank
              : 0;
          return range(
            identToIndex.size + start,
            identToIndex.size + ellipsisRank!,
          );
        }
        return identToIndex.get(c)!;
      });

    let ellipsisRank: number = 0; // Number of dimensions that "..." represents

    const [lhs, rhs] = expr.split("->");
    const lhsComponents = lhs
      .split(",")
      .map((part) => [...part.matchAll(EINSUM_COMPONENT_RE).map((m) => m[0])]);
    const rhsComponents = [...rhs.matchAll(EINSUM_COMPONENT_RE)].map(
      (m) => m[0],
    );

    // Compute the rank of the ellipsis by looking at the lhs operands, we choose
    // the maximum rank because of broadcasting.
    for (const [i, components] of lhsComponents.entries()) {
      const shape = shapes[i];
      const ellipsisIndex = components.indexOf("...");
      if (ellipsisIndex !== -1) {
        if (components.lastIndexOf("...") !== ellipsisIndex)
          throw new Error(
            "Multiple ellipses in one einsum operand is not allowed",
          );
        const numExplicit = components.length - 1;
        if (shape.length < numExplicit)
          throw new Error(
            `Einsum operand ${i} has shape ${JSON.stringify(shape)} but indexed with "${components.join("")}"`,
          );
        ellipsisRank = Math.max(ellipsisRank, shape.length - numExplicit);
      }
    }

    const lhsIndices = lhsComponents.map((components, i) =>
      componentsToIndices(components, shapes[i].length),
    );
    const rhsIndex = componentsToIndices(rhsComponents);
    return { shapes, lhsIndices, rhsIndex };
  });
}

export class EinsumPath {
  /** Parsed and normalized input for the einsum. */
  readonly input: EinsumInput;

  /** Mapping of each index number to its size in the shape array. */
  readonly sizeMap: Map<number, number>;

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
    sizeMap: Map<number, number>,
    path: [number, number][],
  ) {
    this.input = input;
    this.sizeMap = sizeMap;
    this.path = path;
  }

  /** Shape of the final output tensor. */
  get outputShape(): number[] {
    return this.input.rhsIndex.map((i) => this.sizeMap.get(i)!);
  }

  /** Estimate the number of FLOPs to execute this einsum path. */
  get approximateFlops(): bigint {
    return approximatePathFlops(this.input, this.sizeMap, this.path);
  }
}

function approximatePathFlops(
  input: EinsumInput,
  sizeMap: Map<number, number>,
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
    const newIndex = indexGroup.filter((x) => !indexReduced.includes(x));
    for (const idx of newIndex) indexUsageCounts[idx]++;
    indices.push(newIndex);
  }
  return totalFlops;
}

function approximateCountFlops(
  indexGroup: number[],
  hasReduction: boolean,
  numTerms: number,
  sizeMap: Map<number, number>,
): bigint {
  const elements = bprod(...indexGroup.map((i) => sizeMap.get(i)!));
  const flopsPerLoopIteration =
    BigInt(numTerms) - 1n + (hasReduction ? 1n : 0n);
  return elements * flopsPerLoopIteration;
}

/** Compute size for each index in the einsum expression, also validates input. */
function computeSizeMap({
  shapes,
  lhsIndices,
  rhsIndex,
}: EinsumInput): Map<number, number> {
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

  const sizeMap = new Map<number, number>();
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

const einsumPathCache = new Map<string, EinsumPath>();

/** @inline */
export type ComputePathMethod = "naive" | "optimal";

export function computeEinsumPath(
  input: EinsumInput,
  method?: ComputePathMethod,
): EinsumPath {
  if (!method) {
    method = input.shapes.length <= 5 ? "optimal" : "naive";
  }
  return runWithCache(einsumPathCache, [input, method], () => {
    const sizeMap = computeSizeMap(input);
    if (input.shapes.length === 1) {
      // Trivial case, just return the empty path.
      return new EinsumPath(input, sizeMap, []);
    }
    switch (method) {
      case "naive":
        return computePathNaive(input, sizeMap);
      case "optimal":
        return computePathOptimal(input, sizeMap);
      default:
        throw new Error(`Unknown computePath method: ${method}`);
    }
  });
}

function computePathNaive(input: EinsumInput, sizeMap: Map<number, number>) {
  const n = input.shapes.length;
  const path: [number, number][] = [];
  let lastTensorIndex = 0;
  for (let i = 1; i < n; i++) {
    path.push([lastTensorIndex, i]);
    lastTensorIndex = n + i - 1;
  }
  return new EinsumPath(input, sizeMap, path);
}

function computePathOptimal(input: EinsumInput, sizeMap: Map<number, number>) {
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
      const newTensors = tensors.filter((t) => t !== pair[0] && t !== pair[1]);
      newTensors.push(next);
      for (const subpath of allPaths(newTensors, next + 1)) {
        yield [pair, ...subpath];
      }
    }
  }
}
