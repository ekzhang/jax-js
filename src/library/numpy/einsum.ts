import { range } from "../../utils";

const bprod = (...xs: bigint[]) => xs.reduce((acc, x) => acc * x, 1n);
const bmax = (...xs: bigint[]) => xs.reduce((max, x) => (x > max ? x : max));

const EINSUM_INDEX_RE = /\p{ID_Start}/gu;

// (shape, lhsIndices, shape, lhsIndices, rhsIndex)
// (x, (0,), y, (0,), ())
export interface EinsumInput {
  shapes: bigint[][];
  lhsIndices: number[][];
  rhsIndex: number[];
}

export function parseEinsumExpression(
  expr: string,
): Pick<EinsumInput, "lhsIndices" | "rhsIndex"> {
  const idents = [
    ...expr
      .split("->")[0]
      .matchAll(EINSUM_INDEX_RE)
      .map((m) => m[0]),
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
      .sort((a, b) => a.codePointAt(0)! - b.codePointAt(0)!);
    expr += "->" + outputIndices.join("");
  }

  const identToIndex = new Map<string, number>(
    [...new Set(idents)].sort().map((c, i) => [c, i]),
  );

  const [lhs, rhs] = expr.split("->");
  const lhsParts = lhs.split(",");
  const lhsIndices = lhsParts.map((part) =>
    [...part.matchAll(EINSUM_INDEX_RE)].map((m) => identToIndex.get(m[0])!),
  );
  const rhsIndex = [...rhs.matchAll(EINSUM_INDEX_RE)].map((m) => {
    const idx = identToIndex.get(m[0]);
    if (idx === undefined)
      throw new Error(`Output index ${m[0]} not present in inputs`);
    return idx;
  });

  return { lhsIndices, rhsIndex };
}

export class EinsumPath {
  /** Shape of each original tensor input for the einsum. */
  readonly input: EinsumInput;

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

  /** Mapping of each index number to its size in the shape array. */
  readonly sizeMap: Map<number, bigint>;

  constructor(
    input: EinsumInput,
    path: [number, number][],
    sizeMap: Map<number, bigint>,
  ) {
    this.input = input;
    this.path = path;
    this.sizeMap = sizeMap;
  }

  /** Shape of the final output tensor. */
  get outputShape(): bigint[] {
    return this.input.rhsIndex.map((i) => this.sizeMap.get(i)!);
  }

  /** Estimate the number of FLOPs to execute this einsum path. */
  get approximateFlops(): bigint {
    if (this.path.length == 0) {
      // Special case: 0-length path returned if there's only one input tensor.
      // This is the case if we take the trace or transpose.
      if (this.input.shapes.length !== 1)
        throw new Error("internal: invariant, empty path for multiple tensors");
      return bprod(...this.outputShape.map(BigInt));
    }
    let totalFlops = 0n;

    // Will include indices and shapes of intermediate tensors.
    const indices = [...this.input.lhsIndices];

    const indexUsageCounts = new Map<number, number>();
    for (const idx of [
      ...this.input.lhsIndices.flat(),
      ...this.input.rhsIndex,
    ]) {
      indexUsageCounts.set(idx, (indexUsageCounts.get(idx) ?? 0) + 1);
    }

    for (const tensorGroup of this.path) {
      const indexReduced: number[] = [];
      const indexGroup: number[] = [];
      for (const tensorIdx of tensorGroup) {
        for (const idx of indices[tensorIdx]) {
          if (!indexGroup.includes(idx)) {
            indexGroup.push(idx);
          }
          // If the index is not in the output and isn't in any other inputs,
          // we can consider it reduced here.
          const otherUsages = indexUsageCounts.get(idx)! - 1;
          indexUsageCounts.set(idx, otherUsages);
          if (otherUsages === 0) {
            indexReduced.push(idx);
          }
        }
      }
      totalFlops += approximateCountFlops(
        indexGroup,
        indexReduced.length > 0,
        tensorGroup.length,
        this.sizeMap,
      );
      indices.push(indexGroup.filter((x) => !indexReduced.includes(x)).sort());
    }
    return totalFlops;
  }
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

/** Compute size for each index in the einsum expression. */
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
  method: ComputePathMethod = "naive",
): EinsumPath {
  switch (method) {
    case "naive":
      return computeNaivePath(input);
    case "optimal":
      return computeOptimalPath(input);
  }
}

function computeNaivePath(input: EinsumInput) {
  // Validate input and compute size map.
  const sizeMap = computeSizeMap(input);

  return new EinsumPath(input, [range(input.shapes.length)], [], sizeMap);
}

function* permutations(arr: number[]) {
  const c = Array(arr.length).fill(0);
  yield [...arr];

  let i = 0;
  while (i < arr.length) {
    if (c[i] < i) {
      const swapIdx = i % 2 === 0 ? 0 : c[i];
      [arr[i], arr[swapIdx]] = [arr[swapIdx], arr[i]];
      yield [...arr];
      c[i]++;
      i = 0;
    } else {
      c[i] = 0;
      i++;
    }
  }
}

function computeOptimalPath(input: EinsumInput) {
  // Validate input and compute size map.
  const sizeMap = computeSizeMap(input);

  const indices = range(0, input.shapes.length);
  for (const indicesPermutation of permutations(indices)) {
    // ... make every one of these einsum paths
  }

  // TODO: try every combination and pick the best one.

  return new EinsumPath(input, [range(input.shapes.length)], [], sizeMap);
}

// if an index appears in exactly one input, and it is not in the output,
// you can sum it inside that tensor immediately.
