import { AluExp, AluOp, byteWidth, DType, Kernel } from "../../alu";
import { findPow2, gcd } from "../../utils";

export const simdLanes = 4;

export type StrideResult =
  | { kind: "broadcast"; tileSize: number }
  | { kind: "contiguous"; tileSize: number }
  | { kind: "gather" };

export interface ReductionPointerCandidate {
  exp: AluExp;
  gid: number;
  dtype: DType;
  stride: StrideResult;
  baseIndex: AluExp;
  strideBytes: number;
}

export interface ReductionTilePlan {
  tileSize: number;
  tileRows: number;
  tileVectors: number;
  tileK: number;
  microRows: number;
  microVectors: number;
}

export interface ReductionKTilePlan {
  tileSize: number;
  tileRows: number;
  tileCols: number;
  microRows: number;
  microCols: number;
  kUnroll: number;
}

const TILED_SIMD_ROWS = 128;
const TILED_SIMD_COLUMNS = 128;
const TILED_SIMD_K = 64;
const TILED_SIMD_MICRO_ROWS = 4;
const TILED_SIMD_MICRO_VECTORS = 4;
const K_SIMD_MICRO_ROWS = 4;
const K_SIMD_MICRO_COLS = 4;
const K_SIMD_MAX_UNROLL = 4;

function isSymbol(exp: AluExp, name: string): boolean {
  return (
    (exp.op === AluOp.Variable && exp.arg === name) ||
    (exp.op === AluOp.Special && exp.arg[0] === name)
  );
}

function referencesSymbol(exp: AluExp, name: string): boolean {
  return (
    isSymbol(exp, name) || exp.src.some((src) => referencesSymbol(src, name))
  );
}

function referencesGidx(exp: AluExp): boolean {
  return referencesSymbol(exp, "gidx");
}

function hasFragmentRisk(tileSize: number, N: number): boolean {
  return isFinite(tileSize) && tileSize > N && tileSize % N !== 0;
}

function constInt(exp: AluExp): number | null {
  if (exp.op !== AluOp.Const) return null;
  const value = exp.arg as number;
  return Number.isInteger(value) ? value : null;
}

function coefficientOfSymbol(exp: AluExp, name: string): number | null {
  if (!referencesSymbol(exp, name)) return 0;
  if (isSymbol(exp, name)) return 1;
  if (exp.op === AluOp.Add || exp.op === AluOp.Sub) {
    const a = coefficientOfSymbol(exp.src[0], name);
    const b = coefficientOfSymbol(exp.src[1], name);
    if (a === null || b === null) return null;
    return exp.op === AluOp.Add ? a + b : a - b;
  }
  if (exp.op === AluOp.Mul) {
    const lhs = constInt(exp.src[0]);
    if (lhs !== null) {
      const rhsCoeff = coefficientOfSymbol(exp.src[1], name);
      return rhsCoeff === null ? null : lhs * rhsCoeff;
    }
    const rhs = constInt(exp.src[1]);
    if (rhs !== null) {
      const lhsCoeff = coefficientOfSymbol(exp.src[0], name);
      return lhsCoeff === null ? null : rhs * lhsCoeff;
    }
  }
  return null;
}

function rewriteSymbol(
  exp: AluExp,
  name: string,
  rewrite: (node: AluExp) => AluExp,
): AluExp {
  return exp
    .rewrite((node) => (isSymbol(node, name) ? rewrite(node) : undefined))
    .simplify();
}

function repeatsAcrossGidxTile(exp: AluExp, tileSize: number): boolean {
  if (!isFinite(tileSize)) return false;
  const shifted = rewriteSymbol(exp, "gidx", (node) =>
    AluExp.add(node, AluExp.i32(tileSize)),
  );
  return shifted.getHash() === exp.getHash();
}

function divisorAtMost(value: number, limit: number): number {
  for (let i = Math.min(value, limit); i > 1; i--) {
    if (value % i === 0) return i;
  }
  return 1;
}

function commonTileSize(
  kernelSize: number,
  strideMap: Map<AluExp, StrideResult>,
  minTileSize: number,
  unconstrainedTileSize: number | null = null,
): number | null {
  const tileSizes: number[] = [];
  for (const stride of strideMap.values()) {
    if (stride.kind !== "gather" && isFinite(stride.tileSize))
      tileSizes.push(stride.tileSize);
  }
  if (tileSizes.length === 0) return unconstrainedTileSize;

  const tileSize = Math.min(...tileSizes);
  if (
    tileSize < minTileSize ||
    kernelSize % tileSize !== 0 ||
    tileSizes.some((size) => size % tileSize !== 0)
  ) {
    return null;
  }
  return tileSize;
}

function tiledRows(kernelSize: number, tileSize: number): number {
  const rowCount = Math.floor(kernelSize / tileSize);
  return findPow2(0, Math.max(1, Math.min(rowCount / 2, TILED_SIMD_ROWS)));
}

function tiledColumns(tileSize: number, laneWidth: number): number {
  const columns = tileSize / laneWidth;
  return findPow2(
    0,
    Math.max(
      1,
      Math.min(columns / 2, Math.floor(TILED_SIMD_COLUMNS / laneWidth)),
    ),
  );
}

function microTile(tile: number, axis: number, limit: number): number {
  return divisorAtMost(gcd(tile, axis), limit);
}

function microTileWithTail(tile: number, limit: number): number {
  return Math.max(1, Math.min(tile, limit));
}

function periodicStride(exp: AluExp, kind: "broadcast" | "contiguous") {
  if (exp.src[1].op !== AluOp.Const) return null;
  const N = exp.src[1].arg as number;
  const inner = analyzeStride(exp.src[0]);
  if (inner.kind === "broadcast") return inner;
  if (inner.kind !== "contiguous" || hasFragmentRisk(inner.tileSize, N))
    return { kind: "gather" } as const;
  return { kind, tileSize: Math.min(inner.tileSize, N) };
}

function addStrides(lhs: StrideResult, rhs: StrideResult): StrideResult {
  if (lhs.kind === "gather" || rhs.kind === "gather") return { kind: "gather" };

  const tileSize = Math.min(lhs.tileSize, rhs.tileSize);
  if (lhs.kind === "broadcast") return { kind: rhs.kind, tileSize };
  if (rhs.kind === "broadcast") return { kind: lhs.kind, tileSize };

  // Adding two contiguous terms produces stride 2, not a contiguous load.
  return { kind: "gather" };
}

function analyzeStride(exp: AluExp): StrideResult {
  if (!referencesGidx(exp)) return { kind: "broadcast", tileSize: Infinity };
  if (exp.op === AluOp.Special && exp.arg[0] === "gidx")
    return { kind: "contiguous", tileSize: Infinity };

  if (exp.op === AluOp.Idiv || exp.op === AluOp.Mod) {
    const stride = periodicStride(
      exp,
      exp.op === AluOp.Idiv ? "broadcast" : "contiguous",
    );
    if (stride) return stride;
  }

  if (exp.op === AluOp.Mul) {
    for (let i = 0; i < 2; i++) {
      if (exp.src[i].op === AluOp.Const) {
        const inner = analyzeStride(exp.src[1 - i]);
        if (inner.kind === "broadcast") return inner;
        return { kind: "gather" };
      }
    }
  }

  if (exp.op === AluOp.Add) {
    return addStrides(analyzeStride(exp.src[0]), analyzeStride(exp.src[1]));
  }

  return { kind: "gather" };
}

function simdStrideResult(globalIndex: AluExp): StrideResult {
  const index = globalIndex.src[0];
  const result = analyzeStride(index);
  const [_, len] = globalIndex.arg as [number, number];
  if (
    result.kind !== "gather" &&
    (result.tileSize < simdLanes ||
      (isFinite(result.tileSize) && result.tileSize % simdLanes !== 0))
  ) {
    return { kind: "gather" };
  }
  if (result.kind === "contiguous" && (index.min < 0 || index.max >= len)) {
    return { kind: "gather" };
  }
  return result;
}

export function collectSimdStrides(
  exp: AluExp | undefined,
): Map<AluExp, StrideResult> {
  return new Map(
    (exp?.collect((node) => node.op === AluOp.GlobalIndex) ?? []).map(
      (gi) => [gi, simdStrideResult(gi)] as const,
    ),
  );
}

export function reductionPointerCandidates(
  exp: AluExp,
  strideMap?: Map<AluExp, StrideResult>,
): ReductionPointerCandidate[] {
  const candidates: ReductionPointerCandidate[] = [];
  for (const gi of exp.collect((node) => node.op === AluOp.GlobalIndex)) {
    const stride = strideMap?.get(gi) ?? {
      kind: "broadcast",
      tileSize: Infinity,
    };
    if (strideMap && stride.kind === "gather") continue;

    const [gid, len] = gi.arg as [number, number];
    const index = gi.src[0];
    const strideElems = coefficientOfSymbol(index, "ridx");
    if (strideElems === null || !Number.isInteger(strideElems)) continue;
    if (index.min < 0 || index.max >= len) continue;

    candidates.push({
      exp: gi,
      gid,
      dtype: gi.dtype,
      stride,
      baseIndex: rewriteSymbol(index, "ridx", () => AluExp.i32(0)),
      strideBytes: strideElems * byteWidth(gi.dtype),
    });
  }
  return candidates;
}

export function reductionTilePlan(
  kernel: Kernel,
  strideMap: Map<AluExp, StrideResult>,
  canStorePartial: boolean = true,
): ReductionTilePlan | null {
  if (!kernel.reduction) return null;

  const tileSize = commonTileSize(kernel.size, strideMap, simdLanes, simdLanes);
  if (tileSize === null) return null;

  const tileRows = tiledRows(kernel.size, tileSize);
  const tileVectors = tiledColumns(tileSize, simdLanes);
  return {
    tileSize,
    tileRows,
    tileVectors,
    tileK: canStorePartial
      ? divisorAtMost(kernel.reduction.size, TILED_SIMD_K)
      : kernel.reduction.size,
    microRows: microTile(
      tileRows,
      Math.floor(kernel.size / tileSize),
      TILED_SIMD_MICRO_ROWS,
    ),
    microVectors: microTileWithTail(tileVectors, TILED_SIMD_MICRO_VECTORS),
  };
}

export function reductionKTilePlan(
  kernel: Kernel,
  strideMap: Map<AluExp, StrideResult>,
): ReductionKTilePlan | null {
  if (!kernel.reduction) return null;
  if (kernel.reduction.size % simdLanes !== 0) return null;

  const tileSize = commonTileSize(kernel.size, strideMap, 1, 1);
  if (tileSize === null) return null;

  const kUnroll = divisorAtMost(
    kernel.reduction.size / simdLanes,
    K_SIMD_MAX_UNROLL,
  );
  const tileRows = tiledRows(kernel.size, tileSize);
  const tileCols = tiledColumns(tileSize, 1);
  return {
    tileSize,
    tileRows,
    tileCols,
    microRows: microTile(
      tileRows,
      Math.floor(kernel.size / tileSize),
      K_SIMD_MICRO_ROWS,
    ),
    microCols: microTile(tileCols, tileSize, K_SIMD_MICRO_COLS),
    kUnroll,
  };
}

export function pointerShareKey(
  candidate: ReductionPointerCandidate,
  row: number,
  vector: number,
  groupIndex: number,
): string {
  const hash = candidate.exp.getHash().toString();
  const { stride } = candidate;
  if (stride.kind === "broadcast") {
    return isFinite(stride.tileSize) ? `${hash}:row${row}` : `${hash}:all`;
  }
  if (stride.kind === "contiguous" && isFinite(stride.tileSize)) {
    return repeatsAcrossGidxTile(candidate.baseIndex, stride.tileSize)
      ? `${hash}:vec${vector}`
      : `${hash}:row${row}:vec${vector}`;
  }
  return `${hash}:g${groupIndex}`;
}

export function kReductionPointerShareKey(
  candidate: ReductionPointerCandidate,
  outputStride: StrideResult,
  tileSize: number,
  row: number,
  col: number,
  groupIndex: number,
): string {
  const hash = candidate.exp.getHash().toString();
  if (outputStride.kind === "broadcast" && isFinite(outputStride.tileSize)) {
    return `${hash}:row${row}`;
  }
  if (repeatsAcrossGidxTile(candidate.baseIndex, tileSize)) {
    return `${hash}:col${col}`;
  }
  return `${hash}:g${groupIndex}`;
}
