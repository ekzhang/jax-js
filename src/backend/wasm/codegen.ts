import {
  wasm_asin,
  wasm_atan,
  wasm_cos,
  wasm_erf,
  wasm_erfc,
  wasm_exp,
  wasm_log,
  wasm_sin,
  wasm_threefry2x32,
} from "./builtins";
import { hasWasmFeature } from "./featureProbe";
import { hasSharedArrayBuffer } from "./parallel";
import {
  collectSimdStrides,
  kReductionPointerShareKey,
  pointerShareKey,
  reductionKTilePlan,
  type ReductionKTilePlan,
  type ReductionPointerCandidate,
  reductionPointerCandidates,
  reductionTilePlan,
  type ReductionTilePlan,
  simdLanes,
  type StrideResult,
} from "./tilePlan";
import {
  dty,
  dtyF,
  type SimdPointerPlan,
  simdSupportedOps,
  translateExp,
  translateExpSimd,
} from "./translation";
import { CodeGenerator } from "./wasmblr";
import {
  AluExp,
  AluOp,
  byteWidth,
  DType,
  isFloatDtype,
  Kernel,
} from "../../alu";
import { tuneNullopt } from "../../tuner";
import { DEBUG, mapSetUnion, rep } from "../../utils";

interface WasmCodegenResult {
  bytes: Uint8Array<ArrayBuffer>;
  workSize: number;
  chunkAlignment: number;
  minWorkPerWorker: number;
}

type ReductionPointerPlan = ReductionPointerCandidate & SimdPointerPlan;

interface ReductionGroup {
  gidx: number;
  row: number;
  vector: number;
}

interface KReductionGroup {
  gidx: number;
  row: number;
  col: number;
}

interface ReductionPointerMaps {
  pointerMaps: Map<AluExp, ReductionPointerPlan>[];
  uniquePointers: ReductionPointerPlan[];
}

type WasmReduction = NonNullable<Kernel["reduction"]>;

function isIdentityEpilogue(exp: AluExp | undefined): boolean {
  return exp?.op === AluOp.Variable && exp.arg === "acc";
}

function initializeReductionPointer(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  candidate: ReductionPointerCandidate,
  ctx: Record<string, number>,
  valueKey?: string,
  ridxOffset?: number,
): ReductionPointerPlan {
  const ptr = cg.local.declare(cg.i32);
  translateExp(cg, funcs, candidate.baseIndex, ctx);
  cg.i32.const(byteWidth(candidate.dtype));
  cg.i32.mul();
  cg.local.get(candidate.gid);
  cg.i32.add();
  if (ridxOffset !== undefined && candidate.strideBytes !== 0) {
    cg.local.get(ridxOffset);
    cg.i32.const(candidate.strideBytes);
    cg.i32.mul();
    cg.i32.add();
  }
  cg.local.set(ptr);
  return { ...candidate, ptr, valueKey };
}

function incrementReductionPointers(
  cg: CodeGenerator,
  pointers: Iterable<ReductionPointerPlan>,
  multiplier: number = 1,
): void {
  for (const pointer of pointers) {
    if (pointer.strideBytes === 0) continue;
    cg.local.get(pointer.ptr);
    cg.i32.const(pointer.strideBytes * multiplier);
    cg.i32.add();
    cg.local.set(pointer.ptr);
  }
}

function emitSimdReductionOp(
  cg: CodeGenerator,
  re: WasmReduction,
  reIsInt: boolean,
  valueAlreadyAccumulated: boolean,
): void {
  if (!reIsInt && valueAlreadyAccumulated) return;

  switch (re.op) {
    case AluOp.Add:
      if (reIsInt) cg.i32x4.add();
      else cg.f32x4.add();
      return;
    case AluOp.Mul:
      if (reIsInt) cg.i32x4.mul();
      else cg.f32x4.mul();
      return;
    case AluOp.Min:
      if (reIsInt) {
        if (re.dtype === DType.Int32) cg.i32x4.min_s();
        else cg.i32x4.min_u();
      } else cg.f32x4.min();
      return;
    case AluOp.Max:
      if (reIsInt) {
        if (re.dtype === DType.Int32) cg.i32x4.max_s();
        else cg.i32x4.max_u();
      } else cg.f32x4.max();
      return;
    default:
      throw new Error(`invalid SIMD reduction op: ${re.op}`);
  }
}

function emitScalarReductionOp(
  cg: CodeGenerator,
  re: WasmReduction,
  acc: number,
): void {
  switch (re.op) {
    case AluOp.Add:
      cg.local.get(acc);
      if (re.dtype === DType.Bool) cg.i32.or();
      else dty(cg, re.op, re.dtype).add();
      return;
    case AluOp.Mul:
      cg.local.get(acc);
      if (re.dtype === DType.Bool) cg.i32.and();
      else dty(cg, re.op, re.dtype).mul();
      return;
    case AluOp.Min:
    case AluOp.Max:
      if (isFloatDtype(re.dtype)) {
        cg.local.get(acc);
        if (re.op === AluOp.Min) dtyF(cg, re.op, re.dtype).min();
        else dtyF(cg, re.op, re.dtype).max();
      } else if ([DType.Int32, DType.Uint32, DType.Bool].includes(re.dtype)) {
        // Wasm has no i32.min/max, so emulate with select.
        const local = cg.local.declare(cg.i32);
        cg.local.tee(local);
        cg.local.get(acc);
        cg.local.get(local);
        cg.local.get(acc);
        if (re.op === AluOp.Min) {
          if (re.dtype === DType.Int32) cg.i32.lt_s();
          else cg.i32.lt_u();
        } else {
          if (re.dtype === DType.Int32) cg.i32.gt_s();
          else cg.i32.gt_u();
        }
        cg.select();
      } else throw new Error(`invalid reduction min/max over ${re.dtype}`);
      return;
    default:
      throw new Error(`invalid wasm reduction op: ${re.op}`);
  }
}

/**
 * Check if a kernel is eligible for SIMD codegen.
 *
 * A kernel qualifies when:
 * - size >= 4 (need at least 4 elements for a SIMD group)
 * - For reductions: the reduction op has a SIMD variant for its dtype
 * - All nodes have a supported dtype (f32, i32, u32, bool) with SIMD variants
 */
function isSimdEligible(tunedExp: AluExp, kernel: Kernel): boolean {
  if (kernel.size < simdLanes) return false;
  if (kernel.reduction) {
    if (!simdSupportedOps.get(kernel.reduction.dtype)?.has(kernel.reduction.op))
      return false;
  }

  const check = (exp: AluExp, visited: Set<AluExp>): boolean => {
    if (visited.has(exp)) return true;
    visited.add(exp);

    if (!simdSupportedOps.get(exp.dtype)?.has(exp.op)) return false;

    // GlobalIndex: skip the index subtree. It is evaluated scalarly
    // (via translateExp), either once for contiguous wide loads or
    // four times with lane offsets for the gather fallback.
    if (exp.op === AluOp.GlobalIndex) return true;
    for (const child of exp.src) {
      if (!check(child, visited)) return false;
    }
    return true;
  };

  return check(tunedExp, new Set());
}

/** Checks if SIMD over the reduced-k dimension is workable. */
function canUseKSimdReduction(
  exp: AluExp,
  re: WasmReduction,
  pointers: ReductionPointerCandidate[],
): boolean {
  if (re.op !== AluOp.Add) return false;

  const globalIndexCount = exp.collect(
    (node) => node.op === AluOp.GlobalIndex,
  ).length;
  return (
    globalIndexCount === pointers.length &&
    pointers.every(
      (candidate) =>
        candidate.dtype === DType.Float32 &&
        (candidate.strideBytes === 0 ||
          candidate.strideBytes === byteWidth(candidate.dtype)),
    )
  );
}

/** Emit a runtime guard: enter the if-block only when [begin, end) is SIMD-aligned. */
function emitAlignmentGuard(
  cg: CodeGenerator,
  paramBegin: number,
  paramEnd: number,
  alignment: number = simdLanes,
): void {
  cg.local.get(paramEnd);
  cg.local.get(paramBegin);
  cg.i32.sub();
  cg.i32.const(alignment);
  cg.i32.rem_u();
  cg.i32.eqz(); // (end - begin) % alignment === 0
  cg.local.get(paramBegin);
  cg.i32.const(alignment);
  cg.i32.rem_u();
  cg.i32.eqz(); // begin % alignment === 0
  cg.i32.and();
  cg.if(cg.void);
}

export function codegenWasm(kernel: Kernel): WasmCodegenResult {
  const tune = tuneNullopt(kernel);
  const re = kernel.reduction;

  if (DEBUG >= 3) {
    console.info(`kernel.exp: ${kernel.exp}\ntune.exp: ${tune.exp}`);
  }

  const simdEligible = isSimdEligible(tune.exp, kernel);
  const hasIdentityEpilogue = isIdentityEpilogue(tune.epilogue);

  // Classify each GlobalIndex w.r.t. gidx for the SIMD body.
  const expStrides = simdEligible
    ? collectSimdStrides(tune.exp)
    : new Map<AluExp, StrideResult>();
  const reductionHasLaneGather =
    re && [...expStrides.values()].some((stride) => stride.kind === "gather");
  const useSimd = simdEligible && !reductionHasLaneGather;
  const simdReductionPointerCandidates =
    useSimd && re ? reductionPointerCandidates(tune.exp, expStrides) : [];
  const reductionPointers = re ? reductionPointerCandidates(tune.exp) : [];
  const useKSimdReduction =
    simdEligible && re && canUseKSimdReduction(tune.exp, re, reductionPointers);
  const kSimdReductionPointerCandidates =
    useKSimdReduction && re
      ? reductionPointers.map((candidate) => ({
          ...candidate,
          stride:
            candidate.strideBytes === 0
              ? ({ kind: "broadcast", tileSize: Infinity } as const)
              : ({ kind: "contiguous", tileSize: Infinity } as const),
        }))
      : [];
  const canStoreSimdPartials =
    hasIdentityEpilogue ||
    (re !== undefined &&
      kernel.dtype === re.dtype &&
      byteWidth(kernel.dtype) === 4);
  const simdTilePlan =
    useSimd && re
      ? reductionTilePlan(kernel, expStrides, canStoreSimdPartials)
      : null;
  const kSimdTilePlan =
    reductionHasLaneGather && useKSimdReduction
      ? reductionKTilePlan(kernel, expStrides)
      : null;
  const useRelaxedMadd =
    hasWasmFeature("relaxed-madd") &&
    re?.op === AluOp.Add &&
    tune.exp.dtype === DType.Float32 &&
    tune.exp.op === AluOp.Mul;

  const cg = new CodeGenerator();
  cg.memory.import("env", "memory");
  if (hasSharedArrayBuffer()) {
    cg.memory.pages(0, 65536).shared(true);
  }

  const distinctOps = mapSetUnion(
    tune.exp.distinctOps(),
    tune.epilogue?.distinctOps(),
  );
  const funcs: Record<string, number> = {};
  if (distinctOps.has(AluOp.Sin)) funcs.sin = wasm_sin(cg);
  if (distinctOps.has(AluOp.Cos)) funcs.cos = wasm_cos(cg);
  if (distinctOps.has(AluOp.Asin)) funcs.asin = wasm_asin(cg);
  if (distinctOps.has(AluOp.Atan)) funcs.atan = wasm_atan(cg);
  if (
    distinctOps.has(AluOp.Exp) ||
    distinctOps.has(AluOp.Erf) ||
    distinctOps.has(AluOp.Erfc)
  )
    funcs.exp = wasm_exp(cg);
  if (distinctOps.has(AluOp.Log)) funcs.log = wasm_log(cg);
  if (distinctOps.has(AluOp.Erf)) funcs.erf = wasm_erf(cg, funcs.exp);
  if (distinctOps.has(AluOp.Erfc)) funcs.erfc = wasm_erfc(cg, funcs.exp);
  if (distinctOps.has(AluOp.Threefry2x32))
    funcs.threefry2x32 = wasm_threefry2x32(cg);

  const paramBegin = kernel.nargs + 1;
  const paramEnd = kernel.nargs + 2;
  const kernelFunc = cg.function(rep(kernel.nargs + 3, cg.i32), [], () => {
    const gidx = cg.local.declare(cg.i32);
    cg.local.get(paramBegin);
    cg.local.set(gidx);

    const emitLocalPlusConst = (local: number, amount: number) => {
      cg.local.get(local);
      cg.i32.const(amount);
      cg.i32.add();
    };

    const bumpLocal = (local: number, amount: number) => {
      emitLocalPlusConst(local, amount);
      cg.local.set(local);
    };

    const setLocalConst = (local: number, value: number) => {
      cg.i32.const(value);
      cg.local.set(local);
    };

    const copyLocal = (target: number, source: number) => {
      cg.local.get(source);
      cg.local.set(target);
    };

    const setRowBase = (
      target: number,
      rowTileBase: number,
      rowOffset: number,
      tileSize: number,
    ) => {
      cg.local.get(rowTileBase);
      cg.local.get(rowOffset);
      cg.i32.const(tileSize);
      cg.i32.mul();
      cg.i32.add();
      cg.local.set(target);
    };

    const emitOutputAddress = (index: number) => {
      cg.local.get(kernel.nargs);
      cg.local.get(index);
      cg.i32.const(byteWidth(kernel.dtype));
      cg.i32.mul();
      cg.i32.add();
    };

    const declareTileGidx = (
      rowBase: number,
      col: number,
      rowOffset: number,
      colOffset: number,
    ) => {
      const local = cg.local.declare(cg.i32);
      cg.local.get(rowBase);
      if (rowOffset !== 0) {
        cg.i32.const(rowOffset);
        cg.i32.add();
      }
      cg.local.get(col);
      cg.i32.add();
      if (colOffset !== 0) {
        cg.i32.const(colOffset);
        cg.i32.add();
      }
      cg.local.set(local);
      return local;
    };

    const emitLoopWithBreaks = (
      emitBreaks: () => void,
      emitBody: () => void,
    ) => {
      cg.loop(cg.void);
      cg.block(cg.void);
      emitBreaks();
      emitBody();
      cg.br(1);
      cg.end();
      cg.end();
    };

    const emitLoopWhileLt = (
      index: number,
      emitBound: () => void,
      emitBody: () => void,
    ) =>
      emitLoopWithBreaks(() => {
        cg.local.get(index);
        emitBound();
        cg.i32.ge_u();
        cg.br_if(0);
      }, emitBody);

    const emitLoopWhileLtAndConstLt = (
      index: number,
      emitBound: () => void,
      constBound: number,
      emitBody: () => void,
    ) =>
      emitLoopWithBreaks(() => {
        cg.local.get(index);
        emitBound();
        cg.i32.ge_u();
        cg.br_if(0);
        cg.local.get(index);
        cg.i32.const(constBound);
        cg.i32.ge_u();
        cg.br_if(0);
      }, emitBody);

    const emitLoopWhileBlockFits = (
      index: number,
      blockSize: number,
      emitBound: () => void,
      constBound: number,
      emitBody: () => void,
    ) =>
      emitLoopWithBreaks(() => {
        emitLocalPlusConst(index, blockSize);
        emitBound();
        cg.i32.gt_u();
        cg.br_if(0);
        emitLocalPlusConst(index, blockSize);
        cg.i32.const(constBound);
        cg.i32.gt_u();
        cg.br_if(0);
      }, emitBody);

    const emitRowTileLoop = (
      rowOffset: number,
      rowTileBase: number,
      tileRows: number,
      tileSize: number,
      emitBody: () => void,
    ) =>
      emitLoopWithBreaks(() => {
        cg.local.get(rowOffset);
        cg.i32.const(tileRows);
        cg.i32.ge_u();
        cg.br_if(0);
        cg.local.get(rowTileBase);
        cg.local.get(rowOffset);
        cg.i32.const(tileSize);
        cg.i32.mul();
        cg.i32.add();
        cg.local.get(paramEnd);
        cg.i32.ge_u();
        cg.br_if(0);
      }, emitBody);

    const emitLoopWhileLocalLt = (
      index: number,
      bound: number,
      emitBody: () => void,
    ) => emitLoopWhileLt(index, () => cg.local.get(bound), emitBody);

    const emitLoopWhileConstLt = (
      index: number,
      bound: number,
      emitBody: () => void,
    ) => emitLoopWhileLt(index, () => cg.i32.const(bound), emitBody);

    const emitSimdExpWithAccumulator = (
      ctx: Record<string, number>,
      pointerMap: ReadonlyMap<AluExp, ReductionPointerPlan>,
      pointerValueCache: Map<string, number>,
      acc: number,
    ): boolean => {
      if (useRelaxedMadd) {
        translateExpSimd(
          cg,
          funcs,
          tune.exp.src[0],
          ctx,
          expStrides,
          pointerMap,
          pointerValueCache,
        );
        translateExpSimd(
          cg,
          funcs,
          tune.exp.src[1],
          ctx,
          expStrides,
          pointerMap,
          pointerValueCache,
        );
        cg.local.get(acc);
        cg.f32x4.relaxed_madd();
        return true;
      }

      translateExpSimd(
        cg,
        funcs,
        tune.exp,
        ctx,
        expStrides,
        pointerMap,
        pointerValueCache,
      );
      cg.local.get(acc);
      return false;
    };

    const emitSimdReductionForGidxs = (
      gidxs: number[],
      pointerMaps: Map<AluExp, ReductionPointerPlan>[],
      uniquePointers: ReductionPointerPlan[],
      ridxStart?: number,
      ridxEnd?: number,
    ) => {
      if (!re) throw new Error("internal: missing reduction");
      const reIsInt =
        kernel.exp.dtype === DType.Int32 || kernel.exp.dtype === DType.Uint32;

      const vecAccs = gidxs.map(() =>
        cg.local.declare(reIsInt ? cg.i32x4 : cg.f32x4),
      );

      const initializeIdentityAccumulators = () => {
        for (const acc of vecAccs) {
          if (reIsInt) {
            cg.i32.const(re.identity);
            cg.i32x4.splat();
          } else {
            cg.f32.const(re.identity);
            cg.f32x4.splat();
          }
          cg.local.set(acc);
        }
      };

      const loadPartialAccumulators = () => {
        for (let i = 0; i < gidxs.length; i++) {
          emitOutputAddress(gidxs[i]);
          if (reIsInt) cg.i32x4.load(4);
          else cg.f32x4.load(4);
          cg.local.set(vecAccs[i]);
        }
      };

      if (ridxStart === undefined) {
        initializeIdentityAccumulators();
      } else {
        cg.local.get(ridxStart);
        cg.i32.eqz();
        cg.if(cg.void);
        initializeIdentityAccumulators();
        cg.else();
        loadPartialAccumulators();
        cg.end();
      }

      const ridx = cg.local.declare(cg.i32);
      const emitReductionStep = () => {
        const pointerValueCache = new Map<string, number>();
        for (let i = 0; i < gidxs.length; i++) {
          const valueAlreadyAccumulated = emitSimdExpWithAccumulator(
            { gidx: gidxs[i], ridx },
            pointerMaps[i],
            pointerValueCache,
            vecAccs[i],
          );
          emitSimdReductionOp(cg, re, reIsInt, valueAlreadyAccumulated);
          cg.local.set(vecAccs[i]);
        }
        incrementReductionPointers(cg, uniquePointers);
      };

      if (ridxStart === undefined) setLocalConst(ridx, 0);
      else copyLocal(ridx, ridxStart);

      const emitReductionLoopBody = () => {
        emitReductionStep();
        bumpLocal(ridx, 1);
      };
      if (ridxEnd === undefined)
        emitLoopWhileConstLt(ridx, re.size, emitReductionLoopBody);
      else emitLoopWhileLocalLt(ridx, ridxEnd, emitReductionLoopBody);

      const storeRawAccumulators = () => {
        for (let i = 0; i < gidxs.length; i++) {
          emitOutputAddress(gidxs[i]);
          cg.local.get(vecAccs[i]);
          cg.v128.store(4);
        }
      };

      if (hasIdentityEpilogue) {
        storeRawAccumulators();
        return;
      }

      const storeEpilogueAccumulators = () => {
        const laneGidx = cg.local.declare(cg.i32);
        const laneAcc = cg.local.declare(reIsInt ? cg.i32 : cg.f32);
        for (let i = 0; i < gidxs.length; i++) {
          for (let lane = 0; lane < simdLanes; lane++) {
            cg.local.get(kernel.nargs);
            cg.local.get(gidxs[i]);
            if (lane > 0) {
              cg.i32.const(lane);
              cg.i32.add();
            }

            cg.local.tee(laneGidx);
            cg.i32.const(byteWidth(kernel.dtype));
            cg.i32.mul();
            cg.i32.add();

            cg.local.get(vecAccs[i]);
            if (reIsInt) cg.i32x4.extract_lane(lane);
            else cg.f32x4.extract_lane(lane);
            cg.local.set(laneAcc);
            translateExp(cg, funcs, tune.epilogue!, {
              acc: laneAcc,
              gidx: laneGidx,
            });

            dty(cg, null, kernel.dtype).store(
              Math.log2(byteWidth(kernel.dtype)),
            );
          }
        }
      };

      if (ridxEnd !== undefined) {
        cg.local.get(ridxEnd);
        cg.i32.const(re.size);
        cg.i32.lt_u();
        cg.if(cg.void);
        storeRawAccumulators();
        cg.else();
        storeEpilogueAccumulators();
        cg.end();
      } else {
        storeEpilogueAccumulators();
      }
    };

    const initializePointerMaps = <TGroup>(
      groups: TGroup[],
      candidates: ReductionPointerCandidate[],
      keyFor: (
        candidate: ReductionPointerCandidate,
        group: TGroup,
        groupIndex: number,
      ) => string,
      ctxFor: (group: TGroup) => Record<string, number>,
      ridxOffset?: number,
    ): ReductionPointerMaps => {
      const pointerMaps = groups.map(
        () => new Map<AluExp, ReductionPointerPlan>(),
      );
      const sharedPointers = new Map<string, ReductionPointerPlan>();
      const uniquePointers: ReductionPointerPlan[] = [];

      for (let i = 0; i < groups.length; i++) {
        const group = groups[i];
        for (const candidate of candidates) {
          const key = keyFor(candidate, group, i);
          let plan = sharedPointers.get(key);
          if (!plan) {
            plan = initializeReductionPointer(
              cg,
              funcs,
              candidate,
              ctxFor(group),
              key,
              ridxOffset,
            );
            sharedPointers.set(key, plan);
            uniquePointers.push(plan);
          }
          pointerMaps[i].set(candidate.exp, plan);
        }
      }

      return { pointerMaps, uniquePointers };
    };

    const emitElementwiseSimdStep = () => {
      emitOutputAddress(gidx);
      translateExpSimd(cg, funcs, tune.exp, { gidx }, expStrides);
      cg.v128.store(4);
      bumpLocal(gidx, simdLanes);
    };

    const emitKSimdReductionForGroups = (
      groups: KReductionGroup[],
      pointerMaps: Map<AluExp, ReductionPointerPlan>[],
      uniquePointers: ReductionPointerPlan[],
    ) => {
      if (!re) throw new Error("internal: missing reduction");
      if (!kSimdTilePlan) throw new Error("internal: missing K SIMD plan");

      const vecAccs = groups.map(() => cg.local.declare(cg.f32x4));
      for (const acc of vecAccs) {
        cg.f32.const(re.identity);
        cg.f32x4.splat();
        cg.local.set(acc);
      }

      const ridx = cg.local.declare(cg.i32);
      setLocalConst(ridx, 0);
      emitLoopWhileConstLt(ridx, re.size, () => {
        for (let u = 0; u < kSimdTilePlan.kUnroll; u++) {
          const pointerValueCache = new Map<string, number>();
          for (let i = 0; i < groups.length; i++) {
            const valueAlreadyAccumulated = emitSimdExpWithAccumulator(
              { gidx: groups[i].gidx, ridx },
              pointerMaps[i],
              pointerValueCache,
              vecAccs[i],
            );
            if (!valueAlreadyAccumulated) {
              cg.f32x4.add();
            }
            cg.local.set(vecAccs[i]);
          }
          incrementReductionPointers(cg, uniquePointers, simdLanes);
        }
        bumpLocal(ridx, simdLanes * kSimdTilePlan.kUnroll);
      });

      for (let i = 0; i < groups.length; i++) {
        const acc = cg.local.declare(cg.f32);
        // Sum all the elements across SIMD accumulator lanes into the scalar
        // accumulator expected by the normal reduction epilogue.
        for (let lane = 0; lane < simdLanes; lane++) {
          cg.local.get(vecAccs[i]);
          cg.f32x4.extract_lane(lane);
          if (lane > 0) cg.f32.add();
        }
        cg.local.set(acc);
        emitOutputAddress(groups[i].gidx);
        translateExp(cg, funcs, tune.epilogue!, {
          acc,
          gidx: groups[i].gidx,
        });
        dty(cg, null, kernel.dtype).store(Math.log2(byteWidth(kernel.dtype)));
      }
    };

    const emitKSimdReductionLoop = (plan: ReductionKTilePlan) => {
      const colTile = cg.local.declare(cg.i32);
      const col = cg.local.declare(cg.i32);
      const rowTileBase = cg.local.declare(cg.i32);
      const rowOffset = cg.local.declare(cg.i32);
      const rowBase = cg.local.declare(cg.i32);

      const emitBlock = () => {
        const groups: KReductionGroup[] = [];
        for (let row = 0; row < plan.microRows; row++) {
          for (let colOffset = 0; colOffset < plan.microCols; colOffset++) {
            groups.push({
              gidx: declareTileGidx(
                rowBase,
                col,
                row * plan.tileSize,
                colOffset,
              ),
              row,
              col: colOffset,
            });
          }
        }

        if (!kSimdTilePlan) throw new Error("internal: missing K SIMD plan");
        const { pointerMaps, uniquePointers } = initializePointerMaps(
          groups,
          kSimdReductionPointerCandidates,
          (candidate, group, i) =>
            kReductionPointerShareKey(
              candidate,
              expStrides.get(candidate.exp) ?? { kind: "gather" },
              kSimdTilePlan.tileSize,
              group.row,
              group.col,
              i,
            ),
          (group) => ({ gidx: group.gidx }),
        );
        emitKSimdReductionForGroups(groups, pointerMaps, uniquePointers);
      };

      emitLoopWhileLocalLt(gidx, paramEnd, () => {
        copyLocal(rowTileBase, gidx);

        setLocalConst(colTile, 0);
        emitLoopWhileConstLt(colTile, plan.tileSize, () => {
          setLocalConst(rowOffset, 0);
          emitRowTileLoop(
            rowOffset,
            rowTileBase,
            plan.tileRows,
            plan.tileSize,
            () => {
              copyLocal(col, colTile);
              emitLoopWhileLtAndConstLt(
                col,
                () => emitLocalPlusConst(colTile, plan.tileCols),
                plan.tileSize,
                () => {
                  setRowBase(rowBase, rowTileBase, rowOffset, plan.tileSize);
                  emitBlock();
                  bumpLocal(col, plan.microCols);
                },
              );
              bumpLocal(rowOffset, plan.microRows);
            },
          );
          bumpLocal(colTile, plan.tileCols);
        });
        bumpLocal(gidx, plan.tileRows * plan.tileSize);
      });
    };

    const emitTiledSimdReductionLoop = (plan: ReductionTilePlan) => {
      const colTile = cg.local.declare(cg.i32);
      const col = cg.local.declare(cg.i32);
      const kTile = cg.local.declare(cg.i32);
      const tileEnd = cg.local.declare(cg.i32);
      const rowTileBase = cg.local.declare(cg.i32);
      const rowOffset = cg.local.declare(cg.i32);
      const rowBase = cg.local.declare(cg.i32);

      const emitRowBlock = (microVectors: number) => {
        const groups: ReductionGroup[] = [];
        for (let row = 0; row < plan.microRows; row++) {
          for (let vector = 0; vector < microVectors; vector++) {
            groups.push({
              gidx: declareTileGidx(
                rowBase,
                col,
                row * plan.tileSize,
                vector * simdLanes,
              ),
              row,
              vector,
            });
          }
        }

        const { pointerMaps, uniquePointers } = initializePointerMaps(
          groups,
          simdReductionPointerCandidates,
          (candidate, group, i) =>
            pointerShareKey(candidate, group.row, group.vector, i),
          (group) => ({ gidx: group.gidx }),
          kTile,
        );
        emitSimdReductionForGidxs(
          groups.map((group) => group.gidx),
          pointerMaps,
          uniquePointers,
          kTile,
          tileEnd,
        );
      };

      emitLoopWhileLocalLt(gidx, paramEnd, () => {
        copyLocal(rowTileBase, gidx);

        setLocalConst(colTile, 0);
        emitLoopWhileConstLt(colTile, plan.tileSize, () => {
          setLocalConst(kTile, 0);
          emitLoopWhileConstLt(kTile, re!.size, () => {
            emitLocalPlusConst(kTile, plan.tileK);
            cg.local.set(tileEnd);

            setLocalConst(rowOffset, 0);
            emitRowTileLoop(
              rowOffset,
              rowTileBase,
              plan.tileRows,
              plan.tileSize,
              () => {
                copyLocal(col, colTile);
                emitLoopWhileBlockFits(
                  col,
                  plan.microVectors * simdLanes,
                  () =>
                    emitLocalPlusConst(colTile, plan.tileVectors * simdLanes),
                  plan.tileSize,
                  () => {
                    setRowBase(rowBase, rowTileBase, rowOffset, plan.tileSize);
                    emitRowBlock(plan.microVectors);
                    bumpLocal(col, plan.microVectors * simdLanes);
                  },
                );
                emitLoopWhileLtAndConstLt(
                  col,
                  () =>
                    emitLocalPlusConst(colTile, plan.tileVectors * simdLanes),
                  plan.tileSize,
                  () => {
                    setRowBase(rowBase, rowTileBase, rowOffset, plan.tileSize);
                    emitRowBlock(1);
                    bumpLocal(col, simdLanes);
                  },
                );
                bumpLocal(rowOffset, plan.microRows);
              },
            );
            bumpLocal(kTile, plan.tileK);
          });
          bumpLocal(colTile, plan.tileVectors * simdLanes);
        });
        bumpLocal(gidx, plan.tileRows * plan.tileSize);
      });
    };

    const emitGuardedFastPath = (alignment: number, emit: () => void): void => {
      emitAlignmentGuard(cg, paramBegin, paramEnd, alignment);
      emit();
      cg.return();
      cg.end();
    };

    if (kSimdTilePlan) {
      emitGuardedFastPath(
        kSimdTilePlan.microRows * kSimdTilePlan.tileSize,
        () => emitKSimdReductionLoop(kSimdTilePlan),
      );
    }

    if (useSimd) {
      if (simdTilePlan) {
        emitGuardedFastPath(
          simdTilePlan.microRows * simdTilePlan.tileSize,
          () => emitTiledSimdReductionLoop(simdTilePlan),
        );
      }

      if (!re) {
        emitGuardedFastPath(simdLanes, () => {
          emitLoopWhileLocalLt(gidx, paramEnd, emitElementwiseSimdStep);
        });
      }
    }

    emitLoopWhileLocalLt(gidx, paramEnd, () => {
      emitOutputAddress(gidx);

      if (re) {
        const acc = cg.local.declare(dty(cg, null, kernel.exp.dtype));
        dty(cg, null, kernel.exp.dtype).const(re.identity);
        cg.local.set(acc);

        const ridx = cg.local.declare(cg.i32);
        const emitReductionStep = () => {
          translateExp(cg, funcs, tune.exp, { gidx, ridx });
          emitScalarReductionOp(cg, re, acc);
          cg.local.set(acc);
        };

        setLocalConst(ridx, 0);
        emitLoopWhileConstLt(ridx, re.size, () => {
          emitReductionStep();
          bumpLocal(ridx, 1);
        });

        translateExp(cg, funcs, tune.epilogue!, { acc, gidx });
      } else {
        translateExp(cg, funcs, tune.exp, { gidx });
      }

      dty(cg, null, kernel.dtype).store(Math.log2(byteWidth(kernel.dtype)));
      bumpLocal(gidx, 1);
    });
  });
  cg.export(kernelFunc, "kernel");

  const tiledPlan = simdTilePlan ?? kSimdTilePlan;
  return {
    bytes: cg.finish(),
    workSize: kernel.size,
    chunkAlignment: tiledPlan ? tiledPlan.tileRows * tiledPlan.tileSize : 16,
    minWorkPerWorker:
      tiledPlan && kernel.size / tiledPlan.tileSize >= 1024
        ? tiledPlan.tileSize * 32
        : 256,
  };
}
