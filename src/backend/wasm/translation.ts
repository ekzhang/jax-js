import { simdLanes, type StrideResult } from "./tilePlan";
import { CodeGenerator } from "./wasmblr";
import {
  AluExp,
  AluGroup,
  AluOp,
  byteWidth,
  DType,
  isFloatDtype,
} from "../../alu";
import { UnsupportedOpError } from "../../backend";

export function translateExp(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  exp: AluExp,
  ctx: Record<string, number>,
  pointerMap: ReadonlyMap<AluExp, PointerPlan> = new Map(),
) {
  const references = new Map<AluExp, number>();
  const seen = new Set<AluExp>();
  const countReferences = (exp: AluExp) => {
    references.set(exp, (references.get(exp) ?? 0) + 1);
    if (!seen.has(exp)) {
      seen.add(exp);
      for (const src of exp.src) countReferences(src);
    }
  };

  const expContext = new Map<AluExp, number>();
  const gen = (exp: AluExp) => {
    if (expContext.has(exp)) return cg.local.get(expContext.get(exp)!);
    const { op, src, dtype, arg } = exp;

    // Some of these cases early `return` to force-inline them (no local.set).
    if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
      (gen(src[0]), gen(src[1]));
      if (op === AluOp.Add) {
        if (dtype === DType.Bool) cg.i32.or();
        else dty(cg, op, dtype).add();
      } else if (op === AluOp.Sub) {
        dty(cg, op, dtype).sub();
      } else if (op === AluOp.Mul) {
        if (dtype === DType.Bool) cg.i32.and();
        else dty(cg, op, dtype).mul();
      } else if (op === AluOp.Idiv) {
        if (isFloatDtype(dtype)) {
          dtyF(cg, op, dtype).div();
          dtyF(cg, op, dtype).trunc();
        } else if (dtype === DType.Uint32) cg.i32.div_u();
        else if (dtype === DType.Int32) cg.i32.div_s();
        else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.Mod) {
        if (isFloatDtype(dtype)) {
          // Emulate a % b = a - trunc(a/b)*b
          const dt = dtyF(cg, op, dtype);
          const a = cg.local.declare(dt);
          const b = cg.local.declare(dt);
          cg.local.set(b);
          cg.local.tee(a); // stack: a
          cg.local.get(a);
          cg.local.get(b);
          dt.div();
          dt.trunc(); // stack: a, trunc(a/b)
          cg.local.get(b);
          dt.mul(); // stack: a, trunc(a/b)*b
          dt.sub();
        } else if (dtype === DType.Uint32) cg.i32.rem_u();
        else if (dtype === DType.Int32) cg.i32.rem_s();
        else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.Min || op === AluOp.Max) {
        if (isFloatDtype(dtype)) {
          if (op === AluOp.Min) dtyF(cg, op, dtype).min();
          else dtyF(cg, op, dtype).max();
        } else if (
          dtype === DType.Int32 ||
          dtype === DType.Uint32 ||
          dtype === DType.Bool
        ) {
          // Wasm has no i32.min, so emulate with select.
          const a = cg.local.declare(cg.i32);
          const b = cg.local.declare(cg.i32);
          cg.local.set(b);
          cg.local.tee(a);
          cg.local.get(b);
          cg.local.get(a);
          cg.local.get(b);
          if (dtype === DType.Int32) {
            if (op === AluOp.Min) cg.i32.lt_s();
            else cg.i32.gt_s();
          } else {
            if (op === AluOp.Min) cg.i32.lt_u();
            else cg.i32.gt_u();
          }
          cg.select();
        } else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.BitCombine) {
        if (arg === "and") cg.i32.and();
        else if (arg === "or") cg.i32.or();
        else cg.i32.xor();
      } else if (op === AluOp.BitShift) {
        if (arg === "shl") cg.i32.shl();
        else cg.i32.shr_u();
      } else if (op === AluOp.Cmplt) {
        const srcDtype = src[0].dtype;
        if (isFloatDtype(srcDtype)) dtyF(cg, op, srcDtype).lt();
        else if (srcDtype === DType.Int32) cg.i32.lt_s();
        else if (srcDtype === DType.Uint32) cg.i32.lt_u();
        else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.Cmpne) dty(cg, op, src[0].dtype).ne();
      else throw new UnsupportedOpError(op, dtype, "wasm");
    } else if (AluGroup.Unary.has(op)) {
      // TODO: Our intrinsics are only implemented in f32 precision currently,
      // so we cast to f32 first for other floating-point inputs.
      const callFuncF32 = (func: number): void => {
        if (dtype !== DType.Float32) {
          if (dtype === DType.Float64) cg.f32.demote_f64();
          else throw new UnsupportedOpError(op, dtype, "wasm");
        }
        cg.call(func);
        if (dtype === DType.Float64) cg.f64.promote_f32();
      };
      if (op === AluOp.Sin) (gen(src[0]), callFuncF32(funcs.sin));
      else if (op === AluOp.Cos) (gen(src[0]), callFuncF32(funcs.cos));
      else if (op === AluOp.Asin) (gen(src[0]), callFuncF32(funcs.asin));
      else if (op === AluOp.Atan) (gen(src[0]), callFuncF32(funcs.atan));
      else if (op === AluOp.Exp) (gen(src[0]), callFuncF32(funcs.exp));
      else if (op === AluOp.Log) (gen(src[0]), callFuncF32(funcs.log));
      else if (op === AluOp.Erf) (gen(src[0]), callFuncF32(funcs.erf));
      else if (op === AluOp.Erfc) (gen(src[0]), callFuncF32(funcs.erfc));
      else if (op === AluOp.Sqrt) (gen(src[0]), dtyF(cg, op, dtype).sqrt());
      else if (op === AluOp.Reciprocal) {
        const dt = dtyF(cg, op, dtype);
        (dt.const(1), gen(src[0]), dt.div());
      } else if (op === AluOp.Floor) (gen(src[0]), dtyF(cg, op, dtype).floor());
      else if (op === AluOp.Ceil) (gen(src[0]), dtyF(cg, op, dtype).ceil());
      else if (op === AluOp.Cast) {
        gen(src[0]);
        const dtype0 = src[0].dtype;
        const i32repr =
          dtype0 === DType.Int32 ||
          dtype0 === DType.Uint32 ||
          dtype0 === DType.Bool;
        if (dtype === DType.Int32) {
          if (dtype0 === DType.Float32) cg.i32.trunc_sat_f32_s();
          else if (dtype0 === DType.Float64) cg.i32.trunc_sat_f64_s();
          else if (i32repr) void 0;
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else if (dtype === DType.Uint32) {
          if (dtype0 === DType.Float32) cg.i32.trunc_sat_f32_u();
          else if (dtype0 === DType.Float64) cg.i32.trunc_sat_f64_u();
          else if (i32repr) void 0;
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else if (dtype === DType.Float32) {
          if (dtype0 === DType.Float32) void 0;
          else if (dtype0 === DType.Float64) cg.f32.demote_f64();
          else if (dtype0 === DType.Int32 || dtype0 === DType.Bool)
            cg.f32.convert_i32_s();
          else if (dtype0 === DType.Uint32) cg.f32.convert_i32_u();
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else if (dtype === DType.Float64) {
          if (dtype0 === DType.Float32) cg.f64.promote_f32();
          else if (dtype0 === DType.Float64) void 0;
          else if (dtype0 === DType.Int32 || dtype0 === DType.Bool)
            cg.f64.convert_i32_s();
          else if (dtype0 === DType.Uint32) cg.f64.convert_i32_u();
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else if (dtype === DType.Bool) {
          if (dtype0 === DType.Bool) void 0;
          else if (i32repr) (cg.i32.const(0), cg.i32.ne());
          else if (dtype0 === DType.Float32) (cg.f32.const(0), cg.f32.ne());
          else if (dtype0 === DType.Float64) (cg.f64.const(0), cg.f64.ne());
          else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
        } else throw new UnsupportedOpError(op, dtype, "wasm");
      } else if (op === AluOp.Bitcast) {
        gen(src[0]);
        const dtype0 = src[0].dtype;
        if (dtype !== dtype0) {
          const i32repr = dtype0 === DType.Int32 || dtype0 === DType.Uint32;
          if (dtype === DType.Int32 || dtype === DType.Uint32) {
            if (dtype0 === DType.Float32) cg.i32.reinterpret_f32();
            else if (i32repr) void 0;
            else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
          } else if (dtype === DType.Float32) {
            if (i32repr) cg.f32.reinterpret_i32();
            else if (dtype0 === DType.Float32) void 0;
            else throw new UnsupportedOpError(op, dtype, "wasm", dtype0);
          } else throw new UnsupportedOpError(op, dtype, "wasm");
        }
      } else throw new UnsupportedOpError(op, dtype, "wasm");
    } else if (op === AluOp.Where) {
      gen(src[1]); // t
      gen(src[2]); // f
      gen(src[0]); // cond
      cg.select();
    } else if (op === AluOp.Threefry2x32) {
      for (let i = 0; i < 4; i++) gen(src[i]);
      cg.call(funcs.threefry2x32);
      if (arg === "xor") cg.i32.xor();
      else if (arg === 0) cg.drop();
      else if (arg === 1) {
        const local = cg.local.declare(cg.i32);
        cg.local.set(local);
        cg.drop();
        cg.local.get(local);
      } else throw new UnsupportedOpError(op, dtype, "wasm", arg);
    } else if (op === AluOp.Const) {
      return dty(cg, op, dtype).const(arg as number);
    } else if (op === AluOp.Special) {
      return cg.local.get(ctx[arg[0] as string]);
    } else if (op === AluOp.Variable) {
      return cg.local.get(ctx[arg as string]);
    } else if (op === AluOp.GlobalIndex) {
      const [gid, len] = arg as [number, number];
      const pointer = pointerMap.get(exp);
      if (pointer) {
        cg.local.get(pointer.ptr);
      } else {
        gen(src[0]);

        // If value is out-of-bounds, just set it to be zero.
        // This extra bounds-check is needed in Wasm because otherwise we will get
        // out-of-bounds memory access traps. WebGPU just silently returns 0.
        const local = cg.local.declare(cg.i32);
        cg.local.tee(local);
        cg.i32.const(0);
        (cg.local.get(local), cg.i32.const(len), cg.i32.lt_u());
        cg.select();

        cg.i32.const(byteWidth(dtype));
        cg.i32.mul();
        cg.local.get(gid); // base offset of array
        cg.i32.add();
      }
      dty(cg, op, dtype).load(Math.log2(byteWidth(dtype)));
    } else throw new UnsupportedOpError(op, dtype, "wasm");

    if ((references.get(exp) ?? 0) > 1) {
      const local = cg.local.declare(dty(cg, op, dtype));
      cg.local.tee(local);
      expContext.set(exp, local);
    }
  };

  countReferences(exp);
  gen(exp);
}

export interface PointerPlan {
  ptr: number;
}

export interface SimdPointerPlan extends PointerPlan {
  stride: StrideResult;
  valueKey?: string;
}

/**
 * SIMD version of translateExp. Emits one v128 value for `exp`, interpreting
 * the current `gidx` local as the first lane and the following lanes as
 * `gidx + 1`, `gidx + 2`, etc.
 *
 * GlobalIndex loads are the only places where per-lane address behavior
 * matters:
 * - `strideMap` classifies each GlobalIndex as contiguous, broadcast, or
 *   gather. Contiguous loads become one v128.load, broadcast loads become a
 *   scalar load plus splat, and gather loads fall back to four scalar loads.
 * - `pointerMap` optionally supplies precomputed reduction pointers for
 *   GlobalIndex nodes whose address can be advanced by the surrounding
 *   reduction loop. This avoids re-emitting scalar index math in the hot path.
 * - `pointerValueCache` lets pointer plans with the same `valueKey` share a
 *   single loaded vector within one emitted reduction step.
 */
export function translateExpSimd(
  cg: CodeGenerator,
  funcs: Record<string, number>,
  exp: AluExp,
  ctx: Record<string, number>,
  strideMap: Map<AluExp, StrideResult>,
  pointerMap: ReadonlyMap<AluExp, SimdPointerPlan> = new Map(),
  pointerValueCache: Map<string, number> = new Map(),
): void {
  const references = new Map<AluExp, number>();
  const seen = new Set<AluExp>();
  const countReferences = (exp: AluExp) => {
    references.set(exp, (references.get(exp) ?? 0) + 1);
    if (!seen.has(exp)) {
      seen.add(exp);
      for (const src of exp.src) countReferences(src);
    }
  };

  const expContext = new Map<AluExp, number>();
  const gen = (exp: AluExp) => {
    if (expContext.has(exp)) return cg.local.get(expContext.get(exp)!);
    const { op, src, arg, dtype } = exp;
    const isInt =
      dtype === DType.Int32 || dtype === DType.Uint32 || dtype === DType.Bool;
    const isSigned = dtype === DType.Int32;

    if (op === AluOp.Add) {
      (gen(src[0]), gen(src[1]));
      if (dtype === DType.Bool) cg.v128.or();
      else if (isInt) cg.i32x4.add();
      else cg.f32x4.add();
    } else if (op === AluOp.Sub) {
      (gen(src[0]), gen(src[1]));
      if (isInt) cg.i32x4.sub();
      else cg.f32x4.sub();
    } else if (op === AluOp.Mul) {
      (gen(src[0]), gen(src[1]));
      if (dtype === DType.Bool) cg.v128.and();
      else if (isInt) cg.i32x4.mul();
      else cg.f32x4.mul();
    } else if (op === AluOp.Min) {
      (gen(src[0]), gen(src[1]));
      if (isInt) {
        if (isSigned) cg.i32x4.min_s();
        else cg.i32x4.min_u();
      } else cg.f32x4.min();
    } else if (op === AluOp.Max) {
      (gen(src[0]), gen(src[1]));
      if (isInt) {
        if (isSigned) cg.i32x4.max_s();
        else cg.i32x4.max_u();
      } else cg.f32x4.max();
    } else if (op === AluOp.Sqrt) {
      gen(src[0]);
      cg.f32x4.sqrt();
    } else if (op === AluOp.Floor) {
      gen(src[0]);
      cg.f32x4.floor();
    } else if (op === AluOp.Ceil) {
      gen(src[0]);
      cg.f32x4.ceil();
    } else if (op === AluOp.Const) {
      if (isInt) {
        cg.i32.const(arg as number);
        cg.i32x4.splat();
      } else {
        cg.f32.const(arg as number);
        cg.f32x4.splat();
      }
    } else if (op === AluOp.Cast) {
      gen(src[0]);
      const dtype0 = src[0].dtype;
      const src0IsInt =
        dtype0 === DType.Int32 ||
        dtype0 === DType.Uint32 ||
        dtype0 === DType.Bool;
      if (isInt && !src0IsInt) {
        // f32 to i32/u32
        if (isSigned) cg.i32x4.trunc_sat_f32x4_s();
        else cg.i32x4.trunc_sat_f32x4_u();
      } else if (!isInt && src0IsInt) {
        // i32/bool to f32 (bool uses signed to match scalar path)
        if (dtype0 === DType.Int32 || dtype0 === DType.Bool)
          cg.f32x4.convert_i32x4_s();
        else cg.f32x4.convert_i32x4_u();
      }
      // between i32 and u32: no-op (same bit representation)
    } else if (op === AluOp.Cmplt) {
      (gen(src[0]), gen(src[1]));
      const srcDtype = src[0].dtype;
      if (srcDtype === DType.Float32) cg.f32x4.lt();
      else if (srcDtype === DType.Int32) cg.i32x4.lt_s();
      else if (srcDtype === DType.Uint32) cg.i32x4.lt_u();
      else throw new UnsupportedOpError(op, dtype, "wasm");
      // SIMD comparisons produce 0xFFFFFFFF per lane; normalize to 0/1 to match scalar path.
      cg.i32.const(1);
      cg.i32x4.splat();
      cg.v128.and();
    } else if (op === AluOp.Cmpne) {
      (gen(src[0]), gen(src[1]));
      const srcDtype = src[0].dtype;
      if (srcDtype === DType.Float32) cg.f32x4.ne();
      else cg.i32x4.ne();
      // SIMD comparisons produce 0xFFFFFFFF per lane; normalize to 0/1 to match scalar path.
      cg.i32.const(1);
      cg.i32x4.splat();
      cg.v128.and();
    } else if (op === AluOp.Where) {
      gen(src[1]); // true value
      gen(src[2]); // false value
      // Scalar where uses select (0 = false, nonzero = true), but SIMD only
      // has v128.bitselect which needs a full bitmask per lane (0x00000000
      // or 0xFFFFFFFF). Expand 0/1 conditions with ne(0) to get the bitmask.
      gen(src[0]);
      cg.i32.const(0);
      cg.i32x4.splat();
      cg.i32x4.ne();
      cg.v128.bitselect();
    } else if (op === AluOp.Variable || op === AluOp.Special) {
      // Scalar context variables only appear inside GlobalIndex index
      // subtrees, which are handled below via scalar translateExp. Reaching
      // this path means the SIMD eligibility check missed a free variable.
      throw new Error(`translateExpSimd: unexpected ${op}(${arg})`);
    } else if (op === AluOp.GlobalIndex) {
      const [gid, len] = arg as [number, number];
      const indexSubtree = src[0];
      const pointer = pointerMap.get(exp);
      const stride = pointer?.stride ??
        strideMap.get(exp) ?? { kind: "gather" };

      if (pointer) {
        const cached = pointer.valueKey
          ? pointerValueCache.get(pointer.valueKey)
          : undefined;
        if (cached !== undefined) {
          cg.local.get(cached);
        } else {
          cg.local.get(pointer.ptr);
          if (stride.kind === "contiguous") {
            if (isInt) cg.i32x4.load(4);
            else cg.f32x4.load(4);
          } else if (stride.kind === "broadcast") {
            if (isInt) {
              cg.i32.load(2);
              cg.i32x4.splat();
            } else {
              cg.f32.load(2);
              cg.f32x4.splat();
            }
          } else {
            throw new Error("reduction pointer plan cannot use gather loads");
          }
          if (pointer.valueKey) {
            const local = cg.local.declare(isInt ? cg.i32x4 : cg.f32x4);
            cg.local.tee(local);
            pointerValueCache.set(pointer.valueKey, local);
          }
        }
      } else if (stride.kind === "contiguous") {
        // Wide load: evaluate index subtree once (scalar) to get starting
        // address, then v128.load 4 consecutive elements.
        // If index is out-of-bounds, clamp to len-4 to prevent WASM traps.
        translateExp(cg, funcs, indexSubtree, ctx);
        {
          const maxIdx = Math.max(len - simdLanes, 0);
          const wideIdx = cg.local.declare(cg.i32);
          cg.local.set(wideIdx);
          cg.local.get(wideIdx); // val_true = index
          cg.i32.const(maxIdx); // val_false = maxIdx
          cg.local.get(wideIdx);
          cg.i32.const(maxIdx);
          cg.i32.lt_u(); // condition: index < maxIdx
          cg.select();
        }

        cg.i32.const(byteWidth(dtype));
        cg.i32.mul();
        cg.local.get(gid); // base pointer
        cg.i32.add();
        if (isInt) cg.i32x4.load(4);
        else cg.f32x4.load(4);
      } else if (stride.kind === "broadcast") {
        // Broadcast: index is constant across 4 SIMD lanes.
        // Evaluate once scalarly, load one element, splat to v128.
        translateExp(cg, funcs, indexSubtree, ctx);

        // OOB bounds check (same as scalar path).
        const local = cg.local.declare(cg.i32);
        cg.local.tee(local);
        cg.i32.const(0);
        (cg.local.get(local), cg.i32.const(len), cg.i32.lt_u());
        cg.select();

        cg.i32.const(byteWidth(dtype));
        cg.i32.mul();
        cg.local.get(gid); // base pointer
        cg.i32.add();
        if (isInt) {
          cg.i32.load(2);
          cg.i32x4.splat();
        } else {
          cg.f32.load(2);
          cg.f32x4.splat();
        }
      } else {
        // Gather: evaluate index subtree 4 times with gidx+0,+1,+2,+3,
        // do 4 scalar loads, pack into v128.
        const steppingLocal = ctx["gidx"];
        const origValue = cg.local.declare(cg.i32);
        cg.local.get(steppingLocal);
        cg.local.set(origValue);

        // Start with zeros, replace each lane
        if (isInt) {
          cg.i32.const(0);
          cg.i32x4.splat();
        } else {
          cg.f32.const(0);
          cg.f32x4.splat();
        }
        const vec = cg.local.declare(isInt ? cg.i32x4 : cg.f32x4);
        cg.local.set(vec);

        const idx = cg.local.declare(cg.i32);
        const scalarVal = cg.local.declare(isInt ? cg.i32 : cg.f32);

        for (let lane = 0; lane < simdLanes; lane++) {
          // Set stepping var to original + lane
          cg.local.get(origValue);
          if (lane > 0) {
            cg.i32.const(lane);
            cg.i32.add();
          }
          cg.local.set(steppingLocal);

          // Evaluate index subtree to get flat element index, with OOB clamping.
          // Same bounds check as scalar translateExp's GlobalIndex handler:
          // if index >= len, use 0 instead (prevents WASM memory traps).
          translateExp(cg, funcs, indexSubtree, ctx);
          cg.local.tee(idx);
          cg.i32.const(0);
          (cg.local.get(idx), cg.i32.const(len), cg.i32.lt_u());
          cg.select();

          cg.i32.const(byteWidth(dtype));
          cg.i32.mul();
          cg.local.get(gid); // base pointer
          cg.i32.add();
          if (isInt) cg.i32.load(2);
          else cg.f32.load(2);

          // Pack into v128: replace_lane expects [v128, scalar] on stack
          cg.local.set(scalarVal);
          cg.local.get(vec);
          cg.local.get(scalarVal);
          if (isInt) cg.i32x4.replace_lane(lane);
          else cg.f32x4.replace_lane(lane);
          cg.local.set(vec);
        }

        // Restore original stepping var value
        cg.local.get(origValue);
        cg.local.set(steppingLocal);

        // Push the gathered v128 onto the stack
        cg.local.get(vec);
      }
    } else {
      throw new Error(`translateExpSimd: unsupported op ${op}`);
    }

    // CSE: if this node is used more than once, store in a local
    if ((references.get(exp) ?? 0) > 1) {
      const local = cg.local.declare(isInt ? cg.i32x4 : cg.f32x4);
      cg.local.tee(local);
      expContext.set(exp, local);
    }
  };

  countReferences(exp);
  gen(exp);
}

export function dty(cg: CodeGenerator, op: AluOp | null, dtype: DType) {
  switch (dtype) {
    case DType.Float32:
      return cg.f32;
    case DType.Float64:
      return cg.f64;
    case DType.Int32:
    case DType.Uint32:
    case DType.Bool:
      return cg.i32;
    default:
      throw new UnsupportedOpError(op, dtype, "wasm");
  }
}

export function dtyF(
  cg: CodeGenerator,
  op: AluOp | null,
  dtype: DType,
): CodeGenerator["f32" | "f64"] {
  switch (dtype) {
    case DType.Float32:
      return cg.f32;
    case DType.Float64:
      return cg.f64;
    default:
      throw new UnsupportedOpError(op, dtype, "wasm");
  }
}

/** Subset of operations supported in SIMD compilation mode. */
export const simdSupportedOps: Map<DType, Set<AluOp>> = new Map();
{
  simdSupportedOps.set(
    DType.Float32,
    new Set([
      AluOp.Add,
      AluOp.Sub,
      AluOp.Mul,
      AluOp.Floor,
      AluOp.Ceil,
      AluOp.Min,
      AluOp.Max,
      AluOp.Sqrt,
      AluOp.Cast,
      AluOp.Where,
      AluOp.Const,
      AluOp.GlobalIndex,
    ]),
  );
  simdSupportedOps.set(
    DType.Int32,
    new Set([
      AluOp.Add,
      AluOp.Sub,
      AluOp.Mul,
      AluOp.Min,
      AluOp.Max,
      AluOp.Cast,
      AluOp.Where,
      AluOp.Const,
      AluOp.GlobalIndex,
    ]),
  );
  simdSupportedOps.set(DType.Uint32, simdSupportedOps.get(DType.Int32)!);
  simdSupportedOps.set(
    DType.Bool,
    new Set([
      AluOp.Add,
      AluOp.Mul,
      AluOp.Min,
      AluOp.Max,
      AluOp.Cmplt,
      AluOp.Cmpne,
      AluOp.Const,
      AluOp.GlobalIndex,
    ]),
  );
}
