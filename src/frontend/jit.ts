// Handle Jit operations by translating Jaxprs into dispatched Kernels.

import {
  AluExp,
  AluOp,
  AluVar,
  byteWidth,
  DType,
  Kernel,
  Reduction,
} from "../alu";
import { Backend, Executable, Slot } from "../backend";
import type { NativeScanGeneralParams } from "../backend/wasm";
import { PPrint } from "../pprint";
import { Routine, Routines } from "../routine";
import { Pair, ShapeTracker, unravelAlu } from "../shape";
import {
  DEBUG,
  deepEqual,
  FpHash,
  generalBroadcast,
  prod,
  range,
  rep,
  reportScanPath,
  type ScanPath,
} from "../utils";
import { aluCompare, PendingExecute } from "./array";
import { pool, poolTranspose, prepareConv } from "./convolution";
import {
  Primitive,
  PrimitiveParams,
  promoteAvals,
  routinePrimitives,
  ShapedArray,
} from "./core";
import { Jaxpr, Lit, Var } from "./jaxpr";

export type JitId = number;

export type JitStep =
  | {
      type: "execute";
      source: Kernel | Routine;
      inputs: JitId[]; // mapped to backend Slot
      outputs: JitId[]; // mapped to backend Slot
    }
  | {
      type: "malloc";
      size: number;
      output: JitId;
    }
  | {
      type: "incref";
      input: JitId;
    }
  | {
      type: "free";
      input: JitId;
    }
  | {
      type: "scan";
      bodyProgram: JitProgram;
      bodyJaxpr: Jaxpr;
      length: number;
      numCarry: number;
      numConsts: number;
      numX: number;
      numY: number;
      reverse: boolean;
      consts: JitId[];
      initCarry: JitId[];
      xs: JitId[];
      xsAvals: ShapedArray[]; // xs avals from the scan's input (for shape tracking)
      outputs: JitId[]; // [carry_out..., stacked_ys...]
    }
  | {
      type: "native-scan";
      executable: Executable;
      length: number;
      numCarry: number;
      numConsts: number;
      numY: number;
      reverse: boolean;
      consts: JitId[];
      initCarry: JitId[];
      xs: JitId[];
      outputs: JitId[]; // [carry_out..., stacked_ys...]
      /** For general native scan: parameters including internal sizes, aux buffer, etc. */
      generalParams?: NativeScanGeneralParams;
    }
  | {
      type: "batched-scan";
      /** Batched scan params (stored for dispatch). */
      batchedParams: any; // BatchedScanParams from webgpu.ts
      length: number;
      numCarry: number;
      numConsts: number;
      numX: number;
      numY: number;
      reverse: boolean;
      consts: JitId[];
      initCarry: JitId[];
      xs: JitId[];
      outputs: JitId[]; // [carry_out..., stacked_ys...]
    };

/** Callback type for running scan steps during JitProgram execution. */
export type ScanRunner = (
  bodyProgram: JitProgram,
  backend: Backend,
  jaxpr: Jaxpr,
  length: number,
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  reverse: boolean,
  constSlots: Slot[],
  initCarrySlots: Slot[],
  xsSlots: Slot[],
  xsAvals: ShapedArray[],
  outputSlots: Slot[],
) => { outputs: Slot[]; pending: PendingExecute[] };

/**
 * Check if a chosen scan path satisfies the requirePath constraint.
 * Returns an error message if the path is not allowed, or null if OK.
 */
function checkRequiredPath(
  chosenPath: ScanPath,
  requirePath: string | string[] | undefined,
): string | null {
  if (!requirePath) return null;

  const allowedPaths = Array.isArray(requirePath) ? requirePath : [requirePath];

  if (!allowedPaths.includes(chosenPath)) {
    return (
      `Scan requirePath constraint not satisfied: ` +
      `got "${chosenPath}" but required one of [${allowedPaths.map((p) => `"${p}"`).join(", ")}]`
    );
  }
  return null;
}

/** Result of compiling a Jaxpr. Can be evaluated on a series of inputs. */
export class JitProgram {
  constructor(
    readonly backend: Backend,
    readonly steps: JitStep[],
    readonly inputs: JitId[],
    readonly outputs: JitId[],
  ) {}

  pprint(): PPrint {
    const steps: PPrint[] = this.steps.map((step) => {
      switch (step.type) {
        case "execute": {
          const inputsNice = step.inputs
            .map((id, i) => `${i}: %${id}`)
            .join(", ");
          const outputsNice = step.outputs.map((id) => `%${id}`).join(", ");
          const executeText = `execute (${inputsNice}) -> ${outputsNice}`;
          if (step.source instanceof Kernel) {
            return PPrint.pp(`${executeText}, kernel`).concat(
              step.source.pprint().indent(2),
            );
          } else if (step.source instanceof Routine) {
            return PPrint.pp(`${executeText}, routine ${step.source.name}`);
          } else {
            step.source satisfies never; // static check
            return PPrint.pp(executeText);
          }
        }
        case "malloc":
          return PPrint.pp(`%${step.output} = malloc <${step.size} bytes>`);
        case "incref":
          return PPrint.pp(`incref ${step.input}`);
        case "free":
          return PPrint.pp(`free ${step.input}`);
        case "scan":
          return PPrint.pp(
            `scan length=${step.length} numCarry=${step.numCarry} numConsts=${step.numConsts}`,
          )
            .concat(
              PPrint.pp(
                `  consts=[${step.consts.join(", ")}] initCarry=[${step.initCarry.join(", ")}] xs=[${step.xs.join(", ")}]`,
              ),
            )
            .concat(PPrint.pp(`  outputs=[${step.outputs.join(", ")}]`))
            .concat(
              PPrint.pp("  body=").concat(
                PPrint.pp(step.bodyJaxpr.toString()).indent(4),
              ),
            );
        case "native-scan":
          return PPrint.pp(
            `native-scan length=${step.length} numCarry=${step.numCarry}`,
          )
            .concat(
              PPrint.pp(
                `  initCarry=[${step.initCarry.join(", ")}] xs=[${step.xs.join(", ")}]`,
              ),
            )
            .concat(PPrint.pp(`  outputs=[${step.outputs.join(", ")}]`));
        case "batched-scan":
          return PPrint.pp(
            `batched-scan length=${step.length} numCarry=${step.numCarry} numConsts=${step.numConsts}`,
          )
            .concat(
              PPrint.pp(
                `  initCarry=[${step.initCarry.join(", ")}] xs=[${step.xs.join(", ")}]`,
              ),
            )
            .concat(PPrint.pp(`  outputs=[${step.outputs.join(", ")}]`));
      }
    });
    const display = PPrint.prototype.concat(
      PPrint.pp(`device = ${this.backend.type}`),
      PPrint.pp("inputs = [" + this.inputs.join(", ") + "]"),
      PPrint.pp("outputs = [" + this.outputs.join(", ") + "]"),
      PPrint.pp("steps ="),
      PPrint.prototype.concat(...steps).indent(2),
    );
    return PPrint.pp("{ ").stack(display.stack(PPrint.pp(" }")));
  }

  toString(): string {
    return this.pprint().toString();
  }

  /** Execute the JitProgram with the given inputs.
   * @param scanRunner - Optional callback to run scan steps. Required if program contains scan steps.
   */
  execute(
    inputs: Slot[],
    scanRunner?: ScanRunner,
  ): { outputs: Slot[]; pending: PendingExecute[] } {
    const scope = new Map<JitId, Slot>();
    if (inputs.length !== this.inputs.length) {
      throw new TypeError(
        `Expected ${this.inputs.length} inputs, got ${inputs.length}`,
      );
    }
    for (const [i, id] of this.inputs.entries()) {
      scope.set(id, inputs[i]);
    }
    const pending: PendingExecute[] = [];
    for (const step of this.steps) {
      switch (step.type) {
        case "execute": {
          const inputs = step.inputs.map((id) => scope.get(id)!);
          const outputs = step.outputs.map((id) => scope.get(id)!);
          if (
            inputs.some((s) => s === undefined) ||
            outputs.some((s) => s === undefined)
          ) {
            throw new Error(`internal: JitProgram scope undefined`);
          }
          pending.push(
            new PendingExecute(this.backend, step.source, inputs, outputs),
          );
          break;
        }
        case "malloc": {
          const slot = this.backend.malloc(step.size);
          scope.set(step.output, slot);
          break;
        }
        case "incref": {
          const slot = scope.get(step.input)!;
          this.backend.incRef(slot);
          break;
        }
        case "free": {
          const slot = scope.get(step.input)!;
          this.backend.decRef(slot);
          scope.delete(step.input);
          break;
        }
        case "scan": {
          if (!scanRunner) {
            throw new Error("internal: scan step requires scanRunner callback");
          }

          // IMPORTANT: Submit all pending operations first so input data is computed!
          // This ensures that any preceding kernels (like Transpose) have written
          // their output before the scan tries to read from those buffers.
          for (const p of pending) {
            p.prepareSync();
            p.submit();
          }
          pending.length = 0; // Clear the pending array

          // Get outer scope slots
          const constSlots = step.consts.map((id) => scope.get(id)!);
          const initCarrySlots = step.initCarry.map((id) => scope.get(id)!);
          const xsSlots = step.xs.map((id) => scope.get(id)!);
          const outputSlots = step.outputs.map((id) => scope.get(id)!);

          if (DEBUG >= 2) {
            console.log(`[jit.scan] step.xs=${step.xs}, xsSlots=${xsSlots}`);
            console.log(
              `[jit.scan] step.xsAvals=${JSON.stringify(step.xsAvals?.map((a) => ({ shape: a.shape, dtype: a.dtype })))}`,
            );
            // Read xs data
            for (let i = 0; i < xsSlots.length; i++) {
              const data = this.backend.readSync(xsSlots[i]);
              console.log(
                `[jit.scan] xs[${i}] data:`,
                new Float32Array(data.buffer),
              );
            }
          }

          if (DEBUG >= 2) {
            console.log(
              `[jit.scan] Before scanRunner: outputSlots=${outputSlots}, step.outputs=${step.outputs}`,
            );
          }

          // Delegate to scanRunner callback - it returns the output slots
          const result = scanRunner(
            step.bodyProgram,
            this.backend,
            step.bodyJaxpr,
            step.length,
            step.numCarry,
            step.numConsts,
            step.numX,
            step.numY,
            step.reverse,
            constSlots,
            initCarrySlots,
            xsSlots,
            step.xsAvals,
            outputSlots,
          );

          if (DEBUG >= 2) {
            console.log(
              `[jit.scan] After scanRunner: result.outputs=${result.outputs}`,
            );
          }

          // Put returned outputs into scope
          for (let i = 0; i < step.outputs.length; i++) {
            scope.set(step.outputs[i], result.outputs[i]);
          }

          if (DEBUG >= 2) {
            console.log(
              `[jit.scan] After scope.set: scope.get(${step.outputs[0]})=${scope.get(step.outputs[0])}`,
            );
          }

          pending.push(...result.pending);
          break;
        }
        case "native-scan": {
          // Native scan - dispatch directly to backend
          // IMPORTANT: Submit all pending operations first so input data is computed!
          for (const p of pending) {
            p.prepareSync();
            p.submit();
          }
          pending.length = 0; // Clear the pending array

          const constSlots = step.consts.map((id) => scope.get(id)!);
          const initCarrySlots = step.initCarry.map((id) => scope.get(id)!);
          const xsSlots = step.xs.map((id) => scope.get(id)!);
          const outputSlots = step.outputs.map((id) => scope.get(id)!);

          // Split outputs into carryOut and ysStacked
          const carryOutSlots = outputSlots.slice(0, step.numCarry);
          const ysStackedSlots = outputSlots.slice(step.numCarry);

          // Check if backend supports native scan dispatch
          const backend = this.backend as any;

          // Use general dispatch if generalParams is provided (handles both kernels and routines)
          if (step.generalParams) {
            if (typeof backend.dispatchNativeScanGeneral === "function") {
              backend.dispatchNativeScanGeneral(
                step.executable,
                step.generalParams,
                constSlots,
                initCarrySlots,
                xsSlots,
                carryOutSlots,
                ysStackedSlots,
              );
            } else {
              throw new Error(
                "internal: general native-scan requires backend.dispatchNativeScanGeneral",
              );
            }
          } else if (typeof backend.dispatchNativeScan === "function") {
            backend.dispatchNativeScan(
              step.executable,
              constSlots,
              initCarrySlots,
              xsSlots,
              carryOutSlots,
              ysStackedSlots,
            );
          } else {
            throw new Error(
              "internal: native-scan requires backend.dispatchNativeScan",
            );
          }
          break;
        }
        case "batched-scan": {
          // Batched scan for routine bodies (matmul, conv, etc.)
          // IMPORTANT: Submit all pending operations first so input data is computed!
          for (const p of pending) {
            p.prepareSync();
            p.submit();
          }
          pending.length = 0; // Clear the pending array

          const constSlots = step.consts.map((id) => scope.get(id)!);
          const initCarrySlots = step.initCarry.map((id) => scope.get(id)!);
          const xsSlots = step.xs.map((id) => scope.get(id)!);
          const outputSlots = step.outputs.map((id) => scope.get(id)!);

          // Split outputs into carryOut and ysStacked
          const carryOutSlots = outputSlots.slice(0, step.numCarry);
          const ysStackedSlots = outputSlots.slice(step.numCarry);

          // Check if backend supports batched scan dispatch
          const backend = this.backend as any;
          if (typeof backend.dispatchBatchedScan === "function") {
            backend.dispatchBatchedScan(
              step.batchedParams, // PreparedBatchedScan
              constSlots,
              initCarrySlots,
              xsSlots,
              carryOutSlots,
              ysStackedSlots,
            );
          } else {
            throw new Error(
              "internal: batched-scan requires backend.dispatchBatchedScan",
            );
          }
          break;
        }
        default:
          step satisfies never;
      }
    }
    return {
      outputs: this.outputs.map((id) => scope.get(id)!),
      pending,
    };
  }
}

class JitProgramBuilder {
  backend: Backend;
  #nextId: number;
  steps: JitStep[];

  constructor(backend: Backend, nargs: number) {
    this.backend = backend;
    this.#nextId = nargs;
    this.steps = [];
  }

  pushLit(lit: Lit): JitId {
    const kernel = new Kernel(
      0,
      lit.aval.size,
      AluExp.const(lit.dtype, lit.value),
    );
    return this.pushKernel(kernel, []);
  }

  pushBuffer(size: number): JitId {
    const id = this.#nextId++;
    this.steps.push({
      type: "malloc",
      size,
      output: id,
    });
    return id;
  }

  pushKernel(kernel: Kernel, inputs: JitId[]): JitId {
    const id = this.pushBuffer(kernel.bytes);
    this.steps.push({
      type: "execute",
      source: kernel,
      inputs,
      outputs: [id],
    });
    return id;
  }

  pushRoutine(routine: Routine, inputs: JitId[], outputs: JitId[]): void {
    this.steps.push({
      type: "execute",
      source: routine,
      inputs,
      outputs,
    });
  }

  pushIncref(id: JitId): void {
    this.steps.push({
      type: "incref",
      input: id,
    });
  }

  insertFreeSteps(outputIds: JitId[]): void {
    // Only free malloc'd ids that are not used in the output.
    //
    // Intermediates are allowed to be freed independently, since they are
    // guaranteed to be unused elsewhere after the JitProgram is executed.
    // Meanwhile, inputs and consts are owned / freed elsewhere.
    const ids = this.steps
      .filter((s) => s.type === "malloc")
      .map((s) => s.output);
    for (const id of ids) {
      // Find the last usage of this id.
      if (outputIds.includes(id)) continue;
      const lastUsage = this.steps.findLastIndex(
        (s) =>
          (s.type === "execute" &&
            (s.outputs.includes(id) || s.inputs.includes(id))) ||
          (s.type === "malloc" && s.output === id) ||
          (s.type === "scan" &&
            (s.consts.includes(id) ||
              s.initCarry.includes(id) ||
              s.xs.includes(id) ||
              s.outputs.includes(id))) ||
          (s.type === "native-scan" &&
            (s.consts.includes(id) ||
              s.initCarry.includes(id) ||
              s.xs.includes(id) ||
              s.outputs.includes(id))) ||
          (s.type === "batched-scan" &&
            (s.consts.includes(id) ||
              s.initCarry.includes(id) ||
              s.xs.includes(id) ||
              s.outputs.includes(id))),
      )!;
      this.steps.splice(lastUsage + 1, 0, {
        type: "free",
        input: id,
      });
    }
  }

  pushFree(id: JitId): void {
    // Should be paired with the output of pushKernel() when last used.
    this.steps.push({
      type: "free",
      input: id,
    });
  }
}

type JitValue =
  | { type: "imm"; arg: JitId } // Immediate
  | { type: "exp"; exp: AluExp; args: JitId[] } // Expression, lazily fused
  | { type: "red"; exp: AluExp; reduction: Reduction; args: JitId[] }; // Reduction + epilogue

const jitCompileCache = new Map<string, JitProgram>();

export function jitCompile(backend: Backend, jaxpr: Jaxpr): JitProgram {
  const cacheKey = backend.type + "," + FpHash.hash(jaxpr);

  const cached = jitCompileCache.get(cacheKey);
  if (cached) return cached;

  jaxpr = jaxpr.flatten().simplify();
  const nargs = jaxpr.inBinders.length;
  const builder = new JitProgramBuilder(backend, nargs);

  const blackNodes = splitGraphDataflow(backend, jaxpr);

  // Initialize jaxpr inBinders.
  const ctx = new Map<Var, JitValue>();
  for (let i = 0; i < nargs; i++) {
    const v = jaxpr.inBinders[i];
    ctx.set(v, { type: "imm", arg: i }); // JitId i = input #i
  }

  // Now run each primitive through a set of rules, mirroring implRules.
  for (let i = 0; i < jaxpr.eqns.length; i++) {
    const eqn = jaxpr.eqns[i];

    // If this is a routine, construct and dispatch the routine.
    if (routinePrimitives.has(eqn.primitive)) {
      // The rest of the code collaborates to make sure that all inputs to a
      // routine are "imm" (black node, dispatched) and so is itself.
      const routine = new Routine(
        routinePrimitives.get(eqn.primitive)!,
        {
          inputShapes: eqn.inputs.map((x) => x.aval.shape),
          inputDtypes: eqn.inputs.map((x) => x.aval.dtype),
          outputShapes: eqn.outBinders.map((x) => x.aval.shape),
          outputDtypes: eqn.outBinders.map((x) => x.aval.dtype),
        },
        eqn.params as any,
      );
      const inputs: JitId[] = [];
      for (const input of eqn.inputs) {
        if (input instanceof Var) {
          const jv = ctx.get(input)!;
          if (jv.type !== "imm") {
            throw new Error(
              `jit: routine primitive ${eqn.primitive} input is not imm`,
            );
          }
          inputs.push(jv.arg);
        } else if (input instanceof Lit) {
          inputs.push(builder.pushLit(input));
        }
      }
      const outputs: JitId[] = [];
      for (const outVar of eqn.outBinders) {
        const outId = builder.pushBuffer(
          outVar.aval.size * byteWidth(outVar.aval.dtype),
        );
        outputs.push(outId);
        ctx.set(outVar, { type: "imm", arg: outId });
      }
      builder.pushRoutine(routine, inputs, outputs);
      continue;
    }

    // Handle Scan primitive specially - store jaxpr and run interpreter at execute time
    if (eqn.primitive === Primitive.Scan) {
      const params = eqn.params as PrimitiveParams<typeof Primitive.Scan>;
      const {
        jaxpr: bodyJaxpr,
        numCarry,
        numConsts,
        length,
        reverse,
        requirePath,
      } = params;
      const numX = bodyJaxpr.inBinders.length - numConsts - numCarry;
      const numY = bodyJaxpr.outs.length - numCarry;

      // Get input JitIds from context (layout: [consts..., carry..., xs...])
      const inputs: JitId[] = [];
      for (const input of eqn.inputs) {
        if (input instanceof Var) {
          const jv = ctx.get(input)!;
          if (jv.type !== "imm") {
            throw new Error(`jit: scan primitive input is not imm`);
          }
          inputs.push(jv.arg);
        } else if (input instanceof Lit) {
          inputs.push(builder.pushLit(input));
        }
      }

      // Split inputs by role
      const constsIds = inputs.slice(0, numConsts);
      const initCarryIds = inputs.slice(numConsts, numConsts + numCarry);
      const xsIds = inputs.slice(numConsts + numCarry);

      // Get xs avals from input vars (these are the actual shapes after any transforms like vmap)
      const xsAvals: ShapedArray[] = [];
      const xsInputs = eqn.inputs.slice(numConsts + numCarry);
      for (const input of xsInputs) {
        if (input instanceof Var) {
          xsAvals.push(input.aval);
        } else if (input instanceof Lit) {
          xsAvals.push(input.aval);
        }
      }

      // Create output buffers (layout: [carry_out..., stacked_ys...])
      const outputs: JitId[] = [];
      for (const outVar of eqn.outBinders) {
        const outId = builder.pushBuffer(
          outVar.aval.size * byteWidth(outVar.aval.dtype),
        );
        outputs.push(outId);
        ctx.set(outVar, { type: "imm", arg: outId });
      }

      // Compile the body jaxpr to a JitProgram for efficient per-iteration execution
      const bodyProgram = jitCompile(backend, bodyJaxpr);

      // Try to use native scan (WASM/WebGPU, handles all kernel/routine bodies)
      const generalScanResult = tryPrepareNativeScanGeneral(
        backend,
        bodyProgram,
        bodyJaxpr,
        length,
        numCarry,
        numConsts,
        numX,
        numY,
        reverse,
      );
      const nativeScanExe = generalScanResult?.executable ?? null;

      if (nativeScanExe) {
        // Report fused path (loop runs in native code)
        const pathError = checkRequiredPath("fused", requirePath);
        if (pathError) throw new Error(pathError);
        reportScanPath("fused", backend.type, { numConsts, numCarry, length });

        // Use native scan
        builder.steps.push({
          type: "native-scan",
          executable: nativeScanExe,
          length,
          numCarry,
          numConsts,
          numY,
          reverse,
          consts: constsIds,
          initCarry: initCarryIds,
          xs: xsIds,
          outputs,
          generalParams: generalScanResult?.params,
        });
        continue;
      }

      // Try to use batched scan for routine bodies (WebGPU only, matmul/conv/etc.)
      const batchedParams = tryPrepareBatchedScan(
        backend,
        bodyProgram,
        bodyJaxpr,
        length,
        numCarry,
        numConsts,
        numX,
        numY,
        eqn,
        reverse,
      );

      if (batchedParams) {
        // Use batched scan (pre-encoded dispatches, still fused)
        const pathError = checkRequiredPath("fused", requirePath);
        if (pathError) throw new Error(pathError);
        reportScanPath("fused", backend.type, {
          numConsts,
          numCarry,
          length,
        });
        builder.steps.push({
          type: "batched-scan",
          batchedParams,
          length,
          numCarry,
          numConsts,
          numX,
          numY,
          reverse,
          consts: constsIds,
          initCarry: initCarryIds,
          xs: xsIds,
          outputs,
        });
        continue;
      }

      // Fall back to JS loop scan
      const pathError = checkRequiredPath("fallback", requirePath);
      if (pathError) throw new Error(pathError);
      reportScanPath("fallback", backend.type, {
        numConsts,
        numCarry,
        length,
      });
      builder.steps.push({
        type: "scan",
        bodyProgram,
        bodyJaxpr,
        length,
        numCarry,
        numConsts,
        numX,
        numY,
        reverse,
        consts: constsIds,
        initCarry: initCarryIds,
        xs: xsIds,
        xsAvals,
        outputs,
      });
      continue;
    }

    // Transform each input into an AluExp to start, and normalize any arguments
    // as needed.
    const inputExps: AluExp[] = []; // len(inputs)
    const inputAvals: ShapedArray[] = []; // len(inputs)
    const inputArgs: JitId[] = [];

    let inputReduction: (JitValue & { type: "red" }) | null = null;

    // May need to reindex gids to match order, returns array of new gids.
    const addArgs = (args: JitId[]): number[] => {
      const newGids: number[] = [];
      for (const jitId of args) {
        let newGid = inputArgs.indexOf(jitId);
        if (newGid === -1) {
          newGid = inputArgs.length;
          inputArgs.push(jitId);
        }
        newGids.push(newGid);
      }
      return newGids;
    };

    for (const input of eqn.inputs) {
      if (input instanceof Var) {
        const jv = ctx.get(input)!;
        if (jv.type === "exp") {
          const newGids = addArgs(jv.args);
          inputExps.push(jv.exp.reindexGids(newGids));
        } else if (jv.type === "imm") {
          const [gid] = addArgs([jv.arg]);
          const st = ShapeTracker.fromShape(input.aval.shape); // "imm" is realized
          const indices = unravelAlu(st.shape, AluVar.gidx);
          inputExps.push(AluExp.globalView(input.aval.dtype, gid, st, indices));
        } else if (jv.type === "red") {
          // Special case: We are consuming a 'red' JitValue, so we must be in the
          // fused epilogue of a reduction.
          if (inputReduction)
            throw new Error("jit: unexpected, multiple red inputs");
          const newGids = addArgs(jv.args);
          inputExps.push(jv.reduction.epilogue.reindexGids(newGids));
          inputReduction = jv;
        } else {
          jv satisfies never; // static check
        }
        inputAvals.push(input.aval);
      } else if (input instanceof Lit) {
        inputExps.push(AluExp.const(input.dtype, input.value));
        inputAvals.push(input.aval);
      } else {
        throw new TypeError(`Unexpected input in Jaxpr: ${input}`);
      }
    }

    // Produce a new expression and/or reduction for the operation based on the
    // jit() implementation of the primitive.
    const rule = jitRules[eqn.primitive];
    if (!rule)
      throw new TypeError(`JIT not implemented for primitive ${eqn.primitive}`);

    let exp: AluExp[];
    let reduction: Reduction | undefined;

    if (inputReduction) {
      // Special case: we are in the fused epilogue of a reduction.
      const jv = inputReduction;
      const newEpilogue = rule(inputExps, inputAvals, eqn.params as any).exp[0];
      exp = [jv.exp.reindexGids(addArgs(jv.args))];
      reduction = new Reduction(
        jv.reduction.dtype,
        jv.reduction.op,
        jv.reduction.size,
        newEpilogue,
      );
    } else {
      const ruleOutput = rule(inputExps, inputAvals, eqn.params as any);
      exp = ruleOutput.exp;
      reduction = ruleOutput.reduction;
    }

    // Then dispatch the kernel, if it is a "black" node as determined from
    // dataflow analysis above.
    for (let i = 0; i < eqn.outBinders.length; i++) {
      const outVar = eqn.outBinders[i];
      if (blackNodes.has(outVar)) {
        const nargs = inputArgs.length;
        const size = outVar.aval.size;
        const kernel = new Kernel(nargs, size, exp[i], reduction);
        const outId = builder.pushKernel(kernel, inputArgs);
        ctx.set(outVar, { type: "imm", arg: outId });
      } else if (reduction) {
        // Reduction but not black, means it will have an epilogue.
        ctx.set(outVar, {
          type: "red",
          exp: exp[i],
          reduction,
          args: inputArgs,
        });
      } else {
        // Otherwise, fuse the kernel into the next expression.
        ctx.set(outVar, { type: "exp", exp: exp[i], args: inputArgs });
      }
    }
  }

  // Finally, loop through the outputs.
  const outputIds: JitId[] = [];
  for (const out of jaxpr.outs) {
    if (out instanceof Var) {
      const jitValue = ctx.get(out)!;
      if (jitValue.type !== "imm")
        throw new Error("internal: Expected imm, since outs are black nodes");
      outputIds.push(jitValue.arg);
    } else if (out instanceof Lit) {
      outputIds.push(builder.pushLit(out));
    } else {
      out satisfies never; // static check
    }
  }

  // Each output should have its own backend reference. If an output slot is
  // returned twice, or if an input is returned directly, insert "incref" steps
  // to balance the books.
  const outputNeedsRef = new Set<JitId>(range(nargs)); // inputs
  for (const outputId of outputIds) {
    if (outputNeedsRef.has(outputId)) {
      builder.pushIncref(outputId);
    } else {
      // If this output is seen again, increment its ref at that point.
      outputNeedsRef.add(outputId);
    }
  }

  // Emit free steps after last usage of any intermediates.
  builder.insertFreeSteps(outputIds);

  const jp = new JitProgram(backend, builder.steps, range(0, nargs), outputIds);
  jitCompileCache.set(cacheKey, jp);
  return jp;
}

/**
 * Rule for fusing the operation into a JIT expression to the backend.
 *
 * This takes in the expressions of the `src[]` inputs and produces a subsequent
 * expression, as well as optionally a reduction. The expressions use
 * AluVar.gidx (output index) and AluVar.ridx (reduction index).
 */
type JitRule<P extends Primitive> = (
  exps: AluExp[],
  avals: ShapedArray[],
  params: PrimitiveParams<P>,
) => {
  exp: AluExp[]; // One expression for each output
  reduction?: Reduction;
};

function reshapeViews(
  exp: AluExp,
  mapping: (st: ShapeTracker) => ShapeTracker | undefined,
  reduceAxis: boolean = false,
): AluExp {
  return exp.rewrite((exp) => {
    if (exp.op === AluOp.GlobalView) {
      const [gid, st]: [number, ShapeTracker] = exp.arg;
      const newSt = mapping(st);
      if (newSt) {
        const indices = reduceAxis
          ? unravelAlu(newSt.shape.slice(0, -1), AluVar.gidx).concat(
              AluVar.ridx,
            )
          : unravelAlu(newSt.shape, AluVar.gidx);
        return AluExp.globalView(exp.dtype, gid, newSt, indices);
      }
    } else if (exp.op === AluOp.GlobalIndex) {
      throw new Error("internal: reshapeViews() called with GlobalIndex op");
    }
  });
}

// JIT handler for a broadcasted operation on at least 1 input.
function broadcastedJit<P extends Primitive>(
  fn: (exps: AluExp[], params: PrimitiveParams<P>) => AluExp,
  opts?: { skipCastIdx?: number[] },
): JitRule<P> {
  return (exps, avals, params) => {
    let { shape: newShape, dtype: newDtype } = avals.reduce(promoteAvals);

    const skipCastIdx = opts?.skipCastIdx ?? [];
    if (skipCastIdx.length) {
      // Skip casting some indices to a shared dtype.
      newDtype = avals
        .filter((_, i) => !skipCastIdx.includes(i))
        .reduce(promoteAvals).dtype;
    }

    // Perform a broadcast on each of the input expressions.
    //
    // Only GlobalView is affected. GlobalIndex is not used here, and neither is
    // AluVar.idx, since those are realized before jit().
    exps = exps.map((exp, i) => {
      exp = reshapeViews(exp, (st) => {
        if (!deepEqual(st.shape, newShape))
          return st.broadcast(
            newShape,
            range(newShape.length - st.shape.length),
          );
      });
      if (exp.dtype !== newDtype && !skipCastIdx.includes(i)) {
        exp = AluExp.cast(newDtype, exp);
      }
      return exp;
    });

    // Then, we can call the function to produce a new expression.
    return { exp: [fn(exps, params)] };
  };
}

// Simpler JIT handler, equivalent to broadcastedJit for unary ops.
function unopJit<P extends Primitive>(
  fn: (exp: AluExp, params: PrimitiveParams<P>) => AluExp,
): JitRule<P> {
  return ([a], [_as], params) => {
    return { exp: [fn(a, params)] };
  };
}

function reshapeJit<P extends Primitive>(
  fn: (st: ShapeTracker, params: PrimitiveParams<P>) => ShapeTracker,
): JitRule<P> {
  return ([a], [_as], params) => {
    return { exp: [reshapeViews(a, (st) => fn(st, params))] };
  };
}

function routineNoJit<P extends Primitive>(): JitRule<P> {
  return () => {
    throw new Error("jit: rule is not implemented for routines");
  };
}

const jitRules: { [P in Primitive]: JitRule<P> } = {
  [Primitive.Add]: broadcastedJit(([a, b]) => AluExp.add(a, b)),
  [Primitive.Mul]: broadcastedJit(([a, b]) => AluExp.mul(a, b)),
  [Primitive.Idiv]: broadcastedJit(([a, b]) => AluExp.idiv(a, b)),
  [Primitive.Mod]: broadcastedJit(([a, b]) => AluExp.mod(a, b)),
  [Primitive.Min]: broadcastedJit(([a, b]) => AluExp.min(a, b)),
  [Primitive.Max]: broadcastedJit(([a, b]) => AluExp.max(a, b)),
  [Primitive.Neg]: unopJit((a) => AluExp.sub(AluExp.const(a.dtype, 0), a)),
  [Primitive.Reciprocal]: unopJit(AluExp.reciprocal),
  [Primitive.Floor]: unopJit(AluExp.floor),
  [Primitive.Ceil]: unopJit(AluExp.ceil),
  [Primitive.StopGradient]: unopJit((a) => a), // No-op, just return the input.
  [Primitive.Cast]: unopJit((a, { dtype }) => AluExp.cast(dtype, a)),
  [Primitive.Bitcast]: unopJit((a, { dtype }) => AluExp.bitcast(dtype, a)),
  [Primitive.Sin]: unopJit(AluExp.sin),
  [Primitive.Cos]: unopJit(AluExp.cos),
  [Primitive.Asin]: unopJit(AluExp.asin),
  [Primitive.Atan]: unopJit(AluExp.atan),
  [Primitive.Exp]: unopJit(AluExp.exp),
  [Primitive.Log]: unopJit(AluExp.log),
  [Primitive.Erf]: unopJit(AluExp.erf),
  [Primitive.Erfc]: unopJit(AluExp.erfc),
  [Primitive.Sqrt]: unopJit(AluExp.sqrt),
  [Primitive.Reduce]([a], [as], { op, axis }) {
    const keptAxes: number[] = [];
    const shiftedAxes: number[] = [];
    const newShape: number[] = [];
    for (let i = 0; i < as.shape.length; i++) {
      if (axis.includes(i)) shiftedAxes.push(i);
      else {
        keptAxes.push(i);
        newShape.push(as.shape[i]);
      }
    }
    const reductionSize = prod(shiftedAxes.map((ax) => as.shape[ax]));
    newShape.push(reductionSize);

    const perm = keptAxes.concat(shiftedAxes);
    a = reshapeViews(a, (st) => st.permute(perm).reshape(newShape), true);
    const reduction = new Reduction(a.dtype, op, reductionSize);
    return { exp: [a], reduction };
  },
  [Primitive.Pool]: reshapeJit((st, { window, strides }) =>
    pool(st, window, strides),
  ),
  [Primitive.PoolTranspose]([a], [as], { inShape, window, strides }) {
    let stX = poolTranspose(
      ShapeTracker.fromShape(as.shape),
      inShape,
      window,
      strides,
    );
    stX = stX.reshape([...inShape, prod(stX.shape.slice(inShape.length))]); // Combine all reduce axes.
    a = reshapeViews(a, (st) => st.compose(stX), true);
    const reduction = new Reduction(
      a.dtype,
      AluOp.Add,
      stX.shape[stX.shape.length - 1],
    );
    return { exp: [a], reduction };
  },
  [Primitive.Dot]([a, b], [as, bs]) {
    // Dot is just Mul->Reduce in sequence.
    const k1 = jitRules[Primitive.Mul]([a, b], [as, bs], {});
    const [c] = k1.exp;
    const cs = promoteAvals(as, bs);
    return jitRules[Primitive.Reduce]([c], [cs], {
      op: AluOp.Add,
      axis: [cs.ndim - 1],
    });
  },
  [Primitive.Conv]([a, b], [as, bs], params) {
    const [stX, stY] = prepareConv(
      ShapeTracker.fromShape(as.shape),
      ShapeTracker.fromShape(bs.shape),
      params,
    );
    a = reshapeViews(a, (st) => st.compose(stX));
    b = reshapeViews(b, (st) => st.compose(stY));
    as = new ShapedArray(stX.shape, as.dtype, as.weakType);
    bs = new ShapedArray(stY.shape, bs.dtype, bs.weakType);
    return jitRules[Primitive.Dot]([a, b], [as, bs], {});
  },
  [Primitive.Compare]: broadcastedJit(([a, b], { op }) => aluCompare(a, b, op)),
  [Primitive.Where]: broadcastedJit(
    ([cond, a, b]) => AluExp.where(cond, a, b),
    { skipCastIdx: [0] },
  ),
  [Primitive.Concatenate](exps, avals, { axis }) {
    const ndim = avals[0].ndim;
    const sizes = avals.map((x) => x.shape[axis]);
    const finalSize = sizes.reduce((a, b) => a + b, 0);
    const { dtype: dtypeOut } = avals
      .map((x) => x.scalar())
      .reduce(promoteAvals);
    const makePadAxis = (start: number, end: number): Pair[] =>
      range(ndim).map((i) => (i === axis ? [start, end] : [0, 0]));
    let cum = 0;
    const src: AluExp[] = [];
    for (let i = 0; i < exps.length; i++) {
      const padding = makePadAxis(cum, finalSize - cum - sizes[i]);
      src.push(
        reshapeViews(AluExp.cast(dtypeOut, exps[i]), (st) => st.pad(padding)),
      );
      cum += sizes[i];
    }
    return { exp: [src.reduce(AluExp.add)] };
  },
  [Primitive.Split]([a], [as], { axis, sizes }) {
    const exp: AluExp[] = [];
    let start = 0;
    for (const size of sizes) {
      const slice = range(as.ndim).map<Pair>((d) =>
        d === axis ? [start, start + size] : [0, as.shape[d]],
      );
      exp.push(reshapeViews(a, (st) => st.shrink(slice)));
      start += size;
    }
    return { exp };
  },
  [Primitive.RandomBits]: (keys, keyShapes, { shape, mode }) => {
    const keyShape = keyShapes[0].shape;
    const mapping = (st: ShapeTracker): ShapeTracker | undefined => {
      if (!deepEqual(st.shape, shape))
        return st.broadcast(shape, range(st.shape.length, shape.length));
    };
    const k0 = reshapeViews(keys[0], mapping);
    const k1 = reshapeViews(keys[1], mapping);
    const c0 = AluExp.u32(0);
    const c1 = AluExp.mod(
      AluExp.cast(DType.Uint32, AluVar.gidx),
      // max(..., 1) to avoid mod-by-zero compile error in degenerate case
      AluExp.u32(Math.max(prod(shape.slice(keyShape.length)), 1)),
    );
    const exp = AluExp.threefry2x32(k0, k1, c0, c1, mode);
    return { exp: [exp] };
  },
  [Primitive.Gather](
    [x, ...indices],
    [xs, ...indicesShapes],
    { axis, outDim },
  ) {
    const axisSet = new Set(axis);

    // First, broadcast each integer array in `indices`.
    const indexShape = indicesShapes
      .map((c) => c.shape)
      .reduce(generalBroadcast);
    const finalShape = xs.shape.filter((_, i) => !axisSet.has(i));
    finalShape.splice(outDim, 0, ...indexShape);

    // Make variables for expression indices for gathered axes, and non-axis.
    const idxAll = unravelAlu(finalShape, AluVar.gidx);
    const idxNonaxis = [...idxAll];
    const _idxAxis = idxNonaxis.splice(outDim, indexShape.length);

    // Then, construct a kernel expression that gathers the data.
    const src: AluExp[] = [...idxNonaxis];
    for (let i = 0; i < xs.shape.length; i++) {
      // insert 'null' as axis placeholder, overwritten below as src[axis[i]].
      if (axisSet.has(i)) src.splice(i, 0, null as any);
    }

    for (const [i, iexp] of indices.entries()) {
      // Index iexp by the idxAxis variables, after broadcasting (via GlobalView).
      // [ ... | outDim | ... <iexp> | outDim + indexShape.length | ... ]
      src[axis[i]] = AluExp.cast(
        DType.Int32,
        reshapeViews(iexp, (st) =>
          st.broadcast(finalShape, [
            // Broadcast indices (aligned to the right), plus leading before outDim.
            ...range(outDim + indexShape.length - st.shape.length),
            // Indices to the right of outDim.
            ...range(outDim + indexShape.length, finalShape.length),
          ]),
        ),
      );
    }

    // Finally, index into "x" by replacing its gidx with a flat accessor into
    // the gathered indices.
    const [index, valid] = ShapeTracker.fromShape(xs.shape).toAluExp(src);
    if (!valid.resolve())
      throw new Error("internal: expected full validity mask in Gather");
    return { exp: [x.substitute({ gidx: index })] };
  },
  [Primitive.Transpose]: reshapeJit((st, { perm }) => st.permute(perm)),
  [Primitive.Broadcast]: reshapeJit((st, { shape, axis }) =>
    st.broadcast(shape, axis),
  ),
  [Primitive.Reshape]: reshapeJit((st, { shape }) => st.reshape(shape)),
  [Primitive.Flip]: reshapeJit((st, { axis }) => {
    const arg = rep(st.shape.length, false);
    for (const ax of axis) arg[ax] = true;
    return st.flip(arg);
  }),
  [Primitive.Shrink]: reshapeJit((st, { slice }) => st.shrink(slice)),
  [Primitive.Pad]: reshapeJit((st, { width }) => st.pad(width)),
  [Primitive.Sort]: routineNoJit(),
  [Primitive.Argsort]: routineNoJit(),
  [Primitive.TriangularSolve]: routineNoJit(),
  [Primitive.Cholesky]: routineNoJit(),
  [Primitive.LU]: routineNoJit(),
  [Primitive.Jit]() {
    throw new Error(
      "internal: Jit should have been flattened before JIT compilation",
    );
  },
  [Primitive.Scan]() {
    throw new Error(
      "internal: Scan is handled specially in jitCompile, not via jitRules",
    );
  },
};

/** Determines how to split the Jaxpr into kernels via dataflow analysis. */
function splitGraphDataflow(backend: Backend, jaxpr: Jaxpr): Set<Var> {
  const varToDefn = new Map<Var, number>(); // Var -> eqn index of definition
  const varToUsages: Map<Var, number[]> = new Map(); // Var -> eqn indices of usages
  for (let i = 0; i < jaxpr.eqns.length; i++) {
    const eqn = jaxpr.eqns[i];
    for (const v of eqn.outBinders) {
      if (v instanceof Var) varToDefn.set(v, i);
    }
    for (const input of eqn.inputs) {
      if (input instanceof Var) {
        const usages = varToUsages.get(input);
        if (usages) usages.push(i);
        else varToUsages.set(input, [i]);
      }
    }
  }

  // Calculate reduction epilogues.
  //
  // A reduction can be fused with one or more operations that use its output,
  // which are either 1) unary or 2) binary ops with a literal, or an array not
  // larger than it.
  //
  // We also need to make sure we don't fuse two reductions together.
  const reducePrimitives = [
    Primitive.Reduce,
    Primitive.Dot,
    Primitive.Conv,
    Primitive.PoolTranspose,
  ];
  const reductionEpilogueEqns = new Set<number>();
  const reductionEndpointEqns = new Set<number>();
  for (let i = 0; i < jaxpr.eqns.length; i++) {
    const eqn = jaxpr.eqns[i];
    if (reducePrimitives.includes(eqn.primitive)) {
      let head = i;
      while (true) {
        reductionEpilogueEqns.add(head);

        // Try moving outVar forward through the graph.
        const outVar = jaxpr.eqns[head].outBinders[0];
        const usages = varToUsages.get(outVar) ?? [];
        if (jaxpr.outs.includes(outVar) || usages.length !== 1) break;

        // Next is already fused into a reduction epilogue, can't fuse again.
        if (reductionEpilogueEqns.has(usages[0])) break;

        const nextEqn = jaxpr.eqns[usages[0]];
        switch (nextEqn.primitive) {
          // We can always fuse unary operations.
          case Primitive.Neg:
          case Primitive.Reciprocal:
          case Primitive.Floor:
          case Primitive.Ceil:
          case Primitive.StopGradient:
          case Primitive.Cast:
          case Primitive.Bitcast:
          case Primitive.Sin:
          case Primitive.Cos:
          case Primitive.Asin:
          case Primitive.Atan:
          case Primitive.Exp:
          case Primitive.Log:
          case Primitive.Erf:
          case Primitive.Erfc:
          case Primitive.Sqrt:
            head = usages[0];
            continue;

          // We can fuse binary operations with a literal, or with an array such
          // that the array doesn't lead to broadcasting thus recomputation.
          case Primitive.Add:
          case Primitive.Mul:
          case Primitive.Idiv:
          case Primitive.Mod:
          case Primitive.Min:
          case Primitive.Max: {
            const otherInput = nextEqn.inputs.find((v) => v !== outVar)!;
            if (
              otherInput instanceof Lit ||
              deepEqual(
                generalBroadcast(otherInput.aval.shape, outVar.aval.shape),
                outVar.aval.shape,
              )
            ) {
              head = usages[0];
              continue;
            }
            break;
          }
        }
        break; // Can't move forward anymore.
      }
      reductionEndpointEqns.add(head);
    }
  }

  // Move backwards through the program and compute "black" endpoints.
  //
  // Black nodes are the endpoints where we dispatch a kernel to the backend
  // rather than producing intermediates. This includes:
  //
  // - Kernel outputs
  // - Reductions, except when fused with epilogue
  // - Gather/RandomBits operations (violates rule that kernels must have
  //   homogeneous GlobalView indices)
  // - Inputs to Pad operations, which need clean inputs
  //
  // Also, mark a node black if there are at least two black nodes that can be
  // reached from it, while only going through non-black nodes.
  //
  // TODO: Don't do the above for 'simple' nodes: reshape, cast, etc.
  const blackNodes = new Set<Var>();
  const p1NextBlack = new Map<Var, Var>();
  for (const v of jaxpr.outs) {
    if (v instanceof Var) {
      blackNodes.add(v);
      p1NextBlack.set(v, v);
    }
  }
  const heterogeneousViewPrimitives = [
    // These primitives generate heterogeneous GlobalView outputs, there are
    // multiple views in the expression with different indexing.
    Primitive.RandomBits,
    Primitive.Gather,
  ];
  const needsCleanShapePrimitives = [
    // Concatenate is based on Pad internally.
    Primitive.Concatenate,
    // If Pad is applied to a non-clean input, the reshaped padding would apply
    // to the view _inside_ of the expression. Imagine `GlobalView(...)+1`: if
    // you reshape each view, it adds zeros into the inner expression, so the
    // effect is to pad the intermediate with 1s instead of 0s!
    Primitive.Pad,
  ];
  for (let i = jaxpr.eqns.length - 1; i >= 0; i--) {
    const eqn = jaxpr.eqns[i];
    if (
      reductionEndpointEqns.has(i) ||
      heterogeneousViewPrimitives.includes(eqn.primitive) ||
      routinePrimitives.has(eqn.primitive) ||
      eqn.outBinders.some((v) => blackNodes.has(v))
    ) {
      for (const v of eqn.outBinders) {
        blackNodes.add(v);
        p1NextBlack.set(v, v);
      }
      continue;
    }
    const reach = new Set<Var>();
    let needsCleanOutput = false;
    outer: for (const v of eqn.outBinders) {
      for (const j of varToUsages.get(v) ?? []) {
        if (
          needsCleanShapePrimitives.includes(jaxpr.eqns[j].primitive) ||
          routinePrimitives.has(jaxpr.eqns[j].primitive)
        ) {
          needsCleanOutput = true;
          break outer;
        }
        for (const o of jaxpr.eqns[j].outBinders) {
          const u = p1NextBlack.get(o);
          if (u) reach.add(u);
        }
      }
    }
    if (reach.size > 1 || needsCleanOutput) {
      for (const v of eqn.outBinders) {
        blackNodes.add(v);
        p1NextBlack.set(v, v);
      }
    } else if (reach.size === 1) {
      const b = reach.values().next().value!;
      for (const v of eqn.outBinders) p1NextBlack.set(v, b);
    }
  }

  // Also, mark nodes black if the maximum number of arguments per kernel is
  // exceeded (i.e., maxComputeBuffersPerShaderStage for WebGPU). This needs to
  // be done in a second forward pass over the equations list.
  const p2Deps = new Map<Var, Set<Var>>(); // -> members are Var (black) or inBinders.
  for (const v of jaxpr.inBinders) {
    p2Deps.set(v, new Set([v])); // Each input is a dependency of itself.
  }
  let p2idx = 0;
  while (p2idx < jaxpr.eqns.length) {
    const eqn = jaxpr.eqns[p2idx++];
    const deps: Set<Var>[] = [];
    for (const input of eqn.inputs) {
      if (input instanceof Var) {
        if (blackNodes.has(input)) deps.push(new Set([input]));
        else deps.push(p2Deps.get(input)!);
      } else {
        deps.push(new Set());
      }
    }
    const depCounter = new Map<Var, number>(); // includes counts
    for (const depSet of deps) {
      for (const dep of depSet) {
        depCounter.set(dep, (depCounter.get(dep) ?? 0) + 1);
      }
    }
    if (depCounter.size > backend.maxArgs) {
      // We have too many dependencies, so we need to backtrack and mark one of
      // the inputs as black. By heuristic, we'll mark the one with the most
      // unique dependencies.
      let maxUniqueDeps = 0;
      let assocInput = -1;
      for (let i = 0; i < eqn.inputs.length; i++) {
        const input = eqn.inputs[i];
        if (input instanceof Var && varToDefn.has(input)) {
          let uniqueDeps = 0;
          for (const dep of deps[i]) {
            if (depCounter.get(dep) === 1) uniqueDeps++;
          }
          if (uniqueDeps > maxUniqueDeps) {
            maxUniqueDeps = uniqueDeps;
            assocInput = i;
          }
        }
      }
      if (assocInput === -1) {
        throw new Error(
          `internal: maxArgs, no input found to mark as black in Jaxpr equation ${eqn}`,
        );
      }
      const assocVar = eqn.inputs[assocInput] as Var;
      p2idx = varToDefn.get(assocVar)!; // backtrack to that equation
      for (const out of jaxpr.eqns[p2idx++].outBinders) {
        blackNodes.add(out);
      }
    } else {
      const s = new Set(depCounter.keys());
      for (const out of eqn.outBinders) p2Deps.set(out, s);
    }
  }

  return blackNodes;
}

/**
 * Try to prepare a native scan executable.
 * Returns the executable if native scan is possible, null otherwise.
 *
 * Native scan is only supported when:
 * 1. Backend is WASM or WebGPU
 * 2. Body program contains exactly one execute step with a Kernel (no routines)
 * 3. No constants (for MVP simplicity)
 * 4. No reduction in the body kernel
 */
/**
 * Try to prepare a batched scan for routine bodies (matmul, conv, etc.).
 * Returns the PreparedBatchedScan params if possible, null otherwise.
 *
 * Batched scan is only supported when:
 * 1. Backend is WebGPU
 * 2. Body program contains exactly one execute step with a Routine (not Kernel)
 * 3. MVP: No constants support
 * 4. MVP: numCarry === numY (carry and output are the same)
 */
function tryPrepareBatchedScan(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  length: number,
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  eqn: { inputs: (Var | Lit)[]; outBinders: Var[] },
  reverse: boolean,
): any | null {
  // Returns PreparedBatchedScan or null
  // Only WebGPU backend supports batched scan
  if (backend.type !== "webgpu") {
    if (DEBUG >= 2) console.log("Batched scan: skipped, unsupported backend");
    return null;
  }

  // Constants are supported - they're bound with no offset (same each iteration)

  // Find the single execute step with a Routine
  const executeSteps = bodyProgram.steps.filter((s) => s.type === "execute");
  if (executeSteps.length !== 1) {
    if (DEBUG >= 2)
      console.log(
        `Batched scan: skipped, ${executeSteps.length} execute steps (need exactly 1)`,
      );
    return null;
  }

  const execStep = executeSteps[0] as Extract<JitStep, { type: "execute" }>;
  if (!(execStep.source instanceof Routine)) {
    if (DEBUG >= 2) console.log("Batched scan: skipped, not a Routine");
    return null;
  }

  const bodyRoutine = execStep.source;

  // MVP: Only support case where carry and y are the same (like cumulative matmul)
  // This means numCarry === numY and the outputs reference the same variables
  if (numCarry !== numY) {
    if (DEBUG >= 2)
      console.log(
        `Batched scan: skipped, numCarry=${numCarry} !== numY=${numY}`,
      );
    return null;
  }

  // Get avals for computing sizes and strides
  const carryAvals = bodyJaxpr.inBinders
    .slice(numConsts, numConsts + numCarry)
    .map((v) => v.aval);
  const xAvals = bodyJaxpr.inBinders
    .slice(numConsts + numCarry)
    .map((v) => v.aval);

  // Compute carry sizes in bytes
  const carrySizes = carryAvals.map((a) => a.size * byteWidth(a.dtype));

  // Compute xs strides (ELEMENTS per iteration along axis 0)
  // xs has leading dimension = length, so stride = size_without_leading_dim
  const xsElemStrides = xAvals.map((a) => a.size); // elements per slice

  // Compute ys strides (same as carry for MVP)
  const ysElemStrides = carryAvals.map((a) => a.size); // elements per slice

  // Try to prepare the routine executable
  const webgpuBackend = backend as any;
  if (typeof webgpuBackend.prepareRoutineSync !== "function") {
    if (DEBUG >= 2)
      console.log("Batched scan: skipped, backend has no prepareRoutineSync");
    return null;
  }

  let bodyRoutineExe;
  try {
    bodyRoutineExe = webgpuBackend.prepareRoutineSync(bodyRoutine);
  } catch (e) {
    if (DEBUG >= 2) console.warn("Batched scan: prepareRoutineSync failed:", e);
    return null;
  }

  // Try to prepare batched scan
  if (typeof webgpuBackend.prepareBatchedScan !== "function") {
    if (DEBUG >= 2)
      console.log("Batched scan: skipped, backend has no prepareBatchedScan");
    return null;
  }

  const batchedScanParams = {
    length,
    carrySizes,
    xsElemStrides,
    ysElemStrides,
    bodyRoutine: bodyRoutineExe,
    numCarry,
    numX,
    numY,
    numConsts,
    reverse,
  };

  try {
    const prepared = webgpuBackend.prepareBatchedScan(batchedScanParams);
    if (prepared) {
      if (DEBUG >= 1)
        console.log(
          `Batched scan: SUCCESS! Using WebGPU batched scan for ${bodyRoutine.name}`,
        );
    }
    return prepared;
  } catch (e) {
    if (DEBUG >= 2) {
      console.warn("Batched scan preparation failed:", e);
    }
    return null;
  }
}

/**
 * Try to prepare a native scan for WASM/WebGPU backends.
 * This is the unified implementation that handles all scan body types:
 * - Single kernel bodies (like cumsum)
 * - Multiple independent kernels (like Kalman filter with 2 matmuls)
 * - Bodies with data dependencies between steps
 * - Bodies where numCarry !== numY
 * - Routine steps (Cholesky, Sort) embedded in the scan loop
 */
function tryPrepareNativeScanGeneral(
  backend: Backend,
  bodyProgram: JitProgram,
  bodyJaxpr: Jaxpr,
  length: number,
  numCarry: number,
  numConsts: number,
  numX: number,
  numY: number,
  reverse: boolean,
): {
  executable: Executable;
  internalSizes: number[];
  params: NativeScanGeneralParams;
} | null {
  // Only WASM backend for now
  if (backend.type !== "wasm") {
    if (DEBUG >= 1)
      console.log(
        `[general-scan] skipped, backend=${backend.type} (need wasm)`,
      );
    return null;
  }

  if (DEBUG >= 2)
    console.log(
      `[general-scan] trying with backend=${backend.type}, numCarry=${numCarry}, numY=${numY}`,
    );

  // Get all execute steps
  const executeSteps = bodyProgram.steps.filter(
    (s) => s.type === "execute",
  ) as Extract<JitStep, { type: "execute" }>[];
  if (executeSteps.length === 0) {
    if (DEBUG >= 1) console.log("[general-scan] skipped, no execute steps");
    return null;
  }

  // Check which routines are used and if they're supported
  const usedRoutines = new Set<Routines>();
  for (const step of executeSteps) {
    if (step.source instanceof Routine) {
      // Native scan no longer supports routines (they use pre-compiled AS modules)
      // Fall back to JS loop for any routine in scan body
      if (DEBUG >= 1)
        console.log(
          `[general-scan] skipped, routines use pre-compiled AS modules: ${step.source.name}`,
        );
      return null;
    }
  }

  if (DEBUG >= 1) {
    const routineNames = [...usedRoutines].map((r) => Routines[r]);
    console.log(
      `[general-scan] Analyzing body: ${executeSteps.length} execute steps, numCarry=${numCarry}, numY=${numY}` +
        (routineNames.length > 0
          ? `, routines: ${routineNames.join(", ")}`
          : ""),
    );
  }

  // Number of jaxpr inputs
  const numInputs = numConsts + numCarry + numX;

  // Build internal buffer mapping: each execute step writes to an internal buffer
  // The internal buffer index = step index
  const numInternal = executeSteps.length;

  // Get avals for sizes and strides
  const constAvals = bodyJaxpr.inBinders.slice(0, numConsts).map((v) => v.aval);
  const carryAvals = bodyJaxpr.inBinders
    .slice(numConsts, numConsts + numCarry)
    .map((v) => v.aval);
  const xAvals = bodyJaxpr.inBinders
    .slice(numConsts + numCarry, numConsts + numCarry + numX)
    .map((v) => v.aval);
  const yAvals = bodyJaxpr.outs.slice(numCarry).map((v) => v.aval);

  const constSizes = constAvals.map((a) => a.size * byteWidth(a.dtype));
  const carrySizes = carryAvals.map((a) => a.size * byteWidth(a.dtype));
  const xsStrides = xAvals.map((a) => a.size * byteWidth(a.dtype));
  const ysStrides = yAvals.map((a) => a.size * byteWidth(a.dtype));

  // Build a mapping from JitId (output slot) to internal buffer index
  const slotToInternal = new Map<JitId, number>();
  for (let i = 0; i < executeSteps.length; i++) {
    const step = executeSteps[i];
    for (const outSlot of step.outputs) {
      slotToInternal.set(outSlot, i);
    }
  }

  // Build internal sizes from each step's output
  const internalSizes: number[] = [];
  for (let i = 0; i < executeSteps.length; i++) {
    const source = executeSteps[i].source;
    if (source instanceof Kernel) {
      internalSizes.push(source.size * byteWidth(source.dtype));
    } else {
      // Routine: use first output shape
      const routine = source as Routine;
      const outShape = routine.type.outputShapes[0];
      const outDtype = routine.type.outputDtypes[0];
      internalSizes.push(prod(outShape) * byteWidth(outDtype));
    }
  }

  // Calculate aux buffer size for routines that need it
  let auxBufferSize = 0;
  let elementSize: 4 | 8 = 4;
  for (const step of executeSteps) {
    if (step.source instanceof Routine) {
      const routine = step.source;
      const dtype = routine.type.inputDtypes[0];
      elementSize = byteWidth(dtype) as 4 | 8;
      if (routine.name === Routines.Sort) {
        // Sort needs aux buffer of size sortDim * elementSize
        const inputShape = routine.type.inputShapes[0];
        const sortDim = inputShape[inputShape.length - 1];
        auxBufferSize = Math.max(auxBufferSize, sortDim * elementSize);
      }
    }
  }

  // Build input slot mapping for each step
  // Each step's source has inputs that reference jaxpr inputs or internal buffers
  // - [0, numConsts): constant
  // - [numConsts, numConsts+numCarry): carry
  // - [numConsts+numCarry, numInputs): xs
  // - [numInputs, ...): internal buffer from previous step
  type LocalGeneralScanStep = {
    source: Kernel | Routine;
    inputSlots: number[];
    outputInternalIdx: number;
    outputInternalIndices?: number[];
  };

  const steps: LocalGeneralScanStep[] = [];
  for (let i = 0; i < executeSteps.length; i++) {
    const step = executeSteps[i];
    const source = step.source;

    // step.inputs are JitIds that the source reads from
    // We need to classify each: is it a jaxpr input or an internal buffer?
    const inputSlots: number[] = [];
    for (const inputId of step.inputs) {
      if (inputId < numInputs) {
        // It's a jaxpr input (const, carry, or xs)
        inputSlots.push(inputId);
      } else {
        // It's an internal buffer - find which step produced it
        const internalIdx = slotToInternal.get(inputId);
        if (internalIdx === undefined) {
          if (DEBUG >= 1)
            console.log(
              `[general-scan] skipped, input ${inputId} not found in slot mapping`,
            );
          return null;
        }
        // Internal buffers are indexed after jaxpr inputs
        inputSlots.push(numInputs + internalIdx);
      }
    }

    if (source instanceof Kernel) {
      // Reindex kernel gids to use our inputSlots mapping
      const reindexMap = inputSlots;
      const reindexedExp = source.exp.reindexGids(reindexMap);
      // Also reindex the reduction's epilogue if present
      const reindexedReduction = source.reduction?.reindexGids(reindexMap);
      const reindexedKernel = new Kernel(
        numInputs + numInternal, // nargs: can read from jaxpr inputs + all internals
        source.size,
        reindexedExp,
        reindexedReduction,
      );

      steps.push({
        source: reindexedKernel,
        inputSlots,
        outputInternalIdx: i,
      });
    } else {
      // Routine: no reindexing needed, just pass through
      steps.push({
        source,
        inputSlots,
        outputInternalIdx: i,
      });
    }
  }

  // Find carry output sources: which internal buffer provides each carry output
  // Also handle passthrough (carry input returned as carry output unchanged)
  const carryOutSlots = bodyProgram.outputs.slice(0, numCarry);
  const carryInputSlots = bodyProgram.inputs.slice(
    numConsts,
    numConsts + numCarry,
  );

  type CarryOutputSource = {
    type: "passthrough" | "internal";
    carryIdx?: number;
    internalIdx?: number;
  };
  const carryOutSources: CarryOutputSource[] = [];
  for (const slot of carryOutSlots) {
    // Check if it's a passthrough from carry input
    const carryIdx = carryInputSlots.indexOf(slot);
    if (carryIdx !== -1) {
      carryOutSources.push({ type: "passthrough", carryIdx });
      continue;
    }
    // Otherwise it should be from an internal buffer
    const internalIdx = slotToInternal.get(slot);
    if (internalIdx === undefined) {
      if (DEBUG >= 1)
        console.log(
          `[general-scan] skipped, carry output slot ${slot} not produced by any execute step`,
        );
      return null;
    }
    carryOutSources.push({ type: "internal", internalIdx });
  }

  // Find Y output sources: either passthrough from carry input or from internal buffer
  type YOutputSource = {
    type: "passthrough" | "internal";
    carryIdx?: number;
    internalIdx?: number;
  };

  const yOutputSlots = bodyProgram.outputs.slice(numCarry);
  const yOutputSources: YOutputSource[] = [];

  for (const slot of yOutputSlots) {
    // Check if it's a passthrough from carry input
    const carryIdx = carryInputSlots.indexOf(slot);
    if (carryIdx !== -1) {
      yOutputSources.push({ type: "passthrough", carryIdx });
      continue;
    }

    // Otherwise it should be from an internal buffer
    const internalIdx = slotToInternal.get(slot);
    if (internalIdx === undefined) {
      if (DEBUG >= 1)
        console.log(`[general-scan] skipped, Y output slot ${slot} not found`);
      return null;
    }
    yOutputSources.push({ type: "internal", internalIdx });
  }

  // Try to prepare general native scan
  const nativeBackend = backend as any;
  if (typeof nativeBackend.prepareNativeScanGeneral !== "function") {
    if (DEBUG >= 2)
      console.log("[general-scan] backend has no prepareNativeScanGeneral");
    return null;
  }

  const params = {
    length,
    numConsts,
    constSizes,
    numCarry,
    carrySizes,
    numX,
    xsStrides,
    numY,
    ysStrides,
    internalSizes,
    steps,
    carryOutSources,
    yOutputSources,
    reverse,
    auxBufferSize,
    elementSize,
  };

  try {
    const exe = nativeBackend.prepareNativeScanGeneral(params);
    if (exe) {
      if (DEBUG >= 1) {
        const hasRoutines = steps.some((s) => s.source instanceof Routine);
        console.log(
          `[general-scan] SUCCESS! Using ${backend.type.toUpperCase()} general native scan with ${steps.length} steps` +
            (hasRoutines ? " (includes routines)" : ""),
        );
      }
      return { executable: exe, internalSizes, params };
    }
    return null;
  } catch (e) {
    if (DEBUG >= 2) {
      console.warn("[general-scan] preparation failed:", e);
    }
    return null;
  }
}
