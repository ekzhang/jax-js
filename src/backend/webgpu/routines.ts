// WebGPU implementations of Routines (sort, argsort, cholesky, etc.)

import {
  calculateGrid,
  dtypeToWgsl,
  gridOffsetY,
  headerWgsl,
  maxValueWgsl,
  ShaderInfo,
} from "./codegen";
import { DType, isFloatDtype } from "../../alu";
import { UnsupportedRoutineError } from "../../backend";
import { Routine, Routines, RoutineType } from "../../routine";
import { findPow2, prod } from "../../utils";

type BitonicSortPass = {
  kind: "sort" | "merge"; // sort = full sort (stages 0..k), merge is only merge steps
  mergeStep?: number; // half_block = 2^step, only used for 'merge'
  mergeStage?: number; // stage, only used for 'merge'
};

function bitonicSortUniform(pass: BitonicSortPass): Uint8Array<ArrayBuffer> {
  const ar = new Uint32Array(3);
  ar[0] = pass.kind === "sort" ? 0 : 1;
  ar[1] = pass.mergeStep ?? 0;
  ar[2] = pass.mergeStage ?? 0;
  return new Uint8Array(ar.buffer);
}

/**
 * Generate a bitonic sort shader.
 *
 * We implement a variant of bitonic sort that [only has forward comparators](
 * <https://sortingalgos.miraheze.org/wiki/Bitonic_Sort#Bitonic_Sort_using_Forward_Comparators>),
 * so we don't need to allocate memory for power-of-two padding.
 *
 * This uses workgroup shared memory up to `2*workgroupSize` elements, for each
 * array in `batches`. For larger arrays, multiple passes are done:
 *
 * - Initial "sort" pass: each workgroup sorts its `2*workgroupSize` elements.
 * - Subsequent "merge" passes: each pass merges sorted sequences of size
 *   `2^(step+1)` with multiple workgroups. This doesn't use shared memory.
 *
 * The total number of passes is roughly `log2(n / workgroupSize)^2 / 2`.
 */
function bitonicSortShader(
  device: GPUDevice,
  dtype: DType,
  n: number,
  batches: number,
  outputIndices: boolean,
): ShaderInfo[] {
  const ty = dtypeToWgsl(dtype, true);
  const paddedN = 1 << Math.ceil(Math.log2(n || 1));
  const numThreads = Math.ceil(paddedN / 2); // 2 elements per thread

  // If this is less than numThreads, we need to do multiple dispatches.
  const workgroupSize = findPow2(
    numThreads,
    device.limits.maxComputeWorkgroupSizeX,
  );
  const workgroupsPerBatch = numThreads / workgroupSize;
  const numStages = Math.log2(paddedN);
  const numLocalStages = Math.min(numStages, Math.log2(workgroupSize * 2));

  const needsF16 = dtype === DType.Float16;
  const padValue = isFloatDtype(dtype) ? `${ty}(nan())` : maxValueWgsl(dtype);

  const code = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

struct Uniforms {
  kind: u32, // 0 = sort, 1 = merge
  merge_step: u32, // half_block = 2^step
  merge_stage: u32, // only used for merge
}

@group(0) @binding(0) var<storage, read> input: array<${ty}>;
@group(0) @binding(1) var<storage, read_write> output: array<${ty}>;
${outputIndices ? `@group(0) @binding(2) var<storage, read_write> output_idx: array<i32>;` : ""}

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

var<workgroup> shared_vals: array<${ty}, ${workgroupSize * 2}>;
${outputIndices ? `var<workgroup> shared_idx: array<i32, ${workgroupSize * 2}>;` : ""}

fn compare(a: ${ty}, b: ${ty}) -> bool {
${
  // Roundabout way to handle NaNs, they sort to end
  isFloatDtype(dtype)
    ? `
  let min_value = min(a, b);
  return a == min_value && b != min_value;`
    : "  return a < b;"
}
}

fn compare_and_swap(i: u32, j: u32) {
  let val_i = shared_vals[i];
  let val_j = shared_vals[j];
  if (compare(val_j, val_i)) {
    shared_vals[i] = val_j;
    shared_vals[j] = val_i;
${
  outputIndices
    ? `
    let tmp_idx = shared_idx[i];
    shared_idx[i] = shared_idx[j];
    shared_idx[j] = tmp_idx;`
    : ""
}
  }
}

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let blockid = wg_id.x + wg_id.y * ${gridOffsetY}u;
  let batch = blockid / ${workgroupsPerBatch}u;
  let wg_in_batch = blockid % ${workgroupsPerBatch}u;

  let tid = local_id.x;
  let base = batch * ${n}u;

  if (uniforms.kind == 0u || (uniforms.kind == 1u && uniforms.merge_step == ${numLocalStages - 1}u)) {
    let wg_base = wg_in_batch * ${workgroupSize * 2}u;

    // Load data into shared memory (2 elements per thread)
    let idx0 = tid * 2u;
    let idx1 = tid * 2u + 1u;
    // Load from input for initial 'sort' pass, then from output (read-write) for 'merge' passes.
    if (uniforms.kind == 0u) {
      shared_vals[idx0] = select(${padValue}, input[base + wg_base + idx0], wg_base + idx0 < ${n}u);
      shared_vals[idx1] = select(${padValue}, input[base + wg_base + idx1], wg_base + idx1 < ${n}u);
${
  outputIndices
    ? `
      shared_idx[idx0] = i32(wg_base + idx0);
      shared_idx[idx1] = i32(wg_base + idx1);`
    : ""
}
    } else {
      shared_vals[idx0] = select(${padValue}, output[base + wg_base + idx0], wg_base + idx0 < ${n}u);
      shared_vals[idx1] = select(${padValue}, output[base + wg_base + idx1], wg_base + idx1 < ${n}u);
${
  outputIndices
    ? `
      shared_idx[idx0] = select(${n}, output_idx[base + wg_base + idx0], wg_base + idx0 < ${n}u);
      shared_idx[idx1] = select(${n}, output_idx[base + wg_base + idx1], wg_base + idx1 < ${n}u);`
    : ""
}
    }
    workgroupBarrier();

    let initial_stage = select(0u, ${numLocalStages - 1}u, uniforms.kind != 0u);
    for (var stage = initial_stage; stage < ${numLocalStages}u; stage++) {
      for (var step1 = stage + 1u; step1 > 0u; step1--) {
        let step = step1 - 1u;
        let half_block = 1u << step;
        let is_first_step = uniforms.kind == 0u && step == stage;

        let block_offset = (tid / half_block) * half_block;
        let local_offset = tid % half_block;
        let i = block_offset * 2u + local_offset;
        let j = select(i + half_block, i ^ (half_block * 2u - 1u), is_first_step);
        compare_and_swap(i, j);

        workgroupBarrier();
      }
    }

    if (wg_base + idx0 < ${n}u) {
      output[base + wg_base + idx0] = shared_vals[idx0];
      ${outputIndices ? `output_idx[base + wg_base + idx0] = shared_idx[idx0];` : ""}
    }
    if (wg_base + idx1 < ${n}u) {
      output[base + wg_base + idx1] = shared_vals[idx1];
      ${outputIndices ? `output_idx[base + wg_base + idx1] = shared_idx[idx1];` : ""}
    }
  } else {
    // Execute single merge pass for a step >= numLocalStages.
    let half_block = 1u << uniforms.merge_step;  // half_block >= workgroupSize * 2
    let thread_in_batch = wg_in_batch * ${workgroupSize} + tid;
    let is_first_step = uniforms.merge_step == uniforms.merge_stage;

    let block_offset = (thread_in_batch / half_block) * half_block;
    let local_offset = thread_in_batch % half_block;
    let i = block_offset * 2u + local_offset;
    let j = select(i + half_block, i ^ (half_block * 2u - 1u), is_first_step);

    // Global version of compare_and_swap()
    if (j < ${n}u) {
      let val_i = output[base + i];
      let val_j = output[base + j];
      if (compare(val_j, val_i)) {
        output[base + i] = val_j;
        output[base + j] = val_i;
${
  outputIndices
    ? `
        let tmp_idx = output_idx[base + i];
        output_idx[base + i] = output_idx[base + j];
        output_idx[base + j] = tmp_idx;`
    : ""
}
      }
    }
  }
}
`.trim();

  const grid = calculateGrid(batches * workgroupsPerBatch);
  const passes: BitonicSortPass[] = [{ kind: "sort" }];
  for (let mergeStage = numLocalStages; mergeStage < numStages; mergeStage++) {
    for (
      let mergeStep = mergeStage;
      mergeStep >= numLocalStages - 1;
      mergeStep--
    ) {
      passes.push({ kind: "merge", mergeStep, mergeStage });
    }
  }

  return [
    {
      code,
      numInputs: 1,
      numOutputs: outputIndices ? 2 : 1,
      hasUniform: true,
      passes: passes.map((pass) => ({
        grid,
        uniform: bitonicSortUniform(pass),
      })),
    },
  ];
}

function createSort(device: GPUDevice, type: RoutineType): ShaderInfo[] {
  const dtype = type.inputDtypes[0];
  const shape = type.inputShapes[0];
  const n = shape[shape.length - 1];
  const batches = prod(shape.slice(0, -1));
  return bitonicSortShader(device, dtype, n, batches, false);
}

function createArgsort(device: GPUDevice, type: RoutineType): ShaderInfo[] {
  const dtype = type.inputDtypes[0];
  const shape = type.inputShapes[0];
  const n = shape[shape.length - 1];
  const batches = prod(shape.slice(0, -1));
  return bitonicSortShader(device, dtype, n, batches, true);
}

export function createRoutineShader(
  device: GPUDevice,
  routine: Routine,
): ShaderInfo[] {
  switch (routine.name) {
    case Routines.Sort:
      return createSort(device, routine.type);
    case Routines.Argsort:
      return createArgsort(device, routine.type);
    default:
      throw new UnsupportedRoutineError(routine.name, "webgpu");
  }
}
