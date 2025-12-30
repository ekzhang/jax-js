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
import { prod } from "../../utils";

/**
 * Generate a single-dispatch bitonic sort shader using workgroup shared memory.
 *
 * Each workgroup sorts one batch independently using shared memory and barriers.
 * Each thread handles 2 elements, allowing arrays up to 2x the workgroup size limit.
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

  if (numThreads > device.limits.maxComputeWorkgroupSizeX) {
    // TODO: Multi-pass, global memory sorting for large arrays (radix sort?).
    throw new Error(
      `sort: array size ${n} (padded to ${paddedN}) exceeds device limit of ${device.limits.maxComputeWorkgroupSizeX * 2}.`,
    );
  }

  const numStages = Math.ceil(Math.log2(paddedN));
  const needsF16 = dtype === DType.Float16;
  const padValue = isFloatDtype(dtype) ? `${ty}(nan())` : maxValueWgsl(dtype);

  const shader = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> input: array<${ty}>;
@group(0) @binding(1) var<storage, read_write> output: array<${outputIndices ? "i32" : ty}>;

var<workgroup> shared_vals: array<${ty}, ${paddedN}>;
${outputIndices ? `var<workgroup> shared_idx: array<i32, ${paddedN}>;` : ""}

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

fn compare_and_swap(i: u32, j: u32, ascending: bool) {
  let val_i = shared_vals[i];
  let val_j = shared_vals[j];
  // Swap if out of order: ascending wants smaller value at lower index
  let should_swap = select(compare(val_i, val_j), compare(val_j, val_i), ascending);
  if (should_swap) {
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

@compute @workgroup_size(${numThreads})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let batch = wg_id.x + wg_id.y * ${gridOffsetY}u;
  if (batch >= ${batches}u) { return; }
  let tid = local_id.x;
  let base = batch * ${n}u;

  // Load data into shared memory (2 elements per thread)
  let idx0 = tid * 2u;
  let idx1 = tid * 2u + 1u;
  shared_vals[idx0] = select(${padValue}, input[base + min(idx0, ${n - 1}u)], idx0 < ${n}u);
  shared_vals[idx1] = select(${padValue}, input[base + min(idx1, ${n - 1}u)], idx1 < ${n}u);
${
  outputIndices
    ? `
  shared_idx[idx0] = i32(idx0);
  shared_idx[idx1] = i32(idx1);`
    : ""
}
  workgroupBarrier();

  for (var stage = 0u; stage < ${numStages}u; stage++) {
    for (var step = stage + 1u; step > 0u; step--) {
      let actual_step = step - 1u;
      let half_block = 1u << actual_step;

      let block_offset = (tid / half_block) * half_block;
      let local_offset = tid % half_block;
      let i = block_offset * 2u + local_offset;
      let j = i + half_block;

      // Direction: ascending if in first half of merge block for this stage
      let ascending = ((i >> (stage + 1u)) & 1u) == 0u;
      compare_and_swap(i, j, ascending);

      workgroupBarrier(); // log^2(n) / 2 barriers total
    }
  }

  if (idx0 < ${n}u) {
    output[base + idx0] = ${outputIndices ? "shared_idx" : "shared_vals"}[idx0];
  }
  if (idx1 < ${n}u) {
    output[base + idx1] = ${outputIndices ? "shared_idx" : "shared_vals"}[idx1];
  }
}
`.trim();

  return [
    {
      shader,
      passes: [{ grid: calculateGrid(batches) }],
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
