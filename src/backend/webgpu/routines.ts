// WebGPU implementations of Routines (sort, argsort, cholesky, etc.)

import { dtypeToWgsl, headerWgsl, maxValueWgsl, ShaderInfo } from "./codegen";
import { DType, isFloatDtype } from "../../alu";
import { UnsupportedRoutineError } from "../../backend";
import { Routine, Routines, RoutineType } from "../../routine";
import { prod } from "../../utils";

/**
 * Generate a single-dispatch bitonic sort shader using workgroup shared memory.
 *
 * Each workgroup sorts one batch independently using shared memory and barriers.
 * One thread per element in the padded array.
 */
function bitonicSortShader(
  device: GPUDevice,
  dtype: DType,
  n: number,
  batches: number,
  outputIndices: boolean,
): ShaderInfo[] {
  const ty = dtypeToWgsl(dtype, true);
  // Round up to next power of 2 for bitonic sort (need paddedN >= n).
  const paddedN = 1 << Math.ceil(Math.log2(n || 1));

  if (paddedN > device.limits.maxComputeWorkgroupSizeX) {
    // TODO: Multi-pass, global memory sorting for large arrays (radix sort?).
    throw new Error(
      `sort: array size ${n} (padded to ${paddedN}) exceeds device limit of ${device.limits.maxComputeWorkgroupSizeX}.`,
    );
  }

  const numStages = Math.ceil(Math.log2(paddedN));
  const needsF16 = dtype === DType.Float16;

  const shader = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> input: array<${ty}>;
@group(0) @binding(1) var<storage, read_write> output: array<${outputIndices ? "i32" : ty}>;

var<workgroup> shared_vals: array<${ty}, ${paddedN}>;
${outputIndices ? `var<workgroup> shared_idx: array<i32, ${paddedN}>;` : ""}

fn compare(a: ${ty}, b: ${ty}) -> bool {
${
  isFloatDtype(dtype)
    ? `  // Roundabout way to handle NaNs, they sort to end
  let min_value = min(a, b);
  return a == min_value && b != min_value;
`
    : "  return a < b;"
}
}

@compute @workgroup_size(${paddedN})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let batch = wg_id.x;
  if (batch >= ${batches}u) { return; }
  let tid = local_id.x;
  let base = batch * ${n}u;

  // Load data into shared memory (1 element per thread)
  shared_vals[tid] = select(${isFloatDtype(dtype) ? `${ty}(nan())` : maxValueWgsl(dtype)}, input[base + min(tid, ${n - 1}u)], tid < ${n}u);
  ${outputIndices ? `shared_idx[tid] = select(${n}, i32(tid), tid < ${n}u);` : ""}
  workgroupBarrier();

  // Bitonic sort in shared memory
  for (var stage = 0u; stage < ${numStages}u; stage++) {
    for (var step = stage + 1u; step > 0u; step--) {
      let actual_step = step - 1u;
      let half_block = 1u << actual_step;

      // Compute partner index for this thread
      let ixj = tid ^ half_block;

      if (ixj > tid) {
        // Direction: ascending if in first half of merge block
        let ascending = ((tid >> (stage + 1u)) & 1u) == 0u;
        let val_tid = shared_vals[tid];
        let val_ixj = shared_vals[ixj];
        // Swap if out of order: ascending wants smaller at tid (lower index)
        let should_swap = select(compare(val_tid, val_ixj), compare(val_ixj, val_tid), ascending);
        if (should_swap) {
          shared_vals[tid] = val_ixj;
          shared_vals[ixj] = val_tid;
${
  outputIndices
    ? `
          let tmp_idx = shared_idx[tid];
          shared_idx[tid] = shared_idx[ixj];
          shared_idx[ixj] = tmp_idx;`
    : ""
}
        }
      }
      workgroupBarrier();
    }
  }

  // Write results back
  if (tid < ${n}u) { output[base + tid] = ${outputIndices ? "shared_idx[tid]" : "shared_vals[tid]"}; }
}
`.trim();

  return [
    {
      shader,
      grid: [batches, 1],
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
