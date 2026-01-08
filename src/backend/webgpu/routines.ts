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

/**
 * Generate a triangular solve shader.
 *
 * Solves A @ X.T = B.T for X, where A is upper-triangular.
 * Uses a parallelized back-substitution:
 *   1. Copy b to x
 *   2. For j = n-1 down to 0:
 *      - Divide x[j] by a[j,j] (single thread)
 *      - All threads subtract x[j] * a[i,j] from x[i] for i < j in parallel
 */
function createTriangularSolve(
  device: GPUDevice,
  type: RoutineType,
  params: { unitDiagonal: boolean },
): ShaderInfo[] {
  const dtype = type.inputDtypes[0];
  const aShape = type.inputShapes[0]; // [..., n, n]
  const bShape = type.inputShapes[1]; // [..., batch, n]

  const n = aShape[aShape.length - 1]; // Matrix dimension
  const numRhs = bShape[bShape.length - 2]; // Number of RHS vectors per matrix
  const numMatrices = prod(aShape.slice(0, -2)); // Number of matrices in batch

  const needsF16 = dtype === DType.Float16;
  const ty = dtypeToWgsl(dtype, true);

  // Each workgroup handles one (matrix, rhs) pair
  const workgroupSize = findPow2(n, device.limits.maxComputeWorkgroupSizeX);

  const code = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> a: array<${ty}>;
@group(0) @binding(1) var<storage, read> b: array<${ty}>;
@group(0) @binding(2) var<storage, read_write> x: array<${ty}>;

// Shared memory for the current pivot value x[j]
var<workgroup> x_j: ${ty};

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let wg_idx = wg_id.x + wg_id.y * ${gridOffsetY}u;
  let mat_idx = wg_idx / ${numRhs}u;
  let rhs_idx = wg_idx % ${numRhs}u;

  if (mat_idx >= ${numMatrices}u) {
    return;
  }

  let a_base = mat_idx * ${n * n}u;
  let bx_base = (mat_idx * ${numRhs}u + rhs_idx) * ${n}u;
  let tid = local_id.x;

  // Step 1: Copy b to x (threads collaborate)
  for (var idx = tid; idx < ${n}u; idx += ${workgroupSize}u) {
    x[bx_base + idx] = b[bx_base + idx];
  }
  storageBarrier();

  // Step 2: Back-substitution from j = n-1 down to 0
  for (var jj = 0u; jj < ${n}u; jj++) {
    let j = ${n - 1}u - jj;

    // Thread 0 computes x[j] = x[j] / a[j,j]
    if (tid == 0u) {
      ${params.unitDiagonal ? `x_j = x[bx_base + j];` : `x_j = x[bx_base + j] / a[a_base + j * ${n}u + j];`}
      x[bx_base + j] = x_j;
    }
    workgroupBarrier();  // Sync shared memory x_j

    // All threads subtract x[j] * a[i,j] from x[i] for i < j
    for (var i = tid; i < j; i += ${workgroupSize}u) {
      x[bx_base + i] -= x_j * a[a_base + i * ${n}u + j];
    }
    workgroupBarrier();
    storageBarrier();
  }
}
`.trim();

  const totalWorkgroups = numMatrices * numRhs;
  const grid = calculateGrid(totalWorkgroups);
  return [
    {
      code,
      numInputs: 2,
      numOutputs: 1,
      hasUniform: false,
      passes: [{ grid }],
    },
  ];
}

/**
 * Generate a Cholesky decomposition shader.
 *
 * Computes the lower triangular matrix L such that A = L * L^T for each
 * positive semi-definite matrix in the batch. Uses the Cholesky-Crout
 * algorithm which processes column-by-column.
 *
 * For each column j:
 *   1. All threads compute their row's sum in parallel and store to output
 *   2. Thread 0 computes L[j][j] = sqrt(output[j][j]) and stores to shared memory
 *   3. All threads divide their output[i][j] by L[j][j] in parallel
 */
function createCholesky(device: GPUDevice, type: RoutineType): ShaderInfo[] {
  const dtype = type.inputDtypes[0];
  const shape = type.inputShapes[0];
  const n = shape[shape.length - 1]; // Matrix dimension (n x n)
  const batches = prod(shape.slice(0, -2)); // Number of matrices in batch

  const needsF16 = dtype === DType.Float16;
  const ty = dtypeToWgsl(dtype, true);

  // Use workgroup size to parallelize column computation
  const workgroupSize = findPow2(n, device.limits.maxComputeWorkgroupSizeX);

  const code = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> input: array<${ty}>;
@group(0) @binding(1) var<storage, read_write> output: array<${ty}>;

// Shared memory for the diagonal element
var<workgroup> L_jj: ${ty};

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let batch = wg_id.x + wg_id.y * ${gridOffsetY}u;
  if (batch >= ${batches}u) {
    return;
  }

  let base = batch * ${n * n}u;
  let tid = local_id.x;

  // Zero out output and copy lower triangle from input (threads collaborate)
  for (var idx = tid; idx < ${n * n}u; idx += ${workgroupSize}u) {
    let row = idx / ${n}u;
    let col = idx % ${n}u;
    output[base + idx] = select(0, input[base + idx], col <= row);
  }
  storageBarrier();

  // Cholesky-Crout algorithm: process column by column
  for (var j = 0u; j < ${n}u; j++) {
    // Step 1: All threads compute sum for their rows i >= j in parallel
    // sum = A[i][j] - sum(L[i][k] * L[j][k] for k < j)
    for (var i = j + tid; i < ${n}u; i += ${workgroupSize}u) {
      var sum = output[base + i * ${n}u + j];
      for (var k = 0u; k < j; k++) {
        sum -= output[base + i * ${n}u + k] * output[base + j * ${n}u + k];
      }
      output[base + i * ${n}u + j] = sum;
    }
    storageBarrier();

    // Step 2: Thread 0 computes L[j][j] = sqrt(output[j][j])
    if (tid == 0u) {
      L_jj = sqrt(output[base + j * ${n}u + j]);
      output[base + j * ${n}u + j] = L_jj;
    }
    workgroupBarrier();

    // Step 3: All threads divide output[i][j] by L[j][j] for i > j
    for (var i = j + 1u + tid; i < ${n}u; i += ${workgroupSize}u) {
      output[base + i * ${n}u + j] /= L_jj;
    }
    storageBarrier();
  }
}
`.trim();

  const grid = calculateGrid(batches);
  return [
    {
      code,
      numInputs: 1,
      numOutputs: 1,
      hasUniform: false,
      passes: [{ grid }],
    },
  ];
}

/**
 * Generate an LU decomposition shader with partial pivoting.
 *
 * Computes PA = LU where P is a permutation matrix, L is lower triangular
 * with unit diagonal, and U is upper triangular.
 *
 * For each column j:
 *   1. Find pivot row (max absolute value in column j, rows >= j)
 *   2. Swap rows j and pivot row
 *   3. Compute L[i][j] = A[i][j] / A[j][j] for i > j
 *   4. Update submatrix: A[i][k] -= L[i][j] * A[j][k] for i > j, k > j
 */
function createLU(device: GPUDevice, type: RoutineType): ShaderInfo[] {
  const dtype = type.inputDtypes[0];
  const shape = type.inputShapes[0];
  const m = shape[shape.length - 2]; // rows
  const n = shape[shape.length - 1]; // cols
  const r = Math.min(m, n);
  const batches = prod(shape.slice(0, -2));

  const needsF16 = dtype === DType.Float16;
  const ty = dtypeToWgsl(dtype, true);

  const workgroupSize = findPow2(
    Math.max(m, n),
    device.limits.maxComputeWorkgroupSizeX,
  );

  const code = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> input: array<${ty}>;
@group(0) @binding(1) var<storage, read_write> lu: array<${ty}>;
@group(0) @binding(2) var<storage, read_write> pivots: array<i32>;
@group(0) @binding(3) var<storage, read_write> perm: array<i32>;

var<workgroup> pivot_row: u32;
var<workgroup> pivot_val: ${ty};

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let batch = wg_id.x + wg_id.y * ${gridOffsetY}u;
  if (batch >= ${batches}u) {
    return;
  }

  let lu_base = batch * ${m * n}u;
  let piv_base = batch * ${r}u;
  let perm_base = batch * ${m}u;
  let tid = local_id.x;

  // Copy input to lu
  for (var idx = tid; idx < ${m * n}u; idx += ${workgroupSize}u) {
    lu[lu_base + idx] = input[lu_base + idx];
  }
  // Initialize permutation
  for (var idx = tid; idx < ${m}u; idx += ${workgroupSize}u) {
    perm[perm_base + idx] = i32(idx);
  }
  storageBarrier();

  // LU decomposition with partial pivoting
  for (var j = 0u; j < ${r}u; j++) {
    // Step 1: Thread 0 finds pivot (max abs value in column j, rows >= j)
    if (tid == 0u) {
      var max_val = abs(lu[lu_base + j * ${n}u + j]);
      var max_row = j;
      for (var i = j + 1u; i < ${m}u; i++) {
        let val = abs(lu[lu_base + i * ${n}u + j]);
        if (val > max_val) {
          max_val = val;
          max_row = i;
        }
      }
      pivot_row = max_row;
      pivot_val = lu[lu_base + max_row * ${n}u + j];
      pivots[piv_base + j] = i32(max_row);
    }
    workgroupBarrier();

    // Step 2: Swap rows j and pivot_row (threads collaborate)
    let pr = pivot_row;
    if (pr != j) {
      for (var col = tid; col < ${n}u; col += ${workgroupSize}u) {
        let tmp = lu[lu_base + j * ${n}u + col];
        lu[lu_base + j * ${n}u + col] = lu[lu_base + pr * ${n}u + col];
        lu[lu_base + pr * ${n}u + col] = tmp;
      }
      if (tid == 0u) {
        let tmp_p = perm[perm_base + j];
        perm[perm_base + j] = perm[perm_base + pr];
        perm[perm_base + pr] = tmp_p;
      }
    }
    storageBarrier();

    // Step 3: Compute L[i][j] and update submatrix
    // Each thread handles one row i > j
    for (var i = j + 1u + tid; i < ${m}u; i += ${workgroupSize}u) {
      let factor = lu[lu_base + i * ${n}u + j] / pivot_val;
      lu[lu_base + i * ${n}u + j] = factor; // L[i][j]
      for (var k = j + 1u; k < ${n}u; k++) {
        lu[lu_base + i * ${n}u + k] -= factor * lu[lu_base + j * ${n}u + k];
      }
    }
    storageBarrier();
  }
}
`.trim();

  const grid = calculateGrid(batches);
  return [
    {
      code,
      numInputs: 1,
      numOutputs: 3,
      hasUniform: false,
      passes: [{ grid }],
    },
  ];
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
    case Routines.TriangularSolve:
      return createTriangularSolve(device, routine.type, routine.params);
    case Routines.Cholesky:
      return createCholesky(device, routine.type);
    case Routines.LU:
      return createLU(device, routine.type);
    default:
      throw new UnsupportedRoutineError(routine.name, "webgpu");
  }
}
