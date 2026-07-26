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
import {
  Routine,
  Routines,
  RoutineType,
  ScatterOp,
  type ScatterParams,
} from "../../routine";
import { View } from "../../shape";
import { findPow2, prod, range } from "../../utils";

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
 *
 * If `outputIndices` is true, the shader also tracks the original indices of
 * the sorted elements (argsort) and outputs them to a separate buffer. This
 * also makes the sorting algorithm stable.
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
${
  outputIndices
    ? `
  if (
    compare(val_j, val_i) ||
    (!compare(val_i, val_j) && shared_idx[j] < shared_idx[i])
  ) {
    shared_vals[i] = val_j;
    shared_vals[j] = val_i;
    let tmp_idx = shared_idx[i];
    shared_idx[i] = shared_idx[j];
    shared_idx[j] = tmp_idx;
  }`
    : `
  if (compare(val_j, val_i)) {
    shared_vals[i] = val_j;
    shared_vals[j] = val_i;
  }`
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
${
  outputIndices
    ? `
      let idx_i = output_idx[base + i];
      let idx_j = output_idx[base + j];
      if (compare(val_j, val_i) || (!compare(val_i, val_j) && idx_j < idx_i)) {
        output[base + i] = val_j;
        output[base + j] = val_i;
        output_idx[base + i] = idx_j;
        output_idx[base + j] = idx_i;`
    : `
      if (compare(val_j, val_i)) {
        output[base + i] = val_j;
        output[base + j] = val_i;`
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

function wgslCoordinate(linearIndex: string, view: View, axis: number): string {
  if (view.shape[axis] <= 1) return "0u";
  const stride = view.strides[axis];
  const divided = stride === 1 ? linearIndex : `(${linearIndex} / ${stride}u)`;
  return `(${divided} % ${view.shape[axis]}u)`;
}

function wgslBroadcastIndex(
  linearIndex: string,
  updateView: View,
  indexView: View,
  outDim: number,
  indexRank: number,
): string {
  const firstUpdateDim = outDim + indexRank - indexView.ndim;
  const terms: string[] = [];
  for (let i = 0; i < indexView.ndim; i++) {
    const stride = indexView.strides[i];
    if (stride === 0) continue;
    const coord = wgslCoordinate(linearIndex, updateView, firstUpdateDim + i);
    terms.push(stride === 1 ? coord : `(${coord} * ${stride}u)`);
  }
  return terms.length === 0 ? "0u" : terms.join(" + ");
}

type ScatterUpdateSource = {
  statement: string;
  helpers: string;
};

/**
 * Generate the update statement for scatter-add.
 *
 * WGSL only has native integer atomics. Float32 uses a compare-exchange loop,
 * while float16 updates one lane of a packed u32 with the same technique.
 */
function scatterAddWgsl(dtype: DType): ScatterUpdateSource {
  switch (dtype) {
    case DType.Float32:
      return {
        statement: "atomic_add_f32(&output[output_index], updates[global]);",
        helpers: `
fn atomic_add_f32(address: ptr<storage, atomic<u32>, read_write>, value: f32) {
  var old_bits = atomicLoad(address);
  loop {
    let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + value);
    let result = atomicCompareExchangeWeak(address, old_bits, new_bits);
    if (result.exchanged) {
      break;
    }
    old_bits = result.old_value;
  }
}`,
      };
    case DType.Float16:
      return {
        statement: `
  atomic_add_f16(
    &output[output_index / 2u],
    output_index % 2u != 0u,
    updates[global],
  );`.trim(),
        helpers: `
fn atomic_add_f16(
  address: ptr<storage, atomic<u32>, read_write>,
  high: bool,
  value: f16,
) {
  var old_bits = atomicLoad(address);
  loop {
    var pair = unpack2x16float(old_bits);
    if (high) {
      pair.y += f32(value);
    } else {
      pair.x += f32(value);
    }
    let new_bits = pack2x16float(pair);
    let result = atomicCompareExchangeWeak(address, old_bits, new_bits);
    if (result.exchanged) {
      break;
    }
    old_bits = result.old_value;
  }
}`,
      };
    case DType.Uint32:
    case DType.Int32:
      return {
        statement: "atomicAdd(&output[output_index], updates[global]);",
        helpers: "",
      };
    case DType.Bool:
      return {
        statement: "atomicOr(&output[output_index], updates[global]);",
        helpers: "",
      };
    default:
      throw new Error(`Unsupported atomic Scatter dtype for WebGPU: ${dtype}`);
  }
}

function createScatter(
  device: GPUDevice,
  type: RoutineType,
  {
    op,
    shape: outputShape,
    axis: indexedAxes,
    outDim,
    uniqueIndices,
  }: ScatterParams,
): ShaderInfo[] {
  const dtype = type.inputDtypes[0];
  const updateView = View.create(type.inputShapes[0]);
  const indexViews = type.inputShapes
    .slice(1)
    .map((shape) => View.create(shape));
  const indexRank = Math.max(...indexViews.map((view) => view.ndim));
  const updateCount = updateView.size;
  const elementType = dtypeToWgsl(dtype, true);
  const needsF16 = dtype === DType.Float16;
  if (needsF16 && !device.features.has("shader-f16")) {
    throw new Error("WebGPU device does not support shader-f16 feature");
  }

  // Updates have the free output dimensions, with the broadcasted index
  // dimensions inserted at outDim.
  const freeUpdateDims = [
    ...range(outDim),
    ...range(outDim + indexRank, updateView.ndim),
  ];

  // Each index input is right-aligned within the broadcasted index dimensions.
  const indexLoads = indexViews.map((indexView, i) => {
    const indexOffset = wgslBroadcastIndex(
      "global",
      updateView,
      indexView,
      outDim,
      indexRank,
    );
    return `  let scatter_index_${i} = i32(index_${i}[${indexOffset}]);`;
  });
  const bounds = indexedAxes.map(
    (outputAxis, i) =>
      `scatter_index_${i} < 0 || scatter_index_${i} >= ${outputShape[outputAxis]}`,
  );

  const outputView = View.create(outputShape);
  // Indexed output axes come from index buffers; all others come directly from
  // the corresponding update coordinate.
  let freeDim = 0;
  const outputTerms = range(outputShape.length).map((outputAxis) => {
    const indexInput = indexedAxes.indexOf(outputAxis);
    let coord: string;
    if (indexInput === -1) {
      coord = wgslCoordinate("global", updateView, freeUpdateDims[freeDim++]);
    } else {
      coord = `u32(scatter_index_${indexInput})`;
    }
    return outputView.strides[outputAxis] === 1
      ? coord
      : `(${coord} * ${outputView.strides[outputAxis]}u)`;
  });
  const outputIndex = outputTerms.join(" + ");

  const atomicAdd = op === ScatterOp.Add && !uniqueIndices;
  const usesSignedAtomics = dtype === DType.Int32 || dtype === DType.Bool;
  const outputStorageType = atomicAdd
    ? `atomic<${usesSignedAtomics ? "i32" : "u32"}>`
    : elementType;
  const updateSource = atomicAdd
    ? scatterAddWgsl(dtype)
    : {
        statement: "output[output_index] = updates[global];",
        helpers: "",
      };

  const maxThreads = device.limits.maxComputeWorkgroupSizeX;
  const dispatchSize = Math.max(updateCount, 1);
  const workgroupSize = findPow2(
    Math.min(dispatchSize, maxThreads),
    maxThreads,
  );
  const code = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}
${updateSource.helpers}

@group(0) @binding(0) var<storage, read> updates: array<${elementType}>;
${indexViews
  .map(
    (_, i) =>
      `@group(0) @binding(${i + 1}) var<storage, read> index_${i}: array<${dtypeToWgsl(type.inputDtypes[i + 1], true)}>;`,
  )
  .join("\n")}
@group(0) @binding(${type.inputShapes.length}) var<storage, read_write> output: array<${outputStorageType}>;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let global = global_id.x + global_id.y * ${gridOffsetY * workgroupSize}u;
  if (global >= ${updateCount}u) {
    return;
  }
${indexLoads.join("\n")}
  if (${bounds.join(" || ")}) {
    return;
  }
  let output_index = ${outputIndex};
  ${updateSource.statement}
}
`.trim();

  return [
    {
      code,
      numInputs: type.inputShapes.length,
      numOutputs: 1,
      hasUniform: false,
      clearOutputs: true,
      passes: [{ grid: calculateGrid(Math.ceil(updateCount / workgroupSize)) }],
    },
  ];
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

const CholeskyPhase = {
  Unblocked: 0,
  InitBlocked: 1,
  FactorBlock: 2,
  SolvePanel: 3,
  UpdateTrailing: 4,
} as const;

const CHOLESKY_BLOCK_SIZE = 16;
const CHOLESKY_BLOCK_THRESHOLD = 256;
const CHOLESKY_UPDATE_TILE_SIZE = 16;

function choleskyUniform(
  phase: number,
  k: number,
  blockSize: number,
  rowsBelow: number,
): Uint8Array<ArrayBuffer> {
  const ar = new Uint32Array(4);
  ar[0] = phase;
  ar[1] = k;
  ar[2] = blockSize;
  ar[3] = rowsBelow;
  return new Uint8Array(ar.buffer);
}

/**
 * Generate a Cholesky decomposition shader.
 *
 * Small matrices use the original one-pass Cholesky-Crout kernel. Larger
 * matrices use a blocked multi-pass variant so the trailing update can run
 * across many workgroups.
 */
function createCholesky(device: GPUDevice, type: RoutineType): ShaderInfo[] {
  const dtype = type.inputDtypes[0];
  const shape = type.inputShapes[0];
  const n = shape[shape.length - 1]; // Matrix dimension (n x n)
  const batches = prod(shape.slice(0, -2)); // Number of matrices in batch

  const needsF16 = dtype === DType.Float16;
  const ty = dtypeToWgsl(dtype, true);
  const workgroupSize = Math.min(
    256,
    findPow2(0, device.limits.maxComputeWorkgroupSizeX),
  );
  const useBlocked = n >= CHOLESKY_BLOCK_THRESHOLD;
  const blockSize = useBlocked ? CHOLESKY_BLOCK_SIZE : n;

  const code = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> input: array<${ty}>;
@group(0) @binding(1) var<storage, read_write> output: array<${ty}>;

struct CholeskyParams {
  phase: u32,
  k: u32,
  block_size: u32,
  rows_below: u32,
}

@group(1) @binding(0) var<uniform> params: CholeskyParams;

// Shared memory for the diagonal element
var<workgroup> L_jj: ${ty};

fn mat_idx(base: u32, row: u32, col: u32) -> u32 {
  return base + row * ${n}u + col;
}

@compute @workgroup_size(${workgroupSize})
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(global_invocation_id) global_id: vec3<u32>,
) {
  let tid = local_id.x;

  if (params.phase == ${CholeskyPhase.Unblocked}u) {
    let batch = wg_id.x + wg_id.y * ${gridOffsetY}u;
    if (batch >= ${batches}u) {
      return;
    }

    let base = batch * ${n * n}u;

    // Zero out output and copy lower triangle from input.
    for (var idx = tid; idx < ${n * n}u; idx += ${workgroupSize}u) {
      let row = idx / ${n}u;
      let col = idx % ${n}u;
      output[base + idx] = select(0, input[base + idx], col <= row);
    }
    storageBarrier();

    // Cholesky-Crout algorithm: process column by column.
    for (var j = 0u; j < ${n}u; j++) {
      for (var i = j + tid; i < ${n}u; i += ${workgroupSize}u) {
        var sum = output[mat_idx(base, i, j)];
        for (var k = 0u; k < j; k++) {
          sum -= output[mat_idx(base, i, k)] * output[mat_idx(base, j, k)];
        }
        output[mat_idx(base, i, j)] = sum;
      }
      storageBarrier();

      if (tid == 0u) {
        L_jj = sqrt(output[mat_idx(base, j, j)]);
        output[mat_idx(base, j, j)] = L_jj;
      }
      workgroupBarrier();

      for (var i = j + 1u + tid; i < ${n}u; i += ${workgroupSize}u) {
        output[mat_idx(base, i, j)] /= L_jj;
      }
      storageBarrier();
    }
    return;
  }

  if (params.phase == ${CholeskyPhase.InitBlocked}u) {
    let batch = wg_id.x + wg_id.y * ${gridOffsetY}u;
    if (batch >= ${batches}u) {
      return;
    }

    let base = batch * ${n * n}u;

    for (var idx = tid; idx < ${n * n}u; idx += ${workgroupSize}u) {
      let row = idx / ${n}u;
      let col = idx % ${n}u;
      output[base + idx] = select(0, input[base + idx], col <= row);
    }
    return;
  }

  if (params.phase == ${CholeskyPhase.FactorBlock}u) {
    let batch = wg_id.x + wg_id.y * ${gridOffsetY}u;
    if (batch >= ${batches}u) {
      return;
    }

    let base = batch * ${n * n}u;
    let k0 = params.k;
    let b = params.block_size;

    for (var j = 0u; j < b; j++) {
      let col = k0 + j;
      for (var i = j + tid; i < b; i += ${workgroupSize}u) {
        let row = k0 + i;
        var sum = output[mat_idx(base, row, col)];
        for (var r = 0u; r < j; r++) {
          sum -= output[mat_idx(base, row, k0 + r)] *
            output[mat_idx(base, col, k0 + r)];
        }
        output[mat_idx(base, row, col)] = sum;
      }
      storageBarrier();

      if (tid == 0u) {
        L_jj = sqrt(output[mat_idx(base, col, col)]);
        output[mat_idx(base, col, col)] = L_jj;
      }
      workgroupBarrier();

      for (var i = j + 1u + tid; i < b; i += ${workgroupSize}u) {
        output[mat_idx(base, k0 + i, col)] /= L_jj;
      }
      storageBarrier();
    }
    return;
  }

  if (params.phase == ${CholeskyPhase.SolvePanel}u) {
    let rows = params.rows_below;
    if (rows == 0u) {
      return;
    }

    let global = global_id.x + global_id.y * ${gridOffsetY * workgroupSize}u;
    if (global >= rows * ${batches}u) {
      return;
    }

    let batch = global / rows;
    let row = params.k + params.block_size + (global % rows);
    let base = batch * ${n * n}u;

    for (var j = 0u; j < params.block_size; j++) {
      let col = params.k + j;
      var sum = output[mat_idx(base, row, col)];
      for (var r = 0u; r < j; r++) {
        sum -= output[mat_idx(base, row, params.k + r)] *
          output[mat_idx(base, col, params.k + r)];
      }
      output[mat_idx(base, row, col)] =
        sum / output[mat_idx(base, col, col)];
    }
    return;
  }

  if (params.phase == ${CholeskyPhase.UpdateTrailing}u) {
    let rows = params.rows_below;
    if (rows == 0u) {
      return;
    }

    let tile_count = (rows + ${CHOLESKY_UPDATE_TILE_SIZE - 1}u) / ${CHOLESKY_UPDATE_TILE_SIZE}u;
    let tiles_per_batch = tile_count * (tile_count + 1u) / 2u;
    let wg_global = wg_id.x + wg_id.y * ${gridOffsetY}u;
    if (wg_global >= tiles_per_batch * ${batches}u) {
      return;
    }

    let batch = wg_global / tiles_per_batch;
    let tile = wg_global % tiles_per_batch;
    var tile_row = u32((sqrt(f32(8u * tile + 1u)) - 1.0) * 0.5);
    // Correct for GPU sqrt rounding around exact triangular-number boundaries.
    var row_start = tile_row * (tile_row + 1u) / 2u;
    if (row_start > tile) {
      tile_row -= 1u;
      row_start = tile_row * (tile_row + 1u) / 2u;
    }
    let next_row_start = (tile_row + 1u) * (tile_row + 2u) / 2u;
    if (tile >= next_row_start) {
      tile_row += 1u;
      row_start = next_row_start;
    }
    let tile_col = tile - row_start;
    let local_row = tid / ${CHOLESKY_UPDATE_TILE_SIZE}u;
    let local_col = tid % ${CHOLESKY_UPDATE_TILE_SIZE}u;
    let row_local = tile_row * ${CHOLESKY_UPDATE_TILE_SIZE}u + local_row;
    let col_local = tile_col * ${CHOLESKY_UPDATE_TILE_SIZE}u + local_col;
    if (row_local >= rows || col_local >= rows) {
      return;
    }
    if (row_local < col_local) {
      return;
    }

    let row = params.k + params.block_size + row_local;
    let col = params.k + params.block_size + col_local;
    let base = batch * ${n * n}u;
    var sum = output[mat_idx(base, row, col)];
    for (var r = 0u; r < params.block_size; r++) {
      sum -= output[mat_idx(base, row, params.k + r)] *
        output[mat_idx(base, col, params.k + r)];
    }
    output[mat_idx(base, row, col)] = sum;
  }
}
`.trim();

  const passes: ShaderInfo["passes"] = [];
  if (!useBlocked) {
    passes.push({
      grid: calculateGrid(batches),
      uniform: choleskyUniform(CholeskyPhase.Unblocked, 0, n, 0),
    });
  } else {
    passes.push({
      grid: calculateGrid(batches),
      uniform: choleskyUniform(CholeskyPhase.InitBlocked, 0, blockSize, 0),
    });
    for (let k = 0; k < n; k += blockSize) {
      const b = Math.min(blockSize, n - k);
      const rowsBelow = n - k - b;
      passes.push({
        grid: calculateGrid(batches),
        uniform: choleskyUniform(CholeskyPhase.FactorBlock, k, b, rowsBelow),
      });
      if (rowsBelow > 0) {
        passes.push({
          grid: calculateGrid(Math.ceil((rowsBelow * batches) / workgroupSize)),
          uniform: choleskyUniform(CholeskyPhase.SolvePanel, k, b, rowsBelow),
        });
        passes.push({
          grid: calculateGrid(
            (Math.ceil(rowsBelow / CHOLESKY_UPDATE_TILE_SIZE) *
              (Math.ceil(rowsBelow / CHOLESKY_UPDATE_TILE_SIZE) + 1) *
              batches) /
              2,
          ),
          uniform: choleskyUniform(
            CholeskyPhase.UpdateTrailing,
            k,
            b,
            rowsBelow,
          ),
        });
      }
    }
  }

  return [
    {
      code,
      numInputs: 1,
      numOutputs: 1,
      hasUniform: true,
      passes,
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

function createJacobiEigh(
  device: GPUDevice,
  type: RoutineType,
  params: { maxSweeps: number; tolerance: number },
): ShaderInfo[] {
  const dtype = type.inputDtypes[0];
  const shape = type.inputShapes[0];
  const n = shape[shape.length - 1];
  const batches = prod(shape.slice(0, -2));

  const needsF16 = dtype === DType.Float16;
  const ty = dtypeToWgsl(dtype, true);
  const tolerance = `${ty}(${params.tolerance})`;
  const workgroupSize = findPow2(
    Math.max(n, 1),
    device.limits.maxComputeWorkgroupSizeX,
  );

  const code = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> input: array<${ty}>;
@group(0) @binding(1) var<storage, read_write> diagonalized: array<${ty}>;
@group(0) @binding(2) var<storage, read_write> vectors: array<${ty}>;

var<workgroup> done: u32;
var<workgroup> rot_active: u32;
var<workgroup> rot_c: ${ty};
var<workgroup> rot_s: ${ty};
var<workgroup> rot_app: ${ty};
var<workgroup> rot_aqq: ${ty};
var<workgroup> rot_apq: ${ty};

fn mat_idx(base: u32, row: u32, col: u32) -> u32 {
  return base + row * ${n}u + col;
}

fn sym_idx(base: u32, row: u32, col: u32) -> u32 {
  return mat_idx(base, max(row, col), min(row, col));
}

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

  for (var idx = tid; idx < ${n * n}u; idx += ${workgroupSize}u) {
    let row = idx / ${n}u;
    let col = idx % ${n}u;
    diagonalized[base + idx] = select(
      ${ty}(0),
      input[base + idx],
      row >= col,
    );
    vectors[base + idx] = select(${ty}(0), ${ty}(1), row == col);
  }
  storageBarrier();

  for (var sweep = 0u; sweep < ${params.maxSweeps}u; sweep++) {
    if (tid == 0u) {
      var max_abs = ${ty}(1);
      var max_offdiag = ${ty}(0);
      for (var idx = 0u; idx < ${n * n}u; idx++) {
        let row = idx / ${n}u;
        let col = idx % ${n}u;
        let value = abs(diagonalized[base + idx]);
        max_abs = max(max_abs, value);
        if (row > col) {
          max_offdiag = max(max_offdiag, value);
        }
      }
      done = select(0u, 1u, max_offdiag <= ${tolerance} * max_abs);
    }
    let done_uniform = workgroupUniformLoad(&done);
    if (done_uniform != 0u) {
      break;
    }

    for (var p = 0u; p + 1u < ${n}u; p++) {
      for (var q = p + 1u; q < ${n}u; q++) {
        if (tid == 0u) {
          rot_app = diagonalized[mat_idx(base, p, p)];
          rot_aqq = diagonalized[mat_idx(base, q, q)];
          rot_apq = diagonalized[sym_idx(base, p, q)];
          if (rot_apq == ${ty}(0)) {
            rot_active = 0u;
            rot_c = ${ty}(1);
            rot_s = ${ty}(0);
          } else {
            let tau = (rot_aqq - rot_app) / (${ty}(2) * rot_apq);
            let tau_sign = select(${ty}(-1), ${ty}(1), tau >= ${ty}(0));
            let t = tau_sign / (abs(tau) + sqrt(tau * tau + ${ty}(1)));
            rot_c = inverseSqrt(t * t + ${ty}(1));
            rot_s = t * rot_c;
            rot_active = 1u;
          }
        }
        workgroupBarrier();

        if (rot_active != 0u) {
          for (var k = tid; k < ${n}u; k += ${workgroupSize}u) {
            if (k != p && k != q) {
              let kp = sym_idx(base, k, p);
              let kq = sym_idx(base, k, q);
              let akp = diagonalized[kp];
              let akq = diagonalized[kq];
              let next_kp = rot_c * akp - rot_s * akq;
              let next_kq = rot_s * akp + rot_c * akq;
              diagonalized[kp] = next_kp;
              diagonalized[kq] = next_kq;
            } else if (k == p) {
              diagonalized[mat_idx(base, p, p)] =
                rot_c * rot_c * rot_app - ${ty}(2) * rot_s * rot_c * rot_apq + rot_s * rot_s * rot_aqq;
              diagonalized[sym_idx(base, p, q)] = ${ty}(0);
            } else {
              diagonalized[mat_idx(base, q, q)] =
                rot_s * rot_s * rot_app + ${ty}(2) * rot_s * rot_c * rot_apq + rot_c * rot_c * rot_aqq;
            }

            let vp = mat_idx(base, k, p);
            let vq = mat_idx(base, k, q);
            let vkp = vectors[vp];
            let vkq = vectors[vq];
            vectors[vp] = rot_c * vkp - rot_s * vkq;
            vectors[vq] = rot_s * vkp + rot_c * vkq;
          }
        }
        storageBarrier();
      }
    }
  }
}
`.trim();

  const grid = calculateGrid(batches);
  return [
    {
      code,
      numInputs: 1,
      numOutputs: 2,
      hasUniform: false,
      passes: [{ grid }],
    },
  ];
}

function fftUniform(
  phase: number,
  radix: number,
  prev: number,
  normalize: boolean,
): Uint8Array<ArrayBuffer> {
  return new Uint8Array(
    new Uint32Array([phase, radix, prev, normalize ? 1 : 0]).buffer,
  );
}

function createFft(
  device: GPUDevice,
  type: RoutineType,
  params: { factors: number[]; inverse: boolean },
): ShaderInfo[] {
  const dtype = type.inputDtypes[0];
  const shape = type.inputShapes[0];
  const n = shape[shape.length - 1];
  const batches = prod(shape.slice(0, -1));
  if (prod(params.factors) !== n) {
    throw new Error(
      `fft: factorization ${params.factors} does not match size ${n}`,
    );
  }

  const needsF16 = dtype === DType.Float16;
  const ty = dtypeToWgsl(dtype, true);
  const workgroupSize = Math.min(
    256,
    findPow2(0, device.limits.maxComputeWorkgroupSizeX),
  );
  const maxFactor = Math.max(1, ...params.factors);
  const angleScale = params.inverse
    ? "6.283185307179586"
    : "-6.283185307179586";
  const digitReversal = params.factors
    .map(
      (factor) => `
  digit = remaining % ${factor}u;
  remaining = remaining / ${factor}u;
  stride = stride * ${factor}u;
  reversed = reversed + digit * (${n}u / stride);`,
    )
    .join("");

  const code = `
${needsF16 ? "enable f16;" : ""}
${headerWgsl}

@group(0) @binding(0) var<storage, read> input_real: array<${ty}>;
@group(0) @binding(1) var<storage, read> input_imag: array<${ty}>;
@group(0) @binding(2) var<storage, read_write> output_real: array<${ty}>;
@group(0) @binding(3) var<storage, read_write> output_imag: array<${ty}>;

struct FftParams {
  phase: u32,
  radix: u32,
  prev: u32,
  normalize: u32,
}

@group(1) @binding(0) var<uniform> fft_params: FftParams;

fn digit_reversed_index(index: u32) -> u32 {
  var remaining = index;
  var stride = 1u;
  var reversed = 0u;
  var digit = 0u;
${digitReversal}
  return reversed;
}

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let global = global_id.x + global_id.y * ${gridOffsetY * workgroupSize}u;

  if (fft_params.phase == 0u) {
    if (global >= ${batches * n}u) {
      return;
    }
    let batch = global / ${n}u;
    let out_idx = global % ${n}u;
    let source = batch * ${n}u + digit_reversed_index(out_idx);
    output_real[global] = input_real[source];
    output_imag[global] = input_imag[source];
    return;
  }

  let butterflies_per_batch = ${n}u / fft_params.radix;
  if (global >= ${batches}u * butterflies_per_batch) {
    return;
  }

  let batch = global / butterflies_per_batch;
  let local = global % butterflies_per_batch;
  let j = local % fft_params.prev;
  let group = local / fft_params.prev;
  let span = fft_params.prev * fft_params.radix;
  let start = batch * ${n}u + group * span + j;
  let scale = select(1.0, 1.0 / f32(${n}u), fft_params.normalize != 0u);

  var scratch_real: array<f32, ${maxFactor}>;
  var scratch_imag: array<f32, ${maxFactor}>;

  for (var q = 0u; q < fft_params.radix; q++) {
    let idx = start + q * fft_params.prev;
    let angle = ${angleScale} * f32(q * j) / f32(span);
    let c = cos(angle);
    let s = sin(angle);
    let xr = f32(output_real[idx]);
    let xi = f32(output_imag[idx]);
    scratch_real[q] = xr * c - xi * s;
    scratch_imag[q] = xr * s + xi * c;
  }

  for (var p = 0u; p < fft_params.radix; p++) {
    var sum_real = 0.0;
    var sum_imag = 0.0;
    for (var q = 0u; q < fft_params.radix; q++) {
      let angle = ${angleScale} * f32(q * p) / f32(fft_params.radix);
      let c = cos(angle);
      let s = sin(angle);
      let xr = scratch_real[q];
      let xi = scratch_imag[q];
      sum_real += xr * c - xi * s;
      sum_imag += xr * s + xi * c;
    }
    let idx = start + p * fft_params.prev;
    output_real[idx] = ${ty}(sum_real * scale);
    output_imag[idx] = ${ty}(sum_imag * scale);
  }
}
`.trim();

  const passes = [
    {
      grid: calculateGrid(Math.ceil((batches * n) / workgroupSize)),
      uniform: fftUniform(0, 1, 1, false),
    },
  ];
  let prev = 1;
  for (let i = 0; i < params.factors.length; i++) {
    const radix = params.factors[i];
    passes.push({
      grid: calculateGrid(Math.ceil((batches * n) / radix / workgroupSize)),
      uniform: fftUniform(
        1,
        radix,
        prev,
        params.inverse && i === params.factors.length - 1,
      ),
    });
    prev *= radix;
  }

  return [
    {
      code,
      numInputs: 2,
      numOutputs: 2,
      hasUniform: true,
      passes,
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
    case Routines.Scatter:
      return createScatter(device, routine.type, routine.params);
    case Routines.TriangularSolve:
      return createTriangularSolve(device, routine.type, routine.params);
    case Routines.Cholesky:
      return createCholesky(device, routine.type);
    case Routines.LU:
      return createLU(device, routine.type);
    case Routines.JacobiEigh:
      return createJacobiEigh(device, routine.type, routine.params);
    case Routines.Fft:
      return createFft(device, routine.type, routine.params);
    default:
      throw new UnsupportedRoutineError(routine.name, "webgpu");
  }
}
