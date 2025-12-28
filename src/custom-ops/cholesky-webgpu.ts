import { byteWidth, DType } from "../alu.js";
import { getBackend } from "../backend.js";
import { Array as JAXArray } from "../frontend/array.js";
import { ShapeTracker } from "../shape.js";

export interface CholeskyParams {
  lower: boolean;
}

/**
 * WebGPU implementation of Cholesky decomposition using custom compute shader
 *
 * Algorithm: Column-wise right-looking Cholesky with workgroup barriers
 *
 * For each column j (sequential kernel dispatch):
 *   - Compute diagonal: L[j,j] = sqrt(A[j,j] - sum(L[j,k]^2))
 *   - Workgroup barrier
 *   - Compute off-diagonal: L[i,j] = (A[i,j] - sum(...)) / L[j,j] for i > j
 *
 * Note: Sequential column dependencies prevent full GPU parallelization.
 * This approach minimizes kernel dispatch overhead by batching.
 */
export function choleskyWebGPU(inputA: JAXArray, params: CholeskyParams): JAXArray {
  const a = inputA.ref;

  // Validate input
  if (a.shape.length !== 2) {
    throw new Error(`Cholesky requires 2D matrix, got shape ${a.shape}`);
  }
  const n = a.shape[0];
  if (n !== a.shape[1]) {
    throw new Error(`Cholesky requires square matrix, got shape ${a.shape}`);
  }

  const dtype = a.dtype;
  const device = a.device;

  if (device !== "webgpu") {
    throw new Error("choleskyWebGPU only works with webgpu device");
  }
  if (dtype !== DType.Float32) {
    throw new Error("choleskyWebGPU currently only supports Float32");
  }

  // Get the WebGPU backend
  const backend: any = getBackend(device);
  const gpuDevice: GPUDevice = backend.device;

  if (!gpuDevice) {
    throw new Error("WebGPU device not available");
  }

  // Realize input to get the GPU buffer
  const inputSlot = a._realizeSource();
  const inputBufferInfo = backend.buffers.get(inputSlot);
  if (!inputBufferInfo) {
    throw new Error("Failed to get input buffer");
  }
  const inputBuffer: GPUBuffer = inputBufferInfo.buffer;

  // Allocate output buffer
  const outputSize = n * n * byteWidth(dtype);
  const outputSlot = backend.malloc(outputSize);
  const outputBufferInfo = backend.buffers.get(outputSlot);
  if (!outputBufferInfo) {
    throw new Error("Failed to allocate output buffer");
  }
  const outputBuffer: GPUBuffer = outputBufferInfo.buffer;

  // Copy input to output (we modify it in-place)
  const copyEncoder = gpuDevice.createCommandEncoder();
  copyEncoder.copyBufferToBuffer(inputBuffer, 0, outputBuffer, 0, outputSize);
  gpuDevice.queue.submit([copyEncoder.finish()]);

  // Column-wise Cholesky shader
  const shaderCode = `
@group(0) @binding(0) var<storage, read_write> L: array<f32>;
@group(0) @binding(1) var<uniform> params: vec2<u32>; // params.x = n, params.y = j

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  let n = params.x;
  let j = params.y;

  // Compute diagonal element (only thread j)
  if (i == j && i < n) {
    var sum_diag: f32 = 0.0;
    for (var k = 0u; k < j; k = k + 1u) {
      let ljk = L[j * n + k];
      sum_diag = sum_diag + (ljk * ljk);
    }
    L[j * n + j] = sqrt(max(L[j * n + j] - sum_diag, 1e-10));
  }

  workgroupBarrier();

  // Compute off-diagonal elements (threads i > j)
  if (i > j && i < n) {
    var sum_off: f32 = 0.0;
    for (var k = 0u; k < j; k = k + 1u) {
      sum_off = sum_off + (L[i * n + k] * L[j * n + k]);
    }
    L[i * n + j] = (L[i * n + j] - sum_off) / L[j * n + j];
  } else if (i < j && i < n) {
    // Zero upper triangle
    L[i * n + j] = 0.0;
  }
}
`;

  // Compile shader once
  const shaderModule = gpuDevice.createShaderModule({ code: shaderCode });
  const pipeline = gpuDevice.createComputePipeline({
    layout: "auto",
    compute: { module: shaderModule, entryPoint: "main" },
  });

  const workgroupSize = 256;
  const numWorkgroups = Math.ceil(n / workgroupSize);

  // Batch size: number of columns to submit in one command buffer
  // Larger batches reduce submission overhead but increase latency
  const BATCH_SIZE = 32;

  // Pre-allocate uniform buffers and bind groups for the entire batch
  const uniformBuffers: GPUBuffer[] = [];
  const bindGroups: GPUBindGroup[] = [];

  for (let batchStart = 0; batchStart < n; batchStart += BATCH_SIZE) {
    const batchEnd = Math.min(batchStart + BATCH_SIZE, n);
    const batchCount = batchEnd - batchStart;

    // Create uniform buffers and bind groups for this batch
    for (let i = 0; i < batchCount; i++) {
      const j = batchStart + i;
      const uniformData = new Uint32Array([n, j]);
      const uniformBuffer = gpuDevice.createBuffer({
        size: 8,
        usage: GPUBufferUsage.UNIFORM,
        mappedAtCreation: true,
      });
      new Uint32Array(uniformBuffer.getMappedRange()).set(uniformData);
      uniformBuffer.unmap();
      uniformBuffers.push(uniformBuffer);

      const bindGroup = gpuDevice.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: outputBuffer } },
          { binding: 1, resource: { buffer: uniformBuffer } },
        ],
      });
      bindGroups.push(bindGroup);
    }

    // Submit all columns in this batch in a single command buffer
    const commandEncoder = gpuDevice.createCommandEncoder();
    for (let i = 0; i < batchCount; i++) {
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroups[uniformBuffers.length - batchCount + i]);
      passEncoder.dispatchWorkgroups(numWorkgroups);
      passEncoder.end();
    }
    gpuDevice.queue.submit([commandEncoder.finish()]);
  }

  // Cleanup
  for (const buf of uniformBuffers) {
    buf.destroy();
  }

  a.dispose();

  // Return result as Array
  return new JAXArray({
    source: outputSlot,
    st: ShapeTracker.fromShape([n, n]),
    dtype,
    weakType: false,
    backend,
    committed: true,
  });
}
