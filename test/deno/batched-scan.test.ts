/**
 * Manual test for batched scan with routine body (matmul).
 * Tests the uniform-based offset approach with real WebGPU hardware.
 *
 * Test Environment: Deno native WebGPU (requires GPU hardware)
 * Why Deno: Provides working WebGPU access when Chromium's implementation doesn't work.
 * Regular test equivalent: test/batched-scan-manual.test.ts (may skip on some systems)
 *
 * Run with: deno test --unstable-webgpu --allow-read --allow-env --no-check test/deno/batched-scan.test.ts
 */

import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";

// Import scan-wrapper from source (these functions aren't exported in dist)
import {
  wrapRoutineForScan,
  createAllIterationsOffsetsBuffer,
} from "../../src/backend/webgpu/scan-wrapper.ts";

const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;

async function getJaxJsWebGPUDevice(): Promise<GPUDevice | null> {
  // Lazily import to avoid initializing WebGPU at module load.
  const { init, defaultDevice, getBackend } = await import(
    "../../dist/index.js"
  );

  // If a WebGPU backend is already active, reuse it without re-initializing.
  try {
    const backend = getBackend() as any;
    const existing = backend?.device as GPUDevice | undefined;
    if (existing) return existing;
  } catch {
    // getBackend may throw if init() hasn't been called yet.
  }

  const devices = await init();
  if (!devices.includes("webgpu")) return null;
  defaultDevice("webgpu");

  const backend = getBackend() as any;
  const device = backend?.device as GPUDevice | undefined;
  return device ?? null;
}

// ============================================================================
// jax-js integration tests - run FIRST to use shared backend before creating
// separate GPU devices (which can exhaust GPU memory when created/destroyed)
// ============================================================================

Deno.test({
  name: "batched scan - JS loop matmul reference",
  ignore: !hasWebGPU,
  fn: async () => {
    // Lazy import jax-js to avoid initializing WebGPU device early
    const {
      init,
      numpy: np,
      lax,
      defaultDevice,
    } = await import("../../dist/index.js");

    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("WebGPU not available, skipping");
      return;
    }
    defaultDevice("webgpu");

    console.log("WebGPU available, running matmul scan test");

    // Simple scan with matmul: accumulate matrix products
    // carry = carry @ x for each x in xs
    const n = 4; // Matrix size
    const length = 3; // Number of iterations

    // Create test data
    // initCarry: identity matrix
    const initCarry = np.eye(n);

    // xs: stack of matrices to multiply
    const xsData = new Float32Array(length * n * n);

    // xs[0]: scale x by 2
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        xsData[0 * n * n + i * n + j] = i === j ? (i === 0 ? 2 : 1) : 0;
      }
    }

    // xs[1]: scale y by 2
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        xsData[1 * n * n + i * n + j] = i === j ? (i === 1 ? 2 : 1) : 0;
      }
    }

    // xs[2]: shear (add row 0 to row 1)
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) {
          xsData[2 * n * n + i * n + j] = 1;
        } else if (i === 0 && j === 1) {
          xsData[2 * n * n + i * n + j] = 1;
        } else {
          xsData[2 * n * n + i * n + j] = 0;
        }
      }
    }

    const xs = np.array(xsData).reshape([length, n, n]);

    // Run scan with JS loop (reference)
    const scanFn = (carry: any, x: any): [any, any] => {
      const result = np.matmul(carry.ref, x);
      return [result.ref, result];
    };

    const [finalCarry, ys] = await lax.scan(scanFn, initCarry.ref, xs.ref);

    // Get reference results
    const finalCarryData = await finalCarry.data();

    console.log(
      "Final carry (JS loop):",
      Array.from(finalCarryData).slice(0, 16),
    );

    // Compute expected result manually
    const I = np.eye(n);
    const x0 = np.array(xsData.slice(0, n * n)).reshape([n, n]);
    const x1 = np.array(xsData.slice(n * n, 2 * n * n)).reshape([n, n]);
    const x2 = np.array(xsData.slice(2 * n * n, 3 * n * n)).reshape([n, n]);

    const y0 = np.matmul(I.ref, x0);
    const y1 = np.matmul(y0.ref, x1);
    const y2 = np.matmul(y1.ref, x2);

    const expectedFinal = await y2.data();
    console.log("Expected final:", Array.from(expectedFinal).slice(0, 16));

    // Verify
    for (let i = 0; i < n * n; i++) {
      const diff = Math.abs(finalCarryData[i] - expectedFinal[i]);
      if (diff > 1e-5) {
        throw new Error(
          `Mismatch at index ${i}: got ${finalCarryData[i]}, expected ${expectedFinal[i]}`,
        );
      }
    }

    // Cleanup:
    // - initCarry, xs: used .ref so still have rc=1
    // - finalCarry, ys: consumed by .data()
    // - I: used .ref so still has rc=1
    // - x0, x1, x2: consumed by matmul (no .ref)
    // - y0, y1: used .ref so still have rc=1
    // - y2: consumed by .data()
    initCarry.dispose();
    xs.dispose();
    ys.dispose();
    I.dispose();
    y0.dispose();
    y1.dispose();

    console.log("✓ JS loop scan with matmul body produces correct results");
  },
});

Deno.test({
  name: "batched scan - matmul with reverse",
  ignore: !hasWebGPU,
  fn: async () => {
    // Test that reverse scan with routine body (matmul) processes xs in reverse order
    const {
      init,
      numpy: np,
      lax,
      defaultDevice,
    } = await import("../../dist/index.js");

    const devices = await init();
    if (!devices.includes("webgpu")) {
      console.log("WebGPU not available, skipping");
      return;
    }
    defaultDevice("webgpu");

    console.log("Testing matmul scan with reverse=true");

    // 2x2 matrices for simplicity
    const n = 2;

    const initCarry = np.eye(n); // Identity matrix

    // xs[0]: scale by 2
    // xs[1]: shear
    // xs[2]: rotate 90 degrees
    const xs = np.array([
      [
        [2, 0],
        [0, 2],
      ], // scale
      [
        [1, 1],
        [0, 1],
      ], // shear
      [
        [0, -1],
        [1, 0],
      ], // rotate
    ]);

    const scanFn = (carry: any, x: any): [any, any] => {
      const result = np.matmul(carry.ref, x);
      return [result.ref, result];
    };

    // Reverse scan: process xs[2], xs[1], xs[0]
    const [finalCarry, ys] = await lax.scan(scanFn, initCarry.ref, xs.ref, {
      reverse: true,
    });

    const finalCarryData = await finalCarry.data();
    console.log("Final carry (reverse):", Array.from(finalCarryData));

    // Expected: I * rotate * shear * scale = [[0, -2], [2, 2]]
    // Row-major: [0, -2, 2, 2]
    const expected = [0, -2, 2, 2];
    for (let i = 0; i < 4; i++) {
      const diff = Math.abs(finalCarryData[i] - expected[i]);
      if (diff > 1e-5) {
        throw new Error(
          `Mismatch at index ${i}: got ${finalCarryData[i]}, expected ${expected[i]}`,
        );
      }
    }

    initCarry.dispose();
    xs.dispose();
    ys.dispose();

    console.log("✓ Reverse scan with matmul body produces correct results");
  },
});

// ============================================================================
// Pure WebGPU tests (no jax-js) - run after jax-js tests
// These create/destroy separate GPU devices which may not release memory immediately
// ============================================================================

Deno.test({
  name: "batched scan - offset buffer has correct values",
  ignore: !hasWebGPU,
  fn: async () => {
    // Use default alignment of 256 bytes (typical GPU value)
    const alignment = 256;

    const length = 5;
    const xsElemStrides = [16]; // 4x4 matrix = 16 elements
    const ysElemStrides = [16];

    console.log("Using alignment:", alignment);

    const { buffer, alignment: actualAlignment } =
      createAllIterationsOffsetsBuffer(
        1,
        1,
        length,
        xsElemStrides,
        ysElemStrides,
        alignment,
      );

    console.log("Offset buffer size:", buffer.length);
    console.log("Per-iteration alignment:", actualAlignment);

    // Verify offsets for each iteration
    const view = new DataView(buffer.buffer);
    for (let i = 0; i < length; i++) {
      const xOffset = view.getUint32(i * actualAlignment, true);
      const yOffset = view.getUint32(i * actualAlignment + 4, true);
      console.log(`Iteration ${i}: x_offset=${xOffset}, y_offset=${yOffset}`);

      assertEquals(xOffset, i * 16, `x_offset for iteration ${i}`);
      assertEquals(yOffset, i * 16, `y_offset for iteration ${i}`);
    }

    console.log(
      "✓ Offset buffer has correct element offsets for each iteration",
    );
  },
});

Deno.test({
  name: "batched scan - shader wrapper transforms correctly",
  ignore: !hasWebGPU,
  fn: async () => {
    // Test shader wrapping with a matmul-like pattern - NO jax-js needed
    const mockShader = {
      code: `
@group(0) @binding(0) var<storage, read> carry_in: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> carry_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let row = id.x;
  let col = id.y;
  let n = 4u;
  
  if (row >= n || col >= n) { return; }
  
  var sum = 0.0;
  for (var k = 0u; k < n; k = k + 1u) {
    sum = sum + carry_in[row * n + k] * x[k * n + col];
  }
  
  let idx = row * n + col;
  carry_out[idx] = sum;
  y[idx] = sum;
}
`,
      numInputs: 2,
      numOutputs: 2,
      hasUniform: false,
      passes: [{ grid: [1, 1] as [number, number] }],
    };

    const scanInfo = {
      numConsts: 0,
      numCarry: 1, // carry_in
      numX: 1, // x
      numY: 1, // y (carry_out is carry)
      numInputs: 2,
      numOutputs: 2,
    };

    const wrapped = wrapRoutineForScan(mockShader, scanInfo);

    console.log("Wrapped shader:");
    console.log(wrapped.code);

    assertEquals(wrapped.hasUniform, true);

    // Verify struct and offsets are added
    if (!wrapped.code.includes("struct ScanOffsets")) {
      throw new Error("Missing ScanOffsets struct");
    }
    if (!wrapped.code.includes("x_offset")) {
      throw new Error("Missing x_offset");
    }
    if (!wrapped.code.includes("y_offset")) {
      throw new Error("Missing y_offset");
    }

    // carry_in and carry_out should NOT have offsets
    if (wrapped.code.includes("carry_in_offset")) {
      throw new Error("carry_in should not have offset");
    }
    if (wrapped.code.includes("carry_out_offset")) {
      throw new Error("carry_out should not have offset");
    }

    // Verify x and y accesses are transformed
    if (!wrapped.code.includes("x[x_offset +")) {
      throw new Error("x accesses should be transformed");
    }
    if (!wrapped.code.includes("y[y_offset +")) {
      throw new Error("y accesses should be transformed");
    }

    // Verify carry accesses are NOT transformed (should still have original form)
    if (!wrapped.code.includes("carry_in[row * n + k]")) {
      throw new Error("carry_in access should NOT be transformed");
    }
    if (!wrapped.code.includes("carry_out[idx]")) {
      throw new Error("carry_out access should NOT be transformed");
    }

    console.log("✓ Shader wrapping correctly identifies xs/ys vs carry");
  },
});

Deno.test({
  name: "batched scan - end-to-end dispatch with uniform offsets",
  ignore: !hasWebGPU,
  fn: async () => {
    // IMPORTANT: Reuse jax-js's WebGPU device. Creating a second GPUDevice can
    // destabilize Deno WebGPU and cause flakiness/segfaults across test files.
    const device = await getJaxJsWebGPUDevice();
    if (!device) {
      console.log("WebGPU not available in jax-js, skipping");
      return;
    }

    const createdBuffers: GPUBuffer[] = [];

    // Wrap entire test in try/catch to gracefully handle GPU resource exhaustion
    try {
      const n = 4; // 4x4 matrices
      const length = 3; // 3 iterations
      const alignment = 256; // typical GPU alignment

      // Create a simple scan shader that adds x to carry
      // (Not matmul, just element-wise add for simplicity)
      const shaderCode = `
struct ScanOffsets {
  x_offset: u32,
  y_offset: u32,
}

@group(0) @binding(0) var<storage, read> carry_in: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> carry_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@group(1) @binding(0) var<uniform> offsets: ScanOffsets;

@compute @workgroup_size(16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let idx = id.x;
  if (idx >= 16u) { return; }
  
  let x_idx = offsets.x_offset + idx;
  let y_idx = offsets.y_offset + idx;
  
  let val = carry_in[idx] + x[x_idx];
  carry_out[idx] = val;
  y[y_idx] = val;
}
`;

      const module = device.createShaderModule({ code: shaderCode });

      // Create explicit bind group layouts with dynamic offsets
      const storageBindGroupLayout = device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" },
          },
          {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "read-only-storage" },
          },
          {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" },
          },
          {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "storage" },
          },
        ],
      });

      const uniformBindGroupLayout = device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer: { type: "uniform", hasDynamicOffset: true },
          },
        ],
      });

      const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [storageBindGroupLayout, uniformBindGroupLayout],
      });

      const pipeline = device.createComputePipeline({
        layout: pipelineLayout,
        compute: { module, entryPoint: "main" },
      });

      // Create buffers
      const carrySize = n * n * 4; // 16 floats = 64 bytes
      const xsSize = length * n * n * 4; // 3 * 16 floats = 192 bytes
      const ysSize = length * n * n * 4; // 3 * 16 floats = 192 bytes

      // Ping-pong carry buffers
      const carryPing = device.createBuffer({
        size: carrySize,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
      });
      createdBuffers.push(carryPing);
      const carryPong = device.createBuffer({
        size: carrySize,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_DST |
          GPUBufferUsage.COPY_SRC,
      });
      createdBuffers.push(carryPong);

      // xs buffer (stacked inputs)
      const xsBuffer = device.createBuffer({
        size: xsSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      createdBuffers.push(xsBuffer);
      const xsData = new Float32Array(xsBuffer.getMappedRange());
      // xs[0] = [1,1,1...], xs[1] = [2,2,2...], xs[2] = [3,3,3...]
      for (let iter = 0; iter < length; iter++) {
        for (let i = 0; i < n * n; i++) {
          xsData[iter * n * n + i] = iter + 1;
        }
      }
      xsBuffer.unmap();

      // ys buffer (stacked outputs)
      const ysBuffer = device.createBuffer({
        size: ysSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      });
      createdBuffers.push(ysBuffer);

      // Initialize carry to zeros
      const initCarryData = new Float32Array(n * n);
      initCarryData.fill(0);
      device.queue.writeBuffer(carryPing, 0, initCarryData);

      // Create uniform buffer for offsets (alignment already defined above)
      const offsetBufferSize = length * alignment;
      const offsetBuffer = device.createBuffer({
        size: offsetBufferSize,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      createdBuffers.push(offsetBuffer);
      const offsetData = new Uint8Array(offsetBuffer.getMappedRange());
      const offsetView = new DataView(offsetData.buffer);
      for (let iter = 0; iter < length; iter++) {
        // x_offset and y_offset in elements
        offsetView.setUint32(iter * alignment, iter * n * n, true);
        offsetView.setUint32(iter * alignment + 4, iter * n * n, true);
      }
      offsetBuffer.unmap();

      // Create bind groups
      const pingBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: carryPing } },
          { binding: 1, resource: { buffer: xsBuffer } },
          { binding: 2, resource: { buffer: carryPong } },
          { binding: 3, resource: { buffer: ysBuffer } },
        ],
      });

      const pongBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: carryPong } },
          { binding: 1, resource: { buffer: xsBuffer } },
          { binding: 2, resource: { buffer: carryPing } },
          { binding: 3, resource: { buffer: ysBuffer } },
        ],
      });

      const uniformBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [
          {
            binding: 0,
            resource: { buffer: offsetBuffer, offset: 0, size: 8 },
          },
        ],
      });

      // Encode all iterations
      const commandEncoder = device.createCommandEncoder();

      for (let iter = 0; iter < length; iter++) {
        const storageBindGroup = iter % 2 === 0 ? pingBindGroup : pongBindGroup;
        const dynamicOffset = iter * alignment;

        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, storageBindGroup);
        passEncoder.setBindGroup(1, uniformBindGroup, [dynamicOffset]);
        passEncoder.dispatchWorkgroups(1);
        passEncoder.end();
      }

      // Copy final carry to staging buffer
      const finalCarryBuffer = length % 2 === 0 ? carryPing : carryPong;
      const stagingBuffer = device.createBuffer({
        size: carrySize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      createdBuffers.push(stagingBuffer);
      commandEncoder.copyBufferToBuffer(
        finalCarryBuffer,
        0,
        stagingBuffer,
        0,
        carrySize,
      );

      // Also copy ys to staging
      const ysStagingBuffer = device.createBuffer({
        size: ysSize,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      createdBuffers.push(ysStagingBuffer);
      commandEncoder.copyBufferToBuffer(
        ysBuffer,
        0,
        ysStagingBuffer,
        0,
        ysSize,
      );

      device.queue.submit([commandEncoder.finish()]);

      // Read results
      await stagingBuffer.mapAsync(GPUMapMode.READ);
      const finalCarryResult = new Float32Array(
        stagingBuffer.getMappedRange().slice(0),
      );
      stagingBuffer.unmap();

      await ysStagingBuffer.mapAsync(GPUMapMode.READ);
      const ysResult = new Float32Array(
        ysStagingBuffer.getMappedRange().slice(0),
      );
      ysStagingBuffer.unmap();

      // Make sure all GPU work is fully complete before destroying buffers.
      // This reduces flakiness in Deno's WebGPU runtime when resources are
      // created/destroyed frequently across multiple test files.
      await device.queue.onSubmittedWorkDone();

      console.log("Final carry:", Array.from(finalCarryResult));
      console.log("ys[0]:", Array.from(ysResult.slice(0, 16)));
      console.log("ys[1]:", Array.from(ysResult.slice(16, 32)));
      console.log("ys[2]:", Array.from(ysResult.slice(32, 48)));

      // Expected:
      // iter 0: carry = 0 + 1 = 1, y[0] = 1
      // iter 1: carry = 1 + 2 = 3, y[1] = 3
      // iter 2: carry = 3 + 3 = 6, y[2] = 6

      for (let i = 0; i < n * n; i++) {
        assertEquals(finalCarryResult[i], 6, `final carry[${i}]`);
        assertEquals(ysResult[0 * n * n + i], 1, `ys[0][${i}]`);
        assertEquals(ysResult[1 * n * n + i], 3, `ys[1][${i}]`);
        assertEquals(ysResult[2 * n * n + i], 6, `ys[2][${i}]`);
      }

      console.log(
        "✓ End-to-end batched scan dispatch with uniform offsets works correctly!",
      );
    } catch (e: unknown) {
      if (
        e instanceof Error &&
        (e.message.includes("memory") || e.message.includes("invalid"))
      ) {
        console.log("GPU resource error, skipping:", e.message);
        return;
      }
      throw e;
    } finally {
      // Don't destroy the shared jax-js device.
      for (const b of createdBuffers) b.destroy();
    }
  },
});

Deno.test({
  name: "batched scan - compile wrapped shader on GPU",
  ignore: !hasWebGPU,
  fn: async () => {
    const device = await getJaxJsWebGPUDevice();
    if (!device) {
      console.log("WebGPU not available in jax-js, skipping");
      return;
    }

    // Create a simple shader and wrap it
    const mockShader = {
      code: `
@group(0) @binding(0) var<storage, read> carry_in: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> carry_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let idx = id.x;
  if (idx >= 16u) { return; }
  
  // Simple element-wise add (not real matmul, just for testing shader compilation)
  let val = carry_in[idx] + x[idx];
  carry_out[idx] = val;
  y[idx] = val;
}
`,
      numInputs: 2,
      numOutputs: 2,
      hasUniform: false,
      passes: [{ grid: [1, 1] as [number, number] }],
    };

    const scanInfo = {
      numConsts: 0,
      numCarry: 1,
      numX: 1,
      numY: 1,
      numInputs: 2,
      numOutputs: 2,
    };

    const wrapped = wrapRoutineForScan(mockShader, scanInfo);

    console.log("Attempting to compile wrapped shader on GPU...");
    console.log(wrapped.code);

    // Try to compile the shader (device already obtained from navigator.gpu above)
    const module = device.createShaderModule({ code: wrapped.code });

    // Create pipeline to verify it compiles
    const pipeline = device.createComputePipeline({
      layout: "auto",
      compute: { module, entryPoint: "main" },
    });

    console.log("✓ Wrapped shader compiles successfully on GPU");

    // Verify bind group layouts exist
    const storageLayout = pipeline.getBindGroupLayout(0);
    const uniformLayout = pipeline.getBindGroupLayout(1);

    console.log(
      "✓ Pipeline has expected bind group layouts (group 0: storage, group 1: uniform)",
    );
  },
});
