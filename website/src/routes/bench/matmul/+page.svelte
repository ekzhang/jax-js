<script lang="ts">
  import { browser } from "$app/environment";

  const n = 4096;

  let result: Record<string, number> = $state({});

  const randomBuffer = new Float32Array(
    [...new Array(n * n)].map(() => Math.random()),
  );

  function printBufferItems(buf: Float32Array | Float16Array) {
    // Print a couple items from the buffer.
    console.log(
      buf[0],
      buf[1],
      buf[2],
      buf[3],
      buf[Math.floor((n * n) / 2)],
      buf[n * n - 1],
    );
  }

  // Uncomment to intercept and print out shader code.
  // onMount(() => {
  //   const csm = GPUDevice.prototype.createShaderModule;
  //   GPUDevice.prototype.createShaderModule = function (
  //     descriptor: GPUShaderModuleDescriptor,
  //   ) {
  //     console.info("Intercepted createShaderModule:\n\n" + descriptor.code);
  //     return csm.call(this, descriptor);
  //   };
  // });

  abstract class Strategy {
    abstract name: string;
    abstract run(): Promise<number>;
  }

  abstract class GpuStrategy extends Strategy {
    abstract kernel(): string;
    abstract workgroups(): [number, number, number];

    async run(): Promise<number> {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: "high-performance",
      });
      if (!adapter) {
        alert("WebGPU not supported");
        return -1;
      }

      let device: GPUDevice;
      try {
        device = await adapter.requestDevice({
          requiredLimits: {
            maxComputeInvocationsPerWorkgroup:
              adapter.limits.maxComputeInvocationsPerWorkgroup,
            maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
            maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
            maxComputeWorkgroupSizeZ: adapter.limits.maxComputeWorkgroupSizeZ,
            maxComputeWorkgroupStorageSize:
              adapter.limits.maxComputeWorkgroupStorageSize,
            maxComputeWorkgroupsPerDimension:
              adapter.limits.maxComputeWorkgroupsPerDimension,
            maxStorageBufferBindingSize:
              adapter.limits.maxStorageBufferBindingSize,
          },
        });
      } catch (error) {
        console.warn("Error when creating device:", error);
        return -1;
      }

      const usage =
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST;
      const a = device.createBuffer({ size: n * n * 4, usage });
      const b = device.createBuffer({ size: n * n * 4, usage });
      const c = device.createBuffer({ size: n * n * 4, usage });
      const staging = device.createBuffer({
        size: n * n * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

      device.queue.writeBuffer(a, 0, randomBuffer);
      device.queue.writeBuffer(b, 0, randomBuffer);

      try {
        const pipeline = await device.createComputePipelineAsync({
          compute: {
            module: device.createShaderModule({ code: this.kernel() }),
            entryPoint: "main",
          },
          layout: "auto",
        });

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: a } },
            { binding: 1, resource: { buffer: b } },
            { binding: 2, resource: { buffer: c } },
          ],
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(...this.workgroups());
        passEncoder.end();
        commandEncoder.copyBufferToBuffer(c, 0, staging, 0, n * n * 4);
        device.queue.submit([commandEncoder.finish()]);

        const start = performance.now();

        await staging.mapAsync(GPUMapMode.READ, 0, n * n * 4);
        const buf = new Float32Array(staging.getMappedRange());
        printBufferItems(buf);
        staging.unmap(); // Do not need to actually read it.

        return (performance.now() - start) / 1000;
      } finally {
        a.destroy();
        b.destroy();
        c.destroy();
        staging.destroy();
      }
    }
  }

  class NaiveStrategy extends GpuStrategy {
    name: string;
    blocksize: number;

    constructor(block: number) {
      super();
      this.name = `naive-${block}`;
      this.blocksize = block;
    }

    kernel() {
      return `
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;

// Define the dimensions for the square matrices.
const DIM : u32 = ${n}u;

// Helper function for reading from matrix A.
fn mm_readA(row: u32, col: u32) -> f32 {
  // Matrix A is stored in row-major order.
  return A[row * DIM + col];
}

// Helper function for reading from matrix B.
fn mm_readB(row: u32, col: u32) -> f32 {
  // Matrix B is stored in row-major order.
  return B[row * DIM + col];
}

// Helper function for writing the computed result to matrix C.
fn mm_write(row: u32, col: u32, value: f32) {
  C[row * DIM + col] = value;
}

@compute @workgroup_size(${this.blocksize}, ${this.blocksize}, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  // Compute the output coordinates.
  let row: u32 = global_id.y;
  let col: u32 = global_id.x;

  // If the workgroup size oversubscribes the matrix dimensions, ensure we don't access out of bounds.
  if (row >= DIM || col >= DIM) {
    return;
  }

  var sum: f32 = 0.0;
  // Compute the dot product for the (row, col) position.
  for (var k: u32 = 0u; k < DIM; k = k + 1u) {
    sum = sum + mm_readA(row, k) * mm_readB(k, col);
  }

  // Write the computed sum to the output matrix.
  mm_write(row, col, sum);
}
`;
    }

    workgroups(): [number, number, number] {
      return [n / this.blocksize, n / this.blocksize, 1];
    }
  }

  class ShmemTilingStrategy extends GpuStrategy {
    name: string;
    blocksize: number;

    constructor(block: number) {
      super();
      this.name = `shmem-tiling-${block}`;
      this.blocksize = block;
    }

    kernel() {
      return `
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;

// Declare workgroup (shared) memory tiles for A and B.
var<workgroup> Asub: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> Bsub: array<array<f32, TILE_SIZE>, TILE_SIZE>;

// Define matrix dimensions.
const M : u32 = ${n}u;  // Number of rows of A and C.
const N : u32 = ${n}u;  // Number of columns of B and C.
const K : u32 = ${n}u;  // Number of columns of A and rows of B.
const TILE_SIZE : u32 = ${this.blocksize}u; // Workgroup tile size.

// Helper functions similar to tfjs codegen.
fn mm_readA(row: u32, col: u32) -> f32 {
  // Reads the element at A[row, col].
  return A[row * K + col];
}

fn mm_readB(row: u32, col: u32) -> f32 {
  // Reads the element at B[row, col].
  return B[row * N + col];
}

fn mm_write(row: u32, col: u32, value: f32) {
  // Writes the computed value to C[row, col].
  C[row * N + col] = value;
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(global_invocation_id) global_id: vec3<u32>) {
  // Compute the output coordinate [row, col] for this thread.
  let row: u32 = global_id.y;
  let col: u32 = global_id.x;

  var sum: f32 = 0.0;

  for (var t: u32 = 0u; t < (K / TILE_SIZE); t = t + 1u) {
    // Each thread loads one element of A and one element of B into shared memory.
    // For A, load element at (row, t*TILE_SIZE + local_x).
    let aRow: u32 = row;
    let aCol: u32 = t * TILE_SIZE + local_id.x;
    Asub[local_id.y][local_id.x] = mm_readA(aRow, aCol);

    // For B, load element at (t*TILE_SIZE + local_y, col).
    let bRow: u32 = t * TILE_SIZE + local_id.y;
    let bCol: u32 = col;
    Bsub[local_id.y][local_id.x] = mm_readB(bRow, bCol);

    // Ensure the full tile has been loaded before computation.
    workgroupBarrier();

    // Compute the partial dot product for the tile.
    for (var k_inner: u32 = 0u; k_inner < TILE_SIZE; k_inner = k_inner + 1u) {
      sum = sum + Asub[local_id.y][k_inner] * Bsub[k_inner][local_id.x];
    }

    // Synchronize before loading the next tile.
    workgroupBarrier();
  }

  // Write the computed value to the output matrix C.
  mm_write(row, col, sum);
}`;
    }

    workgroups(): [number, number, number] {
      return [n / this.blocksize, n / this.blocksize, 1];
    }
  }

  class Unroll4Strategy extends GpuStrategy {
    name: string;
    bx: number;
    by: number;

    constructor(bx: number, by: number) {
      super();
      this.name = `unroll4-${bx}-${by}`;
      this.bx = bx;
      this.by = by;
    }

    kernel() {
      return `@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;

// Define the dimensions for the square matrices.
const DIM : u32 = ${n}u;

// Helper function for reading from matrix A (row-major order).
fn mm_readA(row: u32, col: u32) -> f32 {
  return A[row * DIM + col];
}

// Helper function for reading from matrix B (row-major order).
fn mm_readB(row: u32, col: u32) -> f32 {
  return B[row * DIM + col];
}

// Helper function for writing to matrix C.
fn mm_write(row: u32, col: u32, value: f32) {
  C[row * DIM + col] = value;
}

@compute @workgroup_size(${this.bx}, ${this.by}, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  // Each thread now computes a 4x4 block.
  let base_row: u32 = global_id.y * 4u;
  let base_col: u32 = global_id.x * 4u;

  if (base_row >= DIM || base_col >= DIM) {
    return;
  }

  // Initialize the 4x4 accumulators.
  var sum00: f32 = 0.0; var sum01: f32 = 0.0; var sum02: f32 = 0.0; var sum03: f32 = 0.0;
  var sum10: f32 = 0.0; var sum11: f32 = 0.0; var sum12: f32 = 0.0; var sum13: f32 = 0.0;
  var sum20: f32 = 0.0; var sum21: f32 = 0.0; var sum22: f32 = 0.0; var sum23: f32 = 0.0;
  var sum30: f32 = 0.0; var sum31: f32 = 0.0; var sum32: f32 = 0.0; var sum33: f32 = 0.0;

  // Loop over k dimension.
  for (var k: u32 = 0u; k < DIM; k = k + 1u) {
    // Load 4 elements from A: one for each row of the 4x4 block.
    let a0: f32 = mm_readA(base_row + 0u, k);
    let a1: f32 = mm_readA(base_row + 1u, k);
    let a2: f32 = mm_readA(base_row + 2u, k);
    let a3: f32 = mm_readA(base_row + 3u, k);

    // Load 4 elements from B: one for each column of the 4x4 block.
    let b0: f32 = mm_readB(k, base_col + 0u);
    let b1: f32 = mm_readB(k, base_col + 1u);
    let b2: f32 = mm_readB(k, base_col + 2u);
    let b3: f32 = mm_readB(k, base_col + 3u);

    // Accumulate the product for all 4x4 entries.
    sum00 = sum00 + a0 * b0;
    sum01 = sum01 + a0 * b1;
    sum02 = sum02 + a0 * b2;
    sum03 = sum03 + a0 * b3;

    sum10 = sum10 + a1 * b0;
    sum11 = sum11 + a1 * b1;
    sum12 = sum12 + a1 * b2;
    sum13 = sum13 + a1 * b3;

    sum20 = sum20 + a2 * b0;
    sum21 = sum21 + a2 * b1;
    sum22 = sum22 + a2 * b2;
    sum23 = sum23 + a2 * b3;

    sum30 = sum30 + a3 * b0;
    sum31 = sum31 + a3 * b1;
    sum32 = sum32 + a3 * b2;
    sum33 = sum33 + a3 * b3;
  }

  // Write out the 4x4 block to matrix C.
  mm_write(base_row + 0u, base_col + 0u, sum00);
  mm_write(base_row + 0u, base_col + 1u, sum01);
  mm_write(base_row + 0u, base_col + 2u, sum02);
  mm_write(base_row + 0u, base_col + 3u, sum03);

  mm_write(base_row + 1u, base_col + 0u, sum10);
  mm_write(base_row + 1u, base_col + 1u, sum11);
  mm_write(base_row + 1u, base_col + 2u, sum12);
  mm_write(base_row + 1u, base_col + 3u, sum13);

  mm_write(base_row + 2u, base_col + 0u, sum20);
  mm_write(base_row + 2u, base_col + 1u, sum21);
  mm_write(base_row + 2u, base_col + 2u, sum22);
  mm_write(base_row + 2u, base_col + 3u, sum23);

  mm_write(base_row + 3u, base_col + 0u, sum30);
  mm_write(base_row + 3u, base_col + 1u, sum31);
  mm_write(base_row + 3u, base_col + 2u, sum32);
  mm_write(base_row + 3u, base_col + 3u, sum33);
}
`;
    }

    workgroups(): [number, number, number] {
      return [n / this.bx, n / this.by, 1];
    }
  }

  class Unroll4x2Strategy extends GpuStrategy {
    name: string;
    bx: number;
    by: number;

    constructor(bx: number, by: number) {
      super();
      this.name = `unroll4x2-${bx}-${by}`;
      this.bx = bx;
      this.by = by;
    }

    kernel() {
      return `
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;

// Define the dimensions for the square matrices.
const DIM : u32 = ${n}u;

// Helper function for reading from matrix A (row-major order).
fn mm_readA(row: u32, col: u32) -> f32 {
  return A[row * DIM + col];
}

// Helper function for reading from matrix B (row-major order).
fn mm_readB(row: u32, col: u32) -> f32 {
  return B[row * DIM + col];
}

// Helper function for writing to matrix C.
fn mm_write(row: u32, col: u32, value: f32) {
  C[row * DIM + col] = value;
}

@compute @workgroup_size(${this.bx}, ${this.by}, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  // Each thread computes a 4x4 block.
  let base_row: u32 = global_id.y * 4u;
  let base_col: u32 = global_id.x * 4u;

  // Bounds check (if DIM is not a multiple of 4 you may need extra checks).
  if (base_row >= DIM || base_col >= DIM) {
    return;
  }

  // Initialize the 4x4 accumulators.
  var sum00: f32 = 0.0; var sum01: f32 = 0.0; var sum02: f32 = 0.0; var sum03: f32 = 0.0;
  var sum10: f32 = 0.0; var sum11: f32 = 0.0; var sum12: f32 = 0.0; var sum13: f32 = 0.0;
  var sum20: f32 = 0.0; var sum21: f32 = 0.0; var sum22: f32 = 0.0; var sum23: f32 = 0.0;
  var sum30: f32 = 0.0; var sum31: f32 = 0.0; var sum32: f32 = 0.0; var sum33: f32 = 0.0;

  // Loop over the k dimension, unrolled by 2.
  for (var k: u32 = 0u; k < DIM; k = k + 2u) {
    // Load two elements from A for each row of the 4x4 block.
    let a0_0: f32 = mm_readA(base_row + 0u, k + 0u);
    let a0_1: f32 = mm_readA(base_row + 0u, k + 1u);

    let a1_0: f32 = mm_readA(base_row + 1u, k + 0u);
    let a1_1: f32 = mm_readA(base_row + 1u, k + 1u);

    let a2_0: f32 = mm_readA(base_row + 2u, k + 0u);
    let a2_1: f32 = mm_readA(base_row + 2u, k + 1u);

    let a3_0: f32 = mm_readA(base_row + 3u, k + 0u);
    let a3_1: f32 = mm_readA(base_row + 3u, k + 1u);

    // Load two elements from B for each column of the 4x4 block.
    let b0_0: f32 = mm_readB(k + 0u, base_col + 0u);
    let b0_1: f32 = mm_readB(k + 0u, base_col + 1u);
    let b0_2: f32 = mm_readB(k + 0u, base_col + 2u);
    let b0_3: f32 = mm_readB(k + 0u, base_col + 3u);

    let b1_0: f32 = mm_readB(k + 1u, base_col + 0u);
    let b1_1: f32 = mm_readB(k + 1u, base_col + 1u);
    let b1_2: f32 = mm_readB(k + 1u, base_col + 2u);
    let b1_3: f32 = mm_readB(k + 1u, base_col + 3u);

    // Unrolled accumulator updates: row 0.
    sum00 = sum00 + a0_0 * b0_0 + a0_1 * b1_0;
    sum01 = sum01 + a0_0 * b0_1 + a0_1 * b1_1;
    sum02 = sum02 + a0_0 * b0_2 + a0_1 * b1_2;
    sum03 = sum03 + a0_0 * b0_3 + a0_1 * b1_3;

    // Row 1.
    sum10 = sum10 + a1_0 * b0_0 + a1_1 * b1_0;
    sum11 = sum11 + a1_0 * b0_1 + a1_1 * b1_1;
    sum12 = sum12 + a1_0 * b0_2 + a1_1 * b1_2;
    sum13 = sum13 + a1_0 * b0_3 + a1_1 * b1_3;

    // Row 2.
    sum20 = sum20 + a2_0 * b0_0 + a2_1 * b1_0;
    sum21 = sum21 + a2_0 * b0_1 + a2_1 * b1_1;
    sum22 = sum22 + a2_0 * b0_2 + a2_1 * b1_2;
    sum23 = sum23 + a2_0 * b0_3 + a2_1 * b1_3;

    // Row 3.
    sum30 = sum30 + a3_0 * b0_0 + a3_1 * b1_0;
    sum31 = sum31 + a3_0 * b0_1 + a3_1 * b1_1;
    sum32 = sum32 + a3_0 * b0_2 + a3_1 * b1_2;
    sum33 = sum33 + a3_0 * b0_3 + a3_1 * b1_3;
  }

  // Write out the computed 4x4 block to matrix C.
  mm_write(base_row + 0u, base_col + 0u, sum00);
  mm_write(base_row + 0u, base_col + 1u, sum01);
  mm_write(base_row + 0u, base_col + 2u, sum02);
  mm_write(base_row + 0u, base_col + 3u, sum03);

  mm_write(base_row + 1u, base_col + 0u, sum10);
  mm_write(base_row + 1u, base_col + 1u, sum11);
  mm_write(base_row + 1u, base_col + 2u, sum12);
  mm_write(base_row + 1u, base_col + 3u, sum13);

  mm_write(base_row + 2u, base_col + 0u, sum20);
  mm_write(base_row + 2u, base_col + 1u, sum21);
  mm_write(base_row + 2u, base_col + 2u, sum22);
  mm_write(base_row + 2u, base_col + 3u, sum23);

  mm_write(base_row + 3u, base_col + 0u, sum30);
  mm_write(base_row + 3u, base_col + 1u, sum31);
  mm_write(base_row + 3u, base_col + 2u, sum32);
  mm_write(base_row + 3u, base_col + 3u, sum33);
}
`;
    }

    workgroups(): [number, number, number] {
      return [n / this.bx, n / this.by, 1];
    }
  }

  class Unroll4x4Strategy extends GpuStrategy {
    name: string;
    bx: number;
    by: number;

    constructor(bx: number, by: number) {
      super();
      this.name = `unroll4x4-${bx}-${by}`;
      this.bx = bx;
      this.by = by;
    }

    kernel() {
      return `
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> B : array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;

// Define the dimensions for the square matrices.
const DIM : u32 = ${n}u;

// Helper function for reading from matrix A (row-major order).
fn mm_readA(row: u32, col: u32) -> f32 {
  return A[row * DIM + col];
}

// Helper function for reading from matrix B (row-major order).
fn mm_readB(row: u32, col: u32) -> f32 {
  return B[row * DIM + col];
}

// Helper function for writing to matrix C.
fn mm_write(row: u32, col: u32, value: f32) {
  C[row * DIM + col] = value;
}

@compute @workgroup_size(${this.bx}, ${this.by}, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  // Each thread now computes a 4x4 block.
  let base_row: u32 = global_id.y * 4u;
  let base_col: u32 = global_id.x * 4u;

  // Bounds check (assuming DIM is a multiple of 4, this is safe; otherwise you may
  // need per-element checks when writing the 4x4 block).
  if (base_row >= DIM || base_col >= DIM) {
    return;
  }

  // Initialize the 4x4 accumulators.
  var sum00: f32 = 0.0; var sum01: f32 = 0.0; var sum02: f32 = 0.0; var sum03: f32 = 0.0;
  var sum10: f32 = 0.0; var sum11: f32 = 0.0; var sum12: f32 = 0.0; var sum13: f32 = 0.0;
  var sum20: f32 = 0.0; var sum21: f32 = 0.0; var sum22: f32 = 0.0; var sum23: f32 = 0.0;
  var sum30: f32 = 0.0; var sum31: f32 = 0.0; var sum32: f32 = 0.0; var sum33: f32 = 0.0;

  // Loop over k dimension, unrolled by 4.
  for (var k: u32 = 0u; k < DIM; k = k + 4u) {
    // Load 4 elements from A for each of the 4 rows.
    let a0_0: f32 = mm_readA(base_row + 0u, k + 0u);
    let a0_1: f32 = mm_readA(base_row + 0u, k + 1u);
    let a0_2: f32 = mm_readA(base_row + 0u, k + 2u);
    let a0_3: f32 = mm_readA(base_row + 0u, k + 3u);

    let a1_0: f32 = mm_readA(base_row + 1u, k + 0u);
    let a1_1: f32 = mm_readA(base_row + 1u, k + 1u);
    let a1_2: f32 = mm_readA(base_row + 1u, k + 2u);
    let a1_3: f32 = mm_readA(base_row + 1u, k + 3u);

    let a2_0: f32 = mm_readA(base_row + 2u, k + 0u);
    let a2_1: f32 = mm_readA(base_row + 2u, k + 1u);
    let a2_2: f32 = mm_readA(base_row + 2u, k + 2u);
    let a2_3: f32 = mm_readA(base_row + 2u, k + 3u);

    let a3_0: f32 = mm_readA(base_row + 3u, k + 0u);
    let a3_1: f32 = mm_readA(base_row + 3u, k + 1u);
    let a3_2: f32 = mm_readA(base_row + 3u, k + 2u);
    let a3_3: f32 = mm_readA(base_row + 3u, k + 3u);

    // Load 4 elements from B for each of the 4 columns.
    let b0_0: f32 = mm_readB(k + 0u, base_col + 0u);
    let b0_1: f32 = mm_readB(k + 0u, base_col + 1u);
    let b0_2: f32 = mm_readB(k + 0u, base_col + 2u);
    let b0_3: f32 = mm_readB(k + 0u, base_col + 3u);

    let b1_0: f32 = mm_readB(k + 1u, base_col + 0u);
    let b1_1: f32 = mm_readB(k + 1u, base_col + 1u);
    let b1_2: f32 = mm_readB(k + 1u, base_col + 2u);
    let b1_3: f32 = mm_readB(k + 1u, base_col + 3u);

    let b2_0: f32 = mm_readB(k + 2u, base_col + 0u);
    let b2_1: f32 = mm_readB(k + 2u, base_col + 1u);
    let b2_2: f32 = mm_readB(k + 2u, base_col + 2u);
    let b2_3: f32 = mm_readB(k + 2u, base_col + 3u);

    let b3_0: f32 = mm_readB(k + 3u, base_col + 0u);
    let b3_1: f32 = mm_readB(k + 3u, base_col + 1u);
    let b3_2: f32 = mm_readB(k + 3u, base_col + 2u);
    let b3_3: f32 = mm_readB(k + 3u, base_col + 3u);

    // Unrolled accumulation for the 4x4 block.
    sum00 = sum00 + a0_0 * b0_0 + a0_1 * b1_0 + a0_2 * b2_0 + a0_3 * b3_0;
    sum01 = sum01 + a0_0 * b0_1 + a0_1 * b1_1 + a0_2 * b2_1 + a0_3 * b3_1;
    sum02 = sum02 + a0_0 * b0_2 + a0_1 * b1_2 + a0_2 * b2_2 + a0_3 * b3_2;
    sum03 = sum03 + a0_0 * b0_3 + a0_1 * b1_3 + a0_2 * b2_3 + a0_3 * b3_3;

    sum10 = sum10 + a1_0 * b0_0 + a1_1 * b1_0 + a1_2 * b2_0 + a1_3 * b3_0;
    sum11 = sum11 + a1_0 * b0_1 + a1_1 * b1_1 + a1_2 * b2_1 + a1_3 * b3_1;
    sum12 = sum12 + a1_0 * b0_2 + a1_1 * b1_2 + a1_2 * b2_2 + a1_3 * b3_2;
    sum13 = sum13 + a1_0 * b0_3 + a1_1 * b1_3 + a1_2 * b2_3 + a1_3 * b3_3;

    sum20 = sum20 + a2_0 * b0_0 + a2_1 * b1_0 + a2_2 * b2_0 + a2_3 * b3_0;
    sum21 = sum21 + a2_0 * b0_1 + a2_1 * b1_1 + a2_2 * b2_1 + a2_3 * b3_1;
    sum22 = sum22 + a2_0 * b0_2 + a2_1 * b1_2 + a2_2 * b2_2 + a2_3 * b3_2;
    sum23 = sum23 + a2_0 * b0_3 + a2_1 * b1_3 + a2_2 * b2_3 + a2_3 * b3_3;

    sum30 = sum30 + a3_0 * b0_0 + a3_1 * b1_0 + a3_2 * b2_0 + a3_3 * b3_0;
    sum31 = sum31 + a3_0 * b0_1 + a3_1 * b1_1 + a3_2 * b2_1 + a3_3 * b3_1;
    sum32 = sum32 + a3_0 * b0_2 + a3_1 * b1_2 + a3_2 * b2_2 + a3_3 * b3_2;
    sum33 = sum33 + a3_0 * b0_3 + a3_1 * b1_3 + a3_2 * b2_3 + a3_3 * b3_3;
  }

  // Write out the computed 4x4 block to matrix C.
  mm_write(base_row + 0u, base_col + 0u, sum00);
  mm_write(base_row + 0u, base_col + 1u, sum01);
  mm_write(base_row + 0u, base_col + 2u, sum02);
  mm_write(base_row + 0u, base_col + 3u, sum03);

  mm_write(base_row + 1u, base_col + 0u, sum10);
  mm_write(base_row + 1u, base_col + 1u, sum11);
  mm_write(base_row + 1u, base_col + 2u, sum12);
  mm_write(base_row + 1u, base_col + 3u, sum13);

  mm_write(base_row + 2u, base_col + 0u, sum20);
  mm_write(base_row + 2u, base_col + 1u, sum21);
  mm_write(base_row + 2u, base_col + 2u, sum22);
  mm_write(base_row + 2u, base_col + 3u, sum23);

  mm_write(base_row + 3u, base_col + 0u, sum30);
  mm_write(base_row + 3u, base_col + 1u, sum31);
  mm_write(base_row + 3u, base_col + 2u, sum32);
  mm_write(base_row + 3u, base_col + 3u, sum33);
}
`;
    }

    workgroups(): [number, number, number] {
      return [n / this.bx, n / this.by, 1];
    }
  }

  class TfjsStrategy extends Strategy {
    name = "tfjs";

    async run(): Promise<number> {
      const tf = await import("@tensorflow/tfjs");
      await import("@tensorflow/tfjs-backend-webgpu");
      await tf.setBackend("webgpu");

      const a = tf.tensor(randomBuffer, [n, n]);
      const b = tf.tensor(randomBuffer, [n, n]);
      await Promise.all([a.data(), b.data()]); // Make sure tensors are ready.

      performance.mark("tfjs-start");
      const start = performance.now();
      const c = tf.matMul(a, b);
      const ar = (await c.data()) as Float32Array;
      printBufferItems(ar);
      const time = performance.now() - start;
      performance.mark("tfjs-end");
      performance.measure("tfjs", "tfjs-start", "tfjs-end");

      a.dispose();
      b.dispose();
      c.dispose();

      return time / 1000; // seconds
    }
  }

  class OnnxStrategy extends Strategy {
    name: string;
    dtype: "fp16" | "fp32";

    constructor(dtype: "fp16" | "fp32") {
      super();
      this.name = `onnx-${dtype}`;
      this.dtype = dtype;
    }

    // Helper function to create a simple ONNX model with just a MatMul operation
    async createMatMulModel(dim: number): Promise<Uint8Array> {
      const { onnx } = await import("onnx-proto");

      const elemType = {
        fp32: onnx.TensorProto.DataType.FLOAT,
        fp16: onnx.TensorProto.DataType.FLOAT16,
      }[this.dtype];

      // Create input tensors
      const input1 = onnx.ValueInfoProto.create({
        name: "A",
        type: onnx.TypeProto.create({
          tensorType: onnx.TypeProto.Tensor.create({
            elemType,
            shape: onnx.TensorShapeProto.create({
              dim: [
                onnx.TensorShapeProto.Dimension.create({ dimValue: dim }),
                onnx.TensorShapeProto.Dimension.create({ dimValue: dim }),
              ],
            }),
          }),
        }),
      });

      const input2 = onnx.ValueInfoProto.create({
        name: "B",
        type: onnx.TypeProto.create({
          tensorType: onnx.TypeProto.Tensor.create({
            elemType,
            shape: onnx.TensorShapeProto.create({
              dim: [
                onnx.TensorShapeProto.Dimension.create({ dimValue: dim }),
                onnx.TensorShapeProto.Dimension.create({ dimValue: dim }),
              ],
            }),
          }),
        }),
      });

      // Create output tensor
      const output = onnx.ValueInfoProto.create({
        name: "C",
        type: onnx.TypeProto.create({
          tensorType: onnx.TypeProto.Tensor.create({
            elemType,
            shape: onnx.TensorShapeProto.create({
              dim: [
                onnx.TensorShapeProto.Dimension.create({ dimValue: dim }),
                onnx.TensorShapeProto.Dimension.create({ dimValue: dim }),
              ],
            }),
          }),
        }),
      });

      // Create MatMul node
      const matmulNode = onnx.NodeProto.create({
        input: ["A", "B"],
        output: ["C"],
        opType: "MatMul",
        name: "matmul_node",
      });

      // Create the graph
      const graph = onnx.GraphProto.create({
        node: [matmulNode],
        name: "matmul_graph",
        input: [input1, input2],
        output: [output],
      });

      // Create the model
      const model = onnx.ModelProto.create({
        irVersion: 8,
        opsetImport: [onnx.OperatorSetIdProto.create({ version: 14 })],
        graph: graph,
      });

      // Serialize to bytes
      return onnx.ModelProto.encode(model).finish();
    }

    async run(): Promise<number> {
      const ort = await import("onnxruntime-web/webgpu");
      let session: import("onnxruntime-web/webgpu").InferenceSession | null =
        null;

      try {
        const model = await this.createMatMulModel(n);
        session = await ort.InferenceSession.create(model, {
          executionProviders: ["webgpu"],
        });

        // Prepare input tensors
        let buffer: any;
        let ortType: any;
        if (this.dtype === "fp16") {
          buffer = new Float16Array(randomBuffer);
          ortType = "float16";
        } else {
          buffer = randomBuffer;
          ortType = "float32";
        }
        const tensorA = new ort.Tensor(ortType, buffer, [n, n]);
        const tensorB = new ort.Tensor(ortType, buffer, [n, n]);

        // Warm-up run to ensure everything is loaded
        await session.run({ A: tensorA, B: tensorB });

        // Actual benchmark run
        performance.mark("onnx-start");
        const start = performance.now();
        const results = await session.run({ A: tensorA, B: tensorB });
        const outputData = results.C.data as Float32Array | Float16Array;
        printBufferItems(outputData);
        const time = performance.now() - start;
        performance.mark("onnx-end");
        performance.measure("onnx", "onnx-start", "onnx-end");

        return time / 1000; // seconds
      } catch (error) {
        console.error("ONNX Runtime error:", error);
        return -1;
      } finally {
        // Clean up session resources
        if (session) {
          session.release();
        }
      }
    }
  }

  class JaxJsStrategy extends Strategy {
    name: string;
    fp16: boolean;

    constructor(fp16: boolean = false) {
      super();
      this.fp16 = fp16;
      this.name = fp16 ? "jax-js-fp16" : "jax-js";
    }

    async run(): Promise<number> {
      const jax = await import("@jax-js/jax");
      await jax.init();
      jax.defaultDevice("webgpu");
      const np = jax.numpy;

      const a = np
        .array(randomBuffer, { shape: [n, n] })
        .astype(this.fp16 ? np.float16 : np.float32);
      const b = np
        .array(randomBuffer, { shape: [n, n] })
        .astype(this.fp16 ? np.float16 : np.float32);
      await jax.blockUntilReady([a, b]); // Make sure tensors are ready.

      performance.mark("jax-start");
      const start = performance.now();
      const c = np.dot(a, b);
      const ar = (await c.data()) as Float16Array;
      printBufferItems(ar);
      const time = performance.now() - start;
      performance.mark("jax-end");
      performance.measure("jax", "jax-start", "jax-end");

      return time / 1000; // seconds
    }
  }

  const strategiesList: Strategy[] = [
    new NaiveStrategy(1),
    new NaiveStrategy(16),
    new NaiveStrategy(32),
    new ShmemTilingStrategy(16),
    new ShmemTilingStrategy(32),
    new Unroll4Strategy(8, 8),
    new Unroll4Strategy(8, 16),
    new Unroll4Strategy(16, 16),
    new Unroll4x2Strategy(8, 8),
    new Unroll4x2Strategy(8, 16),
    new Unroll4x2Strategy(16, 16),
    new Unroll4x4Strategy(8, 8),
    new Unroll4x4Strategy(8, 16),
    new Unroll4x4Strategy(16, 16),
    new OnnxStrategy("fp16"),
    new OnnxStrategy("fp32"),
    new TfjsStrategy(),
    new JaxJsStrategy(),
    new JaxJsStrategy(true),
  ];

  const strategies = Object.fromEntries(strategiesList.map((s) => [s.name, s]));

  async function bench(variant: string) {
    console.log(`Running ${variant}...`);
    const time = await strategies[variant].run();
    if (time >= 0) {
      result[variant] = time;
    } else {
      console.error(`Error running ${variant}`);
    }
  }
</script>

<main class="p-4">
  <h1 class="text-2xl mb-2">matmul benchmark</h1>

  <p class="mb-4">
    Running a few different WebGPU matmul programs on {n}x{n} matrices.
  </p>

  <div class="flex flex-wrap gap-2 mb-4">
    {#each strategiesList as strategy (strategy.name)}
      <button
        class="border px-2 hover:bg-gray-100 active:scale-95"
        onclick={() => bench(strategy.name)}
      >
        {strategy.name}
      </button>
    {/each}
  </div>

  {#if browser && !navigator.gpu}
    <p class="text-red-500 mb-4">
      WebGPU is not supported. Benchmarks will not work.
    </p>
  {/if}

  {#each Object.entries(result) as [variant, time]}
    <div>
      <span class="font-bold">{variant}:</span>
      {time.toFixed(3)} seconds,
      {((2 * n * n * n) / 1e9 / time).toFixed(2)} GFLOP/s
    </div>
  {/each}
</main>

<style lang="postcss">
  @reference "$app.css";

  button {
    @apply border rounded px-2 hover:bg-gray-100 active:scale-95;
  }
</style>
