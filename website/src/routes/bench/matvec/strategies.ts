import type { Device } from "@jax-js/jax";
import type tf from "@tensorflow/tfjs";

import {
  createRandomBuffer,
  getWebgpuDevice,
  importTfjs,
  logSampleBuffer,
  runBenchmark,
  type Strategy,
} from "$lib/benchmark";

export const n = 4096;
export const repeats = 20;
const matrixSize = n * n;

const randomMatrixBuffer = createRandomBuffer(matrixSize, true);
const randomVectorBuffer = createRandomBuffer(n, true);

abstract class GpuStrategy implements Strategy {
  abstract name: string;
  abstract kernel(): string;
  abstract workgroups(): [number, number, number];

  async run(): Promise<number> {
    const device = await getWebgpuDevice();

    const usage =
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST;
    const a = device.createBuffer({ size: matrixSize * 4, usage });
    const x = device.createBuffer({ size: n * 4, usage });
    const y = device.createBuffer({ size: n * 4, usage });
    const staging = device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    device.queue.writeBuffer(a, 0, randomMatrixBuffer);
    device.queue.writeBuffer(x, 0, randomVectorBuffer);

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
          { binding: 1, resource: { buffer: x } },
          { binding: 2, resource: { buffer: y } },
        ],
      });

      return await runBenchmark("webgpu", async () => {
        for (let i = 0; i < repeats; i++) {
          const commandEncoder = device.createCommandEncoder();
          const passEncoder = commandEncoder.beginComputePass();
          passEncoder.setPipeline(pipeline);
          passEncoder.setBindGroup(0, bindGroup);
          passEncoder.dispatchWorkgroups(...this.workgroups());
          passEncoder.end();
          commandEncoder.copyBufferToBuffer(y, 0, staging, 0, n * 4);
          device.queue.submit([commandEncoder.finish()]);

          await staging.mapAsync(GPUMapMode.READ, 0, n * 4);
          if (i < repeats - 1) staging.unmap();
        }
        const buf = new Float32Array(staging.getMappedRange());
        logSampleBuffer(buf);
        staging.unmap();
      });
    } finally {
      a.destroy();
      x.destroy();
      y.destroy();
      staging.destroy();
    }
  }
}

class NaiveStrategy extends GpuStrategy {
  name: string;
  blocksize: number;

  constructor(blocksize: number) {
    super();
    this.name = `naive-${blocksize}`;
    this.blocksize = blocksize;
  }

  kernel() {
    return `
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> X : array<f32>;
@group(0) @binding(2) var<storage, read_write> Y : array<f32>;

const M : u32 = ${n}u;
const N : u32 = ${n}u;

@compute @workgroup_size(${this.blocksize}, 1, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let row: u32 = global_id.x;
  if (row >= M) {
    return;
  }

  let rowBase: u32 = row * N;
  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < N; k = k + 1u) {
    sum = sum + A[rowBase + k] * X[k];
  }
  Y[row] = sum;
}
`;
  }

  workgroups(): [number, number, number] {
    return [Math.ceil(n / this.blocksize), 1, 1];
  }
}

class UnrollStrategy extends GpuStrategy {
  name: string;
  blocksize: number;
  unroll: number;

  constructor(blocksize: number, unroll: number) {
    super();
    this.name = `unroll${unroll}-${blocksize}`;
    this.blocksize = blocksize;
    this.unroll = unroll;
  }

  kernel() {
    const unroll = this.unroll;
    const loads = [...new Array(unroll)]
      .map((_, i) => `    let x${i}: f32 = X[k + ${i}u];`)
      .join("\n");
    const fmas = [...new Array(unroll)]
      .map((_, i) => `    sum = sum + A[rowBase + k + ${i}u] * x${i};`)
      .join("\n");
    const tail = [...new Array(unroll)]
      .map(
        (_, i) => `    if (k + ${i}u < N) {
      sum = sum + A[rowBase + k + ${i}u] * X[k + ${i}u];
    }`,
      )
      .join("\n");

    return `
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> X : array<f32>;
@group(0) @binding(2) var<storage, read_write> Y : array<f32>;

const M : u32 = ${n}u;
const N : u32 = ${n}u;

@compute @workgroup_size(${this.blocksize}, 1, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let row: u32 = global_id.x;
  if (row >= M) {
    return;
  }

  let rowBase: u32 = row * N;
  var sum: f32 = 0.0;
  let mainEnd: u32 = N / ${unroll}u * ${unroll}u;

  for (var k: u32 = 0u; k < mainEnd; k = k + ${unroll}u) {
${loads}
${fmas}
  }

  for (var k: u32 = mainEnd; k < N; k = k + ${unroll}u) {
${tail}
  }

  Y[row] = sum;
}
`;
  }

  workgroups(): [number, number, number] {
    return [Math.ceil(n / this.blocksize), 1, 1];
  }
}

class ShmemVectorTilingStrategy extends GpuStrategy {
  name: string;
  tileCols: number;
  rowsPerGroup: number;

  constructor(tileCols: number, rowsPerGroup: number) {
    super();
    this.name = `shmem-${tileCols}x${rowsPerGroup}`;
    this.tileCols = tileCols;
    this.rowsPerGroup = rowsPerGroup;
  }

  kernel() {
    return `
@group(0) @binding(0) var<storage, read> A : array<f32>;
@group(0) @binding(1) var<storage, read> X : array<f32>;
@group(0) @binding(2) var<storage, read_write> Y : array<f32>;

const M : u32 = ${n}u;
const N : u32 = ${n}u;
const TILE_COLS : u32 = ${this.tileCols}u;
const ROWS_PER_GROUP : u32 = ${this.rowsPerGroup}u;

var<workgroup> xTile: array<f32, TILE_COLS>;
var<workgroup> partialSums: array<array<f32, TILE_COLS>, ROWS_PER_GROUP>;

@compute @workgroup_size(${this.tileCols}, ${this.rowsPerGroup}, 1)
fn main(
  @builtin(workgroup_id) workgroup_id : vec3<u32>,
  @builtin(local_invocation_id) local_id : vec3<u32>,
) {
  let lane: u32 = local_id.x;
  let rowInGroup: u32 = local_id.y;
  let row: u32 = workgroup_id.x * ROWS_PER_GROUP + rowInGroup;
  let rowBase: u32 = row * N;

  var sum: f32 = 0.0;

  for (var kBase: u32 = 0u; kBase < N; kBase = kBase + TILE_COLS) {
    if (rowInGroup == 0u) {
      if (kBase + lane < N) {
        xTile[lane] = X[kBase + lane];
      } else {
        xTile[lane] = 0.0;
      }
    }
    workgroupBarrier();

    if (row < M && kBase + lane < N) {
      sum = sum + A[rowBase + kBase + lane] * xTile[lane];
    }
    workgroupBarrier();
  }

  partialSums[rowInGroup][lane] = sum;
  workgroupBarrier();

  var stride: u32 = TILE_COLS / 2u;
  loop {
    if (stride == 0u) {
      break;
    }
    if (lane < stride) {
      partialSums[rowInGroup][lane] =
        partialSums[rowInGroup][lane] + partialSums[rowInGroup][lane + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (row < M && lane == 0u) {
    Y[row] = partialSums[rowInGroup][0];
  }
}
`;
  }

  workgroups(): [number, number, number] {
    return [Math.ceil(n / this.rowsPerGroup), 1, 1];
  }
}

class TfjsStrategy implements Strategy {
  name: string;
  wasm: boolean;

  constructor(wasm = false) {
    this.name = wasm ? "tfjs-wasm" : "tfjs";
    this.wasm = wasm;
  }

  async run(): Promise<number> {
    const tf = await importTfjs(this.wasm ? "wasm" : "webgpu");

    const a = tf.tensor(randomMatrixBuffer, [n, n]);
    const x = tf.tensor(randomVectorBuffer, [n, 1]);
    await Promise.all([a.data(), x.data()]);

    try {
      return await runBenchmark("tfjs", async () => {
        let y: tf.Tensor | null = null;
        for (let i = 0; i < repeats; i++) {
          y?.dispose();
          y = tf.matMul(a, x);
        }
        const ar = (await y!.data()) as Float32Array;
        logSampleBuffer(ar);
        y!.dispose();
      });
    } finally {
      a.dispose();
      x.dispose();
    }
  }
}

class OnnxStrategy implements Strategy {
  name = "onnx";

  async createMatVecModel(dim: number): Promise<Uint8Array> {
    const { create, toBinary } = await import("@bufbuild/protobuf");
    const {
      GraphProtoSchema,
      ModelProtoSchema,
      NodeProtoSchema,
      OperatorSetIdProtoSchema,
      TensorProto_DataType,
      TensorShapeProto_DimensionSchema,
      TensorShapeProtoSchema,
      TypeProto_TensorSchema,
      TypeProtoSchema,
      ValueInfoProtoSchema,
    } = await import("onnx-buf");

    const dimension = (value: number | bigint) => {
      return create(TensorShapeProto_DimensionSchema, {
        value: {
          case: "dimValue",
          value: BigInt(value),
        },
      });
    };

    const matrix = create(ValueInfoProtoSchema, {
      name: "A",
      type: create(TypeProtoSchema, {
        value: {
          case: "tensorType",
          value: create(TypeProto_TensorSchema, {
            elemType: TensorProto_DataType.FLOAT,
            shape: create(TensorShapeProtoSchema, {
              dim: [dimension(dim), dimension(dim)],
            }),
          }),
        },
      }),
    });

    const vector = create(ValueInfoProtoSchema, {
      name: "X",
      type: create(TypeProtoSchema, {
        value: {
          case: "tensorType",
          value: create(TypeProto_TensorSchema, {
            elemType: TensorProto_DataType.FLOAT,
            shape: create(TensorShapeProtoSchema, {
              dim: [dimension(dim), dimension(1)],
            }),
          }),
        },
      }),
    });

    const output = create(ValueInfoProtoSchema, {
      name: "Y",
      type: create(TypeProtoSchema, {
        value: {
          case: "tensorType",
          value: create(TypeProto_TensorSchema, {
            elemType: TensorProto_DataType.FLOAT,
            shape: create(TensorShapeProtoSchema, {
              dim: [dimension(dim), dimension(1)],
            }),
          }),
        },
      }),
    });

    const matmulNode = create(NodeProtoSchema, {
      input: ["A", "X"],
      output: ["Y"],
      opType: "MatMul",
      name: "matvec_node",
    });

    const graph = create(GraphProtoSchema, {
      node: [matmulNode],
      name: "matvec_graph",
      input: [matrix, vector],
      output: [output],
    });

    const model = create(ModelProtoSchema, {
      irVersion: 8n,
      opsetImport: [create(OperatorSetIdProtoSchema, { version: 14n })],
      graph,
    });

    return toBinary(ModelProtoSchema, model);
  }

  async run(): Promise<number> {
    const ort = await import("onnxruntime-web/webgpu");
    let session: import("onnxruntime-web/webgpu").InferenceSession | null =
      null;

    try {
      const model = await this.createMatVecModel(n);
      session = await ort.InferenceSession.create(model, {
        executionProviders: ["webgpu"],
      });

      const tensorA = new ort.Tensor("float32", randomMatrixBuffer, [n, n]);
      const tensorX = new ort.Tensor("float32", randomVectorBuffer, [n, 1]);

      return await runBenchmark("onnx", async () => {
        let results: Record<
          string,
          import("onnxruntime-web/webgpu").Tensor
        > | null = null;
        for (let i = 0; i < repeats; i++) {
          results = await session!.run({ A: tensorA, X: tensorX });
        }
        const outputData = results!.Y.data as Float32Array;
        logSampleBuffer(outputData);
      });
    } catch (error) {
      console.error("ONNX Runtime error:", error);
      return -1;
    } finally {
      if (session) {
        session.release();
      }
    }
  }
}

class JaxJsStrategy implements Strategy {
  name: string;
  device: Device;

  constructor(device: Device = "webgpu") {
    this.device = device;
    this.name = "jax-js";
    if (device !== "webgpu") this.name += `-${device}`;
  }

  async run(): Promise<number> {
    const jax = await import("@jax-js/jax");
    await jax.init();
    jax.defaultDevice(this.device);
    const np = jax.numpy;

    const a = np.array(randomMatrixBuffer, { shape: [n, n] });
    const x = np.array(randomVectorBuffer);
    await jax.blockUntilReady([a, x]);

    try {
      return await runBenchmark("jax", async () => {
        let y: typeof a | null = null;
        for (let i = 0; i < repeats; i++) {
          y?.dispose();
          y = np.dot(a.ref, x.ref);
          await y.blockUntilReady();
        }
        const ar = (await y!.data()) as Float32Array;
        logSampleBuffer(ar);
      });
    } finally {
      a.dispose();
      x.dispose();
    }
  }
}

export const strategies: Strategy[] = [
  new NaiveStrategy(64),
  new NaiveStrategy(256),
  new UnrollStrategy(256, 4),
  new UnrollStrategy(256, 8),
  new ShmemVectorTilingStrategy(32, 4),
  new ShmemVectorTilingStrategy(32, 8),
  new ShmemVectorTilingStrategy(64, 4),
  new OnnxStrategy(),
  new TfjsStrategy(),
  new TfjsStrategy(true),
  new JaxJsStrategy("webgpu"),
  new JaxJsStrategy("wasm"),
];
