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

export const batchSize = 1;
export const channels = 64;
export const height = 256;
export const width = 256;
export const filterHeight = 3;
export const filterWidth = 3;
export const outChannels = 128;

const inputSize = batchSize * channels * height * width;
const filterSize = outChannels * channels * filterHeight * filterWidth;
const outputSize = batchSize * outChannels * height * width; // assuming same padding

const randomInput = createRandomBuffer(inputSize);
const randomFilter = createRandomBuffer(filterSize);

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
    const input = device.createBuffer({ size: inputSize * 4, usage });
    const filter = device.createBuffer({ size: filterSize * 4, usage });
    const output = device.createBuffer({ size: outputSize * 4, usage });
    const staging = device.createBuffer({
      size: outputSize * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    device.queue.writeBuffer(input, 0, randomInput);
    device.queue.writeBuffer(filter, 0, randomFilter);

    try {
      const pipeline = await device.createComputePipelineAsync({
        compute: {
          module: device.createShaderModule({ code: this.kernel() }),
          entryPoint: "main",
        },
        layout: "auto",
      });

      return await runBenchmark("webgpu", async () => {
        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: input } },
            { binding: 1, resource: { buffer: filter } },
            { binding: 2, resource: { buffer: output } },
          ],
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(...this.workgroups());
        passEncoder.end();
        commandEncoder.copyBufferToBuffer(
          output,
          0,
          staging,
          0,
          outputSize * 4,
        );
        device.queue.submit([commandEncoder.finish()]);

        await staging.mapAsync(GPUMapMode.READ, 0, outputSize * 4);
        const buf = new Float32Array(staging.getMappedRange());
        logSampleBuffer(buf);
        staging.unmap();
      });
    } finally {
      input.destroy();
      filter.destroy();
      output.destroy();
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
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> weights : array<f32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;

const BATCH_SIZE: u32 = ${batchSize}u;
const IN_CHANNELS: u32 = ${channels}u;
const HEIGHT: u32 = ${height}u;
const WIDTH: u32 = ${width}u;
const FILTER_HEIGHT: u32 = ${filterHeight}u;
const FILTER_WIDTH: u32 = ${filterWidth}u;
const OUT_CHANNELS: u32 = ${outChannels}u;

fn input_idx(b: u32, c: u32, h: u32, w: u32) -> u32 {
  return b * IN_CHANNELS * HEIGHT * WIDTH + c * HEIGHT * WIDTH + h * WIDTH + w;
}

fn weights_idx(oc: u32, ic: u32, fh: u32, fw: u32) -> u32 {
  return oc * IN_CHANNELS * FILTER_HEIGHT * FILTER_WIDTH + ic * FILTER_HEIGHT * FILTER_WIDTH + fh * FILTER_WIDTH + fw;
}

fn output_idx(b: u32, oc: u32, h: u32, w: u32) -> u32 {
  return b * OUT_CHANNELS * HEIGHT * WIDTH + oc * HEIGHT * WIDTH + h * WIDTH + w;
}

@compute @workgroup_size(${this.blocksize}, ${this.blocksize}, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let out_h: u32 = global_id.y;
  let out_w: u32 = global_id.x;
  let out_c: u32 = global_id.z;

  if (out_h >= HEIGHT || out_w >= WIDTH || out_c >= OUT_CHANNELS) {
    return;
  }

  for (var b: u32 = 0u; b < BATCH_SIZE; b = b + 1u) {
    var sum: f32 = 0.0;

    for (var ic: u32 = 0u; ic < IN_CHANNELS; ic = ic + 1u) {
      for (var fh: u32 = 0u; fh < FILTER_HEIGHT; fh = fh + 1u) {
        for (var fw: u32 = 0u; fw < FILTER_WIDTH; fw = fw + 1u) {
          let in_h: i32 = i32(out_h) + i32(fh) - i32(FILTER_HEIGHT / 2u);
          let in_w: i32 = i32(out_w) + i32(fw) - i32(FILTER_WIDTH / 2u);

          if (in_h >= 0 && in_h < i32(HEIGHT) && in_w >= 0 && in_w < i32(WIDTH)) {
            let input_val: f32 = input[input_idx(b, ic, u32(in_h), u32(in_w))];
            let weights_val: f32 = weights[weights_idx(out_c, ic, fh, fw)];
            sum = sum + input_val * weights_val;
          }
        }
      }
    }

    output[output_idx(b, out_c, out_h, out_w)] = sum;
  }
}
`;
  }

  workgroups(): [number, number, number] {
    return [
      Math.ceil(width / this.blocksize),
      Math.ceil(height / this.blocksize),
      outChannels,
    ];
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

    // Use shared random data with NCHW format for input.
    //
    // However, even though tfjs has a "NCHW" format in their documentation,
    // it appears to produce generate invalid kernels in their WebGPU backend
    // as the output is wrong. Probably just a bug in the tfjs-backend-webgpu,
    // since it works in tfjs-backend-webgl (but that is much slower).
    //
    // That's not an issue though, since we can just transpose the input and
    // output in lieu of debugging tfjs to find a fix.
    const input = tf
      .tensor4d(randomInput, [batchSize, channels, height, width])
      .transpose<tf.Tensor4D>([0, 2, 3, 1]); // NHWC format

    // Convert filter from OIHW to HWIO format using transpose
    const filterOIHW = tf.tensor4d(randomFilter, [
      outChannels,
      channels,
      filterHeight,
      filterWidth,
    ]);
    const filter = tf.transpose(filterOIHW, [2, 3, 1, 0]); // OIHW -> HWIO
    await Promise.all([input.data(), filter.data()]);

    return await runBenchmark("tfjs", async () => {
      const output = tf
        .conv2d(input, filter, 1, "same", "NHWC")
        .transpose<tf.Tensor4D>([0, 3, 1, 2]); // NHWC -> NCHW
      const ar = (await output.data()) as Float32Array;
      logSampleBuffer(ar);
      input.dispose();
      filterOIHW.dispose();
      filter.dispose();
      output.dispose();
    });
  }
}

class OnnxStrategy implements Strategy {
  name: string;
  dtype: "fp16" | "fp32";

  constructor(fp16: boolean = false) {
    this.name = fp16 ? "onnx-fp16" : "onnx";
    this.dtype = fp16 ? "fp16" : "fp32";
  }

  // Helper function to create a simple ONNX model with a Conv operation
  async createConvModel(): Promise<Uint8Array> {
    const { create, toBinary } = await import("@bufbuild/protobuf");
    const {
      AttributeProto_AttributeType,
      AttributeProtoSchema,
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

    const elemType = {
      fp32: TensorProto_DataType.FLOAT,
      fp16: TensorProto_DataType.FLOAT16,
    }[this.dtype];

    const dimension = (value: number | bigint) => {
      return create(TensorShapeProto_DimensionSchema, {
        value: {
          case: "dimValue",
          value: BigInt(value),
        },
      });
    };

    // Create input tensor (NCHW format)
    const input = create(ValueInfoProtoSchema, {
      name: "input",
      type: create(TypeProtoSchema, {
        value: {
          case: "tensorType",
          value: create(TypeProto_TensorSchema, {
            elemType,
            shape: create(TensorShapeProtoSchema, {
              dim: [
                dimension(batchSize),
                dimension(channels),
                dimension(height),
                dimension(width),
              ],
            }),
          }),
        },
      }),
    });

    // Create filter tensor (OIHW format)
    const filter = create(ValueInfoProtoSchema, {
      name: "filter",
      type: create(TypeProtoSchema, {
        value: {
          case: "tensorType",
          value: create(TypeProto_TensorSchema, {
            elemType,
            shape: create(TensorShapeProtoSchema, {
              dim: [
                dimension(outChannels),
                dimension(channels),
                dimension(filterHeight),
                dimension(filterWidth),
              ],
            }),
          }),
        },
      }),
    });

    // Create output tensor
    const output = create(ValueInfoProtoSchema, {
      name: "output",
      type: create(TypeProtoSchema, {
        value: {
          case: "tensorType",
          value: create(TypeProto_TensorSchema, {
            elemType,
            shape: create(TensorShapeProtoSchema, {
              dim: [
                dimension(batchSize),
                dimension(outChannels),
                dimension(height),
                dimension(width),
              ],
            }),
          }),
        },
      }),
    });

    // Create Conv node with appropriate attributes
    const convNode = create(NodeProtoSchema, {
      input: ["input", "filter"],
      output: ["output"],
      opType: "Conv",
      name: "conv_node",
      attribute: [
        // Set padding to "same" mode
        create(AttributeProtoSchema, {
          name: "pads",
          type: AttributeProto_AttributeType.INTS,
          ints: [1n, 1n, 1n, 1n], // [top, left, bottom, right]
        }),
        // Set strides
        create(AttributeProtoSchema, {
          name: "strides",
          type: AttributeProto_AttributeType.INTS,
          ints: [1n, 1n],
        }),
      ],
    });

    // Create the graph
    const graph = create(GraphProtoSchema, {
      node: [convNode],
      name: "conv_graph",
      input: [input, filter],
      output: [output],
    });

    // Create the model
    const model = create(ModelProtoSchema, {
      irVersion: 8n,
      opsetImport: [create(OperatorSetIdProtoSchema, { version: 14n })],
      graph: graph,
    });

    // Serialize to bytes
    return toBinary(ModelProtoSchema, model);
  }

  async run(): Promise<number> {
    const ort = await import("onnxruntime-web/webgpu");
    let session: import("onnxruntime-web/webgpu").InferenceSession | null =
      null;

    try {
      const model = await this.createConvModel();
      session = await ort.InferenceSession.create(model, {
        executionProviders: ["webgpu"],
      });

      // Prepare input tensors
      let inputBuffer: any;
      let filterBuffer: any;
      let ortType: any;
      if (this.dtype === "fp16") {
        inputBuffer = new Float16Array(randomInput);
        filterBuffer = new Float16Array(randomFilter);
        ortType = "float16";
      } else {
        inputBuffer = randomInput;
        filterBuffer = randomFilter;
        ortType = "float32";
      }

      const tensorInput = new ort.Tensor(ortType, inputBuffer, [
        batchSize,
        channels,
        height,
        width,
      ]);
      const tensorFilter = new ort.Tensor(ortType, filterBuffer, [
        outChannels,
        channels,
        filterHeight,
        filterWidth,
      ]);

      // Actual benchmark run
      return await runBenchmark("onnx", async () => {
        const results = await session!.run({
          input: tensorInput,
          filter: tensorFilter,
        });
        const outputData = results.output.data as Float32Array;
        logSampleBuffer(outputData);
      });
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

class JaxJsStrategy implements Strategy {
  name: string;
  device: Device;
  fp16: boolean;

  constructor(device: Device = "webgpu", fp16: boolean = false) {
    this.device = device;
    this.fp16 = fp16;
    this.name = "jax-js";
    if (device !== "webgpu") this.name += `-${device}`;
    if (fp16) this.name += "-fp16";
  }

  async run(): Promise<number> {
    const jax = await import("@jax-js/jax");
    await jax.init();
    jax.defaultDevice(this.device);
    const np = jax.numpy;

    const x = np
      .array(randomInput, {
        shape: [batchSize, channels, height, width],
      })
      .astype(this.fp16 ? np.float16 : np.float32);
    const filter = np
      .array(randomFilter, {
        shape: [outChannels, channels, filterHeight, filterWidth],
      })
      .astype(this.fp16 ? np.float16 : np.float32);
    await jax.blockUntilReady([x, filter]);

    return await runBenchmark("jax", async () => {
      const output = jax.lax.convGeneralDilated(x, filter, [1, 1], "SAME");
      const ar = (await output.data()) as Float32Array;
      logSampleBuffer(ar);
    });
  }
}

export const strategies: Strategy[] = [
  new NaiveStrategy(8),
  new NaiveStrategy(16),
  new NaiveStrategy(32),
  new OnnxStrategy(),
  new OnnxStrategy(true),
  new TfjsStrategy(),
  new TfjsStrategy(true),
  new JaxJsStrategy("webgpu"),
  new JaxJsStrategy("webgpu", true),
  new JaxJsStrategy("wasm"),
];
