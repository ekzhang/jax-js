import { AluExp, AluGroup, AluOp, DType } from "../alu";
import { Backend, Slot, SlotError } from "../backend";
import { DEBUG } from "../utils";

/** Implementation of `Backend` that uses WebGPU in browsers. */
export class WebGPUBackend implements Backend {
  readonly pipelines: ShaderPipelineCache;
  readonly buffers: Map<Slot, { ref: number; buffer: GPUBuffer }>;
  nextSlot: number;

  constructor(readonly device: GPUDevice) {
    if (DEBUG >= 3) {
      console.info(
        "webgpu adapter:",
        device.adapterInfo.vendor,
        device.adapterInfo.architecture,
      );
    }
    this.pipelines = new ShaderPipelineCache(device);
    this.buffers = new Map();
    this.nextSlot = 1;
  }

  malloc(size: number, initialData?: ArrayBuffer): Slot {
    let buffer: GPUBuffer;
    if (initialData) {
      if (initialData.byteLength !== size) {
        throw new Error("initialData size does not match buffer size");
      }
      buffer = this.#createBuffer(size, { mapped: true });
      new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(initialData));
      buffer.unmap();
    } else {
      buffer = this.#createBuffer(size);
    }

    const slot = this.nextSlot++;
    this.buffers.set(slot, { buffer, ref: 1 });
    return slot;
  }

  incRef(slot: Slot): void {
    const buffer = this.buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref++;
  }

  decRef(slot: Slot): void {
    const buffer = this.buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref--;
    if (buffer.ref === 0) {
      this.buffers.delete(slot);
      buffer.buffer.destroy();
    }
  }

  async read(slot: Slot, start?: number, count?: number): Promise<ArrayBuffer> {
    const buffer = this.#getBuffer(slot);
    if (start === undefined) start = 0;
    if (count === undefined) count = buffer.size - start;

    // Need a GPUBuffer with MAP_READ usage when transfering data to host.
    const staging = this.#createBuffer(count, { read: true });
    try {
      const commandEncoder = this.device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(buffer, start, staging, 0, count);
      this.device.queue.submit([commandEncoder.finish()]);

      await staging.mapAsync(GPUMapMode.READ);
      const arrayBuffer = staging.getMappedRange();
      const data = new Float32Array(arrayBuffer);
      return data.slice();
    } finally {
      staging.destroy();
    }
  }

  readSync(slot: Slot, start?: number, count?: number): ArrayBuffer {
    // TODO: WebGL hack
    // https://github.com/tensorflow/tfjs/blob/2644bd0d6cea677f80e44ed4a44bea5e04aabeb3/tfjs-backend-webgl/src/backend_webgl.ts#L271
    throw new Error("readSync() not implemented for WebGPU");
  }

  async execute(exp: AluExp, inputs: Slot[], outputs: Slot[]): Promise<void> {
    const inputBuffers = inputs.map((slot) => this.#getBuffer(slot));
    const outputBuffers = outputs.map((slot) => this.#getBuffer(slot));
    const nargs = inputs.length;
    const pipeline = await this.pipelines.get(pipelineSource(nargs, exp));
    pipelineSubmit(this.device, pipeline, inputBuffers, outputBuffers);
  }

  executeSync(exp: AluExp, inputs: Slot[], outputs: Slot[]): void {
    const inputBuffers = inputs.map((slot) => this.#getBuffer(slot));
    const outputBuffers = outputs.map((slot) => this.#getBuffer(slot));
    const nargs = inputs.length;
    const pipeline = this.pipelines.getSync(pipelineSource(nargs, exp));
    pipelineSubmit(this.device, pipeline, inputBuffers, outputBuffers);
  }

  #getBuffer(slot: Slot): GPUBuffer {
    const buffer = this.buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    return buffer.buffer;
  }

  /**
   * Create a GPU buffer.
   *
   * By default, this creates a general-purpose buffer with the given size.
   *
   * - If `mapped` is true, initialize the buffer in mapped mode so that it can
   *   be populated with data from the CPU. (Call `.unmap()` later.)
   * - If `read` is true, create a staging buffer for returning data to CPU.
   *   (Call `.mapAsync()` later.)
   */
  #createBuffer(
    size: number,
    { mapped = false, read = false } = {},
  ): GPUBuffer {
    if (read && mapped) {
      throw new Error("mapped and read cannot both be true");
    }
    const buffer = this.device.createBuffer({
      size,
      usage: read
        ? GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        : GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      mappedAtCreation: mapped,
    });
    return buffer;
  }
}

function dtypeToWgsl(dtype: DType): string {
  switch (dtype) {
    case DType.Bool:
      return "bool";
    case DType.Int32:
      return "i32";
    case DType.Float32:
      return "f32";
    default:
      throw new Error(`Unsupported dtype: ${dtype}`);
  }
}

function constToWgsl(dtype: DType, value: any): string {
  if (dtype === DType.Bool) return value ? "true" : "false";
  if (dtype === DType.Int32) return value.toString();
  if (dtype === DType.Float32) {
    let s = value.toString();
    if (!s.includes(".")) s += ".0";
    return s;
  }
  throw new Error(`Unsupported const dtype: ${dtype}`);
}

/** Compiles an expression into WebGPU shader source code. */
function pipelineSource(nargs: number, exp: AluExp): string {
  exp = exp.simplify();
  const args = Array.from({ length: nargs }, (_, i) => `in${i}`);

  // binding(0): uniforms
  // binding(1..n): input buffers
  // binding(n+1): output buffer

  const kernel: string[] = []; // line-separated
  kernel.push(
    "struct Uniforms {",
    "  len: u32,",
    "};",
    "@group(0) @binding(0) var<uniform> uniforms : Uniforms;",
  );

  for (let i = 0; i < nargs; i++) {
    kernel.push(
      `@group(0) @binding(${i + 1}) var<storage, read> ${args[i]} : array<f32>;`,
    );
  }
  kernel.push(
    `@group(0) @binding(${nargs + 1}) var<storage, read_write> result : array<f32>;`,
  );

  kernel.push(
    "\n@compute @workgroup_size(64)",
    "fn main(@builtin(global_invocation_id) id : vec3<u32>) {",
    "  if (id.x >= uniforms.len) { return; }",
    "  let gidx: i32 = i32(id.x);",
  );

  // Generate code for each AluExp operation.
  // Some expressions may be used twice, so we keep track of them.
  let gensymCount = 0;
  const gensym = () => `alu${gensymCount++}`;

  const expToVar = new Map<AluExp, string>();
  const gen = (exp: AluExp): string => {
    if (expToVar.has(exp)) return expToVar.get(exp)!;
    const { op, src, dtype, arg } = exp;

    // Some of these cases early `return` to inline it.
    let source = "";
    if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
      const a = gen(src[0]);
      const b = gen(src[1]);
      if (op === AluOp.Add) source = `${a} + ${b}`;
      else if (op === AluOp.Sub) source = `${a} - ${b}`;
      else if (op === AluOp.Mul) source = `${a} * ${b}`;
      else if (op === AluOp.Idiv)
        source = dtype === DType.Int32 ? `${a} / ${b}` : `floor(${a} / ${b})`;
      else if (op === AluOp.Mod) source = `${a} % ${b}`;
      else if (op === AluOp.Cmplt) source = `${a} < ${b}`;
      else if (op === AluOp.Cmpne) source = `${a} != ${b}`;
    } else if (AluGroup.Unary.has(op)) {
      const a = gen(src[0]);
      if (op === AluOp.Sin) source = `sin(${a})`;
      else if (op === AluOp.Cos) source = `cos(${a})`;
    } else if (op === AluOp.Where) {
      // select(f, t, cond) -> cond ? t : f
      source = `select(${gen(src[2])}, ${gen(src[1])}, ${gen(src[0])})`;
    } else if (op === AluOp.Const) {
      return constToWgsl(dtype, arg);
    } else if (op === AluOp.Special) {
      return arg[0] as string;
    } else if (op === AluOp.GlobalIndex) {
      source = `${args[arg]}[${gen(src[0])}]`;
    }

    if (!source) throw new Error(`Missing impl for op: ${op}`);
    const typeName = dtypeToWgsl(dtype);
    const name = gensym();
    expToVar.set(exp, name);
    kernel.push(`  let ${name}: ${typeName} = ${source};`);
    return name;
  };

  kernel.push(`  result[gidx] = ${gen(exp)};`, "}");
  return kernel.join("\n");
}

function pipelineSubmit(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  inputs: GPUBuffer[],
  outputs: GPUBuffer[],
) {
  if (
    inputs.length + outputs.length >
    device.limits.maxStorageBuffersPerShaderStage
  ) {
    // This is a hard limit in WebGPU. All platforms have at least 8 storage
    // buffers per shader stage, and >99% support 10. If you pass more than this
    // many inputs then you risk running into this limit.
    const actual = inputs.length + outputs.length;
    const max = device.limits.maxStorageBuffersPerShaderStage;
    throw new Error(
      `Too many buffers (${actual}) for WebGPU pipeline (max: ${max})`,
    );
  }

  const len = outputs[0].size;
  const uniform = device.createBuffer({
    size: 4, // bytes
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  new Uint32Array(uniform.getMappedRange()).set([len]);
  uniform.unmap();

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      ...inputs.map((buffer, i) => {
        return { binding: i + 1, resource: { buffer } };
      }),
      { binding: inputs.length + 1, resource: { buffer: outputs[0] } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(Math.ceil(inputs[0].size / 64));
  passEncoder.end();
  device.queue.submit([commandEncoder.finish()]);
}

/**
 * A cache for compiled GPU compute pipelines, keyed by the shader source.
 *
 * This supports both async compilation (recommended) and a synchronous variant.
 * If the pipeline is not in the cache, it will be compiled and added. For async
 * compilation, only one compilation will be in progress at a time for a given
 * shader source.
 */
class ShaderPipelineCache {
  cache: Map<string, GPUComputePipeline>;
  inProgress: Map<string, Promise<GPUComputePipeline>>;

  constructor(readonly device: GPUDevice) {
    this.cache = new Map();
    this.inProgress = new Map();
  }

  async get(code: string): Promise<GPUComputePipeline> {
    const existingPipeline = this.cache.get(code);
    if (existingPipeline) {
      return existingPipeline;
    }
    const existingPromise = this.inProgress.get(code);
    if (existingPromise) {
      return await existingPromise;
    }
    if (DEBUG >= 2) {
      console.info("=========== WebGPU shader ===========\n" + code);
    }

    const shaderModule = this.device.createShaderModule({ code });
    const promise = (async () => {
      this.device.pushErrorScope("validation");
      try {
        const pipeline = await this.device.createComputePipelineAsync({
          layout: "auto",
          compute: {
            module: shaderModule,
            entryPoint: "main",
          },
        });
        await this.device.popErrorScope();
        return pipeline;
      } catch (e) {
        // This can race with other compilations, but it shouldn't happen in
        // correct code. Any validation error here is a bug in `jax-js`.
        const scope = await this.device.popErrorScope();
        const emsg = await compileError(shaderModule, scope, code);
        throw new Error(emsg);
      }
    })();
    this.inProgress.set(code, promise);

    // This could race against getSync(), but it's okay since shader pipeline
    // creation is deterministic + idempotent.
    const pipeline = await promise;
    this.cache.set(code, pipeline);
    return pipeline;
  }

  getSync(code: string): GPUComputePipeline {
    const existingPipeline = this.cache.get(code);
    if (existingPipeline) {
      return existingPipeline;
    }
    if (DEBUG >= 2) {
      console.info("=========== WebGPU shader ===========\n" + code);
    }

    const shaderModule = this.device.createShaderModule({ code });
    this.device.pushErrorScope("validation");
    const pipeline = this.device.createComputePipeline({
      layout: "auto",
      compute: {
        module: shaderModule,
        entryPoint: "main",
      },
    });
    this.device.popErrorScope().then(async (scope) => {
      // This happens asynchronously, so we can't throw here. But shader syntax
      // validation errors should never occur in correct code. Any issues here
      // reflect bugs in jax-js.
      if (scope !== null) {
        const emsg = await compileError(shaderModule, scope, code);
        console.error(emsg);
      }
    });
    this.cache.set(code, pipeline);
    return pipeline;
  }
}

/** Gather information about a compilation error and format it. */
async function compileError(
  shaderModule: GPUShaderModule,
  scope: GPUError | null,
  code: string,
): Promise<string> {
  let message = `Failed to compile shader: ${scope ? scope.message : "(no error scope)"}`;
  const info = await shaderModule.getCompilationInfo();
  for (const msg of info.messages) {
    message += `\n  [${msg.type} at ${msg.lineNum}:${msg.linePos}] ${msg.message}`;
  }
  if (code) {
    message += `\n\n${code}`;
  }
  return message;
}
