import { AluExp, AluOp, DType, Kernel } from "../alu";
import { Backend, Device, Executable, Slot, SlotError } from "../backend";
import { Routine } from "../routine";
import { tuneWebgpu } from "../tuner";
import { DEBUG, findPow2, FpHash, prod, range, strip1 } from "../utils";
import {
  calculateGrid,
  constToWgsl,
  dtypeToWgsl,
  reduceOpWgsl,
  ShaderInfo,
  WgslBuilder,
  WgslExpCodegen,
} from "./webgpu/codegen";
import { nullaryKernelSource } from "./webgpu/nullaryKernel";
import { SyncReader } from "./webgpu/reader";
import { createRoutineShader } from "./webgpu/routines";
import { maybeAcquireTracingSlot, recordTrace } from "./webgpu/tracing";

interface ShaderDispatch extends ShaderInfo {
  pipeline: GPUComputePipeline; // Compiled pipeline for the shader.
}

const MAX_REUSABLE_BUFFER_BYTES = 64 * 1024 * 1024;
const MAX_REUSABLE_BUFFERS_PER_SIZE = 64;

/** Implementation of `Backend` that uses WebGPU in browsers. */
export class WebGPUBackend implements Backend {
  readonly type: Device = "webgpu";
  readonly maxArgs: number;

  readonly pipelines: ShaderPipelineCache;
  readonly syncReader: SyncReader;
  readonly buffers: Map<
    Slot,
    {
      ref: number;
      size: number; // Refers to "true size" requested, less padding.
      allocatedSize: number; // Actual GPUBuffer size.
      buffer: GPUBuffer;
    }
  >;
  nextSlot: number;

  #cachedShaderMap = new Map<bigint, ShaderInfo>();
  #reusableZsb: GPUBuffer;
  #bufferPool = new Map<number, GPUBuffer[]>();

  constructor(readonly device: GPUDevice) {
    if (DEBUG >= 3 && device.adapterInfo) {
      console.info(
        "webgpu adapter:",
        device.adapterInfo.vendor,
        device.adapterInfo.architecture,
      );
    }
    this.maxArgs = this.device.limits.maxStorageBuffersPerShaderStage - 1;
    this.pipelines = new ShaderPipelineCache(device);
    this.syncReader = new SyncReader(device);
    this.buffers = new Map();
    this.nextSlot = 1;

    // Special "zero-size buffer" that's reused across all allocations of size
    // zero, backing slots for those allocations.
    //
    // WebGPU allows creating buffers of size 0, but you cannot actually make
    // bindings of size 0 when calling `createBindGroup()`. The simplest way to
    // handle this is to just create a buffer of minimum size (4 bytes) and
    // reuse that across all zero-size allocations.
    this.#reusableZsb = this.#createBuffer(4);

    device.addEventListener("uncapturederror", (event) => {
      console.error("Uncaptured error in WebGPU backend:", event.error.message);
    });
  }

  malloc(size: number, initialData?: Uint8Array<ArrayBuffer>): Slot {
    // All GPUBuffer must be a multiple of 4 bytes in length, to support copy
    // operations. Pad it to a multiple of 4.
    if (initialData && initialData.byteLength !== size) {
      throw new Error("initialData size does not match buffer size");
    }
    const allocatedSize = Math.ceil(size / 4) * 4 || 4;
    const buffer =
      size === 0 ? this.#reusableZsb : this.#acquireBuffer(allocatedSize);

    if (initialData && size > 0) {
      if (initialData.byteLength % 4 === 0) {
        this.device.queue.writeBuffer(buffer, 0, initialData);
      } else {
        // Copy all but the last few bytes, then copy 4 bytes as remainder.
        const aligned = initialData.byteLength - (initialData.byteLength % 4);
        if (aligned > 0) {
          this.device.queue.writeBuffer(buffer, 0, initialData, 0, aligned);
        }
        const remainder = new Uint8Array(4);
        remainder.set(initialData.subarray(aligned));
        this.device.queue.writeBuffer(buffer, aligned, remainder);
      }
    }

    const slot = this.nextSlot++;
    this.buffers.set(slot, { buffer, size, allocatedSize, ref: 1 });
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
      if (buffer.buffer !== this.#reusableZsb) {
        this.#releaseBuffer(buffer.buffer, buffer.allocatedSize);
      }
    }
  }

  async read(
    slot: Slot,
    start?: number,
    count?: number,
  ): Promise<Uint8Array<ArrayBuffer>> {
    const { buffer, size } = this.#getBuffer(slot);
    if (buffer === this.#reusableZsb) return new Uint8Array();
    if (start === undefined) start = 0;
    if (count === undefined) count = size - start;

    // Need a GPUBuffer with MAP_READ usage when transfering data to host.
    const paddedSize = Math.ceil(count / 4) * 4;
    const staging = this.#createBuffer(paddedSize, { read: true });
    try {
      const commandEncoder = this.device.createCommandEncoder();
      commandEncoder.copyBufferToBuffer(buffer, start, staging, 0, paddedSize);
      this.device.queue.submit([commandEncoder.finish()]);

      await staging.mapAsync(GPUMapMode.READ);
      const arrayBuffer = staging.getMappedRange();
      return new Uint8Array(arrayBuffer.slice(), 0, count);
    } finally {
      staging.destroy();
    }
  }

  readSync(
    slot: Slot,
    start?: number,
    count?: number,
  ): Uint8Array<ArrayBuffer> {
    const { buffer, size } = this.#getBuffer(slot);
    if (buffer === this.#reusableZsb) return new Uint8Array();
    if (start === undefined) start = 0;
    if (count === undefined) count = size - start;
    return this.syncReader.read(buffer, start, count);
  }

  #cachedShader(kernel: Kernel): ShaderInfo {
    const cacheKey = FpHash.hash(kernel);
    let result = this.#cachedShaderMap.get(cacheKey);
    if (!result) {
      result = pipelineSource(this.device, kernel);
      this.#cachedShaderMap.set(cacheKey, result);
    }
    return result;
  }

  async prepareKernel(kernel: Kernel): Promise<Executable<ShaderDispatch[]>> {
    const shader = this.#cachedShader(kernel);
    const pipeline = await this.pipelines.prepare(shader);
    return new Executable(kernel, [{ ...shader, pipeline }]);
  }

  prepareKernelSync(kernel: Kernel): Executable<ShaderDispatch[]> {
    const shader = this.#cachedShader(kernel);
    const pipeline = this.pipelines.prepareSync(shader);
    return new Executable(kernel, [{ ...shader, pipeline }]);
  }

  async prepareRoutine(
    routine: Routine,
  ): Promise<Executable<ShaderDispatch[]>> {
    const shaders = createRoutineShader(this.device, routine);
    const dispatches = await Promise.all(
      shaders.map(async (shader) => {
        const pipeline = await this.pipelines.prepare(shader);
        return { ...shader, pipeline };
      }),
    );
    return new Executable(routine, dispatches);
  }

  prepareRoutineSync(routine: Routine): Executable<ShaderDispatch[]> {
    const shaders = createRoutineShader(this.device, routine);
    const dispatches = shaders.map((shader) => {
      const pipeline = this.pipelines.prepareSync(shader);
      return { ...shader, pipeline };
    });
    return new Executable(routine, dispatches);
  }

  dispatch(
    exe: Executable<ShaderDispatch[]>,
    inputs: Slot[],
    outputs: Slot[],
  ): void {
    const inputBuffers = inputs.map((slot) => this.#getBuffer(slot).buffer);
    const outputBuffers = outputs.map((slot) => this.#getBuffer(slot).buffer);
    pipelineSubmit(this.device, exe, inputBuffers, outputBuffers);
  }

  #getBuffer(slot: Slot): { buffer: GPUBuffer; size: number } {
    const buffer = this.buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    return { buffer: buffer.buffer, size: buffer.size };
  }

  #acquireBuffer(size: number): GPUBuffer {
    if (size > MAX_REUSABLE_BUFFER_BYTES) return this.#createBuffer(size);
    const bucket = this.#bufferPool.get(size);
    const buffer = bucket?.pop();
    if (bucket && bucket.length === 0) this.#bufferPool.delete(size);
    return buffer ?? this.#createBuffer(size);
  }

  #releaseBuffer(buffer: GPUBuffer, size: number): void {
    if (size > MAX_REUSABLE_BUFFER_BYTES) {
      buffer.destroy();
      return;
    }
    const bucket = this.#bufferPool.get(size);
    if (!bucket) {
      this.#bufferPool.set(size, [buffer]);
      return;
    }
    if (bucket.length >= MAX_REUSABLE_BUFFERS_PER_SIZE) {
      buffer.destroy();
      return;
    }
    bucket.push(buffer);
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

/**
 * Compiles an expression into WebGPU shader source code.
 *
 * Returns the shader source and the number of workgroups to dispatch along x
 * and y axes, to run the kernel.
 */
function pipelineSource(device: GPUDevice, kernel: Kernel): ShaderInfo {
  const nullaryKernel = nullaryKernelSource(device, kernel);
  if (nullaryKernel) return nullaryKernel;

  const tune = tuneWebgpu(kernel);
  if (DEBUG >= 3) {
    console.info(`kernel.exp: ${kernel.exp}\ntune.exp: ${tune.exp}`);
  }

  const { nargs, reduction: re } = kernel;
  const args = Array.from({ length: nargs }, (_, i) => `in${i}`);

  const wb = new WgslBuilder();
  wb.emitPreamble(device, [tune.exp, tune.epilogue]);

  const usedArgs: (DType | null)[] = Array.from({ length: nargs }, () => null);
  tune.exp.fold((exp) => {
    if (exp.op === AluOp.GlobalIndex) usedArgs[exp.arg[0]] = exp.dtype;
  });
  tune.epilogue?.fold((exp) => {
    if (exp.op === AluOp.GlobalIndex) usedArgs[exp.arg[0]] = exp.dtype;
  });

  // binding(0..n-1): input buffers
  // binding(n): output buffer
  for (let i = 0; i < nargs; i++) {
    // If not used, just assume float32, all that matters is size / alignment.
    const ty = dtypeToWgsl(usedArgs[i] ?? DType.Float32, true);
    wb.emit(
      `@group(0) @binding(${i}) var<storage, read> ${args[i]} : array<${ty}>;`,
    );
  }

  const resultTy = dtypeToWgsl(kernel.dtype, true);
  wb.emit(
    `@group(0) @binding(${nargs}) var<storage, read_write> result : array<${resultTy}>;`,
  );

  const groupCount = re ? (tune.size.groups ?? 1) : 1;
  const groupedReduction = re && groupCount > 1;
  if (groupedReduction && tune.threadCount % groupCount !== 0) {
    throw new Error("WebGPU grouped reduction has invalid thread count");
  }
  if (groupedReduction && groupCount > device.limits.maxComputeWorkgroupSizeX) {
    throw new Error("WebGPU grouped reduction exceeds workgroup size limit");
  }
  const workgroupSize = groupedReduction
    ? groupCount
    : findPow2(tune.threadCount, 256);

  // Determine grid size, may need to be 3D due to limits on X.
  // maxComputeWorkgroupsPerDimension ~ 65535, so we use 16384 when exceeded.
  const gridSize = groupedReduction
    ? tune.threadCount / groupCount
    : Math.ceil(tune.threadCount / workgroupSize);
  const [gridX, gridY] = calculateGrid(gridSize);

  if (groupedReduction) {
    const partialTy = dtypeToWgsl(re.dtype);
    for (let i = 0; i < (tune.size.upcast ?? 1); i++) {
      wb.emit(
        `var<workgroup> partial${i}: array<${partialTy}, ${groupCount}>;`,
      );
    }
  }

  wb.emit("", `@compute @workgroup_size(${workgroupSize})`);
  if (groupedReduction) {
    wb.emit(
      "fn main(",
      wb.pushIndent,
      "@builtin(local_invocation_id) lid : vec3<u32>,",
      "@builtin(workgroup_id) wg_id : vec3<u32>,",
      wb.popIndent,
      ") {",
      wb.pushIndent,
    );
    if (gridY === 1) {
      wb.emit(
        `if (wg_id.x >= ${gridSize}u) { return; }`,
        "let gidx: i32 = i32(wg_id.x);",
      );
    } else {
      wb.emit(
        `if (${gridX}u * wg_id.y + wg_id.x >= ${gridSize}u) { return; }`,
        `let gidx: i32 = i32(${gridX}u * wg_id.y + wg_id.x);`,
      );
    }
    wb.emit("let group: i32 = i32(lid.x);");
  } else {
    wb.emit(
      "fn main(@builtin(global_invocation_id) id : vec3<u32>) {",
      wb.pushIndent,
    );
    if (gridY === 1) {
      wb.emit(
        `if (id.x >= ${tune.threadCount}) { return; }`,
        "let gidx: i32 = i32(id.x);",
      );
    } else {
      const sizeX = gridX * workgroupSize;
      wb.emit(
        `if (${sizeX} * id.y + id.x >= ${tune.threadCount}) { return; }`,
        `let gidx: i32 = i32(${sizeX} * id.y + id.x);`,
      );
    }
  }

  wb.emitPhonyAssignments(args);

  const gen = new WgslExpCodegen(wb, args);
  if (!re) {
    gen.countReferences(tune.exp);
    let rhs = strip1(gen.run(tune.exp));
    if (resultTy !== dtypeToWgsl(tune.exp.dtype)) rhs = `${resultTy}(${rhs})`;
    wb.emit(`result[gidx] = ${rhs};`);
  } else {
    const unroll = tune.size.unroll ?? 1;
    const upcast = tune.size.upcast ?? 1;

    const acc = [...Array(upcast)].map((_, i) => `acc${i}`);
    for (let i = 0; i < upcast; i++) {
      wb.emit(
        `var ${acc[i]}: ${dtypeToWgsl(re.dtype)} = ${constToWgsl(re.dtype, re.identity)};`,
      ); // Initialize accumulators.
    }

    wb.emit(
      `for (var ridx: i32 = 0; ridx < ${tune.size.reduce}; ridx++) {`,
      wb.pushIndent,
    );

    // Now generate (shared) expressions for each accumulator and unroll value.
    const exps: AluExp[][] = [];
    const cache = new Map<bigint, AluExp>();
    for (let up = 0; up < upcast; up++) {
      exps.push([]);
      for (let un = 0; un < unroll; un++) {
        const exp = tune.exp.substitute({
          upcast: AluExp.i32(up),
          unroll: AluExp.i32(un),
        });
        exps[up].push(exp.simplify(cache));
        gen.countReferences(exps[up][un]);
      }
    }

    // After references are counted, we can generate the code.
    const items = exps.map((ar) => ar.map((x) => gen.run(x)).map(strip1));
    for (let i = 0; i < upcast; i++) {
      let rhs = items[i][0];
      for (let j = 1; j < unroll; j++) {
        if (re.op === AluOp.Add) rhs = `${rhs} + ${items[i][j]}`;
        else if (re.op === AluOp.Mul) rhs = `${rhs} * ${items[i][j]}`;
        else if (re.op === AluOp.Min) {
          // For booleans, min is AND; for numerics, use min()
          rhs =
            re.dtype === DType.Bool
              ? `(${rhs} && ${items[i][j]})`
              : `min(${rhs}, ${items[i][j]})`;
        } else if (re.op === AluOp.Max) {
          // For booleans, max is OR; for numerics, use max()
          rhs =
            re.dtype === DType.Bool
              ? `(${rhs} || ${items[i][j]})`
              : `max(${rhs}, ${items[i][j]})`;
        } else throw new Error(`Unsupported reduction op: ${re.op}`);
      }
      if (re.op === AluOp.Add) wb.emit(`${acc[i]} += ${rhs};`);
      else if (re.op === AluOp.Mul) wb.emit(`${acc[i]} *= ${rhs};`);
      else if (re.op === AluOp.Min) {
        // For booleans, min is AND; for numerics, use min()
        if (re.dtype === DType.Bool)
          wb.emit(`${acc[i]} = ${acc[i]} && ${rhs};`);
        else wb.emit(`${acc[i]} = min(${acc[i]}, ${rhs});`);
      } else if (re.op === AluOp.Max) {
        // For booleans, max is OR; for numerics, use max()
        if (re.dtype === DType.Bool)
          wb.emit(`${acc[i]} = ${acc[i]} || ${rhs};`);
        else wb.emit(`${acc[i]} = max(${acc[i]}, ${rhs});`);
      } else throw new Error(`Unsupported reduction op: ${re.op}`);
    }
    wb.emit(wb.popIndent, "}");

    if (groupedReduction) {
      for (let i = 0; i < upcast; i++)
        wb.emit(`partial${i}[lid.x] = ${acc[i]};`);
      wb.emit("workgroupBarrier();");
      for (let stride = groupCount / 2; stride >= 1; stride /= 2) {
        wb.emit(`if (lid.x < ${stride}u) {`, wb.pushIndent);
        for (let i = 0; i < upcast; i++) {
          wb.emit(
            `partial${i}[lid.x] = ${reduceOpWgsl(
              re.op,
              re.dtype,
              `partial${i}[lid.x]`,
              `partial${i}[lid.x + ${stride}u]`,
            )};`,
          );
        }
        wb.emit(wb.popIndent, "}", "workgroupBarrier();");
      }
    }

    // Exited the reduction loop scope. Erase any local variables.
    gen.reset();

    const outputIdxExps: AluExp[] = [];
    const fusionExps: AluExp[] = [];
    for (let i = 0; i < upcast; i++) {
      const exp = tune.outputIdxExp.substitute({ upcast: AluExp.i32(i) });
      outputIdxExps.push(exp.simplify(cache));
      gen.countReferences(outputIdxExps[i]);
      fusionExps.push(
        tune
          .epilogue!.substitute({
            acc: AluExp.variable(re.dtype, acc[i]),
            upcast: AluExp.i32(i),
          })
          .simplify(cache),
      );
      gen.countReferences(fusionExps[i]);
    }
    if (groupedReduction) {
      wb.emit("if (lid.x == 0u) {", wb.pushIndent);
      for (let i = 0; i < upcast; i++) wb.emit(`${acc[i]} = partial${i}[0u];`);
    }
    for (let i = 0; i < upcast; i++) {
      const index = strip1(gen.run(outputIdxExps[i]));
      let rhs = strip1(gen.run(fusionExps[i]));
      if (resultTy !== dtypeToWgsl(fusionExps[i].dtype))
        rhs = `${resultTy}(${rhs})`;
      wb.emit(`result[${index}] = ${rhs};`);
    }
    if (groupedReduction) wb.emit(wb.popIndent, "}");
  }

  wb.emit(wb.popIndent, "}");
  return {
    code: wb.toString(),
    numInputs: nargs,
    numOutputs: 1,
    hasUniform: false,
    passes: [{ grid: [gridX, gridY] }],
  };
}

function pipelineSubmit(
  device: GPUDevice,
  exe: Executable<ShaderDispatch[]>,
  inputs: GPUBuffer[],
  outputs: GPUBuffer[],
) {
  const { data: pipelines, source } = exe;
  const commandEncoder = device.createCommandEncoder();
  for (const { pipeline, ...shader } of pipelines) {
    if (
      inputs.length !== shader.numInputs ||
      outputs.length !== shader.numOutputs
    ) {
      throw new Error(
        `webgpu: expected ${shader.numInputs} inputs and ${shader.numOutputs} outputs, ` +
          `got ${inputs.length} inputs and ${outputs.length} outputs`,
      );
    }

    const filteredPasses = shader.passes.filter(({ grid }) => prod(grid) > 0);
    if (filteredPasses.length === 0) continue; // No work to do.

    const slot = maybeAcquireTracingSlot(device);
    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        ...inputs.map((buffer, i) => ({
          binding: i,
          resource: { buffer },
        })),
        ...outputs.map((buffer, i) => ({
          binding: inputs.length + i,
          resource: { buffer },
        })),
      ],
    });

    let uniformBindGroup: GPUBindGroup | null = null;
    let uniformAlignment = 0;
    if (shader.hasUniform) {
      // This shader requires uniforms, create a shared buffer with uniform
      // values for each pass of the shader (use dynamic offsets).
      const uniforms = filteredPasses.map(({ uniform }) => uniform!);
      const [uniformBuffer, alignment] = combineUniforms(device, uniforms);
      uniformAlignment = alignment;
      uniformBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer, size: alignment } },
        ],
      });
    }

    for (let i = 0; i < filteredPasses.length; i++) {
      const { grid } = filteredPasses[i];
      let timestampWrites: GPUComputePassTimestampWrites | undefined;
      if (slot) {
        const isFirst = i === 0;
        const isLast = i === filteredPasses.length - 1;
        if (isFirst || isLast) {
          timestampWrites = {
            querySet: slot.batch.querySet,
            ...(isFirst ? { beginningOfPassWriteIndex: slot.beginIndex } : {}),
            ...(isLast ? { endOfPassWriteIndex: slot.endIndex } : {}),
          };
        }
      }
      const passEncoder = commandEncoder.beginComputePass({ timestampWrites });
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      if (uniformBindGroup)
        passEncoder.setBindGroup(1, uniformBindGroup, [i * uniformAlignment]);
      passEncoder.dispatchWorkgroups(grid[0], grid[1]);
      passEncoder.end();
    }

    if (slot) {
      recordTrace(device, slot, source, filteredPasses.length, shader.code);
    }
  }

  device.queue.submit([commandEncoder.finish()]);
}

function combineUniforms(
  device: GPUDevice,
  uniforms: Uint8Array<ArrayBuffer>[],
): [GPUBuffer, number] {
  for (const buf of uniforms) {
    if (
      !buf ||
      buf.byteLength === 0 ||
      buf.byteLength !== uniforms[0].byteLength
    ) {
      throw new Error("webgpu: Uniform mismatch between shader passes");
    }
  }
  const minAlign = device.limits.minUniformBufferOffsetAlignment;
  const alignment = Math.ceil(uniforms[0].byteLength / minAlign) * minAlign;
  const buffer = device.createBuffer({
    size: alignment * uniforms.length,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  const bufferMapped = new Uint8Array(buffer.getMappedRange());
  for (let i = 0; i < uniforms.length; i++)
    bufferMapped.set(uniforms[i], i * alignment);
  buffer.unmap();
  return [buffer, alignment];
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

  #getLayout(shader: ShaderInfo): GPUPipelineLayout {
    if (
      shader.numInputs + shader.numOutputs >
      this.device.limits.maxStorageBuffersPerShaderStage
    ) {
      // This is a hard limit in WebGPU. All platforms have at least 8 storage
      // buffers per shader stage, and >99% support 10. If you pass more than this
      // many inputs then you risk running into this limit.
      const actual = shader.numInputs + shader.numOutputs;
      const max = this.device.limits.maxStorageBuffersPerShaderStage;
      throw new Error(
        `Too many buffers (${actual}) for WebGPU pipeline (max: ${max})`,
      );
    }
    const bindGroupLayouts: GPUBindGroupLayout[] = [
      this.device.createBindGroupLayout({
        entries: range(shader.numInputs + shader.numOutputs).map((i) => ({
          binding: i,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: i < shader.numInputs ? "read-only-storage" : "storage",
          },
        })),
      }),
    ];
    if (shader.hasUniform) {
      bindGroupLayouts.push(
        this.device.createBindGroupLayout({
          entries: [
            {
              binding: 0,
              visibility: GPUShaderStage.COMPUTE,
              buffer: { type: "uniform", hasDynamicOffset: true },
            },
          ],
        }),
      );
    }
    return this.device.createPipelineLayout({ bindGroupLayouts });
  }

  async prepare(shader: ShaderInfo): Promise<GPUComputePipeline> {
    const existingPipeline = this.cache.get(shader.code);
    if (existingPipeline) return existingPipeline;

    const existingPromise = this.inProgress.get(shader.code);
    if (existingPromise) return await existingPromise;

    if (DEBUG >= 2) {
      console.info("=========== WebGPU shader ===========\n" + shader.code);
    }

    const shaderModule = this.device.createShaderModule({ code: shader.code });
    const promise = (async () => {
      this.device.pushErrorScope("validation");
      try {
        const pipeline = await this.device.createComputePipelineAsync({
          layout: this.#getLayout(shader),
          compute: {
            module: shaderModule,
            entryPoint: "main",
          },
        });
        await this.device.popErrorScope();
        return pipeline;
      } catch (_error: unknown) {
        // This can race with other compilations, but it shouldn't happen in
        // correct code. Any validation error here is a bug in `jax-js`.
        const scope = await this.device.popErrorScope();
        const emsg = await compileError(shaderModule, scope, shader.code);
        throw new Error(emsg);
      }
    })();
    this.inProgress.set(shader.code, promise);

    // This could race against getSync(), but it's okay since shader pipeline
    // creation is deterministic + idempotent.
    const pipeline = await promise;
    this.cache.set(shader.code, pipeline);
    return pipeline;
  }

  prepareSync(shader: ShaderInfo): GPUComputePipeline {
    const existingPipeline = this.cache.get(shader.code);
    if (existingPipeline) return existingPipeline;

    if (DEBUG >= 2) {
      console.info("=========== WebGPU shader ===========\n" + shader.code);
    }

    const shaderModule = this.device.createShaderModule({ code: shader.code });
    this.device.pushErrorScope("validation");
    const pipeline = this.device.createComputePipeline({
      layout: this.#getLayout(shader),
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
        const emsg = await compileError(shaderModule, scope, shader.code);
        console.error(emsg);
      }
    });
    this.cache.set(shader.code, pipeline);
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
