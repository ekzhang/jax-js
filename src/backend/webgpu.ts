import { AluExp, AluGroup, AluOp, DType, isFloatDtype, Kernel } from "../alu";
import {
  Backend,
  Device,
  Executable,
  Slot,
  SlotError,
  UnsupportedOpError,
} from "../backend";
import { tuneWebgpu } from "../tuner";
import { DEBUG, findPow2, FpHash, strip1, union } from "../utils";

interface ShaderInfo {
  shader: string;
  grid: [number, number];
}

interface ShaderDispatch extends ShaderInfo {
  pipeline: GPUComputePipeline;
}

/** Implementation of `Backend` that uses WebGPU in browsers. */
export class WebGPUBackend implements Backend {
  readonly type: Device = "webgpu";
  readonly maxArgs: number;

  readonly pipelines: ShaderPipelineCache;
  readonly syncReader: SyncReader;
  readonly buffers: Map<Slot, { ref: number; buffer: GPUBuffer }>;
  nextSlot: number;

  #cachedShaderMap = new Map<bigint, ShaderInfo>();

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
  }

  malloc(size: number, initialData?: Uint8Array): Slot {
    let buffer: GPUBuffer;
    if (initialData) {
      if (initialData.byteLength !== size) {
        throw new Error("initialData size does not match buffer size");
      }
      if (initialData.byteLength < 4096) {
        buffer = this.#createBuffer(size, { mapped: true });
        new Uint8Array(buffer.getMappedRange()).set(initialData);
        buffer.unmap();
      } else {
        // getMappedRange() seems slower for large buffers, use writeBuffer() instead.
        buffer = this.#createBuffer(size);
        this.device.queue.writeBuffer(buffer, 0, initialData);
      }
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
      // The GPUBuffer.destroy() method does not actually free the memory until
      // pending work is done.
      buffer.buffer.destroy();
    }
  }

  async read(slot: Slot, start?: number, count?: number): Promise<Uint8Array> {
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
      return new Uint8Array(arrayBuffer.slice());
    } finally {
      staging.destroy();
    }
  }

  readSync(slot: Slot, start?: number, count?: number): Uint8Array {
    const buffer = this.#getBuffer(slot);
    if (start === undefined) start = 0;
    if (count === undefined) count = buffer.size - start;
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

  async prepare(kernel: Kernel): Promise<Executable<ShaderDispatch>> {
    const { shader, grid } = this.#cachedShader(kernel);
    const pipeline = await this.pipelines.prepare(shader);
    return new Executable(kernel, { shader, grid, pipeline });
  }

  prepareSync(kernel: Kernel): Executable<ShaderDispatch> {
    const { shader, grid } = this.#cachedShader(kernel);
    const pipeline = this.pipelines.prepareSync(shader);
    return new Executable(kernel, { shader, grid, pipeline });
  }

  dispatch(
    exe: Executable<ShaderDispatch>,
    inputs: Slot[],
    outputs: Slot[],
  ): void {
    const inputBuffers = inputs.map((slot) => this.#getBuffer(slot));
    const outputBuffers = outputs.map((slot) => this.#getBuffer(slot));
    pipelineSubmit(this.device, exe.data, inputBuffers, outputBuffers);
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

function dtypeToWgsl(dtype: DType, storage: boolean = false): string {
  switch (dtype) {
    case DType.Bool:
      return storage ? "i32" : "bool"; // WebGPU does not support bools in buffers.
    case DType.Int32:
      return "i32";
    case DType.Uint32:
      return "u32"; // WebGPU supports uint32 in buffers.
    case DType.Float32:
      return "f32";
    default:
      throw new Error(`Unsupported dtype: ${dtype}`);
  }
}

function constToWgsl(dtype: DType, value: any): string {
  if (dtype === DType.Bool) return value ? "true" : "false";
  if (dtype === DType.Int32) return value.toString();
  if (dtype === DType.Uint32) return value.toString() + "u"; // WebGPU uses 'u' suffix for uint32.
  if (dtype === DType.Float32) {
    if (Number.isNaN(value)) return "nan()";
    if (!Number.isFinite(value)) return value > 0 ? "inf()" : "-inf()";
    return "f32(" + value.toString() + ")";
  }
  throw new Error(`Unsupported const dtype: ${dtype}`);
}

/**
 * Compiles an expression into WebGPU shader source code.
 *
 * Returns the shader source and the number of workgroups to dispatch along x
 * and y axes, to run the kernel.
 */
function pipelineSource(device: GPUDevice, kernel: Kernel): ShaderInfo {
  const tune = tuneWebgpu(kernel);
  if (DEBUG >= 3) {
    console.info(`kernel.exp: ${kernel.exp}\ntune.exp: ${tune.exp}`);
  }

  const { nargs, reduction: re } = kernel;
  const args = Array.from({ length: nargs }, (_, i) => `in${i}`);

  // binding(0..n-1): input buffers
  // binding(n): output buffer

  const shader: string[] = []; // line-separated
  let indent = "";
  const pushIndent = Symbol("pushIndent");
  const popIndent = Symbol("popIndent");
  const emit = (...lines: (string | symbol)[]) => {
    for (const line of lines) {
      if (line === pushIndent) indent += "  ";
      else if (line === popIndent) indent = indent.slice(0, -2);
      else shader.push(line ? indent + (line as string) : line);
    }
  };

  // Global functions at the start of the shader..
  emit(
    // Const-folded infinity and NaN result in compile errors.
    "fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }",
    "fn inf() -> f32 { let bits = 0x7f800000u; return bitcast<f32>(bits); }",
  );

  const distinctOps = union(tune.exp.distinctOps(), re?.fusion.distinctOps());
  if (distinctOps.has(AluOp.Threefry2x32)) {
    emit(threefrySrc);
  }

  // End global function definitions.
  emit("");

  const usedArgs: (DType | null)[] = Array.from({ length: nargs }, () => null);
  tune.exp.fold((exp) => {
    if (exp.op === AluOp.GlobalIndex) usedArgs[exp.arg[0]] = exp.dtype;
  });

  for (let i = 0; i < nargs; i++) {
    // If not used, just assume float32, all that matters is size / alignment.
    const ty = dtypeToWgsl(usedArgs[i] ?? DType.Float32, true);
    emit(
      `@group(0) @binding(${i}) var<storage, read> ${args[i]} : array<${ty}>;`,
    );
  }

  const resultTy = dtypeToWgsl(kernel.dtype, true);
  emit(
    `@group(0) @binding(${nargs}) var<storage, read_write> result : array<${resultTy}>;`,
  );

  const workgroupSize = findPow2(tune.threadCount, 256);

  // Determine grid size, may need to be 3D due to limits on X.
  // maxComputeWorkgroupsPerDimension ~ 65535, so we use 16384 when exceeded.
  const gridSize = Math.ceil(tune.threadCount / workgroupSize);
  let gridX = gridSize;
  let gridY = 1;
  if (gridSize > device.limits.maxComputeWorkgroupsPerDimension) {
    gridX = 16384; // 2^14
    gridY = Math.ceil(gridSize / gridX);
  }

  emit(
    "",
    `@compute @workgroup_size(${workgroupSize})`,
    "fn main(@builtin(global_invocation_id) id : vec3<u32>) {",
    pushIndent,
  );
  if (gridY === 1) {
    emit(
      `if (id.x >= ${tune.threadCount}) { return; }`,
      "let gidx: i32 = i32(id.x);",
    );
  } else {
    const sizeX = gridX * workgroupSize;
    emit(
      `if (${sizeX} * id.y + id.x >= ${tune.threadCount}) { return; }`,
      `let gidx: i32 = i32(${sizeX} * id.y + id.x);`,
    );
  }

  // Generate code for each AluExp operation.
  // Some expressions may be used twice, so we keep track of them.
  let gensymCount = 0;
  const gensym = () => `alu${gensymCount++}`;

  // Insert phony assignments for inputs that are not in use.
  // https://github.com/gpuweb/gpuweb/discussions/4582#discussioncomment-9146686
  for (let i = 0; i < args.length; i++) {
    if (!usedArgs[i]) emit(`_ = &${args[i]};`);
  }

  const references = new Map<AluExp, number>();
  const seen = new Set<AluExp>();
  const countReferences = (exp: AluExp) => {
    references.set(exp, (references.get(exp) ?? 0) + 1);
    if (!seen.has(exp)) {
      seen.add(exp);
      for (const src of exp.src) countReferences(src);
    }
  };

  const expContext = new Map<AluExp, string>();
  const gen = (exp: AluExp): string => {
    if (expContext.has(exp)) return expContext.get(exp)!;
    const { op, src, dtype, arg } = exp;

    // Some of these cases early `return` to force-inline them.
    let source = "";
    if (AluGroup.Binary.has(op) || AluGroup.Compare.has(op)) {
      const a = gen(src[0]);
      const b = gen(src[1]);
      if (op === AluOp.Add) {
        if (dtype === DType.Bool) source = `(${a} || ${b})`;
        else source = `(${a} + ${b})`;
      } else if (op === AluOp.Sub) source = `(${a} - ${b})`;
      else if (op === AluOp.Mul) {
        if (dtype === DType.Bool) source = `(${a} && ${b})`;
        else source = `(${a} * ${b})`;
      } else if (op === AluOp.Idiv)
        source = isFloatDtype(dtype) ? `trunc(${a} / ${b})` : `(${a} / ${b})`;
      else if (op === AluOp.Mod) source = `(${a} % ${b})`;
      else if (op === AluOp.Min) source = `min(${strip1(a)}, ${strip1(b)})`;
      else if (op === AluOp.Max) source = `max(${strip1(a)}, ${strip1(b)})`;
      else if (op === AluOp.Cmplt) source = `(${a} < ${b})`;
      else if (op === AluOp.Cmpne) source = `(${a} != ${b})`;
    } else if (AluGroup.Unary.has(op)) {
      const a = gen(src[0]);
      if (op === AluOp.Sin) source = `sin(${a})`;
      else if (op === AluOp.Cos) source = `cos(${a})`;
      else if (op === AluOp.Exp) source = `exp(${a})`;
      else if (op === AluOp.Log) source = `log(${a})`;
      else if (op === AluOp.Sqrt) source = `sqrt(${a})`;
      else if (op === AluOp.Reciprocal) source = `(1.0 / ${a})`;
      else if (op === AluOp.Cast)
        source = `${dtypeToWgsl(dtype)}(${strip1(a)})`;
      else if (op === AluOp.Bitcast)
        source = `bitcast<${dtypeToWgsl(dtype)}>(${strip1(a)})`;
    } else if (op === AluOp.Where) {
      // select(f, t, cond) -> cond ? t : f
      source = `select(${strip1(gen(src[2]))}, ${strip1(gen(src[1]))}, ${strip1(gen(src[0]))})`;
    } else if (op === AluOp.Threefry2x32) {
      const x = gensym(); // temporary to hold the `vec2<u32>(x0, x1)`
      const [k0, k1, c0, c1] = src.map((x) => strip1(gen(x)));
      emit(`let ${x} = threefry2x32(vec2(${k0}, ${k1}), vec2(${c0}, ${c1}));`);
      if (arg === "xor") source = `(${x}.x ^ ${x}.y)`;
      else if (arg === 0) source = `${x}.x`;
      else if (arg === 1) source = `${x}.y`;
      else throw new UnsupportedOpError(op, dtype, "webgpu", arg);
    } else if (op === AluOp.Const) {
      return constToWgsl(dtype, arg);
    } else if (op === AluOp.Special) {
      return arg[0] as string;
    } else if (op === AluOp.Variable) {
      return arg as string;
    } else if (op === AluOp.GlobalIndex) {
      source = `${args[arg[0]]}[${strip1(gen(src[0]))}]`;
      if (dtype === DType.Bool) source = `(${source} != 0)`; // bool is represented as i32
    }

    if (!source) throw new UnsupportedOpError(op, dtype, "webgpu", arg);
    const typeName = dtypeToWgsl(dtype);
    if ((references.get(exp) ?? 0) > 1) {
      const name = gensym();
      expContext.set(exp, name);
      emit(`let ${name}: ${typeName} = ${strip1(source)};`);
      return name;
    } else {
      expContext.set(exp, source);
      return source;
    }
  };

  if (!re) {
    countReferences(tune.exp);
    let rhs = strip1(gen(tune.exp));
    if (resultTy !== dtypeToWgsl(tune.exp.dtype)) rhs = `${resultTy}(${rhs})`;
    emit(`result[gidx] = ${rhs};`);
  } else {
    if ((tune.size.groups ?? 1) > 1) {
      throw new Error("WebGPU backend does not support group optimization yet");
    }
    const unroll = tune.size.unroll ?? 1;
    const upcast = tune.size.upcast ?? 1;

    const acc = [...Array(upcast)].map((_, i) => `acc${i}`);
    for (let i = 0; i < upcast; i++) {
      emit(
        `var ${acc[i]}: ${dtypeToWgsl(re.dtype)} = ${constToWgsl(re.dtype, re.identity)};`,
      ); // Initialize accumulators.
    }

    emit(
      `for (var ridx: i32 = 0; ridx < ${tune.size.reduce}; ridx++) {`,
      pushIndent,
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
        countReferences(exps[up][un]);
      }
    }

    // After references are counted, we can generate the code.
    const items = exps.map((ar) => ar.map(gen).map(strip1));
    for (let i = 0; i < upcast; i++) {
      let rhs = items[i][0];
      for (let j = 1; j < unroll; j++) {
        if (re.op === AluOp.Add) rhs = `${rhs} + ${items[i][j]}`;
        else if (re.op === AluOp.Mul) rhs = `${rhs} * ${items[i][j]}`;
        else if (re.op === AluOp.Min) rhs = `min(${rhs}, ${items[i][j]})`;
        else if (re.op === AluOp.Max) rhs = `max(${rhs}, ${items[i][j]})`;
        else throw new Error(`Unsupported reduction op: ${re.op}`);
      }
      if (re.op === AluOp.Add) emit(`${acc[i]} += ${rhs};`);
      else if (re.op === AluOp.Mul) emit(`${acc[i]} *= ${rhs};`);
      else if (re.op === AluOp.Min) emit(`${acc[i]} = min(${acc[i]}, ${rhs});`);
      else if (re.op === AluOp.Max) emit(`${acc[i]} = max(${acc[i]}, ${rhs});`);
      else throw new Error(`Unsupported reduction op: ${re.op}`);
    }
    emit(popIndent, "}");

    // Exited the reduction loop scope. Erase any local variables.
    expContext.clear();
    references.clear();
    seen.clear();

    const outputIdxExps: AluExp[] = [];
    const fusionExps: AluExp[] = [];
    for (let i = 0; i < upcast; i++) {
      const exp = tune.outputIdxExp.substitute({ upcast: AluExp.i32(i) });
      outputIdxExps.push(exp.simplify(cache));
      countReferences(outputIdxExps[i]);
      fusionExps.push(
        re.fusion
          .substitute({ acc: AluExp.variable(re.dtype, acc[i]) })
          .simplify(cache),
      );
      countReferences(fusionExps[i]);
    }
    for (let i = 0; i < upcast; i++) {
      const index = strip1(gen(outputIdxExps[i]));
      let rhs = strip1(gen(fusionExps[i]));
      if (resultTy !== dtypeToWgsl(fusionExps[i].dtype))
        rhs = `${resultTy}(${rhs})`;
      emit(`result[${index}] = ${rhs};`);
    }
  }

  emit(popIndent, "}");
  return {
    shader: shader.join("\n"),
    grid: [gridX, gridY],
  };
}

function pipelineSubmit(
  device: GPUDevice,
  { pipeline, grid }: ShaderDispatch,
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

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      ...inputs.map((buffer, i) => {
        return { binding: i, resource: { buffer } };
      }),
      { binding: inputs.length, resource: { buffer: outputs[0] } },
    ],
  });

  const commandEncoder = device.createCommandEncoder();
  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(grid[0], grid[1]);
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

  async prepare(code: string): Promise<GPUComputePipeline> {
    const existingPipeline = this.cache.get(code);
    if (existingPipeline) return existingPipeline;

    const existingPromise = this.inProgress.get(code);
    if (existingPromise) return await existingPromise;

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
      } catch (_error: unknown) {
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

  prepareSync(code: string): GPUComputePipeline {
    const existingPipeline = this.cache.get(code);
    if (existingPipeline) return existingPipeline;

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

/**
 * Graphics state used to synchronously read data from WebGPU buffers.
 *
 * This trick is borrowed from TensorFlow.js. Basically, the idea is to create
 * an offscreen canvas with one pixel for every 4 bytes ("device storage"), then
 * configure it with a WebGPU context. Copy the buffer to a texture, then draw
 * the canvas onto another offscreen canvas with '2d' context ("host storage").
 *
 * Once it's on host storage, we can use `getImageData()` to read the pixels
 * from the image directly.
 *
 * We use 256x256 canvases here (256 KiB). The performance of this is bad
 * because it involves multiple data copies, but it still works. We also
 * actually need to copy the image twice: once in "opaque" mode for the RGB
 * values, and once in "premultiplied" mode for the alpha channel.
 *
 * https://github.com/tensorflow/tfjs/blob/tfjs-v4.22.0/tfjs-backend-webgpu/src/backend_webgpu.ts#L379
 */
class SyncReader {
  static readonly alphaModes: GPUCanvasAlphaMode[] = [
    "opaque",
    "premultiplied",
  ];
  static readonly width = 256;
  static readonly height = 256;

  initialized = false;
  deviceStorage?: OffscreenCanvas[];
  deviceContexts?: GPUCanvasContext[];
  hostStorage?: OffscreenCanvas;
  hostContext?: OffscreenCanvasRenderingContext2D;

  constructor(readonly device: GPUDevice) {}

  #init() {
    const makeCanvas = () =>
      new OffscreenCanvas(SyncReader.width, SyncReader.height);
    this.deviceStorage = SyncReader.alphaModes.map(makeCanvas);
    this.deviceContexts = this.deviceStorage.map((canvas, i) => {
      const context = canvas.getContext("webgpu")!;
      context.configure({
        device: this.device,
        // rgba8unorm is not supported on Chrome for macOS.
        // https://bugs.chromium.org/p/chromium/issues/detail?id=1298618
        format: "bgra8unorm",
        usage: GPUTextureUsage.COPY_DST,
        alphaMode: SyncReader.alphaModes[i],
      });
      return context;
    });
    this.hostStorage = makeCanvas();
    this.hostContext = this.hostStorage.getContext("2d", {
      willReadFrequently: true,
    })!;
    this.initialized = true;
  }

  read(buffer: GPUBuffer, start: number, count: number): Uint8Array {
    if (!this.initialized) this.#init();

    if (count % 4 !== 0) {
      throw new Error("Read size must be a multiple of 4 bytes");
    }

    const deviceStorage = this.deviceStorage!;
    const deviceContexts = this.deviceContexts!;
    const hostContext = this.hostContext!;

    const pixelsSize = count / 4;
    const bytesPerRow = SyncReader.width * 4;
    const valsGPU = new ArrayBuffer(count);

    for (let i = 0; i < deviceContexts.length; i++) {
      const texture = deviceContexts[i].getCurrentTexture();
      // Read data using a (width, height) image at `offset` in valsGPU.
      const readData = (width: number, height: number, offset: number) => {
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToTexture(
          { buffer, bytesPerRow, offset: offset + start },
          { texture },
          { width, height, depthOrArrayLayers: 1 },
        );
        const commandBuffer = encoder.finish();
        this.device.queue.submit([commandBuffer]);

        hostContext.clearRect(0, 0, width, height);
        hostContext.drawImage(deviceStorage[i], 0, 0);
        const values = hostContext.getImageData(0, 0, width, height).data;
        const span = new Uint8ClampedArray(valsGPU, offset, 4 * width * height);
        const alphaMode = SyncReader.alphaModes[i];
        for (let k = 0; k < span.length; k += 4) {
          if (alphaMode === "premultiplied") {
            span[k + 3] = values[k + 3];
          } else {
            span[k] = values[k + 2]; // opaque (BGRA)
            span[k + 1] = values[k + 1];
            span[k + 2] = values[k];
          }
        }
      };

      const pixelsPerCanvas = SyncReader.width * SyncReader.height;
      const wholeChunks = Math.floor(pixelsSize / pixelsPerCanvas);
      let remainder = pixelsSize % pixelsPerCanvas;
      const remainderRows = Math.floor(remainder / SyncReader.width);
      remainder = remainder % SyncReader.width;

      let offset = 0;
      // Read entire canvases.
      for (let j = 0; j < wholeChunks; j++) {
        readData(SyncReader.width, SyncReader.height, offset);
        offset += pixelsPerCanvas * 4;
      }
      // Read a partial canvas with whole rows.
      if (remainderRows > 0) {
        readData(SyncReader.width, remainderRows, offset);
        offset += remainderRows * SyncReader.width * 4;
      }
      // Read a partial canvas with some columns in the first row.
      if (remainder > 0) readData(remainder, 1, offset);
    }

    return new Uint8Array(valsGPU);
  }
}

// Port of the JavaScript `threefry2x32()` function, see alu.ts for details.
const threefrySrc = `
fn threefry2x32(key: vec2<u32>, ctr: vec2<u32>) -> vec2<u32> {
  let ks0: u32 = key.x;
  let ks1: u32 = key.y;
  let ks2: u32 = ks0 ^ ks1 ^ 0x1BD11BDAu;

  var x0: u32 = ctr.x + ks0;
  var x1: u32 = ctr.y + ks1;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks1;
  x1 += ks2 + 1u;

  x0 += x1; x1 = (x1 << 17u) | (x1 >> 15u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 29u) | (x1 >> 3u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 16u) | (x1 >> 16u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 24u) | (x1 >> 8u); x1 ^= x0;
  x0 += ks2;
  x1 += ks0 + 2u;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks0;
  x1 += ks1 + 3u;

  x0 += x1; x1 = (x1 << 17u) | (x1 >> 15u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 29u) | (x1 >> 3u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 16u) | (x1 >> 16u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 24u) | (x1 >> 8u); x1 ^= x0;
  x0 += ks1;
  x1 += ks2 + 4u;

  x0 += x1; x1 = (x1 << 13u) | (x1 >> 19u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 15u) | (x1 >> 17u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 26u) | (x1 >> 6u); x1 ^= x0;
  x0 += x1; x1 = (x1 << 6u) | (x1 >> 26u); x1 ^= x0;
  x0 += ks2;
  x1 += ks0 + 5u;

  return vec2<u32>(x0, x1);
}`;
