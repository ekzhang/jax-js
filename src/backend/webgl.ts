import { AluExp, AluGroup, AluOp, DType, isFloatDtype, Kernel } from "../alu";
import {
  Backend,
  Device,
  Executable,
  Slot,
  SlotError,
  UnsupportedOpError,
  UnsupportedRoutineError,
} from "../backend";
import { Routine } from "../routine";
import { tuneNullopt } from "../tuner";
import { DEBUG, range, strip1 } from "../utils";
import { erfSrc, threefrySrc } from "./webgl/builtins";

/** Information about a compiled WebGL shader program. */
interface ShaderInfo {
  code: string;
  numInputs: number;
  outputSize: [number, number]; // Width and height of the output texture in texels.
  outputDtype: DType; // Informational, dtype of outputs.
}

/** A compiled shader dispatch ready for execution. */
interface ShaderDispatch extends ShaderInfo {
  program: WebGLProgram;
  inputLocations: (WebGLUniformLocation | null)[];
}

/** Buffer stored as a texture in WebGL. */
interface WebGLSlot {
  ref: number;
  size: number; // in bytes
  texture: WebGLTexture;
  width: number;
  height: number;
}

/**
 * No-frills backend that uses WebGL2 textures and shaders for compute.
 *
 * WebGL2 is available in almost all modern browsers, and it has options for
 * floating-point numbers and integers in textures. However, it's still not a
 * "real" compute API, and only float32 arithmetic is available.
 *
 * We make this backend available in case users want a fallback option compared
 * to WebGPU, which is only on newer browsers and iOS 26+.
 *
 * Implementation notes:
 * - All data is stored in typed RGBA32F textures, regardless of original dtype.
 *   They are converted to the correct data type when loaded in shaders.
 * - Each texel holds 4 float32 values (128 bits).
 * - Compute is done by rendering a full-screen quad with a fragment shader.
 * - Output is rendered to a framebuffer-attached texture, then read back.
 */
export class WebGLBackend implements Backend {
  readonly type: Device = "webgl";
  readonly maxArgs = 8; // See: https://web3dsurvey.com/webgl/parameters/MAX_TEXTURE_IMAGE_UNITS

  readonly gl: WebGL2RenderingContext;
  readonly #fbo: WebGLFramebuffer;
  #buffers: Map<Slot, WebGLSlot>;
  #programCache: Map<string, ShaderDispatch>;
  #nextSlot: number;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.#fbo = gl.createFramebuffer();
    this.#buffers = new Map();
    this.#programCache = new Map();
    this.#nextSlot = 1;
  }

  /** Return the number of currently allocated slots (for leak detection). */
  slotCount(): number {
    return this.#buffers.size;
  }

  /**
   * Allocate a buffer with a specific dtype.
   *
   * All buffers use RGBA32F texture format internally. Data is stored as raw
   * bits and reinterpreted using floatBitsToInt/intBitsToFloat in shaders.
   * This mirrors how WebGPU handles untyped byte buffers.
   */
  malloc(size: number, initialData?: Uint8Array): Slot {
    const gl = this.gl;

    // Calculate number of floats needed (4 bytes per float, 4 floats per texel)
    const numFloats = Math.ceil(size / 4) || 1;
    const numTexels = Math.ceil(numFloats / 4) || 1;
    const { width, height } = computeTextureDimensions(numTexels);

    // Create the texture
    const texture = gl.createTexture();
    if (!texture) throw new Error("Failed to create texture");

    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const totalFloats = width * height * 4;
    let pixels: Float32Array<ArrayBuffer> | null = null;
    if (initialData) {
      // Copy initial data as raw bytes into float array
      pixels = new Float32Array(totalFloats);
      new Uint8Array(pixels.buffer).set(initialData);
    }
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA32F,
      width,
      height,
      0,
      gl.RGBA,
      gl.FLOAT,
      pixels,
    );
    gl.bindTexture(gl.TEXTURE_2D, null);

    const slot = this.#nextSlot++;
    this.#buffers.set(slot, { ref: 1, size, texture, width, height });
    return slot;
  }

  incRef(slot: Slot): void {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref++;
  }

  decRef(slot: Slot): void {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    buffer.ref--;
    if (buffer.ref === 0) {
      this.gl.deleteTexture(buffer.texture);
      this.#buffers.delete(slot);
    }
  }

  async read(
    slot: Slot,
    start?: number,
    count?: number,
  ): Promise<Uint8Array<ArrayBuffer>> {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);

    const gl = this.gl;
    if (start === undefined) start = 0;
    if (count === undefined) count = buffer.size - start;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.#fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      buffer.texture,
      0,
    );

    const totalBytes = buffer.width * buffer.height * 4 * 4; // 4 floats * 4 bytes
    const floatData = new Float32Array(totalBytes / 4);

    // Create PBO for async readback
    const pbo = gl.createBuffer();
    if (!pbo) throw new Error("Failed to create PBO");

    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
    gl.bufferData(gl.PIXEL_PACK_BUFFER, totalBytes, gl.STREAM_READ);

    // Kick off GPU readback into the PBO (returns immediately)
    gl.readPixels(0, 0, buffer.width, buffer.height, gl.RGBA, gl.FLOAT, 0);

    // Check for errors after readPixels
    const readError = gl.getError();
    if (readError !== gl.NO_ERROR) {
      gl.deleteBuffer(pbo);
      throw new Error(`WebGL error after readPixels: ${readError}`);
    }

    // Create fence to know when GPU is done
    const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    if (!sync) throw new Error("Failed to create sync object");
    gl.flush();

    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    // Poll for completion without blocking
    await new Promise<void>((resolve, reject) => {
      const poll = () => {
        const status = gl.clientWaitSync(sync, 0, 0);
        if (status === gl.TIMEOUT_EXPIRED) {
          // Try again after 5 milliseconds
          setTimeout(poll, 5);
          return;
        }
        if (status === gl.WAIT_FAILED) {
          gl.deleteSync(sync);
          gl.deleteBuffer(pbo);
          reject(new Error("clientWaitSync failed"));
          return;
        }
        resolve();
      };
      poll();
    });

    // GPU done: copy from PBO into floatData
    gl.deleteSync(sync);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
    gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, floatData);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    gl.deleteBuffer(pbo);

    // Convert to bytes and extract requested range
    const byteData = new Uint8Array(floatData.buffer as ArrayBuffer);
    return new Uint8Array(byteData.slice(start, start + count));
  }

  readSync(
    slot: Slot,
    start?: number,
    count?: number,
  ): Uint8Array<ArrayBuffer> {
    const buffer = this.#buffers.get(slot);
    if (!buffer) throw new SlotError(slot);

    const gl = this.gl;
    if (start === undefined) start = 0;
    if (count === undefined) count = buffer.size - start;

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.#fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      buffer.texture,
      0,
    );

    const totalFloats = buffer.width * buffer.height * 4;
    const floatData = new Float32Array(totalFloats);
    gl.readPixels(
      0,
      0,
      buffer.width,
      buffer.height,
      gl.RGBA,
      gl.FLOAT,
      floatData,
    );
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const byteData = new Uint8Array(floatData.buffer as ArrayBuffer);
    return new Uint8Array(byteData.slice(start, start + count));
  }

  async prepareKernel(kernel: Kernel): Promise<Executable<ShaderDispatch>> {
    return this.prepareKernelSync(kernel);
  }

  prepareKernelSync(kernel: Kernel): Executable<ShaderDispatch> {
    const shader = generateShader(kernel);
    const cached = this.#programCache.get(shader.code);
    if (cached) return new Executable(kernel, cached);
    const dispatch = compileShader(this.gl, shader);
    this.#programCache.set(shader.code, dispatch);
    return new Executable(kernel, dispatch);
  }

  prepareRoutine(routine: Routine): Promise<Executable> {
    throw new UnsupportedRoutineError(routine.name, "webgl");
  }

  prepareRoutineSync(routine: Routine): Executable {
    throw new UnsupportedRoutineError(routine.name, "webgl");
  }

  dispatch(
    exe: Executable<ShaderDispatch>,
    inputs: Slot[],
    outputs: Slot[],
  ): void {
    const gl = this.gl;
    if (gl.isContextLost())
      throw new Error("WebGL context lost - cannot dispatch");

    const { program, inputLocations } = exe.data;

    if (inputs.length !== exe.data.numInputs)
      throw new Error(
        `Expected ${exe.data.numInputs} inputs, got ${inputs.length}`,
      );
    if (outputs.length !== 1)
      throw new Error(`Expected 1 output, got ${outputs.length}`);

    const outputBuffer = this.#buffers.get(outputs[0]);
    if (!outputBuffer) throw new SlotError(outputs[0]);

    // Bind output texture to framebuffer
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.#fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      outputBuffer.texture,
      0,
    );

    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) {
      throw new Error(`Framebuffer incomplete: ${status}`);
    }

    gl.viewport(0, 0, outputBuffer.width, outputBuffer.height);
    gl.useProgram(program);
    for (let i = 0; i < inputs.length; i++) {
      const inputBuffer = this.#buffers.get(inputs[i]);
      if (!inputBuffer) throw new SlotError(inputs[i]);
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, inputBuffer.texture);
      if (inputLocations[i] !== null) {
        gl.uniform1i(inputLocations[i], i);
      }
    }
    gl.drawArrays(gl.TRIANGLES, 0, 3); // Full-screen triangle draw

    // Check for GL errors after draw
    const error = gl.getError();
    if (error !== gl.NO_ERROR) {
      let errorName: string;
      if (error === gl.INVALID_ENUM) errorName = "INVALID_ENUM";
      else if (error === gl.INVALID_VALUE) errorName = "INVALID_VALUE";
      else if (error === gl.INVALID_OPERATION) errorName = "INVALID_OPERATION";
      else if (error === gl.INVALID_FRAMEBUFFER_OPERATION)
        errorName = "INVALID_FRAMEBUFFER_OPERATION";
      else if (error === gl.OUT_OF_MEMORY) errorName = "OUT_OF_MEMORY";
      else if (error === gl.CONTEXT_LOST_WEBGL)
        errorName = "CONTEXT_LOST_WEBGL";
      else errorName = `UNKNOWN(${error})`;
      throw new Error(`WebGL error after drawArrays: ${errorName}`);
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.useProgram(null);
  }
}

function generateShader(kernel: Kernel): ShaderInfo {
  const tune = tuneNullopt(kernel);
  if (DEBUG >= 3) {
    console.info(`webgl kernel.exp: ${kernel.exp}\ntune.exp: ${tune.exp}`);
  }

  const { nargs, reduction: re } = kernel;
  const outputDtype = kernel.dtype;

  // Calculate output texture dimensions (4 elements per texel)
  const numTexels = Math.ceil(kernel.size / 4) || 1;
  const outputSize = computeTextureDimensions(numTexels);

  // Collect input dtypes from GlobalIndex operations and detect builtins needed
  const inputDtypes: DType[] = Array(nargs).fill(DType.Float32);
  const builtins = { erf: false, threefry: false };
  const collectInfo = (exp: AluExp) => {
    if (exp.op === AluOp.GlobalIndex) {
      inputDtypes[exp.arg[0]] = exp.dtype;
    } else if (exp.op === AluOp.Erf || exp.op === AluOp.Erfc) {
      builtins.erf = true;
    } else if (exp.op === AluOp.Threefry2x32) {
      builtins.threefry = true;
    }
  };
  tune.exp.fold(collectInfo);
  tune.epilogue?.fold(collectInfo);

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

  emit(
    // Basic WebGL2 header, use high precision (32-bit) float and ints
    "#version 300 es",
    "precision highp float;",
    "precision highp int;",
    "",
  );

  // All samplers are float textures (RGBA32F) - we use bit reinterpretation
  const args = Array.from({ length: nargs }, (_, i) => `in${i}`);
  const resultType = glslType(outputDtype);
  for (let i = 0; i < nargs; i++) {
    emit(`uniform highp sampler2D ${args[i]};`);
  }

  // Output is always vec4 (RGBA32F) - we convert non-float types with bit casts
  emit("out vec4 out0;");

  // Generate fetch functions for each input dtype
  // All textures are RGBA32F, but we reinterpret bits based on dtype
  const fetchFunctions = new Set<DType>();
  for (const dtype of inputDtypes) {
    fetchFunctions.add(dtype);
  }
  for (const dtype of fetchFunctions) {
    emit(generateLoadFunction(dtype));
  }

  // Emit builtin functions
  if (builtins.erf) emit(erfSrc);
  if (builtins.threefry) emit(threefrySrc);

  // Begin compute() function
  emit(
    `${resultType} compute(int gidx) {`,
    pushIndent,
    `${resultType} result = ${constToGlsl(outputDtype, 0)};`,
    `if (gidx < ${kernel.size}) {`,
    pushIndent,
  );
  if (!re) {
    // Non-reduction kernel: simple element-wise computation
    const code = generateExpression(tune.exp, args, inputDtypes);
    emit(`result = ${strip1(code)};`);
  } else {
    // Reduction kernel: need accumulator and reduction loop
    const accType = glslType(re.dtype);
    const accInit = constToGlsl(re.dtype, re.identity);
    emit(
      `${accType} acc = ${accInit};`,
      `for (int ridx = 0; ridx < ${tune.size.reduce}; ridx++) {`,
      pushIndent,
    );
    const code = generateExpression(tune.exp, args, inputDtypes);
    if (re.op === AluOp.Add) emit(`acc += ${strip1(code)};`);
    else if (re.op === AluOp.Mul) emit(`acc *= ${strip1(code)};`);
    else if (re.op === AluOp.Min) {
      if (re.dtype !== DType.Bool) emit(`acc = min(acc, ${strip1(code)});`);
      else emit(`acc = acc && ${code};`);
    } else if (re.op === AluOp.Max) {
      if (re.dtype !== DType.Bool) emit(`acc = max(acc, ${strip1(code)});`);
      else emit(`acc = acc || ${code};`);
    } else {
      throw new Error(`Unsupported reduction op: ${re.op}`);
    }
    emit(popIndent, "}"); // End reduction loop
    emit(`result = ${generateExpression(tune.epilogue!, args, inputDtypes)};`);
  }
  emit(popIndent, "}", "return result;", popIndent, "}\n"); // End compute() function

  emit(
    "void main() {",
    pushIndent,
    "ivec2 fragCoord = ivec2(gl_FragCoord.xy);",
    `int texelIdx = fragCoord.y * ${outputSize.width} + fragCoord.x;`,
    `${resultType} result0 = compute(texelIdx * 4);`,
    `${resultType} result1 = compute(texelIdx * 4 + 1);`,
    `${resultType} result2 = compute(texelIdx * 4 + 2);`,
    `${resultType} result3 = compute(texelIdx * 4 + 3);`,
    // Convert output to vec4 (RGBA32F) - use bit casts for non-float types
    `out0 = vec4(${range(4)
      .map((i) => toRGBA32F(outputDtype, `result${i}`))
      .join(", ")});`,
  );
  emit(popIndent, "}");

  return {
    code: shader.join("\n"),
    numInputs: nargs,
    outputSize: [outputSize.width, outputSize.height],
    outputDtype,
  };
}

function compile(gl: WebGL2RenderingContext, type: GLenum, src: string) {
  const s = gl.createShader(type)!;
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(s) ?? "Unknown shader compile error");
  }
  return s;
}

function link(gl: WebGL2RenderingContext, vsSrc: string, fsSrc: string) {
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS))
    throw new Error(gl.getProgramInfoLog(p) ?? "Unknown program link error");
  return p;
}

// Vertex shader for full-screen triangle
const vertexShaderSource = `#version 300 es
precision highp float;
const vec2 pos[3] = vec2[](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
void main() { gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0); }
`;

function compileShader(
  gl: WebGL2RenderingContext,
  shader: ShaderInfo,
): ShaderDispatch {
  if (DEBUG >= 1) {
    console.info("=========== WebGL shader ===========\n" + shader.code);
  }
  const program = link(gl, vertexShaderSource, shader.code);
  const inputLocations: (WebGLUniformLocation | null)[] = [];
  for (let i = 0; i < shader.numInputs; i++) {
    inputLocations.push(gl.getUniformLocation(program, `in${i}`));
  }
  return {
    ...shader,
    program,
    inputLocations,
  };
}

/** Compute 2D texture dimensions for a given number of texels. */
function computeTextureDimensions(numTexels: number): {
  width: number;
  height: number;
} {
  // Try to make texture roughly square, but ensure width is power of 2 for efficiency
  const maxDim = 16384; // WebGL2 max texture dimension
  let width = Math.min(Math.ceil(Math.sqrt(numTexels)), maxDim);
  // Round up to next power of 2 for width (optional, but helps with some GPUs)
  width = Math.min(1 << Math.ceil(Math.log2(width)), maxDim);
  const height = Math.min(Math.ceil(numTexels / width), maxDim);
  return { width, height };
}

function glslType(dtype: DType): string {
  switch (dtype) {
    case DType.Float32:
      return "float";
    case DType.Int32:
      return "int";
    case DType.Uint32:
      return "uint";
    case DType.Bool:
      return "bool";
    default:
      throw new Error(`Unsupported dtype for WebGL: ${dtype}`);
  }
}

function generateLoadFunction(dtype: DType): string {
  const funcName = `load_${dtype}`;
  const returnType = glslType(dtype);

  // All textures are sampler2D (RGBA32F), read as vec4
  let conversion: string;
  if (isFloatDtype(dtype)) {
    conversion = "val"; // No conversion needed for floats
  } else if (dtype === DType.Int32) {
    conversion = "floatBitsToInt(val)";
  } else if (dtype === DType.Uint32) {
    conversion = "floatBitsToUint(val)";
  } else if (dtype === DType.Bool) {
    conversion = "floatBitsToInt(val) != 0";
  } else {
    throw new Error(`Unsupported dtype for WebGL fetch: ${dtype}`);
  }

  return `
${returnType} ${funcName}(highp sampler2D tex, int idx) {
  ivec2 texSize = textureSize(tex, 0);
  int texel = idx / 4;
  int component = idx - texel * 4;
  ivec2 coord = ivec2(texel % texSize.x, texel / texSize.x);
  vec4 texVal = texelFetch(tex, coord, 0);
  float val;
  if (component == 0) val = texVal.x;
  else if (component == 1) val = texVal.y;
  else if (component == 2) val = texVal.z;
  else val = texVal.w;
  return ${conversion};
}
`;
}

function toRGBA32F(dtype: DType, source: string): string {
  switch (dtype) {
    case DType.Float32:
      return source;
    case DType.Int32:
      return `intBitsToFloat(${source})`;
    case DType.Uint32:
      return `uintBitsToFloat(${source})`;
    case DType.Bool:
      return `intBitsToFloat(${source} ? 1 : 0)`;
    default:
      throw new Error(`Unsupported dtype for WebGL output: ${dtype}`);
  }
}

function constToGlsl(dtype: DType, value: number): string {
  switch (dtype) {
    case DType.Bool:
      return value ? "true" : "false";
    case DType.Int32:
      return value.toString();
    case DType.Uint32:
      return value.toString() + "u";
    case DType.Float32:
      if (Number.isNaN(value)) return "uintBitsToFloat(0x7fc00000u)";
      if (!Number.isFinite(value)) {
        return value > 0
          ? "uintBitsToFloat(0x7f800000u)"
          : "uintBitsToFloat(0xff800000u)";
      }
      return "float(" + value.toString() + ")";
    default:
      throw new Error(`Unsupported dtype for WebGL constant: ${dtype}`);
  }
}

/** Generate GLSL expression code from an AluExp. */
function generateExpression(
  exp: AluExp,
  args: string[],
  inputDtypes: DType[],
): string {
  const expContext = new Map<AluExp, string>();

  const gen = (e: AluExp): string => {
    if (expContext.has(e)) return expContext.get(e)!;

    const { op, src, dtype, arg } = e;
    let source = "";

    if (AluGroup.Binary.has(op)) {
      const a = gen(src[0]);
      const b = gen(src[1]);
      if (op === AluOp.Add) {
        if (dtype === DType.Bool) source = `(${a} || ${b})`;
        else source = `(${a} + ${b})`;
      } else if (op === AluOp.Sub) source = `(${a} - ${b})`;
      else if (op === AluOp.Mul) {
        if (dtype === DType.Bool) source = `(${a} && ${b})`;
        else source = `(${a} * ${b})`;
      } else if (op === AluOp.Idiv) {
        if (isFloatDtype(dtype)) source = `trunc(${a} / ${b})`;
        else source = `(${a} / ${b})`;
      } else if (op === AluOp.Mod) {
        if (isFloatDtype(dtype)) source = `(${a} - ${b} * trunc(${a} / ${b}))`;
        else source = `(${a} % ${b})`;
      } else if (op === AluOp.Min) {
        if (dtype === DType.Bool) source = `(${a} && ${b})`;
        else source = `min(${a}, ${b})`;
      } else if (op === AluOp.Max) {
        if (dtype === DType.Bool) source = `(${a} || ${b})`;
        else source = `max(${a}, ${b})`;
      }
    } else if (AluGroup.Compare.has(op)) {
      const a = gen(src[0]);
      const b = gen(src[1]);
      if (op === AluOp.Cmplt) source = `(${a} < ${b})`;
      else if (op === AluOp.Cmpne) {
        // NaN detection: In GLSL, NaN comparisons have unspecified behavior.
        // Use min() to reliably detect NaN: min(x, inf) returns x unless x is NaN.
        if (isFloatDtype(src[0].dtype))
          source = `(${a} != ${b} || isnan(${a}) || isnan(${b}))`;
        else source = `(${a} != ${b})`;
      }
    } else if (AluGroup.Unary.has(op)) {
      const a = gen(src[0]);
      if (op === AluOp.Sin) source = `sin(${strip1(a)})`;
      else if (op === AluOp.Cos) source = `cos(${strip1(a)})`;
      else if (op === AluOp.Asin) source = `asin(${strip1(a)})`;
      else if (op === AluOp.Atan) source = `atan(${strip1(a)})`;
      else if (op === AluOp.Exp) source = `exp(${strip1(a)})`;
      else if (op === AluOp.Log) source = `log(${strip1(a)})`;
      else if (op === AluOp.Erf) source = `erf(${strip1(a)})`;
      else if (op === AluOp.Erfc) source = `erfc(${strip1(a)})`;
      else if (op === AluOp.Sqrt) source = `sqrt(${strip1(a)})`;
      else if (op === AluOp.Floor) source = `floor(${strip1(a)})`;
      else if (op === AluOp.Ceil) source = `ceil(${strip1(a)})`;
      else if (op === AluOp.Reciprocal) source = `(1.0 / ${a})`;
      else if (op === AluOp.Cast) source = `${glslType(dtype)}(${strip1(a)})`;
      else if (op === AluOp.Bitcast) {
        const dtype0 = src[0].dtype;
        if (dtype === dtype0) source = a;
        else if (dtype === DType.Float32) {
          if (dtype0 === DType.Int32) source = `intBitsToFloat(${strip1(a)})`;
          else if (dtype0 === DType.Uint32)
            source = `uintBitsToFloat(${strip1(a)})`;
        } else if (dtype === DType.Int32) {
          if (dtype0 === DType.Float32) source = `floatBitsToInt(${strip1(a)})`;
          else if (dtype0 === DType.Uint32) source = `int(${strip1(a)})`;
        } else if (dtype === DType.Uint32) {
          if (dtype0 === DType.Float32)
            source = `floatBitsToUint(${strip1(a)})`;
          else if (dtype0 === DType.Int32) source = `uint(${strip1(a)})`;
        }
      }
    } else if (op === AluOp.Threefry2x32) {
      const [k0, k1, c0, c1] = src.map((x) => strip1(gen(x)));
      const mode = arg as string | number;
      const call = `threefry2x32(uvec2(${k0}, ${k1}), uvec2(${c0}, ${c1}))`;
      if (mode === "xor") source = `(${call}.x ^ ${call}.y)`;
      else if (mode === 0) source = `${call}.x`;
      else if (mode === 1) source = `${call}.y`;
    } else if (op === AluOp.Where) {
      const [cond, t, f] = src.map(gen);
      source = `(${cond} ? ${t} : ${f})`;
    } else if (op === AluOp.Const) {
      source = constToGlsl(dtype, arg);
    } else if (op === AluOp.Special) {
      source = arg[0];
    } else if (op === AluOp.Variable) {
      source = arg;
    } else if (op === AluOp.GlobalIndex) {
      const gid: number = arg[0];
      const bufidx = gen(src[0]);
      source = `load_${inputDtypes[gid]}(${args[gid]}, ${strip1(bufidx)})`;
    }
    if (!source) throw new UnsupportedOpError(op, dtype, "webgl", arg);
    expContext.set(e, source);
    return source;
  };

  return gen(exp);
}
