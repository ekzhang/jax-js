import {
  AluExp,
  AluGroup,
  AluOp,
  byteWidth,
  DType,
  isFloatDtype,
  Kernel,
  MultiKernel,
} from "../alu";
import {
  Backend,
  Device,
  Executable,
  Slot,
  SlotError,
  UnsupportedOpError,
} from "../backend";
import { Routine } from "../routine";
import { tuneNullopt, tuneWebgpu } from "../tuner";
import {
  DEBUG,
  findPow2,
  FpHash,
  mapSetUnion,
  prod,
  range,
  strip1,
} from "../utils";
import { erfSrc, threefrySrc } from "./webgpu/builtins";
import {
  calculateGrid,
  constToWgsl,
  dtypeToWgsl,
  headerWgsl,
  ShaderInfo,
} from "./webgpu/codegen";
import { SyncReader } from "./webgpu/reader";
import { createRoutineShader } from "./webgpu/routines";
import {
  createAllIterationsOffsetsBuffer,
  ScanBindingInfo,
  wrapRoutineForScan,
} from "./webgpu/scan-wrapper";

interface ShaderDispatch extends ShaderInfo {
  pipeline: GPUComputePipeline; // Compiled pipeline for the shader.
}

/** Parameters for native scan execution on WebGPU (elementwise kernel body). */
export interface NativeScanParams {
  /** Number of scan iterations (length of xs along axis 0). */
  length: number;
  /** Number of constant arrays. */
  numConsts: number;
  /** Sizes of each constant buffer in bytes. */
  constSizes: number[];
  /** Sizes of each carry buffer in bytes. */
  carrySizes: number[];
  /** Strides (in bytes) along axis 0 for each xs input. */
  xsStrides: number[];
  /** Strides (in bytes) along axis 0 for each stacked y output. */
  ysStrides: number[];
  /** The body kernel to execute each iteration. */
  bodyKernel: Kernel;
  /** Number of carry arrays. */
  numCarry: number;
  /** Whether to scan in reverse order. */
  reverse?: boolean;
}

/** Describes a single step in a multi-kernel scan body (WebGPU). */
export interface NativeScanMultiStep {
  /** The kernel to execute. */
  kernel: Kernel;
  /**
   * Input mapping: indices into [consts, carry, xs] flattened.
   * For a step, these are the indices of inputs it reads from.
   */
  inputs: number[];
  /**
   * Which carry slot this kernel writes to (0..numCarry-1).
   */
  outputCarryIdx: number;
  /** Size of output in elements (not bytes). */
  outputSize: number;
}

/** Parameters for multi-kernel native scan execution on WebGPU. */
export interface NativeScanMultiParams {
  /** Number of scan iterations (length of xs along axis 0). */
  length: number;
  /** Number of constant arrays (passed to body but unchanged). */
  numConsts: number;
  /** Sizes of each constant buffer in bytes. */
  constSizes: number[];
  /** Number of carry arrays. */
  numCarry: number;
  /** Sizes of each carry buffer in bytes. */
  carrySizes: number[];
  /** Number of x inputs per iteration. */
  numX: number;
  /** Strides (in bytes) along axis 0 for each xs input. */
  xsStrides: number[];
  /** Number of y outputs per iteration. */
  numY: number;
  /** Strides (in bytes) along axis 0 for each stacked y output. */
  ysStrides: number[];
  /** The sequence of kernels to execute each iteration. */
  steps: NativeScanMultiStep[];
  /** Whether to scan in reverse order. */
  reverse?: boolean;
}

/** Parameters for batched scan execution on WebGPU (routine body like matmul). */
export interface BatchedScanParams {
  /** Number of scan iterations (length of xs along axis 0). */
  length: number;
  /** Sizes of each carry buffer in bytes. */
  carrySizes: number[];
  /** Strides (in ELEMENTS) along axis 0 for each xs input. Used for uniform offsets. */
  xsElemStrides: number[];
  /** Strides (in ELEMENTS) along axis 0 for each stacked y output. Used for uniform offsets. */
  ysElemStrides: number[];
  /** The prepared routine executable for the body. */
  bodyRoutine: Executable<ShaderDispatch[]>;
  /** Number of carry arrays. */
  numCarry: number;
  /** Number of xs inputs. */
  numX: number;
  /** Number of ys outputs. */
  numY: number;
  /** Number of const inputs (bound before carry). */
  numConsts: number;
  /** Whether to scan in reverse order. */
  reverse?: boolean;
}

/** Prepared batched scan with wrapped shaders and offset buffer. */
export interface PreparedBatchedScan {
  params: BatchedScanParams;
  /** Shaders with uniform offset support. */
  wrappedShaders: ShaderDispatch[];
  /** GPU buffer containing all iteration offsets. */
  offsetBuffer: GPUBuffer;
  /** Alignment of each iteration's offset data in the buffer. */
  offsetAlignment: number;
}

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
      buffer: GPUBuffer;
    }
  >;
  nextSlot: number;

  #cachedShaderMap = new Map<bigint, ShaderInfo>();
  #reusableZsb: GPUBuffer;

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
    let buffer: GPUBuffer;
    // All GPUBuffer must be a multiple of 4 bytes in length, to support copy
    // operations. Pad it to a multiple of 4.
    const paddedSize = Math.ceil(size / 4) * 4;
    if (size === 0) {
      buffer = this.#reusableZsb;
    } else if (initialData) {
      if (initialData.byteLength !== size) {
        throw new Error("initialData size does not match buffer size");
      }
      if (initialData.byteLength < 4096) {
        buffer = this.#createBuffer(paddedSize, { mapped: true });
        new Uint8Array(buffer.getMappedRange(), 0, size).set(initialData);
        buffer.unmap();
      } else {
        // getMappedRange() seems slower for large buffers, use writeBuffer() instead.
        buffer = this.#createBuffer(paddedSize);
        if (initialData.byteLength % 4 === 0) {
          this.device.queue.writeBuffer(buffer, 0, initialData);
        } else {
          // Copy all but the last few bytes, then copy 4 bytes as remainder.
          const aligned = initialData.byteLength - (initialData.byteLength % 4);
          this.device.queue.writeBuffer(buffer, 0, initialData, 0, aligned);
          const remainder = new Uint8Array(4);
          remainder.set(initialData.subarray(aligned));
          this.device.queue.writeBuffer(buffer, aligned, remainder);
        }
      }
    } else {
      buffer = this.#createBuffer(paddedSize);
    }

    const slot = this.nextSlot++;
    this.buffers.set(slot, { buffer, size, ref: 1 });
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
      if (buffer.buffer !== this.#reusableZsb) {
        buffer.buffer.destroy();
      }
    }
  }

  slotCount(): number {
    return this.buffers.size;
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

  async prepareMultiKernel(
    _multiKernel: MultiKernel,
  ): Promise<Executable<ShaderDispatch[]>> {
    // For now, WebGPU falls back to executing multi-output kernels separately.
    // TODO: Implement native multi-output kernel support for WebGPU.
    throw new Error(
      "MultiKernel not yet implemented for WebGPU - should fall back to single kernels",
    );
  }

  prepareMultiKernelSync(
    _multiKernel: MultiKernel,
  ): Executable<ShaderDispatch[]> {
    throw new Error(
      "MultiKernel not yet implemented for WebGPU - should fall back to single kernels",
    );
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
    pipelineSubmit(this.device, exe.data, inputBuffers, outputBuffers);
  }

  /**
   * Prepare a native scan operation for efficient execution.
   * Returns null if the scan cannot be natively executed.
   */
  prepareNativeScan(
    params: NativeScanParams,
  ): Executable<ShaderDispatch[]> | null {
    const { bodyKernel } = params;
    if (!bodyKernel) return null;

    try {
      const shader = nativeScanShaderSource(this.device, params);
      const pipeline = this.pipelines.prepareSync(shader);
      const syntheticKernel = new Kernel(
        bodyKernel.nargs,
        bodyKernel.size,
        bodyKernel.exp,
        bodyKernel.reduction,
      );
      return new Executable(syntheticKernel, [{ ...shader, pipeline }]);
    } catch (e) {
      if (DEBUG >= 2) {
        console.warn("WebGPU native scan codegen failed:", e);
      }
      return null;
    }
  }

  /**
   * Dispatch a native scan operation.
   * @param exe - The prepared native scan executable
   * @param consts - Constant buffer slots
   * @param initCarry - Initial carry buffer slots
   * @param xs - Input xs buffer slots
   * @param carryOut - Output carry buffer slots
   * @param ysStacked - Output stacked ys buffer slots
   */
  dispatchNativeScan(
    exe: Executable<ShaderDispatch[]>,
    consts: Slot[],
    initCarry: Slot[],
    xs: Slot[],
    carryOut: Slot[],
    ysStacked: Slot[],
  ): void {
    const constsBuffers = consts.map((slot) => this.#getBuffer(slot).buffer);
    const initCarryBuffers = initCarry.map(
      (slot) => this.#getBuffer(slot).buffer,
    );
    const xsBuffers = xs.map((slot) => this.#getBuffer(slot).buffer);
    const carryOutBuffers = carryOut.map(
      (slot) => this.#getBuffer(slot).buffer,
    );
    const ysStackedBuffers = ysStacked.map(
      (slot) => this.#getBuffer(slot).buffer,
    );

    // Dispatch the scan shader
    const commandEncoder = this.device.createCommandEncoder();
    for (const { pipeline, ...shader } of exe.data) {
      // Bind all buffers: consts, initCarry, xs, carryOut, ysStacked
      const allBuffers = [
        ...constsBuffers,
        ...initCarryBuffers,
        ...xsBuffers,
        ...carryOutBuffers,
        ...ysStackedBuffers,
      ];
      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: allBuffers.map((buffer, i) => ({
          binding: i,
          resource: { buffer },
        })),
      });

      for (const { grid } of shader.passes) {
        if (prod(grid) === 0) continue;
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(pipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(grid[0], grid[1]);
        passEncoder.end();
      }
    }
    this.device.queue.submit([commandEncoder.finish()]);
  }

  /**
   * Prepare a multi-kernel native scan operation for efficient execution.
   * Handles scan bodies with multiple independent kernels (e.g., 2 matmuls).
   * Returns null if the scan cannot be natively executed.
   */
  prepareNativeScanMulti(
    params: NativeScanMultiParams,
  ): Executable<ShaderDispatch[]> | null {
    const { steps } = params;
    if (!steps || steps.length === 0) return null;

    try {
      const shader = nativeScanMultiShaderSource(this.device, params);
      const pipeline = this.pipelines.prepareSync(shader);
      // Use the first kernel as the "representative" for the executable
      const firstKernel = steps[0].kernel;
      return new Executable(firstKernel, [{ ...shader, pipeline }]);
    } catch (e) {
      if (DEBUG >= 2) {
        console.warn("WebGPU native scan multi codegen failed:", e);
      }
      return null;
    }
  }

  /**
   * Dispatch a multi-kernel native scan operation.
   * Same buffer layout as dispatchNativeScan.
   */
  dispatchNativeScanMulti(
    exe: Executable<ShaderDispatch[]>,
    consts: Slot[],
    initCarry: Slot[],
    xs: Slot[],
    carryOut: Slot[],
    ysStacked: Slot[],
  ): void {
    // Same dispatch logic as single-kernel scan
    this.dispatchNativeScan(exe, consts, initCarry, xs, carryOut, ysStacked);
  }

  /**
   * Check if batched scan can be used for a routine body.
   * Returns the minimum uniform buffer offset alignment for dynamic offsets.
   */
  getBatchedScanAlignment(): number {
    // Use minUniformBufferOffsetAlignment for dynamic uniform offsets
    // This is typically 256 bytes on most GPUs
    return this.device.limits.minUniformBufferOffsetAlignment ?? 256;
  }

  /**
   * Prepare a batched scan operation for routine bodies (matmul, conv, etc.).
   * Returns the prepared executable if successful, null otherwise.
   *
   * Batched scan encodes all iteration dispatches in a single command buffer,
   * eliminating JS roundtrip overhead per iteration. Uses ping-pong buffers
   * for carry state and uniform-based offset bindings for xs/ys slicing.
   *
   * This approach avoids minStorageBufferOffsetAlignment issues by:
   * 1. Binding full buffers (no offset in GPUBufferBinding)
   * 2. Adding uniform offset variables to the shader
   * 3. Using dynamic uniform buffer offsets for per-iteration offsets
   */
  prepareBatchedScan(params: BatchedScanParams): PreparedBatchedScan | null {
    const {
      xsElemStrides,
      ysElemStrides,
      bodyRoutine,
      numConsts,
      numCarry,
      numX,
      numY,
      length,
      reverse,
    } = params;

    // Verify the routine is valid
    if (!bodyRoutine || bodyRoutine.data.length === 0) {
      if (DEBUG >= 2) console.log("Batched scan: invalid routine");
      return null;
    }

    // Skip if no xs/ys to offset (pure carry operation)
    if (numX === 0 && numY === 0) {
      if (DEBUG >= 2)
        console.log("Batched scan: no xs/ys, using direct dispatch");
      // Could still optimize with batched command buffer, but simpler to fall back
      return null;
    }

    const scanInfo: ScanBindingInfo = {
      numConsts,
      numCarry,
      numX,
      numY,
      numInputs: bodyRoutine.data[0]?.numInputs ?? 0,
      numOutputs: bodyRoutine.data[0]?.numOutputs ?? 0,
    };

    // Wrap each shader in the routine with offset support
    const wrappedShaders: ShaderDispatch[] = [];
    for (const shader of bodyRoutine.data) {
      // Skip routines that already use uniforms (like Sort) - they conflict with our offset uniform
      if (shader.hasUniform) {
        if (DEBUG >= 2)
          console.log("Batched scan: shader already has uniform, skipping");
        return null;
      }

      const wrapped = wrapRoutineForScan(shader, scanInfo);
      if (!wrapped.hasUniform) {
        // No bindings need offsets, fall back
        if (DEBUG >= 2)
          console.log("Batched scan: shader doesn't need offsets");
        return null;
      }

      // Create new pipeline with wrapped shader
      const module = this.device.createShaderModule({ code: wrapped.code });
      const pipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });

      wrappedShaders.push({
        ...shader,
        code: wrapped.code,
        hasUniform: true,
        pipeline,
      });
    }

    // Create the combined uniform buffer with all iteration offsets
    const alignment = this.getBatchedScanAlignment();
    const { buffer: offsetData, alignment: offsetAlignment } =
      createAllIterationsOffsetsBuffer(
        numX,
        numY,
        length,
        xsElemStrides,
        ysElemStrides,
        alignment,
        reverse,
      );

    // Create GPU buffer for offsets
    const offsetBuffer = this.device.createBuffer({
      size: offsetData.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint8Array(offsetBuffer.getMappedRange()).set(offsetData);
    offsetBuffer.unmap();

    if (DEBUG >= 1) {
      console.log(
        `Batched scan: prepared for ${length} iterations with uniform offsets`,
      );
    }

    return {
      params,
      wrappedShaders,
      offsetBuffer,
      offsetAlignment,
    };
  }

  /**
   * Dispatch a batched scan operation with routine body.
   *
   * Uses ping-pong buffers for carry and uniform-based offsets for xs/ys.
   * All iteration dispatches are encoded in a single command buffer.
   * Dynamic uniform buffer offsets are used for per-iteration offset values.
   */
  dispatchBatchedScan(
    prepared: PreparedBatchedScan,
    constSlots: Slot[],
    initCarrySlots: Slot[],
    xsSlots: Slot[],
    carryOutSlots: Slot[],
    ysStackedSlots: Slot[],
  ): void {
    const { params, wrappedShaders, offsetBuffer, offsetAlignment } = prepared;
    const { length, carrySizes, numCarry, numX, numY, numConsts } = params;

    const constBuffers = constSlots.map((slot) => this.#getBuffer(slot).buffer);
    const initCarryBuffers = initCarrySlots.map(
      (slot) => this.#getBuffer(slot).buffer,
    );
    const xsBuffers = xsSlots.map((slot) => this.#getBuffer(slot).buffer);
    const carryOutBuffers = carryOutSlots.map(
      (slot) => this.#getBuffer(slot).buffer,
    );
    const ysStackedBuffers = ysStackedSlots.map(
      (slot) => this.#getBuffer(slot).buffer,
    );

    // Create ping-pong buffers for carry state
    const carryPing = carrySizes.map((size) => this.#createBuffer(size));
    const carryPong = carrySizes.map((size) => this.#createBuffer(size));

    const commandEncoder = this.device.createCommandEncoder();

    // Copy initCarry to carryPing
    for (let i = 0; i < numCarry; i++) {
      commandEncoder.copyBufferToBuffer(
        initCarryBuffers[i],
        0,
        carryPing[i],
        0,
        carrySizes[i],
      );
    }

    // Create bind groups for each shader with full buffer bindings
    // The uniform offset buffer uses dynamic offsets per iteration
    for (const shader of wrappedShaders) {
      const { pipeline, numInputs: _numInputs, passes } = shader;

      // Build storage buffer entries (group 0)
      // Layout: inputs = [consts..., carry_in..., x...], outputs = [carry_out..., y...]

      // Create bind groups for ping and pong configurations
      // Even iterations: read from carryPing, write to carryPong
      // Odd iterations: read from carryPong, write to carryPing

      const createStorageBindGroup = (
        readCarry: GPUBuffer[],
        writeCarry: GPUBuffer[],
      ): GPUBindGroup => {
        const entries: GPUBindGroupEntry[] = [];
        let binding = 0;

        // Inputs: [consts..., carry_in..., x...]
        // Constants (same buffer each iteration - no offset needed)
        for (let c = 0; c < numConsts; c++) {
          entries.push({
            binding: binding++,
            resource: { buffer: constBuffers[c] },
          });
        }

        // Carry inputs (read from ping or pong)
        for (let c = 0; c < numCarry; c++) {
          entries.push({
            binding: binding++,
            resource: { buffer: readCarry[c] },
          });
        }

        // Xs inputs (full buffers - offsets handled by uniform)
        for (let x = 0; x < numX; x++) {
          entries.push({
            binding: binding++,
            resource: { buffer: xsBuffers[x] },
          });
        }

        // Outputs: [carry_out..., y...]
        // Carry outputs (write)
        for (let c = 0; c < numCarry; c++) {
          entries.push({
            binding: binding++,
            resource: { buffer: writeCarry[c] },
          });
        }

        // Ys outputs (full buffers - offsets handled by uniform)
        for (let y = 0; y < numY; y++) {
          entries.push({
            binding: binding++,
            resource: { buffer: ysStackedBuffers[y] },
          });
        }

        return this.device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries,
        });
      };

      const pingBindGroup = createStorageBindGroup(carryPing, carryPong);
      const pongBindGroup = createStorageBindGroup(carryPong, carryPing);

      // Create uniform bind group for offsets (group 1)
      // This uses dynamic offsets - one binding, different offset per iteration
      const uniformBindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [
          {
            binding: 0,
            resource: {
              buffer: offsetBuffer,
              offset: 0,
              size: offsetAlignment, // Size of one iteration's offsets
            },
          },
        ],
      });

      // Dispatch all iterations
      const filteredPasses = passes.filter(({ grid }) => prod(grid) > 0);

      for (let iter = 0; iter < length; iter++) {
        const storageBindGroup = iter % 2 === 0 ? pingBindGroup : pongBindGroup;
        const dynamicOffset = iter * offsetAlignment;

        for (const { grid } of filteredPasses) {
          const passEncoder = commandEncoder.beginComputePass();
          passEncoder.setPipeline(pipeline);
          passEncoder.setBindGroup(0, storageBindGroup);
          passEncoder.setBindGroup(1, uniformBindGroup, [dynamicOffset]);
          passEncoder.dispatchWorkgroups(grid[0], grid[1]);
          passEncoder.end();
        }
      }
    }

    // Copy final carry to carryOut
    const finalCarry = length % 2 === 0 ? carryPing : carryPong;
    for (let i = 0; i < numCarry; i++) {
      commandEncoder.copyBufferToBuffer(
        finalCarry[i],
        0,
        carryOutBuffers[i],
        0,
        carrySizes[i],
      );
    }

    // Submit all commands in one batch
    this.device.queue.submit([commandEncoder.finish()]);

    // Clean up ping-pong buffers (temporary, created per-dispatch)
    for (const buf of [...carryPing, ...carryPong]) {
      buf.destroy();
    }
    // Note: offsetBuffer is NOT destroyed here - it's owned by PreparedBatchedScan
    // and may be reused if the JIT-compiled function is called multiple times.
  }

  #getBuffer(slot: Slot): { buffer: GPUBuffer; size: number } {
    const buffer = this.buffers.get(slot);
    if (!buffer) throw new SlotError(slot);
    return { buffer: buffer.buffer, size: buffer.size };
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

  if (
    tune.exp.some((exp) => exp.dtype === DType.Float16) ||
    tune.epilogue?.some((exp) => exp.dtype === DType.Float16)
  ) {
    if (!device.features.has("shader-f16"))
      throw new Error("WebGPU device does not support shader-f16 feature");
    emit("enable f16;");
  }

  emit(headerWgsl);

  // Global functions at the start of the shader.
  const distinctOps = mapSetUnion(
    tune.exp.distinctOps(),
    tune.epilogue?.distinctOps(),
  );
  if (distinctOps.has(AluOp.Threefry2x32)) {
    emit(threefrySrc);
  }
  if (distinctOps.has(AluOp.Erf) || distinctOps.has(AluOp.Erfc)) {
    emit(erfSrc);
  }

  // End global function definitions.
  emit("");

  const usedArgs: (DType | null)[] = Array.from({ length: nargs }, () => null);
  tune.exp.fold((exp) => {
    if (exp.op === AluOp.GlobalIndex) usedArgs[exp.arg[0]] = exp.dtype;
  });
  tune.epilogue?.fold((exp) => {
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
  const [gridX, gridY] = calculateGrid(gridSize);

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
  const isGensym = (text: string) => text.match(/^alu[0-9]+$/);

  // Insert phony assignments, in case some inputs are not in use.
  // https://github.com/gpuweb/gpuweb/discussions/4582#discussioncomment-9146686
  if (args.length > 0) {
    emit(args.map((arg) => `_ = &${arg};`).join(" "));
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
      else if (op === AluOp.Min) {
        if (dtype === DType.Bool) source = `(${a} && ${b})`;
        else source = `min(${strip1(a)}, ${strip1(b)})`;
      } else if (op === AluOp.Max) {
        if (dtype === DType.Bool) source = `(${a} || ${b})`;
        else source = `max(${strip1(a)}, ${strip1(b)})`;
      } else if (op === AluOp.Cmplt) source = `(${a} < ${b})`;
      else if (op === AluOp.Cmpne) {
        // Edge case: WebGPU doesn't handle NaN correctly, it's unspecified.
        // This is a reliable way I found to detect NaNs, since the spec says
        // for `max()`: if one operand is a NaN, the other is returned.
        if (isFloatDtype(src[0].dtype)) {
          const x = isGensym(a) ? a : gensym();
          if (x !== a) emit(`let ${x} = ${a};`);
          source = `(${x} != ${b} || min(${x}, ${dtypeToWgsl(src[0].dtype)}(inf())) != ${x})`;
        } else {
          source = `(${a} != ${b})`;
        }
      }
    } else if (AluGroup.Unary.has(op)) {
      if (op === AluOp.Reciprocal && src[0].op === AluOp.Sqrt) {
        // Special case: 1/sqrt(x) is optimized as rsqrt(x)
        const a = gen(src[0].src[0]);
        source = `inverseSqrt(${a})`;
      } else {
        const a = gen(src[0]);
        if (op === AluOp.Sin) source = `sin(${strip1(a)})`;
        else if (op === AluOp.Cos) source = `cos(${strip1(a)})`;
        else if (op === AluOp.Asin) source = `asin(${strip1(a)})`;
        else if (op === AluOp.Atan) source = `atan(${strip1(a)})`;
        else if (op === AluOp.Exp) source = `exp(${strip1(a)})`;
        else if (op === AluOp.Log) source = `log(${strip1(a)})`;
        else if (op === AluOp.Erf || op === AluOp.Erfc) {
          const funcName = op === AluOp.Erf ? "erf" : "erfc";
          if (dtype !== DType.Float32) {
            // Always compute special functions in f32 for precision.
            source = `${dtypeToWgsl(dtype)}(${funcName}(f32(${strip1(a)})))`;
          } else {
            source = `${funcName}(${strip1(a)})`;
          }
        } else if (op === AluOp.Sqrt) source = `sqrt(${strip1(a)})`;
        else if (op === AluOp.Reciprocal) source = `(1.0 / ${a})`;
        else if (op === AluOp.Floor) source = `floor(${strip1(a)})`;
        else if (op === AluOp.Ceil) source = `ceil(${strip1(a)})`;
        else if (op === AluOp.Cast)
          source = `${dtypeToWgsl(dtype)}(${strip1(a)})`;
        else if (op === AluOp.Bitcast)
          source = `bitcast<${dtypeToWgsl(dtype)}>(${strip1(a)})`;
      }
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
      if (re.op === AluOp.Add) emit(`${acc[i]} += ${rhs};`);
      else if (re.op === AluOp.Mul) emit(`${acc[i]} *= ${rhs};`);
      else if (re.op === AluOp.Min) {
        // For booleans, min is AND; for numerics, use min()
        if (re.dtype === DType.Bool) emit(`${acc[i]} = ${acc[i]} && ${rhs};`);
        else emit(`${acc[i]} = min(${acc[i]}, ${rhs});`);
      } else if (re.op === AluOp.Max) {
        // For booleans, max is OR; for numerics, use max()
        if (re.dtype === DType.Bool) emit(`${acc[i]} = ${acc[i]} || ${rhs};`);
        else emit(`${acc[i]} = max(${acc[i]}, ${rhs});`);
      } else throw new Error(`Unsupported reduction op: ${re.op}`);
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
        tune
          .epilogue!.substitute({
            acc: AluExp.variable(re.dtype, acc[i]),
            upcast: AluExp.i32(i),
          })
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
    code: shader.join("\n"),
    numInputs: nargs,
    numOutputs: 1,
    hasUniform: false,
    passes: [{ grid: [gridX, gridY] }],
  };
}

/**
 * Generate a WGSL shader for native scan with inlined body kernel.
 *
 * CRITICAL INVARIANT: This shader is only correct for per-element-independent kernels.
 * Each GPU thread i operates exclusively on carry[i] and xs[iter, i] — no cross-thread
 * communication occurs. This invariant is enforced at JIT compile time: only elementwise
 * kernels (no cross-element dependencies) qualify for native scan fusion.
 *
 * Without this invariant, the lack of global barriers between iterations would cause
 * data races. WGSL barriers are workgroup-scoped only, not global across all threads.
 *
 * Buffer layout:
 *   - binding 0..numCarry-1: initCarry buffers (read)
 *   - binding numCarry..numCarry+numX-1: xs buffers (read)
 *   - binding numCarry+numX..numCarry+numX+numCarry-1: carryOut buffers (read_write)
 *   - binding numCarry+numX+numCarry..: ysStacked buffers (write)
 */
function nativeScanShaderSource(
  device: GPUDevice,
  params: NativeScanParams,
): ShaderInfo {
  const {
    length,
    numConsts,
    constSizes: _constSizes,
    carrySizes: _carrySizes,
    xsStrides,
    ysStrides,
    bodyKernel,
    numCarry,
    reverse,
  } = params;

  const re = bodyKernel.reduction;
  const tune = tuneNullopt(bodyKernel);

  const numX = xsStrides.length;
  const numY = ysStrides.length;
  const dtype = bodyKernel.dtype;
  const resultTy = dtypeToWgsl(dtype, true);

  // For MVP, we support single carry/output with matching sizes
  if (numCarry !== 1 || numY !== 1) {
    throw new Error("Native scan: only single carry/output supported for now");
  }

  // kernelSize = number of output elements per iteration
  const kernelSize = bodyKernel.size;

  const shader: string[] = [];
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

  // Check for f16 requirement
  if (dtype === DType.Float16) {
    if (!device.features.has("shader-f16")) {
      throw new Error("WebGPU device does not support shader-f16 feature");
    }
    emit("enable f16;");
  }

  emit(headerWgsl);

  // Global functions needed by body kernel (include epilogue ops)
  const distinctOps = mapSetUnion(
    tune.exp.distinctOps(),
    tune.epilogue?.distinctOps(),
  );
  if (distinctOps.has(AluOp.Threefry2x32)) emit(threefrySrc);
  if (distinctOps.has(AluOp.Erf) || distinctOps.has(AluOp.Erfc)) emit(erfSrc);

  emit("");

  // Buffer declarations
  // Buffer layout: [consts..., initCarry..., xs..., carryOut..., ysStacked...]
  let bindingIdx = 0;

  // const buffers (read only)
  for (let i = 0; i < numConsts; i++) {
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read> const${i}: array<${resultTy}>;`,
    );
  }
  // initCarry buffers (read only)
  for (let i = 0; i < numCarry; i++) {
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read> initCarry${i}: array<${resultTy}>;`,
    );
  }
  // xs buffers (read only)
  for (let i = 0; i < numX; i++) {
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read> xs${i}: array<${resultTy}>;`,
    );
  }
  // carryOut buffers (read_write - used as working buffer)
  for (let i = 0; i < numCarry; i++) {
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read_write> carry${i}: array<${resultTy}>;`,
    );
  }
  // ysStacked buffers (write)
  for (let i = 0; i < numY; i++) {
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read_write> ys${i}: array<${resultTy}>;`,
    );
  }

  // Workgroup size: use kernel size clamped to 256
  const workgroupSize = Math.min(Math.max(kernelSize, 1), 256);
  const [gridX, gridY] = calculateGrid(
    Math.ceil(Math.max(kernelSize, 1) / workgroupSize),
  );

  emit(
    "",
    `@compute @workgroup_size(${workgroupSize})`,
    "fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {",
    pushIndent,
  );

  emit(`let gidx = i32(id.x);`);
  emit(`let inBounds = gidx < ${kernelSize};`);
  emit("");

  // Step 1: Copy initCarry to carryOut (working buffer)
  emit("// Initialize carry from initCarry");
  emit("if (inBounds) {");
  emit(pushIndent);
  for (let i = 0; i < numCarry; i++) {
    emit(`carry${i}[gidx] = initCarry${i}[gidx];`);
  }
  emit(popIndent, "}");
  emit("");

  // Step 2: Main scan loop
  emit(`// Main scan loop over ${length} iterations`);
  emit(`for (var iter: u32 = 0u; iter < ${length}u; iter++) {`, pushIndent);
  // Compute dataIdx = reverse ? (length - 1 - iter) : iter
  if (reverse) {
    emit(`let dataIdx = ${length - 1}u - iter;`);
  } else {
    emit(`let dataIdx = iter;`);
  }
  emit("if (inBounds) {");
  emit(pushIndent);

  const ysElemStride = ysStrides[0] / byteWidth(dtype);

  if (re) {
    // Reduction kernel: inner ridx loop + epilogue
    const accTy = dtypeToWgsl(re.dtype, true);
    emit(`// Reduction: accumulate over ${re.size} elements`);
    emit(`var acc: ${accTy} = ${constToWgsl(re.dtype, re.identity)};`);
    emit(
      `for (var ridx: i32 = 0; ridx < ${tune.size.reduce}; ridx++) {`,
      pushIndent,
    );

    // Generate the expression that produces values to reduce
    const expCode = genScanExpressionWithRidx(
      tune.exp,
      dtype,
      numConsts,
      numCarry,
    );
    emit(`let val = ${expCode};`);

    // Accumulate based on reduction op
    if (re.op === AluOp.Add) {
      emit(`acc = acc + val;`);
    } else if (re.op === AluOp.Mul) {
      emit(`acc = acc * val;`);
    } else if (re.op === AluOp.Min) {
      emit(`acc = min(acc, val);`);
    } else if (re.op === AluOp.Max) {
      emit(`acc = max(acc, val);`);
    } else {
      throw new Error(`Unsupported reduction op: ${re.op}`);
    }

    emit(popIndent, "}");

    // Apply epilogue (transforms acc into result)
    const epilogueCode = genScanExpressionWithRidx(
      tune.epilogue!,
      dtype,
      numConsts,
      numCarry,
    );
    emit(`let result_val: ${resultTy} = ${epilogueCode};`);
  } else {
    // Elementwise kernel: no reduction
    emit("// Compute body expression");
    const expCode = genScanExpressionWithRidx(
      tune.exp,
      dtype,
      numConsts,
      numCarry,
    );
    emit(`let result_val: ${resultTy} = ${expCode};`);
  }

  // Write to ysStacked at dataIdx * stride + gidx
  emit(`ys0[i32(dataIdx) * ${ysElemStride} + gidx] = result_val;`);

  // Update carry for next iteration
  emit(`carry0[gidx] = result_val;`);

  emit(popIndent, "}");
  emit(popIndent, "}");

  emit(popIndent, "}");

  // Buffer layout: [consts..., initCarry..., xs..., carryOut..., ysStacked...]
  // Read-only: consts + initCarry + xs
  // Read-write: carryOut + ysStacked
  const numReadOnlyInputs = numConsts + numCarry + numX;
  const numReadWriteOutputs = numCarry + numY;

  return {
    code: shader.join("\n"),
    numInputs: numReadOnlyInputs,
    numOutputs: numReadWriteOutputs,
    hasUniform: false,
    passes: [{ grid: [gridX, gridY] }],
  };
}

/**
 * Generate WGSL expression code for a scan body.
 * Handles the input layout: [consts..., carry..., xs...]
 * - gid < numConsts: constant buffers (no iteration offset)
 * - gid < numConsts + numCarry: carry buffers
 * - gid >= numConsts + numCarry: xs buffers (with iteration offset via dataIdx)
 */
function genScanExpressionWithRidx(
  exp: AluExp,
  dtype: DType,
  numConsts: number,
  numCarry: number,
): string {
  const gen = (e: AluExp): string => {
    const { op, src, dtype: eDtype, arg } = e;

    if (op === AluOp.GlobalIndex) {
      // arg[0] = buffer index (gid), src[0] = element index expression
      const gid = arg[0] as number;
      const idxCode = gen(src[0]);

      if (gid < numConsts) {
        // Constant input (no iteration offset)
        return `const${gid}[${idxCode}]`;
      } else if (gid < numConsts + numCarry) {
        // Carry input
        const carryIdx = gid - numConsts;
        return `carry${carryIdx}[${idxCode}]`;
      } else {
        // X input with iteration offset (uses dataIdx for reverse support)
        // arg[1] is the stride (elements per iteration)
        const xIdx = gid - numConsts - numCarry;
        const stride = arg[1] as number;
        return `xs${xIdx}[i32(dataIdx) * ${stride} + ${idxCode}]`;
      }
    }

    if (op === AluOp.Const) {
      return constToWgsl(eDtype, arg);
    }

    if (op === AluOp.Special) {
      const name = Array.isArray(arg) ? arg[0] : arg;
      if (name === "gidx") return "gidx";
      if (name === "ridx") return "ridx";
      return name as string;
    }

    if (op === AluOp.Variable) {
      if (arg === "acc") return "acc";
      if (arg === "gidx") return "gidx";
      if (arg === "ridx") return "ridx";
      return arg as string;
    }

    if (op === AluOp.Add) {
      if (eDtype === DType.Bool) return `(${gen(src[0])} || ${gen(src[1])})`;
      return `(${gen(src[0])} + ${gen(src[1])})`;
    }
    if (op === AluOp.Sub) {
      return `(${gen(src[0])} - ${gen(src[1])})`;
    }
    if (op === AluOp.Mul) {
      if (eDtype === DType.Bool) return `(${gen(src[0])} && ${gen(src[1])})`;
      return `(${gen(src[0])} * ${gen(src[1])})`;
    }
    if (op === AluOp.Min) {
      if (eDtype === DType.Bool) return `(${gen(src[0])} && ${gen(src[1])})`;
      return `min(${strip1(gen(src[0]))}, ${strip1(gen(src[1]))})`;
    }
    if (op === AluOp.Max) {
      if (eDtype === DType.Bool) return `(${gen(src[0])} || ${gen(src[1])})`;
      return `max(${strip1(gen(src[0]))}, ${strip1(gen(src[1]))})`;
    }
    if (op === AluOp.Reciprocal) {
      return `(1.0 / ${gen(src[0])})`;
    }
    if (op === AluOp.Sqrt) {
      return `sqrt(${gen(src[0])})`;
    }
    if (op === AluOp.Cast) {
      return `${dtypeToWgsl(eDtype)}(${strip1(gen(src[0]))})`;
    }
    if (op === AluOp.Where) {
      return `select(${strip1(gen(src[2]))}, ${strip1(gen(src[1]))}, ${strip1(gen(src[0]))})`;
    }
    if (op === AluOp.Cmplt) {
      return `(${gen(src[0])} < ${gen(src[1])})`;
    }
    if (op === AluOp.Cmpne) {
      return `(${gen(src[0])} != ${gen(src[1])})`;
    }
    if (op === AluOp.Idiv) {
      return isFloatDtype(eDtype)
        ? `trunc(${gen(src[0])} / ${gen(src[1])})`
        : `(${gen(src[0])} / ${gen(src[1])})`;
    }
    if (op === AluOp.Mod) {
      return `(${gen(src[0])} % ${gen(src[1])})`;
    }
    if (op === AluOp.Sin) return `sin(${strip1(gen(src[0]))})`;
    if (op === AluOp.Cos) return `cos(${strip1(gen(src[0]))})`;
    if (op === AluOp.Asin) return `asin(${strip1(gen(src[0]))})`;
    if (op === AluOp.Atan) return `atan(${strip1(gen(src[0]))})`;
    if (op === AluOp.Exp) return `exp(${strip1(gen(src[0]))})`;
    if (op === AluOp.Log) return `log(${strip1(gen(src[0]))})`;
    if (op === AluOp.Floor) return `floor(${strip1(gen(src[0]))})`;
    if (op === AluOp.Ceil) return `ceil(${strip1(gen(src[0]))})`;
    if (op === AluOp.Bitcast) {
      return `bitcast<${dtypeToWgsl(eDtype)}>(${strip1(gen(src[0]))})`;
    }

    throw new Error(`genScanExpressionWithRidx: unsupported op ${AluOp[op]}`);
  };

  return strip1(gen(exp));
}

/**
 * Generate a WGSL shader for native scan with multiple kernel steps.
 *
 * Each kernel step can have a different size, so we use conditional execution.
 * Kernels are assumed to be independent (each writes to its own carry buffer),
 * so no workgroup barrier is needed between kernels.
 *
 * Buffer layout:
 *   - binding 0..numConsts-1: constant buffers (read)
 *   - binding numConsts..numConsts+numCarry-1: initCarry buffers (read)
 *   - binding numConsts+numCarry..: xs buffers (read)
 *   - binding ...: carryOut buffers (read_write)
 *   - binding ...: ysStacked buffers (write)
 */
function nativeScanMultiShaderSource(
  device: GPUDevice,
  params: NativeScanMultiParams,
): ShaderInfo {
  const {
    length,
    numConsts,
    constSizes: _constSizes,
    carrySizes,
    xsStrides: _xsStrides,
    ysStrides,
    steps,
    numCarry,
    numX,
    numY,
    reverse,
  } = params;

  // Compute element sizes for each carry (assume uniform dtype across all)
  const dtype = steps[0]?.kernel.dtype ?? DType.Float32;
  const resultTy = dtypeToWgsl(dtype, true);
  const elemSize = byteWidth(dtype);

  // Find the maximum kernel size across all steps
  const maxKernelSize = Math.max(...steps.map((s) => s.kernel.size), 1);

  const shader: string[] = [];
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

  // Check for f16 requirement
  if (dtype === DType.Float16) {
    if (!device.features.has("shader-f16")) {
      throw new Error("WebGPU device does not support shader-f16 feature");
    }
    emit("enable f16;");
  }

  emit(headerWgsl);

  // Global functions needed by all kernels
  const allDistinctOps = new Set<AluOp>();
  for (const step of steps) {
    const tune = tuneNullopt(step.kernel);
    for (const [op] of tune.exp.distinctOps()) allDistinctOps.add(op);
    if (tune.epilogue) {
      for (const [op] of tune.epilogue.distinctOps()) allDistinctOps.add(op);
    }
  }
  if (allDistinctOps.has(AluOp.Threefry2x32)) emit(threefrySrc);
  if (allDistinctOps.has(AluOp.Erf) || allDistinctOps.has(AluOp.Erfc))
    emit(erfSrc);

  emit("");

  // Buffer declarations
  let bindingIdx = 0;

  // const buffers (read only)
  for (let i = 0; i < numConsts; i++) {
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read> const${i}: array<${resultTy}>;`,
    );
  }
  // initCarry buffers (read only)
  for (let i = 0; i < numCarry; i++) {
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read> initCarry${i}: array<${resultTy}>;`,
    );
  }
  // xs buffers (read only)
  for (let i = 0; i < numX; i++) {
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read> xs${i}: array<${resultTy}>;`,
    );
  }
  // carryOut buffers (read_write - used as working buffer)
  for (let i = 0; i < numCarry; i++) {
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read_write> carry${i}: array<${resultTy}>;`,
    );
  }
  // ysStacked buffers (write)
  for (let i = 0; i < numY; i++) {
    emit(
      `@group(0) @binding(${bindingIdx++}) var<storage, read_write> ys${i}: array<${resultTy}>;`,
    );
  }

  // Workgroup size: use max kernel size clamped to 256
  const workgroupSize = Math.min(Math.max(maxKernelSize, 1), 256);
  const [gridX, gridY] = calculateGrid(
    Math.ceil(Math.max(maxKernelSize, 1) / workgroupSize),
  );

  emit(
    "",
    `@compute @workgroup_size(${workgroupSize})`,
    "fn main(@builtin(global_invocation_id) id: vec3<u32>) {",
    pushIndent,
  );

  emit(`let gidx = i32(id.x);`);
  emit("");

  // Step 1: Copy initCarry to carryOut (working buffer) for each carry
  // Only copy elements within bounds for each carry
  emit("// Initialize carry from initCarry");
  for (let i = 0; i < numCarry; i++) {
    const carrySize = carrySizes[i] / elemSize;
    emit(`if (gidx < ${carrySize}) {`);
    emit(pushIndent);
    emit(`carry${i}[gidx] = initCarry${i}[gidx];`);
    emit(popIndent, "}");
  }
  emit("");

  // Step 2: Main scan loop
  emit(`// Main scan loop over ${length} iterations`);
  emit(`for (var iter: u32 = 0u; iter < ${length}u; iter++) {`, pushIndent);

  // Compute dataIdx = reverse ? (length - 1 - iter) : iter
  if (reverse) {
    emit(`let dataIdx = ${length - 1}u - iter;`);
  } else {
    emit(`let dataIdx = iter;`);
  }

  // Execute each kernel step
  for (let stepIdx = 0; stepIdx < steps.length; stepIdx++) {
    const step = steps[stepIdx];
    const kernel = step.kernel;
    const tune = tuneNullopt(kernel);
    const carryIdx = step.outputCarryIdx;
    const kernelSize = kernel.size;
    const ysElemStride = ysStrides[carryIdx] / elemSize;

    emit("");
    emit(`// Step ${stepIdx}: kernel writes to carry${carryIdx}`);
    emit(`if (gidx < ${kernelSize}) {`);
    emit(pushIndent);

    const re = kernel.reduction;
    if (re) {
      // Reduction kernel: inner ridx loop + epilogue
      const accTy = dtypeToWgsl(re.dtype, true);
      emit(`var acc: ${accTy} = ${constToWgsl(re.dtype, re.identity)};`);
      emit(
        `for (var ridx: i32 = 0; ridx < ${tune.size.reduce}; ridx++) {`,
        pushIndent,
      );

      const expCode = genScanExpressionWithRidx(
        tune.exp,
        dtype,
        numConsts,
        numCarry,
      );
      emit(`let val = ${expCode};`);

      // Accumulate based on reduction op
      if (re.op === AluOp.Add) {
        emit(`acc = acc + val;`);
      } else if (re.op === AluOp.Mul) {
        emit(`acc = acc * val;`);
      } else if (re.op === AluOp.Min) {
        emit(`acc = min(acc, val);`);
      } else if (re.op === AluOp.Max) {
        emit(`acc = max(acc, val);`);
      } else {
        throw new Error(`Unsupported reduction op: ${re.op}`);
      }

      emit(popIndent, "}");

      // Apply epilogue
      const epilogueCode = genScanExpressionWithRidx(
        tune.epilogue!,
        dtype,
        numConsts,
        numCarry,
      );
      emit(`let result_val_${stepIdx}: ${resultTy} = ${epilogueCode};`);
    } else {
      // Elementwise kernel: no reduction
      const expCode = genScanExpressionWithRidx(
        tune.exp,
        dtype,
        numConsts,
        numCarry,
      );
      emit(`let result_val_${stepIdx}: ${resultTy} = ${expCode};`);
    }

    // Write to ysStacked at dataIdx * stride + gidx
    emit(
      `ys${carryIdx}[i32(dataIdx) * ${ysElemStride} + gidx] = result_val_${stepIdx};`,
    );

    // Update carry for next iteration
    emit(`carry${carryIdx}[gidx] = result_val_${stepIdx};`);

    emit(popIndent, "}");
  }

  emit(popIndent, "}");
  emit(popIndent, "}");

  // Buffer layout: [consts..., initCarry..., xs..., carryOut..., ysStacked...]
  const numReadOnlyInputs = numConsts + numCarry + numX;
  const numReadWriteOutputs = numCarry + numY;

  return {
    code: shader.join("\n"),
    numInputs: numReadOnlyInputs,
    numOutputs: numReadWriteOutputs,
    hasUniform: false,
    passes: [{ grid: [gridX, gridY] }],
  };
}

function pipelineSubmit(
  device: GPUDevice,
  pipelines: ShaderDispatch[],
  inputs: GPUBuffer[],
  outputs: GPUBuffer[],
) {
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
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      if (uniformBindGroup)
        passEncoder.setBindGroup(1, uniformBindGroup, [i * uniformAlignment]);
      passEncoder.dispatchWorkgroups(grid[0], grid[1]);
      passEncoder.end();
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
    // Workaround: Deno's createComputePipelineAsync has a WebIDL binding bug
    // where the 'compute' field is not recognized. Use sync version instead.
    // See: https://github.com/denoland/deno/issues/XXXXX

    if (typeof (globalThis as any).Deno !== "undefined") {
      return this.prepareSync(shader);
    }

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
    // Workaround: Deno's popErrorScope() may return null or non-Promise instead of Promise
    const errorScopePromise = this.device.popErrorScope();
    if (
      errorScopePromise &&
      typeof (errorScopePromise as Promise<unknown>).then === "function"
    ) {
      (errorScopePromise as Promise<GPUError | null>).then(async (scope) => {
        // This happens asynchronously, so we can't throw here. But shader syntax
        // validation errors should never occur in correct code. Any issues here
        // reflect bugs in jax-js.
        if (scope !== null) {
          const emsg = await compileError(shaderModule, scope, shader.code);
          console.error(emsg);
        }
      });
    }
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
