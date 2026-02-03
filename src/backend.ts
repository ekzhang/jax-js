// Shared interfaces and code for the low-level backend API.
//
// Think of each backend as a _connector_ to a specific hardware or software
// implementation of the array API.
//
// Backends do not share any of the built-in operational semantics of the
// library. This is a private API. You must allocate and free buffers manually,
// and dispatch happens on the level of each shader. Buffers are untyped.
//
// The "cpu" backend is very slow and used for debugging. Prefer "wasm".

import { AluOp, DType, Kernel, MultiKernel } from "./alu";
import { CpuBackend } from "./backend/cpu";
import { WasmBackend } from "./backend/wasm";
import { Routine, Routines } from "./routine";

export type Device = "cpu" | "wasm" | "webgpu" | "webgl";
export const devices: Device[] = ["cpu", "wasm", "webgpu", "webgl"];

const initializedBackends = new Map<Device, Backend>();

// Default backends, initialized at startup.
initializedBackends.set("cpu", new CpuBackend());
if (typeof WebAssembly !== "undefined") {
  initializedBackends.set("wasm", new WasmBackend());
}

let defaultBackend: Device = initializedBackends.has("wasm") ? "wasm" : "cpu";

/** Configure the default device for arrays. */
export function defaultDevice(device?: Device): Device {
  if (device !== undefined) {
    if (initializedBackends.has(device)) {
      defaultBackend = device;
    } else {
      throw new Error(`Backend not initialized: ${device}`);
    }
  }
  return defaultBackend;
}

/**
 * Initialize `jax-js` library backends.
 *
 * By default, this will initialize all available backends. If one or more
 * backends is provided, only attempt to initialize those. Returns a list of
 * available backends.
 */
export async function init(...devicesToInit: Device[]): Promise<Device[]> {
  if (devicesToInit.length === 0) {
    devicesToInit = devices;
  }
  const promises: Promise<void>[] = [];
  for (const device of new Set(devicesToInit)) {
    if (!initializedBackends.has(device)) {
      promises.push(
        (async () => {
          const backend = await createBackend(device);
          if (backend) {
            initializedBackends.set(device, backend);
          }
        })(),
      );
    }
  }
  await Promise.all(promises);
  return Array.from(initializedBackends.keys());
}

/** Create a backend, if available. Internal function called by `init()`. */
async function createBackend(device: Device): Promise<Backend | null> {
  if (device === "cpu") {
    return new CpuBackend();
  } else if (device === "wasm") {
    if (typeof WebAssembly === "undefined") return null; // WebAssembly is not available.
    return new WasmBackend();
  } else if (device === "webgpu") {
    if (!navigator.gpu) return null; // WebGPU is not available.
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: "high-performance",
    });
    if (!adapter) return null;

    const { WebGPUBackend } = await import("./backend/webgpu");

    const importantLimits: Exclude<keyof GPUSupportedLimits, "__brand">[] = [
      "maxBufferSize",
      "maxComputeInvocationsPerWorkgroup",
      "maxComputeWorkgroupSizeX", // All of our workgroups use X or Y.
      "maxComputeWorkgroupSizeY",
      "maxComputeWorkgroupSizeZ",
      "maxComputeWorkgroupStorageSize",
      "maxComputeWorkgroupsPerDimension", // Grid size limited to 65535 due to AMD storage in u16.
      "maxStorageBufferBindingSize",
      "maxStorageBuffersPerShaderStage",
      "maxStorageTexturesPerShaderStage",
    ];

    const requestedFeatures: GPUFeatureName[] = [
      "shader-f16", // "enable f16;" feature support for f16 data type
      "timestamp-query", // Performance timing queries.
    ];

    try {
      const device = await adapter.requestDevice({
        requiredLimits: Object.fromEntries(
          importantLimits.map((limit) => [limit, adapter.limits[limit]]),
        ),
        requiredFeatures: requestedFeatures.filter((feature) =>
          adapter.features.has(feature),
        ),
      });
      return new WebGPUBackend(device);
    } catch (error) {
      // Browsers can throw a TypeError if features are not supported by the
      // adapter, or limits have not been set properly.
      console.error("Unexpected error requesting WebGPU device:", error);
      return null;
    }
  } else if (device === "webgl") {
    if (typeof WebGL2RenderingContext === "undefined") return null; // WebGL2 is not available.
    const canvas = new OffscreenCanvas(0, 0);
    const gl = canvas.getContext("webgl2", {
      alpha: false,
      antialias: false,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      depth: false,
      stencil: false,
      failIfMajorPerformanceCaveat: true,
    });
    if (!gl) return null;
    // Required extension for rendering to float textures.
    if (!gl.getExtension("EXT_color_buffer_float")) return null;
    const { WebGLBackend } = await import("./backend/webgl");
    return new WebGLBackend(gl);
  } else {
    device satisfies never;
    throw new Error(`Backend not found: ${device}`);
  }
}

/** Retrieve a backend that has been initialized. */
export function getBackend(device?: Device): Backend {
  device = device ?? defaultBackend;
  const backend = initializedBackends.get(device);
  if (!backend) {
    throw new Error(`${device} backend not ready, call init() first`);
  }
  return backend;
}

/** Unique identifier for an allocated, on-device buffer. */
export type Slot = number;

/** A device backend. */
export interface Backend {
  /** The name of the backend as a string. */
  readonly type: Device;

  /** Maximum number of arguments per dispatched kernel. */
  readonly maxArgs: number;

  /** Allocate a new slot with reference count 1. */
  malloc(size: number, initialData?: Uint8Array): Slot;

  /** Increment the reference count of the slot. */
  incRef(slot: Slot): void;

  /**
   * Decrement the reference count of the slot. If the reference count reaches
   * zero, it is freed. This should throw if the slot was already freed.
   */
  decRef(slot: Slot): void;

  /** Get the number of currently allocated slots (for leak detection). */
  slotCount(): number;

  /** Read a range of bytes from a buffer. */
  read(
    slot: Slot,
    start?: number,
    count?: number,
  ): Promise<Uint8Array<ArrayBuffer>>;

  /** Read a range of bytes from a buffer, blocking variant. */
  readSync(slot: Slot, start?: number, count?: number): Uint8Array<ArrayBuffer>;

  /** Prepare an expression to be executed later. */
  prepareKernel(kernel: Kernel): Promise<Executable>;

  /** Prepare an expression to be executed later, blocking variant. */
  prepareKernelSync(kernel: Kernel): Executable;

  /** Prepare a multi-output kernel to be executed later. */
  prepareMultiKernel(multiKernel: MultiKernel): Promise<Executable>;

  /** Prepare a multi-output kernel to be executed later, blocking variant. */
  prepareMultiKernelSync(multiKernel: MultiKernel): Executable;

  /** Prepare an advanced routine to be executed later. */
  prepareRoutine(routine: Routine): Promise<Executable>;

  /** Prepare an advanced routine to be executed later, blocking variant. */
  prepareRoutineSync(routine: Routine): Executable;

  /**
   * Run a backend operation that was previously prepared.
   *
   * The operation may not run immediately, but operations are guaranteed to run
   * in the dispatch order. Also, `read()` will wait for all pending operations
   * on that slot to finish.
   */
  dispatch(exe: Executable, inputs: Slot[], outputs: Slot[]): void;

  // Optional scan capabilities (implemented by WASM and WebGPU backends)

  /** Prepare a general native scan operation (WASM backend). */
  prepareNativeScanGeneral?(params: any): Executable | null;

  /** Dispatch a general native scan operation (WASM backend). */
  dispatchNativeScanGeneral?(
    exe: Executable,
    params: any,
    consts: Slot[],
    initCarry: Slot[],
    xs: Slot[],
    carryOut: Slot[],
    ysStacked: Slot[],
  ): void;

  /** Prepare a native scan operation (WebGPU backend). */
  prepareNativeScan?(params: any): Executable | null;

  /** Dispatch a native scan operation (WebGPU backend). */
  dispatchNativeScan?(
    exe: Executable,
    consts: Slot[],
    initCarry: Slot[],
    xs: Slot[],
    carryOut: Slot[],
    ysStacked: Slot[],
  ): void;

  /** Prepare a batched scan operation (WebGPU backend). */
  prepareBatchedScan?(params: any): any | null;

  /** Dispatch a batched scan operation (WebGPU backend). */
  dispatchBatchedScan?(
    prepared: any,
    consts: Slot[],
    initCarry: Slot[],
    xs: Slot[],
    carryOut: Slot[],
    ysStacked: Slot[],
  ): void;
}

export class Executable<T = any> {
  constructor(
    /** The `Kernel`, `MultiKernel`, or `Routine` that was prepared. */
    readonly source: Kernel | MultiKernel | Routine,
    /** Extra data specific to the backend running this executable. */
    readonly data: T,
  ) {}
}

export class SlotError extends Error {
  constructor(slot: Slot) {
    super(`Used a buffer that is invalid or already freed: ${slot}`);
  }
}

export class UnsupportedOpError extends Error {
  constructor(op: AluOp | null, dtype: DType, device: Device, arg?: any) {
    let msg = `${op || ""}<${dtype}> not supported in ${device} backend`;
    if (arg !== undefined) msg += ` with arg ${JSON.stringify(arg)}`;
    super(msg);
  }
}

export class UnsupportedRoutineError extends Error {
  constructor(name: Routines, device: Device) {
    super(`routine '${name}' is not supported in ${device} backend`);
  }
}
