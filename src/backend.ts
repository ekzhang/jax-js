/**
 * @file Shared interfaces and code for the low-level backend API.
 *
 * Think of each backend as a _connector_ to a specific hardware or software
 * implementation of the array API.
 *
 * Backends do not share any of the built-in operational semantics of the
 * library. This is a private API. You must allocate and free buffers manually,
 * and dispatch happens on the level of each shader. Buffers are untyped.
 */

import { AluExp, DType } from "./alu";
import { ShapeTracker, unravelAlu } from "./shape";

export type BackendType = "cpu" | "webgpu";
export const backendTypes: BackendType[] = ["cpu", "webgpu"];

let defaultBackend: BackendType = "webgpu";
const initializedBackends = new Map<BackendType, Backend>();

/**
 * Initialize `jax-js` library backends.
 *
 * By default, this will initialize all available backends. If one or more
 * backends is provided, only attempt to initialize those. Returns a list of
 * available backends.
 */
export async function init(...backends: BackendType[]): Promise<BackendType[]> {
  if (backends.length === 0) {
    backends = backendTypes;
  }
  const promises: Promise<void>[] = [];
  for (const backendType of new Set(backends)) {
    if (!initializedBackends.has(backendType)) {
      promises.push(
        (async () => {
          const backend = await createBackend(backendType);
          if (backend) {
            initializedBackends.set(backendType, backend);
          }
        })(),
      );
    }
  }
  await Promise.all(promises);
  return Array.from(initializedBackends.keys());
}

/** Create a backend, if available. Internal function called by `init()`. */
async function createBackend(
  backendType: BackendType,
): Promise<Backend | null> {
  if (backendType === "cpu") {
    const { CPUBackend } = await import("./backend/cpu");
    return new CPUBackend();
  } else if (backendType === "webgpu") {
    const { WebGPUBackend } = await import("./backend/webgpu");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;

    const importantLimits: Exclude<keyof GPUSupportedLimits, "__brand">[] = [
      "maxBufferSize",
      "maxComputeInvocationsPerWorkgroup",
      "maxComputeWorkgroupSizeX", // All of our workgroups use X.
      "maxComputeWorkgroupSizeY",
      "maxComputeWorkgroupSizeZ",
      "maxComputeWorkgroupStorageSize",
      "maxComputeWorkgroupsPerDimension",
      "maxStorageBufferBindingSize",
      "maxStorageBuffersPerShaderStage",
      "maxStorageTexturesPerShaderStage",
    ];

    const device = await adapter.requestDevice({
      requiredFeatures: ["timestamp-query"],
      requiredLimits: Object.fromEntries(
        importantLimits.map((feature) => [feature, adapter.limits[feature]]),
      ),
    });

    return new WebGPUBackend(device);
  } else {
    throw new Error(`Backend not found: ${backendType}`);
  }
}

/** Retrieve a backend that has been initialized. */
export function getBackend(backendType?: BackendType): Backend {
  backendType = backendType ?? defaultBackend;
  const backend = initializedBackends.get(backendType);
  if (!backend) {
    throw new Error(`${backendType} backend not ready, call init() first`);
  }
  return backend;
}

/** Unique identifier for an allocated, on-device buffer. */
export type Slot = number;

/** A device backend. */
export interface Backend {
  /** The name of the backend as a string. */
  type: BackendType;

  /** Allocate a new slot with reference count 1. */
  malloc(size: number, initialData?: ArrayBuffer): Slot;

  /** Increment the reference count of the slot. */
  incRef(slot: Slot): void;

  /**
   * Decrement the reference count of the slot. If the reference count reaches
   * zero, it is freed. This should throw if the slot was already freed.
   */
  decRef(slot: Slot): void;

  /** Read a range of bytes from a buffer. */
  read(slot: Slot, start?: number, count?: number): Promise<ArrayBuffer>;

  /** Read a range of bytes from a buffer, blocking variant. */
  readSync(slot: Slot, start?: number, count?: number): ArrayBuffer;

  /** Prepare an expression to be executed later. */
  prepare(nargs: number, exp: AluExp): Promise<Executable>;

  /** Prepare an expression to be executed later, blocking variant. */
  prepareSync(nargs: number, exp: AluExp): Executable;

  /**
   * Run a backend operation that was previously prepared.
   *
   * The operation may not run immedaitely, but operations are guaranteed to run
   * in the dispatch order. Also, `read()` will wait for all pending operations
   * on that slot to finish.
   */
  dispatch(exe: Executable, inputs: Slot[], outputs: Slot[]): void;
}

export class Executable<T = any> {
  constructor(
    readonly nargs: number,
    readonly exp: AluExp,
    /** Backends store extra data here, and it's only used by that backend. */
    readonly data: T,
  ) {}
}

export class SlotError extends Error {
  constructor(slot: Slot) {
    super(`Used a buffer that is invalid or already freed: ${slot}`);
  }
}

/** Expression for accessing `offset` in input array with the given shape. */
export function accessorGlobal(
  gid: number,
  st: ShapeTracker,
  offset: AluExp,
): AluExp {
  const [index, valid] = st.toAluExp(unravelAlu(st.shape, offset));
  return AluExp.where(
    valid,
    AluExp.globalIndex(DType.Float32, gid, index),
    AluExp.f32(0),
  );
}

/** Expression for accessing `offset` in an array recipe. */
export function accessorAluExp(
  exp: AluExp,
  st: ShapeTracker,
  offset: AluExp,
): AluExp {
  const [_index, valid] = st.toAluExp(unravelAlu(st.shape, offset));
  return AluExp.where(valid, exp, AluExp.f32(0));
}
