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

export async function getBackend(backendName: string): Promise<Backend | null> {
  if (backendName === "cpu") {
    const { CPUBackend } = await import("./backend/cpu");
    return new CPUBackend();
  } else if (backendName === "webgpu") {
    const { WebGPUBackend } = await import("./backend/webgpu");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return null;
    const device = await adapter.requestDevice();
    return new WebGPUBackend(device);
  } else {
    throw new Error(`Backend not found: ${backendName}`);
  }
}

/** Unique identifier for an allocated, on-device buffer. */
export type Slot = number;

/** A device backend. */
export interface Backend {
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
  prepare(nargs: number, exp: AluExp): Promise<Executable<any>>;

  /** Prepare an expression to be executed later, blocking variant. */
  prepareSync(nargs: number, exp: AluExp): Executable<any>;

  /**
   * Run a backend operation that was previously prepared.
   *
   * The operation may not run immedaitely, but operations are guaranteed to run
   * in the dispatch order. Also, `read()` will wait for all pending operations
   * on that slot to finish.
   */
  dispatch(exe: Executable<any>, inputs: Slot[], outputs: Slot[]): void;
}

export class Executable<T> {
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
export function accessorAlu(
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
