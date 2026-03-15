import { Kernel } from "../../alu";
import { Routine } from "../../routine";
import {
  emitTrace,
  isTracing,
  onFlushTrace,
  traceSourceInfo,
} from "../../tracing";

const MAX_TIMESTAMP_QUERIES = 4096;

interface TracingEntry {
  label: string;
  properties: [string, string][];
  beginIndex: number;
  endIndex: number;
}

interface TracingBatch {
  querySet: GPUQuerySet;
  resolve: GPUBuffer;
  dst: GPUBuffer;
  nextIndex: number;
  entries: TracingEntry[];
}

const activeBatch = new WeakMap<GPUDevice, TracingBatch>();

function createTracingBatch(device: GPUDevice): TracingBatch {
  return {
    querySet: device.createQuerySet({
      type: "timestamp",
      count: MAX_TIMESTAMP_QUERIES,
    }),
    resolve: device.createBuffer({
      size: MAX_TIMESTAMP_QUERIES * 8,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    }),
    dst: device.createBuffer({
      size: MAX_TIMESTAMP_QUERIES * 8,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    }),
    nextIndex: 0,
    entries: [],
  };
}

export interface TracingSlot {
  batch: TracingBatch;
  beginIndex: number;
  endIndex: number;
}

function acquireTracingSlot(device: GPUDevice): TracingSlot | undefined {
  if (!device.features.has("timestamp-query")) return undefined;

  let batch = activeBatch.get(device);
  if (batch && batch.nextIndex >= MAX_TIMESTAMP_QUERIES) {
    flushTracingBatch(device, batch);
    batch = undefined;
  }
  if (!batch) {
    batch = createTracingBatch(device);
    activeBatch.set(device, batch);
    onFlushTrace(() => {
      const b = activeBatch.get(device);
      if (b && b.entries.length > 0) {
        flushTracingBatch(device, b);
      }
      activeBatch.delete(device);
    });
  }

  const beginIndex = batch.nextIndex;
  const endIndex = beginIndex + 1;
  batch.nextIndex += 2;
  return { batch, beginIndex, endIndex };
}

/**
 * If tracing is active, acquire a slot for timestamp queries.
 *
 * Returns undefined if tracing is not active or the device doesn't support
 * timestamp queries.
 */
export function maybeAcquireTracingSlot(
  device: GPUDevice,
): TracingSlot | undefined {
  if (!isTracing()) return undefined;
  return acquireTracingSlot(device);
}

/**
 * Record a tracing entry for a pipeline dispatch and schedule an auto-flush.
 */
export function recordTrace(
  device: GPUDevice,
  slot: TracingSlot,
  source: Kernel | Routine,
  numPasses: number,
  wgslSource: string,
): void {
  const { label, properties } = traceSourceInfo(source);
  properties.push(["passes", `${numPasses}`]);
  properties.push(["source", wgslSource]);
  slot.batch.entries.push({
    label,
    properties,
    beginIndex: slot.beginIndex,
    endIndex: slot.endIndex,
  });
  scheduleAutoFlush(device);
}

/**
 * If the active batch has pending entries, flush and replace it so traces
 * are emitted without waiting for the batch to fill or stopTrace().
 *
 * Called after each dispatch records its entry via a microtask so that
 * synchronous back-to-back dispatches are still batched together.
 */
function scheduleAutoFlush(device: GPUDevice): void {
  queueMicrotask(() => {
    const batch = activeBatch.get(device);
    if (batch && batch.entries.length > 0) {
      flushTracingBatch(device, batch);
      activeBatch.set(device, createTracingBatch(device));
    }
  });
}

function flushTracingBatch(device: GPUDevice, batch: TracingBatch): void {
  if (batch.entries.length === 0) return;

  const usedQueries = batch.nextIndex;
  const encoder = device.createCommandEncoder();
  encoder.resolveQuerySet(batch.querySet, 0, usedQueries, batch.resolve, 0);
  encoder.copyBufferToBuffer(batch.resolve, 0, batch.dst, 0, usedQueries * 8);
  device.queue.submit([encoder.finish()]);

  const { entries } = batch;
  batch.dst.mapAsync(GPUMapMode.READ).then(() => {
    try {
      const times = new BigInt64Array(batch.dst.getMappedRange());

      // Establish baseline on the flush: pair the last GPU timestamp with
      // flushCpuMs. We need to do this QuerySet because browsers insert jitter
      // to prevent them being used as absolute timestamps.
      const anchorGpuNs = times[entries[entries.length - 1].endIndex];
      const anchorCpuMs = performance.now();

      for (const entry of entries) {
        const startMs =
          anchorCpuMs + Number(times[entry.beginIndex] - anchorGpuNs) / 1e6;
        const endMs =
          anchorCpuMs + Number(times[entry.endIndex] - anchorGpuNs) / 1e6;
        emitTrace("webgpu", entry.label, entry.properties, startMs, endMs);
      }
    } finally {
      batch.dst.unmap();
      batch.querySet.destroy();
      batch.resolve.destroy();
      batch.dst.destroy();
    }
  });
}
