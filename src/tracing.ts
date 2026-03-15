// Global tracing state for the profiler API.

import { Kernel } from "./alu";
import { Routine } from "./routine";

let traceEnabled = false;
const flushCallbacks: (() => void)[] = [];

/**
 * Start collecting kernel traces.
 *
 * Traces appear in developer tools under the "Performance" tab, and they are
 * useful for measuring fine-grained kernel execution time.
 */
export function startTrace(): void {
  traceEnabled = true;
}

/**
 * Stop collecting kernel traces.
 *
 * Traces appear in developer tools under the "Performance" tab, and they are
 * useful for measuring fine-grained kernel execution time.
 */
export function stopTrace(): void {
  traceEnabled = false;
  for (const cb of flushCallbacks) cb();
}

/** Check if tracing is currently enabled. */
export function isTracing(): boolean {
  return traceEnabled;
}

/** Register a callback to flush pending trace data when tracing stops. */
export function onFlushTrace(cb: () => void): void {
  flushCallbacks.push(cb);
}

/** Build a trace label and properties from a kernel or routine source. */
export function traceSourceInfo(source: Kernel | Routine): {
  label: string;
  properties: [string, string][];
} {
  const properties: [string, string][] = [];
  let label: string;
  if (source instanceof Kernel) {
    label = `Kernel[${source.size}]`;
    properties.push(["exp", `${source.exp}`]);
    properties.push(["size", `${source.size}`]);
    properties.push(["nargs", `${source.nargs}`]);
    if (source.reduction) {
      properties.push([
        "reduction",
        `${source.reduction.op}:${source.reduction.size}`,
      ]);
    }
  } else {
    label = source.name;
    properties.push([
      "inputShapes",
      source.type.inputShapes.map((s) => `[${s}]`).join(", "),
    ]);
    properties.push([
      "outputShapes",
      source.type.outputShapes.map((s) => `[${s}]`).join(", "),
    ]);
    properties.push(["dtype", source.type.inputDtypes.join(", ")]);
  }
  return { label, properties };
}

/** Emit a trace entry as a `performance.measure` with devtools metadata. */
export function emitTrace(
  track: string,
  label: string,
  properties: [string, string][],
  start: number,
  end: number,
): void {
  performance.measure(label, {
    detail: {
      devtools: {
        trackGroup: "JAX Profiler",
        track,
        properties,
      },
    },
    start,
    end,
  });
}
