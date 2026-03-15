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

export interface TraceInfo {
  label: string;
  color: string;
  properties: [string, string][];
}

/** Build a trace label, properties, and color from a kernel or routine source. */
export function traceSourceInfo(source: Kernel | Routine): TraceInfo {
  const properties: [string, string][] = [];
  let label: string;
  let color: string;
  if (source instanceof Kernel) {
    label = `Kernel[${source.size}]`;
    properties.push(["exp", `${source.exp}`]);
    properties.push(["size", `${source.size}`]);
    properties.push(["nargs", `${source.nargs}`]);
    if (!source.reduction) {
      color = "primary";
    } else {
      color = "secondary";
      properties.push([
        "reduction",
        `${source.reduction.op}:${source.reduction.size}`,
      ]);
    }
  } else {
    color = "tertiary";
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
  return { label, color, properties };
}

/** Emit a trace entry as a `performance.measure` with devtools metadata. */
export function emitTrace(
  track: string,
  info: TraceInfo,
  start: number,
  end: number,
): void {
  performance.measure(info.label, {
    detail: {
      devtools: {
        trackGroup: "JAX Profiler",
        track,
        color: info.color,
        properties: info.properties,
      },
    },
    start,
    end,
  });
}
