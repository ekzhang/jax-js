// Global tracing state for the profiler API.

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
