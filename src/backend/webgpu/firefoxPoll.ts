// Firefox drives WebGPU completion from a 100 ms timer (`POLL_TIME_MS` in
// dom/webgpu/ipc/WebGPUParent.cpp, line 38), so awaiting `mapAsync()` takes about ~100  ms
// even when nothing was submitted. Mozilla tracks this as bug 1870699,
// https://bugzilla.mozilla.org/show_bug.cgi?id=1870699, "Don't
// poll WebGPU from a timer".
//
// A comment there says submitting continuously gets you an implicit poll for
// free, and that dropping the timer to 5 ms took one CTS test from 12s to 1.2s:
// https://bugzilla.mozilla.org/show_bug.cgi?id=1870699#c6
// So while waiting on a map, we submit empty command buffers to trigger those
// polls ourselves.
//
// The rates below are tuned on a Macbook M1 Max to nudge every 1 ms while checking on
// every task, which brought the wait down to ~1.4 ms. Being too eager kind of backfires
// though — submitting on every yield floods the IPC channel until the GPU
// process just hangs or has a long wait time.
//
// Checking frequency is a bit separate from nudge frequency, since `setTimeout` can't see a
// completion sooner than its ~5 ms clamp and adds ~10 ms of overhead per wait on its own.
//
// Your mileage may vary on other machines, so these numbers are worth revisiting.
//
// There's already work underway upstream to poll on a background thread only
// while GPU work is in flight, so this should be deletable before long.

const NUDGE_INTERVAL_MS = 1;
// Checking on every task is only cheap while the wait is short.
const BUSY_MS = 20;
const BACKOFF_MS = 5;
// A lost device never resolves the map, so stop submitting into it eventually.
const DEADLINE_MS = 1000;

const isFirefox =
  typeof navigator !== "undefined" &&
  navigator.userAgent.includes("Firefox") &&
  !navigator.userAgent.includes("Seamonkey");

// Resolves on the next task, without `setTimeout`'s clamp.
let channel: MessageChannel | undefined;
const resolvers: (() => void)[] = [];
function yieldTask(): Promise<void> {
  const port = (channel ??= (() => {
    const created = new MessageChannel();
    created.port1.onmessage = () => resolvers.shift()?.();
    created.port1.start();
    return created;
  })()).port2;
  return new Promise((resolve) => {
    resolvers.push(resolve);
    port.postMessage(0);
  });
}

/** Map a staging buffer for reading, without Firefox's ~100 ms poll latency. */
export async function mapAsyncRead(
  device: GPUDevice,
  staging: GPUBuffer,
): Promise<void> {
  const mapped = staging.mapAsync(GPUMapMode.READ);
  if (!isFirefox) return mapped;

  let settled = false;
  // `finally` so a rejected map stops the loop too.
  const tracked = mapped.finally(() => {
    settled = true;
  });
  // An already-complete map shouldn't pay for a submit.
  await yieldTask();

  const start = performance.now();
  let lastNudge = -Infinity;
  while (!settled) {
    const elapsed = performance.now() - start;
    if (elapsed >= DEADLINE_MS) break;
    if (performance.now() - lastNudge >= NUDGE_INTERVAL_MS) {
      lastNudge = performance.now();
      device.queue.submit([device.createCommandEncoder().finish()]);
    }
    if (elapsed < BUSY_MS) await yieldTask();
    else await new Promise((resolve) => setTimeout(resolve, BACKOFF_MS));
  }
  return tracked;
}
