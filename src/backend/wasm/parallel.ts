// Parallel execution support for the WASM backend using Web Workers.
//
// This requires `crossOriginIsolated` to be true, which is only the case if
// the page is served with headers:
//
// ```text
// Cross-Origin-Opener-Policy: same-origin
// Cross-Origin-Embedder-Policy: require-corp
// ```

/** Check if SharedArrayBuffer is available. */
export function hasSharedArrayBuffer(): boolean {
  // Node.js has SharedArrayBuffer but not Worker, so check both.
  return (
    typeof SharedArrayBuffer !== "undefined" && typeof Worker !== "undefined"
  );
}

const MIN_ELEMS_PER_THREAD = 256;

const WORKER_SOURCE = `
let memory = null;
let cachedModule = null;
let cachedFunc = null;

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "init") {
    memory = msg.memory;
    postMessage({ type: "ready" });
    return;
  }
  try {
    const { module, ptrs, begin, end } = msg;
    if (module !== cachedModule) {
      cachedModule = module;
      const instance = new WebAssembly.Instance(module, { env: { memory } });
      cachedFunc = instance.exports.kernel;
    }
    cachedFunc(...ptrs, begin, end);
    postMessage({ type: "done", ok: true });
  } catch (err) {
    postMessage({ type: "done", ok: false, error: String(err) });
  }
};
`;

/** Pool of Web Workers for parallel WASM kernel dispatch. */
export class WasmWorkerPool {
  #memory: WebAssembly.Memory;
  #numWorkers: number;
  #workers: Worker[] = [];
  #ready: Promise<void> = Promise.resolve();
  /** Serializes dispatches so concurrent read() calls don't clobber onmessage. */
  #queue: Promise<void> = Promise.resolve();
  #epoch: bigint = 0n;
  #epochEnd: bigint = 0n;
  #hooks: Map<bigint, (() => void)[]> = new Map();

  constructor(memory: WebAssembly.Memory, numWorkers: number) {
    if (numWorkers <= 0) {
      throw new Error("numWorkers must be positive");
    }
    this.#memory = memory;
    this.#numWorkers = numWorkers;
  }

  get epoch(): bigint {
    return this.#epoch;
  }

  waitForEpoch(target: bigint): Promise<void> {
    if (target <= this.#epoch) return Promise.resolve();
    return new Promise((resolve) => {
      if (target <= this.#epoch) return resolve();
      const hooks = this.#hooks.get(target);
      if (hooks) hooks.push(resolve);
      else this.#hooks.set(target, [resolve]);
    });
  }

  #ensureInit() {
    if (this.#workers.length > 0) return;
    // Called lazily to avoid creating workers if not used.
    const blob = new Blob([WORKER_SOURCE], { type: "application/javascript" });
    const url = URL.createObjectURL(blob);
    this.#workers = [];
    const readyPromises: Promise<void>[] = [];
    for (let i = 0; i < this.#numWorkers; i++) {
      const worker = new Worker(url, { type: "module" });
      this.#workers.push(worker);
      readyPromises.push(
        new Promise<void>((resolve, reject) => {
          worker.onmessage = () => resolve();
          worker.onerror = (e) =>
            reject(new Error(e.message || "Worker failed to load"));
        }),
      );
      worker.postMessage({ type: "init", memory: this.#memory });
    }
    this.#ready = Promise.all(readyPromises).then(() => {
      URL.revokeObjectURL(url);
    });
    this.#queue = this.#ready;
  }

  /**
   * Dispatch a kernel across multiple workers.
   *
   * Returns an epoch that can be used to wait for the ongoing work to complete,
   * which is guaranteed to be monotonically increasing.
   */
  dispatch(module: WebAssembly.Module, ptrs: number[], size: number): bigint {
    this.#ensureInit();
    this.#epochEnd++;
    const result = this.#queue.then(() =>
      this.#dispatchNow(module, ptrs, size),
    );
    this.#queue = result
      .then(
        () => {},
        () => {}, // Swallow errors to avoid blocking the queue.
      )
      .then(() => {
        this.#epoch++;
        const hooks = this.#hooks.get(this.#epoch);
        if (hooks) {
          for (const hook of hooks) hook();
          this.#hooks.delete(this.#epoch);
        }
      });
    return this.#epochEnd;
  }

  async #dispatchNow(
    module: WebAssembly.Module,
    ptrs: number[],
    size: number,
  ): Promise<void> {
    if (size === 0) return;
    const n = Math.min(
      this.#workers.length,
      Math.ceil(size / MIN_ELEMS_PER_THREAD),
    );
    const chunkSize = Math.ceil(size / n);
    const promises: Promise<void>[] = [];
    for (let i = 0; i < n; i++) {
      const begin = i * chunkSize;
      const end = Math.min(begin + chunkSize, size);
      if (begin >= size) break;
      const worker = this.#workers[i];
      promises.push(
        new Promise<void>((resolve, reject) => {
          worker.onmessage = (e) => {
            if (e.data.ok) resolve();
            else reject(new Error(`Worker error: ${e.data.error}`));
          };
          worker.postMessage({ module, ptrs, begin, end });
        }),
      );
    }
    await Promise.all(promises);
  }
}

/** Try to create a worker pool. Returns null if workers are unavailable. */
export function createWorkerPool(
  memory: WebAssembly.Memory,
): WasmWorkerPool | null {
  if (!hasSharedArrayBuffer()) return null;
  try {
    const numWorkers = Math.max(
      1,
      (typeof navigator !== "undefined" && navigator.hardwareConcurrency) || 4,
    );
    return new WasmWorkerPool(memory, numWorkers);
  } catch {
    return null;
  }
}
