/**
 * Check if SharedArrayBuffer is supported, and if so, use it for parallel
 * async execution in worker threads when available.
 *
 * This requires `window.crossOriginIsolated` to be true, which is only the case
 * if the page is served with headers:
 *
 * ```text
 * Cross-Origin-Opener-Policy: same-origin
 * Cross-Origin-Embedder-Policy: require-corp
 * ```
 */
export function hasSharedArrayBuffer(): boolean {
  return typeof SharedArrayBuffer !== "undefined";
}
