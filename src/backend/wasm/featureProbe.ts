// Minimal module that validates `f32x4.relaxed_madd` support:
// (func (param v128 v128 v128) (result v128)
//   local.get 0
//   local.get 1
//   local.get 2
//   f32x4.relaxed_madd)
const RELAXED_MADD_TEST_BYTES = new Uint8Array([
  0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x01, 0x60, 0x03,
  0x7b, 0x7b, 0x7b, 0x01, 0x7b, 0x03, 0x02, 0x01, 0x00, 0x0a, 0x0d, 0x01, 0x0b,
  0x00, 0x20, 0x00, 0x20, 0x01, 0x20, 0x02, 0xfd, 0x85, 0x02, 0x0b,
]);

let relaxedMaddSupported: boolean | undefined;

/** Detects if this environment supports `f32x4.relaxed_madd` (Relaxed SIMD). */
export function hasRelaxedMadd(): boolean {
  if (relaxedMaddSupported !== undefined) return relaxedMaddSupported;
  try {
    relaxedMaddSupported =
      typeof WebAssembly !== "undefined" &&
      WebAssembly.validate(RELAXED_MADD_TEST_BYTES);
  } catch {
    relaxedMaddSupported = false;
  }
  return relaxedMaddSupported;
}
