export type WasmFeature = "relaxed-madd" | string;

// Minimal modules that validate individual Wasm CPU features.
const featureProbes: Record<WasmFeature, string> = {
  // (func (param v128 v128 v128) (result v128)
  //   local.get 0
  //   local.get 1
  //   local.get 2
  //   f32x4.relaxed_madd)
  "relaxed-madd":
    "0061736d0100000001080160037b7b7b017b030201000a0d010b00200020012002fd85020b",
};

const featureSupportCache = new Map<WasmFeature, boolean>();

/** Detects whether this environment supports a probed Wasm CPU feature. */
export function hasWasmFeature(feature: WasmFeature): boolean {
  const cached = featureSupportCache.get(feature);
  if (cached !== undefined) return cached;

  const testHex = featureProbes[feature];
  let supported = false;
  try {
    supported =
      typeof WebAssembly !== "undefined" &&
      WebAssembly.validate(decodeHex(testHex));
  } catch {
    supported = false;
  }
  featureSupportCache.set(feature, supported);
  return supported;
}

function decodeHex(hex: string): Uint8Array<ArrayBuffer> {
  const bytes = new Uint8Array(new ArrayBuffer(hex.length / 2));
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return bytes;
}
