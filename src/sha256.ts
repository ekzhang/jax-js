// Original implementation of NIST FIPS 180-4, sections 4–6.
// https://doi.org/10.6028/NIST.FIPS.180-4
const roundConstants = new Uint32Array([
  0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
  0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
  0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
  0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
  0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
  0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
  0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
  0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
  0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
  0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
  0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]);
const blockBytes = 64;
const lengthBytes = 8;

function rotateRight(word: number, bits: number): number {
  return (word >>> bits) | (word << (32 - bits));
}

function compress(
  state: Uint32Array,
  words: Uint32Array,
  block: DataView,
  offset: number,
): void {
  for (let i = 0; i < 16; i++) words[i] = block.getUint32(offset + 4 * i);
  for (let i = 16; i < 64; i++) {
    const x = words[i - 15];
    const y = words[i - 2];
    const sigma0 = rotateRight(x, 7) ^ rotateRight(x, 18) ^ (x >>> 3);
    const sigma1 = rotateRight(y, 17) ^ rotateRight(y, 19) ^ (y >>> 10);
    words[i] = words[i - 16] + sigma0 + words[i - 7] + sigma1;
  }

  let [a, b, c, d, e, f, g, h] = state;
  for (let i = 0; i < 64; i++) {
    const sum1 = rotateRight(e, 6) ^ rotateRight(e, 11) ^ rotateRight(e, 25);
    const choice = (e & f) ^ (~e & g);
    const first = h + sum1 + choice + roundConstants[i] + words[i];
    const sum0 = rotateRight(a, 2) ^ rotateRight(a, 13) ^ rotateRight(a, 22);
    const majority = (a & b) ^ (a & c) ^ (b & c);
    h = g;
    g = f;
    f = e;
    e = (d + first) | 0;
    d = c;
    c = b;
    b = a;
    a = (first + sum0 + majority) | 0;
  }
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

/** Compute a SHA-256 digest synchronously and return its lowercase hex form. */
export function sha256Hex(input: Uint8Array): string {
  const state = new Uint32Array([
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c,
    0x1f83d9ab, 0x5be0cd19,
  ]);
  const words = new Uint32Array(64);
  const remaining = input.length % blockBytes;
  const complete = input.length - remaining;
  const data = new DataView(input.buffer, input.byteOffset, input.byteLength);
  for (let offset = 0; offset < complete; offset += blockBytes)
    compress(state, words, data, offset);

  // Append a one bit and the 64-bit bit length, using a second block if needed.
  const tail = new Uint8Array(
    remaining < blockBytes - lengthBytes ? blockBytes : 2 * blockBytes,
  );
  tail.set(input.subarray(complete));
  tail[remaining] = 0x80;
  const padded = new DataView(tail.buffer);
  const bitLength = input.length * 8;
  padded.setUint32(tail.length - lengthBytes, Math.floor(bitLength / 2 ** 32));
  padded.setUint32(tail.length - 4, bitLength);
  for (let offset = 0; offset < tail.length; offset += blockBytes)
    compress(state, words, padded, offset);

  let hex = "";
  for (const word of state) hex += word.toString(16).padStart(8, "0");
  return hex;
}
