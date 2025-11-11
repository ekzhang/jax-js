import { cachedFetch } from "./opfs";

/** Supported tokenizer types. */
export type BpeEncodingName = "clip" | (string & {});

/** Get a tokenizer by name. */
export async function getBpe(name: BpeEncodingName): Promise<BpeEncoding> {
  if (name === "clip") {
    const vocab = await loadClipData();
    return new ClipEncoding(vocab);
  }

  throw new Error(`Unsupported tokenizer: ${name}`);
}

function _bytePairMerge(
  ranks: Map<string, number>,
  piece: string,
): [number, number][] {
  // Note: keys of `ranks` are hex-encoded, and piece is also hex-encoded.
  const RANK_MAX = ~(1 << 31); // Large number for no pair

  const n = piece.length / 2; // Should be integer
  const parts: [number, number][] = [];
  let minRank: [number, number] = [RANK_MAX, -1];
  for (let i = 0; i < n - 1; i++) {
    const rank = ranks.get(piece.substring(2 * i, 2 * (i + 2))) ?? RANK_MAX;
    if (rank < minRank[0]) {
      minRank = [rank, i];
    }
    parts.push([i, rank]);
  }
  parts.push([n - 1, RANK_MAX]);
  parts.push([n, RANK_MAX]);

  const getRank = (i: number) => {
    if (i + 3 < parts.length) {
      // Similar to `piece[i..i + 2]` above. The +3 is because we haven't yet deleted
      // parts[i + 1], see comment in the main loop.
      return (
        ranks.get(piece.substring(2 * parts[i][0], 2 * parts[i + 3][0])) ??
        RANK_MAX
      );
    }
    return RANK_MAX;
  };

  while (minRank[0] !== RANK_MAX) {
    const i = minRank[1];
    // Update parts[i] and parts[i - 1] before removing parts[i + 1], since
    // `parts.remove(i + 1)` will thrash the cache.
    if (i > 0) parts[i - 1][1] = getRank(i - 1);
    parts[i][1] = getRank(i);
    parts.splice(i + 1, 1);

    minRank = [RANK_MAX, -1];
    for (let j = 0; j < parts.length - 1; j++) {
      if (parts[j][1] < minRank[0]) {
        minRank = [parts[j][1], j];
      }
    }
  }
  return parts;
}

function bytePairEncode(piece: string, ranks: Map<string, number>): number[] {
  if (piece.length / 2 === 1) return [ranks.get(piece)!];

  const parts = _bytePairMerge(ranks, piece);
  const tokens: number[] = [];
  for (let i = 0; i < parts.length - 1; i++) {
    tokens.push(
      ranks.get(piece.substring(2 * parts[i][0], 2 * parts[i + 1][0]))!,
    );
  }
  return tokens;
}

/**
 * Byte-pair encoding tokenizer, based on the [Tiktoken] library.
 *
 * This handles special tokens and correctly merges adjacent pairs in order of
 * priority (lowest ranks first). This is enough to support LLMs, although some
 * models like CLIP have particular behavior (BOS/EOS, padding,
 * case-insensitivity, whitespace) implemented in subclasses.
 *
 * The internals of this class work in hex strings instead of Uint8Array because
 * strings are more optimized in JavaScript.
 *
 * [Tiktoken]: <https://github.com/openai/tiktoken/blob/0.12.0/src/lib.rs>
 */
export class BpeEncoding {
  encoder: Map<string, number>; // bytes (hex) -> rank
  specialTokensEncoder: Map<string, number>; // special token (string) -> rank
  decoder: Map<number, string>; // rank -> bytes (hex)
  specialTokensDecoder: Map<number, string>; // rank -> special token bytes (hex)
  regex: RegExp; // pattern for pre-tokenization
  specialRegex: RegExp; // pattern for special tokens

  /** Construct a new BPE encoding. */
  constructor(
    encoder: Map<string, number>,
    specialTokensEncoder: Map<string, number>,
    regex: RegExp,
  ) {
    if (!regex.global) {
      throw new Error("Regex for BPE pattern should have global flag set");
    }
    const specialRegex = new RegExp(
      [...specialTokensEncoder.keys()].map(_escapeRegex).join("|"),
      "g",
    );

    const decoder = new Map();
    for (const [bytes, rank] of encoder.entries()) {
      if (decoder.has(rank))
        throw new Error(`Duplicate rank in encoder: ${rank}`);
      decoder.set(rank, bytes);
    }

    const specialTokensDecoder = new Map();
    for (const [token, rank] of specialTokensEncoder.entries()) {
      if (decoder.has(rank) || specialTokensDecoder.has(rank))
        throw new Error(`Duplicate rank in special tokens: ${rank}`);
      specialTokensDecoder.set(
        rank,
        _bytesToHex(new TextEncoder().encode(token)),
      );
    }

    this.encoder = encoder;
    this.specialTokensEncoder = specialTokensEncoder;
    this.decoder = decoder;
    this.specialTokensDecoder = specialTokensDecoder;
    this.regex = regex;
    this.specialRegex = specialRegex;
  }

  /**
   * Decode tokens into a string.
   *
   * May be lossy if the tokens output bytes that don't correspond to a valid
   * UTF-8 string.
   */
  decode(tokens: number[]): string {
    return new TextDecoder().decode(this.decodeBytes(tokens));
  }

  /** Decode tokens into a byte array (may not be UTF-8). */
  decodeBytes(tokens: number[]): Uint8Array {
    tokens = this._beforeDecode(tokens);
    const decodedHex: string[] = [];
    for (const token of tokens) {
      let bytes = this.decoder.get(token);
      if (bytes === undefined) bytes = this.specialTokensDecoder.get(token);
      if (bytes === undefined) {
        throw new Error(`Unknown token during decode: ${token}`);
      }
      decodedHex.push(bytes);
    }
    return _bytesFromHex(decodedHex.join(""));
  }

  /** Encode a text string into tokens, optionally supporting special tokens. */
  encode(text: string, allowedSpecial?: Set<string>): number[] {
    text = this._beforeEncode(text);
    const ret: number[] = [];

    let start = 0;
    while (true) {
      let nextSpecial: RegExpExecArray | null = null;
      this.specialRegex.lastIndex = start;
      while (true) {
        nextSpecial = this.specialRegex.exec(text);
        if (nextSpecial === null) break;
        if (allowedSpecial?.has(nextSpecial[0])) break;
        this.specialRegex.lastIndex = nextSpecial.index + 1; // start+1
      }
      const end = nextSpecial ? nextSpecial.index : text.length;

      // Now encode the next fragment [start, end)
      for (const mat of text.slice(start, end).matchAll(this.regex)) {
        const pieceBytes = new TextEncoder().encode(mat[0]);
        const piece = _bytesToHex(pieceBytes);

        const token = this.encoder.get(piece);
        if (token !== undefined) {
          ret.push(token);
        } else {
          const tokens = bytePairEncode(piece, this.encoder);
          ret.push(...tokens);
        }
      }

      // And handle the special token if any
      if (nextSpecial !== null) {
        const piece = nextSpecial[0];
        const token = this.specialTokensEncoder.get(piece)!;
        ret.push(token);
        start = nextSpecial.index + nextSpecial.length;
      } else {
        break;
      }
    }
    return this._afterEncode(ret);
  }

  /** Retrieve a list of special tokens in this encoding. */
  specialTokens(): Set<string> {
    return new Set(this.specialTokensEncoder.keys());
  }

  /** Encode text with all special tokens allowed. */
  encodeWithSpecialTokens(text: string): number[] {
    return this.encode(text, this.specialTokens());
  }

  /** Can be overridden to change behavior of decode(). */
  _beforeDecode(tokens: number[]): number[] {
    return tokens;
  }

  /** Can be overridden to change behavior of encode(). */
  _beforeEncode(text: string): string {
    return text;
  }

  /** Can be overridden to change behavior of encode(). */
  _afterEncode(tokens: number[]): number[] {
    return tokens;
  }
}

/** BPE encoding with modifications for OpenAI CLIP models. */
class ClipEncoding extends BpeEncoding {
  static readonly pattern =
    /(?:'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^\s\w]+) ?/g;

  constructor(encoder: Map<string, number>) {
    const specialTokensEncoder = new Map<string, number>([
      ["<|startoftext|>", encoder.size], // 49406
      ["<|endoftext|>", encoder.size + 1], // 49407
    ]);
    super(encoder, specialTokensEncoder, ClipEncoding.pattern);
  }

  _beforeEncode(text: string): string {
    // CLIP lowercases and collapses whitespace
    text = text.toLowerCase().replace(/\s+/g, " ").trim();
    // CLIP adds spaces (really "</w>") to each token
    return [...text.matchAll(ClipEncoding.pattern).map((m) => m[0] + " ")].join(
      "",
    );
  }

  _afterEncode(tokens: number[]): number[] {
    // Add bos/eos tokens and pad to length 77
    const bosToken = this.specialTokensEncoder.get("<|startoftext|>")!;
    const eosToken = this.specialTokensEncoder.get("<|endoftext|>")!;
    const padToken = 0;
    const result: number[] = [bosToken, ...tokens, eosToken];
    while (result.length < 77) result.push(padToken);
    return result.slice(0, 77);
  }

  _beforeDecode(tokens: number[]): number[] {
    return tokens.filter((t) => t !== 0); // Remove padding tokens (0)
  }
}

function _escapeRegex(s: string): string {
  // Replace with RegExp.escape() when available (2025 baseline).
  if ("escape" in RegExp) {
    return (RegExp as any).escape(s);
  }
  return s.replace(/[/\-\\^$*+?.()|[\]{}]/g, "\\$&");
}

function _bytesToHex(arr: Uint8Array): string {
  // Use Uint8Array.toHex() if available (2025 baseline).
  if ("toHex" in Uint8Array.prototype) {
    return (arr as any).toHex();
  }
  return Array.from(arr)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function _bytesFromHex(hex: string): Uint8Array {
  // Use Uint8Array.fromHex() if available (2025 baseline).
  if ("fromHex" in Uint8Array) {
    return (Uint8Array as any).fromHex(hex);
  }
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = parseInt(hex.substring(2 * i, 2 * i + 2), 16);
  }
  return bytes;
}

/** Convert a text stream into an async iterator of lines. */
async function* streamLines(
  stream: ReadableStream<string>,
): AsyncIterableIterator<string> {
  const reader = stream.getReader();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += value;
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // Keep the last incomplete line
      for (const line of lines) {
        yield line;
      }
    }
    if (buffer) {
      yield buffer;
    }
  } finally {
    reader.releaseLock();
  }
}

/** Load CLIP tokenizer data from OpenAI CLIP repository. */
async function loadClipData(): Promise<Map<string, number>> {
  const url =
    "https://cdn.jsdelivr.net/gh/mlfoundations/open_clip@v3.2.0/src/open_clip/bpe_simple_vocab_16e6.txt.gz";

  const gzippedData = await cachedFetch(url);

  // Stream decompression and text decoding
  const textStream = new Blob([gzippedData])
    .stream()
    .pipeThrough(new DecompressionStream("gzip"))
    .pipeThrough(new TextDecoderStream());

  const merges: string[] = [];

  // Parse merge rules from file (each line has two tokens separated by space)
  // Skip first line (header) and take merges[1:48895] which is 48894 merges
  // This gives us total vocab size of: 256 + 256 + 48894 = 49406
  const maxMerges = 48894;
  let lineNumber = 0;
  for await (const line of streamLines(textStream)) {
    if (lineNumber > 0 && merges.length < maxMerges) {
      const trimmed = line.trim();
      if (trimmed) {
        const parts = trimmed.split(/\s+/);
        if (parts.length === 2) {
          merges.push(`${parts[0]} ${parts[1]}`);
        }
      }
    }
    lineNumber++;
    if (merges.length >= maxMerges) break;
  }

  // Port of byte-based format in data_gym_to_mergeable_bpe_ranks()
  //
  // Most printable characters are mapped to themselves, but non-printable bytes
  // are mapped to an arbitrary 256+n range of Unicode characters.
  const rankToIntbyte: number[] = [];
  for (let i = 33; i <= 126; i++) rankToIntbyte.push(i);
  for (let i = 161; i <= 172; i++) rankToIntbyte.push(i);
  for (let i = 174; i <= 255; i++) rankToIntbyte.push(i);

  const dataGymByteToByte = new Map<string, number>(
    rankToIntbyte.map((b) => [String.fromCharCode(b), b]),
  );
  let n = 0;
  let escapedSpace = " ";
  for (let b = 0; b < 256; b++) {
    if (!rankToIntbyte.includes(b)) {
      rankToIntbyte.push(b);
      dataGymByteToByte.set(String.fromCharCode(256 + n), b);
      if (b === 0x20) escapedSpace = String.fromCharCode(256 + n);
      n++;
    }
  }

  const decodeDataGym = (value: string): string => {
    value = value.replace(/<\/w>/g, escapedSpace);
    return _bytesToHex(
      new Uint8Array(value.split("").map((c) => dataGymByteToByte.get(c)!)),
    );
  };

  // First replace </w> with the CLIP unicode character for space
  const encoder = new Map<string, number>();
  for (const [i, b] of rankToIntbyte.entries()) {
    encoder.set(_bytesToHex(new Uint8Array([b])), i);
  }
  for (const [i, b] of rankToIntbyte.entries()) {
    encoder.set(_bytesToHex(new Uint8Array([b, 0x20])), i + 256);
  }
  for (const line of merges) {
    const [first, second] = line.split(" ");
    encoder.set(decodeDataGym(first) + decodeDataGym(second), encoder.size);
  }
  return encoder;
}
