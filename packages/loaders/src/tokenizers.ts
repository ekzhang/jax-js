import { fromBinary } from "@bufbuild/protobuf";
import {
  type ModelProto,
  ModelProto_SentencePiece_Type,
  ModelProtoSchema,
} from "sentencepiece-buf/model";

import { cachedFetch } from "./opfs";

/** Supported tokenizer types. */
export type BpeEncodingName =
  | "clip"
  | "r50k_base"
  | "p50k_base"
  | "p50k_edit"
  | "cl100k_base"
  | "o200k_base"
  | "o200k_harmony"
  | (string & {});

// Reference: https://github.com/openai/tiktoken/blob/0.12.0/tiktoken_ext/openai_public.py
const r50kPattern =
  /'(?:[sdmt]|ll|ve|re)| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

/** Get a tokenizer by name. */
export async function getBpe(name: BpeEncodingName): Promise<BpeEncoding> {
  if (name === "clip") {
    const vocab = await loadClipData();
    return new ClipEncoding(vocab);
  } else if (name === "r50k_base") {
    // The "https://openaipublic.blob.core.windows.net" URLs do not have CORS headers, so we use
    // this random NPM library's redistributed CDN instead.
    const encoder = await loadTiktokenBpe(
      "https://cdn.jsdelivr.net/npm/gpt-tokenizer@3.0.1/data/r50k_base.tiktoken",
    );
    return new BpeEncoding(encoder, { "<|endoftext|>": 50256 }, r50kPattern);
  } else if (name === "p50k_base" || name === "p50k_edit") {
    const encoder = await loadTiktokenBpe(
      "https://cdn.jsdelivr.net/npm/gpt-tokenizer@3.0.1/data/p50k_base.tiktoken",
    );
    const specialTokens: Record<string, number> = { "<|endoftext|>": 50256 };
    if (name === "p50k_edit") {
      specialTokens["<|fim_prefix|>"] = 50281;
      specialTokens["<|fim_middle|>"] = 50282;
      specialTokens["<|fim_suffix|>"] = 50283;
    }
    return new BpeEncoding(encoder, specialTokens, r50kPattern);
  } else if (name === "cl100k_base") {
    const encoder = await loadTiktokenBpe(
      "https://cdn.jsdelivr.net/npm/gpt-tokenizer@3.0.1/data/cl100k_base.tiktoken",
    );
    const specialTokens = {
      "<|endoftext|>": 100257,
      "<|fim_prefix|>": 100258,
      "<|fim_middle|>": 100259,
      "<|fim_suffix|>": 100260,
      "<|endofprompt|>": 100276,
    };
    return new BpeEncoding(
      encoder,
      specialTokens,
      /'(?:[sdmtSDMT]|[lL]{2}|[vV][eE]|[rR][eE])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s+$|\s*[\r\n]|\s+(?!\S)|\s/gu,
    );
  } else if (name === "o200k_base" || name === "o200k_harmony") {
    const encoder = await loadTiktokenBpe(
      "https://cdn.jsdelivr.net/npm/gpt-tokenizer@3.0.1/data/o200k_base.tiktoken",
    );
    const specialTokens: Record<string, number> = {
      "<|endoftext|>": 199999,
      "<|endofprompt|>": 200018,
    };
    if (name === "o200k_harmony") {
      delete specialTokens["<|endofprompt|>"];
      specialTokens["<|startoftext|>"] = 199998;
      specialTokens["<|reserved_200000|>"] = 200000;
      specialTokens["<|reserved_200001|>"] = 200001;
      specialTokens["<|return|>"] = 200002;
      specialTokens["<|constrain|>"] = 200003;
      specialTokens["<|reserved_200004|>"] = 200004;
      specialTokens["<|channel|>"] = 200005;
      specialTokens["<|start|>"] = 200006;
      specialTokens["<|end|>"] = 200007;
      specialTokens["<|message|>"] = 200008;
      specialTokens["<|reserved_200009|>"] = 200009;
      specialTokens["<|reserved_200010|>"] = 200010;
      specialTokens["<|reserved_200011|>"] = 200011;
      specialTokens["<|call|>"] = 200012;
      for (let i = 200013; i < 201088; i++) {
        specialTokens[`<|reserved_${i}|>`] = i;
      }
    }
    const pattern = new RegExp(
      [
        /[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?:'[stmdSTMD]|'[rR][eE]|'[vV][eE]|'[lL]{2})?/u
          .source +
          /[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?:'[stmdSTMD]|'[rR][eE]|'[vV][eE]|'[lL]{2})?/u
            .source +
          /\p{N}{1,3}/u.source +
          / ?[^\s\p{L}\p{N}]+[\r\n/]*/u.source +
          /\s*[\r\n]+/.source +
          /\s+(?!\S)/.source +
          /\s+/.source,
      ].join("|"),
      "gu",
    );
    return new BpeEncoding(encoder, specialTokens, pattern);
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
    specialTokens: Record<string, number>,
    regex: RegExp,
  ) {
    if (!regex.global) {
      throw new Error("Regex for BPE pattern should have global flag set");
    }
    const specialTokensEncoder = new Map(Object.entries(specialTokens));
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
    const specialTokens = {
      "<|startoftext|>": encoder.size, // 49406
      "<|endoftext|>": encoder.size + 1, // 49407
    };
    super(encoder, specialTokens, ClipEncoding.pattern);
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

async function loadTiktokenBpe(url: string): Promise<Map<string, number>> {
  const data = await cachedFetch(url);
  const encoder = new Map<string, number>();
  for (const line of new TextDecoder().decode(data).split("\n")) {
    if (!line) continue;
    const [token, rank] = line.split(/\s+/);
    encoder.set(
      _bytesToHex(Uint8Array.from(atob(token), (c) => c.charCodeAt(0))),
      parseInt(rank),
    );
  }
  return encoder;
}

/** The whitespace meta-symbol used by SentencePiece. */
const SPIECE_UNDERLINE = "\u2581"; // ▁

/** Trie node for efficient vocabulary lookup. */
interface TrieNode {
  children: Map<string, TrieNode>;
  /** Token info if this node represents a complete piece. */
  token?: { id: number; score: number };
}

function createTrieNode(): TrieNode {
  return { children: new Map() };
}

/**
 * SentencePiece Unigram tokenizer.
 *
 * This implements the Viterbi-based unigram language model tokenization
 * algorithm used by SentencePiece. It finds the most likely segmentation
 * of input text based on learned piece scores (log probabilities).
 *
 * Uses a trie for efficient O(n * maxPieceLen) vocabulary lookup.
 */
export class Unigram {
  #trie: TrieNode; // trie for piece lookup
  #decoder: Map<number, string>; // id -> piece
  #byteFallback: Map<string, number>; // byte hex -> id (for <0xXX> tokens)
  #unkId: number;
  #bosId: number;
  #eosId: number;

  /** Normalizer settings */
  #addDummyPrefix: boolean;
  #removeExtraWhitespaces: boolean;

  constructor(model: ModelProto) {
    this.#trie = createTrieNode();
    this.#decoder = new Map();
    this.#byteFallback = new Map();

    // Get special token IDs from trainer spec or use defaults
    this.#unkId = model.trainerSpec?.unkId ?? 0;
    this.#bosId = model.trainerSpec?.bosId ?? 1;
    this.#eosId = model.trainerSpec?.eosId ?? 2;

    // Get normalizer settings
    this.#addDummyPrefix = model.normalizerSpec?.addDummyPrefix ?? true;
    this.#removeExtraWhitespaces =
      model.normalizerSpec?.removeExtraWhitespaces ?? true;

    // Build vocabulary maps
    for (let i = 0; i < model.pieces.length; i++) {
      const piece = model.pieces[i];
      const pieceStr = piece.piece;
      const score = piece.score;
      const type = piece.type;

      this.#decoder.set(i, pieceStr);

      if (type === ModelProto_SentencePiece_Type.BYTE) {
        // Byte fallback tokens like <0x00>, <0x01>, etc.
        const match = pieceStr.match(/^<0x([0-9A-Fa-f]{2})>$/);
        if (match) {
          this.#byteFallback.set(match[1].toLowerCase(), i);
        }
      } else if (
        type === ModelProto_SentencePiece_Type.NORMAL ||
        type === ModelProto_SentencePiece_Type.USER_DEFINED
      ) {
        // Insert into trie
        this.#insertIntoTrie(pieceStr, i, score);
      }
      // CONTROL, UNKNOWN, UNUSED types are handled specially
    }
  }

  static fromBinary(data: Uint8Array): Unigram {
    const model = fromBinary(ModelProtoSchema, data);
    return new Unigram(model);
  }

  /** Insert a piece into the trie. */
  #insertIntoTrie(piece: string, id: number, score: number): void {
    let node = this.#trie;
    for (const char of piece) {
      let child = node.children.get(char);
      if (!child) {
        child = createTrieNode();
        node.children.set(char, child);
      }
      node = child;
    }
    node.token = { id, score };
  }

  /**
   * Find all pieces in the vocabulary that start at position `start` in `text`.
   * Returns array of [endPosition, tokenId, score] tuples.
   */
  #findPiecesAt(text: string, start: number): Array<[number, number, number]> {
    const results: Array<[number, number, number]> = [];
    let node = this.#trie;

    for (let i = start; i < text.length; i++) {
      const char = text[i];
      const child = node.children.get(char);
      if (!child) break;
      node = child;
      if (node.token) {
        results.push([i + 1, node.token.id, node.token.score]);
      }
    }

    return results;
  }

  /** Normalize input text according to SentencePiece rules. */
  #normalize(text: string): string {
    if (this.#removeExtraWhitespaces) {
      text = text.replace(/\s+/g, " ").trim();
    }
    if (text.length === 0) return "";
    if (this.#addDummyPrefix) {
      text = " " + text;
    }
    // Replace spaces with the SentencePiece meta-symbol
    text = text.replace(/ /g, SPIECE_UNDERLINE);
    return text;
  }

  /**
   * Encode text into token IDs using Viterbi algorithm.
   *
   * Finds the most likely segmentation by computing the best path through
   * all possible segmentations, where scores are log probabilities.
   */
  encode(text: string): number[] {
    text = this.#normalize(text);
    if (text.length === 0) return [];

    const n = text.length;

    // best[i] = best score to reach position i
    // prev[i] = [start position, token ids] of the best path ending at i
    // We store an array of token IDs to handle multi-byte characters
    const best: number[] = new Array(n + 1).fill(-Infinity);
    const prev: [number, number[]][] = new Array(n + 1).fill(null);
    best[0] = 0;

    for (let i = 0; i < n; i++) {
      if (best[i] === -Infinity) continue;

      // Try all possible pieces starting at position i using trie
      const matches = this.#findPiecesAt(text, i);
      for (const [end, id, score] of matches) {
        const newScore = best[i] + score;
        if (newScore > best[end]) {
          best[end] = newScore;
          prev[end] = [i, [id]];
        }
      }

      // Byte fallback: only use if no vocabulary piece covers position i+1
      // This ensures vocabulary matches are always preferred over byte fallback
      if (prev[i + 1] === null) {
        const char = text[i];
        const bytes = new TextEncoder().encode(char);
        const byteTokens: number[] = [];

        for (const byte of bytes) {
          const hex = byte.toString(16).padStart(2, "0");
          const byteId = this.#byteFallback.get(hex);
          if (byteId !== undefined) {
            byteTokens.push(byteId);
          } else {
            byteTokens.push(this.#unkId);
          }
        }

        if (byteTokens.length > 0) {
          best[i + 1] = best[i]; // Byte fallback has score 0
          prev[i + 1] = [i, byteTokens];
        }
      }
    }

    // Backtrack to get the best segmentation
    const tokens: number[] = [];
    let pos = n;
    while (pos > 0) {
      if (prev[pos] === null) {
        // Should not happen if algorithm is correct, but fallback just in case
        const char = text[pos - 1];
        const bytes = new TextEncoder().encode(char);
        for (let j = bytes.length - 1; j >= 0; j--) {
          const hex = bytes[j].toString(16).padStart(2, "0");
          const byteId = this.#byteFallback.get(hex);
          tokens.push(byteId ?? this.#unkId);
        }
        pos--;
      } else {
        const [start, tokenIds] = prev[pos];
        // Add tokens in reverse order (we'll reverse at the end)
        for (let j = tokenIds.length - 1; j >= 0; j--) {
          tokens.push(tokenIds[j]);
        }
        pos = start;
      }
    }

    tokens.reverse();
    return tokens;
  }

  /** Decode token IDs back to text. */
  decode(tokens: number[]): string {
    const pieces: string[] = [];

    let i = 0;
    while (i < tokens.length) {
      const tokenId = tokens[i];
      const piece = this.#decoder.get(tokenId);

      if (piece === undefined) {
        // Unknown token
        pieces.push("�");
        i++;
        continue;
      }

      // Check if this is a byte token
      const byteMatch = piece.match(/^<0x([0-9A-Fa-f]{2})>$/);
      if (byteMatch) {
        // Collect consecutive byte tokens
        const bytes: number[] = [parseInt(byteMatch[1], 16)];
        i++;

        while (i < tokens.length) {
          const nextPiece = this.#decoder.get(tokens[i]);
          const nextByteMatch = nextPiece?.match(/^<0x([0-9A-Fa-f]{2})>$/);
          if (nextByteMatch) {
            bytes.push(parseInt(nextByteMatch[1], 16));
            i++;
          } else {
            break;
          }
        }
        pieces.push(new TextDecoder().decode(new Uint8Array(bytes)));
      } else {
        pieces.push(piece);
        i++;
      }
    }

    // Join and convert meta-symbol back to space
    let result = pieces
      .join("")
      .replace(new RegExp(SPIECE_UNDERLINE, "g"), " ");

    // Remove leading space if dummy prefix was added
    if (this.#addDummyPrefix && result.startsWith(" ")) {
      result = result.slice(1);
    }

    return result;
  }

  /** Get the beginning-of-sequence token ID. */
  get bosToken(): number {
    return this.#bosId;
  }

  /** Get the end-of-sequence token ID. */
  get eosToken(): number {
    return this.#eosId;
  }

  /** Get the unknown token ID. */
  get unkToken(): number {
    return this.#unkId;
  }

  /** Get vocabulary size. */
  get vocabSize(): number {
    return this.#decoder.size;
  }
}

/** Load a SentencePiece model from a URL. */
export async function loadSentencePiece(url: string): Promise<Unigram> {
  const data = await cachedFetch(url);
  return Unigram.fromBinary(data);
}
