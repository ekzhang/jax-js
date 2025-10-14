import { cachedFetch } from "./opfs";

/** Supported tokenizer types. */
export type TokenizerType = "clip" | (string & {});

/** Tokenizer interface for encoding/decoding text. */
export interface Tokenizer {
  /** Encode text to token IDs. */
  encode(text: string, options?: { addSpecialTokens?: boolean }): Uint32Array;

  /** Decode token IDs back to text. */
  decode(tokens: Uint32Array | number[]): string;
}

/** Get a tokenizer by name. */
export async function get(name: TokenizerType): Promise<Tokenizer> {
  if (name === "clip") {
    const { vocab, merges } = await loadClipTokenizer();
    return new BpeTokenizer({
      vocab,
      merges,
      bosTokenId: vocab.size - 2, // 49406
      eosTokenId: vocab.size - 1, // 49407
      padTokenId: 0,
      contextLength: 77,
    });
  }

  throw new Error(`Unsupported tokenizer: ${name}`);
}

/** Configuration for BPE tokenizer. */
export type BpeTokenizerConfig = {
  /** BPE vocabulary mappings */
  vocab: Map<string, number>;
  /** BPE merge rules */
  merges: string[];
  /** Token IDs for special tokens */
  bosTokenId: number;
  eosTokenId: number;
  padTokenId: number;
  unkTokenId?: number;
  /** Context length / max sequence length */
  contextLength: number;
};

/** Byte-pair encoding (BPE) tokenizer implementation. */
export class BpeTokenizer implements Tokenizer {
  #vocab: Map<string, number>;
  #invVocab: Map<number, string>;
  #merges: Map<string, number>;
  #bosTokenId: number;
  #eosTokenId: number;
  #padTokenId: number;
  #unkTokenId: number | undefined;
  #contextLength: number;
  #byteEncoder: Map<number, string>;
  #byteDecoder: Map<string, number>;

  constructor(config: BpeTokenizerConfig) {
    this.#vocab = config.vocab;
    this.#invVocab = new Map(
      Array.from(config.vocab.entries()).map(([k, v]) => [v, k]),
    );

    // Create merge priority map
    this.#merges = new Map(config.merges.map((merge, i) => [merge, i]));

    this.#bosTokenId = config.bosTokenId;
    this.#eosTokenId = config.eosTokenId;
    this.#padTokenId = config.padTokenId;
    this.#unkTokenId = config.unkTokenId;
    this.#contextLength = config.contextLength;

    // Initialize byte encoder/decoder for BPE
    this.#byteEncoder = new Map();
    this.#byteDecoder = new Map();
    this.#initByteEncoding();
  }

  /** Initialize byte-level encoding mappings. */
  #initByteEncoding(): void {
    // Create a mapping from bytes to unicode characters
    // This follows the OpenAI GPT-2 byte encoder
    const chars: string[] = [];

    // Add printable ASCII characters (excluding space)
    for (let i = 33; i <= 126; i++) {
      chars.push(String.fromCharCode(i));
    }
    for (let i = 161; i <= 172; i++) {
      chars.push(String.fromCharCode(i));
    }
    for (let i = 174; i <= 255; i++) {
      chars.push(String.fromCharCode(i));
    }

    const bytes: number[] = Array.from(chars, (c) => c.charCodeAt(0));
    let n = 0;

    // Map remaining bytes to unicode characters
    for (let b = 0; b < 256; b++) {
      if (!bytes.includes(b)) {
        bytes.push(b);
        chars.push(String.fromCharCode(256 + n));
        n++;
      }
    }

    for (let i = 0; i < bytes.length; i++) {
      this.#byteEncoder.set(bytes[i], chars[i]);
      this.#byteDecoder.set(chars[i], bytes[i]);
    }
  }

  /** Apply BPE merges to a word. */
  #bpe(token: string): string {
    if (token.length <= 1) return token + "</w>";

    // Add end-of-word marker to last character
    const chars = token.split("");
    chars[chars.length - 1] = chars[chars.length - 1] + "</w>";
    let word = chars;
    let pairs = this.#getPairs(word);

    while (true) {
      if (pairs.length === 0) break;

      // Find the pair with the lowest merge priority
      let minPair: [string, string] | null = null;
      let minRank = Infinity;

      for (const pair of pairs) {
        const pairStr = `${pair[0]} ${pair[1]}`;
        const rank = this.#merges.get(pairStr);
        if (rank !== undefined && rank < minRank) {
          minRank = rank;
          minPair = pair;
        }
      }

      if (!minPair) break;

      const [first, second] = minPair;
      const newWord: string[] = [];
      let i = 0;

      while (i < word.length) {
        const j = word.indexOf(first, i);
        if (j === -1) {
          newWord.push(...word.slice(i));
          break;
        }

        newWord.push(...word.slice(i, j));
        if (
          word[j] === first &&
          j < word.length - 1 &&
          word[j + 1] === second
        ) {
          newWord.push(first + second);
          i = j + 2;
        } else {
          newWord.push(word[j]);
          i = j + 1;
        }
      }

      word = newWord;
      if (word.length === 1) break;
      pairs = this.#getPairs(word);
    }

    return word.join(" ");
  }

  /** Get all adjacent pairs in a word. */
  #getPairs(word: string[]): Array<[string, string]> {
    const pairs: Array<[string, string]> = [];
    for (let i = 0; i < word.length - 1; i++) {
      pairs.push([word[i], word[i + 1]]);
    }
    return pairs;
  }

  /** Encode text to token IDs. */
  encode(text: string, options?: { addSpecialTokens?: boolean }): Uint32Array {
    const addSpecialTokens = options?.addSpecialTokens ?? true;

    // Normalize whitespace and convert to lowercase for CLIP
    text = text.toLowerCase().replace(/\s+/g, " ").trim();

    // Tokenize using CLIP's pattern: handles contractions, letters, numbers, and other chars
    // Pattern: 's|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^\\s\\w]+
    const pattern = /'s|'t|'re|'ve|'m|'ll|'d|[a-z]+|[0-9]|[^\s\w]+/gi;
    const tokens: number[] = [];

    // Add BOS token if needed
    if (addSpecialTokens) {
      tokens.push(this.#bosTokenId);
    }

    // Process each word/token
    const matches = text.match(pattern) || [];
    for (const match of matches) {
      // Convert to bytes then to BPE characters
      const encoder = new TextEncoder();
      const bytes = encoder.encode(match);
      const bpeChars = Array.from(bytes, (b) => this.#byteEncoder.get(b) ?? "");
      const bpeToken = bpeChars.join("");

      // Apply BPE and convert to token IDs
      const bpeResult = this.#bpe(bpeToken);
      const parts = bpeResult.split(" ");

      for (const part of parts) {
        const tokenId = this.#vocab.get(part);
        if (tokenId !== undefined) {
          tokens.push(tokenId);
        } else {
          if (this.#unkTokenId === undefined) {
            throw new Error(`Unknown token encountered: ${part}`);
          }
          tokens.push(this.#unkTokenId); // Handle unknown tokens
        }
      }
    }

    // Add EOS token if needed
    if (addSpecialTokens) {
      tokens.push(this.#eosTokenId);
    }

    // Truncate or pad to context length
    const result = new Uint32Array(this.#contextLength);
    if (this.#padTokenId !== 0) {
      result.fill(this.#padTokenId);
    }

    const length = Math.min(tokens.length, this.#contextLength);
    for (let i = 0; i < length; i++) {
      result[i] = tokens[i];
    }

    return result;
  }

  /** Decode token IDs back to text. */
  decode(tokens: Uint32Array | number[]): string {
    const words: string[] = [];
    let currentWord = "";

    for (const tokenId of tokens) {
      // Skip padding tokens (0) and special tokens
      if (
        tokenId === 0 ||
        tokenId === this.#bosTokenId ||
        tokenId === this.#eosTokenId
      ) {
        continue;
      }

      const token = this.#invVocab.get(tokenId);
      if (token !== undefined) {
        // Check if token ends with </w> (end of word marker)
        if (token.endsWith("</w>")) {
          // Remove </w> marker and add to current word
          currentWord += token.slice(0, -4);
          // Decode current word and add to words
          const bytes: number[] = [];
          for (const char of currentWord) {
            const byte = this.#byteDecoder.get(char);
            if (byte !== undefined) bytes.push(byte);
          }
          const decoder = new TextDecoder();
          words.push(decoder.decode(new Uint8Array(bytes)));
          currentWord = "";
        } else {
          // Continue building current word
          currentWord += token;
        }
      }
    }

    // Handle any remaining characters
    if (currentWord.length > 0) {
      const bytes: number[] = [];
      for (const char of currentWord) {
        const byte = this.#byteDecoder.get(char);
        if (byte !== undefined) bytes.push(byte);
      }
      const decoder = new TextDecoder();
      words.push(decoder.decode(new Uint8Array(bytes)));
    }

    return words.join(" ");
  }
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
async function loadClipTokenizer(): Promise<{
  vocab: Map<string, number>;
  merges: string[];
}> {
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
  // This gives us total vocab size of: 256 + 256 + 48894 + 2 = 49408
  // Special tokens at indices 49406 and 49407
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

    // Exit early if we have all the merges we need
    if (merges.length >= maxMerges) {
      break;
    }
  }

  // Build vocabulary following CLIP's exact approach
  // vocab = 256 bytes + 256 bytes</w> + merges + special tokens
  const vocab = new Map<string, number>();
  let vocabIndex = 0;

  // Build byte encoder mapping (bytes_to_unicode in Python)
  const byteEncoder: string[] = [];
  for (let i = 33; i <= 126; i++) {
    byteEncoder.push(String.fromCharCode(i));
  }
  for (let i = 161; i <= 172; i++) {
    byteEncoder.push(String.fromCharCode(i));
  }
  for (let i = 174; i <= 255; i++) {
    byteEncoder.push(String.fromCharCode(i));
  }

  const bytes = Array.from(byteEncoder, (c) => c.charCodeAt(0));
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bytes.includes(b)) {
      bytes.push(b);
      byteEncoder.push(String.fromCharCode(256 + n));
      n++;
    }
  }

  // Add byte-level tokens to vocab (first 256)
  for (const char of byteEncoder) {
    vocab.set(char, vocabIndex++);
  }

  // Add byte-level tokens with end-of-word marker (next 256)
  for (const char of byteEncoder) {
    vocab.set(char + "</w>", vocabIndex++);
  }

  // Add all merged tokens (don't add if already exists from the base+</w> set)
  for (const merge of merges) {
    const merged = merge.replace(/\s+/g, "");
    if (!vocab.has(merged)) {
      vocab.set(merged, vocabIndex++);
    }
  }

  // At this point, vocabIndex should be exactly 49406
  // (256 bytes + 256 bytes</w> + up to 48894 unique merges = 49406)
  // Add special tokens at the end
  vocab.set("<|startoftext|>", vocabIndex++); // Should be 49406
  vocab.set("<|endoftext|>", vocabIndex++); // Should be 49407

  return { vocab, merges };
}
