import { tokenizers } from "@jax-js/loaders";

type AddedToken = {
  id: number;
  content: string;
  special: boolean;
};

type HuggingFaceBpeData = {
  added_tokens: AddedToken[];
  pre_tokenizer: {
    type: "Sequence";
    pretokenizers: [
      {
        type: "Split";
        pattern: { Regex: string };
      },
      { type: "ByteLevel" },
    ];
  };
  model: {
    type: "BPE";
    vocab: Record<string, number>;
  };
};

/** Minimal Hugging Face byte-level BPE tokenizer used by LFM2.5. */
export class HuggingFaceBpeTokenizer {
  readonly bosToken = 1;
  readonly eosToken = 7;
  readonly padToken = 0;
  readonly specialTokenIds: Set<number>;
  readonly #encoding: tokenizers.BpeEncoding;

  constructor(data: HuggingFaceBpeData) {
    if (data.model.type !== "BPE") {
      throw new Error(`Expected a BPE tokenizer, found ${data.model.type}`);
    }

    const addedTokens = new Map(
      data.added_tokens.map((token) => [token.id, token]),
    );
    const specialTokens: Record<string, number> = {};
    this.specialTokenIds = new Set<number>();
    for (const token of data.added_tokens) {
      if (token.special || !(token.content in data.model.vocab)) {
        specialTokens[token.content] = token.id;
      }
      if (token.special) this.specialTokenIds.add(token.id);
    }

    const byteDecoder = createByteDecoder();
    const encoder = new Map<string, number>();
    for (const [piece, id] of Object.entries(data.model.vocab)) {
      if (addedTokens.get(id)?.special) continue;
      encoder.set(decodeByteLevelPiece(piece, byteDecoder), id);
    }

    const split = data.pre_tokenizer.pretokenizers[0];
    if (split.type !== "Split" || !("Regex" in split.pattern)) {
      throw new Error("Expected the LFM2.5 byte-level split pre-tokenizer");
    }
    // JavaScript does not support the inline `(?i:...)` modifier used by
    // tokenizers.json. LFM only uses it for contractions, so the global `i`
    // flag is equivalent for this pattern.
    const pattern = split.pattern.Regex.replace(/^\(\?i:([^)]*)\)/, "(?:$1)");
    this.#encoding = new tokenizers.BpeEncoding(
      encoder,
      specialTokens,
      new RegExp(pattern, "giu"),
    );
  }

  static fromBinary(data: Uint8Array): HuggingFaceBpeTokenizer {
    const parsed = JSON.parse(new TextDecoder().decode(data));
    return new HuggingFaceBpeTokenizer(parsed);
  }

  encode(text: string): number[] {
    return this.#encoding.encodeWithSpecialTokens(text);
  }

  decode(tokens: number[]): string {
    return this.#encoding.decode(tokens);
  }
}

function createByteDecoder(): Map<string, number> {
  const bytes: number[] = [];
  for (let i = 33; i <= 126; i++) bytes.push(i);
  for (let i = 161; i <= 172; i++) bytes.push(i);
  for (let i = 174; i <= 255; i++) bytes.push(i);

  const chars = [...bytes];
  let extra = 0;
  for (let byte = 0; byte < 256; byte++) {
    if (bytes.includes(byte)) continue;
    bytes.push(byte);
    chars.push(256 + extra++);
  }
  return new Map(
    chars.map((char, i) => [String.fromCodePoint(char), bytes[i]]),
  );
}

function decodeByteLevelPiece(
  piece: string,
  byteDecoder: Map<string, number>,
): string {
  let hex = "";
  for (const char of piece) {
    const byte = byteDecoder.get(char);
    if (byte === undefined) {
      throw new Error(`Invalid byte-level tokenizer character: ${char}`);
    }
    hex += byte.toString(16).padStart(2, "0");
  }
  return hex;
}
