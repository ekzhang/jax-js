export class WhisperTokenizer {
  readonly #decoder = new Map<number, string>();
  readonly #byteDecoder: Map<string, number>;

  constructor(vocab: Record<string, number>) {
    for (const [token, id] of Object.entries(vocab)) {
      this.#decoder.set(id, token);
    }
    this.#byteDecoder = bytesToUnicodeDecoder();
  }

  static fromVocabBytes(bytes: Uint8Array): WhisperTokenizer {
    return new WhisperTokenizer(JSON.parse(new TextDecoder().decode(bytes)));
  }

  decode(tokens: number[]): string {
    const bytes: number[] = [];
    for (const token of tokens) {
      if (token >= 50256) continue;
      const piece = this.#decoder.get(token);
      if (piece === undefined) continue;
      for (const char of piece) {
        const byte = this.#byteDecoder.get(char);
        if (byte !== undefined) bytes.push(byte);
      }
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
  }
}

function bytesToUnicodeDecoder(): Map<string, number> {
  const bs: number[] = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);

  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }

  return new Map(cs.map((c, i) => [String.fromCharCode(c), bs[i]]));
}
