import { expect, suite, test } from "vitest";

import { getBpe, loadSentencePiece } from "./tokenizers";

/** Helper function to check CLIP tokenizer results */
function expectClipTokens(tokens: number[], expectedTokens: number[]) {
  expect(tokens.length).toBe(77);

  // Create expected array with padding
  const expected = new Uint32Array(77);
  expected.fill(0);
  for (let i = 0; i < expectedTokens.length; i++) {
    expected[i] = expectedTokens[i];
  }

  expect(new Uint32Array(tokens)).toEqual(expected);
}

suite("CLIP tokenizer", async () => {
  const tokenizer = await getBpe("clip");

  test('should encode "a photo of a cat"', () => {
    const tokens = tokenizer.encode("a photo of a cat");
    expectClipTokens(tokens, [49406, 320, 1125, 539, 320, 2368, 49407]);
  });

  test('should encode "Hello, world!"', () => {
    const tokens = tokenizer.encode("Hello, world!");
    expectClipTokens(tokens, [49406, 3306, 267, 1002, 256, 49407]);
  });

  test("should encode empty string or whitespace", () => {
    expectClipTokens(tokenizer.encode(""), [49406, 49407]);
    expectClipTokens(tokenizer.encode("    \t\n"), [49406, 49407]);
  });

  test("should handle long text with truncation", () => {
    const tokens = tokenizer.encode(
      "A very long sentence that goes on and on and should definitely exceed the normal context length to see how the tokenizer handles truncation when we have way too many tokens",
    );
    expectClipTokens(
      tokens,
      [
        49406, 320, 1070, 1538, 12737, 682, 2635, 525, 537, 525, 537, 1535,
        4824, 32521, 518, 5967, 13089, 10130, 531, 862, 829, 518, 32634, 23895,
        22124, 16163, 21367, 827, 649, 720, 923, 1256, 1346, 23562, 49407,
      ],
    );
  });

  test("should encode special characters", () => {
    const tokens = tokenizer.encode("Special characters: !@#$%^&*()");
    expectClipTokens(
      tokens,
      [49406, 1689, 6564, 281, 0, 31, 2, 3, 4, 61, 5, 9, 8475, 49407],
    );
  });

  test("should encode emoji", () => {
    const tokens = tokenizer.encode("😀 😃 😄");
    expectClipTokens(tokens, [49406, 7334, 8520, 7624, 49407]);
  });

  test("should encode Chinese characters", () => {
    const tokens = tokenizer.encode("你好世界");
    expectClipTokens(
      tokens,
      [49406, 47466, 254, 29290, 121, 19759, 244, 163, 243, 490, 49407],
    );
  });

  test("should encode special tokens as literal text", () => {
    const tokens = tokenizer.encode("<|endoftext|>");
    expectClipTokens(tokens, [49406, 27, 347, 40786, 4160, 91, 285, 49407]);
  });

  test("should encode text with special tokens in the middle", () => {
    const tokens = tokenizer.encode("text with <|startoftext|> in the middle");
    expectClipTokens(
      tokens,
      [
        49406, 4160, 593, 27, 347, 993, 6659, 4160, 91, 285, 530, 518, 3694,
        49407,
      ],
    );
  });

  test("should decode tokens back to text", () => {
    const tokens = [49406, 320, 1125, 539, 320, 2368, 49407];
    const decoded = tokenizer.decode(tokens);
    expect(decoded).toBe("<|startoftext|>a photo of a cat <|endoftext|>");
  });
});

suite("tiktoken encodings", () => {
  test("r50k_base", async () => {
    const enc = await getBpe("r50k_base");
    expect(enc.encode("hello world")).toEqual([31373, 995]);
    expect(enc.encode("")).toEqual([]);
  });

  test("p50k_base", async () => {
    const enc = await getBpe("p50k_base");
    expect(enc.encode("hello world")).toEqual([31373, 995]);
  });
});

// Expected outputs generated with Python sentencepiece library:
// uv run --with sentencepiece python3 -c "from sentencepiece import SentencePieceProcessor; ..."
suite("SentencePiece Unigram tokenizer", async () => {
  const tokenizer = await loadSentencePiece(
    "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/678309e/tokenizer.model",
  );

  test("should have correct special token IDs", () => {
    expect(tokenizer.bosToken).toBe(1);
    expect(tokenizer.eosToken).toBe(2);
    expect(tokenizer.unkToken).toBe(0);
    expect(tokenizer.vocabSize).toBe(4000);
  });

  test('should encode "Hello, world!"', () => {
    const tokens = tokenizer.encode("Hello, world!");
    expect(tokens).toEqual([2994, 262, 578, 682]);
  });

  test("should encode longer sentence", () => {
    const tokens = tokenizer.encode(
      "The quick brown fox jumps over the lazy dog.",
    );
    expect(tokens).toEqual([
      364, 976, 3683, 521, 1923, 1609, 261, 408, 265, 697, 690, 327, 1497, 263,
    ]);
  });

  test("should encode mixed case text", () => {
    const tokens = tokenizer.encode("SentencePiece is a tokenizer.");
    expect(tokens).toEqual([
      446, 530, 599, 759, 720, 576, 277, 267, 266, 536, 306, 1577, 295, 263,
    ]);
  });

  test("should encode Chinese characters using byte fallback", () => {
    const tokens = tokenizer.encode("你好世界");
    expect(tokens).toEqual([
      260, 232, 193, 164, 233, 169, 193, 232, 188, 154, 235, 153, 144,
    ]);
  });

  test("should encode French text", () => {
    const tokens = tokenizer.encode("Bonjour le monde!");
    expect(tokens).toEqual([1132, 306, 1347, 513, 1206, 712, 306, 608, 682]);
  });

  test("should encode single character", () => {
    const tokens = tokenizer.encode("A");
    expect(tokens).toEqual([383]);
  });

  test("should encode empty string", () => {
    const tokens = tokenizer.encode("");
    expect(tokens).toEqual([]);
  });

  test("should decode tokens back to original text", () => {
    const testCases = [
      "Hello, world!",
      "The quick brown fox jumps over the lazy dog.",
      "SentencePiece is a tokenizer.",
      "Bonjour le monde!",
      "A",
    ];

    for (const text of testCases) {
      const tokens = tokenizer.encode(text);
      const decoded = tokenizer.decode(tokens);
      expect(decoded).toBe(text);
    }
  });

  test("should decode Chinese characters from byte tokens", () => {
    const tokens = tokenizer.encode("你好世界");
    const decoded = tokenizer.decode(tokens);
    expect(decoded).toBe("你好世界");
  });
});
