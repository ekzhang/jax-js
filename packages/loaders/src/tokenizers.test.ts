import { expect, suite, test } from "vitest";

import { getBpe } from "./tokenizers";

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
