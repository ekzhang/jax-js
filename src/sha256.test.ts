import { expect, test } from "vitest";

import { sha256Hex } from "./sha256";

test.each([
  ["", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"],
  ["abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"],
  [
    "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
    "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
  ],
])("SHA-256 known digest for %j", (message, expected) => {
  expect(sha256Hex(new TextEncoder().encode(message))).toBe(expected);
});

test("SHA-256 agrees with Web Crypto at padding boundaries", async () => {
  const data = Uint8Array.from({ length: 1030 }, (_, i) => i * 17);
  const lengths = [
    0, 1, 55, 56, 57, 63, 64, 65, 119, 120, 121, 127, 128, 129, 1024,
  ];
  for (const length of lengths) {
    const input = data.subarray(3, length + 3);
    const expected = await crypto.subtle.digest("SHA-256", input);
    const hex = Array.from(new Uint8Array(expected), (byte) =>
      byte.toString(16).padStart(2, "0"),
    ).join("");
    expect(sha256Hex(input), `length ${length}`).toBe(hex);
  }
});
