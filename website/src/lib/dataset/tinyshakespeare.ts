import { cachedFetch } from "@jax-js/loaders";

export const tinyShakespeareUrl =
  "https://cdn.jsdelivr.net/gh/karpathy/char-rnn@master/data/tinyshakespeare/input.txt";

export type CharDataset = {
  text: string;
  encoded: Int32Array<ArrayBuffer>;
  charToIdx: Record<string, number>;
  idxToChar: string[];
};

export async function fetchTinyShakespeare(): Promise<string> {
  const bytes = await cachedFetch(tinyShakespeareUrl);
  return new TextDecoder().decode(bytes).replaceAll("\r\n", "\n");
}

export function makeCharDataset(text: string): CharDataset {
  const idxToChar = Array.from(new Set(text)).sort();
  const charToIdx = Object.fromEntries(
    idxToChar.map((char, idx) => [char, idx]),
  );
  const encoded = new Int32Array(text.length);
  for (let i = 0; i < text.length; i++) {
    encoded[i] = charToIdx[text[i]];
  }
  return { text, encoded, charToIdx, idxToChar };
}
