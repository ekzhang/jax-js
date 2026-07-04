import { tokenizers } from "@jax-js/loaders";

// Load a browser-friendly BPE tokenizer and encode text.
const enc = await tokenizers.getBpe("clip");
const ids = enc.encode("jax-js runs NumPy-style code in the browser");

console.log("token ids =", ids.slice(0, 16));
console.log("decoded =", enc.decode(ids));
