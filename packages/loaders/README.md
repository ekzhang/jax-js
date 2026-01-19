# @jax-js/loaders

Utility package for `jax-js` that can load tensors from various formats and cache large downloads.

- [OPFS](#opfs)
- [Safetensors](#safetensors)
- [Tokenizers](#tokenizers)
- [WeightMapper](#weightmapper)

It has zero dependencies (except Protobuf) and can be used independently of `@jax-js/jax`.

## OPFS

The `opfs` object provides a browser-based cache for large files like model weights downloaded from
CDN, using the Origin Private File System (OPFS).

This is useful because weights and datasets should be:

- **Stored persistently:** Users don't need to repeatedly download the same files across sessions
- **Cleared when stale:** Only the application can determine when files are outdated and need
  refreshing

The basic `opfs` object allows you to access the file system and store data. Keys can be any string,
not just typical file names.

```ts
import { opfs } from "@jax-js/loaders";

await opfs.write("foo", new Uint8Array([1, 2, 3]));
await opfs.read("foo"); // => Uint8Array
```

These methods return `FileInfo` objects, which have a `name`, `lastModified`, and `size` (in bytes).

```ts
import { opfs } from "@jax-js/loaders";

await opfs.info("foo"); // => FileInfo | null
await opfs.list(); // => FileInfo[]

await opfs.remove("foo"); // => FileInfo | null
```

The library also supports a convenient `fetch()` wrapper that caches the request body directly keyed
by URL.

```ts
import { cachedFetch } from "@jax-js/loaders";

const url = "https://huggingface.co/ekzhang/jax-js-models/resolve/main/mobileclip_s0.safetensors";

await cachedFetch(url); // Also takes `RequestInit` as second parameter
```

## Safetensors

A loader for [Safetensors](https://github.com/huggingface/safetensors) files, which returns the
tensors as native
[typed array views](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Typed_arrays).

```ts
import { safetensors } from "@jax-js/loaders";

const buf = await fetch("model.safetensors").then((resp) => resp.bytes());
safetensors.parse(buf); // => { tensors: { ... } };
```

## Tokenizers

Tokenization for preparing the inputs to a model. It currently supports the
[byte-pair encoding (BPE)](https://github.com/openai/tiktoken) format for various transformers and
CLIP.

Since tokenizer definitions can be nontrivially large (~1 MB), their data is fetched from CDN as
needed.

```ts
import { tokenizers } from "@jax-js/loaders";

const enc = await tokenizers.getBpe("clip");

const tokens = enc.encode("Hello, world!"); // => [ 49406, 3306, 267, 1002, ... ]
enc.decode(tokens); // => "Hello, world!"
```

## WeightMapper

Utility for translating object keys based on matching substrings or prefixes/suffixes, useful for
loading model weights from a different format.

```ts
import { WeightMapper } from "@jax-js/loaders";

const weightMapper = new WeightMapper({
  prefix: {
    "model.transformer.layers": "text_encoder.transformer",
  },
  suffix: {
    ".up_proj": ".up.weight",
    ".up_proj_bias": ".up.bias",
  },
  substring: {
    ".qkv_fused.": ".qkv.",
  },
});
```
