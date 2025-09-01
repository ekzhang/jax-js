# @jax-js/opfs

A browser-based cache for large files like model weights downloaded from CDN, using the Origin Private File System (OPFS).

## Why?

Model weights and datasets should be:

- **Stored persistently:** Users don't need to repeatedly download the same files across sessions
- **Cleared when stale:** Only the application can determine when files are outdated and need refreshing

This package provides a simple API for storing large blobs in OPFS, reading them later, and listing files with modification times.

There's also a `fetch()` wrapper for making requests that are cached.

## API

The basic `opfs` object allows you to access the file system and store data. Keys can be any string, not just typical file names.

```ts
import { opfs } from "@jax-js/opfs";

await opfs.write("foo", new Uint8Array([1, 2, 3]));
await opfs.read("foo"); // => Uint8Array
```

These methods return `FileInfo` objects, which have a `name`, `lastModified`, and `size` (in bytes).

```ts
import { opfs } from "@jax-js/opfs";

await opfs.info("foo"); // => FileInfo | null
await opfs.list(); // => FileInfo[]

await opfs.remove("foo"); // => FileInfo | null
```

The library also supports a convenient `fetch()` wrapper that caches the request body directly keyed by URL.

```ts
import { cachedFetch } from "@jax-js/opfs";

const url =
  "https://huggingface.co/ekzhang/jax-js-models/resolve/main/mobileclip_s0.safetensors";

await cachedFetch(url); // Also takes `RequestInit`
```
