# jax-js

## Learning workflow

- When explaining internals, show me the actual source code, not just descriptions.
- When touching the tracer/interpreter stack, walk through what happens step by step.

## Context

- My evolving architecture notes are in `docs/notes.md`.
- Arrays use `.ref` for Rust-like move semantics and reference counting instead of GC.
- Author's blog posts: [announcing jax-js](https://substack.com/inbox/post/179060245),
  [how the JIT compiler works](https://substack.com/home/post/p-163548742).
