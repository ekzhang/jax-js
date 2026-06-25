import {
  blockUntilReady,
  defaultDevice,
  init,
  jit,
  nn,
  numpy as np,
  random,
} from "@jax-js/jax";
import { afterAll, bench, suite } from "vitest";

const devices = await init("webgpu");

type AttentionOpts = { isCausal?: boolean };

function attentionForwardNaive(
  q: np.Array,
  k: np.Array,
  v: np.Array,
  opts: AttentionOpts = {},
): np.Array {
  return nn.dotProductAttention(q, k, v, {
    implementation: "naive",
    isCausal: opts.isCausal,
  });
}

function attentionForwardFlash(
  q: np.Array,
  k: np.Array,
  v: np.Array,
  opts: AttentionOpts = {},
): np.Array {
  return nn.dotProductAttention(q, k, v, {
    implementation: "flash",
    isCausal: opts.isCausal,
  });
}

const attentionForwardNaiveJit = jit(function attentionForwardNaiveJit(
  q: np.Array,
  k: np.Array,
  v: np.Array,
): np.Array {
  return attentionForwardNaive(q, k, v);
});

const attentionForwardNaiveCausalJit = jit(
  function attentionForwardNaiveCausalJit(
    q: np.Array,
    k: np.Array,
    v: np.Array,
  ): np.Array {
    return attentionForwardNaive(q, k, v, { isCausal: true });
  },
);

afterAll(() => {
  attentionForwardNaiveJit.dispose();
  attentionForwardNaiveCausalJit.dispose();
});

const cases = [
  { name: "B1 H16 S512 D64", batch: 1, heads: 16, seq: 512, dim: 64 },
  { name: "B8 H16 S512 D64", batch: 8, heads: 16, seq: 512, dim: 64 },
  { name: "B1 H16 S1024 D64", batch: 1, heads: 16, seq: 1024, dim: 64 },
];
const dtypes = [
  { name: "fp32", dtype: np.float32 },
  { name: "fp16", dtype: np.float16 },
];
const modes = [
  { name: "dense", isCausal: false },
  { name: "causal", isCausal: true },
];

for (const { name, batch, heads, seq, dim } of cases) {
  for (const { name: dtypeName, dtype } of dtypes) {
    for (const { name: modeName, isCausal } of modes) {
      suite.skipIf(!devices.includes("webgpu"))(
        `gpu attention forward ${name} ${dtypeName} ${modeName}`,
        async () => {
          defaultDevice("webgpu");

          // jax.nn.dotProductAttention uses [B, sequence, heads, head_dim].
          const shape = [batch, seq, heads, dim];
          const q = random.uniform(random.key(3), shape).astype(dtype);
          const k = random.uniform(random.key(4), shape).astype(dtype);
          const v = random.uniform(random.key(5), shape).astype(dtype);
          await blockUntilReady([q, k, v]);

          const opts = { isCausal };
          const jitFn = isCausal
            ? attentionForwardNaiveCausalJit
            : attentionForwardNaiveJit;

          // Warm up kernels / JIT cache outside measured iterations.
          let out = attentionForwardNaive(q.ref, k.ref, v.ref, opts);
          await out.blockUntilReady();
          out.dispose();

          out = jitFn(q.ref, k.ref, v.ref);
          await out.blockUntilReady();
          out.dispose();

          out = attentionForwardFlash(q.ref, k.ref, v.ref, opts);
          await out.blockUntilReady();
          out.dispose();

          afterAll(() => {
            q.dispose();
            k.dispose();
            v.dispose();
          });

          bench("naive eager", async () => {
            const out = attentionForwardNaive(q.ref, k.ref, v.ref, opts);
            await out.blockUntilReady();
            out.dispose();
          });

          bench("naive jit", async () => {
            const out = jitFn(q.ref, k.ref, v.ref);
            await out.blockUntilReady();
            out.dispose();
          });

          bench("flash", async () => {
            const out = attentionForwardFlash(q.ref, k.ref, v.ref, opts);
            await out.blockUntilReady();
            out.dispose();
          });
        },
      );
    }
  }
}
