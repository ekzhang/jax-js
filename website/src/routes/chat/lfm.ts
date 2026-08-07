import { blockUntilReady, jit, nn, numpy as np } from "@jax-js/jax";
import { safetensors, WeightMapper } from "@jax-js/loaders";

export const LFM_CONFIG = {
  bosTokenId: 1,
  eosTokenId: 7,
  padTokenId: 0,
  vocabSize: 65_536,
  hiddenSize: 1024,
  intermediateSize: 4608,
  numHiddenLayers: 16,
  numAttentionHeads: 16,
  numKeyValueHeads: 8,
  headDim: 64,
  convCacheLength: 3,
  rmsNormEps: 1e-5,
  ropeTheta: 1_000_000,
  layerTypes: [
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
    "full_attention",
    "conv",
  ],
} as const;

type Linear = {
  weight: np.Array;
};

type RMSNorm = {
  weight: np.Array;
};

type LfmMLP = {
  w1: Linear;
  w2: Linear;
  w3: Linear;
};

type LfmAttention = {
  qProj: Linear;
  kProj: Linear;
  vProj: Linear;
  outProj: Linear;
  qLayernorm: RMSNorm;
  kLayernorm: RMSNorm;
};

type LfmShortConv = {
  conv: Linear;
  inProj: Linear;
  outProj: Linear;
};

type LfmLayerBase = {
  operatorNorm: RMSNorm;
  ffnNorm: RMSNorm;
  feedForward: LfmMLP;
};

type LfmAttentionLayer = LfmLayerBase & {
  selfAttn: LfmAttention;
};

type LfmConvLayer = LfmLayerBase & {
  conv: LfmShortConv;
};

export type LfmModel = {
  embedTokens: Linear;
  embeddingNorm: RMSNorm;
  layers: (LfmAttentionLayer | LfmConvLayer)[];
};

type LfmAttentionCache = {
  kind: "attention";
  key: np.Array;
  value: np.Array;
};

type LfmAttentionArrays = {
  key: np.Array;
  value: np.Array;
};

type LfmConvCache = {
  kind: "conv";
  value: np.Array;
};

type LfmCache = LfmAttentionCache | LfmConvCache;

export type LfmState = {
  caches: LfmCache[];
  position: number;
  capacity: number;
};

const ATTENTION_SCALE = 1 / Math.sqrt(LFM_CONFIG.headDim);
const KV_CACHE_BLOCK_SIZE = 512;

const runLinear = jit(function runLinear(
  { weight }: Linear,
  x: np.Array,
): np.Array {
  return np.dot(x, weight.transpose());
});

const runEmbedding = jit(function runEmbedding(
  { weight }: Linear,
  tokenIds: np.Array,
): np.Array {
  // Keep the residual stream in fp32. LFM is trained with bf16 activations,
  // whose range cannot be represented safely by fp16.
  return weight.slice(tokenIds).astype(np.float32);
});

const runRMSNorm = jit(function runRMSNorm(
  { weight }: RMSNorm,
  x: np.Array,
): np.Array {
  const dtype = x.dtype;
  x = x.astype(np.float32);
  const rms = x.ref.mul(x.ref).mean(-1, { keepdims: true });
  x = x.div(np.sqrt(rms.add(LFM_CONFIG.rmsNormEps)));
  return x.mul(weight.astype(np.float32)).astype(dtype);
});

function runMLP({ w1, w2, w3 }: LfmMLP, x: np.Array): np.Array {
  const gate = nn.silu(runLinear(w1, x.ref));
  const up = runLinear(w3, x);
  return runLinear(w2, gate.mul(up));
}

function rotateHalf(x: np.Array): np.Array {
  const [x1, x2] = np.split(x, 2, -1);
  return np.concatenate([x2.mul(-1), x1], -1);
}

function applyRoPE(
  q: np.Array,
  k: np.Array,
  offset: number,
): [np.Array, np.Array] {
  const [T, , D] = q.shape;
  const halfD = D / 2;
  const dim = np.arange(halfD, undefined, undefined, { dtype: np.float32 });
  const invFreq = np.exp(dim.mul((-Math.log(LFM_CONFIG.ropeTheta) * 2) / D));
  const positions = np
    .arange(T, undefined, undefined, { dtype: np.float32 })
    .add(offset)
    .reshape([T, 1]);
  const freqs = positions.mul(invFreq);

  const cosHalf = np.cos(freqs.ref).astype(q.dtype);
  const sinHalf = np.sin(freqs).astype(q.dtype);
  const cos = np.concatenate([cosHalf.ref, cosHalf], -1).reshape([T, 1, D]);
  const sin = np.concatenate([sinHalf.ref, sinHalf], -1).reshape([T, 1, D]);
  const qOut = q.ref.mul(cos.ref).add(rotateHalf(q).mul(sin.ref));
  const kOut = k.ref.mul(cos).add(rotateHalf(k).mul(sin));
  return [qOut, kOut];
}

function runAttentionPrefill(
  { qProj, kProj, vProj, outProj, qLayernorm, kLayernorm }: LfmAttention,
  x: np.Array,
): { output: np.Array; key: np.Array; value: np.Array } {
  const T = x.shape[0];
  let q = runLinear(qProj, x.ref).reshape([
    T,
    LFM_CONFIG.numAttentionHeads,
    LFM_CONFIG.headDim,
  ]);
  let k = runLinear(kProj, x.ref).reshape([
    T,
    LFM_CONFIG.numKeyValueHeads,
    LFM_CONFIG.headDim,
  ]);
  const v = runLinear(vProj, x).reshape([
    T,
    LFM_CONFIG.numKeyValueHeads,
    LFM_CONFIG.headDim,
  ]);

  q = runRMSNorm(qLayernorm, q);
  k = runRMSNorm(kLayernorm, k);
  [q, k] = applyRoPE(q, k, 0);
  const attn = nn.dotProductAttention(q, k.ref, v.ref, {
    isCausal: true,
    scale: ATTENTION_SCALE,
  });
  const output = runLinear(
    outProj,
    attn.reshape([T, LFM_CONFIG.numAttentionHeads * LFM_CONFIG.headDim]),
  );
  return { output, key: k, value: v };
}

function runAttentionStep(
  { qProj, kProj, vProj, outProj, qLayernorm, kLayernorm }: LfmAttention,
  cache: LfmAttentionArrays,
  x: np.Array,
  position: number,
  slot: number,
  validLength: number,
): { output: np.Array; cache: LfmAttentionArrays } {
  const T = 1;
  let q = runLinear(qProj, x.ref).reshape([
    T,
    LFM_CONFIG.numAttentionHeads,
    LFM_CONFIG.headDim,
  ]);
  let k = runLinear(kProj, x.ref).reshape([
    T,
    LFM_CONFIG.numKeyValueHeads,
    LFM_CONFIG.headDim,
  ]);
  const v = runLinear(vProj, x).reshape([
    T,
    LFM_CONFIG.numKeyValueHeads,
    LFM_CONFIG.headDim,
  ]);

  q = runRMSNorm(qLayernorm, q);
  k = runRMSNorm(kLayernorm, k);
  [q, k] = applyRoPE(q, k, position);

  const capacity = cache.key.shape[0];
  const slotMask = np.arange(capacity).equal(slot).reshape([capacity, 1, 1]);
  const key = np.where(slotMask.ref, np.tile(k, [capacity, 1, 1]), cache.key);
  const value = np.where(slotMask, np.tile(v, [capacity, 1, 1]), cache.value);
  const validMask = np.arange(capacity).less(validLength);
  const attn = nn.dotProductAttention(q, key.ref, value.ref, {
    mask: validMask,
    scale: ATTENTION_SCALE,
  });
  const output = runLinear(
    outProj,
    attn.reshape([T, LFM_CONFIG.numAttentionHeads * LFM_CONFIG.headDim]),
  );
  return { output, cache: { key, value } };
}

function runConvPrefill(
  { conv, inProj, outProj }: LfmShortConv,
  x: np.Array,
): { output: np.Array; cache: np.Array } {
  const T = x.shape[0];
  const [b, c, gate] = np.split(runLinear(inProj, x), 3, -1);
  const bx = b.mul(gate);
  const cache =
    T >= LFM_CONFIG.convCacheLength
      ? bx.ref.slice([T - LFM_CONFIG.convCacheLength], [])
      : np.pad(bx.ref, {
          0: [LFM_CONFIG.convCacheLength - T, 0],
        });

  const padded = np.pad(bx, {
    0: [LFM_CONFIG.convCacheLength - 1, 0],
  });
  const windows = np.stack(
    [
      padded.ref.slice([0, T], []),
      padded.ref.slice([1, T + 1], []),
      padded.slice([2, T + 2], []),
    ],
    -1,
  );
  const convOut = windows
    .mul(
      conv.weight.reshape([LFM_CONFIG.hiddenSize, LFM_CONFIG.convCacheLength]),
    )
    .sum(-1);
  return { output: runLinear(outProj, c.mul(convOut)), cache };
}

function runConvStep(
  { conv, inProj, outProj }: LfmShortConv,
  cache: np.Array,
  x: np.Array,
): { output: np.Array; cache: np.Array } {
  const [b, c, gate] = np.split(runLinear(inProj, x), 3, -1);
  const bx = b.mul(gate);
  const updatedCache = np.concatenate([cache.slice([1], []), bx], 0);
  const convOut = updatedCache.ref
    .mul(
      conv.weight
        .reshape([LFM_CONFIG.hiddenSize, LFM_CONFIG.convCacheLength])
        .transpose(),
    )
    .sum(0)
    .reshape([1, LFM_CONFIG.hiddenSize]);
  return {
    output: runLinear(outProj, c.mul(convOut)),
    cache: updatedCache,
  };
}

function padAttentionCache(
  key: np.Array,
  value: np.Array,
  capacity: number,
): LfmAttentionArrays {
  const T = key.shape[0];
  if (T > capacity) {
    throw new Error(`Prompt length ${T} exceeds cache capacity ${capacity}`);
  }
  if (T === capacity) return { key, value };
  return {
    key: np.pad(key, { 0: [0, capacity - T] }),
    value: np.pad(value, { 0: [0, capacity - T] }),
  };
}

const runAttentionLayerPrefill = jit(
  function runAttentionLayerPrefill(
    { operatorNorm, ffnNorm, feedForward, selfAttn }: LfmAttentionLayer,
    x: np.Array,
    capacity: number,
  ): [np.Array, LfmAttentionArrays] {
    const residual = x.ref;
    x = runRMSNorm(operatorNorm, x);
    const { output, key, value } = runAttentionPrefill(selfAttn, x);
    x = residual.add(output);

    const residual2 = x.ref;
    x = runMLP(feedForward, runRMSNorm(ffnNorm, x));
    return [residual2.add(x), padAttentionCache(key, value, capacity)];
  },
  { staticArgnums: [2] },
);

const runAttentionLayerStep = jit(function runAttentionLayerStep(
  { operatorNorm, ffnNorm, feedForward, selfAttn }: LfmAttentionLayer,
  cache: LfmAttentionArrays,
  x: np.Array,
  position: number,
  slot: number,
  validLength: number,
): [np.Array, LfmAttentionArrays] {
  const residual = x.ref;
  x = runRMSNorm(operatorNorm, x);
  const { output, cache: updatedCache } = runAttentionStep(
    selfAttn,
    cache,
    x,
    position,
    slot,
    validLength,
  );
  x = residual.add(output);

  const residual2 = x.ref;
  x = runMLP(feedForward, runRMSNorm(ffnNorm, x));
  return [residual2.add(x), updatedCache];
});

const runConvLayerPrefill = jit(function runConvLayerPrefill(
  { operatorNorm, ffnNorm, feedForward, conv }: LfmConvLayer,
  x: np.Array,
): [np.Array, np.Array] {
  const residual = x.ref;
  x = runRMSNorm(operatorNorm, x);
  const { output, cache } = runConvPrefill(conv, x);
  x = residual.add(output);

  const residual2 = x.ref;
  x = runMLP(feedForward, runRMSNorm(ffnNorm, x));
  return [residual2.add(x), cache];
});

const runConvLayerStep = jit(function runConvLayerStep(
  { operatorNorm, ffnNorm, feedForward, conv }: LfmConvLayer,
  cache: np.Array,
  x: np.Array,
): [np.Array, np.Array] {
  const residual = x.ref;
  x = runRMSNorm(operatorNorm, x);
  const { output, cache: updatedCache } = runConvStep(conv, cache, x);
  x = residual.add(output);

  const residual2 = x.ref;
  x = runMLP(feedForward, runRMSNorm(ffnNorm, x));
  return [residual2.add(x), updatedCache];
});

function isAttentionLayer(index: number): boolean {
  return LFM_CONFIG.layerTypes[index] === "full_attention";
}

function roundCacheCapacity(requiredCapacity: number): number {
  return Math.max(
    KV_CACHE_BLOCK_SIZE,
    Math.ceil(requiredCapacity / KV_CACHE_BLOCK_SIZE) * KV_CACHE_BLOCK_SIZE,
  );
}

export function createLfmState({
  capacity = KV_CACHE_BLOCK_SIZE,
  dtype = np.float16,
}: { capacity?: number; dtype?: np.DType } = {}): LfmState {
  capacity = roundCacheCapacity(capacity);
  return {
    capacity,
    position: 0,
    caches: LFM_CONFIG.layerTypes.map((type) =>
      type === "full_attention"
        ? {
            kind: "attention" as const,
            key: np.zeros(
              [capacity, LFM_CONFIG.numKeyValueHeads, LFM_CONFIG.headDim],
              { dtype },
            ),
            value: np.zeros(
              [capacity, LFM_CONFIG.numKeyValueHeads, LFM_CONFIG.headDim],
              { dtype },
            ),
          }
        : {
            kind: "conv" as const,
            value: np.zeros(
              [LFM_CONFIG.convCacheLength, LFM_CONFIG.hiddenSize],
              { dtype },
            ),
          },
    ),
  };
}

function ensureStateCapacity(state: LfmState, requiredCapacity: number) {
  if (state.capacity >= requiredCapacity) return;
  const oldCapacity = state.capacity;
  const newCapacity = roundCacheCapacity(requiredCapacity);
  for (const cache of state.caches) {
    if (cache.kind !== "attention") continue;
    cache.key = np.pad(cache.key, { 0: [0, newCapacity - oldCapacity] });
    cache.value = np.pad(cache.value, { 0: [0, newCapacity - oldCapacity] });
  }
  state.capacity = newCapacity;
}

export function runLfmPrefill(
  model: LfmModel,
  tokenIds: np.Array,
  state: LfmState,
): np.Array {
  ensureStateCapacity(state, tokenIds.shape[0]);
  let x = runEmbedding({ weight: model.embedTokens.weight.ref }, tokenIds);

  for (let i = 0; i < LFM_CONFIG.numHiddenLayers; i++) {
    const cache = state.caches[i];
    if (isAttentionLayer(i)) {
      if (cache.kind !== "attention")
        throw new Error("Invalid attention cache");
      cache.key.dispose();
      cache.value.dispose();
      const [nextX, nextCache] = runAttentionLayerPrefill(
        model.layers[i] as LfmAttentionLayer,
        x,
        state.capacity,
      );
      x = nextX;
      state.caches[i] = { kind: "attention", ...nextCache };
    } else {
      if (cache.kind !== "conv") throw new Error("Invalid convolution cache");
      cache.value.dispose();
      const [nextX, nextCache] = runConvLayerPrefill(
        model.layers[i] as LfmConvLayer,
        x,
      );
      x = nextX;
      state.caches[i] = { kind: "conv", value: nextCache };
    }
  }

  x = runRMSNorm(model.embeddingNorm, x).slice([-1]);
  const logits = runLinear(model.embedTokens, x).reshape([
    LFM_CONFIG.vocabSize,
  ]);
  state.position = tokenIds.shape[0];
  return logits;
}

export function runLfmStep(
  model: LfmModel,
  tokenId: number,
  state: LfmState,
): np.Array {
  ensureStateCapacity(state, state.position + 1);
  const tokenIds = np.array([tokenId], { dtype: np.uint32 });
  let x = runEmbedding({ weight: model.embedTokens.weight.ref }, tokenIds);
  const position = state.position;
  const slot = position;
  const validLength = position + 1;

  for (let i = 0; i < LFM_CONFIG.numHiddenLayers; i++) {
    const cache = state.caches[i];
    if (isAttentionLayer(i)) {
      if (cache.kind !== "attention")
        throw new Error("Invalid attention cache");
      const [nextX, nextCache] = runAttentionLayerStep(
        model.layers[i] as LfmAttentionLayer,
        { key: cache.key, value: cache.value },
        x,
        position,
        slot,
        validLength,
      );
      x = nextX;
      state.caches[i] = { kind: "attention", ...nextCache };
    } else {
      if (cache.kind !== "conv") throw new Error("Invalid convolution cache");
      const [nextX, nextCache] = runConvLayerStep(
        model.layers[i] as LfmConvLayer,
        cache.value,
        x,
      );
      x = nextX;
      state.caches[i] = { kind: "conv", value: nextCache };
    }
  }

  x = runRMSNorm(model.embeddingNorm, x);
  const logits = runLinear(model.embedTokens, x).reshape([
    LFM_CONFIG.vocabSize,
  ]);
  state.position++;
  return logits;
}

const mapper = new WeightMapper({
  prefix: {
    "model.": "",
  },
  substring: {
    embed_tokens: "embedTokens",
    embedding_norm: "embeddingNorm",
    operator_norm: "operatorNorm",
    ffn_norm: "ffnNorm",
    feed_forward: "feedForward",
    in_proj: "inProj",
    out_proj: "outProj",
    self_attn: "selfAttn",
    q_proj: "qProj",
    k_proj: "kProj",
    v_proj: "vProj",
    q_layernorm: "qLayernorm",
    k_layernorm: "kLayernorm",
  },
});

function tensorToArray(
  tensor: safetensors.Tensor,
  dtype: np.DType = np.float16,
): np.Array {
  if (tensor.dtype !== "F16") {
    throw new Error(
      `Expected fp16 LFM2.5 weights, but tensor has dtype ${tensor.dtype}. ` +
        `Use model-fp16.safetensors.`,
    );
  }
  switch (dtype) {
    case np.float16:
      return np.array(tensor.data as Float16Array<ArrayBuffer>, {
        shape: tensor.shape,
        dtype: np.float16,
      });
    case np.float32:
      return np.array(
        new Float32Array(tensor.data as Float16Array<ArrayBuffer>),
        { shape: tensor.shape, dtype: np.float32 },
      );
    default:
      throw new Error(`Unsupported dtype ${dtype}`);
  }
}

export async function lfmFromSafetensors(
  file: safetensors.File,
  dtype: np.DType = np.float16,
): Promise<LfmModel> {
  const hydrated: Record<string, np.Array> = {};
  for (const [key, tensor] of Object.entries(file.tensors)) {
    hydrated[mapper.mapKey(key)] = tensorToArray(tensor, dtype);
  }

  const model = safetensors.toNested(hydrated) as LfmModel;
  if (model.layers.length !== LFM_CONFIG.numHiddenLayers) {
    throw new Error(
      `Expected ${LFM_CONFIG.numHiddenLayers} LFM2.5 layers, ` +
        `found ${model.layers.length}`,
    );
  }

  return blockUntilReady(model);
}
