import { jit, nn, numpy as np } from "@jax-js/jax";
import { safetensors, WeightMapper } from "@jax-js/loaders";

export const GEMMA_CONFIG = {
  bosTokenId: 2,
  eosTokenId: 1,
  padTokenId: 0,
  vocabSize: 262_144,
  hiddenSize: 640,
  intermediateSize: 2048,
  numHiddenLayers: 18,
  numAttentionHeads: 4,
  numKeyValueHeads: 1,
  headDim: 256,
  rmsNormEps: 1e-6,
  queryPreAttnScalar: 256,
  ropeTheta: 1_000_000,
  ropeLocalBaseFreq: 10_000,
  layerTypes: [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
  ],
} as const;

export type Linear = {
  weight: np.Array;
  bias?: np.Array;
};

export type RMSNorm = {
  weight: np.Array;
};

export type GemmaMLP = {
  gateProj: Linear;
  upProj: Linear;
  downProj: Linear;
};

export type GemmaAttention = {
  qProj: Linear;
  kProj: Linear;
  vProj: Linear;
  oProj: Linear;
  qNorm: RMSNorm;
  kNorm: RMSNorm;
};

export type GemmaDecoderLayer = {
  inputLayernorm: RMSNorm;
  postAttentionLayernorm: RMSNorm;
  preFeedforwardLayernorm: RMSNorm;
  postFeedforwardLayernorm: RMSNorm;
  selfAttn: GemmaAttention;
  mlp: GemmaMLP;
};

export type GemmaModel = {
  embedTokens: Linear;
  layers: GemmaDecoderLayer[];
  norm: RMSNorm;
};

export type GemmaKVCache = {
  key: np.Array; // [capacity, num_key_value_heads, head_dim]
  value: np.Array; // [capacity, num_key_value_heads, head_dim]
};

export type GemmaState = {
  caches: GemmaKVCache[];
  position: number;
  capacity: number;
};

const ATTENTION_SCALE = 1 / Math.sqrt(GEMMA_CONFIG.queryPreAttnScalar);
const EMBED_SCALE = Math.sqrt(GEMMA_CONFIG.hiddenSize);
const KV_CACHE_BLOCK_SIZE = 512;

function runLinear({ weight, bias }: Linear, x: np.Array): np.Array {
  x = np.dot(x, weight.transpose());
  if (bias) x = x.add(bias);
  return x;
}

const runEmbedding = jit(function runEmbedding(
  { weight }: Linear,
  tokenIds: np.Array,
): np.Array {
  // Gemma is trained for bf16 activations; fp16 residuals overflow in this
  // browser implementation, so keep the residual stream in fp32 while storing
  // weights in fp16.
  return weight.slice(tokenIds).astype(np.float32).mul(EMBED_SCALE);
});

function runGemmaRMSNorm(
  { weight }: RMSNorm,
  x: np.Array,
  eps: number = GEMMA_CONFIG.rmsNormEps,
): np.Array {
  const dtype = x.dtype;
  x = x.astype(np.float32);
  const rms = x.ref.mul(x.ref).mean(-1, { keepdims: true });
  x = x.div(np.sqrt(rms.add(eps)));

  // Gemma RMSNorm weights are zero-centered: output = norm(x) * (1 + weight).
  const scale = weight.astype(np.float32).add(1);
  return x.mul(scale).astype(dtype);
}

function rotateHalf(x: np.Array): np.Array {
  const [x1, x2] = np.split(x, 2, -1);
  return np.concatenate([x2.mul(-1), x1], -1);
}

function applyGemmaRoPE(
  q: np.Array, // [T, num_heads, head_dim]
  k: np.Array, // [T, num_key_value_heads, head_dim]
  offset: number,
  theta: number,
): [np.Array, np.Array] {
  const [T, , D] = q.shape;
  const halfD = D / 2;

  const dim = np.arange(halfD, undefined, undefined, { dtype: np.float32 });
  const invFreq = np.exp(dim.mul((-Math.log(theta) * 2) / D));
  const positions = np
    .arange(T, undefined, undefined, { dtype: np.float32 })
    .add(offset)
    .reshape([T, 1]);
  const freqs = positions.mul(invFreq); // [T, head_dim / 2]

  const cosHalf = np.cos(freqs.ref).astype(q.dtype);
  const sinHalf = np.sin(freqs).astype(q.dtype);
  const cos = np.concatenate([cosHalf.ref, cosHalf], -1).reshape([T, 1, D]);
  const sin = np.concatenate([sinHalf.ref, sinHalf], -1).reshape([T, 1, D]);

  const qOut = q.ref.mul(cos.ref).add(rotateHalf(q).mul(sin.ref));
  const kOut = k.ref.mul(cos).add(rotateHalf(k).mul(sin));
  return [qOut, kOut];
}

function runMLP({ gateProj, upProj, downProj }: GemmaMLP, x: np.Array) {
  const gate = nn.gelu(runLinear(gateProj, x.ref), { approximate: true });
  const up = runLinear(upProj, x);
  return runLinear(downProj, gate.mul(up));
}

function runAttentionPrefill(
  { qProj, kProj, vProj, oProj, qNorm, kNorm }: GemmaAttention,
  x: np.Array,
  ropeTheta: number,
): { output: np.Array; key: np.Array; value: np.Array } {
  const T = x.shape[0];
  let q = runLinear(qProj, x.ref).reshape([
    T,
    GEMMA_CONFIG.numAttentionHeads,
    GEMMA_CONFIG.headDim,
  ]);
  let k = runLinear(kProj, x.ref).reshape([
    T,
    GEMMA_CONFIG.numKeyValueHeads,
    GEMMA_CONFIG.headDim,
  ]);
  const v = runLinear(vProj, x).reshape([
    T,
    GEMMA_CONFIG.numKeyValueHeads,
    GEMMA_CONFIG.headDim,
  ]);

  q = runGemmaRMSNorm(qNorm, q);
  k = runGemmaRMSNorm(kNorm, k);
  [q, k] = applyGemmaRoPE(q, k, 0, ropeTheta);

  const attn = nn.dotProductAttention(q, k.ref, v.ref, {
    isCausal: true,
    scale: ATTENTION_SCALE,
  });
  const output = runLinear(
    oProj,
    attn.reshape([T, GEMMA_CONFIG.numAttentionHeads * GEMMA_CONFIG.headDim]),
  );
  return { output, key: k, value: v };
}

function runAttentionStep(
  { qProj, kProj, vProj, oProj, qNorm, kNorm }: GemmaAttention,
  cache: GemmaKVCache,
  x: np.Array,
  position: number,
  slot: number,
  validLength: number,
  ropeTheta: number,
): { output: np.Array; cache: GemmaKVCache } {
  const T = 1;
  let q = runLinear(qProj, x.ref).reshape([
    T,
    GEMMA_CONFIG.numAttentionHeads,
    GEMMA_CONFIG.headDim,
  ]);
  let k = runLinear(kProj, x.ref).reshape([
    T,
    GEMMA_CONFIG.numKeyValueHeads,
    GEMMA_CONFIG.headDim,
  ]);
  const v = runLinear(vProj, x).reshape([
    T,
    GEMMA_CONFIG.numKeyValueHeads,
    GEMMA_CONFIG.headDim,
  ]);

  q = runGemmaRMSNorm(qNorm, q);
  k = runGemmaRMSNorm(kNorm, k);
  [q, k] = applyGemmaRoPE(q, k, position, ropeTheta);

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
    oProj,
    attn.reshape([T, GEMMA_CONFIG.numAttentionHeads * GEMMA_CONFIG.headDim]),
  );
  return { output, cache: { key, value } };
}

function padCache(
  key: np.Array,
  value: np.Array,
  capacity: number,
): GemmaKVCache {
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

export const runGemmaDecoderLayerPrefill = jit(
  function runGemmaDecoderLayerPrefill(
    {
      inputLayernorm,
      postAttentionLayernorm,
      preFeedforwardLayernorm,
      postFeedforwardLayernorm,
      selfAttn,
      mlp,
    }: GemmaDecoderLayer,
    x: np.Array,
    ropeTheta: number,
    capacity: number,
  ): [np.Array, GemmaKVCache] {
    const residual = x.ref;
    x = runGemmaRMSNorm(inputLayernorm, x);
    const {
      output: attnOut,
      key,
      value,
    } = runAttentionPrefill(selfAttn, x, ropeTheta);
    x = runGemmaRMSNorm(postAttentionLayernorm, attnOut);
    x = residual.add(x);

    const residual2 = x.ref;
    x = runGemmaRMSNorm(preFeedforwardLayernorm, x);
    x = runMLP(mlp, x);
    x = runGemmaRMSNorm(postFeedforwardLayernorm, x);
    x = residual2.add(x);

    return [x, padCache(key, value, capacity)];
  },
  { staticArgnums: [2, 3] },
);

export const runGemmaDecoderLayerStep = jit(
  function runGemmaDecoderLayerStep(
    {
      inputLayernorm,
      postAttentionLayernorm,
      preFeedforwardLayernorm,
      postFeedforwardLayernorm,
      selfAttn,
      mlp,
    }: GemmaDecoderLayer,
    cache: GemmaKVCache,
    x: np.Array,
    position: number,
    slot: number,
    validLength: number,
    ropeTheta: number,
  ): [np.Array, GemmaKVCache] {
    const residual = x.ref;
    x = runGemmaRMSNorm(inputLayernorm, x);
    const { output: attnOut, cache: updatedCache } = runAttentionStep(
      selfAttn,
      cache,
      x,
      position,
      slot,
      validLength,
      ropeTheta,
    );
    x = runGemmaRMSNorm(postAttentionLayernorm, attnOut);
    x = residual.add(x);

    const residual2 = x.ref;
    x = runGemmaRMSNorm(preFeedforwardLayernorm, x);
    x = runMLP(mlp, x);
    x = runGemmaRMSNorm(postFeedforwardLayernorm, x);
    x = residual2.add(x);

    return [x, updatedCache];
  },
  { staticArgnums: [6] },
);

function layerRopeTheta(layerIndex: number): number {
  return GEMMA_CONFIG.layerTypes[layerIndex] === "full_attention"
    ? GEMMA_CONFIG.ropeTheta
    : GEMMA_CONFIG.ropeLocalBaseFreq;
}

function roundCacheCapacity(requiredCapacity: number): number {
  return Math.max(
    KV_CACHE_BLOCK_SIZE,
    Math.ceil(requiredCapacity / KV_CACHE_BLOCK_SIZE) * KV_CACHE_BLOCK_SIZE,
  );
}

export function createGemmaState(
  capacity: number = KV_CACHE_BLOCK_SIZE,
): GemmaState {
  capacity = roundCacheCapacity(capacity);
  return {
    capacity,
    position: 0,
    caches: Array.from({ length: GEMMA_CONFIG.numHiddenLayers }, () => ({
      key: np.zeros(
        [capacity, GEMMA_CONFIG.numKeyValueHeads, GEMMA_CONFIG.headDim],
        { dtype: np.float16 },
      ),
      value: np.zeros(
        [capacity, GEMMA_CONFIG.numKeyValueHeads, GEMMA_CONFIG.headDim],
        { dtype: np.float16 },
      ),
    })),
  };
}

function ensureGemmaStateCapacity(state: GemmaState, requiredCapacity: number) {
  if (state.capacity >= requiredCapacity) return;

  const oldCapacity = state.capacity;
  const newCapacity = roundCacheCapacity(requiredCapacity);
  for (const cache of state.caches) {
    cache.key = np.pad(cache.key, { 0: [0, newCapacity - oldCapacity] });
    cache.value = np.pad(cache.value, { 0: [0, newCapacity - oldCapacity] });
  }
  state.capacity = newCapacity;
}

export function runGemmaPrefill(
  model: GemmaModel,
  tokenIds: np.Array,
  state: GemmaState,
): np.Array {
  ensureGemmaStateCapacity(state, tokenIds.shape[0]);

  let x = runEmbedding({ weight: model.embedTokens.weight.ref }, tokenIds);

  for (let i = 0; i < GEMMA_CONFIG.numHiddenLayers; i++) {
    state.caches[i].key.dispose();
    state.caches[i].value.dispose();
    [x, state.caches[i]] = runGemmaDecoderLayerPrefill(
      model.layers[i],
      x,
      layerRopeTheta(i),
      state.capacity,
    );
  }

  x = runGemmaRMSNorm(model.norm, x);
  x = x.slice([-1]);
  const logits = runLinear(model.embedTokens, x).reshape([
    GEMMA_CONFIG.vocabSize,
  ]);
  state.position = tokenIds.shape[0];
  return logits;
}

export function runGemmaStep(
  model: GemmaModel,
  tokenId: number,
  state: GemmaState,
): np.Array {
  ensureGemmaStateCapacity(state, state.position + 1);

  const tokenIds = np.array([tokenId], { dtype: np.uint32 });
  let x = runEmbedding({ weight: model.embedTokens.weight.ref }, tokenIds);
  const position = state.position;
  const slot = position;
  const validLength = position + 1;

  for (let i = 0; i < GEMMA_CONFIG.numHiddenLayers; i++) {
    [x, state.caches[i]] = runGemmaDecoderLayerStep(
      model.layers[i],
      state.caches[i],
      x,
      position,
      slot,
      validLength,
      layerRopeTheta(i),
    );
  }

  x = runGemmaRMSNorm(model.norm, x);
  const logits = runLinear(model.embedTokens, x).reshape([
    GEMMA_CONFIG.vocabSize,
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
    input_layernorm: "inputLayernorm",
    post_attention_layernorm: "postAttentionLayernorm",
    pre_feedforward_layernorm: "preFeedforwardLayernorm",
    post_feedforward_layernorm: "postFeedforwardLayernorm",
    self_attn: "selfAttn",
    q_proj: "qProj",
    k_proj: "kProj",
    v_proj: "vProj",
    o_proj: "oProj",
    q_norm: "qNorm",
    k_norm: "kNorm",
    gate_proj: "gateProj",
    up_proj: "upProj",
    down_proj: "downProj",
  },
});

function tensorToArray(tensor: safetensors.Tensor): np.Array {
  if (tensor.dtype !== "F16") {
    throw new Error(
      `Expected fp16 Gemma weights, but tensor has dtype ${tensor.dtype}. ` +
        `Use model-fp16.safetensors.`,
    );
  }
  return np.array(tensor.data as Float16Array<ArrayBuffer>, {
    shape: tensor.shape,
    dtype: np.float16,
  });
}

export function fromSafetensors(file: safetensors.File): GemmaModel {
  const hydrated: Record<string, np.Array> = {};
  for (const [key, tensor] of Object.entries(file.tensors)) {
    hydrated[mapper.mapKey(key)] = tensorToArray(tensor);
  }

  const model = safetensors.toNested(hydrated) as GemmaModel;
  if (model.layers.length !== GEMMA_CONFIG.numHiddenLayers) {
    throw new Error(
      `Expected ${GEMMA_CONFIG.numHiddenLayers} Gemma layers, ` +
        `found ${model.layers.length}`,
    );
  }
  return model;
}
