import { blockUntilReady, jit, lax, nn, numpy as np, tree } from "@jax-js/jax";
import { safetensors, WeightMapper } from "@jax-js/loaders";

const EN_SUPPRESS_TOKENS = [
  1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91,
  92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303,
  1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600,
  4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907,
  13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724,
  22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549, 47282,
  49146, 50257, 50357, 50358, 50359, 50360, 50361,
];

type WhisperSize = {
  dModel: number;
  layers: number;
  heads: number;
};

type WhisperTokens = {
  vocabSize: number;
  sotToken: number;
  eosToken: number;
  noTimestampsToken: number;
  timestampBeginToken: number;
  promptTokens: number[];
  beginSuppressTokens: number[];
};

export type WhisperModelId =
  | "tiny"
  | "tiny.en"
  | "base"
  | "base.en"
  | "small"
  | "small.en"
  | "medium"
  | "medium.en";

export type WhisperConfig = {
  id: WhisperModelId;
  label: string;
  repo: string;
  dModel: number;
  encoderLayers: number;
  decoderLayers: number;
  heads: number;
  headDim: number;
  ffnDim: number;
  vocabSize: number;
  sotToken: number;
  eosToken: number;
  noTimestampsToken: number;
  timestampBeginToken: number;
  timestampStepSeconds: number;
  maxSourcePositions: number;
  maxTargetPositions: number;
  beginSuppressTokens: number[];
  suppressTokens: number[];
  promptTokens: number[];
};

const SIZES: Record<"tiny" | "base" | "small" | "medium", WhisperSize> = {
  tiny: { dModel: 384, layers: 4, heads: 6 },
  base: { dModel: 512, layers: 6, heads: 8 },
  small: { dModel: 768, layers: 12, heads: 12 },
  medium: { dModel: 1024, layers: 24, heads: 16 },
};

const ENGLISH_TOKENS: WhisperTokens = {
  vocabSize: 51864,
  sotToken: 50257,
  eosToken: 50256,
  noTimestampsToken: 50362,
  timestampBeginToken: 50363,
  promptTokens: [50257],
  beginSuppressTokens: [220, 50256],
};

const MULTILINGUAL_TOKENS: WhisperTokens = {
  vocabSize: 51865,
  sotToken: 50258,
  eosToken: 50257,
  noTimestampsToken: 50363,
  timestampBeginToken: 50364,
  // Force English transcription so multilingual checkpoints behave like the
  // .en variants for this demo.
  promptTokens: [50258, 50259, 50359],
  beginSuppressTokens: [220, 50257],
};

function whisperConfig(id: WhisperModelId): WhisperConfig {
  const family = id.replace(".en", "") as keyof typeof SIZES;
  const size = SIZES[family];
  const tokens = id.endsWith(".en") ? ENGLISH_TOKENS : MULTILINGUAL_TOKENS;
  return {
    id,
    label: id,
    repo: `mlx-community/whisper-${id}-asr-fp16`,
    dModel: size.dModel,
    encoderLayers: size.layers,
    decoderLayers: size.layers,
    heads: size.heads,
    headDim: size.dModel / size.heads,
    ffnDim: size.dModel * 4,
    timestampStepSeconds: 0.02,
    maxSourcePositions: 1500,
    maxTargetPositions: 448,
    suppressTokens: EN_SUPPRESS_TOKENS,
    ...tokens,
  };
}

export const WHISPER_MODELS = (
  [
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
  ] as const
).map(whisperConfig);

export const DEFAULT_WHISPER_CONFIG = WHISPER_MODELS[0];

export type Linear = {
  weight: np.Array;
  bias?: np.Array;
};

export type Conv1d = {
  weight: np.Array;
  bias: np.Array;
};

export type LayerNorm = {
  weight: np.Array;
  bias: np.Array;
};

export type Attention = {
  qProj: Linear;
  kProj: Linear;
  vProj: Linear;
  outProj: Linear;
};

export type WhisperEncoderLayer = {
  selfAttn: Attention;
  selfAttnLayerNorm: LayerNorm;
  fc1: Linear;
  fc2: Linear;
  finalLayerNorm: LayerNorm;
};

export type WhisperDecoderLayer = {
  selfAttn: Attention;
  encoderAttn: Attention;
  selfAttnLayerNorm: LayerNorm;
  encoderAttnLayerNorm: LayerNorm;
  fc1: Linear;
  fc2: Linear;
  finalLayerNorm: LayerNorm;
};

export type WhisperModel = {
  encoder: {
    conv1: Conv1d;
    conv2: Conv1d;
    embedPositions: { weight: np.Array };
    layers: WhisperEncoderLayer[];
    layerNorm: LayerNorm;
  };
  decoder: {
    embedTokens: { weight: np.Array };
    embedPositions: { weight: np.Array };
    layers: WhisperDecoderLayer[];
    layerNorm: LayerNorm;
  };
};

export type KVCache = {
  key: np.Array;
  value: np.Array;
};

export type WhisperState = {
  position: number;
  caches: KVCache[];
};

type CrossAttentionKV = {
  kProj: Linear;
  vProj: Linear;
};

export function createWhisperState(
  maxTokens: number,
  dtype: np.DType,
  config: WhisperConfig,
): WhisperState {
  return {
    position: 0,
    caches: Array.from({ length: config.decoderLayers }, () => ({
      key: np.zeros([maxTokens, config.heads, config.headDim], {
        dtype,
      }),
      value: np.zeros([maxTokens, config.heads, config.headDim], { dtype }),
    })),
  };
}

const encoderJitCache = new Map<
  WhisperModelId,
  (encoder: WhisperModel["encoder"], features: np.Array) => np.Array
>();
const crossKVJitCache = new Map<
  WhisperModelId,
  (layers: CrossAttentionKV[], encoderHidden: np.Array) => KVCache[]
>();
const decoderStepJitCache = new Map<
  WhisperModelId,
  (
    decoder: WhisperModel["decoder"],
    crossKV: KVCache[],
    caches: KVCache[],
    tokenIds: np.Array,
    position: np.Array,
    positionIds: np.Array,
  ) => [np.Array, KVCache[]]
>();

function encoderJit(config: WhisperConfig) {
  let fn = encoderJitCache.get(config.id);
  if (!fn) {
    fn = jit(function runWhisperEncoderJit(
      encoder: WhisperModel["encoder"],
      features: np.Array,
    ): np.Array {
      let x = encoderConv(encoder, features);
      for (const layer of encoder.layers) {
        x = encoderLayer(config, layer, x);
      }
      return runLayerNorm(encoder.layerNorm, x);
    });
    encoderJitCache.set(config.id, fn);
  }
  return fn;
}

export function runWhisperEncoder(
  encoder: WhisperModel["encoder"],
  features: np.Array,
  config: WhisperConfig,
): np.Array {
  return encoderJit(config)(tree.ref(encoder), features);
}

function crossKVJit(config: WhisperConfig) {
  let fn = crossKVJitCache.get(config.id);
  if (!fn) {
    fn = jit(function runWhisperCrossKV(
      layers: CrossAttentionKV[],
      encoderHidden: np.Array,
    ): KVCache[] {
      const caches: KVCache[] = [];
      for (let i = 0; i < layers.length; i++) {
        const layer = layers[i];
        caches.push({
          key: splitHeads(config, runLinear(layer.kProj, encoderHidden.ref)),
          value: splitHeads(
            config,
            runLinear(
              layer.vProj,
              i === layers.length - 1 ? encoderHidden : encoderHidden.ref,
            ),
          ),
        });
      }
      return caches;
    });
    crossKVJitCache.set(config.id, fn);
  }
  return fn;
}

export function prepareWhisperCrossKV(
  decoder: WhisperModel["decoder"],
  encoderHidden: np.Array,
  config: WhisperConfig,
): KVCache[] {
  return crossKVJit(config)(
    decoder.layers.map((layer) =>
      tree.ref({
        kProj: layer.encoderAttn.kProj,
        vProj: layer.encoderAttn.vProj,
      }),
    ),
    encoderHidden,
  );
}

function decoderStepJit(config: WhisperConfig) {
  let fn = decoderStepJitCache.get(config.id);
  if (!fn) {
    fn = jit(function runWhisperDecoderStepJit(
      decoder: WhisperModel["decoder"],
      crossKV: KVCache[],
      caches: KVCache[],
      tokenIds: np.Array,
      position: np.Array,
      positionIds: np.Array,
    ): [np.Array, KVCache[]] {
      let x = decoder.embedTokens.weight.ref
        .slice(tokenIds)
        .add(decoder.embedPositions.weight.ref.slice(positionIds));
      const nextCaches: KVCache[] = [];
      for (let i = 0; i < decoder.layers.length; i++) {
        const [nextX, nextCache] = decoderLayerStep(
          config,
          decoder.layers[i],
          caches[i],
          crossKV[i],
          x,
          i === decoder.layers.length - 1 ? position : position.ref,
        );
        x = nextX;
        nextCaches.push(nextCache);
      }
      x = runLayerNorm(decoder.layerNorm, x);
      return [
        np
          .dot(x, decoder.embedTokens.weight.ref.transpose())
          .reshape([config.vocabSize]),
        nextCaches,
      ];
    });
    decoderStepJitCache.set(config.id, fn);
  }
  return fn;
}

export function runWhisperDecoderStep(
  decoder: WhisperModel["decoder"],
  crossKV: KVCache[],
  state: WhisperState,
  token: number,
  config: WhisperConfig,
): np.Array {
  const tokenIds = np.array([token], { dtype: np.uint32 });
  const position = np.array(state.position, { dtype: np.int32 });
  const positionIds = np.array([state.position], { dtype: np.uint32 });
  const [logits, caches] = decoderStepJit(config)(
    tree.ref(decoder),
    tree.ref(crossKV),
    state.caches,
    tokenIds,
    position,
    positionIds,
  );
  state.caches = caches;
  state.position++;
  return logits;
}

function encoderConv(
  { conv1, conv2, embedPositions }: WhisperModel["encoder"],
  features: np.Array,
): np.Array {
  let x = lax
    .conv(features, conv1.weight.ref, [1], [[1, 1]])
    .add(conv1.bias.ref.reshape([1, conv1.bias.shape[0], 1]));
  x = nn.gelu(x, { approximate: false });
  x = lax
    .conv(x, conv2.weight.ref, [2], [[1, 1]])
    .add(conv2.bias.ref.reshape([1, conv2.bias.shape[0], 1]));
  const dtype = x.dtype;
  return nn
    .gelu(x, { approximate: false })
    .slice(0)
    .transpose()
    .add(embedPositions.weight.ref.astype(dtype));
}

function encoderLayer(
  config: WhisperConfig,
  layer: WhisperEncoderLayer,
  x: np.Array,
): np.Array {
  const residual = x.ref;
  x = residual.add(
    runSelfAttention(
      config,
      layer.selfAttn,
      runLayerNorm(layer.selfAttnLayerNorm, x),
    ),
  );
  return x.ref.add(
    runLinear(
      layer.fc2,
      nn.gelu(runLinear(layer.fc1, runLayerNorm(layer.finalLayerNorm, x)), {
        approximate: false,
      }),
    ),
  );
}

function decoderLayerStep(
  config: WhisperConfig,
  layer: WhisperDecoderLayer,
  selfCache: KVCache,
  crossCache: KVCache,
  x: np.Array,
  position: np.Array,
): [np.Array, KVCache] {
  const residual = x.ref;
  const [selfOut, nextSelfCache] = runDecoderSelfAttention(
    config,
    layer.selfAttn,
    selfCache,
    runLayerNorm(layer.selfAttnLayerNorm, x),
    position,
  );
  x = residual.add(selfOut);
  x = x.ref.add(
    runDecoderCrossAttention(
      config,
      layer.encoderAttn,
      crossCache,
      runLayerNorm(layer.encoderAttnLayerNorm, x),
    ),
  );
  return [
    x.ref.add(
      runLinear(
        layer.fc2,
        nn.gelu(runLinear(layer.fc1, runLayerNorm(layer.finalLayerNorm, x)), {
          approximate: false,
        }),
      ),
    ),
    nextSelfCache,
  ];
}

function runLinear({ weight, bias }: Linear, x: np.Array): np.Array {
  x = np.dot(x, weight.ref.transpose());
  if (bias) x = x.add(bias.ref);
  return x;
}

function runLayerNorm(
  { weight, bias }: LayerNorm,
  x: np.Array,
  eps = 1e-5,
): np.Array {
  const dtype = x.dtype;
  x = x.astype(np.float32);
  const mean = x.ref.mean(-1, { keepdims: true });
  const centered = x.sub(mean);
  const variance = centered.ref.mul(centered.ref).mean(-1, { keepdims: true });
  const scale = weight.ref.astype(np.float32);
  const offset = bias.ref.astype(np.float32);
  return centered
    .div(np.sqrt(variance.add(eps)))
    .mul(scale)
    .add(offset)
    .astype(dtype);
}

function splitHeads(config: WhisperConfig, x: np.Array): np.Array {
  return x.reshape([x.shape[0], config.heads, config.headDim]);
}

function mergeHeads(config: WhisperConfig, x: np.Array): np.Array {
  return x.reshape([x.shape[0], config.heads * config.headDim]);
}

function runSelfAttention(
  config: WhisperConfig,
  attn: Attention,
  x: np.Array,
): np.Array {
  const q = splitHeads(config, runLinear(attn.qProj, x.ref));
  const k = splitHeads(config, runLinear(attn.kProj, x.ref));
  const v = splitHeads(config, runLinear(attn.vProj, x));
  return runLinear(
    attn.outProj,
    mergeHeads(config, nn.dotProductAttention(q, k, v)),
  );
}

function runDecoderSelfAttention(
  config: WhisperConfig,
  attn: Attention,
  cache: KVCache,
  x: np.Array,
  position: np.Array,
): [np.Array, KVCache] {
  const q = splitHeads(config, runLinear(attn.qProj, x.ref));
  const k = splitHeads(config, runLinear(attn.kProj, x.ref));
  const v = splitHeads(config, runLinear(attn.vProj, x));
  const capacity = cache.key.shape[0];
  const slotMask = np
    .arange(capacity, undefined, undefined, { dtype: np.int32 })
    .equal(position.ref)
    .reshape([capacity, 1, 1]);
  const key = np.where(slotMask.ref, np.tile(k, [capacity, 1, 1]), cache.key);
  const value = np.where(slotMask, np.tile(v, [capacity, 1, 1]), cache.value);
  const validMask = np
    .arange(capacity, undefined, undefined, { dtype: np.int32 })
    .less(position.add(1));
  const out = nn.dotProductAttention(q, key.ref, value.ref, {
    mask: validMask,
  });
  return [runLinear(attn.outProj, mergeHeads(config, out)), { key, value }];
}

function runDecoderCrossAttention(
  config: WhisperConfig,
  attn: Attention,
  crossCache: KVCache,
  x: np.Array,
): np.Array {
  const q = splitHeads(config, runLinear(attn.qProj, x));
  return runLinear(
    attn.outProj,
    mergeHeads(
      config,
      nn.dotProductAttention(q, crossCache.key.ref, crossCache.value.ref),
    ),
  );
}

const mapper = new WeightMapper({
  prefix: {
    "model.": "",
  },
  substring: {
    ".blocks.": ".layers.",
    cross_attn_ln: "encoderAttnLayerNorm",
    attn_ln: "selfAttnLayerNorm",
    mlp_ln: "finalLayerNorm",
    ln_post: "layerNorm",
    "decoder.ln.": "decoder.layerNorm.",
    positional_embedding: "embedPositions.weight",
    token_embedding: "embedTokens",
    cross_attn: "encoderAttn",
    ".attn.": ".selfAttn.",
    ".query.": ".qProj.",
    ".key.": ".kProj.",
    ".value.": ".vProj.",
    ".out.": ".outProj.",
    mlp1: "fc1",
    mlp2: "fc2",
    self_attn: "selfAttn",
    encoder_attn: "encoderAttn",
    q_proj: "qProj",
    k_proj: "kProj",
    v_proj: "vProj",
    out_proj: "outProj",
    embed_tokens: "embedTokens",
    embed_positions: "embedPositions",
  },
  autoCamelCase: true,
});

function tensorToArray(tensor: safetensors.Tensor, dtype: np.DType): np.Array {
  if (tensor.dtype === "F16") {
    if (dtype === np.float32) {
      return np.array(
        new Float32Array(tensor.data as Float16Array<ArrayBuffer>),
        {
          shape: tensor.shape,
          dtype: np.float32,
        },
      );
    }
    const arr = np.array(tensor.data as Float16Array<ArrayBuffer>, {
      shape: tensor.shape,
      dtype: np.float16,
    });
    return dtype === np.float16 ? arr : arr.astype(dtype);
  }
  if (tensor.dtype === "F32") {
    const arr = np.array(tensor.data as Float32Array<ArrayBuffer>, {
      shape: tensor.shape,
      dtype: np.float32,
    });
    return dtype === np.float32 ? arr : arr.astype(dtype);
  }
  throw new Error(`Expected F16 or F32 Whisper tensor, got ${tensor.dtype}`);
}

export async function fromSafetensors(
  file: safetensors.File,
  dtype: np.DType,
  config: WhisperConfig,
): Promise<WhisperModel> {
  const hydrated: Record<string, np.Array> = {};
  for (const [key, tensor] of Object.entries(file.tensors)) {
    hydrated[mapper.mapKey(key)] = tensorToArray(tensor, dtype);
  }

  const model = safetensors.toNested(hydrated) as WhisperModel;
  model.encoder.embedPositions ??= {
    weight: sinusoidalPositions(
      config.maxSourcePositions,
      config.dModel,
      dtype,
    ),
  };
  normalizeConvWeight(model.encoder.conv1, 80);
  normalizeConvWeight(model.encoder.conv2, config.dModel);
  if (model.encoder.layers.length !== config.encoderLayers) {
    throw new Error(
      `Expected ${config.encoderLayers} encoder layers, found ${model.encoder.layers.length}`,
    );
  }
  if (model.decoder.layers.length !== config.decoderLayers) {
    throw new Error(
      `Expected ${config.decoderLayers} decoder layers, found ${model.decoder.layers.length}`,
    );
  }
  return blockUntilReady(model);
}

function normalizeConvWeight(conv: Conv1d, inputChannels: number) {
  const shape = conv.weight.shape;
  if (shape.length !== 3) return;
  if (shape[1] === inputChannels) return;
  if (shape[2] === inputChannels) {
    conv.weight = conv.weight.transpose([0, 2, 1]);
    return;
  }
  throw new Error(
    `Unexpected Whisper conv weight shape [${shape.join(", ")}] for ${inputChannels} input channels`,
  );
}

function sinusoidalPositions(
  length: number,
  channels: number,
  dtype: np.DType,
): np.Array {
  const half = Math.floor(channels / 2);
  const data = new Float32Array(length * channels);
  const logStep = Math.log(10_000) / Math.max(half - 1, 1);
  for (let pos = 0; pos < length; pos++) {
    for (let i = 0; i < half; i++) {
      const scaled = pos * Math.exp(-logStep * i);
      data[pos * channels + i] = Math.sin(scaled);
      data[pos * channels + half + i] = Math.cos(scaled);
    }
  }
  const arr = np.array(data, { shape: [length, channels], dtype: np.float32 });
  return dtype === np.float32 ? arr : arr.astype(dtype);
}
