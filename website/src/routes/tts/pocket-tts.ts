import { jit, lax, nn, numpy as np, random, tree } from "@jax-js/jax";
import { safetensors, WeightMapper } from "@jax-js/loaders";

// Kyutai Pocket TTS model weights interfaces and forward pass.

export type KVCache = {
  key: np.Array; // [T_cache, H, D]
  value: np.Array; // [T_cache, H, D]
};

export function emptyKVCache(): KVCache {
  return {
    key: np.zeros([0], { dtype: np.float16 }),
    value: np.zeros([0], { dtype: np.float16 }),
  };
}

export type PocketTTS = {
  flowLM: FlowLMModel;
  mimi: MimiModel;
};

export type FlowLMModel = {
  bosEmb: np.Array; // embedding for NaN values in latent (BOS position)
  conditionerEmbed: np.Array; // sentencepiece token embeds, [vocab_size=4001, embed_dim=1024]
  embMean: np.Array; // multiply latent by this during decode
  embStd: np.Array; // multiply latent by this during decode
  flowNet: SimpleMLPAdaLN;
  inputLinear: Linear;
  outNorm: LayerNorm;
  outEos: Linear;
  speakerProjWeight: np.Array; // 512->1024 for speaker audio latents -> conditioning
  transformer: StreamingTransformerLayer[];
};

export type FlowLMState = {
  kvCaches: KVCache[];
  kvCacheLen: number; // position offset
};

export function createFlowLMState(model: FlowLMModel): FlowLMState {
  return {
    kvCaches: model.transformer.map(() => emptyKVCache()),
    kvCacheLen: 0,
  };
}

export function runFlowLMStep(
  {
    bosEmb,
    conditionerEmbed,
    embMean,
    embStd,
    flowNet,
    inputLinear,
    outNorm,
    outEos,
    speakerProjWeight,
    transformer,
  }: FlowLMModel,
  { kvCaches, kvCacheLen }: FlowLMState,
  key: np.Array, // random key
  sequence: np.Array, // [S, ldim] - latent sequence, NaN for BOS
  embeds: np.Array | null, // [T, dim] - conditioning, text and voice
  offset: number, // position offset
  lsdDecodeSteps: number = 1,
  temperature: number = 0.7,
  noiseClamp: number | null = null,
  eosThreshold: number = -4.0,
): { latent: np.Array; isEos: np.Array; state: FlowLMState } {
  // unused fields
  bosEmb.dispose();
  conditionerEmbed.dispose();
  embMean.dispose();
  embStd.dispose();
  speakerProjWeight.dispose();

  const ldim = bosEmb.shape[0];

  // Project input from 32 -> 1024
  let input = runLinear(inputLinear, sequence);

  // Concatenate text/voice embeddings with input
  if (embeds !== null) input = np.concatenate([embeds, input], 0);

  for (let i = 0; i < transformer.length; i++) {
    // If kv cache is not large enough, expand it to next multiple of 64.
    if (kvCacheLen > 0 && kvCaches[i].key.shape[0] === kvCacheLen) {
      const newCapacity = Math.ceil((kvCacheLen + 1) / 64) * 64;
      kvCaches[i].key = np.pad(kvCaches[i].key, {
        0: [0, newCapacity - kvCacheLen],
      });
      kvCaches[i].value = np.pad(kvCaches[i].value, {
        0: [0, newCapacity - kvCacheLen],
      });
    }
    const layer = transformer[i];
    [input, kvCaches[i]] = runStreamingTransformerLayer(
      layer,
      kvCaches[i],
      input,
      offset,
      kvCacheLen,
      { numHeads: 16 },
    );
  }
  kvCacheLen += input.shape[0];

  let transformerOut = runLayerNorm(outNorm, input);

  // Get last position output (for next token prediction)
  transformerOut = transformerOut.slice([-1]); // [1, dim]

  // Check EOS
  const eosLogit = runLinear(outEos, transformerOut.ref);
  const isEos = np.greater(eosLogit, eosThreshold); // [1, 1]

  const noiseShape = [1, ldim]; // [T, ldim] with T=1
  const std = Math.sqrt(temperature);
  let noise = random.normal(key, noiseShape).mul(std);
  if (noiseClamp !== null) {
    // Truncated normal - clamp to [-noiseClamp, noiseClamp]
    noise = np.clip(noise, -noiseClamp, noiseClamp);
  }

  // Decode using LSD
  const conditionedFlow = (s: np.Array, t: np.Array, x: np.Array) =>
    runSimpleMLPAdaLN(tree.ref(flowNet), transformerOut.ref, s, t, x);
  const latent = lsdDecode(conditionedFlow, noise, lsdDecodeSteps);
  tree.dispose([flowNet, transformerOut]);

  return { latent, isEos, state: { kvCaches, kvCacheLen } };
}

export type SimpleMLPAdaLN = {
  timeEmbed: TimestepEmbedder[]; // [num_time_conds=2]
  condEmbed: Linear;
  inputProj: Linear;
  resBlocks: ResBlock[]; // [num_res_blocks=6]
  finalLayer: {
    // layernorm without elementwise_affine, eps=1e-6
    linear: Linear;
    adaLNModulation: [undefined, Linear]; // [SiLU, Linear]
  };
};

export const runSimpleMLPAdaLN = jit(function runSimpleMLPAdaLN(
  { timeEmbed, condEmbed, inputProj, resBlocks, finalLayer }: SimpleMLPAdaLN,
  c: np.Array, // conditioning from AR transformer
  s: np.Array, // start time tensor
  t: np.Array, // target time tensor
  x: np.Array, // input [N, C]
): np.Array {
  x = runLinear(inputProj, x);

  // Combine time conditions (average of s and t embeddings)
  const sEmb = runTimestepEmbedder(timeEmbed[0], s);
  const tEmb = runTimestepEmbedder(timeEmbed[1], t);
  const tCombined = sEmb.add(tEmb).div(2);

  // Embed condition and combine with time
  const cEmb = runLinear(condEmbed, c);
  const y = tCombined.add(cEmb);

  // Apply residual blocks
  for (const block of resBlocks) {
    x = runResBlock(block, x, y.ref);
  }

  // Final layer: LayerNorm (no affine) + AdaLN modulation + Linear
  const [, finalAdaLNLinear] = finalLayer.adaLNModulation;
  const finalMod = runLinear(finalAdaLNLinear, nn.silu(y));
  const [shift, scale] = np.split(finalMod, 2, -1);

  x = runLayerNorm({}, x, 1e-6); // LayerNorm without affine
  x = modulate(x, shift, scale);
  x = runLinear(finalLayer.linear, x);

  return x;
});

export function runRope(
  q: np.Array, // [T, H, D]
  k: np.Array, // [T, H, D]
  offset: np.Array,
  maxPeriod: number = 10000,
): [np.Array, np.Array] {
  const [T, H, D] = q.shape;
  const halfD = D / 2;

  // Compute frequency basis
  const ds = np.arange(halfD, undefined, undefined, { dtype: np.float32 });
  const freqs = np.exp(ds.mul((-Math.log(maxPeriod) * 2) / D));

  // Position indices
  const ts = np.arange(T).add(offset).astype(np.float32).reshape([T, 1, 1]);

  // Reshape q and k to separate real and imaginary parts
  const qReshaped = q.reshape([T, H, halfD, 2]);
  const kReshaped = k.reshape([T, H, halfD, 2]);

  // Get real and imaginary components via slicing
  let [qr, qi] = np.split(qReshaped, 2, -1);
  let [kr, ki] = np.split(kReshaped, 2, -1);
  qr = np.squeeze(qr, -1);
  qi = np.squeeze(qi, -1);
  kr = np.squeeze(kr, -1);
  ki = np.squeeze(ki, -1);

  const angles = freqs.mul(ts); // [T, 1, D/2] broadcast
  const rotr = np.cos(angles.ref).astype(qr.dtype);
  const roti = np.sin(angles).astype(qr.dtype);

  // Apply rotation (complex multiplication)
  const qor = qr.ref.mul(rotr.ref).sub(qi.ref.mul(roti.ref));
  const qoi = qr.mul(roti.ref).add(qi.mul(rotr.ref));
  const kor = kr.ref.mul(rotr.ref).sub(ki.ref.mul(roti.ref));
  const koi = kr.mul(roti).add(ki.mul(rotr));

  // Stack and reshape back
  const qo = np.stack([qor, qoi], -1).reshape([T, H, D]);
  const ko = np.stack([kor, koi], -1).reshape([T, H, D]);

  return [qo, ko];
}

export type MimiStreamingMultiheadAttention = {
  outProj: Linear; // no bias
  inProj: Linear; // no bias
};

export function runMimiStreamingMultiheadAttention(
  { inProj, outProj }: MimiStreamingMultiheadAttention,
  kvCache: KVCache,
  query: np.Array, // [T, embed_dim]
  offset: np.Array, // scalar, position offset
  kvCacheLen: np.Array, // scalar, length of kvCache
  context: number,
  numHeads: number,
  maxPeriod: number = 10000,
): [np.Array, KVCache] {
  const [T, embedDim] = query.shape;
  const headDim = embedDim / numHeads;

  const projected = runLinear(inProj, query); // [T, 3 * embed_dim]
  const qkv = projected.reshape([T, 3 * numHeads, headDim]);
  const [q_, k_, v] = np.split(qkv, 3, 1); // each is [T, H, D]
  const [q, k] = runRope(q_, k_, offset, maxPeriod);

  const isPrefill = kvCache.key.size === 0; // Empty kv cache = prefill
  let x: np.Array;
  if (isPrefill) {
    tree.dispose([kvCache, kvCacheLen]);
    x = nn.dotProductAttention(q, k.ref, v.ref, {
      isCausal: true,
      localWindowSize: context ? [context - 1, 0] : undefined,
    });
    kvCache = { key: k, value: v };
  } else {
    // Decode step
    // Update kvCache with new k,v
    const capacity = kvCache.key.shape[0];
    const cacheMask = np
      .arange(capacity)
      .reshape([-1, 1, 1])
      .less(kvCacheLen.ref);
    kvCache.key = np.where(
      cacheMask.ref,
      kvCache.key,
      np.tile(k, [capacity / T, 1, 1]), // Hack: Assume T divides into kv cache length
    );
    kvCache.value = np.where(
      cacheMask,
      kvCache.value,
      np.tile(v, [capacity / T, 1, 1]), // Hack: Assume T divides into kv cache length
    );
    // Casual attention mask offset by kvCacheLen
    const maskDelta = np
      .arange(capacity)
      .sub(np.arange(T).reshape([T, 1]))
      .sub(kvCacheLen); // [T, capacity]
    const mask = context
      ? maskDelta.ref.lessEqual(0).mul(maskDelta.greater(-context))
      : maskDelta.lessEqual(0);
    x = nn.dotProductAttention(q, kvCache.key.ref, kvCache.value.ref, { mask });
  }
  x = x.reshape([T, embedDim]);
  x = runLinear(outProj, x);
  return [x, kvCache];
}

export type StreamingTransformerLayer = {
  selfAttn: MimiStreamingMultiheadAttention;
  norm1: LayerNorm; // eps=1e-5
  norm2: LayerNorm; // eps=1e-5
  linear1: Linear; // 1024->4096, no bias
  linear2: Linear; // 4096->1024, no bias
  layerScale1?: np.Array; // shape [1024], just multiplicative if present
  layerScale2?: np.Array; // shape [1024], just multiplicative if present
};

export const runStreamingTransformerLayer = jit(
  function runStreamingTransformerLayer(
    {
      selfAttn,
      norm1,
      norm2,
      linear1,
      linear2,
      layerScale1,
      layerScale2,
    }: StreamingTransformerLayer,
    kvCache: KVCache,
    x: np.Array, // [T, D]
    offset: np.Array, // scalar, position offset of x
    kvCacheLen: np.Array, // scalar, length of kvCache
    {
      context = 0, // infinite context
      numHeads,
      maxPeriod = 10000,
    }: { context?: number; numHeads: number; maxPeriod?: number },
  ): [np.Array, KVCache] {
    // Self-attention block with pre-norm
    const xOrig = x.ref;
    x = runLayerNorm(norm1, x);
    let update: np.Array;
    [update, kvCache] = runMimiStreamingMultiheadAttention(
      selfAttn,
      kvCache,
      x,
      offset,
      kvCacheLen,
      context,
      numHeads,
      maxPeriod,
    );
    if (layerScale1) {
      update = update.mul(layerScale1);
    }
    x = xOrig.add(update);

    // FFN block with pre-norm
    const xOrig2 = x.ref;
    x = runLayerNorm(norm2, x);
    let ffnOut = runLinear(linear1, x);
    ffnOut = nn.gelu(ffnOut, { approximate: false });
    ffnOut = runLinear(linear2, ffnOut);
    if (layerScale2) {
      ffnOut = ffnOut.mul(layerScale2);
    }
    x = xOrig2.add(ffnOut);

    return [x, kvCache];
  },
  { staticArgnums: [5, 6, 7] },
);

export type SEANetResnetBlock = {
  // Alternating [ELU, Conv1d, ELU, Conv1d], with residual at the end
  block: [undefined, StreamingConv1d, undefined, StreamingConv1d];
};

export function runSEANetResnetBlock(
  { block }: SEANetResnetBlock,
  states: (np.Array | null)[],
  x: np.Array, // [1, C, T]
): [np.Array, np.Array[]] {
  let v = x.ref;
  let stateIdx = 0;
  for (const layer of block) {
    if (layer === undefined) {
      // ELU activation
      v = nn.elu(v);
    } else {
      // StreamingConv1d
      [v, states[stateIdx]] = runConv1d(layer.conv, states[stateIdx], v);
      stateIdx++;
    }
  }
  // Residual connection
  return [x.add(v), states as np.Array[]];
}

export type SEANetEncoder = {
  model: [
    StreamingConv1d,

    // ratio=6
    SEANetResnetBlock,
    undefined, // ELU
    StreamingConv1d,

    // ratio=5
    SEANetResnetBlock,
    undefined, // ELU
    StreamingConv1d,

    // ratio=4
    SEANetResnetBlock,
    undefined, // ELU
    StreamingConv1d,

    // final two layers with indices 10, 11
    undefined, // ELU
    StreamingConv1d,
  ];
};

export function runSEANetEncoder(
  { model }: SEANetEncoder,
  x: np.Array, // [C, T] - audio waveform
): np.Array {
  // Process through model layers with appropriate strides
  // model structure: [Conv1d, (ResBlock, ELU, Conv1d) * 3, ELU, Conv1d]
  const ratios = [4, 5, 6]; // reversed from decoder [6, 5, 4]

  // Initial conv (index 0)
  x = np.expandDims(x, 0); // [1, C, T]
  [x] = runConv1d(model[0].conv, null, x);

  // Encoder blocks (ratio=4, ratio=5, ratio=6)
  let idx = 1;
  for (let i = 0; i < 3; i++) {
    // ResBlock
    let states: any = [null, null];
    [x, states] = runSEANetResnetBlock(
      model[idx] as SEANetResnetBlock,
      states,
      x,
    );
    tree.dispose(states);
    idx++;
    // ELU
    x = nn.elu(x);
    idx++;
    // Strided Conv (downsampling)
    const stride = ratios[i];
    [x] = runConv1d((model[idx] as StreamingConv1d).conv, null, x, stride);
    idx++;
  }

  // Final ELU + Conv
  x = nn.elu(x);
  [x] = runConv1d((model[11] as StreamingConv1d).conv, null, x);

  return x.slice(0);
}

export type SEANetDecoder = {
  model: [
    StreamingConv1d,

    // ratio=6
    undefined, // ELU
    StreamingConvTranspose1d,
    SEANetResnetBlock,

    // ratio=5
    undefined, // ELU
    StreamingConvTranspose1d,
    SEANetResnetBlock,

    // ratio=4
    undefined, // ELU
    StreamingConvTranspose1d,
    SEANetResnetBlock,

    // final two layers with indices 10, 11
    undefined, // ELU
    StreamingConv1d,
  ];
};

export function createSEANetDecoderState({
  model,
}: SEANetDecoder): SEANetDecoderState {
  return {
    conv1: createConv1dState(model[0].conv),
    blocks: [
      {
        convtr: createConvTranspose1dState(model[2].convtr, 6),
        res: [
          createConv1dState(model[3].block[1].conv),
          createConv1dState(model[3].block[3].conv),
        ],
      },
      {
        convtr: createConvTranspose1dState(model[5].convtr, 5),
        res: [
          createConv1dState(model[6].block[1].conv),
          createConv1dState(model[6].block[3].conv),
        ],
      },
      {
        convtr: createConvTranspose1dState(model[8].convtr, 4),
        res: [
          createConv1dState(model[9].block[1].conv),
          createConv1dState(model[9].block[3].conv),
        ],
      },
    ],
    conv2: createConv1dState(model[11].conv),
  };
}

export const runSEANetDecoder = jit(function runSEANetDecoder(
  { model }: SEANetDecoder,
  state: SEANetDecoderState,
  x: np.Array, // [C, T] - encoded representation
): [np.Array, SEANetDecoderState] {
  // Process through model layers with appropriate strides
  // model structure: [Conv1d, (ELU, ConvTr, ResBlock) * 3, ELU, Conv1d]
  const ratios = [6, 5, 4]; // upsampling ratios

  // Initial conv (index 0)
  x = np.expandDims(x, 0); // [1, C, T]
  [x, state.conv1] = runConv1d(model[0].conv, state.conv1, x);

  // Decoder blocks
  let idx = 1;
  for (let i = 0; i < 3; i++) {
    const blockState = state.blocks[i];
    // ELU
    x = nn.elu(x);
    idx++;
    // Transposed Conv (upsampling)
    const stride = ratios[i];
    [x, blockState.convtr] = runConvTranspose1d(
      (model[idx] as StreamingConvTranspose1d).convtr,
      blockState.convtr,
      x,
      stride,
    );
    idx++;
    // ResBlock
    [x, blockState.res] = runSEANetResnetBlock(
      model[idx] as SEANetResnetBlock,
      blockState.res,
      x,
    );
    idx++;
  }

  // Final ELU + Conv
  x = nn.elu(x);
  [x, state.conv2] = runConv1d(model[11].conv, state.conv2, x);

  return [x.slice(0), state];
});

export type MimiModel = {
  encoder: SEANetEncoder;
  decoder: SEANetDecoder;
  encoderTransformer: StreamingTransformerLayer[];
  decoderTransformer: StreamingTransformerLayer[];
  quantizer: {
    outputProj: { weight: np.Array }; // DummyQuantizer, plain conv1d [512, 32, 1], kernel size 1
  };
  downsample: StreamingConv1d;
  upsample: StreamingConvTranspose1d; // note: depthwise
};

export function runMimiEncode(
  {
    encoder,
    encoderTransformer,
    decoder,
    decoderTransformer,
    quantizer,
    downsample,
    upsample,
  }: MimiModel,
  x: np.Array, // [C, T] - audio waveform at 24kHz
): np.Array {
  tree.dispose([decoder, decoderTransformer, quantizer, upsample]);
  x = runSEANetEncoder(encoder, x);

  // Encoder transformer (with transpose for [T, D] format)
  x = x.transpose([1, 0]); // [C, T] -> [T, C]
  const offset = np.array(0, { dtype: np.int32, device: x.device });
  for (const layer of encoderTransformer) {
    let kvCache = emptyKVCache();
    [x, kvCache] = runStreamingTransformerLayer(
      layer,
      kvCache,
      x,
      offset.ref,
      0,
      { context: 250, numHeads: 8 },
    );
    tree.dispose(kvCache);
  }
  offset.dispose();
  x = x.transpose([1, 0]); // back to [C, T]

  // Downsample (stride 16)
  [x] = runConv1d(downsample.conv, null, x, 16);
  return x;
}

export type MimiDecodeState = {
  kvCaches: KVCache[];
  kvCacheLen: number;
  initialConvState: np.Array;
  seanetStates: SEANetDecoderState;
};

export type SEANetDecoderState = {
  conv1: np.Array;
  blocks: {
    convtr: np.Array;
    res: np.Array[];
  }[];
  conv2: np.Array;
};

export function createMimiDecodeState(mimi: MimiModel): MimiDecodeState {
  return {
    kvCaches: mimi.decoderTransformer.map(() => emptyKVCache()),
    kvCacheLen: 0,
    initialConvState: createConvTranspose1dState(mimi.upsample.convtr, 16, 512),
    seanetStates: createSEANetDecoderState(mimi.decoder),
  };
}

export function runMimiDecode(
  {
    encoder,
    encoderTransformer,
    decoder,
    decoderTransformer,
    quantizer,
    downsample,
    upsample,
  }: MimiModel,
  { kvCaches, kvCacheLen, initialConvState, seanetStates }: MimiDecodeState,
  latent: np.Array, // [T, 32] - bottleneck representation
  offset: number, // scalar, position offset
): [np.Array, MimiDecodeState] {
  tree.dispose([encoder, encoderTransformer, downsample]);

  // Run through "dummy quantizer"
  latent = np.expandDims(latent.transpose([1, 0]), 0); // [1, 32, T]
  latent = lax.conv(latent, quantizer.outputProj.weight, [1], "VALID"); // [1, 512, T]

  // Upsample (stride 16), depthwise
  let x: np.Array;
  [x, initialConvState] = runConvTranspose1d(
    upsample.convtr,
    initialConvState,
    latent,
    16,
    latent.shape[1],
  ); // [1, 512, 16*T]
  x = x.slice(0);

  // Decoder transformer
  x = x.transpose([1, 0]); // [C, 16*T] -> [16*T, C]
  for (let i = 0; i < decoderTransformer.length; i++) {
    const layer = decoderTransformer[i];
    [x, kvCaches[i]] = runStreamingTransformerLayer(
      layer,
      kvCaches[i],
      x,
      offset * 16,
      kvCacheLen,
      { context: 250, numHeads: 8 },
    );
  }
  x = x.transpose([1, 0]); // [C, 16*T]

  // Decode through SEANet decoder
  [x, seanetStates] = runSEANetDecoder(decoder, seanetStates, x); // [1, 1920*T]

  // Maintain and update KV caches as needed.
  kvCacheLen += 16;
  if (kvCaches[0].key.shape[0] !== 272) {
    // Pad it to a constant [272] in length, more than 250 context + 16 for next pass.
    const padAmount = 272 - kvCaches[0].key.shape[0];
    for (const c of kvCaches) {
      c.key = np.pad(c.key, { 0: [0, padAmount] });
      c.value = np.pad(c.value, { 0: [0, padAmount] });
    }
  }
  if (kvCacheLen === 272) {
    // Cycle room for one more kv cache entry.
    kvCacheLen -= 16;
    for (const c of kvCaches) {
      c.key = np.pad(c.key.slice([16]), { 0: [0, 16] });
      c.value = np.pad(c.value.slice([16]), { 0: [0, 16] });
    }
  }

  return [
    x,
    {
      kvCaches,
      kvCacheLen: kvCacheLen,
      initialConvState,
      seanetStates,
    },
  ];
}

export function lsdDecode(
  flowNet: (s: np.Array, t: np.Array, x: np.Array) => np.Array,
  x0: np.Array,
  numSteps: number = 1,
): np.Array {
  // Lagrangian Self Distillation decoding
  // Rebuilds the data sample from starting point x0
  let current = x0;
  for (let i = 0; i < numSteps; i++) {
    const s = i / numSteps;
    const t = (i + 1) / numSteps;
    const sArr = np.full(x0.shape.slice(0, -1).concat([1]), s);
    const tArr = np.full(x0.shape.slice(0, -1).concat([1]), t);
    const flowDir = flowNet(sArr, tArr, current.ref);
    current = current.add(flowDir.div(numSteps));
  }
  return current;
}

export type TimestepEmbedder = {
  mlp: [Linear, undefined, Linear, RMSNorm]; // [Linear, SiLU, Linear, RMSNorm]
  freqs: np.Array; // [128], half of freq embedding size
};

export function runTimestepEmbedder(
  { mlp, freqs }: TimestepEmbedder,
  t: np.Array,
): np.Array {
  // t: scalar or [N] tensor of timesteps
  // freqs: [128] precomputed frequency basis
  // mlp: [Linear, SiLU, Linear, RMSNorm]
  const [linear1, , linear2, rmsNorm] = mlp;
  const args = t.mul(freqs); // [N, 128] or [128]
  const embedding = np.concatenate([np.cos(args.ref), np.sin(args)], -1); // [N, 256]
  let x = runLinear(linear1, embedding);
  x = nn.silu(x);
  x = runLinear(linear2, x);
  x = runRMSNorm(rmsNorm, x);
  return x;
}

function modulate(x: np.Array, shift: np.Array, scale: np.Array): np.Array {
  // x * (1 + scale) + shift
  return x.mul(scale.add(1)).add(shift);
}

export type ResBlock = {
  inLN: LayerNorm; // eps=1e-6
  mlp: [Linear, undefined, Linear]; // [Linear, SiLU, Linear]
  adaLNModulation: [undefined, Linear]; // [SiLU, Linear]
};

export function runResBlock(
  { inLN, mlp, adaLNModulation }: ResBlock,
  x: np.Array,
  y: np.Array,
): np.Array {
  // y is the combined time + condition embedding
  // AdaLN modulation: [SiLU, Linear] -> 3 * channels for shift, scale, gate
  const [, adaLNLinear] = adaLNModulation;
  const modulation = runLinear(adaLNLinear, nn.silu(y));
  const [shiftMlp, scaleMlp, gateMlp] = np.split(modulation, 3, -1);

  // Apply AdaLN then MLP
  let h = runLayerNorm(inLN, x.ref, 1e-6);
  h = modulate(h, shiftMlp, scaleMlp);

  // MLP: [Linear, SiLU, Linear]
  const [mlpLinear1, , mlpLinear2] = mlp;
  h = runLinear(mlpLinear1, h);
  h = nn.silu(h);
  h = runLinear(mlpLinear2, h);

  // Residual with gate
  return x.add(gateMlp.mul(h));
}

export type Linear = {
  weight: np.Array; // [out, in]
  bias?: np.Array; // [out]
};

export function runLinear({ weight, bias }: Linear, x: np.Array): np.Array {
  x = np.dot(x, weight.transpose());
  if (bias) x = x.add(bias);
  return x;
}

export type LayerNorm = {
  // LayerNorm with `elementwise_affine`, i.e. has weight and bias
  weight: np.Array;
  bias: np.Array;
};

export const runLayerNorm = jit(
  function runLayerNorm(
    { weight, bias }: Partial<LayerNorm> = {},
    x: np.Array,
    eps: number = 1e-5,
  ) {
    const dtype = x.dtype;
    x = x.astype(np.float32); // LayerNorm in high precision to avoid numerics issues.
    const mean = x.ref.mean(-1, { keepdims: true });
    const var_ = np.var_(x.ref, -1, {
      mean: mean.ref,
      correction: 0,
      keepdims: true,
    });
    x = x.sub(mean).div(np.sqrt(var_.add(eps)));
    if (weight) {
      x = x.mul(weight).add(bias!);
    }
    return x.astype(dtype);
  },
  { staticArgnums: [2] },
);

export type RMSNorm = {
  alpha: np.Array; // [dim]
};

export function runRMSNorm(
  { alpha }: RMSNorm,
  x: np.Array,
  eps: number = 1e-5,
) {
  // RMSNorm: x * alpha / sqrt(var + eps)
  const dtype = x.dtype;
  x = x.astype(np.float32); // RMSNorm in high precision to avoid numerics issues.
  const var_ = np.var_(x.ref, -1, { correction: 0, keepdims: true });
  x = x.mul(alpha).div(np.sqrt(var_.add(eps)));
  return x.astype(dtype);
}

export type Conv1d = {
  weight: np.Array; // [C_out, C_in, kernel_size]
  bias?: np.Array; // [C_out]
};

export function createConv1dState(
  { weight }: Conv1d,
  stride: number = 1,
): np.Array {
  return np.zeros(
    [
      1, // batch size
      weight.shape[1], // in channels
      weight.shape[2] - stride, // kernel size - stride
    ],
    { dtype: np.float16 },
  );
}

export function runConv1d(
  { weight, bias }: Conv1d,
  state: np.Array | null,
  x: np.Array,
  stride: number = 1,
): [np.Array, np.Array] {
  // x: [1, C_in, T_in]
  state ??= createConv1dState({ weight }, stride);
  x = np.concatenate([state, x], 2); // pad with state
  state = x.ref.slice([], [], [x.shape[2] - state.shape[2]]);
  let y = lax.conv(x, weight, [stride], "VALID");
  if (bias) y = y.add(np.expandDims(bias, -1));
  return [y, state];
}

export type ConvTranspose1d = {
  weight: np.Array; // [C_in, C_out, kernel_size]
  bias?: np.Array; // [C_out]
};

export function createConvTranspose1dState(
  { weight }: ConvTranspose1d,
  stride: number = 1,
  groups: number = 1,
): np.Array {
  return np.zeros(
    [
      1, // batch size
      weight.shape[1] * groups, // out channels
      weight.shape[2] - stride, // kernel size - stride
    ],
    { dtype: np.float16 },
  );
}

export function runConvTranspose1d(
  { weight, bias }: ConvTranspose1d,
  state: np.Array | null,
  x: np.Array,
  stride: number = 1,
  groups: number = 1,
): [np.Array, np.Array] {
  state ??= createConvTranspose1dState({ weight }, stride);
  // Depthwise needs to flip spatial dims and flip C_in,C_out -> C_out,C_in.
  const [cIn, cOut, kernelSize] = weight.shape;
  weight = np.flip(weight, -1);
  if (groups > 1) {
    weight = weight
      .reshape([groups, cIn / groups, cOut, kernelSize])
      .transpose([0, 2, 1, 3])
      .reshape([cOut * groups, cIn / groups, kernelSize]);
  } else {
    weight = weight.transpose([1, 0, 2]);
  }

  let y = lax.convGeneralDilated(
    x, // x: [1, C_in, T_in]
    weight,
    [1],
    // To match padding, we need to pad left and right with (kernel_size-1).
    // This is different from JAX's `lax.convTranspose()`!
    [[kernelSize - 1, kernelSize - 1]],
    {
      lhsDilation: [stride],
      featureGroupCount: groups,
    },
  );
  y = y.add(np.pad(state, { 2: [0, y.shape[2] - state.shape[2]] }));
  [y, state] = np.split(y, [y.shape[2] - state.shape[2]], 2);
  if (bias) y = y.add(np.expandDims(bias, -1));
  return [y, state];
}

export type StreamingConv1d = {
  // TODO: Actually make this streaming, needs to init `kernel-stride` zeros
  // state and maintain this on each successive forward pass.
  conv: Conv1d;
};

export type StreamingConvTranspose1d = {
  // TODO: Actually make this streaming, needs to init `kernel-stride` zeros
  // state and maintain this on each successive forward pass.
  convtr: ConvTranspose1d;
};

const weightMapper = new WeightMapper({
  prefix: {
    "flow_lm.": "flowLM.",
    "mimi.decoder_transformer.transformer.layers": "mimi.decoderTransformer",
    "mimi.encoder_transformer.transformer.layers": "mimi.encoderTransformer",
  },
  suffix: {
    ".conditioner.embed.weight": ".conditionerEmbed",
    ".layer_scale_1.scale": ".layerScale1",
    ".layer_scale_2.scale": ".layerScale2",
  },
  substring: {
    ".conv.conv.": ".conv.",
    ".convtr.convtr.": ".convtr.",
    ".in_ln.": ".inLN.",
    ".transformer.layers.": ".transformer.",
  },
  autoCamelCase: true,
});

export function fromSafetensors(file: safetensors.File): PocketTTS {
  const mappedWeights = weightMapper.mapObject(file.tensors);
  const hydrated: Record<string, np.Array> = {};
  for (const [key, value] of Object.entries(mappedWeights)) {
    if (value.dtype === "F16") {
      hydrated[key] = np.array(value.data as Float16Array<ArrayBuffer>, {
        dtype: np.float16,
        shape: value.shape,
      });
    } else {
      throw new Error(`Unexpected dtype ${value.dtype} for weight ${key}`);
    }
  }
  return safetensors.toNested(hydrated);
}
