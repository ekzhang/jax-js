import { lax, numpy as np } from "@jax-js/jax";
import { safetensors, WeightMapper } from "@jax-js/loaders";

// Kyutai Pocket TTS model weights interfaces and forward pass.

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

export type TimestepEmbedder = {
  mlp: [Linear, undefined, Linear, { alpha: np.Array }]; // [Linear, SiLU, Linear, RMSNorm]
  freqs: np.Array; // [128], half of freq embedding size?
};

export type ResBlock = {
  inLN: LayerNorm; // eps=1e-6
  mlp: [Linear, undefined, Linear]; // [Linear, SiLU, Linear]
  adaLNModulation: [undefined, Linear]; // [SiLU, Linear]
};

export type MimiModel = {
  encoder: SEANetEncoder;
  decoder: SEANetDecoder;
  encoderTransformer: StreamingTransformerLayer[];
  decoderTransformer: StreamingTransformerLayer[];
  quantizer: {
    outputProj: Conv1d; // DummyQuantizer
  };
  downsample: StreamingConv1d;
  upsample: StreamingConvTranspose1d;
};

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

export type SEANetResnetBlock = {
  // Alternating [ELU, Conv1d, ELU, Conv1d], with residual at the end
  block: (StreamingConv1d | undefined)[];
};

export type StreamingTransformerLayer = {
  selfAttn: MimiStreamingMultiheadAttention;
  norm1: LayerNorm; // eps=1e-5
  norm2: LayerNorm; // eps=1e-5
  linear1: Linear; // 1024->4096, no bias
  linear2: Linear; // 4096->1024, no bias
  layerScale1?: np.Array; // shape [1024], just multiplicative if present
  layerScale2?: np.Array; // shape [1024], just multiplicative if present
};

export type MimiStreamingMultiheadAttention = {
  outProj: Linear; // no bias
  inProj: Linear; // no bias
};

export type Linear = {
  weight: np.Array; // [out, in]
  bias?: np.Array; // [out]
};

export function runLinear({ weight, bias }: Linear, x: np.Array): np.Array {
  x = np.vecdot(x, weight);
  if (bias) x = x.add(bias);
  return x;
}

export type LayerNorm = {
  // LayerNorm with `elementwise_affine`, i.e. has weight and bias
  weight: np.Array;
  bias: np.Array;
};

export function runLayerNorm(
  { weight, bias }: Partial<LayerNorm> = {},
  x: np.Array,
  eps: number = 1e-5,
) {
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
  return x;
}

export type Conv1d = {
  weight: np.Array; // [C_out, C_in, kernel_size]
  bias?: np.Array; // [C_out]
};

export function runConv1d(
  { weight, bias }: Conv1d,
  x: np.Array,
  stride: number = 1,
): np.Array {
  // x: [C_in, T_in]
  const y = lax.conv(x, weight, [stride], "SAME");
  if (bias) return y.add(bias);
  return y;
}

export type ConvTranspose1d = {
  weight: np.Array; // [C_in, C_out, kernel_size]
  bias?: np.Array; // [C_out]
};

export function runConvTranspose1d(
  { weight, bias }: ConvTranspose1d,
  x: np.Array,
  stride: number = 1,
): np.Array {
  // x: [C_in, T_in]
  const y = lax.convTranspose(x, weight, [stride], "SAME", {
    transposeKernel: true,
  });
  if (bias) return y.add(bias);
  return y;
}

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
    ".transformer.layers.": ".transformer.",
  },
  autoCamelCase: true,
});

export function fromSafetensors(file: safetensors.File): PocketTTS {
  const mappedWeights = weightMapper.mapObject(file.tensors);
  const hydrated: Record<string, np.Array> = {};
  for (const [key, value] of Object.entries(mappedWeights)) {
    // console.log(key, value);
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
