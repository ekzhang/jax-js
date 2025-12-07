import { nn, numpy as np } from "@jax-js/jax";
import { safetensors, WeightMapper } from "@jax-js/loaders";

// MobileCLIP2 model weights interfaces and forward pass.

export type MobileCLIP = {
  text: MobileCLIPTextEncoder;
  visual: any; // TODO
  logitScale: np.Array;
};

const weightMapper = new WeightMapper({
  exact: {
    logit_scale: "logitScale",
    "text.token_embedding.weight": "text.tokenEmbedding",
    "text.positional_embedding": "text.positionalEmbedding",
    "text.text_projection": "text.textProjection",
  },
  prefix: {
    "text.transformer.resblocks.": "text.transformer.",
  },
  substring: {
    ".ln_final.": ".lnFinal.",
    ".ln_1.": ".ln1.",
    ".ln_2.": ".ln2.",
    ".mlp.c_fc.": ".mlpUp.",
    ".mlp.c_proj.": ".mlpDown.",
    ".attn.in_proj_": ".attn.qkvProj.",
    ".attn.out_proj.": ".attn.outProj.",
  },
});

export function fromSafetensors(file: safetensors.File): MobileCLIP {
  const mappedWeights = weightMapper.mapObject(file.tensors);
  const hydrated: Record<string, np.Array> = {};
  for (const [key, value] of Object.entries(mappedWeights)) {
    console.log(key);
    console.log(value);
    if (value.dtype === "F16") {
      hydrated[key] = np
        .array(value.data as Float16Array<ArrayBuffer>, {
          dtype: np.float16,
          shape: value.shape,
        })
        .astype(np.float32);
    } else if (value.dtype === "I64") {
      // Ignored, these are metadata for BatchNorm.
      continue;
    } else {
      throw new Error(`Unexpected dtype ${value.dtype} for weight ${key}`);
    }
  }
  return safetensors.toNested(hydrated);
}

export type MobileCLIPTextEncoder = {
  tokenEmbedding: np.Array;
  positionalEmbedding: np.Array;
  transformer: MobileCLIPTextBlock[];
  lnFinal: LayerNorm;
  textProjection: np.Array;
};

export function runMobileCLIPTextEncoder(
  {
    tokenEmbedding,
    positionalEmbedding,
    transformer,
    lnFinal,
    textProjection,
  }: MobileCLIPTextEncoder,
  textTokens: np.Array,
): np.Array {
  // Embed tokens and add positional embeddings
  let x = tokenEmbedding.slice(textTokens.ref); // [L, D]
  x = x.add(positionalEmbedding);

  for (const block of transformer) {
    x = runMobileCLIPTextBlock(block, x);
  }
  x = runLayerNorm(lnFinal, x.ref);

  const finalFeatures = x.slice(np.argmax(textTokens, -1));
  return np.dot(textProjection.transpose(), finalFeatures); // [D_out]
}

export type MobileCLIPTextBlock = {
  ln1: LayerNorm;
  attn: MultiHeadAttention;
  ln2: LayerNorm;
  mlpUp: Linear;
  mlpDown: Linear;
};

export function runMobileCLIPTextBlock(
  { ln1, attn, ln2, mlpUp, mlpDown }: MobileCLIPTextBlock,
  x: np.Array,
): np.Array {
  // Pre-norm attention block
  const normed1 = runLayerNorm(ln1, x.ref);
  const attnOut = runMultiHeadAttention(attn, normed1);
  x = x.add(attnOut); // Residual connection

  // Pre-norm MLP block
  const normed2 = runLayerNorm(ln2, x.ref);
  let mlpOut = runLinear(mlpUp, normed2);
  mlpOut = nn.gelu(mlpOut);
  mlpOut = runLinear(mlpDown, mlpOut);
  x = x.add(mlpOut); // Residual connection

  return x;
}

export type MultiHeadAttention = {
  qkvProj: Linear;
  outProj: Linear;
};

export function runMultiHeadAttention(
  { qkvProj, outProj }: MultiHeadAttention,
  x: np.Array,
): np.Array {
  const numHeads = 8;

  // x shape: [seqLen, embed]
  const [seqLen, embed] = x.shape;
  const headDim = embed / numHeads;

  // Project to Q, K, V
  const qkv = runLinear(qkvProj, x); // [seqLen, 3 * embed]

  // Split into Q, K, V by slicing along the last dimension
  const q_ = qkv.ref.slice([], [0, embed]);
  const k_ = qkv.ref.slice([], [embed, 2 * embed]);
  const v_ = qkv.slice([], [2 * embed, 3 * embed]);

  // Reshape for multi-head attention: [numHeads, seqLen, headDim]
  const q = q_.reshape([seqLen, numHeads, headDim]).transpose([1, 0, 2]);
  const k = k_.reshape([seqLen, numHeads, headDim]).transpose([1, 0, 2]);
  const v = v_.reshape([seqLen, numHeads, headDim]).transpose([1, 0, 2]);

  // Compute attention scores: Q @ K^T / sqrt(headDim)
  const scores = np.matmul(q, k.transpose([0, 2, 1])); // [numHeads, seqLen, seqLen]
  const scaledScores = scores.mul(1 / Math.sqrt(headDim));

  // Apply softmax
  const attnWeights = nn.softmax(scaledScores, -1);

  // Apply attention to values
  const attnOutput = np.matmul(attnWeights, v); // [numHeads, seqLen, headDim]

  // Transpose back and reshape: [seqLen, numHeads, headDim] -> [seqLen, embed]
  const output = attnOutput.transpose([1, 0, 2]).reshape([seqLen, embed]);

  // Final projection
  return runLinear(outProj, output);
}

export type Linear = {
  weight: np.Array; // [Out, In]
  bias: np.Array; // [Out]
};

export function runLinear({ weight, bias }: Linear, x: np.Array): np.Array {
  return np.dot(x, weight.transpose()).add(bias);
}

export type LayerNorm = {
  weight: np.Array;
  bias: np.Array;
};

export function runLayerNorm(
  { weight, bias }: LayerNorm,
  x: np.Array,
): np.Array {
  // Normalize with respect to the last dimension of x.
  const dimSize = x.shape[x.ndim - 1];
  const avg = x.ref.mean(-1, { keepdims: true });
  x = x.sub(avg);
  const denom = np
    .sqrt(
      np
        .square(x.ref)
        .mul(1 / dimSize)
        .sum(-1, { keepdims: true }),
    )
    .add(1e-5);
  return x.div(denom).mul(weight).add(bias);
}
