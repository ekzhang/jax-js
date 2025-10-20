import { nn, numpy as np } from "@jax-js/jax";

// Interfaces for model weights.

export interface MobileCLIPTextEncoder {
  tokenEmbedding: np.Array;
  positionalEmbedding: np.Array;
  transformer: MobileCLIPTextBlock[];
  lnFinal: LayerNorm;
  textProjection: np.Array;
}

export interface MobileCLIPTextBlock {
  ln1: LayerNorm;
  attn: MultiHeadAttention;
  ln2: LayerNorm;
  mlpUp: Linear;
  mlpDown: Linear;
}

export function runMobileCLIPTextBlock(
  { ln1, attn, ln2, mlpUp, mlpDown }: MobileCLIPTextBlock,
  x: np.Array,
  numHeads: number,
): np.Array {
  // Pre-norm attention block
  const normed1 = runLayerNorm(ln1, x.ref);
  const attnOut = runMultiHeadAttention(attn, normed1, numHeads);
  x = x.add(attnOut); // Residual connection

  // Pre-norm MLP block
  const normed2 = runLayerNorm(ln2, x.ref);
  let mlpOut = runLinear(mlpUp, normed2);
  mlpOut = nn.gelu(mlpOut);
  mlpOut = runLinear(mlpDown, mlpOut);
  x = x.add(mlpOut); // Residual connection

  return x;
}

export interface MultiHeadAttention {
  qkvProj: Linear;
  outProj: Linear;
}

export function runMultiHeadAttention(
  { qkvProj, outProj }: MultiHeadAttention,
  x: np.Array,
  numHeads: number,
): np.Array {
  // x shape: [seqLen, embed]
  const [seqLen, embed] = x.shape;
  const headDim = embed / numHeads;

  // Project to Q, K, V
  const qkv = runLinear(qkvProj, x); // [seqLen, 3 * embed]

  // Split into Q, K, V by slicing along the last dimension
  const q_ = qkv.ref.slice([], [0, embed]);
  const k_ = qkv.ref.slice([], [embed, 2 * embed]);
  const v_ = qkv.ref.slice([], [2 * embed, 3 * embed]);

  // Reshape for multi-head attention: [seqLen, numHeads, headDim]
  const q = q_.reshape([seqLen, numHeads, headDim]);
  const k = k_.reshape([seqLen, numHeads, headDim]);
  const v = v_.reshape([seqLen, numHeads, headDim]);

  // Transpose for attention: [numHeads, seqLen, headDim]
  const qT = q.transpose([1, 0, 2]);
  const kT = k.transpose([1, 0, 2]);
  const vT = v.transpose([1, 0, 2]);

  // Compute attention scores: Q @ K^T / sqrt(headDim)
  const scores = np.matmul(qT, kT.transpose([0, 2, 1])); // [numHeads, seqLen, seqLen]
  const scaledScores = scores.mul(1 / Math.sqrt(headDim));

  // Apply softmax
  const attnWeights = nn.softmax(scaledScores, -1);

  // Apply attention to values
  const attnOutput = np.matmul(attnWeights, vT); // [numHeads, seqLen, headDim]

  // Transpose back and reshape: [seqLen, numHeads, headDim] -> [seqLen, embed]
  const output = attnOutput.transpose([1, 0, 2]).reshape([seqLen, embed]);

  // Final projection
  return runLinear(outProj, output);
}

export interface Linear {
  weight: np.Array; // [Out, In]
  bias: np.Array; // [Out]
}

export function runLinear({ weight, bias }: Linear, x: np.Array): np.Array {
  return x.mul(weight.transpose()).add(bias);
}

export interface LayerNorm {
  weight: np.Array;
  bias: np.Array;
}

export function runLayerNorm(
  { weight, bias }: LayerNorm,
  x: np.Array,
): np.Array {
  // Normalize with respect to the last dimension of x.
  const dimSize = x.shape[x.ndim - 1];
  const avg = x.ref.mean(-1, { keepDims: true });
  x = x.sub(avg);
  const denom = np
    .sqrt(
      np
        .square(x.ref)
        .mul(1 / dimSize)
        .sum(-1, { keepDims: true }),
    )
    .add(np.scalar(1e-5, { dtype: x.dtype, device: x.device }));
  return x.div(denom).mul(weight).add(bias);
}
