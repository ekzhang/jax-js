import { numpy as np, tree } from "@jax-js/jax";
import { safetensors, tokenizers } from "@jax-js/loaders";

import {
  createGemmaState,
  fromSafetensors,
  GEMMA_CONFIG,
  type GemmaModel,
  type GemmaState,
  runGemmaPrefill,
  runGemmaStep,
} from "./gemma";
import { HuggingFaceBpeTokenizer } from "./huggingface-tokenizer";
import {
  createLfmState,
  lfmFromSafetensors,
  type LfmModel,
  type LfmState,
  runLfmPrefill,
  runLfmStep,
} from "./lfm";

// Gemma chat-template control tokens in tokenizer.model.
const START_OF_TURN_TOKEN = 105;
const END_OF_TURN_TOKEN = 106;

export type ChatMessage = {
  id: number;
  role: "user" | "assistant";
  content: string;
};

export type SamplingDefaults = {
  temperature: number;
  topK: number;
  topP: number;
  repetitionPenalty: number;
};

/** A tokenizer with model-specific output filtering hidden behind it. */
export type ChatTokenizer = {
  bosToken: number;
  eosToken: number;
  encode(text: string): number[];
  decode(tokens: number[]): string;
  decodeGenerated(tokens: number[]): string;
};

/** One stateful prefill/decode sequence. */
export type ChatModelSession = {
  prefill(tokenIds: np.Array): np.Array;
  step(token: number): np.Array;
  dispose(): void;
};

/** A checkpoint loaded onto the currently selected device. */
export type LoadedChatModel = {
  definition: ChatModel;
  createSession(): ChatModelSession;
  dispose(): void;
};

/** Everything the chat UI needs to know about a model family. */
export type ChatModel<Id extends string = string> = {
  id: Id;
  label: string;
  downloadSize: string;
  weightsUrl: string;
  tokenizerUrl: string;
  defaults: SamplingDefaults;
  createTokenizer(data: Uint8Array): ChatTokenizer;
  formatPrompt(history: ChatMessage[]): string;
  encodePrompt(tokenizer: ChatTokenizer, history: ChatMessage[]): number[];
  stopTokens(tokenizer: ChatTokenizer): number[];
  loadCheckpoint(
    data: Uint8Array<ArrayBuffer>,
    dtype: np.DType,
  ): Promise<LoadedChatModel>;
};

type BaseTokenizer = {
  bosToken: number;
  eosToken: number;
  encode(text: string): number[];
  decode(tokens: number[]): string;
};

type ChatModelImplementation<
  Id extends string,
  Model,
  State,
  Tokenizer extends BaseTokenizer,
> = {
  id: Id;
  label: string;
  downloadSize: string;
  weightsUrl: string;
  tokenizerUrl: string;
  defaults: SamplingDefaults;
  createTokenizer(data: Uint8Array): Tokenizer;
  decodeGenerated(tokenizer: Tokenizer, tokens: number[]): string;
  formatPrompt(history: ChatMessage[]): string;
  stopTokens(tokenizer: ChatTokenizer): number[];
  loadModel(file: safetensors.File, dtype: np.DType): Promise<Model>;
  createState(dtype: np.DType): State;
  prefill(model: Model, tokenIds: np.Array, state: State): np.Array;
  step(model: Model, token: number, state: State): np.Array;
};

/**
 * Keeps each model implementation fully typed while exposing a small,
 * type-erased interface to the page.
 */
function defineChatModel<
  const Id extends string,
  Model,
  State,
  Tokenizer extends BaseTokenizer,
>(
  implementation: ChatModelImplementation<Id, Model, State, Tokenizer>,
): ChatModel<Id> {
  const definition: ChatModel<Id> = {
    id: implementation.id,
    label: implementation.label,
    downloadSize: implementation.downloadSize,
    weightsUrl: implementation.weightsUrl,
    tokenizerUrl: implementation.tokenizerUrl,
    defaults: implementation.defaults,

    createTokenizer(data) {
      const tokenizer = implementation.createTokenizer(data);
      return {
        bosToken: tokenizer.bosToken,
        eosToken: tokenizer.eosToken,
        encode: (text) => tokenizer.encode(text),
        decode: (tokens) => tokenizer.decode(tokens),
        decodeGenerated: (tokens) =>
          implementation.decodeGenerated(tokenizer, tokens),
      };
    },

    formatPrompt: implementation.formatPrompt,
    encodePrompt(tokenizer, history) {
      return [
        tokenizer.bosToken,
        ...tokenizer.encode(implementation.formatPrompt(history)),
      ];
    },
    stopTokens: implementation.stopTokens,

    async loadCheckpoint(data, dtype) {
      const model = await implementation.loadModel(
        safetensors.parse(data),
        dtype,
      );
      let disposed = false;

      return {
        definition,

        createSession() {
          if (disposed) throw new Error(`${definition.label} is disposed`);
          const state = implementation.createState(dtype);
          let sessionDisposed = false;

          const assertActive = () => {
            if (sessionDisposed) {
              throw new Error(`${definition.label} session is disposed`);
            }
          };

          return {
            prefill(tokenIds) {
              assertActive();
              return implementation.prefill(tree.ref(model), tokenIds, state);
            },

            step(token) {
              assertActive();
              return implementation.step(tree.ref(model), token, state);
            },

            dispose() {
              if (sessionDisposed) return;
              sessionDisposed = true;
              tree.dispose(state);
            },
          };
        },

        dispose() {
          if (disposed) return;
          disposed = true;
          tree.dispose(model);
        },
      };
    },
  };

  return definition;
}

function gemmaPrompt(history: ChatMessage[]): string {
  // Matches the Gemma chat template, excluding the BOS token; encodePrompt
  // adds BOS as an ID so SentencePiece does not treat it as ordinary text.
  let text = "";
  for (const message of history) {
    const content = message.content.trim();
    if (content === "") continue;
    const role = message.role === "assistant" ? "model" : "user";
    text += `<start_of_turn>${role}\n${content}<end_of_turn>\n`;
  }
  return `${text}<start_of_turn>model\n`;
}

function lfmPrompt(history: ChatMessage[]): string {
  let text = "";
  for (const message of history) {
    const content = message.content.trim();
    if (content === "") continue;
    text += `<|im_start|>${message.role}\n${content}<|im_end|>\n`;
  }
  return `${text}<|im_start|>assistant\n`;
}

const gemma = defineChatModel<
  "gemma-3-270m",
  GemmaModel,
  GemmaState,
  tokenizers.SentencePiece
>({
  id: "gemma-3-270m",
  label: "Gemma 3 270M",
  downloadSize: "536 MB",
  weightsUrl:
    "https://huggingface.co/ekzhang/jax-js-models/resolve/main/gemma-3-270m/model-it-fp16.safetensors",
  tokenizerUrl:
    "https://huggingface.co/ekzhang/jax-js-models/resolve/main/gemma-3-270m/tokenizer.model",
  defaults: {
    temperature: 0.8,
    topK: 64,
    topP: 0.95,
    repetitionPenalty: 1,
  },
  createTokenizer: tokenizers.SentencePiece.fromBinary,
  decodeGenerated: (tokenizer, tokens) =>
    tokenizer.decode(
      tokens.filter(
        (token) =>
          token !== GEMMA_CONFIG.padTokenId &&
          token !== tokenizer.bosToken &&
          token !== tokenizer.eosToken &&
          token !== START_OF_TURN_TOKEN &&
          token !== END_OF_TURN_TOKEN,
      ),
    ),
  formatPrompt: gemmaPrompt,
  stopTokens: (tokenizer) => [tokenizer.eosToken, END_OF_TURN_TOKEN],
  loadModel: fromSafetensors,
  createState: (dtype) => createGemmaState({ dtype }),
  prefill: runGemmaPrefill,
  step: runGemmaStep,
});

const lfm = defineChatModel<
  "lfm2.5-350m",
  LfmModel,
  LfmState,
  HuggingFaceBpeTokenizer
>({
  id: "lfm2.5-350m",
  label: "LFM2.5 350M",
  downloadSize: "676 MB",
  weightsUrl:
    "https://huggingface.co/ekzhang/jax-js-models/resolve/main/lfm2.5-350m/model-fp16.safetensors",
  tokenizerUrl:
    "https://huggingface.co/LiquidAI/LFM2.5-350M/resolve/9e6c6ccf47cd318696e137d381a7ded8fe4df09f/tokenizer.json",
  defaults: {
    temperature: 0.1,
    topK: 50,
    topP: 1,
    repetitionPenalty: 1.05,
  },
  createTokenizer: HuggingFaceBpeTokenizer.fromBinary,
  decodeGenerated: (tokenizer, tokens) =>
    tokenizer.decode(
      tokens.filter((token) => !tokenizer.specialTokenIds.has(token)),
    ),
  formatPrompt: lfmPrompt,
  stopTokens: (tokenizer) => [tokenizer.eosToken],
  loadModel: lfmFromSafetensors,
  createState: (dtype) => createLfmState({ dtype }),
  prefill: runLfmPrefill,
  step: runLfmStep,
});

export const CHAT_MODELS = {
  [gemma.id]: gemma,
  [lfm.id]: lfm,
};

export type ChatModelId = keyof typeof CHAT_MODELS;

export const CHAT_MODEL_IDS = Object.keys(CHAT_MODELS) as ChatModelId[];
export const DEFAULT_CHAT_MODEL_ID: ChatModelId = lfm.id;
