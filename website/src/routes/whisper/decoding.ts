import { numpy as np } from "@jax-js/jax";

import type { WhisperConfig } from "./model";
import { WhisperTokenizer } from "./tokenizer";

export type TranscriptSegment = {
  text: string;
  start: number;
  end: number;
  row: number;
};

type Gate = {
  tokens: readonly number[];
  duration: number;
  forceTimestamp: boolean;
};

export async function sampleGreedy(
  logits: np.Array,
  tokens: readonly number[],
  duration: number,
  config: WhisperConfig,
): Promise<number> {
  const values = (await logits.data()) as ArrayLike<number>;
  const gate = { tokens, duration, forceTimestamp: false };
  gate.forceTimestamp =
    timestampScore(values, gate, config) > textScore(values, gate, config);

  let best: number = config.eosToken;
  let bestValue = -Infinity;
  for (let token = 0; token < values.length; token++) {
    if (!canSample(token, gate, config)) continue;
    if (values[token] > bestValue) {
      best = token;
      bestValue = values[token];
    }
  }
  return best;
}

export function decodeTranscriptTokens(
  tokens: readonly number[],
  tokenizer: WhisperTokenizer,
  config: WhisperConfig,
): string {
  return tokenizer.decode(
    tokens.filter((token) => !isTimestampToken(token, config)),
  );
}

export function buildTimestampSegments(
  tokens: readonly number[],
  tokenizer: WhisperTokenizer,
  totalDuration: number,
  config: WhisperConfig,
): TranscriptSegment[] {
  const groups: Omit<TranscriptSegment, "row">[] = [];
  let start: number | null = null;
  let textTokens: number[] = [];

  for (const token of tokens) {
    if (isTimestampToken(token, config)) {
      flushSegment(
        groups,
        tokenizer,
        textTokens,
        start,
        timestampSeconds(token, totalDuration, config),
        config,
      );
      textTokens = [];
      start = timestampSeconds(token, totalDuration, config);
    } else if (token < config.eosToken) {
      textTokens.push(token);
    }
  }
  flushSegment(
    groups,
    tokenizer,
    textTokens,
    start ?? 0,
    Math.min(totalDuration, 30),
    config,
  );

  return groups.map((group) => ({ ...group, row: 0 }));
}

function canSample(token: number, gate: Gate, config: WhisperConfig): boolean {
  if (gate.forceTimestamp && !isTimestampToken(token, config)) return false;
  if (isTimestampToken(token, config)) return canTimestamp(token, gate, config);
  if (
    config.suppressTokens.includes(token) ||
    token === config.noTimestampsToken
  )
    return false;
  if (token > config.eosToken && token < config.timestampBeginToken)
    return false;
  if (gate.tokens.length === 0 && config.beginSuppressTokens.includes(token))
    return false;
  if (timestampPhase(gate.tokens, config) === "timestamp-or-eos") {
    return token === config.eosToken;
  }
  return token <= config.eosToken;
}

function canTimestamp(
  token: number,
  { tokens, duration }: Gate,
  config: WhisperConfig,
): boolean {
  if (timestampPhase(tokens, config) === "text") return false;
  const last = findLastTimestampToken(tokens, config);
  const maxTimestamp =
    config.timestampBeginToken +
    Math.ceil(
      Math.min(Math.max(duration, 0.02), 30) / config.timestampStepSeconds,
    ) +
    2;
  return (last === null || token >= last) && token <= maxTimestamp;
}

function timestampPhase(tokens: readonly number[], config: WhisperConfig) {
  const last = tokens[tokens.length - 1];
  if (!isTimestampToken(last, config)) return "any";
  return tokens.length < 2 ||
    isTimestampToken(tokens[tokens.length - 2], config)
    ? "text"
    : "timestamp-or-eos";
}

function timestampScore(
  values: ArrayLike<number>,
  gate: Gate,
  config: WhisperConfig,
): number {
  return logSumExp(values, config.timestampBeginToken, values.length, (token) =>
    canTimestamp(token, gate, config),
  );
}

function textScore(
  values: ArrayLike<number>,
  gate: Gate,
  config: WhisperConfig,
): number {
  let best = -Infinity;
  for (let token = 0; token < config.timestampBeginToken; token++) {
    if (canSample(token, gate, config)) best = Math.max(best, values[token]);
  }
  return best;
}

function logSumExp(
  values: ArrayLike<number>,
  start: number,
  end: number,
  include: (token: number) => boolean,
): number {
  let max = -Infinity;
  for (let token = start; token < end; token++) {
    if (include(token)) max = Math.max(max, values[token]);
  }
  if (max === -Infinity) return max;

  let sum = 0;
  for (let token = start; token < end; token++) {
    if (include(token)) sum += Math.exp(values[token] - max);
  }
  return max + Math.log(sum);
}

function isTimestampToken(
  token: number | undefined,
  config: WhisperConfig,
): boolean {
  return (
    token !== undefined &&
    token >= config.timestampBeginToken &&
    token < config.vocabSize
  );
}

function findLastTimestampToken(
  tokens: readonly number[],
  config: WhisperConfig,
): number | null {
  for (let i = tokens.length - 1; i >= 0; i--) {
    const token = tokens[i];
    if (isTimestampToken(token, config)) return token;
  }
  return null;
}

function timestampSeconds(
  token: number,
  totalDuration: number,
  config: WhisperConfig,
): number {
  const seconds =
    (token - config.timestampBeginToken) * config.timestampStepSeconds;
  return Math.max(0, Math.min(Math.min(totalDuration, 30), seconds));
}

function flushSegment(
  groups: Omit<TranscriptSegment, "row">[],
  tokenizer: WhisperTokenizer,
  textTokens: number[],
  start: number | null,
  end: number,
  config: WhisperConfig,
) {
  const text = tokenizer.decode(textTokens).trim();
  if (!text) return;
  const from = start ?? 0;
  groups.push({
    text,
    start: from,
    end: Math.max(from + config.timestampStepSeconds, end),
  });
}
