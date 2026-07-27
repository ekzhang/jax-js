import { numpy as np, random, tree } from "@jax-js/jax";

import type { AudioPlayer } from "./audio";
import {
  createFlowLMState,
  createMimiDecodeState,
  type PocketTTS,
  runFlowLMStep,
  runMimiDecode,
} from "./pocket-tts";

export interface TTSSamplingProgress {
  framesGenerated: number;
  elapsedMs: number;
  framesPerSecond: number;
  realTimeFactor: number;
}

export interface PlayTTSOptions {
  framesAfterEos: number;
  seed: number | null;
  lsdDecodeSteps: number;
  temperature: number;
  noiseClamp: number | null;
  onProgress: (progress: TTSSamplingProgress) => void;
}

export async function playTTS(
  player: AudioPlayer,
  model: PocketTTS,
  embeds: np.Array,
  {
    framesAfterEos = 0,
    seed = null,
    lsdDecodeSteps = 1,
    temperature = 0.7,
    noiseClamp = null,
    onProgress,
  }: Partial<PlayTTSOptions> = {},
): Promise<void> {
  let lastLatent = model.flowLM.bosEmb.ref.reshape([1, -1]); // [1, 32]
  let audioPromise: Promise<void> = Promise.resolve();

  if (seed === null) seed = Math.floor(Math.random() * 2 ** 32);
  let key = random.key(seed);

  try {
    let flowLMState = createFlowLMState(model.flowLM);
    let mimiState = createMimiDecodeState(model.mimi);
    let eosStep: number | null = null;

    console.log("Starting TTS generation...");
    let lastTimestamp = performance.now();
    const samplingStart = lastTimestamp;
    let lastFrameTimestamp = samplingStart;
    const recentFrameDurationsMs: number[] = [];
    let framesGenerated = 0;

    for (let step = 0; step < 1000; step++) {
      let stepKey: np.Array;
      [key, stepKey] = random.split(key);
      const {
        latent,
        isEos,
        state: newFlowLMState,
      } = runFlowLMStep(
        tree.ref(model.flowLM),
        flowLMState,
        stepKey,
        lastLatent.ref,
        step === 0 ? embeds.ref : null,
        flowLMState.kvCacheLen, // same as offset
        lsdDecodeSteps,
        temperature,
        noiseClamp,
      );
      flowLMState = newFlowLMState;

      const isEosData = await isEos.data();
      if (isEosData[0] && eosStep === null) {
        console.log(`🛑 EOS at step ${step}!`);
        eosStep = step;
      }
      if (eosStep !== null && step >= eosStep + framesAfterEos) {
        console.log(
          `Generation ended at step ${step}, ${framesAfterEos} frames after EOS.`,
        );
        latent.dispose();
        break;
      }

      const prevLatent = lastLatent;
      lastLatent = latent;
      prevLatent.dispose();

      const timestamp = performance.now();
      console.log(
        `Generated step ${step} in ${(timestamp - lastTimestamp).toFixed(1)} ms`,
      );
      lastTimestamp = timestamp;

      const mimiInput = latent.ref
        .mul(model.flowLM.embStd.ref)
        .add(model.flowLM.embMean.ref);

      const [audio, newMimiState] = runMimiDecode(
        tree.ref(model.mimi),
        mimiState,
        mimiInput,
      );
      mimiState = newMimiState;

      const lastAudioPromise = audioPromise;
      audioPromise = (async () => {
        const audioPcm = (await np
          .clip(audio.slice(0), -1, 1)
          .astype(np.float32)
          .data()) as Float32Array;
        if (audioPcm.length !== 1920) {
          throw new Error(
            `expected 1920 audio samples, got ${audioPcm.length}`,
          );
        }

        // Update progress metrics for audio playback.
        framesGenerated++;
        const frameTimestamp = performance.now();
        recentFrameDurationsMs.push(frameTimestamp - lastFrameTimestamp);
        lastFrameTimestamp = frameTimestamp;
        if (recentFrameDurationsMs.length > 10) recentFrameDurationsMs.shift();
        const windowElapsedSeconds =
          recentFrameDurationsMs.reduce((sum, duration) => sum + duration, 0) /
          1000;
        const framesPerSecond =
          recentFrameDurationsMs.length / windowElapsedSeconds;
        onProgress?.({
          framesGenerated,
          elapsedMs: frameTimestamp - samplingStart,
          framesPerSecond,
          realTimeFactor: framesPerSecond / 12.5,
        });

        await lastAudioPromise;
        await player.playChunk(audioPcm);
      })();
    }
  } finally {
    lastLatent.dispose();
    tree.dispose([model, embeds]);
    await audioPromise;
  }
}
