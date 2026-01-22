import { numpy as np, tree } from "@jax-js/jax";

import type { AudioPlayer } from "./audio";
import {
  createFlowLMState,
  createMimiDecodeState,
  type PocketTTS,
  runFlowLMStep,
  runMimiDecode,
} from "./pocket-tts";

export async function playTTS(
  player: AudioPlayer,
  model: PocketTTS,
  embeds: np.Array,
  framesAfterEos: number,
): Promise<void> {
  let sequence = model.flowLM.bosEmb.ref.reshape([1, -1]); // [1, 32]
  let audioPromise: Promise<void> = Promise.resolve();

  try {
    let flowLMState = createFlowLMState(model.flowLM);
    let mimiState = createMimiDecodeState(model.mimi);
    let eosStep: number | null = null;

    console.log("Starting TTS generation...");
    let lastTimestamp = performance.now();

    for (let step = 0; step < 1000; step++) {
      const {
        latent,
        isEos,
        state: newFlowLMState,
      } = runFlowLMStep(
        tree.ref(model.flowLM),
        flowLMState,
        step === 0 ? sequence.ref : sequence.ref.slice([-1]),
        step === 0 ? embeds.ref : null,
        flowLMState.kvCacheLen, // same as offset
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

      sequence = np.concatenate([sequence, latent]);

      const timestamp = performance.now();
      console.log(
        `Generated step ${step} in ${(timestamp - lastTimestamp).toFixed(1)} ms`,
      );
      lastTimestamp = timestamp;

      let mimiInput = sequence.ref.slice([-1]);
      mimiInput = mimiInput
        .mul(model.flowLM.embStd.ref)
        .add(model.flowLM.embMean.ref);

      const [audio, newMimiState] = runMimiDecode(
        tree.ref(model.mimi),
        mimiState,
        mimiInput,
        step,
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
        await lastAudioPromise;
        player.playChunk(audioPcm);
      })();
    }
  } finally {
    sequence.dispose();
    tree.dispose([model, embeds]);
    await audioPromise;
  }
}
