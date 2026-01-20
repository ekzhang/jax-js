<script lang="ts">
  /* eslint-disable @typescript-eslint/no-unused-vars */
  import {
    defaultDevice,
    init,
    jit,
    numpy as np,
    tree,
    vmap,
  } from "@jax-js/jax";
  import { cachedFetch, opfs, safetensors, tokenizers } from "@jax-js/loaders";

  import DownloadManager from "$lib/common/DownloadManager.svelte";
  import { createStreamingPlayer } from "./audio";
  import {
    fromSafetensors,
    type PocketTTS,
    runFlowLMStep,
    runMimiDecode,
  } from "./pocket-tts";

  // Model configuration
  const latentDim = 32;
  const flowDim = 512;
  const modelDim = 1024;
  const framesAfterEos = 5;

  // Cached large objects to download.
  let _weights: safetensors.File | null = null;
  let _model: any | null = null;
  let _tokenizer: any | null = null;

  let downloadManager: DownloadManager;

  let isDownloadingWeights = $state(false);
  let hasModel = $state(false);

  async function downloadClipWeights(): Promise<safetensors.File> {
    if (_weights) return _weights;
    isDownloadingWeights = true;
    try {
      const weightsUrl =
        "https://huggingface.co/ekzhang/jax-js-models/resolve/main/kyutai-pocket-tts_b6369a24-fp16.safetensors";

      const data = await downloadManager.fetch("model weights", weightsUrl);
      const result = safetensors.parse(data);
      _weights = result;
      return result;
    } catch (error) {
      alert("Error downloading weights: " + error);
      throw error;
    } finally {
      isDownloadingWeights = false;
    }
  }

  async function getModel(): Promise<PocketTTS> {
    if (_model) return _model;
    const weights = await downloadClipWeights();
    _model = fromSafetensors(weights);
    hasModel = true;
    return _model;
  }

  const HF_URL_PREFIX =
    "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/fbf8280";

  const predefinedVoices = {
    alba: HF_URL_PREFIX + `/embeddings/alba.safetensors`,
    azelma: HF_URL_PREFIX + `/embeddings/azelma.safetensors`,
    cosette: HF_URL_PREFIX + `/embeddings/cosette.safetensors`,
    eponine: HF_URL_PREFIX + `/embeddings/eponine.safetensors`,
    fantine: HF_URL_PREFIX + `/embeddings/fantine.safetensors`,
    javert: HF_URL_PREFIX + `/embeddings/javert.safetensors`,
    jean: HF_URL_PREFIX + `/embeddings/jean.safetensors`,
    marius: HF_URL_PREFIX + `/embeddings/marius.safetensors`,
  };

  async function getTokenizer(): Promise<tokenizers.Unigram> {
    if (!_tokenizer)
      _tokenizer = await tokenizers.loadSentencePiece(
        "https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/fbf8280/tokenizer.model",
      );
    return _tokenizer;
  }

  let audioPromise: Promise<void> | null = null;

  async function run() {
    const devices = await init();
    if (devices.includes("webgpu")) {
      defaultDevice("webgpu");
    } else {
      alert("WebGPU not supported on this device, required for inference");
      return;
    }

    const model = await getModel();
    const tokenizer = await getTokenizer();
    console.log("Model:", model);

    const prompt = "The sun is shining, and the birds are singing.";
    const tokens = tokenizer.encode(prompt);
    console.log("Tokenizer:", tokens);

    const audioPrompt = safetensors.parse(
      await cachedFetch(predefinedVoices["fantine"]),
    ).tensors.audio_prompt;
    const voiceEmbed = np
      .array(audioPrompt.data as Float32Array<ArrayBuffer>, {
        shape: audioPrompt.shape,
        dtype: np.float32,
      })
      .slice(0)
      .astype(np.float16);

    const tokensAr = np.array(tokens, { dtype: np.uint32 });
    let embeds = model.flowLM.conditionerEmbed.ref.slice(tokensAr); // [seq_len, 1024]
    embeds = np.concatenate([voiceEmbed, embeds], 0);

    let sequence = np.full([1, latentDim], np.nan, { dtype: np.float16 });
    audioPromise = new Promise((resolve) => setTimeout(resolve, 2_000));
    const player = createStreamingPlayer();

    const jitMimiDecode = jit(runMimiDecode);
    try {
      let eosStep: number | null = null;
      let baseSeqLength = embeds.shape[0] + sequence.shape[0];
      for (let step = 0; step < 1000; step++) {
        const { latent, isEos } = runFlowLMStep(
          tree.ref(model.flowLM),
          sequence.ref,
          embeds.ref,
          baseSeqLength + step,
        );

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
        console.log("Sequence shape:", sequence.shape);

        // Hack: for mimi decode, actually pass the last 4 latents since we
        // don't have stateful convolution yet.
        let mimiInput = sequence.ref.slice([
          Math.max(1, sequence.shape[0] - 4),
        ]);

        mimiInput = mimiInput
          .mul(model.flowLM.embStd.ref)
          .add(model.flowLM.embMean.ref);
        const audio = jitMimiDecode(tree.ref(model.mimi), mimiInput);

        const lastAudioPromise = audioPromise;
        audioPromise = (async () => {
          const audioPcm = (await np
            .clip(audio.slice(0, [-1920]), -1, 1)
            .astype(np.float32)
            .data()) as Float32Array;
          await lastAudioPromise;
          player.playChunk(audioPcm);
        })();
      }
    } finally {
      sequence.dispose();
      embeds.dispose();
      jitMimiDecode.dispose();
      await audioPromise;
    }
  }
</script>

<DownloadManager bind:this={downloadManager} />

<button class="btn" onclick={run}>Run</button>

<style lang="postcss">
  @reference "$app.css";

  .btn {
    @apply flex items-center justify-center gap-2 px-5 py-2.5 border-2 border-black;
    @apply enabled:hover:bg-black enabled:hover:text-white disabled:opacity-50 disabled:cursor-not-allowed transition-colors;
  }
</style>
