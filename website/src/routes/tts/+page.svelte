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
  import { playPcm } from "./audio";
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

    const prompt = "This is TTS generated from jax-js!";
    const tokens = tokenizer.encode(prompt);
    console.log("Tokenizer:", tokens);

    const audioPrompt = safetensors.parse(
      await cachedFetch(predefinedVoices["azelma"]),
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

    const sequence = np.full([1, latentDim], np.nan, { dtype: np.float16 });
    const { latent, isEos } = runFlowLMStep(
      tree.ref(model.flowLM),
      sequence,
      embeds,
    );
    console.log("isEos?", isEos.js());
    console.log("Generated latent:", latent.ref.js());

    let mimiInput = latent
      .mul(model.flowLM.embStd.ref)
      .add(model.flowLM.embMean.ref);
    const audio = runMimiDecode(tree.ref(model.mimi), mimiInput);
    console.log(audio.shape);
    console.log("Generated audio:", audio.ref.js());

    const audioPcm = (await np
      .clip(audio, -1, 1)
      .astype(np.float32)
      .data()) as Float32Array;
    await playPcm(audioPcm);
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
