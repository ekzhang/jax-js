<script lang="ts">
  import {
    defaultDevice,
    init,
    jit,
    numpy as np,
    tree,
    vmap,
  } from "@jax-js/jax";
  import { safetensors, tokenizers } from "@jax-js/loaders";

  import DownloadManager from "$lib/common/DownloadManager.svelte";
  import { fromSafetensors, type PocketTTS } from "./pocket-tts";

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

  async function getTokenizer() {
    if (!_tokenizer) _tokenizer = await tokenizers.getBpe("clip");
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
