<script lang="ts">
  import { defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
  import { cachedFetch, safetensors, tokenizers } from "@jax-js/loaders";
  import { AudioLinesIcon, DownloadIcon } from "@lucide/svelte";

  import DownloadManager from "$lib/common/DownloadManager.svelte";
  import { createStreamingPlayer } from "./audio";
  import { playTTS } from "./inference";
  import { fromSafetensors, type PocketTTS } from "./pocket-tts";

  // Cached large objects to download.
  let _weights: safetensors.File | null = null;
  let _model: any | null = null;
  let _tokenizer: any | null = null;

  let downloadManager: DownloadManager;

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  let isDownloadingWeights = $state(false);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  let hasModel = $state(false);

  let prompt = $state("The sun is shining, and the birds are singing.");
  let selectedVoice = $state("azelma");
  let playing = $state(false);
  let audioBlob = $state<Blob | null>(null);

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

  const predefinedVoices: Record<string, string> = {
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

  function prepareTextPrompt(text: string): [string, number] {
    // Ported from the Python repository.
    text = text.trim();
    if (text === "") throw new Error("Prompt cannot be empty");
    text = text.replace(/\s+/g, " ");
    const numberOfWords = text.split(" ").length;
    let framesAfterEosGuess = 3;
    if (numberOfWords <= 4) {
      framesAfterEosGuess = 5;
    }

    // Make sure it starts with an uppercase letter
    text = text.replace(/^(\p{Ll})/u, (c) => c.toLocaleUpperCase());

    // Let's make sure it ends with some kind of punctuation
    // If it ends with a letter or digit, we add a period.
    if (/[\p{L}\p{N}]$/u.test(text)) {
      text = text + ".";
    }

    // The model does not perform well when there are very few tokens, so
    // we can add empty spaces at the beginning to increase the token count.
    if (text.split(" ").length < 5) {
      text = " ".repeat(8) + text;
    }

    return [text, framesAfterEosGuess];
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

    const [text, framesAfterEos] = prepareTextPrompt(prompt);
    const tokens = tokenizer.encode(text);
    console.log("Tokens:", tokens);

    const audioPrompt = safetensors.parse(
      await cachedFetch(predefinedVoices[selectedVoice]),
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
    embeds = np.concatenate([voiceEmbed, embeds]);

    const player = createStreamingPlayer();
    try {
      await playTTS(player, tree.ref(model), embeds, framesAfterEos);
      audioBlob = player.toWav();
    } finally {
      await player.close();
    }
  }
</script>

<title>Kyutai Pocket TTS (Web)</title>

<DownloadManager bind:this={downloadManager} />

<main class="mx-4 my-8">
  <h1 class="text-2xl font-semibold mb-1">Kyutai Pocket TTS</h1>
  <p class="text-lg text-gray-500">
    Text-to-speech AI voice model, running in your browser with <a
      href="/"
      class="text-primary hover:underline">jax-js</a
    >.
  </p>

  <form
    class="mt-6"
    onsubmit={async (event) => {
      event.preventDefault();
      audioBlob = null;
      playing = true;
      try {
        await run();
      } finally {
        playing = false;
      }
    }}
  >
    <textarea
      class="border-2 rounded p-2 w-full max-w-md"
      rows={6}
      placeholder="Enter your prompt here…"
      bind:value={prompt}
    ></textarea>

    <div class="flex gap-3 mt-1 h-9">
      <select class="border-2 rounded p-1" bind:value={selectedVoice}>
        {#each Object.keys(predefinedVoices) as voice}
          <option value={voice}
            >{voice.charAt(0).toLocaleUpperCase() + voice.slice(1)}</option
          >
        {/each}
      </select>
      <button
        class="btn"
        type="submit"
        disabled={playing || prompt.trim() === ""}
      >
        {#if playing}
          <AudioLinesIcon size={20} class="animate-pulse" />
        {:else}
          Play
        {/if}
      </button>

      {#if audioBlob}
        <a
          class="btn"
          href={URL.createObjectURL(audioBlob)}
          download="tts_output.wav"
        >
          <DownloadIcon size={20} />
        </a>
      {/if}
    </div>
  </form>
</main>

<style lang="postcss">
  @reference "$app.css";

  .btn {
    @apply flex items-center justify-center gap-2 px-3 rounded py-1 border-2 border-black;
    @apply disabled:opacity-50 disabled:cursor-wait transition-colors;
    @apply not-disabled:hover:bg-black not-disabled:hover:text-white;
  }
</style>
