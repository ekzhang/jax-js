<script lang="ts">
  import { defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
  import { cachedFetch, safetensors, tokenizers } from "@jax-js/loaders";
  import { AudioLinesIcon, DownloadIcon, GithubIcon } from "@lucide/svelte";
  import { onMount } from "svelte";

  import DownloadManager from "$lib/common/DownloadManager.svelte";
  import { createStreamingPlayer } from "./audio";
  import { playTTS, type TTSSamplingProgress } from "./inference";
  import { fromSafetensors, type PocketTTS } from "./pocket-tts";

  type TTSBackend = "webgpu" | "wasm";

  const BACKEND_LABEL: Record<TTSBackend, string> = {
    webgpu: "WebGPU",
    wasm: "Wasm",
  };
  const TTS_BACKENDS: TTSBackend[] = ["webgpu", "wasm"];

  // Cached large objects to download.
  let _weights: safetensors.File | null = null;
  let _model: PocketTTS | null = null;
  let _modelBackend: TTSBackend | null = null;
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
  let backend = $state<TTSBackend>("webgpu");
  let availableBackends = $state<TTSBackend[]>([]);
  let checkedBackends = $state(false);
  let samplingProgress = $state<TTSSamplingProgress | null>(null);

  // Advanced options
  let seed = $state<number | null>(null);
  let temperature = $state(0.7);
  let lsdDecodeSteps = $state(1);

  onMount(() => {
    void initializeBackendOptions().catch((error) => {
      console.warn("Failed to initialize Pocket TTS backends", error);
    });
  });

  async function initializeBackendOptions() {
    const devices = await init(...TTS_BACKENDS);
    availableBackends = TTS_BACKENDS.filter((device) =>
      devices.includes(device),
    );
    checkedBackends = true;
    if (availableBackends.length > 0 && !availableBackends.includes(backend)) {
      backend = availableBackends.includes("webgpu") ? "webgpu" : "wasm";
    }
  }

  function disposeModel() {
    if (_model) tree.dispose(_model);
    _model = null;
    _modelBackend = null;
    hasModel = false;
  }

  function handleBackendChange(event: Event) {
    const nextBackend = (event.currentTarget as HTMLSelectElement)
      .value as TTSBackend;
    if (nextBackend === backend) return;
    backend = nextBackend;
    samplingProgress = null;
    audioBlob = null;
    disposeModel();
  }

  async function setupDevice() {
    if (!checkedBackends) await initializeBackendOptions();

    let selectedBackend = backend;
    if (!availableBackends.includes(selectedBackend)) {
      if (selectedBackend === "webgpu" && availableBackends.includes("wasm")) {
        selectedBackend = "wasm";
      } else {
        throw new Error(
          `${BACKEND_LABEL[selectedBackend]} is not available in this browser.`,
        );
      }
    }

    backend = selectedBackend;
    defaultDevice(selectedBackend);
  }

  async function downloadModelWeights(): Promise<safetensors.File> {
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
    if (_model && _modelBackend === backend) return _model;
    if (_model) disposeModel();
    const weights = await downloadModelWeights();
    const weightDtype = backend === "wasm" ? np.float32 : np.float16;
    _model = fromSafetensors(weights, weightDtype);
    _modelBackend = backend;
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

  async function getTokenizer(): Promise<tokenizers.SentencePiece> {
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
    await setupDevice();
    const model = await getModel();
    const tokenizer = await getTokenizer();
    console.log("Model:", model);

    const [text, framesAfterEos] = prepareTextPrompt(prompt);
    const tokens = tokenizer.encode(text);
    console.log("Tokens:", tokens);

    const audioPrompt = safetensors.parse(
      await cachedFetch(predefinedVoices[selectedVoice]),
    ).tensors.audio_prompt;
    const weightDtype = backend === "wasm" ? np.float32 : np.float16;
    const voiceEmbed = np
      .array(audioPrompt.data as Float32Array<ArrayBuffer>, {
        shape: audioPrompt.shape,
        dtype: np.float32,
      })
      .slice(0)
      .astype(weightDtype);

    const tokensAr = np.array(tokens, { dtype: np.uint32 });
    let embeds = model.flowLM.conditionerEmbed.ref.slice(tokensAr); // [seq_len, 1024]
    embeds = np.concatenate([voiceEmbed, embeds]);

    const player = createStreamingPlayer();
    try {
      await playTTS(player, tree.ref(model), embeds, {
        framesAfterEos,
        seed,
        temperature,
        lsdDecodeSteps,
        onProgress: (progress) => {
          samplingProgress = progress;
        },
      });
      audioBlob = player.toWav();
    } finally {
      await player.close();
    }
  }
</script>

<title>Kyutai Pocket TTS (Web)</title>

<DownloadManager bind:this={downloadManager} />

<main class="mx-4 my-8">
  <h1 class="text-2xl font-semibold mb-1">
    Kyutai Pocket TTS
    <a
      target="_blank"
      href="https://github.com/ekzhang/jax-js/tree/main/website/src/routes/tts"
    >
      <GithubIcon class="inline-block ml-2 -mt-1" />
    </a>
  </h1>
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
      samplingProgress = null;
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
      bind:value={prompt}></textarea>

    <div class="flex flex-wrap items-end gap-3 mt-2">
      <label class="text-xs text-gray-600">
        Voice
        <select
          class="block mt-1 h-9 border-2 rounded px-2 text-base text-black"
          bind:value={selectedVoice}
          disabled={playing}
        >
          {#each Object.keys(predefinedVoices) as voice}
            <option value={voice}
              >{voice.charAt(0).toLocaleUpperCase() + voice.slice(1)}</option
            >
          {/each}
        </select>
      </label>

      <label class="text-xs text-gray-600">
        Backend
        <select
          class="block mt-1 h-9 border-2 rounded px-2 text-base text-black"
          value={backend}
          onchange={handleBackendChange}
          disabled={playing ||
            !checkedBackends ||
            availableBackends.length === 0}
        >
          {#if checkedBackends && availableBackends.length > 0}
            {#each availableBackends as device}
              <option value={device}>{BACKEND_LABEL[device]}</option>
            {/each}
          {:else if checkedBackends}
            <option>No supported backend</option>
          {:else}
            <option value={backend}>Initializing…</option>
          {/if}
        </select>
      </label>

      <button
        class="btn h-9"
        type="submit"
        disabled={playing ||
          prompt.trim() === "" ||
          !checkedBackends ||
          availableBackends.length === 0}
      >
        {#if playing}
          <AudioLinesIcon size={20} class="animate-pulse" />
        {:else}
          Play
        {/if}
      </button>

      {#if audioBlob}
        <a
          class="btn h-9"
          href={URL.createObjectURL(audioBlob)}
          download="tts_output.wav"
        >
          <DownloadIcon size={20} />
        </a>
      {/if}
    </div>

    <div
      class="mt-3 min-h-6 text-sm text-gray-600 tabular-nums"
      aria-live="polite"
    >
      {#if playing && !samplingProgress}
        Warming up {BACKEND_LABEL[backend]}…
      {:else if samplingProgress}
        {playing ? "Sampling" : "Completed"} · RTF {samplingProgress.realTimeFactor.toFixed(
          2,
        )}× · {samplingProgress.framesPerSecond.toFixed(1)} frame/s ·
        {samplingProgress.framesGenerated} frames
      {/if}
    </div>

    <details class="mt-8 max-w-md">
      <summary class="cursor-pointer text-gray-600 hover:text-gray-800"
        >Advanced options</summary
      >
      <div class="mt-3 space-y-4 pl-2">
        <p class="text-xs text-gray-500">
          WebGPU runs on fp16 activations; Wasm uses fp32. Performance metrics
          average the latest 10 frames. RTF is real-time factor: 12.5 FPS.
        </p>

        <div>
          <label class="block text-sm text-gray-700">
            Seed
            <input
              type="number"
              class="block mt-1 border-2 rounded p-1 w-32"
              placeholder="(random)"
              bind:value={seed}
            />
          </label>
        </div>

        <div>
          <label class="block text-sm text-gray-700">
            Temperature: {temperature.toFixed(2)}
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              class="mt-1 w-full"
              bind:value={temperature}
            />
          </label>
        </div>

        <div>
          <label class="block text-sm text-gray-700">
            LSD Decode Steps: {lsdDecodeSteps}
            <input
              type="range"
              min="1"
              max="4"
              step="1"
              class="mt-1 w-full"
              bind:value={lsdDecodeSteps}
            />
          </label>
        </div>
      </div>
    </details>
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
