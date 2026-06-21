<script lang="ts">
  import { browser } from "$app/environment";
  import { resolve } from "$app/paths";

  import {
    blockUntilReady,
    defaultDevice,
    init,
    numpy as np,
    tree,
  } from "@jax-js/jax";
  import { safetensors } from "@jax-js/loaders";
  import {
    ActivityIcon,
    GithubIcon,
    LoaderCircleIcon,
    MicIcon,
    PauseIcon,
    PlayIcon,
    RadioTowerIcon,
    SquareIcon,
    UploadIcon,
    WavesIcon,
  } from "@lucide/svelte";
  import { onMount, tick } from "svelte";

  import DownloadManager from "$lib/common/DownloadManager.svelte";
  import {
    buildTimestampSegments,
    decodeTranscriptTokens,
    sampleGreedy,
    type TranscriptSegment,
  } from "./decoding";
  import {
    decodeAudioFromUrl,
    type DecodedAudio,
    type WhisperFeatures,
    whisperLogMel,
  } from "./features";
  import {
    createWhisperState,
    DEFAULT_WHISPER_CONFIG,
    fromSafetensors,
    type KVCache,
    prepareWhisperCrossKV,
    runWhisperDecoderStep,
    runWhisperEncoder,
    WHISPER_MODELS,
    type WhisperConfig,
    type WhisperModel,
    type WhisperModelId,
    type WhisperState,
  } from "./model";
  import { WhisperTokenizer } from "./tokenizer";

  type WhisperBackend = "webgpu" | "wasm";

  const SAMPLE_AUDIO_URL =
    "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/dialogue.wav";
  const SAMPLE_AUDIO_NAME = "dialogue.wav";
  const MAX_NEW_TOKENS = 96;
  const BACKEND_LABEL: Record<WhisperBackend, string> = {
    webgpu: "WebGPU",
    wasm: "Wasm",
  };

  let downloadManager: DownloadManager;
  let audioElement = $state<HTMLAudioElement | undefined>();
  let waveformCanvas: HTMLCanvasElement | undefined;
  let spectrogramCanvas: HTMLCanvasElement | undefined;
  let spectrogramScratchCanvas: HTMLCanvasElement | undefined;

  let _model: WhisperModel | null = null;
  let _modelBackend: WhisperBackend | null = null;
  let _modelId: WhisperModelId | null = null;
  let _tokenizer: WhisperTokenizer | null = null;
  let _tokenizerModelId: WhisperModelId | null = null;

  let selectedModel = $state<WhisperModelId>(DEFAULT_WHISPER_CONFIG.id);
  let backend = $state<WhisperBackend>("webgpu");
  let availableBackends = $state<WhisperBackend[]>([]);
  let checkedBackends = $state(false);
  let status = $state("standing by");
  let audioUrl = $state<string | null>(null);
  let playbackUrl = $state<string | null>(null);
  let audio = $state<DecodedAudio | null>(null);
  let features = $state<WhisperFeatures | null>(null);
  let transcript = $state("");
  let segments = $state<TranscriptSegment[]>([]);
  let loadMs = $state<number | null>(null);
  let prefillMs = $state<number | null>(null);
  let decodeMs = $state<number | null>(null);
  let recording = $state(false);
  let busy = $state(false);
  let errorMessage = $state("");
  let generatedCount = $state(0);
  let playbackTime = $state(0);
  let playing = $state(false);

  let ownedAudioUrl: string | null = null;
  let decodedAudioUrl: string | null = null;
  let recorder: MediaRecorder | null = null;
  let recordStream: MediaStream | null = null;
  let recordChunks: BlobPart[] = [];
  let playbackFrame: number | null = null;

  const duration = $derived(audio?.duration ?? 0);
  const selectedConfig = $derived(
    WHISPER_MODELS.find((model) => model.id === selectedModel) ??
      DEFAULT_WHISPER_CONFIG,
  );
  const playheadPercent = $derived(
    duration > 0
      ? Math.max(0, Math.min(100, (100 * playbackTime) / duration))
      : 0,
  );
  const showPlayhead = $derived(Boolean(audioUrl && duration > 0));

  onMount(() => {
    void initializeBackendOptions().catch((error) => {
      console.warn("Failed to initialize Whisper backends", error);
      checkedBackends = true;
    });
    return stopPlaybackTicker;
  });

  $effect(() => {
    void waveformCanvas;
    void spectrogramCanvas;
    void audio;
    void features;
    drawSignal();
  });

  $effect(() => {
    if (!browser) return;
    const redraw = () => drawSignal();
    window.addEventListener("resize", redraw);
    return () => window.removeEventListener("resize", redraw);
  });

  async function initializeBackendOptions() {
    const devices = await init("webgpu", "wasm");
    availableBackends = (["webgpu", "wasm"] as WhisperBackend[]).filter((d) =>
      devices.includes(d),
    );
    checkedBackends = true;
    if (!availableBackends.includes(backend)) {
      backend = availableBackends.includes("webgpu") ? "webgpu" : "wasm";
    }
  }

  async function setupDevice() {
    if (!checkedBackends) await initializeBackendOptions();
    if (!availableBackends.includes(backend)) {
      throw new Error(`${BACKEND_LABEL[backend]} is not available`);
    }
    defaultDevice(backend);
  }

  function disposeModel() {
    if (_model) tree.dispose(_model);
    _model = null;
    _modelBackend = null;
    _modelId = null;
  }

  function hfFileUrl(config: WhisperConfig, file: string): string {
    return `https://huggingface.co/${config.repo}/resolve/main/${file}`;
  }

  function changeWhisperModel() {
    disposeModel();
    _tokenizer = null;
    _tokenizerModelId = null;
    loadMs = null;
    prefillMs = null;
    decodeMs = null;
    generatedCount = 0;
    transcript = "";
    segments = [];
    status = `${selectedConfig.label} selected`;
  }

  async function getTokenizer(): Promise<WhisperTokenizer> {
    const config = selectedConfig;
    if (_tokenizer && _tokenizerModelId === config.id) return _tokenizer;
    status = "downloading tokenizer";
    const data = await downloadManager.fetch(
      `Whisper ${config.label} tokenizer`,
      hfFileUrl(config, "vocab.json"),
    );
    _tokenizer = WhisperTokenizer.fromVocabBytes(data);
    _tokenizerModelId = config.id;
    return _tokenizer;
  }

  async function getModel(): Promise<WhisperModel> {
    const config = selectedConfig;
    if (_model && _modelBackend === backend && _modelId === config.id)
      return _model;
    disposeModel();

    const start = performance.now();
    status = "downloading checkpoint";
    const data = await downloadManager.fetch(
      `Whisper ${config.label} weights`,
      hfFileUrl(config, "model.safetensors"),
    );
    status = "parsing checkpoint";
    const weights = safetensors.parse(data);
    const dtype = backend === "wasm" ? np.float32 : np.float16;
    status =
      backend === "wasm" ? "preparing fp32 weights" : "uploading fp16 weights";
    _model = await fromSafetensors(weights, dtype, config);
    _modelBackend = backend;
    _modelId = config.id;
    loadMs = performance.now() - start;
    return _model;
  }

  function setAudioSource(url: string, name: string, owned: boolean) {
    if (ownedAudioUrl) URL.revokeObjectURL(ownedAudioUrl);
    if (decodedAudioUrl) URL.revokeObjectURL(decodedAudioUrl);
    ownedAudioUrl = owned ? url : null;
    decodedAudioUrl = null;
    audioUrl = url;
    playbackUrl = url;
    audio = null;
    features = null;
    playbackTime = 0;
    playing = false;
    transcript = "";
    segments = [];
    prefillMs = null;
    decodeMs = null;
    generatedCount = 0;
    errorMessage = "";
    status = "signal queued";
  }

  async function loadAudio(url: string) {
    status = "decoding audio";
    await tick();
    audio = await decodeAudioFromUrl(url);
    if (decodedAudioUrl) URL.revokeObjectURL(decodedAudioUrl);
    decodedAudioUrl = audio.playbackUrl;
    playbackUrl = decodedAudioUrl;
    status = "extracting mel features";
    await tick();
    features = whisperLogMel(audio.samples);
    status = "signal locked";
    drawSignal();
  }

  async function ensureAudio() {
    if (!audioUrl) {
      setAudioSource(SAMPLE_AUDIO_URL, SAMPLE_AUDIO_NAME, false);
    }
    if (!audioUrl) throw new Error("No audio source");
    if (!features) await loadAudio(audioUrl);
  }

  async function loadSample() {
    if (busy || recording) return;
    busy = true;
    try {
      setAudioSource(SAMPLE_AUDIO_URL, SAMPLE_AUDIO_NAME, false);
      await loadAudio(SAMPLE_AUDIO_URL);
    } catch (error) {
      setError(error);
    } finally {
      busy = false;
    }
  }

  async function handleFileChange(event: Event) {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    busy = true;
    try {
      const url = URL.createObjectURL(file);
      setAudioSource(url, file.name, true);
      await loadAudio(url);
    } catch (error) {
      setError(error);
    } finally {
      busy = false;
      input.value = "";
    }
  }

  async function toggleRecording() {
    if (recording) {
      recorder?.stop();
      return;
    }

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("microphone capture is not available");
      }
      recordChunks = [];
      recordStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recorder = new MediaRecorder(recordStream);
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) recordChunks.push(event.data);
      };
      recorder.onstop = async () => {
        recording = false;
        busy = true;
        recordStream?.getTracks().forEach((track) => track.stop());
        recordStream = null;
        try {
          const blob = new Blob(recordChunks, { type: recorder?.mimeType });
          const url = URL.createObjectURL(blob);
          setAudioSource(url, `mic-${new Date().toLocaleTimeString()}`, true);
          await loadAudio(url);
        } catch (error) {
          setError(error);
        } finally {
          busy = false;
        }
      };
      recorder.start();
      recording = true;
      status = "recording";
    } catch (error) {
      setError(error);
    }
  }

  async function transcribe() {
    if (busy || recording) return;
    busy = true;
    errorMessage = "";
    transcript = "";
    segments = [];
    generatedCount = 0;
    prefillMs = null;
    decodeMs = null;

    let inputFeatures: np.Array | null = null;
    let encoded: np.Array | null = null;
    let crossKV: KVCache[] | null = null;
    let state: WhisperState | null = null;
    let logits: np.Array | null = null;

    try {
      await setupDevice();
      await ensureAudio();
      if (!features || !audio) throw new Error("Audio features are missing");
      const config = selectedConfig;
      const tokenizer = await getTokenizer();
      const model = await getModel();
      const dtype = backend === "wasm" ? np.float32 : np.float16;

      status = "running encoder";
      const prefillStart = performance.now();
      inputFeatures = np
        .array(features.data as Float32Array<ArrayBuffer>, {
          shape: [1, features.mels, features.frames],
          dtype: np.float32,
        })
        .astype(dtype);
      encoded = runWhisperEncoder(model.encoder, inputFeatures.ref, config);
      await encoded.blockUntilReady();

      status = "precomputing cross attention";
      crossKV = prepareWhisperCrossKV(model.decoder, encoded, config);
      encoded = null;
      await blockUntilReady(crossKV);

      state = createWhisperState(
        MAX_NEW_TOKENS + config.promptTokens.length + 8,
        dtype,
        config,
      );
      for (const token of config.promptTokens) {
        logits?.dispose();
        logits = runWhisperDecoderStep(
          model.decoder,
          crossKV,
          state,
          token,
          config,
        );
      }
      await logits?.blockUntilReady();
      prefillMs = performance.now() - prefillStart;

      const generated: number[] = [];
      const decodeStart = performance.now();
      for (let i = 0; i < MAX_NEW_TOKENS; i++) {
        status = `decoding token ${i + 1}/${MAX_NEW_TOKENS}`;
        const sampledLogits = logits;
        if (!sampledLogits) throw new Error("Decoder logits were not ready");
        logits = null;
        const next = await sampleGreedy(
          sampledLogits,
          generated,
          audio.duration,
          config,
        );
        if (next === config.eosToken) break;
        generated.push(next);
        generatedCount = generated.length;
        transcript = decodeTranscriptTokens(
          generated,
          tokenizer,
          config,
        ).trimStart();
        segments = buildTimestampSegments(
          generated,
          tokenizer,
          audio.duration,
          config,
        );
        await tick();
        logits = runWhisperDecoderStep(
          model.decoder,
          crossKV,
          state,
          next,
          config,
        );
      }

      decodeMs = performance.now() - decodeStart;
      status = "transcript resolved";
      if (!transcript) transcript = "(no speech decoded)";
    } catch (error) {
      setError(error);
    } finally {
      logits?.dispose();
      encoded?.dispose();
      inputFeatures?.dispose();
      if (crossKV) tree.dispose(crossKV);
      if (state) tree.dispose(state);
      busy = false;
    }
  }

  function setError(error: unknown) {
    console.error(error);
    errorMessage = error instanceof Error ? error.message : String(error);
    status = "fault";
  }

  function formatTime(seconds: number): string {
    if (!Number.isFinite(seconds) || seconds <= 0) return "0:00";
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60)
      .toString()
      .padStart(2, "0");
    return `${minutes}:${secs}`;
  }

  function formatMs(ms: number | null): string {
    if (ms === null) return "-";
    if (ms < 1000) return `${Math.round(ms)} ms`;
    return `${(ms / 1000).toFixed(2)} s`;
  }

  function updatePlaybackTime() {
    playbackTime = audioElement?.currentTime ?? 0;
  }

  async function togglePlayback() {
    if (!audioElement) return;
    if (audioElement.paused || audioElement.ended) {
      await audioElement.play();
    } else {
      audioElement.pause();
    }
  }

  function seekAudio(event: Event) {
    const nextTime = Number((event.currentTarget as HTMLInputElement).value);
    seekToAudioTime(nextTime);
  }

  function isSpaceKey(event: KeyboardEvent) {
    return event.code === "Space" || event.key === " ";
  }

  function togglePlaybackFromKeyboard(event: KeyboardEvent) {
    if (event.repeat || !isSpaceKey(event) || !playbackUrl) return false;
    event.preventDefault();
    event.stopPropagation();
    void togglePlayback();
    return true;
  }

  function seekToAudioTime(time: number) {
    const nextTime = Math.max(0, Math.min(duration, time));
    playbackTime = nextTime;
    if (audioElement) audioElement.currentTime = nextTime;
  }

  function seekSignal(event: PointerEvent) {
    if (!playbackUrl || duration <= 0) return;
    const frame = event.currentTarget as HTMLElement;
    frame.focus({ preventScroll: true });
    const rect = frame.getBoundingClientRect();
    if (rect.width <= 0) return;
    const progress = Math.max(
      0,
      Math.min(1, (event.clientX - rect.left) / rect.width),
    );
    seekToAudioTime(progress * duration);
  }

  function seekSignalKeyboard(event: KeyboardEvent) {
    if (!playbackUrl || duration <= 0) return;
    let nextTime = playbackTime;
    if (togglePlaybackFromKeyboard(event)) return;
    if (event.key === "Home") nextTime = 0;
    else if (event.key === "End") nextTime = duration;
    else if (event.key === "ArrowLeft") nextTime -= 1;
    else if (event.key === "ArrowRight") nextTime += 1;
    else if (event.key === "PageDown") nextTime -= 5;
    else if (event.key === "PageUp") nextTime += 5;
    else return;
    event.preventDefault();
    seekToAudioTime(nextTime);
  }

  function handleTransportRangeKeydown(event: KeyboardEvent) {
    togglePlaybackFromKeyboard(event);
  }

  function handlePageKeydown(event: KeyboardEvent) {
    if (event.defaultPrevented || event.repeat || !isSpaceKey(event)) return;
    if (isInteractiveTarget(event.target)) return;
    togglePlaybackFromKeyboard(event);
  }

  function isInteractiveTarget(target: EventTarget | null) {
    if (!(target instanceof HTMLElement)) return false;
    return (
      target.isContentEditable ||
      ["A", "BUTTON", "INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)
    );
  }

  function startPlaybackTicker() {
    if (!browser || playbackFrame !== null) return;
    const tick = () => {
      updatePlaybackTime();
      if (audioElement && !audioElement.paused && !audioElement.ended) {
        playbackFrame = requestAnimationFrame(tick);
      } else {
        playbackFrame = null;
      }
    };
    playbackFrame = requestAnimationFrame(tick);
  }

  function stopPlaybackTicker() {
    if (!browser || playbackFrame === null) return;
    cancelAnimationFrame(playbackFrame);
    playbackFrame = null;
  }

  function handleAudioPlay() {
    updatePlaybackTime();
    playing = true;
    startPlaybackTicker();
  }

  function handleAudioStop() {
    updatePlaybackTime();
    playing = false;
    stopPlaybackTicker();
  }

  function segmentStyle(segment: TranscriptSegment) {
    const total = Math.max(0.1, duration);
    const left = (100 * segment.start) / total;
    const width = Math.max(5, (100 * (segment.end - segment.start)) / total);
    return `left: ${left.toFixed(3)}%; width: ${Math.min(width, 100 - left).toFixed(3)}%;`;
  }

  function isActiveSegment(segment: TranscriptSegment) {
    return (
      playing && playbackTime >= segment.start && playbackTime < segment.end
    );
  }

  function setupCanvas(canvas: HTMLCanvasElement | undefined) {
    if (!browser || !canvas) return null;
    const ratio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext("2d");
    if (!context) return null;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { context, width: canvas.clientWidth, height: canvas.clientHeight };
  }

  function drawSignal() {
    drawWaveform();
    drawSpectrogram();
  }

  function drawWaveform() {
    const canvas = setupCanvas(waveformCanvas);
    if (!canvas) return;
    const { context, width, height } = canvas;
    context.fillStyle = "#070a0d";
    context.fillRect(0, 0, width, height);
    drawGrid(context, width, height, 18, "#17323a");

    if (!audio) {
      return;
    }

    const samples = audio.samples;
    const mid = height / 2;
    const amplitude = mid * 1.42;
    context.strokeStyle = "#d8f23d";
    context.lineWidth = 1.4;
    context.beginPath();
    for (let x = 0; x < width; x++) {
      const start = Math.floor((x / width) * samples.length);
      const end = Math.max(
        start + 1,
        Math.floor(((x + 1) / width) * samples.length),
      );
      let min = 1;
      let max = -1;
      for (let i = start; i < end; i++) {
        const sample = samples[i] ?? 0;
        min = Math.min(min, sample);
        max = Math.max(max, sample);
      }
      context.moveTo(x, mid + min * amplitude);
      context.lineTo(x, mid + max * amplitude);
    }
    context.stroke();

    context.strokeStyle = "#55f0ff";
    context.globalAlpha = 0.55;
    context.beginPath();
    context.moveTo(0, mid);
    context.lineTo(width, mid);
    context.stroke();
    context.globalAlpha = 1;
  }

  function drawSpectrogram() {
    const canvas = setupCanvas(spectrogramCanvas);
    if (!canvas) return;
    const { context, width, height } = canvas;
    context.fillStyle = "#050609";
    context.fillRect(0, 0, width, height);

    if (!features) {
      drawGrid(context, width, height, 16, "#241831");
      return;
    }

    const displayFrames = Math.max(
      1,
      Math.min(features.frames, Math.ceil(Math.max(duration, 0.01) * 100)),
    );
    const imageData = new Uint8ClampedArray(displayFrames * features.mels * 4);
    for (let y = 0; y < features.mels; y++) {
      for (let x = 0; x < displayFrames; x++) {
        const v = Math.max(
          0,
          Math.min(
            1,
            features.data[(features.mels - 1 - y) * features.frames + x],
          ),
        );
        const dst = (y * displayFrames + x) * 4;
        writePlasmaRail(imageData, dst, v);
        imageData[dst + 3] = 255;
      }
    }
    const scratch = (spectrogramScratchCanvas ??=
      document.createElement("canvas"));
    scratch.width = displayFrames;
    scratch.height = features.mels;
    scratch
      .getContext("2d")
      ?.putImageData(
        new ImageData(imageData, displayFrames, features.mels),
        0,
        0,
      );
    context.imageSmoothingEnabled = false;
    context.drawImage(scratch, 0, 0, width, height);
    context.fillStyle = "rgba(5, 6, 9, 0.25)";
    context.fillRect(0, 0, width, height);
    drawGrid(context, width, height, 24, "rgba(216, 242, 61, 0.12)");
  }

  function writePlasmaRail(
    out: Uint8ClampedArray,
    offset: number,
    value: number,
  ) {
    const v = Math.max(0, Math.min(1, value));
    out[offset] = Math.round(18 + 230 * Math.max(0, v - 0.35) ** 0.8);
    out[offset + 1] = Math.round(
      26 + 210 * Math.sin((Math.PI * v) / 1.35) ** 2,
    );
    out[offset + 2] = Math.round(36 + 190 * (1 - Math.abs(v - 0.55)) ** 2);
  }

  function drawGrid(
    context: CanvasRenderingContext2D,
    width: number,
    height: number,
    step: number,
    color: string,
  ) {
    context.strokeStyle = color;
    context.lineWidth = 1;
    context.beginPath();
    for (let x = 0; x <= width; x += step) {
      context.moveTo(x, 0);
      context.lineTo(x, height);
    }
    for (let y = 0; y <= height; y += step) {
      context.moveTo(0, y);
      context.lineTo(width, y);
    }
    context.stroke();
  }
</script>

<svelte:head>
  <title>Whisper ASR - jax-js</title>
</svelte:head>

<DownloadManager bind:this={downloadManager} />

<svelte:window onkeydown={handlePageKeydown} />

<main
  class="min-h-screen overflow-hidden bg-[#f7f7f3] font-tiktok text-[#202019]"
>
  <header
    class="mx-auto flex max-w-screen-xl items-center justify-between px-5 py-3 md:px-8"
  >
    <a href={resolve("/")} class="text-sm font-medium text-[#202019]">jax-js</a>
    <a
      href="https://github.com/ekzhang/jax-js/tree/main/website/src/routes/whisper"
      target="_blank"
      class="inline-flex items-center gap-[0.45rem] text-sm text-gray-600 hover:text-[#6d7619]"
    >
      <GithubIcon size={18} />
      Source
    </a>
  </header>

  <section class="mx-auto max-w-screen-xl px-5 pb-10 md:px-8">
    <div
      class="mb-2 mt-8 flex items-end justify-between gap-4 max-[900px]:grid max-[900px]:items-start"
    >
      <h1 class="text-[clamp(2rem,4.4vw,3.2rem)] font-medium leading-[0.95]">
        Whisper ASR
      </h1>
    </div>

    <div
      class="flex items-center justify-between gap-4 border-b border-[#d7d7cf] py-3.5 max-[900px]:grid max-[900px]:items-start"
    >
      <div class="flex flex-wrap items-center justify-start gap-4">
        <label class="control-label">
          Model
          <select
            class="model-select"
            bind:value={selectedModel}
            disabled={busy || recording}
            onchange={changeWhisperModel}
          >
            {#each WHISPER_MODELS as model}
              <option value={model.id}>{model.label}</option>
            {/each}
          </select>
        </label>

        <label class="control-label">
          Backend
          <select
            bind:value={backend}
            disabled={busy || recording || !checkedBackends}
          >
            {#if checkedBackends}
              {#each availableBackends as device}
                <option value={device}>{BACKEND_LABEL[device]}</option>
              {/each}
            {:else}
              <option value={backend}>Initializing</option>
            {/if}
          </select>
        </label>
      </div>

      <div
        class="flex flex-wrap items-center justify-end gap-2 max-[900px]:justify-start max-[520px]:grid max-[520px]:grid-cols-2"
      >
        <label class="icon-button max-[520px]:w-full" title="Upload audio">
          <UploadIcon size={17} />
          <span>Upload</span>
          <input
            class="sr-only"
            type="file"
            accept="audio/*"
            disabled={busy || recording}
            onchange={handleFileChange}
          />
        </label>

        <button
          class="icon-button max-[520px]:w-full"
          type="button"
          onclick={toggleRecording}
          disabled={busy}
          title={recording ? "Stop recording" : "Record audio"}
        >
          {#if recording}
            <SquareIcon size={17} />
            <span>Stop</span>
          {:else}
            <MicIcon size={17} />
            <span>Record</span>
          {/if}
        </button>

        <button
          class="icon-button max-[520px]:w-full"
          type="button"
          onclick={loadSample}
          disabled={busy || recording}
          title="Load sample audio"
        >
          <RadioTowerIcon size={17} />
          <span>Sample</span>
        </button>

        <button
          class="run-button max-[520px]:w-full"
          type="button"
          onclick={transcribe}
          disabled={busy || recording}
        >
          {#if busy}
            <LoaderCircleIcon size={17} class="animate-spin" />
          {:else}
            <PlayIcon size={17} />
          {/if}
          <span>Run</span>
        </button>
      </div>
    </div>

    <div class="grid grid-cols-6 gap-4 py-3.5 max-[900px]:grid-cols-2">
      <div class="min-w-0">
        <span class="metric-label">Status</span>
        <strong class="metric-value">{status}</strong>
      </div>
      <div class="min-w-0">
        <span class="metric-label">Duration</span>
        <strong class="metric-value">{formatTime(duration)}</strong>
      </div>
      <div class="min-w-0">
        <span class="metric-label">Weights</span>
        <strong class="metric-value">{formatMs(loadMs)}</strong>
      </div>
      <div class="min-w-0">
        <span class="metric-label">Prefill</span>
        <strong class="metric-value">{formatMs(prefillMs)}</strong>
      </div>
      <div class="min-w-0">
        <span class="metric-label">Decode</span>
        <strong class="metric-value">{formatMs(decodeMs)}</strong>
      </div>
      <div class="min-w-0">
        <span class="metric-label">Tokens</span>
        <strong class="metric-value">{generatedCount}</strong>
      </div>
    </div>

    {#if errorMessage}
      <p
        class="border-l-[3px] border-red-600 bg-red-100 px-3.5 py-3 font-mono text-red-900"
      >
        {errorMessage}
      </p>
    {/if}

    <section class="mt-6">
      <div class="grid gap-0">
        <div class="audio-transport">
          <audio
            bind:this={audioElement}
            class="native-audio"
            src={playbackUrl ?? ""}
            preload="metadata"
            onplay={handleAudioPlay}
            onpause={handleAudioStop}
            onended={handleAudioStop}
            onseeked={updatePlaybackTime}
            ontimeupdate={updatePlaybackTime}
          ></audio>
          <button
            class="transport-button"
            type="button"
            onclick={togglePlayback}
            disabled={!playbackUrl}
            aria-label={playing ? "Pause audio" : "Play audio"}
          >
            {#if playing}
              <PauseIcon size={15} />
            {:else}
              <PlayIcon size={15} />
            {/if}
          </button>
          <span class="transport-time">{formatTime(playbackTime)}</span>
          <input
            class="transport-range"
            type="range"
            min="0"
            max={Math.max(duration, 0.01)}
            step="0.01"
            value={playbackTime}
            disabled={!playbackUrl || duration <= 0}
            style={`--progress: ${playheadPercent.toFixed(2)}%;`}
            aria-label="Audio position"
            oninput={seekAudio}
            onkeydown={handleTransportRangeKeydown}
          />
          <span class="transport-time">{formatTime(duration)}</span>
        </div>
        <div
          class="relative grid gap-1 overflow-hidden rounded-b border border-t-0 border-[#1f3036] bg-[#151b1f] outline-none focus-visible:ring-1 focus-visible:ring-[#96a026] {playbackUrl
            ? 'cursor-pointer'
            : 'cursor-default'}"
          aria-disabled={!playbackUrl || duration <= 0}
          aria-label="Seek audio"
          aria-valuemax={Math.max(duration, 0)}
          aria-valuemin={0}
          aria-valuenow={playbackTime}
          onkeydown={seekSignalKeyboard}
          onpointerdown={seekSignal}
          role="slider"
          tabindex={playbackUrl ? 0 : -1}
        >
          <div class="relative min-w-0">
            <span class="plot-label"><WavesIcon size={13} /> waveform</span>
            <canvas
              bind:this={waveformCanvas}
              class="block h-[clamp(4rem,9vh,5.7rem)] w-full border-0"
            ></canvas>
          </div>
          <div class="relative min-w-0">
            <span class="plot-label"
              ><ActivityIcon size={13} /> log-mel spectrogram</span
            >
            <canvas
              bind:this={spectrogramCanvas}
              class="block h-[clamp(6.25rem,15vh,9rem)] w-full border-0"
            ></canvas>
          </div>
          {#if showPlayhead}
            <div
              class="playhead"
              style={`left: ${playheadPercent.toFixed(3)}%`}
            ></div>
          {/if}
        </div>
      </div>

      <div
        class="relative min-h-[3.25rem] bg-[linear-gradient(90deg,rgba(150,160,38,0.1)_1px,transparent_1px)] bg-[length:8.33%_100%] pt-3"
      >
        {#if segments.length > 0}
          {#each segments as segment}
            <div
              class="annotation"
              class:is-active={isActiveSegment(segment)}
              style={segmentStyle(segment)}
            >
              [{segment.text}]
            </div>
          {/each}
        {/if}
      </div>

      <div class="mt-5 border-t border-[#d7d7cf] pt-4">
        <p
          class="m-0 text-[clamp(1.5rem,3vw,2.4rem)] leading-snug {transcript
            ? ''
            : 'text-[#7c8178]'}"
        >
          {transcript || "Upload or record audio, then run Whisper in jax-js."}
        </p>
      </div>
    </section>
  </section>
</main>

<style lang="postcss">
  .control-label,
  .plot-label,
  .icon-button,
  .run-button {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
  }

  .control-label {
    gap: 0.65rem;
    color: #6b7280;
    font-size: 0.9rem;
  }

  select,
  .icon-button,
  .run-button {
    min-height: 2.35rem;
    border: 1px solid #c8c8bf;
    border-radius: 4px;
    background: white;
    color: #202019;
    padding: 0 0.75rem;
    font-size: 0.9rem;
  }

  select {
    width: 8.5rem;
    padding-block: 0.45rem;
    outline: none;
  }

  .model-select {
    width: 9.5rem;
  }

  select:focus {
    border-color: #96a026;
  }

  .icon-button:hover {
    color: #6d7619;
    border-color: #96a026;
  }

  :disabled,
  .icon-button:has(input:disabled) {
    opacity: 0.5;
  }

  .run-button {
    border-color: #202019;
    background: #202019;
    color: white;
  }

  .run-button:hover:not(:disabled) {
    background: #96a026;
    border-color: #96a026;
  }

  .metric-label,
  .annotation,
  .plot-label,
  .transport-time {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }

  .metric-label,
  .plot-label,
  .transport-time {
    color: #7c8178;
    font-size: 0.72rem;
    text-transform: uppercase;
  }

  .metric-label,
  .metric-value {
    display: block;
  }

  .metric-value {
    display: block;
    overflow: hidden;
    font:
      500 0.8rem ui-monospace,
      SFMono-Regular,
      Menlo,
      monospace;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .audio-transport {
    display: grid;
    grid-template-columns: auto auto minmax(0, 1fr) auto;
    align-items: center;
    gap: 0.7rem;
    width: 100%;
    margin-top: 0.15rem;
    border: 1px solid #1f3036;
    border-bottom: 0;
    border-radius: 4px 4px 0 0;
    background: #151b1f;
    padding: 0.45rem 0.65rem;
  }
  .native-audio {
    display: none;
  }
  .transport-button {
    display: grid;
    width: 1.75rem;
    height: 1.75rem;
    place-items: center;
    border: 1px solid rgba(216, 242, 61, 0.25);
    border-radius: 999px;
    background: rgba(7, 10, 13, 0.75);
    color: #d8f23d;
  }
  .transport-button:hover {
    border-color: #d8f23d;
  }
  .transport-time {
    color: #b8c0b1;
    font-size: 0.72rem;
    letter-spacing: 0;
    text-transform: none;
    white-space: nowrap;
  }
  .transport-range {
    width: 100%;
    height: 0.35rem;
    appearance: none;
    border-radius: 999px;
    background:
      linear-gradient(#d8f23d, #d8f23d) 0 0 / var(--progress) 100% no-repeat,
      #2a3438;
    outline: none;
  }
  .transport-range::-webkit-slider-thumb {
    width: 0.85rem;
    height: 0.85rem;
    appearance: none;
    border: 1px solid #070a0d;
    border-radius: 999px;
    background: #d8f23d;
  }
  .transport-range::-moz-range-thumb {
    width: 0.85rem;
    height: 0.85rem;
    border: 1px solid #070a0d;
    border-radius: 999px;
    background: #d8f23d;
  }

  .plot-label {
    position: absolute;
    left: 0.7rem;
    top: 0.55rem;
    z-index: 1;
    gap: 0.35rem;
    border: 1px solid rgba(216, 242, 61, 0.12);
    border-radius: 999px;
    background: rgba(7, 10, 13, 0.42);
    color: rgba(224, 231, 216, 0.72);
    padding: 0.22rem 0.45rem;
    backdrop-filter: blur(3px);
  }

  .playhead {
    position: absolute;
    inset-block: 0;
    z-index: 2;
    width: 2px;
    background: #f43f3f;
    box-shadow: 0 0 0 1px rgba(7, 10, 13, 0.35);
    pointer-events: none;
    transform: translateX(-1px);
  }

  .annotation {
    position: absolute;
    top: 0.9rem;
    overflow: hidden;
    border-left: 1px solid #96a026;
    color: #4b5313;
    font-size: clamp(0.68rem, 1.3vw, 0.86rem);
    line-height: 1.15;
    padding-left: 0.35rem;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .annotation.is-active {
    color: #202019;
    font-weight: 700;
  }
  .annotation::before {
    position: absolute;
    left: 0;
    top: -0.72rem;
    height: 0.55rem;
    border-left: 1px solid #96a026;
    content: "";
  }
</style>
