<script lang="ts">
  import { browser } from "$app/environment";

  import {
    blockUntilReady,
    defaultDevice,
    init,
    jit,
    numpy as np,
    profiler,
    tree,
  } from "@jax-js/jax";
  import { ONNXModel } from "@jax-js/onnx";
  import {
    Camera,
    Image as ImageIcon,
    Link,
    Play,
    Square,
    Upload,
  } from "@lucide/svelte";

  import { runBenchmark } from "$lib/benchmark";
  import DownloadManager from "$lib/common/DownloadManager.svelte";
  import { countMethodCalls } from "$lib/profiling";
  import { COCO_CLASSES, stringToColor } from "../detr-resnet-50/coco";

  const width = 640;
  const height = 640;
  const defaultModelUrl =
    "https://huggingface.co/bukuroo/D-FINE-ONNX/resolve/main/dfine_s_obj2coco.onnx";
  const exampleUrls = [
    "https://upload.wikimedia.org/wikipedia/commons/0/00/Gats_domestics.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Desk333.JPG/1280px-Desk333.JPG",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Stanton_Cafe_and_Bar_in_Brisbane%2C_Queensland_09.jpg/1280px-Stanton_Cafe_and_Bar_in_Brisbane%2C_Queensland_09.jpg",
  ];
  const exampleLiveDelayMs = 1200;
  const coco80Classes = COCO_CLASSES.filter((name) => name !== "N/A");

  let canvasEl: HTMLCanvasElement;
  let videoEl: HTMLVideoElement;
  let downloadManager: DownloadManager;
  let onnxModel: ONNXModel | null = null;
  let onnxModelRun: any;
  let loadedModelKey = "";
  let warmedModelKey = "";
  let fileModelBytes: Uint8Array<ArrayBuffer> | null = null;

  let exampleIndex = 0;
  let nextLoopTimer: ReturnType<typeof setTimeout> | null = null;
  let webcamStream: MediaStream | null = null;
  let isFrontCamera = $state(false);
  let isLooping = $state(false);
  let isBusy = $state(false);
  let inputSource: "example" | "webcam" = $state("example");
  let modelUrl = $state(
    browser
      ? (new URLSearchParams(location.search).get("model") ?? defaultModelUrl)
      : defaultModelUrl,
  );
  let fileModelName = $state("");
  let confidenceThreshold = $state(0.45);
  let status = $state("Idle");
  let lastSeconds: number | null = $state(null);
  let lastFps: number | null = $state(null);
  let lastDispatches: number | null = $state(null);
  let lastBufferCreates: number | null = $state(null);
  let detections: Detection[] = [];

  interface Detection {
    label: string;
    score: number;
    x1: number;
    y1: number;
    x2: number;
    y2: number;
  }

  function labelFor(id: number): string {
    return coco80Classes[id] ?? COCO_CLASSES[id] ?? `class ${id}`;
  }

  function drawDetections(items: Detection[]) {
    const ctx = canvasEl.getContext("2d")!;
    for (const { label, score, x1, y1, x2, y2 } of items) {
      const left = Math.max(0, Math.min(width, x1));
      const top = Math.max(0, Math.min(height, y1));
      const right = Math.max(0, Math.min(width, x2));
      const bottom = Math.max(0, Math.min(height, y2));
      const boxW = Math.max(0, right - left);
      const boxH = Math.max(0, bottom - top);
      if (boxW < 1 || boxH < 1) continue;

      const color = stringToColor(label);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(left, top, boxW, boxH);

      const text = `${label} ${(score * 100).toFixed(1)}%`;
      ctx.font = "bold 14px sans-serif";
      const textW = Math.ceil(ctx.measureText(text).width) + 8;
      const labelY = Math.max(0, top - 18);
      ctx.fillStyle = color;
      ctx.fillRect(left, labelY, Math.min(textW, width - left), 18);
      ctx.fillStyle = "#fff";
      ctx.fillText(text, left + 4, labelY + 14);
    }
  }

  function nextExampleUrl(): string {
    return exampleUrls[exampleIndex++ % exampleUrls.length];
  }

  async function startWebcam() {
    if (webcamStream) return;
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width, height },
    });
    videoEl.srcObject = webcamStream;
    await videoEl.play();

    const track = webcamStream.getVideoTracks()[0];
    const settings = track.getSettings();
    isFrontCamera = settings.facingMode !== "environment";
  }

  function stopWebcam() {
    if (!webcamStream) return;
    webcamStream.getTracks().forEach((track) => track.stop());
    webcamStream = null;
    videoEl.srcObject = null;
  }

  function clearScheduledLoop() {
    if (nextLoopTimer === null) return;
    clearTimeout(nextLoopTimer);
    nextLoopTimer = null;
  }

  function scheduleNextRun() {
    if (!isLooping) return;
    if (inputSource === "example") {
      clearScheduledLoop();
      nextLoopTimer = setTimeout(() => {
        nextLoopTimer = null;
        if (isLooping) loadAndRun();
      }, exampleLiveDelayMs);
    } else {
      requestAnimationFrame(() => {
        if (isLooping) loadAndRun();
      });
    }
  }

  async function onInputSourceChange() {
    if (inputSource === "webcam") {
      await startWebcam();
    } else {
      stopWebcam();
      isLooping = false;
      clearScheduledLoop();
    }
  }

  async function loadImage(source: "example" | "webcam"): Promise<{
    images: np.Array;
    origTargetSizes: np.Array;
  }> {
    canvasEl.width = width;
    canvasEl.height = height;
    const ctx = canvasEl.getContext("2d", { willReadFrequently: true })!;

    if (source === "webcam") {
      const { videoWidth: origW, videoHeight: origH } = videoEl;
      const cropWidth = Math.min(origW, (origH * width) / height);
      const cropHeight = Math.min(origH, (origW * height) / width);
      const sx = (origW - cropWidth) / 2;
      const sy = (origH - cropHeight) / 2;
      if (isFrontCamera) {
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(
          videoEl,
          sx,
          sy,
          cropWidth,
          cropHeight,
          -width,
          0,
          width,
          height,
        );
        ctx.restore();
      } else {
        ctx.drawImage(
          videoEl,
          sx,
          sy,
          cropWidth,
          cropHeight,
          0,
          0,
          width,
          height,
        );
      }
    } else {
      const img = new Image();
      img.crossOrigin = "anonymous";
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = nextExampleUrl();
      });
      const cropWidth = Math.min(img.width, (img.height * width) / height);
      const cropHeight = Math.min(img.height, (img.width * height) / width);
      const sx = (img.width - cropWidth) / 2;
      const sy = (img.height - cropHeight) / 2;
      ctx.drawImage(img, sx, sy, cropWidth, cropHeight, 0, 0, width, height);
    }

    const pixels = ctx.getImageData(0, 0, width, height).data;
    if (source === "webcam" && detections.length > 0) {
      drawDetections(detections);
    }
    const images = np
      .array(new Float32Array(new Uint8Array(pixels.buffer)), {
        shape: [height, width, 4],
      })
      .slice([], [], [0, 3])
      .mul(1 / 255)
      .transpose([2, 0, 1])
      .reshape([1, 3, height, width]);
    const origTargetSizes = np.array(new Int32Array([height, width]), {
      shape: [1, 2],
      dtype: np.int32,
    });

    await blockUntilReady(images);
    return { images, origTargetSizes };
  }

  async function onModelFileChange(event: Event) {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    fileModelBytes = new Uint8Array(await file.arrayBuffer());
    fileModelName = file.name;
    status = `Loaded ${file.name}`;
  }

  async function ensureModel() {
    const url = modelUrl.trim();
    const modelKey = fileModelBytes
      ? `file:${fileModelName}:${fileModelBytes.byteLength}`
      : `url:${url}`;

    if (!fileModelBytes && !url) {
      throw new Error("Enter a D-FINE ONNX model URL or choose a file.");
    }

    if (onnxModel && modelKey === loadedModelKey) return;

    const modelBytes =
      fileModelBytes ?? (await downloadManager.fetch("D-FINE model", url));
    onnxModel?.dispose();
    onnxModel = new ONNXModel(modelBytes);
    onnxModelRun = jit(onnxModel.run, { staticArgnums: [1] });
    loadedModelKey = modelKey;
    warmedModelKey = "";

    profiler.startTrace();
  }

  function getOutput(
    outputs: Record<string, np.Array>,
    candidates: string[],
  ): np.Array {
    for (const name of candidates) {
      if (outputs[name]) return outputs[name];
    }
    throw new Error(
      `Missing output ${candidates.join("/")} in [${Object.keys(outputs).join(", ")}]`,
    );
  }

  function parseDetections(
    labelsData: ArrayLike<number>,
    boxesData: ArrayLike<number>,
    scoresData: ArrayLike<number>,
  ): Detection[] {
    const count = Math.min(
      labelsData.length,
      scoresData.length,
      Math.floor(boxesData.length / 4),
    );
    const items: Detection[] = [];
    for (let i = 0; i < count; i++) {
      const score = Number(scoresData[i]);
      if (score < confidenceThreshold) continue;
      const boxOffset = i * 4;
      const x1 = Number(boxesData[boxOffset]);
      const y1 = Number(boxesData[boxOffset + 1]);
      const x2 = Number(boxesData[boxOffset + 2]);
      const y2 = Number(boxesData[boxOffset + 3]);
      const normalized = Math.max(x1, y1, x2, y2) <= 1.5;
      items.push({
        label: labelFor(Number(labelsData[i])),
        score,
        x1: normalized ? x1 * width : x1,
        y1: normalized ? y1 * height : y1,
        x2: normalized ? x2 * width : x2,
        y2: normalized ? y2 * height : y2,
      });
    }
    return items.sort((a, b) => b.score - a.score).slice(0, 80);
  }

  async function loadAndRun() {
    if (isBusy) return;
    isBusy = true;
    status = "Preparing";

    try {
      const devices = await init("webgpu");
      if (!devices.includes("webgpu")) {
        throw new Error("WebGPU is not supported on this device/browser.");
      }
      defaultDevice("webgpu");
      await ensureModel();

      const { images, origTargetSizes } = await loadImage(inputSource);

      if (warmedModelKey !== loadedModelKey) {
        status = "Compiling";
        const warmOutputs = onnxModelRun({
          images: images.ref,
          orig_target_sizes: origTargetSizes.ref,
        });
        await blockUntilReady(warmOutputs);
        tree.dispose(warmOutputs);
        warmedModelKey = loadedModelKey;
      }

      status = "Running";
      const dispatchCount = countMethodCalls(
        GPUComputePassEncoder,
        "dispatchWorkgroups",
      );
      const bufferCreateCount = countMethodCalls(GPUDevice, "createBuffer");

      let labelsData: ArrayLike<number>;
      let boxesData: ArrayLike<number>;
      let scoresData: ArrayLike<number>;

      const seconds = await runBenchmark("d-fine", async () => {
        const outputs = onnxModelRun({
          images,
          orig_target_sizes: origTargetSizes,
        });
        await blockUntilReady(outputs);
        labelsData = await getOutput(outputs, ["labels"]).data();
        boxesData = await getOutput(outputs, ["boxes"]).data();
        scoresData = await getOutput(outputs, ["scores"]).data();
      });

      lastSeconds = seconds;
      lastFps = 1 / seconds;
      lastDispatches = dispatchCount();
      lastBufferCreates = bufferCreateCount();
      detections = parseDetections(labelsData!, boxesData!, scoresData!);
      drawDetections(detections);
      status = `Found ${detections.length} detections`;
    } catch (error) {
      console.error(error);
      status = error instanceof Error ? error.message : String(error);
      isLooping = false;
    } finally {
      isBusy = false;
      scheduleNextRun();
    }
  }

  function runOnce() {
    if (inputSource === "example") detections = [];
    loadAndRun();
  }

  function startLoop() {
    clearScheduledLoop();
    isLooping = true;
    loadAndRun();
  }

  function stopLoop() {
    isLooping = false;
    clearScheduledLoop();
  }
</script>

<svelte:head>
  <title>D-FINE – jax-js</title>
</svelte:head>

<DownloadManager bind:this={downloadManager} />

<main class="font-tiktok p-4 md:p-6 max-w-screen-xl mx-auto">
  <div class="mb-5">
    <h1 class="text-2xl md:text-3xl leading-tight">D-FINE detection</h1>
  </div>

  <div class="grid lg:grid-cols-[22rem_1fr] gap-5 items-start">
    <section class="border rounded-lg p-4 space-y-4">
      <label class="block">
        <span class="text-sm text-gray-600 inline-flex items-center gap-1.5">
          <Link size={15} /> Model URL
        </span>
        <input
          bind:value={modelUrl}
          oninput={() => {
            fileModelBytes = null;
            fileModelName = "";
          }}
          placeholder="https://.../model.onnx"
          class="mt-1 w-full border rounded px-2 py-1.5 text-sm"
        />
      </label>

      <label class="block">
        <span class="text-sm text-gray-600 inline-flex items-center gap-1.5">
          <Upload size={15} /> ONNX file
        </span>
        <input
          type="file"
          accept=".onnx"
          onchange={onModelFileChange}
          class="mt-1 w-full text-sm"
        />
        {#if fileModelName}
          <span class="block mt-1 text-xs text-gray-500">{fileModelName}</span>
        {/if}
      </label>

      <label class="block">
        <span class="text-sm text-gray-600">Input source</span>
        <select
          bind:value={inputSource}
          onchange={onInputSourceChange}
          class="mt-1 w-full border rounded px-2 py-1.5 text-sm"
        >
          <option value="example">Example images</option>
          <option value="webcam">Webcam</option>
        </select>
      </label>

      <label class="block">
        <span class="text-sm text-gray-600">
          Confidence {(confidenceThreshold * 100).toFixed(0)}%
        </span>
        <input
          type="range"
          min="0.05"
          max="0.95"
          step="0.05"
          bind:value={confidenceThreshold}
          class="mt-1 w-full"
        />
      </label>

      <div class="flex flex-wrap gap-2">
        <button onclick={runOnce} disabled={isBusy}>
          {#if inputSource === "webcam"}
            <Camera size={16} />
          {:else}
            <ImageIcon size={16} />
          {/if}
          Run
        </button>
        {#if isLooping}
          <button onclick={stopLoop}>
            <Square size={16} />
            Stop
          </button>
        {:else}
          <button onclick={startLoop} disabled={isBusy}>
            <Play size={16} />
            Live
          </button>
        {/if}
      </div>

      <div class="text-sm border-t pt-4 space-y-1">
        <div class="flex justify-between gap-3">
          <span class="text-gray-500">Status</span>
          <span class="text-right">{status}</span>
        </div>
        <div class="flex justify-between gap-3">
          <span class="text-gray-500">Latency</span>
          <span>
            {lastSeconds === null
              ? "--"
              : `${(lastSeconds * 1000).toFixed(1)} ms`}
          </span>
        </div>
        <div class="flex justify-between gap-3">
          <span class="text-gray-500">FPS</span>
          <span>{lastFps === null ? "--" : lastFps.toFixed(1)}</span>
        </div>
        <div class="flex justify-between gap-3">
          <span class="text-gray-500">Dispatches</span>
          <span>{lastDispatches ?? "--"}</span>
        </div>
        <div class="flex justify-between gap-3">
          <span class="text-gray-500">Buffers</span>
          <span>{lastBufferCreates ?? "--"}</span>
        </div>
      </div>
    </section>

    <section class="-mx-4 sm:mx-0">
      <video
        bind:this={videoEl}
        class="hidden"
        class:-scale-x-100={isFrontCamera}
        {width}
        {height}
        playsinline
        muted
      ></video>
      <canvas
        bind:this={canvasEl}
        class="w-full max-w-[640px] bg-black"
        {width}
        {height}
      ></canvas>
    </section>
  </div>
</main>

<style lang="postcss">
  @reference "$app.css";

  button {
    @apply inline-flex items-center gap-1.5 border rounded px-2.5 py-1.5 text-sm hover:bg-gray-100 active:scale-95 disabled:opacity-50 disabled:active:scale-100;
  }
</style>
