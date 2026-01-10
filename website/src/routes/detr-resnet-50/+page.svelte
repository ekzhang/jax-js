<script lang="ts">
  import {
    blockUntilReady,
    defaultDevice,
    init,
    jit,
    lax,
    nn,
    numpy as np,
  } from "@jax-js/jax";
  import { ONNXModel } from "@jax-js/onnx";

  import { runBenchmark } from "$lib/benchmark";
  import DownloadManager from "$lib/common/DownloadManager.svelte";
  import { countMethodCalls, isMobile } from "$lib/profiling";
  import { COCO_CLASSES, stringToColor } from "./coco";

  const width = 800;
  const height = 800;
  const downsampleFactor = isMobile() ? 2 : 1;

  let canvasEl: HTMLCanvasElement;
  let videoEl: HTMLVideoElement;

  let downloadManager: DownloadManager;
  let onnxModel: ONNXModel;
  let onnxModelRun: any;
  let ortSession: import("onnxruntime-web/webgpu").InferenceSession | null =
    null;

  let runCount = 0;
  let webcamStream: MediaStream | null = null;
  let inputSource: "example" | "webcam" = $state("example");
  let isFrontCamera = $state(false);
  let isLooping = $state(false);

  let detections: Detection[] = [];

  interface Detection {
    label: string;
    prob: number;
    cx: number;
    cy: number;
    w: number;
    h: number;
  }

  function drawDetections(detections: Detection[]) {
    const ctx = canvasEl.getContext("2d")!;

    const imgW = canvasEl.width;
    const imgH = canvasEl.height;

    for (const { label, prob, cx, cy, w, h } of detections) {
      // Convert from center format to corner format
      const x1 = (cx - w / 2) * imgW;
      const y1 = (cy - h / 2) * imgH;
      const boxW = w * imgW;
      const boxH = h * imgH;

      const color = stringToColor(label);

      // Draw bounding box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, boxW, boxH);

      // Draw label background
      const text = `${label}: ${(prob * 100).toFixed(1)}%`;
      ctx.font = "bold 14px sans-serif";
      const textMetrics = ctx.measureText(text);
      const textH = 14;
      ctx.fillStyle = color;
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1 - textH, textMetrics.width + 4, textH);
      ctx.fillRect(x1, y1 - textH, textMetrics.width + 4, textH);

      // Draw label text
      ctx.fillStyle = "#ffffff";
      ctx.fillText(text, x1 + 2, y1 - 2);
    }
  }

  function getImageUrl() {
    const imageUrls = [
      "https://upload.wikimedia.org/wikipedia/commons/0/00/Gats_domestics.png",
      "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Desk333.JPG/1280px-Desk333.JPG",
      "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Afrykarium_tunel.jpg/1280px-Afrykarium_tunel.jpg",
      "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Stanton_Cafe_and_Bar_in_Brisbane%2C_Queensland_09.jpg/1280px-Stanton_Cafe_and_Bar_in_Brisbane%2C_Queensland_09.jpg",
    ];
    return imageUrls[runCount++ % imageUrls.length];
  }

  async function startWebcam() {
    if (webcamStream) return;
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width, height },
    });
    videoEl.srcObject = webcamStream;
    await videoEl.play();

    // Check if we got a front-facing camera
    // On desktop, facingMode is often undefined, so assume front camera (mirror by default)
    const track = webcamStream.getVideoTracks()[0];
    const settings = track.getSettings();
    isFrontCamera = settings.facingMode !== "environment";
  }

  function stopWebcam() {
    if (webcamStream) {
      webcamStream.getTracks().forEach((track) => track.stop());
      webcamStream = null;
      videoEl.srcObject = null;
    }
  }

  async function onInputSourceChange() {
    if (inputSource === "webcam") {
      await startWebcam();
    } else {
      stopWebcam();
    }
  }

  async function loadImage(source: "example" | "webcam"): Promise<{
    pixelValues: np.Array;
    pixelMask: np.Array;
  }> {
    canvasEl.width = width;
    canvasEl.height = height;
    const ctx = canvasEl.getContext("2d", { willReadFrequently: true })!;

    if (source === "webcam") {
      // Draw current video frame, center cropped to square
      const { videoWidth: origW, videoHeight: origH } = videoEl;
      const cropWidth = Math.min(origW, (origH * width) / height);
      const cropHeight = Math.min(origH, (origW * height) / width);
      const sx = (origW - cropWidth) / 2;
      const sy = (origH - cropHeight) / 2;
      if (isFrontCamera) {
        // Mirror front camera horizontally
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(
          videoEl,
          sx,
          sy,
          cropWidth,
          cropHeight,
          -cropWidth,
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
      // Load example image
      const img = new Image();
      img.crossOrigin = "anonymous";
      const url = getImageUrl();
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = url;
      });

      // Center crop to square, then resize to WxH
      const { width: origW, height: origH } = img;
      const cropWidth = Math.min(origW, (origH * width) / height);
      const cropHeight = Math.min(origH, (origW * height) / width);
      const sx = (origW - cropWidth) / 2;
      const sy = (origH - cropHeight) / 2;
      ctx.drawImage(img, sx, sy, cropWidth, cropHeight, 0, 0, width, height);
    }

    const imageData = ctx.getImageData(0, 0, width, height);
    drawDetections(detections);
    const pixels = imageData.data; // RGBA

    // ImageNet normalization constants
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    // Convert to [1, 3, W, H] float32, with ImageNet normalization
    let pixelValues = np
      .array(new Float32Array(new Uint8Array(pixels.buffer)), {
        shape: [width, height, 4],
      })
      .slice([], [], [0, 3]) // RGB channels
      .mul(1 / 255)
      .sub(np.array(mean))
      .div(np.array(std))
      .transpose([2, 0, 1]) // to [3, W, H]
      .reshape([1, 3, width, height]);

    if (downsampleFactor > 1) {
      // Downsample by factor of 2 using average pooling
      pixelValues = lax.reduceWindow(
        pixelValues,
        np.mean,
        [downsampleFactor, downsampleFactor],
        [downsampleFactor, downsampleFactor],
      );
    }

    // Pixel mask: Transformers.js hardcodes this to [batch, 64, 64]
    const pixelMask = np.ones([1, 64, 64], { dtype: np.int32 });

    await blockUntilReady(pixelValues);
    return { pixelValues, pixelMask };
  }

  async function processOutput(
    logitsData: Float32Array<ArrayBuffer>,
    boxesData: Float32Array<ArrayBuffer>,
  ) {
    const logits = np.array(logitsData, { shape: [100, 92] });
    const boxes = np.array(boxesData, { shape: [100, 4] });
    const probs = await nn.softmax(logits, -1).jsAsync();
    const predBoxes = await boxes.jsAsync();

    // Find detections (excluding "no object" class at index 91)
    const detections: Detection[] = [];
    for (let i = 0; i < 100; i++) {
      let bestClass = 0;
      let bestProb = 0;
      for (let c = 0; c < 91; c++) {
        if (probs[i][c] > bestProb) {
          bestProb = probs[i][c];
          bestClass = c;
        }
      }
      // Filter by confidence threshold
      if (bestProb > 0.75) {
        const [cx, cy, w, h] = predBoxes[i];
        detections.push({
          label: COCO_CLASSES[bestClass],
          prob: bestProb,
          cx,
          cy,
          w,
          h,
        });
      }
    }

    return detections;
  }

  async function loadAndRun() {
    const devices = await init("webgpu");
    if (!devices.includes("webgpu")) {
      alert("WebGPU is not supported on this device/browser.");
      return;
    }
    defaultDevice("webgpu");

    if (typeof onnxModel === "undefined") {
      // const modelUrl =
      // "https://huggingface.co/ekzhang/jax-js-models/resolve/main/detr-resnet-50-fp16.onnx";
      const modelUrl =
        "https://huggingface.co/Xenova/detr-resnet-50/resolve/main/onnx/model_fp16.onnx";
      const modelBytes = await downloadManager.fetch("model weights", modelUrl);
      onnxModel = new ONNXModel(modelBytes);
      onnxModelRun = jit(onnxModel.run, { staticArgnums: [1] });
      console.log("ONNX Model loaded:", onnxModel);
    }

    console.log(`Loading image from: ${inputSource}`);
    const { pixelValues, pixelMask } = await loadImage(inputSource);
    console.log("Image loaded:", pixelValues.shape);

    console.log("Running forward pass...");

    const dispatchCount = countMethodCalls(
      GPUComputePassEncoder,
      "dispatchWorkgroups",
    );
    const bufferCreateCount = countMethodCalls(GPUDevice, "createBuffer");

    let logitsData: Float32Array<ArrayBuffer>;
    let boxesData: Float32Array<ArrayBuffer>;

    const seconds = await runBenchmark("detr-resnet-50", async () => {
      const outputs = onnxModelRun(
        {
          pixel_values: pixelValues,
          pixel_mask: pixelMask,
        },
        // { verbose: true },
      );
      await blockUntilReady(outputs);
      console.log(
        "Outputs:",
        `logits=${outputs.logits.aval}, pred_boxe=${outputs.pred_boxes.aval}`,
      );

      logitsData = await outputs.logits.slice(0).jsAsync(); // [100, 92]
      boxesData = await outputs.pred_boxes.slice(0).jsAsync(); // [100, 4]
    });

    console.log(`Forward pass took ${seconds.toFixed(3)} s`);
    console.log(
      `jax-js dispatch count: ${dispatchCount()}, buffer creates: ${bufferCreateCount()}`,
    );

    detections = await processOutput(logitsData!, boxesData!);
    drawDetections(detections);

    if (isLooping) {
      requestAnimationFrame(() => loadAndRun());
    }
  }

  async function loadAndRunOrt() {
    const ort = await import("onnxruntime-web/webgpu");

    if (!ortSession) {
      const modelUrl =
        "https://huggingface.co/Xenova/detr-resnet-50/resolve/main/onnx/model_fp16.onnx";
      const modelBytes = await downloadManager.fetch(
        "model weights (ort)",
        modelUrl,
      );
      ortSession = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ["webgpu"],
      });
      console.log("ORT Session loaded:", ortSession);
    }

    console.log(`Loading image from: ${inputSource}`);
    const loadedImage = await loadImage(inputSource);
    const pixelValues = (await loadedImage.pixelValues.data()) as Float32Array;
    const pixelMask = new BigInt64Array(
      [...(await loadedImage.pixelMask.data())].map(BigInt),
    );
    console.log("Image loaded for ORT");

    console.log("Running forward pass with ORT...");

    let logitsData: Float32Array<ArrayBuffer>;
    let boxesData: Float32Array<ArrayBuffer>;

    const dispatchCount = countMethodCalls(
      GPUComputePassEncoder,
      "dispatchWorkgroups",
    );
    const bufferCreateCount = countMethodCalls(GPUDevice, "createBuffer");

    const seconds = await runBenchmark("detr-resnet-50-ort", async () => {
      const tensorPixelValues = new ort.Tensor("float32", pixelValues, [
        1,
        3,
        width / downsampleFactor,
        height / downsampleFactor,
      ]);
      const tensorPixelMask = new ort.Tensor("int64", pixelMask, [1, 64, 64]);

      const outputs = await ortSession!.run({
        pixel_values: tensorPixelValues,
        pixel_mask: tensorPixelMask,
      });
      console.log("ORT Outputs:", outputs);

      // Get logits and pred_boxes from outputs
      logitsData = outputs.logits.data as Float32Array<ArrayBuffer>; // [1, 100, 92]
      boxesData = outputs.pred_boxes.data as Float32Array<ArrayBuffer>; // [1, 100, 4]
    });
    console.log(`ORT Forward pass took ${seconds.toFixed(3)} s`);
    console.log(
      `ORT dispatch count: ${dispatchCount()}, buffer creates: ${bufferCreateCount()}`,
    );

    detections = await processOutput(logitsData!, boxesData!);
    drawDetections(detections);

    if (isLooping) {
      requestAnimationFrame(() => loadAndRunOrt());
    }
  }

  function stopLoop() {
    isLooping = false;
  }
</script>

<DownloadManager bind:this={downloadManager} />

<main class="p-4">
  <div class="flex flex-col gap-2">
    <div>
      <label>
        Input source:
        <select
          bind:value={inputSource}
          onchange={onInputSourceChange}
          class="border px-1 rounded"
        >
          <option value="example">Example images</option>
          <option value="webcam">Webcam</option>
        </select>
      </label>
    </div>
    <div class="flex gap-2 items-center">
      {#if isLooping}
        <button onclick={stopLoop} class="border px-2">Stop</button>
      {:else}
        <button
          onclick={() => {
            if (inputSource === "webcam") isLooping = true;
            else detections = [];
            loadAndRun();
          }}
        >
          Load & Run (jax-js)
        </button>
        <button
          onclick={() => {
            if (inputSource === "webcam") isLooping = true;
            else detections = [];
            loadAndRunOrt();
          }}
        >
          Load & Run (onnx)
        </button>
      {/if}
    </div>
  </div>
  <div class="mt-4 -mx-4 sm:mx-0">
    <video
      bind:this={videoEl}
      class="hidden"
      class:-scale-x-100={isFrontCamera}
      {width}
      {height}
      playsinline
      muted
    ></video>
    <canvas bind:this={canvasEl} class="max-w-full" {width} {height}></canvas>
  </div>
</main>

<style lang="postcss">
  @reference "$app.css";

  button {
    @apply border rounded px-2 hover:bg-gray-100 active:scale-95;
  }
</style>
