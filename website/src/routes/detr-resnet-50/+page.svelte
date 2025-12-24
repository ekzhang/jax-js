<script lang="ts">
  import {
    blockUntilReady,
    defaultDevice,
    init,
    nn,
    numpy as np,
  } from "@jax-js/jax";
  import { ONNXModel } from "@jax-js/onnx";

  import { runBenchmark } from "$lib/benchmark";
  import DownloadManager from "$lib/common/DownloadManager.svelte";
  import { COCO_CLASSES, stringToColor } from "./coco";

  let canvasEl: HTMLCanvasElement;
  let downloadManager: DownloadManager;
  let onnxModel: ONNXModel;

  let runCount = 0;

  function drawDetections(
    detections: {
      label: string;
      prob: number;
      cx: number;
      cy: number;
      w: number;
      h: number;
    }[],
  ) {
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
      const textH = 18;
      ctx.fillStyle = color;
      ctx.fillRect(x1, y1 - textH, textMetrics.width + 8, textH);

      // Draw label text
      ctx.fillStyle = "#ffffff";
      ctx.fillText(text, x1 + 4, y1 - 4);
    }
  }

  async function loadImage(url: string): Promise<{
    pixelValues: np.Array;
    pixelMask: np.Array;
  }> {
    const img = new Image();
    img.crossOrigin = "anonymous";
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = url;
    });

    // Center crop to square, then resize to 800x800
    const size = 800;
    const { width: origW, height: origH } = img;
    const cropSize = Math.min(origW, origH);
    const sx = (origW - cropSize) / 2;
    const sy = (origH - cropSize) / 2;

    // Draw on the visible canvas
    canvasEl.width = size;
    canvasEl.height = size;
    const ctx = canvasEl.getContext("2d", { willReadFrequently: true })!;
    ctx.drawImage(img, sx, sy, cropSize, cropSize, 0, 0, size, size);

    const imageData = ctx.getImageData(0, 0, size, size);
    const pixels = imageData.data; // RGBA

    // ImageNet normalization constants
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    // Convert to [1, 3, 800, 800] float32, with ImageNet normalization
    const pixelValues = np
      .array(new Float32Array(new Uint8Array(pixels.buffer)), {
        shape: [size, size, 4],
      })
      .slice([], [], [0, 3]) // RGB channels
      .mul(1 / 255)
      .sub(np.array(mean))
      .div(np.array(std))
      .transpose([2, 0, 1]) // to [3, 800, 800]
      .reshape([1, 3, size, size]);

    // Pixel mask: Transformers.js hardcodes this to [batch, 64, 64]
    const pixelMask = np.ones([1, 64, 64], { dtype: np.int32 });

    await blockUntilReady(pixelValues);
    return { pixelValues, pixelMask };
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
      console.log("ONNX Model loaded:", onnxModel);
    }

    const imageUrls = [
      "https://upload.wikimedia.org/wikipedia/commons/0/00/Gats_domestics.png",
      "https://upload.wikimedia.org/wikipedia/commons/d/d9/Desk333.JPG",
      "https://upload.wikimedia.org/wikipedia/commons/3/36/Afrykarium_tunel.jpg",
      "https://upload.wikimedia.org/wikipedia/commons/b/b4/Stanton_Cafe_and_Bar_in_Brisbane%2C_Queensland_09.jpg",
    ];
    const imageUrl = imageUrls[runCount++ % imageUrls.length];

    console.log(`Loading image: ${imageUrl}`);
    const { pixelValues, pixelMask } = await loadImage(imageUrl);
    console.log("Image loaded:", pixelValues.shape);
    console.log(
      "pixel_values min:",
      await np.min(pixelValues.ref).jsAsync(),
      "max:",
      await np.max(pixelValues.ref).jsAsync(),
    );

    console.log("Running forward pass...");
    let probs: number[][] = [];
    let predBoxes: number[][] = [];
    const seconds = await runBenchmark("detr-resnet-50", async () => {
      const outputs = onnxModel.run(
        {
          pixel_values: pixelValues,
          pixel_mask: pixelMask,
        },
        // { verbose: true },
      );
      await blockUntilReady(outputs);
      console.log("Outputs:", outputs);
      console.log("Outputs dtype:", outputs.logits.dtype);
      console.log("logits shape:", outputs.logits.shape);
      console.log("pred_boxes shape:", outputs.pred_boxes.shape);
      console.log("Logits:", await outputs.logits.ref.slice(0).jsAsync());

      probs = await nn.softmax(outputs.logits, -1).slice(0).jsAsync(); // [100, 92]
      predBoxes = await outputs.pred_boxes.slice(0).jsAsync(); // [100, 4]
    });
    console.log(`Forward pass took ${seconds.toFixed(3)} s`);

    // Find detections (excluding "no object" class at index 91)
    const detections: {
      label: string;
      prob: number;
      cx: number;
      cy: number;
      w: number;
      h: number;
    }[] = [];
    // console.log("Detections:");
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
      if (bestProb > 0.5) {
        const [cx, cy, w, h] = predBoxes[i];
        detections.push({
          label: COCO_CLASSES[bestClass],
          prob: bestProb,
          cx,
          cy,
          w,
          h,
        });
        // console.log(
        //   `  ${COCO_CLASSES[bestClass]}: ${(bestProb * 100).toFixed(1)}% at [cx=${cx.toFixed(2)}, cy=${cy.toFixed(2)}, w=${w.toFixed(2)}, h=${h.toFixed(2)}]`,
        // );
      }
    }

    // Draw detections on the canvas
    drawDetections(detections);
  }
</script>

<DownloadManager bind:this={downloadManager} />

<main class="p-4">
  <button onclick={loadAndRun} class="border px-2">
    Load & Run DETR ResNet-50
  </button>
  <div class="mt-4">
    <canvas bind:this={canvasEl} class="border" width="800" height="800"
    ></canvas>
  </div>
</main>
