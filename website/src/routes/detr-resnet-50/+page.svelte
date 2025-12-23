<script lang="ts">
  import {
    blockUntilReady,
    defaultDevice,
    init,
    numpy as np,
  } from "@jax-js/jax";
  import { cachedFetch } from "@jax-js/loaders";
  import { ONNXModel } from "@jax-js/onnx";

  import { runBenchmark } from "$lib/benchmark";

  async function loadAndRun() {
    await init("webgpu");
    defaultDevice("webgpu");

    const modelUrl =
      "https://huggingface.co/Xenova/detr-resnet-50/resolve/main/onnx/model_fp16.onnx";
    const modelBytes = await cachedFetch(modelUrl);
    const onnxModel = new ONNXModel(modelBytes);
    console.log("ONNX Model loaded:", onnxModel);

    // Create dummy inputs
    // pixel_values: [batch, channels, height, width] - float16
    // pixel_mask: [batch, 64, 64] - int32
    const pixelValues = np.ones([1, 3, 800, 800], { dtype: np.float16 });
    const pixelMask = np.ones([1, 64, 64], { dtype: np.int32 });

    console.log("Running forward pass...");
    const seconds = await runBenchmark("detr-resnet-50", async () => {
      const outputs = onnxModel.run({
        pixel_values: pixelValues,
        pixel_mask: pixelMask,
      });
      await blockUntilReady(outputs);
      console.log("Outputs:", outputs);
      console.log("Outputs dtype:", outputs.logits.dtype);
      console.log("logits shape:", outputs.logits.shape);
      console.log("pred_boxes shape:", outputs.pred_boxes.shape);
      console.log("Logits:", outputs.logits.slice(0).js());
      console.log("Pred boxes:", outputs.pred_boxes.slice(0).js());
    });

    console.log(`Forward pass took ${seconds.toFixed(3)} s`);

    onnxModel.dispose();
  }
</script>

<main class="p-4">
  <button onclick={loadAndRun} class="border px-2">
    Load & Run DETR ResNet-50
  </button>
</main>
