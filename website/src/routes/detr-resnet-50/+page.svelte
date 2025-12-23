<script lang="ts">
  import { numpy as np } from "@jax-js/jax";
  import { cachedFetch } from "@jax-js/loaders";
  import { ONNXModel } from "@jax-js/onnx";

  async function loadAndRun() {
    const modelUrl =
      "https://huggingface.co/Xenova/detr-resnet-50/resolve/main/onnx/model_fp16.onnx";
    const modelBytes = await cachedFetch(modelUrl);
    const onnxModel = new ONNXModel(modelBytes);
    console.log("ONNX Model loaded:", onnxModel);

    // Create dummy inputs
    // pixel_values: [batch, channels, height, width] - float32
    // pixel_mask: [batch, 64, 64] - int64
    const pixelValues = np.ones([1, 3, 800, 800], { dtype: np.float32 });
    const pixelMask = np.ones([1, 64, 64], { dtype: np.int32 });

    console.log("Running forward pass...");
    const startTime = performance.now();

    const outputs = onnxModel.run({
      pixel_values: pixelValues,
      pixel_mask: pixelMask,
    });

    const endTime = performance.now();
    console.log(`Forward pass took ${(endTime - startTime).toFixed(2)}ms`);
    console.log("Outputs:", outputs);
    console.log("logits shape:", outputs.logits.shape);
    console.log("pred_boxes shape:", outputs.pred_boxes.shape);

    // Clean up
    outputs.logits.dispose();
    outputs.pred_boxes.dispose();
    onnxModel.dispose();
  }
</script>

<main class="p-4">
  <button onclick={loadAndRun} class="border px-2">
    Load & Run DETR ResNet-50
  </button>
</main>
