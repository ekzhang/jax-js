<script lang="ts">
  import BenchmarkPage from "../BenchmarkPage.svelte";
  import {
    batchSize,
    channels,
    filterHeight,
    filterWidth,
    height,
    outChannels,
    strategies,
    width,
  } from "./strategies";

  const summary = `Benchmarking fp32 conv2d kernels on a ${batchSize}×${channels}×${height}×${width} input with ${outChannels} filters of size ${filterHeight}×${filterWidth}.`;

  const notes = [
    '"naive" is a simple nested-loop WebGPU kernel.',
    '"onnx" runs a Conv operator in onnxruntime-web.',
    '"tfjs" runs tf.conv2d() with NHWC format.',
    '"jax-js" runs jax.lax.convGeneralDilated().',
  ];

  function formatResult(time: number) {
    const flops =
      2 *
      batchSize *
      outChannels *
      channels *
      height *
      width *
      filterHeight *
      filterWidth;
    return `${(flops / 1e9 / time).toFixed(2)} GFLOP/s`;
  }
</script>

<BenchmarkPage
  title="Conv2d benchmark"
  {summary}
  {notes}
  {strategies}
  {formatResult}
/>
