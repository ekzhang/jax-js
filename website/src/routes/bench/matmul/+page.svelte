<script lang="ts">
  import BenchmarkPage from "../BenchmarkPage.svelte";
  import { n, strategies } from "./strategies";

  const summary = `Benchmarking fp32 matmul kernels on ${n}×${n} matrices.`;

  const notes = [
    '"naive" does a simple loop reduction with a WebGPU block size.',
    '"shmem-tiling" uses tiled reduction with workgroup memory.',
    '"unroll4" has each thread compute a 4×4 output block.',
    '"unroll4x2" and "unroll4x4" add inner-loop unrolling.',
    '"onnx", "tfjs", and "jax-js" run the corresponding framework kernels.',
  ];

  function formatResult(time: number) {
    return `${((2 * n * n * n) / 1e9 / time).toFixed(2)} GFLOP/s`;
  }
</script>

<BenchmarkPage
  title="Matmul benchmark"
  {summary}
  {notes}
  {strategies}
  {formatResult}
/>
