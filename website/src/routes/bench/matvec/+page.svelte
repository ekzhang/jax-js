<script lang="ts">
  import BenchmarkPage from "../BenchmarkPage.svelte";
  import { n, repeats, strategies } from "./strategies";

  const summary = `Benchmarking fp32 matrix-vector kernels on a ${n}×${n} matrix, repeated ${repeats}× per timed run to amortize dispatch and readback overhead.`;

  const notes = [
    '"naive" assigns one thread to each output row.',
    '"unroll" assigns one thread per output row with inner-loop unrolling.',
    '"shmem" reuses tiles of the input vector across multiple rows in a workgroup.',
    '"onnx" runs a column-vector MatMul in onnxruntime-web.',
    '"tfjs" and "jax-js" run the corresponding framework dot or matMul kernels.',
  ];

  function formatResult(time: number) {
    const gflops = (repeats * 2 * n * n) / 1e9 / time;
    return `${gflops.toFixed(2)} GFLOP/s`;
  }
</script>

<BenchmarkPage
  title="Matvec benchmark"
  {summary}
  {notes}
  {strategies}
  {formatResult}
/>
