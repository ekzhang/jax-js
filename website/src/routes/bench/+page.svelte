<script lang="ts">
  import { resolve } from "$app/paths";

  const links = [
    {
      title: "Matmul",
      href: resolve("/bench/matmul"),
      description:
        "Square matrix multiplication across WebGPU kernels and ML frameworks.",
    },
    {
      title: "Matvec",
      href: resolve("/bench/matvec"),
      description:
        "Matrix-vector multiplication repeated to amortize dispatch overhead.",
    },
    {
      title: "Conv2d",
      href: resolve("/bench/conv2d"),
      description: "2D convolution in NCHW layout with framework comparisons.",
    },
  ];
</script>

<svelte:head>
  <title>Benchmarks – jax-js</title>
</svelte:head>

<section class="mx-auto max-w-6xl border border-zinc-950/10 bg-stone-50/95">
  <div class="border-b border-zinc-950 p-5 sm:p-6">
    <h1 class="max-w-3xl text-4xl font-semibold text-zinc-950 sm:text-5xl">
      Select a workload
    </h1>
    <p class="mt-4 max-w-2xl text-base leading-relaxed text-zinc-600">
      Run benchmark variants in your browser and compare custom WebGPU kernels
      against ONNX Runtime, TensorFlow.js, and jax-js.
    </p>
  </div>

  <div
    class="grid divide-y divide-zinc-950/10 md:grid-cols-3 md:divide-x md:divide-y-0"
  >
    {#each links as link, i}
      <a href={link.href} class="group block p-5 transition hover:bg-white">
        <p class="font-mono text-xs text-zinc-400 group-hover:text-primary">
          {String(i + 1).padStart(2, "0")}
        </p>
        <h2 class="mt-6 font-medium text-zinc-950">{link.title}</h2>
        <p class="mt-2 text-sm leading-snug text-zinc-600">
          {link.description}
        </p>
      </a>
    {/each}
  </div>
</section>
