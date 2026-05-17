<script lang="ts">
  import { resolve } from "$app/paths";
  import { page } from "$app/state";

  let { children } = $props();

  const links = [
    {
      title: "Matmul",
      href: resolve("/bench/matmul"),
      detail: "4096×4096 matrix multiply",
    },
    {
      title: "Matvec",
      href: resolve("/bench/matvec"),
      detail: "4096×4096 by vector",
    },
    {
      title: "Conv2d",
      href: resolve("/bench/conv2d"),
      detail: "NCHW 3×3 convolution",
    },
  ];

  function isActive(href: string) {
    return page.url.pathname === href || page.url.pathname === `${href}/`;
  }
</script>

<div class="bench-shell min-h-screen font-tiktok text-zinc-950">
  <div class="mx-auto grid max-w-screen-2xl md:grid-cols-[17rem_minmax(0,1fr)]">
    <aside
      class="border-b border-zinc-950/10 bg-stone-100/90 md:sticky md:top-0 md:min-h-screen md:border-b-0 md:border-r"
    >
      <div class="p-5 md:p-6">
        <a
          href={resolve("/")}
          class="mb-4 inline-flex text-lg font-medium tracking-[-0.03em] text-zinc-950 hover:text-primary"
        >
          jax-js
        </a>

        <p class="text-sm leading-snug text-zinc-600">
          Benchmarks of handwritten WebGPU kernels against jax-js, tfjs, and
          onnxruntime.
        </p>

        <nav class="mt-6 space-y-px" aria-label="Benchmark pages">
          {#each links as link}
            <a
              href={link.href}
              class="bench-link group block border-l-2 border-transparent px-3 py-3 text-sm transition hover:border-zinc-400 hover:bg-white/50"
              class:active={isActive(link.href)}
            >
              <span class="block font-medium">{link.title}</span>
              <span class="block text-xs leading-tight text-zinc-500"
                >{link.detail}</span
              >
            </a>
          {/each}
        </nav>
      </div>
    </aside>

    <main class="min-w-0 p-4 sm:p-6 lg:p-8">
      {@render children()}
    </main>
  </div>
</div>

<style lang="postcss">
  @reference "$app.css";

  .bench-shell {
    background-color: #f7f6ef;
    background-image:
      linear-gradient(to right, rgb(24 24 27 / 0.045) 1px, transparent 1px),
      linear-gradient(to bottom, rgb(24 24 27 / 0.045) 1px, transparent 1px);
    background-size: 32px 32px;
  }

  .bench-link.active {
    @apply border-primary bg-white text-zinc-950;
  }
</style>
