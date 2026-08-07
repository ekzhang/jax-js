<script lang="ts">
  import { browser } from "$app/environment";
  import { resolve } from "$app/paths";

  import { ArrowUpRight, ExternalLinkIcon } from "@lucide/svelte";

  import logo from "$lib/assets/logo.svg";
  import DemoThumbnail from "$lib/homepage/DemoThumbnail.svelte";
  import MatmulPerfDemo from "$lib/homepage/MatmulPerfDemo.svelte";
  import EmbeddedRepl from "$lib/repl/EmbeddedRepl.svelte";

  const installText = {
    npm: `npm install @jax-js/jax`,
    web:
      `<` +
      String.raw`script type="module">
  import * as jax from "https://esm.sh/@jax-js/jax";
</script` +
      `>`,
  };

  let installMode = $state<"npm" | "web">("npm");

  const examples: {
    title: string;
    href: string;
    thumbnail: string;
    description: string;
  }[] = [
    {
      title: "LLM Chat",
      href: resolve("/chat"),
      thumbnail: "chat",
      description:
        "Run a small language model locally in your browser, with GPU and CPU-based inference.",
    },
    {
      title: "Neural Cellular Automata",
      href: resolve("/nca-growing"),
      thumbnail: "nca",
      description: "Grow image patterns with a tiny differentiable automaton.",
    },
    {
      title: "Heat Method Geodesics",
      href: resolve("/heat-method"),
      thumbnail: "heat",
      description:
        "Compute surface distances on triangle meshes with dense solves.",
    },
    {
      title: "Whisper ASR",
      href: resolve("/whisper"),
      thumbnail: "whisper",
      description: "Transcribe audio locally with OpenAI Whisper.",
    },
    {
      title: "Kyutai Pocket TTS",
      href: resolve("/tts"),
      thumbnail: "tts",
      description: "Voice cloning AI model that runs in your browser.",
    },
    {
      title: "Fluid Simulation",
      href: resolve("/fluid-sim"),
      thumbnail: "fluid",
      description: "Interactive Navier-Stokes flow with WebGPU kernels.",
    },
    {
      title: "D-FINE Detection",
      href: resolve("/d-fine"),
      thumbnail: "detection",
      description: "Run an ONNX object detector locally with WebGPU.",
    },
    {
      title: "MobileCLIP2 Inference",
      href: resolve("/mobileclip"),
      thumbnail: "mobileclip",
      description: "Compute embeddings for book passages.",
    },
    {
      title: "MNIST Training",
      href: resolve("/mnist"),
      thumbnail: "mnist",
      description: "Train a neural network on handwritten digits in-browser.",
    },
    {
      title: "Principal Component Analysis",
      href: resolve("/pca"),
      thumbnail: "pca",
      description: "Explore components of a 3D point cloud.",
    },
    {
      title: "Benchmarks",
      href: resolve("/bench/matmul"),
      thumbnail: "benchmarks",
      description: "Compare jax-js kernels with other web ML libraries.",
    },
  ];

  const resourceLinks: {
    title: string;
    href: string;
    description: string;
  }[] = [
    {
      title: "GitHub Repository",
      href: "https://github.com/ekzhang/jax-js",
      description: "Check out the code and tutorial.",
    },
    {
      title: "REPL",
      href: resolve("/repl"),
      description: "Try jax-js in this browser-based REPL.",
    },
    {
      title: "API Reference",
      href: "https://jax-js.com/docs/",
      description: "View the generated API documentation.",
    },
  ];
</script>

<svelte:head>
  <title>jax-js – ML for the web</title>
</svelte:head>

<!-- Header -->
<header
  class="px-6 py-4 flex items-center justify-between max-w-screen-xl mx-auto font-tiktok gap-6"
>
  <div class="flex items-center gap-3 shrink-0">
    <a href={resolve("/")}>
      <img src={logo} alt="jax-js logo" class="h-8" />
    </a>
  </div>
  <nav class="flex items-center gap-6">
    <a href={resolve("/repl")} class="hidden sm:block hover:text-primary"
      >REPL</a
    >
    <a rel="external" href="https://jax-js.com/docs/" class="hover:text-primary"
      >Docs</a
    >
    <a
      href="https://github.com/ekzhang/jax-js"
      target="_blank"
      class="bg-primary/15 hover:bg-primary/25 px-4 py-1 rounded-full"
    >
      GitHub
      <ExternalLinkIcon size={16} class="inline-block mb-1 ml-0.5 opacity-60" />
    </a>
  </nav>
</header>

<main class="font-tiktok">
  <!-- Hero section -->
  <section class="px-6 py-14 md:py-20 max-w-screen-xl mx-auto">
    <div class="grid md:grid-cols-[5fr_3fr] gap-x-12 gap-y-16">
      <div class="lg:py-8">
        <h1 class="text-3xl sm:text-4xl mb-6 leading-tight max-w-2xl">
          jax-js is <span class="hidden sm:inline">a machine learning</span
          ><span class="sm:hidden">an ML</span> library and compiler for the web
        </h1>
        <p class="text-lg text-gray-700 leading-snug mb-8 max-w-2xl">
          High-performance WebGPU and WebAssembly kernels in JavaScript. Run
          neural networks, image algorithms, simulations, and numerical code,
          all JIT compiled in your browser.
        </p>

        <!-- Add to project box -->
        <div class="bg-primary/5 rounded-lg p-4">
          <h2 class="text-xl font-medium mb-1.5">Add jax-js to your project</h2>
          <p class="text-gray-600 text-sm mb-4">
            Zero dependencies. All major browsers, with <button
              class="enabled:underline"
              onclick={() => (installMode = "npm")}
              disabled={installMode === "npm"}>bundlers</button
            >
            and in
            <button
              class="enabled:underline"
              onclick={() => (installMode = "web")}
              disabled={installMode === "web"}>JS modules</button
            >.
          </p>
          <div
            class="bg-primary/5 border-1 border-primary rounded-lg px-3 py-2 font-mono whitespace-pre-wrap"
          >
            <span
              class="text-primary/50 select-none"
              class:hidden={installMode === "web"}>&gt;&nbsp;</span
            >{installText[installMode]}
          </div>
        </div>
      </div>

      <!-- Performance Chart -->
      <MatmulPerfDemo />
    </div>
  </section>

  <!-- Explainer section -->
  <section class="mx-auto max-w-screen-xl my-8 sm:px-6 hidden">
    <div class="sm:rounded-xl bg-primary/5 px-8 py-8">
      <div class="mx-auto max-w-2xl">
        <h2 class="text-xl font-medium text-center mb-6">
          Like JAX and PyTorch in your browser
        </h2>

        <p class="mb-6">
          jax-js is a end-to-end ML library inspired by JAX, but in pure
          JavaScript:
        </p>

        <ul
          class="space-y-2 pl-4 mb-6 list-disc list-inside marker:text-gray-400"
        >
          <li>Runs completely client-side (Chrome, Firefox, iOS, Android).</li>
          <li>
            Has close <a
              href="https://github.com/ekzhang/jax-js/blob/main/FEATURES.md"
              target="_blank"
              class="underline hover:text-primary">API compatibility</a
            > with NumPy/JAX.
          </li>
          <li>Is written from scratch, with zero external dependencies.</li>
        </ul>

        <p class="mb-6">
          jax-js is likely the most portable GPU ML framework, since it runs
          anywhere a browser can run. It's also simple but optimized, including
          a lightweight compiler that translates your high-level operations into
          WebGPU and WebAssembly kernels.
        </p>

        <p>
          The goal of jax-js is to make numerical code accessible and deployable
          to everyone, so compute-intensive apps can run fast and locally on
          consumer hardware.
        </p>
      </div>
    </div>
  </section>

  <!-- Live Editor section -->
  <section class="px-6 py-8 max-w-screen-xl mx-auto">
    <h2 class="text-xl mb-2">Try it out!</h2>

    <p class="mb-4 text-sm text-gray-600">
      This is a live editor, the code is running in your browser{browser &&
      navigator.gpu
        ? " with WebGPU"
        : ""}.
    </p>

    <EmbeddedRepl
      initialText={String.raw`import { grad, numpy as np, vmap } from "@jax-js/jax";

const f = (x: np.Array) => np.sqrt(x.ref.mul(x).sum());
const x = np.array([1, 2, 3, 4]);

console.log(f(x.ref));
console.log(grad(f)(x.ref));
console.log(vmap(grad(np.square))(x));
`}
    />
  </section>

  <!-- Demo gallery section -->
  <section class="px-6 py-16 max-w-screen-xl mx-auto">
    <div class="mb-6 max-w-2xl">
      <h2 class="text-xl mb-1">Live examples</h2>
      <p class="text-sm text-gray-600">
        Interactive demos for local AI models, simulations, geometry processing,
        and numerical methods.
      </p>
    </div>

    <div class="grid sm:grid-cols-2 lg:grid-cols-3 gap-5">
      {#each examples as demo}
        <a href={demo.href} class="demo-card group">
          <div
            class="relative overflow-hidden border-b border-black/10 bg-primary/5 h-36 sm:h-40"
            aria-hidden="true"
          >
            <DemoThumbnail name={demo.thumbnail} />
          </div>

          <div class="p-4">
            <h3 class="text-lg mb-1 flex items-start justify-between gap-3">
              <span>{demo.title}</span>
              <ArrowUpRight
                size={18}
                class="text-gray-400 mt-1 shrink-0 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5"
              />
            </h3>
            <p class="text-sm text-gray-600 leading-snug">
              {demo.description}
            </p>
          </div>
        </a>
      {/each}
    </div>
  </section>

  <!-- Resource section -->
  <section class="px-6 pb-16 max-w-screen-xl mx-auto">
    <h2 class="text-xl mb-4">Resources</h2>
    <div class="grid md:grid-cols-3 gap-5">
      {#each resourceLinks as { title, href, description }}
        <a
          {href}
          class="block rounded-lg border border-black/10 bg-primary/5 px-4 py-3 transition-colors hover:border-primary/40 hover:bg-primary/15"
        >
          <h3 class="mb-1 text-base">
            {title}
            <ArrowUpRight size={17} class="inline-block text-gray-400 mb-px" />
          </h3>
          <p class="text-sm leading-snug text-gray-600">{description}</p>
        </a>
      {/each}
    </div>
  </section>
</main>

<style lang="postcss">
  @reference "$app.css";

  .demo-card {
    @apply block overflow-hidden rounded-lg border border-black/10 bg-white transition-colors hover:border-primary/45 hover:bg-primary/5;
  }
</style>
