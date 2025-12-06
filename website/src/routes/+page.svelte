<script lang="ts">
  import { browser } from "$app/environment";
  import { resolve } from "$app/paths";

  import { ArrowUpRight } from "lucide-svelte";

  import logo from "$lib/assets/logo.svg";
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
    <a
      href="https://github.com/ekzhang/jax-js"
      target="_blank"
      class="hover:text-primary">GitHub</a
    >
    <a href={resolve("/repl")} class="hidden sm:block hover:text-primary"
      >REPL</a
    >
    <a
      rel="external"
      target="_blank"
      href="https://www.ekzhang.com/jax-js/docs/"
      class="hover:text-primary transition-colors">Reference</a
    >
  </nav>
</header>

<main class="font-tiktok">
  <!-- Hero section -->
  <section class="px-6 py-12 md:py-16 max-w-screen-xl mx-auto">
    <div class="grid md:grid-cols-[5fr_3fr] gap-x-12 gap-y-16 items-center">
      <div>
        <h1 class="text-3xl sm:text-4xl mb-6 leading-tight max-w-2xl">
          jax-js is <span class="hidden sm:inline">a machine learning</span
          ><span class="sm:hidden">an ML</span> library and compiler for the web
        </h1>
        <p class="text-lg text-gray-700 leading-snug mb-8 max-w-2xl">
          High-performance WebGPU and WebAssembly kernels in JavaScript. Run AI
          training and inference, image algorithms, simulations, and numerical
          code on arrays, all JIT compiled in your browser.
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
      <section class="flex flex-col justify-center items-center text-center">
        <h3 class="text-lg mb-1">Matrix multiplication</h3>
        <p class="text-gray-700 text-sm mb-8 max-w-[32ch]">
          Billions of floating point operations (GFLOPs) per second
        </p>
        <!-- Bar chart -->
        <div
          class="h-60 w-80 border bg-red-50 p-8 flex justify-center items-center text-red-500 text-sm"
        >
          widget / perf demo under construction
        </div>
      </section>
    </div>
  </section>

  <!-- Explainer section -->
  <section class="mx-auto max-w-screen-xl my-8 sm:px-6">
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
  <section class="px-6 py-12 max-w-screen-xl mx-auto">
    <h2 class="text-xl mb-2">Try it out!</h2>

    <p class="mb-4 text-sm text-gray-600">
      This is a live editor, the code is running in your browser{browser &&
      navigator.gpu
        ? " with WebGPU"
        : ""}.
    </p>

    <EmbeddedRepl
      initialText={String.raw`import { grad, numpy as np } from "@jax-js/jax";

const f = (x: np.Array) => np.sqrt(x.ref.mul(x).sum());
const df = grad(f);

const x = np.array([1, 2, 3, 4]);
console.log(f(x.ref).js());
console.log(df(x).js());
`}
    />
  </section>

  <!-- Learn More section -->
  <section class="px-6 py-16 max-w-screen-xl mx-auto">
    <h2 class="text-xl mb-6">Learn more</h2>

    <div class="grid sm:grid-cols-3 gap-x-6 md:gap-x-8 gap-y-4">
      <a
        href="https://github.com/ekzhang/jax-js"
        class="bg-primary/5 hover:bg-primary/15 transition-colors p-4 rounded-lg"
      >
        <h3 class="mb-2">
          GitHub Repository <ArrowUpRight
            size={18}
            class="inline-block text-gray-400 mb-px"
          />
        </h3>
        <p class="text-sm text-gray-600">
          Get started with jax-js and check out the code.
        </p>
      </a>

      <a
        href={resolve("/mnist")}
        class="bg-primary/5 hover:bg-primary/15 transition-colors p-4 rounded-lg"
      >
        <h3 class="mb-2">
          MNIST Training <ArrowUpRight
            size={18}
            class="inline-block text-gray-400 mb-px"
          />
        </h3>
        <p class="text-sm text-gray-600">
          Demo of training a neural network on MNIST.
        </p>
      </a>

      <a
        href={resolve("/repl")}
        class="bg-primary/5 hover:bg-primary/15 transition-colors p-4 rounded-lg"
      >
        <h3 class="mb-2">
          REPL <ArrowUpRight
            size={18}
            class="inline-block text-gray-400 mb-px"
          />
        </h3>
        <p class="text-sm text-gray-600">
          Try out the library in your browser.
        </p>
      </a>
    </div>
  </section>
</main>
