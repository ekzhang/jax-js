<script lang="ts">
  import { base } from "$app/paths";
</script>

<svelte:head>
  <title>jax-js</title>
</svelte:head>

<main class="mx-auto my-12 px-4 sm:px-6 max-w-screen-md">
  <div class="mb-10">
    <h1 class="text-6xl text-center font-bold mb-4">jax-js</h1>
    <p class="italic text-center">Numerical foundations for the web</p>
  </div>

  <p class="mb-6">
    Have you ever wanted to use NumPy or PyTorch in your browser?
  </p>

  <pre class="mb-6 text-center"><code>npm install @jax-js/core</code></pre>

  <p class="mb-6">
    This is a machine learning framework that brings high-performance,
    <a href="https://docs.jax.dev/en/latest/" target="_blank" class="link"
      >JAX</a
    >-style Wasm and WebGPU kernels to JavaScript. Run neural networks, image
    algorithms, simulations, and all sorts of other numerical applications
    without leaving the browser.
  </p>

  <div class="mb-6 grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-4">
    <!-- TODO: Images -->
    <div class="aspect-video bg-gray-500">
      <span class="text-white text-xs italic">julia set</span>
    </div>
    <div class="aspect-video bg-gray-500">
      <span class="text-white text-xs italic">mnist</span>
    </div>
    <div class="aspect-video bg-gray-500">
      <span class="text-white text-xs italic">llama chatbot</span>
    </div>
    <div class="aspect-video bg-gray-500">
      <span class="text-white text-xs italic">bayesian regression</span>
    </div>
  </div>

  <p class="mb-6">
    <code>jax-js</code> is likely the most portable ML framework, since it runs
    anywhere a browser can run (Chrome, Firefox, Safari, iOS, and Android). It's
    also simple but optimized, including a JIT compiler and GPU kernel scheduler
    based on a pared-down version of
    <a href="https://github.com/tinygrad/tinygrad" target="_blank" class="link"
      >tinygrad</a
    >.
  </p>

  <div class="mb-6 border h-48 flex items-center justify-center square-grid">
    <p class="text-sm italic text-center">
      <!-- TODO: Insert graph -->
      Performance graph of flops: cpu, wasm, webgl, webgpu
    </p>
  </div>

  <p class="mb-6">
    The built-in compiler lets it achieve best-in-class performance on the web.
    <code>jax-js</code> compiles math into optimized WebAssembly, WebGL, and
    WebGPU code on modern browsers. Its generated code is multiple times faster
    than other browser ML frameworks like
    <a href="https://www.tensorflow.org/js/" target="_blank" class="link"
      >TensorFlow.js</a
    > (Google).
  </p>

  <p class="mb-6">
    <span class="italic"
      >XXX Performance benchmarks coming soon—though see
      <a href="{base}/matmul" class="link">here</a> for some initial results.</span
    >
  </p>

  <p class="mb-6">
    NumPy, PyTorch, and JAX revolutionized numerical computing in Python.
    <code>jax-js</code> aspires to provide that same general foundation for JavaScript—the
    world's largest software platform.
  </p>

  <p>
    It's still in development, but you can try it in the
    <a href="{base}/repl" class="link">jax-js REPL</a>.
  </p>
</main>

<style lang="postcss">
  @reference "$app.css";

  .square-grid {
    background-size: 40px 40px;
    background-image:
      linear-gradient(to right, var(--color-gray-200) 1px, transparent 1px),
      linear-gradient(to bottom, var(--color-gray-200) 1px, transparent 1px);
    background-position: center center;
  }

  a.link {
    @apply underline;
  }
</style>
