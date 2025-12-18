<script lang="ts">
  import { LoaderCircle, SquareMousePointerIcon } from "@lucide/svelte";
  import { onMount } from "svelte";
  import { fade } from "svelte/transition";

  interface PerfResults {
    flops: {
      Wasm: number;
      WebGPU: number;
      "WebGPU-fp16": number | undefined;
    };
    browser: string;
    live: boolean;
  }

  // Fall back to these if
  const ericLaptopResults: PerfResults = {
    flops: {
      Wasm: 2.72,
      WebGPU: 2071,
      "WebGPU-fp16": 3343,
    },
    browser: "Chrome on Apple M3 Pro",
    live: false,
  };

  let results = $state<PerfResults | null>(null);

  async function benchFlops(
    n: number,
    device: any,
    dtype: any,
  ): Promise<number> {
    try {
      const jax = await import("@jax-js/jax");
      await jax.init(device);
      jax.defaultDevice(device);

      const np = jax.numpy;

      const measurements: number[] = [];

      // 1 warmup + 2 timed runs
      for (let i = 0; i < 3; i++) {
        const key = jax.random.key(0);
        const [k1, k2] = jax.random.split(key, 2);

        const A = jax.random.uniform(k1, [n, n]).astype(dtype);
        const B = jax.random.uniform(k2, [n, n]).astype(dtype);
        await jax.blockUntilReady([A, B]);

        const start = performance.now();
        const C = np.matmul(A, B);
        await jax.blockUntilReady(C);
        C.dispose();
        const end = performance.now();

        if (i > 0) {
          measurements.push((end - start) / 1000);
        }
      }

      const gflops = (2 * n * n * n) / 1e9;
      const seconds =
        measurements.reduce((a, b) => a + b, 0) / measurements.length;

      return gflops / seconds;
    } catch (error: any) {
      console.error("Benchmark error:", error);
      return 0;
    }
  }

  async function measurePerf(): Promise<PerfResults> {
    // See if the current browser supports WebGPU.
    if (!navigator?.gpu?.requestAdapter) {
      // Fall back to laptop results if WebGPU is not supported.
      return ericLaptopResults;
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return ericLaptopResults;
    const hasF16 = adapter.features.has("shader-f16");

    const isApple = /iPad|iPhone|iPod|Macintosh/.test(navigator.userAgent);
    const isMobile = /Mobi/.test(navigator.userAgent);
    const isNvidia = /NVIDIA/i.test(adapter.info.vendor);
    const isGoodGpu = /NVIDIA|AMD|Qualcomm|ARM/i.test(adapter.info.vendor); // "Good" GPUs, probably won't crash

    // Large matmuls put pressure on mobile browsers like iOS Safari, and it can
    // lead to page crashes. Also the measured FLOPs is lower.
    //
    // It also looks like some Windows PCs have trouble with GPU, keeping things
    // suitably small to really avoid accidentally crashing someone's browser on
    // page load. Nvidia GPUs are definitely fine though.
    //
    // Trying to get as good of a first impression as possible while not
    // stressing out people's computers too much.
    //
    // References:
    // - https://github.com/gpuweb/gpuweb/blob/WGSLv1/design/AdapterIdentifiers.md
    // - https://github.com/gpuweb/gpuweb/pull/2660
    // - https://github.com/milhidaka/webgpu-blas
    // - https://github.com/chromium/chromium/blob/145.0.7583.2/content/utility/on_device_model/on_device_model_sandbox_init.cc#L41-L44
    // - https://github.com/ProfSynapse/nexus/blob/3.2.0/src/services/llm/adapters/webllm/WebLLMVRAMDetector.ts
    // - https://github.com/1karess/wasm-fingerprint/blob/main/src/webgpu-detection.js#L726-L740
    let gpuDim: number;
    if (isApple) {
      gpuDim = isMobile ? 2048 : 4096;
    } else if (isNvidia) {
      gpuDim = 4096;
    } else if (isGoodGpu) {
      gpuDim = isMobile ? 1024 : 2048;
    } else {
      gpuDim = 1024;
    }

    return {
      flops: {
        Wasm: await benchFlops(128, "wasm", "float32"),
        WebGPU: await benchFlops(gpuDim, "webgpu", "float32"),
        "WebGPU-fp16": hasF16
          ? await benchFlops(gpuDim, "webgpu", "float16")
          : undefined,
      },
      browser: "Your browser (live)",
      live: true,
    };
  }

  let measuring = $state(false);

  async function measurementTask() {
    if (measuring) return;
    measuring = true;
    try {
      results = null;
      results = await measurePerf();
    } finally {
      measuring = false;
    }
  }

  onMount(() => {
    measurementTask();
  });

  // Bar chart configuration
  const barWidth = 80;
  const barGap = 24;
  const paddingX = 16;
  const paddingTop = 24;
  const paddingBottom = 28;

  const allBackends = ["Wasm", "WebGPU", "WebGPU-fp16"] as const;
  const barColors: Record<(typeof allBackends)[number], string> = {
    Wasm: "#6366f1",
    WebGPU: "#8b5cf6",
    "WebGPU-fp16": "#a855f7",
  };

  // Always render all backends for smooth transitions
  const chartWidth =
    paddingX * 2 +
    allBackends.length * barWidth +
    (allBackends.length - 1) * barGap;
  const chartHeight = 220;

  // Calculate max value for scaling
  const maxFlops = $derived(
    results
      ? Math.max(
          ...allBackends.map((b) => results!.flops[b]).filter((v) => v != null),
        )
      : ericLaptopResults.flops["WebGPU-fp16"]!,
  );

  // Get bar height as a percentage (0-1)
  function getBarHeight(backend: (typeof allBackends)[number]): number {
    if (!results) return 0;
    const value = results.flops[backend];
    if (value == null) return 0;
    return value / maxFlops;
  }

  // Check if a backend is available
  function isAvailable(backend: (typeof allBackends)[number]): boolean {
    if (!results) return true; // Show all as placeholders before results
    return results.flops[backend] != null;
  }

  // Format number with commas
  function formatNumber(num: number): string {
    const fractionDigits = num < 10 ? 2 : num < 100 ? 1 : 0;
    return num.toLocaleString(undefined, {
      minimumFractionDigits: fractionDigits,
      maximumFractionDigits: fractionDigits,
    });
  }
</script>

<section class="flex flex-col justify-center items-center text-center">
  <h3 class="text-lg mb-1">Matrix multiplication</h3>
  <p class="text-gray-700 text-sm mb-6 max-w-[30ch]">
    Billions of floating-point operations (GFLOPs) per second
  </p>

  <!-- Bar chart -->
  <svg
    viewBox="0 0 {chartWidth} {chartHeight}"
    class="overflow-visible max-w-full"
    style="width: {chartWidth}px; height: auto;"
  >
    <!-- X-axis (subtle) -->
    <line
      x1={0}
      y1={chartHeight - paddingBottom}
      x2={chartWidth}
      y2={chartHeight - paddingBottom}
      stroke="#e2e8f0"
      stroke-width="1"
    />

    {#each allBackends as backend, i}
      {@const available = isAvailable(backend)}
      {@const xPos = paddingX + (barWidth + barGap) * i}
      {@const heightPercent = getBarHeight(backend)}
      {@const availableHeight = chartHeight - paddingBottom - paddingTop}
      {@const minBarHeight = 4}
      {@const barHeight = Math.max(
        heightPercent * availableHeight,
        results ? minBarHeight : 0,
      )}
      {@const yPos = chartHeight - paddingBottom - barHeight}
      {@const value = results?.flops[backend] ?? 0}

      <g
        style="opacity: {available ? 1 : 0}; transition: opacity 0.3s ease-in;"
      >
        <!-- Bar -->
        <rect
          x={xPos}
          y={yPos}
          width={barWidth}
          height={barHeight}
          fill={barColors[backend]}
          rx="4"
          style="transition: height 0.5s ease-out, y 0.5s ease-out;"
        />

        <!-- Value label on top of bar -->
        {#if results}
          <text
            x={xPos + barWidth / 2}
            y={yPos - 8}
            text-anchor="middle"
            class="text-sm font-semibold"
            fill="#1e293b"
            in:fade={{ delay: 200, duration: 300 }}
          >
            {formatNumber(value)}
          </text>
        {/if}

        <!-- Backend label below bar -->
        <text
          x={xPos + barWidth / 2}
          y={chartHeight - paddingBottom + 20}
          text-anchor="middle"
          class="text-xs"
          fill="#64748b"
        >
          {backend}
        </text>
      </g>
    {/each}
  </svg>

  <div
    class="flex items-center gap-2 mt-4 text-sm"
    class:animate-pulse={!results}
  >
    {#if !results}
      <LoaderCircle size={16} class="animate-spin text-gray-400" />
      <p class="text-gray-500">Running benchmark…</p>
    {:else}
      <button
        class="flex items-center gap-2"
        onclick={() => {
          if (results?.live) {
            measurementTask();
          }
        }}
        disabled={!results.live || measuring}
      >
        {#if results.live}
          <SquareMousePointerIcon size={16} class="text-gray-500" />
        {/if}
        <p class="text-gray-800">
          {results.browser}
        </p>
      </button>
    {/if}
  </div>
</section>
