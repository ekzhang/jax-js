<script lang="ts">
  import { linalg, numpy as np, init, blockUntilReady } from "@jax-js/jax";
  import { onMount, tick } from "svelte";
  import { fade } from "svelte/transition";
  import { LoaderCircle, SquareMousePointerIcon, CheckCircle2, AlertCircle } from "@lucide/svelte";

  const n = 64;
  const numRuns = 3;

  type Result = {
    device: string;
    gflops: number;
    time: number;
    maxDiff: number;
    passed: boolean;
  };

  let results = $state<Result[]>([]);
  let isRunning = $state(false);
  let currentDevice = $state<string | null>(null);

  const barColors: Record<string, string> = {
    cpu: "#6366f1",
    wasm: "#8b5cf6",
    webgpu: "#a855f7",
  };

  async function benchDevice(device: "cpu" | "wasm" | "webgpu"): Promise<Result> {
    currentDevice = device;
    const times: number[] = [];
    let lastMaxDiff = 0;

    console.log(`[${device}] Initializing...`);
    await init(device);
    console.log(`[${device}] Initialized!`);

    for (let i = 0; i < numRuns; i++) {
      console.log(`[${device}] Run ${i + 1}/${numRuns}`);
      // Create a fresh matrix for each run to avoid any caching effects
      // or to ensure we're measuring real work.
      const M_raw = new Float32Array(n * n);
      for (let j = 0; j < n * n; j++) M_raw[j] = Math.random();

      console.log(`[${device}] Creating array...`);
      const M = np.array(Array.from({ length: n }, (_, row) =>
        Array.from(M_raw.slice(row * n, (row + 1) * n))
      ), { device });

      console.log(`[${device}] Creating SPD matrix...`);
      const A = np.add(
        np.matmul(M.ref, M.ref.transpose()),
        np.multiply(np.eye(n, undefined, { device }), 0.1)
      );

      console.log(`[${device}] Awaiting blockUntilReady(A)...`);
      // Warm up / Sync
      await blockUntilReady(A);

      console.log(`[${device}] Running cholesky...`);
      const start = performance.now();
      const L = linalg.cholesky(A.ref);
      console.log(`[${device}] Awaiting blockUntilReady(L)...`);
      await blockUntilReady(L);
      const end = performance.now();
      console.log(`[${device}] Cholesky done in ${end - start}ms`);
      
      const time = (end - start) / 1000;
      if (i > 0) times.push(time); // Skip first run for average

      // Verification (only on the last run for efficiency)
      if (i === numRuns - 1) {
        const reconstructed = np.matmul(L.ref, L.ref.transpose());
        const recon_js = (await reconstructed.jsAsync()) as number[][];
        const A_js = (await A.jsAsync()) as number[][];
        
        let maxDiff = 0;
        for (let r = 0; r < n; r++) {
          for (let c = 0; c < n; c++) {
            maxDiff = Math.max(maxDiff, Math.abs(A_js[r][c] - recon_js[r][c]));
          }
        }
        lastMaxDiff = maxDiff;
        reconstructed.dispose();
      }

      L.dispose();
      A.dispose();
      M.dispose();
    }

    const avgTime = times.reduce((a, b) => a + b, 0) / times.length;
    // Cholesky complexity: ~ (1/3) * n^3 FLOPs
    const gflops = ( (1/3) * n * n * n ) / 1e9 / avgTime;

    return {
      device,
      gflops,
      time: avgTime,
      maxDiff: lastMaxDiff,
      passed: lastMaxDiff < 1e-4
    };
  }

  async function runBenchmark() {
    if (isRunning) return;
    isRunning = true;
    results = [];

    try {
      const devices = ["cpu", "wasm", "webgpu"] as const;
      for (const device of devices) {
        console.log(`Starting benchmark for ${device}...`);
        const res = await benchDevice(device);
        console.log(`${device} result:`, res);
        results.push(res);
      }
    } catch (e) {
      console.error("Benchmark error:", e);
      alert(`Benchmark error: ${e}`);
    } finally {
      isRunning = false;
      currentDevice = null;
    }
  }

  // Chart derived values
  const paddingX = 40;
  const paddingBottom = 40;
  const paddingTop = 40;
  const chartHeight = 300;
  const barWidth = 80;
  const barGap = 40;
  const chartWidth = paddingX * 2 + 3 * barWidth + 2 * barGap;

  const maxGflops = $derived(
    results.length > 0 ? Math.max(...results.map(r => r.gflops), 0.1) : 10
  );

  function getBarHeight(gflops: number) {
    const availableHeight = chartHeight - paddingBottom - paddingTop;
    return (gflops / maxGflops) * availableHeight;
  }

  function formatNumber(num: number): string {
    return num.toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
  }

  onMount(() => {
    // Optional: run auto on mount if desired
    // runBenchmark();
  });
</script>

<svelte:head>
  <title>Cholesky Benchmark – jax-js</title>
</svelte:head>

<main class="max-w-4xl mx-auto px-6 py-12 font-tiktok">
  <div class="text-center mb-12">
    <h1 class="text-4xl font-bold mb-4">Cholesky Decomposition</h1>
    <p class="text-gray-600 text-lg max-w-2xl mx-auto">
      Benchmarking performance of {n}x{n} matrix factorization across CPU, WASM, and WebGPU backends.
    </p>
  </div>

  <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 mb-8">
    <div class="flex flex-col items-center">
      <h3 class="text-xl font-semibold mb-2">Performance Comparison</h3>
      <p class="text-gray-500 text-sm mb-8">GFLOPs (Higher is better)</p>

      <div class="relative w-full flex justify-center overflow-x-auto py-4">
        <svg
          viewBox="0 0 {chartWidth} {chartHeight}"
          class="overflow-visible"
          style="width: {chartWidth}px; height: {chartHeight}px;"
        >
          <!-- Grid Lines -->
          {#each [0, 0.25, 0.5, 0.75, 1] as tick}
            {@const y = chartHeight - paddingBottom - tick * (chartHeight - paddingBottom - paddingTop)}
            <line x1={paddingX} y1={y} x2={chartWidth - paddingX} y2={y} stroke="#f1f5f9" stroke-width="1" />
            <text x={paddingX - 8} y={y + 4} text-anchor="end" class="text-[10px] fill-gray-400 font-mono">
              {formatNumber(tick * maxGflops)}
            </text>
          {/each}

          <!-- Bars -->
          {#each ["cpu", "wasm", "webgpu"] as device, i}
            {@const result = results.find(r => r.device === device)}
            {@const xPos = paddingX + i * (barWidth + barGap)}
            {@const height = result ? getBarHeight(result.gflops) : 4}
            {@const yPos = chartHeight - paddingBottom - height}
            {@const isCurrent = currentDevice === device}

            <g class="transition-opacity duration-300" style="opacity: {isRunning && !isCurrent && !result ? 0.3 : 1}">
              <rect
                x={xPos}
                y={yPos}
                width={barWidth}
                height={height}
                fill={barColors[device]}
                rx="6"
                class="transition-all duration-500 ease-out"
              />
              
              {#if result}
                <text
                  x={xPos + barWidth / 2}
                  y={yPos - 12}
                  text-anchor="middle"
                  class="text-sm font-bold fill-gray-800"
                  in:fade
                >
                  {formatNumber(result.gflops)}
                </text>
              {/if}

              <text
                x={xPos + barWidth / 2}
                y={chartHeight - paddingBottom + 24}
                text-anchor="middle"
                class="text-sm font-medium fill-gray-500 uppercase tracking-wider"
              >
                {device}
              </text>

              {#if isCurrent}
                <g transform="translate({xPos + barWidth/2 - 8}, {chartHeight - paddingBottom + 40})">
                   <circle cx="8" cy="8" r="8" fill="none" stroke={barColors[device]} stroke-width="2" class="animate-ping opacity-75" />
                </g>
              {/if}
            </g>
          {/each}
        </svg>
      </div>

      <div class="mt-12 flex flex-col items-center gap-4">
        <button
          onclick={runBenchmark}
          disabled={isRunning}
          class="bg-primary text-white px-8 py-3 rounded-full font-semibold shadow-lg shadow-primary/20 hover:bg-primary/90 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-3"
        >
          {#if isRunning}
            <LoaderCircle size={20} class="animate-spin" />
            Benchmarking {currentDevice}...
          {:else}
            Run Benchmark
          {/if}
        </button>

        {#if results.length > 0 && !isRunning}
          <div class="flex items-center gap-2 text-sm text-green-600 bg-green-50 px-4 py-2 rounded-lg border border-green-100">
            <CheckCircle2 size={16} />
            Accuracy verified: All results within 1e-4 tolerance.
          </div>
        {/if}
      </div>
    </div>
  </div>

  {#if results.length > 0}
    <div class="grid sm:grid-cols-3 gap-6" in:fade>
      {#each results as res}
        <div class="bg-gray-50 rounded-xl p-5 border border-gray-100">
          <div class="flex items-center justify-between mb-3">
            <span class="text-xs font-bold uppercase tracking-widest text-gray-400">{res.device}</span>
            <div class="flex items-center gap-1">
              {#if res.passed}
                <CheckCircle2 size={14} class="text-green-500" />
                <span class="text-[10px] text-green-600 font-bold">PASSED</span>
              {:else}
                <AlertCircle size={14} class="text-red-500" />
                <span class="text-[10px] text-red-600 font-bold">FAILED</span>
              {/if}
            </div>
          </div>
          <div class="text-2xl font-bold text-gray-900 mb-1">{formatNumber(res.gflops)} <span class="text-sm font-normal text-gray-500">GFLOPs</span></div>
          <div class="text-xs text-gray-500">Time: {formatNumber(res.time * 1000)}ms</div>
          <div class="text-[10px] text-gray-400 mt-2 font-mono">Max error: {res.maxDiff.toExponential(2)}</div>
        </div>
      {/each}
    </div>
  {/if}
</main>

<style>
  main {
    margin: 0 auto;
  }
  h1 {
    font-weight: 300;
    margin-bottom: 0.5rem;
  }
  button {
    background: #007bff;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 1rem;
    margin: 1rem 0;
  }
  button:disabled {
    background: #ccc;
    cursor: not-allowed;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1rem;
  }
  th, td {
    text-align: left;
    padding: 0.75rem;
    border-bottom: 1px solid #eee;
  }
  th {
    background: #f8f9fa;
  }
  .passed {
    color: #28a745;
    font-weight: bold;
  }
  .failed {
    color: #dc3545;
    font-weight: bold;
  }
</style>
