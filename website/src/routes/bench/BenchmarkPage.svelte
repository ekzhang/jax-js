<script lang="ts">
  import { browser } from "$app/environment";

  import type { Strategy } from "$lib/benchmark";

  let {
    title,
    summary,
    notes,
    strategies,
    formatResult,
  }: {
    title: string;
    summary: string;
    notes: string[];
    strategies: Strategy[];
    formatResult: (time: number) => string;
  } = $props();

  let results: Record<string, number> = $state({});
  let running: string | null = $state(null);
  let runningAll = $state(false);
  let error: string | null = $state(null);

  function sortedResults() {
    return Object.entries(results).sort(([, a], [, b]) => a - b);
  }

  function throughputBarWidth(time: number) {
    const fastest = sortedResults()[0]?.[1];
    if (!fastest || time <= 0) return "0%";
    return `${Math.min(100, (fastest / time) * 100)}%`;
  }

  function formatTime(time: number) {
    if (time < 1) return `${(time * 1000).toFixed(0)} ms`;
    return `${time.toFixed(3)} s`;
  }

  async function bench(strategy: Strategy) {
    if (running) return;

    console.log(`Running ${strategy.name}...`);
    running = strategy.name;
    error = null;

    try {
      await strategy.run(); // warmup
      const time = await strategy.run();
      if (time >= 0) {
        results[strategy.name] = time;
      } else {
        error = `Error running ${strategy.name}`;
      }
    } catch (cause) {
      console.error(`Error running ${strategy.name}`, cause);
      error = cause instanceof Error ? cause.message : String(cause);
    } finally {
      running = null;
    }
  }

  async function runAll() {
    if (running) return;
    runningAll = true;
    try {
      for (const strategy of strategies) {
        await bench(strategy);
      }
    } finally {
      runningAll = false;
    }
  }
</script>

<svelte:head>
  <title>{title} – jax-js</title>
</svelte:head>

<section class="mx-auto max-w-6xl space-y-5">
  <header class="panel p-5 sm:p-6">
    <div class="grid gap-6 lg:grid-cols-[1fr_auto] lg:items-end">
      <div>
        <h1 class="max-w-3xl text-3xl font-medium text-zinc-950 sm:text-5xl">
          {title}
        </h1>
        <p class="mt-4 max-w-3xl text-base leading-relaxed text-zinc-600">
          {summary}
        </p>
      </div>
    </div>

    {#if browser && !navigator.gpu}
      <p class="notice mt-5 border-amber-300 bg-amber-50 text-amber-950">
        WebGPU is not supported in this browser. WebGPU-backed strategies will
        not run, but WebAssembly variants may still work.
      </p>
    {/if}
  </header>

  <div class="grid gap-5 lg:grid-cols-[minmax(0,1fr)_18rem]">
    <section class="panel">
      <div class="section-bar">
        <div>
          <h2 class="text-xl font-semibold">Strategies</h2>
        </div>
        <div class="flex gap-2">
          <button
            class="button secondary"
            disabled={!!running || runningAll}
            onclick={() => (results = {})}
          >
            Clear
          </button>
          <button
            class="button primary"
            disabled={!!running || runningAll}
            onclick={runAll}
          >
            {runningAll ? "Running…" : "Run all"}
          </button>
        </div>
      </div>

      <div class="border-t border-zinc-950/10 p-3 sm:p-4">
        <p class="mb-3 text-sm text-zinc-500">
          One untimed warmup run, followed by one measured run.
        </p>

        <div class="strategy-grid">
          {#each strategies as strategy (strategy.name)}
            <button
              class="strategy-button"
              disabled={!!running || runningAll}
              onclick={() => bench(strategy)}
            >
              <span
                class="status-dot"
                class:running={running === strategy.name}
                class:done={results[strategy.name] !== undefined &&
                  running !== strategy.name}
              ></span>
              <span class="name">{strategy.name}</span>
            </button>
          {/each}
        </div>

        {#if error}
          <p class="notice mt-4 border-red-300 bg-red-50 text-red-700">
            {error}
          </p>
        {/if}
      </div>
    </section>

    <aside class="panel self-start">
      <div class="section-bar block">
        <h2 class="text-xl font-semibold">Details</h2>
      </div>
      <ol
        class="divide-y divide-zinc-950/10 border-t border-zinc-950/10 text-sm leading-snug text-zinc-600"
      >
        {#each notes as note, i}
          <li class="grid grid-cols-[2rem_1fr] gap-1 px-4 py-3">
            <span class="font-mono text-xs text-zinc-400"
              >{String(i + 1).padStart(2, "0")}</span
            >
            <span>{note}</span>
          </li>
        {/each}
      </ol>
    </aside>
  </div>

  <section class="panel">
    <div class="section-bar">
      <div>
        <h2 class="text-xl font-semibold">Results</h2>
        <p class="mt-1 text-sm text-zinc-500">Sorted fastest first.</p>
      </div>
      {#if sortedResults().length}
        <p class="text-sm text-zinc-500">
          {sortedResults().length} rows
        </p>
      {/if}
    </div>

    {#if sortedResults().length}
      <div class="overflow-x-auto border-t border-zinc-950/10">
        <table class="w-full text-left text-sm">
          <thead class="text-sm text-zinc-700">
            <tr class="border-b border-zinc-950/10">
              <th class="px-4 py-3 font-normal">Strategy</th>
              <th class="w-[28rem] px-4 py-3 font-normal">Throughput</th>
              <th class="px-4 py-3 font-normal">Time</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-zinc-950/10">
            {#each sortedResults() as [variant, time]}
              <tr class="result-row">
                <td class="px-4 py-3 font-mono text-xs text-zinc-950"
                  >{variant}</td
                >
                <td class="px-4 py-3 tabular-nums text-zinc-700">
                  <div class="throughput-cell">
                    <span class="throughput-track" aria-hidden="true">
                      <span
                        class="throughput-bar"
                        style:width={throughputBarWidth(time)}
                      ></span>
                    </span>
                    <span class="throughput-value">{formatResult(time)}</span>
                  </div>
                </td>
                <td class="px-4 py-3 tabular-nums">{formatTime(time)}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>
    {:else}
      <div class="border-t border-zinc-950/10 p-10 text-center">
        <p class="text-sm text-zinc-500">No samples yet.</p>
        <p class="mt-2 text-sm text-zinc-500">
          Run an individual strategy or start the full queue.
        </p>
      </div>
    {/if}
  </section>
</section>

<style lang="postcss">
  @reference "$app.css";

  .panel {
    @apply border border-zinc-950/10 bg-stone-50/95;
    box-shadow: 0 1px 0 rgb(24 24 27 / 0.08);
  }

  .section-bar {
    @apply flex items-center justify-between gap-4 p-4;
  }

  .notice {
    @apply border px-4 py-3 text-sm;
  }

  .button {
    @apply border px-3 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-40;
  }

  .button.primary {
    @apply border-primary bg-primary text-white hover:bg-primary/90 active:translate-y-px;
  }

  .button.secondary {
    @apply border-zinc-950/15 bg-white text-zinc-700 hover:border-zinc-950/30 active:translate-y-px;
  }

  .strategy-grid {
    @apply grid gap-px overflow-hidden border border-zinc-950/10 bg-zinc-950/10 sm:grid-cols-2 xl:grid-cols-3;
  }

  .strategy-button {
    @apply grid grid-cols-[auto_1fr] items-center gap-3 bg-white px-3 py-3 text-left transition hover:bg-stone-100 disabled:cursor-not-allowed disabled:opacity-60;
  }

  .strategy-button .status-dot {
    @apply size-2.5 rounded-full border border-zinc-300 bg-transparent;
  }

  .strategy-button .status-dot.running {
    @apply border-yellow-500 bg-yellow-400;
    animation: bench-pulse 1s ease-in-out infinite;
  }

  .strategy-button .status-dot.done {
    @apply border-green-600 bg-green-600;
  }

  .strategy-button .name {
    @apply truncate font-mono text-xs text-zinc-900;
  }

  .throughput-cell {
    display: grid;
    grid-template-columns: 3.5rem max-content;
    align-items: center;
    column-gap: 0.75rem;
    min-width: 14rem;
  }

  .throughput-value {
    @apply tabular-nums;
  }

  .throughput-track {
    @apply h-4 overflow-hidden bg-zinc-200;
  }

  .throughput-bar {
    @apply block h-full bg-primary;
  }

  .result-row {
    @apply transition-colors hover:bg-zinc-950/[0.025];
  }

  @keyframes bench-pulse {
    0%,
    100% {
      box-shadow: 0 0 0 0 rgb(234 179 8 / 0.35);
      transform: scale(1);
    }
    50% {
      box-shadow: 0 0 0 5px rgb(234 179 8 / 0);
      transform: scale(0.9);
    }
  }
</style>
