<script lang="ts">
  import { building } from "$app/environment";
  import { afterNavigate, goto } from "$app/navigation";
  import { resolve } from "$app/paths";
  import { page } from "$app/state";

  import type { Device } from "@jax-js/jax";
  import {
    ArrowRightIcon,
    LoaderIcon,
    PaletteIcon,
    PlayIcon,
    ShareIcon,
  } from "@lucide/svelte";
  import { SplitPane } from "@rich_harris/svelte-split-pane";

  import ConsoleLine from "$lib/repl/ConsoleLine.svelte";
  import ReplEditor from "$lib/repl/ReplEditor.svelte";
  import { decodeContent, encodeContent } from "$lib/repl/encode";
  import { ReplRunner } from "$lib/repl/runner.svelte";

  const src: Record<string, string> = import.meta.glob("./*.ts", {
    eager: true,
    query: "?raw",
    import: "default",
  });

  const codeSamples: {
    title: string;
    id: string;
  }[] = [
    { title: "Arrays", id: "01-arrays" },
    { title: "Tracing Jaxprs", id: "02-tracing" },
    { title: "Logistic regression", id: "03-logistic-regression" },
    { title: "Mandelbrot set", id: "04-mandelbrot" },
  ];

  interface URLSelection {
    sample?: string;
    content?: string;
  }

  function getSelectionFromUrl(url: URL): URLSelection {
    const selection: URLSelection = {};

    if (!building) {
      const sample = url.searchParams.get("sample");
      if (sample && codeSamples.some((x) => x.id === sample)) {
        selection.sample = sample;
      }

      const contentZipB64 = url.searchParams.get("content");
      if (contentZipB64) {
        selection.content = decodeContent(contentZipB64);
      }

      if (!selection.content && !selection.sample) {
        selection.sample = codeSamples[0].id;
      }
    }

    return selection;
  }

  let selectionOnLoad = getSelectionFromUrl(page.url);

  let sample = $state(selectionOnLoad.sample);
  let device: Device = $state("webgpu");
  let replEditor: ReplEditor;
  let replRunner = new ReplRunner();

  let consoleLines = $derived(replRunner.consoleLines);
  let mockConsole = replRunner.mockConsole;
  let runDurationMs = $state<number | null>(null);

  afterNavigate(({ type }) => {
    if (type === "enter") return; // Already handled on load
    let selection = getSelectionFromUrl(page.url);
    if (selection.sample !== undefined) {
      sample = selection.sample;
      replEditor.setText(src[`./${sample}.ts`]);
    }
    if (selection.content !== undefined) {
      replEditor.setText(selection.content);
    }
  });

  function chooseSample(id: string) {
    replEditor.setText(src[`./${id}.ts`]);
    sample = id;
    goto(page.url.pathname + `?sample=${sample}`);
  }

  async function handleFormat() {
    const { formatWithCursor } = await import("prettier");
    const prettierParserTypescript = await import("prettier/parser-typescript");
    const prettierPluginEstree = await import("prettier/plugins/estree");

    const code = replEditor.getText();
    try {
      const { formatted, cursorOffset } = await formatWithCursor(code, {
        parser: "typescript",
        plugins: [prettierParserTypescript, prettierPluginEstree as any],
        cursorOffset: replEditor.getCursorOffset(),
      });
      replEditor.setText(formatted);
      replEditor.setCursorOffset(cursorOffset);
    } catch (e: any) {
      mockConsole.error(e);
    }
  }

  async function handleShare() {
    const code = replEditor.getText();

    const url = new URL(page.url.origin + page.url.pathname);
    url.searchParams.set("content", encodeContent(code));

    try {
      goto(url, { replaceState: true });
      await navigator.clipboard.writeText(url.toString());
      mockConsole.info("Link copied to clipboard!");
    } catch (e: any) {
      mockConsole.error("Failed to copy link:", e);
    }
  }

  async function handleRun() {
    const startMs = performance.now();
    runDurationMs = null;
    await replRunner.runProgram(replEditor.getText(), device);
    runDurationMs = performance.now() - startMs;
  }
</script>

<svelte:head>
  <title>jax-js REPL</title>
</svelte:head>

<div class="h-dvh">
  <SplitPane
    type="horizontal"
    pos="280px"
    min="200px"
    max="40%"
    --color="var(--color-gray-200)"
    --thickness="16px"
  >
    {#snippet a()}
      <div
        class="bg-gray-50 px-4 pt-4 pb-12 !overflow-y-auto"
        style:scrollbar-width="thin"
      >
        <h1 class="text-xl font-light mb-4">
          <a target="_blank" href={resolve("/")}
            ><span class="font-medium">jax-js</span> REPL</a
          >
        </h1>

        <hr class="mb-6 border-gray-200" />

        <p class="text-sm mb-4">
          Try out jax-js. Numerical and GPU computing for the web!
        </p>
        <p class="text-sm mb-4">
          The goal is having NumPy and JAX-like APIs <em>in the browser</em>, on
          Wasm or WebGPU — with JIT compilation.
        </p>

        <pre class="mb-4 text-sm bg-gray-100 px-2 py-1 rounded"><code
            >npm i @jax-js/jax</code
          ></pre>

        <h2 class="text-lg mt-8 mb-2">Examples</h2>
        <div class="text-sm flex flex-col">
          {#each codeSamples as { title, id } (id)}
            <button
              class="px-2 py-1 text-left rounded flex items-center hover:bg-gray-100 active:bg-gray-200 transition-colors"
              class:font-semibold={id === sample}
              onclick={() => chooseSample(id)}
            >
              <span class="mr-2">
                <ArrowRightIcon size={16} />
              </span>
              {title}
            </button>
          {/each}
        </div>
      </div>
    {/snippet}
    {#snippet b()}
      <SplitPane
        type="vertical"
        pos="-240px"
        min="10%"
        max="-64px"
        --color="var(--color-gray-200)"
        --thickness="16px"
      >
        {#snippet a()}
          <div class="flex flex-col min-w-0 !overflow-visible">
            <div class="px-4 py-2 flex items-center gap-1">
              <button
                class="bg-emerald-100 hover:bg-emerald-200 active:scale-105 transition-all rounded-md text-sm px-3 py-0.5 flex items-center disabled:opacity-50"
                onclick={handleRun}
                disabled={replRunner.running}
              >
                <PlayIcon size={14} class="mr-1.5" />
                Run
              </button>
              <button
                class="hover:bg-gray-100 active:scale-105 transition-all rounded-md text-sm px-3 py-0.5 flex items-center disabled:opacity-50"
                onclick={handleFormat}
              >
                <PaletteIcon size={14} class="mr-1.5" />
                Format
              </button>
              <button
                class="hover:bg-gray-100 active:scale-105 transition-all rounded-md text-sm px-3 py-0.5 flex items-center disabled:opacity-50"
                onclick={handleShare}
              >
                <ShareIcon size={14} class="mr-1.5" />
                Share
              </button>

              <!-- Device selector -->
              <select
                bind:value={device}
                class="ml-auto border border-gray-300 rounded-md text-sm px-1 py-0.5"
              >
                <option value="webgpu">WebGPU</option>
                <option value="wasm">Wasm</option>
                <option value="cpu">CPU (slow)</option>
              </select>
            </div>
            <div class="flex-1 min-h-0">
              <ReplEditor
                initialText={selectionOnLoad.content !== undefined
                  ? selectionOnLoad.content!
                  : src[`./${sample ?? codeSamples[0].id}.ts`]}
                bind:this={replEditor}
                onformat={handleFormat}
                onrun={handleRun}
              />
            </div>
          </div>
        {/snippet}
        {#snippet b()}
          <div class="flex flex-col h-full">
            <p class="text-gray-500 text-sm py-2 px-4 select-none shrink-0">
              Console
              {#if replRunner.running}
                <LoaderIcon
                  size={16}
                  class="inline-block animate-spin ml-1 mb-[3px]"
                />
              {:else}
                {#if runDurationMs !== null}
                  <span class="ml-1 text-gray-400">({Math.round(runDurationMs)} ms)</span>
                {/if}
                {#if consoleLines.length === 0}
                  <span>(empty)</span>
                {/if}
              {/if}
            </p>
            <div
              class="pb-2 px-4 flex flex-col grow overflow-y-auto text-[13px]"
              style:scrollbar-width="thin"
            >
              {#each consoleLines as line, i (i)}
                <ConsoleLine {line} showTime />
              {/each}
            </div>
          </div>
        {/snippet}
      </SplitPane>
    {/snippet}
  </SplitPane>
</div>

<style lang="postcss">
  @reference "$app.css";

  /* Prevent diagnostics or hover hints from editor from being cut off. */
  div :global(svelte-split-pane-section) {
    overflow: visible !important;
    min-height: 0;
    min-width: 0;
  }
</style>
