<script lang="ts">
  import { building } from "$app/environment";
  import { afterNavigate, goto } from "$app/navigation";
  import { resolve } from "$app/paths";
  import { page } from "$app/state";

  import type { Device } from "@jax-js/jax";
  import {
    ChevronLeftIcon,
    ChevronRightIcon,
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

  type CodeSample = {
    title: string;
    id: string;
  };

  const sampleGroups: {
    title: string;
    samples: CodeSample[];
  }[] = [
    {
      title: "Basics",
      samples: [
        { title: "Arrays + jit", id: "01-arrays-jit" },
        { title: "Broadcasting", id: "02-broadcasting" },
        { title: "Reference counting", id: "03-reference-counting" },
        { title: "Devices", id: "04-devices" },
        { title: "jit static args", id: "05-jit-static-args" },
      ],
    },
    {
      title: "Autodiff",
      samples: [
        { title: "Gradients", id: "06-gradients" },
        { title: "makeJaxpr", id: "07-make-jaxpr" },
        { title: "vmap", id: "08-vmap" },
        { title: "jvp / vjp", id: "09-jvp-vjp" },
        { title: "Hessian", id: "10-hessian" },
        { title: "Tree gradients", id: "11-tree-gradients" },
      ],
    },
    {
      title: "Linear algebra",
      samples: [
        { title: "PCA / SVD", id: "12-pca-svd" },
        { title: "Matmul / einsum", id: "13-matmul-einsum" },
        { title: "Least squares", id: "14-least-squares" },
        { title: "Cholesky sampling", id: "15-cholesky-sampling" },
        { title: "Solve / det / LU", id: "16-solve-det-lu" },
        { title: "Inverse gradient", id: "17-inverse-gradient" },
        { title: "Sorting / topK", id: "18-sorting-topk" },
      ],
    },
    {
      title: "Signals and images",
      samples: [
        { title: "FFT filter", id: "19-fft-filter" },
        { title: "Reaction-diffusion", id: "20-reaction-diffusion" },
        { title: "Frequency peaks", id: "21-frequency-peaks" },
        { title: "Low-pass filter", id: "22-low-pass-filter" },
        { title: "Image denoise", id: "23-image-denoise" },
        { title: "Convolution theorem", id: "24-convolution-theorem" },
        { title: "Spectrogram", id: "25-spectrogram" },
        { title: "1D convolution", id: "26-conv1d" },
        { title: "Edge detection", id: "27-edge-detection" },
        { title: "Max pooling", id: "28-max-pooling" },
        { title: "Batched convolution", id: "29-batched-conv" },
        { title: "Mandelbrot", id: "30-mandelbrot" },
        { title: "Heat equation", id: "31-heat-equation" },
      ],
    },
    {
      title: "Machine learning",
      samples: [
        { title: "Tiny neural net", id: "32-tiny-neural-net" },
        { title: "Logistic regression", id: "33-logistic-regression" },
        { title: "MLP classifier", id: "34-mlp-classifier" },
        { title: "Adam vs SGD", id: "35-adam-vs-sgd" },
        { title: "Gradient clipping", id: "36-gradient-clipping" },
        { title: "Activations", id: "37-activations" },
        { title: "Softmax", id: "38-softmax" },
        { title: "Attention", id: "39-attention" },
        { title: "Causal mask", id: "40-causal-mask" },
      ],
    },
    {
      title: "Data utilities",
      samples: [
        { title: "Keys / split", id: "41-keys-split" },
        { title: "vmap sampling", id: "42-vmap-sampling" },
        { title: "Monte Carlo pi", id: "43-monte-carlo-pi" },
        { title: "Multivariate normal", id: "44-multivariate-normal" },
        { title: "Categorical / Gumbel", id: "45-categorical-gumbel" },
        { title: "Tokenizer", id: "46-tokenizer" },
        { title: "Safetensors keys", id: "47-safetensors-keys" },
        { title: "WeightMapper", id: "48-weight-mapper" },
      ],
    },
  ];
  const codeSamples = sampleGroups.flatMap((group) => group.samples);

  function groupForSample(id: string | undefined) {
    return (
      sampleGroups.find((group) =>
        group.samples.some((sample) => sample.id === id),
      ) ?? sampleGroups[0]
    );
  }

  interface URLSelection {
    sample?: string;
    content?: string;
  }

  function hasSample(id: string) {
    return codeSamples.some((x) => x.id === id);
  }

  function getSelectionFromUrl(url: URL): URLSelection {
    const selection: URLSelection = {};

    if (!building) {
      const sample = url.searchParams.get("sample");
      if (sample && hasSample(sample)) {
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

  function getNormalizedUrl(url: URL, selection: URLSelection): URL | null {
    const sample = url.searchParams.get("sample");
    if (!sample || hasSample(sample)) return null;

    const normalizedUrl = new URL(url);
    if (selection.content !== undefined) {
      normalizedUrl.searchParams.delete("sample");
    } else {
      normalizedUrl.searchParams.set("sample", codeSamples[0].id);
    }
    return normalizedUrl.href === url.href ? null : normalizedUrl;
  }

  let selectionOnLoad = getSelectionFromUrl(page.url);

  let sample = $state(selectionOnLoad.sample);
  let selectedGroupTitle = $state(groupForSample(selectionOnLoad.sample).title);
  let selectedGroup = $derived(
    sampleGroups.find((group) => group.title === selectedGroupTitle) ??
      sampleGroups[0],
  );
  let currentSampleIndex = $derived(
    sample === undefined
      ? -1
      : codeSamples.findIndex(({ id }) => id === sample),
  );
  let hasSampleSelection = $derived(currentSampleIndex !== -1);
  let hasPreviousSample = $derived(currentSampleIndex > 0);
  let hasNextSample = $derived(
    hasSampleSelection && currentSampleIndex < codeSamples.length - 1,
  );
  let device: Device = $state("webgpu");
  let replEditor: ReplEditor;
  let replRunner = new ReplRunner();

  let consoleLines = $derived(replRunner.consoleLines);
  let mockConsole = replRunner.mockConsole;
  let runDurationMs = $derived(replRunner.runDurationMs);

  $effect(() => {
    if (building) return;
    const selection = getSelectionFromUrl(page.url);
    const normalizedUrl = getNormalizedUrl(page.url, selection);
    if (normalizedUrl) {
      goto(normalizedUrl, {
        replaceState: true,
        keepFocus: true,
        noScroll: true,
      });
    }
  });

  afterNavigate(({ type }) => {
    if (type === "enter") return; // Already handled on load
    let selection = getSelectionFromUrl(page.url);
    if (selection.sample !== undefined) {
      sample = selection.sample;
      selectedGroupTitle = groupForSample(sample).title;
      replEditor.setText(src[`./${sample}.ts`]);
    }
    if (selection.content !== undefined) {
      replEditor.setText(selection.content);
    }
  });

  function chooseSample(id: string) {
    const nextSource = src[`./${id}.ts`];
    if (nextSource === undefined) {
      mockConsole.error(`Missing REPL example source: ${id}`);
      return;
    }
    replEditor.setText(nextSource);
    sample = id;
    selectedGroupTitle = groupForSample(id).title;
    goto(page.url.pathname + `?sample=${sample}`);
  }

  function chooseAdjacentSample(offset: -1 | 1) {
    const index = currentSampleIndex === -1 ? 0 : currentSampleIndex;
    const nextSample = codeSamples[index + offset];
    if (nextSample) chooseSample(nextSample.id);
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
    await replRunner.runProgram(replEditor.getText(), device);
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
          The library has NumPy and JAX-like APIs <em>in the browser</em>, on
          Wasm and WebGPU, with JIT compilation.
        </p>

        <pre class="mb-4 text-sm bg-gray-100 px-2 py-1 rounded"><code
            >npm i @jax-js/jax</code
          ></pre>

        <h2 class="text-lg mt-8 mb-2">Examples</h2>
        <label class="sr-only" for="example-category">Example category</label>
        <div class="flex gap-1 mb-3">
          <select
            id="example-category"
            bind:value={selectedGroupTitle}
            class="min-w-0 flex-1 border border-gray-300 rounded-md bg-white text-sm px-2 py-1"
          >
            {#each sampleGroups as group (group.title)}
              <option value={group.title}>{group.title}</option>
            {/each}
          </select>
          {#if hasSampleSelection}
            <button
              class="w-8 border border-gray-300 rounded-md bg-white text-gray-700 hover:bg-gray-100 active:scale-105 transition-all disabled:opacity-40 disabled:active:scale-100 flex items-center justify-center"
              aria-label="Previous example"
              title="Previous example"
              disabled={!hasPreviousSample}
              onclick={() => chooseAdjacentSample(-1)}
            >
              <ChevronLeftIcon size={16} />
            </button>
            <button
              class="w-8 border border-gray-300 rounded-md bg-white text-gray-700 hover:bg-gray-100 active:scale-105 transition-all disabled:opacity-40 disabled:active:scale-100 flex items-center justify-center"
              aria-label="Next example"
              title="Next example"
              disabled={!hasNextSample}
              onclick={() => chooseAdjacentSample(1)}
            >
              <ChevronRightIcon size={16} />
            </button>
          {/if}
        </div>
        <div class="text-sm flex flex-col">
          {#each selectedGroup.samples as { title, id } (id)}
            <button
              class="px-2 py-1 text-left rounded flex items-center gap-2 hover:bg-gray-100 active:bg-gray-200 transition-colors"
              class:font-semibold={id === sample}
              onclick={() => chooseSample(id)}
            >
              <span class="w-6 shrink-0 font-mono text-xs text-gray-500">
                {id.slice(0, 2)}
              </span>
              <span>{title}</span>
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
                <option value="webgl">WebGL</option>
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
                  size={14}
                  class="inline-block animate-spin ml-1 mb-[3px]"
                />
              {:else if consoleLines.length === 0}
                <span>(empty)</span>
              {:else if runDurationMs !== null}
                <span class="ml-1 text-gray-400"
                  >({Math.round(runDurationMs).toLocaleString()} ms)</span
                >
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
