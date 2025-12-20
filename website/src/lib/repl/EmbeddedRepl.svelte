<script lang="ts">
  import { resolve } from "$app/paths";

  import type { Device } from "@jax-js/jax";
  import {
    CheckIcon,
    ExternalLinkIcon,
    FoldVerticalIcon,
    LoaderIcon,
    PlayIcon,
    TerminalIcon,
    UnfoldVerticalIcon,
  } from "@lucide/svelte";
  import { SplitPane } from "@rich_harris/svelte-split-pane";

  import ConsoleLine from "./ConsoleLine.svelte";
  import ReplEditor from "./ReplEditor.svelte";
  import { encodeContent } from "./encode";
  import { ReplRunner } from "./runner.svelte";

  let {
    initialText,
  }: {
    initialText: string;
  } = $props();

  let editor: ReplEditor;
  let runner = new ReplRunner();
  let currentText = $state(initialText);
  let replLink = $derived(
    resolve("/repl") + `?content=${encodeContent(currentText)}`,
  );
  let device = $state<Device>("webgpu");

  let expanded = $state(false);
  let runDurationMs = $state<number | null>(null);

  async function handleRun() {
    const startMs = performance.now();
    runDurationMs = null;
    await runner.runProgram(editor.getText(), device);
    runDurationMs = performance.now() - startMs;
  }
</script>

<div
  class="flex flex-col border border-gray-200 rounded-lg"
  style:height={expanded ? "720px" : "360px"}
>
  <div
    class="shrink-0 border-b border-gray-200 flex items-center px-2 py-1.5 gap-2"
  >
    <button
      class="bg-green-100 hover:bg-green-200 disabled:opacity-50 px-2 py-0.5 rounded flex text-sm items-center gap-1.5"
      onclick={handleRun}
      disabled={runner.running}
    >
      <PlayIcon size={16} />
      Run
    </button>
    <button
      class="hover:bg-gray-100 px-2 py-0.5 rounded flex text-sm items-center gap-1.5"
      onclick={() => (expanded = !expanded)}
    >
      {#if expanded}
        <FoldVerticalIcon size={16} />
        Collapse
      {:else}
        <UnfoldVerticalIcon size={16} />
        Expand
      {/if}
    </button>
    <a
      href={replLink}
      class="hidden sm:flex hover:bg-gray-100 px-2 py-0.5 rounded text-sm items-center gap-1.5 text-gray-600"
    >
      <ExternalLinkIcon size={16} />
      Open REPL
    </a>

    <div class="ml-auto"></div>
    <select
      bind:value={device}
      class="hover:bg-gray-100 rounded text-sm pt-[3px] py-0.5"
    >
      <option value="webgpu">WebGPU</option>
      <option value="wasm">Wasm</option>
    </select>
  </div>
  <div class="flex-1 min-h-0 split-pane-container">
    <SplitPane
      type="vertical"
      pos="-112px"
      min="40px"
      max="-40px"
      --color="var(--color-gray-200)"
      --thickness="16px"
    >
      {#snippet a()}
        <div class="!overflow-visible">
          <ReplEditor
            bind:this={editor}
            {initialText}
            editorOptions={{
              lineNumbersMinChars: 4,
              padding: { top: 8, bottom: 8 },
              scrollbar: {
                alwaysConsumeMouseWheel: false,
                horizontalScrollbarSize: 10,
                useShadows: false,
                verticalScrollbarSize: 10,
              },
              scrollBeyondLastLine: false,
            }}
            onchange={() => {
              currentText = editor.getText();
            }}
            onrun={handleRun}
          />
        </div>
      {/snippet}
      {#snippet b()}
        {#if !runner.running && !runner.finished}
          <div class="flex px-3 py-2 select-none">
            <TerminalIcon size={20} class="text-gray-400 mr-2" />
            <p class="text-sm text-gray-500">Run code to see output here.</p>
          </div>
        {:else if runner.consoleLines.length === 0 && runner.finished}
          <div class="flex px-3 py-2 select-none">
            <CheckIcon size={20} class="text-gray-400 mr-2" />
            <p class="text-sm text-gray-500">No output.</p>
          </div>
        {:else}
          <div class="flex flex-col px-3 text-[13px]">
            <p class="shrink-0 text-gray-500 pt-2 pb-1">
              Console
              {#if runner.running}
                <LoaderIcon
                  size={14}
                  class="inline-block animate-spin ml-1 mb-0.5"
                />
              {:else if runDurationMs !== null}
                <span class="ml-1 text-gray-400"
                  >({Math.round(runDurationMs)} ms)</span
                >
              {/if}
            </p>
            <div class="flex-1 py-0.5 font-mono overflow-y-auto">
              {#each runner.consoleLines as line}
                <ConsoleLine {line} />
              {/each}
            </div>
          </div>
        {/if}
      {/snippet}
    </SplitPane>
  </div>
</div>

<style lang="postcss">
  @reference "$app.css";

  /* Prevent diagnostics or hover hints from editor from being cut off. */
  .split-pane-container :global(svelte-split-pane-section) {
    overflow: visible !important;
    min-height: 0;
    min-width: 0;
  }
</style>
