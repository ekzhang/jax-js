<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import type * as Monaco from "monaco-editor/esm/vs/editor/editor.api";

  export let initialText: string;

  let containerEl: HTMLDivElement;
  let editor: Monaco.editor.IStandaloneCodeEditor;
  let monaco: typeof Monaco;

  export function getText() {
    return editor?.getValue() ?? "";
  }

  export function setText(text: string) {
    editor?.setValue(text);
  }

  onMount(async () => {
    monaco = (await import("$lib/monaco")).default;

    editor = monaco.editor.create(containerEl, {
      fontSize: 14,
      automaticLayout: true,
    });
    const model = monaco.editor.createModel(
      initialText,
      "typescript",
      monaco.Uri.parse("file:///main.ts"),
    );
    editor.setModel(model);
  });

  onDestroy(() => {
    monaco?.editor.getModels().forEach((model) => model.dispose());
    editor?.dispose();
  });
</script>

<div class="h-full" bind:this={containerEl}></div>
