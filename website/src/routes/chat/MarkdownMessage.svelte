<script lang="ts">
  import { browser } from "$app/environment";

  let { content }: { content: string } = $props();

  let html = $state("");
  let renderVersion = 0;

  $effect(() => {
    const markdown = content;
    const version = ++renderVersion;

    if (!browser || markdown === "") {
      html = "";
      return;
    }

    void (async () => {
      const [{ marked }, DOMPurify] = await Promise.all([
        import("marked"),
        import("dompurify"),
      ]);
      const rendered = await marked.parse(markdown, {
        async: false,
        breaks: true,
        gfm: true,
      });
      if (version === renderVersion) {
        html = DOMPurify.default.sanitize(rendered);
      }
    })();
  });
</script>

{#if html}
  <!-- eslint-disable-next-line svelte/no-at-html-tags -- sanitized with DOMPurify -->
  <div class="markdown-message">{@html html}</div>
{:else}
  <div class="whitespace-pre-wrap">{content}</div>
{/if}

<style lang="postcss">
  @reference "$app.css";

  .markdown-message {
    @apply whitespace-normal;
  }

  .markdown-message :global(:first-child) {
    margin-top: 0;
  }

  .markdown-message :global(:last-child) {
    margin-bottom: 0;
  }

  .markdown-message :global(p) {
    @apply my-3;
  }

  .markdown-message :global(ul),
  .markdown-message :global(ol) {
    @apply my-3 pl-6;
  }

  .markdown-message :global(ul) {
    @apply list-disc;
  }

  .markdown-message :global(ol) {
    @apply list-decimal;
  }

  .markdown-message :global(li) {
    @apply my-1;
  }

  .markdown-message :global(a) {
    @apply underline underline-offset-2;
  }

  .markdown-message :global(code) {
    @apply rounded bg-gray-200 px-1 py-0.5 font-mono text-sm;
  }

  .markdown-message :global(pre) {
    @apply my-3 overflow-x-auto rounded-xl bg-gray-950 p-3 text-gray-50;
  }

  .markdown-message :global(pre code) {
    @apply bg-transparent p-0 text-inherit;
  }

  .markdown-message :global(blockquote) {
    @apply my-3 border-l-4 border-gray-300 pl-4 text-gray-700;
  }

  .markdown-message :global(h1),
  .markdown-message :global(h2),
  .markdown-message :global(h3) {
    @apply mt-4 mb-2 font-semibold;
  }

  .markdown-message :global(h1) {
    @apply text-xl;
  }

  .markdown-message :global(h2) {
    @apply text-lg;
  }

  .markdown-message :global(table) {
    @apply my-3 border-collapse text-sm;
  }

  .markdown-message :global(th),
  .markdown-message :global(td) {
    @apply border border-gray-300 px-2 py-1;
  }
</style>
