<script lang="ts">
  import { init, numpy as np, setDevice } from "@jax-js/jax";
  import { cachedFetch, opfs, safetensors, tokenizers } from "@jax-js/loaders";
  import { FileTextIcon, ImageIcon } from "lucide-svelte";

  import DownloadToast, {
    type Props as DownloadToastProps,
  } from "$lib/common/DownloadToast.svelte";
  import { fromSafetensors, runMobileCLIPTextEncoder } from "./clipInference";

  const weightsUrl =
    "https://huggingface.co/ekzhang/jax-js-models/resolve/main/mobileclip2-s0.safetensors";
  let weights: safetensors.File | null = null;

  let isDownloading = $state(false);
  let downloadState = $state<DownloadToastProps | null>(null);

  let searchQuery = $state("");
  let computeTime = $state<number | null>(null);
  let hasCorpus = $state(false); // Track if user has added any data

  async function downloadClipWeights(): Promise<safetensors.File> {
    if (weights) return weights;
    isDownloading = true;
    try {
      downloadState = {
        status: "downloading",
      };

      const data = await cachedFetch(weightsUrl, {}, (progress) => {
        downloadState = {
          status: "downloading",
          loaded: progress.loadedBytes,
          total: progress.totalBytes,
        };
      });

      const result = safetensors.parse(data);

      downloadState = {
        status: "success",
        loaded: downloadState.loaded,
      };
      setTimeout(() => {
        downloadState = null;
      }, 3000);

      weights = result;
      return result;
    } catch (error) {
      downloadState = {
        status: "error",
        errorMessage:
          error instanceof Error ? error.message : "Download aborted",
      };
      setTimeout(() => {
        downloadState = null;
      }, 4000);
      throw error;
    } finally {
      isDownloading = false;
    }
  }

  async function main() {
    if (isDownloading) return;

    await init("webgpu");
    setDevice("webgpu");

    try {
      weights = await downloadClipWeights();
      const model = fromSafetensors(weights);
      console.log("-------- WEIGHTS --------");
      console.log(weights);
      console.log("-------- MODEL --------");
      console.log(model);

      const tokenizer = await tokenizers.getBpe("clip");
      console.log(tokenizer.encode("hello world"));

      const tokens = np.array(tokenizer.encode("hello world"), {
        dtype: np.uint32,
      });
      const encoded = runMobileCLIPTextEncoder(model.text, tokens);
      console.log("-------- ENCODED --------");
      console.log(encoded.js());
    } catch (error) {
      console.error("Error in main:", error);
    }
  }

  async function clearCache() {
    try {
      await opfs.clear();
      console.log("Cache cleared");
    } catch (error) {
      console.error("Error clearing cache:", error);
    }
  }

  async function handleSearch() {
    // TODO: Implement search logic
    const start = performance.now();
    // Compute embeddings here
    const end = performance.now();
    computeTime = end - start;
  }
</script>

{#if downloadState}
  <DownloadToast {...downloadState} />
{/if}

<div class="min-h-screen bg-white">
  <!-- Header with search bar -->
  <header class="border-b border-gray-200">
    <div class="max-w-4xl mx-auto px-4 py-8">
      <div class="flex items-center gap-4 mb-4">
        <button
          onclick={main}
          disabled={isDownloading}
          class="px-4 py-2 border-2 border-black hover:bg-black hover:text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isDownloading ? "Loading…" : "Load Model"}
        </button>
        <button
          onclick={clearCache}
          class="px-4 py-2 border-2 border-black hover:bg-black hover:text-white transition-colors"
        >
          Clear Cache
        </button>
      </div>
      <div class="relative">
        <input
          type="text"
          class="w-full px-6 py-4 text-xl border-2 border-black focus:outline-none focus:border-black"
          placeholder="Search or describe…"
          bind:value={searchQuery}
          oninput={handleSearch}
        />
        {#if computeTime !== null}
          <span
            class="absolute right-4 top-1/2 -translate-y-1/2 text-sm text-gray-500"
          >
            computed in {computeTime.toFixed(0)}ms
          </span>
        {/if}
      </div>
    </div>
  </header>

  <!-- Main content area -->
  <main class="max-w-4xl mx-auto px-4 py-8">
    {#if !hasCorpus}
      <!-- Empty state -->
      <div class="text-center py-16">
        <h2 class="text-3xl font-normal mb-2">No data yet</h2>
        <p class="text-lg text-gray-600 mb-12">
          Add images and text to start searching with MobileClip
        </p>

        <div class="flex flex-col items-center gap-8">
          <div class="w-full max-w-md">
            <h3 class="text-base font-medium mb-4">Upload your own</h3>
            <div class="flex flex-col gap-3">
              <button
                class="flex items-center justify-center gap-2 px-6 py-3.5 border-2 border-black hover:bg-black hover:text-white transition-colors"
              >
                <ImageIcon size={20} />
                Upload images
              </button>
              <button
                class="flex items-center justify-center gap-2 px-6 py-3.5 border-2 border-black hover:bg-black hover:text-white transition-colors"
              >
                <FileTextIcon size={20} />
                Upload text file
              </button>
            </div>
          </div>

          <div class="text-sm text-gray-400 uppercase tracking-wider">or</div>

          <div class="w-full max-w-md">
            <h3 class="text-base font-medium mb-4">Load a prepared dataset</h3>
            <div class="flex flex-col gap-3">
              <button
                class="px-6 py-3.5 border-2 border-black hover:bg-black hover:text-white transition-colors"
              >
                Art museum images
              </button>
              <button
                class="px-6 py-3.5 border-2 border-black hover:bg-black hover:text-white transition-colors"
              >
                The Little Prince text
              </button>
            </div>
          </div>
        </div>
      </div>
    {:else}
      <!-- Results view (placeholder for now) -->
      <div class="pt-4">
        <!-- Three star matches -->
        <section class="mb-12">
          <h3
            class="flex items-center gap-2 text-base font-medium pb-2 border-b border-gray-200 mb-4"
          >
            <span>★★★</span>
            <span class="text-gray-600 font-normal">Close match</span>
          </h3>
          <div class="flex flex-col gap-4">
            <!-- Match items will go here -->
          </div>
        </section>

        <!-- Two star matches -->
        <section class="mb-12">
          <h3
            class="flex items-center gap-2 text-base font-medium pb-2 border-b border-gray-200 mb-4"
          >
            <span>★★</span>
            <span class="text-gray-600 font-normal">Match</span>
          </h3>
          <div class="flex flex-col gap-4">
            <!-- Match items will go here -->
          </div>
        </section>

        <!-- One star matches -->
        <section class="mb-12">
          <h3
            class="flex items-center gap-2 text-base font-medium pb-2 border-b border-gray-200 mb-4"
          >
            <span>★</span>
            <span class="text-gray-600 font-normal">Slight match</span>
          </h3>
          <div class="flex flex-col gap-4">
            <!-- Match items will go here -->
          </div>
        </section>
      </div>
    {/if}
  </main>
</div>
