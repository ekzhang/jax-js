<script lang="ts">
  import { defaultDevice, init, numpy as np, tree } from "@jax-js/jax";
  import { safetensors, tokenizers } from "@jax-js/loaders";
  import { GithubIcon } from "@lucide/svelte";
  import { onMount, tick } from "svelte";

  import DownloadManager from "$lib/common/DownloadManager.svelte";
  import MarkdownMessage from "./MarkdownMessage.svelte";
  import {
    createGemmaState,
    fromSafetensors,
    GEMMA_CONFIG,
    type GemmaModel,
    runGemmaPrefill,
    runGemmaStep,
  } from "./gemma";

  const MODEL_BASE =
    "https://huggingface.co/ekzhang/jax-js-models/resolve/main/gemma-3-270m";
  const WEIGHTS_URL = `${MODEL_BASE}/model-it-fp16.safetensors`;
  const TOKENIZER_URL = `${MODEL_BASE}/tokenizer.model`;

  // Gemma chat-template control tokens in tokenizer.model.
  const START_OF_TURN_TOKEN = 105;
  const END_OF_TURN_TOKEN = 106;

  type ChatMessage = {
    id: number;
    role: "user" | "assistant";
    content: string;
  };

  type ChatBackend = "webgpu" | "wasm";

  const BACKEND_LABEL: Record<ChatBackend, string> = {
    webgpu: "WebGPU",
    wasm: "Wasm",
  };
  const CHAT_BACKENDS: ChatBackend[] = ["webgpu", "wasm"];

  let _model: GemmaModel | null = null;
  let _modelBackend: ChatBackend | null = null;
  let _tokenizer: tokenizers.SentencePiece | null = null;

  let downloadManager: DownloadManager;
  let scrollContainer: HTMLElement;
  let nextMessageId = 0;

  let hasModel = $state(false);
  let messages = $state<ChatMessage[]>([]);
  let input = $state("");
  let running = $state(false);
  let status = $state("");
  let prefillCount = $state(0);
  let prefillElapsedMs = $state(0);
  let decodeCount = $state(0);
  let decodeElapsedMs = $state(0);

  let maxNewTokens = $state(2048);
  let temperature = $state(0.8);
  let topK = $state(64);
  let topP = $state(0.95);
  let backend = $state<ChatBackend>("webgpu");
  let availableBackends = $state<ChatBackend[]>([]);
  let checkedBackends = $state(false);
  let optionsOpen = $state(false);
  let optionsDetails: HTMLDetailsElement | undefined;

  const prefillTps = $derived(
    prefillElapsedMs > 0 ? prefillCount / (prefillElapsedMs / 1000) : 0,
  );
  const decodeTps = $derived(
    decodeElapsedMs > 0 ? decodeCount / (decodeElapsedMs / 1000) : 0,
  );

  async function scrollToBottom() {
    await tick();
    scrollContainer?.scrollTo({
      top: scrollContainer.scrollHeight,
      behavior: "smooth",
    });
  }

  onMount(() => {
    void initializeBackendOptions().catch((error) => {
      console.warn("Failed to initialize chat backends", error);
    });
  });

  async function initializeBackendOptions() {
    const devices = await init(...CHAT_BACKENDS);
    availableBackends = CHAT_BACKENDS.filter((device) =>
      devices.includes(device),
    );
    checkedBackends = true;
    if (availableBackends.length > 0 && !availableBackends.includes(backend)) {
      backend = availableBackends.includes("webgpu") ? "webgpu" : "wasm";
    }
  }

  function disposeModel() {
    if (_model) tree.dispose(_model);
    _model = null;
    _modelBackend = null;
    hasModel = false;
  }

  function handleBackendChange(event: Event) {
    const nextBackend = (event.currentTarget as HTMLSelectElement)
      .value as ChatBackend;
    if (nextBackend === backend) return;
    backend = nextBackend;
    disposeModel();
  }

  async function setupDevice() {
    status = `Initializing ${BACKEND_LABEL[backend]}…`;
    if (!checkedBackends) await initializeBackendOptions();

    let selectedBackend = backend;
    if (!availableBackends.includes(selectedBackend)) {
      if (selectedBackend === "webgpu" && availableBackends.includes("wasm")) {
        selectedBackend = "wasm";
      } else {
        throw new Error(
          `${BACKEND_LABEL[selectedBackend]} is not available in this browser.`,
        );
      }
    }

    backend = selectedBackend;
    defaultDevice(selectedBackend);
  }

  async function getTokenizer(): Promise<tokenizers.SentencePiece> {
    if (_tokenizer) return _tokenizer;
    status = "Downloading tokenizer…";
    const data = await downloadManager.fetch("Gemma tokenizer", TOKENIZER_URL);
    _tokenizer = tokenizers.SentencePiece.fromBinary(data);
    return _tokenizer;
  }

  async function getModel(): Promise<GemmaModel> {
    if (_model && _modelBackend === backend) return _model;
    if (_model) disposeModel();

    status = "Downloading model weights…";
    const data = await downloadManager.fetch(
      "Gemma 3 270M weights",
      WEIGHTS_URL,
    );

    status = "Parsing checkpoint…";
    const weights = safetensors.parse(data);

    const weightDtype = backend === "wasm" ? np.float32 : np.float16;
    status =
      backend === "wasm"
        ? "Preparing float32 weights for Wasm…"
        : "Uploading weights to WebGPU…";
    _model = await fromSafetensors(weights, weightDtype);
    _modelBackend = backend;
    hasModel = true;
    return _model;
  }

  function visibleTokens(
    tokenizer: tokenizers.SentencePiece,
    tokens: number[],
  ) {
    return tokens.filter(
      (token) =>
        token !== GEMMA_CONFIG.padTokenId &&
        token !== tokenizer.bosToken &&
        token !== tokenizer.eosToken &&
        token !== START_OF_TURN_TOKEN &&
        token !== END_OF_TURN_TOKEN,
    );
  }

  function decodeTokens(tokenizer: tokenizers.SentencePiece, tokens: number[]) {
    return tokenizer.decode(visibleTokens(tokenizer, tokens));
  }

  function formatPrompt(history: ChatMessage[]) {
    // Matches the Gemma chat template, excluding the BOS token; we add BOS as
    // token ID separately so SentencePiece doesn't treat it as normal text.
    let text = "";
    for (const message of history) {
      const content = message.content.trim();
      if (content === "") continue;
      const role = message.role === "assistant" ? "model" : "user";
      text += `<start_of_turn>${role}\n${content}<end_of_turn>\n`;
    }
    text += "<start_of_turn>model\n";
    return text;
  }

  function updateMessage(id: number, content: string) {
    messages = messages.map((message) =>
      message.id === id ? { ...message, content } : message,
    );
  }

  function sampleLogits(
    logits: Float32Array,
    opts: {
      temperature: number;
      topK: number;
      topP: number;
    },
  ): number {
    const k = Math.max(1, Math.min(opts.topK, logits.length));
    const candidates: { id: number; logit: number }[] = [];

    for (let id = 0; id < logits.length; id++) {
      const logit = logits[id];
      if (Number.isNaN(logit)) continue;

      // Keep the candidates in order of descending logit value, maintain insertion.
      if (
        candidates.length < k ||
        logit > candidates[candidates.length - 1].logit
      ) {
        let insertIndex = 0;
        while (
          insertIndex < candidates.length &&
          logit < candidates[insertIndex].logit
        ) {
          insertIndex++;
        }
        candidates.splice(insertIndex, 0, { id, logit });
        if (candidates.length > k) {
          candidates.pop();
        }
      }
    }

    if (candidates.length === 0) {
      throw new Error("Model returned all-NaN logits.");
    }
    if (opts.temperature <= 0) return candidates[0].id;

    const maxLogit = candidates[0].logit;
    if (!Number.isFinite(maxLogit)) return candidates[0].id;

    const probs = candidates.map((candidate) =>
      Math.exp((candidate.logit - maxLogit) / opts.temperature),
    );
    const total = probs.reduce((a, b) => a + b, 0);
    if (!Number.isFinite(total) || total <= 0) return candidates[0].id;

    let keptTotal = 0;
    let kept = 0;
    for (; kept < candidates.length; kept++) {
      keptTotal += probs[kept];
      if (keptTotal / total >= opts.topP) {
        kept++;
        break;
      }
    }
    if (kept === 0) kept = 1;

    let r = Math.random() * keptTotal;
    for (let i = 0; i < kept; i++) {
      r -= probs[i];
      if (r <= 0) return candidates[i].id;
    }
    return candidates[kept - 1].id;
  }

  async function sampleNextToken(logits: np.Array): Promise<number> {
    const data = await logits.data();
    return sampleLogits(data as Float32Array, {
      temperature,
      topK,
      topP,
    });
  }

  async function runChat(history: ChatMessage[], assistantMessageId: number) {
    prefillCount = prefillElapsedMs = 0;
    decodeCount = decodeElapsedMs = 0;

    await setupDevice();
    const tokenizer = await getTokenizer();
    const model = await getModel();

    const promptTokens = [
      tokenizer.bosToken,
      ...tokenizer.encode(formatPrompt(history)),
    ];
    const generatedTokens: number[] = [];
    const state = createGemmaState({
      dtype: backend === "wasm" ? np.float32 : np.float16,
    });
    const inputIds = np.array(promptTokens, { dtype: np.uint32 });
    let logits: np.Array | null = null;
    const startTime = performance.now();

    try {
      status = `Reading ${promptTokens.length} context tokens…`;
      logits = runGemmaPrefill(tree.ref(model), inputIds, state);
      scrollToBottom();

      for (let i = 0; i < maxNewTokens; i++) {
        status = `Sampling ${i + 1}/${maxNewTokens}…`;
        const sampledLogits = logits;
        logits = null; // logits.data() consumes this array; avoid disposing it again in finally.
        const stopTokens = [tokenizer.eosToken, END_OF_TURN_TOKEN];
        // if (i === 4 || i === 5) profiler.startTrace();
        const nextToken = await sampleNextToken(sampledLogits);
        // if (i === 4 || i === 5) profiler.stopTrace();

        console.debug("Gemma sampled token", {
          nextToken,
          piece: tokenizer.decode([nextToken]),
        });

        if (stopTokens.includes(nextToken)) {
          break;
        }

        generatedTokens.push(nextToken);
        if (i === 0) {
          prefillCount = promptTokens.length;
          prefillElapsedMs = performance.now() - startTime;
        } else {
          decodeCount++;
          decodeElapsedMs = performance.now() - startTime - prefillElapsedMs;
        }
        updateMessage(
          assistantMessageId,
          decodeTokens(tokenizer, generatedTokens),
        );
        scrollToBottom();

        if (i === maxNewTokens - 1) break;
        status = `Running token ${i + 1}/${maxNewTokens}…`;
        logits = runGemmaStep(tree.ref(model), nextToken, state);
      }

      status = "✔️ Done";
      if (generatedTokens.length === 0)
        updateMessage(assistantMessageId, "(end of text)");
    } finally {
      logits?.dispose();
      tree.dispose(state);
    }
  }

  async function sendMessage() {
    const text = input.trim();
    if (text === "" || running) return;

    input = "";
    status = "";
    running = true;

    const userMessage: ChatMessage = {
      id: nextMessageId++,
      role: "user",
      content: text,
    };
    const assistantMessage: ChatMessage = {
      id: nextMessageId++,
      role: "assistant",
      content: "",
    };
    const history = [...messages, userMessage];
    messages = [...history, assistantMessage];
    await scrollToBottom();

    try {
      await runChat(history, assistantMessage.id);
    } catch (error) {
      console.error(error);
      updateMessage(
        assistantMessage.id,
        error instanceof Error ? `Error: ${error.message}` : `Error: ${error}`,
      );
    } finally {
      running = false;
      await scrollToBottom();
    }
  }

  function newChat() {
    if (running) return;
    messages = [];
    input = "";
    status = "";
    prefillCount = prefillElapsedMs = 0;
    decodeCount = decodeElapsedMs = 0;
  }
</script>

<title>jax-js model chat</title>

<!-- Close options menu on outside click. -->
<svelte:window
  onclick={(event: MouseEvent) => {
    if (
      optionsOpen &&
      event.target instanceof Node &&
      !optionsDetails?.contains(event.target)
    ) {
      optionsOpen = false;
    }
  }}
/>

<DownloadManager bind:this={downloadManager} />

<main class="h-dvh overflow-hidden bg-white text-gray-950 flex flex-col">
  <header class="shrink-0 border-b border-gray-200 px-4 py-2">
    <div class="mx-auto max-w-4xl flex items-center justify-between gap-4">
      <div>
        <h1 class="font-semibold">
          Chat
          <span
            class="font-normal border rounded-md px-1 ml-1 text-sm text-gray-500 border-gray-300"
            >Gemma 3 270M</span
          >
        </h1>
        <p class="text-sm text-gray-500">
          Running locally with jax-js + {BACKEND_LABEL[backend]}
        </p>
      </div>

      <div class="flex items-center gap-2">
        <details
          bind:this={optionsDetails}
          bind:open={optionsOpen}
          class="relative"
        >
          <summary class="small-btn list-none cursor-pointer">Options</summary>
          <div
            class="absolute right-0 z-10 mt-2 w-72 rounded-2xl border border-gray-200 bg-white p-4 shadow-xl"
          >
            <div class="space-y-4 text-sm">
              <div>
                <label class="block text-gray-700">
                  Backend
                  <select
                    class="mt-1 w-full rounded-lg border border-gray-300 px-2 py-1"
                    value={backend}
                    onchange={handleBackendChange}
                    disabled={running || !checkedBackends}
                  >
                    {#if checkedBackends}
                      {#each availableBackends as device}
                        <option value={device}>{BACKEND_LABEL[device]}</option>
                      {/each}
                    {:else}
                      <option value={backend}>Initializing…</option>
                    {/if}
                  </select>
                </label>

                <p class="mt-2 text-xs text-gray-500">
                  WebGPU uses fp16 weights. Wasm casts weights to fp32 on load.
                </p>
              </div>

              <hr class="border-gray-200" />

              <label class="block text-gray-700">
                Max new tokens
                <input
                  type="number"
                  min="1"
                  max="8192"
                  class="mt-1 w-full rounded-lg border border-gray-300 px-2 py-1"
                  bind:value={maxNewTokens}
                />
              </label>

              <label class="block text-gray-700">
                Top-k
                <input
                  type="number"
                  min="1"
                  max="256"
                  class="mt-1 w-full rounded-lg border border-gray-300 px-2 py-1"
                  bind:value={topK}
                />
              </label>

              <label class="block text-gray-700">
                Temperature: {temperature.toFixed(2)}
                <input
                  type="range"
                  min="0"
                  max="1.5"
                  step="0.05"
                  class="mt-1 w-full"
                  bind:value={temperature}
                />
              </label>

              <label class="block text-gray-700">
                Top-p: {topP.toFixed(2)}
                <input
                  type="range"
                  min="0.1"
                  max="1"
                  step="0.01"
                  class="mt-1 w-full"
                  bind:value={topP}
                />
              </label>

              <p class="text-xs text-gray-500">
                KV cache is allocated dynamically for the current chat.
              </p>
            </div>
          </div>
        </details>

        <button class="small-btn" onclick={newChat} disabled={running}>
          New chat
        </button>
        <a
          class="small-btn"
          target="_blank"
          aria-label="View source"
          href="https://github.com/ekzhang/jax-js/tree/main/website/src/routes/chat"
        >
          <GithubIcon size={18} />
        </a>
      </div>
    </div>
  </header>

  <section
    bind:this={scrollContainer}
    class="min-h-0 flex-1 overflow-y-auto px-4 py-6"
  >
    <div class="mx-auto max-w-3xl">
      {#if messages.length === 0}
        <div class="py-24 text-center">
          <h2 class="text-2xl font-semibold mb-2">Talk to an LLM</h2>
          <p class="text-gray-500 max-w-md mx-auto">
            The first message downloads and caches a 536&nbsp;MB fp16
            checkpoint. Everything runs locally in your browser.
          </p>
        </div>
      {:else}
        <div class="space-y-5">
          {#each messages as message (message.id)}
            <div
              class="flex"
              class:justify-end={message.role === "user"}
              class:justify-start={message.role === "assistant"}
            >
              <div
                class="message-bubble"
                class:user-bubble={message.role === "user"}
                class:assistant-bubble={message.role === "assistant"}
              >
                {#if message.content === "" && message.role === "assistant" && running}
                  <span class="inline-flex gap-1" aria-label="Generating">
                    <span class="typing-dot"></span>
                    <span class="typing-dot animation-delay-150"></span>
                    <span class="typing-dot animation-delay-300"></span>
                  </span>
                {:else if message.role === "assistant"}
                  <MarkdownMessage content={message.content} />
                {:else}
                  {message.content}
                {/if}
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </div>
  </section>

  <footer class="shrink-0 border-t border-gray-200 bg-white px-4 py-4">
    <form
      class="mx-auto max-w-3xl"
      onsubmit={(event) => {
        event.preventDefault();
        void sendMessage();
      }}
    >
      <div class="rounded-2xl border border-gray-300 bg-white p-2">
        <textarea
          class="min-h-11 max-h-40 w-full resize-none px-2 py-2 outline-none disabled:bg-white disabled:text-gray-400"
          rows="2"
          placeholder={hasModel
            ? "Send a message…"
            : "Send a message… (will download model)"}
          bind:value={input}
          disabled={running}
          onkeydown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void sendMessage();
            }
          }}></textarea>

        <div
          class="flex items-center justify-between gap-3 border-t border-gray-100 pt-2"
        >
          <div class="min-h-5 text-xs text-gray-500 tabular-nums">
            {#if status}
              {status}
              {#if prefillCount > 0}
                · Prefill {prefillTps.toFixed(1)} tok/s
              {/if}
              {#if decodeCount > 0}
                · Decode {decodeTps.toFixed(1)} tok/s
              {/if}
            {/if}
          </div>

          <button
            class="send-btn"
            type="submit"
            disabled={running || input.trim() === ""}
          >
            {running ? "Generating" : "Send"}
          </button>
        </div>
      </div>
    </form>
  </footer>
</main>

<style lang="postcss">
  @reference "$app.css";

  .small-btn {
    @apply inline-flex items-center justify-center rounded-full border border-gray-300 px-3 py-1.5 text-sm whitespace-nowrap;
    @apply hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed;
  }

  .send-btn {
    @apply rounded-full bg-black px-4 py-1.5 text-sm font-medium text-white;
    @apply disabled:cursor-not-allowed disabled:bg-gray-300;
  }

  .message-bubble {
    @apply max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-3 leading-relaxed;
  }

  .user-bubble {
    @apply bg-black text-white rounded-br-md;
  }

  .assistant-bubble {
    @apply bg-gray-100 text-gray-950 rounded-bl-md;
  }

  .typing-dot {
    @apply h-2 w-2 rounded-full bg-gray-400 animate-pulse;
  }

  .animation-delay-150 {
    animation-delay: 150ms;
  }

  .animation-delay-300 {
    animation-delay: 300ms;
  }
</style>
