<script lang="ts">
  import {
    blockUntilReady,
    defaultDevice,
    init,
    jit,
    lax,
    nn,
    numpy as np,
    random,
    tree,
    valueAndGrad,
  } from "@jax-js/jax";
  import {
    adam,
    applyUpdates,
    chain,
    clipByGlobalNorm,
    type OptState,
  } from "@jax-js/optax";
  import { shuffle } from "es-toolkit";
  import { onDestroy, onMount } from "svelte";

  import LineChart from "$lib/chart/LineChart.svelte";
  import {
    type CharDataset,
    fetchTinyShakespeare,
    makeCharDataset,
    tinyShakespeareUrl,
  } from "$lib/dataset/tinyshakespeare";

  type Params = {
    embed: np.Array;
    wx1: np.Array;
    wh1: np.Array;
    b1: np.Array;
    wx2: np.Array;
    wh2: np.Array;
    b2: np.Array;
    wy: np.Array;
    by: np.Array;
  };

  const batchSize = 128;
  const seqLength = 16;
  const embedSize = 128;
  const hiddenSize = 512;

  let logs = $state<string[]>([]);
  let trainMetrics = $state<
    { iteration: number; loss: number; perplexity: number }[]
  >([]);
  let epochMetrics = $state<
    { epoch: number; loss: number; perplexity: number }[]
  >([]);
  let sampleText = $state(
    "Click Run to train a 2-layer LSTM and sample Shakespeare-ish text here.",
  );
  let datasetSummary = $state("");
  let running = $state(false);
  let generating = $state(false);
  let stopping = false;
  let currentDevice = $state(defaultDevice());
  let showSettings = $state(false);

  let epochs = $state(8);
  let batchesPerEpoch = $state(30);
  let learningRate = $state(0.005);
  let temperature = $state(0.85);
  let sampleLength = $state(600);
  let seedText = $state("ROMEO:\n");

  let dataset = $state.raw<CharDataset | null>(null);
  let latestParams = $state.raw<Params | null>(null);

  function log(message: string) {
    logs.push(message);
    console.log(message);
  }

  function uniform(key: np.Array, shape: number[], scale: number): np.Array {
    return random.uniform(key, shape, { minval: -scale, maxval: scale });
  }

  function initParams(key: np.Array, vocabSize: number): Params {
    const [ke, kx1, kh1, kx2, kh2, ky] = random.split(key, 6);
    const embed = uniform(ke, [vocabSize, embedSize], 0.05);
    const wx1 = uniform(
      kx1,
      [embedSize, 4 * hiddenSize],
      1 / Math.sqrt(embedSize),
    );
    const wh1 = uniform(
      kh1,
      [hiddenSize, 4 * hiddenSize],
      1 / Math.sqrt(hiddenSize),
    );
    const b1 = np.concatenate([
      np.zeros([hiddenSize]),
      np.ones([hiddenSize]), // forget-gate bias
      np.zeros([2 * hiddenSize]),
    ]);
    const wx2 = uniform(
      kx2,
      [hiddenSize, 4 * hiddenSize],
      1 / Math.sqrt(hiddenSize),
    );
    const wh2 = uniform(
      kh2,
      [hiddenSize, 4 * hiddenSize],
      1 / Math.sqrt(hiddenSize),
    );
    const b2 = np.concatenate([
      np.zeros([hiddenSize]),
      np.ones([hiddenSize]), // forget-gate bias
      np.zeros([2 * hiddenSize]),
    ]);
    const wy = uniform(ky, [hiddenSize, vocabSize], 1 / Math.sqrt(hiddenSize));
    const by = np.zeros([vocabSize]);
    return { embed, wx1, wh1, b1, wx2, wh2, b2, wy, by };
  }

  function lstmLayer(
    input: np.Array,
    h: np.Array,
    c: np.Array,
    wx: np.Array,
    wh: np.Array,
    b: np.Array,
  ): [np.Array, np.Array] {
    const z = np.dot(input, wx).add(np.dot(h, wh)).add(b);

    const inputGate = nn.sigmoid(z.ref.slice([], [0, hiddenSize]));
    const forgetGate = nn.sigmoid(
      z.ref.slice([], [hiddenSize, 2 * hiddenSize]),
    );
    const candidate = np.tanh(
      z.ref.slice([], [2 * hiddenSize, 3 * hiddenSize]),
    );
    const outputGate = nn.sigmoid(
      z.slice([], [3 * hiddenSize, 4 * hiddenSize]),
    );

    c = forgetGate.mul(c).add(inputGate.mul(candidate));
    h = outputGate.mul(np.tanh(c.ref));
    return [h, c];
  }

  function makeLoss(vocabSize: number) {
    return jit((params: Params, x: np.Array, y: np.Array): np.Array => {
      const denom = x.shape[0] * seqLength;
      let h1 = np.zeros([x.shape[0], hiddenSize]);
      let c1 = np.zeros([x.shape[0], hiddenSize]);
      let h2 = np.zeros([x.shape[0], hiddenSize]);
      let c2 = np.zeros([x.shape[0], hiddenSize]);
      let total = np.zeros([]);

      for (let t = 0; t < seqLength; t++) {
        const xT = x.ref.slice([], t);
        const xOneHot = lax.stopGradient(nn.oneHot(xT, vocabSize));
        const emb = np.dot(xOneHot, params.embed.ref);
        [h1, c1] = lstmLayer(
          emb,
          h1,
          c1,
          params.wx1.ref,
          params.wh1.ref,
          params.b1.ref,
        );
        [h2, c2] = lstmLayer(
          h1.ref,
          h2,
          c2,
          params.wx2.ref,
          params.wh2.ref,
          params.b2.ref,
        );

        const logits = np.dot(h2.ref, params.wy.ref).add(params.by.ref);
        const yT = y.ref.slice([], t);
        const yOneHot = lax.stopGradient(nn.oneHot(yT, vocabSize));
        const targetLogProb = nn.logSoftmax(logits).mul(yOneHot).sum();
        total = total.sub(targetLogProb);
      }

      h1.dispose();
      c1.dispose();
      h2.dispose();
      c2.dispose();
      x.dispose();
      y.dispose();
      tree.dispose(params);
      return total.div(denom);
    });
  }

  const lstmStep = jit(
    (
      params: Params,
      x: np.Array,
      h1: np.Array,
      c1: np.Array,
      h2: np.Array,
      c2: np.Array,
    ) => {
      const emb = params.embed.ref.slice(x);
      [h1, c1] = lstmLayer(
        emb,
        h1,
        c1,
        params.wx1.ref,
        params.wh1.ref,
        params.b1.ref,
      );
      [h2, c2] = lstmLayer(
        h1.ref,
        h2,
        c2,
        params.wx2.ref,
        params.wh2.ref,
        params.b2.ref,
      );
      const logits = np.dot(h2.ref, params.wy.ref).add(params.by.ref);
      tree.dispose(params);
      return [logits, h1, c1, h2, c2] as [
        np.Array,
        np.Array,
        np.Array,
        np.Array,
        np.Array,
      ];
    },
  );

  async function ensureDataset(): Promise<CharDataset> {
    if (dataset) return dataset;

    log("=> Loading Tiny Shakespeare from jsDelivr or OPFS cache...");
    const started = performance.now();
    const text = await fetchTinyShakespeare();
    dataset = makeCharDataset(text);
    const duration = performance.now() - started;
    datasetSummary = `${dataset.text.length.toLocaleString()} chars, ${dataset.idxToChar.length} unique symbols`;
    log(`=> Loaded ${datasetSummary} in ${duration.toFixed(1)} ms`);
    return dataset;
  }

  function batchStarts(encoded: Int32Array): number[] {
    const batchChars = Math.floor((encoded.length - 1) / batchSize);
    const starts: number[] = [];
    for (let start = 0; start + seqLength < batchChars; start += seqLength) {
      starts.push(start);
    }
    return starts;
  }

  function makeBatch(encoded: Int32Array, start: number) {
    const batchChars = Math.floor((encoded.length - 1) / batchSize);
    const xBuf = new Int32Array(batchSize * seqLength);
    const yBuf = new Int32Array(batchSize * seqLength);

    for (let batch = 0; batch < batchSize; batch++) {
      const offset = batchChars * batch + start;
      for (let t = 0; t < seqLength; t++) {
        const i = batch * seqLength + t;
        xBuf[i] = encoded[offset + t];
        yBuf[i] = encoded[offset + t + 1];
      }
    }

    return {
      x: np.array(xBuf, { dtype: np.int32 }).reshape([batchSize, seqLength]),
      y: np.array(yBuf, { dtype: np.int32 }).reshape([batchSize, seqLength]),
    };
  }

  function perplexity(loss: number): number {
    return Math.exp(Math.min(loss, 20));
  }

  function sanitizeSeed(seed: string, data: CharDataset): string {
    const filtered = Array.from(seed)
      .filter((char) => data.charToIdx[char] !== undefined)
      .join("");
    return filtered || "ROMEO:\n";
  }

  function sampleFromLogits(logits: number[], temp: number): number {
    const temperature = Math.max(temp, 0.05);
    const scaled = logits.map((x) => x / temperature);
    const maxLogit = Math.max(...scaled);
    const weights = scaled.map((x) => Math.exp(x - maxLogit));
    const sum = weights.reduce((a, b) => a + b, 0);

    if (!Number.isFinite(sum) || sum <= 0) {
      return logits.indexOf(Math.max(...logits));
    }

    let r = Math.random() * sum;
    for (let i = 0; i < weights.length; i++) {
      r -= weights[i];
      if (r <= 0) return i;
    }
    return weights.length - 1;
  }

  async function generateSample(
    params: Params,
    data: CharDataset,
    seed: string,
    length: number,
    temp: number,
  ): Promise<string> {
    const cleanSeed = sanitizeSeed(seed, data);
    let output = cleanSeed;
    let h1 = np.zeros([1, hiddenSize]);
    let c1 = np.zeros([1, hiddenSize]);
    let h2 = np.zeros([1, hiddenSize]);
    let c2 = np.zeros([1, hiddenSize]);
    let logits: np.Array | null = null;

    try {
      for (const char of cleanSeed) {
        logits?.dispose();
        const x = np.array([data.charToIdx[char]], { dtype: np.int32 });
        const [nextLogits, nextH1, nextC1, nextH2, nextC2] = lstmStep(
          tree.ref(params),
          x,
          h1,
          c1,
          h2,
          c2,
        );
        logits = nextLogits;
        h1 = nextH1;
        c1 = nextC1;
        h2 = nextH2;
        c2 = nextC2;
      }

      for (let i = 0; i < length; i++) {
        if (!logits) break;
        const row = (await logits.slice(0).jsAsync()) as number[];
        logits = null;
        const idx = sampleFromLogits(row, temp);
        output += data.idxToChar[idx];

        const x = np.array([idx], { dtype: np.int32 });
        const [nextLogits, nextH1, nextC1, nextH2, nextC2] = lstmStep(
          tree.ref(params),
          x,
          h1,
          c1,
          h2,
          c2,
        );
        logits = nextLogits;
        h1 = nextH1;
        c1 = nextC1;
        h2 = nextH2;
        c2 = nextC2;

        if (i % 32 === 0) await blockUntilReady([h1, c1, h2, c2]);
      }

      return output;
    } finally {
      logits?.dispose();
      h1.dispose();
      c1.dispose();
      h2.dispose();
      c2.dispose();
      tree.dispose(params);
    }
  }

  async function regenerate() {
    if (!latestParams || !dataset || running || generating) return;
    generating = true;
    try {
      sampleText = await generateSample(
        tree.ref(latestParams),
        dataset,
        seedText,
        sampleLength,
        temperature,
      );
    } finally {
      generating = false;
    }
  }

  async function run() {
    if (running) return;
    running = true;
    stopping = false;
    generating = false;
    logs = [];
    trainMetrics = [];
    epochMetrics = [];
    sampleText = "Training... first sample appears after epoch 1.";

    tree.dispose(latestParams);
    latestParams = null;

    let params: Params | null = null;
    let optState: OptState | null = null;

    try {
      const data = await ensureDataset();
      const loss = makeLoss(data.idxToChar.length);
      const starts = batchStarts(data.encoded);
      const numBatches = Math.min(batchesPerEpoch, starts.length);
      const totalTokens = numBatches * batchSize * seqLength;

      log(
        `=> Model: 2-layer LSTM, vocab=${data.idxToChar.length}, embedding=${embedSize}, hidden=${hiddenSize}`,
      );
      log(
        `=> Training ${numBatches} batches/epoch (${totalTokens.toLocaleString()} characters) for ${epochs} epochs`,
      );
      log("=> First batch traces the unrolled LSTM; it may take a moment.");

      params = initParams(random.key(0), data.idxToChar.length);
      await blockUntilReady(params);

      const solver = chain(clipByGlobalNorm(1.0), adam(learningRate));
      optState = solver.init(tree.ref(params));

      let iteration = 0;
      for (let epoch = 0; epoch < epochs; epoch++) {
        log(`=> Epoch ${epoch + 1}/${epochs}`);
        const epochLosses: number[] = [];
        const shuffledStarts = shuffle(starts).slice(0, numBatches);

        for (let batch = 0; batch < shuffledStarts.length; batch++) {
          if (stopping) break;
          const { x, y } = makeBatch(data.encoded, shuffledStarts[batch]);
          const started = performance.now();
          const [lossVal, grads] = valueAndGrad(loss)(tree.ref(params), x, y);
          let updates: Params;
          [updates, optState] = solver.update(grads, optState);
          params = applyUpdates(params, updates);
          await blockUntilReady(params);
          const duration = performance.now() - started;
          const lossNumber = (await lossVal.jsAsync()) as number;
          const ppl = perplexity(lossNumber);

          iteration += 1;
          epochLosses.push(lossNumber);
          trainMetrics.push({ iteration, loss: lossNumber, perplexity: ppl });
          log(
            `batch ${batch + 1}/${numBatches} in ${duration.toFixed(1)} ms, loss ${lossNumber.toFixed(3)}, ppl ${ppl.toFixed(1)}`,
          );
        }

        if (epochLosses.length > 0) {
          const avgLoss =
            epochLosses.reduce((sum, loss) => sum + loss, 0) /
            epochLosses.length;
          const avgPpl = perplexity(avgLoss);
          epochMetrics.push({
            epoch: epoch + 1,
            loss: avgLoss,
            perplexity: avgPpl,
          });
          log(
            `=> Epoch ${epoch + 1} avg loss ${avgLoss.toFixed(3)}, ppl ${avgPpl.toFixed(1)}`,
          );
        }

        tree.dispose(latestParams);
        latestParams = tree.ref(params);

        if (stopping) break;
        log("=> Sampling from the current model...");
        sampleText = await generateSample(
          tree.ref(params),
          data,
          seedText,
          sampleLength,
          temperature,
        );
      }

      if (stopping) {
        log("=> Stopped.");
      } else {
        log(
          "=> Done. Try changing the seed or temperature and sampling again.",
        );
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      log(`ERROR: ${message}`);
      throw error;
    } finally {
      tree.dispose(params);
      tree.dispose(optState);
      running = false;
      stopping = false;
    }
  }

  function stop() {
    stopping = true;
  }

  onMount(async () => {
    const devices = await init("webgpu");
    // if (devices.includes("webgpu")) defaultDevice("webgpu");
    currentDevice = defaultDevice();
  });

  onDestroy(() => {
    tree.dispose(latestParams);
    latestParams = null;
  });
</script>

<svelte:head>
  <title>char-rnn + jax-js</title>
</svelte:head>

<main class="p-4">
  <section class="max-w-3xl">
    <h1 class="text-2xl mb-4">char-rnn + jax-js</h1>

    <p class="mb-4">
      A browser training job in the spirit of Karpathy's
      <a
        class="underline"
        target="_blank"
        href="https://karpathy.github.io/2015/05/21/rnn-effectiveness/"
        >Unreasonable Effectiveness of Recurrent Neural Networks</a
      >. This trains a small character-level LSTM on Karpathy's Tiny Shakespeare
      dataset, then samples new Elizabethan-ish text.
    </p>

    <p class="mb-4">
      The model is a hand-written 2-layer LSTM, an embedding table, and a linear
      readout. Training uses <code>valueAndGrad()</code>, Adam, gradient
      clipping, and <code>jit()</code>-compiled kernels.
    </p>

    <p class="mb-4 text-sm">
      Data source:
      <a class="underline" target="_blank" href={tinyShakespeareUrl}
        >karpathy/char-rnn/data/tinyshakespeare/input.txt</a
      >. Running on <strong>{currentDevice}</strong>{datasetSummary
        ? ` with ${datasetSummary}`
        : ""}.
    </p>

    <div class="mb-8">
      <div class="flex flex-wrap gap-2 items-center">
        {#if !running}
          <button onclick={run}>Run</button>
        {:else}
          <button onclick={stop}>Stop</button>
        {/if}
        <button
          onclick={regenerate}
          disabled={!latestParams || !dataset || running || generating}
        >
          {generating ? "Sampling..." : "Sample again"}
        </button>
        <button onclick={() => (showSettings = !showSettings)}>
          Settings
          <span
            class="inline-block transform transition-transform {showSettings
              ? 'rotate-180'
              : ''}"
            style="font-size: 10px;">▼</span
          >
        </button>
      </div>

      {#if showSettings}
        <div class="mt-2 p-3 border rounded bg-gray-50 text-sm space-y-3">
          <div class="grid sm:grid-cols-2 gap-3">
            <label>
              Epochs
              <input
                type="number"
                min="1"
                max="50"
                bind:value={epochs}
                disabled={running}
              />
            </label>
            <label>
              Batches / epoch
              <input
                type="number"
                min="1"
                max="250"
                bind:value={batchesPerEpoch}
                disabled={running}
              />
            </label>
            <label>
              Learning rate
              <input
                type="number"
                min="0.0001"
                max="0.02"
                step="0.0001"
                bind:value={learningRate}
                disabled={running}
              />
            </label>
            <label>
              Sampling temperature
              <input
                type="number"
                min="0.1"
                max="2"
                step="0.05"
                bind:value={temperature}
              />
            </label>
            <label>
              Sample length
              <input
                type="number"
                min="80"
                max="2000"
                bind:value={sampleLength}
              />
            </label>
          </div>
          <label>
            Seed text
            <textarea rows="3" bind:value={seedText}></textarea>
          </label>
        </div>
      {/if}
    </div>
  </section>

  <div class="grid lg:grid-cols-3 gap-4 my-6">
    <div class="h-[220px] border border-gray-400 rounded">
      <LineChart
        title="Train Loss"
        data={trainMetrics}
        x="iteration"
        y="loss"
      />
    </div>
    <div class="h-[220px] border border-gray-400 rounded">
      <LineChart
        title="Epoch Loss & Perplexity"
        data={epochMetrics}
        x="epoch"
        y={["loss", "perplexity"]}
      />
    </div>
    <div class="h-[220px] border border-gray-400 rounded overflow-hidden">
      <div class="h-full flex flex-col">
        <p class="shrink-0 text-sm text-center my-1">
          Generated Shakespeare sample
        </p>
        <pre
          class="grow min-h-0 overflow-auto px-3 pb-3 text-xs whitespace-pre-wrap font-mono">{sampleText}</pre>
      </div>
    </div>
  </div>

  <div
    class="font-mono text-sm rounded bg-gray-900 px-4 py-2 h-[520px] overflow-y-scroll mt-8"
  >
    {#each logs as line}
      <div class="text-white whitespace-pre-wrap">{line}</div>
    {/each}
  </div>
</main>

<style lang="postcss">
  @reference "$app.css";

  button {
    @apply border rounded px-2 hover:bg-gray-100 active:scale-95 disabled:opacity-50 disabled:active:scale-100;
  }

  label {
    @apply flex flex-col gap-1 font-semibold;
  }

  input,
  textarea {
    @apply border rounded px-2 py-1 font-normal bg-white;
  }
</style>
