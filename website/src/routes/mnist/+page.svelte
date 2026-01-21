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
  import { adam, applyUpdates } from "@jax-js/optax";
  import { range, shuffle } from "es-toolkit";
  import pThrottle from "p-throttle";
  import { onMount } from "svelte";

  import LineChart from "$lib/chart/LineChart.svelte";
  import { fetchMnist } from "$lib/dataset/mnist";

  let logs = $state<string[]>([]);

  function log(msg: string) {
    logs.push(msg);
    console.log(msg);
  }

  type Params = { [key: string]: np.Array };
  type ModelType = {
    init(key: np.Array): Params;
    predict(params: Params, x: np.Array): np.Array;
  };

  function maxPool2x2(x: np.Array): np.Array {
    return lax.reduceWindow(x, np.max, [2, 2], [2, 2]);
  }

  const MLP: ModelType = {
    init(key: np.Array): Params {
      const [d0, d1, d2, d3] = [784, 256, 128, 10]; // Hidden layer dimensions
      const [k11, k12, k21, k22, k31, k32] = random.split(key, 6);
      const w1 = random.uniform(k11, [d0, d1], {
        minval: -1 / Math.sqrt(d0),
        maxval: 1 / Math.sqrt(d0),
      });
      const b1 = random.uniform(k12, [d1], {
        minval: -1 / Math.sqrt(d0),
        maxval: 1 / Math.sqrt(d0),
      });
      const w2 = random.uniform(k21, [d1, d2], {
        minval: -1 / Math.sqrt(d1),
        maxval: 1 / Math.sqrt(d1),
      });
      const b2 = random.uniform(k22, [d2], {
        minval: -1 / Math.sqrt(d1),
        maxval: 1 / Math.sqrt(d1),
      });
      const w3 = random.uniform(k31, [d2, d3], {
        minval: -1 / Math.sqrt(d2),
        maxval: 1 / Math.sqrt(d2),
      });
      const b3 = random.uniform(k32, [d3], {
        minval: -1 / Math.sqrt(d2),
        maxval: 1 / Math.sqrt(d2),
      });
      return { w1, b1, w2, b2, w3, b3 };
    },

    predict: jit((params: Params, x: np.Array): np.Array => {
      // Forward pass through the network
      x = x.reshape([-1, 784]);
      const z1 = np.dot(x, params.w1).add(params.b1);
      const a1 = nn.relu(z1);
      const z2 = np.dot(a1, params.w2).add(params.b2);
      const a2 = nn.relu(z2);
      const z3 = np.dot(a2, params.w3).add(params.b3);
      return nn.logSoftmax(z3);
    }),
  };

  const ConvNet: ModelType = {
    init(key: np.Array): Params {
      const [k11, k12, k21, k22, k31, k32, k41, k42] = random.split(key, 8);
      const w1 = random.uniform(k11, [24, 1, 5, 5], {
        minval: -1 / Math.sqrt(5 * 5),
        maxval: 1 / Math.sqrt(5 * 5),
      });
      const b1 = random.uniform(k12, [24, 1, 1], {
        minval: -1 / Math.sqrt(5 * 5),
        maxval: 1 / Math.sqrt(5 * 5),
      });
      const w2 = random.uniform(k21, [32, 24, 3, 3], {
        minval: -1 / Math.sqrt(24 * 3 * 3),
        maxval: 1 / Math.sqrt(24 * 3 * 3),
      });
      const b2 = random.uniform(k22, [32, 1, 1], {
        minval: -1 / Math.sqrt(24 * 3 * 3),
        maxval: 1 / Math.sqrt(24 * 3 * 3),
      });
      const w3 = random.uniform(k31, [800, 128], {
        minval: -1 / Math.sqrt(800),
        maxval: 1 / Math.sqrt(800),
      });
      const b3 = random.uniform(k32, [128], {
        minval: -1 / Math.sqrt(800),
        maxval: 1 / Math.sqrt(800),
      });
      const w4 = random.uniform(k41, [128, 10], {
        minval: -1 / Math.sqrt(128),
        maxval: 1 / Math.sqrt(128),
      });
      const b4 = random.uniform(k42, [10], {
        minval: -1 / Math.sqrt(128),
        maxval: 1 / Math.sqrt(128),
      });
      return { w1, b1, w2, b2, w3, b3, w4, b4 };
    },

    predict: jit((params: Params, x: np.Array): np.Array => {
      // Forward pass through the network
      x = x.reshape([-1, 1, 28, 28]);
      const z1 = maxPool2x2(
        lax.convGeneralDilated(x, params.w1, [1, 1], "VALID").add(params.b1),
      );
      const a1 = nn.relu(z1); // [batch, 24, 12, 12]
      const z2 = maxPool2x2(
        lax.convGeneralDilated(a1, params.w2, [1, 1], "VALID").add(params.b2),
      );
      const a2 = nn.relu(z2); // [batch, 32, 5, 5]
      const a2flat = a2.reshape([-1, 800]); // Flatten to [batch, 800]
      const z3 = np.dot(a2flat, params.w3).add(params.b3);
      const a3 = nn.relu(z3);
      const z4 = np.dot(a3, params.w4).add(params.b4);
      return nn.logSoftmax(z4);
    }),
  };

  function lossFn(predict: (params: Params, x: np.Array) => np.Array) {
    return (params: Params, x: np.Array, y: np.Array): np.Array => {
      // Compute the negative log-likelihood loss
      const batchSize = y.shape[0];
      const logits = predict(params, x);
      return logits
        .mul(nn.oneHot(y, 10))
        .sum()
        .mul(-1 / batchSize);
    };
  }

  async function loadData(): Promise<{
    X_train: np.Array;
    y_train: np.Array;
    X_test: np.Array;
    y_test: np.Array;
  }> {
    const mnist = await fetchMnist();
    return {
      X_train: np
        .array(new Float32Array(mnist.train.images.data))
        .mul(1 / 255)
        .reshape([-1, 28, 28]),
      y_train: np.array(mnist.train.labels.data),
      X_test: np
        .array(new Float32Array(mnist.test.images.data))
        .mul(1 / 255)
        .reshape([-1, 28, 28]),
      y_test: np.array(mnist.test.labels.data),
    };
  }

  // Training metrics
  let trainMetrics = $state<{ iteration: number; loss: number }[]>([]);
  let testMetrics = $state<{ epoch: number; loss: number; acc: number }[]>([]);

  // Training and inference state
  let latestParams: Params | null = null;
  let probs: number[] = $state.raw([]);
  let running = $state(false);
  let stopping = false;

  // Settings
  let learningRate = $state(0.005);
  let logLearningRate = $state(Math.log10(0.005));
  let showSettings = $state(false);

  $effect(() => {
    learningRate = parseFloat(Math.pow(10, logLearningRate).toFixed(6));
  });
  $effect(() => {
    logLearningRate = Math.log10(learningRate);
  });

  let Model: ModelType = undefined!; // Initialized below.
  let batchSize: number = undefined!;

  let selectedModel = $state("MLP");
  // svelte-ignore state_referenced_locally
  changeModelType(selectedModel);

  function changeModelType(modelType: string) {
    if (running) return;
    switch (modelType) {
      case "MLP":
        Model = MLP;
        batchSize = 1000;
        break;
      case "ConvNet":
        Model = ConvNet;
        batchSize = 250;
        break;
      default:
        throw new Error(`Unknown model type: ${modelType}`);
    }
    tree.dispose(latestParams);
    latestParams = null;
  }

  async function run() {
    running = true;
    stopping = false;
    logs = [];
    trainMetrics = [];
    testMetrics = [];

    let params = Model.init(random.key(0));
    await blockUntilReady(params);

    const loss = lossFn(Model.predict);

    tree.dispose(latestParams);
    latestParams = tree.ref(params);

    log(`=> Loading MNIST database from CDN or cache...`);
    const startTime = performance.now();
    const { X_train, y_train, X_test, y_test } = await loadData();
    const duration = performance.now() - startTime;
    log(`=> Data loaded in ${duration.toFixed(1)} ms`);

    const solver = adam(learningRate);
    let optState = solver.init(tree.ref(params));
    let updates: Params;

    try {
      const numBatches = Math.ceil(X_train.shape[0] / batchSize);
      for (let epoch = 0; epoch < 10; epoch++) {
        log(`=> Epoch ${epoch + 1}`);
        const randomIndices = shuffle(range(X_train.shape[0]));

        for (let i = 0; i < numBatches; i++) {
          if (stopping) break;
          const indices = np.array(
            randomIndices.slice(i * batchSize, (i + 1) * batchSize),
            { dtype: np.int32 },
          );

          const startTime = performance.now();
          const X = X_train.ref.slice(indices.ref);
          const y = y_train.ref.slice(indices);
          const [lossVal, lossGrad] = valueAndGrad(loss)(
            tree.ref(params),
            X,
            y,
          );
          [updates, optState] = solver.update(lossGrad, optState);
          params = applyUpdates(params, updates);

          await blockUntilReady(params);
          const duration = performance.now() - startTime;
          const lossNumber = (await lossVal.jsAsync()) as number;
          log(
            `batch ${i}/${numBatches} completed in ${duration.toFixed(1)} ms, loss: ${lossNumber.toFixed(4)}`,
          );
          trainMetrics.push({
            iteration: epoch * numBatches + i + 1,
            loss: lossNumber,
          });
        }

        tree.dispose(latestParams);
        latestParams = tree.ref(params);

        // Retrigger the inference demo if the user has drawn something.
        if (hasDrawn) inferenceDemo();

        if (stopping) break;

        log(`=> Evaluating on test set...`);
        const testStartTime = performance.now();
        const testSize = X_test.shape[0];
        const testLoss: number[] = [];
        const testAcc: number[] = [];
        for (let i = 0; i + batchSize <= testSize; i += batchSize) {
          const X = X_test.ref.slice([i, i + batchSize]).reshape([-1, 784]);
          const y = y_test.ref.slice([i, i + batchSize]);
          testLoss.push(await loss(tree.ref(params), X.ref, y.ref).jsAsync());
          testAcc.push(
            await np
              .argmax(Model.predict(tree.ref(params), X), 1)
              .equal(y)
              .astype(np.uint32)
              .sum()
              .jsAsync(),
          );
        }
        const testDuration = performance.now() - testStartTime;
        const testLossAvg = testLoss.reduce((a, b) => a + b) / testLoss.length;
        const testAccAvg = testAcc.reduce((a, b) => a + b) / testSize;
        log(
          `=> Test acc: ${testAccAvg.toFixed(4)}, loss: ${testLossAvg.toFixed(4)}, in ${testDuration.toFixed(1)} ms`,
        );
        testMetrics.push({
          epoch: epoch + 1,
          loss: testLossAvg,
          acc: testAccAvg,
        });
      }
    } finally {
      X_train.dispose();
      y_train.dispose();
      X_test.dispose();
      y_test.dispose();
      tree.dispose(params);
      running = false;
    }
  }

  function stop() {
    stopping = true;
  }

  onMount(async () => {
    await init("webgpu");
    defaultDevice("webgpu");

    ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  });

  function normalizeImage(X: np.Array): np.Array {
    // X.shape === [28, 28]
    const [xgrid, ygrid] = np.meshgrid(
      [np.arange(28).astype(np.float32), np.arange(28).astype(np.float32)],
      { indexing: "ij" },
    );
    const dx = Math.round(13.5 - X.ref.mul(xgrid).sum().div(X.ref.sum()).js());
    const dy = Math.round(13.5 - X.ref.mul(ygrid).sum().div(X.ref.sum()).js());
    if (dx > 0) X = np.pad(X, { 0: [dx, 0] }).slice([0, 28], []);
    if (dx < 0) X = np.pad(X, { 0: [0, -dx] }).slice([-dx], []);
    if (dy > 0) X = np.pad(X, { 1: [dy, 0] }).slice([], [0, 28]);
    if (dy < 0) X = np.pad(X, { 1: [0, -dy] }).slice([], [-dy]);
    return X;
  }

  const inferenceDemo = pThrottle({ limit: 0, interval: 30 })(async () => {
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    // First, construct a 784-dimensional vector from the image data.
    const ar = new Float32Array(784);
    for (let i = 0; i < 28; i++) {
      for (let j = 0; j < 28; j++) {
        for (let l = i * 10; l < (i + 1) * 10; l++) {
          for (let k = j * 10; k < (j + 1) * 10; k++) {
            const idx = (l * 280 + k) * 4;
            const r = imgData.data[idx];
            const g = imgData.data[idx + 1];
            const b = imgData.data[idx + 2];
            // Average the RGB values to get a grayscale value.
            ar[i * 28 + j] += (1 - (r + g + b) / 3 / 255) / 100;
          }
        }
      }
    }

    if (latestParams === null) {
      log("No model available for inference. Train the model first.");
      return;
    }
    const params = tree.ref(latestParams);
    let image = np.array(ar).reshape([28, 28]);
    image = normalizeImage(image); // Mimic the MNIST train set preprocessing.
    const logits = Model.predict(params, image.reshape([1, 28, 28]));
    probs = (await np.exp(logits).slice(0).jsAsync()) as number[];
  });

  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;
  let hasDrawn = $state(false);
  let drawing = false;
  let lastPos = [0, 0];
  const lineWidth = 28;

  function coords(event: PointerEvent): [number, number] {
    const rect = canvas.getBoundingClientRect();
    return [
      (event.offsetX / rect.width) * canvas.width,
      (event.offsetY / rect.height) * canvas.height,
    ];
  }

  function drawStart(event: PointerEvent) {
    event.preventDefault();
    const [x, y] = coords(event);
    drawing = true;
    ctx.fillStyle = "black";
    ctx.beginPath();
    ctx.ellipse(x, y, lineWidth / 2, lineWidth / 2, 0, 0, Math.PI * 2);
    ctx.fill();
    lastPos = [x, y];
    hasDrawn = true;
    inferenceDemo();
  }

  function drawMove(event: PointerEvent) {
    if (!drawing) return;
    event.preventDefault();
    const [x, y] = coords(event);
    ctx.beginPath();
    ctx.moveTo(lastPos[0], lastPos[1]);
    ctx.lineTo(x, y);
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "round";
    ctx.stroke();
    lastPos = [x, y];
    inferenceDemo();
  }

  function drawEnd() {
    drawing = false;
  }

  function clearCanvas() {
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    hasDrawn = false;
    probs = [];
  }
</script>

<svelte:head>
  <title>mnist + jax-js</title>
</svelte:head>

<main class="p-4">
  <section class="max-w-3xl">
    <h1 class="text-2xl mb-4">mnist + jax-js</h1>

    <p class="mb-4">
      Let's train a neural network to classify MNIST digits, in your browser
      with <code>jax-js</code>.
    </p>

    <p class="mb-4">
      The model is a 3-layer MLP or 4-layer convolutional neural network trained
      with Adam. Each epoch has 60 (MLP) or 240 (ConvNet) randomized batches,
      with 60,000 images in total in the train set.
    </p>

    <p class="mb-4 text-sm">
      Note: This demo requires a <a
        class="underline"
        target="_blank"
        href="https://browserleaks.com/webgpu">WebGPU</a
      >-enabled browser. Works best on Chrome.
    </p>

    <div class="mb-8">
      <div class="flex gap-2">
        <select
          bind:value={selectedModel}
          onchange={() => changeModelType(selectedModel)}
          disabled={running}
        >
          <option value="MLP">MLP</option>
          <option value="ConvNet">ConvNet</option>
        </select>
        <button
          onclick={() => (showSettings = !showSettings)}
          class="text-sm flex items-center gap-1"
        >
          Settings
          <span
            class="transform transition-transform {showSettings
              ? 'rotate-180'
              : ''}"
            style="font-size: 10px;">▼</span
          >
        </button>
        {#if !running}
          <button onclick={run}>Run</button>
        {:else}
          <button onclick={stop}>Stop</button>
        {/if}
      </div>

      {#if showSettings}
        <div class="mt-2 p-3 border rounded bg-gray-50">
          <div class="flex items-center gap-3 mb-2">
            <label for="learning-rate-slider" class="font-semibold text-sm"
              >Learning rate:</label
            >
            <input
              id="learning-rate-slider"
              type="range"
              min="-3"
              max="-2"
              step="0.01"
              bind:value={logLearningRate}
              class="w-32"
              disabled={running}
            />
            <input
              type="number"
              min="0.001"
              max="0.01"
              step="any"
              bind:value={learningRate}
              class="w-24 px-2 py-1 border rounded text-sm"
              aria-label="Learning rate numerical input"
              disabled={running}
            />
          </div>
          <div class="flex justify-end">
            <button
              onclick={() => {
                learningRate = 0.005;
              }}
              disabled={running}
            >
              Reset to default
            </button>
          </div>
        </div>
      {/if}
    </div>
  </section>

  <div class="grid md:grid-cols-2 xl:grid-cols-3 gap-4 my-6">
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
        title="Test Loss & Accuracy"
        data={testMetrics}
        x="epoch"
        y={["loss", "acc"]}
      />
    </div>
    <div class="h-[220px] border border-gray-400 rounded">
      <div class="flex flex-col h-full">
        <p class="shrink-0 text-sm text-center my-1">Inference Demo</p>
        <div class="grow flex px-2 pb-2 min-h-0">
          <div
            class="relative aspect-square h-full border-4 border-gray-200 rounded-md"
          >
            <canvas
              width="280"
              height="280"
              class="w-full h-full"
              onpointerdown={drawStart}
              onpointermove={drawMove}
              onpointerleave={drawEnd}
              onpointerup={drawEnd}
              bind:this={canvas}
            ></canvas>

            {#if hasDrawn}
              <button
                class="absolute bottom-1 right-1"
                onclick={(event) => {
                  event.stopPropagation();
                  clearCanvas();
                }}>Clear</button
              >
            {:else}
              <p
                class="absolute top-18 left-6 -rotate-15 animate-bounce italic text-gray-400 pointer-events-none"
              >
                draw a digit here!
              </p>
            {/if}
          </div>
          <div class="grow ml-2">
            {#if probs.length > 0}
              <p class="text-xs font-bold">Probabilities:</p>
              {#each probs as prob, i}
                <div class="flex items-center text-xs tabular-nums">
                  <span class="w-4 text-right">{i}:</span>
                  <span
                    class="ml-1 bg-gray-400 h-3 rounded-sm"
                    style:width="{72 * prob}%"
                  ></span>
                  <span class="ml-2">
                    {(prob * 100).toFixed(1)}%
                  </span>
                </div>
              {/each}
            {/if}
          </div>
        </div>
      </div>
    </div>
  </div>

  <div
    class="font-mono text-sm rounded bg-gray-900 px-4 py-2 h-[600px] overflow-y-scroll mt-8"
  >
    {#each logs as log}
      <div class="text-white whitespace-pre-wrap">{log}</div>
    {/each}
  </div>
</main>

<style lang="postcss">
  @reference "$app.css";

  button {
    @apply border rounded px-2 hover:bg-gray-100 active:scale-95;
  }

  select {
    @apply border rounded px-1 text-sm;
  }
</style>
