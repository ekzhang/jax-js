<script lang="ts">
  import {
    init,
    jit,
    nn,
    numpy as np,
    random,
    setDevice,
    tree,
    valueAndGrad,
  } from "@jax-js/jax";
  import pThrottle from "p-throttle";
  import { onMount } from "svelte";

  import LineChart from "$lib/chart/LineChart.svelte";
  import { fetchMnist } from "$lib/dataset/mnist";

  let logs = $state<string[]>([]);

  let trainMetrics = $state<{ iteration: number; loss: number }[]>([]);
  let testMetrics = $state<{ epoch: number; loss: number; acc: number }[]>([]);

  function log(msg: string) {
    logs.push(msg);
    console.log(msg);
  }

  type Params = {
    w1: np.Array; // [784, 1024]
    b1: np.Array; // [1024]
    w2: np.Array; // [1024, 128]
    b2: np.Array; // [128]
    w3: np.Array; // [128, 10]
    b3: np.Array; // [10]
  };

  async function initializeParams(): Promise<Params> {
    const [k11, k12, k21, k22, k31, k32] = random.split(random.key(0), 6);
    const w1 = random.uniform(k11, [784, 1024], {
      minval: -1 / Math.sqrt(784),
      maxval: 1 / Math.sqrt(784),
    });
    const b1 = random.uniform(k12, [1024], {
      minval: -1 / Math.sqrt(784),
      maxval: 1 / Math.sqrt(784),
    });
    const w2 = random.uniform(k21, [1024, 128], {
      minval: -1 / Math.sqrt(1024),
      maxval: 1 / Math.sqrt(1024),
    });
    const b2 = random.uniform(k22, [128], {
      minval: -1 / Math.sqrt(1024),
      maxval: 1 / Math.sqrt(1024),
    });
    const w3 = random.uniform(k31, [128, 10], {
      minval: -1 / Math.sqrt(128),
      maxval: 1 / Math.sqrt(128),
    });
    const b3 = random.uniform(k32, [10], {
      minval: -1 / Math.sqrt(128),
      maxval: 1 / Math.sqrt(128),
    });
    const params = { w1, b1, w2, b2, w3, b3 };
    // Wait for all the arrays to be created on the device.
    await Promise.all(tree.leaves(params).map((ar) => ar.ref.wait()));
    return params;
  }

  function predict(params: Params, x: np.Array): np.Array {
    // Forward pass through the network
    const z1 = np.dot(x, params.w1).add(params.b1);
    const a1 = nn.relu(z1);
    const z2 = np.dot(a1, params.w2).add(params.b2);
    const a2 = nn.relu(z2);
    const z3 = np.dot(a2, params.w3).add(params.b3);
    return nn.logSoftmax(z3);
  }

  const predictJit = jit(predict);

  function loss(params: Params, x: np.Array, y: np.Array): np.Array {
    // Compute the negative log-likelihood loss
    const batchSize = y.shape[0];
    const logits = predictJit(params, x);
    return logits
      .mul(nn.oneHot(y, 10))
      .sum()
      .mul(-1 / batchSize);
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

  let latestParams: Params | null = null;
  let probs: number[] = $state([]);

  async function run() {
    logs = [];
    trainMetrics = [];
    testMetrics = [];

    let params = await initializeParams();

    if (latestParams !== null)
      tree.map((x: np.Array) => x.dispose(), latestParams);
    latestParams = tree.ref(params);

    const startTime = performance.now();
    const { X_train, y_train, X_test, y_test } = await loadData();
    const duration = performance.now() - startTime;
    log(`=> Data loaded in ${duration.toFixed(1)} ms`);

    const lr = 5e-2;

    try {
      const batchSize = 1000;
      const numBatches = Math.ceil(X_train.shape[0] / batchSize);
      for (let epoch = 0; epoch < 10; epoch++) {
        log(`=> Epoch ${epoch + 1}`);
        const randomIndices = [];
        for (let i = 0; i < X_train.shape[0]; i++) {
          randomIndices.push(i);
        }
        for (let i = randomIndices.length - 1; i > 0; i--) {
          const j = Math.floor(Math.random() * (i + 1));
          [randomIndices[i], randomIndices[j]] = [
            randomIndices[j],
            randomIndices[i],
          ];
        }

        for (let i = 0; i < numBatches; i++) {
          const indices = np.array(
            randomIndices.slice(i * batchSize, (i + 1) * batchSize),
            { dtype: np.int32 },
          );

          const startTime = performance.now();
          const X = X_train.ref.slice(indices.ref).reshape([-1, 784]);
          const y = y_train.ref.slice(indices);
          const [lossVal, lossGrad] = valueAndGrad(loss)(
            tree.ref(params),
            X,
            y,
          );
          params = tree.map(
            (a: np.Array, b: np.Array) => a.add(b.mul(-lr)),
            params,
            lossGrad,
          );
          for (const val of Object.values(params)) {
            await val.ref.wait();
          }
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

        if (latestParams !== null)
          tree.map((x: np.Array) => x.dispose(), latestParams);
        latestParams = tree.ref(params);

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
              .argmax(predictJit(tree.ref(params), X), 1)
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
      tree.map((x: np.Array) => x.dispose(), params);
    }
  }

  onMount(async () => {
    await init("webgpu");
    setDevice("webgpu");

    ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  });

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
    const logits = predictJit(params, np.array(ar).reshape([1, 784]));
    probs = (await np.exp(logits).slice(0).jsAsync()) as number[];
  });

  let canvas: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D;
  let hasDrawn = $state(false);
  let drawing = false;
  let lastPos = [0, 0];

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
    ctx.ellipse(x, y, 10, 10, 0, 0, Math.PI * 2);
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
    ctx.lineWidth = 20;
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

<main class="p-4">
  <h1 class="text-2xl mb-2">mnist + jax-js</h1>

  <p class="mb-2">
    Let's try and train a neural network to classify MNIST digits, in your
    browser with <code>jax-js</code>.
  </p>

  <p class="mb-2">Note: This is pretty scuffed right now. To do:</p>

  <ul class="list-disc pl-6 mb-4">
    <li>Implement Adam</li>
    <li>Convolutional layers</li>
    <li>Make JIT less inefficient</li>
  </ul>

  <button onclick={run}>Run</button>

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
              <button class="absolute bottom-1 right-1" onclick={clearCanvas}
                >Clear</button
              >
            {:else}
              <p
                class="absolute top-18 left-6 -rotate-15 animate-bounce italic text-gray-400 select-none"
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
                    style:width="{80 * prob}%"
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
    class="font-mono text-sm bg-gray-900 px-4 py-2 h-[600px] overflow-y-scroll mt-8"
  >
    {#each logs as log}
      <div class="text-white whitespace-pre-wrap">{log}</div>
    {/each}
  </div>
</main>

<style lang="postcss">
  @reference "$app.css";

  button {
    @apply border px-2 hover:bg-gray-100 active:scale-95;
  }
</style>
