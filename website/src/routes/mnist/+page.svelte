<script lang="ts">
  import {
    init,
    jit,
    nn,
    numpy as np,
    setDevice,
    tree,
    valueAndGrad,
  } from "@jax-js/jax";
  import { onMount } from "svelte";

  import { fetchMnist } from "$lib/dataset/mnist";

  let logs = $state<string[]>([]);

  function log(msg: string) {
    logs.push(msg);
    console.log(msg);
  }

  type Params = {
    w1: np.Array; // [784, 128]
    b1: np.Array; // [128]
    w2: np.Array; // [128, 128]
    b2: np.Array; // [128]
    w3: np.Array; // [128, 10]
    b3: np.Array; // [10]
  };

  async function initializeParams(): Promise<Params> {
    // We don't have random() yet, hopefully this is good enough? :/
    const w1 = np
      .sin(np.linspace(-1e6, 1e6, 784 * 128).reshape([784, 128]))
      .mul(0.1);
    const b1 = np.linspace(-1, 1, 128);
    const w2 = np
      .sin(np.linspace(-1e6, 1e6, 128 * 128).reshape([128, 128]))
      .mul(0.1);
    const b2 = np.linspace(-1, 1, 128);
    const w3 = np
      .sin(np.linspace(-1e6, 1e6, 128 * 10).reshape([128, 10]))
      .mul(0.1);
    const b3 = np.zeros([10]);
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

  async function run() {
    logs = [];

    let params = await initializeParams();

    const startTime = performance.now();
    const { X_train, y_train, X_test, y_test } = await loadData();
    const duration = performance.now() - startTime;
    log(`=> Data loaded in ${duration.toFixed(1)} ms`);

    const lr = 5e-2;

    try {
      const batchSize = 2000;
      for (let epoch = 0; epoch < 10; epoch++) {
        log(`=> Epoch ${epoch + 1}`);
        for (let i = 0; i + batchSize <= X_train.shape[0]; i += batchSize) {
          const startTime = performance.now();
          const X = X_train.ref.slice([i, i + batchSize]).reshape([-1, 784]);
          const y = y_train.ref.slice([i, i + batchSize]);
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
          log(
            `batch ${i} completed in ${duration.toFixed(1)} ms, loss: ${((await lossVal.jsAsync()) as number).toFixed(4)}`,
          );
        }

        log(`=> Evaluating on test set...`);
        const testSize = X_test.shape[0];
        const testLoss: number[] = [];
        const testAcc: number[] = [];
        for (let i = 0; i + batchSize <= testSize; i += batchSize) {
          const X = X_test.ref.slice([i, i + batchSize]).reshape([-1, 784]);
          const y = y_test.ref.slice([i, i + batchSize]);
          testLoss.push(
            (await loss(tree.ref(params), X.ref, y.ref).jsAsync()) as number,
          );
          // console.log(
          //   await np.argmax(predictJit(tree.ref(params), X.ref), 1).jsAsync(),
          // );
          testAcc.push(
            (await np
              .where(
                np.argmax(predictJit(tree.ref(params), X), 1).equal(y),
                1,
                0,
              )
              .sum()
              .jsAsync()) as number,
          );
        }
        const testLossAvg = testLoss.reduce((a, b) => a + b) / testLoss.length;
        const testAccAvg = testAcc.reduce((a, b) => a + b) / testSize;
        log(`=> Test acc: ${testAccAvg.toFixed(4)}`);
        log(`=> Test loss: ${testLossAvg.toFixed(4)}`);
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
  });
</script>

<main class="p-4">
  <h1 class="text-2xl mb-2">mnist + jax-js</h1>

  <p class="mb-4">
    Let's try to train a neural network to classify MNIST digits, in your
    browser.
  </p>

  <button onclick={run}>Run</button>

  <div class="font-mono text-sm bg-black h-[600px] overflow-y-scroll mt-8">
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
