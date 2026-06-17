<script lang="ts">
  import { resolve } from "$app/paths";

  import {
    blockUntilReady,
    defaultDevice,
    init,
    jit,
    lax,
    nn,
    numpy as np,
    profiler,
    tree,
  } from "@jax-js/jax";
  import type { Device } from "@jax-js/jax";
  import {
    ChevronDownIcon,
    GithubIcon,
    PauseIcon,
    PlayIcon,
    RotateCcwIcon,
    SkipForwardIcon,
  } from "@lucide/svelte";
  import { onDestroy, onMount } from "svelte";

  const CELLS2PIXELS_CDN_BASE =
    "https://cdn.jsdelivr.net/gh/Cells2Pixels/Cells2Pixels.github.io@8e45a3c39b25cae4cd1555bae889d7b1c914e328";
  const MODEL_URL = `${CELLS2PIXELS_CDN_BASE}/2d_growing_demo/models.json`;
  const TARGET_BASE_URL = `${CELLS2PIXELS_CDN_BASE}/2d_growing_demo/target_images`;

  const SOURCE_URL =
    "https://github.com/ekzhang/jax-js/blob/main/website/src/routes/nca-growing/+page.svelte";
  const ORIGINAL_DEMO_URL = "https://cells2pixels.github.io/#growing";
  const PAPER_URL = "https://arxiv.org/abs/2506.22899";

  const GRID_SIZE = 128;
  const CHANNELS = 32;
  const DEFAULT_MODEL = "Chameleon";
  const VIEW_RADIUS = 0.7;
  const DAMAGE_RADIUS = 4;

  type NcaBackend = Extract<Device, "webgpu" | "wasm">;

  const BACKEND_LABEL: Record<NcaBackend, string> = {
    webgpu: "WebGPU",
    wasm: "WebAssembly",
  };

  type RawTensor = {
    shape: number[];
    data64: string;
  };

  type RawModel = Record<string, RawTensor>;
  type RawModels = Record<string, RawModel>;

  type NcaWeights = {
    w1: np.Array;
    b1: np.Array;
    w2t: np.Array;
    perceptionKernel: np.Array;
  };

  type LppnWeights = {
    weights: np.Array[];
    biases: np.Array[];
  };

  type LoadedModel = {
    nca: NcaWeights;
    lppn: LppnWeights;
  };

  type DecodeGrid = {
    coords: np.Array;
    i00: np.Array;
    i01: np.Array;
    i10: np.Array;
    i11: np.Array;
    w00: np.Array;
    w01: np.Array;
    w10: np.Array;
    w11: np.Array;
  };

  type GridPosition = { x: number; y: number };

  let canvas: HTMLCanvasElement;
  let running = $state(false);
  let loading = $state(true);
  let stepping = $state(false);
  let frameRequest = 0;
  let stepCount = $state(0);
  let lastFrameMs = $state(0);
  let selectedBackend = $state<NcaBackend>("webgpu");
  let backendOptions = $state<NcaBackend[]>([]);
  let selectedModel = $state(DEFAULT_MODEL);
  let modelPickerOpen = $state(false);
  let modelNames = $state<string[]>([]);
  let renderScale = $state(3);
  let damaging = false;

  let rawModels: RawModels | null = null;
  let loadedModel: LoadedModel | null = null;
  let ncaState: np.Array | null = null;
  let decodeGrid: DecodeGrid | null = null;
  let decodeGridScale = 0;
  let pendingDamagePositions: GridPosition[] = [];

  function targetUrl(name: string): string {
    return `${TARGET_BASE_URL}/${name.toLowerCase()}.png`;
  }

  function stopAnimation() {
    running = false;
    if (frameRequest) {
      cancelAnimationFrame(frameRequest);
      frameRequest = 0;
    }
  }

  function startAnimation() {
    if (running) return;
    running = true;
    animate();
  }

  function disposeRuntimeState() {
    ncaState?.dispose();
    ncaState = null;
    tree.dispose(decodeGrid);
    decodeGrid = null;
    decodeGridScale = 0;
    tree.dispose(loadedModel);
    loadedModel = null;
  }

  function setBackend(backend: NcaBackend) {
    defaultDevice(backend);
    selectedBackend = backend;
  }

  function maxAlive(x: np.Array): np.Array {
    const alpha = x.slice([], [], 3);
    const alive = lax.reduceWindow(
      np.pad(alpha, { 0: [1, 1], 1: [1, 1] }),
      np.max,
      [3, 3],
      [1, 1],
    );
    return alive.greater(0.1).astype(np.float32);
  }

  const ncaStep = jit(function ncaStep(
    x: np.Array,
    w1: np.Array,
    b1: np.Array,
    w2t: np.Array,
    perceptionKernel: np.Array,
    updateMask: np.Array,
  ): np.Array {
    const [h, w] = x.shape;
    const alive = maxAlive(x.ref);
    const edgeFeatures = lax
      .convGeneralDilated(
        x.ref.transpose([2, 0, 1]).reshape([1, CHANNELS, h, w]),
        perceptionKernel,
        [1, 1],
        "SAME",
        { featureGroupCount: CHANNELS },
      )
      .reshape([CHANNELS, 3, h, w])
      .transpose([1, 0, 2, 3])
      .reshape([3 * CHANNELS, h, w])
      .transpose([1, 2, 0])
      .reshape([h * w, 3 * CHANNELS]);
    const perception = np.concatenate(
      [x.ref.reshape([h * w, CHANNELS]), edgeFeatures],
      1,
    );
    const hidden = nn.relu(np.dot(perception, w1).add(b1));
    const delta = np.dot(hidden, w2t).reshape([h, w, CHANNELS]).mul(updateMask);
    return x.add(delta).mul(alive.reshape([h, w, 1]));
  });

  function tensorData(tensor: RawTensor): Float32Array<ArrayBuffer> {
    const bytes = Uint8Array.from(atob(tensor.data64), (c) => c.charCodeAt(0));
    return new Float32Array(bytes.buffer) as Float32Array<ArrayBuffer>;
  }

  function tensorArray(tensor: RawTensor): np.Array {
    return np.array(tensorData(tensor)).reshape(tensor.shape);
  }

  function createPerceptionKernel(): np.Array {
    const filters = [
      [-1, 0, 1, -2, 0, 2, -1, 0, 1],
      [-1, -2, -1, 0, 0, 0, 1, 2, 1],
      [1, 2, 1, 2, -12, 2, 1, 2, 1],
    ];
    const data = new Float32Array(CHANNELS * filters.length * 3 * 3);
    for (let channel = 0; channel < CHANNELS; channel++) {
      for (let filter = 0; filter < filters.length; filter++) {
        const offset = (channel * filters.length + filter) * 9;
        data.set(filters[filter], offset);
      }
    }
    return np.array(data).reshape([CHANNELS * filters.length, 1, 3, 3]);
  }

  function loadModelFromRaw(raw: RawModel): LoadedModel {
    const w1 = tensorArray(raw["nca.w1.weight"]).transpose();
    const b1 = tensorArray(raw["nca.w1.bias"]);
    const w2t = tensorArray(raw["nca.w2.weight.T"]);
    const perceptionKernel = createPerceptionKernel();
    const weights: np.Array[] = [];
    const biases: np.Array[] = [];
    for (let i = 0; i < 4; i++) {
      const prefix = `lppn.net.${i}${i === 3 ? "" : ".linear"}`;
      weights.push(tensorArray(raw[`${prefix}.weight`]).transpose());
      biases.push(tensorArray(raw[`${prefix}.bias`]));
    }
    return {
      nca: { w1, b1, w2t, perceptionKernel },
      lppn: { weights, biases },
    };
  }

  function createSeed(): np.Array {
    const data = new Float32Array(GRID_SIZE * GRID_SIZE * CHANNELS);
    const x = Math.floor(GRID_SIZE / 2);
    const y = Math.floor(GRID_SIZE / 2);
    const offset = (y * GRID_SIZE + x) * CHANNELS;
    data[offset + 3] = 1;
    for (let c = 4; c < CHANNELS; c++) data[offset + c] = 1;
    return np.array(data).reshape([GRID_SIZE, GRID_SIZE, CHANNELS]);
  }

  function createUpdateMask(): np.Array {
    const data = new Float32Array(GRID_SIZE * GRID_SIZE);
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.random() < 0.5 ? 1 : 0;
    }
    return np.array(data).reshape([GRID_SIZE, GRID_SIZE, 1]);
  }

  function createDamageMask(positions: GridPosition[]): np.Array {
    const data = new Float32Array(GRID_SIZE * GRID_SIZE);
    data.fill(1);
    const radiusSq = DAMAGE_RADIUS * DAMAGE_RADIUS;
    for (const { x: centerX, y: centerY } of positions) {
      const minX = Math.max(0, Math.floor(centerX - DAMAGE_RADIUS));
      const maxX = Math.min(GRID_SIZE - 1, Math.ceil(centerX + DAMAGE_RADIUS));
      const minY = Math.max(0, Math.floor(centerY - DAMAGE_RADIUS));
      const maxY = Math.min(GRID_SIZE - 1, Math.ceil(centerY + DAMAGE_RADIUS));

      for (let y = minY; y <= maxY; y++) {
        const dy = y - centerY;
        for (let x = minX; x <= maxX; x++) {
          const dx = x - centerX;
          if (dx * dx + dy * dy <= radiusSq) data[y * GRID_SIZE + x] = 0;
        }
      }
    }
    return np.array(data).reshape([GRID_SIZE, GRID_SIZE, 1]);
  }

  function queueDamagePosition(pos: GridPosition) {
    pendingDamagePositions.push(pos);
    if (pendingDamagePositions.length > 64) {
      pendingDamagePositions = pendingDamagePositions.slice(-64);
    }
  }

  function consumeDamageMask(): np.Array | null {
    if (pendingDamagePositions.length === 0) return null;
    const positions = pendingDamagePositions;
    pendingDamagePositions = [];
    return createDamageMask(positions);
  }

  function wrapGridIndex(index: number): number {
    return ((index % GRID_SIZE) + GRID_SIZE) % GRID_SIZE;
  }

  function createDecodeGrid(scale: number): DecodeGrid {
    const h = GRID_SIZE * scale;
    const w = GRID_SIZE * scale;
    const size = h * w;
    const coords = new Float32Array(size * 4);
    const i00 = new Int32Array(size);
    const i01 = new Int32Array(size);
    const i10 = new Int32Array(size);
    const i11 = new Int32Array(size);
    const w00 = new Float32Array(size);
    const w01 = new Float32Array(size);
    const w10 = new Float32Array(size);
    const w11 = new Float32Array(size);

    for (let y = 0; y < h; y++) {
      const fetchY = 0.5 + ((y + 0.5) / h - 0.5) * VIEW_RADIUS;
      const gridY = fetchY * GRID_SIZE;
      const sourceY = gridY - 0.5;
      const y0Raw = Math.floor(sourceY);
      const y1Raw = y0Raw + 1;
      const yMix = sourceY - y0Raw;
      const y0 = wrapGridIndex(y0Raw);
      const y1 = wrapGridIndex(y1Raw);
      const py = (gridY - Math.floor(gridY) - 0.5) * 2;

      for (let x = 0; x < w; x++) {
        const fetchX = 0.5 + ((x + 0.5) / w - 0.5) * VIEW_RADIUS;
        const gridX = fetchX * GRID_SIZE;
        const sourceX = gridX - 0.5;
        const x0Raw = Math.floor(sourceX);
        const x1Raw = x0Raw + 1;
        const xMix = sourceX - x0Raw;
        const x0 = wrapGridIndex(x0Raw);
        const x1 = wrapGridIndex(x1Raw);
        const px = (gridX - Math.floor(gridX) - 0.5) * 2;
        const index = y * w + x;
        const coordOffset = index * 4;

        coords[coordOffset] = Math.sin(Math.PI * py);
        coords[coordOffset + 1] = Math.sin(Math.PI * px);
        coords[coordOffset + 2] = Math.cos(Math.PI * py);
        coords[coordOffset + 3] = Math.cos(Math.PI * px);

        i00[index] = y0 * GRID_SIZE + x0;
        i01[index] = y0 * GRID_SIZE + x1;
        i10[index] = y1 * GRID_SIZE + x0;
        i11[index] = y1 * GRID_SIZE + x1;
        w00[index] = (1 - yMix) * (1 - xMix);
        w01[index] = (1 - yMix) * xMix;
        w10[index] = yMix * (1 - xMix);
        w11[index] = yMix * xMix;
      }
    }

    return {
      coords: np.array(coords).reshape([h, w, 4]),
      i00: np.array(i00, { dtype: np.int32 }),
      i01: np.array(i01, { dtype: np.int32 }),
      i10: np.array(i10, { dtype: np.int32 }),
      i11: np.array(i11, { dtype: np.int32 }),
      w00: np.array(w00).reshape([size, 1]),
      w01: np.array(w01).reshape([size, 1]),
      w10: np.array(w10).reshape([size, 1]),
      w11: np.array(w11).reshape([size, 1]),
    };
  }

  function getDecodeGrid(): DecodeGrid {
    if (!decodeGrid || decodeGridScale !== renderScale) {
      tree.dispose(decodeGrid);
      decodeGrid = createDecodeGrid(renderScale);
      decodeGridScale = renderScale;
    }
    return decodeGrid;
  }

  async function waitForIdle(): Promise<void> {
    while (stepping) {
      await new Promise<void>((resolve) =>
        requestAnimationFrame(() => resolve()),
      );
    }
  }

  const bilinearUpsampleState = jit(function bilinearUpsampleState(
    x: np.Array,
    grid: DecodeGrid,
  ): np.Array {
    grid.coords.dispose(); // Unused
    const flat = x.reshape([GRID_SIZE * GRID_SIZE, CHANNELS]);
    const v00 = np.take(flat.ref, grid.i00.ref, 0);
    const v01 = np.take(flat.ref, grid.i01.ref, 0);
    const v10 = np.take(flat.ref, grid.i10.ref, 0);
    const v11 = np.take(flat, grid.i11.ref, 0);
    const [h, w] = grid.coords.shape;
    return v00
      .mul(grid.w00.ref)
      .add(v01.mul(grid.w01.ref))
      .add(v10.mul(grid.w10.ref))
      .add(v11.mul(grid.w11.ref))
      .reshape([h, w, CHANNELS]);
  });

  const decodeImage = jit(function decodeImage(
    x: np.Array,
    lppn: LppnWeights,
    grid: DecodeGrid,
  ): np.Array {
    const gridCoords = grid.coords.ref;
    const up = bilinearUpsampleState(x, grid);
    const [h, w] = up.shape;
    const alpha = up.ref.slice([], [], 3);
    const mask = lax
      .reduceWindow(
        np.pad(alpha, { 0: [1, 1], 1: [1, 1] }),
        np.max,
        [3, 3],
        [1, 1],
      )
      .greater(0.1)
      .astype(np.float32)
      .reshape([h, w, 1]);
    let y = np.concatenate(
      [gridCoords.reshape([h * w, 4]), up.reshape([h * w, CHANNELS])],
      1,
    );

    for (let i = 0; i < 4; i++) {
      y = np.dot(y, lppn.weights[i]).add(lppn.biases[i]);
      if (i < 3) y = np.sin(y.mul(10));
    }
    return y.reshape([h, w, 4]).mul(mask);
  });

  const packCanvasPixels = jit(function packCanvasPixels(
    rgba: np.Array,
  ): np.Array {
    const [h, w] = rgba.shape;
    const alpha = np.clip(rgba.ref.slice([], [], 3), 0, 1);
    const rgb = np
      .clip(
        rgba
          .slice([], [], [0, 3])
          .add(1)
          .sub(alpha.reshape([h, w, 1])),
        0,
        1,
      )
      .mul(255)
      .add(0.5)
      .astype(np.uint32);
    const r = rgb.ref.slice([], [], 0);
    const g = rgb.ref.slice([], [], 1);
    const b = rgb.slice([], [], 2);
    return r.add(g.mul(256)).add(b.mul(65536)).add(0xff000000);
  });

  const damageState = jit(function damageState(
    x: np.Array,
    mask: np.Array,
  ): np.Array {
    return x.mul(mask);
  });

  function resetState() {
    ncaState?.dispose();
    ncaState = createSeed();
    stepCount = 0;
    lastFrameMs = 0;
  }

  async function selectModel(name: string) {
    if (!rawModels) return;
    modelPickerOpen = false;
    if (name === selectedModel) return;
    const resume = running;
    stopAnimation();
    await waitForIdle();
    loading = true;
    selectedModel = name;
    disposeRuntimeState();
    loadedModel = loadModelFromRaw(rawModels[name]);
    resetState();
    await render();
    loading = false;
    if (resume) startAnimation();
  }

  async function selectBackend(backend: NcaBackend) {
    if (backend === defaultDevice() || !rawModels) return;
    const resume = running;
    stopAnimation();
    await waitForIdle();
    loading = true;

    try {
      disposeRuntimeState();
      setBackend(backend);
      loadedModel = loadModelFromRaw(rawModels[selectedModel]);
      resetState();
      if (ncaState) await blockUntilReady(ncaState);
      await render();
      if (resume) startAnimation();
    } catch (error) {
      window.alert(error);
    } finally {
      loading = false;
    }
  }

  async function selectRenderScale(scale: number) {
    if (scale === renderScale) return;
    const resume = running;
    stopAnimation();
    await waitForIdle();
    renderScale = scale;
    await render();
    if (resume) startAnimation();
  }

  async function stepOnce(renderAfter = true) {
    profiler.startTrace();
    if (!loadedModel || !ncaState || stepping) return;
    stepping = true;
    const start = performance.now();
    try {
      const damageMask = consumeDamageMask();
      if (damageMask) ncaState = damageState(ncaState, damageMask);
      ncaState = ncaStep(
        ncaState,
        loadedModel.nca.w1.ref,
        loadedModel.nca.b1.ref,
        loadedModel.nca.w2t.ref,
        loadedModel.nca.perceptionKernel.ref,
        createUpdateMask(),
      );
      stepCount += 1;
      if (renderAfter) await render();
      lastFrameMs = performance.now() - start;
    } finally {
      stepping = false;
    }
  }

  function pointerGridPosition(event: PointerEvent): GridPosition | null {
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const u = (event.clientX - rect.left) / rect.width;
    const v = (event.clientY - rect.top) / rect.height;
    if (u < 0 || u > 1 || v < 0 || v > 1) return null;

    return {
      x: GRID_SIZE * (0.5 + (u - 0.5) * VIEW_RADIUS),
      y: GRID_SIZE * (0.5 + (v - 0.5) * VIEW_RADIUS),
    };
  }

  async function damageAtPointer(event: PointerEvent) {
    const pos = pointerGridPosition(event);
    if (!pos || !loadedModel || !ncaState) return;
    event.preventDefault();
    queueDamagePosition(pos);

    if (running || stepping) return;

    stepping = true;
    try {
      const damageMask = consumeDamageMask();
      if (damageMask) ncaState = damageState(ncaState, damageMask);
      await render();
    } finally {
      stepping = false;
    }
  }

  function startDamage(event: PointerEvent) {
    if (event.button !== 0) return;
    damaging = true;
    (event.currentTarget as HTMLCanvasElement).setPointerCapture(
      event.pointerId,
    );
    void damageAtPointer(event);
  }

  function dragDamage(event: PointerEvent) {
    if (!damaging) return;
    void damageAtPointer(event);
  }

  function stopDamage(event: PointerEvent) {
    damaging = false;
    const target = event.currentTarget as HTMLCanvasElement;
    if (target.hasPointerCapture(event.pointerId)) {
      target.releasePointerCapture(event.pointerId);
    }
  }

  async function render() {
    if (!canvas || !loadedModel || !ncaState) return;
    const scale = renderScale;
    const width = GRID_SIZE * scale;
    const height = GRID_SIZE * scale;
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;

    const rgba = decodeImage(
      ncaState.ref,
      tree.ref(loadedModel.lppn),
      tree.ref(getDecodeGrid()),
    );
    const packed = packCanvasPixels(rgba);
    const values = (await packed.data()) as Uint32Array;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const pixels = new Uint8ClampedArray(
      values.buffer as ArrayBuffer,
      values.byteOffset,
      values.byteLength,
    );
    const image = new ImageData(pixels, width, height, { colorSpace: "srgb" });
    ctx.putImageData(image, 0, 0);
  }

  function animate() {
    if (!running) return;
    frameRequest = requestAnimationFrame(animate);
    void stepOnce(true);
  }

  async function toggleRunning() {
    if (running) {
      stopAnimation();
    } else {
      startAnimation();
    }
  }

  async function hardReset() {
    stopAnimation();
    resetState();
    await render();
  }

  onMount(async () => {
    try {
      const available = await init("webgpu", "wasm");
      backendOptions = (["webgpu", "wasm"] as const).filter((backend) =>
        available.includes(backend),
      );
      if (backendOptions.length === 0) {
        throw new Error("No WebGPU or WebAssembly backend is available.");
      }
      setBackend(
        backendOptions.includes("webgpu") ? "webgpu" : backendOptions[0],
      );

      const response = await fetch(MODEL_URL);
      rawModels = (await response.json()) as RawModels;
      modelNames = Object.keys(rawModels);
      loadedModel = loadModelFromRaw(
        rawModels[selectedModel] ?? rawModels[modelNames[0]],
      );
      selectedModel = rawModels[selectedModel] ? selectedModel : modelNames[0];
      resetState();

      if (ncaState) await blockUntilReady(ncaState);
      await render();
      loading = false;
      startAnimation();
    } catch (error) {
      window.alert(error);
      loading = false;
    }
  });

  onDestroy(() => {
    stopAnimation();
    disposeRuntimeState();
    ncaStep.dispose();
    bilinearUpsampleState.dispose();
    packCanvasPixels.dispose();
    damageState.dispose();
  });
</script>

<svelte:head>
  <title>Neural Cellular Automata - jax-js</title>
</svelte:head>

<main class="min-h-screen bg-[#f8f8f4] text-[#202019] font-tiktok">
  <section class="mx-auto max-w-screen-xl px-4 py-4 sm:px-6">
    <div class="mb-4 flex flex-wrap items-center justify-between gap-3">
      <a href={resolve("/")} class="text-sm text-gray-600 hover:text-primary">
        jax-js
      </a>
      <a
        href={SOURCE_URL}
        target="_blank"
        rel="noreferrer"
        class="flex items-center gap-1.5 text-sm text-gray-600 hover:text-primary"
      >
        <GithubIcon size={16} />
        <span>Source</span>
      </a>
    </div>

    <div class="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)_240px]">
      <aside class="space-y-3">
        <div class="rounded-lg border border-black/10 bg-white p-3">
          <label class="mb-1 block text-sm text-gray-600" for="backend">
            Backend
          </label>
          <select
            id="backend"
            class="w-full rounded-md border border-black/15 bg-white px-2 py-2"
            value={selectedBackend}
            onchange={(e) => selectBackend(e.currentTarget.value as NcaBackend)}
          >
            {#each backendOptions as backend}
              <option value={backend}>{BACKEND_LABEL[backend]}</option>
            {/each}
          </select>
        </div>

        <div class="rounded-lg border border-black/10 bg-white p-3">
          <div
            class="relative"
            onfocusout={(event) => {
              const nextTarget = event.relatedTarget;
              if (
                nextTarget instanceof Node &&
                event.currentTarget.contains(nextTarget)
              ) {
                return;
              }
              modelPickerOpen = false;
            }}
          >
            <div id="target-label" class="mb-1 block text-sm text-gray-600">
              Target
            </div>
            <button
              type="button"
              class="flex w-full items-center gap-2 rounded-md border border-black/15 bg-white px-2 py-2 text-left hover:bg-black/[0.03] disabled:opacity-50"
              aria-haspopup="listbox"
              aria-expanded={modelPickerOpen}
              aria-labelledby="target-label"
              onclick={() => (modelPickerOpen = !modelPickerOpen)}
            >
              <img
                class="h-9 w-9 shrink-0 rounded border border-black/10 object-cover"
                src={targetUrl(selectedModel)}
                alt=""
              />
              <span class="min-w-0 flex-1 truncate">{selectedModel}</span>
              <ChevronDownIcon size={16} class="shrink-0 opacity-60" />
            </button>

            {#if modelPickerOpen}
              <div
                class="absolute z-20 mt-2 max-h-[52vh] w-full overflow-y-auto rounded-md border border-black/15 bg-white p-1 shadow-lg"
                role="listbox"
                aria-labelledby="target-label"
              >
                {#each modelNames as name}
                  <button
                    type="button"
                    class:target-option-selected={name === selectedModel}
                    class="target-option"
                    role="option"
                    aria-selected={name === selectedModel}
                    onclick={() => selectModel(name)}
                  >
                    <img
                      class="h-10 w-10 shrink-0 rounded border border-black/10 object-cover"
                      src={targetUrl(name)}
                      alt=""
                      loading="lazy"
                      decoding="async"
                    />
                    <span class="min-w-0 truncate">{name}</span>
                  </button>
                {/each}
              </div>
            {/if}
          </div>
        </div>

        <div class="rounded-lg border border-black/10 bg-white p-3">
          <label class="mb-1 block text-sm text-gray-600" for="scale">
            LPPN scale
          </label>
          <select
            id="scale"
            class="w-full rounded-md border border-black/15 bg-white px-2 py-2"
            value={renderScale}
            onchange={(e) => selectRenderScale(Number(e.currentTarget.value))}
          >
            <option value={1}>x1</option>
            <option value={2}>x2</option>
            <option value={3}>x3</option>
            <option value={4}>x4</option>
            <option value={5}>x5</option>
            <option value={6}>x6</option>
          </select>
        </div>

        <div class="grid grid-cols-3 gap-2">
          <button
            class="icon-button"
            title={running ? "Pause" : "Play"}
            onclick={toggleRunning}
            disabled={loading}
          >
            {#if running}
              <PauseIcon size={18} />
            {:else}
              <PlayIcon size={18} />
            {/if}
          </button>
          <button
            class="icon-button"
            title="Step"
            onclick={() => stepOnce(true)}
            disabled={loading || running || stepping}
          >
            <SkipForwardIcon size={18} />
          </button>
          <button
            class="icon-button"
            title="Reset"
            onclick={hardReset}
            disabled={loading || stepping}
          >
            <RotateCcwIcon size={18} />
          </button>
        </div>
      </aside>

      <section
        class="overflow-hidden rounded-lg border border-black/10 bg-white"
      >
        <div class="flex min-h-[420px] items-center justify-center p-3">
          <canvas
            bind:this={canvas}
            class="image-rendering-auto aspect-square w-full max-w-[min(78vh,760px)] cursor-crosshair touch-none rounded bg-white shadow-sm"
            aria-label="Neural cellular automata canvas"
            onpointerdown={startDamage}
            onpointermove={dragDamage}
            onpointerup={stopDamage}
            onpointercancel={stopDamage}
          ></canvas>
        </div>
      </section>

      <aside class="space-y-3">
        <div class="rounded-lg border border-black/10 bg-white p-3">
          <div class="text-sm text-gray-600">Steps</div>
          <div class="text-2xl tabular-nums">{stepCount}</div>
        </div>
        <div class="rounded-lg border border-black/10 bg-white p-3">
          <div class="text-sm text-gray-600">Last frame</div>
          <div class="text-2xl tabular-nums">{lastFrameMs.toFixed(1)} ms</div>
        </div>
        {#if selectedModel}
          <div class="rounded-lg border border-black/10 bg-white p-3">
            <div class="text-sm text-gray-600">Target</div>
            <img
              class="mt-2 aspect-square w-full rounded border border-black/10 object-cover"
              src={targetUrl(selectedModel)}
              alt={`${selectedModel} target`}
            />
          </div>
        {/if}
      </aside>
    </div>

    <div class="mt-4 text-center text-sm leading-6 text-gray-600">
      Port of the
      <a
        href={ORIGINAL_DEMO_URL}
        target="_blank"
        rel="noreferrer"
        class="text-primary hover:underline"
      >
        Cells2Pixels growing demo
      </a>
      to jax-js, based on
      <a
        href={PAPER_URL}
        target="_blank"
        rel="noreferrer"
        class="text-primary hover:underline"
      >
        Neural Cellular Automata: From Cells to Pixels
      </a>
      (Pajouheshgar et al., SIGGRAPH '26).
    </div>
  </section>
</main>

<style lang="postcss">
  @reference "$app.css";

  .icon-button {
    @apply flex h-10 items-center justify-center rounded-md border border-black/15 bg-white hover:bg-black/[0.03] disabled:opacity-50;
  }

  .target-option {
    @apply flex w-full items-center gap-2 rounded border border-transparent px-2 py-1.5 text-left text-sm hover:bg-black/[0.03];
  }

  .target-option-selected {
    @apply border-primary bg-primary/10;
  }
</style>
