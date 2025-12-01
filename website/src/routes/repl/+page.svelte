<script lang="ts">
  import { building } from "$app/environment";
  import { afterNavigate, goto } from "$app/navigation";
  import { base } from "$app/paths";
  import { page } from "$app/state";

  import type { Device, numpy as np } from "@jax-js/jax";
  import { SplitPane } from "@rich_harris/svelte-split-pane";
  import type { Plugin } from "@rollup/browser";
  import { gunzipSync, gzipSync } from "fflate";
  import { Base64 } from "js-base64";
  import {
    AlertTriangleIcon,
    ArrowRightIcon,
    ChevronRightIcon,
    ImageIcon,
    InfoIcon,
    LoaderIcon,
    PaletteIcon,
    PlayIcon,
    ShareIcon,
    XIcon,
  } from "lucide-svelte";

  import ReplEditor from "$lib/repl/ReplEditor.svelte";

  const src: Record<string, string> = import.meta.glob("./*.ts", {
    eager: true,
    query: "?raw",
    import: "default",
  });

  const codeSamples: {
    title: string;
    id: string;
  }[] = [
    { title: "Arrays", id: "01-arrays" },
    { title: "Tracing Jaxprs", id: "02-tracing" },
    { title: "Logistic regression", id: "03-logistic-regression" },
    { title: "Mandelbrot set", id: "04-mandelbrot" },
  ];

  interface URLSelection {
    sample?: string;
    content?: string;
  }

  function getSelectionFromUrl(url: URL): URLSelection {
    const selection: URLSelection = {};

    if (!building) {
      const sample = url.searchParams.get("sample");
      if (sample && codeSamples.some((x) => x.id === sample)) {
        selection.sample = sample;
      }

      const contentZipB64 = url.searchParams.get("content");
      if (contentZipB64) {
        const contentZip = Base64.toUint8Array(contentZipB64); // Supports URL-safe base64
        selection.content = new TextDecoder().decode(gunzipSync(contentZip));
      }
    }

    return selection;
  }

  let selectionOnLoad = getSelectionFromUrl(page.url);

  let sample = $state(selectionOnLoad.sample);
  let device: Device = $state("webgpu");
  let replEditor: ReplEditor;

  afterNavigate(({ type }) => {
    if (type === "enter") return; // Already handled on load
    let selection = getSelectionFromUrl(page.url);
    if (selection.sample !== undefined) {
      sample = selection.sample;
      replEditor.setText(src[`./${sample}.ts`]);
    }
    if (selection.content !== undefined) {
      replEditor.setText(selection.content);
    }
  });

  function chooseSample(id: string) {
    replEditor.setText(src[`./${id}.ts`]);
    sample = id;
    goto(page.url.pathname + `?sample=${sample}`);
  }

  async function handleFormat() {
    const { formatWithCursor } = await import("prettier");
    const prettierParserTypescript = await import("prettier/parser-typescript");
    const prettierPluginEstree = await import("prettier/plugins/estree");

    const code = replEditor.getText();
    try {
      const { formatted, cursorOffset } = await formatWithCursor(code, {
        parser: "typescript",
        plugins: [prettierParserTypescript, prettierPluginEstree as any],
        cursorOffset: replEditor.getCursorOffset(),
      });
      replEditor.setText(formatted);
      replEditor.setCursorOffset(cursorOffset);
    } catch (e: any) {
      mockConsole.error(e);
    }
  }

  async function handleShare() {
    const code = replEditor.getText();

    // Encode the code as gzipped base64
    const encoded = new TextEncoder().encode(code);
    const compressed = gzipSync(encoded, { mtime: 0 });
    const base64Content = Base64.fromUint8Array(compressed, true); // URL-safe base64

    const url = new URL(page.url.origin + page.url.pathname);
    url.searchParams.set("content", base64Content);

    try {
      goto(url, { replaceState: true });
      await navigator.clipboard.writeText(url.toString());
      mockConsole.info("Link copied to clipboard!");
    } catch (e: any) {
      mockConsole.error("Failed to copy link:", e);
    }
  }

  async function handleRun() {
    if (running) return;
    running = true;

    const [jax, optax, loaders] = await Promise.all([
      import("@jax-js/jax"),
      import("@jax-js/optax"),
      import("@jax-js/loaders"),
    ]);
    const ts = await import("typescript");
    const { rollup } = await import("@rollup/browser");

    // Builtins for the REPL environment.
    const np = jax.numpy;
    const displayImage = async (ar: np.Array) => {
      if (ar.ndim !== 2 && ar.ndim !== 3) {
        throw new Error(
          "displayImage() only supports 2D (H, W) or 3D (H, W, C) array",
        );
      }
      await ar.blockUntilReady();

      if (ar.ndim === 2) {
        // If 2D, convert to (H, W, 1)
        ar = ar.reshape([...ar.shape, 1]);
      }
      const height = ar.shape[0];
      const width = ar.shape[1];
      const channels = ar.shape[2];

      if (ar.dtype === np.float32 || ar.dtype === np.float16) {
        // If float32, normalize [0, 1) to [0, 256)
        ar = np.clip(ar.mul(256), 0, 255).astype(np.uint32);
      } else if (ar.dtype === np.bool) {
        // If bool, convert to 0 or 255
        ar = ar.astype(np.uint32).mul(255);
      }

      let rgbaArray: np.Array;
      if (channels === 1) {
        ar = np.repeat(ar, 3, 2);
        const alphas = np.full([height, width, 1], 255, {
          dtype: ar.dtype,
          device: ar.device,
        });
        rgbaArray = np.concatenate([ar, alphas], 2);
      } else if (channels === 3) {
        const alphas = np.full([height, width, 1], 255, {
          dtype: ar.dtype,
          device: ar.device,
        });
        rgbaArray = np.concatenate([ar, alphas], 2);
      } else if (channels === 4) {
        rgbaArray = ar;
      } else {
        throw new Error(
          "displayImage() only supports 1, 3, or 4 channels in the last dimension",
        );
      }

      const buf = await rgbaArray.data();
      const dataArray = new Uint8ClampedArray(buf);
      const imageData = new ImageData(dataArray, width, height, {
        colorSpace: "srgb",
      });

      // Create a temporary <canvas> to draw and produce a data URL.
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d")!;
      ctx.putImageData(imageData, 0, 0);
      const dataUrl = canvas.toDataURL();

      // Append the image to the console output.
      consoleLines.push({
        level: "image",
        data: [dataUrl],
        time: Date.now(),
      });
    };

    mockConsole.clear();

    const devices = await jax.init();
    if (devices.includes(device)) {
      jax.defaultDevice(device);
    } else {
      mockConsole.warn(`${device} not supported, falling back to Wasm`);
      jax.defaultDevice("wasm");
    }

    const userCode = replEditor.getText();

    // Create a simple virtual module plugin to resolve our in-memory modules.
    const virtualPlugin: Plugin = {
      name: "virtual",
      resolveId(id) {
        // We treat 'index.ts' as the user code entry point.
        if (id === "index.ts") {
          return id;
        } else {
          throw new Error("Module not found: " + id);
        }
      },
      load(id) {
        if (id === "index.ts") {
          return userCode;
        } else {
          return null;
        }
      },
    };

    const typescriptPlugin: Plugin = {
      name: "typescript",
      transform(code, id) {
        if (id.endsWith(".ts")) {
          return ts.transpileModule(code, {
            compilerOptions: {
              module: ts.ModuleKind.ESNext,
              target: ts.ScriptTarget.ES2022,
            },
          }).outputText;
        }
        return null;
      },
    };

    try {
      // Use @rollup/browser to bundle the code.
      const bundle = await rollup({
        input: "index.ts",
        plugins: [typescriptPlugin, virtualPlugin],
        external: ["@jax-js/jax", "@jax-js/optax", "@jax-js/loaders"],
      });

      // We use the "system" format because it allows you to use async/await.
      // https://rollupjs.org/repl/
      const { output } = await bundle.generate({
        file: "bundle.js",
        format: "system",
      });

      const header = `
      const console = _BUILTINS.console;
      const displayImage = _BUILTINS.displayImage;

      const System = { register(externals, f) {
        const { execute, setters } = f();
        for (let i = 0; i < externals.length; i++) {
          setters[i](_MODULES[externals[i]]);
        }
        this.f = execute;
      } };`;
      const trailer = `;await (async () => System.f())()`;
      const bundledCode = header + output[0].code + trailer;

      // AsyncFunction constructor, analogous to Function.
      const AsyncFunction: typeof Function = async function () {}
        .constructor as any;

      await new AsyncFunction("_MODULES", "_BUILTINS", bundledCode)(
        // _MODULES
        {
          "@jax-js/jax": jax,
          "@jax-js/optax": optax,
          "@jax-js/loaders": loaders,
        },
        // _BUILTINS
        {
          console: mockConsole,
          displayImage: displayImage,
        },
      );
    } catch (e: any) {
      mockConsole.error(e);
    } finally {
      running = false;
    }
  }

  type ConsoleLine = {
    level: "log" | "info" | "warn" | "error" | "image";
    data: string[];
    time: number;
  };

  let consoleLines: ConsoleLine[] = $state([]);
  let running = $state(false);

  // Intercepted methods similar to console.log().
  const consoleMethods = [
    "clear",
    "error",
    "info",
    "log",
    "time",
    "timeEnd",
    "timeLog",
    "trace",
    "warn",
  ] as const;
  const consoleTimers = new Map<string, number>();

  function handleMockConsole(
    method: (typeof consoleMethods)[number],
    ...args: any[]
  ) {
    if (
      method === "log" ||
      method === "info" ||
      method === "warn" ||
      method === "error"
    ) {
      consoleLines.push({
        level: method,
        data: args.map((x) =>
          typeof x === "string"
            ? x
            : x instanceof Error
              ? x.toString()
              : JSON.stringify(x, null, 2),
        ),
        time: Date.now(),
      });
    } else if (method === "clear") {
      consoleLines = [];
    } else if (method === "trace") {
      consoleLines.push({
        level: "error",
        data: ["Received stack trace, see console for details."],
        time: Date.now(),
      });
    } else if (method === "time") {
      consoleTimers.set(args[0], performance.now());
    } else if (method === "timeLog") {
      const start = consoleTimers.get(args[0]);
      if (start !== undefined) {
        const elapsed = performance.now() - start;
        consoleLines.push({
          level: "log",
          data: [`${args[0]}: ${elapsed.toFixed(1)}ms`],
          time: Date.now(),
        });
      }
    } else if (method === "timeEnd") {
      const start = consoleTimers.get(args[0]);
      if (start !== undefined) {
        const elapsed = performance.now() - start;
        consoleLines.push({
          level: "log",
          data: [`${args[0]}: ${elapsed.toFixed(1)}ms - timer ended`],
          time: Date.now(),
        });
        consoleTimers.delete(args[0]);
      }
    }
  }

  const mockConsole = new Proxy(console, {
    get(target, prop, receiver) {
      if (consoleMethods.some((m) => m === prop)) {
        return (...args: any[]) => {
          handleMockConsole(prop as any, ...args);
          Reflect.get(target, prop, receiver)(...args);
        };
      }
      return Reflect.get(target, prop, receiver);
    },
  });
</script>

<svelte:head>
  <title>jax-js REPL</title>
</svelte:head>

<div class="h-dvh">
  <SplitPane
    type="horizontal"
    pos="280px"
    min="200px"
    max="40%"
    --color="var(--color-gray-200)"
  >
    {#snippet a()}
      <div
        class="bg-gray-50 px-4 pt-4 pb-12 !overflow-y-auto"
        style:scrollbar-width="thin"
      >
        <h1 class="text-xl font-light mb-4">
          <a href="{base}/"><span class="font-medium">jax-js</span> REPL</a>
        </h1>

        <hr class="mb-6 border-gray-200" />

        <p class="text-sm mb-4">
          Try out jax-js. Numerical and GPU computing for the web!
        </p>
        <p class="text-sm mb-4">
          The goal is having NumPy and JAX-like APIs <em>in the browser</em>, on
          Wasm or WebGPU — with JIT compilation.
        </p>

        <pre class="mb-4 text-sm bg-gray-100 px-2 py-1 rounded"><code
            >npm i @jax-js/jax</code
          ></pre>

        <h2 class="text-lg mt-8 mb-2">Examples</h2>
        <div class="text-sm flex flex-col">
          {#each codeSamples as { title, id } (id)}
            <button
              class="px-2 py-1 text-left rounded flex items-center hover:bg-gray-100 active:bg-gray-200 transition-colors"
              class:font-semibold={id === sample}
              onclick={() => chooseSample(id)}
            >
              <span class="mr-2">
                <ArrowRightIcon size={16} />
              </span>
              {title}
            </button>
          {/each}
        </div>
      </div>
    {/snippet}
    {#snippet b()}
      <SplitPane
        type="vertical"
        pos="-240px"
        min="10%"
        max="-64px"
        --color="var(--color-gray-200)"
      >
        {#snippet a()}
          <div class="flex flex-col min-w-0">
            <div class="px-4 py-2 flex items-center gap-1">
              <button
                class="bg-emerald-100 hover:bg-emerald-200 active:scale-105 transition-all rounded-md text-sm px-3 py-0.5 flex items-center disabled:opacity-50"
                onclick={handleRun}
                disabled={running}
              >
                <PlayIcon size={14} class="mr-1.5" />
                Run
              </button>
              <button
                class="hover:bg-gray-100 active:scale-105 transition-all rounded-md text-sm px-3 py-0.5 flex items-center disabled:opacity-50"
                onclick={handleFormat}
              >
                <PaletteIcon size={14} class="mr-1.5" />
                Format
              </button>
              <button
                class="hover:bg-gray-100 active:scale-105 transition-all rounded-md text-sm px-3 py-0.5 flex items-center disabled:opacity-50"
                onclick={handleShare}
              >
                <ShareIcon size={14} class="mr-1.5" />
                Share
              </button>

              <!-- Device selector -->
              <select
                bind:value={device}
                class="ml-auto border border-gray-300 rounded-md text-sm px-1 py-0.5"
              >
                <option value="webgpu">WebGPU</option>
                <option value="wasm">Wasm</option>
                <option value="cpu">CPU (slow)</option>
              </select>
            </div>
            <!-- <div class="ml-4 text-sm text-gray-700 pb-1 flex items-center">
              <FileIcon size={14} class="mr-1 text-sky-600" /> index.ts
            </div> -->
            <div class="flex-1 min-h-0">
              <ReplEditor
                initialText={selectionOnLoad.content !== undefined
                  ? selectionOnLoad.content!
                  : src[`./${sample ?? codeSamples[0].id}.ts`]}
                bind:this={replEditor}
                onformat={handleFormat}
                onrun={handleRun}
              />
            </div>
          </div>
        {/snippet}
        {#snippet b()}
          <div class="flex flex-col h-full">
            <p class="text-gray-500 text-sm py-2 px-4 select-none shrink-0">
              Console
              {#if running}
                <LoaderIcon
                  size={16}
                  class="inline-block animate-spin ml-1 mb-[3px]"
                />
              {:else if consoleLines.length === 0}
                <span>(empty)</span>
              {/if}
            </p>
            <div
              class="pb-2 px-4 flex flex-col grow overflow-y-auto text-[13px]"
              style:scrollbar-width="thin"
            >
              {#each consoleLines as line, i (i)}
                <div
                  class={[
                    "py-0.5 border-t flex items-start gap-x-2",
                    line.level === "error"
                      ? "border-red-200 bg-red-50"
                      : line.level === "warn"
                        ? "border-yellow-200 bg-yellow-50"
                        : "border-gray-200",
                  ]}
                >
                  {#if line.level === "log"}
                    <ChevronRightIcon
                      size={18}
                      class="shrink-0 text-gray-300"
                    />
                  {:else if line.level === "info"}
                    <InfoIcon size={18} class="shrink-0 text-blue-500" />
                  {:else if line.level === "warn"}
                    <AlertTriangleIcon
                      size={18}
                      class="shrink-0 text-yellow-500"
                    />
                  {:else if line.level === "error"}
                    <XIcon size={18} class="shrink-0 text-red-500" />
                  {:else if line.level === "image"}
                    <ImageIcon size={18} class="shrink-0 text-gray-400" />
                  {/if}
                  <p class="font-mono whitespace-pre-wrap">
                    {#if line.level === "image"}
                      <img
                        src={line.data[0]}
                        alt="Output from displayImage()"
                        class="max-w-full my-0.5"
                      />
                    {:else}
                      {line.data.join(" ")}
                    {/if}
                  </p>
                  <p
                    class="hidden md:block ml-auto shrink-0 font-mono text-gray-400 select-none"
                  >
                    {new Date(line.time).toLocaleTimeString()}
                  </p>
                </div>
              {/each}
            </div>
          </div>
        {/snippet}
      </SplitPane>
    {/snippet}
  </SplitPane>
</div>

<style lang="postcss">
  @reference "$app.css";
</style>
