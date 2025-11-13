<script lang="ts">
  import { building } from "$app/environment";
  import { goto } from "$app/navigation";
  import { base } from "$app/paths";
  import { page } from "$app/state";

  import { SplitPane } from "@rich_harris/svelte-split-pane";
  import type { Plugin } from "@rollup/browser";
  import {
    AlertTriangleIcon,
    ArrowRightIcon,
    ChevronRightIcon,
    InfoIcon,
    LoaderIcon,
    PaletteIcon,
    PlayIcon,
    TerminalIcon,
    X,
  } from "lucide-svelte";

  import ReplEditor from "$lib/repl/ReplEditor.svelte";

  const src: Record<string, string> = import.meta.glob("./*.ts", {
    eager: true,
    query: "?raw",
    import: "default",
  });

  const codeSamples: {
    title: string;
    code: string;
  }[] = [
    { title: "Arrays", code: src["./01-arrays.ts"] },
    { title: "Tracing Jaxprs", code: src["./02-tracing.ts"] },
    { title: "Logistic regression", code: src["./03-logistic-regression.ts"] },
    { title: "Mandelbrot set", code: src["./04-mandelbrot.ts"] },
  ];

  function getSampleFromUrl(url: URL) {
    const str = building ? "0" : (url.searchParams.get("sample") ?? "0");
    const i = parseInt(str);
    if (Number.isInteger(i) && i >= 0 && i < codeSamples.length) return i;
    return 0;
  }

  let selected = $state(getSampleFromUrl(page.url));
  let replEditor: ReplEditor;

  $effect(() => {
    // When selected changes, update the query string in the URL.
    if (selected !== getSampleFromUrl(page.url)) {
      goto(page.url.pathname + `?sample=${selected}`, { replaceState: true });
    }
  });

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

    mockConsole.clear();

    const devices = await jax.init();
    if (devices.includes("webgpu")) {
      jax.setDevice("webgpu");
    } else {
      mockConsole.warn(`WebGPU not supported, falling back to Wasm`);
      jax.setDevice("wasm");
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

      const header = `const System = { register(externals, f) {
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

      await new AsyncFunction("_MODULES", "console", bundledCode)(
        {
          "@jax-js/jax": jax,
          "@jax-js/optax": optax,
          "@jax-js/loaders": loaders,
        },
        mockConsole,
      );
    } catch (e: any) {
      mockConsole.error(e);
    } finally {
      running = false;
    }
  }

  type ConsoleLine = {
    level: "log" | "info" | "warn" | "error";
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
          {#each codeSamples as { title, code }, i (i)}
            <button
              class="px-2 py-1 text-left rounded flex items-center hover:bg-gray-100 active:bg-gray-200 transition-colors"
              class:font-semibold={i === selected}
              onclick={() => {
                selected = i;
                replEditor.setText(code);
              }}
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
        min="33%"
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
            </div>
            <!-- <div class="ml-4 text-sm text-gray-700 pb-1 flex items-center">
              <FileIcon size={14} class="mr-1 text-sky-600" /> index.ts
            </div> -->
            <div class="flex-1 min-h-0">
              <ReplEditor
                initialText={codeSamples[selected].code}
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
              class="pb-2 px-4 flex flex-col grow overflow-y-auto"
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
                    <ChevronRightIcon size={18} class="text-gray-300" />
                  {:else if line.level === "info"}
                    <InfoIcon size={18} class="text-blue-500" />
                  {:else if line.level === "warn"}
                    <AlertTriangleIcon size={18} class="text-yellow-500" />
                  {:else if line.level === "error"}
                    <X size={18} class="text-red-500" />
                  {/if}
                  <p class="text-sm font-mono whitespace-pre-wrap">
                    {line.data.join(" ")}
                  </p>
                  <p
                    class="hidden md:block ml-auto shrink-0 text-sm font-mono text-gray-400 select-none"
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
