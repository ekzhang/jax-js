<script lang="ts">
  import { SplitPane } from "@rich_harris/svelte-split-pane";
  import type { Plugin } from "@rollup/browser";
  import { ArrowRightIcon, PaletteIcon, PlayIcon } from "lucide-svelte";
  import { format } from "prettier";
  import * as prettierParserTypescript from "prettier/parser-typescript";
  import * as prettierPluginEstree from "prettier/plugins/estree";

  import ReplEditor from "$lib/repl/ReplEditor.svelte";
  import { jaxSrc } from "$lib/jax-js-source";

  const codeSamples: {
    title: string;
    code: string;
  }[] = [
    {
      title: "Arrays",
      code: String.raw`import { grad, numpy as np } from "@jax-js/core";

const f = (x: np.Array) => x.mul(x);
const df = grad(f);

const x = np.array(3);
console.log(f(x).js());
console.log(df(x).js());
`,
    },
    {
      title: "Logistic regression",
      code: String.raw`import { numpy as np } from "@jax-js/core";

const X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]]);
const y = np.dot(X, np.array([1, 2])).add(3);

// TODO
`,
    },
    {
      title: "Mandelbrot set",
      code: String.raw`import { numpy as np } from "@jax-js/core";
// TODO
`,
    },
  ];

  let selected = $state(0);
  let replEditor: ReplEditor;

  async function handleFormat() {
    const code = replEditor.getText();
    try {
      const formattedCode = await format(code, {
        parser: "typescript",
        plugins: [prettierParserTypescript, prettierPluginEstree as any],
      });
      replEditor.setText(formattedCode);
    } catch (e: any) {
      // TODO: Display the error in the console.
      alert(e.toString());
    }
  }

  async function handleRun() {
    const ts = await import("typescript");
    const { rollup } = await import("@rollup/browser");

    const userCode = replEditor.getText();

    // Create a simple virtual module plugin to resolve our in-memory modules.
    const virtualPlugin: Plugin = {
      name: "virtual",
      resolveId(id) {
        console.log("resolving", id);
        // We treat 'index.ts' as the user code entry point.
        if (id === "index.ts" || id === "@jax-js/core") {
          return id;
        } else {
          console.log("Module not found: " + id);
          return "<empty>";
        }
      },
      load(id) {
        if (id === "index.ts") {
          return userCode;
        } else if (id === "@jax-js/core") {
          return jaxSrc;
        } else if (id === "<empty>") {
          return "export default {}";
        } else {
          return null;
        }
      },
    };

    const typescriptPlugin: Plugin = {
      name: "typescript",
      transform(code, id) {
        console.log(code.length, id);
        if (id.endsWith(".ts")) {
          return ts.transpileModule(code, {
            compilerOptions: {
              module: ts.ModuleKind.ESNext,
              target: ts.ScriptTarget.ES2017,
            },
          }).outputText;
        }
        return null;
      },
    };

    // Use @rollup/browser to bundle the code.
    const bundle = await rollup({
      input: "index.ts",
      plugins: [typescriptPlugin, virtualPlugin],
    });

    const { output } = await bundle.generate({
      file: "bundle.js",
      format: "iife",
    });

    const bundledCode = output[0].code;

    eval(bundledCode);
  }
</script>

<div class="h-dvh">
  <SplitPane type="horizontal" pos="288px" min="240px" max="40%">
    {#snippet a()}
      <div class="shrink-0 bg-gray-50 px-4 py-4">
        <h1 class="text-xl font-light mb-4">
          <a href="/"><span class="font-medium">jax-js</span> REPL</a>
        </h1>

        <hr class="mb-6 border-gray-300" />

        <p class="text-sm mb-4">
          Try out jax-js in this in-browser editor. Machine learning and
          numerical computing on the web!
        </p>
        <p class="text-sm mb-4">
          The goal is to <em>just use</em> NumPy and JAX in the browser, on WASM
          or GPU.
        </p>

        <pre class="mb-4 text-sm bg-gray-100 px-2 py-1 rounded"><code
            >npm i @jax-js/core</code
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
        pos="-120px"
        min="400px"
        max="40px"
        --color="black"
      >
        {#snippet a()}
          <div class="flex flex-col min-w-0">
            <div class="px-4 py-1 flex items-center gap-4">
              <button
                class="bg-green-600 hover:bg-green-500 text-white px-4 py-0.5 flex items-center"
                onclick={handleRun}
              >
                <PlayIcon size={16} class="mr-1.5" />
                Run
              </button>
              <button
                class="bg-gray-600 hover:bg-gray-500 text-white px-4 py-0.5 flex items-center"
                onclick={handleFormat}
              >
                <PaletteIcon size={16} class="mr-1.5" />
                Format
              </button>
            </div>
            <div class="flex-1 min-h-0">
              <ReplEditor
                initialText={codeSamples[selected].code}
                bind:this={replEditor}
              />
            </div>
          </div>
        {/snippet}
        {#snippet b()}
          <div>hello</div>
        {/snippet}
      </SplitPane>
    {/snippet}
  </SplitPane>
</div>

<style lang="postcss">
  @reference "$app.css";
</style>
