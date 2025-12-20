import type { Device, numpy as np } from "@jax-js/jax";
import type { Plugin } from "@rollup/browser";

export type ConsoleLine = {
  level: "log" | "info" | "warn" | "error" | "image";
  data: string[];
  time: number;
};

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

export class ReplRunner {
  running = $state(false);
  finished = $state(false);
  consoleLines: ConsoleLine[] = $state([]);
  runDurationMs = $state<number | null>(null);
  consoleTimers = new Map<string, number>();
  mockConsole: Console;

  constructor() {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const runner = this;

    this.mockConsole = new Proxy(console, {
      get(target, prop, receiver) {
        if (consoleMethods.some((m) => m === prop)) {
          return (...args: any[]) => {
            runner.#handleMockConsole(prop as any, ...args);
            Reflect.get(target, prop, receiver)(...args);
          };
        }
        return Reflect.get(target, prop, receiver);
      },
    });
  }

  async runProgram(source: string, device: Device) {
    if (this.running) return;
    this.running = true;
    this.finished = false;
    this.runDurationMs = null;
    const startTime = performance.now();
    try {
      await _runProgram(source, device, this);
    } finally {
      const endTime = performance.now();
      const duration = endTime - startTime;
      this.runDurationMs = duration;
      if (duration < 100) {
        // Take at least 100ms, otherwise it's unclear it actually ran.
        await new Promise((resolve) => setTimeout(resolve, 100 - duration));
      }
      this.running = false;
      this.finished = true;
    }
  }

  #handleMockConsole(method: (typeof consoleMethods)[number], ...args: any[]) {
    if (
      method === "log" ||
      method === "info" ||
      method === "warn" ||
      method === "error"
    ) {
      this.consoleLines.push({
        level: method,
        data: args.map((x) =>
          typeof x === "string"
            ? x
            : x instanceof Error
              ? x.toString()
              : formatObject(x),
        ),
        time: Date.now(),
      });
    } else if (method === "clear") {
      this.consoleLines = [];
    } else if (method === "trace") {
      this.consoleLines.push({
        level: "error",
        data: ["Received stack trace, see console for details."],
        time: Date.now(),
      });
    } else if (method === "time") {
      this.consoleTimers.set(args[0], performance.now());
    } else if (method === "timeLog") {
      const start = this.consoleTimers.get(args[0]);
      if (start !== undefined) {
        const elapsed = performance.now() - start;
        this.consoleLines.push({
          level: "log",
          data: [`${args[0]}: ${elapsed.toFixed(1)}ms`],
          time: Date.now(),
        });
      }
    } else if (method === "timeEnd") {
      const start = this.consoleTimers.get(args[0]);
      if (start !== undefined) {
        const elapsed = performance.now() - start;
        this.consoleLines.push({
          level: "log",
          data: [`${args[0]}: ${elapsed.toFixed(1)}ms - timer ended`],
          time: Date.now(),
        });
        this.consoleTimers.delete(args[0]);
      }
    }
  }
}

async function _runProgram(source: string, device: Device, runner: ReplRunner) {
  const [jax, optax, loaders] = await Promise.all([
    import("@jax-js/jax"),
    import("@jax-js/optax"),
    import("@jax-js/loaders"),
  ]);
  const ts = await import("typescript");
  const { rollup } = await import("@rollup/browser");

  const mockConsole = runner.mockConsole;

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
    runner.consoleLines.push({
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
    mockConsole.warn(`${device} not supported, using wasm`);
    jax.defaultDevice("wasm");
  }

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
        return source;
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
  }
}

function formatObject(obj: any): string {
  const buffer: string[] = [];
  obj = _convertJaxArrays(obj);
  _formatObject(obj, "", buffer);
  return buffer.join("");
}

function _convertJaxArrays(obj: any): any {
  if (typeof obj !== "object" || obj === null) {
    return obj;
  } else if (typeof obj["js"] === "function") {
    return obj.js();
  } else if (Array.isArray(obj)) {
    return obj.map((x) => _convertJaxArrays(x));
  } else {
    const newObj: any = {};
    for (const [k, v] of Object.entries(obj)) {
      newObj[k] = _convertJaxArrays(v);
    }
    return newObj;
  }
}

/**
 * Format an object with indentation, in JSON style, and keeping lists / objects
 * inline if they don't exceed 120 characters in width.
 */
function _formatObject(obj: any, indent: string, buffer: string[]) {
  if (typeof obj !== "object" || obj === null) {
    buffer.push(_stringifyOneLine(obj));
    return;
  }

  const strRep = _stringifyOneLine(obj);
  if (strRep.length <= 120) {
    buffer.push(strRep);
  } else {
    if (Array.isArray(obj)) {
      buffer.push("[\n");
      const newIndent = indent + "  ";
      for (let i = 0; i < obj.length; i++) {
        buffer.push(newIndent);
        _formatObject(obj[i], newIndent, buffer);
        buffer.push(",\n");
      }
      buffer.push(indent + "]");
    } else {
      buffer.push("{\n");
      const newIndent = indent + "  ";
      const keys = Object.keys(obj);
      for (let i = 0; i < keys.length; i++) {
        const key = keys[i];
        buffer.push(newIndent + _stringifyKey(key) + ": ");
        _formatObject(obj[key], newIndent, buffer);
        buffer.push(",\n");
      }
      buffer.push(indent + "}");
    }
  }
}

function _stringifyOneLine(obj: any): string {
  if (typeof obj === "number") {
    // Format numbers with up to 7 significant digits.
    return obj.toPrecision(7).replace(/\.?0+$/, "");
  } else if (typeof obj !== "object" || obj === null) {
    return JSON.stringify(obj);
  } else if (Array.isArray(obj)) {
    return "[" + obj.map(_stringifyOneLine).join(", ") + "]";
  } else {
    return (
      "{ " +
      Object.entries(obj)
        .map(([k, v]) => `${_stringifyKey(k)}: ${_stringifyOneLine(v)}`)
        .join(", ") +
      " }"
    );
  }
}

function _stringifyKey(key: string): string {
  // If key is a valid identifier, return as is.
  if (/^[A-Za-z_$][A-Za-z0-9_$]*$/.test(key)) {
    return key;
  }
  return JSON.stringify(key);
}
