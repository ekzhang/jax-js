import { readdir } from "node:fs/promises";

import { defineConfig, type Options } from "tsdown";

const watchMode = process.env.TSDOWN_WATCH_MODE === "1";

// Common options for all packages.
const opts: Options = {
  external: (id: string) => {
    // Externalize all imports by default, except for runtime helpers generated
    // by the compiler / bundler toolchain.
    if (id.startsWith("@oxc-project/runtime")) return false;
    return Boolean(id.match(/^[^./]/));
  },
  format: ["cjs", "esm"],
  platform: "browser",
  dts: {
    // Without newContext, dts generation in watch mode has trouble updating
    // when files are changed.
    // https://github.com/rolldown/tsdown/issues/396
    newContext: true,
  },
};

export default defineConfig([
  {
    name: "jax",
    ...opts,
    watch: watchMode && "src",
  },

  ...(await readdir("packages")).map((pkg) => ({
    name: pkg,
    ...opts,
    cwd: `packages/${pkg}`,
    watch: watchMode && `packages/${pkg}/src`, // Unaffected by cwd.
  })),
] as Options[]);
