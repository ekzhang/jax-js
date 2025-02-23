import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["cjs", "esm"],
  dts: true,
  clean: true,

  // TODO: Needed for REPL but this is inefficient. Fix before release.
  // https://github.com/egoist/tsup/issues/619
  noExternal: [/(.*)/],
  splitting: false,
});
