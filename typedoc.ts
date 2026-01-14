// Executable script for generating typedoc documentation.

import { spawn } from "node:child_process";
import { readdir } from "node:fs/promises";

/** Run a shell command, echo its output, and throw on non‑zero exit. */
async function sh(
  strings: TemplateStringsArray,
  ...values: (string | number)[]
): Promise<void> {
  // Use raw strings to preserve backslashes (like String.raw)
  const cmd = strings.raw.reduce(
    (acc, str, i) => acc + str + (values[i] ?? ""),
    "",
  );

  return new Promise((resolve, reject) => {
    const child = spawn(cmd, { shell: true, stdio: "inherit" });
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`Command failed with exit code ${code}: ${cmd}`));
      } else {
        resolve();
      }
    });
  });
}

function sourceLinkOptions(dir: string): string {
  return [
    // Disable automatic Git detection and pass in specific link.
    "--disableGit",
    "--gitRevision main",
    `--sourceLinkTemplate "https://github.com/ekzhang/jax-js/blob/{gitRevision}/${dir}{path}#L{line}"`,
  ].join(" ");
}

const cmds: Promise<void>[] = [];

// Docs for @jax-js/jax.
cmds.push(
  sh`pnpm typedoc src/index.ts ${sourceLinkOptions("")} --readme none --json docs-json/jax.json`,
);

// Generate docs for each package in the packages directory.
for (const pkg of await readdir("packages")) {
  cmds.push(
    sh`pnpm typedoc packages/${pkg}/src/index.ts ${sourceLinkOptions(`packages/${pkg}/src/`)} --json docs-json/${pkg}.json`,
  );
}

await Promise.all(cmds);

// Merge all package docs into a single HTML output.
await sh`
pnpm typedoc \
  --name jax-js \
  --entryPointStrategy merge \
  --favicon website/static/favicon.ico \
  --titleLink "https://jax-js.com" \
  --searchInDocuments \
  --searchInComments \
  --router structure \
  --navigation.includeFolders false \
  --projectDocuments "README.md" \
  --readme none \
  --plugin typedoc-theme-fresh \
  --theme fresh \
  "docs-json/*.json"
`;
