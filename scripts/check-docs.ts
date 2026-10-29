// Check every generated source link against the repository checkout.
import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import { isAbsolute, relative, resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const sourceLines = new Map<string, number>();

function checkSources(value: unknown): number {
  if (!value || typeof value !== "object") return 0;
  if (Array.isArray(value)) {
    return value.reduce<number>(
      (count, child) => count + checkSources(child),
      0,
    );
  }

  let count = 0;
  for (const [key, child] of Object.entries(value)) {
    if (key !== "sources") {
      count += checkSources(child);
      continue;
    }
    assert(Array.isArray(child), "Expected a source reference array");
    for (const source of child) {
      assert.equal(typeof source.fileName, "string");
      const fileName = source.fileName as string;
      const path = relative(root, resolve(root, fileName));
      assert(!isAbsolute(fileName) && !path.startsWith(".."), fileName);
      assert(
        fileName.startsWith("src/") || /^packages\/[^/]+\/src\//.test(fileName),
        `Source must be relative to the repository root: ${fileName}`,
      );
      if (!sourceLines.has(fileName)) {
        sourceLines.set(
          fileName,
          readFileSync(resolve(root, fileName), "utf8").split("\n").length,
        );
      }
      assert(Number.isInteger(source.line) && source.line > 0, fileName);
      assert(source.line <= sourceLines.get(fileName)!, fileName);
      assert.equal(
        source.url,
        `https://github.com/ekzhang/jax-js/blob/main/${fileName}#L${source.line}`,
        `Incorrect source link for ${fileName}`,
      );
      count++;
    }
  }
  return count;
}

for (const pkg of ["jax", ...readdirSync(resolve(root, "packages"))]) {
  const project: unknown = JSON.parse(
    readFileSync(resolve(root, "docs-json", `${pkg}.json`), "utf8"),
  );
  const count = checkSources(project);
  assert(count > 0, `No source references found for ${pkg}`);
  console.log(`Checked ${count} source references for ${pkg}`);
}
