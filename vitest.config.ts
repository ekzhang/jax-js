import { playwright } from "@vitest/browser-playwright";
import { configDefaults, defineConfig } from "vitest/config";

const BROWSER = process.env.BROWSER || "chromium";

export default defineConfig({
  esbuild: {
    supported: {
      using: false, // Needed to lower 'using' statements in tests.
    },
  },
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  test: {
    browser: {
      enabled: true,
      // Explicitly set to false, but enabled in "args" below. We don't want to
      // use the `chromium-headless-shell` build because that is not compiled
      // with WebGPU support.
      headless: BROWSER !== "chromium",
      ui: false,
      screenshotFailures: false,
      provider: playwright({
        launchOptions: {
          args: ["--headless=new", "--no-sandbox"], // Chromium
          firefoxUserPrefs: {
            // GitHub Actions does not have WebGPU for Firefox, throws UnsupportedError.
            "dom.webgpu.enabled": !process.env.GITHUB_ACTIONS,
          },
        },
      }),
      // https://vitest.dev/config/browser/playwright.html
      instances: [{ browser: BROWSER as any }],
    },
    coverage: {
      // coverage is disabled by default, run with `pnpm test:coverage`.
      enabled: false,
      provider: "v8",
    },
    exclude: [
      ...configDefaults.exclude,
      ...(BROWSER === "webkit"
        ? [
            // TODO: Fails due to UnknownError and no logs, haven't debugged yet.
            "packages/loaders/src/tokenizers.test.ts",
          ]
        : []),
    ],
    isolate: false,
    passWithNoTests: true,
    setupFiles: ["test/setup.ts"],
  },
});
