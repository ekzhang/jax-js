import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

const BROWSER = process.env.BROWSER || "chromium";

export default defineConfig({
  esbuild: {
    supported: {
      using: false, // Needed to lower 'using' statements in tests.
    },
  },
  test: {
    isolate: false,
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
          args:
            BROWSER === "chromium" ? ["--headless=new", "--no-sandbox"] : [],
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
    passWithNoTests: true,
    setupFiles: ["test/setup.ts"],
  },
});
