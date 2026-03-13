import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

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
      headless: false,
      screenshotFailures: false,
      provider: playwright({
        launchOptions: {
          args: [
            "--headless=new",
            "--no-sandbox",
            "--enable-unsafe-webgpu", // for Linux
            "--enable-features=Vulkan", // for Linux
          ],
        },
      }),
      // https://vitest.dev/config/browser/playwright.html
      instances: [{ browser: "chromium" }],
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
