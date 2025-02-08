/// <reference types="@vitest/browser/providers/playwright" />

import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    setupFiles: ["test/setup.ts"],
    browser: {
      enabled: true,
      headless: true,
      screenshotFailures: false,
      provider: "playwright",
      // https://vitest.dev/guide/browser/playwright
      instances: [{ browser: "chromium" }],
    },
  },
});
