/**
 * Deno test harness with memory leak detection for jax-js WebGPU tests.
 *
 * Usage:
 *   import { withLeakCheck, assertNoLeaks, getSlotCount } from "./harness.ts";
 *
 *   Deno.test("my test", withLeakCheck(async () => {
 *     // test code - all arrays should be consumed by end
 *   }));
 *
 * Or manually:
 *   Deno.test("my test", async () => {
 *     const before = getSlotCount();
 *     // test code
 *     assertNoLeaks(before, "test name");
 *   });
 */

import { assertEquals } from "https://deno.land/std@0.224.0/assert/mod.ts";
import { getBackend, defaultDevice } from "../../dist/index.js";

/**
 * Get the current slot count from the active backend.
 * Returns 0 if no backend is initialized.
 */
export function getSlotCount(): number {
  try {
    const device = defaultDevice();
    if (!device) return 0;
    const backend = getBackend(device);
    return backend.slotCount();
  } catch {
    return 0;
  }
}

/**
 * Assert that no memory was leaked since the baseline.
 * @param baselineSlots - Slot count before the test
 * @param testName - Optional test name for error message
 */
export function assertNoLeaks(baselineSlots: number, testName?: string): void {
  const currentSlots = getSlotCount();
  const leaked = currentSlots - baselineSlots;
  if (leaked > 0) {
    const msg = testName
      ? `Memory leak in "${testName}": ${leaked} slot(s) leaked (${baselineSlots} -> ${currentSlots})`
      : `Memory leak: ${leaked} slot(s) leaked (${baselineSlots} -> ${currentSlots})`;
    throw new Error(msg);
  }
}

/**
 * Wrapper that adds memory leak checking to a test function.
 * Use with Deno.test:
 *
 *   Deno.test("my test", withLeakCheck(async () => { ... }));
 *
 * @param fn - The test function
 * @param options - Optional settings
 */
export function withLeakCheck(
  fn: () => void | Promise<void>,
  options?: { allowedLeaks?: number },
): () => Promise<void> {
  const allowedLeaks = options?.allowedLeaks ?? 0;

  return async () => {
    const before = getSlotCount();
    try {
      await fn();
    } finally {
      const after = getSlotCount();
      const leaked = after - before;
      if (leaked > allowedLeaks) {
        throw new Error(
          `Memory leak: ${leaked} slot(s) leaked (before=${before}, after=${after}, allowed=${allowedLeaks})`,
        );
      }
    }
  };
}

/**
 * Test options with memory leak checking.
 */
export interface LeakCheckTestOptions extends Deno.TestDefinition {
  /** Number of leaked slots allowed (default: 0) */
  allowedLeaks?: number;
}

/**
 * Enhanced test registration with automatic leak checking.
 *
 * Usage:
 *   leakCheckTest({
 *     name: "my test",
 *     fn: async () => { ... },
 *   });
 */
export function leakCheckTest(options: LeakCheckTestOptions): void {
  const { allowedLeaks, fn, ...rest } = options;
  Deno.test({
    ...rest,
    fn: withLeakCheck(fn, { allowedLeaks }),
  });
}
