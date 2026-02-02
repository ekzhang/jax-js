import { beforeAll, describe, it } from "vitest";

import { dlmFit, type DlmMode } from "./dlm";
// Import JSON fixtures directly (bundled by Vitest)
import nileInput from "./niledemo-in.json";
import nileKeys from "./niledemo-keys.json";
import nileReference from "./niledemo-out-m.json";
import { deepAlmostEqual, filterKeys } from "./utils";
import { defaultDevice, Device, DType, init } from "../../src";

describe("niledemo output", () => {
  beforeAll(async () => {
    await init();
  });

  const runTest = async (mode: DlmMode, label: string) => {
    const devices = await init();
    let useDevice: Device = "cpu";
    let useDType = DType.Float64;
    if (devices.includes("webgpu")) {
      useDevice = "webgpu";
      useDType = DType.Float32;
    } else if (devices.includes("wasm")) {
      useDevice = "wasm";
    }
    console.log(`Using device: ${useDevice}, dtype: ${useDType}`);
    defaultDevice(useDevice);
    const startTime = performance.now();
    const result = await dlmFit(
      nileInput.y,
      nileInput.s,
      nileInput.w as [number, number],
      useDType,
      mode,
    );
    const endTime = performance.now();
    console.log(`[${label}] Time: ${(endTime - startTime).toFixed(0)}ms`);

    // Filter to compared keys only
    const filteredResult = filterKeys(result, nileKeys);
    const filteredReference = filterKeys(nileReference, nileKeys);

    const relativeTolerance = 1e-10;
    const cmp = deepAlmostEqual(
      filteredResult,
      filteredReference,
      relativeTolerance,
    );
    if (!cmp.equal) {
      throw new Error(
        `[${label}] Output does not match reference.\n` +
          `First mismatch at: ${cmp.path}\n` +
          `Result value:    ${JSON.stringify(cmp.a)}\n` +
          `Reference value: ${JSON.stringify(cmp.b)}`,
      );
    }
  };

  it(`should match reference using for-loops`, async () => {
    await runTest("for", "for-loop");
  });

  it(`should match reference using lax.scan`, async () => {
    await runTest("scan", "lax.scan");
  });

  it(`should match reference using jit(scan)`, async () => {
    // First run includes JIT compilation overhead
    await runTest("jit", "jit(scan) warmup");
    // Second run uses cached compiled code
    await runTest("jit", "jit(scan) cached");
  });
});
