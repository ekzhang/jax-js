import { expect, test } from "vitest";

import { wasm_exp, wasm_log } from "./builtins";
import { CodeGenerator } from "./wasmblr";

test("wasm_exp has relative error < 2e-5", async () => {
  const cg = new CodeGenerator();

  const expFunc = wasm_exp(cg);
  cg.export(expFunc, "exp");

  const wasmBytes = cg.finish();
  const { instance } = await WebAssembly.instantiate(wasmBytes);
  const { exp } = instance.exports as { exp(x: number): number };

  // Test a range of values
  const testValues = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 10];

  for (const x of testValues) {
    const wasmResult = exp(x);
    const jsResult = Math.exp(x);

    const relativeError =
      Math.abs(wasmResult - jsResult) / (Math.abs(jsResult) + 1);
    expect(relativeError).toBeLessThan(2e-5);
  }
});

test("wasm_log has relative error < 2e-5", async () => {
  const cg = new CodeGenerator();

  const logFunc = wasm_log(cg);
  cg.export(logFunc, "log");

  const wasmBytes = cg.finish();
  const { instance } = await WebAssembly.instantiate(wasmBytes);
  const { log } = instance.exports as { log(x: number): number };

  // Test a range of positive values (log domain is x > 0)
  const testValues = [0.01, 0.1, 0.5, 1, 1.5, 2, Math.E, 5, 10, 100];

  for (const x of testValues) {
    const wasmResult = log(x);
    const jsResult = Math.log(x);

    const relativeError =
      Math.abs(wasmResult - jsResult) / (Math.abs(jsResult) + 1);

    expect(relativeError).toBeLessThan(2e-5);
  }

  // Test edge case: log(x <= 0) should return NaN
  expect(log(0)).toBeNaN();
  expect(log(-1)).toBeNaN();
});
