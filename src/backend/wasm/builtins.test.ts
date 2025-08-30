import { expect, test } from "vitest";

import { wasm_cos, wasm_exp, wasm_log, wasm_sin } from "./builtins";
import { CodeGenerator } from "./wasmblr";

function relativeError(wasmResult: number, jsResult: number): number {
  return Math.abs(wasmResult - jsResult) / (Math.abs(jsResult) + 1);
}

test("wasm_exp has relative error < 2e-5", async () => {
  const cg = new CodeGenerator();

  const expFunc = wasm_exp(cg);
  cg.export(expFunc, "exp");

  const wasmBytes = cg.finish();
  const { instance } = await WebAssembly.instantiate(wasmBytes);
  const { exp } = instance.exports as { exp(x: number): number };

  const testValues = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 10];
  for (const x of testValues) {
    expect(relativeError(exp(x), Math.exp(x))).toBeLessThan(2e-5);
  }
});

test("wasm_log has relative error < 2e-5", async () => {
  const cg = new CodeGenerator();

  const logFunc = wasm_log(cg);
  cg.export(logFunc, "log");

  const wasmBytes = cg.finish();
  const { instance } = await WebAssembly.instantiate(wasmBytes);
  const { log } = instance.exports as { log(x: number): number };

  const testValues = [0.01, 0.1, 0.5, 1, 1.5, 2, Math.E, 5, 10, 100];
  for (const x of testValues) {
    expect(relativeError(log(x), Math.log(x))).toBeLessThan(2e-5);
  }

  // Test edge case: log(x <= 0) should return NaN
  expect(log(0)).toBeNaN();
  expect(log(-1)).toBeNaN();
});

test("wasm_sin has absolute error < 1e-5", async () => {
  const cg = new CodeGenerator();

  const sinFunc = wasm_sin(cg);
  cg.export(sinFunc, "sin");

  const wasmBytes = cg.finish();
  const { instance } = await WebAssembly.instantiate(wasmBytes);
  const { sin } = instance.exports as { sin(x: number): number };

  // Test a range of values including critical points
  const testValues = [
    -2 * Math.PI,
    -Math.PI,
    -Math.PI / 2,
    -Math.PI / 4,
    0,
    Math.PI / 6,
    Math.PI / 4,
    Math.PI / 3,
    Math.PI / 2,
    Math.PI,
    (3 * Math.PI) / 2,
    2 * Math.PI,
    5,
    10,
    -5,
    -10,
  ];

  for (const x of testValues) {
    expect(Math.abs(sin(x) - Math.sin(x))).toBeLessThan(1e-5);
  }
});

test("wasm_cos has absolute error < 1e-5", async () => {
  const cg = new CodeGenerator();

  const cosFunc = wasm_cos(cg);
  cg.export(cosFunc, "cos");

  const wasmBytes = cg.finish();
  const { instance } = await WebAssembly.instantiate(wasmBytes);
  const { cos } = instance.exports as { cos(x: number): number };

  // Test a range of values including critical points
  const testValues = [
    -2 * Math.PI,
    -Math.PI,
    -Math.PI / 2,
    -Math.PI / 4,
    0,
    Math.PI / 6,
    Math.PI / 4,
    Math.PI / 3,
    Math.PI / 2,
    Math.PI,
    (3 * Math.PI) / 2,
    2 * Math.PI,
    5,
    10,
    -5,
    -10,
  ];

  for (const x of testValues) {
    expect(Math.abs(cos(x) - Math.cos(x))).toBeLessThan(1e-5);
  }
});
