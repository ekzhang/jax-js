import { expect, test } from "vitest";

import {
  wasm_cos,
  wasm_exp,
  wasm_log,
  wasm_sin,
  wasm_threefry2x32,
} from "./builtins";
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

test("wasm_threefry2x32 produces expected results", async () => {
  const cg = new CodeGenerator();

  const threefryFunc = wasm_threefry2x32(cg);
  cg.export(threefryFunc, "threefry2x32");

  const wasmBytes = cg.finish();
  const { instance } = await WebAssembly.instantiate(wasmBytes);

  const threefry2x32 = instance.exports.threefry2x32 as CallableFunction;

  // Test known vector: all zeros input
  const result0 = threefry2x32(0, 0, 0, 0) as [number, number];
  expect(Array.isArray(result0)).toBe(true);
  expect(result0).toHaveLength(2);

  // Convert to unsigned 32-bit for comparison
  const x0 = result0[0] >>> 0; // Convert to unsigned 32-bit
  const x1 = result0[1] >>> 0;

  expect(x0).toBe(1797259609);
  expect(x1).toBe(2579123966);

  // Test that different inputs produce different outputs
  const result1 = threefry2x32(0, 0, 0, 0) as [number, number];
  const result2 = threefry2x32(0, 0, 0, 1) as [number, number];
  const result3 = threefry2x32(1, 0, 0, 0) as [number, number];

  expect(result1).not.toEqual(result2);
  expect(result1).not.toEqual(result3);
  expect(result2).not.toEqual(result3);

  // Test with non-zero keys
  const result4 = threefry2x32(
    0xdeadbeef,
    0xcafebabe,
    0x12345678,
    0x87654321,
  ) as [number, number];
  expect(Array.isArray(result4)).toBe(true);
  expect(result4).toHaveLength(2);
});
