import { expect, test, vi } from "vitest";

import { defaultDevice, getBackend, init } from "../backend";
import { DType, jit, numpy as np } from "../index";
import { Primitive, ShapedArray } from "./core";
import { Jaxpr, JaxprEqn, Lit, Var } from "./jaxpr";

await init("cpu");

test("splits kernels that exceed maxArgs when an input is already black", () => {
  // WebGPU and WebGL allow 7-8 buffers per kernel. Lower the limit on the cpu
  // backend so a small program has to be split.
  defaultDevice("cpu");
  const maxArgs = vi
    .spyOn(getBackend("cpu"), "maxArgs", "get")
    .mockReturnValue(3);
  try {
    // The where() sees 4 buffers: w through its condition, and a, b, c through
    // y and sin(y). The condition is marked black first, which is not enough,
    // so y must be marked black next instead of the condition again.
    const f = jit((w: np.Array, a: np.Array, b: np.Array, c: np.Array) => {
      const y = a.add(b).add(c);
      return np.where(w.greater(0), y.ref, np.sin(y));
    });
    const out = f(
      np.array([1, -1]),
      np.array([1, 2]),
      np.array([3, 4]),
      np.array([5, 6]),
    );
    const [x0, x1] = out.js() as number[];
    expect(x0).toBeCloseTo(9);
    expect(x1).toBeCloseTo(Math.sin(12));
  } finally {
    maxArgs.mockRestore();
  }
});

test("JIT caches equal scalar literals separately by dtype", () => {
  const previousDevice = defaultDevice();
  defaultDevice("wasm");
  const f = jit(() => np.array(2, { dtype: DType.Float32 }));
  const g = jit(() => np.array(2, { dtype: DType.Int32 }));
  try {
    expect(f().js()).toBe(2);
    expect(g().js()).toBe(2);
  } finally {
    g.dispose();
    f.dispose();
    defaultDevice(previousDevice);
  }
});

test("Jaxpr cache keys use typed atoms and binder order", () => {
  const scalar = new ShapedArray([], DType.Int32, false);
  const weakScalar = new ShapedArray([], DType.Int32, true);
  const x = new Var(scalar);
  const y = new Var(scalar);
  const bound = new Jaxpr([x], [], [x]);
  const literal = new Jaxpr([x], [], [new Lit(scalar, 1)]);
  const weakLiteral = new Jaxpr([x], [], [new Lit(weakScalar, 1)]);
  expect(bound.getCacheKey()).toMatch(/^[a-f0-9]{64}$/);
  expect(bound.getHash()).toBe(BigInt(`0x${bound.getCacheKey()}`));
  expect(bound.getCacheKey()).toBe(new Jaxpr([y], [], [y]).getCacheKey());
  expect(bound.getCacheKey()).not.toBe(literal.getCacheKey());
  expect(literal.getCacheKey()).not.toBe(weakLiteral.getCacheKey());
  expect(
    new Jaxpr([new Var(weakScalar)], [], literal.outs).getCacheKey(),
  ).not.toBe(literal.getCacheKey());
});

test("Jaxpr cache keys preserve available NaN payloads", () => {
  const buffer = new DataView(new ArrayBuffer(8));
  const scalar = new ShapedArray([], DType.Float64, false);
  const literalKey = (bits: bigint) => {
    buffer.setBigUint64(0, bits);
    const value = buffer.getFloat64(0);
    expect(Number.isNaN(value)).toBe(true);
    buffer.setFloat64(0, value);
    return {
      bits: buffer.getBigUint64(0),
      key: new Jaxpr([], [], [new Lit(scalar, value)]).getCacheKey(),
    };
  };
  const first = literalKey(0x7ff8000000000001n);
  const second = literalKey(0x7ff8000000000002n);
  expect(first.key === second.key).toBe(first.bits === second.bits);
});

test("Jaxpr cache keys canonicalize nested graphs and retain special parameters", () => {
  const matrix = new ShapedArray([2, 2], DType.Float32, false);
  const vector = new ShapedArray([2], DType.Float32, false);
  const compileKey = (tolerance: number, reverseFields = false) => {
    const x = new Var(matrix);
    const values = new Var(vector);
    const vectors = new Var(matrix);
    const params = reverseFields
      ? { tolerance, maxSweeps: 10 }
      : { maxSweeps: 10, tolerance };
    const inner = new Jaxpr(
      [x],
      [new JaxprEqn(Primitive.JacobiEigh, [x], params, [values, vectors])],
      [values, vectors],
    );
    const outer = new Jaxpr(
      [x],
      [
        new JaxprEqn(
          Primitive.Jit,
          [x],
          { name: "inner", jaxpr: inner, numConsts: 0 },
          [values, vectors],
        ),
      ],
      [values, vectors],
    );
    return outer.getCacheKey();
  };
  expect(compileKey(1e-5)).toBe(compileKey(1e-5, true));
  expect(
    new Set([NaN, Infinity, -Infinity, 0, -0].map((x) => compileKey(x))).size,
  ).toBe(5);
});
