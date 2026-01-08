// Make sure .ref move semantics are working correctly, and that arrays are
// freed at the right time.

import { grad, numpy as np } from "@jax-js/jax";
import { expect, suite, test } from "vitest";

suite("refcount through grad", () => {
  test("add and sum", () => {
    const f = (x: np.Array) => x.ref.add(x).sum();
    const df = grad(f);

    const x = np.array([1, 2, 3, 4]);
    expect(df(x).js()).toEqual([2, 2, 2, 2]);
    expect(() => x.dispose()).toThrowError(ReferenceError);
    expect(() => df(x).js()).toThrowError(ReferenceError);
  });

  test("multiply and sum", () => {
    const f = (x: np.Array) => x.ref.mul(x).sum();
    const df = grad(f);

    const x = np.array([1, 2, 3, 4]);
    expect(df(x).js()).toEqual([2, 4, 6, 8]);
    expect(() => x.dispose()).toThrowError(ReferenceError);
    expect(() => df(x).js()).toThrowError(ReferenceError);
  });
});

suite("refCount property", () => {
  test("initial refCount is 1", () => {
    const x = np.array([1, 2, 3]);
    expect(x.refCount).toBe(1);
    x.dispose();
  });

  test("refCount increments after .ref", () => {
    const x = np.array([1, 2, 3]);
    expect(x.refCount).toBe(1);
    const y = x.ref;
    expect(x.refCount).toBe(2);
    expect(y.refCount).toBe(2); // same array
    x.dispose();
    y.dispose();
  });

  test("refCount is 0 after final dispose", () => {
    const x = np.array([1, 2, 3]);
    expect(x.refCount).toBe(1);
    x.dispose();
    expect(x.refCount).toBe(0);
  });

  test("refCount is readable on disposed arrays", () => {
    const x = np.array([1, 2, 3]);
    x.dispose();
    // Should not throw - refCount is readable for debugging
    expect(x.refCount).toBe(0);
  });
});
