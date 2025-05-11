import { grad, init, jvp, numpy as np } from "@jax-js/core";
import { expect, suite, test } from "vitest";

await init("cpu");

suite("jax.numpy.eye()", () => {
  test("computes a square matrix", () => {
    const x = np.eye(3);
    expect(x).toBeAllclose([
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ]);
  });

  test("computes a rectangular matrix", () => {
    const x = np.eye(2, 3);
    expect(x).toBeAllclose([
      [1, 0, 0],
      [0, 1, 0],
    ]);
  });

  test("can be multiplied", () => {
    const x = np.eye(3, 5).mul(-42);
    expect(x.sum()).toBeAllclose(-126);
    expect(x).toBeAllclose([
      [-42, 0, 0, 0, 0],
      [0, -42, 0, 0, 0],
      [0, 0, -42, 0, 0],
    ]);
  });
});

suite("jax.numpy.where()", () => {
  test("computes where", () => {
    const x = np.array([1, 2, 3]);
    const y = np.array([4, 5, 6]);
    const z = np.array([true, false, true]);
    const result = np.where(z, x, y);
    expect(result.js()).toEqual([1, 5, 3]);
  });

  test("works with jvp", () => {
    const x = np.array([1, 2, 3]);
    const y = np.array([4, 5, 6]);
    const z = np.array([true, false, true]);
    const result = jvp(
      (x: np.Array, y: np.Array) => np.where(z, x, y),
      [x, y],
      [np.array([1, 1, 1]), np.zeros([3])],
    );
    expect(result[0].js()).toEqual([1, 5, 3]);
    expect(result[1].js()).toEqual([1, 0, 1]);
  });

  test("works with grad reverse-mode", () => {
    const x = np.array([1, 2, 3]);
    const y = np.array([4, 5, 6]);
    const z = np.array([true, false, true]);
    const f = ({ x, y }: { x: np.Array; y: np.Array }) =>
      np.where(z, x, y).sum();
    const grads = grad(f)({ x, y });
    expect(grads.x.js()).toEqual([1, 0, 1]);
    expect(grads.y.js()).toEqual([0, 1, 0]);
  });
});
