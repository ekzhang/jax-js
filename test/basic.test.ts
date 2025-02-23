import { expect, suite, test } from "vitest";
import { numpy as np, jvp, jacfwd, vmap } from "@jax-js/core";

// test("has webgpu", async () => {
//   const adapter = await navigator.gpu?.requestAdapter();
//   const device = await adapter?.requestDevice();
//   if (!adapter || !device) {
//     throw new Error("No adapter or device");
//   }
//   console.log(device.adapterInfo.architecture);
//   console.log(device.adapterInfo.vendor);
//   console.log(adapter.limits.maxVertexBufferArrayStride);
// });

/** Take the derivative of a simple function. */
function deriv(f: (x: np.Array) => np.Array): (x: np.ArrayLike) => np.Array {
  return (x) => {
    const [_y, dy] = jvp(f, [x], [1.0]);
    return dy;
  };
}

test("can create array", () => {
  const x = np.array([1, 2, 3]);
  expect(x.js()).toEqual([1, 2, 3]);
});

suite("jax.jvp()", () => {
  test("can take scalar derivatives", () => {
    const x = 3.0;
    expect(np.sin(x)).toBeAllclose(0.141120001);
    expect(deriv(np.sin)(x)).toBeAllclose(-0.989992499);
    expect(deriv(deriv(np.sin))(x)).toBeAllclose(-0.141120001);
    expect(deriv(deriv(deriv(np.sin)))(x)).toBeAllclose(0.989992499);
  });

  test("can take jvp of pytrees", () => {
    const result = jvp(
      (x: { a: np.Array; b: np.Array }) => x.a.mul(x.a).add(x.b),
      [{ a: 1, b: 2 }],
      [{ a: 1, b: 0 }],
    );
    expect(result[0]).toBeAllclose(3);
    expect(result[1]).toBeAllclose(2);
  });

  test("works for vector to scalar functions", () => {
    const f = (x: np.Array) => np.reduceSum(x);
    const x = np.array([1, 2, 3]);
    expect(f(x)).toBeAllclose(6);
    expect(jvp(f, [x], [np.array([1, 1, 1])])[1]).toBeAllclose(3);
  });
});

suite("jax.vmap()", () => {
  test("vectorizes a function over a single axis", () => {
    // f multiplies its input by 2.
    const f = (x: np.Array) => x.mul(2);
    // vmap with inAxes=0 means that the function is applied over the first axis.
    const batchedF = vmap(f, [0]);
    const x = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(batchedF(x).js()).toEqual([
      [2, 4, 6],
      [8, 10, 12],
    ]);
  });

  test("vectorizes a function with multiple arguments", () => {
    // f adds its two array arguments.
    const f = (x: np.Array, y: np.Array) => x.add(y);
    // Both arguments are batched over axis 0.
    const batchedF = vmap(f, [0, 0]);
    const x = np.array([
      [1, 2],
      [3, 4],
    ]);
    const y = np.array([
      [10, 20],
      [30, 40],
    ]);
    expect(batchedF(x, y).js()).toEqual([
      [11, 22],
      [33, 44],
    ]);
  });

  test("vectorizes with a static argument", () => {
    // f multiplies an array by a scalar.
    const f = (x: np.Array, y: number) => x.mul(y);
    // Here we want to batch only over the first argument.
    const batchedF = vmap(f, [0, null]);
    const x = np.array([
      [1, 2],
      [3, 4],
    ]);
    const y = 10;
    expect(batchedF(x, y).js()).toEqual([
      [10, 20],
      [30, 40],
    ]);
  });

  test("vectorizes a function returning a pytree", () => {
    // f returns an object whose leaves are computed from x.
    const f = (x: np.Array) => ({
      double: x.mul(2),
      square: x.mul(x),
    });
    const batchedF = vmap(f, [0]);
    const x = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const result = batchedF(x);
    expect(result.double.js()).toEqual([
      [2, 4, 6],
      [8, 10, 12],
    ]);
    expect(result.square.js()).toEqual([
      [1, 4, 9],
      [16, 25, 36],
    ]);
  });

  test("vectorizes over pytrees inputs", () => {
    // f accepts an object with two array fields and adds them.
    const f = (x: { a: np.Array; b: np.Array }) => x.a.add(x.b);
    const batchedF = vmap(f, [{ a: 0, b: 0 }]);
    const x = {
      a: np.array([
        [1, 2],
        [3, 4],
      ]),
      b: np.array([
        [10, 20],
        [30, 40],
      ]),
    };
    expect(batchedF(x).js()).toEqual([
      [11, 22],
      [33, 44],
    ]);
  });
});

suite("jax.jacfwd()", () => {
  test("computes jacobian of 3d square", () => {
    const f = (x: np.Array) => x.mul(x);
    const x = np.array([1, 2, 3]);
    const j = jacfwd(f, x);
    expect(j).toBeAllclose(
      np.array([
        [2, 0, 0],
        [0, 4, 0],
        [0, 0, 6],
      ]),
    );
  });
});
