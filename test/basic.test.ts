import { jacfwd, jacrev, jvp, numpy as np, vmap } from "@jax-js/jax";
import { expect, suite, test } from "vitest";

test("can create array", () => {
  const x = np.array([1, 2, 3]);
  expect(x.js()).toEqual([1, 2, 3]);
});

suite("jax.jvp()", () => {
  test("can take scalar derivatives", () => {
    /** Take the derivative of a simple function. */
    const deriv: (
      f: (x: np.Array) => np.Array,
    ) => (x: np.ArrayLike) => np.Array = (f) => (x) => {
      const [_y, dy] = jvp(f, [x], [1.0]);
      return dy;
    };
    const x = 3.0;
    expect(np.sin(x)).toBeAllclose(0.141120001);
    expect(deriv(np.sin)(x)).toBeAllclose(-0.989992499);
    expect(deriv(deriv(np.sin))(x)).toBeAllclose(-0.141120001);
    expect(deriv(deriv(deriv(np.sin)))(x)).toBeAllclose(0.989992499);
  });

  test("can take jvp of pytrees", () => {
    const result = jvp(
      (x: { a: np.Array; b: np.Array }) => x.a.ref.mul(x.a).add(x.b),
      [{ a: 1, b: 2 }],
      [{ a: 1, b: 0 }],
    );
    expect(result[0]).toBeAllclose(3);
    expect(result[1]).toBeAllclose(2);
  });

  test("works for vector to scalar functions", () => {
    const f = (x: np.Array) => np.sum(x);
    const x = np.array([1, 2, 3]);
    expect(f(x.ref)).toBeAllclose(6);
    expect(jvp(f, [x], [np.array([1, 1, 1])])[1]).toBeAllclose(3);
  });

  test("can compute jvp of products", () => {
    const x = np.array([1, 2, 3, 4]);

    const jvpProd = jvp(
      (x: np.Array) => np.prod(x),
      [x],
      [np.array([1, 10, 100, 1000])],
    );
    expect(jvpProd[0]).toBeAllclose(24);
    // 1 * 2*3*4 + 10 * 1*3*4 + 100 * 1*2*4 + 1000 * 1*2*3
    expect(jvpProd[1]).toBeAllclose(6944);
  });

  test("can have jvp of min", () => {
    const x = np.array([1, 2, 1, 4]);
    const [pmin, jmin] = jvp(
      (x: np.Array) => np.min(x),
      [x],
      [np.array([0, 10, 5, 1000])],
    );
    expect(pmin).toBeAllclose(1);
    expect(jmin).toBeAllclose(2.5); // (0+5)/2
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
    // Batch x over axis 0 and y over axis 1.
    const batchedF = vmap(f, [0, 1]);
    const x = np.array([
      [1, 2],
      [3, 4],
    ]);
    const y = np.array([
      [10, 20],
      [30, 40],
    ]);
    expect(batchedF(x, y).js()).toEqual([
      [11, 32],
      [23, 44],
    ]);
  });

  test("vectorizes with a static argument", () => {
    const f = (x: np.Array, y: np.Array) => x.mul(y);
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
      double: x.ref.mul(2),
      square: x.ref.mul(x.ref),
      sum: x.sum(),
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
    expect(result.sum.js()).toEqual([6, 15]);
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

  test("can take a single axis number", () => {
    const a = np.array([
      [1, 2],
      [3, 4],
    ]);
    const b = np.array([
      [5, 6],
      [7, 8],
    ]);
    expect(vmap(np.dot, 0)(a.ref, b.ref).js()).toEqual([17, 53]);
    expect(vmap(np.dot, 1)(a, b).js()).toEqual([26, 44]);
  });

  test("can vectorize over iteration and slice", () => {
    const f = (x: np.Array) => [...x];
    const batchedF = vmap(f, 0);
    const x = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const [a, b, c] = batchedF(x);
    expect(a.js()).toEqual([1, 4]);
    expect(b.js()).toEqual([2, 5]);
    expect(c.js()).toEqual([3, 6]);
  });
});

suite("jax.jacfwd()", () => {
  test("computes jacobian of 3d square", () => {
    const f = (x: np.Array) => x.ref.mul(x);
    const x = np.array([1, 2, 3]);
    const j = jacfwd(f)(x);
    expect(j).toBeAllclose(
      np.array([
        [2, 0, 0],
        [0, 4, 0],
        [0, 0, 6],
      ]),
    );
  });

  test("equals jacrev() output", () => {
    const f = (x: np.Array) => np.sin(x.ref).add(np.cos(x));
    const x = np.array([1, 2, 3]);
    const jFwd = jacfwd(f)(x.ref);
    const jRev = jacrev(f)(x);
    expect(jFwd).toBeAllclose(jRev);
  });
});
