import { expect, suite, test } from "vitest";
import { jvp, numpy as np } from "jax-js";

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
    expect(np.sin(x)).toBeClose(0.141120001);
    expect(deriv(np.sin)(x)).toBeClose(-0.989992499);
    expect(deriv(deriv(np.sin))(x)).toBeClose(-0.141120001);
    expect(deriv(deriv(deriv(np.sin)))(x)).toBeClose(0.989992499);
  });

  test("can take jvp of pytrees", () => {
    const result = jvp(
      (x: { a: np.Array; b: np.Array }) => x.a.mul(x.a).add(x.b),
      [{ a: 1, b: 2 }],
      [{ a: 1, b: 0 }]
    );
    expect(result[0]).toBeClose(3);
    expect(result[1]).toBeClose(2);
  });
});
