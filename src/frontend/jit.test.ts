import { expect, test, vi } from "vitest";

import { AluExp } from "../alu";
import { defaultDevice, getBackend, init } from "../backend";
import { jit, makeJaxpr, numpy as np } from "../index";
import { jitCompile } from "./jit";

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

/**
 * Number of distinct expression nodes that jit fusion hands to each Kernel,
 * before Kernel's simplify() dedupes them.
 */
function fusedNodeCounts(f: (...args: any[]) => any, ...args: any[]) {
  const { jaxpr } = makeJaxpr(f)(...args);
  const simplify = vi.spyOn(AluExp.prototype, "simplify");
  try {
    jitCompile(getBackend("cpu"), jaxpr.jaxpr);
    // Kernel calls exp.simplify() with no cache; recursive calls pass one.
    return (simplify.mock.contexts as AluExp[])
      .filter((_, i) => simplify.mock.calls[i][0] === undefined)
      .map((root) => {
        const nodes = new Set<AluExp>();
        root.fold((node) => void nodes.add(node));
        return nodes.size;
      });
  } finally {
    simplify.mockRestore();
  }
}

test("jit keeps operands shared when an op reuses them in swapped order", () => {
  // Pin to cpu: other tests leave the default device set, and WebGL inlines
  // every use of a shared value into the shader.
  defaultDevice("cpu");

  // Each step uses y and x as (y, x) and as (x, y). Fusion used to copy y's
  // expression once per order, doubling the kernel with every step.
  const depth = 16;
  const f = (x: np.Array, y: np.Array) => {
    for (let i = 0; i < depth; i++) {
      y = np.add(
        np.sin(np.subtract(y.ref, x.ref)),
        np.cos(np.subtract(x.ref, y)),
      );
    }
    x.dispose();
    return y;
  };

  const counts = fusedNodeCounts(f, np.zeros([4]), np.zeros([4]));
  expect(counts.length).toBeGreaterThan(0);
  expect(Math.max(...counts)).toBeLessThan(20 * depth);

  const x = [0.5, -1, 2, 0];
  const out = jit(f)(np.array(x), np.array([0, 1, -0.5, 3])).js() as number[];
  const expected = [0, 1, -0.5, 3].map((y0, i) => {
    let y = Math.fround(y0);
    for (let j = 0; j < depth; j++)
      y = Math.fround(Math.sin(y - x[i]) + Math.cos(x[i] - y));
    return y;
  });
  for (let i = 0; i < x.length; i++) expect(out[i]).toBeCloseTo(expected[i], 4);
});
