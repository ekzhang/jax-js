import { describe, expect, test } from "vitest";
import { getBackend, accessorAlu } from "../backend";
import { ShapeTracker } from "../shape";
import { AluExp, DType } from "../alu";
import { range } from "../utils";

describe.each(["cpu", "webgpu"])("Backend '%s'", (backendName) => {
  test("can run simple operations", async ({ skip }) => {
    const backend = await getBackend(backendName);
    if (!backend) return skip();

    const shape = ShapeTracker.fromShape([3]);
    const a = backend.malloc(3 * 4, new Float32Array([1, 2, 3]).buffer);
    const b = backend.malloc(3 * 4, new Float32Array([4, 5, 6]).buffer);
    const c = backend.malloc(3 * 4);

    try {
      const gidx = AluExp.special(DType.Int32, "gidx", 3);
      const arg1 = accessorAlu(0, shape, gidx);
      const arg2 = accessorAlu(1, shape.flip([true]), gidx);

      const exe1 = await backend.prepare(2, AluExp.mul(arg1, arg2));
      backend.dispatch(exe1, [a, b], [c]);

      const buf = await backend.read(c);
      expect(new Float32Array(buf)).toEqual(new Float32Array([6, 10, 12]));

      const exe2 = await backend.prepare(2, AluExp.add(arg1, arg2));
      backend.dispatch(exe2, [a, b], [c]);
      const buf2 = await backend.read(c);
      expect(new Float32Array(buf2)).toEqual(new Float32Array([7, 7, 7]));
    } finally {
      backend.decRef(a);
      backend.decRef(b);
      backend.decRef(c);
    }
  });

  test("can create array from index", async ({ skip }) => {
    const backend = await getBackend(backendName);
    if (!backend) return skip();

    const a = backend.malloc(200 * 4);
    try {
      const gidx = AluExp.special(DType.Int32, "gidx", 200);
      const exe = await backend.prepare(0, AluExp.cast(DType.Float32, gidx));
      backend.dispatch(exe, [], [a]);
      const buf = await backend.read(a);
      expect(new Float32Array(buf)).toEqual(new Float32Array(range(0, 200)));
    } finally {
      backend.decRef(a);
    }
  });

  test("can run synchronous operations", async ({ skip }) => {
    const backend = await getBackend(backendName);
    if (!backend) return skip();

    const a = backend.malloc(4 * 4);
    try {
      const gidx = AluExp.special(DType.Int32, "gidx", 4);
      const exe = backend.prepareSync(0, AluExp.cast(DType.Float32, gidx));
      backend.dispatch(exe, [], [a]);
      const buf = backend.readSync(a);
      expect(new Float32Array(buf)).toEqual(new Float32Array([0, 1, 2, 3]));
    } finally {
      backend.decRef(a);
    }
  });

  test("synchronously reads a buffer", async ({ skip }) => {
    const backend = await getBackend(backendName);
    if (!backend) return skip();

    const array = new Float32Array([1, 1, 2, 3, 5, 7]);
    const a = backend.malloc(6 * 4, array.buffer);
    try {
      let buf = backend.readSync(a);
      expect(new Float32Array(buf)).toEqual(array);
      buf = backend.readSync(a, 3 * 4, 2 * 4);
      expect(new Float32Array(buf)).toEqual(array.slice(3, 5));
    } finally {
      backend.decRef(a);
    }
  });

  test("asynchronously reads a buffer", async ({ skip }) => {
    const backend = await getBackend(backendName);
    if (!backend) return skip();

    const array = new Float32Array([1, 1, 2, 3, 5, 7]);
    const a = backend.malloc(6 * 4, array.buffer);
    try {
      let buf = await backend.read(a);
      expect(new Float32Array(buf)).toEqual(array);
      buf = await backend.read(a, 3 * 4, 2 * 4);
      expect(new Float32Array(buf)).toEqual(array.slice(3, 5));
    } finally {
      backend.decRef(a);
    }
  });
});
