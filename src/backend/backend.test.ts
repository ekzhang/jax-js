import { beforeEach, expect, suite, test } from "vitest";

import {
  accessorGlobal,
  AluExp,
  AluOp,
  AluVar,
  DType,
  Kernel,
  Reduction,
} from "../alu";
import { devices, getBackend, init } from "../backend";
import { ShapeTracker, unravelAlu } from "../shape";
import { range } from "../utils";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);

  beforeEach(({ skip }) => {
    if (skipped) skip();
  });

  test("can run simple operations", async () => {
    const backend = getBackend(device);

    const shape = ShapeTracker.fromShape([3]);
    const a = backend.malloc(
      3 * 4,
      new Uint8Array(new Float32Array([1, 2, 3]).buffer),
    );
    const b = backend.malloc(
      3 * 4,
      new Uint8Array(new Float32Array([4, 5, 6]).buffer),
    );
    const c = backend.malloc(3 * 4);

    try {
      const gidx = AluVar.gidx;
      let arg1 = accessorGlobal(DType.Float32, 0, shape, [gidx]);
      let arg2 = accessorGlobal(DType.Float32, 1, shape.flip([true]), [gidx]);

      const exe1 = await backend.prepareKernel(
        new Kernel(2, 3, AluExp.mul(arg1, arg2)),
      );
      backend.dispatch(exe1, [a, b], [c]);

      const { buffer: buf } = await backend.read(c);
      expect(new Float32Array(buf)).toEqual(new Float32Array([6, 10, 12]));

      const exe2 = await backend.prepareKernel(
        new Kernel(2, 3, AluExp.add(arg1, arg2)),
      );
      backend.dispatch(exe2, [a, b], [c]);
      const { buffer: buf2 } = await backend.read(c);
      expect(new Float32Array(buf2)).toEqual(new Float32Array([7, 7, 7]));

      // Now try it with GlobalView.
      arg1 = AluExp.globalView(DType.Float32, 0, shape, [gidx]);
      arg2 = AluExp.globalView(DType.Float32, 1, shape.flip([true]), [gidx]);
      const exe3 = await backend.prepareKernel(
        new Kernel(2, 3, AluExp.mul(arg1, arg2)),
      );
      backend.dispatch(exe3, [a, b], [c]);
      const { buffer: buf3 } = await backend.read(c);
      expect(new Float32Array(buf3)).toEqual(new Float32Array([6, 10, 12]));
    } finally {
      backend.decRef(a);
      backend.decRef(b);
      backend.decRef(c);
    }
  });

  test("can create array from index", async () => {
    const backend = getBackend(device);
    const a = backend.malloc(200 * 4);
    try {
      const exe = await backend.prepareKernel(
        new Kernel(0, 200, AluExp.cast(DType.Float32, AluVar.gidx)),
      );
      backend.dispatch(exe, [], [a]);
      const { buffer: buf } = await backend.read(a);
      expect(new Float32Array(buf)).toEqual(new Float32Array(range(0, 200)));
    } finally {
      backend.decRef(a);
    }
  });

  test("can run synchronous operations", () => {
    const backend = getBackend(device);
    const a = backend.malloc(4 * 4);
    try {
      const exe = backend.prepareKernelSync(
        new Kernel(0, 4, AluExp.cast(DType.Float32, AluVar.gidx)),
      );
      backend.dispatch(exe, [], [a]);
      const buf = backend.readSync(a).buffer;
      expect(new Float32Array(buf)).toEqual(new Float32Array([0, 1, 2, 3]));
    } finally {
      backend.decRef(a);
    }
  });

  test("synchronously reads a buffer", () => {
    const backend = getBackend(device);
    const array = new Float32Array([1, 1, 2, 3, 5, 7]);
    const a = backend.malloc(6 * 4, new Uint8Array(array.buffer));
    try {
      let buf = backend.readSync(a).buffer;
      expect(new Float32Array(buf)).toEqual(array);
      buf = backend.readSync(a, 3 * 4, 2 * 4).buffer;
      expect(new Float32Array(buf)).toEqual(array.slice(3, 5));
    } finally {
      backend.decRef(a);
    }
  });

  test("asynchronously reads a buffer", async () => {
    const backend = getBackend(device);
    const array = new Float32Array([1, 1, 2, 3, 5, 7]);
    const a = backend.malloc(6 * 4, new Uint8Array(array.buffer));
    try {
      let buf = (await backend.read(a)).buffer;
      expect(new Float32Array(buf)).toEqual(array);
      buf = (await backend.read(a, 3 * 4, 2 * 4)).buffer;
      expect(new Float32Array(buf)).toEqual(array.slice(3, 5));
    } finally {
      backend.decRef(a);
    }
  });

  test("performs reduction", () => {
    const backend = getBackend(device);

    const array = new Float32Array([1, 1, 2, 3, 5, 7]);
    const a = backend.malloc(6 * 4, new Uint8Array(array.buffer));
    const output = backend.malloc(3 * 4);
    try {
      const st = ShapeTracker.fromShape([3, 2]);
      const exp = AluExp.globalView(DType.Float32, 0, st, [
        AluVar.gidx,
        AluVar.ridx,
      ]);
      let reduction = new Reduction(DType.Float32, AluOp.Add, 2);
      let kernel = new Kernel(1, 3, exp, reduction);

      const exe = backend.prepareKernelSync(kernel);
      backend.dispatch(exe, [a], [output]);

      const buf = backend.readSync(output).buffer;
      expect(new Float32Array(buf)).toEqual(new Float32Array([2, 5, 12]));

      // Try a reduction with fused +1.
      reduction = new Reduction(
        DType.Float32,
        AluOp.Add,
        2,
        AluExp.add(AluVar.acc(DType.Float32), AluExp.f32(1)),
      );
      kernel = new Kernel(1, 3, exp, reduction);
      const exe2 = backend.prepareKernelSync(kernel);
      backend.dispatch(exe2, [a], [output]);

      const buf2 = backend.readSync(output).buffer;
      expect(new Float32Array(buf2)).toEqual(new Float32Array([3, 6, 13]));
    } finally {
      backend.decRef(a);
      backend.decRef(output);
    }
  });

  test("bitwise operations", async () => {
    const backend = getBackend(device);

    const shape = ShapeTracker.fromShape([4]);
    const gidx = AluVar.gidx;

    // Test BitCombine (and, or, xor) on uint32
    const aData = new Uint32Array([0xff00ff00, 0x0f0f0f0f, 0xaaaaaaaa, 7]);
    const bData = new Uint32Array([0x00ff00ff, 0xf0f0f0f0, 0x55555555, 3]);
    const a = backend.malloc(4 * 4, new Uint8Array(aData.buffer));
    const b = backend.malloc(4 * 4, new Uint8Array(bData.buffer));
    const c = backend.malloc(4 * 4);

    try {
      const arg1 = accessorGlobal(DType.Uint32, 0, shape, [gidx]);
      const arg2 = accessorGlobal(DType.Uint32, 1, shape, [gidx]);

      // AND
      let exe = await backend.prepareKernel(
        new Kernel(2, 4, AluExp.bitCombine(arg1, arg2, "and")),
      );
      backend.dispatch(exe, [a, b], [c]);
      let buf = (await backend.read(c)).buffer;
      expect(new Uint32Array(buf)).toEqual(
        new Uint32Array([0x00000000, 0x00000000, 0x00000000, 3]),
      );

      // OR
      exe = await backend.prepareKernel(
        new Kernel(2, 4, AluExp.bitCombine(arg1, arg2, "or")),
      );
      backend.dispatch(exe, [a, b], [c]);
      buf = (await backend.read(c)).buffer;
      expect(new Uint32Array(buf)).toEqual(
        new Uint32Array([0xffffffff, 0xffffffff, 0xffffffff, 7]),
      );

      // XOR
      exe = await backend.prepareKernel(
        new Kernel(2, 4, AluExp.bitCombine(arg1, arg2, "xor")),
      );
      backend.dispatch(exe, [a, b], [c]);
      buf = (await backend.read(c)).buffer;
      expect(new Uint32Array(buf)).toEqual(
        new Uint32Array([0xffffffff, 0xffffffff, 0xffffffff, 4]),
      );

      // BitShift left
      const shiftData = new Uint32Array([1, 1, 1, 1]);
      const shiftAmt = new Uint32Array([0, 1, 8, 16]);
      const sa = backend.malloc(4 * 4, new Uint8Array(shiftData.buffer));
      const sb = backend.malloc(4 * 4, new Uint8Array(shiftAmt.buffer));

      const sarg1 = accessorGlobal(DType.Uint32, 0, shape, [gidx]);
      const sarg2 = accessorGlobal(DType.Uint32, 1, shape, [gidx]);

      exe = await backend.prepareKernel(
        new Kernel(2, 4, AluExp.bitShift(sarg1, sarg2, "shl")),
      );
      backend.dispatch(exe, [sa, sb], [c]);
      buf = (await backend.read(c)).buffer;
      expect(new Uint32Array(buf)).toEqual(new Uint32Array([1, 2, 256, 65536]));

      // BitShift right
      const rData = new Uint32Array([256, 65536, 0xffff0000, 8]);
      const ra = backend.malloc(4 * 4, new Uint8Array(rData.buffer));
      const rarg1 = accessorGlobal(DType.Uint32, 0, shape, [gidx]);

      exe = await backend.prepareKernel(
        new Kernel(2, 4, AluExp.bitShift(rarg1, sarg2, "shr")),
      );
      backend.dispatch(exe, [ra, sb], [c]);
      buf = (await backend.read(c)).buffer;
      expect(new Uint32Array(buf)).toEqual(
        new Uint32Array([256, 32768, 0x00ffff00, 0]),
      );

      backend.decRef(sa);
      backend.decRef(sb);
      backend.decRef(ra);
    } finally {
      backend.decRef(a);
      backend.decRef(b);
      backend.decRef(c);
    }
  });

  test("performs 64x64 matmul", async () => {
    const backend = getBackend(device);

    // This should trigger an optimization via Upcast/Unroll.
    const n = 64;
    const array = new Float32Array(n * n);
    for (let i = 0; i < array.length; ++i) array[i] = 1.0;

    const a = backend.malloc(n * n * 4, new Uint8Array(array.buffer));
    const b = backend.malloc(n * n * 4);
    try {
      // Calculate a^2, which should be all n.0 values.
      const st = ShapeTracker.fromShape([n, n]);
      const st1 = st.reshape([n, 1, n]).expand([n, n, n]);
      const st2 = st.permute([1, 0]).reshape([1, n, n]).expand([n, n, n]);
      const indices = [...unravelAlu([n, n], AluVar.gidx), AluVar.ridx];
      const exp = AluExp.mul(
        AluExp.globalView(DType.Float32, 0, st1, indices),
        AluExp.globalView(DType.Float32, 0, st2, indices),
      );
      const reduction = new Reduction(DType.Float32, AluOp.Add, n);
      const kernel = new Kernel(1, n * n, exp, reduction);

      const exe = await backend.prepareKernel(kernel);
      backend.dispatch(exe, [a], [b]);

      const result = new Float32Array((await backend.read(b)).buffer);
      for (let i = 0; i < result.length; i++) {
        expect(result[i]).toBeCloseTo(n);
      }
    } finally {
      backend.decRef(a);
      backend.decRef(b);
    }
  });

  test("pointwise SIMD-eligible kernel with size not divisible by 4", async () => {
    const backend = getBackend(device);
    const data = new Float32Array([1, 2, 3, 4, 5, 6, 7]);
    const a = backend.malloc(7 * 4, new Uint8Array(data.buffer));
    const out = backend.malloc(7 * 4);

    try {
      const shape = ShapeTracker.fromShape([7]);
      const gidx = AluVar.gidx;
      const arg = accessorGlobal(DType.Float32, 0, shape, [gidx]);
      const kernel = new Kernel(
        1,
        7,
        AluExp.mul(AluExp.add(arg, AluExp.f32(2)), AluExp.f32(3)),
      );
      const exe = await backend.prepareKernel(kernel);
      backend.dispatch(exe, [a], [out]);

      const { buffer } = await backend.read(out);
      expect(new Float32Array(buffer)).toEqual(
        new Float32Array([9, 12, 15, 18, 21, 24, 27]),
      );
    } finally {
      backend.decRef(a);
      backend.decRef(out);
    }
  });

  test("pointwise SIMD-eligible kernel with two inputs", async () => {
    const backend = getBackend(device);
    const dataA = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const dataB = new Float32Array([2, 3, 4, 5, 6, 7, 8, 9]);
    const a = backend.malloc(8 * 4, new Uint8Array(dataA.buffer));
    const b = backend.malloc(8 * 4, new Uint8Array(dataB.buffer));
    const out = backend.malloc(8 * 4);

    try {
      const shape = ShapeTracker.fromShape([8]);
      const gidx = AluVar.gidx;
      const arg1 = accessorGlobal(DType.Float32, 0, shape, [gidx]);
      const arg2 = accessorGlobal(DType.Float32, 1, shape, [gidx]);
      const kernel = new Kernel(2, 8, AluExp.mul(arg1, arg2));
      const exe = await backend.prepareKernel(kernel);
      backend.dispatch(exe, [a, b], [out]);

      const { buffer } = await backend.read(out);
      expect(new Float32Array(buffer)).toEqual(
        new Float32Array([2, 6, 12, 20, 30, 42, 56, 72]),
      );
    } finally {
      backend.decRef(a);
      backend.decRef(b);
      backend.decRef(out);
    }
  });
});
