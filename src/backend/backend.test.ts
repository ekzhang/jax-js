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

  if (device === "wasm") {
    // Make sure tests are configured so wasm runs with SharedArrayBuffer, we
    // want to test the multithreaded backend.
    test("SharedArrayBuffer is available", () => {
      if (typeof window !== "undefined") {
        expect(window.crossOriginIsolated).toBe(true);
      }
      expect(typeof SharedArrayBuffer).toBe("function");
    });
  }

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

  test("matmul with output size that misaligns parallel chunks", async () => {
    // Regression test: a matmul with M=131, K=4, N=4 produces 524 output
    // elements. With most worker counts, ceil(524/workers) is not a multiple
    // of 4, so some workers get a `begin` that's not 4-aligned. This would
    // cause SIMD-over-gidx to cross tile boundaries if dispatch isn't aligned.
    //
    // Uses distinct values (not all-ones) so wrong-index reads produce
    // detectably wrong results.
    const backend = getBackend(device);
    const M = 131, K = 4, N = 4;

    // A is [M, K] with A[i,j] = i + j*0.01 (distinct per element).
    const aData = new Float32Array(M * K);
    for (let i = 0; i < M; i++)
      for (let j = 0; j < K; j++)
        aData[i * K + j] = i + j * 0.01;

    // B is [K, N] identity matrix.
    const bData = new Float32Array(K * N);
    for (let i = 0; i < K; i++) bData[i * N + i] = 1.0;

    // C = A @ I = A, so C[i, j] = A[i, j].
    const a = backend.malloc(M * K * 4, new Uint8Array(aData.buffer));
    const b = backend.malloc(K * N * 4, new Uint8Array(bData.buffer));
    const c = backend.malloc(M * N * 4);
    try {
      const stA = ShapeTracker.fromShape([M, K]).reshape([M, 1, K]).expand([M, N, K]);
      const stB = ShapeTracker.fromShape([K, N]).permute([1, 0]).reshape([1, N, K]).expand([M, N, K]);
      const indices = [...unravelAlu([M, N], AluVar.gidx), AluVar.ridx];
      const exp = AluExp.mul(
        AluExp.globalView(DType.Float32, 0, stA, indices),
        AluExp.globalView(DType.Float32, 1, stB, indices),
      );
      const reduction = new Reduction(DType.Float32, AluOp.Add, K);
      const kernel = new Kernel(2, M * N, exp, reduction);

      const exe = await backend.prepareKernel(kernel);
      backend.dispatch(exe, [a, b], [c]);

      const result = new Float32Array((await backend.read(c)).buffer);
      for (let i = 0; i < M; i++) {
        for (let j = 0; j < N; j++) {
          expect(result[i * N + j]).toBeCloseTo(aData[i * K + j], 4);
        }
      }
    } finally {
      backend.decRef(a);
      backend.decRef(b);
      backend.decRef(c);
    }
  });

  test("pointwise f32 add+mul with 7 elements", async () => {
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

  test("pointwise f32 mul with two inputs", async () => {
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

  test("pointwise f32 add on flipped 8-element array", async () => {
    const backend = getBackend(device);
    const data = new Float32Array([10, 20, 30, 40, 50, 60, 70, 80]);
    const a = backend.malloc(8 * 4, new Uint8Array(data.buffer));
    const out = backend.malloc(8 * 4);

    try {
      // Flip reverses the array — stride becomes -1, non-contiguous.
      // On WASM SIMD, this exercises the gather path for pointwise kernels.
      const shape = ShapeTracker.fromShape([8]).flip([true]);
      const gidx = AluVar.gidx;
      const arg = accessorGlobal(DType.Float32, 0, shape, [gidx]);
      const kernel = new Kernel(1, 8, AluExp.add(arg, AluExp.f32(2)));

      const exe = await backend.prepareKernel(kernel);
      backend.dispatch(exe, [a], [out]);

      const { buffer } = await backend.read(out);
      // Flipped: gidx=0 reads element 7 (80), gidx=1 reads element 6 (70), etc.
      // Each +2: [82, 72, 62, 52, 42, 32, 22, 12]
      expect(new Float32Array(buffer)).toEqual(
        new Float32Array([82, 72, 62, 52, 42, 32, 22, 12]),
      );
    } finally {
      backend.decRef(a);
      backend.decRef(out);
    }
  });

  test("reduction sum on [3,8] f32 array", () => {
    const backend = getBackend(device);
    // [3, 8] array: sum each row of 8 elements → 3 outputs
    const data = new Float32Array([
      1, 2, 3, 4, 5, 6, 7, 8,
      10, 20, 30, 40, 50, 60, 70, 80,
      100, 200, 300, 400, 500, 600, 700, 800,
    ]);
    const a = backend.malloc(24 * 4, new Uint8Array(data.buffer));
    const output = backend.malloc(3 * 4);
    try {
      const st = ShapeTracker.fromShape([3, 8]);
      const exp = AluExp.globalView(DType.Float32, 0, st, [
        AluVar.gidx,
        AluVar.ridx,
      ]);
      const kernel = new Kernel(
        1, 3, exp,
        new Reduction(DType.Float32, AluOp.Add, 8),
      );
      const exe = backend.prepareKernelSync(kernel);
      backend.dispatch(exe, [a], [output]);
      const buf = backend.readSync(output).buffer;
      expect(new Float32Array(buf)).toEqual(new Float32Array([36, 360, 3600]));
    } finally {
      backend.decRef(a);
      backend.decRef(output);
    }
  });

  test("reduction sum on permuted [8,4] f32 array", () => {
    const backend = getBackend(device);
    // Transposed [8, 4] array: permute gives shape [4, 8] with strides [1, 4].
    // gidx has stride 1 (contiguous), ridx has stride 4 (non-contiguous).
    // Output size = 4, so SIMD-over-gidx is eligible.
    const data = new Float32Array([
      // Stored in memory as the original [8, 4] row-major layout:
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16,
      17, 18, 19, 20,
      21, 22, 23, 24,
      25, 26, 27, 28,
      29, 30, 31, 32,
    ]);
    const a = backend.malloc(32 * 4, new Uint8Array(data.buffer));
    const output = backend.malloc(4 * 4);
    try {
      // permute([1, 0]) gives shape [4, 8] with strides [1, 4].
      // With indices [gidx, ridx]: gidx has stride 1, ridx has stride 4.
      // Stepping ridx by 1 gives stride-4 access — non-contiguous.
      const st = ShapeTracker.fromShape([8, 4]).permute([1, 0]);
      const exp = AluExp.globalView(DType.Float32, 0, st, [
        AluVar.gidx,
        AluVar.ridx,
      ]);
      const kernel = new Kernel(
        1, 4, exp,
        new Reduction(DType.Float32, AluOp.Add, 8),
      );
      const exe = backend.prepareKernelSync(kernel);
      backend.dispatch(exe, [a], [output]);
      const buf = backend.readSync(output).buffer;
      // Each output[gidx] sums column gidx across 8 rows.
      // gidx=0: 1+5+9+13+17+21+25+29 = 120
      // gidx=1: 2+6+10+14+18+22+26+30 = 128
      // gidx=2: 3+7+11+15+19+23+27+31 = 136
      // gidx=3: 4+8+12+16+20+24+28+32 = 144
      expect(new Float32Array(buf)).toEqual(
        new Float32Array([120, 128, 136, 144]),
      );
    } finally {
      backend.decRef(a);
      backend.decRef(output);
    }
  });

  test("pointwise i32 add on 8-element array", async () => {
    const backend = getBackend(device);
    const data = new Int32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const a = backend.malloc(8 * 4, new Uint8Array(data.buffer));
    const out = backend.malloc(8 * 4);

    try {
      const shape = ShapeTracker.fromShape([8]);
      const gidx = AluVar.gidx;
      const arg = accessorGlobal(DType.Int32, 0, shape, [gidx]);
      const kernel = new Kernel(1, 8, AluExp.add(arg, AluExp.i32(10)));
      const exe = await backend.prepareKernel(kernel);
      backend.dispatch(exe, [a], [out]);

      const { buffer } = await backend.read(out);
      expect(new Int32Array(buffer)).toEqual(
        new Int32Array([11, 12, 13, 14, 15, 16, 17, 18]),
      );
    } finally {
      backend.decRef(a);
      backend.decRef(out);
    }
  });

  test("reduction sum on [3,7] i32 array", () => {
    const backend = getBackend(device);
    // [3, 7] i32 array: sum each row of 7 elements → 3 outputs.
    // Output size 3 < 4, so this runs the scalar fallback path.
    const data = new Int32Array([
      1, 2, 3, 4, 5, 6, 7,
      10, 20, 30, 40, 50, 60, 70,
      100, 200, 300, 400, 500, 600, 700,
    ]);
    const a = backend.malloc(21 * 4, new Uint8Array(data.buffer));
    const output = backend.malloc(3 * 4);
    try {
      const st = ShapeTracker.fromShape([3, 7]);
      const exp = AluExp.globalView(DType.Int32, 0, st, [
        AluVar.gidx,
        AluVar.ridx,
      ]);
      const kernel = new Kernel(
        1, 3, exp,
        new Reduction(DType.Int32, AluOp.Add, 7),
      );
      const exe = backend.prepareKernelSync(kernel);
      backend.dispatch(exe, [a], [output]);
      const buf = backend.readSync(output).buffer;
      expect(new Int32Array(buf)).toEqual(new Int32Array([28, 280, 2800]));
    } finally {
      backend.decRef(a);
      backend.decRef(output);
    }
  });

  test("reduction min on [4,8] f32 array", () => {
    const backend = getBackend(device);
    // Each row has a different minimum at a different position.
    const data = new Float32Array([
      9, 7, 3, 5, 8, 6, 4, 2,   // min = 2
      10, 1, 20, 30, 40, 50, 60, 70, // min = 1
      5, 5, 5, 5, 5, 5, 5, 0.5, // min = 0.5
      99, 98, 97, 96, 95, 94, 93, 92, // min = 92
    ]);
    const a = backend.malloc(32 * 4, new Uint8Array(data.buffer));
    const output = backend.malloc(4 * 4);
    try {
      const st = ShapeTracker.fromShape([4, 8]);
      const exp = AluExp.globalView(DType.Float32, 0, st, [
        AluVar.gidx,
        AluVar.ridx,
      ]);
      const kernel = new Kernel(
        1, 4, exp,
        new Reduction(DType.Float32, AluOp.Min, 8),
      );
      const exe = backend.prepareKernelSync(kernel);
      backend.dispatch(exe, [a], [output]);
      const buf = backend.readSync(output).buffer;
      expect(new Float32Array(buf)).toEqual(
        new Float32Array([2, 1, 0.5, 92]),
      );
    } finally {
      backend.decRef(a);
      backend.decRef(output);
    }
  });

  test("reduction max on [4,8] f32 array", () => {
    const backend = getBackend(device);
    const data = new Float32Array([
      1, 2, 3, 4, 5, 6, 7, 100, // max = 100
      -1, -2, -3, -4, -5, -6, -7, -0.1, // max = -0.1
      0, 0, 0, 0, 0, 0, 0, 0.001, // max = 0.001
      50, 99, 50, 50, 50, 50, 50, 50,  // max = 99
    ]);
    const a = backend.malloc(32 * 4, new Uint8Array(data.buffer));
    const output = backend.malloc(4 * 4);
    try {
      const st = ShapeTracker.fromShape([4, 8]);
      const exp = AluExp.globalView(DType.Float32, 0, st, [
        AluVar.gidx,
        AluVar.ridx,
      ]);
      const kernel = new Kernel(
        1, 4, exp,
        new Reduction(DType.Float32, AluOp.Max, 8),
      );
      const exe = backend.prepareKernelSync(kernel);
      backend.dispatch(exe, [a], [output]);
      const buf = backend.readSync(output).buffer;
      expect(new Float32Array(buf)).toEqual(
        new Float32Array([100, -0.1, 0.001, 99]),
      );
    } finally {
      backend.decRef(a);
      backend.decRef(output);
    }
  });

  test("pointwise f32 add with broadcast input", async () => {
    const backend = getBackend(device);
    // A: 4 elements, each broadcast across 8 columns.
    const dataA = new Float32Array([10, 20, 30, 40]);
    // B: 32 distinct elements.
    const dataB = new Float32Array(range(1, 33));
    const a = backend.malloc(4 * 4, new Uint8Array(dataA.buffer));
    const b = backend.malloc(32 * 4, new Uint8Array(dataB.buffer));
    const out = backend.malloc(32 * 4);

    try {
      const indices = [...unravelAlu([4, 8], AluVar.gidx)];
      // A: [4] → [4,1] → expand [4,8] — dim 1 has stride 0 (broadcast).
      const stA = ShapeTracker.fromShape([4]).reshape([4, 1]).expand([4, 8]);
      const stB = ShapeTracker.fromShape([4, 8]);

      const kernel = new Kernel(
        2,
        32,
        AluExp.add(
          AluExp.globalView(DType.Float32, 0, stA, indices),
          AluExp.globalView(DType.Float32, 1, stB, indices),
        ),
      );
      const exe = await backend.prepareKernel(kernel);
      backend.dispatch(exe, [a, b], [out]);

      const { buffer } = await backend.read(out);
      // Row i: A[i] + B[i*8+0..7]
      expect(new Float32Array(buffer)).toEqual(
        new Float32Array([
          11, 12, 13, 14, 15, 16, 17, 18, // 10 + [1..8]
          29, 30, 31, 32, 33, 34, 35, 36, // 20 + [9..16]
          47, 48, 49, 50, 51, 52, 53, 54, // 30 + [17..24]
          65, 66, 67, 68, 69, 70, 71, 72, // 40 + [25..32]
        ]),
      );
    } finally {
      backend.decRef(a);
      backend.decRef(b);
      backend.decRef(out);
    }
  });

  test("pointwise f32 where with comparison", async () => {
    const backend = getBackend(device);
    // where(x < 5, x * 2, x): doubles values below 5, leaves others unchanged.
    const data = new Float32Array([1, 6, 3, 8, 2, 7, 4, 9]);
    const a = backend.malloc(8 * 4, new Uint8Array(data.buffer));
    const out = backend.malloc(8 * 4);

    try {
      const shape = ShapeTracker.fromShape([8]);
      const gidx = AluVar.gidx;
      const x = accessorGlobal(DType.Float32, 0, shape, [gidx]);
      const kernel = new Kernel(
        1,
        8,
        AluExp.where(
          AluExp.cmplt(x, AluExp.f32(5)),
          AluExp.mul(x, AluExp.f32(2)),
          x,
        ),
      );
      const exe = await backend.prepareKernel(kernel);
      backend.dispatch(exe, [a], [out]);

      const { buffer } = await backend.read(out);
      // x < 5: [T, F, T, F, T, F, T, F] → doubled or unchanged
      expect(new Float32Array(buffer)).toEqual(
        new Float32Array([2, 6, 6, 8, 4, 7, 8, 9]),
      );
    } finally {
      backend.decRef(a);
      backend.decRef(out);
    }
  });

  test("pointwise cast f32 to i32 on 8-element array", async () => {
    const backend = getBackend(device);
    const data = new Float32Array([1.9, -2.7, 3.1, -4.8, 5.5, -6.2, 7.4, -8.9]);
    const a = backend.malloc(8 * 4, new Uint8Array(data.buffer));
    const out = backend.malloc(8 * 4);

    try {
      const shape = ShapeTracker.fromShape([8]);
      const gidx = AluVar.gidx;
      const arg = accessorGlobal(DType.Float32, 0, shape, [gidx]);
      const kernel = new Kernel(1, 8, AluExp.cast(DType.Int32, arg));
      const exe = await backend.prepareKernel(kernel);
      backend.dispatch(exe, [a], [out]);

      const { buffer } = await backend.read(out);
      // trunc_sat: truncates toward zero
      expect(new Int32Array(buffer)).toEqual(
        new Int32Array([1, -2, 3, -4, 5, -6, 7, -8]),
      );
    } finally {
      backend.decRef(a);
      backend.decRef(out);
    }
  });
});
