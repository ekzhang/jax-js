import { beforeAll, expect, suite, test } from "vitest";

import { WebGLBackend } from "./webgl";
import { AluExp, AluVar, DType, Kernel } from "../alu";

// Use a fresh context without enabling EXT_color_buffer_float. Integer render
// targets are part of core WebGL2, including on software-rendered test runners.
const gl =
  typeof OffscreenCanvas === "undefined"
    ? null
    : new OffscreenCanvas(1, 1).getContext("webgl2");

suite("WebGL integer storage", () => {
  let backend: WebGLBackend;

  beforeAll(({ skip }) => {
    if (!gl) {
      skip();
      return;
    }
    backend = new WebGLBackend(gl);
  });

  // Cross several texture rows and leave a partially used final texel.
  const size = 131;
  const load = (dtype: DType) =>
    AluExp.globalIndex(dtype, 0, size, AluVar.gidx);

  test("preserves every bit through upload and both readback paths", async () => {
    // Includes subnormal, signed-zero, infinity, and NaN float bit patterns.
    // This tests storage only; float shader arithmetic may still flush denormals.
    const patterns = [
      0, 1, 2, 0x007fffff, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234,
      0xffffffff,
    ];
    const words = Uint32Array.from(
      { length: size },
      (_, i) => patterns[i % patterns.length],
    );
    const bytes = new Uint8Array(words.buffer);
    const padded = new Uint8Array(bytes.length + 3);
    padded.set(bytes, 3);
    // Exercise an unaligned source view and a partially used final word.
    const input = padded.subarray(3, padded.length - 1);
    const slot = backend.malloc(input.length, input);
    try {
      expect(backend.readSync(slot)).toEqual(input);
      expect(await backend.read(slot)).toEqual(input);
      expect(backend.readSync(slot, 5, 37)).toEqual(input.slice(5, 42));
      expect(await backend.read(slot, 5, 37)).toEqual(input.slice(5, 42));
    } finally {
      backend.decRef(slot);
    }
  });

  test.each([DType.Int32, DType.Uint32])(
    "preserves %s bit patterns across shader boundaries",
    async (dtype) => {
      const patterns = [
        0, 1, 2, 0x007fffff, 0x80000000, 0x7fffffff, 0x7f800001, 0xffffffff,
      ];
      const words = Uint32Array.from(
        { length: size },
        (_, i) => patterns[i % patterns.length],
      );
      const bytes = new Uint8Array(words.buffer);
      const input = backend.malloc(bytes.length, bytes);
      const output = backend.malloc(bytes.length);
      try {
        const exe = backend.prepareKernelSync(new Kernel(1, size, load(dtype)));
        backend.dispatch(exe, [input], [output]);
        expect(backend.readSync(output)).toEqual(bytes);
        expect(await backend.read(output)).toEqual(bytes);
      } finally {
        backend.decRef(input);
        backend.decRef(output);
      }
    },
  );

  test("loads uploaded booleans and stores canonical zero/one words", async () => {
    const values = Uint32Array.from(
      { length: size },
      (_, i) => [0, 1, 2, 0xffffffff][i % 4],
    );
    const input = backend.malloc(
      values.byteLength,
      new Uint8Array(values.buffer),
    );
    const output = backend.malloc(values.byteLength);
    const expected = values.map((value) => Number(value !== 0));
    try {
      const exe = backend.prepareKernelSync(
        new Kernel(1, size, load(DType.Bool)),
      );
      backend.dispatch(exe, [input], [output]);
      expect(new Uint32Array(backend.readSync(output).buffer)).toEqual(expected);
      expect(new Uint32Array((await backend.read(output)).buffer)).toEqual(
        expected,
      );
    } finally {
      backend.decRef(input);
      backend.decRef(output);
    }
  });

  test.each([false, true])("materializes constant %s booleans", async (value) => {
    const output = backend.malloc(size * 4);
    try {
      const exe = backend.prepareKernelSync(
        new Kernel(0, size, AluExp.bool(value)),
      );
      backend.dispatch(exe, [], [output]);
      const expected = new Uint32Array(size).fill(Number(value));
      expect(new Uint32Array(backend.readSync(output).buffer)).toEqual(expected);
      expect(new Uint32Array((await backend.read(output)).buffer)).toEqual(
        expected,
      );
    } finally {
      backend.decRef(output);
    }
  });

  test("uses a materialized comparison mask in another shader", async () => {
    const values = Float32Array.from({ length: size }, (_, i) => (i % 5) - 2);
    const input = backend.malloc(
      values.byteLength,
      new Uint8Array(values.buffer),
    );
    const mask = backend.malloc(size * 4);
    const output = backend.malloc(size * 4);
    try {
      const comparison = backend.prepareKernelSync(
        new Kernel(1, size, AluExp.cmplt(load(DType.Float32), AluExp.f32(0))),
      );
      backend.dispatch(comparison, [input], [mask]);
      const expectedMask = Uint32Array.from(values, (x) => Number(x < 0));
      expect(new Uint32Array(backend.readSync(mask).buffer)).toEqual(
        expectedMask,
      );
      expect(new Uint32Array((await backend.read(mask)).buffer)).toEqual(
        expectedMask,
      );

      const select = backend.prepareKernelSync(
        new Kernel(
          1,
          size,
          AluExp.where(load(DType.Bool), AluExp.i32(-1), AluExp.i32(1)),
        ),
      );
      backend.dispatch(select, [mask], [output]);
      const expected = Int32Array.from(values, (x) => (x < 0 ? -1 : 1));
      expect(new Int32Array(backend.readSync(output).buffer)).toEqual(expected);
      expect(new Int32Array((await backend.read(output)).buffer)).toEqual(
        expected,
      );
    } finally {
      backend.decRef(input);
      backend.decRef(mask);
      backend.decRef(output);
    }
  });

  test("bitcasts float inputs and outputs around arithmetic", async () => {
    const values = Float32Array.from({ length: size }, (_, i) => i / 4 - 16);
    const input = backend.malloc(
      values.byteLength,
      new Uint8Array(values.buffer),
    );
    const output = backend.malloc(values.byteLength);
    try {
      const exe = backend.prepareKernelSync(
        new Kernel(1, size, AluExp.mul(load(DType.Float32), AluExp.f32(2))),
      );
      backend.dispatch(exe, [input], [output]);
      const expected = values.map((x) => x * 2);
      expect(new Float32Array(backend.readSync(output).buffer)).toEqual(
        expected,
      );
      expect(new Float32Array((await backend.read(output)).buffer)).toEqual(
        expected,
      );
    } finally {
      backend.decRef(input);
      backend.decRef(output);
    }
  });
});
