import { expect, suite, test } from "vitest";

import { CodeGenerator } from "./wasmblr";

suite("CodeGenerator", () => {
  test("assembles the add() function", async () => {
    const cg = new CodeGenerator();

    const addFunc = cg.function([cg.f32, cg.f32], [cg.f32], () => {
      cg.local.get(0);
      cg.local.get(1);
      cg.f32.add();
    });
    cg.export(addFunc, "add");

    const wasmBytes = cg.finish();
    const { instance } = await WebAssembly.instantiate(wasmBytes);
    const { add } = instance.exports as { add(a: number, b: number): number };

    expect(add(1, 2)).toBe(3);
    expect(add(3.5, 4.6)).toBeCloseTo(8.1, 5);
  });

  test("assembles recursive factorial", async () => {
    const cg = new CodeGenerator();

    const factorialFunc = cg.function([cg.f32], [cg.f32], () => {
      cg.local.get(0);
      cg.f32.const(1.0);
      cg.f32.lt();
      cg.if(cg.f32); // base case
      {
        cg.f32.const(1.0);
      }
      cg.else();
      {
        cg.local.get(0);
        cg.local.get(0);
        cg.f32.const(1.0);
        cg.f32.sub();
        cg.call(factorialFunc);
        cg.f32.mul();
      }
      cg.end();
    });
    cg.export(factorialFunc, "factorial");

    const wasmBytes = cg.finish();
    const { instance } = await WebAssembly.instantiate(wasmBytes);
    const { factorial } = instance.exports as { factorial(x: number): number };

    expect(factorial(0)).toBe(1);
    expect(factorial(1)).toBe(1);
    expect(factorial(2)).toBe(2);
    expect(factorial(3)).toBe(6);
    expect(factorial(7)).toBe(5040);
  });

  test("can run SIMD on f32x4", async () => {
    const cg = new CodeGenerator();

    cg.memory.pages(1).export("memory");

    const vectorSumFunc = cg.function([cg.i32], [cg.f32], () => {
      const vec = cg.local.declare(cg.f32x4);
      const i = cg.local.declare(cg.i32);

      // All locals are initialized to zero in Wasm.
      // cg.i32.const(0); cg.local.set(i);
      // cg.f32.const(0); cg.f32x4.splat(); cg.local.set(vec);

      cg.loop(cg.void);
      {
        // if (i >= length) break;
        cg.block(cg.void);
        cg.local.get(i);
        cg.local.get(0);
        cg.i32.ge_u();
        cg.br_if(0);

        // vec += array[i..i+4];
        cg.local.get(i);
        cg.i32.const(4); // number of bytes per f32 element
        cg.i32.mul();
        cg.f32x4.load(4);
        cg.local.get(vec);
        cg.f32x4.add();
        cg.local.set(vec);

        // i += 4;
        cg.local.get(i);
        cg.i32.const(4);
        cg.i32.add();
        cg.local.set(i);

        cg.br(1); // continue loop
        cg.end();
      }
      cg.end();

      cg.f32.const(0);
      for (let i = 0; i < 4; i++) {
        cg.local.get(vec);
        cg.f32x4.extract_lane(i);
        cg.f32.add();
      }
    });
    cg.export(vectorSumFunc, "vectorSum");

    const wasmBytes = cg.finish();
    const { instance } = await WebAssembly.instantiate(wasmBytes);

    const { memory, vectorSum } = instance.exports as {
      memory: WebAssembly.Memory;
      vectorSum: (length: number) => number;
    };
    new Float32Array(memory.buffer).set([1, 2, 3, 4, 5, 6, 7, 8]);
    const result = vectorSum(8);
    expect(result).toBe(36);
  });

  test("select() wasm operation", async () => {
    const cg = new CodeGenerator();

    const selectFunc = cg.function([cg.i32, cg.i32, cg.i32], [cg.i32], () => {
      cg.local.get(1); // true value
      cg.local.get(2); // false value
      cg.local.get(0); // condition
      cg.select();
    });
    cg.export(selectFunc, "select");

    const wasmBytes = cg.finish();
    const { instance } = await WebAssembly.instantiate(wasmBytes);
    const { select } = instance.exports as {
      select(condition: number, trueValue: number, falseValue: number): number;
    };
    expect(select(1, 10, 20)).toBe(10);
    expect(select(0, 10, 20)).toBe(20);
  });
});
