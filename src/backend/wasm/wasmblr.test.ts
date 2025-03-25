import { suite, test, expect } from "vitest";
import { CodeGenerator } from "./wasmblr";

suite("CodeGenerator", () => {
  test("should assemble the add() function", async () => {
    const cg = new CodeGenerator();

    const addFunc = cg.function([cg.f32, cg.f32], [cg.f32], () => {
      cg.local.get(0);
      cg.local.get(1);
      cg.f32.add();
    });
    cg.export_(addFunc, "add");

    const wasmBytes = cg.finish();
    const { instance } = await WebAssembly.instantiate(wasmBytes);
    const { add } = instance.exports as { add(a: number, b: number): number };

    expect(add(1, 2)).toBe(3);
    expect(add(3.5, 4.6)).toBeCloseTo(8.1, 5);
  });
});
