import { expect, test } from "vitest";

import { WasmAllocator } from "./allocator";

const WASM_PAGE_SIZE = 65536;
const MAX_MEMORY32_PAGES = 65536;

function createMockMemory(): {
  memory: WebAssembly.Memory;
  growDeltas: number[];
} {
  let byteLength = 0;
  const growDeltas: number[] = [];
  const memory = {
    get buffer() {
      return { byteLength };
    },
    grow(delta: number) {
      if (!Number.isInteger(delta) || delta < 0) {
        throw new TypeError("Memory growth must be a non-negative integer");
      }

      const previousPages = byteLength / WASM_PAGE_SIZE;
      if (previousPages + delta > MAX_MEMORY32_PAGES) {
        throw new RangeError("Memory growth exceeds the memory32 limit");
      }

      growDeltas.push(delta);
      byteLength += delta * WASM_PAGE_SIZE;
      return previousPages;
    },
  } as unknown as WebAssembly.Memory;

  return { memory, growDeltas };
}

test("grows across the signed 32-bit boundary", () => {
  const { memory, growDeltas } = createMockMemory();
  const allocator = new WasmAllocator(memory);
  const gibibyte = 2 ** 30;

  expect(allocator.malloc(gibibyte)).toBe(64);
  expect(allocator.malloc(gibibyte)).toBe(gibibyte + 64);

  expect(allocator.getStats().totalAllocated).toBe(2 ** 31 + 64);
  expect(memory.buffer.byteLength).toBe(2 ** 31 + WASM_PAGE_SIZE);
  expect(growDeltas.every((delta) => delta > 0)).toBe(true);
});

test("supports a single allocation of 2 GiB", () => {
  const { memory } = createMockMemory();
  const allocator = new WasmAllocator(memory);

  expect(allocator.malloc(2 ** 31)).toBe(64);
  expect(allocator.getStats().totalAllocated).toBe(2 ** 31 + 64);
  expect(memory.buffer.byteLength).toBe(2 ** 31 + WASM_PAGE_SIZE);
});

test("grows memory into the final memory32 page", () => {
  const { memory, growDeltas } = createMockMemory();
  const allocator = new WasmAllocator(memory);
  const gibibyte = 2 ** 30;

  allocator.malloc(gibibyte);
  allocator.malloc(gibibyte);
  allocator.malloc(gibibyte);
  allocator.malloc(gibibyte - WASM_PAGE_SIZE);

  expect(memory.buffer.byteLength).toBe(2 ** 32);
  expect(growDeltas.at(-1)).toBe(16383);
  expect(growDeltas.every((delta) => delta > 0)).toBe(true);
});

test("rejects allocations beyond the memory32 address space", () => {
  const { memory } = createMockMemory();
  const allocator = new WasmAllocator(memory);

  expect(() => allocator.malloc(2 ** 32)).toThrow(
    "Allocation exceeds the 4 GiB memory32 limit",
  );
  expect(allocator.getStats().totalAllocated).toBe(64);
  expect(memory.buffer.byteLength).toBe(0);
});
