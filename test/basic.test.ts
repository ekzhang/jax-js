import { x } from "jax-js";

import { expect, test } from "vitest";

test("x is 3", () => {
  expect(x).toBe(3);
});

test("has webgpu", async () => {
  const adapter = await navigator.gpu?.requestAdapter();
  const device = await adapter?.requestDevice();
  if (!adapter || !device) {
    throw new Error("No adapter or device");
  }
  console.log(device.adapterInfo.architecture);
  console.log(device.adapterInfo.vendor);
  console.log(adapter.limits.maxVertexBufferArrayStride);
});
