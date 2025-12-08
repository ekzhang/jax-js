import { defaultDevice, devices, numpy as np } from "@jax-js/jax";
import { expect, test } from "vitest";

defaultDevice("cpu");

test("setup has wasm and cpu devices", () => {
  // We'll need these two devices to test behavior with arrays on different
  // devices when they interact.
  expect(devices).toContain("cpu");
  expect(devices).toContain("wasm");
});

test("binop moves to committed device", () => {
  const x = np.array([1, 2, 3], { device: "wasm" });
  const y = np.array(4);

  expect(x.device).not.toBe(y.device);
  const z = x.add(y);
  expect(z.device).toBe("wasm"); // committed
  expect(z.js()).toEqual([5, 6, 7]);
});

test("devicePut moves device", () => {
  let ar = np.array([1, 2, 3]);
  expect(ar.device).toBe("cpu");
  ar = np.devicePut(ar, "wasm");
  expect(ar.device).toBe("wasm");
  ar = np.devicePut(ar, "wasm");
  expect(ar.device).toBe("wasm");
  ar.dispose();
});

test("devicePut can be called with no device", () => {
  let ar = np.array([1, 2, 3]);
  expect(ar.device).toBe("cpu");
  ar = np.devicePut(ar);
  expect(ar.device).toBe("cpu");

  // ar should still be uncommitted, as devicePut is a no-op in this case.
  ar = ar.add(np.array(2, { device: "wasm" }));
  expect(ar.device).toBe("wasm");
  expect(ar.js()).toEqual([3, 4, 5]);
});

test("devicePut works with scalars", () => {
  const x = np.devicePut(5, "wasm");
  const y = np.devicePut(10);
  expect(x.device).toBe("wasm");
  expect(x.dtype).toBe(np.float32);
  expect(x.weakType).toBe(true);
  expect(y.device).toBe("cpu");
  expect(y.dtype).toBe(np.float32);
  expect(y.weakType).toBe(true);

  const z = x.add(y);
  expect(z.device).toBe("wasm");
  expect(z.js()).toBe(15);
});
