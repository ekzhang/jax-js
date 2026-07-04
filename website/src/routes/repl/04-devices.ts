import { defaultDevice, init, jit, numpy as np } from "@jax-js/jax";

// The same jitted function can run on any available backend.
const available = await init("wasm", "webgpu");
const work = jit((x: np.Array) => {
  for (let i = 0; i < 12; i++) {
    x = np.sin(x.ref).add(np.cos(x));
  }
  return x;
});

// blockUntilReady() measures device work, not just dispatch time.
for (const device of available.filter((d) => d === "wasm" || d === "webgpu")) {
  defaultDevice(device);
  const x = np.linspace(0, 10, 200_000);
  const start = performance.now();
  const y = work(x);
  await y.blockUntilReady();
  console.log(device, `${(performance.now() - start).toFixed(1)} ms`);
}
