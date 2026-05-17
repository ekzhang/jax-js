import type tf from "@tensorflow/tfjs";

export interface Strategy {
  name: string;
  run(): Promise<number>;
}

export function createRandomBuffer(length: number, centered = false) {
  const buffer = new Float32Array(length);
  for (let i = 0; i < buffer.length; i++) {
    buffer[i] = centered ? Math.random() - 0.5 : Math.random();
  }
  return buffer;
}

export function logSampleBuffer(buf: Float32Array | Float16Array) {
  console.log(
    buf[0],
    buf[1],
    buf[2],
    buf[3],
    buf[Math.floor(buf.length / 2)],
    buf[buf.length - 1],
  );
}

export async function runBenchmark(
  name: string,
  fn: () => Promise<void>,
): Promise<number> {
  performance.mark(`${name}-start`);
  const start = performance.now();
  await fn();
  const time = performance.now() - start;
  performance.mark(`${name}-end`);
  performance.measure(name, `${name}-start`, `${name}-end`);
  return time / 1000;
}

export async function importTfjs(
  backend: "wasm" | "webgpu",
): Promise<typeof tf> {
  const tf = await import("@tensorflow/tfjs");
  if (backend === "wasm") {
    if (!isSecureContext || !crossOriginIsolated) {
      alert("tfjs-wasm requires a secure context and cross-origin isolation.");
      throw new Error("Insecure context for tfjs-wasm backend.");
    }
    const { setThreadsCount, setWasmPaths } = await import(
      "@tensorflow/tfjs-backend-wasm"
    );
    setThreadsCount(navigator.hardwareConcurrency);
    setWasmPaths(
      `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tf.version.tfjs}/wasm-out/`,
    );
    await tf.setBackend("wasm");
  } else if (backend === "webgpu") {
    await import("@tensorflow/tfjs-backend-webgpu");
    await tf.setBackend("webgpu");
  } else {
    throw new Error(`Unsupported backend: ${backend}`);
  }
  return tf;
}

export async function getWebgpuDevice(): Promise<GPUDevice> {
  const adapter = await navigator.gpu.requestAdapter({
    powerPreference: "high-performance",
  });
  if (!adapter) {
    alert("WebGPU not supported");
    throw new Error("WebGPU not supported");
  }

  try {
    return await adapter.requestDevice({
      requiredLimits: {
        maxComputeInvocationsPerWorkgroup:
          adapter.limits.maxComputeInvocationsPerWorkgroup,
        maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
        maxComputeWorkgroupSizeY: adapter.limits.maxComputeWorkgroupSizeY,
        maxComputeWorkgroupSizeZ: adapter.limits.maxComputeWorkgroupSizeZ,
        maxComputeWorkgroupStorageSize:
          adapter.limits.maxComputeWorkgroupStorageSize,
        maxComputeWorkgroupsPerDimension:
          adapter.limits.maxComputeWorkgroupsPerDimension,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    });
  } catch (error) {
    throw new Error("Error when creating device: " + error);
  }
}
