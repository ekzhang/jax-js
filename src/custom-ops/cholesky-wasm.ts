import type { Array as JAXArray } from "../frontend/array.js";
import { choleskyCPU, CholeskyParams } from "./cholesky-cpu.js";

/**
 * WASM implementation of Cholesky decomposition
 *
 * Currently delegates to the CPU implementation.
 * A full WASM implementation would use CodeGenerator to emit WebAssembly bytecode.
 *
 * TODO: Implement native WASM version using CodeGenerator for potential performance gains
 */
export function choleskyWASM(a: JAXArray, params: CholeskyParams): JAXArray {
  // Delegate to CPU implementation
  // The CPU backend and WASM backend both use typed arrays,
  // so the optimized JS implementation works efficiently for both
  return choleskyCPU(a, params);
}
