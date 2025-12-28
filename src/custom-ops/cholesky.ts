import type { Device } from "../device.js";
import type { Array as JAXArray } from "../frontend/array.js";
import type { ShapedArray } from "../frontend/jaxpr.js";
import { customOpRegistry, CustomOpImpl } from "./registry.js";
import { choleskyCPU, type CholeskyParams } from "./cholesky-cpu.js";
import { choleskyWASM } from "./cholesky-wasm.js";
import { choleskyWebGPU } from "./cholesky-webgpu.js";
import { choleskyJVP } from "./cholesky-jvp.js";
import { choleskyVJP } from "./cholesky-vjp.js";

/**
 * Cholesky decomposition custom operation
 *
 * Computes the Cholesky decomposition of a positive-definite matrix:
 *   A = L @ L^T  (lower=true, default)
 *   A = U^T @ U  (lower=false)
 *
 * This custom op provides optimized implementations for each backend:
 * - CPU: Right-looking algorithm with cache optimization
 * - WASM: Delegates to CPU (same typed array optimization)
 * - WebGPU: Column-wise parallel GPU shader with batching
 *
 * Supports forward-mode autodiff (JVP).
 * Reverse-mode autodiff (VJP) throws NonlinearError as expected.
 */

// Register the Cholesky custom operation
customOpRegistry.register({
  name: "linalg.cholesky",

  /**
   * Dispatch to appropriate backend implementation
   */
  dispatch(args: JAXArray[], params: CholeskyParams, device: Device): JAXArray {
    const [a] = args;

    switch (device) {
      case "cpu":
        return choleskyCPU(a, params);

      case "wasm":
        return choleskyWASM(a, params);

      case "webgpu":
        return choleskyWebGPU(a, params);

      default:
        throw new Error(`Cholesky not implemented for device: ${device}`);
    }
  },

  /**
   * Forward-mode autodiff (JVP)
   */
  jvp: choleskyJVP,

  /**
   * Reverse-mode autodiff (VJP)
   * Note: Throws NonlinearError because Cholesky is nonlinear
   */
  vjp: choleskyVJP,

  /**
   * Abstract evaluation - computes output shape/dtype
   */
  abstractEval(inputs: ShapedArray[], params: CholeskyParams): ShapedArray {
    const [a] = inputs;

    // Validate input
    if (a.ndim !== 2) {
      throw new TypeError(`cholesky: input must be 2D, got ${a.ndim}D`);
    }
    if (a.shape[0] !== a.shape[1]) {
      throw new TypeError(
        `cholesky: matrix must be square, got ${a.shape[0]}x${a.shape[1]}`,
      );
    }

    // Output has same shape/dtype as input
    return new ShapedArray(a.shape, a.dtype, a.weakType);
  },
} satisfies CustomOpImpl);

/**
 * Export the custom op type for external use
 * (though users should call linalg.cholesky() instead)
 */
export type { CholeskyParams } from "./cholesky-cpu.js";
