import type { Array as JAXArray } from "../frontend/array.js";
import type { ShapedArray } from "../frontend/jaxpr.js";
import type { Device } from "../device.js";

/**
 * Interface for custom operation implementations.
 *
 * Custom ops provide a way to implement domain-specific operations
 * that don't fit neatly into the core primitive set, such as:
 * - Linear algebra: Cholesky, QR, SVD, eigendecomposition
 * - Signal processing: FFT, convolution variants
 * - ML-specific: attention, layer norm, group norm
 * - Misc: scatter-add, sort, unique, etc.
 */
export interface CustomOpImpl {
  /** Unique name for this custom op (e.g., "linalg.cholesky") */
  name: string;

  /**
   * Forward implementation - dispatches to device-specific code
   * @param args Input arrays
   * @param params Operation-specific parameters
   * @param device Target device
   * @returns Output array(s)
   */
  dispatch: (
    args: JAXArray[],
    params: Record<string, any>,
    device: Device,
  ) => JAXArray | JAXArray[] | Promise<JAXArray | JAXArray[]>;

  /**
   * Forward-mode autodiff (JVP) rule
   * @param primals Primal input values
   * @param tangents Tangent input values (derivatives)
   * @param params Operation parameters
   * @returns [primal outputs, tangent outputs]
   */
  jvp?: (
    primals: JAXArray[],
    tangents: JAXArray[],
    params: Record<string, any>,
  ) => [JAXArray[], JAXArray[]];

  /**
   * Reverse-mode autodiff (VJP) transpose rule
   * @param cotangents Upstream gradients
   * @param args Original input arguments
   * @param params Operation parameters
   * @returns Gradient functions for each input
   */
  vjp?: (
    cotangents: JAXArray[],
    args: JAXArray[],
    params: Record<string, any>,
  ) => JAXArray[];

  /**
   * Abstract evaluation - computes output shape/dtype without running computation
   * @param inputs Input shapes and dtypes
   * @param params Operation parameters
   * @returns Output shape(s) and dtype(s)
   */
  abstractEval?: (
    inputs: ShapedArray[],
    params: Record<string, any>,
  ) => ShapedArray | ShapedArray[];
}

/**
 * Global registry of custom operations.
 * Operations register themselves on module load.
 */
class CustomOpRegistry {
  private ops = new Map<string, CustomOpImpl>();

  /**
   * Register a custom operation implementation
   */
  register(impl: CustomOpImpl): void {
    if (this.ops.has(impl.name)) {
      console.warn(`Custom op "${impl.name}" is already registered, overwriting`);
    }
    this.ops.set(impl.name, impl);
  }

  /**
   * Get a custom operation implementation by name
   */
  get(name: string): CustomOpImpl | undefined {
    return this.ops.get(name);
  }

  /**
   * Check if a custom operation is registered
   */
  has(name: string): boolean {
    return this.ops.has(name);
  }

  /**
   * List all registered custom operations
   */
  list(): string[] {
    return Array.from(this.ops.keys());
  }
}

/**
 * Singleton instance of the custom op registry
 */
export const customOpRegistry = new CustomOpRegistry();
