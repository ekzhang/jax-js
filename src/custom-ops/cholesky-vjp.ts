import { Primitive } from "../frontend/core.js";
import type { Tracer } from "../frontend/core.js";
import type { UndefPrimal } from "../frontend/linearize.js";

/**
 * VJP rule for Cholesky decomposition
 *
 * Cholesky is a nonlinear operation, so if x is UndefPrimal (tangent),
 * we cannot compute the transpose directly.
 * This should not happen in practice because Cholesky uses primal values only.
 *
 * For proper reverse-mode autodiff, users should use jvp() or grad() which
 * handles nonlinear operations correctly.
 */
export function choleskyVJP(
  cotangents: Tracer[],
  args: (Tracer | UndefPrimal)[],
  params: { lower: boolean },
): (Tracer | null)[] {
  const [ct] = cotangents;
  const [x] = args;

  // Cholesky is a nonlinear operation
  // Cannot compute VJP directly - this is expected behavior
  throw new Error(
    "Cholesky decomposition is nonlinear and cannot be differentiated in reverse mode. " +
    "Use forward-mode autodiff (jvp) instead, or wrap in a custom_vjp."
  );
}
