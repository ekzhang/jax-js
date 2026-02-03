/**
 * Triangular solve routine - AssemblyScript implementation.
 *
 * Solves A @ X = B for X where A is triangular.
 * Supports both upper and lower triangular, unit and non-unit diagonal.
 *
 * Memory layout:
 *   - aPtr: input triangular matrices A [numBatches x n x n]
 *   - bPtr: input right-hand side B [numBatches x batchRows x n]
 *   - xPtr: output solution X [numBatches x batchRows x n]
 *
 * The same algorithm exists in src/routine.ts as a JS fallback.
 * When modifying this file, also update the JS fallback to match.
 */

// ============================================================================
// f32 core solvers
// ============================================================================

/**
 * Solve upper-triangular system for f32, single vector.
 * A @ x = b where A is [n x n] upper-triangular, b and x are [n].
 */
function solve_upper_f32(
  aPtr: usize,
  bPtr: usize,
  xPtr: usize,
  n: i32,
  unitDiagonal: i32,
): void {
  // Back-substitution: solve from bottom to top
  for (let i: i32 = n - 1; i >= 0; i--) {
    let sum: f32 = load<f32>(bPtr + ((<usize>i) << 2));
    for (let j: i32 = i + 1; j < n; j++) {
      sum -=
        load<f32>(aPtr + ((<usize>(i * n + j)) << 2)) *
        load<f32>(xPtr + ((<usize>j) << 2));
    }
    if (unitDiagonal) {
      store<f32>(xPtr + ((<usize>i) << 2), sum);
    } else {
      store<f32>(
        xPtr + ((<usize>i) << 2),
        sum / load<f32>(aPtr + ((<usize>(i * n + i)) << 2)),
      );
    }
  }
}

/**
 * Solve lower-triangular system for f32, single vector.
 * A @ x = b where A is [n x n] lower-triangular, b and x are [n].
 */
function solve_lower_f32(
  aPtr: usize,
  bPtr: usize,
  xPtr: usize,
  n: i32,
  unitDiagonal: i32,
): void {
  // Forward-substitution: solve from top to bottom
  for (let i: i32 = 0; i < n; i++) {
    let sum: f32 = load<f32>(bPtr + ((<usize>i) << 2));
    for (let j: i32 = 0; j < i; j++) {
      sum -=
        load<f32>(aPtr + ((<usize>(i * n + j)) << 2)) *
        load<f32>(xPtr + ((<usize>j) << 2));
    }
    if (unitDiagonal) {
      store<f32>(xPtr + ((<usize>i) << 2), sum);
    } else {
      store<f32>(
        xPtr + ((<usize>i) << 2),
        sum / load<f32>(aPtr + ((<usize>(i * n + i)) << 2)),
      );
    }
  }
}

/**
 * Batched triangular solve for f32.
 *
 * @param aPtr - Pointer to A matrices [numBatches x n x n]
 * @param bPtr - Pointer to B vectors [numBatches x batchRows x n]
 * @param xPtr - Pointer to X output [numBatches x batchRows x n]
 * @param n - Matrix dimension
 * @param batchRows - Number of RHS vectors per matrix
 * @param numBatches - Number of A matrices
 * @param unitDiagonal - 1 if diagonal is implicitly 1, 0 otherwise
 * @param lower - 1 for lower triangular, 0 for upper triangular
 */
export function triangular_solve_batched_f32(
  aPtr: usize,
  bPtr: usize,
  xPtr: usize,
  n: i32,
  batchRows: i32,
  numBatches: i32,
  unitDiagonal: i32,
  lower: i32,
): void {
  const matrixBytes: usize = (<usize>(n * n)) << 2;
  const vectorBytes: usize = (<usize>n) << 2;

  for (let batch: i32 = 0; batch < numBatches; batch++) {
    const aOff: usize = aPtr + <usize>batch * matrixBytes;

    for (let row: i32 = 0; row < batchRows; row++) {
      const idx: usize = <usize>(batch * batchRows + row);
      const bOff: usize = bPtr + idx * vectorBytes;
      const xOff: usize = xPtr + idx * vectorBytes;

      if (lower) {
        solve_lower_f32(aOff, bOff, xOff, n, unitDiagonal);
      } else {
        solve_upper_f32(aOff, bOff, xOff, n, unitDiagonal);
      }
    }
  }
}

// ============================================================================
// f64 core solvers
// ============================================================================

/**
 * Solve upper-triangular system for f64, single vector.
 */
function solve_upper_f64(
  aPtr: usize,
  bPtr: usize,
  xPtr: usize,
  n: i32,
  unitDiagonal: i32,
): void {
  for (let i: i32 = n - 1; i >= 0; i--) {
    let sum: f64 = load<f64>(bPtr + ((<usize>i) << 3));
    for (let j: i32 = i + 1; j < n; j++) {
      sum -=
        load<f64>(aPtr + ((<usize>(i * n + j)) << 3)) *
        load<f64>(xPtr + ((<usize>j) << 3));
    }
    if (unitDiagonal) {
      store<f64>(xPtr + ((<usize>i) << 3), sum);
    } else {
      store<f64>(
        xPtr + ((<usize>i) << 3),
        sum / load<f64>(aPtr + ((<usize>(i * n + i)) << 3)),
      );
    }
  }
}

/**
 * Solve lower-triangular system for f64, single vector.
 */
function solve_lower_f64(
  aPtr: usize,
  bPtr: usize,
  xPtr: usize,
  n: i32,
  unitDiagonal: i32,
): void {
  for (let i: i32 = 0; i < n; i++) {
    let sum: f64 = load<f64>(bPtr + ((<usize>i) << 3));
    for (let j: i32 = 0; j < i; j++) {
      sum -=
        load<f64>(aPtr + ((<usize>(i * n + j)) << 3)) *
        load<f64>(xPtr + ((<usize>j) << 3));
    }
    if (unitDiagonal) {
      store<f64>(xPtr + ((<usize>i) << 3), sum);
    } else {
      store<f64>(
        xPtr + ((<usize>i) << 3),
        sum / load<f64>(aPtr + ((<usize>(i * n + i)) << 3)),
      );
    }
  }
}

/**
 * Batched triangular solve for f64.
 */
export function triangular_solve_batched_f64(
  aPtr: usize,
  bPtr: usize,
  xPtr: usize,
  n: i32,
  batchRows: i32,
  numBatches: i32,
  unitDiagonal: i32,
  lower: i32,
): void {
  const matrixBytes: usize = (<usize>(n * n)) << 3;
  const vectorBytes: usize = (<usize>n) << 3;

  for (let batch: i32 = 0; batch < numBatches; batch++) {
    const aOff: usize = aPtr + <usize>batch * matrixBytes;

    for (let row: i32 = 0; row < batchRows; row++) {
      const idx: usize = <usize>(batch * batchRows + row);
      const bOff: usize = bPtr + idx * vectorBytes;
      const xOff: usize = xPtr + idx * vectorBytes;

      if (lower) {
        solve_lower_f64(aOff, bOff, xOff, n, unitDiagonal);
      } else {
        solve_upper_f64(aOff, bOff, xOff, n, unitDiagonal);
      }
    }
  }
}
