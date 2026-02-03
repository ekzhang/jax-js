/**
 * Triangular solve routine - AssemblyScript implementation.
 *
 * Solves A @ X.T = B.T for X where A is upper-triangular.
 * Supports both unit-diagonal and non-unit-diagonal modes.
 *
 * Memory layout:
 *   - aPtr: input upper-triangular matrix A [n x n]
 *   - bPtr: input right-hand side B [batch x n]
 *   - xPtr: output solution X [batch x n]
 */

// ============================================================================
// f32 versions
// ============================================================================

/**
 * Solve upper-triangular system for f32, single matrix.
 * A @ x = b where A is [n x n] upper-triangular, b and x are [n].
 */
export function triangular_solve_f32(
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
 * Solve upper-triangular system for f32, batched.
 * A @ X.T = B.T where A is [n x n], B and X are [batch x n].
 */
export function triangular_solve_batched_f32(
  aPtr: usize,
  bPtr: usize,
  xPtr: usize,
  n: i32,
  batch: i32,
  unitDiagonal: i32,
): void {
  const rowBytes: usize = (<usize>n) << 2;
  for (let t: i32 = 0; t < batch; t++) {
    const bRow = bPtr + <usize>t * rowBytes;
    const xRow = xPtr + <usize>t * rowBytes;
    triangular_solve_f32(aPtr, bRow, xRow, n, unitDiagonal);
  }
}

// ============================================================================
// f64 versions
// ============================================================================

/**
 * Solve upper-triangular system for f64, single matrix.
 */
export function triangular_solve_f64(
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
 * Solve upper-triangular system for f64, batched.
 */
export function triangular_solve_batched_f64(
  aPtr: usize,
  bPtr: usize,
  xPtr: usize,
  n: i32,
  batch: i32,
  unitDiagonal: i32,
): void {
  const rowBytes: usize = (<usize>n) << 3;
  for (let t: i32 = 0; t < batch; t++) {
    const bRow = bPtr + <usize>t * rowBytes;
    const xRow = xPtr + <usize>t * rowBytes;
    triangular_solve_f64(aPtr, bRow, xRow, n, unitDiagonal);
  }
}
