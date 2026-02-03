/**
 * LU decomposition routine - AssemblyScript implementation.
 *
 * Computes LU factorization with partial pivoting: P @ A = L @ U
 * where L is lower-triangular with unit diagonal, U is upper-triangular.
 *
 * Memory layout:
 *   - aPtr: input matrix A [m x n]
 *   - luPtr: output LU matrix [m x n] (L and U packed together)
 *   - pivPtr: output pivot indices [min(m,n)] as i32
 *   - permPtr: output permutation [m] as i32
 */

// ============================================================================
// f32 version
// ============================================================================

/**
 * LU decomposition for f32, single matrix.
 */
export function lu_f32(
  aPtr: usize,
  luPtr: usize,
  pivPtr: usize,
  permPtr: usize,
  m: i32,
  n: i32,
): void {
  const r: i32 = m < n ? m : n;

  // Copy input to output
  for (let i: i32 = 0; i < m * n; i++) {
    store<f32>(luPtr + ((<usize>i) << 2), load<f32>(aPtr + ((<usize>i) << 2)));
  }

  // Initialize permutation
  for (let i: i32 = 0; i < m; i++) {
    store<i32>(permPtr + ((<usize>i) << 2), i);
  }

  for (let j: i32 = 0; j < r; j++) {
    // Find pivot: max |LU[i,j]| for i >= j
    let maxVal: f32 = abs<f32>(load<f32>(luPtr + ((<usize>(j * n + j)) << 2)));
    let maxRow: i32 = j;
    for (let i: i32 = j + 1; i < m; i++) {
      const val: f32 = abs<f32>(load<f32>(luPtr + ((<usize>(i * n + j)) << 2)));
      if (val > maxVal) {
        maxVal = val;
        maxRow = i;
      }
    }

    // Store pivot index
    store<i32>(pivPtr + ((<usize>j) << 2), maxRow);

    // Swap rows if needed
    if (maxRow != j) {
      for (let col: i32 = 0; col < n; col++) {
        const tmp: f32 = load<f32>(luPtr + ((<usize>(j * n + col)) << 2));
        store<f32>(
          luPtr + ((<usize>(j * n + col)) << 2),
          load<f32>(luPtr + ((<usize>(maxRow * n + col)) << 2)),
        );
        store<f32>(luPtr + ((<usize>(maxRow * n + col)) << 2), tmp);
      }
      const tmpP: i32 = load<i32>(permPtr + ((<usize>j) << 2));
      store<i32>(
        permPtr + ((<usize>j) << 2),
        load<i32>(permPtr + ((<usize>maxRow) << 2)),
      );
      store<i32>(permPtr + ((<usize>maxRow) << 2), tmpP);
    }

    // Update L and U
    const diag: f32 = load<f32>(luPtr + ((<usize>(j * n + j)) << 2));
    if (diag != 0) {
      for (let i: i32 = j + 1; i < m; i++) {
        const factor: f32 =
          load<f32>(luPtr + ((<usize>(i * n + j)) << 2)) / diag;
        store<f32>(luPtr + ((<usize>(i * n + j)) << 2), factor); // L
        for (let col: i32 = j + 1; col < n; col++) {
          const val: f32 = load<f32>(luPtr + ((<usize>(i * n + col)) << 2));
          store<f32>(
            luPtr + ((<usize>(i * n + col)) << 2),
            val - factor * load<f32>(luPtr + ((<usize>(j * n + col)) << 2)),
          );
        }
      }
    }
  }
}

/**
 * LU decomposition for f32, batched.
 */
export function lu_batched_f32(
  aPtr: usize,
  luPtr: usize,
  pivPtr: usize,
  permPtr: usize,
  m: i32,
  n: i32,
  batchSize: i32,
): void {
  const matrixBytes: usize = (<usize>(m * n)) << 2;
  const r: i32 = m < n ? m : n;
  const pivBytes: usize = (<usize>r) << 2;
  const permBytes: usize = (<usize>m) << 2;

  for (let b: i32 = 0; b < batchSize; b++) {
    lu_f32(
      aPtr + <usize>b * matrixBytes,
      luPtr + <usize>b * matrixBytes,
      pivPtr + <usize>b * pivBytes,
      permPtr + <usize>b * permBytes,
      m,
      n,
    );
  }
}

// ============================================================================
// f64 version
// ============================================================================

/**
 * LU decomposition for f64, single matrix.
 */
export function lu_f64(
  aPtr: usize,
  luPtr: usize,
  pivPtr: usize,
  permPtr: usize,
  m: i32,
  n: i32,
): void {
  const r: i32 = m < n ? m : n;

  // Copy input to output
  for (let i: i32 = 0; i < m * n; i++) {
    store<f64>(luPtr + ((<usize>i) << 3), load<f64>(aPtr + ((<usize>i) << 3)));
  }

  // Initialize permutation
  for (let i: i32 = 0; i < m; i++) {
    store<i32>(permPtr + ((<usize>i) << 2), i);
  }

  for (let j: i32 = 0; j < r; j++) {
    // Find pivot
    let maxVal: f64 = abs<f64>(load<f64>(luPtr + ((<usize>(j * n + j)) << 3)));
    let maxRow: i32 = j;
    for (let i: i32 = j + 1; i < m; i++) {
      const val: f64 = abs<f64>(load<f64>(luPtr + ((<usize>(i * n + j)) << 3)));
      if (val > maxVal) {
        maxVal = val;
        maxRow = i;
      }
    }

    store<i32>(pivPtr + ((<usize>j) << 2), maxRow);

    if (maxRow != j) {
      for (let col: i32 = 0; col < n; col++) {
        const tmp: f64 = load<f64>(luPtr + ((<usize>(j * n + col)) << 3));
        store<f64>(
          luPtr + ((<usize>(j * n + col)) << 3),
          load<f64>(luPtr + ((<usize>(maxRow * n + col)) << 3)),
        );
        store<f64>(luPtr + ((<usize>(maxRow * n + col)) << 3), tmp);
      }
      const tmpP: i32 = load<i32>(permPtr + ((<usize>j) << 2));
      store<i32>(
        permPtr + ((<usize>j) << 2),
        load<i32>(permPtr + ((<usize>maxRow) << 2)),
      );
      store<i32>(permPtr + ((<usize>maxRow) << 2), tmpP);
    }

    const diag: f64 = load<f64>(luPtr + ((<usize>(j * n + j)) << 3));
    if (diag != 0) {
      for (let i: i32 = j + 1; i < m; i++) {
        const factor: f64 =
          load<f64>(luPtr + ((<usize>(i * n + j)) << 3)) / diag;
        store<f64>(luPtr + ((<usize>(i * n + j)) << 3), factor);
        for (let col: i32 = j + 1; col < n; col++) {
          const val: f64 = load<f64>(luPtr + ((<usize>(i * n + col)) << 3));
          store<f64>(
            luPtr + ((<usize>(i * n + col)) << 3),
            val - factor * load<f64>(luPtr + ((<usize>(j * n + col)) << 3)),
          );
        }
      }
    }
  }
}

/**
 * LU decomposition for f64, batched.
 */
export function lu_batched_f64(
  aPtr: usize,
  luPtr: usize,
  pivPtr: usize,
  permPtr: usize,
  m: i32,
  n: i32,
  batchSize: i32,
): void {
  const matrixBytes: usize = (<usize>(m * n)) << 3;
  const r: i32 = m < n ? m : n;
  const pivBytes: usize = (<usize>r) << 2;
  const permBytes: usize = (<usize>m) << 2;

  for (let b: i32 = 0; b < batchSize; b++) {
    lu_f64(
      aPtr + <usize>b * matrixBytes,
      luPtr + <usize>b * matrixBytes,
      pivPtr + <usize>b * pivBytes,
      permPtr + <usize>b * permBytes,
      m,
      n,
    );
  }
}
