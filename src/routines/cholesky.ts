/**
 * Cholesky decomposition for AssemblyScript compilation.
 *
 * This file is compiled to WASM at build time using AssemblyScript.
 * It implements the Cholesky-Banachiewicz algorithm: A = L @ L^T
 *
 * The same algorithm exists in src/routine.ts as a JS fallback.
 * When modifying this file, also update the JS fallback to match.
 */

/**
 * Cholesky decomposition for f32 matrices.
 * Computes lower-triangular L where A = L @ L^T.
 *
 * @param inPtr - Pointer to input matrix (n×n, f32)
 * @param outPtr - Pointer to output matrix (n×n, f32)
 * @param n - Matrix dimension
 */
export function cholesky_f32(inPtr: usize, outPtr: usize, n: i32): void {
  const elemSize: i32 = 4;
  const nn: i32 = n * n;

  // Zero output matrix (upper triangle won't be written by algorithm)
  for (let idx: i32 = 0; idx < nn; idx++) {
    store<f32>(outPtr + <usize>(idx * elemSize), 0.0);
  }

  // Cholesky-Banachiewicz algorithm
  // https://en.wikipedia.org/wiki/Cholesky_decomposition#Computation
  for (let i: i32 = 0; i < n; i++) {
    for (let j: i32 = 0; j <= i; j++) {
      // sum = A[i, j]
      let sum: f32 = load<f32>(inPtr + <usize>((i * n + j) * elemSize));

      // sum -= sum(L[i,k] * L[j,k] for k in 0..j)
      for (let k: i32 = 0; k < j; k++) {
        const lik: f32 = load<f32>(outPtr + <usize>((i * n + k) * elemSize));
        const ljk: f32 = load<f32>(outPtr + <usize>((j * n + k) * elemSize));
        sum -= lik * ljk;
      }

      if (i == j) {
        // Diagonal: L[i,i] = sqrt(sum)
        store<f32>(outPtr + <usize>((i * n + j) * elemSize), sqrt<f32>(sum));
      } else {
        // Off-diagonal: L[i,j] = sum / L[j,j]
        const ljj: f32 = load<f32>(outPtr + <usize>((j * n + j) * elemSize));
        store<f32>(outPtr + <usize>((i * n + j) * elemSize), sum / ljj);
      }
    }
  }
}

/**
 * Cholesky decomposition for f64 matrices.
 * Computes lower-triangular L where A = L @ L^T.
 *
 * @param inPtr - Pointer to input matrix (n×n, f64)
 * @param outPtr - Pointer to output matrix (n×n, f64)
 * @param n - Matrix dimension
 */
export function cholesky_f64(inPtr: usize, outPtr: usize, n: i32): void {
  const elemSize: i32 = 8;
  const nn: i32 = n * n;

  // Zero output matrix
  for (let idx: i32 = 0; idx < nn; idx++) {
    store<f64>(outPtr + <usize>(idx * elemSize), 0.0);
  }

  // Cholesky-Banachiewicz algorithm
  for (let i: i32 = 0; i < n; i++) {
    for (let j: i32 = 0; j <= i; j++) {
      let sum: f64 = load<f64>(inPtr + <usize>((i * n + j) * elemSize));

      for (let k: i32 = 0; k < j; k++) {
        const lik: f64 = load<f64>(outPtr + <usize>((i * n + k) * elemSize));
        const ljk: f64 = load<f64>(outPtr + <usize>((j * n + k) * elemSize));
        sum -= lik * ljk;
      }

      if (i == j) {
        store<f64>(outPtr + <usize>((i * n + j) * elemSize), sqrt<f64>(sum));
      } else {
        const ljj: f64 = load<f64>(outPtr + <usize>((j * n + j) * elemSize));
        store<f64>(outPtr + <usize>((i * n + j) * elemSize), sum / ljj);
      }
    }
  }
}

/**
 * Batched Cholesky decomposition for f32 matrices.
 * Processes multiple n×n matrices in sequence.
 *
 * @param inPtr - Pointer to input matrices (batch × n × n, f32)
 * @param outPtr - Pointer to output matrices (batch × n × n, f32)
 * @param n - Matrix dimension
 * @param batch - Number of matrices
 */
export function cholesky_batched_f32(
  inPtr: usize,
  outPtr: usize,
  n: i32,
  batch: i32,
): void {
  const matrixSize: i32 = n * n * 4; // f32 = 4 bytes
  for (let b: i32 = 0; b < batch; b++) {
    cholesky_f32(
      inPtr + <usize>(b * matrixSize),
      outPtr + <usize>(b * matrixSize),
      n,
    );
  }
}

/**
 * Batched Cholesky decomposition for f64 matrices.
 *
 * @param inPtr - Pointer to input matrices (batch × n × n, f64)
 * @param outPtr - Pointer to output matrices (batch × n × n, f64)
 * @param n - Matrix dimension
 * @param batch - Number of matrices
 */
export function cholesky_batched_f64(
  inPtr: usize,
  outPtr: usize,
  n: i32,
  batch: i32,
): void {
  const matrixSize: i32 = n * n * 8; // f64 = 8 bytes
  for (let b: i32 = 0; b < batch; b++) {
    cholesky_f64(
      inPtr + <usize>(b * matrixSize),
      outPtr + <usize>(b * matrixSize),
      n,
    );
  }
}
