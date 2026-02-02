/**
 * WASM implementation of Cholesky decomposition.
 *
 * This implements the Cholesky-Banachiewicz algorithm to compute
 * the lower-triangular L where A = L @ L^T.
 *
 * Supports both f32 and f64 dtypes. The routine function takes a float type
 * (cg.f32 or cg.f64) and derives element size from it.
 */

import { CodeGenerator } from "./wasmblr";

/**
 * Generate a WASM function for Cholesky decomposition.
 *
 * The function takes (inPtr: i32, outPtr: i32, n: i32) and computes
 * the Cholesky decomposition of a single n×n matrix.
 *
 * Algorithm (Cholesky-Banachiewicz):
 *   for i = 0 to n-1:
 *     for j = 0 to i:
 *       sum = A[i,j]
 *       for k = 0 to j-1:
 *         sum -= L[i,k] * L[j,k]
 *       if i == j:
 *         L[i,j] = sqrt(sum)
 *       else:
 *         L[i,j] = sum / L[j,j]
 *
 * @param cg - The code generator
 * @param ft - The float type (cg.f32 or cg.f64)
 * @returns The function index
 */
export function wasm_cholesky(
  cg: CodeGenerator,
  ft: CodeGenerator["f32"] | CodeGenerator["f64"],
): number {
  const elemSize = ft.name === "f32" ? 4 : 8;

  // Parameters: inPtr (i32), outPtr (i32), n (i32)
  return cg.function([cg.i32, cg.i32, cg.i32], [], () => {
    // Local variables
    const i = cg.local.declare(cg.i32);
    const j = cg.local.declare(cg.i32);
    const k = cg.local.declare(cg.i32);
    const sum = cg.local.declare(ft);
    const _ljj = cg.local.declare(ft); // L[j,j] cached for division
    const ijOffset = cg.local.declare(cg.i32); // (i*n + j) * elemSize
    const jjOffset = cg.local.declare(cg.i32); // (j*n + j) * elemSize
    const ikOffset = cg.local.declare(cg.i32); // (i*n + k) * elemSize
    const jkOffset = cg.local.declare(cg.i32); // (j*n + k) * elemSize
    const nBytes = cg.local.declare(cg.i32); // n * elemSize (row stride in bytes)

    // Aliases for parameters
    const inPtr = 0;
    const outPtr = 1;
    const n = 2;

    // nBytes = n * elemSize (bytes per row)
    cg.local.get(n);
    cg.i32.const(elemSize);
    cg.i32.mul();
    cg.local.set(nBytes);

    // Zero out the output matrix first (upper triangle won't be written)
    // idx = 0; while (idx < n*n) { out[idx] = 0; idx++; }
    const idx = cg.local.declare(cg.i32);
    const nn = cg.local.declare(cg.i32); // n * n

    cg.local.get(n);
    cg.local.get(n);
    cg.i32.mul();
    cg.local.set(nn);

    cg.i32.const(0);
    cg.local.set(idx);

    cg.loop(cg.void);
    {
      cg.block(cg.void);
      cg.local.get(idx);
      cg.local.get(nn);
      cg.i32.ge_u();
      cg.br_if(0);

      // out[idx] = 0.0
      cg.local.get(outPtr);
      cg.local.get(idx);
      cg.i32.const(elemSize);
      cg.i32.mul();
      cg.i32.add();
      ft.const(0.0);
      ft.store(0, 0);

      // idx++
      cg.local.get(idx);
      cg.i32.const(1);
      cg.i32.add();
      cg.local.set(idx);

      cg.br(1);

      cg.end(); // block
    }
    cg.end(); // loop

    // i = 0
    cg.i32.const(0);
    cg.local.set(i);

    // Outer loop: for i = 0 to n-1
    cg.loop(cg.void);
    {
      // if (i >= n) break outer loop
      cg.block(cg.void);
      cg.local.get(i);
      cg.local.get(n);
      cg.i32.ge_u();
      cg.br_if(0);

      // j = 0
      cg.i32.const(0);
      cg.local.set(j);

      // Middle loop: for j = 0 to i (inclusive)
      cg.loop(cg.void);
      {
        // if (j > i) break middle loop
        cg.block(cg.void);
        cg.local.get(j);
        cg.local.get(i);
        cg.i32.gt_u();
        cg.br_if(0);

        // ijOffset = outPtr + (i * n + j) * elemSize
        cg.local.get(outPtr);
        cg.local.get(i);
        cg.local.get(n);
        cg.i32.mul();
        cg.local.get(j);
        cg.i32.add();
        cg.i32.const(elemSize);
        cg.i32.mul();
        cg.i32.add();
        cg.local.set(ijOffset);

        // sum = A[i,j] (from input)
        cg.local.get(inPtr);
        cg.local.get(i);
        cg.local.get(n);
        cg.i32.mul();
        cg.local.get(j);
        cg.i32.add();
        cg.i32.const(elemSize);
        cg.i32.mul();
        cg.i32.add();
        ft.load(0, 0);
        cg.local.set(sum);

        // Inner loop: for k = 0 to j-1
        cg.i32.const(0);
        cg.local.set(k);

        cg.block(cg.void); // Block to break if j == 0 (no inner iterations)
        cg.local.get(j);
        cg.i32.eqz();
        cg.br_if(0);

        cg.loop(cg.void);
        {
          // if (k >= j) break inner loop
          cg.block(cg.void);
          cg.local.get(k);
          cg.local.get(j);
          cg.i32.ge_u();
          cg.br_if(0);

          // ikOffset = outPtr + (i * n + k) * elemSize
          cg.local.get(outPtr);
          cg.local.get(i);
          cg.local.get(n);
          cg.i32.mul();
          cg.local.get(k);
          cg.i32.add();
          cg.i32.const(elemSize);
          cg.i32.mul();
          cg.i32.add();
          cg.local.set(ikOffset);

          // jkOffset = outPtr + (j * n + k) * elemSize
          cg.local.get(outPtr);
          cg.local.get(j);
          cg.local.get(n);
          cg.i32.mul();
          cg.local.get(k);
          cg.i32.add();
          cg.i32.const(elemSize);
          cg.i32.mul();
          cg.i32.add();
          cg.local.set(jkOffset);

          // sum -= L[i,k] * L[j,k]
          cg.local.get(sum);
          cg.local.get(ikOffset);
          ft.load(0, 0);
          cg.local.get(jkOffset);
          ft.load(0, 0);
          ft.mul();
          ft.sub();
          cg.local.set(sum);

          // k++
          cg.local.get(k);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(k);

          // Continue inner loop
          cg.br(1);

          cg.end(); // block (break target)
        }
        cg.end(); // loop

        cg.end(); // block (skip inner loop if j == 0)

        // if i == j: L[i,j] = sqrt(sum)
        // else: L[i,j] = sum / L[j,j]
        cg.local.get(i);
        cg.local.get(j);
        cg.i32.eq();
        cg.if(cg.void);
        {
          // L[i,j] = sqrt(sum)
          cg.local.get(ijOffset);
          cg.local.get(sum);
          ft.sqrt();
          ft.store(0, 0);
        }
        cg.else();
        {
          // jjOffset = outPtr + (j * n + j) * elemSize
          cg.local.get(outPtr);
          cg.local.get(j);
          cg.local.get(n);
          cg.i32.mul();
          cg.local.get(j);
          cg.i32.add();
          cg.i32.const(elemSize);
          cg.i32.mul();
          cg.i32.add();
          cg.local.set(jjOffset);

          // L[i,j] = sum / L[j,j]
          cg.local.get(ijOffset);
          cg.local.get(sum);
          cg.local.get(jjOffset);
          ft.load(0, 0);
          ft.div();
          ft.store(0, 0);
        }
        cg.end(); // if

        // j++
        cg.local.get(j);
        cg.i32.const(1);
        cg.i32.add();
        cg.local.set(j);

        // Continue middle loop
        cg.br(1);

        cg.end(); // block (break target)
      }
      cg.end(); // loop

      // i++
      cg.local.get(i);
      cg.i32.const(1);
      cg.i32.add();
      cg.local.set(i);

      // Continue outer loop
      cg.br(1);

      cg.end(); // block (break target)
    }
    cg.end(); // loop
  });
}

/**
 * Generate a WASM module for batched Cholesky decomposition.
 *
 * The exported function takes (inPtr, outPtr, n, batchSize) and computes
 * Cholesky decomposition for each matrix in the batch.
 *
 * Creates either an f32 or f64 module based on elementSize.
 *
 * @param elementSize - Element size in bytes: 4 for f32, 8 for f64
 * @returns The compiled WebAssembly.Module
 */
export function createCholeskyModule(
  elementSize: 4 | 8 = 4,
): WebAssembly.Module {
  const cg = new CodeGenerator();
  cg.memory.import("env", "memory");

  const ft = elementSize === 4 ? cg.f32 : cg.f64;
  const choleskyFn = wasm_cholesky(cg, ft);

  // Batched wrapper: (inPtr, outPtr, n, batchSize) -> void
  const batchedFn = cg.function([cg.i32, cg.i32, cg.i32, cg.i32], [], () => {
    const batch = cg.local.declare(cg.i32);
    const matrixSize = cg.local.declare(cg.i32); // n * n * elementSize bytes
    const inOffset = cg.local.declare(cg.i32);
    const outOffset = cg.local.declare(cg.i32);

    // matrixSize = n * n * elementSize
    cg.local.get(2); // n
    cg.local.get(2); // n
    cg.i32.mul();
    cg.i32.const(elementSize);
    cg.i32.mul();
    cg.local.set(matrixSize);

    // batch = 0
    cg.i32.const(0);
    cg.local.set(batch);

    cg.loop(cg.void);
    {
      // if (batch >= batchSize) break
      cg.block(cg.void);
      cg.local.get(batch);
      cg.local.get(3); // batchSize
      cg.i32.ge_u();
      cg.br_if(0);

      // inOffset = inPtr + batch * matrixSize
      cg.local.get(0); // inPtr
      cg.local.get(batch);
      cg.local.get(matrixSize);
      cg.i32.mul();
      cg.i32.add();
      cg.local.set(inOffset);

      // outOffset = outPtr + batch * matrixSize
      cg.local.get(1); // outPtr
      cg.local.get(batch);
      cg.local.get(matrixSize);
      cg.i32.mul();
      cg.i32.add();
      cg.local.set(outOffset);

      // Call single-matrix Cholesky
      cg.local.get(inOffset);
      cg.local.get(outOffset);
      cg.local.get(2); // n
      cg.call(choleskyFn);

      // batch++
      cg.local.get(batch);
      cg.i32.const(1);
      cg.i32.add();
      cg.local.set(batch);

      cg.br(1);

      cg.end(); // block
    }
    cg.end(); // loop
  });

  cg.export(batchedFn, "cholesky");

  return new WebAssembly.Module(cg.finish());
}
