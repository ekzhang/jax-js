/**
 * WASM implementation of Triangular Solve.
 *
 * Solves A @ x = b where A is upper or lower triangular.
 * Uses back-substitution (upper) or forward-substitution (lower).
 *
 * The internal routine assumes:
 * - A is shape [n, n] (upper triangular, row-major)
 * - b is shape [n] (column vector)
 * - x is shape [n] (output)
 *
 * For batched/matrix B, the caller loops over columns.
 *
 * Supports both f32 and f64 dtypes. The routine functions take a float type
 * (cg.f32 or cg.f64) and derive element size from it.
 */

import { CodeGenerator } from "./wasmblr";

/**
 * Generate a WASM function for upper-triangular solve (back-substitution).
 *
 * Solves A @ x = b where A is upper triangular.
 *
 * Algorithm:
 *   for i = n-1 down to 0:
 *     sum = b[i]
 *     for j = i+1 to n-1:
 *       sum -= A[i,j] * x[j]
 *     x[i] = sum / A[i,i]  (or just sum if unitDiagonal)
 *
 * Function signature: (aPtr: i32, bPtr: i32, xPtr: i32, n: i32, unitDiagonal: i32) -> void
 *
 * @param cg - The code generator
 * @param ft - The float type (cg.f32 or cg.f64)
 * @returns The function index
 */
export function wasm_triangular_solve_upper(
  cg: CodeGenerator,
  ft: CodeGenerator["f32"] | CodeGenerator["f64"],
): number {
  const elemSize = ft.name === "f32" ? 4 : 8;

  return cg.function([cg.i32, cg.i32, cg.i32, cg.i32, cg.i32], [], () => {
    const i = cg.local.declare(cg.i32);
    const j = cg.local.declare(cg.i32);
    const sum = cg.local.declare(ft);
    const aij = cg.local.declare(ft);
    const xj = cg.local.declare(ft);
    const aii = cg.local.declare(ft);

    // Parameter aliases
    const aPtr = 0;
    const bPtr = 1;
    const xPtr = 2;
    const n = 3;
    const unitDiagonal = 4;

    // i = n - 1
    cg.local.get(n);
    cg.i32.const(1);
    cg.i32.sub();
    cg.local.set(i);

    // Outer loop: for i = n-1 down to 0
    cg.loop(cg.void);
    {
      cg.block(cg.void);

      // if (i < 0) break
      cg.local.get(i);
      cg.i32.const(0);
      cg.i32.lt_s();
      cg.br_if(0);

      // sum = b[i]
      cg.local.get(bPtr);
      cg.local.get(i);
      cg.i32.const(elemSize);
      cg.i32.mul();
      cg.i32.add();
      ft.load(0, 0);
      cg.local.set(sum);

      // j = i + 1
      cg.local.get(i);
      cg.i32.const(1);
      cg.i32.add();
      cg.local.set(j);

      // Inner loop: for j = i+1 to n-1
      cg.loop(cg.void);
      {
        cg.block(cg.void);

        // if (j >= n) break inner
        cg.local.get(j);
        cg.local.get(n);
        cg.i32.ge_u();
        cg.br_if(0);

        // aij = A[i * n + j]
        cg.local.get(aPtr);
        cg.local.get(i);
        cg.local.get(n);
        cg.i32.mul();
        cg.local.get(j);
        cg.i32.add();
        cg.i32.const(elemSize);
        cg.i32.mul();
        cg.i32.add();
        ft.load(0, 0);
        cg.local.set(aij);

        // xj = x[j]
        cg.local.get(xPtr);
        cg.local.get(j);
        cg.i32.const(elemSize);
        cg.i32.mul();
        cg.i32.add();
        ft.load(0, 0);
        cg.local.set(xj);

        // sum -= aij * xj
        cg.local.get(sum);
        cg.local.get(aij);
        cg.local.get(xj);
        ft.mul();
        ft.sub();
        cg.local.set(sum);

        // j++
        cg.local.get(j);
        cg.i32.const(1);
        cg.i32.add();
        cg.local.set(j);

        cg.br(1); // continue inner loop
        cg.end(); // block
      }
      cg.end(); // inner loop

      // if (!unitDiagonal) sum /= A[i,i]
      cg.local.get(unitDiagonal);
      cg.i32.eqz();
      cg.if(cg.void);
      {
        // aii = A[i * n + i]
        cg.local.get(aPtr);
        cg.local.get(i);
        cg.local.get(n);
        cg.i32.mul();
        cg.local.get(i);
        cg.i32.add();
        cg.i32.const(elemSize);
        cg.i32.mul();
        cg.i32.add();
        ft.load(0, 0);
        cg.local.set(aii);

        // sum /= aii
        cg.local.get(sum);
        cg.local.get(aii);
        ft.div();
        cg.local.set(sum);
      }
      cg.end(); // if

      // x[i] = sum
      cg.local.get(xPtr);
      cg.local.get(i);
      cg.i32.const(elemSize);
      cg.i32.mul();
      cg.i32.add();
      cg.local.get(sum);
      ft.store(0, 0);

      // i--
      cg.local.get(i);
      cg.i32.const(1);
      cg.i32.sub();
      cg.local.set(i);

      cg.br(1); // continue outer loop
      cg.end(); // block
    }
    cg.end(); // outer loop
  });
}

/**
 * Generate a WASM function for lower-triangular solve (forward-substitution).
 *
 * Solves A @ x = b where A is lower triangular.
 *
 * Algorithm:
 *   for i = 0 to n-1:
 *     sum = b[i]
 *     for j = 0 to i-1:
 *       sum -= A[i,j] * x[j]
 *     x[i] = sum / A[i,i]  (or just sum if unitDiagonal)
 *
 * Function signature: (aPtr: i32, bPtr: i32, xPtr: i32, n: i32, unitDiagonal: i32) -> void
 *
 * @param cg - The code generator
 * @param ft - The float type (cg.f32 or cg.f64)
 * @returns The function index
 */
export function wasm_triangular_solve_lower(
  cg: CodeGenerator,
  ft: CodeGenerator["f32"] | CodeGenerator["f64"],
): number {
  const elemSize = ft.name === "f32" ? 4 : 8;

  return cg.function([cg.i32, cg.i32, cg.i32, cg.i32, cg.i32], [], () => {
    const i = cg.local.declare(cg.i32);
    const j = cg.local.declare(cg.i32);
    const sum = cg.local.declare(ft);
    const aij = cg.local.declare(ft);
    const xj = cg.local.declare(ft);
    const aii = cg.local.declare(ft);

    // Parameter aliases
    const aPtr = 0;
    const bPtr = 1;
    const xPtr = 2;
    const n = 3;
    const unitDiagonal = 4;

    // i = 0
    cg.i32.const(0);
    cg.local.set(i);

    // Outer loop: for i = 0 to n-1
    cg.loop(cg.void);
    {
      cg.block(cg.void);

      // if (i >= n) break
      cg.local.get(i);
      cg.local.get(n);
      cg.i32.ge_u();
      cg.br_if(0);

      // sum = b[i]
      cg.local.get(bPtr);
      cg.local.get(i);
      cg.i32.const(elemSize);
      cg.i32.mul();
      cg.i32.add();
      ft.load(0, 0);
      cg.local.set(sum);

      // j = 0
      cg.i32.const(0);
      cg.local.set(j);

      // Inner loop: for j = 0 to i-1
      cg.loop(cg.void);
      {
        cg.block(cg.void);

        // if (j >= i) break inner
        cg.local.get(j);
        cg.local.get(i);
        cg.i32.ge_u();
        cg.br_if(0);

        // aij = A[i * n + j]
        cg.local.get(aPtr);
        cg.local.get(i);
        cg.local.get(n);
        cg.i32.mul();
        cg.local.get(j);
        cg.i32.add();
        cg.i32.const(elemSize);
        cg.i32.mul();
        cg.i32.add();
        ft.load(0, 0);
        cg.local.set(aij);

        // xj = x[j]
        cg.local.get(xPtr);
        cg.local.get(j);
        cg.i32.const(elemSize);
        cg.i32.mul();
        cg.i32.add();
        ft.load(0, 0);
        cg.local.set(xj);

        // sum -= aij * xj
        cg.local.get(sum);
        cg.local.get(aij);
        cg.local.get(xj);
        ft.mul();
        ft.sub();
        cg.local.set(sum);

        // j++
        cg.local.get(j);
        cg.i32.const(1);
        cg.i32.add();
        cg.local.set(j);

        cg.br(1); // continue inner loop
        cg.end(); // block
      }
      cg.end(); // inner loop

      // if (!unitDiagonal) sum /= A[i,i]
      cg.local.get(unitDiagonal);
      cg.i32.eqz();
      cg.if(cg.void);
      {
        // aii = A[i * n + i]
        cg.local.get(aPtr);
        cg.local.get(i);
        cg.local.get(n);
        cg.i32.mul();
        cg.local.get(i);
        cg.i32.add();
        cg.i32.const(elemSize);
        cg.i32.mul();
        cg.i32.add();
        ft.load(0, 0);
        cg.local.set(aii);

        // sum /= aii
        cg.local.get(sum);
        cg.local.get(aii);
        ft.div();
        cg.local.set(sum);
      }
      cg.end(); // if

      // x[i] = sum
      cg.local.get(xPtr);
      cg.local.get(i);
      cg.i32.const(elemSize);
      cg.i32.mul();
      cg.i32.add();
      cg.local.get(sum);
      ft.store(0, 0);

      // i++
      cg.local.get(i);
      cg.i32.const(1);
      cg.i32.add();
      cg.local.set(i);

      cg.br(1); // continue outer loop
      cg.end(); // block
    }
    cg.end(); // outer loop
  });
}

/**
 * Generate a WASM module for batched triangular solve.
 *
 * The exported function takes:
 *   (aPtr, bPtr, xPtr, n, batchRows, numBatches, unitDiagonal, lower)
 *
 * Where:
 *   - aPtr: [numBatches, n, n] matrix batch
 *   - bPtr: [numBatches, batchRows, n] RHS vectors
 *   - xPtr: [numBatches, batchRows, n] solution vectors
 *   - n: matrix dimension
 *   - batchRows: number of RHS vectors per matrix
 *   - numBatches: number of matrices
 *   - unitDiagonal: 1 if diagonal is implicitly 1
 *   - lower: 1 for lower triangular, 0 for upper
 *
 * @param elementSize - Element size in bytes: 4 for f32, 8 for f64
 * @returns The compiled WebAssembly.Module
 */
export function createTriangularSolveModule(
  elementSize: 4 | 8 = 4,
): WebAssembly.Module {
  const cg = new CodeGenerator();
  cg.memory.import("env", "memory");

  const ft = elementSize === 4 ? cg.f32 : cg.f64;
  const solveLower = wasm_triangular_solve_lower(cg, ft);
  const solveUpper = wasm_triangular_solve_upper(cg, ft);

  // Batched wrapper
  const batchedFn = cg.function(
    [cg.i32, cg.i32, cg.i32, cg.i32, cg.i32, cg.i32, cg.i32, cg.i32],
    [],
    () => {
      const batch = cg.local.declare(cg.i32);
      const row = cg.local.declare(cg.i32);
      const matrixSize = cg.local.declare(cg.i32); // n * n * elementSize
      const vectorSize = cg.local.declare(cg.i32); // n * elementSize
      const aOffset = cg.local.declare(cg.i32);
      const bOffset = cg.local.declare(cg.i32);
      const xOffset = cg.local.declare(cg.i32);

      // Parameter aliases
      const aPtr = 0;
      const bPtr = 1;
      const xPtr = 2;
      const n = 3;
      const batchRows = 4;
      const numBatches = 5;
      const unitDiagonal = 6;
      const lower = 7;

      // matrixSize = n * n * elementSize
      cg.local.get(n);
      cg.local.get(n);
      cg.i32.mul();
      cg.i32.const(elementSize);
      cg.i32.mul();
      cg.local.set(matrixSize);

      // vectorSize = n * elementSize
      cg.local.get(n);
      cg.i32.const(elementSize);
      cg.i32.mul();
      cg.local.set(vectorSize);

      // batch = 0
      cg.i32.const(0);
      cg.local.set(batch);

      // Outer loop over batches
      cg.loop(cg.void);
      {
        cg.block(cg.void);
        cg.local.get(batch);
        cg.local.get(numBatches);
        cg.i32.ge_u();
        cg.br_if(0);

        // aOffset = aPtr + batch * matrixSize
        cg.local.get(aPtr);
        cg.local.get(batch);
        cg.local.get(matrixSize);
        cg.i32.mul();
        cg.i32.add();
        cg.local.set(aOffset);

        // row = 0
        cg.i32.const(0);
        cg.local.set(row);

        // Inner loop over rows (vectors)
        cg.loop(cg.void);
        {
          cg.block(cg.void);
          cg.local.get(row);
          cg.local.get(batchRows);
          cg.i32.ge_u();
          cg.br_if(0);

          // bOffset = bPtr + (batch * batchRows + row) * vectorSize
          cg.local.get(bPtr);
          cg.local.get(batch);
          cg.local.get(batchRows);
          cg.i32.mul();
          cg.local.get(row);
          cg.i32.add();
          cg.local.get(vectorSize);
          cg.i32.mul();
          cg.i32.add();
          cg.local.set(bOffset);

          // xOffset = xPtr + (batch * batchRows + row) * vectorSize
          cg.local.get(xPtr);
          cg.local.get(batch);
          cg.local.get(batchRows);
          cg.i32.mul();
          cg.local.get(row);
          cg.i32.add();
          cg.local.get(vectorSize);
          cg.i32.mul();
          cg.i32.add();
          cg.local.set(xOffset);

          // Call appropriate solver based on 'lower' flag
          cg.local.get(lower);
          cg.if(cg.void);
          {
            cg.local.get(aOffset);
            cg.local.get(bOffset);
            cg.local.get(xOffset);
            cg.local.get(n);
            cg.local.get(unitDiagonal);
            cg.call(solveLower);
          }
          cg.else();
          {
            cg.local.get(aOffset);
            cg.local.get(bOffset);
            cg.local.get(xOffset);
            cg.local.get(n);
            cg.local.get(unitDiagonal);
            cg.call(solveUpper);
          }
          cg.end();

          // row++
          cg.local.get(row);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(row);

          cg.br(1);
          cg.end(); // block
        }
        cg.end(); // inner loop

        // batch++
        cg.local.get(batch);
        cg.i32.const(1);
        cg.i32.add();
        cg.local.set(batch);

        cg.br(1);
        cg.end(); // block
      }
      cg.end(); // outer loop
    },
  );

  cg.export(batchedFn, "triangularSolve");

  return new WebAssembly.Module(cg.finish());
}
