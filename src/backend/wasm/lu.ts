/**
 * WASM implementation of LU decomposition with partial pivoting.
 *
 * Computes P @ A = L @ U where:
 * - P is a permutation matrix
 * - L is lower triangular with unit diagonal
 * - U is upper triangular
 *
 * Output format matches JAX: combined LU matrix, pivots array, permutation array.
 *
 * Supports both f32 and f64 dtypes. The routine function takes a float type
 * (cg.f32 or cg.f64) and derives element size from it.
 * Note: pivots and permutation are always i32 regardless of float dtype.
 */

import { CodeGenerator } from "./wasmblr";

/**
 * Generate a WASM function for LU decomposition of a single matrix.
 *
 * Algorithm (Gaussian elimination with partial pivoting):
 *   for j = 0 to min(m,n)-1:
 *     // Find pivot (max absolute value in column j, rows j..m-1)
 *     maxRow = j
 *     maxVal = |LU[j,j]|
 *     for i = j+1 to m-1:
 *       if |LU[i,j]| > maxVal:
 *         maxVal = |LU[i,j]|
 *         maxRow = i
 *     pivots[j] = maxRow
 *
 *     // Swap rows j and maxRow
 *     if maxRow != j:
 *       swap row j with row maxRow in LU
 *       swap perm[j] with perm[maxRow]
 *
 *     // Eliminate below pivot
 *     if LU[j,j] != 0:
 *       for i = j+1 to m-1:
 *         factor = LU[i,j] / LU[j,j]
 *         LU[i,j] = factor  // Store L factor
 *         for k = j+1 to n-1:
 *           LU[i,k] -= factor * LU[j,k]
 *
 * Function signature:
 *   (inPtr, luPtr, pivotsPtr, permPtr, m, n) -> void
 *
 * @param cg - The code generator
 * @param ft - The float type (cg.f32 or cg.f64)
 * @returns The function index
 */
export function wasm_lu(
  cg: CodeGenerator,
  ft: CodeGenerator["f32"] | CodeGenerator["f64"],
): number {
  const elemSize = ft.name === "f32" ? 4 : 8;

  return cg.function(
    [cg.i32, cg.i32, cg.i32, cg.i32, cg.i32, cg.i32],
    [],
    () => {
      const i = cg.local.declare(cg.i32);
      const j = cg.local.declare(cg.i32);
      const k = cg.local.declare(cg.i32);
      const r = cg.local.declare(cg.i32); // min(m, n)
      const maxRow = cg.local.declare(cg.i32);
      const maxVal = cg.local.declare(ft);
      const curVal = cg.local.declare(ft);
      const diag = cg.local.declare(ft);
      const factor = cg.local.declare(ft);
      const tmp = cg.local.declare(ft);
      const tmpI = cg.local.declare(cg.i32);
      const offset1 = cg.local.declare(cg.i32);
      const offset2 = cg.local.declare(cg.i32);

      // Parameter aliases
      const inPtr = 0;
      const luPtr = 1;
      const pivotsPtr = 2;
      const permPtr = 3;
      const m = 4;
      const n = 5;

      // Copy input to LU output
      // for idx = 0 to m*n-1: LU[idx] = in[idx]
      cg.i32.const(0);
      cg.local.set(i);
      cg.loop(cg.void);
      {
        cg.block(cg.void);

        // if (i >= m * n) break
        cg.local.get(i);
        cg.local.get(m);
        cg.local.get(n);
        cg.i32.mul();
        cg.i32.ge_u();
        cg.br_if(0);

        // LU[i] = in[i]
        cg.local.get(luPtr);
        cg.local.get(i);
        cg.i32.const(elemSize);
        cg.i32.mul();
        cg.i32.add();

        cg.local.get(inPtr);
        cg.local.get(i);
        cg.i32.const(elemSize);
        cg.i32.mul();
        cg.i32.add();
        ft.load(0, 0);

        ft.store(0, 0);

        cg.local.get(i);
        cg.i32.const(1);
        cg.i32.add();
        cg.local.set(i);

        cg.br(1);
        cg.end();
      }
      cg.end();

      // Initialize permutation: perm[i] = i (always i32)
      cg.i32.const(0);
      cg.local.set(i);
      cg.loop(cg.void);
      {
        cg.block(cg.void);

        cg.local.get(i);
        cg.local.get(m);
        cg.i32.ge_u();
        cg.br_if(0);

        // perm[i] = i
        cg.local.get(permPtr);
        cg.local.get(i);
        cg.i32.const(4); // perm is always i32 (4 bytes)
        cg.i32.mul();
        cg.i32.add();
        cg.local.get(i);
        cg.i32.store(0, 0);

        cg.local.get(i);
        cg.i32.const(1);
        cg.i32.add();
        cg.local.set(i);

        cg.br(1);
        cg.end();
      }
      cg.end();

      // r = min(m, n)
      cg.local.get(m);
      cg.local.get(n);
      cg.i32.lt_u();
      cg.if(cg.i32);
      cg.local.get(m);
      cg.else();
      cg.local.get(n);
      cg.end();
      cg.local.set(r);

      // Main loop: j = 0 to r-1
      cg.i32.const(0);
      cg.local.set(j);

      cg.loop(cg.void);
      {
        cg.block(cg.void);

        cg.local.get(j);
        cg.local.get(r);
        cg.i32.ge_u();
        cg.br_if(0);

        // Find pivot: maxRow = j, maxVal = |LU[j,j]|
        cg.local.get(j);
        cg.local.set(maxRow);

        // maxVal = |LU[j * n + j]|
        cg.local.get(luPtr);
        cg.local.get(j);
        cg.local.get(n);
        cg.i32.mul();
        cg.local.get(j);
        cg.i32.add();
        cg.i32.const(elemSize);
        cg.i32.mul();
        cg.i32.add();
        ft.load(0, 0);
        ft.abs();
        cg.local.set(maxVal);

        // Search for max in column j, rows j+1..m-1
        cg.local.get(j);
        cg.i32.const(1);
        cg.i32.add();
        cg.local.set(i);

        cg.loop(cg.void);
        {
          cg.block(cg.void);

          cg.local.get(i);
          cg.local.get(m);
          cg.i32.ge_u();
          cg.br_if(0);

          // curVal = |LU[i * n + j]|
          cg.local.get(luPtr);
          cg.local.get(i);
          cg.local.get(n);
          cg.i32.mul();
          cg.local.get(j);
          cg.i32.add();
          cg.i32.const(elemSize);
          cg.i32.mul();
          cg.i32.add();
          ft.load(0, 0);
          ft.abs();
          cg.local.set(curVal);

          // if (curVal > maxVal) { maxVal = curVal; maxRow = i; }
          cg.local.get(curVal);
          cg.local.get(maxVal);
          ft.gt();
          cg.if(cg.void);
          {
            cg.local.get(curVal);
            cg.local.set(maxVal);
            cg.local.get(i);
            cg.local.set(maxRow);
          }
          cg.end();

          cg.local.get(i);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(i);

          cg.br(1);
          cg.end();
        }
        cg.end();

        // Store pivot (always i32)
        cg.local.get(pivotsPtr);
        cg.local.get(j);
        cg.i32.const(4); // pivots is always i32
        cg.i32.mul();
        cg.i32.add();
        cg.local.get(maxRow);
        cg.i32.store(0, 0);

        // Swap rows if needed
        cg.local.get(maxRow);
        cg.local.get(j);
        cg.i32.ne();
        cg.if(cg.void);
        {
          // Swap row j and row maxRow in LU
          cg.i32.const(0);
          cg.local.set(k);

          cg.loop(cg.void);
          {
            cg.block(cg.void);

            cg.local.get(k);
            cg.local.get(n);
            cg.i32.ge_u();
            cg.br_if(0);

            // offset1 = luPtr + (j * n + k) * elemSize
            cg.local.get(luPtr);
            cg.local.get(j);
            cg.local.get(n);
            cg.i32.mul();
            cg.local.get(k);
            cg.i32.add();
            cg.i32.const(elemSize);
            cg.i32.mul();
            cg.i32.add();
            cg.local.set(offset1);

            // offset2 = luPtr + (maxRow * n + k) * elemSize
            cg.local.get(luPtr);
            cg.local.get(maxRow);
            cg.local.get(n);
            cg.i32.mul();
            cg.local.get(k);
            cg.i32.add();
            cg.i32.const(elemSize);
            cg.i32.mul();
            cg.i32.add();
            cg.local.set(offset2);

            // tmp = LU[j,k]
            cg.local.get(offset1);
            ft.load(0, 0);
            cg.local.set(tmp);

            // LU[j,k] = LU[maxRow,k]
            cg.local.get(offset1);
            cg.local.get(offset2);
            ft.load(0, 0);
            ft.store(0, 0);

            // LU[maxRow,k] = tmp
            cg.local.get(offset2);
            cg.local.get(tmp);
            ft.store(0, 0);

            cg.local.get(k);
            cg.i32.const(1);
            cg.i32.add();
            cg.local.set(k);

            cg.br(1);
            cg.end();
          }
          cg.end();

          // Swap perm[j] and perm[maxRow] (always i32)
          // offset1 = permPtr + j * 4
          cg.local.get(permPtr);
          cg.local.get(j);
          cg.i32.const(4);
          cg.i32.mul();
          cg.i32.add();
          cg.local.set(offset1);

          // offset2 = permPtr + maxRow * 4
          cg.local.get(permPtr);
          cg.local.get(maxRow);
          cg.i32.const(4);
          cg.i32.mul();
          cg.i32.add();
          cg.local.set(offset2);

          // tmpI = perm[j]
          cg.local.get(offset1);
          cg.i32.load(0, 0);
          cg.local.set(tmpI);

          // perm[j] = perm[maxRow]
          cg.local.get(offset1);
          cg.local.get(offset2);
          cg.i32.load(0, 0);
          cg.i32.store(0, 0);

          // perm[maxRow] = tmpI
          cg.local.get(offset2);
          cg.local.get(tmpI);
          cg.i32.store(0, 0);
        }
        cg.end();

        // Eliminate: compute L factors and update U
        // diag = LU[j,j]
        cg.local.get(luPtr);
        cg.local.get(j);
        cg.local.get(n);
        cg.i32.mul();
        cg.local.get(j);
        cg.i32.add();
        cg.i32.const(elemSize);
        cg.i32.mul();
        cg.i32.add();
        ft.load(0, 0);
        cg.local.set(diag);

        // if (diag != 0)
        cg.local.get(diag);
        ft.const(0);
        ft.ne();
        cg.if(cg.void);
        {
          // for i = j+1 to m-1
          cg.local.get(j);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(i);

          cg.loop(cg.void);
          {
            cg.block(cg.void);

            cg.local.get(i);
            cg.local.get(m);
            cg.i32.ge_u();
            cg.br_if(0);

            // factor = LU[i,j] / diag
            cg.local.get(luPtr);
            cg.local.get(i);
            cg.local.get(n);
            cg.i32.mul();
            cg.local.get(j);
            cg.i32.add();
            cg.i32.const(elemSize);
            cg.i32.mul();
            cg.i32.add();
            cg.local.set(offset1);

            cg.local.get(offset1);
            ft.load(0, 0);
            cg.local.get(diag);
            ft.div();
            cg.local.set(factor);

            // LU[i,j] = factor (store L)
            cg.local.get(offset1);
            cg.local.get(factor);
            ft.store(0, 0);

            // Update row i, columns j+1..n-1
            cg.local.get(j);
            cg.i32.const(1);
            cg.i32.add();
            cg.local.set(k);

            cg.loop(cg.void);
            {
              cg.block(cg.void);

              cg.local.get(k);
              cg.local.get(n);
              cg.i32.ge_u();
              cg.br_if(0);

              // offset1 = luPtr + (i * n + k) * elemSize
              cg.local.get(luPtr);
              cg.local.get(i);
              cg.local.get(n);
              cg.i32.mul();
              cg.local.get(k);
              cg.i32.add();
              cg.i32.const(elemSize);
              cg.i32.mul();
              cg.i32.add();
              cg.local.set(offset1);

              // offset2 = luPtr + (j * n + k) * elemSize
              cg.local.get(luPtr);
              cg.local.get(j);
              cg.local.get(n);
              cg.i32.mul();
              cg.local.get(k);
              cg.i32.add();
              cg.i32.const(elemSize);
              cg.i32.mul();
              cg.i32.add();
              cg.local.set(offset2);

              // LU[i,k] -= factor * LU[j,k]
              cg.local.get(offset1);
              cg.local.get(offset1);
              ft.load(0, 0);
              cg.local.get(factor);
              cg.local.get(offset2);
              ft.load(0, 0);
              ft.mul();
              ft.sub();
              ft.store(0, 0);

              cg.local.get(k);
              cg.i32.const(1);
              cg.i32.add();
              cg.local.set(k);

              cg.br(1);
              cg.end();
            }
            cg.end();

            cg.local.get(i);
            cg.i32.const(1);
            cg.i32.add();
            cg.local.set(i);

            cg.br(1);
            cg.end();
          }
          cg.end();
        }
        cg.end();

        cg.local.get(j);
        cg.i32.const(1);
        cg.i32.add();
        cg.local.set(j);

        cg.br(1);
        cg.end();
      }
      cg.end();
    },
  );
}

/**
 * Generate a WASM module for batched LU decomposition.
 *
 * The exported function takes:
 *   (inPtr, luPtr, pivotsPtr, permPtr, m, n, batchSize)
 *
 * @param elementSize - Element size in bytes: 4 for f32, 8 for f64
 * @returns The compiled WebAssembly.Module
 */
export function createLUModule(elementSize: 4 | 8 = 4): WebAssembly.Module {
  const cg = new CodeGenerator();
  cg.memory.import("env", "memory");

  const ft = elementSize === 4 ? cg.f32 : cg.f64;
  const luFn = wasm_lu(cg, ft);

  // Batched wrapper
  const batchedFn = cg.function(
    [cg.i32, cg.i32, cg.i32, cg.i32, cg.i32, cg.i32, cg.i32],
    [],
    () => {
      const batch = cg.local.declare(cg.i32);
      const matrixSize = cg.local.declare(cg.i32); // m * n * elementSize
      const pivotSize = cg.local.declare(cg.i32); // min(m,n) * 4 (always i32)
      const permSize = cg.local.declare(cg.i32); // m * 4 (always i32)
      const inOffset = cg.local.declare(cg.i32);
      const luOffset = cg.local.declare(cg.i32);
      const pivotsOffset = cg.local.declare(cg.i32);
      const permOffset = cg.local.declare(cg.i32);
      const r = cg.local.declare(cg.i32);

      // Parameter aliases
      const inPtr = 0;
      const luPtr = 1;
      const pivotsPtr = 2;
      const permPtr = 3;
      const m = 4;
      const n = 5;
      const batchSize = 6;

      // matrixSize = m * n * elementSize
      cg.local.get(m);
      cg.local.get(n);
      cg.i32.mul();
      cg.i32.const(elementSize);
      cg.i32.mul();
      cg.local.set(matrixSize);

      // r = min(m, n)
      cg.local.get(m);
      cg.local.get(n);
      cg.i32.lt_u();
      cg.if(cg.i32);
      cg.local.get(m);
      cg.else();
      cg.local.get(n);
      cg.end();
      cg.local.set(r);

      // pivotSize = r * 4 (pivots are always i32)
      cg.local.get(r);
      cg.i32.const(4);
      cg.i32.mul();
      cg.local.set(pivotSize);

      // permSize = m * 4 (perm is always i32)
      cg.local.get(m);
      cg.i32.const(4);
      cg.i32.mul();
      cg.local.set(permSize);

      // batch = 0
      cg.i32.const(0);
      cg.local.set(batch);

      cg.loop(cg.void);
      {
        cg.block(cg.void);
        cg.local.get(batch);
        cg.local.get(batchSize);
        cg.i32.ge_u();
        cg.br_if(0);

        // inOffset = inPtr + batch * matrixSize
        cg.local.get(inPtr);
        cg.local.get(batch);
        cg.local.get(matrixSize);
        cg.i32.mul();
        cg.i32.add();
        cg.local.set(inOffset);

        // luOffset = luPtr + batch * matrixSize
        cg.local.get(luPtr);
        cg.local.get(batch);
        cg.local.get(matrixSize);
        cg.i32.mul();
        cg.i32.add();
        cg.local.set(luOffset);

        // pivotsOffset = pivotsPtr + batch * pivotSize
        cg.local.get(pivotsPtr);
        cg.local.get(batch);
        cg.local.get(pivotSize);
        cg.i32.mul();
        cg.i32.add();
        cg.local.set(pivotsOffset);

        // permOffset = permPtr + batch * permSize
        cg.local.get(permPtr);
        cg.local.get(batch);
        cg.local.get(permSize);
        cg.i32.mul();
        cg.i32.add();
        cg.local.set(permOffset);

        // Call LU
        cg.local.get(inOffset);
        cg.local.get(luOffset);
        cg.local.get(pivotsOffset);
        cg.local.get(permOffset);
        cg.local.get(m);
        cg.local.get(n);
        cg.call(luFn);

        // batch++
        cg.local.get(batch);
        cg.i32.const(1);
        cg.i32.add();
        cg.local.set(batch);

        cg.br(1);
        cg.end();
      }
      cg.end();
    },
  );

  cg.export(batchedFn, "lu");

  return new WebAssembly.Module(cg.finish());
}
