/**
 * WASM implementation of stable argsort.
 *
 * Uses bottom-up merge sort on indices, comparing via the data array.
 * Returns both sorted values and indices.
 *
 * Supports both f32 and f64 dtypes. The routine function takes a float type
 * (cg.f32 or cg.f64) and derives element size from it.
 * Indices are always i32 (4 bytes).
 */

import { CodeGenerator } from "./wasmblr";

/**
 * Generate a WASM function for argsort using merge sort.
 *
 * Sorts indices by comparing data values, producing both sorted data and indices.
 *
 * Function signature: (dataPtr, idxPtr, auxData, auxIdx, n) -> void
 * - dataPtr: input/output data array [n]
 * - idxPtr: output indices array [n] (int32)
 * - auxData: auxiliary data buffer [n]
 * - auxIdx: auxiliary index buffer [n] (int32)
 * - n: array length
 *
 * @param cg - The code generator
 * @param ft - The float type (cg.f32 or cg.f64)
 * @returns The function index
 */
export function wasm_argsort(
  cg: CodeGenerator,
  ft: CodeGenerator["f32"] | CodeGenerator["f64"],
): number {
  const elemSize = ft.name === "f32" ? 4 : 8;
  const idxSize = 4; // Indices are always i32

  // Internal merge function for argsort
  // merge(srcData, srcIdx, dstData, dstIdx, left, mid, right)
  const mergeFn = cg.function(
    [cg.i32, cg.i32, cg.i32, cg.i32, cg.i32, cg.i32, cg.i32],
    [],
    () => {
      const i = cg.local.declare(cg.i32); // left index
      const j = cg.local.declare(cg.i32); // right index
      const k = cg.local.declare(cg.i32); // output index
      const vi = cg.local.declare(ft);
      const vj = cg.local.declare(ft);
      const ii = cg.local.declare(cg.i32);
      const ij = cg.local.declare(cg.i32);

      // Parameter aliases
      const srcData = 0;
      const srcIdx = 1;
      const dstData = 2;
      const dstIdx = 3;
      const left = 4;
      const mid = 5;
      const right = 6;

      // i = left, j = mid, k = left
      cg.local.get(left);
      cg.local.set(i);
      cg.local.get(mid);
      cg.local.set(j);
      cg.local.get(left);
      cg.local.set(k);

      // Main merge loop
      cg.loop(cg.void);
      {
        cg.block(cg.void);

        cg.local.get(k);
        cg.local.get(right);
        cg.i32.ge_u();
        cg.br_if(0);

        // Check if left or right exhausted
        cg.local.get(i);
        cg.local.get(mid);
        cg.i32.ge_u();
        cg.if(cg.void);
        {
          // Left exhausted, copy from right
          // dstData[k] = srcData[j]
          cg.local.get(dstData);
          cg.local.get(k);
          cg.i32.const(elemSize);
          cg.i32.mul();
          cg.i32.add();
          cg.local.get(srcData);
          cg.local.get(j);
          cg.i32.const(elemSize);
          cg.i32.mul();
          cg.i32.add();
          ft.load(0, 0);
          ft.store(0, 0);

          // dstIdx[k] = srcIdx[j] (indices are always i32)
          cg.local.get(dstIdx);
          cg.local.get(k);
          cg.i32.const(idxSize);
          cg.i32.mul();
          cg.i32.add();
          cg.local.get(srcIdx);
          cg.local.get(j);
          cg.i32.const(idxSize);
          cg.i32.mul();
          cg.i32.add();
          cg.i32.load(0, 0);
          cg.i32.store(0, 0);

          cg.local.get(j);
          cg.i32.const(1);
          cg.i32.add();
          cg.local.set(j);
        }
        cg.else();
        {
          cg.local.get(j);
          cg.local.get(right);
          cg.i32.ge_u();
          cg.if(cg.void);
          {
            // Right exhausted, copy from left
            cg.local.get(dstData);
            cg.local.get(k);
            cg.i32.const(elemSize);
            cg.i32.mul();
            cg.i32.add();
            cg.local.get(srcData);
            cg.local.get(i);
            cg.i32.const(elemSize);
            cg.i32.mul();
            cg.i32.add();
            ft.load(0, 0);
            ft.store(0, 0);

            cg.local.get(dstIdx);
            cg.local.get(k);
            cg.i32.const(idxSize);
            cg.i32.mul();
            cg.i32.add();
            cg.local.get(srcIdx);
            cg.local.get(i);
            cg.i32.const(idxSize);
            cg.i32.mul();
            cg.i32.add();
            cg.i32.load(0, 0);
            cg.i32.store(0, 0);

            cg.local.get(i);
            cg.i32.const(1);
            cg.i32.add();
            cg.local.set(i);
          }
          cg.else();
          {
            // Compare srcData[i] and srcData[j]
            cg.local.get(srcData);
            cg.local.get(i);
            cg.i32.const(elemSize);
            cg.i32.mul();
            cg.i32.add();
            ft.load(0, 0);
            cg.local.set(vi);

            cg.local.get(srcData);
            cg.local.get(j);
            cg.i32.const(elemSize);
            cg.i32.mul();
            cg.i32.add();
            ft.load(0, 0);
            cg.local.set(vj);

            // Get original indices for stable sort tiebreaker
            cg.local.get(srcIdx);
            cg.local.get(i);
            cg.i32.const(idxSize);
            cg.i32.mul();
            cg.i32.add();
            cg.i32.load(0, 0);
            cg.local.set(ii);

            cg.local.get(srcIdx);
            cg.local.get(j);
            cg.i32.const(idxSize);
            cg.i32.mul();
            cg.i32.add();
            cg.i32.load(0, 0);
            cg.local.set(ij);

            // NaN-aware stable sort:
            // Use left if:
            // - vj is NaN (NaN sorts to end, so left wins)
            // - OR (!isNaN(vi) && vi < vj)
            // - OR (!isNaN(vi) && !isNaN(vj) && vi == vj && ii < ij)
            // Simplified: isNaN(vj) || (!isNaN(vi) && (vi < vj || (vi == vj && ii < ij)))

            // Check: isNaN(vj)
            cg.local.get(vj);
            cg.local.get(vj);
            ft.ne(); // isNaN(vj) - NaN != NaN is true

            // Check: !isNaN(vi) && vi < vj
            cg.local.get(vi);
            cg.local.get(vi);
            ft.eq(); // !isNaN(vi)
            cg.local.get(vi);
            cg.local.get(vj);
            ft.lt(); // vi < vj
            cg.i32.and(); // !isNaN(vi) && vi < vj

            cg.i32.or(); // isNaN(vj) || (!isNaN(vi) && vi < vj)

            cg.if(cg.void);
            {
              // Use left
              cg.local.get(dstData);
              cg.local.get(k);
              cg.i32.const(elemSize);
              cg.i32.mul();
              cg.i32.add();
              cg.local.get(vi);
              ft.store(0, 0);

              cg.local.get(dstIdx);
              cg.local.get(k);
              cg.i32.const(idxSize);
              cg.i32.mul();
              cg.i32.add();
              cg.local.get(ii);
              cg.i32.store(0, 0);

              cg.local.get(i);
              cg.i32.const(1);
              cg.i32.add();
              cg.local.set(i);
            }
            cg.else();
            {
              // Check for equal values (stable tiebreaker)
              // Both are not NaN here (since we didn't take the left branch due to isNaN(vj))
              // and vi >= vj. Check if vi == vj && ii < ij
              cg.local.get(vi);
              cg.local.get(vj);
              ft.eq();
              cg.local.get(ii);
              cg.local.get(ij);
              cg.i32.lt_s();
              cg.i32.and();
              cg.if(cg.void);
              {
                // Equal values, smaller original index wins (stable)
                cg.local.get(dstData);
                cg.local.get(k);
                cg.i32.const(elemSize);
                cg.i32.mul();
                cg.i32.add();
                cg.local.get(vi);
                ft.store(0, 0);

                cg.local.get(dstIdx);
                cg.local.get(k);
                cg.i32.const(idxSize);
                cg.i32.mul();
                cg.i32.add();
                cg.local.get(ii);
                cg.i32.store(0, 0);

                cg.local.get(i);
                cg.i32.const(1);
                cg.i32.add();
                cg.local.set(i);
              }
              cg.else();
              {
                // Use right
                cg.local.get(dstData);
                cg.local.get(k);
                cg.i32.const(elemSize);
                cg.i32.mul();
                cg.i32.add();
                cg.local.get(vj);
                ft.store(0, 0);

                cg.local.get(dstIdx);
                cg.local.get(k);
                cg.i32.const(idxSize);
                cg.i32.mul();
                cg.i32.add();
                cg.local.get(ij);
                cg.i32.store(0, 0);

                cg.local.get(j);
                cg.i32.const(1);
                cg.i32.add();
                cg.local.set(j);
              }
              cg.end();
            }
            cg.end();
          }
          cg.end();
        }
        cg.end();

        cg.local.get(k);
        cg.i32.const(1);
        cg.i32.add();
        cg.local.set(k);

        cg.br(1);
        cg.end();
      }
      cg.end();
    },
  );

  // Main argsort function
  return cg.function([cg.i32, cg.i32, cg.i32, cg.i32, cg.i32], [], () => {
    const width = cg.local.declare(cg.i32);
    const i = cg.local.declare(cg.i32);
    const left = cg.local.declare(cg.i32);
    const mid = cg.local.declare(cg.i32);
    const right = cg.local.declare(cg.i32);
    const srcData = cg.local.declare(cg.i32);
    const srcIdx = cg.local.declare(cg.i32);
    const dstData = cg.local.declare(cg.i32);
    const dstIdx = cg.local.declare(cg.i32);
    const pass = cg.local.declare(cg.i32);

    // Parameter aliases
    const dataPtr = 0;
    const idxPtr = 1;
    const auxData = 2;
    const auxIdx = 3;
    const n = 4;

    // Initialize indices: idxPtr[i] = i
    cg.i32.const(0);
    cg.local.set(i);

    cg.loop(cg.void);
    {
      cg.block(cg.void);

      cg.local.get(i);
      cg.local.get(n);
      cg.i32.ge_u();
      cg.br_if(0);

      cg.local.get(idxPtr);
      cg.local.get(i);
      cg.i32.const(4);
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

    // width = 1, pass = 0
    cg.i32.const(1);
    cg.local.set(width);
    cg.i32.const(0);
    cg.local.set(pass);

    // Outer loop
    cg.loop(cg.void);
    {
      cg.block(cg.void);

      cg.local.get(width);
      cg.local.get(n);
      cg.i32.ge_u();
      cg.br_if(0);

      // Set src/dst based on pass parity
      cg.local.get(pass);
      cg.i32.const(1);
      cg.i32.and();
      cg.i32.eqz();
      cg.if(cg.void);
      {
        cg.local.get(dataPtr);
        cg.local.set(srcData);
        cg.local.get(idxPtr);
        cg.local.set(srcIdx);
        cg.local.get(auxData);
        cg.local.set(dstData);
        cg.local.get(auxIdx);
        cg.local.set(dstIdx);
      }
      cg.else();
      {
        cg.local.get(auxData);
        cg.local.set(srcData);
        cg.local.get(auxIdx);
        cg.local.set(srcIdx);
        cg.local.get(dataPtr);
        cg.local.set(dstData);
        cg.local.get(idxPtr);
        cg.local.set(dstIdx);
      }
      cg.end();

      // i = 0
      cg.i32.const(0);
      cg.local.set(i);

      // Inner loop
      cg.loop(cg.void);
      {
        cg.block(cg.void);

        cg.local.get(i);
        cg.local.get(n);
        cg.i32.ge_u();
        cg.br_if(0);

        cg.local.get(i);
        cg.local.set(left);

        // mid = min(i + width, n)
        cg.local.get(i);
        cg.local.get(width);
        cg.i32.add();
        cg.local.set(mid);
        cg.local.get(mid);
        cg.local.get(n);
        cg.i32.gt_u();
        cg.if(cg.void);
        {
          cg.local.get(n);
          cg.local.set(mid);
        }
        cg.end();

        // right = min(i + 2*width, n)
        cg.local.get(i);
        cg.local.get(width);
        cg.i32.const(2);
        cg.i32.mul();
        cg.i32.add();
        cg.local.set(right);
        cg.local.get(right);
        cg.local.get(n);
        cg.i32.gt_u();
        cg.if(cg.void);
        {
          cg.local.get(n);
          cg.local.set(right);
        }
        cg.end();

        // merge
        cg.local.get(srcData);
        cg.local.get(srcIdx);
        cg.local.get(dstData);
        cg.local.get(dstIdx);
        cg.local.get(left);
        cg.local.get(mid);
        cg.local.get(right);
        cg.call(mergeFn);

        // i += 2 * width
        cg.local.get(i);
        cg.local.get(width);
        cg.i32.const(2);
        cg.i32.mul();
        cg.i32.add();
        cg.local.set(i);

        cg.br(1);
        cg.end();
      }
      cg.end();

      // width *= 2, pass++
      cg.local.get(width);
      cg.i32.const(2);
      cg.i32.mul();
      cg.local.set(width);

      cg.local.get(pass);
      cg.i32.const(1);
      cg.i32.add();
      cg.local.set(pass);

      cg.br(1);
      cg.end();
    }
    cg.end();

    // If odd passes, copy aux back to data/idx
    cg.local.get(pass);
    cg.i32.const(1);
    cg.i32.and();
    cg.if(cg.void);
    {
      cg.i32.const(0);
      cg.local.set(i);

      cg.loop(cg.void);
      {
        cg.block(cg.void);

        cg.local.get(i);
        cg.local.get(n);
        cg.i32.ge_u();
        cg.br_if(0);

        // dataPtr[i] = auxData[i] (float data uses elemSize)
        cg.local.get(dataPtr);
        cg.local.get(i);
        cg.i32.const(elemSize);
        cg.i32.mul();
        cg.i32.add();
        cg.local.get(auxData);
        cg.local.get(i);
        cg.i32.const(elemSize);
        cg.i32.mul();
        cg.i32.add();
        ft.load(0, 0);
        ft.store(0, 0);

        // idxPtr[i] = auxIdx[i] (indices always use 4 bytes)
        cg.local.get(idxPtr);
        cg.local.get(i);
        cg.i32.const(idxSize);
        cg.i32.mul();
        cg.i32.add();
        cg.local.get(auxIdx);
        cg.local.get(i);
        cg.i32.const(idxSize);
        cg.i32.mul();
        cg.i32.add();
        cg.i32.load(0, 0);
        cg.i32.store(0, 0);

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
  });
}

/**
 * Generate a WASM module for batched argsort.
 *
 * The exported function takes:
 *   (inPtr, outDataPtr, outIdxPtr, auxData, auxIdx, n, batchSize)
 *
 * @param elementSize - Element size in bytes: 4 for f32, 8 for f64
 * @returns The compiled WebAssembly.Module
 */
export function createArgsortModule(
  elementSize: 4 | 8 = 4,
): WebAssembly.Module {
  const cg = new CodeGenerator();
  cg.memory.import("env", "memory");

  const ft = elementSize === 4 ? cg.f32 : cg.f64;
  const argsortFn = wasm_argsort(cg, ft);
  const idxSize = 4; // indices are always i32

  // Batched wrapper
  const batchedFn = cg.function(
    [cg.i32, cg.i32, cg.i32, cg.i32, cg.i32, cg.i32, cg.i32],
    [],
    () => {
      const batch = cg.local.declare(cg.i32);
      const dataArraySize = cg.local.declare(cg.i32); // n * elementSize
      const idxArraySize = cg.local.declare(cg.i32); // n * 4 (indices are always i32)
      const inOffset = cg.local.declare(cg.i32);
      const outDataOffset = cg.local.declare(cg.i32);
      const outIdxOffset = cg.local.declare(cg.i32);
      const i = cg.local.declare(cg.i32);

      // Parameter aliases
      const inPtr = 0;
      const outDataPtr = 1;
      const outIdxPtr = 2;
      const auxData = 3;
      const auxIdx = 4;
      const n = 5;
      const batchSize = 6;

      // dataArraySize = n * elementSize
      cg.local.get(n);
      cg.i32.const(elementSize);
      cg.i32.mul();
      cg.local.set(dataArraySize);

      // idxArraySize = n * 4
      cg.local.get(n);
      cg.i32.const(idxSize);
      cg.i32.mul();
      cg.local.set(idxArraySize);

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

        // Compute offsets (data uses elementSize, indices use 4)
        cg.local.get(inPtr);
        cg.local.get(batch);
        cg.local.get(dataArraySize);
        cg.i32.mul();
        cg.i32.add();
        cg.local.set(inOffset);

        cg.local.get(outDataPtr);
        cg.local.get(batch);
        cg.local.get(dataArraySize);
        cg.i32.mul();
        cg.i32.add();
        cg.local.set(outDataOffset);

        cg.local.get(outIdxPtr);
        cg.local.get(batch);
        cg.local.get(idxArraySize);
        cg.i32.mul();
        cg.i32.add();
        cg.local.set(outIdxOffset);

        // Copy input to outData
        cg.i32.const(0);
        cg.local.set(i);

        cg.loop(cg.void);
        {
          cg.block(cg.void);

          cg.local.get(i);
          cg.local.get(n);
          cg.i32.ge_u();
          cg.br_if(0);

          cg.local.get(outDataOffset);
          cg.local.get(i);
          cg.i32.const(elementSize);
          cg.i32.mul();
          cg.i32.add();

          cg.local.get(inOffset);
          cg.local.get(i);
          cg.i32.const(elementSize);
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

        // Argsort
        cg.local.get(outDataOffset);
        cg.local.get(outIdxOffset);
        cg.local.get(auxData);
        cg.local.get(auxIdx);
        cg.local.get(n);
        cg.call(argsortFn);

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

  cg.export(batchedFn, "argsort");

  return new WebAssembly.Module(cg.finish());
}
