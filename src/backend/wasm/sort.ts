/**
 * WASM implementation of stable merge sort.
 *
 * Uses bottom-up merge sort for O(n log n) stable sorting.
 * Requires O(n) auxiliary space.
 *
 * Supports both f32 and f64 dtypes. The routine function takes a float type
 * (cg.f32 or cg.f64) and derives element size from it.
 */

import { CodeGenerator } from "./wasmblr";

/**
 * Generate a WASM function for merge sort on a single array segment.
 *
 * Uses bottom-up merge sort:
 *   for width = 1, 2, 4, ... while width < n:
 *     for i = 0, 2*width, 4*width, ... while i < n:
 *       merge(arr, i, min(i+width, n), min(i+2*width, n), aux)
 *
 * Function signature: (arrPtr, auxPtr, n) -> void
 * Both arrPtr and auxPtr point to n floats. Result ends up in arrPtr.
 *
 * @param cg - The code generator
 * @param ft - The float type (cg.f32 or cg.f64)
 * @returns The function index
 */
export function wasm_merge_sort(
  cg: CodeGenerator,
  ft: CodeGenerator["f32"] | CodeGenerator["f64"],
): number {
  const elemSize = ft.name === "f32" ? 4 : 8;

  // Internal merge function
  // merge(arrPtr, auxPtr, left, mid, right) - merges arr[left:mid] and arr[mid:right] into aux[left:right]
  const mergeFn = cg.function(
    [cg.i32, cg.i32, cg.i32, cg.i32, cg.i32],
    [],
    () => {
      const i = cg.local.declare(cg.i32); // left index
      const j = cg.local.declare(cg.i32); // right index
      const k = cg.local.declare(cg.i32); // output index
      const vi = cg.local.declare(ft);
      const vj = cg.local.declare(ft);

      // Parameter aliases
      const arrPtr = 0;
      const auxPtr = 1;
      const left = 2;
      const mid = 3;
      const right = 4;

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

        // if (k >= right) break
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
          cg.local.get(auxPtr);
          cg.local.get(k);
          cg.i32.const(elemSize);
          cg.i32.mul();
          cg.i32.add();

          cg.local.get(arrPtr);
          cg.local.get(j);
          cg.i32.const(elemSize);
          cg.i32.mul();
          cg.i32.add();
          ft.load(0, 0);

          ft.store(0, 0);

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
            cg.local.get(auxPtr);
            cg.local.get(k);
            cg.i32.const(elemSize);
            cg.i32.mul();
            cg.i32.add();

            cg.local.get(arrPtr);
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
          }
          cg.else();
          {
            // Compare arr[i] and arr[j]
            cg.local.get(arrPtr);
            cg.local.get(i);
            cg.i32.const(elemSize);
            cg.i32.mul();
            cg.i32.add();
            ft.load(0, 0);
            cg.local.set(vi);

            cg.local.get(arrPtr);
            cg.local.get(j);
            cg.i32.const(elemSize);
            cg.i32.mul();
            cg.i32.add();
            ft.load(0, 0);
            cg.local.set(vj);

            // NaN-aware comparison: NaN should sort to the end
            // useLeft = (!isNaN(vi) && isNaN(vj)) || (!isNaN(vi) && !isNaN(vj) && vi <= vj)
            //         = !isNaN(vi) && (isNaN(vj) || vi <= vj)
            // Also for stability: if both are NaN, use left
            // So: useLeft = isNaN(vj) || (!isNaN(vi) && vi <= vj) || (isNaN(vi) && isNaN(vj))
            //            = isNaN(vj) || (!isNaN(vi) && vi <= vj)
            // Check: vj is NaN
            cg.local.get(vj);
            cg.local.get(vj);
            ft.ne(); // isNaN(vj)

            // Check: !isNaN(vi) && vi <= vj
            cg.local.get(vi);
            cg.local.get(vi);
            ft.eq(); // !isNaN(vi)
            cg.local.get(vi);
            cg.local.get(vj);
            ft.le(); // vi <= vj
            cg.i32.and(); // !isNaN(vi) && vi <= vj

            cg.i32.or(); // isNaN(vj) || (!isNaN(vi) && vi <= vj)

            cg.if(cg.void);
            {
              cg.local.get(auxPtr);
              cg.local.get(k);
              cg.i32.const(elemSize);
              cg.i32.mul();
              cg.i32.add();
              cg.local.get(vi);
              ft.store(0, 0);

              cg.local.get(i);
              cg.i32.const(1);
              cg.i32.add();
              cg.local.set(i);
            }
            cg.else();
            {
              cg.local.get(auxPtr);
              cg.local.get(k);
              cg.i32.const(elemSize);
              cg.i32.mul();
              cg.i32.add();
              cg.local.get(vj);
              ft.store(0, 0);

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

  // Main sort function
  return cg.function([cg.i32, cg.i32, cg.i32], [], () => {
    const width = cg.local.declare(cg.i32);
    const i = cg.local.declare(cg.i32);
    const left = cg.local.declare(cg.i32);
    const mid = cg.local.declare(cg.i32);
    const right = cg.local.declare(cg.i32);
    const srcPtr = cg.local.declare(cg.i32);
    const dstPtr = cg.local.declare(cg.i32);
    const pass = cg.local.declare(cg.i32); // 0 or 1, determines which buffer is source

    // Parameter aliases
    const arrPtr = 0;
    const auxPtr = 1;
    const n = 2;

    // width = 1, pass = 0
    cg.i32.const(1);
    cg.local.set(width);
    cg.i32.const(0);
    cg.local.set(pass);

    // Outer loop: for width = 1, 2, 4, ... while width < n
    cg.loop(cg.void);
    {
      cg.block(cg.void);

      cg.local.get(width);
      cg.local.get(n);
      cg.i32.ge_u();
      cg.br_if(0);

      // Determine src and dst based on pass parity
      cg.local.get(pass);
      cg.i32.const(1);
      cg.i32.and();
      cg.i32.eqz();
      cg.if(cg.void);
      {
        cg.local.get(arrPtr);
        cg.local.set(srcPtr);
        cg.local.get(auxPtr);
        cg.local.set(dstPtr);
      }
      cg.else();
      {
        cg.local.get(auxPtr);
        cg.local.set(srcPtr);
        cg.local.get(arrPtr);
        cg.local.set(dstPtr);
      }
      cg.end();

      // i = 0
      cg.i32.const(0);
      cg.local.set(i);

      // Inner loop: for i = 0, 2*width, ... while i < n
      cg.loop(cg.void);
      {
        cg.block(cg.void);

        cg.local.get(i);
        cg.local.get(n);
        cg.i32.ge_u();
        cg.br_if(0);

        // left = i
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

        // merge(srcPtr, dstPtr, left, mid, right)
        cg.local.get(srcPtr);
        cg.local.get(dstPtr);
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

    // If odd number of passes, result is in aux, copy to arr
    cg.local.get(pass);
    cg.i32.const(1);
    cg.i32.and();
    cg.if(cg.void);
    {
      // Copy auxPtr to arrPtr
      cg.i32.const(0);
      cg.local.set(i);

      cg.loop(cg.void);
      {
        cg.block(cg.void);

        cg.local.get(i);
        cg.local.get(n);
        cg.i32.ge_u();
        cg.br_if(0);

        cg.local.get(arrPtr);
        cg.local.get(i);
        cg.i32.const(elemSize);
        cg.i32.mul();
        cg.i32.add();

        cg.local.get(auxPtr);
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
    }
    cg.end();
  });
}

/**
 * Generate a WASM module for batched sort.
 *
 * The exported function takes:
 *   (inPtr, outPtr, auxPtr, n, batchSize)
 *
 * Where:
 *   - inPtr: input array [batchSize, n]
 *   - outPtr: output array [batchSize, n] (sorted)
 *   - auxPtr: auxiliary buffer [n] (reused for each batch)
 *   - n: length of each array to sort
 *   - batchSize: number of arrays
 *
 * @param elementSize - Element size in bytes: 4 for f32, 8 for f64
 * @returns The compiled WebAssembly.Module
 */
export function createSortModule(elementSize: 4 | 8 = 4): WebAssembly.Module {
  const cg = new CodeGenerator();
  cg.memory.import("env", "memory");

  const ft = elementSize === 4 ? cg.f32 : cg.f64;
  const sortFn = wasm_merge_sort(cg, ft);

  // Batched wrapper
  const batchedFn = cg.function(
    [cg.i32, cg.i32, cg.i32, cg.i32, cg.i32],
    [],
    () => {
      const batch = cg.local.declare(cg.i32);
      const arraySize = cg.local.declare(cg.i32); // n * elementSize
      const inOffset = cg.local.declare(cg.i32);
      const outOffset = cg.local.declare(cg.i32);
      const i = cg.local.declare(cg.i32);

      // Parameter aliases
      const inPtr = 0;
      const outPtr = 1;
      const auxPtr = 2;
      const n = 3;
      const batchSize = 4;

      // arraySize = n * elementSize
      cg.local.get(n);
      cg.i32.const(elementSize);
      cg.i32.mul();
      cg.local.set(arraySize);

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

        // inOffset = inPtr + batch * arraySize
        cg.local.get(inPtr);
        cg.local.get(batch);
        cg.local.get(arraySize);
        cg.i32.mul();
        cg.i32.add();
        cg.local.set(inOffset);

        // outOffset = outPtr + batch * arraySize
        cg.local.get(outPtr);
        cg.local.get(batch);
        cg.local.get(arraySize);
        cg.i32.mul();
        cg.i32.add();
        cg.local.set(outOffset);

        // Copy input to output first
        cg.i32.const(0);
        cg.local.set(i);

        cg.loop(cg.void);
        {
          cg.block(cg.void);

          cg.local.get(i);
          cg.local.get(n);
          cg.i32.ge_u();
          cg.br_if(0);

          cg.local.get(outOffset);
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

        // Sort in place
        cg.local.get(outOffset);
        cg.local.get(auxPtr);
        cg.local.get(n);
        cg.call(sortFn);

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

  cg.export(batchedFn, "sort");

  return new WebAssembly.Module(cg.finish());
}
