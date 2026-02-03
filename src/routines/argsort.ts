/**
 * Argsort routine - AssemblyScript implementation.
 *
 * Returns indices that would sort the array.
 * Uses merge sort on indices for stability.
 *
 * Memory layout:
 *   - dataPtr: input array (read-only)
 *   - outPtr: output sorted values
 *   - idxPtr: output indices
 *   - auxPtr: auxiliary buffer for indices (same size as data)
 *   - n: number of elements
 */

// ============================================================================
// f32 version
// ============================================================================

/**
 * Merge two sorted index subarrays for f32.
 * Compares by dataPtr[idx] values.
 */
function merge_idx_f32(
  dataPtr: usize,
  idxPtr: usize,
  auxPtr: usize,
  left: i32,
  mid: i32,
  right: i32,
): void {
  // Copy indices to aux
  for (let i: i32 = left; i <= right; i++) {
    store<i32>(
      auxPtr + ((<usize>i) << 2),
      load<i32>(idxPtr + ((<usize>i) << 2)),
    );
  }

  let i: i32 = left;
  let j: i32 = mid + 1;
  let k: i32 = left;

  while (i <= mid && j <= right) {
    const idxI: i32 = load<i32>(auxPtr + ((<usize>i) << 2));
    const idxJ: i32 = load<i32>(auxPtr + ((<usize>j) << 2));
    const ai: f32 = load<f32>(dataPtr + ((<usize>idxI) << 2));
    const aj: f32 = load<f32>(dataPtr + ((<usize>idxJ) << 2));

    // NaN-aware: NaN goes to end, preserve stability by preferring left on tie
    if (ai < aj || (ai == aj && i <= j) || isNaN<f32>(aj)) {
      store<i32>(idxPtr + ((<usize>k) << 2), idxI);
      i++;
    } else {
      store<i32>(idxPtr + ((<usize>k) << 2), idxJ);
      j++;
    }
    k++;
  }

  while (i <= mid) {
    store<i32>(
      idxPtr + ((<usize>k) << 2),
      load<i32>(auxPtr + ((<usize>i) << 2)),
    );
    i++;
    k++;
  }

  while (j <= right) {
    store<i32>(
      idxPtr + ((<usize>k) << 2),
      load<i32>(auxPtr + ((<usize>j) << 2)),
    );
    j++;
    k++;
  }
}

/**
 * Argsort for f32, single array.
 */
export function argsort_f32(
  dataPtr: usize,
  outPtr: usize,
  idxPtr: usize,
  auxPtr: usize,
  n: i32,
): void {
  // Initialize indices
  for (let i: i32 = 0; i < n; i++) {
    store<i32>(idxPtr + ((<usize>i) << 2), i);
  }

  // Bottom-up merge sort on indices
  let width: i32 = 1;
  while (width < n) {
    let left: i32 = 0;
    while (left < n - width) {
      const mid: i32 = left + width - 1;
      let right: i32 = left + 2 * width - 1;
      if (right >= n) right = n - 1;
      merge_idx_f32(dataPtr, idxPtr, auxPtr, left, mid, right);
      left += 2 * width;
    }
    width *= 2;
  }

  // Write sorted values
  for (let i: i32 = 0; i < n; i++) {
    const idx: i32 = load<i32>(idxPtr + ((<usize>i) << 2));
    store<f32>(
      outPtr + ((<usize>i) << 2),
      load<f32>(dataPtr + ((<usize>idx) << 2)),
    );
  }
}

/**
 * Argsort batched for f32.
 */
export function argsort_batched_f32(
  dataPtr: usize,
  outPtr: usize,
  idxPtr: usize,
  auxPtr: usize,
  n: i32,
  batchSize: i32,
): void {
  const rowBytes: usize = (<usize>n) << 2;
  for (let b: i32 = 0; b < batchSize; b++) {
    argsort_f32(
      dataPtr + <usize>b * rowBytes,
      outPtr + <usize>b * rowBytes,
      idxPtr + <usize>b * rowBytes,
      auxPtr,
      n,
    );
  }
}

// ============================================================================
// f64 version
// ============================================================================

/**
 * Merge two sorted index subarrays for f64.
 */
function merge_idx_f64(
  dataPtr: usize,
  idxPtr: usize,
  auxPtr: usize,
  left: i32,
  mid: i32,
  right: i32,
): void {
  for (let i: i32 = left; i <= right; i++) {
    store<i32>(
      auxPtr + ((<usize>i) << 2),
      load<i32>(idxPtr + ((<usize>i) << 2)),
    );
  }

  let i: i32 = left;
  let j: i32 = mid + 1;
  let k: i32 = left;

  while (i <= mid && j <= right) {
    const idxI: i32 = load<i32>(auxPtr + ((<usize>i) << 2));
    const idxJ: i32 = load<i32>(auxPtr + ((<usize>j) << 2));
    const ai: f64 = load<f64>(dataPtr + ((<usize>idxI) << 3));
    const aj: f64 = load<f64>(dataPtr + ((<usize>idxJ) << 3));

    if (ai < aj || (ai == aj && i <= j) || isNaN<f64>(aj)) {
      store<i32>(idxPtr + ((<usize>k) << 2), idxI);
      i++;
    } else {
      store<i32>(idxPtr + ((<usize>k) << 2), idxJ);
      j++;
    }
    k++;
  }

  while (i <= mid) {
    store<i32>(
      idxPtr + ((<usize>k) << 2),
      load<i32>(auxPtr + ((<usize>i) << 2)),
    );
    i++;
    k++;
  }

  while (j <= right) {
    store<i32>(
      idxPtr + ((<usize>k) << 2),
      load<i32>(auxPtr + ((<usize>j) << 2)),
    );
    j++;
    k++;
  }
}

/**
 * Argsort for f64, single array.
 */
export function argsort_f64(
  dataPtr: usize,
  outPtr: usize,
  idxPtr: usize,
  auxPtr: usize,
  n: i32,
): void {
  for (let i: i32 = 0; i < n; i++) {
    store<i32>(idxPtr + ((<usize>i) << 2), i);
  }

  let width: i32 = 1;
  while (width < n) {
    let left: i32 = 0;
    while (left < n - width) {
      const mid: i32 = left + width - 1;
      let right: i32 = left + 2 * width - 1;
      if (right >= n) right = n - 1;
      merge_idx_f64(dataPtr, idxPtr, auxPtr, left, mid, right);
      left += 2 * width;
    }
    width *= 2;
  }

  for (let i: i32 = 0; i < n; i++) {
    const idx: i32 = load<i32>(idxPtr + ((<usize>i) << 2));
    store<f64>(
      outPtr + ((<usize>i) << 3),
      load<f64>(dataPtr + ((<usize>idx) << 3)),
    );
  }
}

/**
 * Argsort batched for f64.
 */
export function argsort_batched_f64(
  dataPtr: usize,
  outPtr: usize,
  idxPtr: usize,
  auxPtr: usize,
  n: i32,
  batchSize: i32,
): void {
  const dataRowBytes: usize = (<usize>n) << 3;
  const idxRowBytes: usize = (<usize>n) << 2;
  for (let b: i32 = 0; b < batchSize; b++) {
    argsort_f64(
      dataPtr + <usize>b * dataRowBytes,
      outPtr + <usize>b * dataRowBytes,
      idxPtr + <usize>b * idxRowBytes,
      auxPtr,
      n,
    );
  }
}
