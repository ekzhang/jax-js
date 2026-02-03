/**
 * Sort routine - AssemblyScript implementation.
 *
 * In-place merge sort for f32/f64 arrays.
 * Uses bottom-up merge sort which is O(n log n) and stable.
 *
 * Memory layout:
 *   - dataPtr: input/output array to sort
 *   - auxPtr: auxiliary buffer (same size as data)
 *   - n: number of elements
 */

// ============================================================================
// f32 version
// ============================================================================

/**
 * Merge two sorted subarrays for f32.
 */
function merge_f32(
  dataPtr: usize,
  auxPtr: usize,
  left: i32,
  mid: i32,
  right: i32,
): void {
  // Copy to aux
  for (let i: i32 = left; i <= right; i++) {
    store<f32>(
      auxPtr + ((<usize>i) << 2),
      load<f32>(dataPtr + ((<usize>i) << 2)),
    );
  }

  let i: i32 = left;
  let j: i32 = mid + 1;
  let k: i32 = left;

  while (i <= mid && j <= right) {
    const ai: f32 = load<f32>(auxPtr + ((<usize>i) << 2));
    const aj: f32 = load<f32>(auxPtr + ((<usize>j) << 2));
    // NaN-aware comparison: NaN goes to end
    if (ai <= aj || isNaN<f32>(aj)) {
      store<f32>(dataPtr + ((<usize>k) << 2), ai);
      i++;
    } else {
      store<f32>(dataPtr + ((<usize>k) << 2), aj);
      j++;
    }
    k++;
  }

  while (i <= mid) {
    store<f32>(
      dataPtr + ((<usize>k) << 2),
      load<f32>(auxPtr + ((<usize>i) << 2)),
    );
    i++;
    k++;
  }

  while (j <= right) {
    store<f32>(
      dataPtr + ((<usize>k) << 2),
      load<f32>(auxPtr + ((<usize>j) << 2)),
    );
    j++;
    k++;
  }
}

/**
 * Bottom-up merge sort for f32.
 */
export function sort_f32(dataPtr: usize, auxPtr: usize, n: i32): void {
  // Bottom-up merge sort
  let width: i32 = 1;
  while (width < n) {
    let left: i32 = 0;
    while (left < n - width) {
      const mid: i32 = left + width - 1;
      let right: i32 = left + 2 * width - 1;
      if (right >= n) right = n - 1;
      merge_f32(dataPtr, auxPtr, left, mid, right);
      left += 2 * width;
    }
    width *= 2;
  }
}

/**
 * Sort batched arrays for f32.
 */
export function sort_batched_f32(
  dataPtr: usize,
  auxPtr: usize,
  n: i32,
  batchSize: i32,
): void {
  const rowBytes: usize = (<usize>n) << 2;
  for (let b: i32 = 0; b < batchSize; b++) {
    sort_f32(dataPtr + <usize>b * rowBytes, auxPtr, n);
  }
}

// ============================================================================
// f64 version
// ============================================================================

/**
 * Merge two sorted subarrays for f64.
 */
function merge_f64(
  dataPtr: usize,
  auxPtr: usize,
  left: i32,
  mid: i32,
  right: i32,
): void {
  for (let i: i32 = left; i <= right; i++) {
    store<f64>(
      auxPtr + ((<usize>i) << 3),
      load<f64>(dataPtr + ((<usize>i) << 3)),
    );
  }

  let i: i32 = left;
  let j: i32 = mid + 1;
  let k: i32 = left;

  while (i <= mid && j <= right) {
    const ai: f64 = load<f64>(auxPtr + ((<usize>i) << 3));
    const aj: f64 = load<f64>(auxPtr + ((<usize>j) << 3));
    if (ai <= aj || isNaN<f64>(aj)) {
      store<f64>(dataPtr + ((<usize>k) << 3), ai);
      i++;
    } else {
      store<f64>(dataPtr + ((<usize>k) << 3), aj);
      j++;
    }
    k++;
  }

  while (i <= mid) {
    store<f64>(
      dataPtr + ((<usize>k) << 3),
      load<f64>(auxPtr + ((<usize>i) << 3)),
    );
    i++;
    k++;
  }

  while (j <= right) {
    store<f64>(
      dataPtr + ((<usize>k) << 3),
      load<f64>(auxPtr + ((<usize>j) << 3)),
    );
    j++;
    k++;
  }
}

/**
 * Bottom-up merge sort for f64.
 */
export function sort_f64(dataPtr: usize, auxPtr: usize, n: i32): void {
  let width: i32 = 1;
  while (width < n) {
    let left: i32 = 0;
    while (left < n - width) {
      const mid: i32 = left + width - 1;
      let right: i32 = left + 2 * width - 1;
      if (right >= n) right = n - 1;
      merge_f64(dataPtr, auxPtr, left, mid, right);
      left += 2 * width;
    }
    width *= 2;
  }
}

/**
 * Sort batched arrays for f64.
 */
export function sort_batched_f64(
  dataPtr: usize,
  auxPtr: usize,
  n: i32,
  batchSize: i32,
): void {
  const rowBytes: usize = (<usize>n) << 3;
  for (let b: i32 = 0; b < batchSize; b++) {
    sort_f64(dataPtr + <usize>b * rowBytes, auxPtr, n);
  }
}
