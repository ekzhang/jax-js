/**
 * @file Lazy shape tracking for multidimensional tensors.
 *
 * This module provides an immutable `View` class that can be used to calculate
 * shapes of arrays as operations are applied to them, lazily.
 *
 * Some operations like reshape() may not be representable with a single view,
 * for instance, because composing reshape() with shrink() leads to a
 * non-contiguous range of memory locations. This is why `ShapeTracker` is a
 * list of views.
 *
 * Indexing into a `ShapeTracker` or `View` can be folded into shader code.
 *
 * Originally based on tinygrad's implementation of shape tracking in the
 * `tinygrad.shape` module. But this version is simplified a bit. I'm not really
 * trying to innovate on shape tracking in this library, so if I have doubts on
 * something, it'll just be copied from tinygrad (with comments).
 *
 * This file is a bit longer than the original, since Python is more concise.
 */

import { AluExp } from "./alu";
import {
  deepEqual,
  intdiv,
  isPermutation,
  prod,
  range,
  rep,
  zip,
} from "./utils";

export type Pair = [number, number];

const jstr = JSON.stringify;

/** Remove "1" dimensions from the strides list. */
function canonicalizeStrides(shape: number[], strides: number[]): number[] {
  const newStrides: number[] = [];
  for (let i = 0; i < shape.length; i++) {
    if (shape[i] === 1) newStrides.push(0);
    else newStrides.push(strides[i]);
  }
  return newStrides;
}

/** Get the strides for a shape in default row-major order. */
function defaultStrides(shape: number[]): number[] {
  if (shape.length === 0) return [];
  const strides = rep(shape.length, 1);
  for (let i = shape.length - 1; i > 0; i--) {
    strides[i - 1] = shape[i] * strides[i];
  }
  return canonicalizeStrides(shape, strides);
}

/** Merge contiguous subparts or zero-strided dimensions in a view. */
function mergeDims(
  shape: number[],
  strides: number[],
  mask: Pair[] | null,
): [number, number, number][] {
  if (shape.length === 0) return [];
  if (
    shape.length !== strides.length ||
    (mask && shape.length !== mask.length)
  ) {
    throw new Error("internal: invalid args to mergeDims");
  }

  // Returns: Array of [merged_size, stride, real_size]
  // The "real size" only applies if there's a mask. It's the size after the
  // last dimension with size greater than 1, but the mask only has size 1.
  // Basically, after that point we can have stride 0 since it's masked.
  const ret: [number, number, number][] = [
    [shape[0], strides[0], strides[0] !== 0 ? shape[0] : 0],
  ];

  // Merge this dim to next dim if mask size is 1.
  // As a special case, merging can also be set at the beginning if shape[0] is
  // 1. This is so that initial dimensions of size 1 can be merged in. Any
  // subsequent ones would be skipped by `if (s === 1) continue;`.
  let merging = mask ? mask[0][1] - mask[0][0] === 1 : shape[0] === 1;
  for (let i = 1; i < shape.length; i++) {
    const [s, st] = [shape[i], strides[i]];
    if (s === 1) continue; // Always merge 1
    const [lastS, lastSt, lastPreExpandS] = ret[ret.length - 1];
    if (merging || lastSt === s * st) {
      // Merge last dim with this dim if merging or strides matched.
      // If merging due to mask of size 1, reset real size.
      ret[ret.length - 1] = [lastS * s, st, merging ? s : lastPreExpandS * s];
    } else {
      ret.push([s, st, s]);
    }
    merging = mask ? mask[i][1] - mask[i][0] === 1 : false;
  }

  return ret;
}

/** Return the new mask if a reshape if possible, otherwise `null`. */
function reshapeMask(
  maskInput: Pair[],
  oldShape: number[],
  newShape: number[],
): Pair[] | null {
  const newMask: Pair[] = [];

  // Create iterators for all three arrays in reverse order.
  let rMasksI = maskInput.length;
  let rShapeI = oldShape.length;
  let rNewShapeI = newShape.length;
  const rMasks = () => (rMasksI ? maskInput[--rMasksI] : ([0, 1] as Pair));
  const rShape = () => (rShapeI ? oldShape[--rShapeI] : 1);
  const rNewShape = () => (rNewShapeI ? newShape[--rNewShapeI] : 1);

  let currStride = 1;
  let [oldDim, newDim, mask] = [rShape(), rNewShape(), rMasks()];

  while (newMask.length < newShape.length) {
    const [l, r] = mask;
    const nextStride = newDim * currStride;
    // Need to split mask
    if (oldDim === nextStride) {
      newMask.push([intdiv(l, currStride), intdiv(r - 1, currStride) + 1]);
      currStride = 1;
      [oldDim, newDim, mask] = [rShape(), rNewShape(), rMasks()];
    } else if (oldDim > nextStride) {
      if (oldDim % nextStride !== 0) return null;
      if (
        (l % nextStride !== 0 || r % nextStride !== 0) &&
        intdiv(l, nextStride) !== intdiv(r - 1, nextStride)
      )
        return null; // Stride doesn't divide evenly into the new mask.
      newMask.push([
        intdiv(l % nextStride, currStride),
        intdiv((r - 1) % nextStride, currStride) + 1,
      ]);
      [currStride, newDim] = [nextStride, rNewShape()];
    } else {
      // oldDim < nextStride
      const nextMask = rMasks();
      // Combine if the mask can unfold continuously
      if (
        !deepEqual(mask, [0, oldDim]) &&
        l !== r &&
        nextMask[1] - nextMask[0] !== 1
      )
        return null;
      mask = [nextMask[0] * oldDim + l, (nextMask[1] - 1) * oldDim + r];
      oldDim *= rShape();
    }
  }
  return newMask.reverse();
}

/**
 * A multidimensional view into memory. An array can be thought of as the
 * combination of a linear buffer of memory, along with a `View`.
 *
 * Formula for getting a data point is basically:
 *   1. Check if ∀i. 0 <= dim[i] < shape[i], otherwise out of bounds.
 *   2. If mask exists, and ∃i. dim[i] ∉ mask[i], return 0.
 *   2. Otherwise, look at this memory address: offset + ∑(strides[i] * dim[i]).
 */
export class View {
  // Cached, computed property values.
  #size: number | undefined;
  #contiguous: boolean | undefined;

  private constructor(
    /** The shape of the view (size of each dimension). */
    readonly shape: number[],

    /** How many indices to move in buffer for each hop in one dimension. */
    readonly strides: number[],

    /** Offset from the start of the buffer. */
    readonly offset: number,

    /** Masked out subarray where data is read. All other data is zeroed. */
    readonly mask: Pair[] | null,
  ) {}

  static create(
    shape: number[],
    strides?: number[],
    offset: number = 0,
    mask: Pair[] | null = null,
  ): View {
    if (shape.some((s) => s < 0))
      throw new Error("View shape must be non-negative");

    strides = strides
      ? canonicalizeStrides(shape, strides)
      : defaultStrides(shape);

    // Canonicalize zero-sized arrays.
    if (shape.includes(0)) {
      return new View(shape, rep(shape.length, 0), 0, null);
    }
    // Canonicalize default mask / no mask.
    if (mask !== null && mask.every(([b, e], i) => b === 0 && e === shape[i])) {
      mask = null;
    }
    // If dimension has size greater than 1, but is masked to only one index,
    // then set its stride to zero. Likewise, if any mask is empty, we can just
    // mask out the entire array.
    if (mask !== null) {
      const elimDims: number[] = [];
      let hasNoData = false;
      for (let i = 0; i < shape.length; i++) {
        const [b, e] = mask[i];
        if (b + 1 >= e) elimDims.push(i);
        if (b >= e) hasNoData = true;
      }
      if (elimDims.length) {
        if (hasNoData) {
          strides = rep(shape.length, 0);
          offset = 0;
          mask = rep(shape.length, () => [0, 0] as Pair);
        }
        for (const i of elimDims) {
          offset += strides[i] * mask[i][0];
          strides[i] = 0;
        }
      }
    }
    return new View(shape, strides, offset, mask);
  }

  get ndim(): number {
    return this.shape.length;
  }

  get size(): number {
    if (this.#size === undefined) this.#size = prod(this.shape);
    return this.#size;
  }

  /** Whether this is a default, contiguous, unaltered view of the data (identity). */
  get contiguous(): boolean {
    if (this.#contiguous === undefined) {
      this.#contiguous =
        this.size === 0 ||
        (this.offset === 0 &&
          this.mask === null &&
          deepEqual(this.strides, defaultStrides(this.shape)));
    }
    return this.#contiguous;
  }

  /** Return the range of data being indexed in this view, or [0, 0] if none. */
  dataRange(): [number, number] {
    if (this.size === 0 || (this.mask && this.mask[0][0] === this.mask[0][1]))
      return [0, 0];
    let min = this.offset;
    let max = this.offset;
    for (let i = 0; i < this.ndim; i++) {
      let [lo, hi] = this.mask ? this.mask[i] : [0, this.shape[i]];
      --hi; // make inclusive
      const s = this.strides[i];
      if (s > 0) {
        min += s * lo;
        max += s * hi;
      } else if (s < 0) {
        min += s * hi;
        max += s * lo;
      }
    }
    return [min, max + 1];
  }

  /** Produce an AluExp for evaluating this view at an index. */
  toAluExp(idxs: AluExp[]): [AluExp, AluExp] {
    let iexpr = AluExp.i32(this.offset);
    let vexpr = AluExp.bool(true);
    // Iterate in reverse order to make bottom-up optimizations simpler.
    for (let i = this.ndim - 1; i >= 0; i--) {
      const idx = idxs[i];
      if (this.shape[i] !== 1 && this.strides[i] !== 0) {
        iexpr = AluExp.add(AluExp.mul(idx, AluExp.i32(this.strides[i])), iexpr);
      }
      if (this.mask) {
        if (this.mask[i][0] !== 0)
          vexpr = AluExp.mul(
            AluExp.cmplt(idx, AluExp.i32(this.mask[i][0])).not(),
            vexpr,
          );
        if (this.mask[i][1] !== this.shape[i])
          vexpr = AluExp.mul(
            AluExp.cmplt(idx, AluExp.i32(this.mask[i][1])),
            vexpr,
          );
      }
    }
    return [iexpr, vexpr];
  }

  /**
   * Try to compose this view with another one. `this` view is applied first,
   * followed by the argument. If this is not possible for the specific views,
   * return `null` instead.
   *
   * If composable, return a combined view with the same shape as `v1`.
   *
   * This is very tricky. The shapes of v1 and v2 may be different, and in that
   * case, we do some math to figure out whether they're compatible.
   */
  compose(v1: View): View | null {
    const v2 = this;
    if (v2.contiguous) return v1;
    if (v1.contiguous) {
      if (deepEqual(v1.shape, v2.shape)) return v2;
      if (v1.size === v2.size) {
        const ret = v2.reshape(v1.shape);
        if (ret !== null) return ret;
      }
    }
    // Normalize out any masks in v1, applying them afterward.
    if (v1.mask !== null) {
      const newV1 = v1.shrink(v1.mask);
      const merged = v2.compose(newV1);
      return merged
        ? merged.pad(zip(v1.mask, v1.shape).map(([m, s]) => [m[0], s - m[1]]))
        : null;
    }

    // Project offset and strides.
    //  - origin: the unravelled offset of v1 in v2
    //  - terms: a list of [dim, stride] pairs for each dimension of v2, where
    //    the stride is offset in v2 for one index of that dim of v1.
    //  - strides: the new strides for v1, reduced from terms

    const origin = unravel(v2.shape, v1.offset); // v1 applies after v2
    const terms: Pair[][] = rep(v2.ndim, () => []);
    const strides = rep(v1.ndim, 0);
    for (let d1 = 0; d1 < v1.strides.length; d1++) {
      const st = v1.strides[d1];
      if (st === 0) {
        continue;
      }
      const unravelOffset = unravel(v2.shape, v1.offset + st);
      // compare new unravel with origin
      for (let d2 = 0; d2 < v2.ndim; d2++) {
        const o = origin[d2];
        const diff = unravelOffset[d2] - o;
        if (diff === 0) {
          continue;
        }
        terms[d2].push([d1, diff]);
        strides[d1] += diff * v2.strides[d2];
      }
    }

    // Merge dimensions in v2 if required.
    // This is helpful in cases where the shape of v1 doesn't match that of v2
    // in a concise way, so we need to figure out which dimensions in v2 are
    // joined together. Sometimes this may not be possible.
    let [mergedSize, mergedTermMin, mergedTermMax] = [1, 0, 0];
    const extents: [number, number, number][] = []; // size, vmin, vmax
    for (let i = v2.ndim - 1; i >= 0; i--) {
      const term = terms[i]; // list of [dim in v1, stride in v2.shape[i]]
      const s = v2.shape[i];
      // Figure out the min and max value of this dimension in v2.
      let [tmin, tmax] = [origin[i], origin[i]];
      for (const [d1, s1] of term) {
        if (s1 > 0) tmax += (v1.shape[d1] - 1) * s1;
        else if (s1 < 0) tmin += (v1.shape[d1] - 1) * s1;
      }
      mergedTermMin += tmin * mergedSize;
      mergedTermMax += tmax * mergedSize;
      mergedSize *= s;
      // Only keep this dimension if the term doesn't exceed array bounds.
      if (mergedTermMin >= 0 && mergedTermMax < mergedSize) {
        extents.push([mergedSize, mergedTermMin, mergedTermMax]);
        [mergedSize, mergedTermMin, mergedTermMax] = [1, 0, 0];
      }
    }
    // Unmerged dimensions => invalid, it goes past array bounds.
    if (mergedTermMin !== 0 || mergedTermMax !== 0) return null;
    extents.reverse();

    const v2Shape = extents.map(([s]) => s);
    if (!deepEqual(v2Shape, v2.shape)) {
      const reshapedV2 = v2.reshape(v2Shape);
      if (reshapedV2 === null) return null;
      // NOTE: Unclear why this conditional is needed? Original says it prevents infinite loop.
      if (!deepEqual(reshapedV2.shape, v2.shape)) return reshapedV2.compose(v1);
    }

    // If v2 has a mask, let's try to project it onto v1
    if (v2.mask !== null) {
      const newB = rep(v1.ndim, 0);
      const newE = v1.shape.slice();
      let bad = false;

      for (let d2 = 0; d2 < v2.ndim; d2++) {
        const [b, e] = v2.mask[d2];
        const o = origin[d2];
        const term = terms[d2];
        const [_, tmin, tmax] = extents[d2];
        if (b <= tmin && tmax < e) continue; // v1 doesn't reach outside the mask
        if (term.length !== 1) {
          // otherwise, v1 reaches outside the mask...
          if (term.length === 0 && newE.length) newE[0] = 0;
          else bad = true; // ...and it has two or more terms, so the mask is violated
        } else {
          const [d1, s1] = term[0]; // changes in d1 -> changes in d2, by s1
          newB[d1] = Math.max(
            newB[d1],
            Math.ceil((s1 > 0 ? b - o : e - o - 1) / s1),
          );
          newE[d1] = Math.min(
            newE[d1],
            Math.floor((s1 < 0 ? b - o : e - o - 1) / s1) + 1,
          );
        }
      }

      // If any of v1 was masked off, try again with that mask in place.
      for (let d1 = 0; d1 < v1.ndim; d1++) {
        if (newB[d1] !== 0 || newE[d1] !== v1.shape[d1]) {
          return v2.compose(
            View.create(v1.shape, v1.strides, v1.offset, zip(newB, newE)),
          );
        }
      }

      // Otherwise, if v2's mask was violated, we can't merge :(
      // Note: I don't know why this is below the previous line, but whatever.
      if (bad) return null;
    }

    // Final offset is v2.offset plus sum of origin*d2 strides.
    let finalOffset = v2.offset;
    for (let d2 = 0; d2 < v2.ndim; d2++) {
      finalOffset += origin[d2] * v2.strides[d2];
    }

    // Return the composed view (no mask, see normalization at the beginning).
    return View.create(v1.shape, strides, finalOffset, null);
  }

  /** Attempt to simplify this view into a smaller reshaped form. */
  minify(): View {
    const minShape = mergeDims(this.shape, this.strides, this.mask).map(
      (x) => x[0],
    );
    const nv = this.reshape(minShape);
    return nv ? nv : this;
  }

  /** Pad the view with zeros on each dimension. */
  pad(arg: Pair[]): View {
    if (arg.length !== this.ndim || !arg.every(([b, e]) => b >= 0 && e >= 0)) {
      throw new Error(`invalid pad ${jstr(arg)} for ${jstr(this.shape)}`);
    }
    if (arg.every(([b, e]) => b === 0 && e === 0)) return this;
    const zvarg = arg.map<Pair>(([b, e], i) => [-b, this.shape[i] + e]);
    const mask = arg.map<Pair>(([b, _e], i) => [b, this.shape[i] + b]);
    return this.#unsafeResize(zvarg, mask);
  }

  /** Shrink the view by taking a subarray. */
  shrink(arg: Pair[]): View {
    if (
      arg.length !== this.ndim ||
      !arg.every(([b, e], i) => 0 <= b && b <= e && e <= this.shape[i])
    ) {
      throw new Error(`invalid shrink ${jstr(arg)} for ${jstr(this.shape)}`);
    }
    return this.#unsafeResize(arg);
  }

  #unsafeResize(arg: Pair[], mask?: Pair[]): View {
    const offset = this.strides
      .map((s, i) => s * arg[i][0])
      .reduce((a, b) => a + b, 0);
    if (this.mask) {
      // Move the old mask
      const nmask = this.mask.map<Pair>(([mx, my], i) => [
        Math.max(0, Math.min(mx - arg[i][0], arg[i][1] - arg[i][0])),
        Math.max(0, Math.min(my - arg[i][0], arg[i][1] - arg[i][0])),
      ]);
      // Merge the masks if we have two
      mask = mask
        ? mask.map(([mx, my], i) => [
            Math.max(mx, nmask[i][0]),
            Math.min(my, nmask[i][1]),
          ])
        : nmask;
    }
    return View.create(
      arg.map(([b, e]) => e - b),
      this.strides,
      this.offset + offset,
      mask,
    );
  }

  /** Expand one or more axes with length "1" by repeating the data. */
  expand(newShape: number[]): View {
    if (newShape.length !== this.ndim) {
      throw new Error(
        `Can't expand ${jstr(this.shape)} into ${jstr(newShape)}`,
      );
    }
    for (let i = 0; i < this.ndim; i++) {
      if (newShape[i] !== this.shape[i] && this.shape[i] !== 1) {
        throw new Error(
          `Can't expand ${jstr(this.shape)} into ${jstr(newShape)}`,
        );
      }
    }
    // If it's a zero size array, just return a zero size array.
    if (this.size === 0) return View.create(newShape);
    const mask = this.mask
      ? this.mask.map<Pair>((m, i) =>
          this.shape[i] === newShape[i]
            ? m
            : m[0] === 0 && m[1] === 1
              ? [0, newShape[i]]
              : [0, 0],
        )
      : null;
    return View.create(newShape, this.strides, this.offset, mask);
  }

  /** Permute the axes of an array. */
  permute(axis: number[]): View {
    if (!isPermutation(axis, this.ndim))
      throw new Error(`Invalid permutation ${jstr(axis)} of len ${this.ndim}`);
    const newShape = axis.map((a) => this.shape[a]);
    const newStrides = axis.map((a) => this.strides[a]);
    const newMask = this.mask ? axis.map((a) => this.mask![a]) : null;
    return View.create(newShape, newStrides, this.offset, newMask);
  }

  /** Flip (reverse) one or more axes of the view. */
  flip(arg: boolean[]): View {
    if (arg.length !== this.ndim)
      throw new Error(`Invalid flip ${jstr(arg)} for ${jstr(this.shape)}`);
    const strides = this.strides.slice();
    let offset = this.offset;
    const mask: Pair[] | null = this.mask ? this.mask.slice() : null;
    for (let i = 0; i < this.ndim; i++) {
      const s = this.shape[i];
      if (arg[i]) {
        strides[i] = -strides[i];
        offset += (s - 1) * this.strides[i];
        if (mask) mask[i] = [s - mask[i][1], s - mask[i][0]];
      }
    }
    return View.create(this.shape, strides, offset, mask);
  }

  /** Reshape the view into a new shape. */
  reshape(newShape: number[]): View | null {
    if (deepEqual(this.shape, newShape)) return this;
    if (newShape.some((s) => s < 0))
      throw new Error(`Reshape cannot have negative numbers ${jstr(newShape)}`);
    if (this.size !== prod(newShape))
      throw new Error(`Reshape size ${jstr(this.shape)} -> ${jstr(newShape)}`);

    if (this.size === 0) return View.create(newShape);
    // Edge case: if the new shape is empty, the size is 1. If that element is
    // masked out in the original array, then we can't mask it in the new array
    // because there's no dimension to mask by.
    if (newShape.length === 0 && this.mask?.some(([b, e]) => b === e))
      return null;

    if (this.contiguous) return View.create(newShape); // easy case

    // Now for the tricky part: We have to compute the new strides and mask in
    // the reshaped array. To do this, first we merge dimensions.
    const rStrides: number[] = [];
    const merge = mergeDims(this.shape, this.strides, this.mask);
    let rShapeIdx = newShape.length; // reverse iterator
    for (let i = merge.length - 1; i >= 0; i--) {
      let [mergedSize, newStride, realSize] = merge[i];
      let acc = 1; // how much size we've accumulated for this chunk so far
      while (acc < mergedSize && rShapeIdx > 0) {
        const newDim = newShape[--rShapeIdx];
        rStrides.push(newStride * acc);
        acc *= newDim;
        if (acc >= realSize) newStride = 0; // Can't make it, give up.
      }
      if (acc !== mergedSize) return null; // Not divisible, give up.
    }

    // If we didn't make it through, just fill in the rest with zeros. It won't
    // work, but we can just continue onto the View.create() step.
    const newStrides = rep(newShape.length - rStrides.length, 0).concat(
      rStrides.reverse(),
    );
    if (!this.mask) return View.create(newShape, newStrides, this.offset);

    // Okay, now we gotta compute the new mask and offset — but then we're done!
    const newMask = reshapeMask(this.mask, this.shape, newShape);
    if (!newMask) return null;

    // Since the mask changed, we also have to adjust the offset.
    let newOffset = this.offset;
    for (let i = 0; i < this.ndim; i++)
      newOffset += this.strides[i] * this.mask[i][0];
    for (let i = 0; i < newShape.length; i++)
      newOffset -= newStrides[i] * newMask[i][0];
    return View.create(newShape, newStrides, newOffset, newMask);
  }
}

/**
 * Find position of `offset` in each dimension within an existing shape. Like
 * `numpy.unravel_index` in behavior.
 */
export function unravel(shape: number[], offset: number): number[] {
  let acc = 1;
  const idxs: number[] = [];
  for (let i = shape.length - 1; i >= 0; i--) {
    const d = shape[i];
    idxs.push(Math.floor(offset / acc) % d);
    acc *= d;
  }
  return idxs.reverse();
}

/** Generate a list of AluExp for computing unravel(). */
export function unravelAlu(shape: number[], offset: AluExp): AluExp[] {
  let acc = 1;
  const idxs: AluExp[] = [];
  for (let i = shape.length - 1; i >= 0; i--) {
    const d = shape[i];
    idxs.push(AluExp.mod(AluExp.idiv(offset, AluExp.i32(acc)), AluExp.i32(d)));
    acc *= d;
  }
  return idxs.reverse();
}

/**
 * Array shape after applying movement operations, as a series of views.
 *
 * Each view is applied, then treated as if it were a contiguous array of its
 * shape, then used as the virtual buffer for the next view.
 */
export class ShapeTracker {
  constructor(readonly views: View[]) {} // Views apply left-to-right

  /** Compose this shape tracker with another, applying it after this one. */
  compose(other: ShapeTracker): ShapeTracker {
    if (this.contiguous) return other;
    let ret: ShapeTracker = this;
    for (const v of other.views) {
      ret = new ShapeTracker(ret.views.concat(v)).simplify();
    }
    return ret;
  }

  static fromShape(shape: number[]): ShapeTracker {
    return new ShapeTracker([View.create(shape)]);
  }

  get contiguous(): boolean {
    return this.views.length === 1 && this.views[0].contiguous;
  }

  get consecutive(): boolean {
    // Like contiguous, but may have nonzero offset
    return (
      this.views.length === 1 &&
      this.views[0].mask === null &&
      deepEqual(this.views[0].strides, defaultStrides(this.views[0].shape))
    );
  }

  get lastStrides(): number[] {
    return this.views[this.views.length - 1].strides;
  }

  get shape(): number[] {
    return this.views[this.views.length - 1].shape;
  }

  get size(): number {
    return this.views[this.views.length - 1].size;
  }

  toAluExp(idxs: AluExp[]): [AluExp, AluExp] {
    // Note: Cannot minify the final view since this takes indices.
    let [iexpr, vexpr] = this.views[this.views.length - 1].toAluExp(idxs);
    for (let i = this.views.length - 2; i >= 0; i--) {
      const view = this.views[i].minify();
      const exprs = view.toAluExp(unravelAlu(view.shape, iexpr));
      iexpr = exprs[0];
      vexpr = AluExp.mul(vexpr, exprs[1]);
    }
    return [iexpr.simplify(), vexpr.simplify()];
  }

  simplify(): ShapeTracker {
    const views = this.views.slice();
    while (views.length >= 2) {
      const newView = views[views.length - 2].compose(views[views.length - 1]);
      if (newView === null) break;
      views.splice(views.length - 2, 2, newView);
    }
    return new ShapeTracker(views);
  }

  pad(arg: Pair[]): ShapeTracker {
    return new ShapeTracker(applyLast(this.views, (x) => x.pad(arg)));
  }
  shrink(arg: Pair[]): ShapeTracker {
    return new ShapeTracker(applyLast(this.views, (x) => x.shrink(arg)));
  }
  expand(newShape: number[]): ShapeTracker {
    return new ShapeTracker(applyLast(this.views, (x) => x.expand(newShape)));
  }
  permute(axis: number[]): ShapeTracker {
    return new ShapeTracker(applyLast(this.views, (x) => x.permute(axis)));
  }
  flip(arg: boolean[]): ShapeTracker {
    return new ShapeTracker(applyLast(this.views, (x) => x.flip(arg)));
  }

  reshape(newShape: number[]): ShapeTracker {
    const newView = this.views[this.views.length - 1].reshape(newShape);
    return new ShapeTracker(
      newView === null
        ? this.views.concat(View.create(newShape))
        : this.views.toSpliced(this.views.length - 1, 1, newView),
    );
  }

  // Below this line are "composite" operations.

  /** Broadcast along the given new axes, then expand the shape. */
  broadcast(newShape: number[], axis: number[]): ShapeTracker {
    let st = this as ShapeTracker;
    if (axis.length > 0) {
      // First, unsqueeze each dimension of "axis".
      const unsqueezed = [...st.shape];
      for (const i of axis.toSorted()) {
        unsqueezed.splice(i, 0, 1);
      }
      st = st.reshape(unsqueezed);
    }
    // Then, expand the data to the new shape.
    return st.expand(newShape);
  }

  /**
   * Repeat data in each axis by a positive number of repetitions.
   *
   * - If `tile` is true (default): [1, 2, 3] -> [1, 2, 3, 1, 2, 3].
   * - If `tile` is false: [1, 2, 3] -> [1, 1, 2, 2, 3, 3].
   */
  repeat(reps: number[], tile: boolean = true): ShapeTracker {
    if (reps.length > this.shape.length) {
      throw new Error(
        `Too many repeats ${jstr(reps)} for shape ${jstr(this.shape)}`,
      );
    }
    if (reps.some((c) => c <= 0))
      throw new Error(`Invalid repeats ${jstr(reps)}`);
    if (reps.length === 0) return this;

    // First, duplicate the dimensions in the shape.
    const noop = this.shape.slice(0, -reps.length);
    const shape = this.shape.slice(-reps.length);

    return this.broadcast(
      [
        ...noop,
        ...shape.flatMap((s, i) => (tile ? [reps[i], s] : [s, reps[i]])),
      ],
      shape.map((_, i) => noop.length + 2 * i + (tile ? 0 : 1)),
    ).reshape([...noop, ...shape.map((s, i) => s * reps[i])]);
  }

  /** Move axis i to axis j. */
  moveaxis(i: number, j: number): ShapeTracker {
    const perm = range(this.shape.length);
    perm.splice(i, 1); // remove i
    perm.splice(j, 0, i); // insert i at j
    return this.permute(perm);
  }

  /** Like pad(), but allows for negative values. */
  padOrShrink(arg: Pair[]): ShapeTracker {
    const padArg: Pair[] = [];
    const shrinkArg: Pair[] = [];
    for (let i = 0; i < arg.length; i++) {
      const [b, e] = arg[i];
      if (b < -this.shape[i] || e < -this.shape[i] || b + e < -this.shape[i])
        throw new Error(
          `Invalid padOrShrink ${jstr(arg)} for ${jstr(this.shape)}`,
        );
      padArg.push([Math.max(0, b), Math.max(0, e)]);
      shrinkArg.push([Math.max(0, -b), this.shape[i] - Math.max(0, -e)]);
    }
    return this.shrink(shrinkArg).pad(padArg);
  }
}

function applyLast<T>(ar: T[], f: (x: T) => T): T[] {
  return ar.toSpliced(ar.length - 1, 1, f(ar[ar.length - 1]));
}
