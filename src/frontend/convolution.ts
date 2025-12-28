// Implementation of the Conv primitive (lax.conv_general_dilated).
//
// This handles both forward and transposed convolutions.
//
// Reference:
//  - https://openxla.org/xla/operation_semantics#conv_convolution
//  - https://github.com/jax-ml/jax/blob/main/jax/_src/lax/convolution.py

import { Pair, ShapeTracker } from "../shape";
import {
  deepEqual,
  generalBroadcast,
  prod,
  range,
  rep,
  zip,
  zipn,
} from "../utils";

/** Definition of a general dilated convolution. Should be valid on creation. */
export interface ConvParams {
  vmapDims: number; // number of dims to batch in front for (lhs, rhs)
  strides: number[];
  padding: Pair[];
  lhsDilation: number[];
  rhsDilation: number[];
}

/*
Rules for transposing a convolution:

Backprop of activations:
  y = conv(x, filter) -> x’ = conv(y’, filter), where

- in_channels <-> out_channels
- stride <-> lhs_dilation
- rhs_dilation stays the same
- left_padding -> (dilated kernel_size - 1) - left_padding
- right_padding -> (dilated kernel_size - 1) - right_padding
- kernel -> flip(kernel)

Backprop of filter:
  y = conv(x, filter) -> filter’ = conv(x, y’), where

- in_channels & out_channels are transposed with batch size
- stride <-> rhs_dilation
- lhs_dilation stays the same
- padding stays the same
*/

/**
 * Check that the shapes and parameters passed to convolution are valid.
 * Expected shapes of the lhs and rhs of the convolution are:
 *
 * - `lhsShape = [*vmapDims, batchSize, inChannels, spatialDims...]`
 * - `rhsShape = [*vmapDims, outChannels, inChannels, kernelSize...]`
 *
 * If the check succeeds, returns the output shape.
 */
export function checkConvShape(
  lhsShape: number[],
  rhsShape: number[],
  { vmapDims, strides, padding, lhsDilation, rhsDilation }: ConvParams,
): number[] {
  if (lhsShape.length !== rhsShape.length) {
    throw new Error(
      `conv() requires inputs with the same number of dimensions, got ${lhsShape.length} and ${rhsShape.length}`,
    );
  }
  const n = lhsShape.length - 2 - vmapDims;
  if (n < 0) throw new Error("conv() requires at least 2D inputs");
  if (strides.length !== n) throw new Error("conv() strides != spatial dims");
  if (padding.length !== n) throw new Error("conv() padding != spatial dims");
  if (lhsDilation.length !== n)
    throw new Error("conv() lhsDilation != spatial dimensions");
  if (rhsDilation.length !== n)
    throw new Error("conv() rhsDilation != spatial dimensions");
  if (lhsShape[vmapDims + 1] !== rhsShape[vmapDims + 1])
    throw new Error(`conv() input channels: ${lhsShape[1]} != ${rhsShape[1]}`);

  const outShape = [
    ...generalBroadcast(
      lhsShape.slice(0, vmapDims),
      rhsShape.slice(0, vmapDims),
    ), // vmap dimensions (broadcast)
    lhsShape[vmapDims], // Batch size
    rhsShape[vmapDims], // out_channels
  ];

  // Check each spatial dimension.
  for (let i = 0; i < n; i++) {
    if (strides[i] <= 0 || !Number.isInteger(strides[i]))
      throw new Error(`conv() strides[${i}] must be a positive integer`);
    if (padding[i].length !== 2 || !padding[i].every(Number.isInteger))
      throw new Error(`conv() padding[${i}] must be a 2-tuple of integers`);
    if (lhsDilation[i] <= 0 || !Number.isInteger(lhsDilation[i]))
      throw new Error(`conv() lhsDilation[${i}] must be a positive integer`);
    if (rhsDilation[i] <= 0 || !Number.isInteger(rhsDilation[i]))
      throw new Error(`conv() rhsDilation[${i}] must be a positive integer`);

    const [x, k] = [lhsShape[i + vmapDims + 2], rhsShape[i + vmapDims + 2]];
    if (k <= 0) throw new Error("conv() kernel size must be positive");

    const [pl, pr] = padding[i];
    if (pl < -x || pr < -x || pl + pr < -x)
      throw new Error(
        `conv() padding[${i}]=(${pl},${pr}) is too negative for input size ${x}`,
      );

    const kernelSize = (k - 1) * rhsDilation[i] + 1;
    const inSize = Math.max((x - 1) * lhsDilation[i] + 1, 0) + pl + pr;
    if (kernelSize > inSize)
      throw new Error(
        `conv() kernel size ${kernelSize} > input size ${inSize} in dimension ${i}`,
      );
    outShape.push(Math.ceil((inSize - kernelSize + 1) / strides[i]));
  }
  return outShape;
}

export function checkPoolShape(
  inShape: number[],
  window: number[],
  strides: number[],
): number[] {
  if (strides.length !== window.length)
    throw new Error("pool() strides != window dims");
  if (window.length > inShape.length)
    throw new Error("pool() window has more dimensions than input");

  const outShape = inShape.slice(0, inShape.length - window.length);
  for (let i = 0; i < window.length; i++) {
    const k = window[i];
    const s = strides[i];
    const size = inShape[inShape.length - window.length + i];
    if (k <= 0 || !Number.isInteger(k))
      throw new Error(`pool() window[${i}] must be a positive integer`);
    if (k > size)
      throw new Error(`pool() window[${i}]=${k} > input size ${size}`);
    if (s <= 0 || !Number.isInteger(s))
      throw new Error(`pool() strides[${i}] must be a positive integer`);
    outShape.push(Math.ceil((size - k + 1) / s));
  }
  return outShape.concat(window);
}

/**
 * Takes a shape tracker and a kernel size `ks`, then reshapes it so the last
 * `ks.length` dimensions become `2 * ks.length` dimensions by treating them as
 * spatial dimensions convolved with a kernel.
 *
 * The resulting array can be multiplied with a kernel of shape `ks`, then
 * reduced along the last `ks.length` dimensions for a convolution.
 *
 * Reference: https://github.com/tinygrad/tinygrad/blob/v0.10.3/tinygrad/tensor.py#L2097
 */
export function pool(
  st: ShapeTracker,
  ks: number[],
  strides: number | number[] = 1,
  dilation: number | number[] = 1,
): ShapeTracker {
  if (ks.length === 0) return st; // Dimension 0 kernel, no pooling needed.
  if (st.shape.length < ks.length)
    throw new Error("pool() called with too many dimensions");
  if (typeof strides === "number") strides = rep(ks.length, strides);
  if (typeof dilation === "number") dilation = rep(ks.length, dilation);

  if (strides.some((s) => s <= 0 || !Number.isInteger(s)))
    throw new Error("pool() strides must be positive integers");
  if (dilation.some((d) => d <= 0 || !Number.isInteger(d)))
    throw new Error("pool() dilation must be positive integers");

  const noop = st.shape.slice(0, -ks.length);

  const i_ = st.shape.slice(-ks.length);
  const s_ = strides;
  const d_ = dilation;
  const o_ = zipn(i_, d_, ks, s_).map(([i, d, k, s]) =>
    Math.ceil((i - d * (k - 1)) / s),
  );

  // Alternative implementation for d=1 and k<=s, faster (e.g., max pooling).
  if (d_.every((d) => d === 1) && ks.every((k, j) => k <= s_[j])) {
    // Pad or shrink to shape [..., o_ * s_]
    st = st.padOrShrink([
      ...noop.map<Pair>(() => [0, 0]),
      ...zipn(i_, o_, s_).map<Pair>(([i, o, s]) => [0, o * s - i]),
    ]);
    // Reshape to [..., o_, s_] and then shrink to [..., o_, k_]
    st = st
      .reshape([...noop, ...zip(o_, s_).flatMap(([o, s]) => [o, s])])
      .shrink([
        ...noop.map<Pair>((x) => [0, x]),
        ...zip(o_, ks).flatMap<Pair>(([o, k]) => [
          [0, o],
          [0, k],
        ]),
      ]);
    // Permute k_ dimensions to end.
    st = st.permute([
      ...range(noop.length),
      ...ks.map((_, j) => noop.length + 2 * j), // o_ dimensions
      ...ks.map((_, j) => noop.length + 2 * j + 1), // k_ dimensions
    ]);
    return st;
  }

  // Input size scaling factor to make sure shrink for stride is possible.
  const f_ = zipn(o_, s_, i_, d_, ks).map(
    ([o, s, i, d, k]) => 1 + Number(o * s > i - d * (k - 1)),
  );

  // Number of repeats such that we don't need padding.
  // We basically want k*(i+d) worth of elements, but each from repeated rows of i elements.
  // This wil let us shrink consecutive rows so that the offset will be by d.
  //   [1, 2, 3, 4, 5] -> [1, 2, 3, 4, 5]
  //                      [2, 3, 4, 5, 1]
  //                      [3, 4, 5, 1, 2]
  const kidf = zipn(ks, i_, d_, f_);
  st = st.repeat([
    ...rep(noop.length, 1),
    ...kidf.map(([k, i, d, f]) => Math.ceil((k * (i * f + d)) / i)),
  ]);
  st = st
    .shrink([
      ...noop.map<Pair>((x) => [0, x]),
      ...kidf.map<Pair>(([k, i, d, f]) => [0, k * (i * f + d)]),
    ])
    .reshape([...noop, ...kidf.flatMap(([k, i, d, f]) => [k, i * f + d])]);

  // Next, handle stride by only taking every s-th element.
  //   [1, 2, 3, 4, 5]    [1, 3, 5]
  //   [2, 3, 4, 5, 1] -> [2, 4, 1]
  //   [3, 4, 5, 1, 2]    [3, 5, 2]
  const kos = zipn(ks, o_, s_);
  st = st
    .shrink([
      ...noop.map<Pair>((x) => [0, x]),
      ...kos.flatMap<Pair>(([k, o, s]) => [
        [0, k],
        [0, o * s],
      ]),
    ])
    .reshape([...noop, ...kos.flat(1)]);
  st = st
    .shrink([
      ...noop.map<Pair>((x) => [0, x]),
      ...kos.flatMap<Pair>(([k, o]) => [
        [0, k],
        [0, o],
        [0, 1],
      ]),
    ])
    .reshape([...noop, ...kos.flatMap(([k, o]) => [k, o])]);

  // Finally, permute to move reduction dimensions (k_) to the end.
  st = st.permute([
    ...range(noop.length),
    ...ks.map((_, j) => noop.length + 2 * j + 1), // o_ dimensions
    ...ks.map((_, j) => noop.length + 2 * j), // k_ dimensions
  ]);

  return st;
}

/**
 * Perform the transpose of pool, directly undo-ing a pool() operation.
 *
 * Note that since pool repeats the input, the transpose operation technically
 * should include a sum reduction. This function doesn't perform the reduction,
 * which should be done on the last `k` axes of the returned shape.
 */
export function poolTranspose(
  st: ShapeTracker,
  inShape: number[],
  ks: number[],
  strides: number | number[] = 1,
  dilation: number | number[] = 1,
): ShapeTracker {
  if (ks.length === 0) return st;

  if (typeof strides === "number") strides = rep(ks.length, strides);
  if (typeof dilation === "number") dilation = rep(ks.length, dilation);

  const noop = inShape.slice(0, -ks.length);

  const i_ = inShape.slice(-ks.length);
  const s_ = strides;
  const d_ = dilation;
  const o_ = zipn(i_, d_, ks, s_).map(([i, d, k, s]) =>
    Math.ceil((i - d * (k - 1)) / s),
  );

  // Alternative implementation for d=1 and k<=s, faster (e.g., max pooling).
  if (d_.every((d) => d === 1) && ks.every((k, j) => k <= s_[j])) {
    // Undo moving k_ dimensions to the end.
    st = st.permute([
      ...range(noop.length),
      ...ks.flatMap((_, j) => [noop.length + j, noop.length + o_.length + j]),
    ]);
    // Undo shrinking s_ to k_, and then undo reshaping o_ * s_ to [o_, s_].
    st = st
      .pad([
        ...noop.map<Pair>(() => [0, 0]),
        ...zip(s_, ks).flatMap<Pair>(([s, k]) => [
          [0, 0],
          [0, s - k],
        ]),
      ])
      .reshape([...noop, ...zip(o_, s_).map(([o, s]) => o * s)]);
    // Undo pad or shrink from original shape to o_ * s_.
    st = st.padOrShrink([
      ...noop.map<Pair>(() => [0, 0]),
      ...zipn(i_, o_, s_).map<Pair>(([i, o, s]: number[]) => [0, i - o * s]),
    ]);
    // We need some additional dimensions to match behavior of the behavior of
    // the other implementation, where add |ks| axes for repetitions.
    return st.reshape(st.shape.concat(rep(ks.length, 1)));
  }

  if (!deepEqual(o_, st.shape.slice(noop.length, noop.length + ks.length))) {
    throw new Error("poolTranspose() called with mismatched output shape");
  }

  const f_ = zipn(o_, s_, i_, d_, ks).map(
    ([o, s, i, d, k]) => 1 + Number(o * s > i - d * (k - 1)),
  );

  const kidf = zipn(ks, i_, d_, f_);
  const kos = zipn(ks, o_, s_);

  // Undo permute to move reduction dimensions (k_) to the end.
  st = st.permute([
    ...range(noop.length),
    ...ks.flatMap((_, j) => [noop.length + ks.length + j, noop.length + j]),
  ]);

  // Undo taking every s-th element (stride).
  st = st.reshape([...noop, ...kos.flatMap(([k, o]) => [k, o, 1])]).pad([
    ...noop.map<Pair>(() => [0, 0]),
    ...s_.flatMap<Pair>((s) => [
      [0, 0],
      [0, 0],
      [0, s - 1],
    ]),
  ]);
  st = st.reshape([...noop, ...kos.flatMap(([k, o, s]) => [k, o * s])]).pad([
    ...noop.map<Pair>(() => [0, 0]),
    ...kidf.flatMap<Pair>(([_k, i, d, f], j) => [
      [0, 0],
      [0, i * f + d - o_[j] * s_[j]],
    ]),
  ]);

  // Undo taking repeats to make shrinking possible.
  st = st
    .reshape([...noop, ...kidf.map(([k, i, d, f]) => k * (i * f + d))])
    .pad([
      ...noop.map<Pair>(() => [0, 0]),
      ...kidf.map<Pair>(([k, i, d, f]) => [
        0,
        Math.ceil((k * (i * f + d)) / i) * i - k * (i * f + d),
      ]),
    ]);
  st = st
    .reshape([
      ...noop,
      ...kidf.flatMap(([k, i, d, f]) => [Math.ceil((k * (i * f + d)) / i), i]),
    ])
    .permute([
      ...range(noop.length),
      ...ks.map((_, j) => noop.length + 2 * j + 1), // input dimensions
      ...ks.map((_, j) => noop.length + 2 * j), // repeat dimensions (to be reduced)
    ]);

  return st;
}

/** Applies dilation to an array directly, for transposed convolution. */
function applyDilation(st: ShapeTracker, dilation: number[]): ShapeTracker {
  if (dilation.every((s) => s === 1)) return st;
  // (k) -> (k,1) -[pad]-> (k,s) -> (k*s) -[shrink]-> (k*s-s+1)
  const s_ = dilation;
  const n = s_.length;
  const prefix = st.shape.slice(0, -n);
  const k_ = st.shape.slice(-n);
  st = st.reshape([...prefix, ...k_.flatMap((k) => [k, 1])]);
  st = st.pad([
    ...prefix.map<Pair>(() => [0, 0]),
    ...s_.flatMap<Pair>((s) => [
      [0, 0],
      [0, s - 1],
    ]),
  ]);
  st = st.reshape([...prefix, ...k_.map((k, i) => k * s_[i])]);
  st = st.shrink([
    ...prefix.map<Pair>((p) => [0, p]),
    ...k_.map<Pair>((k, i) => [0, (k - 1) * s_[i] + 1]),
  ]);
  return st;
}

/**
 * Prepare for a convolution between two arrays.
 *
 * This does not check the validity of the shapes, which should be checked
 * beforehand using `checkConvShape()`.
 */
export function prepareConv(
  stX: ShapeTracker,
  stY: ShapeTracker,
  params: ConvParams,
): [ShapeTracker, ShapeTracker] {
  const v = params.vmapDims;
  const n = stX.shape.length - 2 - v; // spatial dimensions count
  const vmapShape = stX.shape.slice(0, v);

  stX = applyDilation(stX, params.lhsDilation);

  const ks = stY.shape.slice(v + 2); // kernel shape, ks.length == n
  stX = stX.padOrShrink([...rep<Pair>(v + 2, [0, 0]), ...params.padding]);
  stX = pool(stX, ks, params.strides, params.rhsDilation);

  // Permute in channels to the end along with ks, to be reduced.
  stX = stX.moveaxis(v + 1, v + n + 1).reshape([
    ...vmapShape, // vmap dimensions
    stX.shape[v], // batch size
    1, // output channels
    ...stX.shape.slice(v + 2, v + n + 2), // spatial dimensions
    stX.shape[v + 1] * prod(ks), // reduction
  ]);
  stY = stY.reshape([
    ...vmapShape, // vmap dimensions
    1, // batch size (broadcasts with stX's batch size)
    stY.shape[v], // output channels
    ...rep(n, 1), // spatial dimensions
    stY.shape[v + 1] * prod(ks), // reduction
  ]);

  return [stX, stY];
}
