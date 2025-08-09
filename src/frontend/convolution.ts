// Implementation of the Conv primitive (lax.conv_general_dilated).
//
// This handles both forward and transposed convolutions.
//
// Reference:
//  - https://openxla.org/xla/operation_semantics#conv_convolution
//  - https://github.com/jax-ml/jax/blob/main/jax/_src/lax/convolution.py

import { Pair, ShapeTracker } from "../shape";
import { prod, range, rep, zipn } from "../utils";

/** Definition of a general dilated convolution. Should be valid on creation. */
export interface ConvParams {
  strides: number[];
  padding: [number, number][];
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
 *
 * If the check succeeds, returns the output shape.
 */
export function checkConvShape(
  lhsShape: number[],
  rhsShape: number[],
  { strides, padding, lhsDilation, rhsDilation }: ConvParams,
): number[] {
  if (lhsShape.length !== rhsShape.length) {
    throw new Error(
      `conv() requires inputs with the same number of dimensions, got ${lhsShape.length} and ${rhsShape.length}`,
    );
  }
  const n = lhsShape.length - 2;
  if (n < 0) throw new Error("conv() requires at least 2D inputs");
  if (strides.length !== n) throw new Error("conv() strides != spatial dims");
  if (padding.length !== n) throw new Error("conv() padding != spatial dims");
  if (lhsDilation.length !== n)
    throw new Error("conv() lhsDilation != spatial dimensions");
  if (rhsDilation.length !== n)
    throw new Error("conv() rhsDilation != spatial dimensions");
  if (lhsShape[1] !== rhsShape[1])
    throw new Error(`conv() input channels: ${lhsShape[1]} != ${rhsShape[1]}`);

  const outShape = [lhsShape[0], rhsShape[0]]; // Batch size and out_channels

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

    const [x, k] = [lhsShape[i + 2], rhsShape[i + 2]];
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

/**
 * Takes a shape tracker and reshapes it such that and a kernel size `ks`, then
 * reshapes the last `ks.length` dimensions into `2 * ks.length` dimensions by
 * treating them as spatial dimensions convolved with a kernel.
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

  const noop = st.shape.slice(0, -ks.length);

  const i_ = st.shape.slice(-ks.length);
  const s_ = strides;
  const d_ = dilation;
  const o_ = zipn(i_, d_, ks, s_).map(([i, d, k, s]) =>
    Math.ceil((i - d * (k - 1)) / s),
  );

  // TODO: Alternative implementation for d=1 and k<=s, faster (e.g., max pooling).

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

/** Applies dilation to an array directly, for transposed convolution. */
function applyDilation(st: ShapeTracker, dilation: number[]): ShapeTracker {
  if (dilation.every((s) => s === 1)) return st;
  // (k) -> (k,1) -[pad]-> (k,s) -> (k*s) -[shrink]-> (k*s-s+1)
  const s_ = dilation;
  const [a, b, ...k_] = st.shape;
  st = st.reshape([a, b, ...k_.flatMap((k) => [k, 1])]);
  st = st.pad([
    [0, 0],
    [0, 0],
    ...s_.flatMap<[number, number]>((s) => [
      [0, 0],
      [0, s - 1],
    ]),
  ]);
  st = st.reshape([a, b, ...k_.map((k, i) => k * s_[i])]);
  st = st.shrink([
    [0, a],
    [0, b],
    ...k_.map<[number, number]>((k, i) => [0, (k - 1) * s_[i] + 1]),
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
  const n = stX.shape.length - 2; // spatial dimensions count

  stX = applyDilation(stX, params.lhsDilation);

  const ks = stY.shape.slice(2); // kernel shape, ks.length == n
  stX = stX.padOrShrink([[0, 0], [0, 0], ...params.padding]);
  stX = pool(stX, ks, params.strides, params.rhsDilation);

  // Permute in channels to the end along with ks, to be reduced.
  stX = stX.moveaxis(1, n + 1).reshape([
    stX.shape[0], // batch size
    1, // output channels
    ...stX.shape.slice(2, n + 2), // spatial dimensions
    stX.shape[1] * prod(ks), // reduction
  ]);
  stY = stY.reshape([
    stY.shape[0], // output channels
    ...rep(n, 1), // spatial dimensions
    stY.shape[1] * prod(ks), // reduction
  ]);

  return [stX, stY];
}
