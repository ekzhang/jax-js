import * as tf from "@tensorflow/tfjs-core";

import { Array, DType } from "./core";
import * as core from "./core";
import { deepEqual } from "./utils";

export { Array, DType };

export const float32 = DType.Float32;
export const int32 = DType.Int32;
export const bool = DType.Bool;
export const complex64 = DType.Complex64;

// Note: These primitive wrappers have fudged types.
//
// They can take any `TracerValue` and return any `Tracer` subclass based on the
// current stack of interpreters. But we hide that away from users to mimic
// JAX's composable tracing transformations.

export type ArrayLike = Array | number | boolean;

export const add = core.add as (x: ArrayLike, y: ArrayLike) => Array;
export const mul = core.mul as (x: ArrayLike, y: ArrayLike) => Array;
export const neg = core.neg as (x: ArrayLike) => Array;
export const sin = core.sin as (x: ArrayLike) => Array;
export const cos = core.cos as (x: ArrayLike) => Array;
export const greater = core.greater as (x: ArrayLike, y: ArrayLike) => Array;
export const less = core.less as (x: ArrayLike, y: ArrayLike) => Array;
export const transpose = core.transpose as (
  x: ArrayLike,
  perm?: number[],
) => Array;
export const broadcast = core.broadcast as (
  x: ArrayLike,
  shape: number[],
  axes: number[],
) => Array;
export const reduceSum = core.reduceSum as (
  x: ArrayLike,
  axis?: number | number[],
) => Array;
export const moveaxis = core.moveaxis as (
  x: ArrayLike,
  src: number,
  dst: number,
) => Array;

/** Compute the number of dimensions of an array. */
export const ndim = core.ndim as (x: ArrayLike) => number;

export function array(
  values: Array | tf.TensorLike,
  { shape, dtype }: { shape?: number[]; dtype?: DType } = {},
): Array {
  if (values instanceof Array) {
    let data = values.data;
    if (shape) {
      data = tf.reshape(data, shape);
    }
    if (dtype) {
      data = tf.cast(data, dtype);
    }
    return new Array(data);
  } else {
    return new Array(tf.tensor(values, shape, dtype));
  }
}

/** Return if two arrays are element-wise equal within a tolerance. */
export function allclose(
  actual: ArrayLike,
  expected: ArrayLike,
  options?: { rtol?: number; atol?: number },
): boolean {
  const { rtol = 1e-5, atol = 1e-8 } = options ?? {};

  const x = array(actual);
  const y = array(expected);
  if (!deepEqual(x.shape, y.shape)) {
    return false;
  }
  return Boolean(
    tf
      .all(
        tf.lessEqual(
          tf.abs(tf.sub(x.data, y.data)),
          tf.add(atol, tf.mul(rtol, y.data.abs())),
        ),
      )
      .dataSync()[0],
  );
}
