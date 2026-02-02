import { DType, numpy as np } from "../../src";

/** TypedArray type for float data - either Float32Array or Float64Array based on dtype */
export type FloatArray = Float32Array | Float64Array;

/** TypedArray constructor type */
export type FloatArrayConstructor = typeof Float32Array | typeof Float64Array;

/**
 * Result from the DLM smoother function (dlmSmo).
 * Contains both filtered and smoothed estimates.
 * @internal - Used only within the library implementation.
 */
export interface DlmSmoResult {
  /** Smoothed state means - array of 2x1 np.Arrays */
  x: np.Array[];
  /** Smoothed state covariances - array of 2x2 np.Arrays */
  C: np.Array[];
  /** Filtered (one-step-ahead prediction) state means - array of 2x1 np.Arrays */
  xf: np.Array[];
  /** Filtered (one-step-ahead prediction) state covariances - array of 2x2 np.Arrays */
  Cf: np.Array[];
  /** Filter predictions (F * x_pred) */
  yhat: FloatArray;
  /** Prediction standard deviations */
  ystd: FloatArray;
  /** Smoothed state standard deviations [time][state_dim] */
  xstd: [number, number][];
  /** Raw residuals (y - yhat) */
  resid0: FloatArray;
  /** Scaled residuals (resid0 / V) */
  resid: FloatArray;
  /** Standardized prediction residuals (v / sqrt(Cp)) */
  resid2: FloatArray;
  /** Innovations (prediction errors) */
  v: FloatArray;
  /** Innovation covariances */
  Cp: FloatArray;
  /** Sum of squared raw residuals */
  ssy: number;
  /** Residual variance */
  s2: number;
  /** Number of observations */
  nobs: number;
  /** -2 * log likelihood */
  lik: number;
  /** Mean squared error of standardized residuals */
  mse: number;
  /** Mean absolute percentage error */
  mape: number;
}

/**
 * Result from the DLM fit function (dlmFit).
 * Numeric arrays use TypedArrays (Float32Array or Float64Array based on dtype).
 */
export interface DlmFitResult {
  /** Filtered state means [state_dim][time] */
  xf: FloatArray[];
  /** Filtered covariances [row][col][time] */
  Cf: FloatArray[][];
  /** Smoothed state means [state_dim][time] */
  x: FloatArray[];
  /** Smoothed covariances [row][col][time] */
  C: FloatArray[][];
  /** Smoothed state standard deviations [time][state_dim] */
  xstd: FloatArray[];
  /** State transition matrix G (2x2) */
  G: number[][];
  /** Observation matrix F (1x2 flattened) */
  F: number[];
  /** State noise covariance W (2x2) */
  W: number[][];
  /** Observations */
  y: FloatArray;
  /** Observation noise standard deviations */
  V: FloatArray;
  /** Initial state mean (after first smoother pass) */
  x0: number[];
  /** Initial state covariance (scaled) */
  C0: number[][];
  /** Covariates (empty for basic model) */
  XX: number[];
  /** Filter predictions */
  yhat: FloatArray;
  /** Prediction standard deviations */
  ystd: FloatArray;
  /** Raw residuals */
  resid0: FloatArray;
  /** Scaled residuals */
  resid: FloatArray;
  /** Sum of squared residuals */
  ssy: number;
  /** Innovations */
  v: FloatArray;
  /** Innovation covariances */
  Cp: FloatArray;
  /** Residual variance */
  s2: number;
  /** Number of observations */
  nobs: number;
  /** -2 * log likelihood */
  lik: number;
  /** Mean squared error */
  mse: number;
  /** Mean absolute percentage error */
  mape: number;
  /** Standardized residuals */
  resid2: FloatArray;
  /** Class identifier */
  class: "dlmfit";
}

/**
 * Helper to dispose multiple np.Arrays at once.
 */
export function disposeAll(...arrays: (np.Array | undefined | null)[]): void {
  for (const arr of arrays) {
    if (arr) {
      arr.dispose();
    }
  }
}

/**
 * Get the appropriate TypedArray constructor based on DType.
 */
export function getFloatArrayType(dtype: DType): FloatArrayConstructor {
  return dtype === DType.Float32 ? Float32Array : Float64Array;
}
