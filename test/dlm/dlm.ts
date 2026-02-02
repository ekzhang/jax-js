import type { DlmFitResult, DlmSmoResult, FloatArray } from "./types";
import { disposeAll, getFloatArrayType } from "./types";
import { DType, jit, lax, numpy as np } from "../../src";

// Public type exports
export type { DlmFitResult, FloatArray } from "./types";

/**
 * Execution mode for the DLM smoother.
 * - 'for-js': Pure JavaScript with TypedArrays (fastest for small matrices, no jax-js overhead)
 * - 'for': Use explicit for-loops with jax-js operations
 * - 'scan': Use lax.scan primitive (slower with interpreter fallback, ~2.8s for n=100)
 *           but enables future JIT compilation and automatic differentiation
 * - 'jit': Use jit-compiled lax.scan (fastest when body is compiled)
 */
export type DlmMode = "for-js" | "for" | "scan" | "jit";

/**
 * DLM Smoother - Kalman filter (forward) + Rauch-Tung-Striebel smoother (backward).
 *
 * Implements the state-space model:
 *
 *   y(t) = F·x(t) + v,    observation equation
 *   x(t) = G·x(t-1) + w,  state transition equation
 *
 * where:
 *   x(1) ~ N(x0, C0)      initial state distribution
 *   v ~ N(0, V)           observation noise
 *   w ~ N(0, W)           state evolution noise
 *
 * The forward Kalman filter computes one-step-ahead predictions.
 * The backward RTS smoother refines estimates using all observations.
 *
 * Reference: Durbin & Koopman (2012), "Time Series Analysis by State Space Methods"
 *
 * @param y - Observations (n×1)
 * @param F - Observation matrix (1×m), maps state to observation
 * @param V_std - Observation noise std devs (n×1)
 * @param x0_data - Initial state mean (m×1 as nested array)
 * @param G - State transition matrix (m×m)
 * @param W - State noise covariance (m×m)
 * @param C0_data - Initial state covariance (m×m as nested array)
 * @param dtype - Computation precision
 * @param mode - Execution mode: 'for' (faster) or 'scan' (enables JIT/AD)
 * @returns Smoothed and filtered state estimates with diagnostics
 * @internal
 */
const dlmSmo = async (
  y: FloatArray,
  F: np.Array,
  V_std: FloatArray,
  x0_data: number[][],
  G: np.Array,
  W: np.Array,
  C0_data: number[][],
  dtype: DType = DType.Float64,
  mode: DlmMode = "for",
): Promise<DlmSmoResult> => {
  if (mode === "for-js") {
    return dlmSmoForJs(y, F, V_std, x0_data, G, W, C0_data, dtype);
  }
  if (mode === "scan") {
    return dlmSmoScan(y, F, V_std, x0_data, G, W, C0_data, dtype, false);
  }
  if (mode === "jit") {
    return dlmSmoScan(y, F, V_std, x0_data, G, W, C0_data, dtype, true);
  }
  return dlmSmoFor(y, F, V_std, x0_data, G, W, C0_data, dtype);
};

/**
 * DLM Smoother implementation using pure JavaScript with TypedArrays.
 * Fastest for small matrices (2×2) by eliminating all jax-js overhead.
 * @internal
 */
const dlmSmoForJs = async (
  y: FloatArray,
  F: np.Array,
  V_std: FloatArray,
  x0_data: number[][],
  G: np.Array,
  W: np.Array,
  C0_data: number[][],
  dtype: DType = DType.Float64,
): Promise<DlmSmoResult> => {
  const n = y.length;
  const FA = getFloatArrayType(dtype);

  // Extract system matrices to plain arrays (once, not per iteration)
  const F_data = await F.ref.data();
  const G_data = await G.ref.data();
  const W_data = await W.ref.data();

  // System matrices as scalars (2×2 layout: row-major)
  // F is 1×2: F = [f0, f1]
  const f0 = F_data[0],
    f1 = F_data[1];
  // G is 2×2: G = [[g00, g01], [g10, g11]]
  const g00 = G_data[0],
    g01 = G_data[1],
    g10 = G_data[2],
    g11 = G_data[3];
  // W is 2×2: W = [[w00, w01], [w10, w11]]
  const w00 = W_data[0],
    w01 = W_data[1],
    w10 = W_data[2],
    w11 = W_data[3];

  // ─────────────────────────────────────────────────────────────────────────
  // Storage using TypedArrays for cache efficiency
  // State vectors stored as pairs: [x0, x1] per timestep
  // Covariance matrices stored as 4-tuples: [C00, C01, C10, C11] per timestep
  // ─────────────────────────────────────────────────────────────────────────
  const x_pred_0 = new FA(n); // x_pred[i][0]
  const x_pred_1 = new FA(n); // x_pred[i][1]
  const C_pred_00 = new FA(n); // C_pred[i][0,0]
  const C_pred_01 = new FA(n); // C_pred[i][0,1]
  const C_pred_10 = new FA(n); // C_pred[i][1,0]
  const C_pred_11 = new FA(n); // C_pred[i][1,1]
  const K_0 = new FA(n); // Kalman gain K[0]
  const K_1 = new FA(n); // Kalman gain K[1]
  const v_array = new FA(n); // Innovations
  const Cp_array = new FA(n); // Innovation covariances

  // Initialize from prior
  x_pred_0[0] = x0_data[0][0];
  x_pred_1[0] = x0_data[1][0];
  C_pred_00[0] = C0_data[0][0];
  C_pred_01[0] = C0_data[0][1];
  C_pred_10[0] = C0_data[1][0];
  C_pred_11[0] = C0_data[1][1];

  // ─────────────────────────────────────────────────────────────────────────
  // Forward Kalman Filter (pure JS)
  // ─────────────────────────────────────────────────────────────────────────
  for (let i = 0; i < n; i++) {
    const xi0 = x_pred_0[i],
      xi1 = x_pred_1[i];
    const Ci00 = C_pred_00[i],
      Ci01 = C_pred_01[i];
    const Ci10 = C_pred_10[i],
      Ci11 = C_pred_11[i];

    // v = y - F·x  (scalar: F is 1×2, x is 2×1)
    const Fx = f0 * xi0 + f1 * xi1;
    const v = y[i] - Fx;
    v_array[i] = v;

    // Cp = F·C·F' + V²  (scalar: 1×1)
    // F·C = [f0*C00 + f1*C10, f0*C01 + f1*C11] (1×2)
    // F·C·F' = (f0*C00 + f1*C10)*f0 + (f0*C01 + f1*C11)*f1
    const FCFt = (f0 * Ci00 + f1 * Ci10) * f0 + (f0 * Ci01 + f1 * Ci11) * f1;
    const Cp = FCFt + V_std[i] * V_std[i];
    Cp_array[i] = Cp;

    // K = G·C·F' / Cp  (2×1: result of 2×2 · 2×2 · 2×1 / scalar)
    // G·C = [[g00*C00 + g01*C10, g00*C01 + g01*C11],
    //        [g10*C00 + g11*C10, g10*C01 + g11*C11]]
    // (G·C)·F' = [row0·F', row1·F'] where each row·F' = row0*f0 + row1*f1
    const GC00 = g00 * Ci00 + g01 * Ci10;
    const GC01 = g00 * Ci01 + g01 * Ci11;
    const GC10 = g10 * Ci00 + g11 * Ci10;
    const GC11 = g10 * Ci01 + g11 * Ci11;
    const GCFt0 = GC00 * f0 + GC01 * f1;
    const GCFt1 = GC10 * f0 + GC11 * f1;
    const k0 = GCFt0 / Cp;
    const k1 = GCFt1 / Cp;
    K_0[i] = k0;
    K_1[i] = k1;

    if (i < n - 1) {
      // L = G - K·F  (2×2 = 2×2 - 2×1 · 1×2)
      // K·F = [[k0*f0, k0*f1], [k1*f0, k1*f1]]
      const L00 = g00 - k0 * f0;
      const L01 = g01 - k0 * f1;
      const L10 = g10 - k1 * f0;
      const L11 = g11 - k1 * f1;

      // x_next = G·x + K·v  (2×1)
      x_pred_0[i + 1] = g00 * xi0 + g01 * xi1 + k0 * v;
      x_pred_1[i + 1] = g10 * xi0 + g11 * xi1 + k1 * v;

      // C_next = G·C·L' + W  (2×2)
      // We already have G·C, now compute (G·C)·L'
      // L' = [[L00, L10], [L01, L11]]
      // (G·C)·L' row 0: [GC00*L00 + GC01*L01, GC00*L10 + GC01*L11]
      // (G·C)·L' row 1: [GC10*L00 + GC11*L01, GC10*L10 + GC11*L11]
      C_pred_00[i + 1] = GC00 * L00 + GC01 * L01 + w00;
      C_pred_01[i + 1] = GC00 * L10 + GC01 * L11 + w01;
      C_pred_10[i + 1] = GC10 * L00 + GC11 * L01 + w10;
      C_pred_11[i + 1] = GC10 * L10 + GC11 * L11 + w11;
    }
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Backward RTS Smoother (pure JS)
  // ─────────────────────────────────────────────────────────────────────────
  const x_smooth_0 = new FA(n);
  const x_smooth_1 = new FA(n);
  const C_smooth_00 = new FA(n);
  const C_smooth_01 = new FA(n);
  const C_smooth_10 = new FA(n);
  const C_smooth_11 = new FA(n);

  // Initialize smoother: r = [0, 0], N = [[0,0],[0,0]]
  let r0 = 0,
    r1 = 0;
  let N00 = 0,
    N01 = 0,
    N10 = 0,
    N11 = 0;

  for (let i = n - 1; i >= 0; i--) {
    const xi0 = x_pred_0[i],
      xi1 = x_pred_1[i];
    const Ci00 = C_pred_00[i],
      Ci01 = C_pred_01[i];
    const Ci10 = C_pred_10[i],
      Ci11 = C_pred_11[i];
    const k0 = K_0[i],
      k1 = K_1[i];
    const vi = v_array[i];
    const Cpi = Cp_array[i];

    // L = G - K·F
    const L00 = g00 - k0 * f0;
    const L01 = g01 - k0 * f1;
    const L10 = g10 - k1 * f0;
    const L11 = g11 - k1 * f1;

    // F'·Cp⁻¹ = [f0/Cp, f1/Cp] (2×1)
    const FtCpInv0 = f0 / Cpi;
    const FtCpInv1 = f1 / Cpi;

    // r_new = F'·Cp⁻¹·v + L'·r
    // L' = [[L00, L10], [L01, L11]]
    // L'·r = [L00*r0 + L10*r1, L01*r0 + L11*r1]
    const r0_new = FtCpInv0 * vi + L00 * r0 + L10 * r1;
    const r1_new = FtCpInv1 * vi + L01 * r0 + L11 * r1;

    // N_new = F'·Cp⁻¹·F + L'·N·L
    // F'·Cp⁻¹·F = [[f0*f0/Cp, f0*f1/Cp], [f1*f0/Cp, f1*f1/Cp]]
    const FtCpInvF00 = FtCpInv0 * f0;
    const FtCpInvF01 = FtCpInv0 * f1;
    const FtCpInvF10 = FtCpInv1 * f0;
    const FtCpInvF11 = FtCpInv1 * f1;

    // L'·N = [[L00*N00 + L10*N10, L00*N01 + L10*N11],
    //         [L01*N00 + L11*N10, L01*N01 + L11*N11]]
    const LtN00 = L00 * N00 + L10 * N10;
    const LtN01 = L00 * N01 + L10 * N11;
    const LtN10 = L01 * N00 + L11 * N10;
    const LtN11 = L01 * N01 + L11 * N11;

    // (L'·N)·L: matrix multiply (L'·N) by L
    // [i,j] = sum_k LtN[i,k] * L[k,j]
    const N00_new = FtCpInvF00 + LtN00 * L00 + LtN01 * L10;
    const N01_new = FtCpInvF01 + LtN00 * L01 + LtN01 * L11;
    const N10_new = FtCpInvF10 + LtN10 * L00 + LtN11 * L10;
    const N11_new = FtCpInvF11 + LtN10 * L01 + LtN11 * L11;

    // x_smooth = x_pred + C_pred·r_new
    x_smooth_0[i] = xi0 + Ci00 * r0_new + Ci01 * r1_new;
    x_smooth_1[i] = xi1 + Ci10 * r0_new + Ci11 * r1_new;

    // C_smooth = C_pred - C_pred·N_new·C_pred
    // C·N = [[Ci00*N00_new + Ci01*N10_new, Ci00*N01_new + Ci01*N11_new],
    //        [Ci10*N00_new + Ci11*N10_new, Ci10*N01_new + Ci11*N11_new]]
    const CN00 = Ci00 * N00_new + Ci01 * N10_new;
    const CN01 = Ci00 * N01_new + Ci01 * N11_new;
    const CN10 = Ci10 * N00_new + Ci11 * N10_new;
    const CN11 = Ci10 * N01_new + Ci11 * N11_new;

    // (C·N)·C
    C_smooth_00[i] = Ci00 - (CN00 * Ci00 + CN01 * Ci10);
    C_smooth_01[i] = Ci01 - (CN00 * Ci01 + CN01 * Ci11);
    C_smooth_10[i] = Ci10 - (CN10 * Ci00 + CN11 * Ci10);
    C_smooth_11[i] = Ci11 - (CN10 * Ci01 + CN11 * Ci11);

    // Update for next iteration
    r0 = r0_new;
    r1 = r1_new;
    N00 = N00_new;
    N01 = N01_new;
    N10 = N10_new;
    N11 = N11_new;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Compute output statistics
  // ─────────────────────────────────────────────────────────────────────────
  const yhat = new FA(n);
  const ystd = new FA(n);
  const xstd: [number, number][] = new Array(n);
  const resid0 = new FA(n);
  const resid = new FA(n);
  const resid2 = new FA(n);
  let ssy = 0;
  let lik = 0;

  for (let i = 0; i < n; i++) {
    yhat[i] = x_pred_0[i]; // F·x = [1,0]·x = x[0]
    xstd[i] = [
      Math.sqrt(Math.abs(C_smooth_00[i])),
      Math.sqrt(Math.abs(C_smooth_11[i])),
    ];
    ystd[i] = Math.sqrt(C_smooth_00[i] + V_std[i] * V_std[i]);

    const r0_val = y[i] - yhat[i];
    resid0[i] = r0_val;
    resid[i] = r0_val / V_std[i];
    resid2[i] = v_array[i] / Math.sqrt(Cp_array[i]);

    ssy += r0_val * r0_val;
    lik += v_array[i] ** 2 / Cp_array[i] + Math.log(Cp_array[i]);
  }

  let s2 = 0,
    mse = 0,
    mape = 0;
  for (let i = 0; i < n; i++) {
    s2 += resid[i] ** 2;
    mse += resid2[i] ** 2;
    mape += Math.abs(resid2[i]) / Math.abs(y[i]);
  }
  s2 /= n;
  mse /= n;
  mape /= n;

  // ─────────────────────────────────────────────────────────────────────────
  // Convert to np.Array outputs (required by DlmSmoResult interface)
  // This is the only jax-js usage - just for output format compatibility
  // ─────────────────────────────────────────────────────────────────────────
  const x_smooth: np.Array[] = new Array(n);
  const C_smooth: np.Array[] = new Array(n);
  const x_pred: np.Array[] = new Array(n);
  const C_pred: np.Array[] = new Array(n);

  for (let i = 0; i < n; i++) {
    x_smooth[i] = np.array([[x_smooth_0[i]], [x_smooth_1[i]]], { dtype });
    C_smooth[i] = np.array(
      [
        [C_smooth_00[i], C_smooth_01[i]],
        [C_smooth_10[i], C_smooth_11[i]],
      ],
      { dtype },
    );
    x_pred[i] = np.array([[x_pred_0[i]], [x_pred_1[i]]], { dtype });
    C_pred[i] = np.array(
      [
        [C_pred_00[i], C_pred_01[i]],
        [C_pred_10[i], C_pred_11[i]],
      ],
      { dtype },
    );
  }

  return {
    x: x_smooth,
    C: C_smooth,
    xf: x_pred,
    Cf: C_pred,
    yhat,
    ystd,
    xstd,
    resid0,
    resid,
    resid2,
    v: v_array,
    Cp: Cp_array,
    ssy,
    s2,
    nobs: n,
    lik,
    mse,
    mape,
  };
};

/**
 * DLM Smoother implementation using explicit for-loops.
 * Faster with interpreter execution.
 * @internal
 */
const dlmSmoFor = async (
  y: FloatArray,
  F: np.Array,
  V_std: FloatArray,
  x0_data: number[][],
  G: np.Array,
  W: np.Array,
  C0_data: number[][],
  dtype: DType = DType.Float64,
): Promise<DlmSmoResult> => {
  const n = y.length;
  const FA = getFloatArrayType(dtype);

  // ─────────────────────────────────────────────────────────────────────────
  // Storage allocation for Kalman filter outputs
  // ─────────────────────────────────────────────────────────────────────────
  const x_pred: np.Array[] = new Array(n); // One-step-ahead state predictions (m×1 each)
  const C_pred: np.Array[] = new Array(n); // Prediction covariances (m×m each)
  const K_array: np.Array[] = new Array(n); // Kalman gains (m×1 each)
  const v_array = new FA(n); // Innovation residuals (scalar per timestep)
  const Cp_array = new FA(n); // Innovation covariances (scalar per timestep)

  // Initialize state from prior distribution x(1) ~ N(x0, C0)
  x_pred[0] = np.array(x0_data, { dtype });
  C_pred[0] = np.array(C0_data, { dtype });

  // Precompute F' for reuse in both filter and smoother loops
  const Ft = np.transpose(F.ref);

  // ─────────────────────────────────────────────────────────────────────────
  // Forward Kalman Filter
  // Computes one-step-ahead predictions: x(t|t-1) and C(t|t-1)
  // ─────────────────────────────────────────────────────────────────────────
  for (let i = 0; i < n; i++) {
    const xi = x_pred[i]; // x(t|t-1): predicted state
    const Ci = C_pred[i]; // C(t|t-1): predicted covariance

    // Innovation (prediction error): v(t) = y(t) - F·x(t|t-1)
    const v = np.subtract(
      np.array([y[i]], { dtype }),
      np.matmul(F.ref, xi.ref),
    );
    v_array[i] = (await v.ref.data())[0];

    // Innovation covariance: Cp(t) = F·C(t|t-1)·F' + V(t)
    const Cp = np.add(
      np.einsum("ij,jk,lk->il", F.ref, Ci.ref, F.ref),
      np.array([[V_std[i] ** 2]], { dtype }),
    );
    Cp_array[i] = (await Cp.ref.data())[0];

    // Kalman gain: K(t) = G·C(t|t-1)·F' / Cp(t)
    // This determines how much the innovation corrects the state prediction
    const K = np.divide(
      np.einsum("ij,jk,lk->il", G.ref, Ci.ref, F.ref),
      Cp_array[i],
    );
    K_array[i] = np.reshape(K.ref, [2, 1]);

    // Propagate state prediction to next timestep (except at final observation)
    if (i < n - 1) {
      // L(t) = G - K(t)·F  (used for covariance propagation)
      const L = np.subtract(G.ref, np.matmul(K.ref, F.ref));

      // State prediction: x(t+1|t) = G·x(t|t-1) + K(t)·v(t)
      x_pred[i + 1] = np.add(
        np.matmul(G.ref, xi.ref),
        np.matmul(K.ref, np.array([[v_array[i]]], { dtype })),
      );

      // Covariance prediction: C(t+1|t) = G·C(t|t-1)·L' + W
      C_pred[i + 1] = np.add(
        np.einsum("ij,jk,lk->il", G.ref, Ci.ref, L.ref),
        W.ref,
      );
    }

    disposeAll(K, v, Cp);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Backward RTS (Rauch-Tung-Striebel) Smoother
  // Refines state estimates using future observations: x(t|n) and C(t|n)
  // Uses the "information form" with auxiliary variables r and N
  // ─────────────────────────────────────────────────────────────────────────
  let r = np.array([[0.0], [0.0]], { dtype }); // Smoothing residual (m×1)
  let N = np.array(
    [
      [0.0, 0.0],
      [0.0, 0.0],
    ],
    { dtype },
  ); // Smoothing precision (m×m)
  const x_smooth: np.Array[] = new Array(n);
  const C_smooth: np.Array[] = new Array(n);

  for (let i = n - 1; i >= 0; i--) {
    const xi = x_pred[i]; // x(t|t-1)
    const Ci = C_pred[i]; // C(t|t-1)
    const Ki = K_array[i]; // K(t)

    // L(t) = G - K(t)·F  (transition minus Kalman correction)
    const L = np.subtract(G.ref, np.matmul(Ki.ref, F.ref));

    // F'·Cp⁻¹ — weighted observation operator
    const FtCpInv = np.divide(Ft.ref, Cp_array[i]);

    // Smoother residual update: r(t-1) = F'·Cp⁻¹·v(t) + L'·r(t)
    const r_new = np.add(
      np.multiply(FtCpInv.ref, v_array[i]),
      np.matmul(np.transpose(L.ref), r.ref),
    );

    // Smoother precision update: N(t-1) = F'·Cp⁻¹·F + L'·N(t)·L
    const N_new = np.add(
      np.matmul(FtCpInv, F.ref),
      np.einsum("ji,jk,kl->il", L.ref, N.ref, L.ref),
    );

    // Smoothed state: x(t|n) = x(t|t-1) + C(t|t-1)·r(t-1)
    x_smooth[i] = np.add(xi.ref, np.matmul(Ci.ref, r_new.ref));

    // Smoothed covariance: C(t|n) = C(t|t-1) - C(t|t-1)·N(t-1)·C(t|t-1)
    C_smooth[i] = np.subtract(
      Ci.ref,
      np.einsum("ij,jk,kl->il", Ci.ref, N_new.ref, Ci.ref),
    );

    disposeAll(r, N);
    r = r_new;
    N = N_new;
  }

  // Cleanup temporary arrays
  disposeAll(r, N, Ft);
  for (let i = 0; i < n; i++) K_array[i].dispose();

  // ─────────────────────────────────────────────────────────────────────────
  // Compute output statistics and diagnostics
  // ─────────────────────────────────────────────────────────────────────────
  const yhat = new FA(n); // Filter predictions: F·x(t|t-1)
  const ystd = new FA(n); // Prediction std: sqrt(C[0,0] + V²)
  const xstd: [number, number][] = new Array(n); // Smoothed state stds
  const resid0 = new FA(n); // Raw residuals: y - yhat
  const resid = new FA(n); // Scaled residuals: resid0 / V
  const resid2 = new FA(n); // Standardized prediction residuals: v / sqrt(Cp)
  let ssy = 0; // Sum of squared raw residuals
  let lik = 0; // -2 × log-likelihood (for model comparison)

  for (let i = 0; i < n; i++) {
    // Extract filter prediction (first state component = level)
    const yhat_i = (await x_pred[i].ref.data())[0];
    const C_s = await C_smooth[i].ref.data();

    yhat[i] = yhat_i;
    // State uncertainty: sqrt of diagonal elements of C(t|n)
    xstd[i] = [Math.sqrt(Math.abs(C_s[0])), Math.sqrt(Math.abs(C_s[3]))];
    // Observation uncertainty: state + observation noise
    ystd[i] = Math.sqrt(C_s[0] + V_std[i] ** 2);

    // Residual diagnostics
    const r0 = y[i] - yhat_i;
    resid0[i] = r0; // Raw residual
    resid[i] = r0 / V_std[i]; // Scaled by obs noise
    resid2[i] = v_array[i] / Math.sqrt(Cp_array[i]); // Standardized innovation

    // Accumulate likelihood components
    ssy += r0 * r0;
    lik += v_array[i] ** 2 / Cp_array[i] + Math.log(Cp_array[i]);
  }

  // Aggregate error metrics
  let s2 = 0,
    mse = 0,
    mape = 0;
  for (let i = 0; i < n; i++) {
    s2 += resid[i] ** 2;
    mse += resid2[i] ** 2;
    mape += Math.abs(resid2[i]) / Math.abs(y[i]);
  }
  s2 /= n; // Residual variance
  mse /= n; // Mean squared error of standardized residuals
  mape /= n; // Mean absolute percentage error

  return {
    x: x_smooth,
    C: C_smooth,
    xf: x_pred,
    Cf: C_pred,
    yhat,
    ystd,
    xstd,
    resid0,
    resid,
    resid2,
    v: v_array,
    Cp: Cp_array,
    ssy,
    s2,
    nobs: n,
    lik,
    mse,
    mape,
  };
};

/**
 * DLM Smoother implementation using lax.scan primitive.
 * When useJit=true, wraps scans with jit() for ~80× speedup.
 * @internal
 */
const dlmSmoScan = async (
  y: FloatArray,
  F: np.Array,
  V_std: FloatArray,
  x0_data: number[][],
  G: np.Array,
  W: np.Array,
  C0_data: number[][],
  dtype: DType = DType.Float64,
  useJit: boolean = false,
): Promise<DlmSmoResult> => {
  const n = y.length;
  const FA = getFloatArrayType(dtype);

  // Stack observations: shape [n, 1, 1] for matmul compatibility
  const y_arr = np.array(
    Array.from(y).map((yi) => [[yi]]),
    { dtype },
  );
  // Stack V² (variance): shape [n, 1, 1]
  const V2_arr = np.array(
    Array.from(V_std).map((v) => [[v * v]]),
    { dtype },
  );

  // Initial state
  const x0 = np.array(x0_data, { dtype });
  const C0 = np.array(C0_data, { dtype });

  // Precompute F' for reuse
  const Ft = np.transpose(F.ref);

  // ─────────────────────────────────────────────────────────────────────────
  // Forward Kalman Filter using lax.scan
  // ─────────────────────────────────────────────────────────────────────────

  type ForwardCarry = { x: np.Array; C: np.Array };
  type ForwardX = { y: np.Array; V2: np.Array };
  type ForwardY = {
    x_pred: np.Array;
    C_pred: np.Array;
    K: np.Array;
    v: np.Array;
    Cp: np.Array;
  };

  const forwardStep = (
    carry: ForwardCarry,
    inp: ForwardX,
  ): [ForwardCarry, ForwardY] => {
    const { x: xi, C: Ci } = carry;
    const { y: yi, V2: V2i } = inp;

    // Innovation: v = y - F·x
    const v = np.subtract(yi.ref, np.matmul(F.ref, xi.ref));

    // Innovation covariance: Cp = F·C·F' + V²
    const Cp = np.add(np.einsum("ij,jk,lk->il", F.ref, Ci.ref, F.ref), V2i.ref);

    // Kalman gain: K = G·C·F' / Cp
    const GCFt = np.einsum("ij,jk,lk->il", G.ref, Ci.ref, F.ref);
    const K = np.divide(GCFt.ref, Cp.ref);

    // L = G - K·F
    const L = np.subtract(G.ref, np.matmul(K.ref, F.ref));

    // Next state prediction: x_next = G·x + K·v
    const x_next = np.add(np.matmul(G.ref, xi.ref), np.matmul(K.ref, v.ref));

    // Next covariance: C_next = G·C·L' + W
    const C_next = np.add(
      np.einsum("ij,jk,lk->il", G.ref, Ci.ref, L.ref),
      W.ref,
    );

    const output: ForwardY = {
      x_pred: xi.ref,
      C_pred: Ci.ref,
      K: K.ref,
      v: v.ref,
      Cp: Cp.ref,
    };

    // Note: Don't dispose K, v, Cp - they are returned via .ref in output
    // The scan will manage their lifecycle through the output pytree
    disposeAll(GCFt, L);

    return [{ x: x_next, C: C_next }, output];
  };

  // Run forward scan (optionally jit-compiled)
  const [finalCarry, forwardOutputs] = useJit
    ? await jit(
        (x0: np.Array, C0: np.Array, y_arr: np.Array, V2_arr: np.Array) =>
          lax.scan(forwardStep, { x: x0, C: C0 }, { y: y_arr, V2: V2_arr }),
      )(x0, C0, y_arr, V2_arr)
    : lax.scan(forwardStep, { x: x0, C: C0 }, { y: y_arr, V2: V2_arr });

  const {
    x_pred: x_pred_stacked,
    C_pred: C_pred_stacked,
    K: K_stacked,
    v: v_stacked,
    Cp: Cp_stacked,
  } = forwardOutputs;

  // Convert v and Cp to TypedArrays
  const v_data = await v_stacked.ref.data();
  const Cp_data = await Cp_stacked.ref.data();
  const v_array = new FA(n);
  const Cp_array = new FA(n);
  for (let i = 0; i < n; i++) {
    v_array[i] = v_data[i];
    Cp_array[i] = Cp_data[i];
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Backward RTS Smoother using lax.scan with reverse
  // ─────────────────────────────────────────────────────────────────────────

  const x_pred_rev = np.flip(x_pred_stacked.ref, 0);
  const C_pred_rev = np.flip(C_pred_stacked.ref, 0);
  const K_rev = np.flip(K_stacked.ref, 0);
  const v_rev = np.flip(v_stacked.ref, 0);
  const Cp_rev = np.flip(Cp_stacked.ref, 0);

  type BackwardCarry = { r: np.Array; N: np.Array };
  type BackwardX = {
    x_pred: np.Array;
    C_pred: np.Array;
    K: np.Array;
    v: np.Array;
    Cp: np.Array;
  };
  type BackwardY = { x_smooth: np.Array; C_smooth: np.Array };

  const backwardStep = (
    carry: BackwardCarry,
    inp: BackwardX,
  ): [BackwardCarry, BackwardY] => {
    const { r, N } = carry;
    const { x_pred: xi, C_pred: Ci, K: Ki, v: vi, Cp: Cpi } = inp;

    // L = G - K·F
    const L = np.subtract(G.ref, np.matmul(Ki.ref, F.ref));

    // F'·Cp⁻¹
    const FtCpInv = np.divide(Ft.ref, Cpi.ref);

    // r_new = F'·Cp⁻¹·v + L'·r
    const r_new = np.add(
      np.multiply(FtCpInv.ref, vi.ref),
      np.matmul(np.transpose(L.ref), r.ref),
    );

    // N_new = F'·Cp⁻¹·F + L'·N·L
    const N_new = np.add(
      np.matmul(FtCpInv.ref, F.ref),
      np.einsum("ji,jk,kl->il", L.ref, N.ref, L.ref),
    );

    // x_smooth = x_pred + C_pred·r_new
    const x_smooth = np.add(xi.ref, np.matmul(Ci.ref, r_new.ref));

    // C_smooth = C_pred - C_pred·N_new·C_pred
    const C_smooth = np.subtract(
      Ci.ref,
      np.einsum("ij,jk,kl->il", Ci.ref, N_new.ref, Ci.ref),
    );

    disposeAll(L, FtCpInv);

    return [
      { r: r_new, N: N_new },
      { x_smooth, C_smooth },
    ];
  };

  const r0 = np.array([[0.0], [0.0]], { dtype });
  const N0 = np.array(
    [
      [0.0, 0.0],
      [0.0, 0.0],
    ],
    { dtype },
  );

  // Run backward scan (optionally jit-compiled)
  const [finalBackward, backwardOutputs] = useJit
    ? await jit(
        (
          r0: np.Array,
          N0: np.Array,
          x_pred_rev: np.Array,
          C_pred_rev: np.Array,
          K_rev: np.Array,
          v_rev: np.Array,
          Cp_rev: np.Array,
        ) =>
          lax.scan(
            backwardStep,
            { r: r0, N: N0 },
            {
              x_pred: x_pred_rev,
              C_pred: C_pred_rev,
              K: K_rev,
              v: v_rev,
              Cp: Cp_rev,
            },
          ),
      )(r0, N0, x_pred_rev, C_pred_rev, K_rev, v_rev, Cp_rev)
    : lax.scan(
        backwardStep,
        { r: r0, N: N0 },
        {
          x_pred: x_pred_rev,
          C_pred: C_pred_rev,
          K: K_rev,
          v: v_rev,
          Cp: Cp_rev,
        },
      );

  // Reverse smoothed outputs back to forward order
  const x_smooth_stacked = np.flip(backwardOutputs.x_smooth.ref, 0);
  const C_smooth_stacked = np.flip(backwardOutputs.C_smooth.ref, 0);

  // Cleanup (lax.scan consumed init and xs inputs)
  disposeAll(finalCarry.x, finalCarry.C, finalBackward.r, finalBackward.N);
  disposeAll(Ft);
  disposeAll(backwardOutputs.x_smooth, backwardOutputs.C_smooth);
  disposeAll(v_stacked, Cp_stacked, K_stacked);

  // Convert stacked outputs to per-timestep arrays
  const x_pred: np.Array[] = new Array(n);
  const C_pred: np.Array[] = new Array(n);
  const x_smooth: np.Array[] = new Array(n);
  const C_smooth: np.Array[] = new Array(n);

  for (let i = 0; i < n; i++) {
    x_pred[i] = np.reshape(x_pred_stacked.ref.slice(i), [2, 1]);
    C_pred[i] = np.reshape(C_pred_stacked.ref.slice(i), [2, 2]);
    x_smooth[i] = np.reshape(x_smooth_stacked.ref.slice(i), [2, 1]);
    C_smooth[i] = np.reshape(C_smooth_stacked.ref.slice(i), [2, 2]);
  }

  disposeAll(
    x_pred_stacked,
    C_pred_stacked,
    x_smooth_stacked,
    C_smooth_stacked,
  );

  // Compute output statistics
  const yhat = new FA(n);
  const ystd = new FA(n);
  const xstd: [number, number][] = new Array(n);
  const resid0 = new FA(n);
  const resid = new FA(n);
  const resid2 = new FA(n);
  let ssy = 0;
  let lik = 0;

  for (let i = 0; i < n; i++) {
    const yhat_i = (await x_pred[i].ref.data())[0];
    const C_s = await C_smooth[i].ref.data();

    yhat[i] = yhat_i;
    xstd[i] = [Math.sqrt(Math.abs(C_s[0])), Math.sqrt(Math.abs(C_s[3]))];
    ystd[i] = Math.sqrt(C_s[0] + V_std[i] ** 2);

    const r0_val = y[i] - yhat_i;
    resid0[i] = r0_val;
    resid[i] = r0_val / V_std[i];
    resid2[i] = v_array[i] / Math.sqrt(Cp_array[i]);

    ssy += r0_val * r0_val;
    lik += v_array[i] ** 2 / Cp_array[i] + Math.log(Cp_array[i]);
  }

  let s2 = 0,
    mse = 0,
    mape = 0;
  for (let i = 0; i < n; i++) {
    s2 += resid[i] ** 2;
    mse += resid2[i] ** 2;
    mape += Math.abs(resid2[i]) / Math.abs(y[i]);
  }
  s2 /= n;
  mse /= n;
  mape /= n;

  return {
    x: x_smooth,
    C: C_smooth,
    xf: x_pred,
    Cf: C_pred,
    yhat,
    ystd,
    xstd,
    resid0,
    resid,
    resid2,
    v: v_array,
    Cp: Cp_array,
    ssy,
    s2,
    nobs: n,
    lik,
    mse,
    mape,
  };
};

/**
 * Fit a local linear trend Dynamic Linear Model (DLM).
 *
 * Implements a two-pass estimation procedure:
 * 1. Initial pass with diffuse prior to estimate starting values
 * 2. Final pass with refined initial state from smoothed estimates
 *
 * The local linear trend model has state x = [level, slope]':
 *   y(t) = level(t) + v(t),           v ~ N(0, s²)
 *   level(t) = level(t-1) + slope(t-1) + w₁(t),  w₁ ~ N(0, w[0]²)
 *   slope(t) = slope(t-1) + w₂(t),    w₂ ~ N(0, w[1]²)
 *
 * System matrices:
 *   F = [1, 0]        (observation extracts level)
 *   G = [[1, 1],      (level evolves with slope)
 *        [0, 1]]      (slope is random walk)
 *   W = diag(w[0]², w[1]²)  (state noise covariance)
 *
 * @param y - Observations (n×1 array)
 * @param s - Observation noise standard deviation
 * @param w - State noise standard deviations [level, slope]
 * @param dtype - Computation precision (default: Float64)
 * @param mode - Execution mode: 'for' (faster) or 'scan' (enables JIT/AD)
 * @returns Complete model fit with smoothed estimates and diagnostics
 */
export const dlmFit = async (
  y: ArrayLike<number>,
  s: number,
  w: [number, number],
  dtype: DType = DType.Float64,
  mode: DlmMode = "for",
): Promise<DlmFitResult> => {
  const n = y.length;
  const FA = getFloatArrayType(dtype);

  // Convert input to TypedArray if needed
  const yArr = y instanceof FA ? y : FA.from(y);
  // Observation noise std dev (constant for all timesteps)
  const V_std = new FA(n).fill(s);

  // ─────────────────────────────────────────────────────────────────────────
  // Define system matrices for local linear trend model
  // ─────────────────────────────────────────────────────────────────────────
  // State transition: x(t) = G·x(t-1) + w
  const G = np.array(
    [
      [1.0, 1.0],
      [0.0, 1.0],
    ],
    { dtype },
  );
  // Observation: y(t) = F·x(t) + v
  const F = np.array([[1.0, 0.0]], { dtype });
  // State noise covariance
  const W = np.array(
    [
      [w[0] ** 2, 0.0],
      [0.0, w[1] ** 2],
    ],
    { dtype },
  );

  // ─────────────────────────────────────────────────────────────────────────
  // Initialize state with diffuse prior
  // Level initialized to mean of first observations; slope initialized to 0
  // ─────────────────────────────────────────────────────────────────────────
  let sum = 0;
  const count = Math.min(12, n);
  for (let i = 0; i < count; i++) sum += yArr[i];
  const mean_y = sum / count;
  // Initial covariance: large uncertainty (diffuse prior)
  const c0_val = (Math.abs(mean_y) * 0.5) ** 2;
  const c0 = c0_val === 0 ? 1e7 : c0_val;
  const x0_data = [[mean_y], [0.0]]; // [level, slope]
  const C0_data = [
    [c0, 0.0],
    [0.0, c0],
  ];

  // ─────────────────────────────────────────────────────────────────────────
  // Pass 1: Initial smoother to refine starting values
  // ─────────────────────────────────────────────────────────────────────────
  const out1 = await dlmSmo(
    yArr,
    F.ref,
    V_std,
    x0_data,
    G.ref,
    W.ref,
    C0_data,
    dtype,
    mode,
  );

  // Update initial state from smoothed estimate at t=1
  const x0_new = await out1.x[0].ref.data();
  const C0_new = await out1.C[0].ref.data();
  const x0_updated = [[x0_new[0]], [x0_new[1]]];
  // Scale initial covariance by 100 for second pass (following MATLAB dlmfit)
  const C0_scaled = [
    [C0_new[0] * 100, C0_new[1] * 100],
    [C0_new[2] * 100, C0_new[3] * 100],
  ];

  // Dispose Pass 1 arrays (no longer needed)
  for (let i = 0; i < n; i++) {
    disposeAll(out1.x[i], out1.C[i], out1.xf[i], out1.Cf[i]);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Pass 2: Final smoother with refined initial state
  // ─────────────────────────────────────────────────────────────────────────
  const out2 = await dlmSmo(
    yArr,
    F,
    V_std,
    x0_updated,
    G,
    W,
    C0_scaled,
    dtype,
    mode,
  );

  // ─────────────────────────────────────────────────────────────────────────
  // Convert np.Array outputs to TypedArray format for API
  // Layout: [state_dim][time] for vectors, [row][col][time] for matrices
  // ─────────────────────────────────────────────────────────────────────────
  const xf = [new FA(n), new FA(n)]; // Filtered states
  const Cf = [
    [new FA(n), new FA(n)],
    [new FA(n), new FA(n)],
  ]; // Filtered covariances
  const x = [new FA(n), new FA(n)]; // Smoothed states
  const C = [
    [new FA(n), new FA(n)],
    [new FA(n), new FA(n)],
  ]; // Smoothed covariances
  const xstd: FloatArray[] = new Array(n); // Smoothed state std devs

  for (let i = 0; i < n; i++) {
    // Extract filtered state: xf[state_dim][time]
    const xfi = await out2.xf[i].ref.data();
    xf[0][i] = xfi[0];
    xf[1][i] = xfi[1];
    out2.xf[i].dispose();

    // Extract filtered covariance: Cf[row][col][time]
    const Cfi = await out2.Cf[i].ref.data();
    Cf[0][0][i] = Cfi[0];
    Cf[0][1][i] = Cfi[1];
    Cf[1][0][i] = Cfi[2];
    Cf[1][1][i] = Cfi[3];
    out2.Cf[i].dispose();

    // Extract smoothed state: x[state_dim][time]
    const xi = await out2.x[i].ref.data();
    x[0][i] = xi[0];
    x[1][i] = xi[1];
    out2.x[i].dispose();

    // Extract smoothed covariance: C[row][col][time]
    const Ci = await out2.C[i].ref.data();
    C[0][0][i] = Ci[0];
    C[0][1][i] = Ci[1];
    C[1][0][i] = Ci[2];
    C[1][1][i] = Ci[3];
    out2.C[i].dispose();

    // State std devs: xstd[time][state_dim] (matches MATLAB format)
    xstd[i] = new FA([out2.xstd[i][0], out2.xstd[i][1]]);
  }

  return {
    // State estimates
    xf,
    Cf,
    x,
    C,
    xstd,
    // System matrices (plain arrays for easy serialization)
    G: [
      [1.0, 1.0],
      [0.0, 1.0],
    ],
    F: [1.0, 0.0],
    W: [
      [w[0] ** 2, 0.0],
      [0.0, w[1] ** 2],
    ],
    // Input data
    y: yArr,
    V: V_std,
    // Initial state (after Pass 1 refinement)
    x0: [x0_updated[0][0], x0_updated[1][0]],
    C0: C0_scaled,
    // Covariates (empty for basic model)
    XX: [],
    // Predictions and residuals
    yhat: out2.yhat,
    ystd: out2.ystd,
    resid0: out2.resid0,
    resid: out2.resid,
    resid2: out2.resid2,
    // Diagnostics
    ssy: out2.ssy, // Sum of squared residuals
    v: out2.v, // Innovations (filter prediction errors)
    Cp: out2.Cp, // Innovation covariances
    s2: out2.s2, // Residual variance
    nobs: out2.nobs, // Number of observations
    lik: out2.lik, // -2×log-likelihood
    mse: out2.mse, // Mean squared error
    mape: out2.mape, // Mean absolute percentage error
    class: "dlmfit",
  };
};
