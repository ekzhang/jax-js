// Convolution operations.
//
// TODO: ConvTranspose (prompt_encoder_mask_decoder)

import { lax, numpy as np } from "@jax-js/jax";

import {
  type Operand,
  operandToJax,
  operandToJs,
  StaticArray,
} from "../tensor";

const padsMapping: Record<string, lax.PaddingType> = {
  SAME_UPPER: "SAME",
  SAME_LOWER: "SAME_LOWER",
  VALID: "VALID",
};

export function Conv(
  inputs: Operand[],
  {
    auto_pad: autoPad = "NOTSET",
    dilations,
    group = 1,
    kernel_shape: _kernelShape, // inferred from weights
    pads,
    strides,
  }: {
    auto_pad?: "NOTSET" | "SAME_LOWER" | "SAME_UPPER" | "VALID";
    dilations?: number[];
    group?: number;
    kernel_shape?: number[];
    pads?: number[];
    strides?: number[];
  },
): Operand[] {
  const [x, w, bias] = inputs.map(operandToJax);
  if (!x || !w) throw new Error("Conv: missing required inputs");
  const [_batchSize, channelsIn, ...xSpatial] = x.shape;
  const [_channelsOut, channelsInGrouped, ...wSpatial] = w.shape;
  if (channelsIn !== channelsInGrouped * group) {
    throw new Error(
      `Conv: input channels ${channelsIn} must match weight channels ${channelsInGrouped} x group ${group}`,
    );
  }
  if (xSpatial.length !== wSpatial.length) {
    throw new Error(
      `Conv: input spatial dims ${xSpatial.length} must match weight spatial dims ${wSpatial.length}`,
    );
  }
  const n = xSpatial.length;
  let output = lax.convGeneralDilated(
    x,
    w,
    strides ?? wSpatial.map(() => 1),
    padsMapping[autoPad] ??
      pads?.slice(0, n).map((p, i) => [p, pads[i + n]]) ??
      "VALID",
    {
      rhsDilation: dilations,
      featureGroupCount: group,
    },
  );
  // Add bias if provided (reshape to [1, C, 1, 1, ...] for broadcasting)
  if (bias) {
    const biasShape = [bias.size, ...xSpatial.map(() => 1)];
    output = output.add(bias.reshape(biasShape));
  }
  return [output];
}

// Pad a tensor with -Infinity along spatial dimensions (for max pooling).
function padWithNegInf(x: np.Array, pads: [number, number][]): np.Array {
  // pads is for spatial dims only, we need to add batch and channel dims
  for (let i = 0; i < pads.length; i++) {
    const [padBefore, padAfter] = pads[i];
    const axis = i + 2; // Skip batch and channel dims
    if (padBefore > 0) {
      const beforeShape = [...x.shape];
      beforeShape[axis] = padBefore;
      const before = np.full(beforeShape, -Infinity, { dtype: x.dtype });
      x = np.concatenate([before, x], axis);
    }
    if (padAfter > 0) {
      const afterShape = [...x.shape];
      afterShape[axis] = padAfter;
      const after = np.full(afterShape, -Infinity, { dtype: x.dtype });
      x = np.concatenate([x, after], axis);
    }
  }
  return x;
}

export function MaxPool(
  [xOp]: Operand[],
  {
    auto_pad: autoPad = "NOTSET",
    ceil_mode: ceilMode = 0,
    dilations,
    kernel_shape: kernelShape,
    pads,
    strides,
  }: {
    auto_pad?: "NOTSET" | "SAME_LOWER" | "SAME_UPPER" | "VALID";
    ceil_mode?: number;
    dilations?: number[];
    kernel_shape: number[];
    pads?: number[];
    strides?: number[];
  },
): Operand[] {
  if (dilations && dilations.some((d) => d !== 1)) {
    throw new Error("MaxPool: dilations != 1 is not supported");
  }
  const x = operandToJax(xOp);
  const n = kernelShape.length;
  const xSpatial = x.shape.slice(2);
  if (xSpatial.length !== n) {
    throw new Error(
      `MaxPool: input spatial dims ${xSpatial.length} must match kernel dims ${n}`,
    );
  }

  // Compute explicit padding
  let explicitPads: [number, number][];
  if (autoPad !== "NOTSET") {
    const effectiveStrides = strides ?? kernelShape.map(() => 1);
    const outShape = xSpatial.map((size, i) =>
      Math.ceil(size / effectiveStrides[i]),
    );
    const padSizes = outShape.map((o, i) => {
      const s = effectiveStrides[i];
      const k = kernelShape[i];
      const inSize = xSpatial[i];
      return Math.max(0, (o - 1) * s + k - inSize);
    });
    explicitPads =
      autoPad === "SAME_UPPER"
        ? padSizes.map((size) => [size >> 1, size - (size >> 1)])
        : padSizes.map((size) => [size - (size >> 1), size >> 1]);
  } else if (pads) {
    explicitPads = pads
      .slice(0, n)
      .map((p, i) => [p, pads[i + n]] as [number, number]);
  } else {
    explicitPads = kernelShape.map(() => [0, 0] as [number, number]);
  }

  if (ceilMode) {
    const effectiveStrides = strides ?? kernelShape.map(() => 1);
    for (let i = 0; i < n; i++) {
      const padTotal = explicitPads[i][0] + explicitPads[i][1];
      const numerator = xSpatial[i] + padTotal - kernelShape[i];
      const remainder = numerator % effectiveStrides[i];
      if (numerator > 0 && remainder !== 0) {
        explicitPads[i] = [
          explicitPads[i][0],
          explicitPads[i][1] + effectiveStrides[i] - remainder,
        ];
      }
    }
  }

  // Apply padding with -Infinity if needed
  const needsPadding = explicitPads.some(([a, b]) => a > 0 || b > 0);
  const padded = needsPadding ? padWithNegInf(x, explicitPads) : x;

  const output = lax.reduceWindow(
    padded,
    np.max,
    kernelShape,
    strides ?? kernelShape.map(() => 1),
  );
  return [output];
}

export function Pad(
  inputs: Operand[],
  { mode = "constant" }: { mode?: string },
): Operand[] {
  const [dataOp, padsOp, constantValueOp, axesOp] = inputs;

  if (mode !== "constant") {
    throw new Error(`Pad: mode '${mode}' is not supported`);
  }

  if (constantValueOp) {
    const fillRaw = operandToJs(constantValueOp);
    const fill = Array.isArray(fillRaw) ? fillRaw[0] : fillRaw;
    if (fill !== 0) {
      throw new Error("Pad: only constant_value=0 is supported");
    }
  }

  const data = operandToJax(dataOp);
  const pads: number[] = operandToJs(padsOp);
  const ndim = data.ndim;
  const axes: number[] = axesOp
    ? (operandToJs(axesOp) as number[]).map((a) => (a < 0 ? ndim + a : a))
    : Array.from({ length: ndim }, (_, i) => i);

  if (pads.length !== 2 * axes.length) {
    throw new Error(
      `Pad: pads length ${pads.length} does not match axes length ${axes.length}`,
    );
  }

  const widths: [number, number][] = Array.from(
    { length: ndim },
    () => [0, 0] as [number, number],
  );
  for (let i = 0; i < axes.length; i++) {
    widths[axes[i]] = [pads[i], pads[i + axes.length]];
  }

  return [np.pad(data, widths)];
}

function inputCoordFor(
  outIdx: number,
  inSize: number,
  outSize: number,
  coordMode: string,
): number {
  switch (coordMode) {
    case "asymmetric":
      return (outIdx * inSize) / outSize;
    case "align_corners":
      return outSize > 1 ? (outIdx * (inSize - 1)) / (outSize - 1) : 0;
    case "half_pixel":
      return ((outIdx + 0.5) * inSize) / outSize - 0.5;
    case "pytorch_half_pixel":
      return outSize > 1 ? ((outIdx + 0.5) * inSize) / outSize - 0.5 : 0;
    default:
      throw new Error(
        `Resize: coordinate_transformation_mode '${coordMode}' is not supported`,
      );
  }
}

export function Resize(
  [xOp, roi, scales, sizes]: Operand[],
  {
    coordinate_transformation_mode: coordMode = "half_pixel",
    mode = "nearest",
    nearest_mode: nearestMode = "round_prefer_floor",
  }: {
    coordinate_transformation_mode?: string;
    mode?: string;
    nearest_mode?: string;
    // Ignored: cubic_coeff_a, exclude_outside, extrapolation_value,
    // keep_aspect_ratio_policy, axes
  },
): Operand[] {
  if (mode !== "nearest" && mode !== "linear") {
    throw new Error(
      `Resize: mode '${mode}' is not supported, only 'nearest' and 'linear'`,
    );
  }
  if (mode === "nearest") {
    if (coordMode !== "asymmetric") {
      throw new Error(
        `Resize: coordinate_transformation_mode '${coordMode}' is not supported for nearest`,
      );
    }
    if (nearestMode !== "floor") {
      throw new Error(
        `Resize: nearest_mode '${nearestMode}' is not supported, only 'floor'`,
      );
    }
  }

  if (roi && !(roi instanceof StaticArray)) {
    // We don't use roi, so just dispose it.
    roi.dispose();
  }

  const x = operandToJax(xOp);
  const inShape = x.shape;
  let outShape: number[];
  if (sizes && sizes.shape[0] > 0) {
    outShape = operandToJs(sizes);
  } else if (scales && scales.shape[0] > 0) {
    const scalesArr: number[] = operandToJs(scales);
    outShape = inShape.map((d, i) => Math.floor(d * scalesArr[i]));
  } else {
    throw new Error("Resize: either scales or sizes must be provided");
  }

  let result = x;
  for (let axis = 0; axis < inShape.length; axis++) {
    const inSize = result.shape[axis];
    const outSize = outShape[axis];
    if (inSize === outSize) continue;

    if (mode === "nearest") {
      const indices = np.array(
        Array.from({ length: outSize }, (_, i) =>
          Math.floor((i * inSize) / outSize),
        ),
        { dtype: np.int32 },
      );
      const sliceArgs: (np.Array | [])[] = result.shape.map(() => [] as []);
      sliceArgs[axis] = indices;
      result = result.slice(...sliceArgs);
      continue;
    }

    const i0Arr = new Int32Array(outSize);
    const i1Arr = new Int32Array(outSize);
    const wArr = new Float32Array(outSize);
    for (let i = 0; i < outSize; i++) {
      const pos = inputCoordFor(i, inSize, outSize, coordMode);
      const i0Raw = Math.floor(pos);
      const i0 = Math.max(0, Math.min(inSize - 1, i0Raw));
      const i1 = Math.max(0, Math.min(inSize - 1, i0Raw + 1));
      i0Arr[i] = i0;
      i1Arr[i] = i1;
      wArr[i] = i0Raw >= 0 && i0Raw < inSize - 1 ? pos - i0Raw : 0;
    }

    const baseShape = result.shape;
    const i0Indices = np.array(i0Arr, { dtype: np.int32 });
    const i1Indices = np.array(i1Arr, { dtype: np.int32 });

    const sliceArgs0: (np.Array | [])[] = baseShape.map(() => [] as []);
    sliceArgs0[axis] = i0Indices;
    const slice0 = result.ref.slice(...sliceArgs0);

    const sliceArgs1: (np.Array | [])[] = baseShape.map(() => [] as []);
    sliceArgs1[axis] = i1Indices;
    const slice1 = result.slice(...sliceArgs1);

    const wShape = baseShape.map(() => 1);
    wShape[axis] = outSize;
    const w = np.array(wArr).reshape(wShape);

    const diff = np.subtract(slice1, slice0.ref);
    result = np.add(slice0, np.multiply(diff, w));
  }

  return [result];
}

export function GridSample(
  [xOp, gridOp]: Operand[],
  {
    mode = "bilinear",
    padding_mode: paddingMode = "zeros",
    align_corners: alignCorners = 0,
  }: { mode?: string; padding_mode?: string; align_corners?: number },
): Operand[] {
  if (mode !== "bilinear") {
    throw new Error(`GridSample: mode '${mode}' is not supported`);
  }
  if (paddingMode !== "zeros") {
    throw new Error(
      `GridSample: padding_mode '${paddingMode}' is not supported`,
    );
  }

  const x = operandToJax(xOp);
  const grid = operandToJax(gridOp);
  if (x.ndim !== 4) {
    throw new Error(`GridSample: only 4D input is supported, got ${x.ndim}D`);
  }
  if (grid.ndim !== 4 || grid.shape[3] !== 2) {
    throw new Error(
      `GridSample: grid must be [N, H_out, W_out, 2], got [${grid.shape}]`,
    );
  }

  const [N, C, Hin, Win] = x.shape;
  const [Ng, Hout, Wout] = grid.shape;
  if (Ng !== N) {
    throw new Error(`GridSample: grid batch ${Ng} != input batch ${N}`);
  }

  const gx = grid.ref.slice([], [], [], 0);
  const gy = grid.slice([], [], [], 1);

  const ax = alignCorners ? (Win - 1) / 2 : Win / 2;
  const ay = alignCorners ? (Hin - 1) / 2 : Hin / 2;
  const bx = (Win - 1) / 2;
  const by = (Hin - 1) / 2;

  const px = np.add(np.multiply(gx, ax), bx);
  const py = np.add(np.multiply(gy, ay), by);
  const x0f = np.floor(px.ref);
  const y0f = np.floor(py.ref);

  const dx = np.subtract(px, x0f.ref);
  const dy = np.subtract(py, y0f.ref);
  const odx = np.subtract(1, dx.ref);
  const ody = np.subtract(1, dy.ref);

  const w00 = np.multiply(ody.ref, odx.ref);
  const w01 = np.multiply(ody, dx.ref);
  const w10 = np.multiply(dy.ref, odx);
  const w11 = np.multiply(dy, dx);

  const x0Inb = np.logicalAnd(
    np.greaterEqual(x0f.ref, 0),
    np.less(x0f.ref, Win),
  );
  const x1Inb = np.logicalAnd(
    np.greaterEqual(x0f.ref, -1),
    np.less(x0f.ref, Win - 1),
  );
  const y0Inb = np.logicalAnd(
    np.greaterEqual(y0f.ref, 0),
    np.less(y0f.ref, Hin),
  );
  const y1Inb = np.logicalAnd(
    np.greaterEqual(y0f.ref, -1),
    np.less(y0f.ref, Hin - 1),
  );

  const mw00 = np.where(np.logicalAnd(x0Inb.ref, y0Inb.ref), w00, 0);
  const mw01 = np.where(np.logicalAnd(x1Inb.ref, y0Inb), w01, 0);
  const mw10 = np.where(np.logicalAnd(x0Inb, y1Inb.ref), w10, 0);
  const mw11 = np.where(np.logicalAnd(x1Inb, y1Inb), w11, 0);

  const x0c = np.astype(np.clip(x0f, 0, Win - 1), np.int32);
  const y0c = np.astype(np.clip(y0f, 0, Hin - 1), np.int32);
  const x1c = np.clip(np.add(x0c.ref, 1), 0, Win - 1);
  const y1c = np.clip(np.add(y0c.ref, 1), 0, Hin - 1);

  const expandC = (t: np.Array) => t.reshape([N, 1, Hout, Wout]);
  const yxFlat = (yi: np.Array, xi: np.Array) =>
    np.add(np.multiply(expandC(yi), Win), expandC(xi));

  const nIdx = np.arange(N).reshape([N, 1, 1, 1]);
  const cIdx = np.arange(C).reshape([1, C, 1, 1]);
  const ncFlat = np.add(
    np.multiply(nIdx, C * Hin * Win),
    np.multiply(cIdx, Hin * Win),
  );

  const idx00 = np.add(ncFlat.ref, yxFlat(y0c.ref, x0c.ref));
  const idx01 = np.add(ncFlat.ref, yxFlat(y0c, x1c.ref));
  const idx10 = np.add(ncFlat.ref, yxFlat(y1c.ref, x0c));
  const idx11 = np.add(ncFlat, yxFlat(y1c, x1c));

  const xFlat = x.reshape([N * C * Hin * Win]);
  const v00 = np.take(xFlat.ref, idx00);
  const v01 = np.take(xFlat.ref, idx01);
  const v10 = np.take(xFlat.ref, idx10);
  const v11 = np.take(xFlat, idx11);

  const t00 = np.multiply(v00, expandC(mw00));
  const t01 = np.multiply(v01, expandC(mw01));
  const t10 = np.multiply(v10, expandC(mw10));
  const t11 = np.multiply(v11, expandC(mw11));

  return [np.add(np.add(t00, t01), np.add(t10, t11))];
}
