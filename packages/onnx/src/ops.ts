/**
 * ONNX operation handlers.
 *
 * Maps ONNX operations to jax-js implementations.
 */

import { nn, numpy as np } from "@jax-js/jax";

import { onnxDtypeToJax, tensorToArray } from "./tensor.js";

/** Attributes parsed from an ONNX node */

export type Attrs = Record<string, any>;

/** An ONNX operation handler function */
export type OnnxOp = (inputs: np.Array[], attrs: Attrs) => np.Array[];

/**
 * Registry of ONNX operations to jax-js implementations.
 */
export const onnxOps: Record<string, OnnxOp> = {
  // ============================================================
  // Element-wise operations
  // ============================================================

  Add: ([a, b]) => [np.add(a, b)],
  Sub: ([a, b]) => [np.subtract(a, b)],
  Mul: ([a, b]) => [np.multiply(a, b)],
  Div: ([a, b]) => [np.divide(a, b)],
  Neg: ([x]) => [np.negative(x)],
  Abs: ([x]) => [np.abs(x)],
  Sqrt: ([x]) => [np.sqrt(x)],
  Exp: ([x]) => [np.exp(x)],
  Log: ([x]) => [np.log(x)],
  Pow: ([x, y]) => [np.pow(x, y)],
  Reciprocal: ([x]) => [np.reciprocal(x)],
  Floor: ([x]) => [np.trunc(x)], // jax-js uses trunc
  Ceil: ([x]) => [np.trunc(x.add(0.999999))], // Approximate ceil
  Sin: ([x]) => [np.sin(x)],
  Cos: ([x]) => [np.cos(x)],
  Tan: ([x]) => [np.tan(x)],
  Sinh: ([x]) => [np.sinh(x)],
  Cosh: ([x]) => [np.cosh(x)],
  Tanh: ([x]) => [np.tanh(x)],
  Asin: ([x]) => [np.asin(x)],
  Acos: ([x]) => [np.acos(x)],
  Atan: ([x]) => [np.atan(x)],
  Asinh: ([x]) => [np.asinh(x)],
  Acosh: ([x]) => [np.acosh(x)],
  Atanh: ([x]) => [np.atanh(x)],
  Sign: ([x]) => [np.sign(x)],

  // Erf is needed for GELU in transformers
  Erf: ([x]) => {
    // Approximate erf using tanh approximation (fast and accurate)
    // erf(x) ≈ tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
    const sqrtTwoOverPi = 0.7978845608028654;
    const a = 0.044715;
    const x3 = np.pow(x.ref, 3);
    const inner = x.add(x3.mul(a)).mul(sqrtTwoOverPi);
    return [np.tanh(inner)];
  },

  // ============================================================
  // Activation functions
  // ============================================================

  Relu: ([x]) => [nn.relu(x)],
  Sigmoid: ([x]) => [nn.sigmoid(x)],
  Softmax: ([x], { axis = -1 }) => [nn.softmax(x, axis)],
  LogSoftmax: ([x], { axis = -1 }) => [nn.logSoftmax(x, axis)],
  LeakyRelu: ([x], { alpha = 0.01 }) => [nn.leakyRelu(x, alpha)],
  Elu: ([x], { alpha = 1.0 }) => [nn.elu(x, alpha)],
  Celu: ([x], { alpha = 1.0 }) => [nn.celu(x, alpha)],
  Selu: ([x]) => {
    // SELU: scale * (max(0,x) + min(0, alpha * (exp(x) - 1)))
    const alpha = 1.6732632423543772;
    const scale = 1.0507009873554805;
    const positive = np.maximum(np.zeros(x.ref.shape), x.ref);
    const negative = np.minimum(
      np.zeros(x.ref.shape),
      np.exp(x).sub(1).mul(alpha),
    );
    return [positive.add(negative).mul(scale)];
  },
  Gelu: ([x]) => [nn.gelu(x)],
  HardSigmoid: ([x], { alpha = 0.2, beta = 0.5 }) => {
    // HardSigmoid: max(0, min(1, alpha * x + beta))
    return [np.clip(x.mul(alpha).add(beta), 0, 1)];
  },
  Softplus: ([x]) => [nn.softplus(x)],
  Softsign: ([x]) => [nn.softSign(x)],
  Mish: ([x]) => [nn.mish(x)],

  // ============================================================
  // Reduction operations
  // ============================================================

  ReduceSum: ([x, axes], { keepdims = 1, noop_with_empty_axes = 0 }) => {
    const axesArr = axes ? axes.js().flat().map(Number) : null;
    if (axesArr?.length === 0 && noop_with_empty_axes) return [x];
    return [np.sum(x, axesArr, { keepdims: !!keepdims })];
  },
  ReduceMean: ([x, axes], { keepdims = 1 }) => {
    const axesArr = axes ? axes.js().flat().map(Number) : null;
    return [np.mean(x, axesArr, { keepdims: !!keepdims })];
  },
  ReduceMax: ([x, axes], { keepdims = 1 }) => {
    const axesArr = axes ? axes.js().flat().map(Number) : null;
    return [np.max(x, axesArr, { keepdims: !!keepdims })];
  },
  ReduceMin: ([x, axes], { keepdims = 1 }) => {
    const axesArr = axes ? axes.js().flat().map(Number) : null;
    return [np.min(x, axesArr, { keepdims: !!keepdims })];
  },
  ReduceProd: ([x, axes], { keepdims = 1 }) => {
    const axesArr = axes ? axes.js().flat().map(Number) : null;
    return [np.prod(x, axesArr, { keepdims: !!keepdims })];
  },

  // ============================================================
  // Matrix operations
  // ============================================================

  MatMul: ([a, b]) => [np.matmul(a, b)],

  Gemm: ([a, b, c], { alpha = 1, beta = 1, transA = 0, transB = 0 }) => {
    if (transA) a = a.transpose();
    if (transB) b = b.transpose();
    let result = np.matmul(a, b);
    if (alpha !== 1) result = result.mul(alpha);
    if (c && beta !== 0) result = result.add(c.ref.mul(beta));
    return [result];
  },

  // ============================================================
  // Shape operations
  // ============================================================

  Reshape: ([x, shape]) => {
    const newShape = shape.js().flat().map(Number);
    return [x.reshape(newShape)];
  },

  Transpose: ([x], { perm }) => {
    if (perm) {
      return [x.transpose(perm)];
    }
    // Default: reverse all axes
    return [x.transpose()];
  },

  Squeeze: ([x, axes], attrs) => {
    // ONNX opset 13+: axes is an input tensor
    // ONNX opset <13: axes is an attribute
    let axesArr: number[] | undefined;
    if (axes) {
      axesArr = axes.js().flat().map(Number);
    } else if (attrs.axes) {
      axesArr = attrs.axes;
    }

    if (axesArr && axesArr.length > 0) {
      // Remove specified axes (must be size 1)
      const newShape = x.shape.filter((_, i) => !axesArr!.includes(i));
      return [x.reshape(newShape)];
    } else {
      // Remove all axes of size 1
      const newShape = x.shape.filter((d) => d !== 1);
      return [x.reshape(newShape)];
    }
  },

  Unsqueeze: ([x, axes], attrs) => {
    // ONNX opset 13+: axes is an input tensor
    // ONNX opset <13: axes is an attribute
    let axesArr: number[];
    if (axes) {
      axesArr = axes.js().flat().map(Number);
    } else if (attrs.axes) {
      axesArr = attrs.axes;
    } else {
      throw new Error("Unsqueeze requires axes");
    }

    // Normalize negative axes and sort
    const ndim = x.ndim + axesArr.length;
    axesArr = axesArr.map((a) => (a < 0 ? ndim + a : a)).sort((a, b) => a - b);

    const shape = [...x.shape];
    for (const axis of axesArr) {
      shape.splice(axis, 0, 1);
    }
    return [x.reshape(shape)];
  },

  Flatten: ([x], { axis = 1 }) => {
    const pre = x.shape.slice(0, axis).reduce((a, b) => a * b, 1);
    const post = x.shape.slice(axis).reduce((a, b) => a * b, 1);
    return [x.reshape([pre, post])];
  },

  Concat: (inputs, { axis }) => {
    return [np.concatenate(inputs, axis)];
  },

  Split: ([x, split], { axis = 0, num_outputs }) => {
    let splitSizes: number[];
    if (split) {
      splitSizes = split.js().flat().map(Number);
    } else if (num_outputs) {
      // Equal split
      const dimSize = x.shape[axis];
      const splitSize = Math.floor(dimSize / num_outputs);
      splitSizes = Array(num_outputs).fill(splitSize);
      // Handle remainder
      const remainder = dimSize % num_outputs;
      for (let i = 0; i < remainder; i++) {
        splitSizes[i]++;
      }
    } else {
      throw new Error("Split requires either split sizes or num_outputs");
    }

    const results: np.Array[] = [];
    let offset = 0;
    for (const size of splitSizes) {
      // Build slice indices for all dimensions
      // slice() takes variadic args, each being [] (full), [start, end], or number
      const sliceArgs: ([] | [number, number])[] = x.shape.map(
        (d: number, i: number): [] | [number, number] =>
          i === axis ? [offset, offset + size] : [],
      );
      results.push(x.ref.slice(...sliceArgs));
      offset += size;
    }
    x.dispose();
    return results;
  },

  // ============================================================
  // Comparison operations
  // ============================================================

  Equal: ([a, b]) => [np.equal(a, b)],
  Less: ([a, b]) => [np.less(a, b)],
  Greater: ([a, b]) => [np.greater(a, b)],
  LessOrEqual: ([a, b]) => [np.lessEqual(a, b)],
  GreaterOrEqual: ([a, b]) => [np.greaterEqual(a, b)],
  Not: ([x]) => [x.equal(0)], // Logical not as comparison to 0

  // ============================================================
  // Utility operations
  // ============================================================

  // eslint-disable-next-line no-empty-pattern
  Constant: ([], { value }) => {
    return [tensorToArray(value)];
  },

  Identity: ([x]) => [x],

  Cast: ([x], { to }) => {
    const dtype = onnxDtypeToJax(to);
    return [x.astype(dtype)];
  },

  Where: ([condition, x, y]) => {
    return [np.where(condition, x, y)];
  },

  Clip: ([x, min, max]) => {
    // min and max can be tensors or undefined
    const minVal = min ? min.js().flat()[0] : -Infinity;
    const maxVal = max ? max.js().flat()[0] : Infinity;
    return [np.clip(x, minVal, maxVal)];
  },

  Shape: ([x]) => {
    return [np.array(x.shape, { dtype: np.int32 })];
  },

  // Dropout in inference mode is just identity
  Dropout: ([x]) => {
    // Return [output, mask] - mask is optional
    return [x];
  },

  // ============================================================
  // Gather and indexing operations
  // ============================================================

  Gather: ([data, indices], { axis = 0 }) => {
    // Normalize axis
    const normalizedAxis = axis < 0 ? data.ndim + axis : axis;

    // Get indices as flat array
    const indicesArr = indices.js().flat().map(Number);

    // For now, implement simple case: gather along one axis
    // This is common for embedding lookups
    if (indices.ndim === 1 || indices.ndim === 0) {
      const gathered = indicesArr.map((idx: number) => {
        // Build slice args: [] for full dimension, [start, end] for slice
        const sliceArgs: ([] | [number, number])[] = data.ref.shape.map(
          (d: number, i: number): [] | [number, number] =>
            i === normalizedAxis ? [idx, idx + 1] : [],
        );
        return data.ref.slice(...sliceArgs);
      });
      data.dispose();

      if (gathered.length === 1) {
        // Squeeze the gathered axis if single index
        const result = gathered[0];
        if (indices.ndim === 0) {
          const newShape = result.shape.filter(
            (_: number, i: number) => i !== normalizedAxis,
          );
          return [result.reshape(newShape)];
        }
        return [result];
      }
      return [np.concatenate(gathered, normalizedAxis)];
    }

    throw new Error(
      `Gather with ${indices.ndim}D indices not yet fully supported`,
    );
  },

  Slice: ([data, starts, ends, axes, steps]) => {
    const startsArr = starts.js().flat().map(Number);
    const endsArr = ends.js().flat().map(Number);
    const axesArr = axes ? axes.js().flat().map(Number) : null;
    const stepsArr = steps ? steps.js().flat().map(Number) : null;

    // Build slice specification for all dimensions (default to full range)
    const sliceRanges: [number, number][] = data.shape.map((d: number) => [
      0,
      d,
    ]);

    const targetAxes = axesArr || startsArr.map((_: number, i: number) => i);
    for (let i = 0; i < targetAxes.length; i++) {
      const axis =
        targetAxes[i] < 0 ? data.ndim + targetAxes[i] : targetAxes[i];
      let start = startsArr[i];
      let end = endsArr[i];
      const step = stepsArr ? stepsArr[i] : 1;

      if (step !== 1) {
        throw new Error("Slice with step != 1 not yet supported");
      }

      // Handle negative indices
      const dimSize = data.shape[axis];
      if (start < 0) start = Math.max(0, dimSize + start);
      if (end < 0) end = dimSize + end;
      // Clamp to valid range
      start = Math.max(0, Math.min(start, dimSize));
      end = Math.max(0, Math.min(end, dimSize));

      sliceRanges[axis] = [start, end];
    }

    // Convert to slice args format: [] for full dim, [start, end] for range
    const sliceArgs: ([] | [number, number])[] = sliceRanges.map(
      ([start, end], i): [] | [number, number] =>
        start === 0 && end === data.shape[i] ? [] : [start, end],
    );
    return [data.slice(...sliceArgs)];
  },

  // ============================================================
  // Normalization operations
  // ============================================================

  LayerNormalization: (
    [x, scale, bias],
    { axis = -1, epsilon = 1e-5, stash_type },
  ) => {
    void stash_type;

    // Normalize axis
    const normalizedAxis = axis < 0 ? x.ndim + axis : axis;

    // Compute mean and variance over the normalized axes
    const axes = [];
    for (let i = normalizedAxis; i < x.ndim; i++) {
      axes.push(i);
    }

    const mean = np.mean(x.ref, axes, { keepdims: true });
    const variance = np.mean(np.square(x.ref.sub(mean.ref)), axes, {
      keepdims: true,
    });

    // Normalize
    const normalized = x.sub(mean).div(np.sqrt(variance.add(epsilon)));

    // Apply scale and bias
    let result = normalized.mul(scale);
    if (bias) {
      result = result.add(bias);
    }

    return [result];
  },
};
