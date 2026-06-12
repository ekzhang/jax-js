// Element-wise operations, mostly simple to wrap directly.

import {
  DType,
  jit,
  nn,
  numpy as np,
  scipySpecial as special,
} from "@jax-js/jax";

import {
  onnxDtypeToJax,
  type Operand,
  operandToJax,
  StaticArray,
} from "../tensor";

function isIntegerDtype(dtype: DType): boolean {
  return dtype === np.int32 || dtype === np.uint32 || dtype === np.bool;
}

function wrapFn(
  fn: (...args: np.Array[]) => np.Array,
  staticFn?: (...args: number[]) => number,
  outDtype?: DType,
): (inputs: Operand[]) => Operand[] {
  return (inputs: Operand[]) => {
    // Static elementwise math stores outputs in Int32Array. Float constants
    // should take the normal path so shape-control values are not rounded.
    if (
      staticFn &&
      inputs.every(
        (op) =>
          op instanceof StaticArray &&
          (op.dtype === np.int32 || op.dtype === np.bool),
      )
    ) {
      const arrays = inputs as StaticArray[];
      // Compute broadcast shape across all inputs
      let outShape = arrays[0].shape;
      for (let i = 1; i < arrays.length; i++) {
        outShape = np.broadcastShapes(outShape, arrays[i].shape);
      }
      const broadcasted = arrays.map((a) => a.broadcastTo(outShape));
      const result = new Int32Array(broadcasted[0].data.length);
      for (let i = 0; i < result.length; i++) {
        result[i] = staticFn(...broadcasted.map((b) => b.data[i]));
      }
      return [
        new StaticArray(
          result,
          outShape,
          outDtype ?? arrays[arrays.length - 1].dtype,
        ),
      ];
    }
    return [fn(...inputs.map(operandToJax))];
  };
}

export const Add = wrapFn(np.add, (a, b) => a + b);
export const Sub = wrapFn(np.subtract, (a, b) => a - b);
export const Mul = wrapFn(np.multiply, (a, b) => a * b);
export const Neg = wrapFn(np.negative, (a) => -a);
export const Abs = wrapFn(np.abs, Math.abs);
export const Sqrt = wrapFn(np.sqrt);
export const Exp = wrapFn(np.exp);
export const Log = wrapFn(np.log);
export const Pow = wrapFn(np.pow, Math.pow);
export const Reciprocal = wrapFn(np.reciprocal);
export const Floor = wrapFn(np.floor);
export const Ceil = wrapFn(np.ceil);
export const Identity = wrapFn((x) => x);

export const Equal = wrapFn(np.equal, (a, b) => Number(a === b), np.bool);
export const Less = wrapFn(np.less, (a, b) => Number(a < b), np.bool);
export const Greater = wrapFn(np.greater, (a, b) => Number(a > b), np.bool);
export const LessOrEqual = wrapFn(
  np.lessEqual,
  (a, b) => Number(a <= b),
  np.bool,
);
export const GreaterOrEqual = wrapFn(
  np.greaterEqual,
  (a, b) => Number(a >= b),
  np.bool,
);

export const Where = wrapFn(np.where, (cond, x, y) => (cond ? x : y));
export const Clip = wrapFn(np.clip);

export const IsNaN = wrapFn(np.isnan);

export function Not([x]: Operand[]): Operand[] {
  return [np.notEqual(operandToJax(x), true)];
}

export const Sin = wrapFn(np.sin);
export const Cos = wrapFn(np.cos);
export const Tan = wrapFn(np.tan);

export const Sinh = wrapFn(np.sinh);
export const Cosh = wrapFn(np.cosh);
export const Tanh = wrapFn(np.tanh);

export const Asin = wrapFn(np.asin);
export const Acos = wrapFn(np.acos);
export const Atan = wrapFn(np.atan);

export const Asinh = wrapFn(np.asinh);
export const Acosh = wrapFn(np.acosh);
export const Atanh = wrapFn(np.atanh);

export const Sign = wrapFn(np.sign);
export const Erf = wrapFn(special.erf);

export const Relu = wrapFn(nn.relu);
export const Sigmoid = wrapFn(nn.sigmoid);
export const Elu = wrapFn(nn.elu);
export const Celu = wrapFn(nn.celu);
export const Softplus = wrapFn(nn.softplus);
export const Softsign = wrapFn(nn.softSign);
export const Mish = wrapFn(nn.mish);

export function Gelu(
  inputs: Operand[],
  { approximate = "none" }: { approximate?: "none" | "tanh" },
): Operand[] {
  const [x] = inputs.map(operandToJax);
  return [nn.gelu(x, { approximate: approximate === "tanh" })];
}

export function Swish(
  inputs: Operand[],
  { alpha = 1.0 }: { alpha?: number },
): Operand[] {
  const [x] = inputs.map(operandToJax);
  if (alpha === 1.0) {
    return [nn.silu(x)];
  }
  return [x.ref.mul(nn.sigmoid(x.mul(alpha)))];
}

export function LeakyRelu(
  inputs: Operand[],
  { alpha = 0.01 }: { alpha?: number },
): Operand[] {
  const [x] = inputs.map(operandToJax);
  return [nn.leakyRelu(x, alpha)];
}

const integerDiv = wrapFn(
  (a, b) => a.div(b),
  (a, b) => Math.trunc(a / b),
);

export function Div(inputs: Operand[]): Operand[] {
  if (inputs.every((op) => isIntegerDtype(op.dtype))) {
    return integerDiv(inputs);
  }
  const [a, b] = inputs.map(operandToJax);
  return [np.divide(a, b)];
}

function layerNormalizationCore(
  x: np.Array,
  scale: np.Array,
  axis: number,
  epsilon: number,
): np.Array {
  const ndim = x.ndim;
  const normAxis = axis < 0 ? ndim + axis : axis;
  const reduceAxes: number[] = [];
  for (let i = normAxis; i < ndim; i++) reduceAxes.push(i);

  const mean = np.mean(x.ref, reduceAxes, { keepdims: true });
  const diff = np.subtract(x, mean);
  const variance = np.mean(np.square(diff.ref), reduceAxes, {
    keepdims: true,
  });
  const invStd = np.reciprocal(np.sqrt(np.add(variance, epsilon)));
  const normalized = np.multiply(diff, invStd);

  return np.multiply(normalized, scale);
}

const layerNormalizationNoBias = jit(layerNormalizationCore, {
  staticArgnums: [2, 3],
});

const layerNormalizationWithBias = jit(
  function layerNormalizationWithBias(
    x: np.Array,
    scale: np.Array,
    bias: np.Array,
    axis: number,
    epsilon: number,
  ): np.Array {
    return np.add(layerNormalizationCore(x, scale, axis, epsilon), bias);
  },
  { staticArgnums: [3, 4] },
);

export function LayerNormalization(
  [xOp, scaleOp, biasOp]: Operand[],
  { axis = -1, epsilon = 1e-5 }: { axis?: number; epsilon?: number },
): Operand[] {
  const x = operandToJax(xOp);
  const scale = operandToJax(scaleOp);

  if (!biasOp) {
    return [layerNormalizationNoBias(x, scale, axis, epsilon)];
  }

  const bias = operandToJax(biasOp);
  return [layerNormalizationWithBias(x, scale, bias, axis, epsilon)];
}

export function Cast([xOp]: Operand[], { to }: { to: number }): Operand[] {
  const dtype = onnxDtypeToJax(to);
  if (dtype === xOp.dtype) return [xOp];
  const x = operandToJax(xOp);
  return [x.astype(dtype)];
}

export function Mod(
  inputs: Operand[],
  { fmod = 0 }: { fmod: number },
): Operand[] {
  const [a, b] = inputs.map(operandToJax);
  if (fmod) return [np.fmod(a, b)]; // Use sign of a.
  return [np.remainder(a, b)]; // Semantics of integer mod in ONNX use the sign of b.
}
