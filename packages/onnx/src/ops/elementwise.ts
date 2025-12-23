// Element-wise operations, mostly simple to wrap directly.

import { nn, numpy as np, scipySpecial as special } from "@jax-js/jax";

import { onnxDtypeToJax } from "../tensor";

function wrapFn(
  fn: (...args: np.Array[]) => np.Array,
): (inputs: np.Array[]) => np.Array[] {
  return (inputs: np.Array[]) => [fn(...inputs)];
}

export const Add = wrapFn(np.add);
export const Sub = wrapFn(np.subtract);
export const Mul = wrapFn(np.multiply);
export const Div = wrapFn(np.divide);
export const Neg = wrapFn(np.negative);
export const Abs = wrapFn(np.abs);
export const Sqrt = wrapFn(np.sqrt);
export const Exp = wrapFn(np.exp);
export const Log = wrapFn(np.log);
export const Reciprocal = wrapFn(np.reciprocal);
export const Floor = wrapFn(np.floor);
export const Ceil = wrapFn(np.ceil);
export const Identity = wrapFn((x) => x);

export function Pow([x, y]: np.Array[]): np.Array[] {
  // ONNX Pow needs to handle negative bases for integer exponents.
  //
  // np.pow uses exp(log(x)*y) which fails for negative x.
  // For even integer exponents, we can use abs(x) since (-x)^2n = x^2n.
  // For x^2 specifically, just use square.
  if (y.ndim === 0) {
    // TODO: This is a hack, please fix properly in `@jax-js/jax`.
    const exp = y.js();
    if (exp === 2) {
      return [np.square(x)];
    }
    if (Number.isInteger(exp) && exp > 0 && exp % 2 === 0) {
      // Even integer exponent: |x|^exp
      return [np.pow(np.abs(x), y)];
    }
  }
  return [np.pow(x, y)];
}

export const Equal = wrapFn(np.equal);
export const Less = wrapFn(np.less);
export const Greater = wrapFn(np.greater);
export const LessOrEqual = wrapFn(np.lessEqual);
export const GreaterOrEqual = wrapFn(np.greaterEqual);

export const Where = wrapFn(np.where);
export const Clip = wrapFn(np.clip);

export const IsNaN = wrapFn(np.isnan);

export function Not([x]: np.Array[]): np.Array[] {
  return [np.notEqual(x, true)];
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
  [x]: np.Array[],
  { approximate = "none" }: { approximate?: "none" | "tanh" },
) {
  return [nn.gelu(x, { approximate: approximate === "tanh" })];
}

export function Swish([x]: np.Array[], { alpha = 1.0 }: { alpha?: number }) {
  if (alpha === 1.0) {
    return [nn.silu(x)];
  }
  return [x.ref.mul(nn.sigmoid(x.mul(alpha)))];
}

export function LeakyRelu(
  [x]: np.Array[],
  { alpha = 0.01 }: { alpha?: number },
): np.Array[] {
  return [nn.leakyRelu(x, alpha)];
}

export function Cast([x]: np.Array[], { to }: { to: number }): np.Array[] {
  const dtype = onnxDtypeToJax(to);
  return [x.astype(dtype)];
}

export function Mod(
  [a, b]: np.Array[],
  { fmod = 0 }: { fmod: number },
): np.Array[] {
  if (fmod) return [np.fmod(a, b)]; // Use sign of a.
  return [np.remainder(a, b)]; // Semantics of integer mod in ONNX use the sign of b.
}
