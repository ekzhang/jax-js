import { DType, isFloatDtype } from "../../alu";
import type { Array } from "../../frontend/array";

/** @inline */
type Casting = "no" | "equiv" | "safe" | "same_kind" | "unsafe";

// Which dtypes each dtype can be safely cast to, following NumPy's rules: a
// cast is safe if every value of the source type is exactly representable in
// the target type. (float32 has a 24-bit mantissa, so int32 and uint32 can
// only be safely cast to float64.)
const safeCasts: Readonly<Record<DType, readonly DType[]>> = {
  [DType.Bool]: [
    DType.Bool,
    DType.Uint32,
    DType.Int32,
    DType.Float16,
    DType.Float32,
    DType.Float64,
  ],
  [DType.Uint32]: [DType.Uint32, DType.Float64],
  [DType.Int32]: [DType.Int32, DType.Float64],
  [DType.Float16]: [DType.Float16, DType.Float32, DType.Float64],
  [DType.Float32]: [DType.Float32, DType.Float64],
  [DType.Float64]: [DType.Float64],
};

// Follow NumPy's "same_kind" ordering: uint32 can cast to int32, but not vice versa.
const kindRank = (dtype: DType): number => {
  switch (dtype) {
    case DType.Bool:
      return 0;
    case DType.Uint32:
      return 1;
    case DType.Int32:
      return 2;
    case DType.Float16:
    case DType.Float32:
    case DType.Float64:
      return 3;
    default:
      dtype satisfies never;
      throw new Error(`canCast: unsupported dtype ${dtype}`);
  }
};

/**
 * Returns true if a cast between data types can occur according to the given
 * casting rule.
 *
 * The casting rules match NumPy:
 *
 * - `"no"` / `"equiv"`: the dtypes must be equal.
 * - `"safe"`: only casts that preserve values are allowed.
 * - `"same_kind"`: safe casts, plus casts within a kind (like float64 to
 *   float16 or uint32 to int32).
 * - `"unsafe"`: any cast is allowed.
 */
export function canCast(
  from: DType | Array,
  to: DType,
  casting: Casting = "safe",
): boolean {
  const src = typeof from === "string" ? from : from.dtype;
  switch (casting) {
    case "no":
    case "equiv":
      return src === to;
    case "safe":
      return safeCasts[src].includes(to);
    case "same_kind":
      return kindRank(to) >= kindRank(src);
    case "unsafe":
      return true;
    default:
      casting satisfies never; // completeness check
      throw new Error(`canCast: unsupported casting rule ${casting}`);
  }
}

/** @inline */
type FInfo = Readonly<{
  /** The number of bits occupied by the type. */
  bits: number;
  /** Returns the _dtype_ for which finfo returns information. */
  dtype: DType;
  /** The difference between 1.0 and the next smallest representable float larger than 1.0. */
  eps: number;
  /** The difference between 1.0 and the next largest representable float smaller than 1.0. */
  epsneg: number;
  /** The exponent that yields `eps`. */
  machep: number;
  /** The largest representable finite number. */
  max: number;
  /** The smallest positive power of the base (2) that causes overflow. */
  maxexp: number;
  /** The smallest representable (most negative) finite number. */
  min: number;
  /** The largest negative power of the base (2) without leading zeros in mantissa. */
  minexp: number;
  /** The exponent that yields `epsneg`. */
  negep: number;
  /** Number of bits in the exponent portion. */
  nexp: number;
  /** Number of bits in the mantissa portion. */
  nmant: number;
  /** The approximate number of decimal digits to which this kind of float is precise. */
  precision: number;
  /** The approximate decimal resolution, i.e., `10 ** -precision`. */
  resolution: number;
  /** The smallest positive normal number. */
  smallestNormal: number;
  /** The smallest positive subnormal number. */
  smallestSubnormal: number;
}>;

/** Machine limits for floating-point types. */
export function finfo(dtype: DType): FInfo {
  if (!isFloatDtype(dtype))
    throw new Error(`finfo: received ${dtype}, must be a floating-point type`);
  switch (dtype) {
    case DType.Float16:
      return Object.freeze({
        bits: 16,
        dtype: DType.Float16,
        eps: 2 ** -10,
        epsneg: 2 ** -11,
        machep: -10,
        max: 65504,
        maxexp: 16,
        min: -65504,
        minexp: -14,
        negep: -24,
        nexp: 5,
        nmant: 10,
        precision: 3,
        resolution: 1e-3,
        smallestNormal: 2 ** -14,
        smallestSubnormal: 2 ** -24,
      });
    case DType.Float32:
      return Object.freeze({
        bits: 32,
        dtype: DType.Float32,
        eps: 2 ** -23,
        epsneg: 2 ** -24,
        machep: -23,
        max: 3.4028234663852886e38, // Enough digits to be exact float64
        maxexp: 128,
        min: -3.4028234663852886e38,
        minexp: -126,
        negep: -24,
        nexp: 8,
        nmant: 23,
        precision: 6,
        resolution: 1e-6,
        smallestNormal: 2 ** -126,
        smallestSubnormal: 2 ** -149,
      });
    case DType.Float64:
      return Object.freeze({
        bits: 64,
        dtype: DType.Float64,
        eps: 2 ** -52,
        epsneg: 2 ** -53,
        machep: -52,
        max: Number.MAX_VALUE,
        maxexp: 1024,
        min: -Number.MAX_VALUE,
        minexp: -1022,
        negep: -53,
        nexp: 11,
        nmant: 52,
        precision: 15,
        resolution: 1e-15,
        smallestNormal: 2 ** -1022,
        smallestSubnormal: 2 ** -1074,
      });
    default:
      dtype satisfies never; // completeness check
      throw new Error(`finfo: unsupported dtype ${dtype}`);
  }
}

/** @inline */
type IInfo = Readonly<{
  /** The number of bits occupied by the type. */
  bits: number;
  /** Returns the _dtype_ for which iinfo returns information. */
  dtype: DType;
  /** The largest representable integer. */
  max: number;
  /** The smallest representable integer. */
  min: number;
}>;

/** Machine limits for integer types. */
export function iinfo(dtype: DType): IInfo {
  switch (dtype) {
    case DType.Int32:
      return Object.freeze({
        bits: 32,
        dtype: DType.Int32,
        max: 2147483647,
        min: -2147483648,
      });
    case DType.Uint32:
      return Object.freeze({
        bits: 32,
        dtype: DType.Uint32,
        max: 4294967295,
        min: 0,
      });
    default:
      throw new Error(`iinfo: unsupported dtype ${dtype}`);
  }
}
