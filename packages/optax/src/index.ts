export { adam, sgd } from "./alias";
export {
  applyUpdates,
  type GradientTransformation,
  identity,
  type OptState,
  setToZero,
} from "./base";
export { l2Loss, squaredError } from "./losses";
export { scaleByAdam, type ScaleByAdamOptions } from "./transform";
