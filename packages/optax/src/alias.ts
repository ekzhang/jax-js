import { GradientTransformation, identity, ScalarOrSchedule } from "./base";
import { chain } from "./combine";
import {
  addDecayedWeights,
  AddDecayedWeightsOptions,
  scaleByAdam,
  ScaleByAdamOptions,
  scaleByLearningRate,
  trace,
} from "./transform";

export type SgdOptions = {
  momentum?: number | null;
  nesterov?: boolean;
};

/** Stochastic gradient descent. */
export function sgd(
  learningRate: ScalarOrSchedule,
  opts: SgdOptions = {},
): GradientTransformation {
  const { momentum = null, nesterov = false } = opts;

  let opt: GradientTransformation;
  if (momentum !== null) {
    opt = trace({ decay: momentum, nesterov });
  } else {
    opt = identity();
  }

  return chain(opt, scaleByLearningRate(learningRate));
}

/** The Adam optimizer. */
export function adam(
  learningRate: ScalarOrSchedule,
  opts?: ScaleByAdamOptions,
): GradientTransformation {
  return chain(scaleByAdam(opts), scaleByLearningRate(learningRate));
}

export type AdamWOptions = ScaleByAdamOptions & AddDecayedWeightsOptions;

/** Adam with weight decay regularization. */
export function adamw(
  learningRate: ScalarOrSchedule,
  opts: AdamWOptions = {},
): GradientTransformation {
  const {
    b1,
    b2,
    eps,
    epsRoot,
    nesterov,
    weightDecay = 1e-4,
    mask,
    ...adamOpts
  } = opts;

  return chain(
    scaleByAdam({
      b1,
      b2,
      eps,
      epsRoot,
      nesterov,
      ...adamOpts,
    }),
    addDecayedWeights({ weightDecay, mask }),
    scaleByLearningRate(learningRate),
  );
}
