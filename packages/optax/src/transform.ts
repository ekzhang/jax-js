import { JsTree, numpy as np, tree } from "@jax-js/jax";

import {
  GradientTransformation,
  identity,
  initEmptyState,
  ScalarOrSchedule,
  Schedule,
} from "./base";
import {
  treeBiasCorrection,
  treeUpdateMoment,
  treeZerosLike,
} from "./treeUtils";

function u32(x: number): np.Array {
  return np.array(x, { dtype: np.uint32 });
}

export type ScaleByAdamOptions = {
  b1?: number;
  b2?: number;
  eps?: number;
  epsRoot?: number;
  nesterov?: boolean;
};

export function scaleByAdam({
  b1 = 0.9,
  b2 = 0.999,
  eps = 1e-8,
  epsRoot = 0.0,
  nesterov = false,
}: ScaleByAdamOptions = {}): GradientTransformation {
  return {
    init(params) {
      const mu = treeZerosLike(tree.ref(params)); // first moment
      const nu = treeZerosLike(params); // second moment
      return { count: u32(0), mu, nu };
    },
    update(updates, state, params) {
      tree.dispose(params);
      let { count, mu, nu } = state as {
        count: np.Array;
        mu: JsTree<np.Array>;
        nu: JsTree<np.Array>;
      };
      mu = treeUpdateMoment(tree.ref(updates), mu, b1, 1);
      nu = treeUpdateMoment(tree.ref(updates), nu, b2, 2);
      count = count.add(1);
      let muHat: typeof mu;
      if (nesterov) {
        muHat = tree.map(
          (m: np.Array, g: np.Array) => m.mul(b1).add(g.mul(1 - b1)),
          treeBiasCorrection(tree.ref(mu), b1, count.ref.add(1)),
          treeBiasCorrection(tree.ref(updates), b1, count.ref),
        );
      } else {
        muHat = treeBiasCorrection(tree.ref(mu), b1, count.ref);
      }
      const nuHat = treeBiasCorrection(tree.ref(nu), b2, count.ref);
      tree.dispose(updates);
      updates = tree.map(
        (m: np.Array, v: np.Array) => m.div(np.sqrt(v.add(epsRoot)).add(eps)),
        muHat,
        nuHat,
      ) as typeof updates;
      return [updates, { count, mu, nu }];
    },
  };
}

/** Scale by a constant step size. */
export function scale(stepSize: number): GradientTransformation {
  return {
    init: initEmptyState,
    update(updates, state, params) {
      tree.dispose(params);
      updates = tree.map((g: np.Array) => g.mul(stepSize), updates);
      return [updates, state];
    },
  };
}

/** Scale updates using a custom schedule for the step size. */
export function scaleBySchedule(stepSizeFn: Schedule): GradientTransformation {
  return {
    init(params) {
      tree.dispose(params);
      return { count: u32(0) }; // initial step
    },
    update(updates, state, params) {
      tree.dispose(params);
      const { count } = state as { count: np.Array };
      const countInt = count.item();
      const stepSize = stepSizeFn(countInt);
      updates = tree.map((g: np.Array) => g.mul(stepSize), updates);
      return [updates, { count: u32(countInt + 1) }];
    },
  };
}

/** Scale by the (negative) learning rate (either as scalar or as schedule). */
export function scaleByLearningRate(
  learningRate: ScalarOrSchedule,
  flipSign = true,
): GradientTransformation {
  if (learningRate === undefined) return identity();
  const m = flipSign ? -1 : 1;
  if (typeof learningRate === "function") {
    return scaleBySchedule((count) => m * learningRate(count));
  }
  return scale(m * learningRate);
}
