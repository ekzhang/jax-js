import { grad, numpy as np } from "@jax-js/jax";
import { adam, applyUpdates, sgd } from "@jax-js/optax";

// Minimize a slightly ill-conditioned scalar objective.
const loss = (x: np.Array) =>
  np.square(x.ref.sub(3)).add(np.sin(x.mul(8)).mul(0.03));
const dloss = grad(loss);

let xAdam = np.array(-4);
let xSgd = np.array(-4);
const adamOpt = adam(0.25);
const sgdOpt = sgd(0.08);
let adamState = adamOpt.init(xAdam.ref);
let sgdState = sgdOpt.init(xSgd.ref);

// Optimizers are small state machines over parameter trees.
for (let step = 0; step < 40; step++) {
  const [adamUpdate, nextAdam] = adamOpt.update(
    dloss(xAdam.ref),
    adamState,
    xAdam.ref,
  );
  const [sgdUpdate, nextSgd] = sgdOpt.update(dloss(xSgd.ref), sgdState);
  adamState = nextAdam;
  sgdState = nextSgd;
  xAdam = applyUpdates(xAdam, adamUpdate);
  xSgd = applyUpdates(xSgd, sgdUpdate);
}

console.log("Adam x =", await xAdam.jsAsync());
console.log("SGD x =", await xSgd.jsAsync());
