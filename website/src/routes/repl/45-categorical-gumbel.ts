import { numpy as np, random } from "@jax-js/jax";

// Categorical sampling uses the Gumbel-max trick.
const probs = np.array([0.1, 0.2, 0.7]);
const draws = random.categorical(random.key(0), np.log(probs), {
  shape: [10_000],
});

const counts = [0, 0, 0];
for (const draw of await draws.jsAsync()) counts[draw]++;

console.log("counts =", counts);
console.log(
  "frequencies =",
  counts.map((x) => x / 10_000),
);
