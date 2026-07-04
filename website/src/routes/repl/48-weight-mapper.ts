import { WeightMapper } from "@jax-js/loaders";

// WeightMapper rewrites checkpoint key names into app-friendly names.
const mapper = new WeightMapper({
  prefix: { "model.encoder.": "encoder." },
  suffix: { ".gamma": ".weight", ".beta": ".bias" },
  substring: { self_attn: "attention" },
  autoCamelCase: true,
});

const checkpoint = {
  "model.encoder.layers.0.self_attn.gamma": 1,
  "model.encoder.layers.0.self_attn.beta": 2,
};

console.log(mapper.mapObject(checkpoint));
