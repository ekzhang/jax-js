import { safetensors } from "@jax-js/loaders";

// Safetensor keys can be converted between flat names and nested objects.
const flat = {
  "layers.0.weight": "w0",
  "layers.0.bias": "b0",
  "layers.1.weight": "w1",
  "layers.1.bias": "b1",
};

const nested = safetensors.toNested(flat);
const roundTrip = safetensors.fromNested(nested);

console.log("nested =", nested);
console.log("flat again =", roundTrip);
