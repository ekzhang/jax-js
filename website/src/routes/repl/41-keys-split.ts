import { random } from "@jax-js/jax";

// PRNG keys are explicit; split one key into independent child keys.
const keys = random.split(random.key(123), 4);

const a = random.normal(keys.ref.slice(0), [3]);
const b = random.normal(keys.slice(1), [3]);

console.log("sample A =", await a.jsAsync());
console.log("sample B =", await b.jsAsync());
