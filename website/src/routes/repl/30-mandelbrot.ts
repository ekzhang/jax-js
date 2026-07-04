import { jit, numpy as np } from "@jax-js/jax";

// Mandelbrot iteration over a whole complex grid at once.
const width = 420;
const height = 300;
const xmin = -2.0;
const xmax = 0.6;
const yspan = ((xmax - xmin) * height) / width;
const [cx, cy] = np.meshgrid([
  np.linspace(xmin, xmax, width),
  np.linspace(-yspan / 2, yspan / 2, height),
]);

const step = jit((x: np.Array, y: np.Array, count: np.Array) => {
  const xx = x.ref.mul(x.ref);
  const yy = y.ref.mul(y.ref);
  count = count.add(xx.ref.add(yy.ref).less(4).astype(np.float32));
  const nextX = xx.sub(yy).add(cx.ref);
  const nextY = x.mul(y).mul(2).add(cy.ref);
  return [nextX, nextY, count];
});

let x = np.zeros([height, width]);
let y = np.zeros([height, width]);
let count = np.zeros([height, width]);
for (let i = 0; i < 80; i++) [x, y, count] = step(x, y, count);

await displayImage(count.div(80));
