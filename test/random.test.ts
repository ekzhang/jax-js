import { defaultDevice, devices, init, numpy as np, random } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init();

suite.each(devices)("device:%s", (device) => {
  const skipped = !devicesAvailable.includes(device);
  beforeEach(({ skip }) => {
    if (skipped) skip();
    defaultDevice(device);
  });

  test("random bits", () => {
    // jax.random.bits(jax.random.key(0))
    const x = random.bits(random.key(0));
    expect(x.shape).toEqual([]);
    expect(x.dtype).toEqual(np.uint32);
    expect(x.js()).toEqual(4070199207);

    // jax.random.bits(jax.random.key(0), shape=(4,))
    const y = random.bits(random.key(10), [4]);
    expect(y.shape).toEqual([4]);
    expect(y.dtype).toEqual(np.uint32);
    expect(y.js()).toEqual([169096361, 1572511259, 2689743692, 2228103506]);
  });

  test("random split is consistent with jax", () => {
    const splits = random.split(random.key(1), 3);
    expect(splits.shape).toEqual([3, 2]);
    expect(splits.dtype).toEqual(np.uint32);
    expect(splits.js()).toEqual([
      [507451445, 1853169794],
      [1948878966, 4237131848],
      [2441914641, 3819641963],
    ]);
  });

  test("generate uniform random", () => {
    const key = random.key(42);
    const [a, b, c] = random.split(key, 3);

    const x = random.uniform(a);
    expect(x.js()).toBeWithinRange(0, 1);

    const y = random.uniform(b, [0]);
    expect(y.js()).toEqual([]);

    const z = random.uniform(c, [2, 3], { minval: 10, maxval: 15 });
    expect(z.shape).toEqual([2, 3]);
    expect(z.dtype).toEqual(np.float32);
    const zx = z.js() as number[][];
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 3; j++) {
        expect(zx[i][j]).toBeWithinRange(10, 15);
      }
    }
  });

  test("uniform is consistent with jax", () => {
    // jax.random.uniform(jax.random.key(51), shape=(4,))
    const x = random.uniform(random.key(51), [4]);
    expect(x).toBeAllclose([0.471269, 0.12344253, 0.17550635, 0.5663593]);
  });

  test("normal distribution", () => {
    const key = random.key(123);
    const count = 5000;
    const values: number[] = random.normal(key, [count]).js();
    let onesigma = 0;
    let twosigma = 0;
    let threesigma = 0;
    for (const v of values) {
      if (Math.abs(v) <= 1) onesigma++;
      if (Math.abs(v) <= 2) twosigma++;
      if (Math.abs(v) <= 3) threesigma++;
    }
    // Approximately 68.27% within 1σ, 95.45% within 2σ, 99.73% within 3σ.
    expect(onesigma / count).toBeCloseTo(0.6827);
    expect(twosigma / count).toBeCloseTo(0.9545);
    expect(threesigma / count).toBeCloseTo(0.9973);
  });

  test("bernoulli distribution", () => {
    const key = random.key(2024);
    const count = 500;
    const p = np.array([[0.25], [0.8]]); // try array of p
    const samples: boolean[][] = random.bernoulli(key, p, [2, count]).js();
    const trues = [0, 0];
    for (let i = 0; i < 2; i++) {
      for (const s of samples[i]) {
        if (s) trues[i]++;
      }
    }
    expect(trues[0] / count).toBeCloseTo(0.25, 1);
    expect(trues[1] / count).toBeCloseTo(0.8, 1);
  });

  test("cauchy distribution", () => {
    const key = random.key(999);
    const count = 20000;
    const samples: number[] = random.cauchy(key, [count]).js();

    // Cauchy has heavy tails, so we can't use mean/variance tests.
    // Instead, check that the median is close to 0 and that quartiles
    // are close to ±1 (since tan(π/4) = 1).
    const sorted = samples.slice().sort((a, b) => a - b);
    const q1 = sorted[Math.floor(count * 0.25)];
    const median = sorted[Math.floor(count * 0.5)];
    const q3 = sorted[Math.floor(count * 0.75)];

    expect(median).toBeCloseTo(0, 1); // Median should be near 0
    expect(q1).toBeCloseTo(-1, 1); // 25th percentile should be near -1
    expect(q3).toBeCloseTo(1, 1); // 75th percentile should be near 1
  });

  test("laplace distribution", () => {
    const key = random.key(888);
    const count = 20000;
    const samples: number[] = random.laplace(key, [count]).js();

    // Laplace(0, 1) has mean 0 and variance 2
    const mean = samples.reduce((a, b) => a + b, 0) / count;
    const variance =
      samples.reduce((a, b) => a + (b - mean) ** 2, 0) / (count - 1);

    expect(mean).toBeCloseTo(0, 1);
    expect(variance).toBeCloseTo(2, 0); // variance = 2 * scale^2 = 2, relaxed to 0 decimal places
  });

  test("gumbel distribution", () => {
    const key = random.key(777);
    const count = 20000;
    const samples: number[] = random.gumbel(key, [count]).js();

    // Gumbel(0, 1) has mean = Euler-Mascheroni constant ≈ 0.5772
    // and variance = π²/6 ≈ 1.6449
    const eulerGamma = 0.5772156649;
    const mean = samples.reduce((a, b) => a + b, 0) / count;
    const variance =
      samples.reduce((a, b) => a + (b - mean) ** 2, 0) / (count - 1);

    expect(mean).toBeCloseTo(eulerGamma, 1);
    expect(variance).toBeCloseTo(Math.PI ** 2 / 6, 1);
  });
});
