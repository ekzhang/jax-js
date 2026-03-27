/**
 * WebAssembly matrix multiplication benchmark using wasmblr directly.
 *
 * Implements the techniques from bwasti's blog post:
 * "WebAssembly Techniques to Speed Up Matrix Multiplication by 120x"
 *
 * Includes a JS baseline, naive wasm scalar, and optimized SIMD+unrolled version
 * with a parameter sweep to find the best unroll configuration.
 *
 * Reports GFlops (billion floating point operations per second).
 */

import { CodeGenerator } from "../src/backend/wasm/wasmblr";

// ─── JS baseline ───────────────────────────────────────────────────────────

function jsMatmul(
  a: Float32Array,
  b: Float32Array,
  c: Float32Array,
  M: number,
  N: number,
  K: number,
) {
  for (let m = 0; m < M; ++m) {
    for (let n = 0; n < N; ++n) {
      let sum = 0;
      for (let k = 0; k < K; ++k) {
        sum += a[m * K + k] * b[k * N + n];
      }
      c[m * N + n] = sum;
    }
  }
}

// ─── Naive scalar wasm matmul ──────────────────────────────────────────────

function buildNaiveMatmul(M: number, N: number, K: number): Uint8Array {
  const cg = new CodeGenerator();
  const pages = ((M * N + K * N + M * K) * 4) / (1 << 16) + 1;
  const A_off = 0;
  const B_off = M * K * 4;
  const C_off = (M * K + K * N) * 4;

  cg.memory.pages(pages).export("mem");

  const fn = cg.function([], [], () => {
    const m = cg.local.declare(cg.i32);
    const n = cg.local.declare(cg.i32);
    const k = cg.local.declare(cg.i32);
    const tmp = cg.local.declare(cg.f32);

    cg.i32.const(0);
    cg.local.set(m);
    cg.loop(cg.void); // M loop

    cg.i32.const(0);
    cg.local.set(n);
    cg.loop(cg.void); // N loop

    // sum = 0
    cg.f32.const(0);
    cg.local.set(tmp);

    cg.i32.const(0);
    cg.local.set(k);
    cg.loop(cg.void); // K loop

    // sum += A[m*K+k] * B[k*N+n]
    cg.local.get(tmp);

    cg.local.get(m);
    cg.i32.const(K);
    cg.i32.mul();
    cg.local.get(k);
    cg.i32.add();
    cg.i32.const(4);
    cg.i32.mul();
    cg.f32.load(0, A_off);

    cg.local.get(k);
    cg.i32.const(N);
    cg.i32.mul();
    cg.local.get(n);
    cg.i32.add();
    cg.i32.const(4);
    cg.i32.mul();
    cg.f32.load(0, B_off);

    cg.f32.mul();
    cg.f32.add();
    cg.local.set(tmp);

    cg.local.get(k);
    cg.i32.const(1);
    cg.i32.add();
    cg.local.tee(k);
    cg.i32.const(K);
    cg.i32.lt_u();
    cg.br_if(0);
    cg.end(); // K

    // C[m*N+n] = sum
    cg.local.get(m);
    cg.i32.const(N);
    cg.i32.mul();
    cg.local.get(n);
    cg.i32.add();
    cg.i32.const(4);
    cg.i32.mul();
    cg.local.get(tmp);
    cg.f32.store(0, C_off);

    cg.local.get(n);
    cg.i32.const(1);
    cg.i32.add();
    cg.local.tee(n);
    cg.i32.const(N);
    cg.i32.lt_u();
    cg.br_if(0);
    cg.end(); // N

    cg.local.get(m);
    cg.i32.const(1);
    cg.i32.add();
    cg.local.tee(m);
    cg.i32.const(M);
    cg.i32.lt_u();
    cg.br_if(0);
    cg.end(); // M
  });
  cg.export(fn, "mm");
  return cg.finish();
}

// ─── Optimized SIMD matmul with unrolling ──────────────────────────────────

function buildOptimizedMatmul(
  M: number,
  N: number,
  K: number,
  M_unroll: number,
  N_unroll: number,
  K_unroll: number,
): Uint8Array {
  const cg = new CodeGenerator();
  const pages = ((M * N + K * N + M * K) * 4) / (1 << 16) + 1;
  const A_off = 0;
  const B_off = M * K * 4;
  const C_off = (M * K + K * N) * 4;

  cg.memory.pages(pages).export("mem");

  const fn = cg.function([], [], () => {
    const m = cg.local.declare(cg.i32);
    const n = cg.local.declare(cg.i32);
    const k = cg.local.declare(cg.i32);

    const load_a: number[] = [];
    const load_b: number[] = [];
    for (let j = 0; j < K_unroll; ++j) {
      for (let i = 0; i < M_unroll; ++i) {
        load_a.push(cg.local.declare(cg.v128));
      }
      for (let i = 0; i < N_unroll; ++i) {
        load_b.push(cg.local.declare(cg.v128));
      }
    }

    const a_off = cg.local.declare(cg.i32);
    const b_off = cg.local.declare(cg.i32);
    const c_off = cg.local.declare(cg.i32);

    const accs: number[] = [];
    for (let i = 0; i < M_unroll * N_unroll; ++i) {
      accs.push(cg.local.declare(cg.v128));
    }

    cg.i32.const(0);
    cg.local.set(m);
    cg.loop(cg.void); // M loop

    cg.local.get(m);
    cg.i32.const(N * 4);
    cg.i32.mul();
    cg.local.set(c_off);

    cg.i32.const(0);
    cg.local.set(n);
    cg.loop(cg.void); // N loop

    // Load C accumulators
    for (let mu = 0; mu < M_unroll; ++mu) {
      for (let nu = 0; nu < N_unroll; ++nu) {
        cg.local.get(c_off);
        cg.v128.load(0, C_off + nu * 4 * 4 + mu * N * 4);
        cg.local.set(accs[mu * N_unroll + nu]);
      }
    }

    cg.local.get(m);
    cg.i32.const(K * 4);
    cg.i32.mul();
    cg.local.set(a_off);

    cg.local.get(n);
    cg.i32.const(4 * 4 * N_unroll);
    cg.i32.mul();
    cg.local.set(b_off);

    cg.i32.const(0);
    cg.local.set(k);
    cg.loop(cg.void); // K loop

    for (let ku = 0; ku < K_unroll; ++ku) {
      for (let mu = 0; mu < M_unroll; ++mu) {
        cg.local.get(a_off);
        cg.v128.load32_splat(0, A_off + (mu * K + ku) * 4);
        cg.local.set(load_a[mu * K_unroll + ku]);
      }

      for (let nu = 0; nu < N_unroll; ++nu) {
        cg.local.get(b_off);
        cg.v128.load(0, B_off + (ku * N + nu * 4) * 4);
        cg.local.set(load_b[nu * K_unroll + ku]);
      }

      for (let mu = 0; mu < M_unroll; ++mu) {
        for (let nu = 0; nu < N_unroll; ++nu) {
          cg.local.get(accs[mu * N_unroll + nu]);
          cg.local.get(load_a[mu * K_unroll + ku]);
          cg.local.get(load_b[nu * K_unroll + ku]);
          cg.f32x4.mul();
          cg.f32x4.add();
          cg.local.set(accs[mu * N_unroll + nu]);
        }
      }
    }

    cg.local.get(a_off);
    cg.i32.const(4 * K_unroll);
    cg.i32.add();
    cg.local.set(a_off);

    cg.local.get(b_off);
    cg.i32.const(N * 4 * K_unroll);
    cg.i32.add();
    cg.local.set(b_off);

    cg.local.get(k);
    cg.i32.const(K_unroll);
    cg.i32.add();
    cg.local.tee(k);
    cg.i32.const(K);
    cg.i32.lt_u();
    cg.br_if(0);
    cg.end(); // K

    for (let mu = 0; mu < M_unroll; ++mu) {
      for (let nu = 0; nu < N_unroll; ++nu) {
        cg.local.get(c_off);
        cg.local.get(accs[mu * N_unroll + nu]);
        cg.v128.store(0, C_off + nu * 4 * 4 + mu * N * 4);
      }
    }

    cg.local.get(c_off);
    cg.i32.const(N_unroll * 4 * 4);
    cg.i32.add();
    cg.local.set(c_off);

    cg.local.get(n);
    cg.i32.const(1);
    cg.i32.add();
    cg.local.tee(n);
    cg.i32.const(N / 4 / N_unroll);
    cg.i32.lt_u();
    cg.br_if(0);
    cg.end(); // N

    cg.local.get(m);
    cg.i32.const(M_unroll);
    cg.i32.add();
    cg.local.tee(m);
    cg.i32.const(M);
    cg.i32.lt_u();
    cg.br_if(0);
    cg.end(); // M
  });
  cg.export(fn, "mm");
  return cg.finish();
}

// ─── Helpers ───────────────────────────────────────────────────────────────

async function instantiateMatmul(bytes: Uint8Array) {
  const module = await WebAssembly.compile(bytes);
  const instance = await WebAssembly.instantiate(module);
  return instance;
}

function fillRandom(arr: Float32Array) {
  for (let i = 0; i < arr.length; ++i) {
    arr[i] = Math.random() * 2 - 1;
  }
}

function computeGflops(
  M: number,
  N: number,
  K: number,
  timeMs: number,
): number {
  return (2 * M * N * K) / (timeMs / 1000) / 1e9;
}

function benchFn(
  fn: () => void,
  warmup: number,
  iters: number,
): { medianMs: number; minMs: number } {
  for (let i = 0; i < warmup; ++i) fn();
  const times: number[] = [];
  for (let i = 0; i < iters; ++i) {
    const t0 = performance.now();
    fn();
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  return {
    medianMs: times[Math.floor(times.length / 2)],
    minMs: times[0],
  };
}

// ─── Benchmark ─────────────────────────────────────────────────────────────

const WARMUP = 5;
const ITERS = 10;
const M = 512,
  N = 512,
  K = 512;
const flops = 2 * M * N * K;

console.log(`\n${"=".repeat(60)}`);
console.log(
  `  Matrix size: ${M}x${K} * ${K}x${N}  (${(flops / 1e6).toFixed(1)}M flops)`,
);
console.log(`${"=".repeat(60)}`);

// JS baseline
{
  const a = new Float32Array(M * K);
  const b = new Float32Array(K * N);
  const c = new Float32Array(M * N);
  fillRandom(a);
  fillRandom(b);
  const { medianMs, minMs } = benchFn(
    () => jsMatmul(a, b, c, M, N, K),
    WARMUP,
    ITERS,
  );
  console.log(
    `  JS baseline:       ${computeGflops(M, N, K, medianMs).toFixed(2)} gflops (median ${medianMs.toFixed(2)}ms, best ${computeGflops(M, N, K, minMs).toFixed(2)} gflops)`,
  );
}

// Naive wasm
{
  const bytes = buildNaiveMatmul(M, N, K);
  const instance = await instantiateMatmul(bytes);
  const mem = (instance.exports as any).mem as WebAssembly.Memory;
  const mm = (instance.exports as any).mm as () => void;
  const a = new Float32Array(mem.buffer, 0, M * K);
  const b = new Float32Array(mem.buffer, M * K * 4, K * N);
  const c = new Float32Array(mem.buffer, (M * K + K * N) * 4, M * N);
  fillRandom(a);
  fillRandom(b);
  const { medianMs, minMs } = benchFn(
    () => {
      c.fill(0);
      mm();
    },
    WARMUP,
    ITERS,
  );
  console.log(
    `  Wasm naive:        ${computeGflops(M, N, K, medianMs).toFixed(2)} gflops (median ${medianMs.toFixed(2)}ms, best ${computeGflops(M, N, K, minMs).toFixed(2)} gflops)`,
  );
}

// SIMD sweep
console.log(`\n  SIMD parameter sweep:`);
console.log(
  `  ${"m".padStart(4)} ${"n".padStart(4)} ${"k".padStart(4)}  ${"gflops".padStart(8)} (median)  ${"gflops".padStart(8)} (best)   ${"ms".padStart(8)} (median)`,
);
console.log(`  ${"-".repeat(56)}`);

let bestGflops = 0;
let bestConfig = "";

for (const mu of [1, 2, 4]) {
  for (const nu of [1, 2, 4]) {
    for (const ku of [1, 2, 4, 8]) {
      if (M % mu !== 0) continue;
      if (N % (nu * 4) !== 0) continue;
      if (K % ku !== 0) continue;

      try {
        const bytes = buildOptimizedMatmul(M, N, K, mu, nu, ku);
        const instance = await instantiateMatmul(bytes);
        const mem = (instance.exports as any).mem as WebAssembly.Memory;
        const mm = (instance.exports as any).mm as () => void;
        const a = new Float32Array(mem.buffer, 0, M * K);
        const b = new Float32Array(mem.buffer, M * K * 4, K * N);
        const c = new Float32Array(mem.buffer, (M * K + K * N) * 4, M * N);
        fillRandom(a);
        fillRandom(b);
        const { medianMs, minMs } = benchFn(
          () => {
            c.fill(0);
            mm();
          },
          WARMUP,
          ITERS,
        );
        const medGf = computeGflops(M, N, K, medianMs);
        const bestGf = computeGflops(M, N, K, minMs);

        if (medGf > bestGflops) {
          bestGflops = medGf;
          bestConfig = `m=${mu},n=${nu},k=${ku}`;
        }

        console.log(
          `  ${String(mu).padStart(4)} ${String(nu).padStart(4)} ${String(ku).padStart(4)}  ${medGf.toFixed(2).padStart(8)}          ${bestGf.toFixed(2).padStart(8)}          ${medianMs.toFixed(2).padStart(8)}`,
        );
      } catch {
        // Some configs may fail
      }
    }
  }
}

console.log(`\n  Best: ${bestConfig} @ ${bestGflops.toFixed(2)} gflops`);
