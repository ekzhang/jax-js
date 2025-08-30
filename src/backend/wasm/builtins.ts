// Complex primitives that need to be implemented in software.

import { CodeGenerator } from "./wasmblr";

/**
 * Approximate e^x.
 *
 * Method: range-reduce x = k*ln2 + r with k = round(x/ln2), |r|<=~0.3466
 *         then e^x = 2^k * P(r), where P is 5th-order poly (Taylor).
 */
export function wasm_exp(cg: CodeGenerator): number {
  return cg.function([cg.f32], [cg.f32], () => {
    const k_f = cg.local.declare(cg.f32);
    const k = cg.local.declare(cg.i32);
    const r = cg.local.declare(cg.f32);
    const p = cg.local.declare(cg.f32);
    const scale = cg.local.declare(cg.f32);

    // k = nearest(x / ln2)
    cg.local.get(0);
    cg.f32.const(1 / Math.LN2);
    cg.f32.mul();
    cg.f32.nearest();
    cg.local.tee(k_f);
    cg.i32.trunc_f32_s();
    cg.local.set(k);

    // r = x - k*ln2
    cg.local.get(0);
    cg.local.get(k_f);
    cg.f32.const(Math.LN2);
    cg.f32.mul();
    cg.f32.sub();
    cg.local.set(r);

    // P(r) ≈ 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120
    // Horner form: 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r/120))))
    cg.f32.const(1 / 120);
    cg.local.get(r);
    cg.f32.mul();
    cg.f32.const(1 / 24);
    cg.f32.add();
    cg.local.get(r);
    cg.f32.mul();
    cg.f32.const(1 / 6);
    cg.f32.add();
    cg.local.get(r);
    cg.f32.mul();
    cg.f32.const(1 / 2);
    cg.f32.add();
    cg.local.get(r);
    cg.f32.mul();
    cg.f32.const(1.0);
    cg.f32.add();
    cg.local.get(r);
    cg.f32.mul();
    cg.f32.const(1.0);
    cg.f32.add();
    cg.local.set(p);

    // scale = 2^k via exponent bits: ((k + 127) << 23)
    cg.local.get(k);
    cg.i32.const(127);
    cg.i32.add();
    cg.i32.const(23);
    cg.i32.shl();
    cg.f32.reinterpret_i32();
    cg.local.set(scale);

    // result = P(r) * 2^k
    cg.local.get(p);
    cg.local.get(scale);
    cg.f32.mul();
  });
}

/**
 * Approximate ln(x), x > 0.
 *
 * Method: decompose x = m * 2^e with m in [1,2), e integer (via bit ops)
 *         ln(x) = e*ln2 + ln(m);  use atanh-style series with t=(m-1)/(m+1)
 *         ln(m) ≈ 2*(t + t^3/3 + t^5/5 + t^7/7)
 */
export function wasm_log(cg: CodeGenerator): number {
  return cg.function([cg.f32], [cg.f32], () => {
    const bits = cg.local.declare(cg.i32);
    const e = cg.local.declare(cg.i32);
    const m = cg.local.declare(cg.f32);
    const t = cg.local.declare(cg.f32);
    const t2 = cg.local.declare(cg.f32);
    const t3 = cg.local.declare(cg.f32);
    const t5 = cg.local.declare(cg.f32);
    const t7 = cg.local.declare(cg.f32);
    const lnm = cg.local.declare(cg.f32);
    const el2 = cg.local.declare(cg.f32);

    // Handle (very) small or non-positive quickly: if x <= 0 -> NaN
    cg.local.get(0);
    cg.f32.const(0.0);
    cg.f32.le();
    cg.if(cg.void);
    {
      cg.f32.const(NaN);
      cg.return();
    }
    cg.end();

    // bits = reinterpret(x)
    cg.local.get(0);
    cg.i32.reinterpret_f32();
    cg.local.tee(bits);

    // e = ((bits >> 23) & 0xff) - 127
    cg.i32.const(23);
    cg.i32.shr_u();
    cg.i32.const(255);
    cg.i32.and();
    cg.i32.const(127);
    cg.i32.sub();
    cg.local.set(e);

    // m_bits = (bits & 0x7fffff) | 0x3f800000  => m in [1,2)
    cg.local.get(bits);
    cg.i32.const(0x7fffff);
    cg.i32.and();
    cg.i32.const(0x3f800000);
    cg.i32.or();
    cg.f32.reinterpret_i32();
    cg.local.set(m);

    // t = (m - 1) / (m + 1)
    cg.local.get(m);
    cg.f32.const(1.0);
    cg.f32.sub();
    cg.local.get(m);
    cg.f32.const(1.0);
    cg.f32.add();
    cg.f32.div();
    cg.local.set(t);

    // powers of t
    cg.local.get(t);
    cg.local.get(t);
    cg.f32.mul();
    cg.local.set(t2); // t^2
    cg.local.get(t);
    cg.local.get(t2);
    cg.f32.mul();
    cg.local.set(t3); // t^3
    cg.local.get(t3);
    cg.local.get(t2);
    cg.f32.mul();
    cg.local.set(t5); // t^5
    cg.local.get(t5);
    cg.local.get(t2);
    cg.f32.mul();
    cg.local.set(t7); // t^7

    // lnm ≈ 2 * ( t + t^3/3 + t^5/5 + t^7/7 )
    cg.local.get(t7);
    cg.f32.const(1 / 7);
    cg.f32.mul();
    cg.local.get(t5);
    cg.f32.const(1 / 5);
    cg.f32.mul();
    cg.f32.add();
    cg.local.get(t3);
    cg.f32.const(1 / 3);
    cg.f32.mul();
    cg.f32.add();
    cg.local.get(t);
    cg.f32.add();
    cg.f32.const(2.0);
    cg.f32.mul();
    cg.local.set(lnm);

    // el2 = e * ln2
    cg.local.get(e);
    cg.f32.convert_i32_s();
    cg.f32.const(Math.LN2);
    cg.f32.mul();
    cg.local.set(el2);

    // ln(x) ≈ e*ln2 + ln(m)
    cg.local.get(el2);
    cg.local.get(lnm);
    cg.f32.add();
  });
}

/**
 * Approximate sin(x).
 *
 * Method: reduce to y in [-π, π], then quadrant via q = round(y/(π/2))
 *         z = y - q*(π/2); use odd polynomial on z:
 *         sin(z) ≈ z + z^3*(-1/6) + z^5*(1/120) + z^7*(-1/5040)
 */
export function wasm_sin(cg: CodeGenerator): number {
  return cg.function([cg.f32], [cg.f32], () => {
    const y = cg.local.declare(cg.f32);
    const qf = cg.local.declare(cg.f32);
    const q = cg.local.declare(cg.i32);
    const z = cg.local.declare(cg.f32);
    const sz = cg.local.declare(cg.f32);
    const cz = cg.local.declare(cg.f32);
    const z2 = cg.local.declare(cg.f32);
    const z3 = cg.local.declare(cg.f32);
    const z4 = cg.local.declare(cg.f32);
    const z5 = cg.local.declare(cg.f32);
    const z6 = cg.local.declare(cg.f32);
    const z7 = cg.local.declare(cg.f32);
    const mag = cg.local.declare(cg.f32);

    // y = x - round(x / (2π)) * (2π)  (mod 2π to [-π, π] implicitly)
    cg.local.get(0);
    cg.local.get(0);
    cg.f32.const(2 * Math.PI);
    cg.f32.div();
    cg.f32.nearest();
    cg.local.tee(qf);
    cg.f32.const(2 * Math.PI);
    cg.f32.mul();
    cg.f32.sub();
    cg.local.set(y);

    // q = round(y / (π/2))
    cg.local.get(y);
    cg.f32.const(Math.PI / 2);
    cg.f32.div();
    cg.f32.nearest();
    cg.local.tee(qf);
    cg.i32.trunc_f32_s();
    cg.local.set(q);

    // z = y - q*(π/2)
    cg.local.get(y);
    cg.local.get(qf);
    cg.f32.const(Math.PI / 2);
    cg.f32.mul();
    cg.f32.sub();
    cg.local.set(z);

    // powers of z
    cg.local.get(z);
    cg.local.get(z);
    cg.f32.mul();
    cg.local.set(z2);
    cg.local.get(z);
    cg.local.get(z2);
    cg.f32.mul();
    cg.local.set(z3);
    cg.local.get(z2);
    cg.local.get(z2);
    cg.f32.mul();
    cg.local.set(z4);
    cg.local.get(z3);
    cg.local.get(z2);
    cg.f32.mul();
    cg.local.set(z5);
    cg.local.get(z4);
    cg.local.get(z2);
    cg.f32.mul();
    cg.local.set(z6);
    cg.local.get(z5);
    cg.local.get(z2);
    cg.f32.mul();
    cg.local.set(z7);

    // sin poly at z: sz = z + z^3*(-1/6) + z^5*(1/120) + z^7*(-1/5040)
    cg.local.get(z);
    cg.local.get(z3);
    cg.f32.const(-1 / 6);
    cg.f32.mul();
    cg.f32.add();
    cg.local.get(z5);
    cg.f32.const(1 / 120);
    cg.f32.mul();
    cg.f32.add();
    cg.local.get(z7);
    cg.f32.const(-1 / 5040);
    cg.f32.mul();
    cg.f32.add();
    cg.local.set(sz);

    // cos poly at z: cz = 1 + z^2*(-1/2) + z^4*(1/24) + z^6*(-1/720)
    cg.f32.const(1.0);
    cg.local.get(z2);
    cg.f32.const(-0.5);
    cg.f32.mul();
    cg.f32.add();
    cg.local.get(z4);
    cg.f32.const(1 / 24);
    cg.f32.mul();
    cg.f32.add();
    cg.local.get(z6);
    cg.f32.const(-1 / 720);
    cg.f32.mul();
    cg.f32.add();
    cg.local.set(cz);

    // magnitude: (q & 1) ? cz : sz
    cg.local.get(cz);
    cg.local.get(sz);
    cg.local.get(q);
    cg.i32.const(1);
    cg.i32.and();
    cg.select();
    cg.local.set(mag);

    // sign: (q & 2) ? -mag : mag
    cg.local.get(mag);
    cg.f32.neg();
    cg.local.get(mag);
    cg.local.get(q);
    cg.i32.const(2);
    cg.i32.and();
    cg.select();
  });
}

/**
 * Approximate cos(x).
 *
 * Same reduction as sinf, then quadrant mapping:
 * k=q mod 4: 0: +cz, 1: -sz, 2: -cz, 3: +sz
 */
export function wasm_cos(cg: CodeGenerator): number {
  return cg.function([cg.f32], [cg.f32], () => {
    const y = cg.local.declare(cg.f32);
    const qf = cg.local.declare(cg.f32);
    const q = cg.local.declare(cg.i32);
    const z = cg.local.declare(cg.f32);
    const sz = cg.local.declare(cg.f32);
    const cz = cg.local.declare(cg.f32);
    const z2 = cg.local.declare(cg.f32);
    const z3 = cg.local.declare(cg.f32);
    const z4 = cg.local.declare(cg.f32);
    const z5 = cg.local.declare(cg.f32);
    const z6 = cg.local.declare(cg.f32);
    const z7 = cg.local.declare(cg.f32);
    const mag = cg.local.declare(cg.f32);

    // y = x - round(x / (2π)) * (2π)
    cg.local.get(0);
    cg.local.get(0);
    cg.f32.const(2 * Math.PI);
    cg.f32.div();
    cg.f32.nearest();
    cg.local.tee(qf);
    cg.f32.const(2 * Math.PI);
    cg.f32.mul();
    cg.f32.sub();
    cg.local.set(y);

    // q = round(y / (π/2)); z = y - q*(π/2)
    cg.local.get(y);
    cg.f32.const(Math.PI / 2);
    cg.f32.div();
    cg.f32.nearest();
    cg.local.tee(qf);
    cg.i32.trunc_f32_s();
    cg.local.set(q);

    cg.local.get(y);
    cg.local.get(qf);
    cg.f32.const(Math.PI / 2);
    cg.f32.mul();
    cg.f32.sub();
    cg.local.set(z);

    // sin/cos polynomials at z
    cg.local.get(z);
    cg.local.get(z);
    cg.f32.mul();
    cg.local.set(z2);
    cg.local.get(z);
    cg.local.get(z2);
    cg.f32.mul();
    cg.local.set(z3);
    cg.local.get(z2);
    cg.local.get(z2);
    cg.f32.mul();
    cg.local.set(z4);
    cg.local.get(z4);
    cg.local.get(z2);
    cg.f32.mul();
    cg.local.set(z6);
    cg.local.get(z3);
    cg.local.get(z2);
    cg.f32.mul();
    cg.local.set(z5);
    cg.local.get(z5);
    cg.local.get(z2);
    cg.f32.mul();
    cg.local.set(z7);

    // sz
    cg.local.get(z);
    cg.local.get(z3);
    cg.f32.const(-1 / 6);
    cg.f32.mul();
    cg.f32.add();
    cg.local.get(z5);
    cg.f32.const(1 / 120);
    cg.f32.mul();
    cg.f32.add();
    cg.local.get(z7);
    cg.f32.const(-1 / 5040);
    cg.f32.mul();
    cg.f32.add();
    cg.local.set(sz);

    // cz
    cg.f32.const(1.0);
    cg.local.get(z2);
    cg.f32.const(-0.5);
    cg.f32.mul();
    cg.f32.add();
    cg.local.get(z4);
    cg.f32.const(1 / 24);
    cg.f32.mul();
    cg.f32.add();
    cg.local.get(z6);
    cg.f32.const(-1 / 720);
    cg.f32.mul();
    cg.f32.add();
    cg.local.set(cz);

    // magnitude: (q & 1) ? sz : cz
    cg.local.get(sz);
    cg.local.get(cz);
    cg.local.get(q);
    cg.i32.const(1);
    cg.i32.and();
    cg.select();
    cg.local.set(mag);

    // sign: ((q+1) & 2) ? -mag : mag
    cg.local.get(mag);
    cg.f32.neg();
    cg.local.get(mag);
    cg.local.get(q);
    cg.i32.const(1);
    cg.i32.add();
    cg.i32.const(2);
    cg.i32.and();
    cg.select();
  });
}
