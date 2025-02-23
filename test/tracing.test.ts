import { expect, suite, test } from "vitest";
import { numpy as np, makeJaxpr, jvp, linearize, vjp, grad } from "jax-js";

suite("jax.makeJaxpr()", () => {
  test("tracks a nullary function", () => {
    const { jaxpr, consts } = makeJaxpr(() => np.mul(2, 2))();
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda  .
        let 
        in ( 4 ) }"
    `);
    expect(consts).toEqual([]);
  });

  test("tracks a unary function", () => {
    const { jaxpr, consts } = makeJaxpr((x: np.Array) => np.mul(x.add(2), x))(
      np.array([
        [2, 4, 10],
        [1, 1, 1],
      ])
    );
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda v_1:float32[2,3] .
        let v_3:float32[2,3] = add v_1 2
            v_4:float32[2,3] = mul v_3 v_1
        in ( v_4 ) }"
    `);
    expect(consts).toEqual([]);
  });

  test("composes with jvp", () => {
    const f = (x: np.Array) => np.mul(x.add(2), x);
    const fdot = (x: np.Array) => jvp(f, [x], [np.array(1)])[1];

    const { jaxpr, consts } = makeJaxpr(fdot)(np.array(2));
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda v_1:float32[] .
        let v_3:float32[] = add v_1 2
            v_10:float32[] = add v_3 v_1
        in ( v_10 ) }"
    `);
    expect(consts).toEqual([]);
  });

  test("composes with grad", () => {
    const f = (x: np.Array) => {
      const y = x.add(2);
      return x.mul(x).add(y);
    };
    const { jaxpr, consts } = makeJaxpr(grad(f))(3);
    expect(consts).toEqual([]);
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda v_1:float32[] .
        let v_16:float32[] = add 1 v_1
            v_18:float32[] = add v_16 v_1
        in ( v_18 ) }"
    `);
  });
});

suite("jax.linearize()", () => {
  test("works for scalars", () => {
    const [y, lin] = linearize(np.sin, 3);
    expect(y).toBeAllclose(np.sin(3));
    expect(lin(1)).toBeAllclose(np.cos(3));
    expect(lin(-42)).toBeAllclose(np.cos(3).mul(-42));
  });

  test("works for simple arrays", () => {
    const [y, lin] = linearize((x: np.Array) => x.mul(x), np.array([2, 3]));
    expect(y).toBeAllclose(np.array([4, 9]));
    expect(lin(np.array([1, 0]))).toBeAllclose(np.array([4, 0]));
    expect(lin(np.array([0, 1]))).toBeAllclose(np.array([0, 6]));
  });

  test("can take and return jstrees", () => {
    const [y, lin] = linearize(
      (x: { a: np.Array; b: np.Array }) => ({
        r1: x.a.mul(x.a).add(x.b),
        r2: x.b,
      }),
      { a: 1, b: 2 }
    );
    expect(y.r1).toBeAllclose(3);
    expect(y.r2).toBeAllclose(2);

    const { r1: r1Dot, r2: r2Dot } = lin({ a: 1, b: 0 });
    expect(r1Dot).toBeAllclose(2);
    expect(r2Dot).toBeAllclose(0);
  });
});

suite("jax.vjp()", () => {
  test("works for scalars", () => {
    const [y, backward] = vjp(np.sin, 3);
    expect(y).toBeAllclose(np.sin(3));
    expect(backward(1)[0]).toBeAllclose(np.cos(3));
  });
});

suite("jax.grad()", () => {
  test("works for a simple scalar function", () => {
    const f = (x: np.Array) => x.mul(x).mul(x); // d/dx (x^3) = 3x^2
    const df = grad(f);
    expect(df(4)).toBeAllclose(48);
    expect(df(5)).toBeAllclose(75);
    expect(df(0)).toBeAllclose(0);
    expect(df(-4)).toBeAllclose(48);
  });

  test("can compute higher derivatives", () => {
    const f = (x: np.Array) => np.sin(np.cos(x));
    const df = grad(f); // d/dx sin(cos(x)) = -sin(x)cos(cos(x))
    const ddf = grad(df); // d^2/dx^2 sin(cos(x)) = -sin^2(x)sin(cos(x)) - cos(x)cos(cos(x))
    expect(df(3)).toBeAllclose(-0.077432003);
    expect(ddf(3)).toBeAllclose(0.559854311);
  });
});
