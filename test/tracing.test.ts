import {
  grad,
  jit,
  jvp,
  linearize,
  makeJaxpr,
  numpy as np,
  vjp,
  vmap,
} from "@jax-js/jax";
import { expect, suite, test } from "vitest";

suite("jax.makeJaxpr()", () => {
  test("tracks a nullary function", () => {
    const { jaxpr, consts } = makeJaxpr(() => np.multiply(2, 2))();
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda  .
        ( 4 ) }"
    `);
    expect(consts).toEqual([]);
  });

  test("tracks a unary function", () => {
    const { jaxpr, consts } = makeJaxpr((x: np.Array) =>
      np.multiply(x.add(2), x),
    )(
      np.array([
        [2, 4, 10],
        [1, 1, 1],
      ]),
    );
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda a:float32[2,3] .
        let b:float32[2,3] = add a 2
            c:float32[2,3] = mul b a
        in ( c ) }"
    `);
    expect(consts).toEqual([]);
  });

  test("composes with jvp", () => {
    const f = (x: np.Array) => np.multiply(x.add(2), x);
    const fdot = (x: np.Array) => jvp(f, [x], [1])[1];

    const { jaxpr, consts } = makeJaxpr(fdot)(np.array(2));
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda a:float32[] .
        let b:float32[] = add a 2
            c:float32[] = add b a
        in ( c ) }"
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
      "{ lambda a:float32[] .
        let b:float32[] = add 1 a
            c:float32[] = add b a
        in ( c ) }"
    `);
  });

  test("can flatten() nested Jaxprs", () => {
    const f = (x: np.Array) => {
      const y = x.add(2);
      return x.mul(x).add(y);
    };
    const jf = jit(f);

    const { jaxpr, consts } = makeJaxpr((x) => f(jf(x)))(3);
    expect(consts).toEqual([]);
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda a:float32[] .
        let b:float32[] = jit_call [ jaxpr={ lambda a:float32[] .
                                       let b:float32[] = add a 2
                                           c:float32[] = mul a a
                                           d:float32[] = add c b
                                       in ( d ) }
                                     numConsts=0 ] a
            c:float32[] = add b 2
            d:float32[] = mul b b
            e:float32[] = add d c
        in ( e ) }"
    `);
    expect(jaxpr.flatten().toString()).toMatchInlineSnapshot(`
      "{ lambda a:float32[] .
        let b:float32[] = add a 2
            c:float32[] = mul a a
            d:float32[] = add c b
            e:float32[] = add d 2
            f:float32[] = mul d d
            g:float32[] = add f e
        in ( g ) }"
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
      { a: 1, b: 2 },
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

suite("jax.jit()", () => {
  test("works for a simple scalar function", () => {
    const f = (x: np.Array) => x.mul(x).mul(x); // d/dx (x^3) = 3x^2
    const f2 = jit(f);
    expect(f(np.array(2))).toBeAllclose(8);
    expect(f2(np.array(2))).toBeAllclose(8);
  });

  test("jit-of-jit", () => {
    const f = jit((x: np.Array) => x.mul(x));
    const g = jit((x: np.Array) => f(f(x)));
    expect(g(3)).toBeAllclose(81);
    expect(jit(jit(g))(3)).toBeAllclose(81);
  });

  test("jvp-of-jit", () => {
    const f = jit((x: np.Array) => x.mul(x));
    expect(jvp(f, [3], [1])).toBeAllclose([9, 6]);
  });

  test("grad-of-jit", () => {
    const f = jit((x: np.Array) => x.mul(x));
    expect(grad(f)(3)).toBeAllclose(6);
    expect(grad(f)(10)).toBeAllclose(20);
    expect(grad(grad(f))(10)).toBeAllclose(2);
    expect(grad(jit(grad(f)))(10)).toBeAllclose(2);
  });

  test("vmap-of-jit", () => {
    const s = jit((x: np.Array) => x.sum());
    const ar = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(s(ar).js()).toEqual(21);
    expect(vmap(s, [0])(ar).js()).toEqual([6, 15]);
    expect(vmap(s, [1])(ar).js()).toEqual([5, 7, 9]);
  });
});
