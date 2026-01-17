import {
  grad,
  jit,
  jvp,
  linearize,
  makeJaxpr,
  nn,
  numpy as np,
  tree,
  valueAndGrad,
  vjp,
  vmap,
} from "@jax-js/jax";
import { expect, suite, test } from "vitest";

suite("jax.makeJaxpr()", () => {
  test("tracks a nullary function", () => {
    const { jaxpr } = makeJaxpr(() => np.multiply(2, 2))();
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda  .
        ( 4 ) }"
    `);
    expect(jaxpr.consts).toEqual([]);
  });

  test("tracks a unary function", () => {
    const { jaxpr } = makeJaxpr((x: np.Array) => np.multiply(x.ref.add(2), x))(
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
    expect(jaxpr.consts).toEqual([]);
  });

  test("composes with jvp", () => {
    const f = (x: np.Array) => np.multiply(x.ref.add(2), x);
    const fdot = (x: np.Array) => {
      const [y, dy] = jvp(f, [x.ref], [1]);
      y.dispose();
      return dy;
    };

    const { jaxpr } = makeJaxpr(fdot)(np.array(2));
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda a:float32[] .
        let b:float32[] = add a 2
            c:float32[] = add b a
        in ( c ) }"
    `);
    expect(jaxpr.consts).toEqual([]);
  });

  test("composes with grad", () => {
    const f = (x: np.Array) => {
      const y = x.ref.add(2);
      return x.ref.mul(x).add(y);
    };
    const { jaxpr } = makeJaxpr(grad(f))(3);
    expect(jaxpr.consts).toEqual([]);
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda a:float32[] .
        let b:float32[] = add 1 a
            c:float32[] = add b a
        in ( c ) }"
    `);
  });

  test("can flatten() nested Jaxprs", () => {
    const f = (x: np.Array) => {
      const y = x.ref.add(2);
      return x.ref.mul(x).add(y);
    };
    const jf = jit(f);

    const { jaxpr } = makeJaxpr((x) => f(jf(x)))(3);
    expect(jaxpr.consts).toEqual([]);
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda a:float32[] .
        let b:float32[] = jit [ name=f
                                jaxpr={ lambda a:float32[] .
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
    expect(jaxpr.jaxpr.flatten().toString()).toMatchInlineSnapshot(`
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
    const [y, lin] = linearize(np.sin, [3]);
    expect(y).toBeAllclose(np.sin(3));
    expect(lin(1)).toBeAllclose(np.cos(3));
    expect(lin(-42)).toBeAllclose(np.cos(3).mul(-42));
  });

  test("works for simple arrays", () => {
    const [y, lin] = linearize(
      (x: np.Array) => x.ref.mul(x),
      [np.array([2, 3])],
    );
    expect(y).toBeAllclose(np.array([4, 9]));
    expect(lin(np.array([1, 0]))).toBeAllclose(np.array([4, 0]));
    expect(lin(np.array([0, 1]))).toBeAllclose(np.array([0, 6]));
  });

  test("can take and return jstrees", () => {
    const [y, lin] = linearize(
      (x: { a: np.Array; b: np.Array }) => ({
        r1: x.a.ref.mul(x.a).add(x.b.ref),
        r2: x.b,
      }),
      [{ a: 1, b: 2 }],
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
    const [y, backward] = vjp(np.sin, [3]);
    expect(y).toBeAllclose(np.sin(3));
    expect(backward(1)[0]).toBeAllclose(np.cos(3));
  });

  test("hasAux returns aux and computes correct gradients", () => {
    const f = (x: np.Array): [np.Array, np.Array] => {
      const loss = x.ref.sum();
      const aux = x.mul(2);
      return [loss, aux];
    };

    const x = np.array([1, 2, 3]);
    const [loss, vjpFn, aux] = vjp(f, [x], { hasAux: true });

    expect(loss).toBeAllclose(6);
    expect(aux).toBeAllclose([2, 4, 6]);

    const [grad] = vjpFn(1);
    expect(grad).toBeAllclose([1, 1, 1]);

    vjpFn.dispose();
  });

  test("hasAux handles pytree aux", () => {
    type Aux = { predictions: np.Array; squared: np.Array };
    const f = (x: np.Array): [np.Array, Aux] => {
      const loss = x.ref.sum();
      const aux = {
        predictions: x.ref.mul(2),
        squared: x.ref.mul(x),
      };
      return [loss, aux];
    };

    const x = np.array([1, 2, 3]);
    const [loss, vjpFn, aux] = vjp(f, [x], { hasAux: true });

    expect(loss).toBeAllclose(6);
    expect(aux.predictions).toBeAllclose([2, 4, 6]);
    expect(aux.squared).toBeAllclose([1, 4, 9]);

    vjpFn.dispose();
  });

  test("hasAux handles pytree main output", () => {
    type Main = { a: np.Array; b: np.Array };
    const f = (x: np.Array): [Main, np.Array] => {
      const main = { a: x.ref.sum(), b: x.ref.prod() };
      const aux = x.mul(2);
      return [main, aux];
    };

    const x = np.array([1, 2, 3]);
    const [main, vjpFn, _aux] = vjp(f, [x], { hasAux: true });

    expect(main.a).toBeAllclose(6);
    expect(main.b).toBeAllclose(6);

    const [grad] = vjpFn({ a: np.ones([]), b: np.ones([]) });
    expect(grad).toBeAllclose([7, 4, 3]);

    vjpFn.dispose();
  });

  test("hasAux throws if function does not return tuple", () => {
    const f = (x: np.Array) => x.sum();
    const x = np.array([1, 2, 3]);
    expect(() => vjp(f as any, [x], { hasAux: true })).toThrow(/tuple/);
  });

  test("hasAux gradients match vjp without aux", () => {
    const fWithAux = (x: np.Array): [np.Array, np.Array] => [
      x.ref.sum(),
      x.mul(2),
    ];
    const fWithoutAux = (x: np.Array): np.Array => x.sum();

    const x = np.array([1, 2, 3]);

    const [, vjpFn1] = vjp(fWithAux, [x.ref], { hasAux: true });
    const [, vjpFn2] = vjp(fWithoutAux, [x]);

    const [grad1] = vjpFn1(np.ones([]));
    const [grad2] = vjpFn2(np.ones([]));

    expect(grad1).toBeAllclose(grad2);

    vjpFn1.dispose();
  });

  test("hasAux works with jit wrapper", () => {
    const f = jit((x: np.Array): [np.Array, np.Array] => [
      x.ref.sum(),
      x.mul(2),
    ]);

    const x = np.array([1, 2, 3]);
    const [loss, vjpFn, aux] = vjp(f, [x], { hasAux: true });

    expect(loss).toBeAllclose(6);
    expect(aux).toBeAllclose([2, 4, 6]);
    const [grad] = vjpFn(np.ones([]));
    expect(grad).toBeAllclose([1, 1, 1]);

    vjpFn.dispose();
  });

  test("hasAux works inside jit", () => {
    const inner = (x: np.Array): [np.Array, np.Array] => [
      x.ref.sum(),
      x.mul(2),
    ];

    const outer = jit((x: np.Array): [np.Array, np.Array] => {
      const [y, vjpFn, aux] = vjp(inner, [x], { hasAux: true });
      tree.dispose(y);
      const [grad] = vjpFn(np.ones([]));
      vjpFn.dispose();
      return [grad, aux];
    });

    const x = np.array([1, 2, 3]);
    const [grad, aux] = outer(x);

    expect(grad).toBeAllclose([1, 1, 1]);
    expect(aux).toBeAllclose([2, 4, 6]);
  });
});

suite("jax.grad()", () => {
  test("works for a simple scalar function", () => {
    const f = (x: np.Array) => x.ref.mul(x.ref).mul(x); // d/dx (x^3) = 3x^2
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

  test("can compute grad of products", () => {
    const x = np.array([1, 2, 3, 4]);
    const gradProd = grad((x: np.Array) => np.prod(x))(x);
    expect(gradProd.js()).toEqual([24, 12, 8, 6]);
  });

  test("passing const argnums", () => {
    const f = (x: np.Array, y: [np.Array, np.Array]) =>
      x.ref.mul(y[0]).add(y[1]).sum();
    const df_dx = grad(f, { argnums: 0 });
    const df_dy = grad(f, { argnums: 1 });
    const x = np.array([2, 3]);
    const y: [np.Array, np.Array] = [np.array([4, 5]), np.array([10, 20])];

    expect(df_dx(x.ref, tree.ref(y)).js()).toEqual([4, 5]); // dy/dx = y0
    const w = df_dy(x.ref, tree.ref(y));
    expect(w[0].js()).toEqual([2, 3]); // dy/dy0 = x
    expect(w[1].js()).toEqual([1, 1]); // dy/dy1 = 1

    // Now try with a tuple of argnums
    const df_both = grad(f, { argnums: [1, 0] });
    const [[dy0, dy1], dx] = df_both(x, y);
    expect(dx.js()).toEqual([4, 5]);
    expect(dy0.js()).toEqual([2, 3]);
    expect(dy1.js()).toEqual([1, 1]);
  });

  test("backprops through auto-broadcast", () => {
    const [dx, dy] = grad(([x, y]: [np.Array, np.Array]) => x.mul(y).sum())([
      np.array([[2], [4]]),
      np.array([4, 5, 6]),
    ]);
    expect(dx.js()).toEqual([[15], [15]]);
    expect(dy.js()).toEqual([6, 6, 6]);

    const [dx2, dy2] = grad(([x, y]: [np.Array, np.Array]) => x.add(y).sum())([
      np.array([[2], [4]]),
      np.array([4, 5, 6]),
    ]);
    expect(dx2.js()).toEqual([[3], [3]]);
    expect(dy2.js()).toEqual([2, 2, 2]);
  });

  test("backprop handles dense layer", () => {
    type Params = {
      w: np.Array;
      b: np.Array;
    };

    // x is of shape [batch, dim_in]
    // params.w is of shape [dim_in, dim_out]
    // params.b is of shape [dim_out]

    // const dense = (params: Params, x: np.Array) =>
    //   np
    //     .multiply(x.reshape([...x.shape, 1]), params.w)
    //     .sum(-2)
    //     .add(params.b);

    const dense = (params: Params, x: np.Array) =>
      np.dot(x, params.w).add(params.b);

    const loss = (params: Params, x: np.Array) =>
      nn.logSoftmax(dense(params, x)).slice([], 0).sum().mul(0.5);

    const params: Params = {
      w: np.array([
        [0.1, 0.2, -0.3, 0.0],
        [0.5, -0.1, 0.3, 0.4],
      ]),
      b: np.array([0, 0, 0, 0]),
    };
    const x = np.array([
      [0.1, 0.2],
      [0.2, 0.3],
    ]);

    // These numbers are checked for consistency with JAX.
    expect(loss(tree.ref(params), x.ref)).toBeAllclose(-1.3165712);

    // console.log(makeJaxpr(grad(loss))(params, x).jaxpr.toString());
    const grads = grad(loss)(params, x);
    expect(grads.w).toBeAllclose([
      [0.1095776, -0.03502218, -0.03585088, -0.03870453],
      [0.18276905, -0.05844341, -0.05986644, -0.06445917],
    ]);
    expect(grads.b).toBeAllclose([
      0.7319144, -0.2342123, -0.24015564, -0.25754642,
    ]);
  });

  test("hasAux returns aux and computes correct gradient", () => {
    const f = (x: np.Array): [np.Array, np.Array] => {
      const loss = x.ref.sum();
      const aux = x.mul(2);
      return [loss, aux];
    };

    const x = np.array([1, 2, 3]);
    const [gradient, aux] = grad(f, { hasAux: true })(x);

    expect(gradient).toBeAllclose([1, 1, 1]);
    expect(aux).toBeAllclose([2, 4, 6]);
  });

  test("hasAux handles pytree aux", () => {
    type Aux = { predictions: np.Array; squared: np.Array };
    const f = (x: np.Array): [np.Array, Aux] => {
      const loss = x.ref.sum();
      const aux = {
        predictions: x.ref.mul(2),
        squared: x.ref.mul(x),
      };
      return [loss, aux];
    };

    const x = np.array([1, 2, 3]);
    const [gradient, aux] = grad(f, { hasAux: true })(x);

    expect(gradient).toBeAllclose([1, 1, 1]);
    expect(aux.predictions).toBeAllclose([2, 4, 6]);
    expect(aux.squared).toBeAllclose([1, 4, 9]);
  });

  test("hasAux gradients match grad without aux", () => {
    const fWithAux = (x: np.Array): [np.Array, np.Array] => [
      x.ref.sum(),
      x.mul(2),
    ];
    const fWithoutAux = (x: np.Array) => x.sum();

    const x = np.array([1, 2, 3]);

    const [grad1] = grad(fWithAux, { hasAux: true })(x.ref);
    const grad2 = grad(fWithoutAux)(x);

    expect(grad1).toBeAllclose(grad2);
  });

  test("hasAux works with jit wrapper", () => {
    const f = jit((x: np.Array): [np.Array, np.Array] => [
      x.ref.sum(),
      x.mul(2),
    ]);

    const x = np.array([1, 2, 3]);
    const [gradient, aux] = grad(f, { hasAux: true })(x);

    expect(gradient).toBeAllclose([1, 1, 1]);
    expect(aux).toBeAllclose([2, 4, 6]);
  });

  test("hasAux throws on non-scalar output", () => {
    const f = (x: np.Array): [np.Array, np.Array] => [x.ref, x.mul(2)];
    const x = np.array([1, 2, 3]);
    expect(() => grad(f, { hasAux: true })(x)).toThrow("scalar");
  });

  test("hasAux throws on non-float dtype", () => {
    const f = (x: np.Array): [np.Array, np.Array] => [x.ref.sum(), x.mul(2)];
    const x = np.array([1, 2, 3], { dtype: np.int32 });
    expect(() => grad(f, { hasAux: true })(x)).toThrow("floating-point");
  });

  test("grad is not stopped in aux values", () => {
    const f = grad((x: np.Array) => {
      const [one, xsquare] = grad(
        (y: np.Array): [np.Array, np.Array] => [y.ref, y.ref.mul(y)],
        { hasAux: true },
      )(x);
      one.dispose();
      return xsquare;
    });
    expect(f(10)).toBeAllclose(20);
  });
});

suite("jax.valueAndGrad()", () => {
  test("returns value and gradient", () => {
    const f = (x: np.Array) => x.ref.mul(x).sum();
    const x = np.array([2, 3]);
    const [value, gradient] = valueAndGrad(f)(x);

    expect(value).toBeAllclose(13); // 4 + 9 = 13
    expect(gradient).toBeAllclose([4, 6]); // 2x = [4, 6]
  });

  test("hasAux returns value, gradient, and aux", () => {
    const f = (x: np.Array): [np.Array, np.Array] => {
      const loss = x.ref.sum();
      const aux = x.mul(2);
      return [loss, aux];
    };

    const x = np.array([1, 2, 3]);
    const [[value, aux], gradient] = valueAndGrad(f, { hasAux: true })(x);

    expect(value).toBeAllclose(6);
    expect(gradient).toBeAllclose([1, 1, 1]);
    expect(aux).toBeAllclose([2, 4, 6]);
  });

  test("hasAux matches valueAndGrad for value and gradient", () => {
    const fWithAux = (x: np.Array): [np.Array, np.Array] => [
      x.ref.sum(),
      x.mul(2),
    ];
    const fWithoutAux = (x: np.Array) => x.sum();

    const x = np.array([1, 2, 3]);

    const [[value1, aux1], grad1] = valueAndGrad(fWithAux, { hasAux: true })(
      x.ref,
    );
    const [value2, grad2] = valueAndGrad(fWithoutAux)(x);
    aux1.dispose();

    expect(value1).toBeAllclose(value2);
    expect(grad1).toBeAllclose(grad2);
  });
});

suite("jax.jit()", () => {
  test("works for a simple scalar function", () => {
    const f = (x: np.Array) => x.ref.mul(x.ref).mul(x); // d/dx (x^3) = 3x^2
    const f2 = jit(f);
    expect(f(np.array(2))).toBeAllclose(8);
    expect(f2(np.array(2))).toBeAllclose(8);
  });

  test("works with identity function", () => {
    const f = jit((x: np.Array) => x);
    const a = f(np.array(3));
    expect(a.js()).toEqual(3);
  });

  test("works with duplicate output", () => {
    const f = jit((x: np.Array) => [x.ref, x]);
    const [a, b] = f(np.array(3));
    expect(a.js()).toEqual(3);
    expect(b.js()).toEqual(3);
  });

  test("processes gather ops", () => {
    const f = jit((x: np.Array) =>
      x.slice(np.array([1, 3, 2, 0], { dtype: np.int32 })),
    );
    const a = f(np.array([10, 20, 30, 40]));
    expect(a.js()).toEqual([20, 40, 30, 10]);
  });

  test("supports staticArgnums", () => {
    const f = jit((x: np.Array, idx: number) => x.slice(idx), {
      staticArgnums: [1],
    });
    expect(f(np.arange(20), 0).js()).toEqual(0);
    expect(f(np.arange(20), 3).js()).toEqual(3);
    expect(f(np.array([30, 1, 20, 11]), 3).js()).toEqual(11);
  });

  test("jit-of-jit", () => {
    const f = jit((x: np.Array) => x.ref.mul(x));
    const g = jit((x: np.Array) => f(f(x)));
    expect(g(3)).toBeAllclose(81);
    expect(jit(jit(g))(3)).toBeAllclose(81);
  });

  test("jvp-of-jit", () => {
    const f = jit((x: np.Array) => x.ref.mul(x));
    expect(jvp(f, [3], [1])).toBeAllclose([9, 6]);
  });

  test("grad-of-jit", () => {
    const f = jit((x: np.Array) => x.ref.mul(x));

    expect(grad(f)(3)).toBeAllclose(6);
    expect(grad(f)(10)).toBeAllclose(20);
    expect(jvp(grad(f), [10], [1])).toBeAllclose([20, 2]);
    expect(grad(grad(f))(10)).toBeAllclose(2);
    expect(grad(jit(grad(f)))(10)).toBeAllclose(2);
  });

  test("vmap-of-jit", () => {
    const s = jit((x: np.Array) => x.sum());
    const ar = np.array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(s(ar.ref).js()).toEqual(21);
    expect(vmap(s)(ar.ref).js()).toEqual([6, 15]);
    expect(vmap(s, 1)(ar).js()).toEqual([5, 7, 9]);
  });
});
