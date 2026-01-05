import { jit, numpy as np, vjp, vjpWithAux } from "@jax-js/jax";
import { expect, suite, test } from "vitest";

suite("vjpWithAux", () => {
  test("returns aux and computes correct gradients", () => {
    const f = (x: np.Array): [np.Array, np.Array] => {
      const loss = x.ref.sum();
      const aux = x.mul(2);
      return [loss, aux];
    };

    const x = np.array([1, 2, 3]);
    const [loss, vjpFn, aux] = vjpWithAux(f, x);

    expect(loss).toBeAllclose(6);
    expect(aux).toBeAllclose([2, 4, 6]);

    const [grad] = vjpFn(np.ones([]));
    expect(grad).toBeAllclose([1, 1, 1]);

    vjpFn.dispose();
  });

  test("handles pytree aux", () => {
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
    const [loss, vjpFn, aux] = vjpWithAux(f, x);

    expect(loss).toBeAllclose(6);
    expect((aux as Aux).predictions).toBeAllclose([2, 4, 6]);
    expect((aux as Aux).squared).toBeAllclose([1, 4, 9]);

    vjpFn.dispose();
  });

  test("handles pytree main output", () => {
    type Main = { a: np.Array; b: np.Array };
    const f = (x: np.Array): [Main, np.Array] => {
      const main = { a: x.ref.sum(), b: x.ref.prod() };
      const aux = x.mul(2);
      return [main, aux];
    };

    const x = np.array([1, 2, 3]);
    const [main, vjpFn, _aux] = vjpWithAux(f, x);

    expect((main as Main).a).toBeAllclose(6);
    expect((main as Main).b).toBeAllclose(6);

    const [grad] = vjpFn({ a: np.ones([]), b: np.ones([]) });
    expect(grad).toBeAllclose([7, 4, 3]);

    vjpFn.dispose();
  });

  test("throws if function does not return tuple", () => {
    const f = (x: np.Array) => x.sum();
    const x = np.array([1, 2, 3]);
    expect(() => vjpWithAux(f as any, x)).toThrow(/tuple/);
  });

  test("gradients match vjp without aux", () => {
    const fWithAux = (x: np.Array): [np.Array, np.Array] => [
      x.ref.sum(),
      x.mul(2),
    ];
    const fWithoutAux = (x: np.Array) => x.sum();

    const x = np.array([1, 2, 3]);

    const [, vjpFn1] = vjpWithAux(fWithAux, x.ref);
    const [, vjpFn2] = vjp(fWithoutAux, x);

    const [grad1] = vjpFn1(np.ones([]));
    const [grad2] = vjpFn2(np.ones([]));

    expect(grad1).toBeAllclose(grad2);

    vjpFn1.dispose();
    // Note: vjpFn2 type doesn't expose dispose(), but cleanup happens on GC
  });

  test("works with jit wrapper", () => {
    const f = jit((x: np.Array): [np.Array, np.Array] => [
      x.ref.sum(),
      x.mul(2),
    ]);

    const x = np.array([1, 2, 3]);
    const [loss, vjpFn, aux] = vjpWithAux(f, x);

    expect(loss).toBeAllclose(6);
    expect(aux).toBeAllclose([2, 4, 6]);
    const [grad] = vjpFn(np.ones([]));
    expect(grad).toBeAllclose([1, 1, 1]);

    vjpFn.dispose();
  });

  test("works inside jit", () => {
    const inner = (x: np.Array): [np.Array, np.Array] => [
      x.ref.sum(),
      x.mul(2),
    ];

    const outer = jit((x: np.Array) => {
      const [, vjpFn, aux] = vjpWithAux(inner, x);
      const [grad] = vjpFn(np.ones([]));
      vjpFn.dispose();
      return [grad, aux];
    });

    const x = np.array([1, 2, 3]);
    const [grad, aux] = outer(x) as [np.Array, np.Array];

    expect(grad).toBeAllclose([1, 1, 1]);
    expect(aux).toBeAllclose([2, 4, 6]);
  });
});
