import { grad, gradWithAux, jit, numpy as np, valueAndGrad, valueAndGradWithAux } from "@jax-js/jax";
import { expect, suite, test } from "vitest";

suite("gradWithAux", () => {
  test("returns aux and computes correct gradient", () => {
    const f = (x: np.Array): [np.Array, np.Array] => {
      const loss = x.ref.sum();
      const aux = x.mul(2);
      return [loss, aux];
    };

    const x = np.array([1, 2, 3]);
    const [grad, aux] = gradWithAux(f)(x);

    expect(grad).toBeAllclose([1, 1, 1]);
    expect(aux).toBeAllclose([2, 4, 6]);
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
    const [grad, aux] = gradWithAux(f)(x);

    expect(grad).toBeAllclose([1, 1, 1]);
    expect((aux as Aux).predictions).toBeAllclose([2, 4, 6]);
    expect((aux as Aux).squared).toBeAllclose([1, 4, 9]);
  });

  test("gradients match grad without aux", () => {
    const fWithAux = (x: np.Array): [np.Array, np.Array] => [
      x.ref.sum(),
      x.mul(2),
    ];
    const fWithoutAux = (x: np.Array) => x.sum();

    const x = np.array([1, 2, 3]);

    const [grad1] = gradWithAux(fWithAux)(x.ref);
    const grad2 = grad(fWithoutAux)(x);

    expect(grad1).toBeAllclose(grad2);
  });

  test("works with jit wrapper", () => {
    const f = jit(
      (x: np.Array): [np.Array, np.Array] => [x.ref.sum(), x.mul(2)],
    );

    const x = np.array([1, 2, 3]);
    const [grad, aux] = gradWithAux(f)(x);

    expect(grad).toBeAllclose([1, 1, 1]);
    expect(aux).toBeAllclose([2, 4, 6]);
  });

  test("throws on non-scalar output", () => {
    const f = (x: np.Array): [np.Array, np.Array] => [x.ref, x.mul(2)];
    const x = np.array([1, 2, 3]);
    expect(() => gradWithAux(f)(x)).toThrow("scalar");
  });

  test("throws on non-float dtype", () => {
    const f = (x: np.Array): [np.Array, np.Array] => [x.ref.sum(), x.mul(2)];
    const x = np.array([1, 2, 3], { dtype: np.int32 });
    expect(() => gradWithAux(f)(x)).toThrow("floating-point");
  });
});

suite("valueAndGradWithAux", () => {
  test("returns value, gradient, and aux", () => {
    const f = (x: np.Array): [np.Array, np.Array] => {
      const loss = x.ref.sum();
      const aux = x.mul(2);
      return [loss, aux];
    };

    const x = np.array([1, 2, 3]);
    const [value, grad, aux] = valueAndGradWithAux(f)(x);

    expect(value).toBeAllclose(6);
    expect(grad).toBeAllclose([1, 1, 1]);
    expect(aux).toBeAllclose([2, 4, 6]);
  });

  test("matches valueAndGrad for value and gradient", () => {
    const fWithAux = (x: np.Array): [np.Array, np.Array] => [
      x.ref.sum(),
      x.mul(2),
    ];
    const fWithoutAux = (x: np.Array) => x.sum();

    const x = np.array([1, 2, 3]);

    const [value1, grad1] = valueAndGradWithAux(fWithAux)(x.ref);
    const [value2, grad2] = valueAndGrad(fWithoutAux)(x);

    expect(value1).toBeAllclose(value2);
    expect(grad1).toBeAllclose(grad2);
  });
});
