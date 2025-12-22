import { expect, suite, test } from "vitest";

import { computePath, parseEinsumExpression } from "./einsum";

suite("parseEinsumExpression()", () => {
  test("can parse expressions", () => {
    let expr = parseEinsumExpression("ij,jk->ik");
    expect(expr.lhsIndices).toEqual([
      [0, 1],
      [1, 2],
    ]);
    expect(expr.rhsIndex).toEqual([0, 2]);

    expr = parseEinsumExpression("ij,jk");
    expect(expr.lhsIndices).toEqual([
      [0, 1],
      [1, 2],
    ]);
    expect(expr.rhsIndex).toEqual([0, 2]);

    expr = parseEinsumExpression("iii");
    expect(expr.lhsIndices).toEqual([[0, 0, 0]]);
    expect(expr.rhsIndex).toEqual([]);

    expr = parseEinsumExpression("iii->i");
    expect(expr.lhsIndices).toEqual([[0, 0, 0]]);
    expect(expr.rhsIndex).toEqual([0]);

    expr = parseEinsumExpression("ji");
    expect(expr.lhsIndices).toEqual([[1, 0]]);
    expect(expr.rhsIndex).toEqual([0, 1]);

    expr = parseEinsumExpression("ji->ji");
    expect(expr.lhsIndices).toEqual([[1, 0]]);
    expect(expr.rhsIndex).toEqual([1, 0]);

    expr = parseEinsumExpression("ij->ji");
    expect(expr.lhsIndices).toEqual([[0, 1]]);
    expect(expr.rhsIndex).toEqual([1, 0]);

    expr = parseEinsumExpression("->");
    expect(expr.lhsIndices).toEqual([[]]);
    expect(expr.rhsIndex).toEqual([]);
  });

  test("throws on invalid einsum expressions", () => {
    expect(() => parseEinsumExpression("->i")).toThrow(Error);
    expect(() => parseEinsumExpression("i->ij")).toThrow(Error);
  });
});

suite("computePath()", () => {
  test("works for matmul", () => {
    const path = computePath({
      ...parseEinsumExpression("ij,jk->ik"),
      shapes: [
        [25n, 30n],
        [30n, 40n],
      ],
    });
    // Matmul has flops: 2 * M * N * K
    expect(path.approximateFlops).toBe(2n * 25n * 40n * 30n);
    expect(path.outputShape).toEqual([25n, 40n]);
    expect(path.path).toEqual([[0, 1]]);
  });

  test("computing 2D trace", () => {
    const path = computePath({
      ...parseEinsumExpression("ii->"),
      shapes: [[50n, 50n]],
    });
    // Trace has flops: N
    expect(path.approximateFlops).toBe(50n);
    expect(path.outputShape).toEqual([]);
    expect(path.path).toEqual([]);
  });

  test("get diagonal of matrix", () => {
    const path = computePath({
      ...parseEinsumExpression("ii->i"),
      shapes: [[60n, 60n]],
    });
    expect(path.approximateFlops).toBe(60n);
    expect(path.outputShape).toEqual([60n]);
    expect(path.path).toEqual([[0]]);
  });

  test("diagonal dot product", () => {
    const path = computePath({
      ...parseEinsumExpression("ii,ii->"),
      shapes: [
        [70n, 70n],
        [70n, 70n],
      ],
    });
    // Diagonal dot product has flops: 2 * N
    expect(path.approximateFlops).toBe(2n * 70n);
    expect(path.outputShape).toEqual([]);
    expect(path.path).toEqual([[0, 1]]);
  });
});
