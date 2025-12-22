import { expect, suite, test } from "vitest";

import { computePath, parseEinsumExpression } from "./einsum";

suite("parseEinsumExpression()", () => {
  test("can parse explicit expressions", () => {
    let expr = parseEinsumExpression("ij,jk->ik", []);
    expect(expr.lhsIndices).toEqual([
      [0, 1],
      [1, 2],
    ]);
    expect(expr.rhsIndex).toEqual([0, 2]);

    expr = parseEinsumExpression("iii->i", []);
    expect(expr.lhsIndices).toEqual([[0, 0, 0]]);
    expect(expr.rhsIndex).toEqual([0]);

    expr = parseEinsumExpression("ji->ji", []);
    expect(expr.lhsIndices).toEqual([[1, 0]]);
    expect(expr.rhsIndex).toEqual([1, 0]);

    expr = parseEinsumExpression("ij->ji", []);
    expect(expr.lhsIndices).toEqual([[0, 1]]);
    expect(expr.rhsIndex).toEqual([1, 0]);

    expr = parseEinsumExpression("->", []);
    expect(expr.lhsIndices).toEqual([[]]);
    expect(expr.rhsIndex).toEqual([]);
  });

  test("can parse implicit expressions", () => {
    let expr = parseEinsumExpression("ij,jk", []);
    expect(expr.lhsIndices).toEqual([
      [0, 1],
      [1, 2],
    ]);
    expect(expr.rhsIndex).toEqual([0, 2]);

    expr = parseEinsumExpression("iii", []);
    expect(expr.lhsIndices).toEqual([[0, 0, 0]]);
    expect(expr.rhsIndex).toEqual([]);

    expr = parseEinsumExpression("ji", []);
    expect(expr.lhsIndices).toEqual([[1, 0]]);
    expect(expr.rhsIndex).toEqual([0, 1]);
  });

  test("parses expression with ...", () => {
    let expr = parseEinsumExpression("...ij,jk->...ik", [
      [2n, 3n, 4n],
      [4n, 5n],
    ]);
    expect(expr.lhsIndices).toEqual([
      [3, 0, 1],
      [1, 2],
    ]);
    expect(expr.rhsIndex).toEqual([3, 0, 2]);

    expr = parseEinsumExpression("...->", [[100n]]);
    expect(expr.lhsIndices).toEqual([[0]]);
    expect(expr.rhsIndex).toEqual([]);
  });

  test("in implicit mode ... goes in front", () => {
    const expr = parseEinsumExpression("...ij,j", [
      [2n, 3n, 4n],
      [4n, 5n],
    ]);
    expect(expr.lhsIndices).toEqual([[2, 0, 1], [1]]);
    expect(expr.rhsIndex).toEqual([2, 0]);
  });
});

suite("computePath()", () => {
  test("works for matmul", () => {
    const path = computePath(
      parseEinsumExpression("ij,jk->ik", [
        [25n, 30n],
        [30n, 40n],
      ]),
    );
    // Matmul has flops: 2 * M * N * K
    expect(path.approximateFlops).toBe(2n * 25n * 40n * 30n);
    expect(path.outputShape).toEqual([25n, 40n]);
    expect(path.path).toEqual([[0, 1]]);
  });

  test("computing 2D trace", () => {
    const path = computePath(parseEinsumExpression("ii->", [[50n, 50n]]));
    // Trace has flops: N
    expect(path.approximateFlops).toBe(50n);
    expect(path.outputShape).toEqual([]);
    expect(path.path).toEqual([]);
  });

  test("get diagonal of matrix", () => {
    const path = computePath(parseEinsumExpression("ii->i", [[60n, 60n]]));
    expect(path.approximateFlops).toBe(60n);
    expect(path.outputShape).toEqual([60n]);
    expect(path.path).toEqual([]);
  });

  test("diagonal dot product", () => {
    const path = computePath(
      parseEinsumExpression("ii,ii->", [
        [70n, 70n],
        [70n, 70n],
      ]),
    );
    // Diagonal dot product has flops: 2 * N
    expect(path.approximateFlops).toBe(2n * 70n);
    expect(path.outputShape).toEqual([]);
    expect(path.path).toEqual([[0, 1]]);
  });

  test("optimal path for 3 tensors", () => {
    const path = computePath(
      parseEinsumExpression("ij,jk,kl->il", [
        [10n, 20n],
        [20n, 30n],
        [30n, 40n],
      ]),
      "optimal",
    );
    // Optimal path is ((AB)C)
    expect(path.path).toEqual([
      [0, 1],
      [2, 3],
    ]);
    expect(path.approximateFlops).toBe(
      2n * (10n * 30n * 20n + 10n * 40n * 30n),
    );
  });
});
