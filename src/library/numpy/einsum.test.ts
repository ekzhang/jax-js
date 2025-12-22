import { expect, suite, test } from "vitest";

import { computeEinsumPath, parseEinsumExpression } from "./einsum";

suite("parseEinsumExpression()", () => {
  test("can parse explicit expressions", () => {
    let expr = parseEinsumExpression("ij,jk->ik", [[], []]);
    expect(expr.lhsIndices).toEqual([
      [0, 1],
      [1, 2],
    ]);
    expect(expr.rhsIndex).toEqual([0, 2]);

    expr = parseEinsumExpression("iii->i", [[]]);
    expect(expr.lhsIndices).toEqual([[0, 0, 0]]);
    expect(expr.rhsIndex).toEqual([0]);

    expr = parseEinsumExpression("ji->ji", [[]]);
    expect(expr.lhsIndices).toEqual([[1, 0]]);
    expect(expr.rhsIndex).toEqual([1, 0]);

    expr = parseEinsumExpression("ij->ji", [[]]);
    expect(expr.lhsIndices).toEqual([[0, 1]]);
    expect(expr.rhsIndex).toEqual([1, 0]);

    expr = parseEinsumExpression("->", [[]]);
    expect(expr.lhsIndices).toEqual([[]]);
    expect(expr.rhsIndex).toEqual([]);
  });

  test("can parse implicit expressions", () => {
    let expr = parseEinsumExpression("ij,jk", [[], []]);
    expect(expr.lhsIndices).toEqual([
      [0, 1],
      [1, 2],
    ]);
    expect(expr.rhsIndex).toEqual([0, 2]);

    expr = parseEinsumExpression("iii", [[]]);
    expect(expr.lhsIndices).toEqual([[0, 0, 0]]);
    expect(expr.rhsIndex).toEqual([]);

    expr = parseEinsumExpression("ji", [[]]);
    expect(expr.lhsIndices).toEqual([[1, 0]]);
    expect(expr.rhsIndex).toEqual([0, 1]);
  });

  test("parses expression with ...", () => {
    let expr = parseEinsumExpression("...ij,jk->...ik", [
      [2, 3, 4],
      [4, 5],
    ]);
    expect(expr.lhsIndices).toEqual([
      [3, 0, 1],
      [1, 2],
    ]);
    expect(expr.rhsIndex).toEqual([3, 0, 2]);

    expr = parseEinsumExpression("...->", [[100]]);
    expect(expr.lhsIndices).toEqual([[0]]);
    expect(expr.rhsIndex).toEqual([]);
  });

  test("in implicit mode ... goes in front", () => {
    const expr = parseEinsumExpression("...ij,j", [
      [2, 3, 4],
      [4, 5],
    ]);
    expect(expr.lhsIndices).toEqual([[2, 0, 1], [1]]);
    expect(expr.rhsIndex).toEqual([2, 0]);
  });

  test("supports numpy broadcasting", () => {
    const expr = parseEinsumExpression("..., ..., ...", [
      [],
      [3, 4],
      [5, 3, 4],
    ]);
    expect(expr.lhsIndices).toEqual([[], [1, 2], [0, 1, 2]]);
    expect(expr.rhsIndex).toEqual([0, 1, 2]);
  });
});

suite("computePath()", () => {
  test("works for matmul", () => {
    const path = computeEinsumPath(
      parseEinsumExpression("ij,jk->ik", [
        [25, 30],
        [30, 40],
      ]),
    );
    // Matmul has flops: 2 * M * N * K
    expect(path.approximateFlops).toBe(2n * 25n * 40n * 30n);
    expect(path.outputShape).toEqual([25, 40]);
    expect(path.path).toEqual([[0, 1]]);
  });

  test("computing 2D trace", () => {
    const path = computeEinsumPath(parseEinsumExpression("ii->", [[50, 50]]));
    // Trace has flops: N
    expect(path.approximateFlops).toBe(50n);
    expect(path.outputShape).toEqual([]);
    expect(path.path).toEqual([]);
  });

  test("get diagonal of matrix", () => {
    const path = computeEinsumPath(parseEinsumExpression("ii->i", [[60, 60]]));
    expect(path.approximateFlops).toBe(60n);
    expect(path.outputShape).toEqual([60]);
    expect(path.path).toEqual([]);
  });

  test("diagonal dot product", () => {
    const path = computeEinsumPath(
      parseEinsumExpression("ii,ii->", [
        [70, 70],
        [70, 70],
      ]),
    );
    // Diagonal dot product has flops: 2 * N
    expect(path.approximateFlops).toBe(2n * 70n);
    expect(path.outputShape).toEqual([]);
    expect(path.path).toEqual([[0, 1]]);
  });

  test("optimal path for 3 tensors", () => {
    const path = computeEinsumPath(
      parseEinsumExpression("ij,jk,kl->il", [
        [10, 20],
        [20, 30],
        [30, 40],
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

  test("optimal path for 5 tensors (perf)", () => {
    const startTime = performance.now();
    const path = computeEinsumPath(
      parseEinsumExpression("ab,bc,cd,de,ef->af", [
        [5, 10],
        [10, 15],
        [15, 20],
        [20, 25],
        [25, 30],
      ]),
      "optimal",
    );
    // Check that it runs within reasonable time.
    const endTime = performance.now();
    expect(endTime - startTime).toBeLessThan(100);
    expect(path.path).toEqual([
      [0, 1],
      [2, 5],
      [3, 6],
      [4, 7],
    ]);
  });
});
