import { numpy as np } from "@jax-js/jax";
import { expect, test } from "vitest";

import { treeMax, treeNorm, treeSum } from "../src/treeUtils";

test("treeSum: sums all elements across arrays", () => {
  const tr = [np.array([1, 2, 3]), np.array([4, 5])];
  const result = treeSum(tr);
  expect(result).toBeAllclose(15);
});

test("treeSum: handles empty tree", () => {
  const tr: np.Array[] = [];
  const result = treeSum(tr);
  expect(result).toBeAllclose(0);
});

test("treeSum: handles nested tree structure", () => {
  const tr = { a: np.array([1, 2]), b: { c: np.array([3, 4, 5]) } };
  const result = treeSum(tr);
  expect(result).toBeAllclose(15);
});

test("treeMax: finds max across arrays", () => {
  const tr = [np.array([1, 2, 3]), np.array([4, 5])];
  const result = treeMax(tr);
  expect(result).toBeAllclose(5);
});

test("treeMax: handles negative numbers", () => {
  const tr = [np.array([-5, -2, -10]), np.array([-1])];
  const result = treeMax(tr);
  expect(result).toBeAllclose(-1);
});

test("treeMax: handles empty tree", () => {
  const tr: np.Array[] = [];
  const result = treeMax(tr);
  expect(result).toBeAllclose(-Infinity);
});

test("treeNorm: L2 norm (default)", () => {
  const tr = [np.array([3]), np.array([4])];
  const result = treeNorm(tr);
  expect(result).toBeAllclose(5);
});

test("treeNorm: L2 norm squared", () => {
  const tr = [np.array([3]), np.array([4])];
  const result = treeNorm(tr, 2, true);
  expect(result).toBeAllclose(25);
});

test("treeNorm: L1 norm", () => {
  const tr = [np.array([-3, 4]), np.array([-5])];
  const result = treeNorm(tr, 1);
  expect(result).toBeAllclose(12);
});

test("treeNorm: inf norm", () => {
  const tr = [np.array([-3, 4]), np.array([-10])];
  const result = treeNorm(tr, "inf");
  expect(result).toBeAllclose(10);
});

test("treeNorm: throws on unsupported ord", () => {
  const tr = [np.array([1, 2, 3])];
  expect(() => treeNorm(tr, 3)).toThrow("Unsupported ord: 3");
});
