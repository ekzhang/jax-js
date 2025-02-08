import { suite, test, expect } from "vitest";
import { JsTreeDef, NodeType, flatten, unflatten } from "./tree";

suite("JsTreeDef.toString()", () => {
  test("should return '*' for a leaf", () => {
    expect(JsTreeDef.leaf.toString()).toBe("JsTreeDef(*)");
  });

  test("should return a string representation for an array node", () => {
    // Create a treedef representing an array of two leaves.
    const arrTreedef = new JsTreeDef(NodeType.Array, null, [
      JsTreeDef.leaf,
      JsTreeDef.leaf,
    ]);
    expect(arrTreedef.toString()).toBe("JsTreeDef([*, *])");
  });

  test("should return a string representation for an object node", () => {
    // Create an object treedef with keys "a" and "b" mapping to leaves.
    const objTreedef = new JsTreeDef(
      NodeType.Object,
      ["a", "b"],
      [JsTreeDef.leaf, JsTreeDef.leaf]
    );
    expect(objTreedef.toString()).toBe("JsTreeDef({a: *, b: *})");
  });

  test("should properly quote keys that are not valid identifiers", () => {
    // Keys with spaces or special characters should be quoted.
    const objTreedef = new JsTreeDef(
      NodeType.Object,
      ["not-valid", "validKey"],
      [JsTreeDef.leaf, JsTreeDef.leaf]
    );
    // "not-valid" is not a valid identifier, so it should be quoted (using JSON.stringify).
    const expected = `JsTreeDef({${JSON.stringify("not-valid")}: *, validKey: *})`;
    expect(objTreedef.toString()).toBe(expected);
  });
});

suite("flatten()", () => {
  test("should flatten a leaf value", () => {
    const [leaves, treedef] = flatten(42);
    expect(leaves).toEqual([42]);
    // A leaf treedef always returns '*' for non-root calls.
    expect(treedef.toString()).toBe("JsTreeDef(*)");
  });

  test("should flatten an array", () => {
    const input = [1, 2, 3];
    const [leaves, treedef] = flatten(input);
    expect(leaves).toEqual([1, 2, 3]);
    // treedef should be an array type with three children (all leaves)
    expect(treedef.nodeType).toBe(NodeType.Array);
    expect(treedef.childTreedefs.length).toBe(3);
    for (const child of treedef.childTreedefs) {
      expect(child.nodeType).toBe(NodeType.Leaf);
    }
    expect(treedef.toString()).toBe("JsTreeDef([*, *, *])");
  });

  test("should flatten an object", () => {
    const input = { a: 1, b: [2, 3] };
    const [leaves, treedef] = flatten(input);
    // The order of keys is determined by Object.entries order.
    // Assuming the natural order: "a", then "b".
    expect(leaves).toEqual([1, 2, 3]);

    expect(treedef.nodeType).toBe(NodeType.Object);
    expect(treedef.nodeMetadata).toEqual(["a", "b"]);
    expect(treedef.childTreedefs.length).toBe(2);

    // Check the first child: a leaf.
    expect(treedef.childTreedefs[0].nodeType).toBe(NodeType.Leaf);
    // Check the second child: an array with two leaves.
    const secondChild = treedef.childTreedefs[1];
    expect(secondChild.nodeType).toBe(NodeType.Array);
    expect(secondChild.childTreedefs.length).toBe(2);
    expect(secondChild.childTreedefs[0].nodeType).toBe(NodeType.Leaf);
    expect(secondChild.childTreedefs[1].nodeType).toBe(NodeType.Leaf);

    // We can also check the string representation.
    expect(treedef.toString()).toBe("JsTreeDef({a: *, b: [*, *]})");
  });
});

suite("unflatten()", () => {
  test("should reconstruct a leaf value", () => {
    const value = "test";
    const [leaves, treedef] = flatten(value);
    const reconstructed = unflatten(treedef, leaves);
    expect(reconstructed).toBe(value);
  });

  test("should reconstruct an array", () => {
    const input = [10, 20, 30];
    const [leaves, treedef] = flatten(input);
    const reconstructed = unflatten(treedef, leaves);
    expect(reconstructed).toEqual(input);
  });

  test("should reconstruct an object", () => {
    const input = { x: 100, y: [200, 300] };
    const [leaves, treedef] = flatten(input);
    const reconstructed = unflatten(treedef, leaves);
    expect(reconstructed).toEqual(input);
  });

  test("should reconstruct a nested tree", () => {
    const input = { a: [1, { b: 2 }], c: "hello" };
    const [leaves, treedef] = flatten<number | string>(input);
    const reconstructed = unflatten(treedef, leaves);
    expect(reconstructed).toEqual(input);
  });

  test("should throw an error when there are too few leaves", () => {
    const input = [1, 2, 3];
    const [leaves, treedef] = flatten(input);
    // Remove one leaf to simulate missing leaves.
    const insufficientLeaves = leaves.slice(0, leaves.length - 1);
    expect(() => {
      unflatten(treedef, insufficientLeaves);
    }).toThrow("Ran out of leaves while unflattening JsTree");
  });
});

suite("JsTreeDef.equals()", () => {
  test("marks identical trees as equal", () => {
    const tree1 = { a: [1, 2], b: { c: 3 } };
    const tree2 = { a: [-40, 222], b: { c: 31 } };

    const [, treedef1] = flatten(tree1);
    const [, treedef2] = flatten(tree2);

    expect(treedef1.equals(treedef2)).toBe(true);
  });

  test("marks different shapes or keys as not equal", () => {
    const tree1 = { a: [1, 2], b: { c: 3 } };
    const tree2 = { a: [1, 2, 3], b: { c: 3 } };
    const tree3 = { a: [1, 2], d: { c: 3 } };

    const [, treedef1] = flatten(tree1);
    const [, treedef2] = flatten(tree2);
    const [, treedef3] = flatten(tree3);

    expect(treedef1.equals(treedef2)).toBe(false);
    expect(treedef1.equals(treedef3)).toBe(false);
  });
});
