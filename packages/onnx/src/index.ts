/**
 * ONNX model loader for jax-js.
 *
 * Parse ONNX models and convert them to jax-js functions.
 */

import { fromBinary } from "@bufbuild/protobuf";
import { numpy as np } from "@jax-js/jax";
import {
  AttributeProto_AttributeType,
  ModelProtoSchema,
  type NodeProto,
  type TensorProto,
} from "onnx-buf";

import * as onnxOps from "./ops";
import { tensorToArray } from "./tensor";

/** Parse attributes from an ONNX node into a plain object. */
function parseAttributes(node: NodeProto): Record<string, any> {
  const attrs: Record<string, any> = {};
  for (const attr of node.attribute) {
    switch (attr.type) {
      case AttributeProto_AttributeType.FLOAT:
        attrs[attr.name] = attr.f;
        break;
      case AttributeProto_AttributeType.INT:
        attrs[attr.name] = Number(attr.i);
        break;
      case AttributeProto_AttributeType.STRING:
        attrs[attr.name] = new TextDecoder().decode(attr.s);
        break;
      case AttributeProto_AttributeType.TENSOR:
        attrs[attr.name] = attr.t;
        break;
      case AttributeProto_AttributeType.FLOATS:
        attrs[attr.name] = attr.floats;
        break;
      case AttributeProto_AttributeType.INTS:
        attrs[attr.name] = attr.ints.map(Number);
        break;
      case AttributeProto_AttributeType.STRINGS:
        attrs[attr.name] = attr.strings.map((s) => new TextDecoder().decode(s));
        break;
      // Skip unsupported attribute types (GRAPH, GRAPHS, TENSORS, etc.)
    }
  }
  return attrs;
}

/**
 * Parse all initializers (constant weights) from an ONNX graph.
 */
function parseInitializers(initializers: TensorProto[]): Map<string, np.Array> {
  const map = new Map<string, np.Array>();
  for (const tensor of initializers) {
    map.set(tensor.name, tensorToArray(tensor));
  }
  return map;
}

/**
 * Execute a single ONNX node.
 */
function executeNode(node: NodeProto, vars: Map<string, np.Array>): np.Array[] {
  const opType = node.opType;
  const handler = (onnxOps as Record<string, any>)[opType];
  if (!handler) throw new Error(`Unsupported ONNX operation: ${opType}`);

  const inputs: (np.Array | undefined)[] = [];
  for (const name of node.input) {
    if (name === "") {
      inputs.push(undefined); // Optional input not provided
      continue;
    }
    const arr = vars.get(name);
    if (!arr) {
      throw new Error(
        `Missing input '${name}' for node '${node.name}' (op: ${opType})`,
      );
    }
    inputs.push(arr.ref);
  }
  const attrs = parseAttributes(node);
  return handler(inputs, attrs);
}

/**
 * Parse an ONNX model and return a jax-js function that evaluates it.
 *
 * The returned function takes input tensors and returns output tensors.
 * Input tensors are consumed (their reference count decremented).
 * Initializers (model weights) are cached and reused across calls.
 *
 * @param modelBytes - The raw bytes of the ONNX model file
 * @returns A function that executes the model
 *
 * @example
 * ```ts
 * import { modelAsJaxFunction } from "@jax-js/onnx";
 * import { numpy as np } from "@jax-js/jax";
 *
 * const modelBytes = await fetch("model.onnx").then(r => r.arrayBuffer());
 * const model = modelAsJaxFunction(new Uint8Array(modelBytes));
 *
 * const input = np.ones([1, 3, 224, 224]);
 * const output = model(input);
 * ```
 */
export function modelAsJaxFunction(
  modelBytes: Uint8Array,
): (inputs: Record<string, np.Array>) => Record<string, np.Array> {
  const model = fromBinary(ModelProtoSchema, modelBytes);
  const graph = model.graph;
  if (!graph) {
    throw new Error("ONNX model has no graph");
  }

  // Extract initializers (weights/biases) as jax arrays.
  // These are cached and reused across calls.
  const initializers = parseInitializers(graph.initializer);

  const inputNames: string[] = [];
  for (const input of graph.input) {
    if (!initializers.has(input.name)) {
      inputNames.push(input.name);
    }
  }
  const outputNames = graph.output.map((o) => o.name);

  return function (inputs: Record<string, np.Array>): Record<string, np.Array> {
    if (Object.keys(inputs).length !== inputNames.length) {
      throw new Error(
        `Expected ${inputNames.length} inputs, but got ${Object.keys(inputs).length}`,
      );
    }
    for (const name of inputNames) {
      if (!Object.hasOwn(inputs, name))
        throw new Error(`Missing input '${name}'`);
    }

    const vars = new Map<string, np.Array>();

    try {
      for (const [name, arr] of initializers) vars.set(name, arr.ref);
      for (const name of inputNames) vars.set(name, inputs[name]);

      for (const node of graph.node) {
        const results = executeNode(node, vars);
        for (const [i, name] of node.output.entries())
          vars.set(name, results[i]);
      }

      const outputs: Record<string, np.Array> = {};
      for (const name of outputNames) {
        const arr = vars.get(name);
        if (!arr) throw new Error(`Missing output '${name}'`);
        outputs[name] = arr;
      }
      for (const name of outputNames) vars.delete(name); // Prevent disposing outputs
      return outputs;
    } finally {
      // Clean up, dispose of all values that weren't returned.
      for (const ar of vars.values()) ar.dispose();
    }
  };
}
