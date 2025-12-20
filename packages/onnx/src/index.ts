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

import { type Attrs, onnxOps } from "./ops.js";
import { tensorToArray } from "./tensor.js";

export { tensorToArray, onnxDtypeToJax } from "./tensor.js";
export { onnxOps } from "./ops.js";

/**
 * Parse attributes from an ONNX node into a plain object.
 */
function parseAttributes(node: NodeProto): Attrs {
  const attrs: Attrs = {};
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
function executeNode(
  node: NodeProto,
  values: Map<string, np.Array>,
): np.Array[] {
  const opType = node.opType;
  const handler = onnxOps[opType];

  if (!handler) {
    throw new Error(`Unsupported ONNX operation: ${opType}`);
  }

  // Gather input arrays (filter out empty string inputs which are optional)
  const inputs: np.Array[] = [];
  for (const name of node.input) {
    if (name === "") continue; // Optional input not provided
    const arr = values.get(name);
    if (!arr) {
      throw new Error(
        `Missing input '${name}' for node '${node.name}' (op: ${opType})`,
      );
    }
    inputs.push(arr);
  }

  // Parse attributes
  const attrs = parseAttributes(node);

  // Execute the operation
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
  // Parse the ONNX model
  const model = fromBinary(ModelProtoSchema, modelBytes);
  const graph = model.graph;

  if (!graph) {
    throw new Error("ONNX model has no graph");
  }

  // Extract initializers (weights/biases) as jax arrays
  // These are cached and reused across calls
  const initializers = parseInitializers(graph.initializer);

  // Build list of actual input names (excluding initializers)
  const inputNames: string[] = [];
  for (const input of graph.input) {
    if (!initializers.has(input.name)) {
      inputNames.push(input.name);
    }
  }

  // Build list of output names
  const outputNames = graph.output.map((o) => o.name);

  // Return the execution function
  return function (inputs: Record<string, np.Array>): Record<string, np.Array> {
    if (Object.keys(inputs).length !== inputNames.length) {
      throw new Error(
        `Expected ${inputNames.length} inputs, but got ${
          Object.keys(inputs).length
        }`,
      );
    }
    for (const name of inputNames) {
      if (!(name in inputs)) {
        throw new Error(`Missing input '${name}'`);
      }
    }

    // Map to hold all intermediate values
    const values = new Map<string, np.Array>();

    // Add initializers (with ref since they're reused)
    for (const [name, arr] of initializers) {
      values.set(name, arr.ref);
    }

    // Add user inputs (consumed, no ref)
    for (const inputName of inputNames) {
      values.set(inputName, inputs[inputName]);
    }

    // Execute nodes in topological order (ONNX guarantees this order)
    for (const node of graph.node) {
      const results = executeNode(node, values);

      // Store outputs
      for (let i = 0; i < node.output.length; i++) {
        const outputName = node.output[i];
        if (outputName !== "" && results[i]) {
          values.set(outputName, results[i]);
        }
      }
    }

    // Gather outputs
    const outputs: Record<string, np.Array> = {};
    for (const name of outputNames) {
      const arr = values.get(name);
      if (!arr) {
        throw new Error(`Missing output '${name}'`);
      }
      outputs[name] = arr;
    }
    return outputs;
  };
}
