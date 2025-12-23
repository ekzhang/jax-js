// ONNX model loader for jax-js.
//
// Parse ONNX models and convert them to jax-js functions.

import { fromBinary } from "@bufbuild/protobuf";
import { numpy as np } from "@jax-js/jax";
import {
  AttributeProto_AttributeType,
  ModelProto,
  ModelProtoSchema,
  type NodeProto,
  type TensorProto,
} from "onnx-buf";

import * as onnxOps from "./ops";
import { tensorToArray } from "./tensor";

/**
 * Loads an ONNX model (`.onnx` file) and provides a jax-js function that
 * evaluates it.
 *
 * The returned function takes input tensors and returns output tensors.
 * Input tensors are consumed (their reference count decremented).
 * Initializers (model weights, data) are cached and reused across calls.
 *
 * @example
 * ```ts
 * import { ONNXModel } from "@jax-js/onnx";
 * import { numpy as np } from "@jax-js/jax";
 *
 * const modelBytes = await fetch("./model.onnx").then((r) => r.bytes());
 * const model = new ONNXModel(modelBytes);
 *
 * const input = np.ones([1, 3, 224, 224]);
 * const { output } = model.run({ input });
 * ```
 */
export class ONNXModel {
  /** The parsed model as a Protobuf object. */
  readonly model: ModelProto;

  /**
   * @function
   * Run a forward pass on the model. This function is bound to `this`, so you
   * don't need to create a separate closure to pass it to transformations such
   * as `jit()` and `grad()`.
   */
  readonly run: (
    inputs: Record<string, np.Array>,
    options?: ONNXRunOptions,
  ) => Record<string, np.Array>;

  /** Cache of initializers (data / weights), needed for `run()` calls. */
  #initializers: Map<string, np.Array>;

  /** Load a new model from binary contents of an `.onnx` file. */
  constructor(modelBytes: Uint8Array<ArrayBuffer>) {
    this.model = fromBinary(ModelProtoSchema, modelBytes);
    if (!this.model.graph) {
      throw new Error("ONNX model has no graph");
    }
    // Extract initializers (weights/biases) as jax arrays.
    // These are cached and reused across calls.
    this.#initializers = parseInitializers(this.model.graph.initializer);
    if (this.model.graph.sparseInitializer.length > 0) {
      throw new Error("ONNX sparse initializers are not supported");
    }
    if (this.model.functions.length > 0) {
      throw new Error(
        "ONNX model custom functions are not supported: " +
          this.model.functions.map((f) => f.name).join(", "),
      );
    }
    this.run = modelAsJaxFunction(this.model, this.#initializers);
  }

  /**
   * Dispose of this model and free model weights.
   *
   * After disposing, `run()` should not be called anymore, it will not be able
   * to find the missing variables.
   */
  dispose(): void {
    if (!this.#initializers) return;
    for (const arr of this.#initializers.values()) arr.dispose();
    this.#initializers.clear();
  }
}

/** Parse attributes from an ONNX node into a plain object. */
function parseAttributes(node: NodeProto): Record<string, any> {
  const attrs: Record<string, any> = {};
  for (const attr of node.attribute) {
    switch (attr.type) {
      case AttributeProto_AttributeType.FLOAT:
        attrs[attr.name] = attr.f;
        break;
      case AttributeProto_AttributeType.FLOATS:
        attrs[attr.name] = attr.floats;
        break;
      case AttributeProto_AttributeType.INT:
        attrs[attr.name] = Number(attr.i);
        break;
      case AttributeProto_AttributeType.INTS:
        attrs[attr.name] = attr.ints.map(Number);
        break;
      case AttributeProto_AttributeType.STRING:
        attrs[attr.name] = new TextDecoder().decode(attr.s);
        break;
      case AttributeProto_AttributeType.STRINGS:
        attrs[attr.name] = attr.strings.map((s) => new TextDecoder().decode(s));
        break;
      case AttributeProto_AttributeType.TENSOR:
        attrs[attr.name] = attr.t;
        break;
      case AttributeProto_AttributeType.TENSORS:
        attrs[attr.name] = attr.tensors;
        break;
      case AttributeProto_AttributeType.SPARSE_TENSOR:
      case AttributeProto_AttributeType.SPARSE_TENSORS:
        throw new Error("ONNX sparse tensor attributes are not supported");

      // Skip other attribute types for now.
      case AttributeProto_AttributeType.GRAPH:
      case AttributeProto_AttributeType.GRAPHS:
      case AttributeProto_AttributeType.TYPE_PROTO:
      case AttributeProto_AttributeType.TYPE_PROTOS:
      default:
    }
  }
  return attrs;
}

/** Parse all initializers (constant weights) from an ONNX graph. */
function parseInitializers(initializers: TensorProto[]): Map<string, np.Array> {
  const map = new Map<string, np.Array>();
  for (const tensor of initializers) {
    map.set(tensor.name, tensorToArray(tensor));
  }
  return map;
}

/** Execute a single ONNX node. */
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

/** Options for running an ONNX model. */
export interface ONNXRunOptions {
  /** Print out names, input and output shapes when running each operation. */
  verbose?: boolean;

  /**
   * Tensors for which to log debug information during execution. When provided,
   * logs statistics (min, max, mean, variance) for intermediate tensors
   * matching these names.
   *
   * This is an unstable API and may be removed without notice.
   */
  debugStats?: string[];
}

function logDebugStats(name: string, arr: np.Array): void {
  arr = arr.astype(np.float32);

  const min = np.min(arr.ref).js() as number;
  const max = np.max(arr.ref).js() as number;
  const mean = np.mean(arr.ref).js() as number;
  const variance = np.var_(arr.ref).js() as number;

  const shortName = name.split("/").pop() || name;
  console.log(`${shortName}
  full: ${name}
  shape: [${arr.shape.join(", ")}], dtype: ${arr.dtype}
  min: ${min.toFixed(6)}, max: ${max.toFixed(6)}
  mean: ${mean.toFixed(6)}, var: ${variance.toFixed(6)}`);
  console.log();

  arr.dispose();
}

function modelAsJaxFunction(
  model: ModelProto,
  initializers: Map<string, np.Array> = new Map(),
): typeof ONNXModel.prototype.run {
  const graph = model.graph;
  if (!graph) {
    throw new Error("ONNX model has no graph");
  }

  const inputNames: string[] = [];
  for (const input of graph.input) {
    if (!initializers.has(input.name)) {
      if (inputNames.includes(input.name)) {
        throw new Error(`ONNX model has duplicate input '${input.name}'`);
      }
      inputNames.push(input.name);
    }
  }
  const outputNames = graph.output.map((o) => o.name);
  const numInitializers = initializers.size;

  return function (
    inputs: Record<string, np.Array>,
    options?: ONNXRunOptions,
  ): Record<string, np.Array> {
    for (const name of inputNames) {
      if (!Object.hasOwn(inputs, name))
        throw new Error(`Missing input '${name}'`);
    }
    if (Object.keys(inputs).length !== inputNames.length) {
      throw new Error(
        `Expected inputs ${JSON.stringify(inputNames)}, but got extra inputs ${JSON.stringify(Object.keys(inputs))}`,
      );
    }
    if (initializers.size !== numInitializers)
      throw new Error("Model has already been disposed");

    const debugStats = new Set(options?.debugStats ?? []);
    const verbose = options?.verbose ?? false;

    const vars = new Map<string, np.Array>();
    try {
      for (const [name, arr] of initializers) vars.set(name, arr.ref);
      for (const name of inputNames) vars.set(name, inputs[name]);

      for (const node of graph.node) {
        const results = executeNode(node, vars);

        if (verbose) {
          const inputs = node.input.map((n) => {
            if (!n) return "_";
            const arr = vars.get(n)!;
            return `${n} ${arr.aval}`;
          });
          const outputs = results.map(
            (arr, i) => `${node.output[i]} ${arr.aval}`,
          );
          console.log(
            `${node.opType} ${node.name}\n` +
              `  ${inputs.join(", ")} -> ${outputs.join(", ")}`,
          );
        }

        for (const [i, name] of node.output.entries()) {
          if (debugStats.has(name)) {
            logDebugStats(name, results[i].ref);
          }
          vars.set(name, results[i]);
        }
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
