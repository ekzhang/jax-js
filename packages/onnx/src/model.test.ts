import { create, toBinary } from "@bufbuild/protobuf";
import { numpy as np } from "@jax-js/jax";
import {
  GraphProtoSchema,
  ModelProtoSchema,
  NodeProtoSchema,
  OperatorSetIdProtoSchema,
  TensorProto_DataType,
  TensorProtoSchema,
  TensorShapeProto_DimensionSchema,
  TensorShapeProtoSchema,
  TypeProto_TensorSchema,
  TypeProtoSchema,
  ValueInfoProtoSchema,
} from "onnx-buf";
import { expect, onTestFinished, test } from "vitest";

import { ONNXModel } from "./index";

/**
 * Helper to create a dimension proto.
 */
function dim(value: number) {
  return create(TensorShapeProto_DimensionSchema, {
    value: { case: "dimValue", value: BigInt(value) },
  });
}

/**
 * Helper to create a ValueInfoProto for a float tensor.
 */
function floatTensorInfo(name: string, shape: number[]) {
  return create(ValueInfoProtoSchema, {
    name,
    type: create(TypeProtoSchema, {
      value: {
        case: "tensorType",
        value: create(TypeProto_TensorSchema, {
          elemType: TensorProto_DataType.FLOAT,
          shape: create(TensorShapeProtoSchema, {
            dim: shape.map(dim),
          }),
        }),
      },
    }),
  });
}

/**
 * Helper to create a constant tensor initializer.
 */
function floatInitializer(name: string, shape: number[], data: number[]) {
  return create(TensorProtoSchema, {
    name,
    dims: shape.map(BigInt),
    dataType: TensorProto_DataType.FLOAT,
    floatData: data,
  });
}

test("should evaluate a simple Add operation", async () => {
  // Create a model: C = A + B
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 14n })],
    graph: create(GraphProtoSchema, {
      name: "add_graph",
      input: [floatTensorInfo("A", [2, 3]), floatTensorInfo("B", [2, 3])],
      output: [floatTensorInfo("C", [2, 3])],
      node: [
        create(NodeProtoSchema, {
          opType: "Add",
          input: ["A", "B"],
          output: ["C"],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  const a = np.array([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
  const b = np.array([10, 20, 30, 40, 50, 60]).reshape([2, 3]);
  const result = onnxModel.run({ A: a, B: b });

  expect(await result.C.data()).toEqual(
    new Float32Array([11, 22, 33, 44, 55, 66]),
  );
});

test("should evaluate Add followed by Mul", async () => {
  // Create a model: D = (A + B) * C
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 14n })],
    graph: create(GraphProtoSchema, {
      name: "add_mul_graph",
      input: [
        floatTensorInfo("A", [2, 2]),
        floatTensorInfo("B", [2, 2]),
        floatTensorInfo("C", [2, 2]),
      ],
      output: [floatTensorInfo("D", [2, 2])],
      node: [
        create(NodeProtoSchema, {
          opType: "Add",
          input: ["A", "B"],
          output: ["AB"],
        }),
        create(NodeProtoSchema, {
          opType: "Mul",
          input: ["AB", "C"],
          output: ["D"],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  const a = np.array([1, 2, 3, 4]).reshape([2, 2]);
  const b = np.array([10, 20, 30, 40]).reshape([2, 2]);
  const c = np.array([2, 2, 2, 2]).reshape([2, 2]);
  const result = onnxModel.run({ A: a, B: b, C: c });

  // (1+10)*2=22, (2+20)*2=44, (3+30)*2=66, (4+40)*2=88
  expect(await result.D.data()).toEqual(new Float32Array([22, 44, 66, 88]));
});

test("should handle initializers (constant weights)", async () => {
  // Create a model: C = A + B where B is a constant
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 14n })],
    graph: create(GraphProtoSchema, {
      name: "add_const_graph",
      input: [
        floatTensorInfo("A", [3]),
        floatTensorInfo("B", [3]), // Listed as input but provided as initializer
      ],
      output: [floatTensorInfo("C", [3])],
      initializer: [floatInitializer("B", [3], [100, 200, 300])],
      node: [
        create(NodeProtoSchema, {
          opType: "Add",
          input: ["A", "B"],
          output: ["C"],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  // Only need to provide A since B is an initializer
  const a = np.array([1, 2, 3]);
  const result = onnxModel.run({ A: a });

  expect(await result.C.data()).toEqual(new Float32Array([101, 202, 303]));
});

test("should evaluate MatMul", async () => {
  // Create a model: C = A @ B
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 14n })],
    graph: create(GraphProtoSchema, {
      name: "matmul_graph",
      input: [floatTensorInfo("A", [2, 3]), floatTensorInfo("B", [3, 2])],
      output: [floatTensorInfo("C", [2, 2])],
      node: [
        create(NodeProtoSchema, {
          opType: "MatMul",
          input: ["A", "B"],
          output: ["C"],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  // A = [[1, 2, 3], [4, 5, 6]]
  // B = [[1, 2], [3, 4], [5, 6]]
  const a = np.array([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
  const b = np.array([1, 2, 3, 4, 5, 6]).reshape([3, 2]);
  const result = onnxModel.run({ A: a, B: b });

  // C = [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
  //   = [[22, 28], [49, 64]]
  expect(await result.C.data()).toEqual(new Float32Array([22, 28, 49, 64]));
});

test("should evaluate Relu", async () => {
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 14n })],
    graph: create(GraphProtoSchema, {
      name: "relu_graph",
      input: [floatTensorInfo("X", [6])],
      output: [floatTensorInfo("Y", [6])],
      node: [
        create(NodeProtoSchema, {
          opType: "Relu",
          input: ["X"],
          output: ["Y"],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  const x = np.array([-3, -1, 0, 1, 2, 3]);
  const result = onnxModel.run({ X: x });

  expect(await result.Y.data()).toEqual(new Float32Array([0, 0, 0, 1, 2, 3]));
});

test("should evaluate Reshape", async () => {
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 14n })],
    graph: create(GraphProtoSchema, {
      name: "reshape_graph",
      input: [floatTensorInfo("X", [2, 3])],
      output: [floatTensorInfo("Y", [3, 2])],
      initializer: [
        create(TensorProtoSchema, {
          name: "shape",
          dims: [2n],
          dataType: TensorProto_DataType.INT64,
          int64Data: [3n, 2n],
        }),
      ],
      node: [
        create(NodeProtoSchema, {
          opType: "Reshape",
          input: ["X", "shape"],
          output: ["Y"],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  const x = np.array([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
  const result = onnxModel.run({ X: x });

  expect(result.Y.shape).toEqual([3, 2]);
  expect(await result.Y.data()).toEqual(new Float32Array([1, 2, 3, 4, 5, 6]));
});

test("should evaluate a chain: Add -> Relu -> MatMul", async () => {
  // Create: Y = Relu(A + B) @ C
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 14n })],
    graph: create(GraphProtoSchema, {
      name: "chain_graph",
      input: [
        floatTensorInfo("A", [2, 3]),
        floatTensorInfo("B", [2, 3]),
        floatTensorInfo("C", [3, 1]),
      ],
      output: [floatTensorInfo("Y", [2, 1])],
      node: [
        create(NodeProtoSchema, {
          opType: "Add",
          input: ["A", "B"],
          output: ["sum"],
        }),
        create(NodeProtoSchema, {
          opType: "Relu",
          input: ["sum"],
          output: ["relu_out"],
        }),
        create(NodeProtoSchema, {
          opType: "MatMul",
          input: ["relu_out", "C"],
          output: ["Y"],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  // A + B has some negative values that Relu will zero out
  const a = np.array([-5, 2, 3, 4, -1, 6]).reshape([2, 3]);
  const b = np.array([1, -5, 1, -10, 2, -10]).reshape([2, 3]);
  // sum = [[-4, -3, 4], [-6, 1, -4]]
  // relu = [[0, 0, 4], [0, 1, 0]]
  const c = np.array([1, 1, 1]).reshape([3, 1]);
  // Y = [[0+0+4], [0+1+0]] = [[4], [1]]

  const result = onnxModel.run({ A: a, B: b, C: c });

  expect(result.Y.shape).toEqual([2, 1]);
  expect(await result.Y.data()).toEqual(new Float32Array([4, 1]));
});
