import { create, toBinary } from "@bufbuild/protobuf";
import { defaultDevice, numpy as np } from "@jax-js/jax";
import {
  AttributeProto_AttributeType,
  AttributeProtoSchema,
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

defaultDevice("wasm");

/**
 * Helper to create a dimension proto.
 */
function dim(value: number) {
  return create(TensorShapeProto_DimensionSchema, {
    value: { case: "dimValue", value: BigInt(value) },
  });
}

/**
 * Helper to create a ValueInfoProto for a tensor.
 */
function tensorInfo(
  name: string,
  shape: number[],
  elemType: TensorProto_DataType,
) {
  return create(ValueInfoProtoSchema, {
    name,
    type: create(TypeProtoSchema, {
      value: {
        case: "tensorType",
        value: create(TypeProto_TensorSchema, {
          elemType,
          shape: create(TensorShapeProtoSchema, {
            dim: shape.map(dim),
          }),
        }),
      },
    }),
  });
}

/**
 * Helper to create a ValueInfoProto for a float tensor.
 */
function floatTensorInfo(name: string, shape: number[]) {
  return tensorInfo(name, shape, TensorProto_DataType.FLOAT);
}

function intTensorInfo(name: string, shape: number[]) {
  return tensorInfo(name, shape, TensorProto_DataType.INT32);
}

function uintTensorInfo(name: string, shape: number[]) {
  return tensorInfo(name, shape, TensorProto_DataType.UINT32);
}

function boolTensorInfo(name: string, shape: number[]) {
  return tensorInfo(name, shape, TensorProto_DataType.BOOL);
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

function int64Initializer(name: string, shape: number[], data: number[]) {
  return create(TensorProtoSchema, {
    name,
    dims: shape.map(BigInt),
    dataType: TensorProto_DataType.INT64,
    int64Data: data.map(BigInt),
  });
}

function intAttr(name: string, value: number) {
  return create(AttributeProtoSchema, {
    name,
    type: AttributeProto_AttributeType.INT,
    i: BigInt(value),
  });
}

function intsAttr(name: string, value: number[]) {
  return create(AttributeProtoSchema, {
    name,
    type: AttributeProto_AttributeType.INTS,
    ints: value.map(BigInt),
  });
}

function floatAttr(name: string, value: number) {
  return create(AttributeProtoSchema, {
    name,
    type: AttributeProto_AttributeType.FLOAT,
    f: value,
  });
}

function stringAttr(name: string, value: string) {
  return create(AttributeProtoSchema, {
    name,
    type: AttributeProto_AttributeType.STRING,
    s: new TextEncoder().encode(value) as Uint8Array<ArrayBuffer>,
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

test("should evaluate Trilu", async () => {
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 14n })],
    graph: create(GraphProtoSchema, {
      name: "trilu_graph",
      input: [floatTensorInfo("X", [2, 3])],
      output: [floatTensorInfo("Y", [2, 3])],
      initializer: [int64Initializer("K", [], [-1])],
      node: [
        create(NodeProtoSchema, {
          opType: "Trilu",
          input: ["X", "K"],
          output: ["Y"],
          attribute: [intAttr("upper", 0)],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  const x = np.array([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
  const result = onnxModel.run({ X: x });

  expect(await result.Y.data()).toEqual(new Float32Array([0, 0, 0, 4, 0, 0]));
});

test("should evaluate EyeLike", async () => {
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 14n })],
    graph: create(GraphProtoSchema, {
      name: "eyelike_graph",
      input: [floatTensorInfo("X", [2, 4])],
      output: [intTensorInfo("Y", [2, 4])],
      node: [
        create(NodeProtoSchema, {
          opType: "EyeLike",
          input: ["X"],
          output: ["Y"],
          attribute: [
            intAttr("dtype", TensorProto_DataType.INT32),
            intAttr("k", 1),
          ],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  const x = np.zeros([2, 4]);
  const result = onnxModel.run({ X: x });

  expect(await result.Y.data()).toEqual(
    new Int32Array([0, 1, 0, 0, 0, 0, 1, 0]),
  );
});

test("should evaluate LpNormalization", async () => {
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 14n })],
    graph: create(GraphProtoSchema, {
      name: "lp_normalization_graph",
      input: [floatTensorInfo("X", [2, 3])],
      output: [floatTensorInfo("Y", [2, 3])],
      node: [
        create(NodeProtoSchema, {
          opType: "LpNormalization",
          input: ["X"],
          output: ["Y"],
          attribute: [intAttr("axis", 1), intAttr("p", 2)],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  const x = np.array([3, 4, 0, 0, 0, 0]).reshape([2, 3]);
  const result = onnxModel.run({ X: x });

  expect(result.Y).toBeAllclose([
    [0.6, 0.8, 0],
    [0, 0, 0],
  ]);
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

test("should evaluate ONNX activation elementwise ops", async () => {
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 18n })],
    graph: create(GraphProtoSchema, {
      name: "activation_elementwise_graph",
      input: [floatTensorInfo("X", [5])],
      output: [
        floatTensorInfo("hard_sigmoid", [5]),
        floatTensorInfo("hard_swish", [5]),
        floatTensorInfo("selu", [5]),
        floatTensorInfo("prelu", [5]),
        floatTensorInfo("thresholded_relu", [5]),
        floatTensorInfo("shrink", [5]),
      ],
      initializer: [floatInitializer("slope", [1], [0.25])],
      node: [
        create(NodeProtoSchema, {
          opType: "HardSigmoid",
          input: ["X"],
          output: ["hard_sigmoid"],
          attribute: [floatAttr("alpha", 0.5), floatAttr("beta", 0.25)],
        }),
        create(NodeProtoSchema, {
          opType: "HardSwish",
          input: ["X"],
          output: ["hard_swish"],
        }),
        create(NodeProtoSchema, {
          opType: "Selu",
          input: ["X"],
          output: ["selu"],
          attribute: [floatAttr("alpha", 2), floatAttr("gamma", 0.5)],
        }),
        create(NodeProtoSchema, {
          opType: "PRelu",
          input: ["X", "slope"],
          output: ["prelu"],
        }),
        create(NodeProtoSchema, {
          opType: "ThresholdedRelu",
          input: ["X"],
          output: ["thresholded_relu"],
          attribute: [floatAttr("alpha", 1)],
        }),
        create(NodeProtoSchema, {
          opType: "Shrink",
          input: ["X"],
          output: ["shrink"],
          attribute: [floatAttr("lambd", 1), floatAttr("bias", 0.5)],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  const x = np.array([-2, -1, 0, 1, 2]);
  const result = onnxModel.run({ X: x });

  expect(result.hard_sigmoid).toBeAllclose([0, 0, 0.25, 0.75, 1]);
  expect(result.hard_swish).toBeAllclose([-1 / 3, -1 / 3, 0, 2 / 3, 5 / 3]);
  expect(result.selu).toBeAllclose([
    Math.exp(-2) - 1,
    Math.exp(-1) - 1,
    0,
    0.5,
    1,
  ]);
  expect(result.prelu).toBeAllclose([-0.5, -0.25, 0, 1, 2]);
  expect(await result.thresholded_relu.data()).toEqual(
    new Float32Array([0, 0, 0, 0, 2]),
  );
  expect(result.shrink).toBeAllclose([-1.5, 0, 0, 0, 1.5]);
});

test("should evaluate logical, numeric classification, and extrema elementwise ops", async () => {
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 20n })],
    graph: create(GraphProtoSchema, {
      name: "logical_numeric_elementwise_graph",
      input: [
        boolTensorInfo("A", [4]),
        boolTensorInfo("B", [4]),
        floatTensorInfo("X", [6]),
        floatTensorInfo("F", [4]),
        floatTensorInfo("M1", [3]),
        floatTensorInfo("M2", [3]),
        floatTensorInfo("M3", [3]),
      ],
      output: [
        boolTensorInfo("and", [4]),
        boolTensorInfo("or", [4]),
        boolTensorInfo("xor", [4]),
        floatTensorInfo("rounded", [6]),
        boolTensorInfo("pos_inf", [4]),
        boolTensorInfo("nan", [4]),
        floatTensorInfo("min", [3]),
        floatTensorInfo("max", [3]),
      ],
      node: [
        create(NodeProtoSchema, {
          opType: "And",
          input: ["A", "B"],
          output: ["and"],
        }),
        create(NodeProtoSchema, {
          opType: "Or",
          input: ["A", "B"],
          output: ["or"],
        }),
        create(NodeProtoSchema, {
          opType: "Xor",
          input: ["A", "B"],
          output: ["xor"],
        }),
        create(NodeProtoSchema, {
          opType: "Round",
          input: ["X"],
          output: ["rounded"],
        }),
        create(NodeProtoSchema, {
          opType: "IsInf",
          input: ["F"],
          output: ["pos_inf"],
          attribute: [
            intAttr("detect_negative", 0),
            intAttr("detect_positive", 1),
          ],
        }),
        create(NodeProtoSchema, {
          opType: "IsNaN",
          input: ["F"],
          output: ["nan"],
        }),
        create(NodeProtoSchema, {
          opType: "Min",
          input: ["M1", "M2", "M3"],
          output: ["min"],
        }),
        create(NodeProtoSchema, {
          opType: "Max",
          input: ["M1", "M2", "M3"],
          output: ["max"],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  const result = onnxModel.run({
    A: np.array([true, true, false, false]),
    B: np.array([true, false, true, false]),
    X: np.array([-0, -4.5, -1.5, 1.5, 2.5, 3.2]),
    F: np.array([-Infinity, Infinity, NaN, 0]),
    M1: np.array([1, 5, -2]),
    M2: np.array([2, 4, -3]),
    M3: np.array([0, 6, -1]),
  });

  expect(await result.and.data()).toEqual(new Int32Array([1, 0, 0, 0]));
  expect(await result.or.data()).toEqual(new Int32Array([1, 1, 1, 0]));
  expect(await result.xor.data()).toEqual(new Int32Array([0, 1, 1, 0]));

  const rounded = await result.rounded.data();
  expect(Object.is(rounded[0], -0)).toBe(true);
  expect(rounded.slice(1)).toEqual(new Float32Array([-4, -2, 2, 2, 3]));

  expect(await result.pos_inf.data()).toEqual(new Int32Array([0, 1, 0, 0]));
  expect(await result.nan.data()).toEqual(new Int32Array([0, 0, 1, 0]));
  expect(await result.min.data()).toEqual(new Float32Array([0, 4, -3]));
  expect(await result.max.data()).toEqual(new Float32Array([2, 6, -1]));
});

test("should evaluate bitwise elementwise ops and BitCast", async () => {
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 18n })],
    graph: create(GraphProtoSchema, {
      name: "bitwise_elementwise_graph",
      input: [
        uintTensorInfo("U", [3]),
        uintTensorInfo("V", [3]),
        intTensorInfo("I", [2]),
      ],
      output: [
        uintTensorInfo("and", [3]),
        uintTensorInfo("or", [3]),
        uintTensorInfo("xor", [3]),
        uintTensorInfo("not", [3]),
        uintTensorInfo("left", [3]),
        uintTensorInfo("right", [3]),
        floatTensorInfo("bitcast", [2]),
      ],
      node: [
        create(NodeProtoSchema, {
          opType: "BitwiseAnd",
          input: ["U", "V"],
          output: ["and"],
        }),
        create(NodeProtoSchema, {
          opType: "BitwiseOr",
          input: ["U", "V"],
          output: ["or"],
        }),
        create(NodeProtoSchema, {
          opType: "BitwiseXor",
          input: ["U", "V"],
          output: ["xor"],
        }),
        create(NodeProtoSchema, {
          opType: "BitwiseNot",
          input: ["U"],
          output: ["not"],
        }),
        create(NodeProtoSchema, {
          opType: "BitShift",
          input: ["U", "V"],
          output: ["left"],
          attribute: [stringAttr("direction", "LEFT")],
        }),
        create(NodeProtoSchema, {
          opType: "BitShift",
          input: ["U", "V"],
          output: ["right"],
          attribute: [stringAttr("direction", "RIGHT")],
        }),
        create(NodeProtoSchema, {
          opType: "BitCast",
          input: ["I"],
          output: ["bitcast"],
          attribute: [intAttr("to", TensorProto_DataType.FLOAT)],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  const result = onnxModel.run({
    U: np.array([1, 6, 0xffffffff], { dtype: np.uint32 }),
    V: np.array([2, 1, 4], { dtype: np.uint32 }),
    I: np.array([1065353216, -1082130432], { dtype: np.int32 }),
  });

  expect(await result.and.data()).toEqual(new Uint32Array([0, 0, 4]));
  expect(await result.or.data()).toEqual(new Uint32Array([3, 7, 0xffffffff]));
  expect(await result.xor.data()).toEqual(new Uint32Array([3, 7, 0xfffffffb]));
  expect(await result.not.data()).toEqual(
    new Uint32Array([0xfffffffe, 0xfffffff9, 0]),
  );
  expect(await result.left.data()).toEqual(
    new Uint32Array([4, 12, 0xfffffff0]),
  );
  expect(await result.right.data()).toEqual(
    new Uint32Array([0, 3, 0x0fffffff]),
  );
  expect(await result.bitcast.data()).toEqual(new Float32Array([1, -1]));
});

test("should evaluate MeanVarianceNormalization", async () => {
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 13n })],
    graph: create(GraphProtoSchema, {
      name: "mean_variance_normalization_graph",
      input: [floatTensorInfo("X", [1, 2, 2, 2])],
      output: [floatTensorInfo("Y", [1, 2, 2, 2])],
      node: [
        create(NodeProtoSchema, {
          opType: "MeanVarianceNormalization",
          input: ["X"],
          output: ["Y"],
          attribute: [intsAttr("axes", [0, 2, 3])],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  const x = np.array([1, 2, 3, 4, 2, 4, 6, 8]).reshape([1, 2, 2, 2]);
  const result = onnxModel.run({ X: x });

  expect(result.Y).toBeAllclose(
    [
      [
        [
          [-1.3416408, -0.4472136],
          [0.4472136, 1.3416408],
        ],
        [
          [-1.3416408, -0.4472136],
          [0.4472136, 1.3416408],
        ],
      ],
    ],
    { atol: 1e-5 },
  );
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

test("should preserve float StaticArray constants for Resize scales", async () => {
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 16n })],
    graph: create(GraphProtoSchema, {
      name: "resize_scales_graph",
      input: [floatTensorInfo("X", [1, 1, 2, 2])],
      output: [floatTensorInfo("Y", [1, 1, 3, 3])],
      initializer: [floatInitializer("scales", [4], [1, 1, 1.5, 1.5])],
      node: [
        create(NodeProtoSchema, {
          opType: "Resize",
          input: ["X", "", "scales", ""],
          output: ["Y"],
          attribute: [
            stringAttr("coordinate_transformation_mode", "asymmetric"),
            stringAttr("mode", "nearest"),
            stringAttr("nearest_mode", "floor"),
          ],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  const x = np.array([1, 2, 3, 4]).reshape([1, 1, 2, 2]);
  const result = onnxModel.run({ X: x });

  expect(result.Y.shape).toEqual([1, 1, 3, 3]);
  expect(await result.Y.data()).toEqual(
    new Float32Array([1, 1, 2, 1, 1, 2, 3, 3, 4]),
  );
});

test("should evaluate GatherElements", async () => {
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 16n })],
    graph: create(GraphProtoSchema, {
      name: "gather_elements_graph",
      input: [floatTensorInfo("X", [2, 3])],
      output: [floatTensorInfo("Y", [2, 2])],
      initializer: [int64Initializer("indices", [2, 2], [2, 0, 1, 2])],
      node: [
        create(NodeProtoSchema, {
          opType: "GatherElements",
          input: ["X", "indices"],
          output: ["Y"],
          attribute: [intAttr("axis", 1)],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  const x = np.array([1, 2, 3, 4, 5, 6]).reshape([2, 3]);
  const result = onnxModel.run({ X: x });

  expect(result.Y.shape).toEqual([2, 2]);
  expect(await result.Y.data()).toEqual(new Float32Array([3, 1, 5, 6]));
});

test("should keep integer Div as truncating division", async () => {
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 16n })],
    graph: create(GraphProtoSchema, {
      name: "integer_div_graph",
      input: [intTensorInfo("X", [6])],
      output: [intTensorInfo("Y", [6])],
      initializer: [int64Initializer("denominator", [], [3])],
      node: [
        create(NodeProtoSchema, {
          opType: "Div",
          input: ["X", "denominator"],
          output: ["Y"],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  const x = np.array([-7, -5, -1, 1, 5, 7], { dtype: np.int32 });
  const result = onnxModel.run({ X: x });

  expect(result.Y.dtype).toBe(np.int32);
  expect(await result.Y.data()).toEqual(new Int32Array([-2, -1, 0, 0, 1, 2]));
});

test("should evaluate TopK", async () => {
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 16n })],
    graph: create(GraphProtoSchema, {
      name: "topk_graph",
      input: [floatTensorInfo("X", [2, 4])],
      output: [
        floatTensorInfo("values", [2, 2]),
        intTensorInfo("indices", [2, 2]),
      ],
      initializer: [int64Initializer("K", [1], [2])],
      node: [
        create(NodeProtoSchema, {
          opType: "TopK",
          input: ["X", "K"],
          output: ["values", "indices"],
          attribute: [intAttr("axis", 1)],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  const x = np.array([1, 4, 2, 3, 9, 7, 8, 6]).reshape([2, 4]);
  const result = onnxModel.run({ X: x });

  expect(await result.values.data()).toEqual(new Float32Array([4, 3, 9, 8]));
  expect(await result.indices.data()).toEqual(new Int32Array([1, 3, 0, 2]));
});

test("should evaluate LayerNormalization", async () => {
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 17n })],
    graph: create(GraphProtoSchema, {
      name: "layer_normalization_graph",
      input: [floatTensorInfo("X", [1, 2, 2])],
      output: [floatTensorInfo("Y", [1, 2, 2])],
      initializer: [
        floatInitializer("scale", [2], [1, 1]),
        floatInitializer("bias", [2], [0, 0]),
      ],
      node: [
        create(NodeProtoSchema, {
          opType: "LayerNormalization",
          input: ["X", "scale", "bias"],
          output: ["Y"],
          attribute: [floatAttr("epsilon", 0), intAttr("axis", -1)],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  const x = np.array([1, 3, 2, 4]).reshape([1, 2, 2]);
  const result = onnxModel.run({ X: x });

  expect(result.Y.shape).toEqual([1, 2, 2]);
  expect(await result.Y.data()).toEqual(new Float32Array([-1, 1, -1, 1]));
});

test("should evaluate ArgMin", async () => {
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 13n })],
    graph: create(GraphProtoSchema, {
      name: "argmin_graph",
      input: [floatTensorInfo("X", [2, 2]), floatTensorInfo("X_ties", [2, 2])],
      output: [
        intTensorInfo("default_axes_keepdims", [1, 2]),
        intTensorInfo("keepdims", [2, 1]),
        intTensorInfo("no_keepdims", [2]),
        intTensorInfo("negative_axis_keepdims", [2, 1]),
        intTensorInfo("select_last_default_axes", [1, 2]),
        intTensorInfo("select_last_keepdims", [2, 1]),
        intTensorInfo("select_last_no_keepdims", [2]),
        intTensorInfo("select_last_negative_axis_keepdims", [2, 1]),
      ],
      node: [
        create(NodeProtoSchema, {
          opType: "ArgMin",
          input: ["X"],
          output: ["default_axes_keepdims"],
          attribute: [intAttr("keepdims", 1)],
        }),
        create(NodeProtoSchema, {
          opType: "ArgMin",
          input: ["X"],
          output: ["keepdims"],
          attribute: [intAttr("axis", 1), intAttr("keepdims", 1)],
        }),
        create(NodeProtoSchema, {
          opType: "ArgMin",
          input: ["X"],
          output: ["no_keepdims"],
          attribute: [intAttr("axis", 1), intAttr("keepdims", 0)],
        }),
        create(NodeProtoSchema, {
          opType: "ArgMin",
          input: ["X"],
          output: ["negative_axis_keepdims"],
          attribute: [intAttr("axis", -1), intAttr("keepdims", 1)],
        }),
        create(NodeProtoSchema, {
          opType: "ArgMin",
          input: ["X_ties"],
          output: ["select_last_default_axes"],
          attribute: [intAttr("keepdims", 1), intAttr("select_last_index", 1)],
        }),
        create(NodeProtoSchema, {
          opType: "ArgMin",
          input: ["X_ties"],
          output: ["select_last_keepdims"],
          attribute: [
            intAttr("axis", 1),
            intAttr("keepdims", 1),
            intAttr("select_last_index", 1),
          ],
        }),
        create(NodeProtoSchema, {
          opType: "ArgMin",
          input: ["X_ties"],
          output: ["select_last_no_keepdims"],
          attribute: [
            intAttr("axis", 1),
            intAttr("keepdims", 0),
            intAttr("select_last_index", 1),
          ],
        }),
        create(NodeProtoSchema, {
          opType: "ArgMin",
          input: ["X_ties"],
          output: ["select_last_negative_axis_keepdims"],
          attribute: [
            intAttr("axis", -1),
            intAttr("keepdims", 1),
            intAttr("select_last_index", 1),
          ],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  const result = onnxModel.run({
    X: np.array([2, 1, 3, 10]).reshape([2, 2]),
    X_ties: np.array([2, 2, 3, 10]).reshape([2, 2]),
  });

  expect(await result.default_axes_keepdims.data()).toEqual(
    new Int32Array([0, 0]),
  );
  expect(await result.keepdims.data()).toEqual(new Int32Array([1, 0]));
  expect(await result.no_keepdims.data()).toEqual(new Int32Array([1, 0]));
  expect(await result.negative_axis_keepdims.data()).toEqual(
    new Int32Array([1, 0]),
  );
  expect(await result.select_last_default_axes.data()).toEqual(
    new Int32Array([0, 0]),
  );
  expect(await result.select_last_keepdims.data()).toEqual(
    new Int32Array([1, 0]),
  );
  expect(await result.select_last_no_keepdims.data()).toEqual(
    new Int32Array([1, 0]),
  );
  expect(await result.select_last_negative_axis_keepdims.data()).toEqual(
    new Int32Array([1, 0]),
  );
});

test("should evaluate ArgMax", async () => {
  const model = create(ModelProtoSchema, {
    irVersion: 8n,
    opsetImport: [create(OperatorSetIdProtoSchema, { version: 13n })],
    graph: create(GraphProtoSchema, {
      name: "argmax_graph",
      input: [floatTensorInfo("X", [2, 2])],
      output: [
        intTensorInfo("default_axes_keepdims", [1, 2]),
        intTensorInfo("keepdims", [2, 1]),
        intTensorInfo("no_keepdims", [2]),
        intTensorInfo("negative_axis_keepdims", [2, 1]),
        intTensorInfo("select_last_default_axes", [1, 2]),
        intTensorInfo("select_last_keepdims", [2, 1]),
        intTensorInfo("select_last_no_keepdims", [2]),
        intTensorInfo("select_last_negative_axis_keepdims", [2, 1]),
      ],
      node: [
        create(NodeProtoSchema, {
          opType: "ArgMax",
          input: ["X"],
          output: ["default_axes_keepdims"],
          attribute: [intAttr("keepdims", 1)],
        }),
        create(NodeProtoSchema, {
          opType: "ArgMax",
          input: ["X"],
          output: ["keepdims"],
          attribute: [intAttr("axis", 1), intAttr("keepdims", 1)],
        }),
        create(NodeProtoSchema, {
          opType: "ArgMax",
          input: ["X"],
          output: ["no_keepdims"],
          attribute: [intAttr("axis", 1), intAttr("keepdims", 0)],
        }),
        create(NodeProtoSchema, {
          opType: "ArgMax",
          input: ["X"],
          output: ["negative_axis_keepdims"],
          attribute: [intAttr("axis", -1), intAttr("keepdims", 1)],
        }),
        create(NodeProtoSchema, {
          opType: "ArgMax",
          input: ["X"],
          output: ["select_last_default_axes"],
          attribute: [intAttr("keepdims", 1), intAttr("select_last_index", 1)],
        }),
        create(NodeProtoSchema, {
          opType: "ArgMax",
          input: ["X"],
          output: ["select_last_keepdims"],
          attribute: [
            intAttr("axis", 1),
            intAttr("keepdims", 1),
            intAttr("select_last_index", 1),
          ],
        }),
        create(NodeProtoSchema, {
          opType: "ArgMax",
          input: ["X"],
          output: ["select_last_no_keepdims"],
          attribute: [
            intAttr("axis", 1),
            intAttr("keepdims", 0),
            intAttr("select_last_index", 1),
          ],
        }),
        create(NodeProtoSchema, {
          opType: "ArgMax",
          input: ["X"],
          output: ["select_last_negative_axis_keepdims"],
          attribute: [
            intAttr("axis", -1),
            intAttr("keepdims", 1),
            intAttr("select_last_index", 1),
          ],
        }),
      ],
    }),
  });

  const onnxModel = new ONNXModel(toBinary(ModelProtoSchema, model));
  onTestFinished(() => onnxModel.dispose());

  const x = np.array([2, 2, 3, 10]).reshape([2, 2]);
  const result = onnxModel.run({ X: x });

  expect(await result.default_axes_keepdims.data()).toEqual(
    new Int32Array([1, 1]),
  );
  expect(await result.keepdims.data()).toEqual(new Int32Array([0, 1]));
  expect(await result.no_keepdims.data()).toEqual(new Int32Array([0, 1]));
  expect(await result.negative_axis_keepdims.data()).toEqual(
    new Int32Array([0, 1]),
  );
  expect(await result.select_last_default_axes.data()).toEqual(
    new Int32Array([1, 1]),
  );
  expect(await result.select_last_keepdims.data()).toEqual(
    new Int32Array([1, 1]),
  );
  expect(await result.select_last_no_keepdims.data()).toEqual(
    new Int32Array([1, 1]),
  );
  expect(await result.select_last_negative_axis_keepdims.data()).toEqual(
    new Int32Array([1, 1]),
  );
});
