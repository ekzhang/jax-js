// Special code generation for nullary kernels, which don't take any arrays as
// input. This includes constructors like `arange` and `full`.
//
// It's common for these functions to be used to create new arrays. But because
// the constant inputs may differ, they often result in kernel recompiles, which
// slows down performance in browsers.
//
// To fix this, we lift all constant inputs into _uniforms_.

import {
  calculateGrid,
  dtypeToWgsl,
  ShaderInfo,
  WgslBuilder,
  WgslExpCodegen,
} from "./codegen";
import { AluExp, AluOp, DType, Kernel } from "../../alu";
import { findPow2, strip1 } from "../../utils";

type ConstantUniform = {
  name: string;
  dtype: DType;
  uniformDtype: DType;
  value: number;
};

function uniformDtype(dtype: DType): DType {
  if (dtype === DType.Float16) return DType.Float32;
  if (dtype === DType.Bool) return DType.Int32;
  return dtype;
}

function replacementFor({
  name,
  dtype,
  uniformDtype,
}: ConstantUniform): AluExp {
  const value = AluExp.variable(uniformDtype, `uniforms.${name}`);
  if (dtype === DType.Float16) return AluExp.cast(DType.Float16, value);
  if (dtype === DType.Bool) return AluExp.cmpne(value, AluExp.i32(0));
  return value;
}

function writeUniform(
  view: DataView,
  offset: number,
  dtype: DType,
  value: number,
) {
  switch (dtype) {
    case DType.Float32:
      view.setFloat32(offset, value, true);
      break;
    case DType.Int32:
      view.setInt32(offset, value, true);
      break;
    case DType.Uint32:
      view.setUint32(offset, value, true);
      break;
    default:
      throw new Error(`Unsupported dtype for constant uniform: ${dtype}`);
  }
}

function liftConstants(exp: AluExp): [AluExp, ConstantUniform[]] {
  const uniforms: ConstantUniform[] = [];
  const lifted = exp.rewrite((node) => {
    if (node.op !== AluOp.Const || node.arg === 0) return;
    const uniform: ConstantUniform = {
      name: `c${uniforms.length}`,
      dtype: node.dtype,
      uniformDtype: uniformDtype(node.dtype),
      value: node.arg,
    };
    uniforms.push(uniform);
    return replacementFor(uniform);
  });
  return [lifted, uniforms];
}

function uniformsData(uniforms: ConstantUniform[]): Uint8Array<ArrayBuffer> {
  const data = new Uint8Array(uniforms.length * 4);
  const view = new DataView(data.buffer);
  uniforms.forEach((u, i) =>
    writeUniform(view, i * 4, u.uniformDtype, u.value),
  );
  return data;
}

export function nullaryKernelSource(
  device: GPUDevice,
  kernel: Kernel,
): ShaderInfo | null {
  if (kernel.nargs !== 0 || kernel.reduction) return null;

  let exp = kernel.exp
    .substitute({ gidx: AluExp.special(DType.Int32, "gidx", kernel.size) })
    .simplify();
  let uniforms: ConstantUniform[] = [];
  [exp, uniforms] = liftConstants(exp);

  const wb = new WgslBuilder();
  wb.emitPreamble(device, [exp]);

  if (uniforms.length > 0) {
    wb.emit(
      "struct Uniforms {",
      wb.pushIndent,
      ...uniforms.map((u) => `${u.name}: ${dtypeToWgsl(u.uniformDtype)},`),
      wb.popIndent,
      "}\n",
    );
  }

  const resultTy = dtypeToWgsl(kernel.dtype, true);
  wb.emit(
    `@group(0) @binding(0) var<storage, read_write> result : array<${resultTy}>;`,
  );
  if (uniforms.length > 0) {
    wb.emit(`@group(1) @binding(0) var<uniform> uniforms: Uniforms;`);
  }

  const workgroupSize = findPow2(kernel.size, 256);
  const gridSize = Math.ceil(kernel.size / workgroupSize);
  const [gridX, gridY] = calculateGrid(gridSize);
  wb.emit(
    "",
    `@compute @workgroup_size(${workgroupSize})`,
    "fn main(@builtin(global_invocation_id) id : vec3<u32>) {",
    wb.pushIndent,
  );
  if (gridY === 1) {
    wb.emit(
      `if (id.x >= ${kernel.size}) { return; }`,
      "let gidx: i32 = i32(id.x);",
    );
  } else {
    const sizeX = gridX * workgroupSize;
    wb.emit(
      `if (${sizeX} * id.y + id.x >= ${kernel.size}) { return; }`,
      `let gidx: i32 = i32(${sizeX} * id.y + id.x);`,
    );
  }

  const gen = new WgslExpCodegen(wb, []);
  gen.countReferences(exp);
  let rhs = strip1(gen.run(exp));
  if (resultTy !== dtypeToWgsl(exp.dtype)) rhs = `${resultTy}(${rhs})`;
  wb.emit(`result[gidx] = ${rhs};`, wb.popIndent, "}");

  return {
    code: wb.toString(),
    numInputs: 0,
    numOutputs: 1,
    hasUniform: uniforms.length > 0,
    passes: [
      {
        grid: [gridX, gridY],
        uniform: uniforms.length > 0 ? uniformsData(uniforms) : undefined,
      },
    ],
  };
}
