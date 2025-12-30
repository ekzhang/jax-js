// Shared codegen helpers for generating WGSL.

import { DType } from "../../alu";

export interface ShaderInfo {
  shader: string; // WGSL shader source code.
  passes: {
    grid: [number, number]; // Grid size (number of workgroups) in x and y.
    uniform?: Uint8Array<ArrayBuffer>; // Optional uniform value.
  }[];
}

export const headerWgsl = String.raw`
fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
fn inf() -> f32 { let bits = 0x7f800000u; return bitcast<f32>(bits); }
`.trim();

export function dtypeToWgsl(dtype: DType, storage: boolean = false): string {
  switch (dtype) {
    case DType.Bool:
      return storage ? "i32" : "bool"; // WebGPU does not support bools in buffers.
    case DType.Int32:
      return "i32";
    case DType.Uint32:
      return "u32"; // WebGPU supports uint32 in buffers.
    case DType.Float32:
      return "f32";
    case DType.Float16:
      return "f16";
    default:
      throw new Error(`Unsupported dtype for WebGPU: ${dtype}`);
  }
}

export function maxValueWgsl(dtype: DType): string {
  switch (dtype) {
    case DType.Bool:
      return "1"; // Using i32 representation.
    case DType.Int32:
      return "2147483647"; // 2^31 - 1
    case DType.Uint32:
      return "4294967295u"; // 2^32 - 1
    case DType.Float32:
      return "inf()";
    case DType.Float16:
      return "f16(inf())";
    default:
      throw new Error(`Unsupported dtype for WebGPU: ${dtype}`);
  }
}

export function constToWgsl(dtype: DType, value: any): string {
  if (dtype === DType.Bool) return value ? "true" : "false";
  if (dtype === DType.Int32) return value.toString();
  if (dtype === DType.Uint32) return value.toString() + "u"; // WebGPU uses 'u' suffix for uint32.
  if (dtype === DType.Float32) {
    if (Number.isNaN(value)) return "nan()";
    if (!Number.isFinite(value)) return value > 0 ? "inf()" : "-inf()";
    return "f32(" + value.toString() + ")";
  }
  if (dtype === DType.Float16) {
    if (Number.isNaN(value)) return "f16(nan())";
    if (!Number.isFinite(value))
      return value > 0 ? "f16(inf())" : "f16(-inf())";
    return "f16(" + value.toString() + ")";
  }
  throw new Error(`Unsupported const dtype: ${dtype}`);
}

export const gridOffsetY = 16384;

export function calculateGrid(gridSize: number): [number, number] {
  let gridX = gridSize;
  let gridY = 1;
  // https://web3dsurvey.com/webgpu/limits/maxComputeWorkgroupsPerDimension
  // device.limits.maxComputeWorkgroupsPerDimension = 65535
  if (gridSize > 65535) {
    gridX = gridOffsetY;
    gridY = Math.ceil(gridSize / gridOffsetY);
  }
  return [gridX, gridY];
}
