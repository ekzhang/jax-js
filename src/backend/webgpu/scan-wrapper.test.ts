/**
 * Tests for WGSL scan wrapper transformation.
 */

import { describe, expect, it } from "vitest";

import {
  createAllIterationsOffsetsBuffer,
  parseBufferBindings,
  ScanBindingInfo,
  transformArrayAccesses,
  wrapRoutineForScan,
} from "./scan-wrapper";

describe("scan-wrapper", () => {
  describe("parseBufferBindings", () => {
    it("parses simple buffer declarations", () => {
      const code = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
`;
      const bindings = parseBufferBindings(code);
      expect(bindings).toHaveLength(2);
      expect(bindings[0]).toEqual({
        group: 0,
        binding: 0,
        name: "input",
        access: "read",
        type: "f32",
      });
      expect(bindings[1]).toEqual({
        group: 0,
        binding: 1,
        name: "output",
        access: "read_write",
        type: "f32",
      });
    });

    it("parses buffers with i32 type", () => {
      const code = `@group(0) @binding(2) var<storage, read_write> indices: array<i32>;`;
      const bindings = parseBufferBindings(code);
      expect(bindings).toHaveLength(1);
      expect(bindings[0].name).toBe("indices");
      expect(bindings[0].type).toBe("i32");
    });
  });

  describe("transformArrayAccesses", () => {
    it("transforms simple array access", () => {
      const code = `output[idx] = input[idx];`;
      const bindings = [
        {
          group: 0,
          binding: 0,
          name: "input",
          access: "read" as const,
          type: "f32",
        },
        {
          group: 0,
          binding: 1,
          name: "output",
          access: "read_write" as const,
          type: "f32",
        },
      ];
      const result = transformArrayAccesses(code, bindings);
      expect(result).toContain("output[output_offset + (idx)]");
      expect(result).toContain("input[input_offset + (idx)]");
    });

    it("transforms nested expressions", () => {
      const code = `output[base + i * n + j] = input[base + j * n + i];`;
      const bindings = [
        {
          group: 0,
          binding: 0,
          name: "input",
          access: "read" as const,
          type: "f32",
        },
        {
          group: 0,
          binding: 1,
          name: "output",
          access: "read_write" as const,
          type: "f32",
        },
      ];
      const result = transformArrayAccesses(code, bindings);
      expect(result).toContain("output[output_offset + (base + i * n + j)]");
      expect(result).toContain("input[input_offset + (base + j * n + i)]");
    });

    it("handles select() with nested array access", () => {
      const code = `shared[idx] = select(pad, input[base + idx], idx < n);`;
      const bindings = [
        {
          group: 0,
          binding: 0,
          name: "input",
          access: "read" as const,
          type: "f32",
        },
      ];
      const result = transformArrayAccesses(code, bindings);
      expect(result).toContain("input[input_offset + (base + idx)]");
      // shared should NOT be transformed (not in bindings)
      expect(result).toContain("shared[idx]");
    });
  });

  describe("wrapRoutineForScan", () => {
    it("transforms only xs/ys bindings based on scan signature", () => {
      // Simulate a scan body: carry(input) + x -> carry(output)
      // Inputs: [carry_in (read), x (read)]
      // Outputs: [carry_out (read_write)]
      // But for scan: carry is ping-pong (no offset), x needs offset, y needs offset
      const shaderInfo = {
        code: `
@group(0) @binding(0) var<storage, read> carry_in: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> carry_out: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let idx = id.x;
  let val = carry_in[idx] + x[idx];
  carry_out[idx] = val;
  y[idx] = val;
}
`,
        numInputs: 2,
        numOutputs: 2,
        hasUniform: false,
        passes: [{ grid: [1, 1] as [number, number] }],
      };

      const scanInfo: ScanBindingInfo = {
        numConsts: 0,
        numCarry: 1, // carry_in is carry
        numX: 1, // x is xs
        numY: 1, // y is ys (carry_out is carry)
        numInputs: 2,
        numOutputs: 2,
      };

      const result = wrapRoutineForScan(shaderInfo, scanInfo);

      // Should add struct with ONLY x and y offsets (not carry)
      expect(result.code).toContain("struct ScanOffsets");
      expect(result.code).toContain("x_offset: u32");
      expect(result.code).toContain("y_offset: u32");
      expect(result.code).not.toContain("carry_in_offset");
      expect(result.code).not.toContain("carry_out_offset");

      // Should add uniform binding
      expect(result.code).toContain(
        "@group(1) @binding(0) var<uniform> scan_offsets: ScanOffsets",
      );

      // Should transform ONLY x and y array accesses
      expect(result.code).toContain("x[x_offset + (idx)]");
      expect(result.code).toContain("y[y_offset + (idx)]");

      // Carry accesses should be UNCHANGED
      expect(result.code).toContain("carry_in[idx]");
      expect(result.code).toContain("carry_out[idx]");

      expect(result.hasUniform).toBe(true);
    });

    it("returns unchanged shader when no xs/ys bindings", () => {
      // Pure carry operation with no xs/ys
      const shaderInfo = {
        code: `
@group(0) @binding(0) var<storage, read> carry_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> carry_out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  carry_out[id.x] = carry_in[id.x] * 2.0;
}
`,
        numInputs: 1,
        numOutputs: 1,
        hasUniform: false,
        passes: [{ grid: [1, 1] as [number, number] }],
      };

      const scanInfo: ScanBindingInfo = {
        numConsts: 0,
        numCarry: 1,
        numX: 0, // No xs
        numY: 0, // No ys (carry_out is all carry)
        numInputs: 1,
        numOutputs: 1,
      };

      const result = wrapRoutineForScan(shaderInfo, scanInfo);

      // Should return unchanged
      expect(result.code).toBe(shaderInfo.code);
      expect(result.hasUniform).toBe(false);
    });
  });

  describe("createAllIterationsOffsetsBuffer", () => {
    it("creates correctly aligned offset buffer", () => {
      const numX = 1;
      const numY = 1;
      const length = 3;
      const xsElemStrides = [10]; // 10 elements per iteration
      const ysElemStrides = [10];
      const minAlignment = 256;

      const { buffer, alignment } = createAllIterationsOffsetsBuffer(
        numX,
        numY,
        length,
        xsElemStrides,
        ysElemStrides,
        minAlignment,
      );

      // Each iteration needs 2 u32s = 8 bytes, padded to 256
      expect(alignment).toBe(256);
      expect(buffer.length).toBe(256 * 3);

      // Check offsets
      const view = new DataView(buffer.buffer);

      // Iteration 0: x_offset=0, y_offset=0
      expect(view.getUint32(0, true)).toBe(0);
      expect(view.getUint32(4, true)).toBe(0);

      // Iteration 1: x_offset=10, y_offset=10
      expect(view.getUint32(256, true)).toBe(10);
      expect(view.getUint32(256 + 4, true)).toBe(10);

      // Iteration 2: x_offset=20, y_offset=20
      expect(view.getUint32(512, true)).toBe(20);
      expect(view.getUint32(512 + 4, true)).toBe(20);
    });
  });
});
