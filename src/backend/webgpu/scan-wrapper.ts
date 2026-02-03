/**
 * WGSL shader transformation for scan-aware routine dispatch.
 *
 * Transforms a routine shader to accept per-iteration offsets for xs/ys buffers,
 * enabling batched command buffer dispatch without minStorageBufferOffsetAlignment
 * constraints.
 *
 * Key insight: We know exactly which bindings need offsets from the scan signature:
 * - Inputs [0, numConsts): constants - NO offset (same each iter)
 * - Inputs [numConsts, numConsts+numCarry): carry - NO offset (ping-pong buffers)
 * - Inputs [numConsts+numCarry, ...): xs - YES offset (iter * stride)
 * - Outputs [0, numCarry): new carry - NO offset (ping-pong buffers)
 * - Outputs [numCarry, ...): ys - YES offset (iter * stride)
 *
 * The transformation:
 * 1. Adds a ScanOffsets uniform struct with offsets for xs/ys bindings only
 * 2. Injects offset variable declarations at the start of main()
 * 3. Transforms array accesses for xs/ys bindings from `buffer[idx]` to `buffer[offset + idx]`
 */

import { ShaderInfo } from "./codegen";

export interface BufferBinding {
  group: number;
  binding: number;
  name: string;
  access: "read" | "read_write";
  type: string; // e.g., "f32", "i32", "vec4<f32>"
}

/** Scan signature metadata - which bindings need offsets */
export interface ScanBindingInfo {
  numConsts: number;
  numCarry: number;
  numX: number;
  numY: number;
  numInputs: number; // total inputs
  numOutputs: number; // total outputs
}

/**
 * Parse buffer declarations from WGSL source.
 * Matches: @group(G) @binding(B) var<storage, ACCESS> NAME: array<TYPE>;
 */
export function parseBufferBindings(code: string): BufferBinding[] {
  const regex =
    /@group\((\d+)\)\s+@binding\((\d+)\)\s+var<storage,\s*(read|read_write)>\s+(\w+)\s*:\s*array<([^>]+)>/g;
  const bindings: BufferBinding[] = [];
  let match;
  while ((match = regex.exec(code)) !== null) {
    bindings.push({
      group: parseInt(match[1]),
      binding: parseInt(match[2]),
      name: match[4],
      access: match[3] as "read" | "read_write",
      type: match[5],
    });
  }
  return bindings;
}

/**
 * Identify which bindings need offsets based on scan signature.
 */
function getOffsetBindings(
  bindings: BufferBinding[],
  scanInfo: ScanBindingInfo,
): { xsBindings: BufferBinding[]; ysBindings: BufferBinding[] } {
  // Sort by binding number
  const sorted = [...bindings].sort((a, b) => a.binding - b.binding);

  // Separate inputs (read) and outputs (read_write)
  const inputs = sorted.filter((b) => b.access === "read");
  const outputs = sorted.filter((b) => b.access === "read_write");

  // XS bindings are inputs after consts and carry
  const xsStart = scanInfo.numConsts + scanInfo.numCarry;
  const xsBindings = inputs.slice(xsStart, xsStart + scanInfo.numX);

  // YS bindings are outputs after carry
  const ysBindings = outputs.slice(
    scanInfo.numCarry,
    scanInfo.numCarry + scanInfo.numY,
  );

  return { xsBindings, ysBindings };
}

/**
 * Generate the ScanOffsets uniform struct declaration.
 * Only includes xs/ys bindings that need offsets.
 */
function generateOffsetsStruct(
  xsBindings: BufferBinding[],
  ysBindings: BufferBinding[],
): string {
  const fields: string[] = [];
  for (const b of xsBindings) {
    fields.push(`  ${b.name}_offset: u32,`);
  }
  for (const b of ysBindings) {
    fields.push(`  ${b.name}_offset: u32,`);
  }
  if (fields.length === 0) {
    return "struct ScanOffsets { _pad: u32, }";
  }
  return `struct ScanOffsets {\n${fields.join("\n")}\n}`;
}

/**
 * Transform array accesses to include offset for specific bindings.
 */
export function transformArrayAccesses(
  code: string,
  bindings: BufferBinding[],
): string {
  let result = code;

  for (const binding of bindings) {
    const name = binding.name;
    // Match: name[ followed by anything until matching ]
    const pattern = new RegExp(`\\b${name}\\s*\\[`, "g");

    let match;
    const replacements: { start: number; end: number; text: string }[] = [];

    while ((match = pattern.exec(result)) !== null) {
      const startBracket = match.index + match[0].length - 1;
      let depth = 1;
      let i = startBracket + 1;

      // Find matching closing bracket
      while (i < result.length && depth > 0) {
        if (result[i] === "[") depth++;
        if (result[i] === "]") depth--;
        i++;
      }

      if (depth !== 0) {
        console.warn(`Unmatched bracket in array access: ${name}`);
        continue;
      }

      const endBracket = i - 1;
      const indexExpr = result.substring(startBracket + 1, endBracket);

      // Create new access with offset
      const newAccess = `${name}[${name}_offset + (${indexExpr})]`;

      replacements.push({
        start: match.index,
        end: i,
        text: newAccess,
      });
    }

    // Apply replacements in reverse order to preserve indices
    for (let r = replacements.length - 1; r >= 0; r--) {
      const rep = replacements[r];
      result =
        result.substring(0, rep.start) + rep.text + result.substring(rep.end);
    }
  }

  return result;
}

/**
 * Find the start of the main() function body.
 * Handles various WGSL main function signatures.
 */
function findMainBodyStart(code: string): number {
  // Match fn main with various parameter patterns, ending with {
  // Use .* to handle nested parens in @builtin(global_invocation_id) etc.
  const mainMatch = code.match(/fn\s+main\s*\(.*\)\s*(?:->[\s\S]*?)?\{/);
  if (!mainMatch || mainMatch.index === undefined) {
    throw new Error("Could not find main() function in shader");
  }
  return mainMatch.index + mainMatch[0].length;
}

/**
 * Generate offset variable declarations to inject at start of main().
 */
function generateOffsetDeclarations(
  xsBindings: BufferBinding[],
  ysBindings: BufferBinding[],
): string {
  const decls: string[] = [];
  for (const b of xsBindings) {
    decls.push(`  let ${b.name}_offset = scan_offsets.${b.name}_offset;`);
  }
  for (const b of ysBindings) {
    decls.push(`  let ${b.name}_offset = scan_offsets.${b.name}_offset;`);
  }
  return decls.join("\n");
}

/**
 * Transform a routine shader for scan-aware dispatch.
 *
 * Only transforms xs/ys buffer accesses (not carry/consts) based on scan signature.
 *
 * @param shaderInfo The original shader info from createRoutineShader
 * @param scanInfo Scan signature info identifying which bindings are xs/ys
 * @returns A new ShaderInfo with transformed code and uniform binding for offsets
 */
export function wrapRoutineForScan(
  shaderInfo: ShaderInfo,
  scanInfo: ScanBindingInfo,
): ShaderInfo {
  const { code } = shaderInfo;

  // Parse buffer bindings
  const allBindings = parseBufferBindings(code);
  if (allBindings.length === 0) {
    throw new Error("No buffer bindings found in shader");
  }

  // Identify which bindings need offsets
  const { xsBindings, ysBindings } = getOffsetBindings(allBindings, scanInfo);

  // If no bindings need offsets, return unchanged
  if (xsBindings.length === 0 && ysBindings.length === 0) {
    return shaderInfo;
  }

  // Generate uniform struct and binding
  const offsetsStruct = generateOffsetsStruct(xsBindings, ysBindings);
  const uniformBinding = `@group(1) @binding(0) var<uniform> scan_offsets: ScanOffsets;`;

  // Find where to inject the uniform declarations (after the last @group(0) binding)
  const lastBindingMatch = [
    ...code.matchAll(
      /@group\(0\)\s+@binding\(\d+\)\s+var<[^>]+>\s+\w+\s*:\s*[^;]+;/g,
    ),
  ];
  if (lastBindingMatch.length === 0) {
    throw new Error("No group(0) bindings found");
  }
  const lastBinding = lastBindingMatch[lastBindingMatch.length - 1];
  const insertPoint = lastBinding.index! + lastBinding[0].length;

  // Build new code
  let newCode = code.substring(0, insertPoint);
  newCode += `\n\n${offsetsStruct}\n${uniformBinding}\n`;
  newCode += code.substring(insertPoint);

  // Transform array accesses ONLY for xs/ys bindings
  const bindingsToTransform = [...xsBindings, ...ysBindings];
  newCode = transformArrayAccesses(newCode, bindingsToTransform);

  // Inject offset declarations at start of main()
  const mainStart = findMainBodyStart(newCode);
  const offsetDecls = generateOffsetDeclarations(xsBindings, ysBindings);
  if (offsetDecls) {
    newCode =
      newCode.substring(0, mainStart) +
      "\n" +
      offsetDecls +
      "\n" +
      newCode.substring(mainStart);
  }

  return {
    ...shaderInfo,
    code: newCode,
    hasUniform: true, // Now requires uniform binding for offsets
  };
}

/**
 * Create a uniform buffer containing offsets for a single scan iteration.
 *
 * Offsets are in ELEMENTS (not bytes), matching how WGSL array indexing works.
 *
 * @param numX Number of xs buffers
 * @param numY Number of ys buffers
 * @param iteration Current iteration index
 * @param xsElemStrides Elements per iteration for each xs buffer
 * @param ysElemStrides Elements per iteration for each ys buffer
 * @returns Uint8Array containing the uniform data
 */
export function createScanOffsetsUniform(
  numX: number,
  numY: number,
  iteration: number,
  xsElemStrides: number[], // elements per iteration
  ysElemStrides: number[], // elements per iteration
): Uint8Array {
  // Each offset is a u32 (4 bytes)
  const totalOffsets = numX + numY;
  const data = new Uint32Array(totalOffsets);

  // XS offsets (element-based)
  for (let i = 0; i < numX; i++) {
    data[i] = iteration * xsElemStrides[i];
  }

  // YS offsets (element-based)
  for (let i = 0; i < numY; i++) {
    data[numX + i] = iteration * ysElemStrides[i];
  }

  return new Uint8Array(data.buffer);
}

/**
 * Create a combined uniform buffer with offsets for ALL iterations.
 * Each iteration's offsets are padded to minUniformBufferOffsetAlignment.
 *
 * This allows using dynamic uniform offsets for efficient iteration.
 *
 * @returns The combined buffer and the alignment (bytes between iterations)
 */
export function createAllIterationsOffsetsBuffer(
  numX: number,
  numY: number,
  length: number,
  xsElemStrides: number[],
  ysElemStrides: number[],
  minAlignment: number,
  reverse?: boolean,
): { buffer: Uint8Array; alignment: number } {
  const offsetsPerIter = numX + numY;
  const bytesPerIter = offsetsPerIter * 4;

  // Pad to alignment
  const alignment = Math.ceil(bytesPerIter / minAlignment) * minAlignment;
  const totalBytes = alignment * length;

  const buffer = new Uint8Array(totalBytes);
  const view = new DataView(buffer.buffer);

  for (let iter = 0; iter < length; iter++) {
    const baseOffset = iter * alignment;

    // XS offsets: read in reverse order when reverse=true
    // For reverse scan, iteration 0 reads from xs[length-1], iteration 1 reads from xs[length-2], etc.
    const xsIter = reverse ? length - 1 - iter : iter;
    for (let i = 0; i < numX; i++) {
      view.setUint32(baseOffset + i * 4, xsIter * xsElemStrides[i], true);
    }

    // YS offsets: always write in forward order
    // Output[0] gets the result of processing xs[length-1] (first in reverse), etc.
    for (let i = 0; i < numY; i++) {
      view.setUint32(
        baseOffset + (numX + i) * 4,
        iter * ysElemStrides[i],
        true,
      );
    }
  }

  return { buffer, alignment };
}
