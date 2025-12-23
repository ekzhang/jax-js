# DETR ResNet-50 ONNX: JIT and Dynamic Shapes Investigation

## Problem

When trying to JIT-compile the ONNX model, operations that call `.js()` to extract tensor values
break tracing. These are "shape ops" where ONNX passes shape/index information as tensor inputs
rather than attributes.

## Shape Op Counts (model_quantized.onnx)

| Op              | Count | Uses `.js()`                    |
| --------------- | ----- | ------------------------------- |
| Unsqueeze       | 307   | Yes (axes)                      |
| Concat          | 181   | No                              |
| Reshape         | 179   | Yes (shape)                     |
| Shape           | 80    | No (produces shape tensors)     |
| Gather          | 72    | No                              |
| Slice           | 11    | Yes (starts, ends, axes, steps) |
| ConstantOfShape | 4     | Yes (shape)                     |
| Expand          | 3     | Yes (shape)                     |
| Tile            | 1     | Yes (repeats)                   |
| Resize          | 1     | Yes (sizes/scales)              |

## How Shape Tensors Flow

The typical pattern in DETR:

```
Shape(activation) → Gather(dim) → Unsqueeze → Concat([..., CONST]) → Reshape
```

For example, attention head reshaping:

1. `Shape(input)` extracts `[batch, seq_len, hidden]`
2. `Gather` picks individual dims (e.g., batch=1, seq_len=625)
3. `Unsqueeze` makes them 1D tensors
4. `Concat` combines with constants to form `[1, 625, 8, 32]`
5. `Reshape` uses this to split heads

## ONNX Optimizer Results

Running `onnxoptimizer.optimize()` reduces some ops:

| Op        | Before | After | Change |
| --------- | ------ | ----- | ------ |
| Unsqueeze | 307    | 97    | -210   |
| Shape     | 80     | 39    | -41    |
| Reshape   | 179    | 177   | -2     |
| Concat    | 181    | 180   | -1     |

The 39 remaining `Shape` ops are on **intermediate activations** (attention outputs, layer norm
outputs, conv outputs) - these can't be constant-folded because they depend on runtime tensor
shapes.

## Remaining Reshape Shape Origins (after optimization)

```
54x Concat([Unsqueeze(Mul), CONST, CONST])
36x Concat([Unsqueeze(Gather(Shape(...))), CONST, CONST, ...])
30x Concat([Unsqueeze(Gather(Shape(...))), CONST, Unsqueeze(Gather(Shape(...))), ...])
18x Concat([Unsqueeze(Gather(Shape(...))), Unsqueeze(Gather(Shape(...))), CONST, ...])
18x Concat([Unsqueeze(Gather(Shape(...))), Unsqueeze(Gather(Shape(...))), Unsqueeze(Gather(Shape(...)))])
12x Concat([Unsqueeze(Mul), Unsqueeze(Gather(Shape(...))), Unsqueeze(Gather(Shape(...)))])
 5x CONST
 4x Concat([Slice, CONST])
```

## Solution: Shape Tensor Fast Path

Since constant folding isn't possible for dynamic shapes, the best approach is to track int64
tensors separately and implement operations on them without going through jax-js tracing.

### Implementation Strategy

1. **Track int64/int32 shape tensors as plain JS arrays** instead of traced `np.Array`
2. **Implement shape ops natively in JS**:
   - `Shape(x)` → returns `x.shape` as `number[]`
   - `Gather(arr, idx)` → `arr[idx]`
   - `Unsqueeze(arr)` → `[arr]` or wrap appropriately
   - `Concat([a, b, c])` → `[...a, ...b, ...c]`
   - `Slice(arr, ...)` → JS array slice
3. **Only convert to np.Array when needed** - when shape tensor is used as actual data

This way the JIT trace sees:

```typescript
// Instead of tracing Shape → Gather → Concat → Reshape
data.reshape([1, 625, 8, 32]); // Concrete numbers, JIT-friendly
```

### Files to Modify

- `packages/onnx/src/index.ts` - Track which tensors are "shape tensors"
- `packages/onnx/src/ops/movement.ts` - Handle shape tensor inputs specially
- `packages/onnx/src/ops/utility.ts` - Shape op returns JS array, not np.Array

### Alternative: Shape Specialization

If the shape tensor approach is too complex, another option is shape specialization:

- JIT compiles for specific input shapes (e.g., 800x800)
- Cache compiled functions per unique input shape
- All shapes become constants after first trace

This is simpler but requires re-tracing for each new input size.
