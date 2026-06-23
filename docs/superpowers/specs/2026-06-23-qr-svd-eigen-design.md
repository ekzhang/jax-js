# QR Algorithm For SVD And Eigenvalues

## Goal

Implement the two algorithmic building blocks requested in
`ekzhang/jax-js#51`:

- Householder reductions: Hessenberg form for square eigenvalue problems and
  bidiagonal form for rectangular SVD problems.
- Repeated shifted QR iterations on the reduced matrices.

The first implementation targets real-valued floating point arrays and follows
the existing jax-js routine model: CPU has the reference implementation, WASM
can reuse CPU routines, and WebGPU support can be added as an optimized routine
only after the CPU behavior is stable.

## Public Surface

Expose SVD first through `jax.numpy.linalg.svd(a, opts?)`, because bidiagonal QR
has a real-valued output contract for real inputs.

Supported options:

- `computeUv?: boolean`, default `true`.
- `fullMatrices?: boolean`, default `true`.

Return shape follows NumPy/JAX conventions:

- If `computeUv` is true: `[u, s, vh]`.
- If `computeUv` is false: `s`.

For eigenvalues, keep the public API conservative:

- Add `jax.numpy.linalg.eigvals(a)` only for real eigenvalues found by the real
  QR routine.
- Detect unreduced `2x2` blocks that represent complex conjugate pairs and
  throw a clear error until complex dtype or an explicit real/imag return API is
  designed.

This still implements Hessenberg reduction and QR iteration, while avoiding an
unstable public complex-number contract.

## Internal Architecture

Add one routine-level primitive for SVD and one for eigenvalues:

- `Primitive.SVD` / `Routines.SVD`
- `Primitive.Eigvals` / `Routines.Eigvals`

The CPU routine owns the numerical algorithms:

1. Convert each matrix in the batch to a local mutable `number[]`.
2. For SVD:
   - Use Householder reflections to reduce `A` to bidiagonal form.
   - Accumulate left and right orthogonal factors when `computeUv` is true.
   - Apply implicit shifted QR steps to the bidiagonal matrix until off-diagonal
     entries deflate.
   - Sort singular values descending and permute singular vectors to match.
3. For eigenvalues:
   - Use Householder reflections to reduce `A` to upper Hessenberg form.
   - Apply shifted QR iteration until subdiagonal entries deflate.
   - Read real eigenvalues from `1x1` diagonal blocks.
   - Throw on remaining `2x2` complex blocks.

The implementation should keep helpers local to `src/routine.ts` unless the file
becomes unwieldy. If extraction is needed, create `src/routine/linalg.ts` and
keep `runCpuRoutine()` as the dispatch owner.

## Shape And Dtype Rules

SVD:

- Input must be at least 2D.
- Last two dimensions are matrix dimensions `m, n`.
- Batched inputs are supported by iterating over leading dimensions.
- Output singular values have shape `[..., min(m, n)]`.
- With `fullMatrices: true`, `u` has shape `[..., m, m]` and `vh` has shape
  `[..., n, n]`.
- With `fullMatrices: false`, `u` has shape `[..., m, k]` and `vh` has shape
  `[..., k, n]`, where `k = min(m, n)`.
- Floating dtypes preserve the input dtype where possible. Integer inputs should
  be rejected or promoted consistently with existing linalg behavior.

Eigenvalues:

- Input must be at least 2D and square on the last two axes.
- Batched square matrices are supported.
- Output has shape `[..., n]`.
- The first version returns real-valued eigenvalues only.

## Error Handling And Convergence

Use a scale-aware tolerance based on machine epsilon and matrix norm. QR
iteration should have a fixed iteration cap per active block. On failure, throw
a descriptive error such as:

- `svd: QR iteration failed to converge`
- `eigvals: QR iteration failed to converge`
- `eigvals: complex eigenvalues are not supported yet`

## Testing

Use test-driven development.

Add focused Vitest coverage:

- SVD reconstructs square, tall, wide, diagonal, zero, and batched matrices.
- `computeUv: false` returns only singular values.
- Singular values are sorted descending and nonnegative.
- `fullMatrices: false` produces reduced shapes.
- SVD agrees with known closed-form small examples.
- Eigenvalues match diagonal, triangular, symmetric, and simple real-spectrum
  nonsymmetric matrices.
- Eigenvalues throw on a real rotation matrix with complex eigenvalues.
- Invalid shapes throw clear errors.

Run at least:

- `pnpm test test/numpy-linalg.test.ts`
- `pnpm check`

If WebGPU lacks the new routine initially, either skip WebGPU tests for these
specific APIs or mark the routine unsupported on WebGPU with a clear error. CPU
and WASM should pass because WASM falls back to CPU routines.

## Non-Goals

- Full complex dtype support.
- Differentiation rules for SVD/eigvals in the first pass.
- Optimized WebGPU kernels for QR iteration.
- Full `np.linalg.eig` eigenvectors for nonsymmetric matrices.
