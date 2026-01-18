# jax-js compatibility table

jax-js strives for _approximate_ API compatibility with the JAX python library (and through that,
NumPy). But some features vary for a few reasons:

1. **Data model:** jax-js has _ownership_ of arrays using the `.ref` system, which obviates the need
   for APIs like `jit()`'s `donate_argnums` and `numpy.asarray()`.
2. **Language primitives:** JavaScript has no named arguments, so method call signatures may take
   objects instead of Python's keyword arguments. Also, PyTrees are translated in spirit to "JsTree"
   in jax-js, but their specification is different.
3. **Maturity:** JAX has various types like `complex64`, advanced functions like `hessenberg()`, and
   advanced higher-order features like `lax.while_loop()` that we haven't implemented. Some of these
   are not easy to implement on GPU.

Other features just aren't implemented yet. But those can probably be added easily!

In the tables below, we use a color legend to refer to functions in JAX:

- 🟢 = supported **(~45%)**
- 🟡 = supported, with API limitations **(~2%)**
- 🟠 = not supported, easy to add (<1 day) **(~35%)**
- 🔴 = not supported **(~18%)**
- ⚪️ = not applicable, will not be supported (see notes)

## [`jax`](https://docs.jax.dev/en/latest/jax.html)

[API docs](https://jax-js.com/docs/modules/_jax-js_jax.html) for these functions.

| API                  | Support | Notes                                           |
| -------------------- | ------- | ----------------------------------------------- |
| `config`             | ⚪️      | no separate config object                       |
| `default_device`     | 🟢      | devices are strings; there is only 1 GPU on web |
| `jit`                | 🟢      |                                                 |
| `make_jaxpr`         | 🟢      |                                                 |
| `eval_shape`         | 🟠      |                                                 |
| `ShapeDtypeStruct`   | 🟠      |                                                 |
| `device_put`         | 🟢      | async-only for performance                      |
| `device_get`         | ⚪️      | no separate "host" device                       |
| `default_backend`    | ⚪️      | XLA feature                                     |
| `named_call`         | ⚪️      | XLA feature                                     |
| `named_scope`        | ⚪️      | XLA feature                                     |
| `block_until_ready`  | 🟢      |                                                 |
| `copy_to_host_async` | ⚪️      | no separate "host" device                       |
| `make_mesh`          | ⚪️      | device sharding                                 |
| `set_mesh`           | ⚪️      | device sharding                                 |
| `grad`               | 🟢      |                                                 |
| `value_and_grad`     | 🟢      |                                                 |
| `jacobian`           | 🟢      |                                                 |
| `jacfwd`             | 🟢      |                                                 |
| `jacrev`             | 🟢      |                                                 |
| `hessian`            | 🟢      |                                                 |
| `jvp`                | 🟢      |                                                 |
| `linearize`          | 🟢      |                                                 |
| `linear_transpose`   | 🟠      |                                                 |
| `vjp`                | 🟢      |                                                 |
| `custom_gradient`    | 🔴      | core engine feature                             |
| `closure_convert`    | 🔴      | core engine feature                             |
| `checkpoint`         | 🔴      | core engine feature                             |
| `vmap`               | 🟡      | some ops do not have vmap support yet           |
| `shard_map`          | ⚪️      | device sharding                                 |
| `smap`               | ⚪️      | device sharding                                 |
| `pmap`               | ⚪️      | device sharding                                 |
| `devices`            | 🟢      | semantics differ, returns all devices           |
| `local_devices`      | ⚪️      | device sharding                                 |
| `process_index`      | ⚪️      | device sharding                                 |
| `device_count`       | ⚪️      | device sharding                                 |
| `local_device_count` | ⚪️      | device sharding                                 |
| `process_count`      | ⚪️      | device sharding                                 |
| `process_indices`    | ⚪️      | device sharding                                 |
| `custom_jvp`         | 🔴      | core engine feature                             |
| `custom_vjp`         | 🔴      | core engine feature                             |
| `custom_batching`    | 🔴      | core engine feature                             |
| `Array`              | 🟢      |                                                 |

Array primitives need to be called with methods like `a.add(b)` / `a.mul(b)` instead of `a + b` and
`a * b` as in Python, which has overloading.

Broadcasting is fully supported. Basic and advanced indexing can be done with `Array.slice()`.

Several other Array convenience methods are supported like `Array.min()` and `Array.sum()`, although
some of them are only available in the `jax.numpy` namespace. This is for performance and to
simplify the core Array prototype, since there's a bit of cruft there with esoteric methods like
`Array.ptp()` — feel free to submit an issue if you disagree.

## [`jax.numpy` module](https://docs.jax.dev/en/latest/jax.numpy.html)

**Data types:** We only support data types that can be efficiently worked with on the web.
[Type promotion](https://docs.jax.dev/en/latest/type_promotion.html) behaves similarly as in JAX,
with "weak types" baked into the compiler IR.
[Complex numbers](https://docs.jax.dev/en/latest/_autosummary/jax.lax.complex.html) are not
supported.

| Data type     | CPU (debug) | Wasm | WebGPU | WebGL(\*) | Notes              |
| ------------- | ----------- | ---- | ------ | --------- | ------------------ |
| `np.bool_`    | 🟢          | 🟢   | 🟢     | 🟢        |                    |
| `np.int8`     | 🟠          | 🟠   | 🟠     | 🟠        | requires emulation |
| `np.uint8`    | 🟠          | 🟠   | 🟠     | 🟠        | requires emulation |
| `np.int16`    | 🟠          | 🟠   | 🟠     | 🟠        | requires emulation |
| `np.uint16`   | 🟠          | 🟠   | 🟠     | 🟠        | requires emulation |
| `np.int32`    | 🟢          | 🟢   | 🟢     | 🟢        |                    |
| `np.uint32`   | 🟢          | 🟢   | 🟢     | 🟢        |                    |
| `np.bfloat16` | 🔴          | 🔴   | 🔴     | 🔴        | lacks support      |
| `np.float16`  | 🟢          | 🔴   | 🟢     | 🔴        | no wasm support    |
| `np.float32`  | 🟢          | 🟢   | 🟢     | 🟢        |                    |
| `np.float64`  | 🟢          | 🟢   | 🔴     | 🔴        | no webgpu support  |

_(\*) The WebGL backend is not guaranteed to be well-supported. It relies on fragment shaders and is
mostly meant for compatibility in browsers that don't support WebGPU, but you should use WebGPU when
possible._

Most operations behave the same way as they do in JAX.
[API docs](https://jax-js.com/docs/modules/_jax-js_jax.numpy.html).

| API                   | Support | Notes                                   |
| --------------------- | ------- | --------------------------------------- |
| `ndarray.at`          | ⚪️      | Python-specific                         |
| `abs`                 | 🟢      |                                         |
| `absolute`            | 🟢      |                                         |
| `acos`                | 🟢      |                                         |
| `acosh`               | 🟢      |                                         |
| `add`                 | 🟢      |                                         |
| `all`                 | 🟢      |                                         |
| `allclose`            | 🟡      | no `equal_nan` option                   |
| `amax`                | ⚪️      | alias of `max`                          |
| `amin`                | ⚪️      | alias of `min`                          |
| `angle`               | ⚪️      | complex numbers                         |
| `any`                 | 🟢      |                                         |
| `append`              | 🟠      |                                         |
| `apply_along_axis`    | 🟠      |                                         |
| `apply_over_axes`     | 🟠      |                                         |
| `arange`              | 🟢      |                                         |
| `arccos`              | 🟢      |                                         |
| `arccosh`             | 🟢      |                                         |
| `arcsin`              | 🟢      |                                         |
| `arcsinh`             | 🟢      |                                         |
| `arctan`              | 🟢      |                                         |
| `arctan2`             | 🟢      |                                         |
| `arctanh`             | 🟢      |                                         |
| `argmax`              | 🟢      |                                         |
| `argmin`              | 🟢      |                                         |
| `argpartition`        | 🟠      | sorting                                 |
| `argsort`             | 🟢      | sorting                                 |
| `argwhere`            | 🟠      | sorting                                 |
| `around`              | ⚪️      | alias of `round`                        |
| `array`               | 🟢      |                                         |
| `array_equal`         | 🔴      |                                         |
| `array_equiv`         | 🔴      |                                         |
| `array_repr`          | 🔴      |                                         |
| `array_split`         | 🟠      | `split` is supported                    |
| `array_str`           | 🔴      |                                         |
| `asarray`             | ⚪️      | alias of `array`                        |
| `asin`                | 🟢      |                                         |
| `asinh`               | 🟢      |                                         |
| `astype`              | 🟢      |                                         |
| `atan`                | 🟢      |                                         |
| `atanh`               | 🟢      |                                         |
| `atan2`               | 🟢      |                                         |
| `atleast_1d`          | ⚪️      | confusing, use `reshape`                |
| `atleast_2d`          | ⚪️      | confusing, use `reshape`                |
| `atleast_3d`          | ⚪️      | confusing, use `reshape`                |
| `average`             | 🟠      | `mean` is supported                     |
| `bartlett`            | 🟠      |                                         |
| `bincount`            | 🟠      |                                         |
| `bitwise_and`         | 🔴      |                                         |
| `bitwise_count`       | 🔴      |                                         |
| `bitwise_invert`      | 🔴      |                                         |
| `bitwise_left_shift`  | 🔴      |                                         |
| `bitwise_not`         | 🔴      |                                         |
| `bitwise_or`          | 🔴      |                                         |
| `bitwise_right_shift` | 🔴      |                                         |
| `bitwise_xor`         | 🔴      |                                         |
| `blackman`            | 🟠      |                                         |
| `block`               | 🟠      |                                         |
| `broadcast_arrays`    | 🟢      |                                         |
| `broadcast_shapes`    | 🟢      |                                         |
| `broadcast_to`        | 🟢      |                                         |
| `c_`                  | ⚪️      | Python-specific                         |
| `can_cast`            | 🟠      |                                         |
| `cbrt`                | 🟢      |                                         |
| `ceil`                | 🟢      |                                         |
| `choose`              | ⚪️      | confusing API                           |
| `clip`                | 🟢      |                                         |
| `column_stack`        | 🟢      |                                         |
| `compress`            | 🔴      |                                         |
| `concat`              | ⚪️      | use `concatenate`                       |
| `concatenate`         | 🟢      |                                         |
| `conj`                | ⚪️      | complex numbers                         |
| `conjugate`           | ⚪️      | complex numbers                         |
| `convolve`            | 🟢      | `lax.conv_general_dilated` is supported |
| `copy`                | ⚪️      | move semantics                          |
| `copysign`            | 🟠      |                                         |
| `corrcoef`            | 🟢      |                                         |
| `correlate`           | 🟢      | `lax.conv_general_dilated` is supported |
| `cos`                 | 🟢      |                                         |
| `cosh`                | 🟢      |                                         |
| `count_nonzero`       | 🟠      |                                         |
| `cov`                 | 🟢      |                                         |
| `cross`               | 🟠      |                                         |
| `cumprod`             | 🟠      |                                         |
| `cumsum`              | 🟡      | Quadratic-time                          |
| `cumulative_prod`     | 🟠      |                                         |
| `cumulative_sum`      | 🟡      | Quadratic-time                          |
| `deg2rad`             | 🟢      |                                         |
| `degrees`             | 🟢      |                                         |
| `delete`              | 🟠      |                                         |
| `diag`                | 🟢      |                                         |
| `diag_indices`        | 🟠      |                                         |
| `diag_indices_from`   | 🟠      |                                         |
| `diagflat`            | 🟠      |                                         |
| `diagonal`            | 🟢      |                                         |
| `diff`                | 🟠      |                                         |
| `digitize`            | 🟠      |                                         |
| `divide`              | 🟢      |                                         |
| `divmod`              | 🟢      |                                         |
| `dot`                 | 🟢      |                                         |
| `dsplit`              | 🟠      | `split` is supported                    |
| `dstack`              | 🟢      |                                         |
| `dtype`               | ⚪️      | can access `Array.dtype`                |
| `ediff1d`             | 🟠      |                                         |
| `einsum`              | 🟢      |                                         |
| `einsum_path`         | ⚪️      | path is currently private               |
| `empty`               | ⚪️      | use `zeros`                             |
| `empty_like`          | ⚪️      | use `zeros_like`                        |
| `equal`               | 🟢      |                                         |
| `exp`                 | 🟢      |                                         |
| `exp2`                | 🟢      |                                         |
| `expand_dims`         | 🟠      |                                         |
| `expm1`               | 🟡      | implemented as `exp(x)-1`               |
| `extract`             | 🔴      |                                         |
| `eye`                 | 🟢      |                                         |
| `fabs`                | ⚪️      | use `abs`                               |
| `fill_diagonal`       | 🟠      |                                         |
| `finfo`               | 🟢      |                                         |
| `fix`                 | ⚪️      | use `trunc`                             |
| `flatnonzero`         | 🔴      |                                         |
| `flip`                | 🟢      |                                         |
| `fliplr`              | 🟢      |                                         |
| `flipud`              | 🟢      |                                         |
| `float_power`         | 🟠      |                                         |
| `floor`               | 🟢      |                                         |
| `floor_divide`        | 🟢      |                                         |
| `fmax`                | 🟠      | use `maximum`                           |
| `fmin`                | 🟠      | use `minimum`                           |
| `fmod`                | 🟢      |                                         |
| `frexp`               | 🟢      |                                         |
| `frombuffer`          | 🟠      |                                         |
| `fromfile`            | ⚪️      | Python-specific                         |
| `fromfunction`        | 🟠      |                                         |
| `fromiter`            | ⚪️      | Python-specific                         |
| `frompyfunc`          | ⚪️      | Python-specific                         |
| `fromstring`          | ⚪️      | Python-specific                         |
| `from_dlpack`         | ⚪️      | Python-specific                         |
| `full`                | 🟢      |                                         |
| `full_like`           | 🟢      |                                         |
| `gcd`                 | 🔴      |                                         |
| `geomspace`           | 🟠      |                                         |
| `get_printoptions`    | ⚪️      | Python-specific                         |
| `gradient`            | 🟠      |                                         |
| `greater`             | 🟢      |                                         |
| `greater_equal`       | 🟢      |                                         |
| `hamming`             | 🟢      |                                         |
| `hanning`             | 🟢      |                                         |
| `heaviside`           | 🟢      |                                         |
| `histogram`           | 🔴      |                                         |
| `histogram_bin_edges` | 🔴      |                                         |
| `histogram2d`         | 🔴      |                                         |
| `histogramdd`         | 🔴      |                                         |
| `hsplit`              | 🟠      | `split` is supported                    |
| `hstack`              | 🟢      |                                         |
| `hypot`               | 🟡      | implemented as `sqrt(x^2 + y^2)`        |
| `i0`                  | 🔴      | transcendental                          |
| `identity`            | 🟢      |                                         |
| `iinfo`               | 🟢      |                                         |
| `imag`                | ⚪️      | complex numbers                         |
| `index_exp`           | ⚪️      | Python-specific                         |
| `indices`             | 🟠      |                                         |
| `inner`               | 🟢      |                                         |
| `insert`              | 🟠      |                                         |
| `interp`              | 🟠      |                                         |
| `intersect1d`         | 🔴      | sorting                                 |
| `invert`              | 🔴      |                                         |
| `isclose`             | ⚪️      | use `allclose`                          |
| `iscomplex`           | ⚪️      | complex numbers                         |
| `iscomplexobj`        | ⚪️      | complex numbers                         |
| `isdtype`             | 🟠      |                                         |
| `isfinite`            | 🟢      |                                         |
| `isin`                | 🔴      |                                         |
| `isinf`               | 🟢      |                                         |
| `isnan`               | 🟢      |                                         |
| `isneginf`            | 🟢      |                                         |
| `isposinf`            | 🟢      |                                         |
| `isreal`              | ⚪️      | complex numbers                         |
| `isrealobj`           | ⚪️      | complex numbers                         |
| `isscalar`            | 🟠      |                                         |
| `issubdtype`          | 🟠      |                                         |
| `iterable`            | ⚪️      | Python-specific                         |
| `ix_`                 | ⚪️      | Python-specific                         |
| `kaiser`              | 🔴      | transcendental                          |
| `kron`                | 🟠      |                                         |
| `lcm`                 | 🔴      |                                         |
| `ldexp`               | 🟢      |                                         |
| `left_shift`          | 🔴      |                                         |
| `less`                | 🟢      |                                         |
| `less_equal`          | 🟢      |                                         |
| `lexsort`             | 🔴      | sorting                                 |
| `linspace`            | 🟢      |                                         |
| `load`                | ⚪️      | file I/O                                |
| `log`                 | 🟢      |                                         |
| `log10`               | 🟢      |                                         |
| `log1p`               | 🟡      | implemented as `log(1+x)`               |
| `log2`                | 🟢      |                                         |
| `logaddexp`           | 🟠      |                                         |
| `logaddexp2`          | 🟠      |                                         |
| `logical_and`         | 🟢      |                                         |
| `logical_not`         | 🟢      |                                         |
| `logical_or`          | 🟢      |                                         |
| `logical_xor`         | 🟢      |                                         |
| `logspace`            | 🟢      |                                         |
| `mask_indices`        | 🟠      |                                         |
| `matmul`              | 🟢      |                                         |
| `matrix_transpose`    | 🟢      |                                         |
| `matvec`              | 🟠      |                                         |
| `max`                 | 🟢      |                                         |
| `maximum`             | 🟢      |                                         |
| `mean`                | 🟢      |                                         |
| `median`              | 🟠      | sorting                                 |
| `meshgrid`            | 🟢      |                                         |
| `mgrid`               | ⚪️      | Python-specific                         |
| `min`                 | 🟢      |                                         |
| `minimum`             | 🟢      |                                         |
| `mod`                 | ⚪️      | Skipped for clarity, use `remainder()`  |
| `modf`                | 🟠      |                                         |
| `moveaxis`            | 🟢      |                                         |
| `multiply`            | 🟢      |                                         |
| `nan_to_num`          | 🟠      |                                         |
| `nanargmax`           | 🟠      |                                         |
| `nanargmin`           | 🟠      |                                         |
| `nancumprod`          | 🟠      |                                         |
| `nancumsum`           | 🟠      |                                         |
| `nanmax`              | 🟠      |                                         |
| `nanmean`             | 🟠      |                                         |
| `nanmedian`           | 🟠      | sorting                                 |
| `nanmin`              | 🟠      |                                         |
| `nanpercentile`       | 🟠      | sorting                                 |
| `nanprod`             | 🟠      |                                         |
| `nanquantile`         | 🟠      | sorting                                 |
| `nanstd`              | 🟠      |                                         |
| `nansum`              | 🟠      |                                         |
| `nanvar`              | 🟠      |                                         |
| `ndarray`             | 🟢      | just `Array` in jax-js                  |
| `ndim`                | 🟢      |                                         |
| `negative`            | 🟢      |                                         |
| `nextafter`           | 🔴      |                                         |
| `nonzero`             | 🔴      |                                         |
| `not_equal`           | 🟢      |                                         |
| `ogrid`               | ⚪️      | Python-specific                         |
| `ones`                | 🟢      |                                         |
| `ones_like`           | 🟢      |                                         |
| `outer`               | 🟢      |                                         |
| `packbits`            | ⚪️      | no uint8 support                        |
| `pad`                 | 🟢      |                                         |
| `partition`           | 🟠      | sorting                                 |
| `percentile`          | 🟠      | sorting                                 |
| `permute_dims`        | 🟢      |                                         |
| `piecewise`           | 🔴      | `lax.switch` control flow               |
| `place`               | 🔴      |                                         |
| `poly`                | 🔴      |                                         |
| `polyadd`             | 🟠      |                                         |
| `polyder`             | 🟠      |                                         |
| `polydiv`             | 🔴      |                                         |
| `polyfit`             | 🔴      |                                         |
| `polyint`             | 🟠      |                                         |
| `polymul`             | 🟠      |                                         |
| `polysub`             | 🟠      |                                         |
| `polyval`             | 🟠      |                                         |
| `positive`            | 🟢      |                                         |
| `pow`                 | 🟢      |                                         |
| `power`               | 🟢      |                                         |
| `printoptions`        | ⚪️      | Python-specific                         |
| `prod`                | 🟢      |                                         |
| `promote_types`       | 🟢      |                                         |
| `ptp`                 | 🟢      |                                         |
| `put`                 | 🟠      |                                         |
| `put_along_axis`      | 🟠      |                                         |
| `quantile`            | 🟠      | sorting                                 |
| `r_`                  | ⚪️      | Python-specific                         |
| `rad2deg`             | 🟢      |                                         |
| `radians`             | 🟢      |                                         |
| `ravel`               | 🟢      |                                         |
| `ravel_multi_index`   | 🟠      |                                         |
| `real`                | ⚪️      | complex numbers                         |
| `reciprocal`          | 🟢      |                                         |
| `remainder`           | 🟢      |                                         |
| `repeat`              | 🟢      |                                         |
| `reshape`             | 🟢      |                                         |
| `resize`              | 🟠      |                                         |
| `result_type`         | 🟠      | see `promote_types`                     |
| `right_shift`         | 🔴      |                                         |
| `rint`                | 🟠      |                                         |
| `roll`                | 🟠      |                                         |
| `rollaxis`            | 🟠      |                                         |
| `roots`               | 🔴      |                                         |
| `rot90`               | 🟠      |                                         |
| `round`               | 🟠      |                                         |
| `s_`                  | ⚪️      | Python-specific                         |
| `save`                | ⚪️      | file I/O                                |
| `savez`               | ⚪️      | file I/O                                |
| `searchsorted`        | 🔴      | sorting                                 |
| `select`              | 🟠      |                                         |
| `set_printoptions`    | ⚪️      | Python-specific                         |
| `setdiff1d`           | ⚪️      | Python-specific                         |
| `setxor1d`            | ⚪️      | Python-specific                         |
| `shape`               | 🟢      |                                         |
| `sign`                | 🟢      |                                         |
| `signbit`             | 🔴      |                                         |
| `sin`                 | 🟢      |                                         |
| `sinc`                | 🟡      | JVP not supported at x=0                |
| `sinh`                | 🟢      |                                         |
| `size`                | 🟢      |                                         |
| `sort`                | 🟢      | sorting                                 |
| `sort_complex`        | ⚪️      | complex numbers                         |
| `spacing`             | 🔴      |                                         |
| `split`               | 🟢      |                                         |
| `sqrt`                | 🟢      |                                         |
| `square`              | 🟢      |                                         |
| `squeeze`             | 🟢      |                                         |
| `stack`               | 🟢      |                                         |
| `std`                 | 🟢      |                                         |
| `subtract`            | 🟢      |                                         |
| `sum`                 | 🟢      |                                         |
| `swapaxes`            | 🟢      |                                         |
| `take`                | 🟢      |                                         |
| `take_along_axis`     | 🟠      |                                         |
| `tan`                 | 🟢      |                                         |
| `tanh`                | 🟢      |                                         |
| `tensordot`           | 🟢      |                                         |
| `tile`                | 🟢      |                                         |
| `trace`               | 🟢      |                                         |
| `trapezoid`           | 🟠      |                                         |
| `transpose`           | 🟢      |                                         |
| `tri`                 | 🟢      |                                         |
| `tril`                | 🟢      |                                         |
| `tril_indices`        | 🟠      |                                         |
| `tril_indices_from`   | 🟠      |                                         |
| `trim_zeros`          | 🟠      |                                         |
| `triu`                | 🟢      |                                         |
| `triu_indices`        | 🟠      |                                         |
| `triu_indices_from`   | 🟠      |                                         |
| `true_divide`         | 🟢      |                                         |
| `trunc`               | 🟢      |                                         |
| `ufunc`               | ⚪️      | Python-specific                         |
| `union1d`             | 🔴      | sorting                                 |
| `unique`              | 🔴      | sorting                                 |
| `unique_all`          | 🔴      | sorting                                 |
| `unique_counts`       | 🔴      | sorting                                 |
| `unique_inverse`      | 🔴      | sorting                                 |
| `unique_values`       | 🔴      | sorting                                 |
| `unpackbits`          | ⚪️      | no uint8 support                        |
| `unravel_index`       | 🟠      |                                         |
| `unstack`             | 🟠      |                                         |
| `unwrap`              | 🔴      |                                         |
| `vander`              | 🟠      |                                         |
| `var`                 | 🟢      |                                         |
| `vdot`                | 🟢      |                                         |
| `vecdot`              | 🟢      |                                         |
| `vecmat`              | 🟠      |                                         |
| `vectorize`           | 🟠      |                                         |
| `vsplit`              | 🟠      | `split` is supported                    |
| `vstack`              | 🟢      |                                         |
| `where`               | 🟢      |                                         |
| `zeros`               | 🟢      |                                         |
| `zeros_like`          | 🟢      |                                         |

## [`jax.numpy.fft` module](https://docs.jax.dev/en/latest/jax.numpy.html#module-jax.numpy.fft)

Basic FFT is supported, but there is no `complex64` data type in the library. All FFT routines take
in pairs of real and imaginary parts.

| API         | Support | Notes            |
| ----------- | ------- | ---------------- |
| `fft`       | 🟡      | only powers of 2 |
| `fft2`      | 🟠      |                  |
| `fftfreq`   | 🟠      |                  |
| `fftn`      | 🟠      |                  |
| `fftshift`  | 🟠      |                  |
| `hfft`      | 🟠      |                  |
| `ifft`      | 🟡      | only powers of 2 |
| `ifft2`     | 🟠      |                  |
| `ifftn`     | 🟠      |                  |
| `ifftshift` | 🟠      |                  |
| `ihfft`     | 🟠      |                  |
| `irfft`     | 🟠      |                  |
| `irfft2`    | 🟠      |                  |
| `irfftn`    | 🟠      |                  |
| `rfft`      | 🟠      |                  |
| `rfft2`     | 🟠      |                  |
| `rfftfreq`  | 🟠      |                  |
| `rfftn`     | 🟠      |                  |

## [`jax.numpy.linalg` module](https://docs.jax.dev/en/latest/jax.numpy.html#module-jax.numpy.linalg)

Similarly, the `linalg` module has some very important operations for linear algebra and matrices.
Most of these will be tricky to implement as routines with backend-specific lowering. We have
Cholesky but are missing other building blocks like:

- LU decomposition (solver)
- Householder iteration (QR, SVD, eigenvalues)

| API                | Support | Notes                                   |
| ------------------ | ------- | --------------------------------------- |
| `cholesky`         | 🟢      |                                         |
| `cond`             | 🔴      |                                         |
| `cross`            | 🟠      |                                         |
| `det`              | 🟢      |                                         |
| `diagonal`         | 🟢      |                                         |
| `eig`              | 🔴      |                                         |
| `eigh`             | 🔴      |                                         |
| `eigvals`          | 🔴      |                                         |
| `eigvalsh`         | 🔴      |                                         |
| `inv`              | 🟢      |                                         |
| `lstsq`            | 🟡      | Cholesky-based, less stable than QR/SVD |
| `matmul`           | 🟢      |                                         |
| `matrix_norm`      | 🟠      |                                         |
| `matrix_power`     | 🟢      |                                         |
| `matrix_rank`      | 🔴      |                                         |
| `matrix_transpose` | 🟢      |                                         |
| `multi_dot`        | 🟠      |                                         |
| `norm`             | 🟠      |                                         |
| `outer`            | 🟢      |                                         |
| `pinv`             | 🔴      |                                         |
| `qr`               | 🔴      |                                         |
| `slogdet`          | 🟢      |                                         |
| `solve`            | 🟢      |                                         |
| `svd`              | 🔴      |                                         |
| `svdvals`          | 🔴      |                                         |
| `tensordot`        | 🟢      |                                         |
| `tensorinv`        | 🔴      |                                         |
| `tensorsolve`      | 🔴      |                                         |
| `trace`            | 🟢      |                                         |
| `vector_norm`      | 🟠      |                                         |
| `vecdot`           | 🟢      |                                         |

## [`jax.lax` module](https://docs.jax.dev/en/latest/jax.lax.html)

Only a few functions in `jax.lax` have been implemented, notably `conv_general_dilated()` for
convolutions and `dot()` for general tensor contractions. Also, `linalg.triangular_solve()` is
available.

In the future, the library may need a rework to add support for `lax` operations, which are
lower-level (semantics-wise, they don't do automatic type promotion). The reason why jax-js did not
start from `lax` is because JAX is built on XLA as foundations and started with `lax` wrappers, but
jax-js was built from scratch.

## [`jax.random` module](https://docs.jax.dev/en/latest/jax.random.html)

JAX uses a [Threefry2x32](https://docs.jax.dev/en/latest/jep/263-prng.html) random number generator.
jax-js implements the same PRNG, with bitwise identical outputs. However, most samplers in the
`random` module have not been implemented yet, these can be added easily.

| API             | Support | Notes                         |
| --------------- | ------- | ----------------------------- |
| `key`           | 🟢      | only 32-bit seeding right now |
| `key_data`      | ⚪️      | keys are just uint32 arrays   |
| `wrap_key_data` | ⚪️      | keys are just uint32 arrays   |
| `fold_in`       | 🟠      |                               |
| `split`         | 🟢      | not vmappable yet             |
| `clone`         | ⚪️      | use `.ref`                    |
| `PRNGKey`       | ⚪️      | legacy                        |

**Samplers:** These are all 🟠 assuming that sampling from distributions is usually easier than
modeling their transcendental CDFs (e.g., normal via Box-Muller).

| API                    | Support | Notes |
| ---------------------- | ------- | ----- |
| `ball`                 | 🟠      |       |
| `bernoulli`            | 🟢      |       |
| `beta`                 | 🟠      |       |
| `binomial`             | 🟠      |       |
| `bits`                 | 🟢      |       |
| `categorical`          | 🟠      |       |
| `cauchy`               | 🟢      |       |
| `chisquare`            | 🟠      |       |
| `choice`               | 🟠      |       |
| `dirichlet`            | 🟠      |       |
| `double_sided_maxwell` | 🟠      |       |
| `exponential`          | 🟢      |       |
| `f`                    | 🟠      |       |
| `gamma`                | 🟠      |       |
| `generalized_normal`   | 🟠      |       |
| `geometric`            | 🟠      |       |
| `gumbel`               | 🟢      |       |
| `laplace`              | 🟢      |       |
| `loggamma`             | 🟠      |       |
| `logistic`             | 🟠      |       |
| `lognormal`            | 🟠      |       |
| `maxwell`              | 🟠      |       |
| `multinomial`          | 🟠      |       |
| `multivariate_normal`  | 🟢      |       |
| `normal`               | 🟢      |       |
| `orthogonal`           | 🟠      |       |
| `pareto`               | 🟠      |       |
| `permutation`          | 🟠      |       |
| `poisson`              | 🟠      |       |
| `rademacher`           | 🟠      |       |
| `randint`              | 🟠      |       |
| `rayleigh`             | 🟠      |       |
| `t`                    | 🟠      |       |
| `triangular`           | 🟠      |       |
| `truncated_normal`     | 🟠      |       |
| `uniform`              | 🟢      |       |
| `wald`                 | 🟠      |       |
| `weibull_min`          | 🟠      |       |

## [`jax.nn` module](https://docs.jax.dev/en/latest/jax.nn.html)

These provide basic helpers for neural networks, though it falls short of a full "neural network
framework" like `torch.nn.Module`. Thinking of trying to port an API like
[Equinox](https://github.com/patrick-kidger/equinox) under the jax-js namespace as well, although it
would need substantial changes to work well in JavaScript.

**Activation functions:**

| API              | Support | Notes |
| ---------------- | ------- | ----- |
| `relu`           | 🟢      |       |
| `relu6`          | 🟢      |       |
| `sigmoid`        | 🟢      |       |
| `softplus`       | 🟢      |       |
| `sparse_plus`    | 🟢      |       |
| `sparse_sigmoid` | 🟢      |       |
| `soft_sign`      | 🟢      |       |
| `silu`           | 🟢      |       |
| `swish`          | 🟢      |       |
| `log_sigmoid`    | 🟢      |       |
| `leaky_relu`     | 🟢      |       |
| `hard_sigmoid`   | 🟢      |       |
| `hard_silu`      | 🟢      |       |
| `hard_swish`     | 🟢      |       |
| `hard_tanh`      | 🟢      |       |
| `tanh`           | 🟢      |       |
| `elu`            | 🟢      |       |
| `celu`           | 🟢      |       |
| `selu`           | 🟢      |       |
| `gelu`           | 🟢      |       |
| `glu`            | 🟢      |       |
| `squareplus`     | 🟢      |       |
| `mish`           | 🟢      |       |
| `identity`       | 🟢      |       |

**Other functions:**

| API                             | Support | Notes                                             |
| ------------------------------- | ------- | ------------------------------------------------- |
| `softmax`                       | 🟢      |                                                   |
| `log_softmax`                   | 🟢      |                                                   |
| `logmeanexp`                    | 🟢      |                                                   |
| `logsumexp`                     | 🟢      |                                                   |
| `standardize`                   | 🟢      |                                                   |
| `one_hot`                       | 🟢      |                                                   |
| `dot_product_attention`         | 🟢      | correct, but can custom-optimize (FlashAttention) |
| `scaled_matmul`                 | 🟠      | for microscaling                                  |
| `get_scaled_dot_general_config` | 🔴      |                                                   |
| `scaled_dot_general`            | 🟠      | for microscaling                                  |
| `log1mexp`                      | 🟠      |                                                   |

## Other `jax.*` modules

The `jax.tree` module is available but differs significantly in how it is implemented. JsTree is
based on nested JavaScript objects and arrays similar to JSON format, and it has generic TypeScript
bindings.

These modules are unimplemented:

- `jax.scipy`
- `jax.sharding`
- `jax.debug`
- `jax.dlpack`
- `jax.distributed`
- `jax.dtypes`
- `jax.ffi`
- `jax.flatten_util`
- `jax.image`
- `jax.ops`
- `jax.profiler`
- `jax.ref`
- `jax.stages`
- `jax.test_util`
- `jax.tree_util`
- `jax.typing`
- `jax.export`
- `jax.extend`
- `jax.example_libraries`
- `jax.experimental`

## [`optax`](https://optax.readthedocs.io/en/latest/index.html)

We have ported a subset of the [Optax](https://github.com/google-deepmind/optax) gradient processing
and optimization library at `@jax-js/optax`. You can install this alongside `@jax-js/jax`.

```bash
npm i @jax-js/optax
```

[API docs](https://jax-js.com/docs/modules/_jax-js_optax.html). Currently, the following optimizers
are supported:

- SGD
- Adam
