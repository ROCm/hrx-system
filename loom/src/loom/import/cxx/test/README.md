# Standalone C++ programs

These programs exercise the full source path: C++ import, ordinary Loom
bytecode linking, config specialization, AMDGPU compilation, and GPU execution.
The existing `iree-test-loom` runner compares outputs against independent
double-precision references and checks both output guards bitwise. Ordinary
integer functions also execute through the VM with exact scalar references.

```sh
iree-bazel-test --config=asan --config=loom-importer-cxx \
  --//runtime/config/hal:drivers=amdgpu,task \
  --//loom/config/emit:enable=amdgpu \
  //loom/src/loom/import/cxx/test:kernels_test

iree-bazel-test --config=asan --config=loom-importer-cxx \
  --//loom/config/target:enable=vm \
  //loom/src/loom/import/cxx/test:functions_test
```

The manifest imports each kernel through an external facade include root, so
the same test works with `--//loom/config/import/cxx:embed_includes=false`.
Launch count bounds become normal config declarations; the runner supplies
three workgroups through `--config` for the numerical kernels. The source
semantics kernels specify one workgroup. Scalar lane kernels use 64 threads;
explicit register-vector kernels use one thread.
The numerical tests explicitly permit approximate mathematical functions.
They test correctness and source compatibility, not kernel performance or
compatibility with complete upstream libraries.

| Source | Numerical coverage | Provenance |
| --- | --- | --- |
| `flash_attention.cpp` | Online softmax, shared key/value tiles, shuffle reductions, partial query/key tiles, and large score differences. | Original standalone f32 attention implementation, head dimension 64 and 16-key tiles. |
| `llama_rms_norm.cpp` | Two 32-lane reductions, shared reduction storage, and columns of length 1, 33, and 129. | [llama.cpp norm.cu](https://github.com/ggml-org/llama.cpp/blob/972d2313bc0bf0a45f634f77d95c9fb03aeab12c/ggml/src/ggml-cuda/norm.cu), MIT. |
| `aiter_swiglu_f16.cpp` | FP16 storage with f32 arithmetic, clamp extremes, reciprocal/exponential calls, and columns of length 1, 31, 65, and 129. | [aiter activation_kernels.cu](https://github.com/ROCm/aiter/blob/df95f04b703bfd7c520f072fcf2560092ec9d5ac/csrc/kernels/activation_kernels.cu), MIT. |
| `control_flow.cpp` | Pre-test, post-test, and nested loops; final scalar values and effectful helper calls in conditions. Seven trip counts including zero are checked bitwise. | Original source-language semantics witness. |
| `scheduled_sum.cpp` | Template-selected unroll factors 1/3 and pipeline depths 1/2 with linear ordering. Exact integer sums for 0, 1, 2, 5, 17, and 33 columns cover startup, tails, and drain under all four schedules. | Original scheduling-contract witness. |
| `short_circuit.cpp` | Bounds-guarded reads, exact conditional call-order traces through seven-comparison chains, discarded boolean expressions and scalar truth conversions across 64 lanes. Lengths 0, 1, 17, 33 and 64 run normally and with device access sanitization and zero expected access reports. | Original source-language semantics witness. |
| `early_returns.cpp` | Per-lane kernel exits, guarded helper returns inside a counted loop, nested return trees, void calls in returns, and local state across continuing paths. Lengths 0, 1, 17, 33 and 64 run normally and with device access sanitization; exited lanes retain their sentinel values. | Original source-language semantics witness. |
| `integer_functions.cpp` | Exact VM results for fixed-point multiply/rescale, byte increment/decrement, signed short decrement, 64-bit wrap and independently promoted shift counts. The 162 cases include negative rescaling, sign boundaries and counts 0/31/32/63. | Original source-language semantics witness using ordinary exported functions. |
| `comparison_functions.cpp` | Seven unparenthesized comparisons execute on the VM for all 128 truth combinations, zero inputs, and each argument at the unsigned maximum. | Original source-language parser and execution witness. |
| `assumptions.cpp` | Conjunctive and repeated bounds, capacity/stride constant expressions, templates, casts, unsigned wrap, unevaluated `sizeof`, scoped refinements and wide values. VM checks include every byte value; AMDGPU checks all outputs and guards, normally and with device access sanitization. | Original source-contract witness. |
| `vector_initializers.cpp` | Typed vector temporaries in returns, arguments, templates and nested expressions. VM lane checks and complete AMDGPU buffers cover empty/single/partial forms, narrow conversions, signed zero, zero-filled lanes and left-to-right initializer effects. | Original source-language initialization witness. |
| `integer_increment.cpp` | Uniform and lane-varying byte/64-bit increments execute on AMDGPU, including byte wrap, low-word carry, bit 63 and full-width wrap. Fifteen input cases check all 64 lanes and both output guards. | Original source-language semantics witness. |
| `increment_values.cpp` | Prefix/postfix results, byte/64-bit wrap, short-circuit and conditional mutations, and loop-condition updates. Exact VM oracles cover 532 scalar cases; native checks include complete pointer streams, skipped updates, unchanged inputs and output guards, normally and with device access sanitization. | Original source-language sequencing witness. |
| `pointer_walk.cpp` | Interior pointers through helper returns, conditional origins and counted/pre-test/post-test loops. Signed backward displacements, zero-trip behavior and final pointer positions are checked for seven lane-specific trip counts under three starting positions, normally and with device access sanitization. Inputs and output guards remain unchanged. | Original source storage-semantics witness. |
| `shaped_intrinsics.cpp` | Register-table lookups preserve integer values and floating-point bits, including signed zero, infinities and NaNs. Mixed-width dots cover all four byte-signedness pairs and wrapping i32 accumulators. The VM checks 264 scalar-return cases; AMDGPU checks complete buffers and guards for zero and nonzero group counts, normally and with device access sanitization. | Original shaped operation-binding witness. |

The llama.cpp extraction specializes `rms_norm_f32`, `block_reduce<SUM>` and
`warp_reduce_sum` for contiguous rows, one channel/sample, block size 64, and
no multiply/add fusion. Two shared values replace dynamic shared allocation.
The HIP PDL hooks are empty and omitted. A 64-bit row displacement forms input
and output row pointers before the reduction and normalization loops. The source
states the reduction loop's `offset < 32` invariant explicitly for subgroup index
analysis.

The aiter extraction specializes `swiglu_act_and_mul_kernel` for scalar vector
width and equal input/output types. It keeps the original clamp/arithmetic
order, AMD reciprocal, and OCML exponential spellings. `_Float16` storage and
explicit casts retain the input/output rounding points. This extraction uses
unsigned flat element indices; surrounding framework dispatch and vector memory
wrappers are omitted from both extractions. Launch annotations
supply the Loom configuration contract. The upstream licenses are included
beside the source files.

`loom/py/loom/gen/test/cxx_kernel_cases.py` generates deterministic, quantized
inputs and reference fixtures using only Python's standard library. Attention
uses a materialized score matrix and double-precision softmax reference; RMSNorm
uses a direct row sum of squares. These references are independent of the
source kernels' reduction and staging algorithms. Python runs only to prepare
test fixtures; the importer, CLI, and public C extension are native.

The integer references use unbounded integer multiplication/division and
explicit modular conversion into the source width. Shift counts remain within
the promoted left operand's width, and signed arithmetic inputs stay in their
defined domain. Arithmetic i64 right shifts are exercised on the VM.
