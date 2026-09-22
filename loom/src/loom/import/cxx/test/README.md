# C++ importer tests

Compiler behavior is authored in `.cxx-test` files and checked through the same
`loom-check` modes, diagnostics, expectations, and updates as `.loom-test` files.
Each `//====` case supplies C++ source; `//----` introduces its expected Loom
output. `// ERROR` annotations check source diagnostics. `// INPUT:` selects
import options such as the language standard, data model, or exported roots.
The [loom-check guide](../../../tools/loom-check/README.md) describes the format
and `--update` workflow.

```sh
iree-bazel-test --config=asan --config=loom-importer-cxx \
  //loom/src/loom/import/cxx/test:compiler_test
```

Program lowering and source rejection cases belong in these files. Native C++
tests exercise API contracts such as callback failures, source lifetimes,
binding identity, and retained analysis facts.

## Source execution tests

These programs exercise the full source path: C++ import, ordinary Loom
bytecode linking, config specialization, AMDGPU compilation, and GPU execution.
The existing `iree-test-loom` runner compares outputs against independent
double-precision references and checks both output guards bitwise. Ordinary
integer functions also execute through the VM with exact scalar references.
Host references generate C++ translation units containing `LOOM_CHECK_CASE`
bodies and include the implementation under test. Each kernel source file and
its generated Loom references form one reusable `loom_test_module` here, with
its runtime arrays and import options. AMDGPU qualification consumes these
modules through `loom_test` in the
[target test package](../../../tooling/target/amdgpu/test/cxx/BUILD.bazel).
The [source authoring guide](../README.md#executable-checks-and-benchmarks)
shows handwritten cases and benchmarks.

```sh
iree-bazel-test --config=asan --config=loom-importer-cxx \
  --//loom/config/target:enable=amdgpu \
  --//runtime/config/hal:drivers=amdgpu \
  --//loom/config/emit:enable=amdgpu \
  --//loom/config/execute:enable=iree_hal \
  //loom/src/loom/tooling/target/amdgpu/test/cxx:kernels_test

iree-bazel-test --config=asan --config=loom-importer-cxx \
  --//loom/config/target:enable=vm \
  //loom/src/loom/import/cxx/test:functions_test
```

Build declarations list source files; host reference selection and test names
follow each filename. GPU sources pair with their reference module and fixture
directory. All kernels in a source file share that module; the cases select the
entries they exercise. Import uses the normal facade headers without per-kernel
root flags or header-path overrides. Generated directories are declared action
outputs and retained as test data, with NPY paths relative to the check source.
Schedule variants are named C++ entry points, and the reference code exercises
each entry against the same numerical oracle. Numerical reference modules bind
three workgroups with ordinary `config.def` values. The source semantics kernels
specify one workgroup. Scalar lane kernels use 64 threads; explicit
register-vector kernels use one thread.
The numerical tests explicitly permit approximate mathematical functions.
They test correctness and source compatibility, not kernel performance or
compatibility with complete upstream libraries.

`functions_test` here aggregates scalar-result VM cases. The AMDGPU package's
`kernels_test` qualifies its selected source modules normally and with device
access sanitization. Every case checks for zero access reports. Individual
targets such as `integer_functions_test` here, and
`structured_continue_test_execute_amdgpu_test` and
`structured_continue_test_execute_amdgpu_access_test` in the target package,
can be run directly. Both native profiles share one imported and linked module,
with reference arrays retained by its source owner. Device availability gates
execution, while the modules and reference generation remain available without
AMDGPU or HAL. A target explicitly selects the modules it supports; adding a
source does not automatically claim support on every backend. Access
instrumentation applies to correctness and benchmark smoke alike. Compiler
rejection witnesses live in `.cxx-test`, including scheduled-loop lowering and
unsupported VM aggregate transport. The corpus has no JSON execution manifests;
JSON fixtures under
`tooling/` exercise CLI options, report fields, and process exit behavior.

| Source | Numerical coverage | Provenance |
| --- | --- | --- |
| `flash_attention.cpp` | Online softmax, shared key/value tiles, shuffle reductions, partial query/key tiles, and large score differences. | Original standalone f32 attention implementation, head dimension 64 and 16-key tiles. |
| `llama_rms_norm.cpp` | Two 32-lane reductions, shared reduction storage, and columns of length 1, 33, and 129. | [llama.cpp norm.cu](https://github.com/ggml-org/llama.cpp/blob/972d2313bc0bf0a45f634f77d95c9fb03aeab12c/ggml/src/ggml-cuda/norm.cu), MIT. |
| `aiter_swiglu_f16.cpp` | FP16 storage with f32 arithmetic, clamp extremes, reciprocal/exponential calls, and columns of length 1, 31, 65, and 129. | [aiter activation_kernels.cu](https://github.com/ROCm/aiter/blob/df95f04b703bfd7c520f072fcf2560092ec9d5ac/csrc/kernels/activation_kernels.cu), MIT. |
| `control_flow.cpp` | Pre-test, post-test, and nested loops; final scalar values and effectful helper calls in conditions. Seven trip counts including zero are checked bitwise. | Original source-language semantics witness. |
| `scheduled_sum.cpp` | Unroll factors 1/3 and pipeline depths 1/2 with linear ordering. Exact integer sums for 0, 1, 2, 5, 17, and 33 columns cover startup, tails, and drain under all four schedules. | Original scheduling-contract witness. |
| `short_circuit.cpp` | Bounds-guarded reads, exact conditional call-order traces through seven-comparison chains, discarded boolean expressions and scalar truth conversions across 64 lanes. Lengths 0, 1, 17, 33 and 64 run normally and with device access sanitization and zero expected access reports. | Original source-language semantics witness. |
| `early_returns.cpp` | Per-lane kernel exits, guarded helper returns inside a counted loop, nested return trees, void calls in returns, and local state across continuing paths. Lengths 0, 1, 17, 33 and 64 run normally and with device access sanitization; exited lanes retain their sentinel values. | Original source-language semantics witness. |
| `integer_functions.cpp` | Exact VM results for fixed-point multiply/rescale, byte increment/decrement, signed short decrement, 64-bit wrap and independently promoted shift counts. The 162 cases include negative rescaling, sign boundaries and counts 0/31/32/63. | Original source-language semantics witness using ordinary exported functions. |
| `enum_values.cpp` | Named constants, signed and unsigned casts, comparisons, boolean enums, inferred 64-bit storage, packed enum promotion, and template-dependent definitions execute in 881 exact VM cases. Twelve AMDGPU cases exercise 8/16/32/64-bit enum parameters and pointer storage through ordinary helper calls, including high bits and wraparound, with unchanged inputs and output guards checked normally and with device access sanitization. | Original source-language representation witness. |
| `comparison_functions.cpp` | Seven unparenthesized comparisons execute on the VM for all 128 truth combinations, zero inputs, and each argument at the unsigned maximum. | Original source-language parser and execution witness. |
| `constexpr_values.cpp` | Source-selected returns and continues, discarded unsupported statements and loop-bound writes, zero-trip loops, and ordinary/constexpr initializers execute on the VM with exact scalar expectations. | Original source-selection and sequencing witness. |
| `assumptions.cpp` | Conjunctive and repeated bounds, capacity/stride constant expressions, templates, casts, unsigned wrap, unevaluated `sizeof`, scoped refinements and wide values. VM checks include every byte value; AMDGPU checks all outputs and guards, normally and with device access sanitization. | Original source-contract witness. |
| `vector_initializers.cpp` | Typed vector temporaries in returns, arguments, templates and nested expressions. VM lane checks and complete AMDGPU buffers cover empty/single/partial forms, narrow conversions, signed zero, zero-filled lanes and left-to-right initializer effects. | Original source-language initialization witness. |
| `integer_increment.cpp` | Uniform and lane-varying byte/64-bit increments execute on AMDGPU, including byte wrap, low-word carry, bit 63 and full-width wrap. Fifteen input cases check all 64 lanes and both output guards. | Original source-language semantics witness. |
| `increment_values.cpp` | Prefix/postfix results, byte/64-bit wrap, short-circuit and conditional mutations, and loop-condition updates. Exact VM oracles cover 532 scalar cases; native checks include complete pointer streams, skipped updates, unchanged inputs and output guards, normally and with device access sanitization. | Original source-language sequencing witness. |
| `structured_continue.cpp` | Conditional iteration exits, shared tails, nested loop targets, shadowed bindings, for increments and pre/post-test conditions. Exact VM references cover 350 scalar cases. Native cases preserve sparse destinations, compacted pointer streams, vector recurrences and filtered reads under four unroll/pipeline schedules, with unchanged inputs and output guards checked normally and with device access sanitization. | Original source-language control witness. |
| `pointer_walk.cpp` | Interior pointers through helper returns, conditional origins and counted/pre-test/post-test loops. Signed backward displacements, zero-trip behavior and final pointer positions are checked for seven lane-specific trip counts under three starting positions, normally and with device access sanitization. Inputs and output guards remain unchanged. | Original source storage-semantics witness. |
| `shaped_intrinsics.cpp` | Register-table lookups preserve integer values and floating-point bits, including signed zero, infinities and NaNs. Mixed-width dots cover all four byte-signedness pairs and wrapping i32 accumulators. The VM checks 264 scalar-return cases; AMDGPU checks complete buffers and guards for zero and nonzero group counts, normally and with device access sanitization. | Original shaped operation-binding witness. |
| `record_values.cpp` | Empty, nested, scalar, vector and pointer records through construction, member mutation, independent copies, helpers, conditional values and loop state. Twenty-four exact VM cases and sixty AMDGPU cases include modular overflow, zero trips, defaults/designators and sequencing. Native cases preserve input buffers and output guards, normally and with access sanitization. Record and explicit-leaf controls express the same algorithm for code/resource comparison. | Original source value and callable-boundary witness. |
| `symbol_exports.cpp` | A kernel and helper use explicit Loom names inherited through source redeclarations. Authored Loom links and launches the renamed kernel for 32 exact modular integer cases, with unchanged inputs and output guards checked normally and with device access sanitization. | Original source-symbol identity witness. |
| `typed_views.cpp` | A dynamic-row/static-column source view preserves a runtime row stride and nonzero pointer origin through a by-value helper, then copies into a dense view at another interior origin. Constexpr specializations also deduce distinct static/dynamic shape types and dense/strided layouts through generic readers. Zero and positive shapes, row padding, unchanged inputs, internal sentinels, outer guards, and device access reports are checked independently. | Original typed-view descriptor and source-lifetime witness. |
| `storage_updates.cpp` | Guarded storage executes bitmap compound assignment, RHS/address sequencing, pointer and element increments, narrow signed/unsigned conversion and floating arithmetic through both the ordinary VM callable ABI and native kernel launches. Volatile updates use the same helper specializations. Native cases assert exact results, input/output guards and zero device access reports. | Original source read/modify/write witness. |
| `volatile_memory.cpp` | Qualified pointers, pointer members, helpers, vector storage, workgroup arrays and typed subviews execute on AMDGPU with exact results, unchanged inputs, guards and access checking. The ordinary-access control uses the same algorithm. A VM caller supplies real object storage to the same C++ helper and verifies results, input preservation and guards. Compiler fixtures independently check repeated and discarded observations through cleanup. | Original source memory-observation witness. |
| `atomics.cpp` | Contended device and workgroup ticket allocation, signed and unsigned combining operations, and CAS success/failure with exact old values, final storage and guards. The VM exercises every integer kind at 32 and 64 bits; native cases select the target's implemented widths and kinds, normally and with access sanitization. Interior and volatile pointers retain their storage identity. | Original source atomic-memory witness. |
| `atomic_builtins.cpp` | GCC exchange, old/new fetch results, signed wraparound, and 32/64-bit CAS. Independent argument counters and aliased expected storage expose duplicate evaluation or an erroneous store on successful CAS. VM and native execution check results, expected storage, and guards; native also runs with access sanitization. | Header-free integer atomic witness. |

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
