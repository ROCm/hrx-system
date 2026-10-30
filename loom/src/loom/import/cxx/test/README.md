# C++ importer tests

Compiler behavior is authored in `.cxx-test` files and checked through the same
`loom-check` modes, diagnostics, expectations, and updates as `.loom-test` files.
Each file owns one coherent language or lowering family in one harness mode.
Related cases extend that file through `//====` sections; a new file represents
a distinct semantic family or run mode rather than one additional witness.
Each `//====` case supplies C++ source; `//----` introduces its expected Loom
output. `// ERROR` annotations check source diagnostics. `// INPUT:` selects
import options such as the language standard, data model, or exported roots.
`// RUN: emit cxx` instead places canonical exported C++ after `//----`; these
files declare `// REQUIRES: loom-export-cxx` and can grow through ordinary
`//====` cases without adding process-level JSON fixtures.
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

Executable importer inputs use `.cxx`, including handwritten cases and shared
kernel implementations. Compiler fixtures use `.cxx-test`; native API test
implementations use `.cc`. The repository's normal clang-format checks include
`.cxx` sources.

These programs exercise the full source path: C++ import, ordinary Loom
bytecode linking, config specialization, AMDGPU compilation, and GPU execution.
Authored `check.scenario` modules generate directed inputs, execute source
functions or kernels, and compare their complete outputs with independent Loom
oracles. Numerical kernels use independent double-precision references. Every
native scenario checks both allocation guards bitwise. AMDGPU qualification
consumes the reusable modules through `loom_test` in the
[target test package](../../../../../target/amdgpu/test/cxx/BUILD.bazel).
The [source authoring guide](../README.md#executable-checks-and-benchmarks)
shows handwritten cases and benchmarks.

```sh
iree-bazel-test --config=asan --config=loom-importer-cxx \
  --config=loom-target-amd-gpu \
  --//runtime/config/hal:drivers=amd-gpu \
  --//libamdf/config:enabled=true --//libamdf/config:families=rdna,cdna \
  //loom/target/amdgpu/test/cxx:kernels_test

iree-bazel-test --config=asan --config=loom-importer-cxx \
  --config=loom-target-vm \
  //loom/src/loom/import/cxx/test:functions_test
```

Build declarations pair each source with its authored scenario module. All
kernels in a source file share that module; the cases select the entries they
exercise. Import uses the normal facade headers without per-kernel root flags or
header-path overrides. Scenario generators construct their inputs at runtime,
and comparison trials run independent oracle functions before transferring the
same physical inputs to the selected target.
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
access sanitization, where any sanitizer report fails the test automatically.
Individual targets such as `integer_functions_test` here, and
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
| `constant_loops.cxx` | Constant-expression bounds, unsigned edge conditions, wrapped constants, mutable bounds and scheduled loop templates execute in 29 function trials with 55 exact observations on VM and AMDGPU. Thirteen native trials check all 64 lanes, complete guarded outputs and unchanged inputs under normal and access-instrumented execution. | Original source constant-loop witness. |
| `control_flow.cxx` | Pre-test, post-test, and nested loops; final scalar values and effectful helper calls in conditions. Seven trip counts including zero are checked bitwise. | Original source-language semantics witness. |
| `scheduled_sum.cxx` | Unroll factors 1/3 and pipeline depths 1/2 with linear ordering. Exact integer sums for 0, 1, 2, 5, 17, and 33 columns cover startup, tails, and drain under all four schedules. | Original scheduling-contract witness. |
| `schedule_values.cxx` | Calls, snapshots, wide and narrow counters, signed counters, unevaluated expressions, initializers and serial loops execute in 10 directed trials with 80 exact observations on VM. | Original source scheduling-semantics witness. |
| `short_circuit.cxx` | Bounds-guarded reads, exact conditional call-order traces through seven-comparison chains, discarded boolean expressions and scalar truth conversions across 64 lanes. Lengths 0, 1, 17, 33 and 64 run normally and with device access sanitization. | Original source-language semantics witness. |
| `early_returns.cxx` | Per-lane kernel exits, guarded helper returns inside a counted loop, nested return trees, void calls in returns, and local state across continuing paths. Lengths 0, 1, 17, 33 and 64 run normally and with device access sanitization; exited lanes retain their sentinel values. | Original source-language semantics witness. |
| `integer_functions.cxx` | Fixed-point multiply/rescale, byte increment/decrement, signed short decrement, 64-bit wrap and independently promoted shift counts execute in 84 directed trials with 162 exact observations on VM. Inputs include negative rescaling, sign boundaries and counts 0/31/32/63. | Original source-language semantics witness using ordinary exported functions. |
| `enum_values.cxx` | Named constants, signed and unsigned casts, comparisons, boolean enums, inferred 64-bit storage, packed enum promotion, and template-dependent definitions execute in 578 directed trials with 1,130 exact scalar observations on VM and AMDGPU. All 256 signed-byte bit patterns are covered. Twelve additional AMDGPU trials exercise 8/16/32/64-bit enum parameters and pointer storage through ordinary helper calls, including high bits and wraparound, with unchanged inputs and complete guarded allocations checked normally and with device access sanitization. | Original source-language representation witness. |
| `comparison_functions.cxx` | Seven unparenthesized comparisons execute in 136 directed trials on VM: all 128 truth combinations, zero inputs, and each argument at the unsigned maximum. | Original source-language parser and execution witness. |
| `constexpr_values.cxx` | Source-selected returns and continues, discarded unsupported statements and loop-bound writes, zero-trip loops, and ordinary/constexpr initializers execute on the VM with exact scalar expectations. | Original source-selection and sequencing witness. |
| `assumptions.cxx` | Conjunctive and repeated bounds, capacity/stride constant expressions, templates, casts, unsigned wrap, unevaluated `sizeof`, scoped refinements, and wide values execute in 346 directed trials with 350 exact scalar observations on VM and AMDGPU. Nine additional native trials check all 64 lanes and both output guards, normally and with device access sanitization. | Original source-contract witness. |
| `vector_initializers.cxx` | Typed vector temporaries in returns, arguments, templates and nested expressions execute in eight directed trials with 288 exact scalar observations on VM and AMDGPU. Eight additional native trials check complete vector storage, narrow packing, signed zero, zero-filled lanes, left-to-right initializer effects, and both allocation guards under normal and access-instrumented execution. | Original source-language initialization witness. |
| `vector_values.cxx` | Unsigned arithmetic, narrow signed division and remainder, comparisons, float bitcasts, NaN predicates, splats, and vector control execute in 42 directed trials with 760 exact scalar observations on VM. AMDGPU executes the supported function trials and tracks signed division and remainder as a strict target xfail. Fifteen native trials check complete control and mask vectors, immutable inputs, and both allocation guards under normal and access-instrumented execution. | Original source-language vector-semantics witness. |
| `integer_increment.cxx` | Uniform and lane-varying byte/64-bit increments execute on AMDGPU, including byte wrap, low-word carry, bit 63 and full-width wrap. Fifteen input cases check all 64 lanes and both output guards. | Original source-language semantics witness. |
| `increment_values.cxx` | Prefix/postfix results, byte/64-bit wrap, short-circuit and conditional mutations, and loop-condition updates execute in 502 directed trials with 532 exact scalar observations on VM and AMDGPU. Seventeen additional native trials cover complete pointer streams, skipped updates, dynamic physical extents, unchanged inputs, and allocation guards, normally and with device access sanitization. | Original source-language sequencing witness. |
| `structured_continue.cxx` | Conditional iteration exits, shared tails, nested loop targets, shadowed bindings, for increments and pre/post-test conditions execute in 49 function trials with 392 exact observations on VM and AMDGPU. Fifty-five native trials preserve sparse destinations, compacted pointer streams, vector recurrences and filtered reads under four unroll/pipeline schedules, with unchanged inputs and output guards checked normally and with access instrumentation. | Original source-language control witness. |
| `pointer_origins.cxx` | ILP32 pointer roots compose with nonzero origins, static member/helper displacements, signed dynamic reads and signed exchange coordinates. An independent Loom oracle checks every returned value, complete mutations, unchanged inputs and allocation guards across VM, hosted Wasm and applicable device profiles. | Original source pointer-ABI witness. |
| `pointer_walk.cxx` | Interior pointers through helper returns, conditional origins and counted/pre-test/post-test loops. Signed backward displacements, zero-trip behavior and final pointer positions are checked for seven lane-specific trip counts under three starting positions, normally and with device access sanitization. Inputs and output guards remain unchanged. | Original source storage-semantics witness. |
| `shaped_intrinsics.cxx` | Register-table lookups preserve integer values and floating-point bits, including signed zero, infinities and NaNs. Mixed-width dots cover all four byte-signedness pairs and wrapping i32 accumulators. Sixty-six directed trials check 264 exact function observations on VM and AMDGPU. Twelve native trials preserve the original zero and nonzero physical extents while checking complete outputs, unchanged inputs, and allocation guards under normal and access-instrumented execution. | Original shaped operation-binding witness. |
| `record_values.cxx` | Empty, nested, scalar, vector and pointer records through construction, member mutation, independent copies, helpers, conditional values and loop state. Twelve function trials check 36 exact observations on VM and AMDGPU; 60 native trials cover modular overflow, zero trips, defaults/designators and sequencing. Native cases preserve input buffers and output guards, normally and with access sanitization. Record and explicit-leaf controls express the same algorithm for code/resource comparison. | Original source value and callable-boundary witness. |
| `operator_values.cxx` | Member and free arithmetic, comparison, compound assignment, and prefix/postfix update over record values. The VM checks old/new results, mutable receiver writeback, member/free operand sequencing, void results, temporary identity, and a reference result selecting a nonreceiver argument. The same source contains a target-neutral kernel that composes record operators over buffer elements. | Original overloaded value-operator and callable-ABI witness. |
| `symbol_exports.cxx` | A kernel and helper use explicit Loom names inherited through source redeclarations. Authored Loom links and launches the renamed kernel for 32 exact modular integer cases, with unchanged inputs and output guards checked normally and with device access sanitization. | Original source-symbol identity witness. |
| `typed_views.cxx` | Rank-generic source views preserve dynamic/static shape types and dense/strided layouts through generic readers and by-value helpers. A rank-three FP8 view composes a dense layout with a storage schema and copies its payload through native AMDGPU lowering. Zero and positive shapes, row padding, pointer origins, unchanged inputs, internal sentinels, outer guards, and device access reports are checked independently. | Original typed-view descriptor and source-lifetime witness. |
| `storage_updates.cxx` | Guarded storage executes bitmap compound assignment, RHS/address sequencing, pointer and element increments, narrow signed/unsigned conversion and floating arithmetic through both the ordinary VM callable ABI and native kernel launches. Volatile updates use the same helper specializations. Native cases assert exact results and input/output guards under normal and access-instrumented execution. | Original source read/modify/write witness. |
| `boolean_storage.cxx` | Addressed parameters, aliases, output-only initialization, conditional updates, Boolean arrays, Boolean enums and volatile observations execute on the VM. Native kernels check exact 0/1 bytes, record offsets, array strides and workgroup storage, with unchanged inputs and every output guard checked normally and with device access sanitization. | Original Boolean value and object-layout witness. |
| `volatile_memory.cxx` | Qualified pointers, pointer members, helpers, vector storage, workgroup arrays and typed subviews execute on AMDGPU with exact results, unchanged inputs, guards and access checking. Top-level volatile scalar and vector parameter objects use unqualified callable ABIs while retaining observable initialization and reads on VM and AMDGPU. The ordinary-access control uses the same algorithm. Compiler fixtures independently check repeated and discarded observations through cleanup. | Original source memory-observation witness. |
| `atomics.cxx` | Contended device and workgroup ticket allocation, signed and unsigned combining operations, and CAS success/failure with exact old values, final storage and guards. The VM exercises every integer kind at 32 and 64 bits; native cases select the target's implemented widths and kinds, normally and with access sanitization. Interior and volatile pointers retain their storage identity. | Original source atomic-memory witness. |
| `atomic_builtins.cxx` | GCC exchange, old/new fetch results, signed wraparound, and 32/64-bit CAS. Independent argument counters and aliased expected storage expose duplicate evaluation or an erroneous store on successful CAS. VM and native execution check results, expected storage, and guards; native also runs with access sanitization. | Header-free integer atomic witness. |

Complete kernels and multi-source authoring examples live in
`//loom/src/loom/test/corpus/authoring/cxx`. This package retains focused
source-language semantics, importer diagnostics, and frontend lowering tests.

The integer references use unbounded integer multiplication/division and
explicit modular conversion into the source width. Shift counts remain within
the promoted left operand's width, and signed arithmetic inputs stay in their
defined domain. Arithmetic i64 right shifts are exercised on the VM.
