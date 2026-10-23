# loom-check

`loom-check` is the golden-test runner for `.loom-test` files. It splits each
file into `// ====` cases, runs the selected `// RUN:` mode, and compares the
actual output or diagnostics against the inline expectation.

Optional source importers feed an ordinary Loom module into the same modes.
Enabling the [C++ importer](../../import/cxx/README.md#compiler-tests) adds
`.cxx-test` files, with source options selected by `// INPUT:`. The linked formats
are listed by `--list-input-formats`; `--input-format` selects one explicitly.

Roundtrip and pass modes print Low in assembly form by default, including when
an explicit `// ----` expectation is present. Creating or updating an expectation
does not change the output representation. `with-locations` retains that
assembly form while including source locations.

Roundtrip mode and successful pass IR output are reparsed and printed again
before comparison. An expectation can only be updated when the output parses
and its canonical text remains stable.

All modes share LoomC's module admission, context, and reusable workspace.
Roundtrip and format modes use the same text and bytecode APIs as other Loom
tools. Native inspection providers borrow admitted modules; they do not parse
another copy or register a second compiler context.

The final binary composes configured target and compiler support through
LoomC. Target capabilities and artifact emitters are selected separately.
Check providers contribute only inspection and requirement handlers; they
borrow that environment rather than registering compiler capabilities. The
test binary separately adds its synthetic target and passes. The shipping
binary accepts `test.*` syntax without those synthetic compiler extensions.

## Choosing The Test Boundary

C++ unit tests own API contracts and focused functionality. IR programs belong
in `.loom-test` files: parsing and printing examples, verification diagnostics,
pass transformations, and compiler analysis or allocation results. The fixture
lives beside the subsystem whose behavior it covers, with input and expected
output visible together. IR formatters and migration tools can then operate on
the program directly, and all such tests share the `loom-check` executable.

A C++ test acquiring `format/text:parser` just to set up an allocator or pass
has crossed that boundary. Embedding source strings, generating whole programs
in C++, or loading external IR into that unit executable all create the same
compiler-stack dependency. Parser API unit tests are different: parsing itself
is the behavior under test. Small valid analysis-data fixtures remain useful
for API boundary cases, paired with `.loom-test` coverage of the real producer.

Each fixture is listed in its owning `test/BUILD.bazel` using
`loom_check_test_suite`. The default runner, `loom-check-test`, includes the
target-neutral test dialect and `test.low.core` descriptor package. Optional
target providers extend this shared runner; their fixtures and check-only
implementations remain beside the target they test. The
[test-file contract](../../testing/test_file.h) describes modes, annotations,
case separators, and expected output.

## Running And Updating Fixtures

### Focused Output Checks

The test runner's `emit low-schedule-query @function` selects scheduler facts
without snapshotting the complete analysis. `order=0,1,2` reports the relative
scheduled order of those source nodes, `issue=0,1,2` reports their issue cycles,
and `descriptor=0,1` reports their selected descriptors. Dependency queries use
`kind=state consumer=7,8` for exact predecessor sets or `kind=state timing=0:7`
for one edge's separation and timing model. The node indices refer to source
order, before scheduling. `scheduled-pressure=test.i32` runs the production
liveness analysis in the selected schedule order and reports that register
class's exact peak live units and values. These compact outputs use ordinary
exact goldens.

Schedule queries build the production function model and scheduler directly.
Tests of unavoidable register debt or explicit residency cliffs use
`emit low-schedule-json ... diagnostics=candidates output=none` with inline
decision annotations. `emit low-allocation ... diagnostics=placement-decisions`
similarly checks storage affinities directly instead of pinning incidental
value IDs, interval endpoints, and physical register choices in JSON.

`emit low-allocation-query @function` selects exact assignments with
`values=%source,%result`, reports spill counts with `spill-count`, and lists
physical edge transfers with `edge-moves`. Optional `fixed=%source:target_id:0:2`
bindings constrain locations through the production allocation interface.
`reverse-blocks` reverses the nonentry block table after parsing, preserving the
CFG and SSA dominance while exercising allocation when physical block order
differs from dominance order. Authored text still defines values before uses.
`spill-free` runs the complete production repair loop before querying the final
allocation. With `spill-count`, it also reports inserted store/reload traffic,
so a zero final spill count cannot hide materialized spills. `materialize-spills`
instead applies one allocation batch and prints the resulting IR.

Focused lowering examples use exact goldens. When a case lowers one operation
and the output is small, the complete before/after IR is the assertion and
`--update` maintains it. `with-checks` is for large transforms and reports whose
output contains unrelated or unstable details that would obscure the property
under test. It keeps those narrow assertions beside the IR without pinning the
rest of the output:

```text
// RUN: with-checks compile-report source-to-low,low-dce
...
// ----
CHECK: COMPILE-REPORT: source_low_memory * unknown_dynamic_packets=0 * dynamic_write_bytes=20 *
CHECK: COMPILE-REPORT: source_low[*] * source_op=view.store * execution_count=5
CHECK-NOT: COMPILE-REPORT: source_low[*] * execution_count=0
```

Each `CHECK:` must match a whole output line after trimming outer whitespace;
`CHECK-NOT:` rejects any matching line. `CHECK-COUNT-N:` requires exactly N
matching lines across the complete output, where N is a positive decimal count.
For example, `CHECK-COUNT-2: * = buffer.alloca*` verifies distinct allocation
operations without fixing their generated SSA names. Checks are independent and
unordered.
`*` matches any sequence within a line and `?` matches one character. Other
characters are literal. A terminal `count=5` cannot match `count=50`. Blank lines
and standalone `//` comments are ignored. At least one positive check is required;
empty patterns and unknown directives are errors. A positive counted check also
satisfies the requirement for a positive check.

This modifier works with every textual-output mode; `verify` uses diagnostic
annotations. Exact goldens remain useful for canonical formatting and complete
output contracts. Formatting preserves check text, and `--update` leaves both
passing and failing checks unchanged. Machine-readable update suggestions also
omit them: changing an assertion requires an intentional edit.

`compile-report <pipeline>` checks reports from source compilation passes.
For authored Low assembly, `emit low-compile-report @function` builds the shared
emission frame and prints its normal report summary, including static and dynamic
instruction counts. This uses the runner's linked target descriptors and the
same report collector as native emission.

### Source Target Lowering

`emit source-low` runs the shared source compilation pipeline. For a function
without an authored target, select its root and a linked target profile together:

```cpp
// RUN: emit source-low @entry target=vm:core output=low
unsigned entry(unsigned value) { return value + 1u; }
```

In a `.cxx-test`, import produces the source function and specialization supplies
its target facts, including reachable helpers. The same request works with
`.loom-test` source IR. With `target=...`, omitting `@entry` selects the sole
function definition or the unique public function or kernel entry among private
helpers. This lets a file-level RUN select one target across cases with different
entry names.
`output=low` compares the resulting Low assembly;
`output=module` includes the rest of the module, and `output=none` checks only
source-located diagnostics. Functions with authored target bindings can use the
existing whole-module form without a function or target option. Pipeline-text
outputs describe the pipeline itself and do not accept specialization requests.

Pass, pass-report, and compile-report modes verify input before any target
specialization or transformation, and verify successful pipeline output before
comparison. Input verification errors use the same diagnostic annotations as
`verify` mode and prevent the pipeline from running.

These modes use LoomC's prepared programs and module-owned compilation state.
Pass IR output inspects the transformed module directly; unlike durable module
serialization, it does not materialize retained target facts as new definitions.
Pass and compile reports are consumed while their module and target providers
remain live.

Pass, pass-report, and compile-report modes accept the same target selection
before the pipeline. An optional `entry=@function` selects an explicit entry;
otherwise the sole definition or unique public function or kernel entry is
selected:

```text
// RUN: with-checks compile-report target=vm:core source-to-low,low-dce
// RUN: pass target=vm:core entry=@entry @named_pipeline
```

Binary output uses `emit vm-dis target=vm:core` or
`emit spirv-dis target=spirv:vulkan1.3+bda input=source-low`. These modes also
accept `@function` for an explicit entry. The selected entry remains a compiler
root even when private; specialization does not change source visibility or
add an export. Its reachable helpers may be inlined and removed normally.

SPIR-V lowering fixtures use `emit spirv-low` to compare Low IR without binary
emission or external SPIR-V tools. They can refine a selected `target=` profile
with named feature facts. `features=FLOAT16,COOPERATIVE_MATRIX_KHR,-INT16` uses
the suffixes of the public `loomc_spirv_feature_t` constants; a leading `-`
records an unavailable feature. Numeric limits use the public
`loomc_spirv_limit_t` suffixes and decimal values, for example
`limits=MAX_WORKGROUP_STORAGE_BYTES:512,MAX_WORKGROUP_SUBGROUP_COUNT:8`.
The profile API validates these facts before compilation.

`emit cxx` projects an admitted High module through the public
`loom-export-cxx` tool. `emit cxx @kernel` selects one kernel from a mixed
module; omitting the root requires every top-level definition to be
projectable. A `.cxx-test` therefore places authored C++ before `// ----` and
its canonical readable C++ projection after it; a `.loom-test` can use the same
mode to check how authored or transformed High IR reads in the C++ API. Files
using the mode declare `// REQUIRES: loom-export-cxx` so the external tool
boundary remains explicit.

### Running Fixtures

Use checked-in Bazel test targets for normal verification:

```bash
iree-bazel-test --config=asan //loom/src/loom/tools/loom-check/test:test
iree-bazel-test --config=asan //loom/src/loom/...
```

To accept intentional output changes, pass the update flag through Bazel:

```bash
iree-bazel-test --config=asan <loom-check-test-target> --test_arg=--update
```

The `iree-bazel-test` wrapper detects `--test_arg=--update` and uses Bazel's
standalone TestRunner strategy so update-capable tests can rewrite checked-in
fixture files. Prefer that path over direct tool invocations when updating
repository tests.

Direct runs are useful for inspection:

```bash
iree-bazel-run //loom/src/loom/tools/loom-check -- path/to/file.loom-test
iree-bazel-run //loom/src/loom/tools/loom-check -- --update path/to/file.loom-test
```

`--update` cannot be used with stdin or verify-mode cases. Agent-oriented usage
is available from:

```bash
iree-bazel-run //loom/src/loom/tools/loom-check -- --agents_md
```

Emit-mode JSON targets should stay small. When a case only needs structured
diagnostics from the emitter, use `output=none` and check the `ERROR@` or
`REMARK@` annotations instead of checking in a large JSON blob.

Emit providers receiving a parsed module run with verification before emission
and again after successful emission. Their scheduling and allocation rewrites
must leave valid IR even when the selected output is native assembly or a report.
