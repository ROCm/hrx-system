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
target-specific cases live with that target's tooling tests and runner. The
[test-file contract](../../testing/test_file.h) describes modes, annotations,
case separators, and expected output.

## Running And Updating Fixtures

### Shared Corpus Templates

A target fixture opts into a shared corpus with a file-preamble `TEMPLATE`
directive. The checked-in fixture is self-contained: ordinary runs and
`--target` compilation use its concrete input without reading the template.
Template sources are not test runtime dependencies and need no `data` entries.

Precommit checks freshness with `loom-check --check-templates`, using the same
synchronizer as `--update`. It checks all tracked Loom sources, including
unchanged consumers when only their shared template changes. This read-only
mode accepts multiple files, ignores files without TEMPLATE, and never executes
RUN directives or compares expectations:

```bash
iree-bazel-run //loom/src/loom/tools/loom-check -- \
  --check-templates --template-root=. path/to/file.loom-test
```

`--update` copies each common case's source, including its declarations,
modifiers, helpers, and comments, before updating output expectations. The
consumer keeps its RUN options, REQUIRES/XFAIL directives, output expectations,
and diagnostic annotations. An annotation whose source line changes requires an
explicit update.
Case identity is the sole function-like definition, or the unique public entry
when a case includes private helpers.

Homogeneous target selection belongs in the RUN compiler options. Synchronizing
a template adds no declarations or target bindings and never substitutes a local
definition for a common declaration. Source edits apply equally to old and new
cases. A case-local RUN can select a different profile or ABI explicitly. A
heterogeneous program with authored target relationships belongs in a dedicated
target fixture.

Architecturally inapplicable cases in a mixed fixture have explicit exclusions:

```text
// TEMPLATE: loom/src/loom/test/corpus/source_low/view_transport.loom-test
// TEMPLATE-EXCLUDE: @correlated_cfg_rotation SPIR-V requires structured control flow.
// RUN: emit source-low target=spirv:vulkan1.3+bda output=low control-flow=structured-low
```

Each exclusion names one exact case and gives a reason. Duplicate or unknown
names fail, so a removed or renamed case requires updating its exclusions.
All remaining cases, including newly added cases, keep their synchronization
contract. `--update` removes excluded cases from the concrete target fixture.
An entirely inapplicable corpus needs no fixture or build registration;
excluding every case is an error. A target's dedicated rejection tests cover
its architectural boundary. Missing implementations within the target's source
contract retain precise diagnostic coverage in the shared-corpus fixture.

### Focused Output Checks

When only a few properties matter, `with-checks` keeps assertions beside the IR
without pinning the rest of a report or transformed program:

```text
// RUN: with-checks compile-report source-to-low,low-dce
...
// ----
CHECK: COMPILE-REPORT: source_low_memory * unknown_dynamic_packets=0 * dynamic_write_bytes=20 *
CHECK: COMPILE-REPORT: source_low[*] * source_op=view.store * execution_count=5
CHECK-NOT: COMPILE-REPORT: source_low[*] * execution_count=0
```

Each `CHECK:` must match a whole output line after trimming outer whitespace;
`CHECK-NOT:` rejects any matching line. Checks are independent and unordered.
`*` matches any sequence within a line and `?` matches one character. Other
characters are literal. A terminal `count=5` cannot match `count=50`. Blank lines
and standalone `//` comments are ignored. At least one positive check is required;
empty patterns and unknown directives are errors.

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

`emit pipeline-plan @pipeline max-instances=<count>` checks diagnostics from
the shared concrete pipeline planner without selecting a target. The input is
verified before generic value facts and the plan are computed. The explicit
resident-instance capacity supplies the planner's allocation bound, normally
provided by a target materializer. This mode produces diagnostics only; source
legality checks that need no propagated facts continue to use `verify`.

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
function definition or the unique public entry among private helpers. This lets
a file-level RUN select one target across cases with different entry names.
`output=low` compares the resulting Low assembly;
`output=module` includes the rest of the module, and `output=none` checks only
source-located diagnostics. Functions with authored target bindings can use the
existing whole-module form without a function or target option. Pipeline-text
outputs describe the pipeline itself and do not accept specialization requests.

Pass, pass-report, and compile-report modes accept the same target selection
before the pipeline. An optional `entry=@function` selects an explicit entry;
otherwise the sole definition or unique public entry is selected:

```text
// RUN: with-checks compile-report target=vm:core source-to-low,low-dce
// RUN: pass target=vm:core entry=@entry @named_pipeline
```

Binary output uses `emit vm-dis target=vm:core` or
`emit spirv-dis target=spirv:vulkan1.3+bda input=source-low`. These modes also
accept `@function` for an explicit entry. The selected entry remains a compiler
root even when private; specialization does not change source visibility or
add an export. Its reachable helpers may be inlined and removed normally.

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
fixture files. It also supplies the checkout root for template synchronization;
ordinary test launches need no template root. Prefer that path over direct tool
invocations when updating repository tests.

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
