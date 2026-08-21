# Test correctness

`iree-test-loom` executes the [`check.case`](../reference/dialects/check/ops/case.md)
programs in an ordinary `.loom` or `.loombc` module. The same source can create
inputs, call reference functions, launch kernels, and state expectations. A
single command verifies and plans the module, executes its cases, writes a
structured report, and returns failure when any planned sample fails.

## Run every case

With the Loom tools on `PATH`, pass the authored module directly to the runner:

```shell
iree-test-loom program.loom
```

Cases execute in source order. Reference-only cases need no device. Cases that
contain [`kernel.launch`](../reference/dialects/kernel/ops/launch.md) use the HAL
device selected for the run:

```shell
iree-test-loom program.loom --device=amdgpu
```

The report is a `loom.test.v0` JSON document on standard output. Its top-level
counts summarize cases, concrete samples, failures, skipped cases, and planning
issues. The `samples` array carries the result of each concrete case sample.

## Select a case or sample

A source file can keep smoke cases, edge cases, and larger validation cases
beside the same implementation. Select one symbol while iterating:

```shell
iree-test-loom program.loom --case=@decode_tail
```

[`check.param.choice`](../reference/dialects/check/ops/param-choice.md) and the
other parameter generators expand one case into concrete samples. Select one
planned sample by ordinal when investigating a failure:

```shell
iree-test-loom program.loom --case=@decode_shapes --sample=1
```

The ordinal selects from the deterministic plan; it is not a random seed. A
generator-heavy case is bounded explicitly when its Cartesian product would be
larger than the evidence needed for one run:

```shell
iree-test-loom program.loom --max-samples-per-case=16
```

## Keep the oracle with the workload

A case is an SSA program rather than a table of runner arguments. It can:

- construct scalar and tensor inputs with `check.literal` and
  `check.generate.*`;
- select named runtime parameters with `check.param.*`;
- call ordinary `func.def` reference implementations;
- launch one or more kernels through `kernel.launch`;
- compare results with `check.expect.*`;
- state target, file, and execution requirements with `check.require.*`.

That ownership boundary keeps the workload, oracle, and comparison policy
versioned with the implementation. A
[`check.benchmark`](../reference/dialects/check/ops/benchmark.md) later selects
assignments from the same case instead of copying its setup into a second
harness.

## Link test wrappers to libraries

Reusable motifs and production kernels do not need test-only entry points in
their published archives. A sibling test module can declare the exact symbols
it calls, own private wrapper kernels and `check.case` programs, and load the
implementation as bytecode libraries:

```shell
iree-test-loom test.loom \
  --library=motifs.loombc \
  --library=kernels.loombc
```

Libraries are linked whole in argument order. They satisfy declarations that
already exist in the authored input; merely placing a definition in a library
does not make an undeclared call valid. This is the same explicit dependency
contract used by
[`loom-link`](link-and-package.md#link-transitive-dependencies-incrementally).
Test wrappers normally keep this declaration-only form because the runner has
already been given the exact archive universe. They do not need
`module.import` unless a test specifically exercises provider-constrained
selection.

This shape lets a library repository keep target-independent helpers under
`motif/` and test-only wrapper kernels under the corresponding `test/`
package. Published motif archives remain free of launch ABIs while the wrappers
still run across every supported target profile.

## Separate configuration from samples

Configuration bindings specialize the private compile copy used for HAL
launches:

```shell
iree-test-loom program.loom \
  --device=amdgpu \
  --config=model.hidden_size=4096
```

Larger configuration objects can be supplied as JSON or JSONC. Nested object
keys flatten with `.` separators:

```shell
iree-test-loom program.loom \
  --device=amdgpu \
  --config-file=model-config.jsonc
```

`config.decl` values are compile-time facts. `check.param.*` values are runtime
case inputs. Keeping the two domains separate lets one compiled candidate run
many shapes represented by case samples while configuration still specializes
the parts of the program the caller declared constant.

## Instrument a test run

Sanitizers are target-pipeline instrumentation, so the same authored case and
expectations remain in use:

```shell
iree-test-loom program.loom \
  --device=amdgpu \
  --sanitizer=access

iree-test-loom program.loom \
  --device=amdgpu \
  --sanitizer=asan \
  --sanitizer-reporting=report-only
```

The sanitizer set includes access, value, operation, race, ASan, UBSan, and
TSan checks. Multiple checks can be joined with `|`, while `all` selects the
complete supported set. Reporting mode controls whether the instrumented
program uses the target default, traps, or reports issues without trapping.

Sanitized execution is correctness evidence. Performance measurements use an
optimized, uninstrumented build because instrumentation changes the program
being measured.

## Read the structured report

Capture the report when a run is part of CI or a larger investigation:

```shell
iree-test-loom program.loom >test-results.json
```

The first useful views are bounded summaries rather than the complete payload:

```shell
jq '{case_count, sample_count, failed_sample_count,
     skipped_case_count, planning_issue_count}' test-results.json

jq '.samples[] | {case, sample_ordinal, passed}' test-results.json

jq '.samples[] | select(.issues) | {case, issues}' test-results.json
```

Skipped cases carry stable provider, operation, and reason-code fields. A skip
means a declared requirement was unavailable; it is distinct from a passed
sample. Planning issues identify a case and source operation that could not be
formed into executable work.

`iree-test-loom` writes the JSON report before returning a failing process
status. CI therefore gets both a hard failure and the structured explanation.

## Continue to measurement

Once every selected sample passes, add or select a `check.benchmark` row and
continue with [Benchmark checked work](benchmark.md). The benchmark runner
repeats the correctness gate before accepting timing evidence.
