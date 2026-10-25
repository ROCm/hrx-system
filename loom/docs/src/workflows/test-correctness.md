# Test correctness

`loom-test` executes [`check.case`](../reference/dialects/check/ops/case.md)
and [`check.scenario`](../reference/dialects/check/ops/scenario.md) programs in
an ordinary `.loom` or `.loombc` module. The same source can create inputs, call
reference functions, launch kernels, compare an unchanged subject across target
and oracle profiles, and state expectations. A single command verifies and
plans the module, executes its selected correctness records, writes a structured
report, and returns failure when any planned sample or trial fails.

## Run every correctness record

With the Loom tools on `PATH`, pass the authored module directly to the runner:

```shell
loom-test program.loom
```

Cases and scenarios execute in source order. Reference-only cases and VM-only
scenarios need no device. Cases that contain
[`kernel.launch`](../reference/dialects/kernel/ops/launch.md), and scenarios
whose target runs through HAL, use the device selected for the run:

```shell
loom-test program.loom --device=legacy-amdgpu
```

The device normally selects the most specific compatible compiler target. An
explicit target keeps cross-device comparisons on a stable compilation
baseline:

```shell
loom-test program.loom \
  --device=legacy-amdgpu \
  --target=amdgpu:gfx11-generic
```

`--target=family:selector` does not select or create a device. It constrains
compilation, and the selected device must advertise a matching loadable target.
A forced generic target remains generic even when the device also advertises a
higher-priority exact target.

The report is a `loom.test.v0` JSON document on standard output. Its top-level
counts summarize cases, scenarios, concrete samples and trials, failures,
skipped cases, and planning issues. The `samples` and `trials` arrays carry each
concrete result.

## Compile without an execution device

`loom-check` can qualify a source fixture or a linked test module for a compiler
profile without opening a device:

```shell
loom-check --target=amdgpu:gfx942 offsets.loom-test
loom-check --target=spirv:vulkan1.3+bda program.loombc
```

Each source case must produce a nonempty final artifact or exactly match its
diagnostic annotations. Linked kernel entries compile independently, just as
independent runtime launches do. This checks compilation, not numerical
correctness: `check.expect` programs still need `loom-test` on the intended
execution device.

Compilation leaves ordinary `RUN` goldens unchanged, ignores their execution
requirements and `XFAIL` markers, and rejects `--update`. Template-backed
fixtures still have to match their source corpus. Compile-time bindings use
`--config=key=value` and `--config-file=model-config.jsonc`.

## Select a case or sample

A source file can keep smoke cases, edge cases, and larger validation cases
beside the same implementation. Select one symbol while iterating:

```shell
loom-test program.loom --case=@decode_tail
```

[`check.param.choice`](../reference/dialects/check/ops/param-choice.md) and the
other parameter generators expand one case into concrete samples. Select one
planned sample by ordinal when investigating a failure:

```shell
loom-test program.loom --case=@decode_shapes --sample=1
```

The ordinal selects from the deterministic plan; it is not a random seed. A
generator-heavy case is bounded explicitly when its Cartesian product would be
larger than the evidence needed for one run:

```shell
loom-test program.loom --max-samples-per-case=16
```

## Compare a target against its oracle

Select a scenario with the same `--case` flag. The standard runner uses its VM
profile as the oracle and, when a HAL device is selected, prepares the target
for that device:

```shell
loom-test program.loom \
  --case=@advance_sweep \
  --device=legacy-amdgpu \
  --target=amdgpu:gfx1151 \
  >test-results.json
```

Every [`check.compare`](../reference/dialects/check/ops/compare.md) trial gets
independent target and oracle storage produced from the same deterministic
recipe. The standard HAL and VM pairing compares a resultless function through
its mutable post-state, aliases, authored guard values, and device outcomes.
Profiles that transport results can check those explicit values as well.
[`check.invoke`](../reference/dialects/check/ops/invoke.md) trials execute only
the target profile.

A scenario runs its complete finite configuration and trial domains. The
`--sample` flag remains a `check.case` selector and is rejected for a selected
scenario. A failing trial records its exact deterministic coordinate and the
source location of the failed expectation:

```shell
jq '.trials[] | select(.passed == false) |
    {scenario, entropy, configuration_ordinal, trial_index, trial_ordinal,
     failures: .expectations.failures}' test-results.json
```

Rerunning the same selected scenario reconstructs the same trial identities and
generated values. Trial execution is internally batched; batch boundaries do
not change entropy reads or the values they produce.

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
loom-test test.loom \
  --library=motifs.loombc \
  --library=kernels.loombc
```

Libraries contribute their exported definitions and template implementations,
but only the selected dependency closure enters the test program. They satisfy
declarations that already exist in the authored input; merely placing a
definition in a library does not make an undeclared call valid. Ambiguous exact
definitions are rejected instead of being selected by argument order. This is
the same explicit dependency contract used by
[`loom-link`](link-and-package.md#link-transitive-dependencies-incrementally).

This shape lets a library repository keep target-independent helpers under
`motif/` and test-only wrapper kernels under the corresponding `test/`
package. Published motif archives remain free of launch ABIs while the wrappers
still run across every supported target profile.

## Separate configuration from samples

Configuration bindings specialize the private compile copy used for HAL
launches:

```shell
loom-test program.loom \
  --device=legacy-amdgpu \
  --config=model.hidden_size=4096
```

Larger configuration objects can be supplied as JSON or JSONC. Nested object
keys flatten with `.` separators:

```shell
loom-test program.loom \
  --device=legacy-amdgpu \
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
loom-test program.loom \
  --device=legacy-amdgpu \
  --sanitizer=access

loom-test program.loom \
  --device=legacy-amdgpu \
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
loom-test program.loom >test-results.json
```

The first useful views are bounded summaries rather than the complete payload:

```shell
jq '{case_count, sample_count, failed_sample_count,
     scenario_count, trial_count, failed_trial_count,
     skipped_case_count, planning_issue_count}' test-results.json

jq '.samples[] | {case, sample_ordinal, passed}' test-results.json

jq '.trials[] | {scenario, configuration_ordinal,
                  trial_index, trial_ordinal, passed}' test-results.json

jq '.samples[] | select(.issues) | {case, issues}' test-results.json
```

Skipped cases carry stable provider, operation, and reason-code fields. A skip
means a declared requirement was unavailable; it is distinct from a passed
sample. Planning issues identify a case and source operation that could not be
formed into executable work.

`loom-test` writes the JSON report before returning a failing process
status. CI therefore gets both a hard failure and the structured explanation.

## Continue to measurement

Once every selected sample and trial passes, add or select a `check.benchmark`
row and continue with [Benchmark checked work](benchmark.md). The benchmark
runner repeats the correctness gate before accepting timing evidence.
