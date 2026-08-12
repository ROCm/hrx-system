# Read compile reports

A Loom compile report joins emitted artifact facts with compiler analysis for
one compilation identity. It is the first place to answer questions about code
size, instruction mix, memory traffic, register pressure, residency, spills,
and the source-to-Low decisions that produced them.

`loom-compile-report` provides three bounded views:

- `show` presents one report without requiring knowledge of its JSON schema;
- `diff` compares one controlled change under an explicit identity contract;
- `suggest` asks the selected target family for evidence-backed experiments.

Raw JSON remains available after one of those views identifies the exact
evidence that needs deeper inspection.

## Capture one report

Request a structured summary beside the emitted artifact:

```shell
loom-compile kernel.loom \
  --backend=amdgpu-hal \
  --target=gfx11-generic \
  --output=kernel.hsaco \
  --compile-report=summary \
  --compile-report-output=kernel.report.json
```

`summary` is the routine tuning product. `details` adds row-level provenance
for source selections, schedule bands, pressure origins, allocations, spills,
waits, target capabilities, and other focused investigations. Text report modes
exist for direct terminal output, but structured JSON is the composable input
to report tools and automation.

Compile reports are version-zero compiler evidence. Analyze and compare reports
with the same Loom tool version that produced them. A schema mismatch fails
with an instruction to regenerate the report rather than guessing at an old
layout.

## Show one report

Start with the target-neutral human view:

```shell
loom-compile-report show kernel.report.json
```

The header states artifact, target, specialization, workload, configuration,
and compile status. Each entry then separates two evidence classes:

- **Artifact facts** come from the emitted code or target-owned artifact
  inspection: code bytes, instruction counts, instruction families, and
  physical memory usage.
- **Compiler analysis** comes from the prepared program and target model:
  scheduled pressure, estimated dispatch traffic, operation counts, residency,
  allocation, and scheduling evidence.

Unavailable fields are omitted instead of rendered as zero. That distinction
matters: zero instructions is a measurement; no target inspector for that
metric is an absence of evidence.

Automation can request the same bounded view as deterministic JSON:

```shell
loom-compile-report show kernel.report.json --format=json \
  >kernel.view.json
```

The view is smaller and more stable for dashboards and agents than the complete
compiler report.

## Diff one controlled change

Capture the same root, configuration, target, and workload before and after one
source or compiler change, then compare them:

```shell
loom-compile-report diff \
  baseline.report.json \
  candidate.report.json
```

The default `exact` contract requires the complete compilation identity to
match: schema, mode, artifact kind and format, backend, module, function,
target, specialization, configuration, workload, and entry set. A mismatch is
an error rather than a misleading delta.

```shell
loom-compile-report diff \
  baseline.report.json \
  candidate.report.json \
  --format=json \
  >candidate.diff.json
```

There is no force switch for incomparable reports. Fix the experiment so that
one independent variable changes, or describe the result as two observations
instead of a causal comparison.

## Compare target specialization

Target comparison deliberately relaxes only the selected target specialization
fields while retaining the backend and target family:

```shell
loom-compile-report diff \
  gfx11-generic.report.json \
  gfx1151.report.json \
  --comparison=target
```

This answers how the same program and configuration change when specialized
within one target family. It does not make AMDGPU and SPIR-V reports comparable,
or allow different workloads and configuration bindings to masquerade as a
target delta.

## Ask the target for experiments

Suggestions are target-owned interpretations of report evidence:

```shell
loom-compile-report suggest kernel.report.json
```

Each finding names an action and cites the exact evidence paths and values that
motivated it. The output is a prioritized experiment queue, not an assertion
that a transformation will improve performance. Recompile, retest, and measure
each accepted experiment.

```shell
loom-compile-report suggest kernel.report.json --format=json \
  >kernel.suggestions.json
```

Default findings require the target provider's high-confidence evidence tier.
Experiments derived from structurally exact but hardware-unvalidated models are
opted into explicitly:

```shell
loom-compile-report suggest kernel.report.json \
  --include-experimental
```

The output preserves the confidence class and evidence for every finding, so
automation can keep high-confidence and exploratory searches distinct.

## Follow one finding into details

Regenerate a detailed report when `show`, `diff`, or `suggest` identifies a
specific question whose answer needs row-level provenance:

```shell
loom-compile kernel.loom \
  --backend=amdgpu-hal \
  --target=gfx1151 \
  --output=kernel.hsaco \
  --compile-report=details \
  --compile-report-output=kernel.details.json
```

For example, inspect the values live at a register-pressure peak:

```shell
jq '.pressure_origin_rows.rows[]? |
    {function, register_class, peak_block, peak_operation,
     origin, semantic_tag, live_units, live_values} |
    with_entries(select(.value != null))' kernel.details.json
```

Or group materialized spill evidence by source origin and semantic tag:

```shell
jq '[.spill_rows.rows[]?] |
    group_by(.origin + ":" + (.semantic_tag // ""))[] |
    {function: .[0].function,
     origin: .[0].origin,
     semantic_tag: .[0].semantic_tag,
     spill_count: length,
     bytes: (map(.byte_size) | add),
     stores: (map(.store_count) | add),
     reloads: (map(.reload_count) | add)}' kernel.details.json
```

The focused tool supplied the question; the JSON path retrieves its provenance.
This is materially different from scanning an undifferentiated disassembly and
trying to recover which source decision produced it.

## Relate reports to benchmark results

`iree-benchmark-loom` can retain the same compile-report tree beside timing
evidence. Debug and full artifact bundles write per-candidate report sidecars;
JSONL compile rows identify candidates so report, correctness, profile, and
benchmark events can be joined.

Timing answers whether a candidate changed performance. A report diff answers
which emitted or modeled compiler properties changed. Neither substitutes for
the other: lower register pressure without a timing improvement is useful
compiler evidence, not a performance claim.

The productive loop is:

1. establish a passing checked case and a representative benchmark;
2. capture a baseline score and compile report;
3. change one source, configuration, provider, pass, or target variable;
4. retest correctness and collect a candidate score;
5. diff comparable reports and inspect the timing interpretation;
6. use suggestions or detailed rows to choose the next experiment.

That loop preserves the path from source fact to compiler decision to native
artifact to measured behavior without making every ordinary compile dump every
intermediate.
