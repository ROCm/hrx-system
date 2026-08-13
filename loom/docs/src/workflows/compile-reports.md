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

The header states artifact, target, specialization, compiled launch geometry,
configuration, and compile status. Launch fields appear only when the compiled
artifact fixes them; a missing workgroup count is the normal result for a
workload-dependent launch function, not missing benchmark evidence. Each entry
then separates two evidence classes:

- **Artifact facts** come from the emitted code or target-owned artifact
  inspection: code bytes, instruction counts, instruction families, and
  physical memory usage.
- **Compiler analysis** comes from the prepared program and target model:
  scheduled pressure, compiled execution economics, residency, allocation, and
  scheduling evidence.

Compiled execution economics never require a static dispatch size. Exact
fixed-trip multiplicities are reported per workitem and, when the artifact
fixes local workgroup size, for one workgroup. They count target-Low operation
effects and descriptor-effect widths, not native instruction issues, memory
transactions, or executed branch paths. Control flow is a statically reachable
block envelope: data-dependent alternatives may both contribute. Whole-
dispatch geometry belongs to a concrete benchmark launch unless the compiled
artifact genuinely specializes it.

Register-move analysis includes compiler-classified causes. Summary reports
preserve the aggregate packet and register-unit counts; detailed reports let
`show` enumerate causes and let `diff` distinguish added, removed, changed, and
unchanged causes. This separates a source-level repacking change from a
target-created operand-bank repair instead of treating both as generic moves.

Fragment memory reports also preserve the producer-proven address relationship
across a complete subgroup. `show` prints each authored memory source and its
packet variants with the per-lane packet width, exact lane-offset formula, and
the byte-interval union for the subgroup. Requested bytes count every lane's
request, unique bytes count their union, and span bytes cover the lowest through
highest requested address. Dense, gapped, and overlapping are exact geometry
facts; they are deliberately not presented as cache-line transaction or cycle
predictions. When the compiler cannot prove a complete subgroup, uniform
dynamic base, uniform control, or packet width, the missing proof and its reason
remain visible instead of manufacturing geometry.

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

When historical or externally produced artifacts cannot be regenerated with a
common identity, `--force` can compare reports containing exactly one compiled
entry each:

```shell
loom-compile-report diff \
  historical-baseline.report.json \
  historical-candidate.report.json \
  --force
```

The forced view preserves both report and entry identities, lists every field
that violates the exact comparison contract, and labels the deltas as
observational rather than causal. Reports with more than one entry remain an
error because entry order is not a semantic pairing contract. Use the strict
default for controlled experiments; `--force` makes otherwise inaccessible
evidence inspectable, but does not repair the experiment.

Wave memory diffs group packet and strategy variants under the stable authored
source identity. A scalar-to-wide lowering therefore appears as removed and
added variants of one source even when the number of packets changes. The tool
does not invent one-to-one matches between ambiguous variants, and it retains
the complete before and after geometry so widening, lane dispersion, and
overlap can be evaluated together.

For a forced single-entry comparison, the explicit entry pair also aligns wave
memory sources whose function was renamed. Packet variants still require exact
semantic identity; the forced pair does not authorize guessed variant matches.

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

The experimental AMDGPU fragment-packet finding requires exact wave geometry
for every cited scalar packet row. It reports per-lane width together with
cross-lane coverage, unique bytes, span, maximum gap, and maximum adjacent-lane
delta. Packet width is not treated as a monotonic objective: wider candidates
that worsen dispersion require a hardware timing win, rather than receiving a
recommendation merely because they reduce static packet count. When the same
packets coincide with target-created operand-bank materialization, the finding
leads with allocator placement and asks the reader to verify that the repairs
belong to those loads before changing the memory hierarchy.

```shell
loom-compile-report suggest kernel.report.json --format=json \
  >kernel.suggestions.json
```

Default findings require the target provider's high-confidence evidence tier.
Exploratory policies and experiments derived from structurally exact but
hardware-unvalidated models are opted into explicitly:

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
