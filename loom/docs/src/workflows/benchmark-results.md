# Read benchmark results

`loom-benchmark` joins planning, compilation, correctness, execution, and
measurement evidence. Its default snapshot is convenient when one process can
consume the complete result. JSONL preserves the same run as a stream of typed
events for long sweeps, live inspection, and interleaved comparisons.

This page owns the result shapes and query recipes. The measurement procedure
and the meaning of each timing boundary live in
[Benchmark checked work](benchmark.md).

## Choose the result shape

| Shape | Selection | Best fit |
| --- | --- | --- |
| Snapshot | `--output-format=snapshot` or the default | One completed run loaded as a document. |
| Event stream | `--output-format=jsonl` | Long-running sweeps, incremental readers, and interleaved comparisons. |

The public output choices are `--output-format=snapshot|jsonl`.

Snapshot output is written after the run completes. JSONL rows are flushed as
evidence becomes available, so an interrupted run retains every completed row.
Both forms distinguish unavailable evidence from a measured zero: absent
objects and fields mean the evidence was not produced, while numeric zero is a
reported value.

## Read a snapshot

The stable author-facing snapshot contains:

| Field | Meaning |
| --- | --- |
| `summary` | Planned, failure, correctness, and artifact totals. |
| `device` | Selected driver, backend, and inexpensive HAL identity when execution selected a device. |
| `compiled_entries` | Case compilation diagnostics, artifact links, and requested reports, shared by launches and samples. |
| `work_items` | Deduplicated physical correctness and measurement results, with links to case compilation evidence. |
| `benchmarks` | Logical benchmark records referring to physical results by `work_item_index`. |
| `failed_samples` | Observed correctness failures; omitted when none were observed. |
| `failures` | Parse, verification, selection, and planning failures; omitted when empty. |
| `repetitions` | Individual interleaved measurement windows when comparison data is retained in the snapshot. |
| `comparisons` | Aggregate baseline and candidate ratios when comparison data is present. |

Physical work items and JSONL `benchmark_result` payloads group the requested
experiment under `policy`, correctness evidence under `correctness`, and timing
evidence under `measurement`. An absent group means that stage did not produce
evidence; it is not an all-zero result.

Each result's `correctness.state` is `unchecked`, `passed`, or `failed`.
Scenario timing is unchecked by default; `--scenario-correctness=true` enables
the authored checks before timing. A top-level `state: "ok"` establishes that
measurement succeeded, not that outputs were checked. Unchecked results have
zero correctness sample counts, `data_cache.validity: "unchecked"`, and no
`correctness_materialization`. Summary correctness totals count only observed
checks and do not certify unchecked results elsewhere in the run.

Begin with the run shape:

```shell
jq '{summary, device, benchmarks, work_items}' results.json
```

Inspect each physical result and select the score appropriate to its
measurement mode. Complete-case timing uses `timing_ns`; HAL dispatch timing
uses `operation_timing_ns`:

```shell
jq '.work_items[] |
    {work_item_index,
     state,
     measure: .policy.measure,
     p50: (.measurement.operation_timing_ns.p50 //
           .measurement.timing_ns.p50),
     stop_reason: .measurement.stop_reason,
     warnings: .measurement.timing_interpretation.warnings}' results.json
```

Join logical names to deduplicated work:

```shell
jq '.benchmarks[] |
    {candidate_id, benchmark, case, work_item_index}' results.json
```

Failures remain short when the run is healthy:

```shell
jq '(.work_items[] | select(.failure)),
    .failed_samples[]?,
    .failures[]?' results.json
```

An entry-specific compilation or execution rejection identifies the affected
kernel as `failure.entry`; the enclosing benchmark remains the logical workload
that selected it.

A `compile` row with `state != "ok"` is a candidate rejection represented as
data. Parse, verification, selection, and planning failures use `failure` rows
because they prevent the candidate lifecycle from beginning.

An unmet `check.requires` or matched `check.skip_if` produces `state: "skipped"`
without a timing measurement or failed-benchmark count. The physical result
retains `skip_reason`, shared by its logical aliases. JSONL includes the same
object in each `benchmark_result`:

```shell
jq '.work_items[] | select(.skip_reason) |
    {work_item_index, case, scenario, skip_reason}' results.json
```

`skip_reason.provider` identifies the predicate provider, and `op` is `requires`
or `skip_if`. Its stable `code` is `provider_not_registered`,
`provider_unavailable`, `requirement_not_satisfied`, or `skip_predicate_matched`.
Optional `provider_code` gives a provider-defined reason; `display_message` is
human-facing text, with an authored `skip_if` reason taking precedence over
provider text. Dry runs and compile-only runs do not evaluate requirements and
therefore do not claim a skip reason.

Executed HAL results retain launches in case-sample and source order. Each
launch separates the ordered, typed `workload` supplied to the launch function
from the resolved `launch_config` used for submission:

```shell
jq '.work_items[] |
    select(.launches) |
    {work_item_index, launches, data_cache}' results.json
```

Benchmark dictionaries bind case parameters. Measurement policy and compiler
configuration remain invocation choices. Concrete parameter samples appear in
the result so several shapes can be joined by values rather than filenames.
`--config=key=value` and `--config-file=path` materialize compile-time
configuration in the candidate copy, while case sample values remain runtime
invocation inputs.
Parameterized snapshot work items retain the same concrete values under
`sample.parameters`.

## Read an event stream

Every JSONL row carries `row` and `run_id`. Candidate-specific rows also carry
`candidate_id` and `candidate_index`. Case samples add `sample_id`,
`sample_index`, and `sample.parameters`; scenario samples identify the
configuration and trial inside `trial_result`. `work_item_index` connects
physical correctness work to its logical benchmark aliases.

| `row` value | Evidence |
| --- | --- |
| `run` | Source path and process-local run identity. |
| `plan` | Selected benchmark and case, effective policy, and command-line overrides. |
| `device` | Selected driver, backend, and inexpensive HAL identity. |
| `compile` | Candidate compile state, diagnostics, and requested compile evidence. |
| `sample` | Correctness result for one concrete case sample or scenario trial. |
| `benchmark` | Measurement or failure evidence, with explicit correctness state. |
| `benchmark.repetition` | One interleaved candidate measurement window. |
| `comparison` | Aggregate baseline and candidate timing ratios. |
| `profile_replay` | Instrumented replay evidence collected outside the scored window. |
| `profile_summary` | Decoded profile bundle state or summary data. |
| `profile_counter` | Decoded counter state, sets, groups, and values. |
| `failure` | Parse, verification, selection, or planning failure. |
| `summary` | Final run totals. |

Watch the rows that usually change an experiment decision:

```shell
tail -f results.jsonl | jq -c '
  select(.row == "compile" or
         .row == "benchmark" or
         .row == "comparison" or
         .row == "summary")'
```

Scenario samples contain the selected configuration, trial domain, trial
ordinal, entropy root, and authored expectation results. With
`--scenario-correctness=true`, each selected trial is checked before timing;
interleaved comparisons check each candidate before
their first window, not before every repetition. A failed trial produces no
timing score, and a checked comparison needs passing baseline and candidate trials.
`check.invoke` trials also execute before timing to check for unhandled device errors,
but have no numerical oracle. A passing result establishes only the checks
authored in the scenario.

```shell
jq 'select(.row == "sample" and .trial_result) |
    {candidate_id, work_item_index, benchmark_sample_index, trial_result}' \
  results.jsonl
```

The default snapshot retains failed trials in `failed_samples[].trial_result`
and counts all observed trials under `summary.correctness`. Successful trials
do not add verbose observation records to the snapshot. Compilation, oracle
execution, and comparisons remain outside the measured window; timing uses
fresh target-only inputs for the same trial. Comparison candidates share the
baseline scenario's entropy domain, so replaying a comparison retains its
baseline selection as well as the reported trial coordinate.

Inspect terminal state and compilation rejection separately:

```shell
jq 'select(.row == "run" or .row == "summary")' results.jsonl

jq 'select(.row == "failure" or
           (.row == "compile" and .state != "ok"))' results.jsonl

jq 'select(.row == "compile" and .state != "ok") |
    .diagnostics[]?' results.jsonl
```

Read dispatch scores and their correctness evidence:

```shell
jq 'select(.row == "benchmark") |
    .benchmark_result |
    {benchmark,
     state,
     p50: .measurement.operation_timing_ns.p50,
     physical_dispatches_per_logical_operation:
       .measurement.physical_dispatches_per_logical_operation,
     mean_physical_dispatch_duration_ns:
       .measurement.mean_physical_dispatch_duration_ns,
     stop_reason: .measurement.stop_reason,
     warnings: .measurement.timing_interpretation.warnings}' results.jsonl
```

Inspect the case and resolved launch evidence carried by that score:

```shell
jq 'select(.row == "benchmark") |
    .benchmark_result |
    {benchmark, case, launches, data_cache}' results.jsonl
```

Join a score to its concrete case parameters:

```shell
jq 'select(.row == "benchmark" and .sample) |
    {candidate_id,
     sample_id,
     sample: .sample.parameters,
     p50: .benchmark_result.measurement.operation_timing_ns.p50}' \
  results.jsonl
```

`--interleave=ABABA` accepts a baseline and one candidate.
`--interleave=round_robin` rotates larger candidate sets; together these form
the `--interleave=ABABA|round_robin` choice. Interleaved
comparisons preserve both individual windows and aggregate ratios:

```shell
jq 'select(.row == "benchmark.repetition") |
    {candidate_id,
     order_index,
     token: .schedule_token,
     p50: .benchmark_result.measurement.operation_timing_ns.p50}' \
  comparison.jsonl

jq 'select(.row == "comparison") |
    {candidate_id,
     baseline_candidate_id,
     ratio_p50,
     speedup_p50,
     ratio_p90,
     speedup_p90,
     candidate_p50_spread_ppm}' comparison.jsonl
```

Scenario repetition rows additionally identify the concrete trial selected by
`--sample`:

```shell
jq 'select(.row == "benchmark.repetition" and has("trial_ordinal")) |
    {candidate_id,
     benchmark_sample_index,
     configuration_ordinal,
     trial_index,
     trial_ordinal}' comparison.jsonl
```

## Follow retained artifacts

`--artifact-bundle-dir=DIR` collects the result, `manifest.json`, file outputs,
and requested profile data. When no explicit result path is present, snapshots
use `results.json` and event streams use `results.jsonl` inside the bundle. The
manifest records the command line, source and device identities, selected
environment, output paths, and path/size/modification-time identities for
observed fixtures and artifacts. It does not content-hash large files or act as
a content-addressed store.

`--artifact-bundle-policy=minimal|debug|full` selects bundle contents, with
`minimal` as the default. The `debug` and `full` policies can also retain
per-candidate compile reports, target-native artifacts, target listings,
artifact manifests, and HAL executables under
`compile_reports/`, `target_artifacts/`, `target_listings/`, and
`hal_executables/`. Compile rows expose the paths without requiring filename
conventions:

```shell
jq 'select(.row == "compile" and .compile_report_path) |
    {candidate_id, path: .compile_report_path}' results.jsonl

jq 'select(.row == "compile" and .artifact_manifest_path) |
    {candidate_id, path: .artifact_manifest_path}' results.jsonl

jq 'select(.row == "compile" and .target_artifact_path) |
    {candidate_id,
     target: .target_artifact_path,
     listing: .target_listing_path,
     hal: .hal_executable_path}' results.jsonl
```

Benchmark rows expose the same links inside `benchmark_result`:

```shell
jq 'select(.row == "benchmark") |
    .benchmark_result |
    {benchmark,
     compile_report_path,
     artifact_manifest_path,
     target_artifact_path,
     target_listing_path,
     hal_executable_path} |
    with_entries(select(.value != null))' results.jsonl
```

`--artifact-manifest=none|summary|details|analysis` controls target-neutral
manifest sidecars and any non-`none` choice requires a `debug` or `full` bundle.
`--compile-report=none|summary|details` controls the canonical `loom-compile`
report tree embedded in compile and benchmark-result payloads; `details` retains
row-level evidence. Start with `summary`, then request `details` only for a
specific compiler question.

`check.file.read.npy` paths resolve relative to the input module.
`check.file.write.npy` paths are relative and resolve under
`--file-output-root`, whose default is a source-specific directory below the
platform temporary directory's `iree-loom-benchmark` root.
`--profile-artifacts-dir=DIR` retains raw HAL profile bundles when counters,
device metrics, executable traces, or later decoding are part of the evidence.

## Read profile rows

Final-batch profiling is a distinct execution after the scored measurement.
Its device timestamps have no subtraction or containment relationship with the
ordinary run. Profile rows are therefore joined to the candidate as additional
evidence rather than treated as a component of host completion time.

Select profiling families with `--profile-data`; repeat `--profile-counter` for
specific implementation-defined counters. An explicit profile-artifact
directory implies `--profile-final-batch=true` unless that flag was explicitly
set false:

```shell
loom-benchmark kernel.loom \
  --benchmark=@kernel_latency \
  --device=amdgpu \
  --measure=dispatch_complete \
  --profile-data=counter-ranges \
  --profile-counter=SQ_WAVES \
  --profile-artifacts-dir=profile-data \
  --artifact-bundle-dir=kernel-run
```

Inspect decoded counter groups and decoder state:

```shell
jq 'select(.row == "profile_counter") |
    .counter |
    select(.type == "counter_group") |
    {key, counter, avg, sum}' results.jsonl

jq 'select(.row == "profile_counter" and
           .counter.type == "counter_decode_state") |
    {candidate_id,
     state: .counter.state,
     state_code: .counter.state_code,
     decoded: .counter.decoded_rows,
     status: .counter.status}' results.jsonl
```

Read target-supplied dispatch statistics retained by the profiled replay:

```shell
jq 'select(.row == "profile_replay") as $event |
    $event.profile_replay.summary.rows[]? |
    select(.type | startswith("dispatch_")) |
    {candidate_id: $event.candidate_id,
     type,
     function_name,
     timing: .timing.mean_ns}' results.jsonl
```

Inspect profile bundle summaries and explicit decode state:

```shell
jq 'select(.row == "profile_summary" and
           .profile_summary.type == "summary") |
    {candidate_id,
     traces: .profile_summary.executable_trace_records,
     trace_bytes: .profile_summary.executable_trace_data_bytes,
     metric_values: .profile_summary.device_metric_values}' results.jsonl

jq 'select(.row == "profile_summary" and
           .profile_summary.type == "profile_summary_state") |
    {candidate_id,
     state: .profile_summary.state,
     state_code: .profile_summary.state_code,
     decoded: .profile_summary.decoded_rows}' results.jsonl
```

When the provider retains per-dispatch timing, inspect it inside the distinct
profile replay rather than combining it with the scored host interval:

```shell
jq 'select(.row == "profile_replay") |
    {candidate_id,
     dispatch_timing: .profile_replay.dispatch_timing,
     comparison: .profile_replay.comparison,
     warnings: .profile_replay.warnings}' results.jsonl
```

## Continue into compiler evidence

With `--compile-report=summary` or `--compile-report=details`, case benchmarks
retain reports in `compiled_entries`, including multi-dispatch sequences. Each
work item's `compile_item_index` selects its compilation group; each launch's
`compile_entry_index` selects an entry within that group. These indexes are
run-local identities, not positions in the output array. Benchmark aliases
share the same physical work and compilation evidence.

For a sequence calling A, B, then A again, the group contains one report for A
and one for B. Both calls of A refer to the first call's report, while retaining
their own workload and resolved launch configuration. The sequence's timing
still measures the complete logical operation. Separate compilation groups
remain distinct even when entry names match. Scenario products retain their
compile report directly on the work item that owns their specialization.

Extract one case entry's canonical report for further inspection:

```shell
jq '.compiled_entries[] |
    select(.compile_item_index == 0 and .compile_entry_index == 0) |
    .compile_report' results.json > entry-report.json

loom-compile-report show entry-report.json
```

The lightweight entry identity and diagnostics remain available without report
capture; the large report payload is present only when requested. JSONL `compile`
rows carry the same entry indexes and payloads as the snapshot collection.

Compile reports embedded in a `compile` row or retained as sidecars use the
same schema as `loom-compile`. Sidecars can be passed directly to
`loom-compile-report`; an embedded report can be extracted without changing its
shape:

```shell
jq 'select(.row == "compile" and .compile_report) |
    .compile_report' results.jsonl
```

To query embedded reports in place, bind the enclosing row, apply a standalone
report query below `.compile_report`, and carry `$event.candidate_id` into the
result. Multi-entry rows are indexed at `.compile_report.entries.rows[]`.
[Query compile evidence](compile-report-queries.md#query-reports-retained-by-a-benchmark)
shows the enclosing-row join. A changed score usually begins with
`loom-compile-report show`, `diff`, or `suggest` before raw fields are needed.
