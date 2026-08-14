# Benchmark checked work

`iree-benchmark-loom` measures workloads selected by
[`check.benchmark`](../reference/dialects/check/ops/benchmark.md). Each benchmark
row names a [`check.case`](../reference/dialects/check/ops/case.md) and may bind
its runtime parameters. The tool owns measurement policy, compilation,
correctness gating, device-buffer materialization, and structured results.

This separation is deliberate: source describes *what* workload to measure;
the invocation describes *how* to measure it.

## Inspect the plan first

Planning catches selection, parameter, requirement, and deduplication mistakes
without compiling or touching a device:

```shell
iree-benchmark-loom program.loom \
  --benchmark=@decode_4096 \
  --dry-run \
  --output=plan.json
```

The plan reports logical benchmark rows and deduplicated physical work items.
Several names may intentionally select the same case and assignments; they then
refer to one physical work item instead of compiling and executing duplicate
work.

```shell
jq '{summary, benchmarks, work_items}' plan.json
```

`--dry-run` verifies and plans the complete module. It performs no correctness
execution, candidate compilation, allocation, or timing.

## Measure the complete case

The default `case_end_to_end` mode measures the executable `check.case` program
as one operation:

```shell
iree-benchmark-loom program.loom \
  --benchmark=@decode_4096 \
  --device=amdgpu \
  --output=results.json
```

This mode is useful when input generation, reference calls, multiple launches,
or output validation are part of the operation whose latency matters. The
runner executes correctness first; timing is omitted when the selected case
sample fails.

## Measure dispatch completion

Kernel tuning usually needs the prepared HAL launch schedule rather than the
whole check program:

```shell
iree-benchmark-loom program.loom \
  --benchmark=@decode_4096 \
  --device=amdgpu \
  --measure=dispatch_complete \
  --batch-size=64 \
  --output=results.json
```

`dispatch_complete` compiles the selected candidate, materializes inputs from
the check program, proves correctness, records a reusable dispatch batch, and
measures host submission through queue completion. The primary score is
normalized per logical operation. Result fields also state the number of
physical dispatches per logical operation, making fused and multi-dispatch
workloads explicit.

A batch size of one exercises isolated dispatch latency. A serialized
multi-dispatch batch amortizes submission overhead and is the stronger primary
shape for sub-microsecond GPU kernels. The batch size is part of the experiment
and remains visible in the result policy.

## Understand benchmark assignments

A benchmark declaration binds case parameters, not compiler or timing policy:

```loom
check.benchmark<@decode_case> @decode_4096 {element_count = 4096}
```

The case still owns input generation and expected results. The command line
selects measurement mode, warmup, duration, batching, profiling, sanitizers,
and configuration. This means the same named workload can be measured under
different policies without editing the program or manufacturing parallel
benchmark symbols.

When the selected case expands into several samples, one ordinal can be
isolated explicitly:

```shell
iree-benchmark-loom program.loom \
  --benchmark=@decode_shapes \
  --sample=1 \
  --dry-run
```

Concrete parameter values appear in the structured result, so reports can be
joined by values rather than filename conventions. Executed HAL results also
record every launch in case-sample and source order. `workload` is the ordered,
typed value vector supplied to the kernel's launch function; `launch_config` is
the separate resolved workgroup count, workgroup size, subgroup size, and any
other launch fields used for that invocation. This makes a measured duration
traceable to its exact runtime launch without pretending those values were
compile-time constants.

```shell
jq '.work_items[] | {benchmark, case, launches}' results.json
```

## Control data reuse

Dispatch measurements rotate through physical device-buffer binding sets. The
default ring targets at least 32 MiB and at least one binding set per dispatch
in the batch, reducing accidental hot-cache results. A deliberate hot-reuse
experiment states that choice directly:

```shell
iree-benchmark-loom program.loom \
  --benchmark=@decode_4096 \
  --device=amdgpu \
  --measure=dispatch_complete \
  --input-ring-count=1 \
  --output=hot-reuse.json
```

The result records the effective ring count and bytes under
`benchmark_result.data_cache`. Comparisons are meaningful when candidates use
the same data-reuse policy.

## Collect instrumented device timing outside the score

Final-batch profiling records and executes a separate metadata-retaining replay
of the same logical workload after the measured window:

```shell
iree-benchmark-loom program.loom \
  --benchmark=@decode_4096 \
  --device=amdgpu \
  --measure=dispatch_complete \
  --batch-size=64 \
  --profile-final-batch=true \
  --output=profiled.json
```

Keeping the profiled replay outside the score prevents instrumentation from
changing the primary host completion timing. It does not make the replay the
device component of that ordinary timing. Retained metadata can add timestamp
packets, barriers, flushes, fixups, or harvest work according to the target.
When the target provides complete timestamps,
`profile_replay.measurement_relationship` records that the replay is a distinct
execution, and
`profile_replay.dispatch_timing.dispatch_distribution.duration_ns` contains
per-dispatch statistics for that instrumented replay.

`profile_replay.warnings` states known comparability and perturbation hazards.
When present, `profile_replay.comparison` means the tool's completeness checks
passed; it does not by itself prove equivalence with another runtime's
instrumentation.

A same-replay device interval must be contained by the synchronous host
interval that encloses it. A separately profiled replay has no subtraction or
containment relationship with the ordinary replay. Cross-backend kernel
comparisons therefore require matched semantic boundaries, device timing on
both sides, and a paired calibration showing that each timing mechanism has
acceptable perturbation. Host queue-completion time and device timestamps
remain different time domains and are not interchangeable scores.

## Compare candidates in one run

Interleaved execution controls for drift that would otherwise contaminate two
independent benchmark processes:

```shell
iree-benchmark-loom program.loom \
  --device=amdgpu \
  --compare=@baseline,@variant \
  --interleave=ABABA \
  --repetitions=2 \
  --measure=dispatch_complete \
  --output-format=jsonl \
  --output=comparison.jsonl
```

ABABA supports a baseline and one candidate. Round-robin interleaving supports
larger candidate sets. Comparison candidates must select the same concrete
parameter layout and values; a parameterized case therefore uses `--sample=N`
when more than one sample remains.

Final profiling is suppressed during interleaved windows so profile collection
cannot bias one candidate. The event stream records every repetition and the
aggregate baseline/candidate ratios.

## Read the result

The default snapshot is one compact JSON document. Its stable author-facing
shape is:

- `summary`: planned, correctness, failure, and artifact counts;
- `device`: selected driver, backend, and HAL identity;
- `work_items`: deduplicated physical execution and timing results;
- `benchmarks`: logical names and their `work_item_index` references;
- `failed_samples` and `failures`: present only when there is evidence to
  explain.

Start with the score and its interpretation:

```shell
jq '.work_items[] |
    {state,
     measure: .policy.measure,
     p50: .measurement.operation_timing_ns.p50,
     warnings: .measurement.timing_interpretation.warnings}' results.json
```

JSONL is the better form for long sweeps and dashboards. Rows are flushed as
planning, compilation, correctness, measurement, profiling, and comparison
events occur. `run_id` and `candidate_id` fields join those events without
loading the complete run in memory.

## Preserve the experiment

An artifact bundle captures the result, command line, source identity, device
identity, fixtures, and requested compiler or profile evidence:

```shell
iree-benchmark-loom program.loom \
  --benchmark=@decode_4096 \
  --device=amdgpu \
  --measure=dispatch_complete \
  --artifact-bundle-dir=decode-run
```

`minimal` is the default bundle policy. `debug` and `full` add target artifacts,
listings, manifests, and compile-report sidecars for investigations that need
them. Those evidence channels are introduced in the focused compilation and
report workflows rather than required for ordinary timing.

Sanitizers, debug compilers, active build jobs, and uncontrolled device work
all change the measurement. Publishable performance evidence comes from an
optimized uninstrumented build, a recorded device identity, a representative
data-reuse policy, and enough work to make timing warnings empty.

## Continue to compiler evidence

When a score changes, compile reports answer which emitted property changed.
Continue with the compilation and report workflows from the
[workflow map](index.md#compilation-and-evidence).
