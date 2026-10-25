# Benchmark checked work

`loom-benchmark` measures workloads selected by
[`check.benchmark`](../reference/dialects/check/ops/benchmark.md). Each benchmark
row names a [`check.case`](../reference/dialects/check/ops/case.md) or
`check.scenario`. Case benchmarks may bind runtime parameters; scenario
benchmarks select from the scenario's finite sample domain. The tool owns
measurement policy, compilation, device-buffer materialization, and structured
results.

This separation is deliberate: source describes *what* workload to measure;
the invocation describes *how* to measure it.

## Inspect the plan first

Planning catches selection, parameter, requirement, and deduplication mistakes
without compiling or touching a device:

```shell
loom-benchmark program.loom \
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

The default `auto` policy resolves a `check.case` benchmark to
`case_end_to_end` and measures the executable case program as one operation:

```shell
loom-benchmark program.loom \
  --benchmark=@decode_4096 \
  --device=legacy-amdgpu \
  --output=results.json
```

This mode is useful when input generation, reference calls, multiple launches,
or output validation are part of the operation whose latency matters. The
runner executes correctness first; timing is omitted when the selected case
sample fails.

The device normally chooses the compiler target. Use the same explicit target
for controlled comparisons that must hold compilation constant across
compatible devices:

```shell
loom-benchmark program.loom \
  --benchmark=@decode_4096 \
  --device=legacy-amdgpu \
  --target=amdgpu:gfx11-generic \
  --output=generic-results.json
```

The target constrains compilation but never changes the device. A device that
cannot load the requested `family:selector` fails before measurement.

## Measure dispatch completion

Kernel tuning usually needs the prepared HAL launch schedule rather than the
whole check program:

```shell
loom-benchmark program.loom \
  --benchmark=@decode_4096 \
  --device=legacy-amdgpu \
  --measure=dispatch_complete \
  --batch-size=64 \
  --output=results.json
```

`dispatch_complete` compiles the selected candidate, materializes inputs from
the check program, records a reusable dispatch batch, and
measures host submission through queue completion. The primary score is
normalized per logical operation. Result fields also state the number of
physical dispatches per logical operation, making fused and multi-dispatch
workloads explicit.

A batch size of one exercises isolated dispatch latency. A serialized
multi-dispatch batch amortizes submission overhead and is the stronger primary
shape for sub-microsecond GPU kernels. The batch size is part of the experiment
and remains visible in the result policy.

A `check.scenario` benchmark resolves `auto` to `dispatch_complete`. Its target
product and target-local inputs are prepared before timing. A comparison trial
is benchmarked as a target-only invocation: the oracle, output comparison, and
readback do not execute in the measured path. The exact configuration, trial,
and ordinal remain in the result so the corresponding correctness trial can be
replayed.

### Opt in to scenario correctness

Scenario benchmarks default to target-only preparation and execution. They do
not compile or run the oracle, allocate its value graphs, compare outputs, or
read results back for correctness. Results report `correctness.state` as
`unchecked`; successful timing is not evidence of numerical correctness.

For occasional validation or a release gate, enable the authored checks:

```shell
loom-benchmark program.loom \
  --benchmark=@decode_4096 \
  --device=legacy-amdgpu \
  --scenario-correctness=true \
  --output=checked-results.json
```

The runner checks each selected configuration/trial coordinate once, before
timing. Aliases share that check, and interleaved comparisons check each
candidate once, not once per timing window. Failed expectations or unhandled
device errors suppress that candidate's score. Both individual kernels and
complete `command.program.def` subjects use this policy. Target compilation is
shared between checking and timing, and oracle storage is never replicated
across the timing ring.

Oracle preparation and execution add to total tool latency even though they are
outside the measured window. On realistic model shapes, a serial CPU/VM oracle
can dominate the run. `--scenario-correctness=false` explicitly restores the
default for benchmark suites. `loom-test` remains the dedicated correctness
tool.

This flag applies to `check.scenario`, not `check.case`. Case benchmarks
retain their correctness preflight; `case_end_to_end` also includes reference
calls and checks in the authored program being timed. Scenario subjects provide
the separation needed for target-only model benchmarking.
`--compile-only` never executes checks and rejects `--scenario-correctness=true`.

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
loom-benchmark program.loom \
  --benchmark=@decode_shapes \
  --sample=1 \
  --dry-run
```

`--sample` indexes the benchmark's sweep after applying its assignments, with
the first unassigned case parameter varying fastest. Pins do not consume a
sweep dimension, and the sample cap limits selected work without restricting
where those samples lie in the full case domain. Reports retain the exact
parameter values and benchmark-local replay index even when that full
Cartesian position exceeds the host's integer range; the optional global case
index is then omitted.

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

Dispatch measurements rotate through physical device-buffer storage sets. The
default ring approaches 32 MiB without using more distinct sets than input uses
in one batch, reducing accidental hot-cache results. The byte target unions
reachable binding ranges within each authored allocation, so repeated,
overlapping, and producer/consumer bindings count the same bytes once. Summed
binding-reference bytes and complete retained authored-input allocations remain
separate report evidence. Scenario-generated result buffers contribute their
owned reachable ranges without being reported as authored input storage.
`--input-ring-min-bytes=BYTES` changes the byte target without fixing the ring
count. A deliberate hot-reuse experiment states that choice directly:

```shell
loom-benchmark program.loom \
  --benchmark=@decode_4096 \
  --device=legacy-amdgpu \
  --measure=dispatch_complete \
  --input-ring-count=1 \
  --output=hot-reuse.json
```

The result records the effective ring count, binding-reference bytes, unique
reachable bytes, and complete retained-input-storage bytes under
`benchmark_result.data_cache`.
Comparisons are meaningful when candidates use the same data-reuse policy.

## Collect instrumented device timing outside the score

Final-batch profiling records and executes a separate metadata-retaining replay
of the same logical workload after the measured window:

```shell
loom-benchmark program.loom \
  --benchmark=@decode_4096 \
  --device=legacy-amdgpu \
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
loom-benchmark program.loom \
  --device=legacy-amdgpu \
  --compare=@baseline,@variant \
  --interleave=ABABA \
  --repetitions=2 \
  --measure=dispatch_complete \
  --output-format=jsonl \
  --output=comparison.jsonl
```

ABABA supports a baseline and one candidate. Round-robin interleaving supports
larger candidate sets. All candidates must benchmark cases, or all must
benchmark scenarios. Case candidates must select the same concrete parameter
layout and values; a parameterized case therefore uses `--sample=N` when more
than one sample remains.

Scenario comparisons likewise select one concrete configuration and trial per
candidate:

```shell
loom-benchmark program.loom \
  --device=legacy-amdgpu \
  --compare=@baseline_scenario,@variant_scenario \
  --sample=3 \
  --interleave=ABABA \
  --output=scenario-comparison.json
```

Separately named scenarios ordinarily have independent entropy namespaces. In
comparison mode, every candidate uses the baseline scenario's entropy namespace
so equivalent recipes produce equivalent inputs. Before timing, the tool checks
the complete retained input ring: invocation types, scalar payloads, tensor
shape and encoding, buffer bytes and ranges, and alias relationships must match.
Compilation and target-product preparation finish before the interleaved
schedule. The same deterministic input ring is rematerialized outside each
timed window, restoring inputs that a previous dispatch may have mutated. Each
repetition records the benchmark sample index and exact scenario configuration,
trial domain, and trial ordinal.

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
     p50: (.measurement.operation_timing_ns.p50 //
           .measurement.timing_ns.p50),
     warnings: .measurement.timing_interpretation.warnings}' results.json
```

JSONL is the better form for long sweeps and dashboards. Rows are flushed as
planning, compilation, correctness, measurement, profiling, and comparison
events occur. `run_id` and `candidate_id` fields join those events without
loading the complete run in memory.

The complete snapshot and event-row contracts, including queries for failures,
interleaved repetitions, retained artifacts, counters, and profile summaries,
live in [Read benchmark results](benchmark-results.md).

## Preserve the experiment

An artifact bundle captures the result, command line, source identity, device
identity, fixtures, and requested compiler or profile evidence:

```shell
loom-benchmark program.loom \
  --benchmark=@decode_4096 \
  --device=legacy-amdgpu \
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
[workflow map](index.md#compilation-and-evidence). The raw result and report
query recipes remain separately indexed by the question they answer.
