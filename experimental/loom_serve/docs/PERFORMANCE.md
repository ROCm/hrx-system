# Establish performance, then change the limiting mechanism

The objective is completed useful work for concurrent, bursty agents under a
memory and power budget. Single-stream decode latency, saturated prefill rate,
and accepted multi-agent output throughput are different measurements. A
portable JIT deployment is useful on its own; a speed claim still needs matched
hardware, checkpoint semantics, and workload evidence.

The [live pi repair workload](../tools/agent_tasks/README.md) adds eight distinct
source repairs with real file tools, executable checks and retained recall.
It records response-dependent trajectories rather than replaying one recording
across every agent. Its completed-task score complements exact-input controls;
the task corpus is intentionally small and does not measure general coding
ability.

## Measurements available to clients and observers

The current HTTP load client is
[`benchmark_service.py`](../models/qwen/benchmark_service.py), a retained-text
workload rather than a general multimodal benchmark command. It targets the
registered deployment selected by `--model`, starts the requested number of
concurrent clients, checks complete SSE responses and retained follow-ups, and
emits per-request measurements plus whole-cohort throughput. The independent
[observer](../tools/observe.py) supplies server-side accounting for any supported
modality without owning the workload or server lifetime.

| Surface | Available measurements | Timing boundary |
| --- | --- | --- |
| Text HTTP client | `first_text_ms`, `request_ms`, usage, finish reason and aggregate selected output tokens/s | Client request start to first nonempty content delta or complete successful SSE, including network and queueing |
| Text owner | `queue_ns`, `first_token_ns`, request duration, consumed inputs, selected outputs and retained tokens | Queue time is separate; first-token time starts at admission and ends at host collection of the first output batch, possibly EOS |
| Image owner | Queue, preparation, parameter admission, submission, execution, PNG and JSON durations; completed images/pixels and PNG bytes | Host-observed phases; execution includes waiting for device readiness and is not a GPU kernel timestamp |
| Embedding/decision owner | Request duration/outcome, input tokens, cohort counts and queued/active ages | Server receipt through owner retirement, not client receipt; no image-style phase breakdown |
| Shared server | Completed/rejected/cancelled work, pool/weight/workspace observations and parameter activation/read counts | Cumulative owner snapshots, independent of retained event history |
| Optional local observer | Process memory/CPU and available power, clocks, temperatures and policy channels | Linux host named by `--local-pid`, not automatically the remote server host |

Time to first text is not interchangeable with first-token availability: a
tool-only response has no nonempty text sample. MTP can return several tokens
in one output batch, so SSE arrival gaps would measure chunk cadence, not
individual token latency. The client records each nonempty text/tool output
chunk's arrival offset in `output_chunk_ms` and the request's monotonic
start/completion timestamps. Its summary reports nearest-rank p50/p95/p99
request and first-text latencies, with explicit sample counts: a missing text
sample is not zero latency. It has no image/vector/decision workload mode. Its
JSON contains full replies, unlike the payload-free server observation stream.

Frozen workloads can give each session a nonnegative integer `start_after_ms`,
relative to the cohort release, and each turn a `pause_ms` before issuing it.
Those delays model agent arrivals and time spent in tools; they are included in
whole-cohort elapsed time. Recorded request timestamps describe actual starts,
including client scheduling delay. The workload must contain at least as many
independent sessions as `--clients`; the client never silently repeats a corpus.
The built-in codeword workload also supports more than eight clients.

`--retention=required` preserves the default assertion that every follow-up
reuses state. Oversubscribed experiments may explicitly select `observe` to
record best-effort cache misses while still validating complete replies and
codeword recall. The summary names that policy and reports cached versus
appended prompt tokens. A cache miss is never reported as retained work.
`followup_cache` separates all subsequent turns into `hits` (any cached prompt
tokens, including partial reuse) and `misses` (zero cached prompt tokens), with
request counts, cached/appended tokens and latency distributions for each.
Initial turns are excluded even if they hit a shared prefix. Sessions with only
one turn contribute no follow-up samples. These groups expose lost reuse under
row or memory pressure without attributing a miss to a cause the client cannot
observe. Their latencies include network and queueing, not just replay cost.

The shared server's text catalog separates `rows` (execution capacity),
`session_cache_capacity` (extra idle conversations), and `checkpoint_capacity`
(explicit pins). Automatic retention defaults to zero. Enabling it lets idle
histories survive row turnover within the shared memory budget; it does not
create more execution rows or reserve full context slabs. The
[idle-session ownership guide](../text/README.md#idle-sessions-beyond-execution-capacity)
describes its memory tradeoff and `text_memory.session_cache` observations.

`prefix_cache_capacity` independently retains exact consumed-token frontiers
for sharing across fresh requests. Zero provides a same-binary disabled control.
Compare initial-turn cached/fresh input, capture readiness, first-content time,
whole-task time and peak memory; a hit count alone is not a throughput result.
Natural prefill boundaries determine which prefixes can match, without forcing
smaller epochs. `text_memory.prefix_cache` and `prefix_capture` expose retention
and readiness separately from completed-session reuse and device epoch work.

`admission_overtakes` controls bounded backfill independently of epoch packing.
The default permits one execution-row cohort of successful overtakes per
blocked oldest request; zero provides a FIFO control. `admit.queue_rank`,
`overtakes` and `overtake_budget` identify each choice. A useful comparison
includes variable completion reservations that actually pressure the pool and
reports completed work, queue/TTFT distributions, and cache reuse together.
Spare logical positions alone are not a measured throughput gain.

Prompt usage counts the actual native context: retained positions plus appended
inputs. It need not equal a fresh tokenization of the same HTTP history. For
example, Qwen's source policy normalizes whitespace in its chat-history key but
continues the original generated token IDs. A trailing generated newline can
remain in a retained context while fresh rendering trims it. Comparisons record
this real work difference separately from visible reply identity; neither a
matching reply nor identical request JSON establishes identical token
trajectories. The reported counts describe those actual trajectories.

Whole-cohort output rate includes selected EOS and uses one elapsed window,
not a sum of individual request rates. A mixed-modality result retains each
model's units and completion latencies; adding tokens, pixels and vectors would
not produce a meaningful common throughput. Request distributions require
complete client samples or an explicit observation-coverage report. Overwritten
server events cannot silently disappear from a latency distribution, even
though cumulative counters remain exact.


## A controlled comparison

The experiment identity includes source commit, executable/configuration,
weight and tokenizer hashes, device/runtime versions, context lengths, active
row distribution, shape table, proposal depth, prompt corpus, and output policy.
Power mode, clocks, temperature, and concurrent work are part of the record.
[`observe.py`](../tools/observe.py) attaches to `/logz` without owning the server;
`--local-pid` explicitly adds system telemetry from the observer host.
[`dashboard.py`](../tools/dashboard.py) presents that stream using cumulative
owner snapshots rather than summing lossy event history. Missing
sensor readings remain missing, not zero power or a healthy temperature.

[`qwen_epoch_check`](../models/qwen/epoch_check.c) establishes retained correctness;
[`qwen_workload`](../models/qwen/workload.c) compares controlled isolated/packed work;
[`qwen_replay`](../models/qwen/replay.c) replays retained text turns with scheduling
windows; [`benchmark_service`](../models/qwen/benchmark_service.py) exercises the actual
HTTP service. Their detailed options are in the [runner README](../README.md)
and [model guide](../models/qwen/README.md). Fixed replay holds work constant;
live agents additionally expose feedback-dependent arrival patterns. Both are
valuable, but they answer different questions.

Host ASAN, device sanitizers, Tracy, and device profiling are diagnostic modes.
Performance uses an explicitly optimized build of the exact executable:

```sh
build_tools/bin/iree-bazel-build --config=opt \
  //experimental/loom_serve/serving:server \
  //experimental/loom_serve/models/qwen:workload
```

Measurement begins after build activity ends, under the runner's exclusion
lease. Remote `-march=native` would describe the build worker, not the inference
machine; an ISA-specific build names its intended runner. Alternating A/B/A/B/A
runs help expose drift but cannot compensate for a different quantization,
output count, prompt distribution, or power regime.

Identical request text and replies also need not produce identical epoch work.
Concurrent arrival order changes physical row assignment; rotating prompt
expansion and independently arriving follow-ups can change batch partitions.
Whole-service comparisons keep that effect in the result. Kernel attribution
additionally checks epoch counts, shape/mix and speculative progress, with
observation gaps disclosed rather than silently treating an incomplete trace as
the complete workload.

The service already exposes useful ablations:

| Comparison | What it isolates |
| --- | --- |
| `--packing=separate` versus `mixed`, both packed | Sharing traversal capacity across prompt/decode cohorts |
| `--scheduler=matched` versus `packed` | Independent versus shared traversals using prefill math for decode controls |
| One `--epoch` versus several cached shapes | Occupancy-based shape selection at fixed model/state semantics |
| `--draft_depth=0` versus a source-supported nonzero depth | Draft residency, proposal, verification, acceptance, and extra committed output |

MTP requires packed scheduling. A comparison against another runtime records
its corresponding settings rather than describing different speculative modes
as equivalent. Proposed or rejected tokens are work, not delivered output.

The scorecard contains aggregate accepted output tokens/second, appended prompt
tokens/second, completed-turn time and latency distribution, useful tokens per
epoch, state/weight/workspace residency, cold JIT/startup time, and energy when
measurable. Service token counters include selected EOS; text-visible output can
use a separate count. Summing per-session token rates is not aggregate throughput.

### Image submission granularity

Image registrations can set `command_quantum` to partition one compiled program
at full-barrier boundaries. Zero submits the whole program. All parts reuse the
same executables, parameters and original workspace; no intermediate tensors
are read back. A positive target bounds command count, not execution time.

Compare complete-image latency with text progress and the longest per-session
silence while both models are active. Arrival phase matters: text prefill and
established decode streams put different pressure on the shared device. Record
workspace overlap and peak memory as well as completion time.

## Concurrent source compilation

The runner shares one application-owned IREE executor across catalogs.
`loom_serve_jit_compile_async` queues product construction and native requests;
the final request assembles the result without a worker waiting for siblings.
Cold constructors still join from their calling thread. `--workers` controls
the shared population, defaulting to eight, rather than eight workers per
model.

The current JIT retains exact native source/root/specialization inputs within
each catalog's fixed compiler and loader domain. Separate command programs keep
their own bindings and executable references; warm execution never queries this
cache. This removes repeated compilation and executable residency, not model
weight traffic or inference math.

Measure cold compilation, process-to-readiness and peak memory separately from
warm inference. Hold the source catalog, shapes and worker count fixed when
isolating cache reuse. Warm execution never queries the native compilation
cache; a startup improvement is not an inference-throughput claim.

## Retained-review benchmark

The checked-in [source-review
workload](../models/qwen/testdata/source_review.json) contains eight
independent two-turn reviews with different prompt lengths. Each turn requests
at most 192 output tokens. Its [corpus
guide](../models/qwen/testdata/README.md) describes the public source excerpts
and closed-loop behavior. A fresh agent needs no private recordings to
reproduce this window.

After the correctness checks in the [packet README](README.md), build the exact
optimized `text:server` target with `--config=opt`. Set `model_dir` to the
checkpoint directory from that README. On the qualified execution host, select
a new `run_dir`, acquire its measurement lease, and start one residency in the
first terminal:

```sh
bazel-bin/experimental/loom_serve/text/server \
  --model=experimental/loom_serve/models/qwen \
  --weights="$model_dir/Qwen3.8-27B-UD-Q5_K_XL.gguf" \
  --tokenizer="$model_dir/tokenizer.json" \
  --prefill_capacity=512 --context_capacity=16384 --rows=8 \
  --epoch=32:8 --epoch=128:8 --epoch=256:8 --epoch=512:8 \
  --draft_depth=3 --scheduler=packed --packing=mixed \
  --port=8080 --heartbeat_ms=1000
```

The output directory already exists; `observe.jsonl` must be new. Readiness is
the JSON `ready` event, followed by a successful `/readyz` response. In a second
terminal, attach the independent observer before issuing the workload:

```sh
python3.12 -B -m experimental.loom_serve.tools.observe \
  --url=http://127.0.0.1:8080 --log="$run_dir/observe.jsonl"
```

An explicit `--local-pid` adds system measurements when this observer runs on
the inference host. It is not a remote process identifier. Run the client in
another terminal on that host while the same measurement lease is held:

```sh
python3.12 -B experimental/loom_serve/models/qwen/benchmark_service.py \
  --url=http://127.0.0.1:8080 --clients=8 --session-prefix=review-0 \
  --workload=experimental/loom_serve/models/qwen/testdata/source_review.json \
  > "$run_dir/clients.jsonl"
```

The eight-session workload issues sixteen requests. Each second turn must have
a nonzero retained-prefix hit when retention is required. The client records
full replies, submitted-history hashes and usage; it rejects empty or
incomplete responses and missing retention. An earlier EOS remains visible in
`finish_reason` and output counts; it is not equivalent completed work merely
because the client exits successfully. These are bounded review continuations,
not a claim that every review finishes within 192 tokens.

After client completion, collect current `/v1/status` and `/logz` snapshots
while the server is still alive: queued/active work must retire, and cumulative
selected-output deltas must agree with client completion usage. Text-owner
`input_tokens` includes both prompt and decode inputs, not just client prompt
usage; the cumulative `text_work.prefill_tokens` delta corresponds to the
client's uncached prompt work. Detailed epoch histories can explain the mix only
when their coverage is complete; a reader gap does not invalidate current owner
counters. SIGINT/SIGTERM to the observer stops only observation. The caller
separately stops the server and verifies its exit
status; an HTTP endpoint cannot guarantee delivery of its own teardown events.
This is whole-cohort elapsed time including prompt work, SSE and retained
follow-ups, excluding model loading
and JIT. Repeating a window means a fresh residency and a new output directory;
otherwise old session rows or warm request histories change the experiment.

Other runtimes need their own diagnostics and host measurements; Loom's
observer requires the Loom observation protocol. Match quantization,
speculation, cache policy and output constraints explicitly. Natural EOS and
different replies can change later requests, so equal output caps do not
establish equal work.

## Traffic and batching hypotheses

For one completed traversal, let `D` be modeled bytes reaching the relevant
memory level, `F` the arithmetic work, `B` sustainable bandwidth, and `P`
sustainable compute throughput. An idealized lower bound is:

```text
epoch time >= max(D / B, F / P)
D = weight traffic + retained-state traffic + activation/scratch traffic
useful throughput <= committed useful tokens / epoch time
```

This is a bound, not a measured duration. Launch cost, synchronization,
dependencies, unhidden latency, and contention can make execution slower.
Encoded checkpoint size is not automatically physical weight traffic: different
tiles may reread data, caches may serve it, and metadata/dequantization have
their own cost. The memory level and reuse assumptions must accompany `D`.

The hybrid-epoch hypothesis is that additional prompt or verification tokens
increase useful work faster than they increase traversal time while weight
traffic dominates. Once compute, KV traffic, workspace, or hold time dominates,
a larger batch can lose. A shape chooser therefore eventually needs measured
cost and admission constraints, not only a count of filled slots. The current
chooser explicitly implements the latter.

MTP wins throughput when fewer target traversals per committed token outweigh
drafting, wider verification, state capture/publication, and foregone useful
prompt work. Acceptance alone is insufficient: a high-acceptance draft can
still cost more than the traversals it saves. Fixed target-only, warm-depth-zero,
and depth-three controls separate those costs.

[`simulate_packing.py`](../tools/simulate_packing.py) can explore recorded arrivals and
shape policies before device runs. Its default minimum-first packer is checked
against the native C implementation for the same ready rows and shape catalog.
The surrounding replay fixes each client's row and assumes resident state;
request admission, KV pressure and output credits still require live evidence.
Fewer modeled epochs are evidence about amortization opportunity, not that each
epoch costs the same. Device costs calibrated by shape, span count, context,
and proposal mode turn that replay
into a stronger predictor; they still need held-out workload validation.

## Read compiler evidence before changing kernels

The canonical [compile-report guide](../../../loom/docs/src/workflows/compile-reports.md)
and [query guide](../../../loom/docs/src/workflows/compile-report-queries.md)
separate emitted artifact facts from compiler analysis. The tool entry points
in this checkout are:

```sh
build_tools/bin/iree-bazel-run //loom/src/loom/tools/loom-compile -- --help
build_tools/bin/iree-bazel-run //loom/py/loom/tools:loom-compile-report -- --help
```

For a self-contained native kernel or a linked, equivalently specialized
inspection input, the documented workflow is:

```sh
loom-compile kernel.loom --format=amdgpu-hsaco --target=amdgpu:gfx1151 \
  --output=/tmp/kernel.hsaco --compile-report=details \
  --compile-report-output=/tmp/kernel.report.json
loom-compile-report show /tmp/kernel.report.json
loom-compile-report diff /tmp/baseline.report.json /tmp/kernel.report.json
loom-compile-report suggest /tmp/kernel.report.json
```

This standalone diagnostic compile is not the serving deployment path. The
equivalent embedding hook is `loomc_compile_report_options_t` attached to
`loomc_emit_options_t.next`, returning a report artifact from the same prepared
module. The current runner does not expose a report-output flag. Adding that
optional cold-path sink would preserve the exact live-profile/configuration
identity; serializing IR alone loses invocation configuration evidence.

Final register counts, LDS allocation, materialized spills/reloads, instruction
mix, source-selected matrix fragments, lane access geometry, and wait reasons
explain why a candidate might help. Missing fields are unavailable evidence.
Reported issued bytes are compiler economics, not measured memory-controller
transactions. Modeled occupancy is constrained jointly by registers, LDS, and
launch resources; reducing one tied limit may buy no additional residency.

The [loop-scheduling walkthrough](../../../loom/docs/src/workflows/tune-loop-schedules.md)
shows `scf.for` read-ahead, unrolling, and recurrence scheduling. A source
pipeline only helps when emitted loads remain outstanding across useful work.
Address-register reuse or added pressure can force early waits or spills.
Depth one is the serial control; matched arithmetic and launch geometry keep
the experiment interpretable.

Panel width and subgroup size change that scheduling problem. The
[signed-byte projection motif](../motifs/tensor/linear_i8.loom) already emitted

Panel width, pipeline depth, subgroup size and per-wave operand reuse are
independent schedule choices. A wave-size change requires corresponding
fragment, accumulator, scale-gather and store ownership. Larger panels can
amortize barriers while increasing LDS and register pressure. Reduced register
use or higher modeled occupancy does not by itself establish a speedup.
Independent arithmetic checks establish mapping correctness; complete consumer
measurements establish whether an isolated gain survives real cache
competition.

## Workgroup traversal is an independent tuning dimension

The [agent authoring
workflow](../../../loom/docs/src/workflows/agent-driven-kernel-development.md#workgroup-traversal-is-a-schedule)
separates lane coalescing, workgroup-local reuse, and reuse between workgroups.
A bijective mapping from physical workgroup IDs to logical tiles can bound
revisited operand panels without changing arithmetic, issued bytes or launch
count. Tail groups must retain unique coverage. Nearby workgroup IDs do not
promise execution order or synchronization.

Traversal selection belongs to the model's shape-specialized caller. Compare
orientations and group sizes across the production shape range; a mapping that
helps a large contraction can hurt a smaller one. Inspect the native coordinate
transform and tails, then return to the full consumer to measure the effect of
its actual cache competition.

## Changes worth testing in this model

The current Qwen sources provide concrete examples, not universal winners:
the fused Q5 gate/up contraction shares encoded weight staging and publishes
SwiGLU directly; Q6 contractions stage metadata and pipeline acquisition; output
projection groups the requested output cohort; MTP proposal rounds feed device
buffers without intermediate host reads. Their [kernel guide](../models/qwen/README.md)
links numerical comparisons at production dimensions and tail shapes.

A candidate record states the bottleneck, one proposed mechanism, baseline,
correctness witness, expected resource change, success threshold, and stop
criterion. A kernel win then returns to the full retained workload: a faster
contraction can increase workspace, reduce batch admission, or shift the
dominant cost elsewhere. A measured end-to-end gain authorizes integration;
more infrastructure around an unconfirmed hypothesis does not.

Larger opportunities change traffic rather than just instruction scheduling:
better output compaction, preparing a machine-specific weight layout once,
sharing prefix blocks, device-owned continuation, and within-stage resident
pipelines. Each needs an ownership/correctness witness and a measured traffic
budget before expanding. NPU projections at 50 or 200 GB/s are hypotheses under
that same accounting, not performance numbers borrowed from a GPU run.
