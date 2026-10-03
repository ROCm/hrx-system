# Establish performance, then change the limiting mechanism

The objective is completed useful work for concurrent, bursty agents under a
memory and power budget. Single-stream decode latency, saturated prefill rate,
and accepted multi-agent output throughput are different measurements. A
portable JIT deployment is useful on its own; a speed claim still needs matched
hardware, checkpoint semantics, and workload evidence.

## A controlled comparison

The experiment identity includes source commit, executable/configuration,
weight and tokenizer hashes, device/runtime versions, context lengths, active
row distribution, shape table, proposal depth, prompt corpus, and output policy.
Power mode, clocks, temperature, and concurrent work are part of the record.
[`observe.py`](../observe.py) combines service events with available system
telemetry; [`dashboard.py`](../dashboard.py) presents that stream. Missing
sensor readings remain missing, not zero power or a healthy temperature.

[`qwen_epoch_check`](../qwen_epoch_check.c) establishes retained correctness;
[`qwen_workload`](../qwen_workload.c) compares controlled isolated/packed work;
[`qwen_replay`](../qwen_replay.c) replays retained text turns with scheduling
windows; [`benchmark_service`](../benchmark_service.py) exercises the actual
HTTP service. Their detailed options are in the [runner README](../README.md)
and [model guide](../models/qwen38/README.md). Fixed replay holds work constant;
live agents additionally expose feedback-dependent arrival patterns. Both are
valuable, but they answer different questions.

Host ASAN, device sanitizers, Tracy, and device profiling are diagnostic modes.
Performance uses an explicitly optimized build of the exact executable:

```sh
build_tools/bin/iree-bazel-build //experimental/loom_serve:qwen_server \
  //experimental/loom_serve:qwen_workload -c opt --features=thin_lto \
  --copt=-O3 --cxxopt=-O3 --host_copt=-O3 --host_cxxopt=-O3
```

Measurement begins after build activity ends, under the runner's exclusion
lease. Remote `-march=native` would describe the build worker, not the inference
machine; an ISA-specific build names its intended runner. Alternating A/B/A/B/A
runs help expose drift but cannot compensate for a different quantization,
output count, prompt distribution, or power regime.

The service already exposes useful ablations:

| Comparison | What it isolates |
| --- | --- |
| `--packing=separate` versus `mixed`, both packed | Sharing traversal capacity across prompt/decode cohorts |
| `--scheduler=matched` versus `packed` | Independent versus shared traversals using prefill math for decode controls |
| One `--epoch` versus several cached shapes | Occupancy-based shape selection at fixed model/state semantics |
| No `--mtp` versus `--mtp --mtp_depth=0` | Draft-state residency and maintenance without speculation |
| MTP depth zero versus three | Proposal, verification, acceptance, and extra committed output |

MTP requires packed scheduling. A comparison against another runtime records
its corresponding settings rather than describing different speculative modes
as equivalent. Proposed or rejected tokens are work, not delivered output.

The scorecard contains aggregate accepted output tokens/second, appended prompt
tokens/second, completed-turn time and latency distribution, useful tokens per
epoch, state/weight/workspace residency, cold JIT/startup time, and energy when
measurable. Service token counters include selected EOS; text-visible output can
use a separate count. Summing per-session token rates is not aggregate throughput.

## Retained-review benchmark

The checked-in [source-review workload](../testdata/source_review.json) contains
eight independent two-turn reviews with different prompt lengths. Each turn
requests at most 192 output tokens. Its [provenance](../testdata/README.md)
records the public source excerpts and immutable corpus hash. A fresh agent
needs no private recordings to reproduce this window.

After the correctness checks in the [packet README](README.md), build the exact
optimized `qwen_server` target with the flags above. Set `model_dir` to the
pinned checkpoint directory from that README. On the qualified execution host,
select a new `run_dir`, acquire its measurement lease, and start one residency:

```sh
python3.12 -B -m experimental.loom_serve.observe \
  --log="$run_dir/observe.jsonl" -- \
  bazel-bin/experimental/loom_serve/qwen_server \
  --model=experimental/loom_serve/models/qwen38 \
  --weights="$model_dir/Qwen3.8-27B-UD-Q5_K_XL.gguf" \
  --tokenizer="$model_dir/tokenizer.json" \
  --prefill_capacity=512 --context_capacity=16384 --rows=8 \
  --epoch=32:8 --epoch=128:8 --epoch=256:8 --epoch=512:8 \
  --mtp --mtp_depth=3 --scheduler=packed --packing=mixed \
  --port=8080 --heartbeat_ms=1000
```

The output directory already exists; `observe.jsonl` must be new. Readiness is
the JSON `ready` event, followed by a successful `/healthz` response. Once ready,
run the client in another terminal on that host while the same lease is held:

```sh
python3.12 -B experimental/loom_serve/benchmark_service.py \
  --url=http://127.0.0.1:8080 --clients=8 --session-prefix=review-0 \
  --workload=experimental/loom_serve/testdata/source_review.json \
  > "$run_dir/clients.jsonl"
```

The qualified Qwen window completes 16 requests, 28,534 appended input tokens,
and 3,072 output tokens. Each second turn has a nonzero retained-prefix hit.
The client records full replies, submitted-history hashes and usage; it rejects
empty or incomplete responses and missing retention. An earlier EOS remains
visible in `finish_reason` and output counts; it is not equivalent completed
work merely because the client exits successfully. These are bounded review
continuations, not a claim that every review finishes within 192 tokens.

After client completion, SIGINT/SIGTERM to the observer drains the owned server.
The final heartbeat must report every issued epoch completed, and `run_end`
must record exit zero. Sum epoch prompt and selected-output counters against
the client summary before using the time. This is whole-cohort elapsed time
including prompt work, SSE and retained follow-ups, excluding model loading
and JIT. Repeating a window means a fresh residency and a new output directory;
otherwise old session rows or warm request histories change the experiment.

The comparison configuration for the pinned Vulkan llama.cpp control is:

```sh
llama-server -m "$model_dir/Qwen3.8-27B-UD-Q5_K_XL.gguf" \
  --host 127.0.0.1 --port 8080 --alias qwen3.8-27b --parallel 8 \
  --kv-unified --kv-unified-per-slot 16384 --cont-batching \
  --flash-attn on --gpu-layers all --device Vulkan0 --fit off \
  --cache-type-k f16 --cache-type-v f16 --batch-size 2048 --ubatch-size 512 \
  --threads 2 --threads-batch 2 --backend-sampling \
  --jinja --reasoning off --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 0 --repeat-penalty 1 --presence-penalty 0 --frequency-penalty 0 \
  --spec-type none
```

Run it through the same observer/client, with readiness from the server and
its `/health` endpoint. Executable and loaded-library identities both matter:
the server executable can be a small loader. This control is explicitly
target-only; Loom uses depth-three MTP. The resulting ratio compares complete
serving configurations, not equal speculative algorithms or equal generated
text. Requalification of newer upstream, another backend, or MTP uses the same
completed-output and retained-state gates before replacing the control.

### Measured source-JIT checkpoint

On 2026-10-02, commit `3ffe90573b` completed the following interleaved window on
a gfx1151 Strix Halo engineering system with 128 GB unified memory. The GPU
policy was `high`, CPU governors were `performance`, and measured package-power
medians were about 100 W. A single measurement lease covered all three windows.

| Run order | Serving configuration | Whole window | Aggregate output tokens/s |
| --- | --- | ---: | ---: |
| 1 | Loom, source JIT, MTP depth 3 | 133.022 s | 23.094 |
| 2 | llama.cpp `f7b384c1e5c5b2c5b321a4a7cefea04b15b54cb7`, Vulkan, target-only | 163.620 s | 18.775 |
| 3 | Loom, same configuration | 132.200 s | 23.238 |

All three complete the same 28,534 appended inputs and 3,072 outputs over 16
retained requests. Client/epoch accounting and clean retirement pass; both
Loom runs retire all 182 epochs. Peak GPU temperatures are 64/62/61 C. The
mean Loom window gives **23.4% higher throughput than this pinned control**,
with 0.62% spread between Loom repeats. This is not a latest-upstream/HIP,
same-MTP, bitwise-output, or universally best-backend claim. Replies differ
across engines and three replies differ between Loom repeats, changing the
follow-up histories despite equal aggregate token work.

The first run's 15 command-stage preparation events total 1.823 seconds,
including native compilation and loading. Source indexing, VM setup and weight
loading are outside that sum. All model inputs are portable source, checkpoint
and tokenizer files; no prepared native image is a deployment input.

Prompt-containing epochs account for about 90.5 seconds, versus 41.6-42.5
seconds in decode-only epochs. This is the reason to profile wide model math
next: removing host decode overhead alone cannot explain away most of the
remaining window. Dispatch profiling is a separate diagnostic run, never one
of the scored controls.

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

[`simulate_packing.py`](../simulate_packing.py) can explore recorded arrivals and
shape policies before device runs. Fewer modeled epochs are evidence about
amortization opportunity, not that each epoch costs the same. Device costs
calibrated by shape, span count, context, and proposal mode turn that replay
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

## Changes worth testing in this model

The current Qwen sources provide concrete examples, not universal winners:
the fused Q5 gate/up contraction shares encoded weight staging and publishes
SwiGLU directly; Q6 contractions stage metadata and pipeline acquisition; output
projection groups the requested output cohort; MTP proposal rounds feed device
buffers without intermediate host reads. Their [kernel guide](../models/qwen38/README.md)
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
