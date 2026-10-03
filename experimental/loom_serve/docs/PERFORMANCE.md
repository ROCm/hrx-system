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
The measured optimized Linux executable is 13,941,680 bytes (13.30 MiB),
including the embedded JIT. That is executable size, not the size of its model
sources, checkpoint, loaded memory, or external ROCr/system libraries.

A subsequent leased refresh used clean upstream
`bed0a856606ee4a24a164066f73d2379447033f5` with the same Vulkan settings:

| Serving configuration | Whole window | Appended inputs | Outputs |
| --- | ---: | ---: | ---: |
| Upstream target-only | 161.991 s | 28,727 | 3,072 |
| Upstream MTP depth 3 | 160.893 s | 28,533 | 2,880 |
| Loom source-JIT control afterward | 131.590 s | 28,534 | 3,072 |

The target-only row passes retained-output and retirement checks, but one
follow-up reprocesses more history, so its input work differs. The MTP row
uses the real draft context and accepts 62.6% of proposals; two requests stop
at EOS, including a four-token follow-up containing only `### Re-e`. Its
shorter window is not equivalent completed work. These observations remain
separate from the repeated pinned-control ratio above; HIP and other tuning
configurations have not been qualified by these runs.

Prompt-containing epochs account for about 90.5 seconds, versus 41.6-42.5
seconds in decode-only epochs. This is the reason to profile wide model math
next: removing host decode overhead alone cannot explain away most of the
remaining window. Dispatch profiling is a separate diagnostic run, never one
of the scored controls.

### FFN block read-ahead checkpoint

Commit `92f4875240` specializes the fused wide feed-forward kernel's model
dimensions through JIT configuration and pipelines next-block weight reads
across current matrix work. The [model guide](../models/qwen38/README.md#fused-feed-forward-projection)
records the source contract and complete source-level differential checks.
Arithmetic, weight format, retained state, scheduling policy and MTP depth
remain unchanged. This adds no HAL or command-program ABI extension.

The following A/B/C/B/A comparison used the same device, power policy, corpus
and settings above under one lease. Both Loom arms used the same optimized
executable, SHA256
`05560c0e167dc7537c3191cb9bf4d4b034d6dd76d609305d0eb4f450550ff452`.
Only `--model` changed: the source catalog at `f13ac7a028` versus the new
catalog. Each launch recorded source hashes, and both control source maps
matched that baseline commit. No precompiled model artifact was substituted.

| Run order | Serving configuration | Whole window | Aggregate output tokens/s |
| --- | --- | ---: | ---: |
| 1 | Loom, previous source | 131.991 s | 23.274 |
| 2 | Loom, block read-ahead | 128.826 s | 23.846 |
| 3 | Pinned llama.cpp `f7b384c1`, Vulkan, target-only | 162.680 s | 18.884 |
| 4 | Loom, block read-ahead | 129.450 s | 23.731 |
| 5 | Loom, previous source | 132.444 s | 23.195 |

Every window completes 16 length-limited responses, 28,534 appended inputs
and 3,072 outputs, with all second turns retaining state. Request frontiers,
output credits, epoch counters and clean shutdown agree with client usage.
Loom retires 183/177/183/182 epochs in its four windows. GPU maxima in run
order are 64/62/62/60/61 C, and package-power medians remain about 100 W.

Mean Loom time improves from 132.217 to 129.138 seconds: **2.38% higher
throughput**, versus 0.34% control spread and 0.48% candidate spread. Both
candidates beat both controls. Mean candidate throughput is **23.789 output
tokens/s, 26.0% ahead of this pinned comparator**. The comparator remains
target-only while Loom uses MTP; the newer-upstream qualification differences
above still apply. This is not a best-backend or same-MTP claim.

Equal token counts do not imply identical trajectories. Six replies differ
between the two controls; the candidate windows differ from the first control
in six and four replies, and all 16 comparator replies differ. Those histories
change subsequent batching. The full-server result therefore measures this
closed-loop workload; isolated kernel tests provide separate causal evidence.
The 15 JIT-stage preparations take 1.974/1.962 seconds for the candidates versus
1.829/1.836 seconds for the controls, excluding indexing, VM setup and loading.

Locked, completion-inclusive kernel A/B/A/B/A measurements use production
K=5120/N=17408, 128/256/512 token capacities, two physical binding sets and
canonical weights. Both read-ahead windows beat every original window at
each capacity, with individual gains of roughly 6-11%. A second comparison
holds dimensions, packet order and unrolling fixed and varies only pipelining;
it also improves all three capacities. Native inspection confirms next-block
loads outstanding across matrix instructions, 51,200 bytes of LDS, no spills,
and unchanged modeled occupancy despite increased register use. This is
partial weight-read overlap, not a claim of fully saturated memory bandwidth.
The host still drains each epoch before scheduling the next cohort.

### Narrow projection read-ahead checkpoint

Commit `ad4229c093` applies whole-block read-ahead to the 32-token Q5
projection. Complete and partial channel tiles specialize the same body;
gate and up remain separate kernels. Canonical encoded weights, arithmetic,
the command interface, scheduler and MTP policy are unchanged.

The source-only A/B/A/B/A comparison below uses baseline `2006b80c6e`, the
same optimized executable and the same eight-client retained-review protocol
as the preceding checkpoint. Every launch records the appropriate source
hashes. One lease covers all five windows on the same hardware and power
policy.

| Run order | Model source | Whole window | Aggregate output tokens/s |
| --- | --- | ---: | ---: |
| 1 | Baseline | 129.292 s | 23.760 |
| 2 | Narrow read-ahead | 126.277 s | 24.328 |
| 3 | Baseline | 128.281 s | 23.947 |
| 4 | Narrow read-ahead | 125.373 s | 24.503 |
| 5 | Baseline | 129.801 s | 23.667 |

Every window completes 16 length-limited responses, 28,534 appended input
tokens and 3,072 outputs, with all second turns retaining state. Client usage,
output credits, epoch frontiers and clean retirement agree. Mean window time
falls from 129.125 to 125.825 seconds: **2.62% higher throughput**, reaching
**24.415 aggregate output tokens/s**. Both candidate windows beat all three
controls; control and candidate spreads are 1.18% and 0.72%. GPU maxima are
61/62/61/61/60 C and package-power medians remain about 100 W.

This is another closed-loop self-comparison, not a refreshed external-runtime
ratio. Relative to the first control, 3/5/4/4 replies differ in the remaining
windows; those histories affect batching. Completed epoch counts are
178/183/178/180/183. The independent kernel differential and access checks
cover every channel-tail class, all production widths and zero workgroups;
real-model checks cover packed/isolated shapes, MTP, retained follow-ups,
request cancellation, conflicting session ownership and reuse after retirement.

Isolated, completion-inclusive ABABA measurements improve all three production
widths by roughly 1.35-1.50x, with two physical binding sets. Native evidence
shows complete packet reads outstanding across 32 matrix instructions. LDS
remains 23,040 bytes with no spills or private storage; VGPR use grows from 40
to 136 while modeled occupancy remains five waves. Native image size grows
from 9,232 to 46,056 bytes. The 15 JIT-stage preparations total about 2.318
seconds versus 1.965 seconds for the control. That cold cost is outside the
scored serving window, and neither instruction overlap nor modeled occupancy
establishes measured DRAM saturation.

### Prepared-weight checkpoint

Commit `f076aa0955` prepares FFN gate/up Q5 blocks in eight-channel order once
at startup. Every target and MTP consumer shares those final bytes; neither
quantization nor arithmetic changes. The source-JIT preparer uses only
workgroup-local scratch. Four independent read/preparation timelines replace
the separate target and auxiliary loading joins. The [model guide](../models/qwen38/README.md#online-weight-residency)
describes the layout and exact source-level comparisons.

This A/B/A/B/A uses baseline `ca760be2a0` and the prepared candidate, both built
on main `1618350bab` with the optimized flags above. One lease covers the same
eight-client, two-turn, MTP3 workload on the same gfx1151 system and power
policy. Executable and model-source identities were captured for every window.
The baseline executable SHA256 is
`063e0d8388eeb8c1d36eb87e4978f6c7b5b9c14d0bd796b36ea42cc95d2ec35f`;
the prepared executable is
`3ae93a14d6a96bf0470a31d0e57fb7f5682cdb75b73bc33d88ae507460beac09`.

| Run order | Weight layout | Whole window | Aggregate output tokens/s |
| --- | --- | ---: | ---: |
| 1 | Canonical | 125.952 s | 24.390 |
| 2 | Prepared | 125.233 s | 24.530 |
| 3 | Canonical | 126.424 s | 24.299 |
| 4 | Prepared | 125.553 s | 24.468 |
| 5 | Canonical | 125.923 s | 24.396 |

Mean time falls from 126.100 to 125.393 seconds: **0.56% higher throughput**,
or **24.499 aggregate output tokens/s**. Both candidates beat all controls;
control and candidate spreads are 0.40% and 0.26%. This is a small closed-loop
self-improvement, not a refreshed llama.cpp ratio. Every window completes
16 requests, 28,534 appended inputs and 3,072 outputs with retained follow-ups,
matching credits/frontiers and clean retirement. Epoch counts are
178/180/183/181/180; reply/usage differences from the first control are
0/4/3/1/3. Equal work totals still do not mean identical histories or cohorts.
GPU maxima are 61/62/61/61/61 C and package-power medians remain about 100 W.

The separate startup profile records four unique weight allocations totaling
20,207,190,016 bytes and 130 in-place preparation dispatches for 7,965,900,800
bytes of FFN tensors. Preparation sums about **73.6 ms of GPU work once at
startup**, not per request or epoch. Warm process-to-ready grows by about
325 ms, including about 256 ms of additional JIT work. Warm filesystem-cache
timing is not cold NVMe throughput, and summed dispatch duration does not
identify the startup critical path. Native consumers have no spills or private
storage and unchanged LDS; the narrow kernel grows from 136 to 144 VGPRs.
All 54 exact/access samples across 17 public cases and real retained/MTP/HTTP
lifecycle checks pass on the final compiler base.

Allocation profiling also exposes a cost that one-image accounting alone
would miss: device-local roots receive 20,207,181,824 bytes through HAL staging.
Two scoped-mappable dual-local alternatives remove that startup copy and pass
the numerical/lifecycle checks, but neither qualifies for inference performance.
Device-preferred placement gives a variable -3.50% mean throughput result;
system placement gives -3.66%, with both candidates slower than every control.
The latter control windows are 126.642/125.885/126.075 seconds versus
128.716/133.287 seconds for the candidates. These experiments change only final
weight allocation policy, not model sources or inference kernels. They do not
establish a precise cache-policy mechanism.

The committed path retains the faster device-local placement and bounded
staging. Direct final-buffer reads without an inference penalty remain an
unmet requirement, as do measured cold-I/O saturation and autonomous device
continuation. Removing a startup transfer is not itself a serving improvement.

### Q6 token-reuse checkpoint

Commit `2446d3198f` increases weight reuse in the FFN-down projection. The
shared metadata contraction gives each subgroup eight output fragments
instead of four, covering 256 token rows per workgroup. The attention and
recurrent-layer command programs select this geometry for their 256- and
512-token recipes. Smaller and intermediate recipes retain their previous
geometry. Runtime active counts remain device data; this changes neither
host scheduling nor canonical weights, accumulation order, shared global
scratch, or in-place residual ownership. The [model guide](../models/qwen38/README.md#q6-metadata-contractions)
contains the source-level comparison recipe.

The 2026-10-03 A/B/A/B/A below uses the same eight-client retained-review
workload, MTP depth three, 16K contexts, gfx1151 system and high/performance
policies. All five runs use the same optimized executable, SHA256
`3ae93a14d6a96bf0470a31d0e57fb7f5682cdb75b73bc33d88ae507460beac09`.
Only the source catalog changes, from `bce7929d3a` to `2446d3198f`; every
launch records its complete model-source identities. One measurement lease
covers the five fresh residencies, without concurrent builds or transfers.

| Run order | FFN-down recipe | Whole window | Aggregate output tokens/s |
| --- | --- | ---: | ---: |
| 1 | Previous | 123.325 s | 24.910 |
| 2 | Wider token reuse | 119.061 s | 25.802 |
| 3 | Previous | 122.678 s | 25.041 |
| 4 | Wider token reuse | 118.899 s | 25.837 |
| 5 | Previous | 122.473 s | 25.083 |

Mean time falls from 122.826 to 118.980 seconds: **3.23% higher throughput**,
or **25.819 aggregate output tokens/s**. Both candidates beat every control;
control and candidate spreads are 0.69% and 0.14%. Every window completes
16 length-limited responses, 28,534 appended inputs and 3,072 outputs, with
retained follow-ups, consistent output credits/frontiers and clean retirement.
Epoch counts are 184/180/184/180/182. Replies or usage differ from the first
control in 0/5/2/6/3 turns; these remain closed-loop cohorts, not identical
token trajectories. This is a controlled self-improvement, not a refreshed
llama.cpp or MTP-matched competitor ratio.

GPU maxima are 60–61 C, NVMe peaks at 37.85 C, and package-power medians remain
about 100 W. Mean cold JIT-stage preparation grows from 2.566 to 2.601 seconds;
mean warm-cache process-to-ready grows from 5.340 to 5.379 seconds. Those
startup differences are outside the scored window and do not measure cold
storage bandwidth.

The six older metadata entries emit byte-identical native code before/after,
both normally and with access instrumentation. The new entry uses 96 VGPRs
and 43,008 bytes of LDS, with no spills or private storage; access checks use
112 VGPRs without spills. All 90 public-library and 64 production-weight
bitwise/access samples pass, followed by host-ASAN full-model/MTP and retained
HTTP lifetime checks. That is focused device-access coverage, not a claim
that every full-model kernel was device-instrumented.

A separate completion-inclusive kernel ABABA screen fixes dispatch capacity
while varying device active count. All nine full and underfilled 256/512
cases improve projection throughput by 11.25–19.12%, beyond their respective
within-arm spreads, with two physical binding sets and no timing warnings.
This does not justify choosing a wide recipe for one ready token instead of
the existing narrow recipe. At capacity 512, the new geometry halves logical
weight/metadata acquisition while preserving activation payload; physical
DRAM traffic and bandwidth saturation are not inferred from source counts.

### Span-cardinality checkpoint

Commit `895113fbc4` keeps more ready sessions in an epoch when cached recipes
admit the same number of useful tokens, before breaking ties by smaller
capacity. A full prompt therefore cannot displace ready decode/verifier peers
merely because a single-span recipe also fills the token budget. The policy
still has no measured execution-cost model.

The following 2026-10-03 experiment changes only the cached shape catalog.
Both arms use that commit's optimized executable, SHA256
`6e8a95dd400f0cc5995616bd037e72a1911a19ffc67cf4069ae6fdf2c2d9587c`,
the model sources at `2446d3198f`, eight resident rows, 16K allocated context,
MTP depth three and the same gfx1151 system and high/performance policies. Control A has
token capacities 32/128/256/512 with eight spans each. Adaptive B caches the
cross-product of those token capacities with 1/2/4/8 spans. One lease covers
both ABABA sequences, with a fresh residency for every window.

| Active clients | A0 seconds | B1 seconds | A2 seconds | B3 seconds | A4 seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 57.162 | 52.740 | 56.937 | 52.489 | 56.848 |
| 8 | 121.472 | 120.975 | 122.038 | 121.052 | 120.564 |

With two active clients, mean window time falls from 56.982 to 52.614 seconds:
**8.30% higher throughput**, reaching **14.597 aggregate output tokens/s**.
Both candidates beat all controls, beyond the 0.55%/0.48% within-arm spreads.
All windows complete 5,783 appended inputs and 768 outputs, with retained
follow-ups, matching credits/frontiers and clean retirement. Controls have
identical replies and 157 epochs; both candidates change one reply and take
150 epochs. The result includes batching and acceptance changes, not only
cheaper dispatches. Authored target-head work falls from 4,832 padded rows to
1,156; actual requests are 1,148/1,112 rows. Logical padding is not physical
memory traffic, and the eight-row MTP proposal recipe remains unchanged.

At eight clients, mean times are 121.358/121.013 seconds. The apparent 0.29%
gain is below the 1.21% control spread, and the last control beats both
candidates: **no full-load speedup qualifies**. Every window still completes
28,534 appended inputs and 3,072 outputs with valid retained state and clean
retirement. Epoch counts are 182/180/183/181/178 and replies differ. Across
client counts the prompt sets also differ, so these rows are not a controlled
concurrency-scaling curve. Neither row is a new external-runtime ratio.

GPU peaks are 59–66 C for the two-client sequence and 60–61 C for eight;
package-power medians remain 99.1–100.0 W. Cold JIT preparation grows from
16 to 52 stages, approximately 2.62 to 8.81 seconds. Warm-filesystem-cache
process-to-ready grows from about 5.42 to 11.63 seconds, outside the scored
window. The logged arenas remain 18.819 GiB of unique weights, 9.169 GiB of
retained state and 0.201 GiB of shared workspace. Loaded native stage images
grow with the catalog; those arena figures are not total residency accounting.
No per-session copy of the weights or programs is introduced.

The existing eight-span benchmark above remains the baseline. To reproduce
the optional adaptive configuration, replace its four `--epoch` arguments
with `"${epoch_options[@]}"` after constructing this array in Bash:

```bash
epoch_options=()
for tokens in 32 128 256 512; do
  for spans in 1 2 4 8; do
    epoch_options+=("--epoch=$tokens:$spans")
  done
done
```

Keep `--rows=8` in both arms and change only the benchmark client's `--clients`
between two and eight. The [narrow-recipe differential](README.md#reproduce-full-qwen-and-retained-http-output)
at `215e9c0385` passes both complementary catalogs and the five-token/single-span
boundary with real retained target/MTP state. Actual eight-client HTTP checks
also pass output limits, retained codewords, natural EOS, ownership conflict,
cancellation after verification and reuse, including host ASAN. These checks
use ordinary device kernels, not full-model device-access instrumentation.
Allocated 16K capacity does not establish performance with 16K occupied history;
that remains a distinct measurement axis.

### Current matched-MTP qualification

A 2026-10-03 refresh pinned llama.cpp to
[`eec18f5d32099fb15d4ba15003a231bcc72757d5`](https://github.com/ggml-org/llama.cpp/tree/eec18f5d32099fb15d4ba15003a231bcc72757d5).
Both Vulkan and HIP were built from that clean source. HIP used ROCm 10.0.0,
targeted gfx1151, and enabled its default graph and matrix implementations.
The server loader, implementation libraries and runtime dependencies were
identified separately. The model, F16 target/draft caches, eight 16K slots,
batch 2048, microbatch 512, greedy sampling and natural EOS policy matched the
retained-review protocol above.

For real MTP, replace `--spec-type none` in the comparator command with
`--spec-type draft-mtp --spec-draft-n-max 3` and add
`--spec-draft-type-k f16 --spec-draft-type-v f16`. HIP selects `--device ROCm0`
and requires its runtime libraries on the execution host. Proposal acceptance
and creation of the MTP context were verified in the server log; merely
passing a speculation flag is not that evidence.

| Qualification window | Outputs, of 3,072 requested | Result |
| --- | ---: | --- |
| Vulkan, target-only | 3,072 | Complete bounded responses |
| Vulkan, MTP depths 1 / 2 / 3 | 2,883 / 2,884 / 2,885 | One short follow-up in each window |
| HIP, target-only | 3,072 | Complete bounded responses |
| HIP, first MTP depth-3 screen | 3,072 | Complete bounded responses |
| HIP, first fresh interleaved MTP arm | 2,516 | Three follow-ups stop after 5 / 5 / 10 tokens |

The interleaved comparison stopped at that failed HIP arm, before measuring a
Loom arm. All 16 requests retired without context truncation or transport
failure, and the GPU peaked at 61 C. A preceding successful HIP screen cannot
replace the failed measured arm. **There is no qualified current MTP-to-MTP
throughput ratio from this experiment.** The preceding Loom self-improvements
stand independently; they are not divided by an incomplete competitor window.

Fresh full-prefill replay of one captured short Vulkan continuation produced
192 tokens from the same 6,357 input IDs. An additional retained Vulkan run
with verbose token logging also completed all requested outputs. The behavior
therefore is not established as deterministic or Vulkan-specific. These
receipts do not identify its cause; the next diagnostic needs the selected
terminal token and target/draft retained frontiers from a failing run without
altering sampling. No EOS suppression, source workaround or filtered samples
were used to turn the failure into a score.

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
