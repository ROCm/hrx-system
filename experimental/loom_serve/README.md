# Loom model runner experiment

This branch-local runner JIT-compiles portable model source into VM control,
command programs, and GPU executables using the public Loom C API. Its private
interfaces are intended to change with real models.
It is not a general HAL VM module or a serving framework.

The [model-authoring packet](docs/README.md) walks through reproduction,
compiler/runtime ownership, a proposed second-model port, and performance
experiments. Its [agent entry point](docs/AGENT_START.md) gives a first assignment,
source-reading order, observable success gates, and failure triage.

## Package ownership

Build rules live beside their implementation and tests. Shared packages have
model-neutral contracts; model packages own checkpoint names, graph assembly,
configuration, numerical policy, and model-specific reference checks.

| Package | Responsibility |
| --- | --- |
| [`runtime/`](runtime) | Source JIT, command materialization, VM imports, weight streaming and queue timelines |
| [`http/`](http) | TCP carrier and bounded HTTP connection/request storage |
| [`text/`](text) | Source-defined packed autoregressive residency, chat protocol, admission and retained token scheduling |
| [`image/`](image) | Source-defined diffusion residency, image request validation, completed-image service and output encoding |
| [`storage/`](storage) | Physical block accounting and logical page maps |
| [`scheduling/`](scheduling) | Allocation-free ready-span packing and token/span shape selection |
| [`motifs/`](motifs) | Reusable tensor math and GGML format kernels, specialized by model source |
| [`models/krea/`](models/krea) | Krea source programs, checkpoint/request policy and model-specific reference checks |
| [`models/qwen/`](models/qwen) | Qwen model programs, chat/state policy and reference checks |
| [`tools/`](tools) | Observation, recording, workload replay, simulation and component checks |

The shared native headers are experimental model-author interfaces, not a
stable ABI. Production text and image binaries have no model package dependency.
Only runtime consumes compiler-private command reflection; model callers use its serving
interfaces. Each model's `:sources` target publishes its source-JIT catalog and
shared motifs, not compiled artifacts. Both generic modality binaries require
an explicit `--model` selecting an external source catalog. The catalog resolves
paths relative to its model directory; copying that directory alone is not a
complete source package.
The [motif guide](motifs/README.md) describes composition and shape contracts.

For example, the image server is
`//experimental/loom_serve/image:server`; the text server is
`//experimental/loom_serve/text:server`. Full serving coverage runs with
`build_tools/bin/iree-bazel-test --config=asan //experimental/loom_serve/...`.

## Shared execution

`runtime/command.c` combines a parsed portable command program with loaded executable
reflection. Recording retains fixed buffers and code. Rebindable slots include
model state and explicit scratch; the materializer allocates no device backing.
There is no semantic-resource extension to the command artifact format.
Barrier recording accounts for all commands in each concurrent wave. Indirect
count consumers require command-processing visibility, and later producers
wait for those reads before reusing count storage. Direct-only waves retain
dispatch/transfer scopes; replay adds no host planning or allocation.

`runtime/module.c` exposes typed `execute_N(i32 stage, N hal.buffer) -> i64` imports
over one retained command table. Model source selects the stage and buffers;
the native call submits and returns without waiting. HAL captures bindings and
retains their buffers independently of the VM invocation. One process can serve
many rows; its preallocated native binding scratch is reused between calls.
The `feedback` import targets bounded host spans registered at startup. Source
can fork feedback between commands without borrowing a VM temporary's memory.

`runtime/execution.c` is the shared execution capability used by the model and host
I/O. Commands and input transfers advance one timeline that serializes shared
scratch use across exact queues. Feedback downloads wait on their producer and
prior feedback, then advance a separate timeline without ordering later model
work. Their device sources remain immutable until feedback completion; host
payloads remain borrowed through that completion. Each returned integer belongs
to its execution object and named timeline, not a global completion namespace.
Queue rejection advances neither frontier. Final drain joins both branches and
propagates their failures before releasing any borrowed storage.

The control integration test compiles token-row state kernels, command programs,
and a VM entry from source. Two retained rows share code, commands, and one VM
process while one row pauses and resumes. Checked outputs cover token history,
position, padding, and EOS. A rejected native binding must leave the execution
domain usable. Feedback forks preserve the work frontier while later VM calls
consume independent retained state. This is an ownership/control witness,
**not a full model run or a performance result**. The test compiles its
portable fixtures in process for the live GPU, using the serving JIT path.

A source-authored continuation command reuses one transient count tuple across
eight producer/consumer steps. A direct heartbeat and an indirect state update
share each consumer wave. Sixteen queued replays run without intermediate host
waits: zero X/Y/Z/all-axis grids suppress exactly half of the state updates,
while all producer and heartbeat steps complete. Both ordinary and retained
profile recording use the same emitted command artifact.

From the worktree root:

```sh
build_tools/bin/iree-bazel-test --config=asan //experimental/loom_serve/runtime:control_test
```

The separate `jit_test` uses the serving compiler path itself: a source-defined
configuration default and explicit override, live GPU profile, native command
loading, VM JIT, and shared retained device state. It destroys compiler/source
storage before executing the prepared commands and verifies the device result.
It also checks that rejected configuration leaves the compiler reusable.

```sh
build_tools/bin/iree-bazel-test --config=asan //experimental/loom_serve/runtime:jit_test
```

Weights, KV pools, model forward stages, scheduling policy, and transport are
separate from this coarse execution boundary. None is inferred from buffer
contents or command names by the native submission code.

## Source-defined diffusion image generation

The [Krea 2 Turbo guide](models/krea/README.md#generate-an-image-natively)
provides a second, non-autoregressive caller: prompt and seed through native
IREE tokenization, source-JIT text encoding, conditioning, eight denoising steps
and the complete VAE to a PPM image, with optional softwatercolor LoRA.
The generic [`generate.c`](image/generate.c) uses the same device, source
compiler, weight loader and execution timelines as Qwen. The
[diffusion-image ABI](image/model.h) lets source bootstrap declare command
stages, checkpoint domains, tokenizer asset and opaque request state. Source VM
control frames prompts, selects a stage and returns one opaque upload payload;
the native input module supplies bounded tokenization and vocabulary lookup.
Source command programs generate numerical inputs and own all model loops and
temporary lifetimes. The generic
[image server](models/krea/README.md#serve-images-over-http) returns native
PNG images over HTTP using one retained model and a bounded request queue.
Its modality-level worker preserves input/output lifetimes while the shared TCP
transport serves concurrent clients. It serializes images rather than batching
them; it does not put image requests through Qwen's token scheduler.

## Shared text execution

[`text/model.{h,c}`](text/model.h) owns a source-defined text residency: shared
parameter storage prepared in place at startup, cached prefill/decode commands,
model VM process, residual buffer and packed workspace. One preallocated arena
partitions private recurrent state among up to sixteen rows. With
`--pool_capacity=N`, attention pages grow from a shared physical budget rather
than reserving every row's logical context. Rows are data, not VM processes.
A single host owner multiplexes their stages through the shared timeline.

Packed target completion forks compact result downloads from cache-only MTP
catch-up. The catch-up stage consumes committed target state, not the downloaded
result records. The synchronous model call joins both branches before publishing
host positions and recycling payloads. Optional bounded continuation pre-issues
a second mixed plan: the device consumes queued prompt chunks and routes
accepted predictions into the next verifier without an intermediate host wait.
Admission and transport are observed between these bounded cohorts; continuous
device admission/output rings are not implemented.

The tools use IREE's standard device profiling flags. One shared device
profiling session begins before model preparation, so it includes startup
transfers and preparation as well as inference. Shutdown drains accepted work
before ending the session and propagates profiling failures. For aggregate
execution statistics, add `--print_device_statistics=true`. Per-dispatch
attribution can use `--device_profiling_mode=dispatch-events` with that flag,
or `--device_profiling_output=/path/to/profile` instead of aggregate printing.
Long captures consume bounded producer event rings with
`--device_profiling_flush_interval_ms=1000`; completing device work does not
itself drain captured records into the sink. Profiling retains command metadata
only when requested. Instrumented device timings explain kernel costs;
throughput comparisons run with profiling off.

The packed server generates its JIT catalog automatically: token classes
32/64/128/256/512 up to `--prefill_capacity`, crossed with span classes
1/2/4/8/16 up to `--rows`. Each axis includes its exact terminal capacity,
including non-power-of-two configurations. Repeated `--epoch=tokens:spans`
options replace this catalog for controlled experiments. Startup specializes
and caches these shapes from the same source and live device profile. Cached
commands share the same weights, retained rows, residual storage, maximum-sized
workspace, and VM process. Each occupies one immutable command-table slot;
selecting another shape allocates no device backing and copies no retained
state. The shared [packer](scheduling/packing.h) evaluates ready spans against
each shape and chooses the most input tokens, then the most participating spans,
then smaller capacities. This occupancy policy is intentionally distinct from
measured cost-based selection.
Supplying one shape gives a fixed-shape control; `--chunk_size` independently
limits each row's prompt contribution. Readiness, cache reservation, and the
default shape catalog remain text-runner policy; the packer has no model identity or
fixed row limit. Epoch records report the selected shape index and both
capacities. The model-local `epoch_check` also accepts repeated `--epoch`
options to cycle commands while comparing retained continuations with isolated
execution in the same residency.

`text:generate` is the CLI caller, not an HTTP service. It round-robins active input
chunks and decode steps, supports retained follow-up turns, and can reset/reuse
the same rows for repeated runs without reloading weights. This is stage-level
interleaving, not batched model math. Greedy selection and device position/history
updates are compiled stages. The host stops on EOS or its requested token limit.

Model math arrives as portable text under `--model`, not native artifacts.
The directory contains `sources.txt` (one relative source path per line),
`control.loom` (the shared VM program), command roots, and kernel libraries.
Editing these files changes the next process's model without rebuilding the
server. The binary embeds the JIT; it never invokes `loom-link` or
`loom-compile`, loads a prepared artifact directory, or falls back to stale code.

`runtime/jit.c` indexes the catalog once and retains the immutable compiler, pipeline,
and live HAL target profile across stage specializations. Its standard loomc
task pool supplies up to eight physical-core workers, each with private reusable
scratch. Stage calls are synchronous; their independent native requests compile
and load concurrently, and the stage queue drains even after a task fails.
`loomc_cmd_program_product_build` produces portable command bytes and
independently owned kernel source requests. Their producer-supplied binding
ordinals map emitted kernels to command entries; filenames and JSON do not
participate. Each request is lowered in memory and emitted from the same module,
preserving concrete target facts through native emission. Loaded executables and
recorded commands have independent ownership. VM control is also compiled at
startup and transferred directly into the VM's trusted in-process loading path.
A compile error terminates preparation with its source diagnostics.

The model's `config.loom` supplies fixed specialization defaults; its source
bootstrap declares run-dependent overrides. Command products supply parameter placement,
launch counts, buffer requirements, and entry mapping.
All target stages must place the shared weights identically; model preparation
checks this before allocating the one weight slab. MTP references existing
target weight views and allocates only its additional parameter groups.
`jit_stage` records expose the root, compiled request count, entry count, pool
worker count, and cold compilation/load duration. Warm execution reuses commands
and storage without compiler tasks.

The [model source guide](models/qwen/README.md) describes the math and
differential checks. The current source and host storage envelope is 512 input
tokens and sixteen resident rows. JIT removes offline preparation as a
prerequisite; larger envelopes still require changing the authored bounds and backing
together, then qualifying the resulting kernels. Shapes are prepared at startup,
not inserted into the fixed command table during a running session.
These are explicit properties of this adapter, not restrictions of the JIT.

The [text package contract](text/README.md) specifies bootstrap results, dynamic
bindings, semantic plans, feedback and chat policy for new models. Neither the
server nor CLI depends on a built-in model package; both require `--model`.
Qwen's isolated commands give the following concrete seven-binding example:

| Slot | Buffer contract |
| --- | --- |
| 0 | Shared residual rows: 512 × 5120 f32 values. |
| 1 | Three i32 values: prefill count/base/EOS, then decode position/count/EOS. |
| 2 | GDN convolution and recurrent state: 156,893,184 bytes. |
| 3 | Sixteen attention layers' KV state: context capacity × 65,536 bytes. |
| 4 | 512 i32 token IDs; after prefill, current token at 0 and a 128-token history ring from 1. |
| 5 | Eight i32 progress values; generated count at 0 and EOS flag at 7. |
| 6 | Shared scratch sized/aligned for the larger requirement of both stages. |

Slots 1 through 5 are private arena spans; 0 and 6 are shared. These isolated
roots use contiguous per-row KV; pooled calls route through packed page maps.
With dense addressing at a 16K context, two retained rows use
2.292 GiB, alongside the 18.504 GiB parameter slab and shared scratch. The host
does not infer these model semantics from kernel reflection.

The selected token returned by a stage has not yet entered the recurrent/KV
state. Decode consumes it; another prefill can instead discard it. A retained
follow-up consumes the last delivered token before appending new role markers
and input. This distinction is essential at EOS and tool-result boundaries.
The CLI retains output tokens on the host while the fixed device ring wraps.

```sh
build_tools/bin/iree-bazel-run --config=asan //experimental/loom_serve/text:generate -- \
  --model=experimental/loom_serve/models/qwen \
  --weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/tokenizer.json \
  --prompt='The secret word is MAPLE. Remember it and reply only READY.' \
  --prompt='The secret word is COBALT. Remember it and reply only READY.' \
  --followup='What is the secret word? Answer with only that word.' \
  --chunk_size=7 --iterations=2 --max_tokens=16
```

The example exercises distinct retained rows, short/padded chunks, follow-ups
and reset/reuse. Each iteration should produce READY/READY then MAPLE/COBALT.
Omitting `--chunk_size` uses the compiled capacity and avoids artificial tiny
chunks. `--prompt_file` accepts an already rendered chat transcript for replay
comparisons. `--rows` repeats the supplied prompt set across a fixed row count.

JSON metric lines on stderr report consumed input, selected output (including
EOS), prefill/decode step counts and completed-stage durations. Model TTFT runs
from round admission through the final input chunk's first prediction; it
includes other rows' intervening work. Round-robin duration covers the whole
set. Cold model creation and final text printing are outside those durations.
For a warm decode rate, divide decode steps by decode seconds; aggregate rate
uses the whole set's elapsed time, not the sum of per-row rates.

Model creation leaves parameter payloads unloaded; explicit activation can
warm before measurement, or the first inference includes loading. Elastic KV
backing grows with actual row/page use. The current host stage API
waits for readback and is single-owner; transport can run independently. These
checks do not establish zero allocations inside VM/HAL submission or batched
throughput. Controlled performance uses an optimized binary and benchmark lock,
not ASAN correctness timings.

## Local TCP transport

`http_server.{h,c}` runs raw `iree/net` TCP carriers on a standard proactor
thread, independently of blocking model stages. Its application owner claims
fully framed requests, sends copied response bytes with bounded send credit,
and finishes or aborts each connection. The transport has no model or chat
semantics. A connection carries one HTTP/1.1 request and a close-delimited
response; retained sessions belong above this connection lifetime.

The listener binds loopback only. Request framing bounds headers and body
storage, rejects ambiguous framing and pipelining, and retains request views
until application release. `--connections` bounds simultaneous peers, defaulting
to 64; excess connections are diagnosed and closed. `--request_body_bytes`
defaults to 8 MiB per peer and is independent of logical token context. Body
storage grows on demand within this bound. Send completion returns credit, and
deactivation joins outstanding operations before a slot is reused. Shutdown
joins both the accept target and its cancellation receipt before stopping the
poll owner. An unrecoverable proactor failure aborts the experimental process
instead of reclaiming storage whose I/O retirement cannot be established.

The executable blocks handled signals before creating any threads. SIGINT and
SIGTERM request application shutdown, allowing the model owner to finish its
current stage before releasing the transport and model. Client errors are
local to that peer; the model scheduler chooses its safe cancellation boundary.

```sh
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/loom_serve/http:request_test \
  //experimental/loom_serve/http:server_test
```

The loopback checks use real sockets/carriers and cover fragmented requests,
ordered streaming bytes, half-close, repeated slot reuse, peer reset and
shutdown with receive/accept work pending. They establish transport ownership,
not a working chat endpoint or a pi session.

## Retained chat service

`text_server` serves the shared model through the TCP transport. The main
application thread owns model state and packs ready prompt chunks and pending
decode tokens into one model epoch. Every admitted row receives its minimum
span before remaining capacity is filled from prompt input: one token for known
input or four for an indivisible MTP verifier. Rotating priority bounds
starvation when token/span capacity cannot fit every ready row and distributes
large prompt chunks. Network progress runs independently. Each row has one
pending copied SSE packet; exhausted carrier credit pauses that row before its
next epoch. The packed path shares matrix work and weight residency, not just
host submission. F16 cache math is the same in dense and pooled layouts.

`--pool_capacity` specifies a shared token-position budget in multiples of 64;
`--context_capacity` specifies the logical ceiling of each row. Admission reserves
enough page-rounded capacity to complete the request, including speculative
writes. Pages are physically assigned only as a row grows. Idle cache can yield
to admission, but active completion guarantees are never overcommitted.
The [model guide](models/qwen/README.md#pooled-kv-and-reserved-admission)
describes the layout, memory accounting, and real-model correctness witness.

[`text/chat.{h,c}`](text/chat.h) owns bounded OpenAI request decoding, canonical
history retention and SSE serialization. The source program owns model identity,
role templates, tool grammar selection and completion text. The supported
endpoint is `POST /v1/chat/completions`, with `stream: true`, greedy generation
and optional function tools. The Qwen source publishes `qwen3.8-27b` and its
non-thinking policy.
Unknown generation options, nonzero temperature and strict constrained sampling
are rejected. Qwen emits XML parameters and explicitly selects a reusable codec
to recover JSON value types from tool schemas. Another source policy can parse
its own grammar and return standard function-call records without that codec.
Complete calls are streamed to the client. Tool execution and full
schema validation remain in the agent. Incomplete or malformed calls produce
an error, never an executable partial call. `GET /healthz` reports readiness.

`X-Loom-Session` is a local cache key of at most 64 ASCII letters, digits,
underscores, hyphens or periods. Matching canonical client history permits
suffix-only prefill. The GPU retains the original generated tokens, including
their original XML spelling; canonical history is comparison data, not replayed
replacement state. A changed history resets and replays explicitly. Idle rows
form an LRU cache and may be evicted by another session. Untagged requests always
replay. Overlapping active or queued requests for one named session return 409.
Requests waiting for rows or completion credit enter a bounded FIFO without
owning device state. `--pending_requests` defaults to 32; exceeding that queue
returns 503. A request too large even for an otherwise empty pool returns 400
with a capacity diagnostic. There is no durable server-side conversation store.

Disconnect cancels at a completed model stage and invalidates the checkpoint.
SIGINT/SIGTERM stop admission, finish the current stage, relinquish request
views, drain transport I/O and release model residency. Request diagnostics are
JSON events on stderr: `enqueue`, `admit`, `complete`, `cancel`, `cancel_queued`,
`evict`, `epoch`, and `heartbeat`. Admission reports retained/appended tokens,
cache hit versus replay, completion reservation, and queue time. Completion adds
per-request prefill/decode epoch counts and end-to-end timing. Each epoch reports
its actual rows, consumed positions, prompt/decode counts, selected outputs,
token shape, traversal count and completed model duration. Shared epoch time is
not attributed in full to every constituent row. An independent host observer
reports current activity, active/backpressured rows, queued requests, pool
capacity/reserved/resident positions, completed counters and interval rates
every second, including while model execution is waiting. Reserved and resident
pool counts overlap: one is a completion guarantee, the other is physical use.
`--heartbeat_ms=0` disables periodic reports. All durations include the host
build's instrumentation and are not automatically performance data.

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //experimental/loom_serve/text:server -- \
  --model=experimental/loom_serve/models/qwen \
  --weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/tokenizer.json \
  --mtp --mtp_depth=3 --port=8080 --max_tokens=384
```

The packed server defaults to sixteen resident slots and a shared pool of
65,536 positions: 4 GiB of target KV plus 256 MiB of draft KV with MTP. This
backing is fixed at startup; individual sessions acquire pages as they grow.
Recurrent state adds 149.625 MiB per slot; `--rows` reduces that separate cost.
`--context_capacity` remains the per-row logical limit (16,384 by default).
For a shared 256K-position budget and a 256K per-session ceiling, set
`--pool_capacity=262144 --context_capacity=262144`. Distribution among sessions
then follows actual admitted request credit, including MTP provisional writes
and page rounding, not a partition into sixteen fixed context slabs.

`--scheduler=isolated` executes the same ready-span plan using ordinary per-row
prefill/decode stages; `--scheduler=matched` uses prefill math for length-one
decode inputs as well.
These controls and the CLI tools retain dense KV by default; explicit
`--pool_capacity=0` also selects dense addressing in the packed server.
Preparing the same explicit epochs and KV layout in all compared modes holds
planner limits and storage fixed. `--chunk_size` caps each row's contribution,
not the whole epoch. Each cached shape still computes padding; the epoch log
exposes useful work separately from its capacity.

`--packing=separate` is a same-kernel scheduling ablation: each epoch batches
only prompts or only decode inputs, selected by the first ready row in the
rotating priority order. A one-token prompt tail remains prompt work. The
default `--packing=mixed` admits both classes together. This choice is
independent of `--scheduler` and the per-row `--chunk_size` cap; ready and epoch
events record it. Comparing separate against mixed with `--scheduler=packed`
isolates cohort mixing, while packed against isolated measures shared versus
per-row traversals. Neither option alone enables speculative decoding.

`--mtp --mtp_depth=3` enables three-token MTP proposals under
packed scheduling. At least one configured epoch must admit the anchor plus
three proposals; an infeasible shape table is rejected before listening. Narrow
shapes can coexist with that wider shape. Only admitted verifier rows are
drafted; known prompt chunks fill the remaining shape capacity. The shared
target command verifies pending
anchors and candidates, commits accepted recurrent transitions, and catches MTP
up to the accepted target hidden state. The host streams every accepted output,
including a rejection replacement or EOS; only the final output stays pending.
Request output credit truncates acceptance before state publication. A row with
one remaining output credit or fewer than four context slots uses ordinary
decode. Cancellation is still observed at the completed-epoch boundary.

Depth zero with an MTP bundle keeps its cache/carry warm without proposing;
omitting the bundle is the target-only control. These are distinct costs.
Epoch and heartbeat records expose proposed tokens and accepted draft inputs
(excluding the pending anchor). `model_ms` includes drafting, verification,
accepted-state replay and catch-up. Proposals feed each other and the verifier
on device; only the final output/progress records are downloaded. Device
dispatch profiling attributes inner work without inserting host waits for
per-stage wall timers. Bundle capacities must match every
loaded target shape. The full canonical vocabulary is used for both proposal
and target selection; all code and weights remain shared across sessions.

For a bounded real HTTP check and initial end-to-end measurement:

```sh
build_tools/bin/iree-bazel-run //experimental/loom_serve/models/qwen:benchmark_service -- \
  --url=http://127.0.0.1:8080 --clients=4 --long-lines=128 --max-tokens=64
```

Alternating short/long prompts produce overlapping decode and prefill demand.
Each client then checks its distinct remembered codeword and requires a retained
prefix hit. JSON output includes response text, usage, first-text latency and
whole-cohort throughput. This is a synthetic HTTP lifecycle workload, not a
coding-agent score. Compare optimized, non-sanitized server runs under the
benchmark lease in interleaved mode order; preserve server epochs and client
results together. Real pi tool continuations remain the product check below.

`--workload=/path/to/workload.json` replaces the counting/codeword prompts with
a frozen text-turn corpus. The object contains a `name` and `sessions`; each
session has a `system` string and a nonempty `turns` array of
`{"content": "user text", "max_tokens": 192}` objects. The first `--clients`
sessions run concurrently, each immediately issuing its next turn with its
actual previous reply in the history. Follow-ups must hit the retained prefix.
This supports code/file-result review workloads without an agent harness in
the timed client. It measures a closed-loop text replay, not tool execution or
hidden reasoning. The summary records the exact corpus SHA256, and each reply
records the submitted history hash. Preserve full replies and usage to expose
work differences when comparing engines or kernel math.

The checked-in [source-review corpus](models/qwen/testdata/source_review.json) freezes eight
two-turn reviews of public compiler tests. Its [provenance](models/qwen/testdata/README.md)
and [measurement recipe](docs/PERFORMANCE.md#retained-review-benchmark) make the
full-model performance witness reproducible without private recordings.

### Fixed-trajectory model replay

`qwen_replay` isolates model/scheduler costs from generated-text divergence.
Its JSON fixture contains `sessions`, each with a `turns` array. Each turn has
the full HTTP `request` object, recorded `response` text, and
`"finish_reason": "length"`. The production chat renderer validates retained
history; the final selected token stays pending and leads the next prompt
append. Recorded text must re-encode to the request's output count. EOS traces
require original token IDs and are rejected by this text-only fixture loader.

Repeated `--epoch` options load shared-state variants; repeated `--window`
options list the zero-based epoch indexes available to the production planner
in each replay. For example, with epochs 128/32/256/512, windows `0`, `0,1`,
`0`, `0,2,3`, `0`, `0,1,2,3`, `0` isolate narrow tails and wider prompts with
interleaved fixed controls. Every window resets rows but reuses weights, commands
and scratch. Sessions issue their next recorded turn immediately on completion.
Selected predictions are logged at fixed logical positions but never feed back
into the input or terminate a trace. Differences from the first window remain
visible. This is teacher-forced completed-work timing, not a generated-quality
score or HTTP latency measurement. `trajectory` events preserve exact input IDs;
`epoch` and `window` events report costs, counts and prediction differences.

### Run telemetry

`observe.py` launches the runner and combines its stdout/stderr with independent
Linux system samples. It uses only the Python standard library; no sensor
subprocesses or OS polling enter the inference process. Build the exact server
target first, then place the observer inside the benchmark lease when measuring:

```sh
python -B -m experimental.loom_serve.tools.observe --log=/private/runs/run.jsonl -- \
  bazel-bin/experimental/loom_serve/text/server \
  --model=experimental/loom_serve/models/qwen \
  --epoch=128:8 \
  --weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/tokenizer.json --rows=8
```

The log is created exclusively; an existing run is never overwritten. The
default system interval is one second (`--interval`), independent of the native
service heartbeat interval. `--quiet` suppresses console mirroring, not logging.
SIGINT/SIGTERM reach the owned server and wait for its normal retirement. A log
or observer failure stops the server and reports failure while continuing to
drain its output through shutdown. There is no silent lossy queue or automatic
log rotation. The caller owns log retention and available disk space. Logs
contain command paths and application diagnostics and are private run data.

Each line is a version-1 envelope containing `sequence`, `unix_time_ns`,
monotonic `elapsed_ns`, `source`, and `data`. Sequence and timestamps describe
observer receipt, not device event time or ordering between stdout and stderr.
Native JSON events remain unchanged inside `data`; plain output becomes a
`diagnostic` event. `run`/`process_start`/`run_end` delimit child lifetime.

The `system_inventory` event gives stable channel indices, labels, source paths,
units and runtime-power guards. Each `system` event supplies the corresponding
`values` array, `unavailable` reasons by index, and errors for missing host
facilities. Readings include available hwmon temperatures, fan speeds, power
and limits; CPU governors/EPP/frequencies; platform policy; GPU occupancy,
memory and clocks; GPU/NPU runtime power state; host and server CPU/memory;
CPU/memory/I/O pressure; and log-filesystem free space. Unavailable readings
are null, never zero or a previous value. CPU percentages use host-wide busy
time for the machine and top-style 100% per busy CPU for the server. First
samples have no CPU delta. Sensor power is device/package power, not wall power.

Discovery is once per run and reflects the kernel's exported interfaces and
current permissions. Device values are skipped when their runtime-power guard
reports suspension or transition, avoiding deliberate wake polling; state can
still change between the guard and attribute reads. Nothing changes power
policy, requests privileged access, or substitutes an estimate for a sensor.

`dashboard.py` reads only this log. It can follow an active run, inspect a
completed one, or stop at a recorded elapsed time without running a model:

```sh
python -B -m experimental.loom_serve.tools.dashboard /private/runs/run.jsonl
python -B -m experimental.loom_serve.tools.dashboard /private/runs/run.jsonl \
  --snapshot --at=16 --width=160
```

Wide terminals place hardware readings beside scheduler progress; narrow ones
scroll vertically. `q` exits, arrows or `j`/`k` scroll, PageUp/PageDown move a
page, Home returns to the top, and `a` exposes all channels with their exact
source paths. The `--all-sensors` option selects that view initially. The
display separates prompt inputs, decode inputs and selected outputs including
EOS; planned token-slot occupancy is not GPU utilization. Heartbeat, last-epoch
and system-sample ages remain independent so a stalled device cannot appear
healthy just because the host sampler is running. A recorded prefix without
`run_end` is not claimed to be a successful or still-running server.

The viewer retains latest state and short histories, not the entire run.
Partial trailing records wait for completion; malformed complete records fail
with an offset. Snapshot mode reports an omitted partial record. Application
text is escaped before terminal rendering. Neither the viewer nor replay can
change the server or hardware state.

### Real pi continuation check

An isolated pi custom-provider configuration uses this `models.json`:

```json
{
  "providers": {
    "loom": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "local",
      "headers": {"X-Loom-Session": "$LOOM_SESSION"},
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "supportsStore": false,
        "supportsStrictMode": false,
        "maxTokensField": "max_tokens"
      },
      "models": [{
        "id": "qwen3.8-27b",
        "reasoning": false,
        "input": ["text"],
        "contextWindow": 16384,
        "maxTokens": 384,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
      }]
    }
  }
}
```

Set `PI_CODING_AGENT_DIR` to that isolated directory. For the bounded witness,
its `settings.json` disables compaction and retries:

```json
{"compaction":{"enabled":false},"retry":{"enabled":false,"provider":{"maxRetries":0}}}
```

Run two pi processes with distinct `LOOM_SESSION` values and session files,
using `--provider loom --model qwen3.8-27b --thinking off --tools read` and
`--no-extensions --no-skills --no-prompt-templates --no-themes
--no-context-files --no-approve --offline`. A short fixed `--system-prompt`
keeps this a focused protocol check. Either RPC mode or repeated print-mode
invocations with the same `--session` file preserve client history.

For each client, ask it to remember a distinct codeword, use `read` with
`offset: 1` and `limit: 8` on
`loom/src/loom/verify/test/function_entry.loom-test`, and name the first
function. It must execute the actual file tool and answer `count_down`. Then
ask for the codeword without tools. The second model invocation (tool result)
and subsequent user turn must report `cache: "hit"`, nonzero retained tokens,
and only appended suffix tokens. Text alone does not prove retention.

The focused host checks are:

```sh
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/loom_serve/models/qwen:chat_test \
  //experimental/loom_serve/http:request_test \
  //experimental/loom_serve/http:server_test
```

### Kernel/scheduler handoff

Kernel changes stay behind the existing stage buffers and parameter placement.
The [model source checks](models/qwen/README.md) cover numerical behavior with
VM oracles. The retained CLI witness, real pi tool continuation, and cancellation
checks cover the state those kernels mutate across stages. The server prepares
artifacts once; restart it after recompiling stages. There is no kernel hot-swap.

For comparisons, preserve both artifact directories and the exact optimized
host binary. Keep GGUF/tokenizer identity, context capacity, active prefill
length, prompt/tool transcript, output count and concurrency fixed. Separate
cold load, prefill, completed decode and whole-request latency; submit-only
numbers do not describe agent latency. Interleave baseline/candidate runs
(ABABA) under the benchmark lock, record hardware load/power/temperature, and
compare generated output as well as timing. A thermally constrained or busy
host permits qualified differentials, not a clean absolute throughput claim.

### Recording agent workloads

Give each pi client a distinct `--session /private/recordings/agent-N.jsonl`
and preserve that file across tool and user turns. The settings above disable
compaction and retries, so each recording contains the actual sequential model
invocations. Raw recordings contain prompts, file contents, and tool arguments;
they belong in private local storage, not source control. Record clients on the
same host or with synchronized wall clocks.

After those clients finish, export their workload counts:

```sh
build_tools/bin/iree-bazel-run //experimental/loom_serve/tools:agent_trace -- \
  --output=/private/recordings/workload.json \
  /private/recordings/agent-0.jsonl /private/recordings/agent-1.jsonl
```

The exporter reads linear pi v3 `openai-completions` histories from one model.
It strips message content, tools, paths, and working directories. The result
retains source hashes, model identity, common clock origin, first-arrival
offsets, uncached input counts (including cache writes), retained prefix counts,
selected output counts, observed response durations, and client delays between
requests. Native session files alone cannot account for compaction's individual
model calls; those require the call recorder below. The exporter rejects
branches, uninstrumented compaction, failed/incomplete turns, duplicate
recordings, and overlapping/backwards request clocks. Output files are created
exclusively to protect earlier evidence. The count format is
`loom-agent-trace-v2`; regenerate older exports from their original recordings.

The inner assistant timestamp is client invocation start; the outer session
entry timestamp is completed-message persistence. Their difference includes
transport and client overhead, not just GPU work. The next invocation depends
on the preceding completion plus its recorded client delay. This dependency is
what a closed-loop scheduler replay must preserve when changing model speed.
Token lengths and recorded cache outcomes are fixed evidence; these files do
not expose server backpressure, cancellation timing, or cache-eviction policy.

For full lifecycles, `pi_record.mjs` wraps the real pi SDK session's provider
stream function. It observes both normal agent calls and summarization calls
without replacing pi's compactor or consuming its event stream. A split-turn
compaction can make two sequential model calls; each retains its own input,
output, cache counts and timing. The saved native compaction entry aggregates
usage and cannot recover those boundaries on its own.

```js
import { VERSION } from "@earendil-works/pi-coding-agent";
import { recordPiSession } from "./experimental/loom_serve/tools/pi_record.mjs";

// session is a configured, idle SDK AgentSession with both retry layers off.
const recorder = recordPiSession(session, "/private/agent.calls.jsonl", VERSION);
await session.prompt(task);
await session.waitForIdle(); // Includes automatic compaction after agent_end.
// Additional user turns use this same session and recorder.
recorder.finish();
```

On failure, abort/wait for the session and call `recorder.close()`; an unfinished
log remains explicit evidence and cannot be exported as a successful lifecycle.
Keep the native pi session file too: the call log deliberately contains no
prompts, tool arguments/results or summaries. Export call logs with the same
`agent_trace` command, without also importing their native sessions (which would
double-count the workload). Call timestamps use a monotonic process clock
anchored to Unix time; completion is provider result availability, not GPU time.

Client settings are per-session metadata, not part of shared model identity.
For example, eight clients can advertise 8192 tokens and eight 16384 while using
the same provider/model. Configure each pi session's model and compaction
settings independently: pi's default reserve of 16384 and kept-history budget
of 20000 are unsuitable for these small windows. A starting experiment can use
2048 reserve tokens, 2048 kept tokens and a 1024-token output ceiling. These
settings need real task-quality validation. Large incoming tool results can
still overflow a window; pi's threshold is not an exact input-admission cap.
The service's compiled context capacity must cover the largest window. Smaller
client budgets do not reduce its current uniform physical row reservations.

The recorder has a CPU integration check using real pi 0.82.1 and a scripted
provider (no network or device). It exercises automatic/manual split-turn
compaction, resumed context, mixed windows and failures:

```sh
node experimental/loom_serve/tools/pi_record_test.mjs \
  /absolute/path/to/node_modules/@earendil-works/pi-coding-agent \
  /private/recordings/new-cpu-witness
```

That check is protocol/lifecycle evidence, not model quality or performance.

### CPU packing replay

`simulate_packing` replays that trace with one outstanding request per client.
First arrivals keep their recorded offsets. Each subsequent request becomes
ready after **simulated** completion plus the recorded client delay; the old
server's response time is not baked into arrival scheduling.

Individual lifecycles can also be collected into one trace and overlaid with
an explicit synthetic composition. For example:

```json
{
  "format": "loom-agent-composition-v1",
  "instances": [
    {"source_session": 0, "start_us": 0},
    {"source_session": 1, "start_us": 2000000, "request_count": 2},
    {"source_session": 0, "start_us": 5000000,
     "pauses": [{"before_request": 1, "duration_us": 3000000}]}
  ]
}
```

Pass this file as `--composition=/private/recordings/composition.json`.
`source_session` is a zero-based index in the input trace; output session
indices refer to instances in this list. Each instance starts from its first
recorded request at `start_us`, ignoring its original wall-clock offset.
`request_count` stops after that many completed model requests (default: the
whole lifecycle). A pause adds to the recorded client gap before the specified
zero-based continuation, after its simulated predecessor completes. These are
request-boundary controls, not in-flight cancellation or absolute-time suspension.

The result embeds the normalized composition beside the original trace hash,
so copies remain identifiable as synthetic instances, not independent measured
clients. Counts, compaction calls, client budgets and cache outcomes remain
those of the source lifecycle. Mixing 8K/16K clients therefore uses recordings
made at those budgets; changing a label would not recreate different compaction
decisions. All instance state is assumed resident, including while paused.
Real concurrent recordings are still needed to check the model against actual
contention, eviction and changes in agent behavior.

```sh
build_tools/bin/iree-bazel-run //experimental/loom_serve/tools:simulate_packing -- \
  /private/recordings/workload.json \
  --capacities=32,64,128 --span-capacity=8 --epoch-us=100000 \
  --output=/private/recordings/packing.json \
  --epochs=/private/recordings/epochs.jsonl
```

The required `--epoch-us` is a hypothetical uniform epoch duration, **not a
measured cost or throughput prediction**. Varying it tests the interaction
between service speed and recorded client delays. All shapes have that same
duration in this experiment; shape occupancy is not hardware utilization.
Measured shape/context costs require separate controlled device evidence.

Both policies use the same admission rule, bounded by token and span capacity.
Each admitted decode reserves one input token; prefill divides the remaining
budget. `single-pass` uses the count arithmetic in `qwen_workload_plan`;
`fair-fill` redistributes short-span leftovers. Both choose the smallest listed
shape fitting their actual plan. Final prefill selects the first prediction,
so a request with N recorded outputs requires N-1 subsequent causal decodes.
The defaults use round-robin admission and immediate dispatch. There is no
speculative lookahead.

`--max-hold-us` experiments with bounded collection: dispatch once the maximum
token budget or span slots are full, or the oldest ready row reaches its hold
target. `--admission=longest` and `--admission=shortest` rank ready spans by
length, with oldest-first priority for rows due before another epoch can
complete. Prefill spans remain splittable, so size ranking matters primarily
when more rows are ready than fit in the span table. These are experimental
policies, not a claim that either ordering wins.

The hold is a ready-to-dispatch target, not a hard real-time guarantee. A request
arriving during a nonpreemptible epoch can already be late when the scheduler
regains control. `collection_us` records deliberate waiting;
`late_service_spans` and `max_hold_overrun_us` expose queueing/dispatch violations.
At the default zero hold, any runnable-to-service delay is an overrun of that
zero-wait target. Token slot occupancy and these progress counters must be
considered together; a fuller batch obtained by idling need not be better.

The optional ledger exposes every epoch's readiness, selected shape, row spans,
positions and output selections. Its three gap counters partition unused
**maximum token budget**: `packing_gap_tokens` is admitted ready work left
unpacked; `span_gap_tokens` is work excluded by span admission;
`causal_gap_tokens` is space with no ready input to fill it. These differ from
`shape_padding_tokens`, which counts only empty slots in the emitted shape.
The summary includes shape occupancy, epoch counts, per-request completion,
and longest runnable-to-service wait. Both policies complete identical recorded
token counts, but their evolving ready frontiers can differ.

All request input is assumed ready at arrival; pi histories do not record
incremental tokenizer publication. All session state is assumed resident and
recorded cache outcomes stay fixed.
The simulator does not establish resident-memory capacity, eviction behavior,
new generated text, MTP acceptance, cancellation, or output-credit behavior.
It is an editable packing experiment, not a replacement for endpoint/device
qualification. Increasing client count requires more recordings; replay does
not silently duplicate sessions or call synthetic replicas real agents.

```sh
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/loom_serve/tools:agent_trace_test \
  //experimental/loom_serve/tools:simulate_packing_test
```
