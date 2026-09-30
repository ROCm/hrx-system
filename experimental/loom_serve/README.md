# Loom model runner experiment

This branch-local runner connects compiled model-control VM code to prepared
HAL commands. Its private interfaces are intended to change with real models.
It is not a general HAL VM module or a serving framework.

`command.c` combines a parsed portable command program with loaded executable
reflection. Recording retains fixed buffers and code. Rebindable slots include
model state and explicit scratch; the materializer allocates no device backing.
There is no semantic-resource extension to the command artifact format.

`module.c` publishes prepared stages as ordinary fixed-signature VM imports:
`(hal.buffer, ... slots ...) -> i64`. Model code selects the stage and buffers;
the native call submits and returns without waiting. HAL captures bindings and
retains their buffers independently of the VM invocation. One process can serve
many rows; its preallocated native binding scratch is reused between calls.

`execution.c` is the shared execution capability used by both the model and
host I/O. Each accepted submission waits on the preceding submission and signals
the next timeline value, including transfers across different exact queues.
This deliberately serializes shared scratch use. The returned integer is scoped
to that execution object, not a global completion handle. Queue rejection does
not advance the accepted frontier. The host drains accepted work before
recycling borrowed host payloads and observes asynchronous failures explicitly.

The GPU integration test compiles Qwen generation-state kernels, command programs,
and a VM entry from source. Two retained rows share code, commands, and one VM
process while one row pauses and resumes. Checked outputs cover token history,
position, padding, and EOS. A rejected native binding must leave the execution
domain usable. This is an ownership/control witness, **not a full Qwen model
run or a performance result**. The test artifacts currently target gfx1151.

From the worktree root:

```sh
build_tools/bin/iree-bazel-test --config=asan //experimental/loom_serve:control_test
```

Weights, KV pools, model forward stages, scheduling policy, and transport are
separate from this coarse execution boundary. None is inferred from buffer
contents or command names by the native submission code.

## Shared Qwen execution

`qwen_model.{h,c}` owns a concrete Qwen3.8-27B UD-Q5_K_XL residency: one GGUF
parameter slab, prepared prefill/decode commands, model VM process, residual
buffer and packed workspace. One preallocated arena partitions private retained
state among up to eight rows. Rows are data, not VM processes. A single host
owner multiplexes their stages through the shared execution timeline.

`qwen` is the CLI caller, not an HTTP service. It round-robins active input
chunks and decode steps, supports retained follow-up turns, and can reset/reuse
the same rows for repeated runs without reloading weights. This is stage-level
interleaving, not batched model math. Greedy selection and device position/history
updates are compiled stages. The host stops on EOS or its requested token limit.

Model math is supplied as separately compiled artifacts, not built into this
binary. Each stage directory contains:

```text
manifest.json                 Version 2 loom-command-set manifest, one root.
config.json                   Compile-time model capacities from the stage driver.
commands/<artifact>           Portable command named by that root.
kernels/<request-stem>.hsaco   Compiled image for each manifest entry.
```

The manifest supplies entry names and the root-local entry mapping. It is not
an execution language: command order, arguments, parameter placement, and
transient requirements come from the portable command artifact. Both stages
must place every parameter identically; the loader compares their keys,
fixed-buffer indices, offsets, lengths, and alignments before sharing weights.
Allocation satisfies both stages' published slab size/alignment requirements.

The [model sources and compiler driver](models/qwen38/README.md) reproduce these
artifacts. Configuration stays with its compiled stage; the loader checks that
both stages declare the same context capacity. Prefill accepts active chunks
up to the compiled launch capacity (at most 512); the caller enforces remaining
context capacity. The seven rebindable slots are:

| Slot | Buffer contract |
| --- | --- |
| 0 | Shared residual rows: 512 × 5120 f32 values. |
| 1 | Three i32 values: prefill count/base/EOS, then decode position/count/EOS. |
| 2 | GDN convolution and recurrent state: 156,893,184 bytes. |
| 3 | Sixteen attention layers' KV state: context capacity × 65,536 bytes. |
| 4 | 512 i32 token IDs; after prefill, current token at 0 and a 128-token history ring from 1. |
| 5 | Eight i32 progress values; generated count at 0 and EOS flag at 7. |
| 6 | Shared scratch sized/aligned for the larger requirement of both stages. |

Slots 1 through 5 are private arena spans; 0 and 6 are shared. The current roots
use contiguous per-row KV, not paged KV. At a 16K context, two retained rows use
2.292 GiB, alongside the 18.504 GiB parameter slab and shared scratch. The host
does not infer these model semantics from kernel reflection.

The selected token returned by a stage has not yet entered the recurrent/KV
state. Decode consumes it; another prefill can instead discard it. A retained
follow-up consumes the last delivered token before appending new role markers
and input. This distinction is essential at EOS and tool-result boundaries.
The CLI retains output tokens on the host while the fixed device ring wraps.

```sh
build_tools/bin/iree-bazel-run --config=asan //experimental/loom_serve:qwen -- \
  --prefill=/path/to/compiled/prefill \
  --decode=/path/to/compiled/decode \
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

All device backing is allocated before generation. The current host stage API
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
until application release. Sixteen connection slots bound admission; excess
connections are diagnosed and closed. Send completion returns credit, and
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
  //experimental/loom_serve:http_request_test \
  //experimental/loom_serve:http_server_test
```

The loopback checks use real sockets/carriers and cover fragmented requests,
ordered streaming bytes, half-close, repeated slot reuse, peer reset and
shutdown with receive/accept work pending. They establish transport ownership,
not a working chat endpoint or a pi session.

## Retained chat service

`qwen_server` serves the shared model through the TCP transport. The main
application thread owns model state and round-robins one prefill chunk or decode
step per ready row. Network progress runs independently. Each row has one
pending copied SSE packet; exhausted carrier credit pauses that row before its
next stage. This is interleaved execution, not batched matrix math.

`qwen_chat.{h,c}` owns the text-only, non-thinking Qwen template and XML tool
translation. The supported endpoint is `POST /v1/chat/completions`, with model
`qwen3.8-27b`, `stream: true`, greedy generation and optional function tools.
Unknown generation options, nonzero temperature and strict constrained sampling
are rejected. The model emits XML parameters; tool schemas recover JSON value
types before a complete call is streamed to the client. Tool execution and full
schema validation remain in the agent. Incomplete or malformed calls produce
an error, never an executable partial call. `GET /healthz` reports readiness.

`X-Loom-Session` is a local cache key of at most 64 ASCII letters, digits,
underscores, hyphens or periods. Matching canonical client history permits
suffix-only prefill. The GPU retains the original generated tokens, including
their original XML spelling; canonical history is comparison data, not replayed
replacement state. A changed history resets and replays explicitly. Idle rows
form an LRU cache and may be evicted by another session. Untagged requests always
replay; overlapping requests for one named session return 409; fully occupied
rows return 503. There is no durable server-side conversation store.

Disconnect cancels at a completed model stage and invalidates the checkpoint.
SIGINT/SIGTERM stop admission, finish the current stage, relinquish request
views, drain transport I/O and release model residency. Request diagnostics are
JSON events on stderr: `admit`, `complete`, `cancel`, and `evict`. Admission
reports retained/appended tokens and cache hit versus replay. Completion adds
prefill/decode counts and completed-stage durations. These durations include
the host build's instrumentation and are not automatically performance data.

```sh
build_tools/bin/iree-bazel-run --config=asan \
  //experimental/loom_serve:qwen_server -- \
  --prefill=/path/to/compiled/prefill \
  --decode=/path/to/compiled/decode \
  --weights=/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --tokenizer=/path/to/tokenizer.json \
  --rows=4 --port=8080 --max_tokens=384
```

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
  //experimental/loom_serve:qwen_chat_test \
  //experimental/loom_serve:http_request_test \
  //experimental/loom_serve:http_server_test
```

### Kernel/scheduler handoff

Kernel changes stay behind the existing stage buffers and parameter placement.
The [model source checks](models/qwen38/README.md) cover numerical behavior with
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
build_tools/bin/iree-bazel-run //experimental/loom_serve:agent_trace -- \
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
import { recordPiSession } from "./experimental/loom_serve/pi_record.mjs";

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
node experimental/loom_serve/pi_record_test.mjs \
  /absolute/path/to/node_modules/@earendil-works/pi-coding-agent \
  /private/recordings/new-cpu-witness
```

That check is protocol/lifecycle evidence, not model quality or performance.

### CPU packing replay

`simulate_packing` replays that trace with one outstanding request per client.
First arrivals keep their recorded offsets. Each subsequent request becomes
ready after **simulated** completion plus the recorded client delay; the old
server's response time is not baked into arrival scheduling.

```sh
build_tools/bin/iree-bazel-run //experimental/loom_serve:simulate_packing -- \
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
  //experimental/loom_serve:agent_trace_test \
  //experimental/loom_serve:simulate_packing_test
```
