# Observe a model server

`loom-serve` exposes current state, cumulative counters and bounded event history
through separate HTTP views. Reading them neither runs inference nor activates
cold weights. Slow observers can miss events without blocking inference or
preventing model removal.

The [serving walkthrough](serve-models.md) provides an arithmetic model for
exercising these endpoints. [Operations](operate-model-server.md) explains the
registration, admission and residency states they describe.

## Watch the live dashboard

Start the server, then visit
`http://127.0.0.1:8080/dashboard/`:

```sh
loom-serve --device=task --pool_backing=fixed \
  --listen=127.0.0.1:8080 \
  --model=vectors/model.json
```

The dashboard is embedded in the executable: no frontend installation,
external assets, database, or second listener. Models appear as they register.
Token and completion rates use a five-second rolling window, with the available
observed duration during warm-up. Lifetime averages divide cumulative work by
elapsed source lifetime, including idle time and work before the page opened.
They survive reloads and reconnects; a source without a known start time leaves
its average unavailable. Serving mix shows recent completions by modality.

Request activity counts arrivals and finishes in roughly one-second buckets, so
short requests remain visible even when they finish between observations. Each
bucket closes on a real sample after at least one second; the received-request
card shows its actual duration. Finishes include all outcomes, while completion
throughput counts only successes. These are counts of work, not concurrency.
Active, queued, and held requests remain separate current samples. Per-model
sparklines, logical KV occupancy, cache counts, and expandable source counters
come from the running services. Logical KV positions are not physical RAM usage;
elastic text-state bytes exclude fixed buffers, weights, and workspace. A modality
without a particular observation leaves it unavailable rather than inventing a
value.

Model badges link to their execution device and storage pool. The Devices section groups current
request activity by that placement; these counts are not hardware utilization or
native queue depth. Device and pool cards remain when models are unregistered.

Each named pool appears once, even when multiple devices consume it. Its card
shows committed serving allocations, observed peaks, reusable backing, and
commitment histories, with links to the permitted devices. Native storage and
elastic backing are distinct backing owners within the pool; their commitment
adds. Workspace is shown as included, never added again. Elastic
virtual reservations are address space, not physical RAM. These views exclude
driver allocations and other processes, and do not invent a shared RAM capacity
or a weights/KV/workspace partition from overlapping observations. Expand a pool
to inspect its source counters.

Unchanged snapshots stay off the wire. A small heartbeat settles the rate window,
then backs off to once per minute while idle. New activity still appears at the
normal sampling cadence; it does not wait for the idle heartbeat.

Observed peaks remain visible after work finishes: rate peaks use full
five-second windows, activity peaks count observed buckets, and active/occupancy
peaks use received gauges.
Peaks belong to this browser page session, survive pause/reconnect, and reset
when the server or corresponding model is replaced. They describe observed work,
not a benchmark of the server's maximum capacity.

The browser retains a fixed history. **Scroll on** moves the charts using the
browser clock, even when the feed is quiet; a dotted tail marks the last received
value, not new samples. **Scroll off** keeps the last activity window visible
while counters and peaks continue updating. Neither mode adds network traffic.
The time-range button switches between 30 seconds and two minutes.
Pause or a hidden tab releases the live connection. Resuming reads current
state; rolling rates and activity buckets do not replay work that happened while
disconnected. Lifetime averages still include that work and elapsed time.

No dashboard flag is needed for same-origin access through `localhost`, a
numeric IP address, or an SSH tunnel. The browser-visible port may differ from
the server's listening port. Relative URLs preserve a proxy mount prefix.

A Unix-socket backend such as `unix:/run/loom/serve.sock` trusts its HTTP relay
to validate public hostnames and control access to the socket. The relay
preserves `Host` and `Origin`, forwards WebSocket upgrades, and replaces
`X-Forwarded-Proto` with `https` when terminating TLS (`http` is the default).
Use a protected socket directory or equivalent namespace isolation. TCP
connections never gain relay trust from forwarded headers.

For a direct DNS hostname or a deliberately separate browser UI, repeat
`--http_allowed_origin=https://console.example` for each additional exact
origin, including any nondefault port. Arbitrary DNS names are not inferred
safe from matching request headers. This browser-origin check is neither
authentication nor a CORS policy; native clients may omit `Origin`. Public
access control belongs to the deployment.

The `/v1/observe` WebSocket sends bounded current-state pages, with at most
one application-unacknowledged page per viewer. Slow clients skip intermediate
states without queuing a history or retaining model resources. Registration
incarnations distinguish a removed model from a replacement with the same
name. Observation pages are live, not an atomic whole-server snapshot.
Persistent collectors use `/metrics`; event history remains in `/logz`.

## Start with status and counters

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/status
curl --fail --silent --show-error http://127.0.0.1:8080/metrics
```

Status includes the current registrations plus a `server` object, even when
no models are loaded. That object reports actual worker count, admission limits,
bound addresses and the named device inventory. Worker count is provisioned
capacity, not merely the requested maximum.

Each model's `loom.status` contains phase, queued/active counts and the oldest
age in each ownership class. Active age includes earlier queue residence,
preparation, capacity waiting and output encoding. A disconnected peer can
remain counted until its accepted work retires. These are scheduler ownership
observations, not device busy time or completed-request latency percentiles.

`loom.parameters` reports independent `primary` and, when declared, `image`
component records. Each has a preparation `phase` and persistent `pinned`
policy. A component without prepared residency has no `active`, `backing` or
byte-accounting fields; absence is not a zero-byte measurement. Prepared
records include activation/deactivation counts and, for elastic backing,
reserved and committed parameter bytes. Image preparation failures and
compilation diagnostics stay in the image record rather than masking healthy
text service. Reading status never starts component preparation.

`/metrics` uses Prometheus text format with `model`, `kind` and registration
`incarnation` labels. Gauges report current phase and ownership counts.
Work counters include requests, completed, rejected,
cancelled, failed, input tokens, output tokens and batches.
`loom_serve_server_*_total` includes final counters from removed registrations.
Reusing a name creates a new incarnation rather than resetting the old series.

Counters are maintained independently of log retention. Service counters include
model-specific controls such as text checkpoints, but exclude router-owned raw
tokenization, health/observation calls and unknown-route rejections. Unavailable
counters are reported as unavailable, not zero. Disabling periodic heartbeats
with `--heartbeat_ms=0` does not disable counting.

Source-owned numeric extensions appear in `loom_serve_model_observation`
(gauges) and `loom_serve_model_observation_total` (counters). Their `topic`,
`schema`, `field`, and `unit` labels describe the source's integer values,
without implicit unit conversion. Text and image services publish these same
facts to the live dashboard. Extension counters belong to their model
incarnation and are not automatically added to server-lifetime totals.

## Read preparation timing

To see how long a model has spent preparing, query its elapsed time alongside
the compilation report:

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/status |
  jq '.data[] | select(.loom.incarnation) | {
    model: .id,
    phase: .loom.registration,
    preparation_seconds: (.loom.preparation_duration_ns / 1e9),
    compilation_stage_seconds:
      (if .loom.compilation then
         .loom.compilation.stage_duration_sum_ns / 1e9
       else null end)
  }'
```

`loom.preparation_duration_ns` counts elapsed time in preparation, including
model-card resolution and synchronous setup. It advances while preparation is
pending and freezes on completion, failure, or completed cancellation. A drained
registration can resolve its card and then return to `declared`; that idle gap
adds no time. Later preparation adds another interval. The field is zero before
any preparation starts. Parameter activation after preparation, inference, and
retirement have their own lifetimes and add no preparation time.

`compilation.stage_duration_sum_ns` sums each compiled stage's interval from
submission through completion. Those intervals include queueing and waits for
shared native kernels, and overlap when stages compile concurrently. For
example, two stages each pending for one second can contribute two seconds to
the sum during one second of elapsed preparation. The sum is neither CPU work
nor model startup latency; dividing it by elapsed time does not measure worker
utilization. Reports are cumulative and contain completed stages, so a pending
stage does not yet contribute its interval.

A lazy image component reports its independent elapsed time at
`loom.parameters.image.preparation_duration_ns`, next to its own `compilation`
report. Reading either field leaves a declared component cold. Retiring a
fixed-backing component and preparing it again adds a new interval; reactivating
an already prepared component's weights does not. During registration retirement,
the component's observations are unavailable and its timing field is omitted;
the registration retains its own elapsed time until removal. Native executable
variants use the same summed-stage field in their compilation reports; staging a
variant does not add to the model's preparation time.

## Distinguish parameter pressure from queued work

Each prepared component in `loom.parameters` reports accepted-consumer `pins` and an
`admission` snapshot: `waiting`, `count`, `duration_seconds`, and `age_seconds`.
The snapshot describes the component's single outstanding interest in shared
parameter admission. A wait can reflect physical backing pressure, an older
waiting model, or asynchronous reclamation that must retire first. A waiting
model keeps its turn until its request or cohort is fully admitted or cancelled:
parameter acquisition alone does not end a turn still waiting for mutable-state
backing. The interval can therefore include activation I/O while a text or audio
row waits for state. It is not one observation per request or execution latency.

The corresponding Prometheus series have a `component` label (`primary` or
`image`) in addition to the model identity:

| Metric suffix after `loom_serve_model_` | Meaning |
| --- | --- |
| `parameter_pins` | Accepted-consumer and explicit-retention pins protecting parameters. |
| `parameter_admission_waiting` | One when new work is waiting for parameter admission; otherwise zero. |
| `parameter_admission_waits_total` | Distinct wait episodes, not repeated scheduler polls. |
| `parameter_admission_wait_seconds_total` | Cumulative waiting, including cancelled waits and the currently open interval. |
| `parameter_admission_wait_age_seconds` | Age of the current wait, or zero when none is open. |

Accepted consumers can retain pins while new work for the same model waits;
their nested execution does not queue behind the work it must unblock. These
pins are separate from the component's persistent `pinned` eviction policy.
Releasing a consumer pin does not clear that policy, and cancelling a new-work
interest does not release another consumer's pin.

Parameter observations have the model's incarnation identity and disappear on
removal. Unlike completed-request and work totals, they have no server-lifetime
aggregate. A registration without parameter observations has no samples, not
synthetic zero values. Polling snapshots neither admits work nor activates
parameters. Current wait age and cumulative duration advance while a wait is
open even when no request has completed.

## Measure completed-request latency

`loom_serve_model_request_duration_seconds` is a cumulative histogram with the
same model identity labels plus `outcome`: `completed`, `rejected`, `cancelled`
or `failed`. Its `_bucket`, `_count` and `_sum` series count each terminal
request once. `loom_serve_server_request_duration_seconds` retains durations
from removed registrations, just like server work counters. Both are independent
of event-ring retention and heartbeat frequency.

For example, this PromQL expression estimates the recent p95 for successful
requests, keeping each model incarnation separate:

```promql
histogram_quantile(0.95,
  sum by (model, kind, incarnation, le) (
    rate(loom_serve_model_request_duration_seconds_bucket{outcome="completed"}[5m])
  )
)
```

Duration starts when a request reaches its model protocol and ends when that
owner retires it. It includes preparation, model admission waits, execution and
encoding. Successful retirement means the response was accepted by the HTTP
carrier, not received by the peer. Time held by a drained router before model
admission is outside this measurement. Cancellation durations include any work
that must retire after the peer leaves. Router-owned utilities and unknown-route
rejections are outside the model protocol, as with work counters.

Buckets span 1 millisecond through 1000 seconds plus an unbounded bucket. Bucket
percentiles are estimates; `_sum / _count` gives the mean without bucket
rounding. An empty distribution is not a zero-latency observation. If a borrowed
service does not publish durations, its histogram is absent and
`loom_serve_server_latency_available` is zero instead of exposing an incomplete
server total. Client completion still requires client measurement; this
histogram does not claim to be TTFT or device execution time.

## Measure first meaningful output

`loom_serve_model_first_output_seconds` measures model protocol receipt through
the first meaningful result accepted by the HTTP carrier. It has the same
identity labels and duration buckets, without an `outcome` label: a streaming
request can publish output and later fail or be cancelled. The server-scoped
`loom_serve_server_first_output_seconds` retains samples from removed models.
Each request contributes at most one sample, independently of log retention.

For streaming text, content, reasoning, or tool calls count as output; HTTP
headers, assistant-role announcements, empty deltas, usage, and errors do not.
An EOS-only empty completion has no sample. Nonstreaming text contributes when
its nonempty response is accepted, not when internal generation first produces
a token. Image, embedding, and decision requests contribute when their complete
result is accepted. Checkpoint controls and raw tokenization are not inference
output. These distinctions make the distributions useful across modalities
without calling every response a token.

The corresponding `first_output` event carries the request ID and `duration_ns`
for correlation with `request_begin` and `request_end`. It reports server-side
publication, not network delivery or client-visible TTFT. Text's separate
`request_stats.first_token_ns` measures admission to host processing of the first
selected token, including EOS; it is not the first-output metric. Router-held
time remains outside both server-side latency distributions.

## Poll bounded history

Without a selector, `/logz` reads server scope. `X-Loom-Model` selects a model:

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/logz \
  -H 'X-Loom-Model: vectors'
```

The response is newline-delimited JSON containing stream metadata, at most 64
retained records, an available current counter snapshot, and a cursor. Metadata
includes the stream `identity`, oldest retained position, publication head and
ring capacity. `partial:true` means this is not a complete view from the start.
A `gap` gives the skipped absolute position interval when publication has
overwritten unread history.

The last `cursor` record's `value` is an opaque resume token. Pass it unchanged
as `X-Loom-Cursor` on the next poll. Each reader owns its cursor independently;
polling does not consume or clear records for another reader. A malformed or
ahead-of-publication cursor returns 400. A cursor from another stream incarnation
returns 409; rediscover the model and start a new stream instead of combining
unrelated histories.

## Stream the same records over SSE

```sh
curl --no-buffer --fail --silent --show-error \
  http://127.0.0.1:8080/logz/stream -H 'X-Loom-Model: vectors'
```

Each `telemetry` SSE event contains the JSONL batch in its data and the next
cursor in its `id`. Reconnect with `Last-Event-ID` to resume. `X-Loom-Cursor`
also works; if both are present they must agree. The client can stop reading
without holding producer slots. Resuming after a wrap reports a gap and current
counters rather than inventing missing events.

Removal emits `stream_end` when output credit permits and closes that model's
observation connection. A new registration with the same name has a new identity.
Readers can rediscover registrations through `/v1/status`; observing a name does
not keep its model alive. Ctrl-C in curl stops the observer, not the server.

The same stream works over a Unix listener:

```sh
curl --unix-socket /srv/run/loom-serve.sock --no-buffer \
  http://localhost/logz/stream -H 'X-Loom-Model: vectors'
```

## Inspect requests and retained sessions

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/requests \
  -H 'X-Loom-Model: vectors'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/sessions \
  -H 'X-Loom-Model: scout'
```

The second command requires a text deployment named `scout`. Stateless image,
embedding and decision services return 501 for session inspection. An absent
name returns 404; an unprepared or retiring registration returns 409. Inspection
does not load a cold model or reopen its gate.

Each page examines at most 64 owner slots. A non-null `next_offset` becomes
`X-Loom-Offset` on the next request; an empty page can still have a next offset.
Pages are live captures, not a frozen snapshot. Deduplicating IDs and restarting
when `incarnation` or telemetry `identity` changes keeps a client from confusing
two registration lifetimes. Router-held and tokenizer requests are reported
separately from model-admitted entries.

Session IDs survive movement between `storage:"row"` and `storage:"cache"`.
`request:null` indicates idle retained history. Positions describe completed,
owner-published work rather than in-flight device writes. Referenced resident
tokens may include shared prefixes, so summing per-session counts does not
measure unique physical memory. Reserved tokens describe completion capacity,
not additional resident tokens. These responses expose no prompts or generated
content.

## Interpret memory and packing evidence

`server.resources.pools[].memory` reports each explicit serving pool's backing,
independently of its consumer devices:

| Field | What it counts |
| --- | --- |
| `storage` | Ordinary native buffers, including fixed parameters and state. With fixed backing, this also includes workspace backing. |
| `retained` | Shared elastic backing for parameters, mutable state, and workspace. It is `null` with fixed backing; elastic `limit_bytes` reports the selected physical capacity or a tighter explicit budget. |
| `workspace` | The workspace portion of `storage` (fixed) or `retained` (elastic), including reusable idle ranges. This is a breakdown, not an additional allocation. |

`storage.committed_bytes` plus non-null `retained.committed_bytes` describes
managed backing. Adding `workspace.committed_bytes` again double-counts it.
`storage.reserved_bytes` counts backing charged to live native reservations;
`storage.abandoned_bytes` is the subset of committed backing whose native return
failed, not reusable capacity. Native release can lag logical buffer retirement.

Workspace values are independent HAL observations, not one coherent device-wide
snapshot. Its cumulative acquire/release/reuse counters survive model removal
and pool trimming. `wait_count` counts reservations needing a reuse prerequisite,
not blocked host threads or elapsed wait time. An exhausted attempt can be
followed by growth and success; it is not a failed-request counter.

Virtual `reserved_bytes` can remain nonzero after physical backing is trimmed:
the reusable addresses are still owned without retaining those physical pages.
These scopes exclude host snapshots, compiler/tokenizer allocations,
executable/driver storage, and hidden physical padding. These values are not
peak process RAM, whole-device VRAM or immediately available memory. Status
does not poll hardware sensors.

| Question | Useful evidence |
| --- | --- |
| Is work waiting outside model admission? | Router-held counts versus queued/active model requests and their oldest ages. |
| Are parameters preventing new work from admitting? | `parameters.admission.waiting`, its age, and peer consumer pins or persistent parameter pinning. |
| Is logical KV capacity full? | `text_memory.pool` resident plus reserved token positions versus capacity. |
| Is output backpressure retaining rows? | `text_activity.backpressured_rows`, separate from prefill/decode rows. |
| Are smaller requests bypassing a blocked head? | `admit.queue_rank`, `overtakes` and `overtake_budget`. |
| Is reuse reducing prompt work? | Response `prompt_tokens_details.cached_tokens` and text-work counters. |
| Is packing completing more useful work? | Client throughput/latency together with epoch token slots, committed work and draft acceptance. |

An `epoch` event covers a completed host-visible cohort including device-fed
continuation. `token_capacity` describes its first bucket; `token_slots` sums
the capacities actually issued across the cohort. Drafts rejected by the
verifier consume slots without producing committed tokens. Dividing continuation
work by only the first bucket misstates fill. A null capacity observation means
the selected execution path does not provide that measure.

## Measure the user-visible result

For text, record client send time, first emitted token and completion. Report
time to first token and the distribution of completion times alongside useful
output tokens per second across the full concurrent cohort. Image latency ends
when the encoded image is available; embedding/decision latency ends when the
complete response is available. Keeping these boundaries explicit is more useful
than assigning every modality a token-rate score.

A controlled comparison uses the same prompts, output limits, cache policy,
arrival pattern, image geometry or vector dimensions. Cold preparation and warm
inference are separate measurements. Draft proposals are not delivered tokens;
cached input is not newly computed prefill. Event gaps preserve counter totals
but prevent exact event-by-event latency reconstruction. Hardware timing,
bandwidth, power and peak-memory claims require corresponding measurements;
queue phase, packed occupancy and allocated-byte accounting do not establish them.

## Inspect host work in Tracy

A Tracy-enabled `loom-serve` exposes serving zones alongside IREE's task,
I/O, and HAL instrumentation. Tracing is a build-time feature, not an HTTP
setting or a server flag. Use a tracing-enabled binary and matching
[Tracy 0.14.1 tools](https://github.com/wolfpld/tracy/releases/tag/v0.14.1).
The release includes the graphical profiler, `tracy-capture`, and
`tracy-capture-daemon`. Source-build instructions live in the
[runtime tracing guide](https://github.com/ROCm/hrx-system/blob/main/runtime/src/iree/base/tracing/README.md).

### Watch a running server

Start the tracing-enabled server with an explicit Tracy TCP port. This example
uses the CPU model from the [serving walkthrough](serve-models.md):

```sh
TRACY_PORT=18086 loom-serve --device=task --pool_backing=fixed \
  --listen=127.0.0.1:8080 --model=vectors/model.json
```

In the Tracy profiler's connection window, enter `127.0.0.1:18086` and connect.
Send ordinary inference requests to HTTP port 8080; their work appears in the
live timeline. Worker lanes retain their actual thread identities; a model does
not create a dedicated thread. Save the capture from the profiler when finished.

For a remote server, forward the Tracy TCP port and connect the profiler to the
same local address:

```sh
ssh -N -L 18086:127.0.0.1:18086 user@inference-host
```

The Tracy listener is separate from HTTP and is not restricted by `--listen`.
It has no authentication and exposes profiling metadata, including source paths.
Firewall it to the intended observers or use a trusted network and SSH tunnel.

### Capture without a graphical profiler

Start this collector in another terminal before launching the server:

```sh
tracy-capture -a 127.0.0.1 -p 18086 -o serving.tracy -s 30 -m 5
```

It waits for the client, records for 30 seconds after connection, and saves
`serving.tracy`. Ctrl+C ends the capture early and saves it. Open the saved file
in the matching Tracy profiler. The collector's `-m 5` is a memory limit expressed
as a percentage of the collector machine's physical RAM, not 5 MiB or a limit on
the server. Capturing changes timings; these traces explain execution structure
rather than establishing production throughput.

For automatic discovery and recording across server launches, run the daemon
before starting servers:

```sh
tracy-capture-daemon -o traces --filter-name loom-serve \
  --filter-port 18086 -m 5
```

The daemon stays in the foreground and connects to discovered clients whose
program name contains `loom-serve` and whose TCP data port is 18086. It writes
one `.tracy` file per capture when that session ends. Stop the server normally
to finish its session, or press Ctrl+C in the daemon to finish and save all
active captures. A service manager must use SIGINT for graceful daemon shutdown;
SIGTERM is not its save-and-exit signal.

Discovery uses UDP port 8086, even when `TRACY_PORT` selects another TCP port.
The daemon must be on the server's broadcast network; an SSH TCP tunnel does not
forward discovery. Running the daemon on the server machine or using explicit
`tracy-capture -a ... -p ...` works without cross-network discovery. The daemon's
`-p` changes its UDP discovery port; `--filter-port` selects the advertised TCP
port.

### Capture lifetime and memory

The current runtime provider records from process startup and supports one
collector connection per process. The live profiler, command-line collector,
and daemon are alternatives, not simultaneous readers. Once a capture
disconnects, a fresh server process is required for another capture; reconnect
and on-demand tracing are not enabled.

Events accumulate in the server before its first connection, so attaching
promptly or starting a collector first matters for long-running services.
Collectors retain the decoded trace in memory and write the file at capture
completion. Neither their memory limit nor the daemon provides a rolling trace
or crash-durable stream. Bounded captures are appropriate for detailed
investigations; `/metrics` and the bounded `/logz` streams remain the continuous
operational views. A daemon-owned capture becomes a file to inspect afterward,
not a second live connection for the graphical profiler.

### Read the serving timeline

The **Sections** view shows independently overlapping ranges with the
deployment name in each label, such as `model-prepare:scout` and
`model-execute:researcher`:

- `model-prepare` spans source preparation, compilation, tokenizer creation,
  and publication of the prepared service.
- `model-warmup` spans an explicit activation request, including preparation
  when needed, capacity waiting, and parameter readiness.
- `model-deactivate` spans an explicit deactivation request through completion,
  cancellation, or failure, including any wait for admitted work to retire.
- `model-retire` spans resource retirement after service borrowers finish.
  A failed release ends its range; an explicit retry starts another.
- `model-execute` spans one accepted text epoch, finite batch, image stage,
  speech batch, or recognition batch through observed device completion and
  feedback readiness.
  CPU input preparation, output encoding, and client backpressure are outside
  this range.

These ranges overlap independently across models and may cross worker threads.
They measure elapsed operation time, including queue waits and completion
delivery, not CPU utilization or timestamped device execution.
`model-warmup` describes `/v1/models/activate`; inference-triggered loading is
visible in the parameter and I/O zones rather than a synthetic warmup range.
Select a section to focus analysis on that part of the capture, then inspect
the ordinary zones inside it for the actual work and waits.

Serving zones distinguish scheduler turns, source compilation, input preparation,
output processing, queue submission, and completion handling. Image output has
separate PNG and JSON encoding zones. Model turns carry the deployment name and
registration incarnation. Service work carries the deployment name, the ordinal
from its `/logz` stream identity, and the relevant request or cohort IDs. These
numeric annotations appear in that order; CPU preparation and encoding zones
also include the phase name. These serving annotations contain no prompts or
generated content.

Each CPU zone stays on the worker that began it and ends before the instrumented
call returns. Asynchronous work gets its own zone on the worker that runs it.
The gap between submission and completion handling can include device execution
and readiness delivery; it is not a CPU zone or a device-duration measurement.
Comparing those boundaries with `/logz` helps distinguish preparation, submission,
and encoding from waiting. Tracing changes timing, so throughput comparisons use
uninstrumented runs of the same workload.
