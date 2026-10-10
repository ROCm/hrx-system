# Serving multiple source models

`//experimental/loom_serve/serving:server` serves named text, image, embedding and
decision deployments through one logical HTTP server with independently owned
TCP, IPv6, and Unix-domain listeners. All deployments borrow
one device, parameter residency cache and private command workspace backing.
Each owns its work and
feedback timelines, source-JIT programs and weights. Text deployments own their
logical KV pool, sessions and checkpoints; image deployments own their finite
request storage. Registering the same checkpoint twice creates two
prepared parameter residencies; it does not deduplicate those weights.

The application owns one [host](../runtime/host.h) with shared IREE task workers
and asynchronous I/O services. Each catalog's JIT queue uses those workers;
adding a model does not create another compiler thread pool. Compilation has
an asynchronous completion API; startup constructors join on the application
thread. HTTP admission and model scheduling run as a cooperative task process:
bounded serialized turns preserve shared memory-policy ownership, while CPU
preparation/output and device work proceed independently. Network, task release,
model frontiers and heartbeat deadlines wake the owner directly. Only the outer
application caller joins final scheduler release; no task waits for device work.
Listeners and accepted connections use the host's shared proactor pool. Adding
a connection or listener creates no private thread. A listener can retire while
its accepted response streams continue, and one failed bind leaves the other
listeners and model state intact. Process signals belong to the application,
not individual listeners.

The catalog selects source packages and deployment options. Model math,
checkpoint interpretation, storage geometry and chat policy remain in the
Loom sources described by the [text model contract](../text/README.md),
[image model contract](../image/model.h), and
[finite-vector contract](../finite/README.md). None of these modalities requires
model-specific C in the executable.

## Launch

The [weight-free three-model catalog](../models/fixtures/README.md) is a complete
first-output recipe for image, embedding and decision packages. The example
below adds real text/image checkpoints; the same executable handles both.

Create a JSON or JSONC catalog, for example:

~~~json
{
  "models": [
    {
      "kind": "text",
      "name": "scout",
      "source": "experimental/loom_serve/models/qwen",
      "weights": "/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf",
      "tokenizer": "/path/to/tokenizer.json",
      "rows": 4,
      "context_capacity": 8192,
      "pool_capacity": 32768,
      "prefill_capacity": 128,
      "mtp_depth": 3,
      "continuation_epochs": 2,
      "checkpoint_capacity": 2
    },
    {
      "kind": "text",
      "name": "researcher",
      "source": "experimental/loom_serve/models/qwen",
      "weights": "/path/to/Qwen3.8-27B-UD-Q5_K_XL.gguf",
      "tokenizer": "/path/to/tokenizer.json",
      "rows": 2,
      "context_capacity": 131072,
      "pool_capacity": 65536,
      "prefill_capacity": 512,
      "mtp_depth": 0
    },
    {
      "kind": "image",
      "name": "illustrator",
      "source": "experimental/loom_serve/models/krea",
      "checkpoint": "/path/to/krea-2-turbo",
      "adapter": "/path/to/style.safetensors",
      "width": 384,
      "height": 384,
      "text_tokens": 512,
      "pending_requests": 4
    },
    {
      "kind": "embedding",
      "name": "vectors",
      "source": "experimental/loom_serve/models/fixtures/embedding",
      "assets": "experimental/loom_serve/models/fixtures/embedding",
      "options": {"batch_units": 64},
      "pending_requests": 16
    }
  ]
}
~~~

Paths resolve from the process working directory. Names are nonempty visible
ASCII identifiers, so the same value fits JSON requests and HTTP headers.
Unknown, duplicate and wrong-modality fields reject instead of silently changing
configuration. Image `checkpoint` is an asset directory, whereas text `weights`
is a checkpoint file; an image's optional `adapter` is interpreted by its source.
A model source can impose additional shape/storage constraints.
Embedding/decision `assets` names the package's asset directory; `options` is passed
unchanged as JSON to its cold source program. There are no native model-specific
embedding configuration fields. The example uses a deterministic ABI fixture,
not a learned embedding model.

~~~sh
build_tools/bin/iree-bazel-run --config=opt \
  //experimental/loom_serve/serving:server -- \
  --models=/path/to/models.json --listen=127.0.0.1:8080 \
  --listen=unix:/path/to/loom-serve.sock \
  --pool_backing=elastic --memory_bytes=38654705664
~~~

The repository's build and hardware-execution policy applies. `--device`,
`--slab_bytes` and `--memory_bytes` configure the shared device. `--workers`
sets the shared CPU worker maximum (eight by default). Connections
and HTTP body storage have independent bounds through `--connections` and
`--request_body_bytes`. Each model's `pending_requests` defaults to 32 and
text `max_tokens` to 512. Image defaults are 384-square pixels and 512 text tokens.
For embedding/decision deployments, `pending_requests` bounds all accepted requests,
including the active cohort. For text/image it bounds the waiting queue.
The budget must fit each model's weights plus live retained state; this example's
36 GiB fits either model separately, not all deployments together.
Other defaults and accepted ranges are in
[configuration.c](configuration.c).

`--listen` is repeatable and defaults to `127.0.0.1:8080`. Numeric IPv4
`host:port`, IPv6 `[host]:port`, and `unix:path` use the same HTTP/SSE
protocol and route to the same model instances and connection budget. A port of
zero selects an ephemeral port. One cold `ready` record reports each bound
address after all bindings succeed. A conflicting Unix socket path is never
removed to force startup; successfully bound filesystem paths are removed on
listener destruction. The deployment owns their namespace exclusively.

For a trusted sandbox-local Unix endpoint:

~~~sh
curl --unix-socket /path/to/loom-serve.sock http://localhost/v1/models
~~~

## Requests and retained state

`POST /tokenize` and `POST /detokenize` use a deployment's already loaded input
tokenizer without activating weights, taking a model row, or waiting for an
inference epoch. Independent finite CPU tasks share the application executor;
one model's device work does not hold these utilities behind its completion.
This raw-text profile uses llama.cpp's field names and defaults, plus the
required catalog `model` selector:

~~~json
{"model":"scout","content":"Hello!","add_special":false,"parse_special":true}
~~~

The response is `{"tokens":[...]}`. Detokenization accepts
`{"model":"scout","tokens":[...]}` and returns `{"content":"..."}`, using the
tokenizer's decoder and preserving special tokens. It can also be used with an
image deployment's prompt tokenizer. These operations do not apply chat
templates or model-specific prompt framing. Empty text/arrays are valid.

Each utility request allows at most 262,144 tokens and 1 MiB of decoded text,
independently of inference context capacity and the transport JSON body limit.
Limits include inserted special tokens and decoded output; exceeding one returns
413 without a truncated success. Tokens must be integer IDs present in the
selected vocabulary. Unknown/duplicate/wrongly typed fields return 400. An
unknown model or absent tokenizer returns 404; a tokenizer execution error
(including an unconfigured decoder) returns 500 for that request without
stopping other models.

The optional llama.cpp `with_pieces=true` mode is not supported: that protocol
requires raw token bytes, whereas the public tokenizer decoder combines partial
UTF-8 and replaces incomplete sequences on finalization. It returns 400 instead
of substituting vocabulary spellings or silently changing the response shape.
`with_pieces=false` is accepted. vLLM's `prompt`/`add_special_tokens` profile and
chat-tokenization fields are not aliases for this raw profile.

`GET /v1/models` lists deployment names. Chat requests use those names, not
the model source's intrinsic identity:

~~~json
{
  "model": "scout",
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "stream": true,
  "max_tokens": 16,
  "temperature": 0
}
~~~

SSE responses report the selected deployment name. `X-Loom-Session` and
`X-Loom-Checkpoint` are scoped to that deployment, so two models can use the
same session/checkpoint string without sharing state. Explicit checkpoint
creation, deletion and suspension select their model with
`X-Loom-Model: scout`. That header may be omitted when exactly one deployment
is registered. Unknown model names return 404; ambiguous checkpoint selectors
return 400. `GET /healthz` does not activate a model.

Images select their deployment with the same `model` field at
`POST /v1/images/generations`:

~~~json
{
  "model": "illustrator",
  "prompt": "a red fox in the snow",
  "seed": "0",
  "strength": 0,
  "response_format": "b64_json"
}
~~~

The response contains one base64 PNG. The configured image shape is fixed;
`strength` selects the loaded adapter's weighting (zero uses base weights).
Without an adapter only the default strength of one is accepted. A text model
does not accept image requests, or vice versa.

Embeddings use `POST /v1/embeddings`:

~~~json
{"model":"vectors","input":["a","bbb"],"dimensions":2,"encoding_format":"float"}
~~~

The input may be a string, ordered string batch, token-ID sequence or batch of
token-ID sequences. The result is an `object:"list"` with ordered embedding
objects, per-request indices and input-token usage. The default wire encoding
is float; base64 contains little-endian IEEE754 F32 values. `dimensions` is an
optional source-supported output width, not a native truncation request. An
optional string `user` is accepted as client metadata. Source programs own
framing, input limits, projection and normalization; unknown options reject
instead of selecting undocumented model behavior.

Different HTTP requests can share a source-selected cohort without sharing
response identity. A cancelled peer loses only its own response. The native
encoder receives compact final vectors, not hidden activations. See the
[fixture](../models/fixtures/embedding/README.md) for exact predictable outputs
and the source replacement/parameter-pressure test.

Decisions use `POST /v1/systemone` with `model`, `state` and keyed typed
`questions`. Choice, score and noul answers come from source-produced compact
probability vectors. A `kind:"decision"` deployment uses the same `source`,
`assets`, `options` and `pending_requests` catalog fields as embeddings, and the
same finite cohort scheduler. See the [decision protocol](../decision/README.md)
and [source fixture](../models/fixtures/decision/README.md) for full request,
response, capacity and model-authoring semantics. An endpoint implementation
does not imply support for a learned decision checkpoint.

## Operational discovery

`GET /v1/models` includes a `loom` object on each deployment with its `kind`,
primary inference `endpoint`, and `tokenize` capability. These fields describe
the registered native modality and loaded tokenizer, not checkpoint identity,
quality, or support for every optional request field. Primary endpoints exclude
the raw utilities and text checkpoint controls documented above.

`GET /v1/status` returns the same list with a copied `loom.status` object:

~~~json
{
  "object": "model",
  "id": "illustrator",
  "loom": {
    "kind": "image",
    "endpoint": "/v1/images/generations",
    "tokenize": true,
    "status": {
      "phase": "generating",
      "queued_requests": 2,
      "active_requests": 1,
      "oldest_queued_age_seconds": 0.25,
      "oldest_active_age_seconds": 2.5
    }
  }
}
~~~

This operational schema is experimental. Counts describe scheduler ownership,
not GPU utilization. Image and finite-vector active requests include preparation,
parameter-capacity waiting, device work and encoding; reset peers still count
until their accepted work retires. Text active requests hold admitted rows;
the source-preparing FIFO head is still queued. Phase labels belong to each
modality. Text uses published counters while a CPU task owns mutable rows;
image and finite-vector heartbeats consume the same phase/count queries.
The age gauges use monotonic time since application acceptance for the oldest
request in each ownership class; empty classes report zero. Active age includes
prior queue residence and continues through preparation, capacity waiting and
output encoding. Cancellation ends the age only when that ownership retires.
These values distinguish prolonged ownership from fresh arrivals without
exposing request IDs or content; they are not device timings or completed-request
latency distributions. Heartbeats publish ages from the same receipt timestamps.
Snapshots neither activate a model nor wait for CPU/device completion, and
transport copies the response so a slow status peer retains no model buffers.
Unmeasured device times and whole-device memory totals are not reported as zero.

`GET /metrics` exposes those same ownership snapshots in
[Prometheus text 0.0.4 format](https://prometheus.io/docs/instrumenting/exposition_formats/).
It samples each deployment once per response, groups samples by metric family,
and returns `Content-Type: text/plain; version=0.0.4; charset=utf-8`. For example:

~~~text
loom_serve_model_queued_requests{model="illustrator",kind="image"} 2
loom_serve_model_active_requests{model="illustrator",kind="image"} 1
loom_serve_model_phase{model="illustrator",kind="image",phase="generating"} 1
loom_serve_model_oldest_queued_age_seconds{model="illustrator",kind="image"} 0.25
loom_serve_model_oldest_active_age_seconds{model="illustrator",kind="image"} 2.5
~~~

These are gauges, not cumulative request counters. The phase family contains
only the current phase with value one; an old phase disappears on the next
scrape. Labels contain deployment/modality/phase identifiers, never prompts,
session IDs or checkpoint paths. Metric names and phase vocabulary are
experimental. A scrape neither activates cold weights nor waits for a model;
it has the same admission and shutdown availability as the other probes.

The same endpoint also exposes cumulative `loom_serve_model_*_total` counters
with `model` and `kind` labels, plus `loom_serve_server_*_total` for the whole
fixed catalog. Fields are `requests`, `completed`, `rejected`, `cancelled`,
`failed`, `input_tokens`, `output_tokens` and `batches`, with the `/logz` semantics
below. Values retain unsigned integer precision. Server totals are sums of
the always-counted service owners, not replayed event history. They include
model-specific controls such as checkpoints but exclude router-owned tokenizer
utilities, health/observation requests and unknown-route rejections.
`loom_serve_server_counters_available` is one only when every service publishes
counters; otherwise server totals are omitted instead of reporting a partial
sum. Weight deactivation does not reset a service's lifetime counters.

`GET /readyz` and `/healthz` return `{"status":"ready"}` after model
construction and router registration. Readiness means the server can route
requests; cold weights, capacity waiters and a full model queue do not make the
entire listener unready. Before registration no application response is served.
Once shutdown stops application admission, the router serves no further probes;
the listener is released after accepted asynchronous callbacks drain. It does
not promise a final 503 response during wind-down. These endpoints do not
promise that a particular request will fit.

### Pull-based event history

Text services publish typed events into a bounded 256-record ring; image,
embedding and decision services use 128 records (about 20 KiB), independently
of their always-on request/model counters. Publication performs no formatting, allocation or sink
I/O. `--heartbeat_ms` adds periodic snapshots; zero disables periodic samples,
not counters or request/invocation events. The HTTP transport owns a separate
stream for peer, capacity, routing and tokenizer errors through shutdown. Its
existing connection-state mutex serializes diagnostic producers; readers never
take it. No ordinary HTTP diagnostic writes to stderr.
Text request/session/epoch events and image preparation/execution/encoding
measurements follow the same ownership contract. CPU borrowers return timings
with completed jobs; only the serialized service owner publishes. Image
`image_prepared` records carry the source-selected stage, tag and input bytes;
`image_generated` marks successful device completion and pin release, before
output encoding. Weight observations count parameter-plan activations,
deactivations and submitted read/preparation bytes, not measured device traffic.
They distinguish retained replay from reloading evicted parameters. Cold startup
compiler/residency diagnostics and terminal CLI failures are separate from
these serving-work streams.

`GET /logz` returns server diagnostics and current catalog-wide work counters. `X-Loom-Model`
instead selects a deployment's stream:

~~~sh
curl -s http://127.0.0.1:8080/logz -H 'X-Loom-Model: vectors'
~~~

The first `stream` line identifies the scope and stream incarnation, current
absolute `head`, oldest retained position and ring capacity. `partial` means
this response does not include the original start of the stream. Retained
events include `sequence`, monotonic `time_ns` and a request lifetime ID (zero
for model-wide events). A `gap` line reports the half-open missed interval
`[from,to)` for this reader. The `counters` line is a fresh cumulative snapshot,
not reconstructed from whatever events remain. Server totals use the same
fixed-catalog owner sum as `/metrics`; a service without counters makes the
server report `counters_available:false`, not invented zero or partial totals.
The final `cursor.value` is
passed verbatim as `X-Loom-Cursor` on the next poll. Each poll considers at most
64 records; polling neither clears records nor adds events to the stream.

Readers have independent cursors and can pause, disconnect or fall behind
without holding ring slots or slowing inference. A late reader without a cursor
starts at zero and receives an explicit gap when the beginning was overwritten.
A cursor from another incarnation returns 409; malformed or ahead-of-publication
cursors return 400. A service without this observation interface returns 501.
Positions/identities are experimental native-lifetime values, not durable log
storage keys. Periodic external pollers can write these JSONL responses to a
file/stdout without putting that sink on a model worker.

The [observer](../tools/observe.py) owns an independent cursor per deployment
and writes file/stdout JSONL; it never launches or signals the server. The
[dashboard](../tools/dashboard.py) displays per-incarnation counter snapshots,
not sums reconstructed from retained events. Gaps therefore lose history but
not cumulative totals. Shared-device memory is labeled separately from model
charges and is never added once per model.

Counters distinguish successful carrier acceptance, rejection, peer/shutdown
cancellation and execution/transport failure. `input_tokens` counts completed
device work, even when the requesting peer disconnected before delivery;
`batches` counts completed device invocations, not clients. Response acceptance
is not proof of remote receipt. Counter samples use host monotonic time, not
device timestamps. Bounded error messages own their bytes and explicitly report
truncation; terminal request/process statuses retain their normal ownership.

The `//experimental/loom_serve/telemetry:stream_benchmark` target separates
timestamped native publication from consumer copying/JSON formatting at the
actual 128/256-record capacities. Build with `--config=opt` and measure on an
excluded runner; its sanitizer smoke test checks execution, not performance.
`BM_PublishSnapshot` reports one published record per iteration;
`BM_CopyFormatBatch64` reports one 64-record batch per iteration. Divide the
latter time by 64 for a per-record cost. Construction and warmed output storage
are outside timing. Neither case includes TCP, file sinks or model execution,
so these measurements are not end-to-end inference speedups.

## Scheduling and memory ownership

The transport proactor handles network I/O independently. One application
owner accepts a bounded batch of requests, then advances each service, rotating
priority. A text turn submits at most one scheduling cohort, which may contain
two device-fed epochs when configured. Model calls and residency transitions
are never concurrent host operations.

Models have independent semaphore timelines even when they use the same exact
queues. Their command allocations share HAL TLSF backing; safe reuse follows
dependency frontiers, not a device-wide execution chain. Registering larger
workspace requirements creates backing lazily without draining other models.
Older pool geometry remains accounted until its roots and queue captures retire.
Model device work is submitted without host completion waits. Successful model
frontier readiness schedules a later owner turn, and finite CPU preparation
and output work releases its task before model reuse. Cold platform memory
mapping calls still execute synchronously on the owner.

Text admission pins parameters and physically reserves its completion high-water.
Image preparation retains host input without pins while waiting for admission;
an admitted invocation pins parameters through successful model completion.
Embedding preparation similarly retains a source-built cohort without pins
while waiting. Its input, output and private command workspace are queue-pooled;
the output's final feedback reader participates in retirement before reuse.
Other models retry queued admission as the owner advances. If the
budget fits only one model, another model waits until those parameter pins
release; the shared LRU can then evict idle weights and stream the requested
model back into its stable virtual roots. Retained KV and explicit checkpoints
remain model-owned. Weight eviction is not KV eviction.

Actual weight-capacity denial starts a FIFO of new-work interests within that
device's residency cache. New requests for an already-active model join behind
the older waiter instead of continually replenishing its pins. Accepted work
keeps its completion reservation and can acquire nested execution pins; it
drains normally. Successful admission or cancellation removes the interest.
Prepared input stays intact while waiting, so retries do not repeat tokenization
or source preparation. A physical denial with no active producer capable of
releasing capacity returns overload; deferral behind another waiter does not.

This policy queues weighted admissions conservatively within one cache after
pressure, not all requests globally. Parameter-free work bypasses it; separate
caches are independent. When all weights fit, no pressure FIFO is established.
It does not preempt accepted generation or choose mutable state to offload.
The router's pressure-release hint uses that same cache identity. Activity in
an independent cache cannot defer an otherwise impossible admission; only
services sharing capacity can supply the hint. A service with no shared
capacity identity is isolated even from other such services.

`--memory_bytes` bounds retained parameter/state VMM commitment, not total
device memory. Heartbeats separately report retained commitment, workspace
commitment/live reservations and workspace reuse. Driver overhead, fixed
buffers and host snapshots are outside that budget. Text JSONL events carry
the deployment name; request/epoch counters remain local to each model.

## Verification

The router tests use actual TCP carriers and borrowed request ownership.
Configuration tests exercise parsing and partial-initialization cleanup;
source-policy tests check that deployment aliases preserve rendered prompts.
The real-weight witness is `models/qwen:check_router`, supplied with the built
shared server, source package, checkpoint, tokenizer and an empty output path.
It compares sequential and overlapping final SSE, independent same-named
sessions/checkpoints, parameter eviction/reload and shared workspace reuse
under a one-model retained budget. Its timings are not a performance baseline.

The mixed-modality witness is
[`models/krea/check_router.py`](../models/krea/check_router.py). It accepts
`--server`, `--text-source`, `--weights`, `--tokenizer`, `--image-source`,
`--checkpoint`, `--adapter`, `--isolated` and `--output`. Run it from the
repository root with `python -m experimental.loom_serve.models.krea.check_router`
on a qualified GPU runner; Pillow decodes the PNG independently. The isolated
directory contains `check_model.py`'s `isolated-0.ppm` for the same Krea assets.
Its default one-worker, 36-GiB retained-budget run compares text-only control
with mixed execution, preserving a pinned conversation through image weight
activation and subsequent text reload. Prepared and queued image cancellation,
health progress, exact native pixels, exact retained text and shared workspace
reuse are checked separately. It does not measure comparative performance.
