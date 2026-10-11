# Serving multiple source models

The [operator guide](../docs/OPERATING.md) walks through empty startup, dynamic
registration, process configuration, observation and removal. This reference
details the serving and ownership contracts behind that flow.

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
and asynchronous I/O services. Every catalog borrows the host's shared JIT task
queue; adding or removing a model neither creates nor joins a compiler pool.
Every modality's registration drives asynchronous source/JIT preparation on
that pool and binds shared memory only on the serving owner. HTTP admission
and model scheduling run as a cooperative task process:
bounded serialized turns preserve shared memory-policy ownership, while CPU
preparation/output and device work proceed independently. Network, task release,
model frontiers and heartbeat deadlines wake the owner directly. Only the outer
application caller joins final scheduler release; no task waits for device work.
Listeners and accepted connections use the host's shared proactor pool. Adding
a connection or listener creates no private thread. A listener can retire while
its accepted response streams continue. The listener API scopes a failed bind
to that listener; this executable requires all configured startup bindings to
succeed before announcing readiness and otherwise unwinds startup. Process
signals belong to the application, not individual listeners.

The [routing owner](router.h) belongs to this application layer. Modality
services implement the [HTTP service contract](../http/service.h) without
depending on the router, listener lifetime, or application run loop. The
single-model executables use the same create, route, retire and destroy
ownership sequence as the shared server.

The catalog selects source packages and deployment options. Model math,
checkpoint interpretation, storage geometry and chat policy remain in the
Loom sources described by the [text model contract](../text/README.md),
[image model contract](../image/model.h), and
[finite-vector contract](../finite/README.md). None of these modalities requires
model-specific C in the executable.

## Launch

The server can start without `--models`, or with an explicit `{"models":[]}`
catalog. It still owns its configured device and shared task/I/O infrastructure;
health, model discovery, status, metrics and diagnostic endpoints remain live.
An empty server publishes empty model/status lists and zero model-work totals.
Unknown-model inference and tokenization requests return HTTP 404. Text, image,
embedding and decision packages can be registered and removed over HTTP;
startup catalog entries use the same asynchronous path. Source preparation and
JIT share the host workers. Initial device uploads use proactor readiness, not
worker waits. A failed package remains inspectable while peers keep serving.

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
      "checkpoint_capacity": 2,
      "session_cache_capacity": 8
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

## Registration and removal

`POST /v1/models/register` takes one model object using the catalog schema.
Optional `admission` is `"open"` by default or `"drained"`. The latter stores
only registration metadata and a bounded observation ring: it does not create
the model, read source/assets, reserve model storage or evict another model.
Opening admission or requesting explicit activation starts preparation. The
same setting is accepted for every modality in startup catalogs.

~~~sh
curl http://127.0.0.1:8080/v1/models/register \
  -H 'Content-Type: application/json' \
  -d '{"kind":"embedding","name":"vectors",
       "source":"experimental/loom_serve/models/fixtures/embedding",
       "assets":"experimental/loom_serve/models/fixtures/embedding",
       "admission":"drained","options":{"batch_units":64}}'
curl http://127.0.0.1:8080/v1/models/undrain \
  -H 'Content-Type: application/json' -d '{"model":"vectors"}'
curl http://127.0.0.1:8080/v1/status
curl http://127.0.0.1:8080/v1/models/unregister \
  -H 'Content-Type: application/json' -d '{"model":"vectors","mode":"drain"}'
~~~

Registration/removal return 202 after accepting server-owned intent. Closing
that connection neither cancels nor rolls back the operation. Discovery gives
managed entries a monotonic `loom.incarnation` and `loom.registration` phase:
`registered`, `preparing`, `ready`, `retiring`, `failed`, or `release_failed`.
Status additionally carries copied compilation evidence and terminal errors.
Inference arriving before preparation completes waits on ordinary holding
credit. A failed source/JIT leaves the name inspectable, rejects its held work
with 503, and leaves peer models serving. Unregistering and registering again
is an explicit new attempt, with a new identity.

Removal defaults to `mode="drain"`: all already-admitted requests finish,
including service queues and CPU tokenizer tasks. `mode="abort"` cancels their
intent; accepted CPU/device work still has to retire. Both seal admission,
answer held arrivals with 410, and retain the name through actual model resource
retirement. A request processed after the name disappears receives 404.
Graceful removal can be explicitly escalated to abort, but cannot be undone by
undrain. Re-registering an occupied name returns 409. A platform release failure
retains the model's remaining resources and shared parents; another unregister
explicitly retries release. No timeout turns graceful removal into cancellation.

`--model_capacity=32` bounds all names, including drained, failed and removing
registrations. Full registration credit returns 429. Discovery order is not a
scheduling-priority contract. Management currently uses trusted server-local
source/asset paths; these endpoints execute source programs and are intended
for the same trusted connections as process administration.

## Reversible model admission

`POST /v1/models/drain` and `POST /v1/models/undrain` accept exactly
`{"model":"name"}`. These controls change admission, independently of model
residency and retained sessions. A drain closes the gate when the serialized
owner processes it. Every request already accepted by the model, including its
waiting queue, continues normally. Later inference, checkpoint and tokenizer
requests wait outside the model until undrain; other models keep running.

~~~sh
curl http://127.0.0.1:8080/v1/models/drain \
  -H 'Content-Type: application/json' -d '{"model":"scout"}'
curl http://127.0.0.1:8080/v1/status
curl http://127.0.0.1:8080/v1/models/undrain \
  -H 'Content-Type: application/json' -d '{"model":"scout"}'
~~~

A successful control response acknowledges the gate change, not completion of
admitted work. Its copied model record and `/v1/status` report `admission` as
`open`, `draining`, `drained`, or `removing`, separately from the model's
scheduler `phase`.
`held_requests` counts requests outside admission; `tokenizer_requests` counts
accepted CPU utilities through final task release. Control and observation
requests bypass admission. Repeated drain or undrain is idempotent, and losing
the control connection does not undo a gate change already processed.

`--held_requests` bounds parked requests across all models. The default is half
of `--connections`; zero makes arrivals at a closed gate fail immediately.
The holding bound must be strictly below the transport connection bound. A
full holding budget returns HTTP 429 instead of consuming the last connection
slots needed to issue undrain. This isolates holding pressure, not every source
of transport pressure: incomplete requests and admitted response streams still
consume the shared connection budget. A reset peer relinquishes its held claim;
server shutdown cancels held requests without admitting them. Requests released
by undrain retain arrival order within each model and enter that modality's
normal bounded admission policy, which can still reject excess work.

Drain itself neither evicts prepared weights nor discards retained KV state.
It does not freeze background maintenance or make model storage safe to destroy;
retirement must also join accepted CPU/device borrowers. Unregister is the
separate irreversible operation; drain alone never deletes retained state.

## Warming and releasing parameters

`POST /v1/models/activate` accepts exactly `{"model":"name"}` for a healthy
managed registration with closed admission. It prepares a metadata-only model,
waits for any already-admitted work, and loads its parameters on the ordinary
model timeline. Both fixed and elastic backing support activation. No dummy
inference is performed; request counters and retained sessions are unchanged.
The 202 response accepts server-owned intent, not completed loading. A client
can disconnect without cancelling it.

~~~sh
curl http://127.0.0.1:8080/v1/models/activate \
  -H 'Content-Type: application/json' -d '{"model":"scout"}'
curl http://127.0.0.1:8080/v1/status
~~~

Status reports `loom.parameter_operation` as `waiting`, `activating`,
`deactivating` or `idle`. `waiting` includes source preparation, admitted work
and physical capacity; `activating` means a parameter-readiness wait has been
submitted. After activation reaches `idle`, check for `loom.error` before
undraining. Admission remains closed throughout. Repeated activation is
idempotent; undrain and the opposite parameter operation return 409 while an
operation is pending. Unregister withdraws waiting admission or joins accepted
activation before retiring resources. No task worker blocks on device readiness.

Warmup is not permanent pinning: subsequent pressure may evict idle weights.
A model whose complete parameters exceed the configured physical pool budget
fails preparation instead of indefinitely blocking the admission queue. A
temporary shortage waits for other work or explicit model/state release.

`POST /v1/models/pin` and `/v1/models/unpin` accept exactly `{"model":"name"}`
and return 200 after changing weight-eviction policy. The idempotent policy
appears as `loom.parameter_pinned`, including for metadata-only declarations.
Pinning does not prepare a model, load weights, reserve capacity, close admission,
or pin any sessions/checkpoints. To retain a warm model, pin its declaration
before activation:

~~~sh
curl http://127.0.0.1:8080/v1/models/pin \
  -H 'Content-Type: application/json' -d '{"model":"scout"}'
curl http://127.0.0.1:8080/v1/models/activate \
  -H 'Content-Type: application/json' -d '{"model":"scout"}'
~~~

Pin/unpin also works with open admission and accepted inference. Unpin restores
LRU eligibility without unloading immediately or removing accepted work's own
protection. A pinned model cannot deactivate; unpin first. Pinning conflicts
with already-accepted deactivation. Unregister still retires a pinned model
normally; a replacement registration starts unpinned. An idle permanent pin is
not work that promises to release capacity: peer requests can still receive
capacity errors, and explicit warmup may wait for an operator to unpin/remove
models. Both fixed and elastic backing retain the policy; fixed backing remains
non-evictable regardless of policy.

`POST /v1/models/deactivate` accepts exactly `{"model":"name"}` after
admission is closed. Its 202 response accepts a registration-owned operation:
already-admitted inference finishes, then reloadable parameter backing is
released. Losing the control connection does not cancel the operation. Programs,
tokenizers, session identities, KV/recurrent state and pinned checkpoints remain
owned by the same model incarnation. This is not full model unloading.

~~~sh
curl http://127.0.0.1:8080/v1/models/drain \
  -H 'Content-Type: application/json' -d '{"model":"scout"}'
curl http://127.0.0.1:8080/v1/models/deactivate \
  -H 'Content-Type: application/json' -d '{"model":"scout"}'
curl http://127.0.0.1:8080/v1/status
~~~

The selected model's `loom.parameter_operation` becomes `idle` when the
operation finishes. On success, `active` is false and `committed_bytes` is zero;
its parameter address reservation remains intact. Requests arriving behind the
drain stay held. Once deactivation finishes, undrain opens admission and the next
inference activates weights on its ordinary device timeline, without recompiling
programs or re-registering the model. Undrain alone does not load weights.
Explicit activation can instead reload them before opening admission.

Deactivation requires elastic backing. Fixed-backed prepared models, open
admission, preparing/failed/removing registrations, and borrowed services return
409 without a partial operation. A drained metadata-only registration is already
cold and succeeds without constructing a model. Repeating deactivation is
idempotent; undrain during a pending deactivation returns 409. Unregister
supersedes pending deactivation and follows its selected drain/abort policy.
An actual parameter-release failure leaves admission closed with `loom.error`
available for inspection; explicit unregister owns subsequent cleanup.

This control waits for model work by advancing the shared task-driven service,
not by blocking a task worker on device completion. Physical slab unmap/free
calls are synchronous cold capacity transitions on the owner, so releasing a
large model can still incur native allocation-management latency. The operation
does not imply a zero-cost swap or release of retained session memory.

## Trimming shared memory

`POST /v1/devices/trim` accepts exactly `{"device":0,"target_bytes":0}`. Both
fields are unsigned integers; `device` selects the immutable index published in
`/v1/status` under `server.resources.devices`. The one-shot target is the sum
of reported workspace and elastic retained commitment, not a changed budget or
a whole-device VRAM limit. Missing, repeated, unknown or invalid fields and
unpublished device indices return 400 before changing memory ownership.

The operation releases eligible idle workspace first and then unpinned parameter
groups in LRU order. It neither closes model admission nor waits for live work.
Execution pins, explicit weight pins, retained KV/recurrent state and checkpoints
survive. Fixed parameter backing cannot be released this way. A busy or retained
allocation can leave memory above the target; this is still successful trim.

HTTP 200 returns `{"target_bytes":0,"device":{...}}`, where `device` has the
same identity and memory schema as the status resource record. Accounting is
observed after trim; workspace values remain independent atomic observations.
Losing the HTTP response does not undo released backing. Later inference reloads
evicted parameters on its ordinary timeline without recompilation or registration
changes. The endpoint is identical over TCP and Unix listeners and works before
any model is registered.

Native unmap/free calls are synchronous cold owner work, not device completion
waits. A platform failure can leave partially released backing; it terminates
the response and starts ordinary server draining. It is not converted into a
request-local error that would permit new inference against that backing.

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

`GET /v1/status` returns the same `object`/`data` list with a copied
`loom.status` object on each model:

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

The response also includes a top-level `server` object, even with zero models.
`worker_count` is the actual provisioned task-worker count, not the requested
maximum. `model_capacity`, `connection_capacity`, `held_request_capacity` and
`request_limits` expose the configured registration, transport, admission and
framing bounds. `held_requests` and `observers` are current router-owned claims.
`observer_capacity` is the combined ceiling for held requests plus live log
observers, one below transport capacity. Ordinary active and incomplete network
requests also use transport credit; these counts are not total connections.

`server.resources.listeners` lists the actual bound addresses, including the
port assigned for `:0` and `unix:` paths. Each entry's `waiting_for_capacity`
reports a listener parked on shared connection pressure. It retains at most
one unadmitted peer while further arrivals stay in the configured kernel
backlog. An owner-reclaimed slot resumes admission without a client retry;
stopping a listener cancels this wait without revoking accepted requests.
Every endpoint observes the same logical server. `server.resources.devices`
lists all logical devices in its
single application group with group-local `index`, HAL `id`, `name`, `driver`,
`backend` and constituent `physical_devices` names/paths/backend ordinals.
Indices have process lifetime; identity strings are HAL descriptions and need
not be unique. This is an inventory, not model placement or a cross-device
visibility guarantee. Embedded routers can publish no devices/listeners when
their caller owns work externally or connects only in-process carriers.

Each device's `memory` separates retained model backing from shared command
workspace. `backing` is `fixed` or `elastic`. Elastic `retained` contains
`limit_bytes` (null for unbounded), `reserved_bytes`, `committed_bytes`,
`peak_bytes` and cumulative `released_bytes`, across all models on that owner.
Fixed `retained` is null, not zero usage. Workspace fields report
`reserved_bytes`, `committed_bytes`, `free_bytes`, `quarantined_bytes`,
`budget_limit` (zero for unbounded) and `reservation_count` using the HAL's
atomic pool observations. They can change independently during a snapshot;
free bytes are not a promise of immediately admissible allocations.
Neither scope includes all fixed buffers, loader/driver storage or hidden
physical padding, so their sum is not VRAM usage. Observation neither waits for
device work nor invokes hardware telemetry queries, residency changes or trim.

Ready managed source models also expose a copied `loom.parameters` object.
`active` means all weight-plan activations have been enqueued, not that their
device work has completed. `backing` is `fixed` or `elastic`. The separate
`loom.parameter_operation` reports explicit residency intent even before a
model finishes preparation. Cumulative `activations` and `deactivations` count
weight-plan transitions (a model can own several plans). Elastic backing reports
parameter-only `reserved_bytes` and `committed_bytes`. Fixed backing omits those
accounting fields instead of presenting its allocated memory as zero. Neither
these bytes nor these counters include the model's retained mutable state.
Cache pressure can also release idle weights, so `active` is not an admission
or pinning flag.

`GET /metrics` exposes scheduler ownership gauges and request counters in
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
server lifetime. Managed models also carry an `incarnation` label on every
metric, so reusing a name starts distinct series instead of resetting counters
in the previous registration's series. Fields are `requests`, `completed`,
`rejected`, `cancelled`, `failed`, `input_tokens`, `output_tokens` and `batches`,
with the `/logz` semantics below. Values retain unsigned integer precision.
Server totals are sums of the always-counted service owners plus final totals
from removed incarnations, not replayed event history. They include
model-specific controls such as checkpoints but exclude router-owned tokenizer
utilities, health/observation requests and unknown-route rejections.
`loom_serve_server_counters_available` is one only when every service publishes
counters; otherwise server totals are omitted instead of reporting a partial
sum. Weight deactivation does not reset a service's lifetime counters. Removing
a model preserves server totals; a new model under the same name begins its
own counters and telemetry incarnation.

`GET /readyz` and `/healthz` return `{"status":"ready"}` once HTTP routing is
available, even with zero models or while asynchronous preparation is running.
Cold weights, capacity waiters, failed packages and full model queues do not
make the entire listener unready. Per-model readiness is reported in status.
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
live-plus-retired owner sum as `/metrics`; a service without counters makes the
server report `counters_available:false`, not invented zero or partial totals.
The final `cursor.value` is
passed verbatim as `X-Loom-Cursor` on the next poll. Each poll considers at most
64 records; polling neither clears records nor adds events to the stream.

Managed registrations own their ring before preparation starts and lend it to
the ready modality service. Preparation, serving and retirement therefore keep
one stream identity; failed source packages have an observable history even
without a published model. Successful unregister destroys that stream only
after its producers retire. HTTP responses already own copies of their bytes
and do not retain the stream.

Readers have independent cursors and can pause, disconnect or fall behind
without holding ring slots or slowing inference. A late reader without a cursor
starts at zero and receives an explicit gap when the beginning was overwritten.
A cursor from another incarnation returns 409; malformed or ahead-of-publication
cursors return 400. A service without this observation interface returns 501.
Positions/identities are experimental native-lifetime values, not durable log
storage keys. Periodic external pollers can write these JSONL responses to a
file/stdout without putting that sink on a model worker.

`GET /logz/stream` keeps the connection open and sends those same bounded
batches as server-sent events. Each `event: telemetry` contains the JSONL batch
as multiple `data:` lines; joining the lines with newlines yields the `/logz`
record schema. Its `id:` is the next opaque cursor. `Last-Event-ID` resumes a
reader, or `X-Loom-Cursor` supplies its initial cursor. Conflicting nonempty
values return 400. Selection and cursor errors are reported before sending 200.

~~~sh
curl --no-buffer --fail --silent --show-error \
  http://127.0.0.1:8080/logz/stream -H 'X-Loom-Model: vectors'
# Resume with the previous event's id, copied verbatim:
curl --no-buffer --fail --silent --show-error \
  http://127.0.0.1:8080/logz/stream -H 'X-Loom-Model: vectors' \
  -H 'Last-Event-ID: <previous-event-id>'
~~~

The initial telemetry event supplies current metadata/counters. Later events need
new records or changed counters; there are no empty keepalive events or private
polling threads. Model heartbeat records, when enabled, also flow here. Each
reader formats only when its carrier has send credit. A stalled reader retains
no model, ring slot or pending rendered batch in the router; it may miss records
and subsequently receive a gap. Counters remain current even after gaps.

Unregister/replacement emits `event: stream_end` with JSON data containing the
old `identity` and `reason: "model_unregistered"`, then closes the stream once
accepted bytes drain. Removal itself never waits for that observer. The terminal
event does not promise an unread tail was preserved; removal before the first
telemetry batch may produce only that terminal event. A stopped server or failed
peer cancels observation immediately; EOF alone is not a completed-model marker.
Consumers end that subscription on `stream_end` and explicitly choose whether
to observe a new incarnation. TCP and Unix listeners have identical semantics.

Held inference and live observers together may claim at most `--connections - 1`
connections; further permanent claims return 429. This preserves one control
credit from those claims, not from arbitrary active inference or unframed socket
traffic. A one-connection server supports finite polling but not live observation.
Each owner turn formats at most one batch per eligible reader and at most the
router's request quota in total, with rotating reader priority. Native model
work advances before observation formatting.

The [observer](../tools/observe.py) owns an independent cursor per deployment
and uses finite polling to write file/stdout JSONL; it never launches or signals
the server. Automatic
discovery follows catalog changes, including attaching before any model is
registered. Explicit `--model` selectors follow those names through absence and
re-registration. Availability transitions and incarnation resets are recorded
explicitly; forgetting a removed name releases its cursor. Protocol and network
failures still terminate this observer instead of masquerading as empty streams.
The [dashboard](../tools/dashboard.py) displays per-incarnation counter snapshots,
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

`//experimental/loom_serve/models/fixtures:catalog_test` exercises the shared
executable with source-JIT text, image, embedding and decision packages using
independent arithmetic oracles, without downloading learned-model weights.
Its dynamic witness starts empty, registers twelve initially drained models,
opens admission, verifies concurrent output, replaces four deployments while
peers progress and returns to empty with cumulative counters preserved. It also
holds 256 incomplete HTTP connections while useful inference continues. Linux
task snapshots assert fixed userspace thread identities and report the kernel's
elastic io_uring workers separately. These are GPU-backed correctness and
ownership checks, not throughput or model-quality measurements.

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
