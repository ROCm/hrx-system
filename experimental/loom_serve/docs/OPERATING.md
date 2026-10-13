# Running and managing the model server

The main serving entry point is
`//experimental/loom_serve/serving:server`. One process owns its device group,
task pool and HTTP listeners. Models have independent registration, admission
and residency lifetimes; adding a compatible source package does not require
rebuilding the executable. The single-model text/image tools are useful for
model development, but are not required to operate this server.

The learned-model deployments use Linux with ROCr and an AMD gfx11 GPU. The CLI
requires an explicit HAL device URI with `--device`; no backend is selected by
default. Repeat `--device` to construct an ordered group, and set each model
document's top-level `device` index when the group has several members. Source
text, embedding and decision fixtures also run on x86 CPU and Vulkan, and the
image fixture runs on Vulkan. CPU and Vulkan use `--pool_backing=fixed`.
All URIs pass to HAL unchanged. No local inference device is added implicitly;
available URI schemes depend on the linked drivers. Each model currently uses
one device, with independent device pools rather than cross-device sharing.
The [resident embedding fixture](../models/fixtures/embedding/README.md#verification)
also exercises actual XDNA HTTP execution with a target-specific source variant;
the ordinary command-based catalog is not an NPU deployment recipe.
[Fresh-machine setup](README.md#fresh-machine-prerequisites) covers compiler
targets, runtime dependencies and device access. Qwen and Krea are learned-model
examples; the embedding and decision fixtures establish their serving contracts,
not useful learned predictions. Model-specific shape and hardware requirements
still apply.

## First output, starting with no models

On a configured GPU runner, from the repository root:

```sh
build_tools/bin/iree-bazel-run --config=opt \
  //experimental/loom_serve/serving:server -- \
  --device=amdgpu --listen=127.0.0.1:8080 \
  --workers=8 --connections=64 --held_requests=32 --model_capacity=32
```

For a download-free x86 CPU first output, use the same command with
`--device=task --pool_backing=fixed`; the
[CPU-only build recipe](README.md#cpu-first-output-without-a-model-download)
supplies the matching compiler/driver settings. The embedding registration below
executes the package's native JIT kernels on the shared task workers. It needs
no GPU or checkpoint. The image fixture's workitem lowering is not yet supported
on CPU; the [fixture guide](../models/fixtures/README.md) distinguishes its GPU path.

Managed build and hardware-runner policies apply. A separate execution host
needs the built executable, its runtime libraries and the source/asset closure;
a successful build does not install ROCr or copy model packages to that host.
The process needs no compiler subprocess or model-specific native library.

After the stderr JSON `ready` record announces the bound address, these requests
work even with no models:

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/readyz
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models
```

The results are `{"status":"ready"}` and `{"object":"list","data":[]}`.
Listener readiness does not mean every registered model has finished loading.

Copy the embedding fixture's complete source package to `/srv/models/vectors`
on the server, then register it initially drained:

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/register \
  -H 'Content-Type: application/json' \
  -d '{"name":"vectors","admission":"drained",
       "base_directory":"/srv/models/vectors",
       "definition":{"kind":"embedding","prepare":"prepare.loom",
       "control":"control.loom","sources":["config.loom","embed.loom"],
       "assets":".","options":{"batch_units":64}}}'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/status
```

HTTP 202 acknowledges registration intent, not successful compilation. This
drained registration holds metadata only: it has not read model files or
allocated model storage. Status shows `loom.registration:"registered"` and
`loom.status.admission:"drained"`. Open it and request actual device output:

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/undrain \
  -H 'Content-Type: application/json' -d '{"model":"vectors"}'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"vectors","input":["a","bbb"],"dimensions":2}'
```

The model progresses through `preparing` to `ready`. Inference arriving during
preparation waits within the holding budget, so a client need not guess a JIT
delay. Undrain returns HTTP 200 with the changed admission state, not a promise
that JIT has completed. Result indices 0 and 1 contain `[1,1]` and `[6,3]`, with
four input tokens. This is deterministic arithmetic through the normal
VM/device/HTTP path, not a
language embedding. The [fixture catalog](../models/fixtures/README.md) adds
image and decision results; the [tiny text package](../models/fixtures/text/README.md)
adds pooled sessions and checkpoints without a model download.

For a learned model, use a source package's asset and deployment instructions:
[Qwen](../models/qwen/README.md) or [Krea](../models/krea/README.md), with the
matching [shared-server catalog fields](../serving/README.md#launch). A Hugging
Face or GGUF file alone is not a runnable package: source must implement the
model graph, weight policy and modality contract.

## Configuration and resource boundaries

Process flags configure resources that outlive models. Repeat
`--model=path/model.json` for initial registrations; omitting it starts empty.
Each file uses exactly the same [complete model document](MODEL_FORMAT.md) as
`POST /v1/models/register`. Ordinary `--flagfile=path` inputs can list
registrations and process settings. Positional arguments are rejected before
startup. Unknown, duplicate, wrongly typed and wrong-modality fields reject.

| Process option | Default and scope |
| --- | --- |
| `--device` | Required; repeat HAL device URIs in group order. Multi-device model documents require a `device` index |
| `--listen` | `127.0.0.1:8080`; repeat for numeric IPv4, `[IPv6]:port` or `unix:path` listeners |
| `--workers` | `8`; shared CPU task workers for JIT, scheduling and media work, not one thread per connection |
| `--connections` | `64`; admitted connections across all listeners, including SSE, incomplete and draining requests. Each listener may park one additional unadmitted peer; further arrivals stay in its kernel backlog |
| `--held_requests` | Half the connection limit; requests held outside closed/preparing model admission. Must be smaller than `connections`; zero rejects instead of holding |
| `--model_capacity` | `32`; all registered names, including drained, failed and retiring entries |
| `--request_body_bytes` | `8388608`; maximum HTTP body per connection, independent of tokenizer/model limits |
| `--pool_backing` | `elastic`; demand-committed parameter/state slabs on each device. `fixed` eagerly backs storage and supports device address sanitization |
| `--slab_bytes` | `2097152`; physical slab granularity on each device. Zero selects the allocator recommendation |
| `--memory_bytes` | `0`; no serving-imposed retained parameter/state budget. A nonzero byte limit requires elastic backing and exactly one device |
| `--heartbeat_ms` | `1000`; periodic model telemetry. Zero disables periodic samples, not counters or request events |

The memory limit is **not a process-RAM or total-VRAM limit**. Private command
workspace, fixed buffers, compiler/driver overhead, HTTP bodies and DRAM
snapshots require additional headroom. Each admitted model must fit its weights
and required retained state; sharing a budget permits idle parameter eviction
and reload, not automatic KV paging to disk. Two registrations of the same
weights currently create separate parameter residencies.

An operator can release idle shared backing without unregistering models:

```sh
curl http://127.0.0.1:8080/v1/devices/trim \
  -H 'Content-Type: application/json' -d '{"device":0,"target_bytes":0}'
```

`device` is the index published in `/v1/status` under `server.resources.devices`.
The one-shot target covers reported workspace plus elastic retained commitment;
it does not change a budget. Trim releases idle workspace first, then eligible
unpinned weights. Accepted work, pinned weights and retained conversations stay
protected. HTTP 200 reports the requested target and the actual device resource
record, even when protected memory prevents reaching that target. Fixed weights
are not reclaimable. Subsequent requests reload evicted weights without re-JIT.

Text `rows` limits simultaneous execution slots (1–16); configured idle-history
and checkpoint capacity can retain additional sessions without execution slots.
`context_capacity` limits each logical context, and `pool_capacity` limits
shared physical token slots. Short sessions grow by pages instead of reserving
their entire logical context. Admission reserves the requested completion
high-water; excess work queues within bounded capacity or returns overload.
`prefill_capacity` bounds the source-JIT shape catalog, not prompt length.
The source may impose a tighter cap than the generic runner; Qwen currently
caps epochs at 512 tokens.
`checkpoint_capacity` reserves explicit pinned endpoints. Independent
`prefix_cache_capacity` enables bounded automatic sharing of exact intermediate
token prefixes, including reuse while the source request remains active. It
defaults to zero; [prefix-cache policy](../text/README.md#automatic-token-prefix-reuse)
explains capture boundaries, recurrent-state cost and eviction. MTP and
two-epoch continuation require source support.

`admission_overtakes` is independent of epoch packing. It defaults to `rows`:
when the oldest text request cannot reserve its complete KV requirement, up to
that many fitting requests can pass it. Zero selects strict FIFO. Arrivals,
failed probes and cancelled tails do not replenish the budget; removing the
oldest request does. The bound is on successful overtakes, not time: an active
slow reader can retain its reservation. Changing this knob requires a new
registration, as with the other deployment capacities. It does not change
kernels, reserve less completion space, or reorder shared weight admission.

With elastic backing, physical pressure first reclaims sufficient idle model
weights. If that cannot satisfy the request, idle text histories can also yield
across models sharing the pool. Reclamation releases one eligible cache owner
at a time and retires its normal asynchronous trim before retrying admission.
Active requests, prepared continuations and explicit checkpoint references
remain protected. An evicted `X-Loom-Session` can replay the full history on its
next request; the name is a cache key, not durable storage. A budget occupied
entirely by protected state still produces overload when no accepted work can
release capacity. Pool pressure does not automatically offload histories.

Each model also has `pending_requests` (default 32). For text/image it bounds
the waiting queue; for embeddings/decisions it includes active requests.
Increasing connection capacity alone does not increase model capacity.

The existing IREE flag-file mechanism supports reproducible process settings.
For example, save this as `serve.flags`:

```text
# Process resources; model registrations remain separate.
--device=amdgpu
--listen=127.0.0.1:8080
--workers=8
--connections=64
--held_requests=32
--model_capacity=32
--pool_backing=elastic
--memory_bytes=38654705664
```

Pass `--flagfile=/path/to/serve.flags` after the Bazel wrapper's `--` in place
of the inline flags. This example caps retained commitment at 36 GiB; it is
not a recommendation for every machine or model. Files contain one literal
flag per line, with no shell variable expansion. Repeated `--listen` entries
add listeners; later scalar flags override earlier values.

Relative `--model` and `--flagfile` arguments use the server's process working
directory. References inside a model document use the document directory;
inline HTTP documents require an explicit absolute `base_directory` when they
contain relative paths. The [path contract](MODEL_FORMAT.md#local-path-binding)
also covers source-declared parameter/policy paths. An external deployment
includes every referenced provider, VM program, tokenizer asset and checkpoint.
Runtime registration does not rewrite startup files; the management client
owns persistence.

## Observe, drain, activate, deactivate and remove

`GET /v1/status` distinguishes registration, admission and scheduler phase.
A source/package failure remains inspectable on its `failed` registration;
other models keep serving. `/healthz` and `/readyz` remain successful in that case.
Correcting the package and explicitly unregistering/re-registering starts a
new attempt and incarnation; there is no silent model substitution.

The same response's `server` object discovers actual worker and admission
limits, bound TCP/Unix addresses, HAL device identities and shared pool memory
accounting without knowing launch flags. It is available before any model is
registered. `server.resources.devices[].memory` separates elastic retained
backing from command workspace; fixed retained accounting is null, not zero.
These are serving-managed pools, not whole-device VRAM totals or immediately
available allocation bytes. See the [status contract](../serving/README.md#operational-discovery)
for the fields and ownership scopes.

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/status
curl --fail --silent --show-error http://127.0.0.1:8080/metrics
curl --fail --silent --show-error http://127.0.0.1:8080/logz \
  -H 'X-Loom-Model: vectors'
```

Metrics provide cumulative counters and current ownership gauges. `/logz` is
bounded JSONL polling with independent cursors. `/logz/stream` sends the same
records and counter snapshots continuously as SSE, over TCP or Unix sockets:

```sh
curl --no-buffer --fail --silent --show-error http://127.0.0.1:8080/logz/stream \
  -H 'X-Loom-Model: vectors'
curl --unix-socket /path/to/loom-serve.sock --no-buffer \
  http://localhost/logz/stream
```

Each telemetry event's `id` can be resumed with `Last-Event-ID`. Slow readers
never block producers or model removal. Overwritten history is reported as
gaps; counters still count. A removed model emits `stream_end` and closes its
observation connection when send credit permits. The optional polling observer
can attach before any models are registered and follow their lifetimes:

```sh
build_tools/bin/iree-bazel-run //experimental/loom_serve/tools:observe -- \
  --url=http://127.0.0.1:8080 --log=/path/to/new-run.jsonl
build_tools/bin/iree-bazel-run //experimental/loom_serve/tools:dashboard -- \
  /path/to/new-run.jsonl
```

The observer creates a new log rather than overwriting one; it and the dashboard
run separately from inference. Ctrl-C stops the selected client, not the server.
The [observation contract](../serving/README.md#operational-discovery) describes
counter scope, gaps, incarnation changes and unmeasured quantities.

### Reading pressure without guessing at utilization

| Question | Evidence |
| --- | --- |
| Is work waiting before model admission? | `/v1/status` separates router-held work from model queued/active requests and their oldest ages |
| Is logical KV capacity occupied? | `text_memory.pool`: `resident_tokens + reserved_tokens` consumes `capacity_tokens`; per-session counts can include shared positions |
| Is a slow client retaining execution capacity? | `text_activity.backpressured_rows` counts rows awaiting output credit, separately from prefill/decode rows |
| Did backfill actually happen? | `admit.queue_rank > 0` records a successful bypass; `overtakes` and `overtake_budget` describe the current oldest request's budget |
| Did retaining more agents preserve useful history? | Client `prompt_tokens_details.cached_tokens` and the benchmark's follow-up hit/miss groups separate reuse from full replay |
| Did more packing complete more useful work? | Client cohort output rate and latency, together with `text_work` prefill/decode counts, traversals and MTP acceptance |

These are distinct limits. A free execution row does not imply a fitting KV
reservation; a free logical KV position does not imply physical backing fits
the retained-memory budget. Server phase and packed slot occupancy are not GPU
utilization or bandwidth measurements. Always-counted totals remain useful
after a log gap, but an exact admission/latency reconstruction requires complete
event coverage. The [comparison procedure](PERFORMANCE.md#a-controlled-comparison)
keeps request work, cache replay and machine state visible together.

An `epoch` record covers a completed host-visible cohort, including any
device-fed continuation. `token_capacity` describes its first bucket;
`token_slots` sums the capacities actually issued across the whole cohort.
The dashboard divides consumed prompt/decode inputs by `token_slots` for its
committed-fill sample. Rejected draft candidates occupy work slots without
counting as committed inputs. Isolated/matched controls publish `token_slots`
as null because their per-span stages do not share the packed bucket; the
dashboard leaves their fill unavailable. Neither doubling the first bucket
nor dividing all continuation work by that first capacity is correct.

### Current requests and retained sessions

Current requests and retained text sessions are available without replaying log
history:

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/requests \
  -H 'X-Loom-Model: vectors'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/sessions \
  -H 'X-Loom-Model: scout'
```

Each page examines at most 64 owner slots. Pass a non-null `next_offset` back as
`X-Loom-Offset` until it becomes null; empty pages may still have a next offset.
These are live captures, not a frozen multi-page snapshot. A client deduplicates
IDs and restarts its scan if the model `incarnation` or telemetry `identity`
changes, including across server restarts. Request IDs match that model telemetry
stream. `held_requests` and `tokenizer_requests` report router-owned work
separately from the model-admitted `data` entries.

Session IDs survive movement between `storage:"row"` and `storage:"cache"`.
`request:null` means an idle retained history. Positions and counters describe
last completed owner-published work, not in-flight device state. Referenced
`resident_tokens` include shared prefixes and must not be summed as unique
physical memory; `reserved_tokens` is the active completion bound, not a count
of additional resident tokens. Explicit checkpoints remain separate from session
lifetimes. Responses contain no prompts, generated text or tool contents.

Inspection never opens admission or loads a cold registration. An unprepared or
retiring registration returns 409; an absent name returns 404. A service without
the requested capability returns 501, including session inspection on stateless
image/embedding/decision services. The same endpoints work on Unix listeners.

### Admission and model lifetime

Drain is reversible: accepted work finishes, new work waits within the holding
budget, and control/observation routes remain available. It does not unload
weights or delete sessions. Unregister is irreversible: it seals admission,
finishes accepted work with `mode:"drain"`, or cancels request intent with
`mode:"abort"`; both wait for actual borrowers to retire before releasing state.

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/drain \
  -H 'Content-Type: application/json' -d '{"model":"vectors"}'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/undrain \
  -H 'Content-Type: application/json' -d '{"model":"vectors"}'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/unregister \
  -H 'Content-Type: application/json' -d '{"model":"vectors","mode":"drain"}'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/status
```

With `--pool_backing=elastic`, `POST /v1/models/deactivate` releases a drained
model's reloadable weights without removing its compiled programs, sessions or
checkpoints:

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/drain \
  -H 'Content-Type: application/json' -d '{"model":"scout"}'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/deactivate \
  -H 'Content-Type: application/json' -d '{"model":"scout"}'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/status
```

The 202 response accepts intent; wait for `loom.parameter_operation:"idle"`,
check for `loom.error`, and confirm `active:false` and `committed_bytes:0` in
`loom.parameters`. Undrain then allows ordinary inference to reload the same
weight plans on demand. Undrain while deactivation is pending returns 409.
Prepared fixed-backed models reject deactivation with 409; a metadata-only
drained registration is already cold and succeeds without preparing a model.
Parameter accounting excludes retained session state, and native slab release
can still take time even though no task worker waits for model execution.

To load weights before reopening admission, use explicit activation. This also
prepares a newly registered drained model without sending an inference request,
and supports fixed or elastic backing:

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/activate \
  -H 'Content-Type: application/json' -d '{"model":"scout"}'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/status
```

The 202 response accepts intent. That model's `loom.parameter_operation` moves
from `waiting` (preparation, admitted work or capacity) to `activating` (device
readiness), then `idle`. Check `loom.error` before undraining. The operation
survives client disconnect; unregister cancels capacity interest or joins
accepted loading before releasing resources. Undrain and opposite operations
return 409 while a parameter operation is pending. A successful warmup does not
pin weights indefinitely: later pool pressure may evict idle parameters.

Failures in accepted native activation/deactivation, or shared backing
reclamation during preparation, stop server admission and drain all accepted
work before returning the error. Reclamation can have partially released
another model's weights, so these failures cannot be isolated to the requesting
model. Callback threads only publish completion; they never tear down shared
queues or pools. Invalid control requests and isolated source/package errors
do not trigger this shared shutdown.

The name remains occupied until removal completes; a second registration
returns 409 during that interval. A final, quiescent release failure reports
`release_failed` and retains its resources; another unregister explicitly
retries release. Restarting the entire process is not part of normal swapping.
SIGINT/SIGTERM stops application admission and joins accepted asynchronous work
before destruction; valid device work has no arbitrary shutdown deadline.

## Carrier and trust boundary

All listeners reach the same catalog, devices and connection budget. A Unix
listener is configured with `--listen=unix:/path/to/loom-serve.sock`; clients
use the same HTTP API:

```sh
curl --unix-socket /path/to/loom-serve.sock http://localhost/v1/models
```

The independent observer uses the same carrier override:

```sh
build_tools/bin/iree-bazel-run //experimental/loom_serve/tools:observe -- \
  --url=http://localhost --unix-socket=/path/to/loom-serve.sock \
  --log=/path/to/new-run.jsonl
```

The socket path is a literal local filesystem path, not URL-encoded. The HTTP
origin supplies the Host header but is not resolved or contacted over TCP.
Unix observation uses plain HTTP; an HTTPS origin with `--unix-socket` is an
error. JSONL records and the dashboard identify both the origin and carrier.
Observer timeouts apply to the socket as they do to TCP, not to model work.

An occupied socket path is a startup error, not permission to delete another
server's endpoint. Successfully owned paths are removed on listener teardown.

These are trusted administration interfaces. Registration executes local Loom
source and reads server-local assets. There is no built-in authentication,
authorization, TLS termination or source sandbox. Loopback, OS socket access
controls, tunnels or an independently configured proxy establish the trust
boundary. Binding a public address does not add an authentication layer.

## Reproducible integration check

```sh
build_tools/bin/iree-bazel-test --config=asan \
  //loom/src/loom/serve/server:configuration_test \
  //loom/src/loom/serve/server:router_test \
  //experimental/loom_serve/models/fixtures:catalog_test \
  //experimental/loom_serve/models/fixtures:catalog_elastic_test \
  //experimental/loom_serve/models/fixtures:catalog_vulkan_test
```

The GPU-backed catalog check runs the actual shared executable from an external
source deployment directory. It covers empty startup, inline/flag-file settings,
TCP/Unix listeners, initially drained registration, source failures that leave
peers serving, exact outputs across all four modalities, and removal/reuse with
counters preserved. It tests ownership and usability, not model quality or
comparative throughput.
