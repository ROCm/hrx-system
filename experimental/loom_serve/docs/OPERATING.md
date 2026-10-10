# Running and managing the model server

The main serving entry point is
`//experimental/loom_serve/serving:server`. One process owns its device group,
task pool and HTTP listeners. Models have independent registration, admission
and residency lifetimes; adding a compatible source package does not require
rebuilding the executable. The single-model text/image tools are useful for
model development, but are not required to operate this server.

The demonstrated deployment is Linux with ROCr and an AMD gfx11 GPU. The CLI
currently selects one AMDGPU device; the device-group foundation is not yet a
multi-device placement interface. [Fresh-machine setup](README.md#fresh-machine-prerequisites)
covers compiler targets, runtime dependencies and device access. Qwen and Krea
are learned-model examples; the embedding and decision fixtures establish their
serving contracts, not useful learned predictions. Model-specific shape and
hardware requirements still apply.

## First output, starting with no models

On a configured GPU runner, from the repository root:

```sh
build_tools/bin/iree-bazel-run --config=opt \
  //experimental/loom_serve/serving:server -- \
  --device=amdgpu --listen=127.0.0.1:8080 \
  --workers=8 --connections=64 --held_requests=32 --model_capacity=32
```

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

Register a small source-defined embedding fixture, initially drained:

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/register \
  -H 'Content-Type: application/json' \
  -d '{"kind":"embedding","name":"vectors",
       "source":"experimental/loom_serve/models/fixtures/embedding",
       "assets":"experimental/loom_serve/models/fixtures/embedding",
       "options":{"batch_units":64},"admission":"drained"}'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/status
```

HTTP 202 acknowledges registration intent, not successful compilation. This
drained registration holds metadata only: it has not read model files or
allocated model storage. Status shows `loom.registration:"registered"` and
`loom.status.admission:"drained"`. Open it and request actual GPU output:

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/models/undrain \
  -H 'Content-Type: application/json' -d '{"model":"vectors"}'
curl --fail --silent --show-error http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"vectors","input":["a","bbb"],"dimensions":2}'
```

The model progresses through `preparing` to `ready`. Inference arriving during
preparation waits within the holding budget, so a client need not guess a JIT
delay. Result indices 0 and 1 contain `[1,1]` and `[6,3]`, with four input tokens.
This is deterministic arithmetic through the normal VM/GPU/HTTP path, not a
language embedding. The [fixture catalog](../models/fixtures/README.md) adds
image and decision results; the [tiny text package](../models/fixtures/text/README.md)
adds pooled sessions and checkpoints without a model download.

For a learned model, use a source package's asset and deployment instructions:
[Qwen](../models/qwen/README.md) or [Krea](../models/krea/README.md), with the
matching [shared-server catalog fields](../serving/README.md#launch). A Hugging
Face or GGUF file alone is not a runnable package: source must implement the
model graph, weight policy and modality contract.

## Configuration and resource boundaries

Process flags configure resources that outlive models. `--models=path.json`
optionally supplies initial registrations as `{"models":[...]}`. Each entry
uses exactly the same schema as `POST /v1/models/register`; an empty array or
omitted catalog starts empty. Unknown, duplicate, wrongly typed and
wrong-modality fields are rejected.

| Process option | Default and scope |
| --- | --- |
| `--device` | `amdgpu`; one HAL device URI for all deployments |
| `--listen` | `127.0.0.1:8080`; repeat for numeric IPv4, `[IPv6]:port` or `unix:path` listeners |
| `--workers` | `8`; shared CPU task workers for JIT, scheduling and media work, not one thread per connection |
| `--connections` | `64`; simultaneous connections across all listeners, including SSE and incomplete requests |
| `--held_requests` | Half the connection limit; requests held outside closed/preparing model admission. Must be smaller than `connections`; zero rejects instead of holding |
| `--model_capacity` | `32`; all registered names, including drained, failed and retiring entries |
| `--request_body_bytes` | `8388608`; maximum HTTP body per connection, independent of tokenizer/model limits |
| `--pool_backing` | `elastic`; demand-committed parameter/state slabs. `fixed` eagerly backs storage and supports device address sanitization |
| `--slab_bytes` | `2097152`; physical slab granularity. Zero selects the allocator recommendation |
| `--memory_bytes` | `0`; no serving-imposed retained parameter/state budget. A nonzero byte limit requires elastic backing |
| `--heartbeat_ms` | `1000`; periodic model telemetry. Zero disables periodic samples, not counters or request events |

The memory limit is **not a process-RAM or total-VRAM limit**. Private command
workspace, fixed buffers, compiler/driver overhead, HTTP bodies and DRAM
snapshots require additional headroom. Each admitted model must fit its weights
and required retained state; sharing a budget permits idle parameter eviction
and reload, not automatic KV paging to disk. Two registrations of the same
weights currently create separate parameter residencies.

Text `rows` limits simultaneously resident sessions (1–16), `context_capacity`
limits each logical context, and `pool_capacity` limits shared physical token
slots. Short sessions grow by pages instead of reserving their entire logical
context. Admission reserves the requested completion high-water; excess work
queues within bounded capacity or returns overload. `prefill_capacity` bounds
the source-JIT shape catalog, not prompt length. The source may impose a tighter
cap than the generic runner; Qwen currently caps epochs at 512 tokens.
`checkpoint_capacity` is opt-in retained endpoint capacity, not automatic
prefix discovery. MTP and two-epoch continuation require source support.

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

Deployment source/asset paths resolve from the **server process working
directory**, not the catalog or flag-file directory and not the HTTP client's
machine. Entries in a model's `sources.txt` resolve from the model package.
External deployment copies include those referenced motif/provider files as
well as cold/warm VM programs, tokenizer assets and weights. Runtime
registration does not rewrite the startup catalog; persistence is the
operator's catalog or management client.

## Observe, drain and remove

`GET /v1/status` distinguishes registration, admission and scheduler phase.
A `failed` registration includes its error and remains inspectable; other
models keep serving. `/healthz` and `/readyz` remain successful in that case.
Correcting the package and explicitly unregistering/re-registering starts a
new attempt and incarnation; there is no silent model substitution.

```sh
curl --fail --silent --show-error http://127.0.0.1:8080/v1/status
curl --fail --silent --show-error http://127.0.0.1:8080/metrics
curl --fail --silent --show-error http://127.0.0.1:8080/logz \
  -H 'X-Loom-Model: vectors'
```

Metrics provide cumulative counters and current ownership gauges. `/logz` is
bounded JSONL polling with independent cursors, not a subscription that can
block producers. Overwritten history is reported as gaps; counters still count.
The optional observer can attach before any models are registered and follow
their lifetimes:

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

The name remains occupied until removal completes; a second registration
returns 409 during that interval. A platform release failure reports
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

An occupied socket path is a startup error, not permission to delete another
server's endpoint. Successfully owned paths are removed on listener teardown.
The current Python observer accepts HTTP(S) origins; `curl --unix-socket` is
the direct Unix-carrier client.

These are trusted administration interfaces. Registration executes local Loom
source and reads server-local assets. There is no built-in authentication,
authorization, TLS termination or source sandbox. Loopback, OS socket access
controls, tunnels or an independently configured proxy establish the trust
boundary. Binding a public address does not add an authentication layer.

## Reproducible integration check

```sh
build_tools/bin/iree-bazel-test --config=asan \
  //experimental/loom_serve/serving:configuration_test \
  //experimental/loom_serve/serving:router_test \
  //experimental/loom_serve/models/fixtures:catalog_test
```

The GPU-backed catalog check runs the actual shared executable from an external
source deployment directory. It covers empty startup, inline/flag-file settings,
TCP/Unix listeners, initially drained registration, source failures that leave
peers serving, exact outputs across all four modalities, and removal/reuse with
counters preserved. It tests ownership and usability, not model quality or
comparative throughput.
