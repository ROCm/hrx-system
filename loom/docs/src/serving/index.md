# loom-serve

`loom-serve` serves source-defined models over HTTP. There is no subcommand and
no built-in model. Devices are explicit process resources; model registrations
have independent lifetimes.

```sh
loom-serve --device=URI [--model=path/model.json ...] [options]
loom-serve --device=task --pool_backing=fixed --listen=127.0.0.1:8080
loom-serve --flagfile=serve.flags
loom-serve --help
loom-serve --agents_md
```

The [serving walkthrough](workflows/serve-models.md) reaches a first response
without weights. [Stream speech](workflows/stream-speech.md) receives incremental
audio and explains how voices share numerical batches while advancing
independently. [Audio transcription](http-profiles.md#audio-transcription) accepts
recorded audio, and the
[client guide](workflows/connect-model-clients.md#transcription-and-dictation) describes
upload formats and Open WebUI dictation. [Model documents](model-documents.md)
specify programs, assets, placement and per-model capacities. [Author model programs](model-programs.md)
describes preparation, execution and the contracts for adding a model package.
The [operation](workflows/operate-model-server.md) and
[observation](workflows/observe-model-server.md) workflows describe lifecycle
controls, pressure, counters and telemetry cursors.

## Process options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--device=URI` | Required, repeatable | HAL device URI; encounter order defines device indices. |
| `--model=path` | None, repeatable | One JSON deployment document per value; omission starts empty. |
| `--listen=address` | `127.0.0.1:8080`, repeatable | Numeric IPv4 `host:port`, IPv6 `[host]:port`, or `unix:path`. |
| `--workers=N` | `8` | Shared CPU workers for JIT, model scheduling and media work. |
| `--execution_scope=MODE:RESOURCES` | One exclusion scope over host and every device | Repeatable interference scopes; `MODE` is `exclude` or `overlap`, and resources are comma-separated `host` or device indices. |
| `--execution_limit=RESOURCE:COUNT` | No additional software limit | Repeatable accepted-unit ceilings per host/device resource; `0` adds no ceiling. |
| `--model_capacity=N` | `32` | Registered names, including drained, failed and removing entries. |
| `--connections=N` | `64` | Simultaneous admitted HTTP connections across all listeners. |
| `--held_requests=N` | `-1` | Closed-gate holding credit; `-1` selects half the connection limit, `0` rejects rather than holds. |
| `--request_body_bytes=N` | `8388608` | Maximum HTTP body bytes per connection. |
| `--heartbeat_ms=N` | `1000` | Per-model stderr JSONL heartbeat interval; `0` disables periodic reporting, not cumulative counters. |
| `--workspace_idle_ms=N` | `0` | Trim reusable command-workspace backing after the device's last workspace allocation retires and stays idle for this interval; `0` disables expiry. |
| `--parameter_idle_ms=N` | `0` | Deactivate unpinned elastic model weights after this idle interval; `0` disables expiry. Sessions and KV/prefix history survive. |
| `--pool_backing=elastic\|fixed` | `elastic` | Demand-committed retained slabs or eagerly backed retained storage. |
| `--weight_slab_bytes=N` | `1073741824` | Reusable elastic parameter slab size in bytes (1 GiB); loading uses independent bounded read windows. `0` uses the allocator recommendation. Ignored for fixed backing. |
| `--state_slab_bytes=N` | `67108864` | Elastic mutable-state commitment/reclamation size in bytes (64 MiB); `0` uses the allocator recommendation. Ignored for fixed backing. |
| `--memory_bytes=N` | `0` | Serving-managed physical parameter/state limit; `0` imposes no serving limit. |

Model, connection and body capacities must be positive. Holding credit must
remain below connection capacity. Heartbeat/idle intervals, slab sizes, memory
limits are nonnegative. A nonzero memory limit
requires elastic backing and exactly one device: independent device pools cannot
enforce a shared byte limit.
It excludes invocation workspace, host allocations and driver overhead; it is
not a whole-process or whole-device memory cap.

`--help` shows serving options; `--help=all` also shows linked runtime options.
`--agents_md` prints compact integration guidance without initializing devices.
Unexpected positional arguments fail instead of being interpreted as models.

## Request priority and interference

Inference requests can carry `X-Loom-QoS: realtime` or
`X-Loom-QoS: background`; omission selects background. Priority applies before
CPU preparation and compatible batch selection as well as before device
submission. An open microphone without ready input or a stalled audio consumer
does not reserve execution merely by retaining its request.

The default gives ready realtime work exclusive turns across the host and
every device. This accounts for CPU, GPU, and NPU work competing for shared
memory bandwidth. Already accepted work retains its storage and runs to its
existing completion boundary; priority cannot interrupt an indivisible kernel
or shorten a previously submitted epoch. Background requests keep their
history and resume when conflicting realtime demand clears. Weight pinning
controls residency independently.

Interference and accepted-unit limits are separate controls. A scope closes
cross-class admission across all its resources; it has no shared count budget.
`--execution_limit=0:1` permits one accepted unit on device zero, while
`--execution_limit=host:8` permits eight host units. These units are source
operations or finite host tasks, with durations determined by the source;
their counts describe neither hardware queues nor compute occupancy. Task
devices claim both their device resource and the shared host resource.

Explicit overlap allows both classes to execute subject to resource limits:

```sh
loom-serve --device=task --pool_backing=fixed \
  --execution_scope=overlap:host,0 --execution_limit=0:2
```

The `host` resource denotes the shared application executor, and numeric
resources use the same zero-based `--device` order as model documents.
Repeated scopes can describe independent devices or shared interference:

```text
--device=amdgpu://0
--device=amdgpu://1
--execution_scope=exclude:host,0
--execution_scope=overlap:1
--execution_limit=0:1
--execution_limit=1:4
```

Here, ready realtime work on the host or device zero excludes background
admission in that scope; device one's native work can overlap it. All models
still use the shared host for preparation. Overlapping scopes compose their
restrictions: an `overlap` declaration cannot erase an `exclude` declaration.
Explicit scopes must cover the host and every device. Omitting all scopes
retains default exclusion even when limits are supplied. Invalid resource
names, duplicate limits, incomplete coverage, and malformed counts fail startup.

The configuration is fixed at process creation and can be kept in an ordinary
flagfile. With no competing realtime work, the policy preserves normal batch
shapes and submission boundaries. Under continuous contention, realtime has
priority for the next contested admission; a positive background share is not
implied by allowing overlap.

## Devices and listeners

Repeated devices form one ordered group. Each model selects one group member
through its top-level `device` index; omission is accepted only for a singleton
group. A URI is resolved by its linked HAL driver. The server neither chooses
an implicit device nor falls back to a different backend.

Device URI query parameters request immutable runtime services. `feedback`
provisions device-to-host diagnostics used by authored kernel assertions and
other UBSAN-style reports. `asan` and `tsan` provision address- and race-
sanitizer state and start the report transport each requires. They are bare,
additive parameters. `--device=amdgpu:?feedback` configures the default AMDGPU
device, while `--device=amdgpu://GPU-0e12865a3bf5b7ab?feedback&asan` configures
a specific device. `?feedback&asan&tsan` is also valid; explicit `feedback`
adds general authored diagnostics to the sanitizer services. A backend that
cannot provide a requested service rejects the URI during device creation.

Models on one device share its memory budget and provisioned queues. Each
loaded model holds a queue assignment until its last execution reference
retires. New assignments choose the fewest live model executions in each
selected queue family; removing and reloading a model reuses the available
inventory. A single-queue device is supported, and models share queues when
there are more live executions than queues.

For independent AMDGPU model submissions, provision extra queues at startup:

```sh
loom-serve --device='amdgpu://0?queue_count=2' \
  --model=models/chat/model.json --model=models/speech/model.json
```

Both model documents select the same device and share its storage and residency
policy. `queue_count` defaults to one per selected GPU and remains fixed for the
device lifetime. Separate queues allow independent submissions to overlap, but
do not reserve compute or memory bandwidth and provide no latency guarantee.
Assignment counts live executions; request batching and residency pinning remain
separate controls. Vulkan uses its provisioned inventory, and task devices use
the application's shared CPU workers.

The model's native source must compile for that device. `task` uses native x86
CPU execution. AMDGPU and Vulkan require the corresponding compiler target,
HAL driver and supported hardware. XDNA packages use resident direct-dispatch
programs rather than assuming a GPU command program also runs on the NPU.
Device-group membership alone does not distribute a model or share its memory
between members.

Elastic backing supports retained-memory budgets and reclamation of idle,
unpinned parameters on task and AMDGPU. It requires virtual reservations,
mapping-source allocations, and bind/unbind operations from the selected device.
Task uses host virtual memory where available; Vulkan requires suitable
sparse-memory capabilities. Fixed backing eagerly retains storage and does not
support parameter deactivation or a retained-memory budget.

Elastic backing fails during device-group initialization when the required
mapping capabilities are unavailable. Model-specific resource requirements are
checked during preparation. Neither path silently changes the memory policy.
Address-sanitized devices require fixed backing so their allocations can be
published to the sanitizer shadow state.

All listeners share one catalog, worker pool and connection budget. A port of
zero requests an ephemeral port. One JSON `ready` record is written to stderr
for each bound address after every startup binding succeeds. An occupied Unix
socket path is a startup error; successfully owned paths are removed on teardown.

```sh
loom-serve --device=task --pool_backing=fixed \
  --listen=127.0.0.1:8080 --listen=unix:/srv/run/loom-serve.sock
curl --unix-socket /srv/run/loom-serve.sock http://localhost/v1/status
```

The socket's parent directory must exist and permit creation. The same HTTP
protocol runs over TCP and Unix sockets; the `localhost` origin in this curl
command does not select another server over TCP.

## Flagfiles and paths

An ordinary flagfile contains one literal flag per line:

```text
--device=task
--pool_backing=fixed
--listen=127.0.0.1:8080
--model=/srv/models/vectors/model.json
--model=/srv/models/scout/model.json
```

Run `loom-serve --flagfile=serve.flags`. Scalar values use the last occurrence;
repeatable flags accumulate in encounter order. There is no shell expansion.
Relative flagfile and model arguments use the process working directory,
including arguments read from a flagfile. References inside a model document
instead use that document's directory or its explicit `base_directory`.

## Readiness, errors and trust

`/healthz` and `/readyz` describe server availability. `/v1/status` reports every
registration and its preparation or failure state. `/v1/models` lists prepared
deployments and their capabilities. A model registration's HTTP 202 means
accepted intent, not completed JIT or weight loading.

A bad source package remains inspectable while peers keep serving. An accepted
native failure affecting shared resources stops admission and drains accepted
work. SIGINT/SIGTERM likewise joins actual asynchronous work before destruction;
there is no arbitrary deadline that makes in-flight device storage safe to free.

The server has no built-in authentication, authorization or TLS termination.
Registration executes trusted source and accesses server-local assets; it is
not a source sandbox. Loopback, socket access controls, tunnels or an external
proxy establish the deployment's trust boundary.
