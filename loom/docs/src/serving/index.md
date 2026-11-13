# loom-serve

`loom-serve` runs models over HTTP from portable model cards or serving
deployment documents. Programs may be Loom source, bytecode, or a combination
of both. Devices are explicit process resources; model registrations have
independent lifetimes.

```sh
loom-serve --device=URI [--model=path ...] [options]
loom-serve --devices=devices.json [--model=path/deployment.json ...] [options]
loom-serve --device=task --pool_backing=fixed --listen=127.0.0.1:8080
loom-serve --flagfile=serve.flags
loom-serve --help
loom-serve --agents_md
```

[Fetch and serve a model](workflows/fetch-and-serve.md) installs a published
Moonshine model and reaches an actual transcription response. The
[arithmetic embedding walkthrough](workflows/serve-models.md) exercises the CPU
server without learned weights. [Stream speech](workflows/stream-speech.md) receives incremental
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

[Sandboxing](sandboxing.md) runs the Linux SDK server without host networking,
with read-only model mounts and explicit GPU/NPU access. The portable commands
above remain the ordinary entry point on every platform.

## Process options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--device=URI` | None | Shortcut constructing device `device` and pool `pool`. Mutually exclusive with `--devices`. |
| `--devices=path` | None | Machine-local JSON declaring named devices and independent pools with their consumer membership, memory limits, and idle policy. |
| `--wait_mode=active\|blocking` | `active` | Completion-wait policy for every device. Overrides the machine file's `wait_mode`. Requests remain asynchronous. |
| `--model=path` | None, repeatable | A local model card, serving deployment document, or directory containing `model.json`; omission starts empty. |
| `--preset=name` | Card selection | Select a preset when loading one card directly. Fetched cards retain the selection made during acquisition. |
| `--listen=address` | `127.0.0.1:8080`, repeatable | Numeric IPv4 `host:port`, IPv6 `[host]:port`, or `unix:path`. |
| `--workers=N` | `8` | Shared CPU workers for JIT, model scheduling and media work. |
| `--model_capacity=N` | `32` | Registered names, including drained, failed and removing entries. |
| `--connections=N` | `64` | Simultaneous admitted HTTP connections across all listeners. |
| `--held_requests=N` | `-1` | Closed-gate holding credit; `-1` selects half the connection limit, `0` rejects rather than holds. |
| `--request_body_bytes=N` | `8388608` | Maximum HTTP body bytes per connection. |
| `--heartbeat_ms=N` | `1000` | Per-model stderr JSONL heartbeat interval; `0` disables periodic reporting, not cumulative counters. |
| `--workspace_idle_ms=N` | `0` | Trim reusable command-workspace backing after the pool's last workspace allocation retires and stays idle for this interval; `0` disables expiry. |
| `--parameter_idle_ms=N` | `0` | Deactivate unpinned elastic model weights after this idle interval; `0` disables expiry. Sessions and KV/prefix history survive. |
| `--pool_backing=elastic\|fixed` | `elastic` | Shared demand-committed parameter/state/workspace slabs or eagerly backed retained storage. |
| `--weight_slab_bytes=N` | `1073741824` | Reusable elastic parameter and shared-workspace slab size in bytes (1 GiB); loading uses independent bounded read windows. `0` uses the allocator recommendation. Ignored for fixed backing. |
| `--state_slab_bytes=N` | `67108864` | Elastic mutable-state commitment/reclamation size in bytes (64 MiB); `0` uses the allocator recommendation. Ignored for fixed backing. |
| `--memory_bytes=N` | `0` | Shared elastic physical-backing limit; `0` uses the selected source's physical capacity. An explicit limit can tighten it. |

The backing, slab, memory, and idle flags apply only to the `--device` shortcut.
With `--devices`, those settings belong to each pool entry; mixing the file with
any single-device flag is rejected, including explicit default values.

### Completion latency and CPU use

`wait_mode` selects how devices observe completion. `active`, the default,
favors latency and permits active observation by existing completion workers.
`blocking` disables optional polling and waits for native completion events,
reducing CPU use at the cost of possible wake-up latency. Both modes preserve
asynchronous requests, device concurrency, and completion guarantees; idle
devices can sleep in either mode.

Set `"wait_mode": "blocking"` at the root of a machine configuration, or use
`--wait_mode=blocking` with either `--device` or `--devices`. An explicit CLI
value overrides the file for every device. The setting is fixed at startup.

On AMD GPU user queues, active mode lets the existing completion worker poll
pending GPU work for up to one second after actual completion progress. Further
progress renews that budget; when no work remains pending, the worker sleeps
without waiting for the budget to expire. This reduces completion wake-up
latency and can improve throughput and energy per request. Polling occupies a
CPU core while active, however, and long GPU operations may use more CPU without
finishing faster. For battery-sensitive or CPU-constrained deployments, select
`--wait_mode=blocking`.

The device URI's `completion_spin_ns` parameter adjusts this budget; for example,
`--device='amdgpu://?completion_spin_ns=10000000'` selects 10 ms. Blocking mode
sets the budget to zero even when the URI specifies a nonzero value. AMD GPU
kernel queues, including the Windows native submission path, use native
completion events and do not use this polling budget. CPU devices report
completion directly from their execution workers; this setting leaves the
shared CPU executor unchanged.

### Memory and admission limits

`--memory_bytes`, `--weight_slab_bytes`, and `--state_slab_bytes` accept decimal
byte counts or lowercase size suffixes: `b`, `kb`, `kib`, `mb`, `mib`, `gb`, and
`gib`. For example, `--memory_bytes=64gib` sets a 68,719,476,736-byte allowance;
`64gb` instead means 64,000,000,000 bytes. Values must be nonnegative whole
numbers, with no spaces, signs, or fractional units. Overflow is rejected.

Model, connection and body capacities must be positive. Holding credit must
remain below connection capacity. Heartbeat/idle intervals, slab sizes and memory
limits are nonnegative. A nonzero memory limit requires elastic backing.
Elastic backing for parameters, permanent model buffers, sparse mutable state,
invocation input/output, and command workspace shares this limit. The reported
storage and workspace values are breakdowns of that physical total, not
additional allocations. Host snapshots, other host allocations, and driver
overhead remain outside it; it is not a whole-process or whole-device memory cap.

`--help` shows serving options; `--help=all` also shows linked runtime options.
`--agents_md` prints compact integration guidance without initializing devices.
Unexpected positional arguments fail instead of being interpreted as models.

## Devices and listeners

Exactly one of `--device` or `--devices` supplies the server's device group.
For explicit placement, a machine-local file names execution devices and storage
pools independently. This example keeps two policies separate even when both
consume the same physical RAM:

```json
{
  "devices": {
    "gpu": {"uri": "amdgpu://0"},
    "npu": {"uri": "xdna://0"}
  },
  "pools": {
    "interactive": {
      "devices": ["gpu"],
      "pool_backing": "elastic",
      "memory_bytes": "48gib",
      "workspace_idle_ms": 30000,
      "parameter_idle_ms": 300000
    },
    "embeddings": {"devices": ["npu"], "pool_backing": "fixed"}
  }
}
```

```sh
loom-serve --devices=devices.json --model=chat-deployment.json
```

The keys are operator-chosen names, not hardware types or ordinals. `gpu0` and
`gpu1` can refer to different drivers. Each deployment selects execution with
`"device": "gpu"` and storage with `"pool": "interactive"`. Each selector may be
omitted only when its inventory contains exactly one entry. Pool membership must
include the selected execution device. The portable model card and its programs
contain no machine placement. The same card can back several deployments with
different names, devices, or pools. `/v1/status` publishes the independent inventories in
`server.resources.devices` and `server.resources.pools`.
A URI is resolved by its linked HAL driver. The server neither chooses an
implicit device nor falls back to a different backend.

Each device entry contains `uri`. Each pool requires a nonempty `devices` array.
The optional root `wait_mode` selects `active` (default) or `blocking` for all
devices, as described under [completion latency and CPU use](#completion-latency-and-cpu-use).
Optional pool fields `pool_backing`, `weight_slab_bytes`,
`state_slab_bytes`, `memory_bytes`, `workspace_idle_ms`, and `parameter_idle_ms`
have the same defaults and units as the single-device flags above. Byte counts
can be JSON integers or strings such as `"64gib"`, `"64mib"`, or `"4096"`.
Strings preserve exact counts when a JSON client uses floating-point numbers.
Idle durations remain JSON integers in milliseconds. Unknown or duplicate
fields, duplicate keys or pool members, unknown devices, empty keys, and malformed values fail startup.
Keys use nonempty visible ASCII; numeric device selectors are rejected.
The file describes one group, not another set of servers or model lifecycles.

An optional pool `placement` object selects the backing domain independently of
the `devices` access list. This configuration requires a GPU-local allocation
domain supporting elastic backing:

```json
{
  "devices": {"gpu": {"uri": "amdgpu://0"}},
  "pools": {
    "weights": {
      "devices": ["gpu"],
      "placement": {"device": "gpu"},
      "pool_backing": "elastic"
    }
  }
}
```

`{"device":"gpu"}` requires that named device's physical allocation domain.
`{"host_numa":0}` requires initial allocation on OS NUMA node 0. The object
contains exactly one selector. Omitting it retains automatic HAL placement.
The same choice applies to fixed storage and elastic physical slabs, including
slabs acquired after trimming or model reactivation. Unsupported placement or
access fails pool creation; it never silently selects another domain. A device
used only as a backing owner may be outside the pool's consumer list, but must
be declared in the group. Device selectors require a single physical owner;
host NUMA placement requires the backend to support the requested OS node.

Placement selects the native allocation domain, not permanent physical
residency against OS eviction. For AMD GPUs, `placement.device` requires the
device's LOCAL domain. On Linux APUs, this means reserved GPU memory and succeeds
only when KFD admits that route. Shared SYSTEM memory allocated through the GPU
can satisfy automatic placement but does not satisfy this explicit LOCAL request.
CPU mapping, cache transitions, atomic access, and sharing remain separate
contracts qualified for the complete consumer set. In particular, host-visible
GPU memory does not imply CPU-cached memory or direct file-I/O support.

Required host-NUMA placement also depends on the backing mode. KFD can register
host-owned pages at fixed GPU addresses under process lifetime. Elastic GPU
aliases need a foreign DMA-BUF import route that discovery excludes because it
charges the same payload twice against the native memory budget. That elastic
combination fails pool creation; it does not substitute GPU-owned SYSTEM memory.

One pool can list several devices, such as `"devices": ["gpu", "npu"]`. The HAL
qualifies storage access for the complete consumer set during startup; an
unsupported combination fails before loading model payloads. Models selecting
that pool share its backing, admission allowance, and reclamation policy. Pool
sharing does not insert copies or make unsupported memory coherency implicit.

Several pools may list the same device. For example, separate `interactive` and
`background` pools can have different limits and idle timeouts. Their names
express policy ownership, not physical-memory equivalence or execution priority.
Limits are independent: the server does not infer an aggregate RAM cap from
hardware identity. The operator sizes their combined allowances to fit the
machine. Conversely, a discrete GPU pool does not budget another GPU's VRAM or
host RAM. No extra per-device pools are created behind this configuration.

Device URI query parameters request immutable runtime services. `feedback`
provisions device-to-host diagnostics used by authored kernel assertions and
other UBSAN-style reports. `asan` and `tsan` provision address- and race-
sanitizer state and start the report transport each requires. They are bare,
additive parameters. `--device=amdgpu:?feedback` configures the default AMDGPU
device, while `--device=amdgpu://0?feedback&asan` configures
a specific device. `?feedback&asan&tsan` is also valid; explicit `feedback`
adds general authored diagnostics to the sanitizer services. A backend that
cannot provide a requested service rejects the URI during device creation.

Models on one device share its provisioned queues, independently of their pools. Each
loaded model holds a queue assignment until its last execution reference
retires. New assignments choose the fewest live model executions in each
selected queue family; removing and reloading a model reuses the available
inventory. A single-queue device is supported, and models share queues when
there are more live executions than queues.

AMDGPU devices provision four compute queues by default, so independent models
need no extra queue configuration:

```sh
loom-serve --device=amdgpu://0 \
  --model=models/chat/model.json --model=models/speech/model.json
```

With this shortcut, both model documents select the same device and pool.
The provisioned queue count remains fixed for the device lifetime;
an explicit URI parameter such as `?queue_count=1` overrides the default.
Separate queues allow independent submissions to overlap, but
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

Discovery does not prove that a container or agent sandbox can open a device.
The [device-access checklist](../getting-started/acquiring-loom.md#check-device-access-inside-a-sandbox)
separates compiled drivers, discovered hardware, device-node access, and an
empty server's successful startup from model execution.

Elastic backing supports shared physical-memory budgets and reclamation of idle,
unpinned parameters on task and AMDGPU. It requires virtual reservations,
mapping-source allocations, and bind/unbind operations from the selected device.
Task uses host virtual memory where available; Vulkan requires suitable
sparse-memory capabilities. Fixed backing eagerly retains storage and does not
support parameter deactivation or a serving physical-memory budget.

Elastic command workspace preserves its virtual ranges across model changes and
physical trim. Logical queue allocations reuse those ranges; physical backing
shares the model-storage pool rather than a separate scratch budget. Idle
workspace is reclaimed before reloadable model data under pressure.

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
