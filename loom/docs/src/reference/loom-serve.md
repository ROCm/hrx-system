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

The [serving walkthrough](../workflows/serve-models.md) reaches a first response
without weights. [Model documents](serving-models.md) specify programs, assets,
placement and per-model capacities.

## Process options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--device=URI` | Required, repeatable | HAL device URI; encounter order defines device indices. |
| `--model=path` | None, repeatable | One JSON deployment document per value; omission starts empty. |
| `--listen=address` | `127.0.0.1:8080`, repeatable | Numeric IPv4 `host:port`, IPv6 `[host]:port`, or `unix:path`. |
| `--workers=N` | `8` | Shared CPU workers for JIT, model scheduling and media work. |
| `--model_capacity=N` | `32` | Registered names, including drained, failed and removing entries. |
| `--connections=N` | `64` | Simultaneous admitted HTTP connections across all listeners. |
| `--held_requests=N` | `-1` | Closed-gate holding credit; `-1` selects half the connection limit, `0` rejects rather than holds. |
| `--request_body_bytes=N` | `8388608` | Maximum HTTP body bytes per connection. |
| `--heartbeat_ms=N` | `1000` | Per-model stderr JSONL heartbeat interval; `0` disables periodic reporting, not cumulative counters. |
| `--pool_backing=elastic\|fixed` | `elastic` | Demand-committed retained slabs or eagerly backed retained storage. |
| `--slab_bytes=N` | `2097152` | Physical slab size in bytes; `0` uses the allocator recommendation. |
| `--memory_bytes=N` | `0` | Serving-managed physical parameter/state limit; `0` imposes no serving limit. |

Model, connection and body capacities must be positive. Holding credit must
remain below connection capacity. Heartbeat intervals, slab sizes and memory
limits are nonnegative. A nonzero memory limit requires elastic backing and
exactly one device: independent device pools cannot enforce a shared byte limit.
It excludes invocation workspace, host allocations and driver overhead; it is
not a whole-process or whole-device memory cap.

`--help` shows serving options; `--help=all` also shows linked runtime options.
`--agents_md` prints compact integration guidance without initializing devices.
Unexpected positional arguments fail instead of being interpreted as models.

## Devices and listeners

Repeated devices form one ordered group. Each model selects one group member
through its top-level `device` index; omission is accepted only for a singleton
group. A URI is resolved by its linked HAL driver. The server neither chooses
an implicit device nor falls back to a different backend.

The model's native source must compile for that device. `task` uses native x86
CPU execution. AMDGPU and Vulkan require the corresponding compiler target,
HAL driver and supported hardware. XDNA packages use resident direct-dispatch
programs rather than assuming a GPU command program also runs on the NPU.
Device-group membership alone does not distribute a model or share its memory
between members.

Fixed backing is suitable for devices without virtual-memory support, including
the task and Vulkan paths. Elastic backing requires the device's virtual-memory
contract. Unsupported resource requirements fail at preparation instead of
silently changing memory policy.

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
