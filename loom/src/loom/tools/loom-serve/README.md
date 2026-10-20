# loom-serve

`loom-serve` runs source-defined text, image, embedding, decision and speech
models over HTTP. One process owns its listeners, shared CPU workers and a device group;
models can be registered, drained and removed independently. Numerical programs
and parameter files are external inputs. Adding a compatible model package does
not require rebuilding the server.

The website owns the [installed-user walkthrough](../../../../docs/src/serving/workflows/serve-models.md),
[command reference](../../../../docs/src/serving/index.md) and
[model-document format](../../../../docs/src/serving/model-documents.md).
The [speech workflow](../../../../docs/src/serving/workflows/stream-speech.md)
covers incremental PCM, independent client progress and source-defined batching.
This page describes the executable's source ownership and repository entry points.

## Start an empty server

```sh
iree-bazel-run //loom/src/loom/tools/loom-serve -- \
  --device=task --pool_backing=fixed --listen=127.0.0.1:8080
```

The installed command is `loom-serve` with the same arguments and no subcommand.
At least one explicit HAL device URI is required. Repeating `--device` builds
one ordered group; model documents select a top-level `device` index when the
group has multiple members. A URI is interpreted by its linked HAL driver,
without an implicit local device or fallback. Fixed backing needs no virtual
reservation. Elastic backing demand-commits retained slabs through the HAL
allocator-level virtual-memory contract. Task and Vulkan have device-group VMM
support, but serving's retained pool does not yet consume that interface; use
fixed backing for those drivers. The [command reference](../../../../docs/src/serving/index.md#devices-and-listeners)
describes the backend and serving requirements separately.

Generic URI parameters request immutable device runtime services. For example,
`--device=amdgpu:?feedback` uses the default AMDGPU device and provisions the
diagnostic channel used by authored kernel assertions. The parameters are
additive: `?feedback&asan` and `?feedback&asan&tsan` are both valid. `asan` and
`tsan` provision address- and race-sanitizer state and the report transport each
requires; explicit `feedback` also enables general authored diagnostics.
Unsupported services fail during device creation. Address-sanitized devices
require fixed backing.

`/healthz` and `/readyz` report server availability, independently of model
preparation. `/v1/status` reports each registration's state and diagnostics;
`/v1/models` lists prepared deployments and their client capabilities. Empty
startup leaves these administration and observation endpoints available.

## Supply model documents

```sh
loom-serve --device=task --pool_backing=fixed \
  --model=/srv/models/vectors/model.json \
  --model=/srv/models/scout/model.json
```

Each JSON document contains a deployment `name` and a `definition` selecting
the modality, preparation/control programs, source providers, asset references
and capacities. Program roles and providers accept both `.loom` and `.loombc`.
The complete source contract is modality-specific; an arbitrary checkpoint
alone is not a runnable model package.

The [source fixtures](../../serve/fixture/README.md) exercise these modalities
without a model download. They use this same executable, JIT and HTTP stack;
their independently checkable results are useful when building a client or
porting a model package.

The same JSON object is accepted by `POST /v1/models/register`. File-document
references resolve from the document directory. JSON-only HTTP registration
requires an explicit absolute `base_directory` when references are relative.
Multipart registration carries source and metadata files in a logical package;
parameter checkpoints remain server-local. The
[live development workflow](../../../../docs/src/serving/workflows/live-model-development.md)
connects uploads to native variant selection and retained-checkpoint comparisons.
Omitted `admission` defaults to `open`; `drained` retains metadata without loading
source or weights. Registration returns HTTP 202 for accepted intent; status reports
when preparation completes. Multiple startup models may be supplied through
repeated flags or an ordinary newline-separated `--flagfile`.

Repeated `--listen` values bind IPv4, IPv6 or `unix:path` endpoints to the same
logical server. Connection and request-body limits are shared across listeners.
Drain/undrain controls model admission; unregister supports graceful completion
or abort, always retaining accepted CPU/device work until its actual retirement.
HTTP has no built-in authentication. Clients and source packages must be
trusted; deployment-owned tunnels and isolation supply any required boundary.

`--help` shows serving flags and `--help=all` includes linked runtime controls.
`--agents_md` describes the model-registration workflow without initializing
devices. `/metrics`, `/logz` and `/logz/stream` expose counters and telemetry;
slow observers do not retain model resources or backpressure record producers.

## Configure execution interference

`X-Loom-QoS: realtime` gives a ready inference request priority over background
work; requests without the header are background. By default, realtime demand
closes new background admission across the shared host and every device.
Accepted work drains to its existing completion boundary before conflicting
work can start.

Repeated `--execution_scope=exclude:host,0` or
`--execution_scope=overlap:host,0` flags describe interference using `host` and
zero-based `--device` ordinals. Explicit scopes must cover every resource.
Repeated `--execution_limit=0:1` or `--execution_limit=host:8` flags independently
bound accepted units; their counts are not hardware queues or core counts.
Omitting scopes retains default exclusion even when limits are configured.
These options also work in flagfiles. The
[execution policy reference](../../../../docs/src/serving/index.md#request-priority-and-interference)
explains cross-device scopes, task-device claims, and explicit overlap.

## Trace live work

A Tracy-enabled build exposes CPU work from serving, JIT compilation, task
workers, and the runtime. The
[runtime tracing guide](../../../../../runtime/src/iree/base/tracing/README.md)
owns build and tool setup. The
[serving observation workflow](../../../../docs/src/serving/workflows/observe-model-server.md#inspect-host-work-in-tracy)
covers live profiler attachment, SSH forwarding, direct capture, automatic
capture across server launches, and the meaning of serving annotations.
The current provider accepts one collector per process; capture memory is
separate from the bounded operational telemetry rings.

## Implementation and tests

[Main](main.c) owns process startup and teardown. [Options](options.h) constructs
shared host/device resources from flags; [signals](signals.h) owns interrupt and
termination subscriptions. The [serving layers](../../serve/README.md) accept
explicit options and do not depend on process-global flags.

Generic command smokes and signal tests are adjacent. The external-package
command test also lives here, with its x86/task-HAL execution registered under
the CPU tooling provider. It exercises runtime source loading, text/bytecode
mixing, multiple devices, first output and dynamic model retirement through the
same executable.
