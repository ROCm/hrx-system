# loom-serve

`loom-serve` runs source-defined text, image, embedding and decision models over
HTTP. One process owns its listeners, shared CPU workers and a device group;
models can be registered, drained and removed independently. Numerical programs
and parameter files are external inputs. Adding a compatible model package does
not require rebuilding the server.

The website owns the [installed-user walkthrough](../../../../docs/src/workflows/serve-models.md),
[command reference](../../../../docs/src/reference/loom-serve.md) and
[model-document format](../../../../docs/src/reference/serving-models.md).
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
without an implicit local device or fallback. Fixed backing works without
virtual-memory support; elastic backing demand-commits retained slabs on
drivers that provide it.

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

The [source fixtures](../../serve/fixture/README.md) exercise all four modalities
without a model download. They use this same executable, JIT and HTTP stack;
their independently checkable results are useful when building a client or
porting a model package.

The same JSON object is accepted by `POST /v1/models/register`. File-document
references resolve from the document directory. HTTP registration requires an
explicit absolute `base_directory` when references are relative. Omitted
`admission` defaults to `open`; `drained` retains metadata without loading source
or weights. Registration returns HTTP 202 for accepted intent; status reports
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
