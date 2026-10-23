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
`--device=legacy-amdgpu:?feedback` uses the default AMDGPU device and provisions the
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

The [native activity caller](activity_test.cc) drives activity detection and
transcription through the registry without an HTTP connection. One application
owns the worker pool and device group across the suite. The CPU provider runs
it with one and eight workers; the AMDGPU provider places activity on CPU and
transcription on GPU:

```sh
iree-bazel-test --config=asan \
  //loom/src/loom/tooling/target/cpu/serve:activity_test \
  //loom/src/loom/tooling/target/cpu/serve:activity_parallel_test \
  //loom/src/loom/tooling/target/amdgpu/serve:activity_test
```

The fixture checks independent histories as requests join and finish, a held
reader while peers complete, cancellation after task acceptance, and output
retained across model replacement. A second path feeds activity-selected audio
to the recognizer through a bounded pre-roll ring, checks successive utterances
and EOF, and verifies that silence creates no recognition request. Physical
entry points and their runtime requirements belong to the tooling providers.

The same executable accepts learned model documents with `--fixture=false`,
`--model`, and optional `--transcription_model`. `--input` supplies mono F32LE
audio at `--sample_rate`; `--reference` supplies independent per-hop F32LE
activity probabilities. Model documents retain their explicit device indices.
Composition qualification supplies `--pre_roll_frames`, `--hangover_frames`,
`--retained_frames`, `--expected_begin`, `--expected_end`, and `--expected_text`.
These are test workload inputs. They do not establish conversation defaults or
derive a source's maximum lookahead from its input credit.
