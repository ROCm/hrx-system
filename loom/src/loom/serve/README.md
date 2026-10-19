# Loom serving components

The serving implementation separates shared execution facilities, transport,
model services and application composition. Numerical model programs are
runtime inputs, not native server dependencies.

The [model-package authoring guide](../../../docs/src/serving/model-programs.md)
traces source preparation, execution and final output. This page maps the
native owners that implement those contracts.

The shared foundation consists of:

- [`runtime/host`](runtime/host/host.h): one application-owned task executor,
  compiler task queue and NUMA-local proactor pool. Models borrow these services
  and retire their own work before releasing the host. Wake notifications and
  periodic timers share the same callback-lifetime contract.
- [`telemetry`](telemetry/stream.h): fixed-size records, cumulative counters and
  bounded independent-reader rings. Publication does not format, allocate,
  perform I/O or wait for consumers. Slow readers report gaps without delaying
  inference; producer-specific event schemas live with their owners.
- [`runtime/execution`](runtime/execution/execution.h): model-local work and
  feedback timelines, direct dispatch, queue-ordered transient workspace, and
  final buffer retirement. A fixed [queue inventory](runtime/execution/queues.h)
  assigns the least borrowed provisioned queues at model creation. Final
  execution release returns the binding, including retained result and VM
  references. Sharing a workspace does not serialize model timelines; owners
  distinguish completion from release of accepted consumers.
- [`runtime/program`](runtime/program/jit.h): source catalogs, task-based JIT,
  command materialization, and the native capabilities available to source VM
  programs. Compiled stages retain their code independently of their catalog.
- [`runtime/storage`](runtime/storage/memory.h): retained-memory reservations,
  physical slab backing, logical blocks, and queue-ordered relocation/snapshots.
  This layer owns storage, not text-session admission or cache policy.
- [`runtime/parameter`](runtime/parameter/weights.h): checkpoint plans,
  streaming in-place preparation, activation, pinning, and residency pressure.
  Parameter-specific telemetry stays here rather than in the shared event ring.
- [`runtime/device`](runtime/device/device.h): one application device group and
  the execution, storage, and residency resources of each member. Model
  services borrow these owners and retire before releasing the group.
- [`transport/http`](transport/http/server.h): bounded request framing,
  connection claims and response credit over shared proactors. Independent
  TCP/Unix listeners feed the same connection owner. Stopping a listener does
  not stop the requests it admitted; this layer has no model dependencies.
- [`transport/websocket`](transport/websocket/README.md): in-place HTTP
  upgrades, bounded message assembly, and caller-owned message receipts. An
  accepted CPU borrower can retain a payload while later controls advance;
  connection ownership joins both the application and network retirement.
- [`service`](service/service.h): the named native and HTTP model service contract,
  including bounded task turns, asynchronous retirement, client capabilities
  and copied inspection state. It consumes transport claims without owning
  socket listeners or process lifetime.
- [`service/finite`](service/finite/model.h): source-defined request cohorts
  returning compact vectors. Embedding and decision protocols share this
  preparation, batching, residency and retirement machinery; their separate
  packages own wire input and output, not learned math.
- [`service/image`](service/image/model.h): staged image invocations with
  source-owned preparation and submission control. The service owns bounded
  admission, asynchronous completion and output encoding; slow clients retain
  copied responses independently of model buffers and parameter pins.
- [`media/audio_decoder`](media/audio_decoder.h): optional in-memory WAV, MP3,
  and FLAC conversion with independent codec and resampler state. The caller
  owns workers, decoded-frame credit, and the encoded bytes' lifetime.
- [`service/recognition`](service/recognition/README.md): continuing PCM input,
  source-defined numerical batches, revisable transcript segments, and typed
  activity observations. Its [service](service/recognition/service.h) shares one
  request ledger between native callers and finite HTTP uploads. Bounded
  decoding runs on the shared CPU executor. [Stream handles](service/recognition/stream.h) retain
  final event packets, documents, and failures independently of model removal;
  input rings and numerical slots retire as soon as accepted work permits.
- [`service/speech`](service/speech/README.md): incremental waveform output with
  separate conditioning and request state, bounded PCM delivery, and explicit
  work/feedback retirement on EOF, disconnect and shutdown.
- [`service/text`](service/text/README.md): retained sessions over one prepared
  model, with bounded admission, mixed prompt/decode epochs and shared caches.
  Source defines model stages and chat policy; native owners retain claims and
  storage through actual CPU/device retirement.
- [`server/http`](server/http/logs.h): endpoint projections for telemetry and
  raw tokenization. Log readers keep independent cursors; tokenizer requests
  use independent CPU states without advancing a model's execution timeline.
- [`server`](server/README.md): deployment documents, stable registration
  lifetimes, modality construction and bounded routing turns. This composition
  layer owns model admission and removal without owning numerical programs.

Each component keeps its unit tests beside its implementation. Native headers
describe ownership contracts; they are implementation interfaces rather than
a separately versioned embedding SDK.

Live execution replacement separates cached code from selection policy.
[`variant_pool`](runtime/program/variant_pool.h) owns immutable choices and
terminal compiler diagnostics; [`execution_reconfiguration`](service/execution_reconfiguration.h)
captures provider locators, joins accepted compilation and retains the service's
preferred choice. Each modality supplies the safe selection boundary and binds
candidate execution to its existing resources. Image selection replaces all
command parts together after the whole image and encoder retire, while finite
services finish their cohort and text finishes its epoch. Preparation/control,
parameter placement and retained state remain unchanged. The
[live development workflow](../../../docs/src/serving/workflows/live-model-development.md)
traces upload, comparison and rollback over the common HTTP endpoints.

The [source fixtures](fixture/README.md) contain deterministic model documents,
VM programs, device kernels and tokenizers. They exercise these same production
paths without learned checkpoints; they are runtime inputs rather than linked
server dependencies. Shared test clients and native harnesses live in
[`testing/`](testing/README.md), separately from production libraries.

Physical CPU, AMDGPU, Vulkan and XDNA test registrations live with target
tooling. Source-model integration suites are compiled once beside their service
owners; provider tests link those suites without acquiring private headers.
Execution facilities have no transport, service, or learned-model dependency.
