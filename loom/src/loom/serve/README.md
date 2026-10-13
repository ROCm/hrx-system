# Loom serving components

The serving implementation separates shared execution facilities, transport,
model services and application composition. Numerical model programs are
runtime inputs, not native server dependencies.

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
  final buffer retirement. Sharing a workspace does not serialize model
  timelines; owners distinguish completion from release of accepted consumers.
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
- [`service`](service/service.h): the named HTTP-facing model service contract,
  including bounded task turns, asynchronous retirement, client capabilities
  and copied inspection state. It consumes transport claims without owning
  socket listeners or process lifetime.
- [`service/finite`](service/finite/model.h): source-defined request cohorts
  returning compact vectors. Embedding and decision protocols share this
  preparation, batching, residency and retirement machinery; their separate
  packages own wire input and output, not learned math.
- [`server/http`](server/http/logs.h): endpoint projections for telemetry and
  raw tokenization. Log readers keep independent cursors; tokenizer requests
  use independent CPU states without advancing a model's execution timeline.

Each component keeps its unit tests beside its implementation. Native headers
describe ownership contracts; they are implementation interfaces rather than
a separately versioned embedding SDK.

Physical AMDGPU test registrations live with target tooling; their native
test sources and source-program inputs remain beside the component they
exercise. Execution facilities have no transport, service, or learned-model
dependency.
