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

Each component keeps its unit tests beside its implementation. Native headers
describe ownership contracts; they are implementation interfaces rather than
a separately versioned embedding SDK.
