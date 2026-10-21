# AMD HAL ownership

The AMD driver separates discovery and memory from execution engines:

```text
driver.{c,h}                     Scheme selection and device creation.
registration/driver_module.*     Registry entries sharing one driver identity.
fabric/
  fabric.{c,h}                   Native instance and passive discovery.
  memory.{c,h}                   Native backing and optional host mappings.
  buffer.{c,h}                   HAL views over prepared backing and bindings.
  file_transfer.{c,h}            Async file I/O through host execution bindings.
  slab_provider.{c,h}            Backing acquisition through the slab interface.
  memory_backend.{c,h}           Complete pool qualification and captured facts.
  memory_transition.{c,h}        Native visibility recipes translated to HAL.
xdna/                           Native NPU contexts, programs, and execution.
gpu/                            Native GPU images, PM4/AQL execution, and retirement.
```

`fabric/` depends on libamdf core and HAL memory interfaces. It has no execution
engine dependency. XDNA owns its extension, execution context, queue resources,
private firmware scope, and firmware address interpretation. GPU owns its
extension, native context, command storage, and completion resources, with the
same dependency direction.

## Discovery lifetime and device selection

The registry retains one lazily created AMD driver per canonical factory ID.
Several URI schemes can name that identity without creating separate native
instances. The original scheme reaches enumeration and device creation, so each
family has its own ordinal namespace. Independent registries and direct driver
construction provide separate instances. The registry uses PROCESS native
lifetime to support Linux KFD host registration. Its primary VM can survive
fabric destruction and prevent a fresh acquisition, so applications retain one
registry/driver for devices using that native process context. Direct driver
construction with INSTANCE bounds native reclamation to the fabric at the cost
of services unsupported by secondary KFD contexts, including host registration.

The `xdna`, `rdna`, `cdna`, and `amdgpu2` schemes share that driver identity. `xdna`
selects NPUs, `rdna` selects RDNA GPUs, `cdna` selects CDNA GPUs, and `amdgpu2` selects all GPUs supported
by this build. Both `rdna:0` and `rdna://0` select the first visible RDNA device;
`xdna:0?columns=4` configures an NPU context. The legacy `amdgpu` driver remains
independent.

Each scheme has its own dense ordinal namespace. Driver creation captures these
optional environment variables:

| Scheme | Ordered visibility list |
| --- | --- |
| `xdna` | `HAL_VISIBLE_XDNA_DEVICES` |
| `rdna` | `HAL_VISIBLE_RDNA_DEVICES` |
| `cdna` | `HAL_VISIBLE_CDNA_DEVICES` |
| `amdgpu2` | `HAL_VISIBLE_AMDGPU2_DEVICES` |

A list indexes that scheme's unmasked supported devices in native enumeration
order. Unset exposes all devices; an empty or whitespace-only value exposes
none. Comma-separated decimal ordinals select and reorder devices. Invalid,
duplicate, or out-of-range ordinals fail driver creation with the variable
named in the diagnostic. The lists are independent: filtering the unified GPU
view does not filter the RDNA view or alter libamdf topology.

For example, given native endpoints NPU A, RDNA B, RDNA C:

```sh
HAL_VISIBLE_RDNA_DEVICES=1,0 HAL_VISIBLE_AMDGPU2_DEVICES=0 iree-run-module ...
```

The process sees `xdna:0` as A, `rdna:0` as C, `rdna:1` as B, and `amdgpu2:0`
as B. All aliases share the same native fabric. Existing drivers retain their
captured selections if the environment changes later.

Enumeration returns `id/<32 hexadecimal digits>` paths for exact native identity
selection within the same visible scheme. An unsupported or hidden identity is
unavailable; identities can expire on reset, removal, or reboot. Opaque numeric
device IDs returned by enumeration name native endpoints across aliases, not
URI ordinals. Visibility configures selection and is not an access-control
boundary. Architecture-local ordinals do not claim to match another runtime's
combined GPU numbering.

GPU discovery captures architecture and usable compute families through query-only
endpoints; it creates no device, address space, queue, or event. Enumeration and
selection then use immutable endpoint-index lists. Queue admission selects an
advertised native protocol and publication mode. The default prefers PM4 on RDNA
and AQL on CDNA among supported families; explicit family selection is independent
of architecture. PM4 supports USER host-producer and KERNEL submission families.
AQL requires USER publication with both host and device producers and native
scratch service. Recorded AQL programs remain device-local: GPU materialization
writes reserved native ring slots and publishes the first header only after the
packet bodies are visible. Both protocols share memory, dependency admission,
and device progress owners. Other native services fail explicitly at construction
or operation admission.

`rdna://0?queue_count=2` provisions two queues sharing the device's execution
workers. `queue_family` selects an exact native family ordinal instead of the
architecture preference. `cdna://0?queue_count=2` provisions the same HAL queue
interface on the first CDNA endpoint. `host_numa_node` selects CPU placement
for the workers and shared proactor; GPU-local memory node IDs are not CPU
NUMA node IDs.

`completion_spin_ns` bounds active completion observation after actual progress
(default 100,000 ns). While actively waiting, the existing retirement worker
reads coherent USER queue completion words; a change resumes authoritative
source drains. At expiry it parks on the native event, with no periodic wakeups.
Setting zero parks immediately. KERNEL queues continue to use native events and
status queries. This trades a bounded CPU tail after the last completion for
lower latency between dependent operations; it does not alter process power
policy, add a thread, or change submission and retirement ordering.

Each materialized device retains the fabric independently. Releasing the
registry and driver leaves those devices operational. The fabric owns the native
instance and its discovery snapshot; engine contexts own their native children.

## Shader diagnostics and sanitizers

The generic URI parameters request runtime services at device creation:

```text
rdna://0?feedback
rdna://0?asan&asan_report_policy=fail-device
rdna://0?tsan&tsan_report_policy=report-only
```

Programmatic callers set the corresponding `runtime_features` in
`iree_hal_device_create_params_t`. ASAN and TSAN also provision feedback.
Executables use the existing compiler-generated configuration globals and report
layouts; requesting a runtime service does not instrument an ordinary HSACO.
Loading an executable requiring an unprovisioned service fails explicitly.

Reports reach the supplied `event_sink`, with executable and source-site identity
retained through delivery. The default sink discards events; applications can
select `iree_hal_device_event_sink_stderr()` or provide their own callback. The
existing completion worker drains the shared channel. There is no additional
feedback thread. Assertion reports preserve the existing diagnostic behavior;
they do not automatically fail submissions.

ASAN defaults to `report-only`; TSAN defaults to `fail-device`. Both policies are
selectable through the URI parameters above. A fail-device report propagates
`ABORTED` for ASAN or `DATA_LOSS` for TSAN and rejects subsequent work. Failure
notification does not cancel already-issued native work or certify its resource
retirement. Pools and caller-owned backing must still outlive their users.

ASAN maintains sparse address shadows, allocation redzones, and freed-backing
quarantine. TSAN maintains per-queue workgroup shadows and immutable executable
placements, including when a recorded command buffer runs on different queues.
Shadow-slot reuse inserts execution joins before prior users can be overwritten.
Sanitizer shadows use local storage when available and qualified system storage
on devices without a separate local heap. Ordinary devices allocate none of this
sanitizer state.

The shared producer requires native shader-triggered events. The current Windows
provider exposes submission completion but lacks that shader event producer, so
feedback/ASAN/TSAN device creation is rejected there. Enabling those services
requires implementing and qualifying the native shader producer; ordinary
Windows PM4 execution does not depend on it.

## GPU queue profiling

The native GPU backend supports queue events, dispatch events, and executable
metadata through the ordinary device profiling API. `LIGHTWEIGHT_STATISTICS`
selects host queue events. Begin and end require an idle device and external
serialization with submission, as specified by the HAL API:

```c
const iree_hal_device_profiling_options_t options = {
    .data_families = IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS,
    .sink = sink,
};
IREE_RETURN_IF_ERROR(iree_hal_device_profiling_begin(device, &options));
// Submit work and wait for its completion.
IREE_RETURN_IF_ERROR(iree_hal_device_profiling_end(device));
```

Each accepted public operation contributes one host-clock record with its queue,
submission identity, payload size, and actual dependency admission strategy.
Repeated command-buffer executions carry the same recording identity and distinct
submission identities. Batched transfers use a `TRANSFER` record; internal file
staging copies and host-call return barriers contribute no additional public
events. Readiness denotes admission to the executor, including native entry
barriers for host calls and mappings; it is not a GPU execution timestamp or a
host callback start timestamp.

Each queue retains up to 4096 records between flushes. Full streams count dropped
records and report `TRUNCATED` chunks. Explicit flush can snapshot completed
records while submissions continue; failed sink writes preserve their records
for retry. Sink callbacks run only during begin, flush, and end, outside queue
and completion locks. End releases the session even when the sink fails.

Disabled sessions allocate no event storage or per-operation profile metadata.
Enabled captures place metadata in the existing submission allocation and use
bounded stream storage. Hardware counters and capture filters are rejected
explicitly by this producer.

### Dispatch timing

Add `IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS` and
`IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA` to the requested families
for attributed device intervals. Direct dispatches supply attribution at
submission. Recorded command buffers must retain it at recording time with
`IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA` or
`IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA`.

Each active dispatch uses two native timestamp words in invocation-owned
argument storage. The normal GPU fixup pass patches their addresses into the
resident program, so concurrent and pipelined invocations have independent
samples without a host walk over recorded commands. Retirement copies completed
samples into the bounded host stream before returning the argument storage.
Export retains host reflection metadata without retaining executable code or
model buffers until a later sink flush.

Intervals join prior compute before the start sample and the dispatch's compute
before the end sample. They measure isolated dispatch completion, including
native issue and join costs; enabling them changes dispatch overlap. Raw ticks
use the exact native family's timestamp frequency. They have no inferred host
epoch and cannot establish host-request-to-first-workgroup latency.

Profile-capable recordings retain optional sample packets and fixups even when
capture is inactive; those packets become NOPs for an unprofiled invocation.
Ordinary recordings retain neither the sample packets nor their fixups. Queue
events remain the lighter option when device intervals are unnecessary.

## Memory construction

An engine captures its native API, instance, device, storage scopes, family
ordinals, address interfaces, minimum native permissions, and private host-access
needs in `iree_hal_amd_fabric_memory_backend_options_t`. The backend borrows these
facts and existing progress services from the device. Each storage choice names
its native scope, locality properties, and any enforced physical affinity. Pool
construction qualifies every participating family together against each eligible
scope and captures the selected placement, access set, binding layout, and
visibility table. Allocation and retirement use the selected storage owner's
existing progress services.

Placement is independent of the access-family set and its order. A pool can
require backing on one physical GPU while exposing that allocation only to a
peer's families or CPU families. Construction qualifies any private native owner
attachment without publishing extra HAL scopes or bindings.
Required placement selects the corresponding physical node in the sealed group;
an unavailable route yields no candidate. Preferred placement can select another
qualified route and reports the achieved placement. Automatic selection prefers
qualified device-local backing when the requested access and host preferences
permit it. SYSTEM scopes make no enforced host NUMA placement promise.

One allocation can publish several address interfaces for several native
devices. A host mapping is acquired only when public or engine-private access
requires it. XDNA currently requires a private read/write host view for its
transfer implementation. Its native memory also requires read/write device
permissions even when the HAL contract grants narrower logical usage.

Pools remain explicit. A sealed device group dominates its pools and their
borrowed native owners. Slabs own backing; buffer views borrow backing and
binding tables through the existing release-callback contract. The slab cache,
suballocation layers, and explicit trim policy retain their existing ownership.
No buffer adds a fabric reference or discovers native addresses at submission.

Owned allocation, host registration and native import/export classes are
qualified together against the selected scope. Imports preserve caller-owned
storage and its physical placement; owned-backing placement does not relocate
an imported allocation. Unqualified native transports and uncached device-access
requirements yield no candidate. Queue visibility uses libamdf's exact
queue-family recipes; program-side visibility remains unqualified by that native
query.

## File transfers

XDNA imports host-memory files and platform file handles through the shared HAL
file implementations. Platform handles used by queues are opened with
`IREE_IO_FILE_MODE_ASYNC`. Queue reads and writes use the existing public API:

```c
IREE_RETURN_IF_ERROR(iree_hal_queue_read(
    queue, prerequisites, loaded, file, file_offset, buffer, buffer_offset,
    length, barriers, IREE_HAL_READ_FLAG_NONE));
// Device work waits for loaded and signals computed.
IREE_RETURN_IF_ERROR(iree_hal_queue_write(
    queue, computed, saved, buffer, buffer_offset, file, file_offset, length,
    barriers, IREE_HAL_WRITE_FLAG_NONE));
```

The fabric transfer owns the exact byte range and continues partial native I/O.
The execution engine owns semaphore dependencies, before/after barriers, and
terminal signaling. A failed transfer fails its dependent signals; it can leave
partial output and does not fail independent queue work. Write completion means
the operating system has consumed the bytes, without promising disk durability.

XDNA resolves its private host execution binding after dependencies complete,
so file operations can target queue-allocated buffers that have not committed at
capture time. Public host mapping permission is unnecessary. The existing host
service acquires source bytes before writes and publishes destination bytes
after reads. Disk I/O runs on the imported file's proactor; its completion
returns to the queue owner before final publication. Neither the proactor nor
the host service blocks waiting for disk I/O. Files on another proactor use the
same completion path, with each progress owner kept live by its caller.

Memory files with device-visible storage already use HAL queue copies. Other
memory files copy on the host service. This implementation allocates no staging
memory and creates no additional threads. Device-only backing without a host
execution binding needs an engine-specific copy/staging route and is rejected by
the direct host path. File and buffer references remain retained through the
operation; callers still own backing retirement and the device lifetime.

### GPU file backing

GPU file transfers use a native host view directly when that exact view
supports file I/O. CPU visibility alone does not establish file eligibility,
for either buffered or direct files. Other ranges use the device's shared SDMA
staging pool, including host-visible UMA allocations and device-local VRAM.
Virtual buffers retain the routing and visibility requirements of each physical
source range.

Staging shares a 16 MiB backing budget per device across file proactors,
growth, and trim, rounded to native allocation granularity. The staging
allocation needs GPU addressing and file access, without virtual aliases; the
destination independently retains its own mapping-source contract. This permits
ordinary model backing with a bounded transfer allocation instead of
registering the full model. Each staged ingress adds one payload read and
write. Explicit file-capable pools avoid that copy when their placement and
native admission costs suit the application.

The device qualifies file-capable backing before creating its SDMA queue.
Without that capability, compute remains available and file operations requiring
staging report `UNAVAILABLE`. Linux process-lifetime registration can attach
ordinary host pages; instance-lifetime native connections require the optional
host exporter for this backing. Native file visibility uses its own pair
queries and does not require CPU-coherent GPU accesses.

### Registered physical backing

A pool can request `IREE_HAL_BUFFER_INTERFACE_REGISTERED_IO` from the family
that will perform file transfers, alongside `MAPPING_SOURCE` when the storage
will back virtual buffers. Construction qualifies native `HOST_FILE_IO` storage
and the family's existing file proactor. The proactor must support concurrent
buffer registration so cold page pinning runs on the allocation caller while
unrelated I/O continues. On Linux, this requires actual `CLONE_BUFFERS` range
replacement support; the proactor probes it before publication. Missing native
or proactor support makes this optional pool profile unavailable at construction.

Physical slabs retain their prepared host view and file registration through
reuse. Virtual aliases preserve the original source binding, and file operations
consume those retained fixed-buffer spans without pinning the payload again.
Trim releases the registration before the native storage. Registration requires
the file to use the captured proactor; a different I/O owner fails admission
instead of silently registering another view. Physical allocation, registration,
and I/O failures propagate after the pool has been selected.

Linux exported-host backing also requires the native provider's huge-page
preparation and `/dev/udmabuf` access, and fixed-buffer registration consumes the
process's locked-memory allowance. These are optional deployment prerequisites,
not requirements for ordinary device backing. The native memory documentation
describes the export limits and supported alias contracts. Registered reads
avoid repeated page lookup and pinning; buffered files still use the operating
system's page cache. Direct file I/O additionally requires the file's advertised
offset, length, and buffer alignment.
