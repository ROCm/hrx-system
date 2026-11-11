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

The `xdna`, `rdna`, `cdna`, and `amdgpu` schemes share that driver identity. `xdna`
selects NPUs, `rdna` selects RDNA GPUs, `cdna` selects CDNA GPUs, and `amdgpu`
selects all GPUs supported by this build. Both `rdna:0` and `rdna://0` select the
first visible RDNA device;
`xdna:0?columns=4` configures an NPU context.

Each scheme has its own dense ordinal namespace. Driver creation captures these
optional environment variables:

| Scheme | Ordered visibility list |
| --- | --- |
| `xdna` | `HAL_VISIBLE_XDNA_DEVICES` |
| `rdna` | `HAL_VISIBLE_RDNA_DEVICES` |
| `cdna` | `HAL_VISIBLE_CDNA_DEVICES` |
| `amdgpu` | `HAL_VISIBLE_AMDGPU_DEVICES` |

A list indexes that scheme's unmasked supported devices in native enumeration
order. Unset exposes all devices; an empty or whitespace-only value exposes
none. Comma-separated decimal ordinals select and reorder devices. Invalid,
duplicate, or out-of-range ordinals fail driver creation with the variable
named in the diagnostic. The lists are independent: filtering the unified GPU
view does not filter the RDNA view or alter libamdf topology.

For example, given native endpoints NPU A, RDNA B, RDNA C:

```sh
HAL_VISIBLE_RDNA_DEVICES=1,0 HAL_VISIBLE_AMDGPU_DEVICES=0 iree-run-module ...
```

The process sees `xdna:0` as A, `rdna:0` as C, `rdna:1` as B, and `amdgpu:0`
as B. All aliases share the same native fabric. Existing drivers retain their
captured selections if the environment changes later.

Enumeration returns `GPU-<16 hexadecimal digits>` when the native persistent
identifier selects one visible GPU. For example,
`amdgpu://GPU-4939e1d93d24ff77` selects that GPU independently of its ordinal.
Linux uses the same native identifier as ROCr. Device partitioning can change
the identifier. Providers without one, or with ambiguous identifiers, return
`id/<32 hexadecimal digits>` endpoint paths. These exact endpoint paths remain
accepted for every device but can expire on reset, removal, or reboot.

Both identity forms require an exact match within the same visible scheme;
unsupported or hidden identities are unavailable. Opaque numeric device IDs
returned by enumeration name native endpoints across aliases, not URI ordinals.
Visibility configures selection and is not an access-control boundary.
Architecture-local ordinals do not claim to match another runtime's combined
GPU numbering. HIP's ROCR/HIP/CUDA visibility variables are frontend policy
applied before activation, independently of these HAL scheme filters.

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

GPU devices provision four compute queues by default, sharing the device's
execution workers. `rdna://0`, `cdna://0`, and `amdgpu://0` use this default
on their selected endpoint. An explicit `queue_count` selects a different
inventory; for example, `rdna://0?queue_count=1` provisions a single compute
queue. `queue_family` selects an exact native family ordinal instead of the
architecture preference. `host_numa_node` selects CPU placement
for the workers and shared proactor; GPU-local memory node IDs are not CPU
NUMA node IDs.

`completion_spin_ns` bounds active completion observation after actual progress
(default 1,000,000,000 ns, or one second). While work remains pending, the
existing retirement worker reads coherent USER queue completion words; a change
resumes authoritative source drains. Actual completion progress renews the
budget. At expiry, or when no coherent work remains pending, the worker parks
on the native event, with no periodic wakeups. Setting zero parks immediately;
device creation with `completion_wait_flags=NONE` also disables observation
even when the URI specifies a nonzero budget.

Active observation reduces completion wake-up latency between dependent GPU
operations, at the cost of occupying a CPU core while polling. It can improve
request throughput and energy per request, but long GPU operations may consume
more CPU without completing faster. Idle devices can sleep. KERNEL queues,
including the Windows native submission path, retain native events and status
queries and do not use this budget. The setting does not alter process power
policy, add a thread, or change submission and retirement ordering.

Each materialized device retains the fabric independently. Releasing the
registry and driver leaves those devices operational. The fabric owns the native
instance and its discovery snapshot; engine contexts own their native children.

## Native GPU execution

PM4, AQL, and SDMA own their packet encodings and publication rules under
`gpu/queue/`. They share dependency admission, captured input ownership, resident
program instances, transfers, and semaphore frontiers. A native FIFO provides the
execution order used to prove reuse; HAL submissions still express their
dependencies with semaphores. Unresolved host dependencies park their captured
operation without blocking independent ready work.

Direct AQL dispatches write complete packet bodies and publish the header last.
Device-written indirect counts use the same rule: the target packet remains
INVALID until the patch dispatch publishes it. Neither producer reads or
performs read-modify-write operations on the native ring. The firmware read
index returns packet credit; separate native completion signals establish when
arguments, bindings, executable references, and other payload ownership can be
retired.

Large PM4 and AQL recordings keep packet templates, fixups, and private data in
device-local storage. Submission uploads invocation inputs, reserves bounded ring ranges,
and issues GPU fixup and materialization kernels. Materialization copies packets
into the reserved slots and publishes the first header after all packet writes
are visible. Host work scales with the number of chunks and supplied inputs,
not a replay of each recorded dispatch. The AQL default ring is 2 MiB, with at
most half the ring's slots used for a chunk's payload packets. This permits
16384 payload packets per default chunk while retaining space for dependencies
and publication. Explicit ring sizes are preserved. PM4 retains its 64 KiB
default and executes resident indirect buffers.

The host may publish multiple AQL chunks ahead while ring credit permits.
Each chunk's ordered terminal packet joins its payload before the next chunk's
materializer runs. This bounds storage and retirement without a device-wide
idle. A recording containing exactly one direct dispatch, at most 512 bytes of
packet/private data, and at most sixteen fixups instead uses a sealed host
template. Submission copies and patches that bounded template directly, avoiding
the two preparation kernels required by the resident path.

Resident instances use vector-clock frontiers to prove safe reuse across queues.
Same-FIFO reuse can pipeline behind the previous invocation; overlapping work
without that proof acquires a separate instance. Scratch growth uses libamdf's
firmware request and caller-owned backing contract, with cold allocation on the
fabric allocation service. Explicit trim releases idle scratch and cached
storage. Synchronous executable and command-buffer construction allocate on the
calling thread.

Executable loading writes directly to qualified host-mapped code when possible.
Unmapped code uses an independent cold queue in the selected compute family:
bounded confirmed `WRITE_DATA` commands upload the bytes, and instruction-cache
publication completes before loading returns. This bootstraps the resident blit
shaders without an SDMA queue or a shader that must already be resident. AQL
always uses this owner for instruction-cache publication; mapped PM4 code uses
its ordinary before-dispatch cache action. Application queue dependencies cannot
block the loader. Accepted uploads retain their target until checked retirement
or native stop, including when shutdown wakes a blocked loading caller.

File reads and writes preserve the selected queue for any GPU copy. Directly
file-accessible backing needs no staging copy. Other ranges use bounded shared
staging storage, with copies submitted through the originating queue's transfer
path and slots retained until that queue retires the copy. The staging owner
manages asynchronous file I/O and storage, and owns no execution queue.

The existing device workers own deferred admission, native completion, host
actions, and feedback. Dynamic queues join those owners and receive their own
native publication resources. A failed public semaphore is not a native
retirement proof: already-published work retains its resources until the native
completion or teardown boundary. Queue and device destruction join their native
users before returning.

### Dedicated transfer queues

Usable native SDMA families appear alongside the selected compute family in the
device queue specification. They advertise TRANSFER and HOST_CALL roles and
dynamic acquisition, with no provisioned queues. `iree_hal_queue_acquire` creates
one exact native queue when requested. Compute initialization and executable
loading do not acquire an SDMA queue; native acquisition failure is returned to
the caller.

An SDMA queue supports ordinary transfer operations, file reads and writes,
pool allocation and mapping, host calls, barriers, and timestamps when its native
clock is available. `iree_hal_queue_execute` accepts TRANSFER command buffers
recorded for that exact family, including late buffer bindings and concurrent
invocations. Private data belongs to separately leased resident instances when
causal reuse cannot be proved. Dispatch and public atomics are absent from the
family's capabilities and are rejected at the HAL boundary.

The queue passed to a file operation determines its copy engine. Compute queues
use their compute-family transfer path; SDMA queues use native copy packets.
Both use the same file, staging, mapping, and pool owners. PM4 and SDMA also share
the bounded DMA traversal that splits transfers, stages host data, and retains
download destinations. Only the packet writer differs. A pool used by both
engines includes both exact families in its access scope before backing is
acquired.

Semaphores connect transfers to compute without a device-wide join. USER
publication carries supported native waits and cache actions in the command
stream. A KERNEL carrier that provides submission-scoped cache transitions
instead waits for incoming producer completion before submission and publishes
outgoing readiness after checked native retirement. Neither case adds a polling
thread per queue.

Native command capacity is separate from host metadata progress. When command
slots are full, queue membership, trim, and completed backing acquisitions still
run under the same publisher. A prepared KERNEL command retains its bytes and
pending causal frontier until accepted or rejected; later native work cannot
overwrite it while metadata grows.

Loom's device execution inventory selects a transfer-only family when one is
available and retains its acquired queue across model activation and reload.
Devices with only a combined compute/transfer family use that family. Models
borrow the inventory, and pools are qualified against those same queue identities.

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

GPU ASAN is unsupported when the host runtime is built with CPU ThreadSanitizer:
their large virtual-address reservations compete for the same process address
space. Device creation rejects that combination before provisioning GPU sanitizer
resources. CPU ThreadSanitizer can still exercise ordinary GPU execution and GPU
TSAN, whose workgroup shadows use bounded per-queue allocations.

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
bounded stream storage. Capture filters select dispatch records by physical
device ordinal, queue ordinal, command-buffer identity, command-index range,
and executable function pattern. The session retains its own pattern bytes.
Whole-invocation exclusions avoid dispatch capture; per-command exclusions are
applied during retirement without a submission-time walk of the recording.
Hardware counters require a native counter service and are rejected explicitly.

### Dispatch timing

Add `IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS` and
`IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA` to the requested families
for attributed device intervals. Direct dispatches supply attribution at
submission. Recorded command buffers must retain it at recording time with
`IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_PROFILE_METADATA` or
`IREE_HAL_COMMAND_BUFFER_MODE_RETAIN_DISPATCH_METADATA`.

PM4 records two native timestamp words per dispatch in invocation-owned argument
storage. Sample packets join prior compute before the start and the dispatch's
compute before the end. They measure isolated dispatch completion, including
native issue and join costs; enabling them changes dispatch overlap. The normal
GPU fixup pass supplies the sample addresses. Profile-capable recordings retain
optional sample packets and fixups; the packets become NOPs for an unprofiled
invocation.

AQL enables the native queue's firmware timestamp mode for dispatch capture.
Each sampled dispatch receives an invocation-owned 64-byte native completion
signal containing its start/end timestamps. The materializer initializes and
patches these signals for resident chunks; the direct path prepares them on the
host. Resident recordings with inactive capture leave dispatch completion fields
zero and use the ordinary chunk completion signal for retirement. Recorded
per-command filters select exported samples without changing the packet layout.

Concurrent and pipelined invocations have independent samples on both protocols.
Each retiring chunk owns its sample range and copies completed timestamps into
the bounded host stream before returning argument storage. Export retains host
reflection metadata without retaining executable code or model buffers until a
later sink flush. Raw ticks use the exact native family's timestamp frequency;
they have no inferred host epoch and cannot establish
host-request-to-first-workgroup latency. Queue events remain the lighter option
when device intervals are unnecessary.

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
Required placement selects the corresponding physical GPU or host NUMA node
in the sealed group when the native profile can enforce it; an unavailable
route yields no candidate. Preferred placement can select another qualified
route and reports the achieved placement. Automatic selection favors qualified
device-local backing, including SYSTEM allocations that libamdf qualifies as
local to every consumer on unified-memory systems. Explicit host requirements
and preferences participate in that selection. Placement identifies the native
allocation domain; it does not prevent native eviction or imply cache coherence.

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
an imported allocation. Unqualified native transports yield no candidate.
`IREE_HAL_POOL_ACCESS_REQUIRE_UNCACHED` requests uncached device access through
libamdf's qualified profile; it does not change public host mapping properties.
Queue and program visibility use their exact native site recipes, captured
alongside the pool's immutable access contract.

### Atomic access and visibility

Pool construction qualifies atomic requirements separately for host accesses,
queue commands, and executable programs. Each site describes operation masks
for 32-bit and 64-bit words at device and system scope. A CPU+GPU+NPU pool can
therefore require CPU/GPU atomics while granting ordinary NPU reads and writes.
`iree_hal_pool_query_atomic_operations` and its buffer equivalent return these
captured facts without native discovery. An absent bit means the contract
makes no guarantee for that operation.

Single-site support does not establish mutual atomicity. The exact
producer/consumer pair's `producer_atomics` and `consumer_atomics` masks
describe which operations are mutually atomic in each width/scope cell.
The guarantee is not transitive: CPU/GPU A and CPU/GPU B guarantees do not
establish GPU A/GPU B atomicity. Payload visibility still requires the pair's
release/acquire actions and an execution dependency. Prepared transition keys
use the existing fixed table lookup during recording.

The `atomics.queue_zero_compute` pool requirement and
`IREE_HAL_ATOMIC_FLAG_ZERO_COMPUTE` operation flag require an implementation
that consumes no compute resources. A shader fallback cannot satisfy them.
The queue family and the target pool must both qualify the operation; the
zero-compute pool/buffer query reports their captured intersection. A wait
can still occupy its hardware FIFO, so its producer needs an independently
progressing queue. This requirement does not promise queue residency under
hardware oversubscription.

### Virtual backing contracts

A virtual buffer declares its candidate source pools before reserving
addresses. Sources contribute at the sites they support: a GPU-local pool
can back GPU-only ranges, while a shared pool can back CPU+GPU ranges in the
same virtual buffer. Each requested site needs at least one source. Each
source must serve at least one requested site.

The resulting host properties, atomic guarantees, and transition requirements
remain fixed for the virtual buffer's lifetime. Site guarantees intersect the
sources serving that site; pair guarantees combine sources serving both sites.
`iree_hal_virtual_buffer_query_pool_compatibility` checks a pool against that
contract without allocation, native calls, or an installed-mapping lookup.
A concrete scope includes every admitted site sharing its native page tables;
`ANY` requires coverage of the complete native root. Mapping rejects backing
that would weaken those guarantees before changing native page tables.

`IREE_HAL_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS` opts into the same numeric
base address at every admitted host and program binding. Zero flags permit
independent native addresses. Common addresses do not grant CPU access to
GPU-only backing. CPU visibility of a physical allocation also does not imply
that its host view can be remapped: cached host backing and CPU-visible VRAM
can have different mapping-source capabilities. Capability queries qualify
the complete source and access set before acquiring addresses or payload.

`iree_hal_queue_map` orders mapping batches through ordinary semaphore
dependencies. The AMD backend maintains an ordered source-range index per
native page-table projection for consumers such as file I/O. Updates reserve
metadata before native mutation and publish affected intervals together under
the existing query lock. They preserve split, clear, and coalesce behavior
without rebuilding the entire index. Installed physical backing remains
caller-owned; it stays live through its last use and checked mapping removal.

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
for either buffered or direct files. Other ranges use shared native staging,
including host-visible UMA allocations and device-local VRAM. GPU copies run
on the exact queue passed to `iree_hal_queue_read` or `iree_hal_queue_write`:
PM4 uses its ordinary transfer packets, and AQL uses its transfer kernels.
Staging manages backing and file progress without acquiring a GPU queue.
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

The device qualifies file-capable backing for the selected compute family.
Without that capability, file operations requiring staging report `UNAVAILABLE`.
Linux process-lifetime registration can attach
ordinary host pages; instance-lifetime native connections require the optional
host exporter for this backing. Native file visibility uses its own pair
queries and does not require CPU-coherent GPU accesses.

Each file request retains its native operands until the selected queue confirms
retirement, including after an early execution failure. A failed request does
not poison the staging shared by independent queues. File-I/O acquire/release
effects execute on the same queue as the copy, alongside the caller's explicit
boundaries; no cross-engine cache handoff is needed.

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
