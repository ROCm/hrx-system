# GPU queues

libamdf creates native GPU queues for directly mapped USER publication or
kernel-mediated KERNEL submission. The runtime owns executable loading,
command encoding, memory visibility, execution dependencies and resource
lifetimes. Those responsibilities remain the same whether the runtime uses
individual dispatches, recorded command buffers or persistent programs.

The [AMD GPU hardware reference](../../docs/reference/amd/gpu/README.md)
describes PM4, SDMA and AQL command semantics.
[GPU timing and performance counters](../../docs/reference/amd/gpu/observability.md)
describes native timestamp, clock and profiling mechanisms.

## Selection and preparation

`endpoint_query_queue_family_info` describes command representation separately
from publication. A runtime selects `GPU_AQL`, `GPU_PM4` or `GPU_SDMA`, a format
version, the required operations, and a supported publication mode before
creating the device. The [GPU header](../include/amdf/gpu.h) specifies each
format's packet layout contract, index units and publication ordering. The
selected family also reports producer modes, priorities, ring limits, optional
format features and cache operations.

The native providers expose these combinations when the endpoint's driver,
engine and storage requirements are satisfied:

| Provider | AQL | PM4 | SDMA |
| --- | --- | --- | --- |
| Linux KFD, CDNA | USER | No compute family. | USER |
| Linux KFD, RDNA | USER | USER | USER |
| Windows WDDM, RDNA | Unavailable. | KERNEL | KERNEL on hardware-scheduled engines. |

Family discovery is the admission contract. SDMA packet features follow the
native SDMA engine independently of compute architecture. Windows engines
requiring software-scheduled SDMA submission have no exposed SDMA family.
For AQL, KFD materializes the architecture-specific descriptor and per-XCC
context storage on admitted CDNA and RDNA endpoints. A compiler still uses
endpoint architecture information to produce compatible kernel code; queue
selection does not make that code portable between architectures.

The caller creates code, kernarg, data and signal backing with the intended
device access. Code requires executable access. Device addresses come from each
memory attachment; a host mapping supplies a separate CPU view. The
[memory contract](memory.md) determines how host writes become visible and which
cache transitions the device requires. A CPU-written local kernarg allocation
therefore uses its exact host-to-queue pair: the caller executes the reported
host release after filling the arguments, then gives the dispatch the reported
queue acquire and retains the bytes through completion.

An executable image is not only its instruction bytes. The caller materializes
the descriptor, every compiler-owned immutable object reachable by the selected
kernel, the linked text and all required alignment or fetch padding at their
linked relative addresses. It publishes that complete initialized image before
dispatch and retains its backing through the final completion of every queue
that can reach it. Kernarg or payload publication does not repair omitted
executable bytes, and packet-slot retirement does not end their lifetime.

## Mapped USER publication

`gpu->user_queue_create` takes the family ordinal, producer mode, priority,
optional ring size and optional fixed scratch. It prepares all native ring,
control, context-save and notification storage before returning. The caller
then uses `user_queue_map(queue, NULL, ...)` and
`user_queue_mapping_query_info` to obtain host ring, index and doorbell
addresses. These addresses remain stable for the mapping lifetime. Queue
creation belongs to device or execution-stream preparation; dispatch publication
reuses its established storage.

Host producers order command and index stores before the write-only MMIO
doorbell with a host-to-device store fence: `SFENCE` on x86-64, followed by one
naturally aligned 64-bit doorbell store. A C/C++ release store alone is
insufficient. This publication fence applies to PM4, SDMA and AQL; it does not
replace the memory site's payload cache transitions or execution dependencies.

`ring_byte_length` is zero for the provider default or an explicit power of
two within the family's range and alignment. KFD PM4, SDMA and AQL queues
default to 4 KiB and admit requests through 2 GiB. That upper bound describes
representable geometry; allocation and native queue creation can still fail.
The returned queue and mapping information carry the actual capacity. A larger
ring permits more commands to be prepared before publication, while command
consumption and final payload use retain their separate reuse boundaries.

### Device-produced SDMA

An SDMA family with `AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER` can establish
publication by its owning GPU. The caller includes that bit in
`required_capabilities` when creating the queue, then calls
`user_queue_map(queue, owning_device, ...)`. Queue creation owns the GPU
notification mapping; the producer mapping borrows it. Creation fails if it
cannot establish the requested capability. Mapping a host-only queue later
does not add device publication, and a different GPU requires its own supported
mapping rather than reusing the owner's addresses.

The device reserves available ring bytes, writes complete packets, and makes
those writes visible at system scope before publishing WPTR. It then writes
the same monotonic byte frontier to the write-only 64-bit doorbell, with system
release ordering. Acquiring RPTR permits command storage reuse. Copy completion,
payload cache transitions and downstream-reader retirement are separate
dependencies; RPTR alone does not establish them.

The caller owns packet generation, resource credits and scheduling. A running
shader can select source/destination ranges, generate a copy and consume its
completion without a libamdf or CPU call for each transfer. The producer stops
using every mapped address before releasing its mapping. Queue destruction
then retires the native queue before releasing its notification and command
storage, following the same terminal cleanup contract as host publication.

### AQL packets and signals

AQL format 1 exposes native 64-byte packet slots with 64-bit packet indices.
Firmware owns the read index. The write index is a reservation frontier, so it
can include packets that another producer has reserved but not yet published.

For each packet, a producer:

1. Reserves an index. Multiple producers use atomic fetch-add on the write index.
2. Waits until the acquired read index permits reuse of the reserved ring slot.
3. Writes the packet body, including addresses and any native signal handles.
4. Release-stores the header/setup dword, making the complete packet valid.
5. Executes a host-to-device store fence (`SFENCE` on x86-64).
6. Stores the packet index to the mapped write-only 64-bit MMIO doorbell.

Single producers notify in increasing order. Multiple producers may finish
publication and notification out of order; an earlier unpublished slot blocks
later consumption. Firmware invalidates consumed slots before returning them
to the producer. Ring capacity limits unreclaimed packet storage, not the number
of workgroups or total execution time.

This path contains no libamdf submission call, scratch allocation or command
translation. The runtime owns reservation and publication synchronization.
Separate queues have independent progress; the runtime expresses dependencies
using the native mechanisms appropriate to its command model.

An AQL family advertising TRANSFER accepts the format-1
[confirmed PM4 transfer subset](../include/amdf/gpu.h). Its vendor packet borrows
an executable indirect buffer; the caller retains the complete immutable IB
through native execution completion. Ring consumption releases the packet slot
without authorizing IB reuse. A multi-XCC device needs the format's PRED_EXEC
prefix around operations that must execute once; a single-XCC device does not.
The family role and endpoint topology supply these decisions independently.

Standard AQL dispatch and AND/OR barriers do not imply the optional vendor
barrier-value packet. A caller using that packet additionally requires
`AMDF_GPU_AQL_FORMAT_FEATURE_BARRIER_VALUE` in the family's `format_features`.

## Queue timing

### AQL dispatch timestamps

An AQL family advertising
`AMDF_GPU_AQL_FORMAT_FEATURE_DISPATCH_TIMESTAMPS` accepts
`AMDF_GPU_USER_QUEUE_FLAG_DISPATCH_TIMESTAMPS` in its queue creation `flags`.
The option enables native firmware capture for the queue lifetime. Zero flags
leave capture disabled; the feature bit alone does not turn it on. PM4 and
SDMA use explicit timestamp commands and reject this queue option.

Each observed dispatch receives an exclusive caller-owned 64-byte native
signal, with its start/end fields initialized to zero. Firmware writes the
unsigned 64-bit start/end ticks at byte offsets 32/40. The caller acquires that
dispatch's completion and copies the timestamps to its own sink before reusing
the signal. Rearming only the signal value leaves old timestamp bytes intact.
A signal shared by several dispatches, or attached only to a final barrier,
does not identify individual dispatch intervals.

Capture selection adds no libamdf call to submission or collection. The caller
can cycle a bounded set of signals throughout execution, with completion and
consumption preceding each reuse. A runtime can enable the queue property at
creation and independently select which dispatches receive observation signals.
Changing the native property on an already-used queue requires firmware state
reload; queue creation establishes it before activation instead.

The interval describes firmware packet processing, not exact first/last shader
instructions. Resident programs produce one outer interval. The nominal queue
clock rate supplies duration units; host correlation uses actual source
observations and bounds as described below. The
[hardware dispatch contract](../../docs/reference/amd/gpu/aql/profiling.md#dispatch-timestamps)
describes native layout and sampling semantics separately from the API.

### SDMA timestamps

An admitted SDMA format-1 queue can write raw `TIMESTAMP_GET_GLOBAL` samples
into caller-owned device-writable storage. The caller brackets its intended
commands with samples and puts the payload's release and completion after the
ending sample. The reader observes that completion and applies the queried
memory edge before reading. Timestamp bytes themselves are not a publication
flag, and command-ring consumption does not make a sample ready.

A runtime can reuse a fixed number of sample slots while execution continues.
The producer waits for each slot's returned credit before writing it. A
separate ordered publication word identifies complete records; the consumer
returns credit only after copying them. This protocol lives in the caller's
command stream and memory layout. libamdf owns no query pool, result collector,
record cursor or sample lifetime, and ordinary submission adds no profiling
work. USER and KERNEL publication retain their native queue contracts.

### Rates and host correlation

GPU extension version 2 exposes the native nominal rate for an admitted queue
family through `device_query_queue_clock_info`. The same passive clock record
is used for XDNA observations. Query once at the caller's profiling setup
boundary; ordinary submission performs no rate query or sampling:

```c
amdf_observation_clock_info_t clock = {0};
amdf_status_t status = gpu->device_query_queue_clock_info(
    device, family.ordinal, &clock);
// Handle a failed query before consuming clock. A successful rate is nonzero.
if (amdf_status_is_ok(status)) {
  // Within one non-wrapping source epoch:
  double duration_ns = (double)(end_ticks - begin_ticks) * 1e9 / clock.frequency_hz;
  // Record duration_ns in caller-owned profiling storage.
}
```

Linux supplies the selected render device's `gpu_counter_freq`, converted from
kHz to Hz. Windows queries the engine node chosen for that family's native
scheduler. An unavailable native rate returns `UNSUPPORTED`; failure leaves
the output unchanged. Meaningful counter width remains zero when the native
interface does not establish it. Shader cycle counters and performance-event
counts use different units.

A nominal rate supplies duration units, not bounded oscillator accuracy or a
shared epoch. In particular, native driver counter samples need not share the
epoch of queue-written timestamps. The caller can bracket an actual queue
sample with its chosen host clock, or surround a source interval with host
submission/completion observations. For an interval `(device_begin,
device_end)` inside `(host_before, host_after)`, a constant-offset model at the
selected rate gives `host_before - device_begin_ns <= offset <= host_after -
device_end_ns`. Subtract a nearby source origin before conversion to retain
precision and avoid overflow. Intersecting independent brackets narrows that
capture's offset range; an empty intersection rejects the model. This is not a
guarantee about future drift, reset, wrap or power transitions. The caller owns
refresh and uncertainty policy. Queue identity and event attribution remain
explicit; equal rates alone correlate nothing.

The [SDMA timing reference](../../docs/reference/amd/gpu/sdma/timing.md) describes
sample placement, publication and instrumentation effects. A sample pair can
include drains and cache work, and its difference is not a measurement of the
host cost of submitting the commands.

## Fixed private memory

AQL kernels declare their private-segment requirement in compiler-produced
metadata. An all-zero scratch request admits only kernels with no private
segment. Otherwise the caller supplies `amdf_gpu_queue_scratch_t` with a
read/write device attachment and a fixed per-workitem capacity. That allocation
is exclusive to the queue through execution completion and destruction.

Retained scratch addresses physical wave slots, including padding for unequal
numbers of active CUs per shader engine. The public endpoint information gives
`C = compute_unit_count`,
`E = xcc_count * shader_engine_count_per_xcc`, and
`S = maximum_scratch_wave_count_per_compute_unit`. The caller sets
`maximum_wave_count = ceil(C / E) * E * S`. Every slot receives
`round_up(maximum_private_segment_byte_length * 64, 1024)` bytes, including
when kernels use wave32. The backing starts at a 4096-byte-aligned GPU address
and covers every physical slot.

For example, 68 private bytes per workitem need 5120 bytes per slot. A topology
reporting 304 CUs, eight XCCs, four shader engines per XCC and 32 scratch slots
per CU has `ceil(304 / 32) * 32 * 32 = 10,240` slots and needs 52,428,800 bytes.
Multiplying only the active CU count by its scratch slots would miss the
shader-engine padding. This capacity is independent of how many workgroups a
dispatch launches. Extra backing does not increase the declared capacity;
each dispatch fits the configured private-segment limit.

Queue creation validates this geometry and programs its per-XCC descriptors.
Smaller scratch pools require a different firmware reclamation protocol and are
rejected by this format. There is no hidden scratch growth or reclaim handler.
The runtime can choose a queue capacity from its executable set before creating
the queue and reuse it across dispatches.

## Kernel-mediated submission

`gpu->kernel_queue_create` takes a KERNEL family's ordinal and the maximum
number of pending submissions. Creation prepares native submission and wait
resources. `kernel_queue_query_info` reports the achieved pending capacity and
maximum command count; the GPU provider accepts one command range per
submission and permits multiple submissions to remain in flight.

The caller encodes and publishes a complete native stream in memory with
EXECUTE access and a stable DEVICE_ADDRESS attachment for the queue's device.
`amdf_gpu_kernel_command_t` identifies that memory, access ordinal, byte offset
and byte length. `gpu->kernel_queue_submit` consumes the descriptor array and
returns an increasing queue-local completion point. It neither copies nor
parses the command bytes, and it does not retain memory owners. The command
range stays immutable and every indirect reference stays live through its
final device use and checked retirement. The [memory contract](memory.md)
supplies the publication and payload cache transitions.

A KERNEL PM4 stream enters as a first-level indirect buffer, whereas a USER
PM4 stream enters on the primary ring. A primary-ring IB call cannot simply
be wrapped inside the submitted KERNEL buffer: compute IB2 call-and-return is
unsupported. Command construction follows the selected entry context and the
format's [indirect-buffer contract](../../docs/reference/amd/gpu/pm4/command-buffers.md).

The base API's `kernel_queue_wait` establishes checked retirement for a
returned point and its earlier accepted prefix. Points need not be dense or
start at a particular value. `kernel_queue_refresh_status` performs a
nonwaiting progress and retirement operation; `kernel_queue_query_status`
only reads already established progress. A wait timeout or a sampled healthy
status permits no additional storage reuse. Independent queues still require
explicit device dependencies and payload visibility.

Queues with `AMDF_KERNEL_QUEUE_CAPABILITY_VIRTUAL_MEMORY` also accept
`kernel_queue_update_memory`. These points belong to the same sequence:
preceding commands finish before replacement, and subsequent commands observe
the new backing. Mapping publication is an explicit cold operation and may
allocate descriptor storage; ordinary command publication is unchanged. The
[mapping contract](memory.md#queue-ordered-replacement) defines backing lifetime,
accepted failure and the host-completed alternative for other providers.

After every accepted point retires, `kernel_queue_destroy` releases the native
owner before the caller releases command backing and the device. An API-domain
BUSY result leaves the queue live; other results consume a valid queue handle,
including native cleanup errors. The caller follows the
[release-failure contract](memory.md#lifetime-and-failure) for any storage that
may remain reachable.

### Event-driven KERNEL completion

Windows GPU KERNEL queues report `AMDF_NATIVE_EVENT_TYPE_BIT_WIN32_EVENT` in
`kernel_queue_query_info().notification_types`. The same operation serves PM4
and SDMA families admitted by discovery. The caller creates an auto-reset or
manual-reset event, establishes event-loop ownership, and requests a wake with
`kernel_queue_request_notification(queue, accepted_point, &event)`. The event
needs `EVENT_MODIFY_STATE` access; the caller separately owns wait/reset access
and event-loop registration.

An unchecked point registers the event against the queue's stable monitored
fence using the native asynchronous CPU-wait operation. An already-completed
point signals immediately; delivery can precede return. Earlier checked points
remain valid after command storage is reused or later work fails. Registration
neither consumes command results nor advances checked retirement. A registration
error preserves its native status and leaves accepted work accepted.

A typical completion worker arms an accepted point, sleeps on its event, consumes
readiness, and calls `kernel_queue_refresh_status`. If more accepted work remains,
it arms the next unchecked point before sleeping again. Several queues can share
one event: a wake prompts a refresh of all participating queues. Repeated requests
are independent and signals may coalesce, so the wake count identifies neither
a queue nor a completed submission. Host control can also wake the worker without
implying device progress. Submission never implicitly arms the event.

The event descriptor is borrowed only for the call. The native event remains live
through outstanding delivery; cancelling a host callback does not cancel native
registrations. For a shared completion worker, its event outlives the participating
native queues: stop new arms, reconcile accepted work and notification delivery,
detach and join host callbacks, release the queues, then close the event. Checked
progress alone is not a count or cancellation of native notifications. Native
destruction failure follows the existing release-failure contract; it is not a
successful join. This ownership belongs to the caller, with no subscription or
lifetime registry inside libamdf.

A device-reset fence sentinel wakes native monitored-fence waiters but produces
terminal failure, not successful retirement. A cached terminal failure rejects
new requests for unchecked work. Queue-local submission or mapping errors do not
promise a device-wide wake; their synchronous status and subsequent checked status
remain authoritative. The [native fence reference](../../docs/reference/amd/gpu/wddm.md#asynchronous-cpu-notification)
describes the OS mechanism and its distinction from packet-originated interrupts.

Explicit arming pays either a native registration or an immediate event signal.
It uses no library allocation, lock, polling loop, worker, or retained descriptor.
The native driver may allocate wait state. Synchronous `kernel_queue_wait` retains
its separate private event and serialization, so it can coexist with caller-event
registration without stealing a wake.

## USER completion and release

AQL completion and barrier dependencies reference complete native 64-byte AMD
signal blocks. A signal is caller-owned device-visible memory with USER kind,
a signed 64-bit value at byte offset eight, and initialized unused fields. The
event padding at byte 28, doorbell-only queue pointer at byte 48, and tail
padding at byte 56 are required zero for USER signals; they are not spare
caller storage. Timestamp fields begin at zero. A packet carries the block's
GPU address. These are native packet operands; libamdf does not create or retain
a signal object for each dispatch. The
[native signal reference](../../docs/reference/amd/gpu/notifications.md#amd-user-signal-layout)
traces the representation and its distinct notification state.

A kernel dispatch decrements its completion signal once after all workgroups
finish. The runtime chooses fence scopes and observes completion with the
required acquire semantics before using results or releasing reachable storage.
Native polling signals do not provide a host event-loop notification by
themselves. The runtime's wait and notification strategy is a separate concern
from allocating the signal block.

There are distinct reuse boundaries:

| Observation | Storage it permits the caller to reuse |
| --- | --- |
| Acquired native read index, or successful `user_queue_wait_consumed` | Consumed packet slots in the command ring. |
| Native execution completion with the required visibility | Complete executable images, kernargs, payloads and completion storage whose final use has completed. |
| Successful `user_queue_destroy`, after execution completion and mapping release | The queue's borrowed scratch backing. |

`user_queue_query_status` samples progress and terminal state. The KFD provider
can enter the driver to inspect VM faults, and also observes firmware queue
errors. It is an explicit status operation, separate from mapped publication or
a direct signal load. A healthy sampled status is not execution completion.

Closing a runtime's producers prevents new work from entering the stream. It
does not remove the publication or execution obligations of already accepted
work. The runtime finishes outstanding publication and continues any host
services that work still requires, such as filling an upload source or
acknowledging consumption of a downloaded result. Waiting for completion while
withholding one of those dependencies can prevent the stream from draining.

A runtime with reusable staging slots tracks prepared inputs separately from
published work. Prepared data that no accepted work can consume requires no
native completion. When the runtime uses per-slot completion values, each final
wait refers to that slot's actual submitted generation. A partial batch can
leave slots at different generations; a rounded-up batch count or the
prepared-input count can include work that will never execute.

Normal release closes acceptance, finishes accepted publication and execution,
observes ring consumption, releases the producer mapping, destroys the queue,
and then releases scratch and other remaining dispatch allocations. The caller
checks every release result. Failed native retirement follows the documented
[resource-release contract](memory.md#lifetime-and-failure), including
preservation of backing that may remain reachable. Instance native-lifetime
policy still controls which underlying KFD reclamation boundary is available;
it does not change the caller's public ownership obligations.
