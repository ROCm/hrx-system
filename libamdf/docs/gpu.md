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
cache transitions the device requires.

## Mapped USER publication

`gpu->user_queue_create` takes the family ordinal, producer mode, priority,
optional ring size and optional fixed scratch. It prepares all native ring,
control, context-save and notification storage before returning. The caller
then uses `user_queue_map(queue, NULL, ...)` and
`user_queue_mapping_query_info` to obtain host ring, index and doorbell
addresses. These addresses remain stable for the mapping lifetime. Queue
creation belongs to device or execution-stream preparation; dispatch publication
reuses its established storage.

### AQL packets and signals

AQL format 1 exposes native 64-byte packet slots with 64-bit packet indices.
Firmware owns the read index. The write index is a reservation frontier, so it
can include packets that another producer has reserved but not yet published.

For each packet, a producer:

1. Reserves an index. Multiple producers use atomic fetch-add on the write index.
2. Waits until the acquired read index permits reuse of the reserved ring slot.
3. Writes the packet body, including addresses and any native signal handles.
4. Release-stores the header/setup dword, making the complete packet valid.
5. Release-stores the packet index to the mapped 64-bit doorbell.

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

After every accepted point retires, `kernel_queue_destroy` releases the native
owner before the caller releases command backing and the device. An API-domain
BUSY result leaves the queue live; other results consume a valid queue handle,
including native cleanup errors. The caller follows the
[release-failure contract](memory.md#lifetime-and-failure) for any storage that
may remain reachable.

## USER completion and release

AQL completion and barrier dependencies reference complete native 64-byte AMD
signal blocks. A signal is caller-owned device-visible memory with USER kind,
a signed 64-bit value at byte offset eight, and initialized unused fields. A
packet carries the block's GPU address. These are native packet operands;
libamdf does not create or retain a signal object for each dispatch.

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
| Native execution completion with the required visibility | Code, kernargs, data and completion storage whose final use has completed. |
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
