# XDNA execution through libamdf

libamdf supplies device admission, scoped memory, addresses, and native queue
submission. A HAL supplies instruction bytes, ELF loading and relocation, tile
programs, and scheduling. ARRAY and CONTROL are image-layer concepts; neither
appears in the driver API.

[XDNA timing, counters, and trace](../../docs/reference/amd/xdna/observability.md) describes tile timer
reads, event counters, trace routing, and native firmware instrumentation.

## One caller flow

The caller passively discovers an endpoint and its target identity, then
explicitly creates a device. `device_query_info` supplies native array geometry,
instruction limits and context admission before the caller creates a context.
Ordinary data comes from the instance's system-memory scope with that live
device in its access set.
Instruction storage comes from the context's private scope with EXECUTE access.
Both use `memory_create`, explicit host mappings, and cached address queries.

The HAL writes and publishes its instruction ranges once. Each native queue
submission identifies `{memory, access_ordinal, byte_offset, byte_length}`.
The caller keeps that range live through retirement; libamdf supplies only the
mandatory native transport packet. It neither copies the instruction stream nor
patches application arguments. The same prepared range can be submitted repeatedly
without additional driver objects or host-side instruction regeneration.

Time-sliced contexts do not reserve application tile state between submissions.
A context switch may reset tile registers, lock credits, or local memory while
the context and its host instruction allocation remain valid. Each independent
command establishes the state it needs. Its initialization and execution bytes
can be prepared together once and reused unchanged. A control-only range that
depends on a previous command's array configuration requires a separately
established native state-retention contract; context identity, fixed placement,
and successful prior completion do not provide that contract.

Ordinary data buffers have no per-submission BO list. Allocation establishes
their native mappings and residency. The caller maintains visibility, ordering,
and lifetime, including uses by independently scheduled consumers such as a
GPU program. Retiring the XDNA command does not join those consumers. Its own
tile workers and transfers must be quiescent before the controller finishes.

For Linux SVA, physical page pinning alone does not establish continuously usable
device translation. Compaction can temporarily invalidate a process mapping
without moving its pinned backing. The [host-memory translation reference](../../docs/reference/amd/xdna/execution.md#host-memory-translation-and-page-pinning)
describes the observed registered-memory failure, the qualified scope of an
allocation-specific mitigation and the progress requirements for resident
workers during native invalidation handling.

For CPU/NPU interchange, the caller queries `memory_query_pair_info` with its
concrete host mapping and device access plus queue-family ordinal. The result
selects the host publication or invalidation operation; XDNA DMA requires no
additional device cache transition. The caller publishes inputs before use and
acquires outputs after the program finishes the relevant DMA and its ordering
edge completes. The query itself neither flushes caches nor orders execution.

## Native entry and placement ownership

The native driver and firmware control which context may use a placement.
Time-sharing transfers that authority between contexts; it does not authorize
unrelated contexts to operate the same physical resources concurrently. libamdf
depends on native context isolation, just as it depends on native address-space
isolation. Applications neither drain another context's work nor coordinate an
exclusion lock with other device users. Context creation and host submission
acceptance are not the point at which the application starts using the array:
its setup executes only when the native provider schedules its command.

The program owns a different obligation: its controller instructions cover the
work using that placement. Before they finish, it quiesces its tile workers and
drains transfers that could interfere with subsequent reconfiguration. Closing
an input includes satisfying already-admitted reads; publishing a stop flag is
insufficient for a worker blocked on another event. A parked worker is quiescent
only when its protocol prevents further interfering work. Native command
completion reports the end of the submitted controller program; it does not
discover or join arbitrary subordinate work on the program's behalf.

| Transition | Native provider responsibility | Program responsibility |
| --- | --- | --- |
| Complete A, then B on one native queue | Execute the accepted controller programs in order. Another context may use the placement between them. | A closes its services before ending; B establishes its complete required state. Both may be submitted before the CPU waits. |
| A different context uses the placement | Isolate the incoming execution from the previous context's activity. This is a native handoff obligation, not a cross-process application protocol. | Initialize required registers, routes, locks and local data without assuming either retained values or a zeroed starting state. |
| An admitted command is rescheduled | Maintain the supported native execution semantics, or report execution failure. | Keep its instructions and reachable memory valid; continue the existing computation rather than applying independent-entry initialization to partially completed work. |

Native isolation is not a promise that every register or FIFO is zero at every
command boundary. Same-context commands can leave state behind, and successful
completion alone does not make an incompletely drained program safe to
reconfigure. Conversely, correct applications are not responsible for repairing
a provider that permits a foreign context's activity to interfere after handing
over the placement. libamdf supplies no application reset sequence, hidden drain,
placement registry or per-dispatch state snapshot.

On Linux, AMD assigns partition setup and context management to firmware and
documents firmware enforcement of context-to-column binding in its
[NPU architecture](https://www.kernel.org/doc/html/latest/accel/amdxdna/amdnpu.html).
On Windows, the native context and hardware queue operate under the
[MCDM execution and scheduling contract](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/mcdm-architecture).
These are the provider dependencies, not a claim that both implementations use
the same reset sequence. The current submission envelopes supply no separate
caller-authored save/restore programs; they do not establish an arbitrary
live-tile checkpoint facility. Cooperative checkpoints require their own native
payload contract and are distinct from complete-command handoff.

Replacing a role or program inside one still-running invocation does not take
this native-entry boundary. Its services remain owned by that invocation. A GPU
consumer can also continue using shared output after NPU retirement, under its
own execution and memory-lifetime protocol; that does not keep the NPU placement
owned. These are separate lifetime boundaries: shared data can outlive a
context, while application tile state requires its own execution protocol.

## Native requirements

The [endpoint profile](../src/xdna/device_profile.c) reports the following
canonical targets with architecture `AMDF_XDNA_ARCHITECTURE_AIE2P`. The native
driver uses separate product names for their PCI identities:

| Canonical target | Native product name | PCI device / revision |
| --- | --- | --- |
| `amd.xdna.strix.17f0_10` | NPU4 | `0x17f0 / 0x10` |
| `amd.xdna.strix_halo.17f0_11` | NPU5 | `0x17f0 / 0x11` |
| `amd.xdna.krackan.17f0_20` | NPU6 | `0x17f0 / 0x20` |

The [pinned native device table](https://github.com/amd/xdna-driver/blob/8dfda66f67a84aecf26cf68336efc9e4cc1756c3/drivers/accel/amdxdna/amdxdna_pci_drv.c#L79-L84)
supplies those product names. Array geometry comes from the activated device;
sharing AIE2P does not imply equal row or column counts. The AIE2IPU register
tables and AIE4 native interfaces discussed in the observability reference are
separately scoped source contracts, not additional execution admission.

| Boundary | Linux modern DRM | Windows MCDM |
| --- | --- | --- |
| Context admission | Native hardware context and negotiated execution support. | Native private adapter query selects direct or metadata partition admission. |
| Instruction storage | Context-qualified DEV backing, with sizes and alignment from its scope. | One 64 MiB native aperture per context; a reserved 32 KiB bootstrap prefix is excluded from the caller's usable range. |
| Instruction submission | Mandatory DRM execution record and command BO referencing the caller's instruction range. | Mandatory native transport record and transaction-interpreter packet referencing that range. |
| Ordinary data addresses | Firmware and shim-DMA address interpretations. | Firmware and shim-DMA address interpretations for standard system backing. |
| Private addresses | The scope reports available firmware and DMA interpretations. | Firmware address returned by native allocation; a private shim-DMA interpretation is not advertised. |

Profiles distinguish backing granularity from address alignment. In particular,
Windows standard backing is rounded to 64 KiB, while KMT mappings guarantee
4 KiB address alignment. Native mapping bounds account for the target's shim-DMA
translation so the complete range fits every advertised address interpretation.

Linux checks the opened device file and driver identity, then requires the
native array metadata and allocation/context operations used by its hardware
architecture. DRM release metadata does not determine admission.

Windows queries the native private adapter interface before preparing a context.
Its reply presence and required size distinguish three coupled context and
submission contracts:

- The baseline provider returns success without populating the 8-byte reply.
  It uses a 272-byte metadata context and 88-byte submission headers.
- A populated basic 8-byte reply selects the expanded 312-byte metadata context
  and 104-byte submission headers. Both metadata protocols supply bootstrap
  identity and partition width without an embedded xclbin and use a shared,
  host-only response allocation.
- A provider requiring the extended 12-byte reply supplies kernel-buffer
  allocation policy for direct partition admission and 120-byte submission
  headers. The context retains its native kernel buffer until destruction;
  the queried policy selects shared or unshared kernel-buffer allocation.

These native interfaces establish the support floor. Compatible newer drivers
are accepted without code changes. Driver build numbers, reserved query fields,
and particular populated hardware-kind values do not select wire layouts.
An invalid extended allocation policy is unsupported before context preparation.
Native query and context failures propagate without guessing
another layout. An escape query on the created device supplies native tile
layout. Native context ID zero is valid.

Windows initialization constructs a minimal PDI/CDO container directly in
reserved private backing. Its single NOP admits the transaction interpreter
without installing a tile program or assigning application DMA, locks, routes,
or tile data. Admission runs once for that backing, independently of application
dispatch and program replacement. The native bootstrap UUID identifies the
admission bytes; target-specific native context accounting remains separate.
Linux direct ELF submission needs no bootstrap container. No public PDI,
program, lane, or argument-patching object is required.

## Submitting a prepared range

The queue and instruction memory below come from the same context. The caller
has already queried alignment and size limits, obtained EXECUTE access, written
its target-native bytes, and performed the required host publication. The
descriptor is consumed during the call; accepted instruction bytes remain
immutable until retirement.

```c
#include "amdf/xdna.h"

amdf_status_t publish_instructions(
    const amdf_xdna_api_t* xdna, amdf_kernel_queue_t* queue,
    amdf_memory_t* instructions, uint32_t access_ordinal,
    uint64_t byte_offset, uint64_t byte_length, uint64_t* out_submission) {
  const amdf_xdna_kernel_command_t command = {
      .memory = instructions,
      .access_ordinal = access_ordinal,
      .byte_offset = byte_offset,
      .byte_length = byte_length,
  };
  const amdf_xdna_kernel_queue_submission_info_t submit = {
      .type = AMDF_STRUCTURE_TYPE_XDNA_KERNEL_QUEUE_SUBMISSION_INFO,
      .structure_size = sizeof(submit),
      .command_count = 1,
      .commands = &command,
  };
  return xdna->kernel_queue_submit(queue, &submit, out_submission);
}
```

Queues admit one instruction range per submission and a configurable number of
unretired submissions. Set `maximum_pending_submission_count` at queue creation;
zero selects the default of 128, and `kernel_queue_query_info` reports the
effective capacity. The library publication path performs no allocation,
instruction parsing, relocation, argument resolution, native submission retry,
sleep or host wait.

The default limits native transport preparation cost while accommodating
overlapped launches. Applications with deeper host-driven pipelines can request
a larger window explicitly. Work scheduled within a persistent program does
not consume additional kernel submission slots.

The queue preallocates native packet and result storage for the complete window.
Completed slots are reclaimed when submission reaches that bound, without an
intermediate host wait. If all slots remain occupied, submission returns `BUSY`
and leaves the output unchanged. Native resource exhaustion can reject a command
before the configured bound. The native driver can also block admission waiting
for its own job credits, independently of the library's packet capacity. KMQ
publication is therefore not a nonblocking OS contract. Multiple contexts can
independently own instruction backing and queues. Those backing lifetimes are
distinct from residency of application state in the physical tiles.

Keeping these public owners alive does not keep an idle device powered. Native
runtime suspension can discard physical tile state while retaining the public
context and its backing. The next submission can synchronously wake the device
and restore native contexts before accepting the command. The caller still
supplies instructions that establish the application state needed for that
independent submission; libamdf does not detect lost tile state, replay program
setup or issue keepalive work. Eager queue preparation therefore does not imply
bounded wake-up latency or preservation of state between submissions.

The returned increasing, opaque submission number identifies accepted work.
Several commands can be published before waiting for the last accepted point;
that wait covers the queue's accepted prefix. The caller performs checked
retirement with `kernel_queue_refresh_status(queue, &status)` to consume the
available completed prefix without waiting, or
`kernel_queue_wait(queue, submission, AMDF_TIMEOUT_INFINITE, 0)` to wait for an
exact accepted point. A zero-time wait polls that point; unlike batch refresh,
it need not discover earlier completion while the requested point is pending.
Retirement includes native completion and command-result inspection.
`kernel_queue_query_status` is a read-only snapshot of retirement already
established by synchronization or capacity reclamation; it does not advance
retirement, even if the hardware has finished. A timeout or wait error is not
cancellation and does not by itself permit instruction storage reuse. The status
query reports established retirement separately from sticky terminal failure.
Batch refresh makes the same distinction: its return status describes native
observation, while the returned snapshot carries checked progress and execution
failure. A successful refresh may report no new progress, including when another
host caller is consuming results. Native observation failure leaves the output
unchanged and does not cancel accepted work.

An ELF decoder and materializer can prepare the native instruction range before
publication. Reusing that range does not repeat image loading or require an
indirect data-buffer list. Each independent submission uses the complete
setup-and-execution range to establish its application tile state; time-sliced
context lifetime alone does not guarantee that state survives between
submissions.

## Program sets and run-local bindings

A native queue has no mutable "current program" or "current arguments" binding.
Each accepted command names its own immutable instruction range. The HAL can
prepare program set A with one work-queue address and program set B with another,
then publish both without waiting for A on the CPU. Multiple prepared ranges can
share one context-private allocation; a pending run does not require a separate
instruction allocation, context, or native queue.

For example, a HAL preparing two pipeline runs owns these distinct ranges:

| Resource | Run A | Run B |
| --- | --- | --- |
| Native instructions | Prepared range A in context-private storage. | Prepared range B in the same allocation. |
| Work input | DMA address of queue A. | DMA address of queue B. |
| Run arguments | Ordinary memory containing A's parameters. | Ordinary memory containing B's parameters. |
| Shared state | Reads or produces state through its declared protocol. | Can consume A's published output directly. |

The HAL obtains each binding's address with `memory_query_address` in the domain
the consuming program requires. A queue's head, tail, entries, generations, and
argument record layout belong to the compiler/runtime ABI. To libamdf these are
ordinary memory, with the same access, visibility, and lifetime contracts as
tensor data. libamdf neither recognizes queue records nor infers dependencies
from their contents. Placement, worker count, and occupancy policy likewise
remain above the native surface.

The in-tree ELF materializer's `iree_hal_amd_xdna_executable_storage_load` and
`iree_hal_amd_xdna_executable_storage_bind` prepare each range before publication.
Its external binding relocations encode declared shim-DMA addresses, including the
base of a queue or argument buffer. This is not a generic scalar-immediate
argument API: a program can DMA its argument record just as it reads any other
bound data. Other argument conventions belong to the compiler and its image
loader; the native submission interface still takes only an instruction range.

Binding B writes B's prepared storage, not a queue-global table or A's accepted
instructions. Those ranges may be loaded from the same executable or different
executables. After publication, each remains immutable until checked native
retirement. A prepared range can then be reused unchanged with the same binding
addresses, while producers publish new records through the bound protocol.
Changing an embedded address requires either a different prepared range or
retirement of all users of the range being rebound. Queue and argument backing
remains live through its actual last device use, including downstream consumers.

A HAL establishes application dependency edges before choosing where and when
to publish native work. Ordering accepted native commands does not add implicit
FIFO semantics to a higher-level queue API or join another engine's users.

Program replacement *within* a resident invocation has a different boundary.
The running program can consume addresses of replacement code from its own
records and arrange device-side transfer and handoff without another libamdf
submission. Replacement source bytes need a usable DMA address; a firmware-only
address from context-private instruction storage is not interchangeable with
one. The caller queries the required address domain when allocating the source
catalog. The program owns instruction-fetch exclusion, transfer completion,
descriptor reuse, and any state carried into the replacement. That ownership
remains inside the native invocation; it does not establish tile-state retention
after native retirement. Independent runs A and B still establish their own
required tile state.

## Resident work and early admission

A resident service can process many logical operations inside one native
invocation. CPU and GPU producers publish application records to shared memory;
tile programs consume them and publish results through their device-visible
protocol. Those records do not require additional libamdf submissions, native
packet slots, or host notification requests. The compiler and HAL own this
protocol, including readiness, backpressure, stop, and drain.

For a frame or pipeline batch, the caller can submit prepared native work before
the CPU has finished constructing its input. The device program waits for an
explicit publication edge before reading each payload. This overlaps native
admission and tile setup with CPU production. Accepted submission is not worker
readiness; a program that needs that distinction publishes its own ready state.
An unused reservation still owns accepted work: the producer closes it through
the program's protocol and lets it drain before releasing its backing. A worker
blocked on a stream read needs that stream's wake mechanism to observe a stop;
changing a separate memory word does not itself unblock the read.

A long-lived service can span several bounded native invocations, or epochs.
Each epoch performs many logical operations and finishes at an application
boundary. Its successor establishes its tile execution from explicit state in
ordinary shared memory. A cursor, accumulator, or queue position can be part of
the predecessor's normal output; the successor can consume it directly without
a CPU copy. Immutable data and state already in shared memory need no checkpoint
copy. The compiler determines the live state, not libamdf or a generic hardware
snapshot. Retaining the public context alone does not preserve tile-local state.

Prepared successors can be queued before an epoch finishes. This permits native
handoff without an intervening CPU completion wait; it does not reserve the
array against other contexts. Each independent invocation still establishes
its required application state. Ordering between queues, devices, and logical
operations remains the caller's responsibility. The epoch's final completion
protocol covers its relevant tile and DMA users before their storage is reused;
a GPU consumer or work admitted to a separate native execution can have a later
last use. NPU workers using this epoch's placement remain covered by this
epoch's native command until they are quiescent.

Native watchdog and preemption policy is separate from logical service progress.
Advancing application records does not necessarily produce driver-visible
progress. Epoch duration includes waiting for producers and downstream credits,
not just arithmetic. The caller chooses work and drain bounds appropriate to
its native execution contract. A fixed operation count does not bound a program
that can wait indefinitely for input. Keeping one invocation alive indefinitely
is not implied by context creation or successful short execution.

`AMDF_TIMEOUT_INFINITE` removes the calling thread's wait deadline; it neither
disables a native watchdog nor extends an execution budget. A timeout does not
cancel accepted work. A native execution failure is not success merely because
some logical outputs arrived. The caller reconciles checked retirement and its
own output protocol before releasing storage or deciding whether an operation
can be repeated; libamdf cannot safely replay opaque application work.

## Power policy and measurement

Sustained XDNA measurements require an explicitly held-active device throughout
the measurement interval. Otherwise an idle gap can turn the next submission
into a hardware/firmware resume measurement even though the caller has retained
all its prepared resources. A warm-up command alone does not hold power across
later gaps. Results identify the power policy separately from instruction
preparation, tile initialization and command completion.

| Measurement | Native power policy | What the result describes |
| --- | --- | --- |
| Sustained execution or submission overhead | Hold runtime power active before timing and throughout the run. | Performance with native wake-up excluded. |
| Deployment-representative latency | Keep the deployment's actual power policy, including autosuspend where enabled. | Request latency with the workload's real idle gaps and wake costs. |

Held-power results are not an approximation of default-policy user latency.
A deployment that deliberately holds its device awake can use that policy in
its representative measurements, but reports it explicitly. Cold-wake runs
observe an actual suspend transition before timing; a fixed sleep does not
prove suspension when another process is using the device. Neither measurement
mode changes the requirement to establish application tile state for an
independent command.

On Linux, the selected accelerator's sysfs `device/power/control` attribute
accepts `on` to prohibit runtime autosuspend and `auto` to permit it. The
benchmark operator or machine-policy manager records the previous value, sets
`on` before the measurement, verifies both `control=on` and
`runtime_status=active`, and restores the recorded value afterward, including
when the benchmark fails. Selecting the device uses its native identity, not
an assumption that every system's intended device is `accel0`.

This is device-wide policy, not a reference owned by a process or open file:
closing the benchmark or a libamdf device does not restore it. The policy owner
serializes changes and owns restoration. The setting does not force maximum
clocks, disable thermal management, reserve tiles or preserve tile state.
Performance-frequency modes are separate controls and are not evidence of a
runtime-power hold. These semantics come from the
[Linux runtime-PM interface](https://docs.kernel.org/power/runtime_pm.html).
The Linux control is not a Windows API; a Windows held-power measurement needs
its own qualified native control and readback before receiving that label.

The measurement record includes native power policy, observed power state,
clock policy, driver/firmware, idle intervals and concurrent CPU/GPU/NPU work.
A cooperative machine benchmark lease serializes participating benchmarks, not
all device users. libamdf issues no keepalive work and changes no machine
power policy.

## Observation result memory

A counter-reply or trace destination is ordinary device-writable memory. The
caller selects a scope and memory profile with the intended NPU and host or
GPU consumers, then obtains backing with `memory_create`, registration, or
`memory_import`.

`memory_query_address(memory, access_ordinal, AMDF_MEMORY_ADDRESS_XDNA_DMA,
&address)` returns the address interpretation used by shim DMA. An offset into
the allocation is added to that address. The firmware address interpretation
is separate; the caller does not translate it using a fixed platform constant.

The prepared observation configuration lives with the other controller
instructions in context-private EXECUTE memory. `kernel_queue_submit`
publishes that range without parsing or modifying it. It does not start a
collector or inspect the observation destination.

After the program's completion edge, the reader applies the visibility recipe
from `memory_query_pair_info`. A host acquire requiring cache invalidation uses
`host_mapping_cache_control` on the relevant range. Cache control establishes
visibility; it does not wait for a still-running writer. A GPU reader can
consume the same backing under its own ordering and lifetime contract. The
[memory reference](memory.md) describes those operations in detail.

## Asynchronous host observation

A caller can keep one native event and one persistent event-loop registration
for a queue. `kernel_queue_query_info` reports `notification_types` for that
activated queue. A zero mask means native event notification is unavailable;
checked refresh and synchronous waits remain available independently.

The caller creates its event and establishes event-loop ownership before calling
`kernel_queue_request_notification(queue, submission, &event)`. The point must
already have been accepted by that exact queue. The descriptor is borrowed only
during the call; the caller keeps the native event live through delivery. The
library stores neither the descriptor nor a subscription.

| Native destination | Registration | Readiness consumption |
| --- | --- | --- |
| Linux nonblocking eventfd | DRM syncobj eventfd request on the exact native fence. | Drain the eventfd counter before requesting another edge. |
| Windows event HANDLE | Asynchronous KMT wait on the exact monitored fence value. | An auto-reset event is consumed by its wait; a manual-reset event is reset by its owner. |

The request is one-shot. It may signal before returning, and repeated requests
add fresh wake obligations rather than replace an earlier request. Notifications
can coalesce; their count is not a completion count. A request for an already
checked point signals immediately, including after its packet slot has been
reused. Submission never implicitly arms or rearms the event.

In the normal single-request flow, the event-loop callback consumes readiness,
calls `kernel_queue_refresh_status`, and processes the checked prefix through its
ordinary completion/release path. If accepted work remains, it requests a wake
for the first unchecked point. When another host caller owns result consumption,
refresh can return unchanged progress: requesting that same point again produces
a fresh hint, without a writer handoff or second retirement mechanism. A caller
publishing work to an idle queue establishes this wake obligation before
returning to sleep; the library has no hidden subscription to do it later.

Readiness is an opportunity to check progress, not successful execution or
permission to release storage. Native first-fence capture contention can produce
an early recheck hint; checked retirement remains authoritative. A notification
error does not reject or replay accepted work. Cancelling an event-loop callback
does not cancel its native request or device execution. Normal teardown consumes
the outstanding wake and reconciles checked last use before releasing the queue
and native event. This single-request flow requires no per-submission event
allocation or notification history.

Each explicit request pays its native registration or signal cost; the kernel
may allocate callback state. Queue creation qualifies the transport. Neither
requesting a notification nor refreshing progress creates a libamdf observer,
event ring, worker, or subscriber registry, and these operations add no work to
the free-capacity submission path. Synchronous callers retain native fence waits
without creating an event. Host observation does not mediate device-to-device
ordering or resident program signaling.

## Ownership

A queue borrows its context. Private memory also borrows its context; ordinary
memory borrows its explicitly requested devices. The HAL releases mappings and
memory after final use and before their owners. There is no hidden retention,
allocation registry, or library suballocator. Several live contexts can own
independent private backing even when their firmware addresses are numerically
equal.

The HAL and compiler own the tile execution model: workgroup placement, core
enable/disable/reset sequences, DMA descriptors, channel and lock protocols,
program replacement, and idle policy. These operations are expressed through
caller-owned target-native instructions and tile programs. Changing between
finite dispatches and resident work queues does not require a different libamdf
submission API. Native retirement establishes when the caller may reuse the
submitted instruction storage. The caller keeps all indirectly referenced
memory live until its tile, DMA and independently scheduled users have finished.
libamdf does not discover or track those uses; the caller supplies the explicit
last-use dependencies required by its execution model.

Native context scheduling and placement constrain those execution models.
Fixed physical backing does not grant exclusive ownership or uninterrupted
residency. The caller queries the admitted scheduling mode and placement
contract; native command completion alone does not establish preservation of
application tile state across scheduling or reset events.

Passive endpoint information contains architecture and compiler target identity.
The activated device owns immutable native row/column metadata and effective
context and instruction capabilities. Instruction encodings, alignment and DMA
address translation are AIE architecture contracts, independent of array size.
Native allocation and admission still enforce resource availability; libamdf
does not publish guessed limits on simultaneously live contexts.

NPU6 (Krackan) uses the AIE2P path and NPU4 firmware bootstrap. AMD's
[driver definition](https://github.com/amd/xdna-driver/blob/8dfda66f67a84aecf26cf68336efc9e4cc1756c3/drivers/accel/amdxdna/npu6_regs.c)
shares NPU4 firmware, hardware operations and feature contracts. Array geometry
is queried from the installed driver on both platforms. Krackan retains its own
PCI and target identity. A compiler or image loader separately establishes
image ABI, instruction format, context bounds and required capabilities; sharing
firmware does not by itself make executable images interchangeable.

The [memory fabric](memory.md) describes the shared scope, address, visibility,
and lifetime contracts used by GPU and XDNA callers.
