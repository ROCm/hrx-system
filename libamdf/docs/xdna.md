# XDNA native execution

libamdf supplies device admission, scoped memory, addresses, and native queue
submission. A HAL supplies instruction bytes, ELF loading and relocation, tile
programs, and scheduling. ARRAY and CONTROL are image-layer concepts; neither
appears in the driver API.

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
and lifetime, including references followed by device-side streaming after a
kernel submission retires. Completion does not discover those references.

For CPU/NPU interchange, the caller queries `memory_query_pair_info` with its
concrete host mapping and device access plus queue-family ordinal. The result
selects the host publication or invalidation operation; XDNA DMA requires no
additional device cache transition. The caller publishes inputs before use and
acquires outputs after the program finishes the relevant DMA and its ordering
edge completes. The query itself neither flushes caches nor orders execution.

## Native requirements

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
zero selects the default of 4096, and `kernel_queue_query_info` reports the
effective capacity. The library publication path performs no allocation,
instruction parsing, relocation, argument resolution, native submission retry,
sleep or host wait.

The queue preallocates native packet and result storage for the complete window.
Completed slots are reclaimed when submission reaches that bound, without an
intermediate host wait. If all slots remain occupied, submission returns `BUSY`
and leaves the output unchanged. Native resource exhaustion can reject a command
before the configured bound. The native driver can also block admission waiting
for its own job credits, independently of the library's packet capacity. KMQ
publication is therefore not a nonblocking OS contract. Multiple contexts can
independently own instruction backing and queues. Those backing lifetimes are
distinct from residency of application state in the physical tiles.

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

The [canonical ELF consumer](../../experimental/xdna/cts/execution_test.cc)
shows the complete flow, including target selection, image loading, relocation,
cold host preparation, independent execution, numerical checks and teardown. The
ELF decoder and materializer live in the runtime image layer; libamdf receives
only the prepared native range. Reusing that range does not repeat image
loading or require an indirect data-buffer list. Each independent submission
uses the complete setup-and-execution range to establish its application tile
state; time-sliced context lifetime alone does not guarantee that state survives
between submissions.

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
memory live until its tile and DMA users have finished; neither libamdf nor the
HAL discovers or tracks those uses.

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
PCI and target identity while the numerical consumer shares the compatible
Strix image profile. Image ABI, instruction format, context bounds and required
capabilities remain checked.

The [memory fabric](memory.md) describes the shared scope, address, visibility,
and lifetime contracts used by GPU and XDNA callers.
