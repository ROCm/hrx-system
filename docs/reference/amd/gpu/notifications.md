# Native signals and host notification

A GPU completion can update a signal value and separately notify a sleeping
host. In the Linux KFD path, a native event connects a GPU interrupt to a host
wait queue; an AMD HSA signal contains the value and the event's notification
fields. Interrupt delivery makes a waiter runnable. The signal condition and
the consumer's memory acquire establish whether its work can proceed.
[Signal representation][signal-abi] · [Native event delivery][event-delivery] ·
[Host wait contract][wait-api]

## Applicability and distinct state

This chapter follows ordinary ROCr USER signals, libhsakmt event services and
Linux KFD. It describes the interrupt paths used by AQL completion, SDMA and
shader notifications; it does not assign Linux event semantics to Windows
KMT events, DXG, or an arbitrary host-visible allocation. A doorbell signal
notifies a GPU queue in the opposite direction and has a different kind and
value interpretation. [Signal kinds][signal-abi] · [Doorbell protocol](aql/publication.md#doorbell-values-and-native-mappings)

| State | Owner and meaning |
| --- | --- |
| Signal value | Signed 64-bit control value in the AMD native USER ABI; the producer's operation and the consumer's comparison define readiness. |
| Event mailbox | Eight-byte slot in the process's native signal page, used to identify pending GPU notifications. It is separate from the signal value. |
| Event ID | Native event identifier, also returned as the slot index and trigger data by the pinned KFD implementation. |
| Event `signaled` and waiter activation | Kernel-owned wake state with manual/automatic reset behavior. |
| Event age | Kernel-owned change history supplied to and from a host wait; it is neither the signal value nor a dispatch-completion count. |

[ABI][signal-abi] · [Event construction][event-create] · [Slot allocation][event-slots]
· [Kernel wake state][event-set] · [Wait ABI][event-uapi]

### AMD USER signal layout

The native block is 64 bytes with 64-byte alignment. A native handle identifies
that block; ROCr's conversion also reaches a surrounding `SharedSignal` and
runtime object. Constructing the native bytes alone does not create those
objects or make the handle valid for arbitrary ROCr APIs.
[Native layout][signal-abi] · [Runtime conversion][signal-convert]

| Byte offset | Size | Field and interpretation |
| --- | ---: | --- |
| 0 | 8 | `kind`: USER is `1`; INVALID is `0`; DOORBELL and LEGACY_DOORBELL are `-1` and `-2`. |
| 8 | 8 | `value` for USER; the union instead contains `hardware_doorbell_ptr` for its doorbell use. |
| 16 | 8 | `event_mailbox_ptr`; zero when this signal has no mailbox. |
| 24 | 4 | `event_id`. |
| 28 | 4 | ABI padding; must be zero. |
| 32, 40 | 8 each | `start_ts`, `end_ts`; [dispatch profiling](aql/profiling.md#dispatch-timestamps) owns their meaning and lifetime. |
| 48 | 8 | `queue_ptr` for doorbell kinds; reserved and zero for USER signals. |
| 56 | 8 | ABI tail padding; must be zero. |

ROCr zeroes the native block when constructing `SharedSignal`, then sets its
initial value and kind. `InterruptSignal` copies the event ID and `HWData2`
mailbox address from its native event. This is initialization, not a protocol
for rewriting the whole block while a device or host waiter uses it.
[Shared initialization][signal-storage] · [Initial value][signal-initial-value]
· [Event binding][interrupt-binding]

The archived AMD Compute ABI names the byte-28 and byte-56 padding as
must-zero and restricts `queue_ptr` to doorbell signals. Those fields are not
available for application metadata, including on a host-polled USER signal.
That document is obsolete as a general compiler and code-object reference;
the signal rule is used here only because the current ROCr representation and
whole-block initialization corroborate it. [Archived signal object][legacy-signal-object]

## Creating the native event

ROCr's ordinary signal factory selects a memory-polled `DefaultSignal` for
IPC, `HSA_AMD_SIGNAL_AMD_GPU_ONLY`, disabled interrupt waits, or an explicit
consumer list containing no CPU agent. Otherwise it selects `InterruptSignal`.
The latter normally borrows an auto-reset event from a runtime pool, creating
one when the pool has none available. After its first allocation failure, the
pool uses only returned events until a pool clear resets that policy. A
caller-supplied event has a separate owner.
[Factory predicate][signal-factory] · [Event pool][event-pool] · [Pool reset][event-pool-type]

If the pool cannot supply an event, the constructor leaves `event_id` and
`event_mailbox_ptr` zero while retaining USER kind. The selected runtime
class alone therefore does not prove that a native notification resource
exists. [Constructor result][interrupt-binding]

The public GPU-only attribute comment says the flag is ignored with a nonzero
consumer count. The pinned factory instead evaluates that flag before its
consumer-list branch. The declared consumer contract and this selection
predicate therefore remain distinct; a native implementation cannot derive
interrupt availability from the comment alone. The public value-pointer API
also limits its use to GPU-only or IPC signals, while the implementation
requires a `BusyWaitSignal`. It does not provide a general way to store into
an interrupt-backed signal without its notification path.
[Attribute and value-pointer contracts][signal-attributes] · [Factory and pointer query][signal-factory]

The thunk passes `auto_reset = !ManualReset` to `CREATE_EVENT`. For its dGPU
path, it first obtains a shared event-page allocation and passes its native
memory handle in `event_page_offset`. KFD maps that GTT backing into the
kernel. The returned offset, slot index and trigger data describe native
resources; none is an ordinary signal value. The thunk exposes the mailbox
byte address as `event_page_base + 8 * event_slot_index` through `HWData2`.
[Thunk construction][thunk-create] · [Native backing][event-backing]

The pinned UAPI sets `KFD_SIGNAL_EVENT_LIMIT = 4096`; each slot occupies eight
bytes, so the complete page requires at least 32 KiB. KFD reserves ID zero
and initializes available slots to `UNSIGNALED_EVENT_SLOT = UINT64_MAX`.
Its allocator limits IDs to the mapped extent. The thunk retains an older
256-event mmap fallback, but the cited upstream native path requires the
user-supplied full-sized backing. Those branches describe different native
interface histories, not two interchangeable sizes for the same constructor.
[Limit and creation ABI][event-uapi] · [Initial state][event-constants]
· [ID reservation][event-zero] · [Size check][event-backing] · [Thunk mapping][thunk-create]

## GPU notification and interrupt decoding

ROCr's classic SDMA completion sequence first updates the output value, then,
when a mailbox exists, emits a DWORD FENCE of the event ID to the mailbox and
a TRAP carrying the same ID. The mailbox slot is eight bytes even though this
producer writes one DWORD. KFD recognizes a pending slot by inequality with
the all-ones sentinel; this is not a 64-bit atomic signal update.
[SDMA completion and notification][sdma-notify] · [TRAP builder][sdma-trap]
· [Slot observation][event-lookup]

The [SDMA interrupt packet](sdma/trap.md) chapter compares the source-selected
context widths, older native command tails and kernel-ring observer. Packet
representation, interrupt decoding and event-allocation limits have distinct
owners.

The gfx125 fused-copy caller also accounts for a separate mailbox/trap
epilogue when its `fused_notify` and mailbox predicates select it. A fused
copy's value update and its host notification retain different command
positions. The [SDMA completion](sdma/fence.md#execution-and-lifetime) and
[atomic protocols](sdma/atomics.md) own the payload-release and update rules.
[Fused caller][sdma-fused]

KFD routes an interrupt to the process identified by its PASID and decodes an
event ID from engine-specific context bits. The following widths are the
bits passed to `kfd_signal_event_interrupt`, not native event-count limits:

| Selected interrupt decoder | CP end-of-pipe | SDMA TRAP | SQ message | Mailbox prerequisite |
| --- | ---: | ---: | ---: | --- |
| CIK | 28 | 28 | 8 | All three paths require a pending mailbox. |
| V9 and V9.4.3 | 32 | 28 | 24 | All three paths require a pending mailbox. |
| V10 | 32 | 28 | 23 | All three paths require a pending mailbox. |
| V11 | 32 | 28 | 24 | All three paths require a pending mailbox. |
| V12.1 | 32 | 28 | 24 | CP passes `signal_mailbox_updated = false`; SDMA and SQ still pass `true`. |

[CIK][irq-cik] · [V9 CP][irq-v9-cp] · [V9 SQ/SDMA][irq-v9-other]
· [V10 CP][irq-v10-cp] · [V10 SQ/SDMA][irq-v10-other]
· [V11][irq-v11] · [V12.1][irq-v12]

The native selector assigns GC9.4.3, 9.4.4 and 9.5.0 to the V9.4.3 class,
whose workqueue decoder is V9 but whose interrupt admission additionally
checks partition/node routing. Its listed earlier GC9 entries use V9; the
listed GC10.1/10.3 entries use V10. The listed GC11.0, 11.5 and 11.7 entries,
plus GC12.0.0 and 12.0.1, use V11. GC12.1.0 selects V12.1. These are native
GC predicates, not compiler-target major-version rules.
[Exact native selector][irq-selection] · [Partition routing][irq-partitions]

With sufficient ID bits, the V12.1 CP path can look up the event without
testing mailbox contents. This is the native decoder's expectation;
it does not remove the mailbox requirement from SDMA or SQ on that device.
Partial-ID lookup searches matching pending slots, and a failed direct lookup
can fall back to scanning pending events. The kernel clears a consumed slot
to the sentinel before updating the event and waking waiters. Notification
order across different events is not a completion-order guarantee.
[Lookup and partial IDs][event-lookup] · [Acknowledgment and delivery][event-delivery]

The V9 interrupt filter also distinguishes firmware versions that supply a
valid CP context ID. Its predicate is MEC firmware at least `0x817a` on
GC9.0.1, at least `0x17a` on GC9.1.0/9.2.1/9.2.2/9.3.0/9.4.0, and native GC
at least 9.4.1 in its default branch. When that predicate holds, zero-context
CP events are discarded instead of invoking the scan. This is a native
firmware workaround, not an event ID that a normal producer can allocate.
[Firmware predicate][irq-context-version] · [Filter][irq-context-filter]

## Sleeping waits and the check-to-sleep race

`InterruptSignal::WaitRelaxed` retains its runtime signal and increments its
waiter count before checking the value. It repeatedly evaluates EQ, NE, LT or
GTE against the signed value. For a blocked-wait hint, it initially polls for
200 microseconds before using the native event wait; an active hint continues
polling, optionally using MWAITX. This threshold is ROCr policy, not measured
interrupt latency or a required delay in another scheduler.
[Single-signal wait][interrupt-wait] · [Comparisons][signal-comparisons]

The wait's event-age input starts at one when ROCr reports support, otherwise
zero. ROCr enables this support for KFD major 1, minor at least 14, and forces
it off for DXG. Without event age, any waiter after the first uses active
waiting. The implementation therefore has a more specific contract than the
older class comment claiming that only one waiter is supported.
[Version predicate][event-age-version] · [Waiter selection][interrupt-wait]
· [Class comment][interrupt-class]

KFD protects event updates and waiter registration with the same event lock:

1. Registration captures `signaled` into the waiter's `activated` flag and
   consumes an auto-reset signal. For a SIGNAL event with nonzero supplied
   age, an age mismatch also activates the waiter.
2. An unactivated waiter is linked into the event wait queue before releasing
   that lock. A concurrent event set therefore either precedes the captured
   state or finds the registered waiter.
3. Before scheduling, the host task becomes interruptible and then rechecks
   activation. A concurrent wake restores runnable state, preventing the
   check-to-sleep race from leaving it asleep after the notification.
4. On completion, activated SIGNAL events with age comparison enabled copy
   their current age back to the host. The thunk round-trips these ages for
   the next wait.

[Locked registration][event-register] · [Locked set][event-set]
· [Sleep transition][event-wait] · [Age output][event-age-copy]
· [Thunk wait][thunk-wait]

An auto-reset event is not a one-waiter semaphore. Setting it activates
**all currently registered waiters**. If there are no waiters, the signaled
state remains available for a later registration; a manual-reset event
remains signaled until reset. Event age starts at one, increments on each
native `set_event`, and avoids zero/one on wrap. Age zero disables comparison;
nonzero ages are compared for inequality, not signed or unsigned ordering.
This preserves history even when another waiter consumed the auto-reset bit.
[Wake/reset behavior][event-set] · [Age comparison][event-register]
· [Initial age][event-create]

Age is not an exact interrupt or dispatch counter. For example, the native
restart path can restore a consumed auto-reset event by calling `set_event`
again. The boolean wake state and mailbox can also combine notifications.
A native wait-for-all observes activation of its registered events; it does
not read HSA signal values or apply the caller's EQ/LT/GTE conditions.
[Restart cleanup][event-wait-cleanup] · [Native condition][event-condition]

After every native wake, ROCr reloads and rechecks the signal. Its acquire
variant adds an acquire fence after the relaxed wait. The public HSA wait
contract also permits a return whose value does not satisfy the condition;
the caller evaluates the returned value and preserves the readiness condition
until dependent users observe it. Native event success alone supplies neither
this condition nor the payload's memory acquire.
[Recheck and acquire][interrupt-wait] · [Public wait contract][wait-api]

## Host producers and grouped observers

ROCr's ordinary host signal stores and RMW operations update the value with
their selected atomic ordering, then call `SetEvent`, which notifies KFD only
when the runtime waiter count is nonzero. Its separate
`StoreReleaseAndNotify` always notifies an existing event, and is used when
adding asynchronous handlers and shutting down the observer thread. Those
callers may race an observer about to enter a native wait without the ordinary
signal-wait registration. A raw value store does not reproduce that protocol.
[Host stores][interrupt-stores] · [Conditional notification][interrupt-notify]
· [Control-store contract][signal-control-store] · [Handler registration][async-registration]
· [Observer shutdown][async-shutdown]

Grouped observation retains event identity as well as values. ROCr deduplicates
shared native events; a group containing a signal without an event uses active
observation. Its asynchronous event loop keeps the last observed age keyed by
event object while handler arrays are compacted or reordered. Resetting that
age to one on each list change would make a previously signaled event appear
new again. This is runtime bookkeeping for native change history, not another
authoritative payload-completion value.
[Grouped waits][signal-group] · [Asynchronous age ownership][async-ages]

## Final users and reuse

An ordinary completed-use sequence has these ownership transitions:

1. Create the signal, event and their mappings; publish the initialized signal
   fields before any GPU command can consume them.
2. The producer completes the intended payload work, performs its required
   release, updates the signal value and issues its engine's notification.
3. The host checks the signal condition, uses the event only to wait for
   changes, and acquires the completed payload before consuming it.
4. Complete every device and host use before rearming or destroying the
   signal, its notification event or backing. A later dependent barrier is
   still a signal consumer even after the original producer completes.

[SDMA producer][sdma-notify] · [Host observation][interrupt-wait]
· [Dependent signal users](aql/barriers.md#native-signal-storage-and-last-consumers)

The SDMA output update precedes its mailbox/trap tail. Observing that value
can therefore finish the payload dependency without retiring notification
storage. Runtime retention protects the signal during a host wait;
`InterruptSignal` destruction returns an owned event to its pool, whose later
clear invokes native destruction. Neither step discovers arbitrary outstanding
device users. [Producer order][sdma-notify] · [Wait retention][interrupt-wait]
· [Pool return][interrupt-binding] · [Pool destruction][event-pool-type]

Native `DESTROY_EVENT` removes the event ID and wakes pending waiters with a
failure state; the thunk frees its `HsaEvent` only after successful native
removal. That operation does not drain GPU queues or complete a pending
mailbox write. Event-ID removal, notification-tail retirement, signal-storage
reuse and process-wide event-page release are separate boundaries.
[Native destruction][event-destroy] · [Wait failure][event-condition]
· [Thunk destruction][thunk-destroy]

Return to [GPU programming](README.md), [AQL](aql/README.md), or the
[primary source map](../sources.md).

[signal-abi]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/inc/amd_hsa_signal.h#L49-L77
[legacy-signal-object]: https://github.com/ROCm/ROCm-ComputeABI-Doc/blob/572ac3831ac5d30ac82f743d489a90bffba14a59/AMDGPU-ABI.md#signal-object-amd_signal_t
[signal-convert]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/inc/signal.h#L295-L326
[signal-storage]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/inc/signal.h#L146-L164
[signal-initial-value]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/signal.cpp#L129-L134
[signal-factory]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/hsa_ext_amd.cpp#L982-L1042
[signal-attributes]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h#L1339-L1425
[signal-comparisons]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/inc/signal.h#L126-L138
[signal-control-store]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/inc/signal.h#L354-L360
[signal-group]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/signal.cpp#L186-L248
[event-pool]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/interrupt_signal.cpp#L50-L92
[event-pool-type]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/inc/interrupt_signal.h#L69-L88
[interrupt-class]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/inc/interrupt_signal.h#L59-L67
[interrupt-binding]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/interrupt_signal.cpp#L94-L116
[interrupt-stores]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/interrupt_signal.cpp#L128-L140
[interrupt-notify]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/interrupt_signal.cpp#L377-L380
[interrupt-wait]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/interrupt_signal.cpp#L142-L211
[wait-api]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/inc/hsa.h#L2005-L2096
[event-age-version]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/inc/runtime.h#L554-L561
[async-registration]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp#L969-L987
[async-ages]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp#L2047-L2091
[async-shutdown]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/runtime.cpp#L3360-L3367
[thunk-create]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/libhsakmt/src/events.c#L69-L170
[thunk-destroy]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/libhsakmt/src/events.c#L173-L190
[thunk-wait]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/libhsakmt/src/events.c#L435-L511
[sdma-notify]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L616-L652
[sdma-fused]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L695-L795
[sdma-trap]: https://github.com/ROCm/rocm-systems/blob/f9ba16bbe70e365b2f59b268e847bef19ad9db6e/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L2978-L2986
[event-uapi]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/include/uapi/linux/kfd_ioctl.h#L258-L370
[event-constants]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_events.h#L35-L75
[event-slots]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_events.c#L49-L85
[event-zero]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_events.c#L200-L217
[event-backing]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_events.c#L267-L334
[event-create]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_events.c#L336-L383
[event-lookup]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_events.c#L97-L146
[event-delivery]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_events.c#L666-L744
[event-set]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_events.c#L586-L637
[event-register]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_events.c#L763-L789
[event-condition]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_events.c#L792-L822
[event-age-copy]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_events.c#L828-L864
[event-wait-cleanup]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_events.c#L884-L901
[event-wait]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_events.c#L903-L1016
[event-destroy]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_events.c#L220-L237
[irq-selection]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_device.c#L125-L189
[irq-cik]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/cik_event_interrupt.c#L88-L104
[irq-v9-cp]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_int_process_v9.c#L364-L385
[irq-v9-other]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_int_process_v9.c#L513-L534
[irq-partitions]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_int_process_v9.c#L578-L609
[irq-context-version]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_int_process_v9.c#L245-L261
[irq-context-filter]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_int_process_v9.c#L333-L344
[irq-v10-cp]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_int_process_v10.c#L208-L214
[irq-v10-other]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_int_process_v10.c#L327-L347
[irq-v11]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_int_process_v11.c#L350-L407
[irq-v12]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_int_process_v12_1.c#L337-L394
