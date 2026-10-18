# Windows SDMA ring submission

ROCr's native Windows SDMA path executes packets from a persistent ring,
publishes its byte frontier through `D3DKMTSubmitCommandToHwQueue`, and writes
a queue progress fence with a conditional interrupt. The host supplies the
packets; Windows accepts each frontier through a kernel call. This differs
from both a directly mapped KFD doorbell and an independently submitted
command-buffer address. [Native producer][publish] [Native submission][submit]

## Applicability and engine selection

The introducing ROCr change describes support for gfx12 and later and excludes
gfx11 and earlier. Its executable admission predicate is more specific:
`HwsInfo::hwsMask.sdma_queue`, an available SDMA engine, and hardware scheduling
enabled for the selected engine. Topology reports `NativeSDMASupported` and
`NativeSDMAEpilogueBytes` from those facts; queue construction checks the native
support bit together with its HWS selection. A compiler target alone does not
establish driver support. The cited change supplies no minimum released driver
version. [Change scope][change] [Topology][topology] [Construction][construct]

WKMI represents PM4, AQL, and native SDMA with separate `HwQueueKind` values.
Its `compute_schedid` is AQL-preferred, while `pm4_compute_schedid` preserves
the PM4 engine. SDMA uses an entry in `sdma_schedid`; these scheduler IDs are
translated to native node ordinals. Sharing the context-creation API does not
make the command representations interchangeable. [SDK representations][sdk]
[Context creation][context]

## Queue storage and native setup

ROCr allocates and clears a system-memory page with `Wkmi::kQueueObject`, which
marks the native allocation as `AmdQueueT`. The driver reports the consumed
byte frontier in this page's `amd_queue_t::read_dispatch_id`. This queue object
is separate from the command ring and the native progress fence. The source
identifies its allocation flag as a requirement for successful queue creation.
[Queue-object owner][construct] [Read-pointer view][read-pointer]

The native setup proceeds as follows:

1. Create a virtual context for the selected SDMA engine, with HWS enabled.
   `FillinContextPrivData(..., rocr_client=true)` selects the required client
   identity in the private context data.
2. Fill hardware-queue private data with `kHwQueueSdma`, ring GPU address and
   capacity, pointer operands, and the queue-object allocation handle.
3. `D3DKMTCreateHwQueue` returns the hardware queue and progress-fence handle,
   CPU mapping, and GPU address. Its private output also supplies a doorbell
   offset used as the SDMA interrupt context.
4. Keep the ring, queue object, and referenced command/payload allocations
   mapped and resident through their native consumers. ROCr's paging helper
   queues a GPU dependency when its required paging fence is outstanding; it
   does not wait synchronously on the submitting CPU.

[Context][context] [Hardware queue][create] [Paging dependency][paging]

## Reservation and publication

For command length `C`, the producer reserves `C + 32` bytes as one contiguous
ring span. The last 32 bytes belong to the native progress epilogue. ROCr's
allocator accounts for that tail when checking capacity and wrapping; its
publisher advances by the same total. The queue submitter fills the reserved
tail without advancing the frontier again. Counting only `C` would let the
next reservation overwrite the epilogue. [Reservation and release][reserve]
[Epilogue publication][publish]

For a reserved span `[start, end)`, the epilogue begins at
`(end - 32) % ring_capacity`. The reservation keeps the complete epilogue
inside the ring. ROCr emits a host release fence before the native call.
The native submission fields have these source-defined values:

| Field | ROCr native SDMA value |
| --- | --- |
| `hHwQueue` | The native SDMA hardware queue. |
| `HwQueueProgressFenceId` | The next queue-local progress ID, incremented once per submission and also embedded in the epilogue. |
| `CommandBuffer` | The persistent ring's GPU base address. |
| `CommandLength` | `static_cast<UINT>(end)`, the low 32 bits of the absolute byte frontier. |
| `FillinSdmaSubmitPrivData(..., WptrInBytes)` | The full 64-bit `end` byte frontier. |

[Submit builder][submit]

`CommandLength` here is not the length of an external IB, nor simply the new
span's byte count. The public wrapper establishes both frontier widths, but
does not expose the KMD's interpretation of their relationship across 32-bit
wrap. Ring modulo wrap, byte-frontier width, and progress-ID width are separate
conditions. The published producer does not demonstrate resetting either
monotonic frontier while work is live. [Submit builder][submit]
[Producer positions][publish]

## Progress epilogue and notification

ROCr emits an eight-DWORD `FENCE_CONDITIONAL_INTERRUPT`:

| DWORD | Emitted value |
| --- | --- |
| 0 | `0x80000105`: opcode 5, suboperation 1, DDW at bit 31, remaining fields zero. The builder identifies the SYSTEM field as zero. |
| 1–2 | Low/high DWORDs of the queue progress fence's GPU address. |
| 3–4 | Low/high DWORDs of the submission's progress ID. |
| 5–6 | Low/high DWORDs of the reference GPU address, set to the same progress-fence address. |
| 7 | Native doorbell offset used as interrupt context. |

The builder describes a 64-bit fence write followed by an interrupt when the
written value is greater than or equal to the referenced value. Using the
same address requests an interrupt on each progress update. The packet's ID
and `HwQueueProgressFenceId` must agree so the KMD's queue-fence reporting path
can advance the OS fence. This is a native queue completion protocol; it does
not use a shader's HSA event mailbox. [Packet builder][epilogue]

The [FENCE chapter](fence.md#conditional-interrupt-declaration) describes the
generation-specific representation. The [WDDM chapter](../wddm.md#hardware-queue-progress-and-command-retirement)
separately describes `RingBufferFenceRelease`, native retirement, and
[CPU event notification](../wddm.md#asynchronous-cpu-notification).
`read_dispatch_id` reports consumed ring bytes, while the progress fence
reports submission IDs. Neither is a payload cache policy or a substitute
for another engine's final-use dependency. [Read-pointer view][read-pointer]
[Native progress owner][epilogue]

## Dependency execution and failure handling

The cited runtime defaults `HSA_ENABLE_SDMA_GPU_POLL` off because its source
reports a KMD problem with GPU polling. Without the opt-in, a host worker
waits on leading HSA signal dependencies, replaces those `POLL_REGMEM` packets
with zeroes, and then submits the span. A span without leading polls can use
the direct host submit path when no earlier worker-owned span is pending.
This preserves a runtime dependency flow by involving a CPU thread; it is
not evidence for device-only dependency waits. The option's existence does
not establish that GPU polling is safe on any particular released driver.
[Option policy][poll-policy] [Host worker][worker] [Publication choice][publish]

The source advances its saved producer frontier before the direct native
submit, and increments the progress ID before that call. Failure is logged;
the worker also decrements its pending count after a failed submission. These
bookkeeping choices do not establish transactional rollback, safe command
replay, or retirement after rejection. Those properties require the native
acceptance and final-use contract. [Failure paths][publish] [Worker][worker]

The native producer emits commands inline. It does not supply a Windows
`INDIRECT` entry with VMID, context-save-area ownership, or a preemption and
return protocol for separately stored commands. Linux's
[scheduled IB protocol](command-buffers.md) has explicit driver-owned context
operands and cannot be transferred to this ring solely because both engines
decode an INDIRECT opcode. [Inline producer][reserve] [Native ring][publish]

[change]: https://github.com/ROCm/rocm-systems/commit/6b7847ef03b76c004f57b8cf40cdef9621ee6f53
[sdk]: https://github.com/ROCm/rocm-systems/blob/6b7847ef03b76c004f57b8cf40cdef9621ee6f53/shared/amdgpu-windows-interop/wkmi/wkmi.h
[topology]: https://github.com/ROCm/rocm-systems/blob/6b7847ef03b76c004f57b8cf40cdef9621ee6f53/projects/rocr-runtime/libhsakmt/src/dxg/topology.cpp#L929-L935
[construct]: https://github.com/ROCm/rocm-systems/blob/6b7847ef03b76c004f57b8cf40cdef9621ee6f53/projects/rocr-runtime/libhsakmt/src/dxg/wddm/queue.cpp#L1266-L1341
[read-pointer]: https://github.com/ROCm/rocm-systems/blob/6b7847ef03b76c004f57b8cf40cdef9621ee6f53/projects/rocr-runtime/libhsakmt/include/impl/wddm/queue.h#L321-L354
[context]: https://github.com/ROCm/rocm-systems/blob/6b7847ef03b76c004f57b8cf40cdef9621ee6f53/projects/rocr-runtime/libhsakmt/src/dxg/wddm/device.cpp#L500-L544
[create]: https://github.com/ROCm/rocm-systems/blob/6b7847ef03b76c004f57b8cf40cdef9621ee6f53/projects/rocr-runtime/libhsakmt/src/dxg/wddm/device.cpp#L1148-L1205
[paging]: https://github.com/ROCm/rocm-systems/blob/6b7847ef03b76c004f57b8cf40cdef9621ee6f53/projects/rocr-runtime/libhsakmt/include/impl/wddm/device.h#L205-L213
[reserve]: https://github.com/ROCm/rocm-systems/blob/6b7847ef03b76c004f57b8cf40cdef9621ee6f53/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L1959-L2070
[publish]: https://github.com/ROCm/rocm-systems/blob/6b7847ef03b76c004f57b8cf40cdef9621ee6f53/projects/rocr-runtime/libhsakmt/src/dxg/wddm/queue.cpp#L1368-L1439
[submit]: https://github.com/ROCm/rocm-systems/blob/6b7847ef03b76c004f57b8cf40cdef9621ee6f53/projects/rocr-runtime/libhsakmt/src/dxg/wddm/device.cpp#L1299-L1323
[epilogue]: https://github.com/ROCm/rocm-systems/blob/6b7847ef03b76c004f57b8cf40cdef9621ee6f53/projects/rocr-runtime/libhsakmt/src/dxg/wddm/queue.cpp#L1343-L1365
[poll-policy]: https://github.com/ROCm/rocm-systems/blob/6b7847ef03b76c004f57b8cf40cdef9621ee6f53/projects/rocr-runtime/libhsakmt/src/dxg/openclose.cpp#L709-L714
[worker]: https://github.com/ROCm/rocm-systems/blob/6b7847ef03b76c004f57b8cf40cdef9621ee6f53/projects/rocr-runtime/libhsakmt/src/dxg/wddm/queue.cpp#L1200-L1264
