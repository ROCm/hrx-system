# Windows command submission and fences

Windows Display Driver Model (WDDM) submission publishes a GPU command-buffer
address to a native context or hardware queue. The native transport supplies
its scheduling and completion protocol around the engine's commands. Command
acceptance, an in-body payload marker, and native command-storage retirement
are separate observations. [Context submission][submit]
[Hardware-queue submission][submit-hardware]

## Contexts, engines, and submission representation

ROCr selects a native node from the requested engine, creates a virtual
context, and sets `HwQueueSupported` when WKMI reports hardware scheduling
enabled for that engine. Its software-scheduled path creates a monitored
fence; its hardware-scheduled path receives a queue progress fence from
`D3DKMTCreateHwQueue`. These are native setup decisions, independent of the
PM4 or SDMA packets placed in the command buffer. [Context creation][context]
[Software queue][software-queue] [Hardware queue][hardware-queue]

| Submission path | Command representation | Completion publication |
| --- | --- | --- |
| `D3DKMTSubmitCommand` | `Commands` is a GPU virtual address; `CommandLength` is bytes. `BroadcastContextCount` is at least one and identifies entries in `BroadcastContext`. | The structure has no hardware-queue progress value. ROCr separately queues a monitored-fence signal on the same context. |
| `D3DKMTSubmitCommandToHwQueue` | `hHwQueue` identifies the queue; `CommandBuffer` is a GPU virtual address and `CommandLength` is bytes. | `HwQueueProgressFenceId` accompanies that submission. The node's fence-release capability determines the update owner. |

Both calls carry host-side driver-private data and its byte extent. ROCr
zero-initializes the native structures, uses one context for software
submission, and fills WKMI private data for the selected transport.
[Context fields][submit] [Hardware-queue fields][submit-hardware]
[ROCr software submission][software-submit]
[ROCr hardware submission][hardware-submit]

ROCr's [native SDMA ring path](sdma/windows.md) uses the hardware-queue call
with a distinct private protocol: `CommandBuffer` is the persistent ring
base, and the private data carries a 64-bit byte frontier. Its
`CommandLength` field contains the low 32 bits of that frontier. It also
reserves an inline progress-fence epilogue. The independent command-buffer
length interpretation in the table above does not describe this ring path.

ROCr's pre-submit path also queues a paging-fence dependency when the observed
paging value is behind its required value. `WaitPagingFence` calls `GpuWait`;
it does not synchronously wait on the submitting CPU. Mappings and residency
remain prerequisites for command, executable, argument, and payload access.
[Paging dependency][paging] [Compute caller][compute-submit]

## Software submission and accepted work

The cited `WDDMDevice::SubmitToSwQueue` performs two native calls in order:

```text
publish command bytes and establish their native memory dependencies
  → D3DKMTSubmitCommand(context, command GPU address, byte length)
  → after success, D3DKMTSignalSynchronizationObjectFromGpu(context, fence, value)
  → observe the requested completion through the fence's native wait protocol
  → retire each resource after its final command, shader, or transfer consumer
```

ROCr's `GpuSignal` supplies the context, object handles, and corresponding
64-bit monitored-fence values. Successfully queuing that operation does not
mean the GPU has already reached the value. [Signal wrapper][signal]
[Signal representation][signal-fields]

The two calls create a partial-progress case: command submission can succeed
and the separate signal call can fail. ROCr returns `false` for either failure
and performs no intervening cancellation. Its compute caller then returns
before advancing its saved completion point. The combined result therefore
does not identify whether commands were accepted, and failure to publish the
expected fence does not establish that their storage is unused.
[Submission order][software-submit] [Caller bookkeeping][compute-submit]

This source flow establishes the need to retain accepted-work ownership when
completion publication fails. It does not establish a recovery operation that
revokes all device access. Likewise, the context-destruction documentation
describes releasing a context but does not define a failed call as proof of
retirement. Those outcomes cannot substitute for a successful final-use
observation. [Context destruction][destroy-context]

## Hardware-queue progress and command retirement

The hardware-queue submission contains both its command address and progress
value. The downstream `DXGKARG_SUBMITCOMMANDTOHWQUEUE` contract assigns progress
updates according to `DXGK_NODEMETADATA_FLAGS::RingBufferFenceRelease`:

| Capability | Progress-fence update owner |
| --- | --- |
| `RingBufferFenceRelease = 0` | The user-mode driver (UMD) inserts the update as the last instruction in its DMA buffer. A kernel submission uses the KMD signaling path. |
| `RingBufferFenceRelease = 1` | The driver/GPU updates progress after neither GPU nor CPU uses the DMA buffer. The native implementation determines the mechanism. |

Only the value-one branch explicitly guarantees that both CPU and GPU use of
the DMA buffer has ended. The value-zero branch specifies the terminal
instruction and its owner; it does not by itself supply that stronger
retirement guarantee. [Progress-fence contract][native-submit]

`ContextSchedulingSupported`, `RingBufferFenceRelease`, and
`UserModeSubmission` are separate node capabilities. The latter is documented
from Windows 11 24H2; it is not implied by selecting a hardware queue.
`MaxInFlightHwQueueBuffers` bounds buffers submitted by the hardware scheduler
to the KMD, with zero meaning no limit. It is not a byte capacity for a user
ring. [Node capabilities][node-flags]

Private submission metadata has a different lifetime from GPU command
backing. ROCr frees its host allocation after the native submission call.
Separately, the hardware-queue DDI permits the downstream KMD private-data
buffer to be freed when that callback returns; its documentation contrasts
this with the longer WDDM 2.0–2.3 lifetime. These are observations at different
API boundaries, not proof that both pointers name the same allocation.
[ROCr metadata owner][hardware-submit] [KMD metadata lifetime][native-submit]

The command graph retains its own shader/transfer joins and cache operations.
A marker in its final body does not by itself establish the transport's final
use of command storage. The [PM4 command-buffer chapter](pm4/command-buffers.md#native-submission-retirement)
composes those owners with indirect calls and chains; the
[SDMA command-buffer chapter](sdma/command-buffers.md) explains the distinct
Linux submission owners.

## Monitored-fence mappings and width

The WDDM 2.x context-monitoring protocol exposes a logically 64-bit fence and
aligned CPU/GPU addresses. `DXGK_VIDSCHCAPS::No64BitAtomics` identifies whether
the underlying GPU update is CPU-visible atomic at 64 bits or only 32 bits.
For 32-bit updates, the OS handles wraparound and requires outstanding wait
and signal values to remain within `UINT_MAX/2` of the last signaled value.
[Atomic-width capability][scheduler-caps] [Fence creation][fence-fields]

| View or field | Access and interpretation |
| --- | --- |
| `FenceValueCPUVirtualAddress` | Normally read-only user mapping. A documented 32-bit-platform exception permits an interlocked 64-bit read through a writable mapping; it does not make CPU stores the signaling API. |
| `FenceValueGPUVirtualAddress` | GPU read/write mapping, subject to the native coherency/addressing contract. The field documentation excludes direct GPU writes on devices without CPU cache coherency. |
| `MonitoredFence.EngineAffinity` | Bits select physical adapters in an LDA link on which the GPU address is committed. Zero selects all adapters; these are not individual SDMA engine indices. |

[Mapping and affinity fields][fence-fields]

The context-monitoring guide describes WB CPU mappings on I/O-coherent
platforms and UC mappings otherwise. It also distinguishes the IOMMU GPU path,
which uses the CPU virtual address for GPU access. CPU signaling uses the
native CPU signal operation. A GPU engine unable to write the fence uses a
queued software signal. Address reachability alone establishes none of those
mapping, atomicity, or notification conditions. [Context monitoring][monitoring]

For a legacy GPU wait, the UMD flushes pending commands and queues the wait;
Dxgkrnl withholds subsequent command buffers until the value is satisfied.
The submitting thread can continue, but this scheduling dependency is not a
native device-only wait. CPU waits can block or arrange event notification.
[Legacy wait ownership][monitoring]

## Asynchronous CPU notification

`D3DKMTWaitForSynchronizationObjectFromCpu` accepts a device handle, an array of
monitored-fence handles, corresponding 64-bit values, and `hAsyncEvent`. A nonnull
event requests notification when the condition is satisfied; a null event makes
the call wait. With `WaitAny = 0`, every supplied fence must reach at least its
requested value. The one-fence form observes an exact point without a host polling
loop. Native registration can fail with an NTSTATUS result independently of the
commands already submitted to that fence. [CPU wait][cpu-wait]
[Event and threshold fields][cpu-wait-fields]

The event is a wake destination, not fence-value storage or an execution-result
record. An auto-reset event releases one waiter and consumes its signaled state;
a manual-reset event remains signaled until its owner resets it. Repeated signals
can coalesce. Sharing an event across fences therefore requires resampling the
participating fences after each wake rather than identifying a completion from
the event alone. The caller owns the event through pending native delivery and
separately joins any host event-loop callbacks. The DDI has no cancellation token
or callback-drain operation; observing a fence value does not count outstanding
event signals. [Win32 event semantics][event-objects]

Microsoft's Direct3D 12 device-removal contract signals monitored fences to
`UINT64_MAX`, which also satisfies pending wait events. That value is a failure
sentinel rather than successful command execution. Removal notification and
successful resource retirement remain separate observations. This contract does
not turn an arbitrary user-mode validation or publication error into a native
device-removal event. [Removal and waited events][removed-events]

This mechanism belongs to the monitored-fence scheduler protocol. It does not
require a command processor interrupt payload, a shader dispatch, or a directly
mapped user-mode command ring, and it does not describe intermediate notifications
inside a still-running persistent invocation.

## Native GPU-fence protocol

Microsoft's native-fence guide describes the extended protocol for WDDM 3.2 /
Windows 11 24H2: GPU-side waits, conditional interrupts for CPU waiters, and
optional GPU-local storage. The capability-field documentation dates
`NativeGpuFence` to WDDM 3.1 and exposes it through
`D3DKMT_WDDM_3_1_CAPS::NativeGpuFenceSupported`. Field availability and the
complete enabled protocol retain their separate source attributions. The KMD
must query OS enablement of `DXGK_FEATURE_NATIVE_FENCE` before advertising
support. [Capability declaration][scheduler-caps] [Feature enablement][native-enable]

The protocol has two separate 64-bit quantities:

| Quantity | Ownership and meaning |
| --- | --- |
| `CurrentValue` | Signaled progress, with untorn CPU/GPU-visible 64-bit updates. User mode reads it; kernel mode and the GPU have their defined write paths. |
| `MonitoredValue` | OS-maintained least pending CPU wait minus one, or `UINT64_MAX` with no CPU waiter. The GPU reads it; user mode does not access it. |

[Native-fence representation][native-design]

The context management processor raises notification when updated progress
exceeds the monitored value. Reliable wakeup also depends on the OS/KMD
monitored-value update handshake: GPU memory ordering and CPU resampling
cover signals racing with waiter changes. This is a native notification
protocol in addition to the progress write. Its documented placement permits
system or GPU-local storage according to fence type; cross-adapter current
values remain in system memory. [Storage and notification ordering][native-ordering]

[submit]: https://github.com/MicrosoftDocs/windows-driver-docs-ddi/blob/7515063cea4c9e98db6a92986c5b4ddb0463fd16/wdk-ddi-src/content/d3dkmthk/ns-d3dkmthk-_d3dkmt_submitcommand.md#L45-L85
[submit-hardware]: https://github.com/MicrosoftDocs/windows-driver-docs-ddi/blob/7515063cea4c9e98db6a92986c5b4ddb0463fd16/wdk-ddi-src/content/d3dkmthk/ns-d3dkmthk-_d3dkmt_submitcommandtohwqueue.md#L49-L78
[context]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/dxg/wddm/device.cpp#L499-L537
[software-queue]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/dxg/wddm/queue.cpp#L74-L139
[hardware-queue]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/dxg/wddm/device.cpp#L1030-L1117
[software-submit]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/dxg/wddm/device.cpp#L855-L888
[hardware-submit]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/dxg/wddm/device.cpp#L1134-L1164
[paging]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/include/impl/wddm/device.h#L201-L209
[compute-submit]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/dxg/wddm/queue.cpp#L628-L658
[signal]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/dxg/wddm/device.cpp#L568-L583
[signal-fields]: https://github.com/MicrosoftDocs/windows-driver-docs-ddi/blob/7515063cea4c9e98db6a92986c5b4ddb0463fd16/wdk-ddi-src/content/d3dkmthk/ns-d3dkmthk-_d3dkmt_signalsynchronizationobjectfromgpu.md#L54-L73
[destroy-context]: https://github.com/MicrosoftDocs/windows-driver-docs-ddi/blob/7515063cea4c9e98db6a92986c5b4ddb0463fd16/wdk-ddi-src/content/d3dkmthk/nf-d3dkmthk-d3dkmtdestroycontext.md#L48-L70
[native-submit]: https://github.com/MicrosoftDocs/windows-driver-docs-ddi/blob/7515063cea4c9e98db6a92986c5b4ddb0463fd16/wdk-ddi-src/content/d3dkmddi/ns-d3dkmddi-_dxgkarg_submitcommandtohwqueue.md#L48-L90
[node-flags]: https://github.com/MicrosoftDocs/windows-driver-docs-ddi/blob/7515063cea4c9e98db6a92986c5b4ddb0463fd16/wdk-ddi-src/content/d3dkmdt/ns-d3dkmdt-_dxgk_nodemetadata_flags.md#L45-L75
[scheduler-caps]: https://github.com/MicrosoftDocs/windows-driver-docs-ddi/blob/7515063cea4c9e98db6a92986c5b4ddb0463fd16/wdk-ddi-src/content/d3dkmddi/ns-d3dkmddi-_dxgk_vidschcaps.md#L89-L108
[fence-fields]: https://github.com/MicrosoftDocs/windows-driver-docs-ddi/blob/7515063cea4c9e98db6a92986c5b4ddb0463fd16/wdk-ddi-src/content/d3dukmdt/ns-d3dukmdt-_d3dddi_synchronizationobjectinfo2.md#L104-L130
[monitoring]: https://github.com/MicrosoftDocs/windows-driver-docs/blob/2d03b8b58143ec94b2ae56e2ddc11ec64133fb2c/windows-driver-docs-pr/display/context-monitoring.md#L15-L58
[native-enable]: https://github.com/MicrosoftDocs/windows-driver-docs/blob/2d03b8b58143ec94b2ae56e2ddc11ec64133fb2c/windows-driver-docs-pr/display/native-gpu-fence-objects.md#L133-L149
[native-design]: https://github.com/MicrosoftDocs/windows-driver-docs/blob/2d03b8b58143ec94b2ae56e2ddc11ec64133fb2c/windows-driver-docs-pr/display/native-gpu-fence-objects.md#L14-L46
[native-ordering]: https://github.com/MicrosoftDocs/windows-driver-docs/blob/2d03b8b58143ec94b2ae56e2ddc11ec64133fb2c/windows-driver-docs-pr/display/native-gpu-fence-objects.md#L63-L130

[cpu-wait]: https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/d3dkmthk/nf-d3dkmthk-d3dkmtwaitforsynchronizationobjectfromcpu
[cpu-wait-fields]: https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/d3dkmthk/ns-d3dkmthk-_d3dkmt_waitforsynchronizationobjectfromcpu
[event-objects]: https://learn.microsoft.com/en-us/windows/win32/sync/event-objects
[removed-events]: https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device5-removedevice
