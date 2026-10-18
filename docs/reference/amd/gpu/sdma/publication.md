# SDMA queue publication

SDMA executes a variable-length DWORD stream. A producer fills an available
ring extent, makes those command bytes visible, then advances the native write
pointer and rings the doorbell. The read pointer permits reuse of consumed
ring bytes; a separate command or signal protocol reports completion of the
transfer and its memory effects. ROCr's direct KFD queues and Linux's scheduled
IB queues have different publication owners. [ROCr publication][blit-publish]
[Linux ring commit][kernel-commit]

[Queue context](context.md) connects the KFD ring and control mappings to
the selected MQD, loader and scheduler. Its family tables distinguish the
legacy 32-bit pointer consumers from the modern 64-bit producer below.

[Windows native rings](windows.md) use a different publication protocol:
kernel-submitted byte frontiers, an `AmdQueueT` read-pointer page, and a
reserved progress-fence epilogue. A direct KFD doorbell store does not supply
that transport.

[Device-generated commands](device-publication.md) use these same native byte
frontiers with a shader as publisher. That path adds GPUVM notification reach,
shader command visibility, lane progress and downstream payload credits to the
queue's ordinary reservation and retirement rules.

## Transport and pointer units

The direct path described here is ROCr's KFD SDMA producer for its selected
GFX9–GFX12 ISA branches. The runtime chooses packet/cache variants by ISA and
transport; these predicates are not native SDMA IP numbers. Linux's
[scheduled IB backends](command-buffers.md#native-generation-and-transport)
are selected by native SDMA IP. The kernel pointer examples here use its
SDMA4.4.2 and SDMA6.x ring interfaces.
[ROCr selection][blit-selection] [SDMA4.4.2 pointers][pointers442]
[SDMA6.x pointers][pointers6]

The cited copy-engine initializer accepts a GPU agent and excludes
`HSA_PROFILE_FULL`. That actual profile check is more precise than its nearby
comment describing the path as dGPU-only. [Copy-engine admission][blit-admission]

| Quantity | Direct KFD SDMA meaning |
| --- | --- |
| Ring capacity `N` | Bytes. ROCr's copy engine allocates an 8 MiB ring with 4096-byte requested alignment. These are runtime allocation choices. |
| Command position | Byte offset `position & (N - 1)` in that power-of-two ring. Packets and padding occupy whole DWORDs. |
| Read pointer `R` | Monotonic 64-bit byte frontier reported by SDMA. |
| Write pointer `W` | Monotonic 64-bit byte frontier immediately after the published commands. |
| Doorbell value | The same new byte frontier `W`; it is neither a DWORD count nor the index of the last packet. |
| Native queue resources | The driver/thunk supplies the read-pointer address, write-pointer address and mapped doorbell address. Their addresses are not derived from the ring's GPU address. |

ROCr allocates and clears the ring before native queue creation, then takes
the three control addresses from the returned queue resources. The thunk
passes ring size in bytes to KFD and returns the mapped doorbell selected by
the driver. Its doorbell allocation uses eight bytes for `gfxv >= 0x90000`
and four bytes for older targets; the 64-bit producer described here is not
an older four-byte doorbell recipe. [Ring size][ring-size]
[Allocation and native resources][blit-create] [Thunk queue creation][thunk-create]
[Doorbell width][thunk-width]

The [KFD queue-storage contract](../architectures.md#kfd-queue-storage) requires
native BO mappings for the ring and control words. ROCr's later copy and
explicit-SDMA allocators request nonpaged system storage to preserve that
backing independently of pageable SVM payloads.

ROCr also exposes a distinct `SdmaQueue` path through its AMD queue-creation
extension. That interface explicitly assigns packet production, capacity,
wrap/padding and publication to a single producer or externally synchronized
callers. Its `hsa_queue_t::size` is **bytes**, unlike an AQL packet count.
It always requests `HSA_QUEUE_SDMA_BY_ENG_ID`, gated by KFD interface 1.17 or
later in the pinned implementation. [Extension contract][queue-contract]
[Size units][queue-size] [Native construction][queue-create]
[Engine admission][queue-engine]

The [engine-selection contract](engine-selection.md) distinguishes ordinary
SDMA, xGMI SDMA, and explicitly selected native engines. A mapped peer address
does not select a suitable engine or replace its transfer-direction checks.

## Reservation, construction and ordered commit

ROCr's ordinary asynchronous-copy producer keeps two private indices:
`cached_reserve_index_` allocates space to host producers, while
`cached_commit_index_` serializes their publication. Neither is the native
write pointer. `SubmitCommand` holds `reservation_lock_` while reserving the
complete command/trailer/padding extent and updating its accounting, then
constructs the commands outside that lock. The reservation primitive uses a
CAS on the private reserve index. [Submission reservation][blit-reserve]
[Space allocation][blit-space]

For a reservation beginning at `B` and ending at `E`, ROCr permits the write
only while the unsigned byte distance satisfies:

```text
E - R < N
```

It rejects a single reservation of `N` bytes or more. The strict inequality
leaves unused space to distinguish a full ring; with DWORD-granular commands,
at least one DWORD remains unoccupied. `R` supplies ring-space ownership, not
a payload visibility fence. [Capacity and wrapping][blit-space]
[Read-frontier check][blit-read]

After filling its extent, a producer waits until the private commit frontier
equals `B`. It then publishes `E` to the native write pointer and doorbell,
and finally advances the private commit frontier to `E`. A later producer
cannot expose a hole belonging to an earlier producer even if it finishes
constructing its packets first. [Ordered commit][blit-publish]

The explicit `SdmaQueue` extension has no equivalent private reservation
protocol. Its public index operations directly access KFD's canonical
hardware control words. The source explicitly excludes treating its CAS/add
methods as a multi-producer reservation mechanism: complete packets and
space/wrap handling precede publication. [Canonical pointer operations][queue-indices]
[Explicit publication][queue-publish]

## Wrap and padding

`BlitSdma` reserves each submission as one contiguous extent. If
`((B + C) & (N - 1)) < C`, for command size `C`, it first reserves the remaining
tail, fills it with zero DWORD NOPs, publishes that tail in commit order and
retries at offset zero. This condition includes an extent that would end
exactly at the boundary. The pad itself undergoes the same read-frontier
capacity check as useful commands. [Wrap decision][blit-space]
[Tail publication][blit-pad]

Submission-size padding is separate from wrapping:

| ROCr predicate | Selected padding |
| --- | --- |
| ISA 9.0.0 through 9.0.4, or exactly 9.0.12 | At least 256 bytes per submission. |
| DXG, when the complete submission is already at least the selected minimum size | Round the complete submission up to 64 bytes. |
| Other selected paths | No extra minimum/alignment padding in this calculation. |

For this padding, ROCr clears the bytes and writes a burst-NOP header with
`(padding_bytes / 4 - 1) << 16`. The count therefore describes the following
DWORDs, not bytes; [NOP representation and firmware selection](ordering.md#nop-representation-and-framing)
distinguish that field from caller-specific framing requirements. The wrap
tail instead remains a sequence of zero DWORD NOPs. These are the producer's
framing choices; they do not establish an
additional payload dependency. [Minimum-size predicate][blit-minimum]
[Padding calculation][blit-reserve] [Padding encoding][blit-completion]

Linux's scheduled ring owner uses another framing rule. It inserts NOPs at
commit to satisfy `ring->funcs->align_mask`: the cited SDMA4.4.2 ring uses
`0xff` (256 DWORDs, 1024 bytes), while SDMA6.x uses `0xf` (16 DWORDs, 64
bytes). Its ring writer wraps individual DWORD positions using `buf_mask`.
These scheduled-ring alignments are neither ROCr's submission minimum nor
the separate eight-DWORD IB-body padding described in
[command-buffer execution](command-buffers.md). [Commit padding][kernel-commit]
[Kernel ring writes][kernel-write] [SDMA4.4.2 ring policy][ring-policy442]
[SDMA6.x ring policy][ring-policy6]

## Visibility, write pointer and doorbell

For its system-memory copy ring, `BlitSdma` writes the native 64-bit WPTR,
then performs a release store of the same value to the doorbell. Its source
attributes the hardware ordering to **x86 with WB/coherent queue state**;
the release operation preserves compiler ordering as well. The code supplies
no universal CPU-memory-type or non-x86 publication rule merely by using
`volatile`. DXG and DTIF additionally invoke the thunk's doorbell operation.
[Publication implementation][blit-publish]

An enabled `sdma_wait_idle` runtime setting adds a wait for the wrapped RPTR
to reach the previous commit frontier before updating WPTR. The source gives
this as a write-pointer workaround without an ASIC predicate at this point;
it is not the unconditional publication sequence. [Optional idle wait][blit-publish]

`SdmaQueue::RingDoorbell` instead release-stores the canonical WPTR, executes
a C++ release fence and writes the doorbell. Its allocator requests executable
and uncached ring storage; device-memory placement additionally requires
Large BAR. These allocation flags belong to that runtime path and do not
identify every CPU mapping's cache type. [Explicit queue allocation][queue-allocation]
[Explicit publication][queue-publish]

The native WPTR is visible to queue management independently of a host's
private reservation state. KFD's v11 MQD builder enables
`F32_WPTR_POLL_ENABLE` and records the WPTR poll address; the v9 builder shown
here does not initialize those fields. The manual GC9.4.3 and GFX11 loading
paths also read the user WPTR while restoring a queue. Although the
`SdmaQueue` wrapper comment describes notification at the doorbell, these
native consumers preclude treating a deferred doorbell as a portable gate
that makes premature WPTR updates safe. The v11 MQD policy alone does not
prove that every loading path programs every poll register: the cited manual
loader does not explicitly write the poll-address registers. Complete command
bytes precede the canonical WPTR update. [v9 MQD][mqd9] [v11 MQD][mqd11]
[GC9.4.3 load][load943] [GFX11 load][load11]
[Wrapper comment and indices][queue-indices]

The [context family and consumer tables](context.md#installation-and-write-pointer-consumers)
extend this distinction through CIK, VI, v9, v10, v11, v12 and v12_1,
including MES's separate process-VA and MC-address WPTR inputs.

Command publication and payload cache maintenance solve different problems.
An in-stream cache command already depends on successful command fetch.
ROCr's asynchronous-copy contract separately requires system-level payload
coherence: normally a sending-device SYSTEM release before the copy and a
receiving-device SYSTEM acquire before destination use. Its packet sequence
selects HDP/GCR or per-packet scopes according to the actual runtime and
transport. The [cache chapter](cache.md) gives those predicates.
[Payload contract][copy-contract] [Runtime selection][blit-selection]

## Completion and storage ownership

The ordinary `SubmitCommand` flow joins dependency signals, emits selected
acquire work, copies the transfer body into the ring, emits selected release
work and updates the output signal. Depending on the platform's 64-bit atomic
support, that update is an atomic decrement or host-computed FENCE writes.
The latter is not a general concurrent signal RMW. With an interrupt signal,
a mailbox FENCE and TRAP follow the value update; NOP padding follows them.
[Dependency and body flow][blit-body] [Completion and notification][blit-completion]
[Atomic/fence selection][blit-signal] [Platform predicate][blit-platform]
[Signal protocols](atomics.md)

For ISA major 9, minor 0, stepping other than 10, pending dependencies select
a host signal-handler workaround before ring reservation. The handler later
resubmits without dependency polls. Thus even a successful asynchronous
submission can precede native ring publication. [Workaround predicate][blit-workaround]
[Deferred submission][blit-deferred]

The ISA-major-12/minor-at-least-5 linear-copy path instead selects a fused
wait/copy/signal body. It uses the same private reservation and commit helpers,
but its signal-bearing copy precedes the optional end timestamp and notification
tail. Its distinct signal protocol is described in [atomics](atomics.md);
the older `SubmitCommand` trailer is not a universal body layout.
[ISA predicate][blit-platform] [Fused-path selection][fused-selection]
[Fused publication][fused-publication]

| Observation | Ownership it establishes |
| --- | --- |
| Reservation or API submission return | Host ownership of space, or acceptance/publication of work; not transfer completion. |
| Native RPTR passes a ring extent | That extent is available for new command bytes under the ring producer's reuse rule. |
| Copy-completion signal reaches its required value, with the recipient's acquire | The copy's payload-completion handoff under the signal and mapping contract. |
| Remaining timestamp/notification commands and their software handling finish | Last use of their result, mailbox/event and handler-owned resources; this is separate from seeing an earlier signal-value store. |
| Successful native queue removal after producers and work are quiescent | End of the queue's native use of its ring/control storage. |

The physical storage borrowed by a packet stays mapped until its final user
finishes. Dependency cells remain live through their polls; payload remains
live through transfer completion and any later consumer; completion and
notification storage remain live through their respective final accesses.
ROCr makes this distinction concrete for gang copies: the leader polls each
member's signal, then performs its final decrement/store to avoid racing
signal destruction. Ring consumption alone cannot replace that join.
[Gang-signal lifetime][blit-gang]

A complete direct-queue sequence is therefore:

1. Create native queue mappings and initialize the ring, payload and control
   storage; establish the payload's producer release.
2. Reserve available ring bytes, including wrap and submission padding, and
   finish the dependency, transfer, cache, completion and notification words.
3. Commit in reservation order; publish the complete extent's byte WPTR, then
   the same byte frontier through the native doorbell operation.
4. Acquire the required completion before consuming the destination. Retain
   every other borrowed object until its own last consumer or notification
   owner finishes; use RPTR separately for ring-byte reuse.
5. Stop further producers before queue teardown. Remove the native queue
   before releasing ring/control backing.

On the successful path, ROCr destroys the native copy queue before freeing
its ring and internal signals. The thunk frees its queue object only after a
successful destroy ioctl. KFD validates and retains references to ring and
pointer mappings at construction and releases queue buffers during teardown.
These successful paths do not establish that a failed queue-removal operation
permits backing release or that queue destruction completes an application's
unfinished payload protocol. [ROCr destruction][blit-destroy]
[Thunk destruction][thunk-destroy] [KFD mapping owner][mapping-owner]
[KFD teardown][native-destroy]

## Scheduled IB publication

PAL's Linux DMA submitter passes command chunks to `AddIb`; that boundary
converts their DWORD count to the DRM `ib_bytes` quantity and requests memory
synchronization on the first IB. The kernel owns primary-ring reservation,
VM/context setup, IB entry/return, applicable cache work and the submission
fence. PAL waits for that native submission fence when waiting for the queue
to become idle. This is a different owner flow from writing a KFD USER ring.
[PAL DMA submission][pal-submit] [PAL IB description][pal-ib]
[PAL completion wait][pal-wait] [Ring reservation and VM setup][kernel-reserve]
[Kernel wrapper][kernel-wrapper]

Linux's generic ring commit pads, executes `mb()`, then invokes the engine's
WPTR setter. For both cited SDMA implementations, software `ring->wptr` is
in DWORDs; the setter publishes `ring->wptr << 2` to native WPTR/doorbell,
and read-pointer access converts hardware bytes back with `>> 2`. The
non-doorbell branch uses kernel register access. That branch and the
kernel-owned submission trailer are not operations inherited by a direct
USER queue. [Kernel commit][kernel-commit] [SDMA4.4.2 publication][pointers442]
[SDMA6.x publication][pointers6]

[ring-size]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L75-L78
[blit-create]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L220-L259
[blit-admission]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L156-L172
[blit-selection]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp#L852-L903
[queue-contract]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h#L3846-L3867
[queue-size]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h#L3925-L3929
[queue-create]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_sdma_queue.cpp#L124-L181
[queue-engine]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp#L911-L917
[queue-allocation]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_sdma_queue.cpp#L88-L110
[queue-publish]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_sdma_queue.cpp#L229-L249
[queue-indices]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_sdma_queue.cpp#L252-L319
[blit-minimum]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L174-L185
[blit-reserve]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L499-L518
[blit-space]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L1954-L1992
[blit-publish]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L1998-L2032
[blit-pad]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L2059-L2084
[blit-read]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L2087-L2094
[blit-body]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L521-L577
[blit-workaround]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L262-L266
[blit-deferred]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L421-L442
[fused-selection]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp#L1419-L1426
[fused-publication]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L714-L803
[blit-signal]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L469-L497
[blit-platform]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L187-L204
[blit-completion]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L614-L663
[blit-gang]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L589-L611
[blit-destroy]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L282-L306
[copy-contract]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h#L2090-L2136
[thunk-width]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/queues.c#L41-L60
[thunk-create]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/queues.c#L765-L851
[thunk-destroy]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/queues.c#L920-L941
[mqd9]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v9.c#L545-L574
[mqd11]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v11.c#L450-L481
[load943]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gc_9_4_3.c#L59-L126
[load11]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gfx_v11.c#L345-L412
[mapping-owner]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_queue.c#L196-L277
[native-destroy]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_process_queue_manager.c#L557-L586
[kernel-commit]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_ring.c#L160-L189
[kernel-write]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_ring.h#L493-L523
[pointers442]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v4_4_2.c#L210-L295
[pointers6]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v6_0.c#L163-L239
[ring-policy442]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v4_4_2.c#L2131-L2138
[ring-policy6]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v6_0.c#L1740-L1747
[pal-submit]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/os/amdgpu/amdgpuQueue.cpp#L1400-L1467
[pal-ib]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/os/amdgpu/amdgpuQueue.cpp#L1938-L1983
[pal-wait]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/os/amdgpu/amdgpuQueue.cpp#L1640-L1672
[kernel-wrapper]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_ib.c#L231-L349
[kernel-reserve]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_ib.c#L199-L229
