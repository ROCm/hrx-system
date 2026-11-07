# Device-generated SDMA commands

A GPU shader can choose transfer addresses and lengths, write new SDMA packets
into a live queue, and notify its command processor while the shader remains
running. SDMA performs the transfer independently; the shader or another
consumer observes its completion and acquires the result. ROCm XIO and
rocSHMEM's Anvil supply public device producers for this model. Their host
owners create the native queues and mappings before launching device work.
[XIO packet construction][xio-compose] [XIO queue owner][xio-owner]
[Anvil publication][anvil-commit]

For a fixed transfer length, [indirect payload addresses](indirect-copy.md)
provide a separate path that ROCr selects for gfx125: the producer publishes
address slots consumed by already constructed copy packets. Generating new
commands also permits new lengths and operation sequences.

## Applicability and address reach

This chapter describes direct Linux KFD queues with 64-bit byte frontiers and
64-bit notification writes. The [queue-publication chapter](publication.md)
owns their pointer units, padding, native WPTR observers and removal protocol.
The individual [packet chapters](README.md) own each engine's representation;
generating a packet in a shader does not change its SDMA-IP requirements.

The publishing shader needs a complete set of device-accessible resources:

| Resource | Device use and ownership |
| --- | --- |
| Primary ring | Shader writes complete packets; SDMA fetches them. Ring space is returned by the native read frontier. |
| Read pointer, RPTR | SDMA reports consumed command bytes; the shader observes this frontier before overwriting their storage. |
| Write pointer, WPTR | The shader publishes the end of a complete contiguous prefix. Native queue management can observe it independently of a doorbell write. |
| Doorbell | The shader writes the same byte frontier through the queue's native notification mapping, using its native store width. |
| Source and destination | Packet operands name mappings accessible to the executing SDMA engine; shader access is an additional requirement when the shader produces or consumes the data. |
| Completion and credit cells | SDMA, the shader and downstream readers access these according to their distinct publication and reuse protocols. |

XIO copies ring, RPTR, WPTR, doorbell and private reservation/commit addresses
into a device handle. Those fields give the shader its addresses; the native
owner establishes actual reach. The handle creates no mapping and selects no
new engine.
[Device handle initialization][xio-handle]
[Engine and transfer-direction selection](engine-selection.md)

## GPUVM doorbells and native ownership

A CPU doorbell mapping alone does not establish shader access. In the pinned
libhsakmt GPUVM path, `hsakmt_fmm_allocate_doorbell` reserves a virtual address
and requests native `DOORBELL | WRITABLE | COHERENT` backing. It maps the CPU
alias at that address; `map_doorbell_dgpu` separately maps the object for GPU
access. If that path fails, `map_doorbell` can fall back to the host mapping.
Successful queue creation therefore does not, by itself, identify a successful
GPUVM notification mapping. [Special allocation][thunk-allocation]
[GPU mapping and host fallback][thunk-map]

KFD binds a `DOORBELL` allocation to the process's own doorbell slice. The
requested allocation size must equal `kfd_doorbell_process_slice`; the driver
obtains the slice address rather than taking an arbitrary physical address
from the caller. The non-MES size is the native doorbell width times the
process queue limit, rounded to `PAGE_SIZE`. The MES helper uses 1024 eight-byte
doorbells before page rounding. These are mapping extents, not ring capacities.
[Native allocation][native-allocation] [Slice calculation][native-slice]
[MES calculation][mes-slice]

The thunk's `DOORBELL_SIZE(gfxv)` selects eight bytes at `gfxv >= 0x90000` and
four bytes below it. For its SOC15 branch, KFD's returned doorbell offset
contains both the mapping selector and the queue's offset within that mapping;
the thunk separates them and returns the selected address. This branch does
not derive a queue's address by multiplying its queue ID by eight.
[Native width][thunk-width] [Returned offset interpretation][thunk-offset]

Doorbell storage has a different owner from ordinary payload memory. The
pinned thunk retains shared per-node process mappings and releases GPUVM
mappings when destroying that process-doorbell state. An application-level
queue handle can borrow such a mapping without owning the whole slice.
Its last publisher must finish before that mapping is released or reused.
[Thunk mapping retirement][thunk-retire]

Peer notification also has a distinct route. In the cited KFD peer branch,
MMIO and doorbell access uses PCIe even when peer VRAM is reachable through
xGMI. A peer doorbell attachment can create a separate scatter/gather BO;
`create_dmamap_sg_bo` copies `COHERENT`/`UNCACHED` BO flags only for USERPTR
backing. The original allocation's flags alone consequently do not describe
every peer notification PTE. Same-GPU reach, peer payload reach and peer
doorbell reach are separate mapping questions.
[Peer route and attachment][peer-attachment] [Attachment flags][peer-flags]

### Shader routing on GC 9.4.3, 9.4.4, and 9.5.0

These GC IPs route shader-originated notifications by the containing 32-byte
block: four eight-byte doorbell indices. Their native engine ranges begin at
QWORD index `0x100 + 10 * physical_engine`. An odd engine's first two indices
share a routing block with the preceding engine and reach that engine instead.
CPU-originated notifications are unaffected. [Routing defect and repair][routing-fix]

KFD assigns each queue `engine_base + (queue_slot >> 1)`, with another 512
indices for odd queue slots. The corrected allocator rounds `engine_base`
up to four indices first. For engine 1, the first mirror changes from indices
266–269 to 268–271; the second mirror has the same alignment. Each engine
uses four of its ten indices, leaving room for this adjustment.
[Corrected allocator][routing-allocator]

The assigned address belongs to the native queue. Rounding a shader's address
alone changes which doorbell it writes without changing the queue's installed
selection. Successful GPUVM mapping or CPU notification therefore does not
establish a valid shader notification route. The actual assignment and routing
rule determine that independently of the command-memory release sequence.

The upstream fix is present from Linux 7.3-rc7 and in these stable release
lines. Earlier kernels in each line retain the defect; a higher patch number
on a different, unfixed line does not establish support.

| Kernel release line | First fixed release |
| --- | --- |
| 6.6 | 6.6.160 |
| 6.12 | 6.12.113 |
| 6.18 | 6.18.57 |
| 7.2 | 7.2.11 |
| Mainline | 7.3-rc7 |

[Stable 6.6 backport][routing-stable66]
[Stable 6.12 backport][routing-stable612]
[Stable 6.18 backport][routing-stable618]
[Stable 7.2 backport][routing-stable72]
[Mainline allocator][routing-mainline]

Separately installed AMDGPU modules have their own patch history. The ROCm
7.2.4, AMDGPU 31.50, and AMDGPU 31.60 release branches still use the unaligned
allocator. AMDGPU 31.40.1's packaged module `6.19.14.31400100` also retains it.
Installing these modules on a fixed upstream kernel does not repair their
allocator. [ROCm allocator][routing-rocm] [AMDGPU 31.50][routing-amd3150]
[AMDGPU 31.60][routing-amd3160] [AMDGPU 31.40.1 package][routing-dkms314]

The native assignment also participates in checkpoint restore. KFD compares
the saved doorbell ID with the newly calculated ID and rejects a mismatch
with `-EINVAL`. An old checkpoint retaining an affected odd-engine address
cannot be accepted merely by changing the allocator: the restored process
would still hold its old notification address. [Restore consequence][routing-fix]

Queue completion does not identify how work was discovered. The CP scheduler's
`MAP_QUEUES` packet carries the WPTR address separately from the doorbell
selection, while manual installation reads the current WPTR before enabling
the ring. These are independent WPTR observers; their existence does not
establish a periodic-polling guarantee or validate a shader notification route.
[Installation and WPTR consumers](context.md#installation-and-write-pointer-consumers)

## Command memory and pre-WPTR visibility

XIO's device-producer ring allocation requests nonpaged, host-accessible,
executable, uncached memory and maps it for GPU access. Its private reserve
and commit frontiers also request uncached device allocations. These are
explicit source policies; a different ring mapping must establish its own
command-fetch visibility. [Ring allocation][xio-ring]
[Private-frontier allocation][xio-handle]

The word `COHERENT` is insufficient to reconstruct a GPU PTE. KFD translates
its allocation flags to BO flags, and the memory-controller implementation
interprets those flags together with the mapping and placement:

| Pinned Linux mapping path | Relevant translation |
| --- | --- |
| `gmc_v10_0_get_vm_pte`, `gmc_v11_0_get_vm_pte` | `COHERENT`, `EXT_COHERENT` or `UNCACHED` on the BO overrides the selected memory type to UC. |
| `gmc_v12_0_get_vm_pte` | The default is NC; `UNCACHED` overrides it to UC. This function has no corresponding `COHERENT`-to-UC branch. |
| `gmc_v9_0_get_coherence_flags` | UC is selected for explicit `UNCACHED`; other choices depend on exact GC IP, local/peer VRAM, APU/NUMA placement and extended coherence. |

[Allocation-to-BO flags][bo-flags] [GMC10 translation][pte10]
[GMC11 translation][pte11] [GMC12 translation][pte12]
[GMC9 translation][pte9]

The GC12.0 IP block selects separate GC12.1 callbacks for native GC12.1.0.
Their placement, coherence, revision and BUS_ATOMICS predicates are described
in the [GC12 mapping comparison](../pm4/atomics.md#gfx12-cache-policy-and-explicit-scope).

The GMC implementation name is not a shader target or SDMA version. NC denotes
non-coherent caching, not uncached access. The
[local-memory discussion](../recipes/local-memory.md#native-hbm-cache-policy)
retains the GC9 locality predicates and later per-page overrides.

Complete command stores must reach the SDMA command-fetch observation point
**before the canonical WPTR advances**. KFD queue restoration and selected
MQD polling can observe WPTR without waiting for the producer's doorbell;
notification cannot hide an incomplete published prefix. An SDMA cache packet
inside that prefix cannot repair its own missing command visibility.
[Native WPTR observers](publication.md#visibility-write-pointer-and-doorbell)

Four edges remain independently necessary:

1. Shader command writes become visible to the command processor before WPTR.
2. Source payload writes become visible to SDMA before it reads them.
3. SDMA completes the copy and required release before reporting completion.
4. The receiving shader observes completion and performs its required acquire
   before using the destination.

ROCr's async-copy contract explicitly separates SYSTEM-scoped sender release
and receiver acquire because DMA can sit outside the shader coherency domain.
A wait-counter instruction retires the instruction classes it names; it is not
by itself a universal replacement for these mapping/scope-dependent cache
operations. [Payload contract][copy-contract]
[Shader release/acquire sequences](../shader-memory.md#global-release-and-acquire-sequences)
[SDMA cache operations](cache.md)

## Reservations, ordered commit and lane progress

XIO's device producer separates a private reservation frontier `cachedWptr`
from `committedWptr` and the canonical WPTR. It reserves a packet group plus
any wrap tail using an agent-scoped CAS, checks the byte distance to RPTR,
and writes the complete padding/packet words before ordered commit. A
reservation ending at `E` is admitted only when `E - R < N`, for ring capacity
`N` and observed read frontier `R`. Reserved space has not yet been published.
[Capacity and reservation][xio-reserve] [Packet and padding stores][xio-place]

Its multi-producer `submitPacket` waits for `committedWptr == B`, where `B`
is the reservation start, then publishes the native WPTR, doorbell and private
commit frontier. The pinned implementation interposes wait-counter,
wave-barrier and compiler-fence operations. WPTR and commit stores are relaxed
agent-scoped atomics; the doorbell store is relaxed SYSTEM-scoped. These exact
operations belong to this source and its allocation/compiler premises.
[XIO ordered commit][xio-commit]

The related Anvil implementation keeps those publication stores **inside** the
loop branch that observes its turn. Its comment explains the SIMD progress
reason: the winning lane advances `committedWptr` before becoming inactive,
so another lane in the same wave can proceed. For `__GFX12__`, its waits name
`s_wait_loadcnt` and `s_wait_storecnt`; the other branch uses
`__builtin_amdgcn_s_waitcnt(0)`. XIO instead leaves its polling loop before
the publication stores. This is a concrete control-flow difference between
related consumers, not independent evidence of a portable multi-lane progress
guarantee. [Anvil commit loop][anvil-commit] [XIO commit loop][xio-commit]

XIO also supplies a one-thread producer without CAS/ordered-commit state. Its
submission writes WPTR before its explicit wait and then writes the doorbell.
That is different from the multi-producer sequence's pre-WPTR wait. The
single-publisher property removes reservation contention; it does not remove
the pre-WPTR visibility obligation. These source sequences cannot be exchanged
without preserving their memory and compiler premises.
[Single-producer reservation and submission][xio-single]

A device protocol additionally needs scheduling progress. A lane-level
reconvergence rule does not guarantee that an independently dispatched
producer or consumer workgroup can become resident. The
[pipeline progress discussion](../../interop/pipelines.md#progress-and-backpressure)
separates readiness dependencies from admission and final drain.

## Command credits, payload credits and drain

RPTR returns **command bytes**. A completion signal plus the receiver's acquire
transfers **payload visibility**. The last downstream reader returns
**payload storage**. These observations can occur at different times: SDMA may
consume the commands for many copies while a consumer still holds their
destinations. A batched producer can append when both ring capacity and the
chosen destination's independent reuse contract permit it.
[Ring and completion ownership](publication.md#completion-and-storage-ownership)
[Payload-slot credits](../../interop/pipelines.md)

XIO's `quietAll` polls RPTR to the last tracked publication, or the current
WPTR when no per-thread state is supplied. Its copy/signal operations have
separate signal stores and waits. An RPTR-based quiet operation alone contains
no receiving-shader payload acquire and no evidence that later readers have
finished. The required payload and downstream-credit protocols therefore
remain explicit. [RPTR wait][xio-quiet] [Copy/signal composition][xio-compose]
[Completion semantics](fence.md)

XIO's compile-time `XIO_SDMA_OSS7` branch can combine a copy and one increment
in `COPY_LINEAR_WAIT_SIGNAL_MI4`. When both a signal and a counter are requested,
the signal uses the fused slot and the counter remains a separate `ATOMIC`
packet. The other branch emits `COPY_LINEAR` and separate atomic increments.
This source selection does not make atomic signaling a requirement of every
device-generated copy; the [signal protocol](atomics.md) retains its own
packet and memory-route applicability. [Packet-group selection][xio-compose]

A complete single-publisher resident flow is:

1. The host establishes the native queue and all producer/SDMA mappings,
   initializes code, arguments and control storage, and launches the shader.
2. The shader computes a source, destination and legal length from live data.
   It obtains command capacity and permission to reuse the destination, then
   constructs complete packets and wrap padding in reserved space.
3. It establishes command and source visibility, publishes the new byte WPTR,
   then writes that frontier to the queue's notification address.
4. SDMA executes the selected copy/cache/completion sequence. The consumer
   observes completion, acquires and uses the result, then returns its payload
   credit after the final read. Further transfer choices can depend on that
   result without a host round trip.
5. Closing admission stops new reservations. Accepted transfers, readers and
   notification users finish; the publisher terminates and ring/control users
   retire before the host releases their mappings and native queue resources.

These steps compose the native publication and payload contracts; they are not
a claim that every packet supports every mapping or signal operation. XIO's
successful owner teardown destroys the native queue before freeing its ring
and private publication state. The creator still has to join its publishing
shader and other users before that teardown. Failed native removal retains
the separate ownership question described by
[queue retirement](publication.md#completion-and-storage-ownership).
[XIO resource release][xio-destroy]

An NPU or another GPU can contribute runtime requests through an established
shared-memory channel while a GPU owns SDMA publication. That composition
inherits both the channel's handoff/credit rules and this queue's mapping and
drain rules. Shared payload access alone supplies no NPU MMIO mapping or
permission to notify another device's queue.
[GPU/array channels](../recipes/gpu-npu.md#resident-programs-and-per-generation-ownership)
[Cross-device transport](../../interop/external-memory.md)

[xio-owner]: https://github.com/ROCm/rocm-xio/blob/cbe97e6392066bef7901121965ffadad19404da4/src/endpoints/sdma-ep/anvil.hip#L183-L292
[xio-ring]: https://github.com/ROCm/rocm-xio/blob/cbe97e6392066bef7901121965ffadad19404da4/src/endpoints/sdma-ep/anvil.hip#L198-L220
[xio-handle]: https://github.com/ROCm/rocm-xio/blob/cbe97e6392066bef7901121965ffadad19404da4/src/endpoints/sdma-ep/anvil.hip#L247-L291
[xio-destroy]: https://github.com/ROCm/rocm-xio/blob/cbe97e6392066bef7901121965ffadad19404da4/src/endpoints/sdma-ep/anvil.hip#L295-L319
[xio-reserve]: https://github.com/ROCm/rocm-xio/blob/cbe97e6392066bef7901121965ffadad19404da4/src/endpoints/sdma-ep/sdma_device.hpp#L288-L350
[xio-place]: https://github.com/ROCm/rocm-xio/blob/cbe97e6392066bef7901121965ffadad19404da4/src/endpoints/sdma-ep/sdma_device.hpp#L364-L394
[xio-commit]: https://github.com/ROCm/rocm-xio/blob/cbe97e6392066bef7901121965ffadad19404da4/src/endpoints/sdma-ep/sdma_device.hpp#L407-L443
[xio-single]: https://github.com/ROCm/rocm-xio/blob/cbe97e6392066bef7901121965ffadad19404da4/src/endpoints/sdma-ep/sdma_device.hpp#L490-L556
[xio-quiet]: https://github.com/ROCm/rocm-xio/blob/cbe97e6392066bef7901121965ffadad19404da4/src/endpoints/sdma-ep/sdma_device.hpp#L446-L470
[xio-compose]: https://github.com/ROCm/rocm-xio/blob/cbe97e6392066bef7901121965ffadad19404da4/src/endpoints/sdma-ep/sdma_device.hpp#L578-L639
[anvil-commit]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocshmem/src/sdma/anvil_device.hpp#L256-L309
[thunk-allocation]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/fmm.c#L1928-L1978
[thunk-map]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/queues.c#L274-L350
[thunk-width]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/queues.c#L41-L43
[thunk-offset]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/queues.c#L822-L849
[thunk-retire]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/src/queues.c#L223-L247
[native-allocation]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_chardev.c#L1161-L1186
[native-slice]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_doorbell.c#L49-L59
[mes-slice]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_mes.c#L32-L39
[peer-attachment]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gpuvm.c#L889-L947
[peer-flags]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gpuvm.c#L330-L359
[bo-flags]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gpuvm.c#L1774-L1779
[pte9]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/gmc_v9_0.c#L1046-L1160
[pte10]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c#L477-L521
[pte11]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/gmc_v11_0.c#L468-L512
[pte12]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/gmc_v12_0.c#L507-L548
[copy-contract]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h#L2090-L2153
[routing-fix]: https://github.com/torvalds/linux/commit/f145d5b3f740de05d85257acdc82f6877180f5c6
[routing-allocator]: https://github.com/torvalds/linux/blob/f145d5b3f740de05d85257acdc82f6877180f5c6/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c#L575-L600
[routing-stable72]: https://cdn.kernel.org/pub/linux/kernel/v7.x/ChangeLog-7.2.11
[routing-stable612]: https://cdn.kernel.org/pub/linux/kernel/v6.x/ChangeLog-6.12.113
[routing-rocm]: https://github.com/ROCm/amdgpu/blob/e5865b4307629a6ad1e44fe2343be68b08818915/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c#L474-L497
[routing-dkms314]: https://repo.radeon.com/amdgpu/31.40.1/ubuntu/pool/main/a/amdgpu-dkms/

[routing-stable66]: https://cdn.kernel.org/pub/linux/kernel/v6.x/ChangeLog-6.6.160
[routing-stable618]: https://cdn.kernel.org/pub/linux/kernel/v6.x/ChangeLog-6.18.57
[routing-mainline]: https://github.com/torvalds/linux/blob/v7.3-rc7/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c#L575-L600
[routing-amd3150]: https://github.com/ROCm/amdgpu/blob/48f2f4486a8dc6122195fa8755b6a1a6ee2ad3e2/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c
[routing-amd3160]: https://github.com/ROCm/amdgpu/blob/820212794d184710298efcecce42c0215bfc9c6a/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c
