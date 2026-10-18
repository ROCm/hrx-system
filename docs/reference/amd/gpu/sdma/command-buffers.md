# SDMA command-buffer execution

Linux's scheduled SDMA path provides a complete first-level indirect-buffer
(IB) workflow: initialize external command storage, enter it from the primary
ring, return to that ring, signal completion and retire the storage. The
driver owns the context operands and surrounding memory operations. The
inspected ordinary KFD USER producers instead copy commands into their primary
ring. Their inline publication contract does not establish external-command
fetch or retirement. [Scheduled owner][schedule] [ROCr producer][rocr-inline]

The [queue-publication chapter](publication.md) describes primary-ring
capacity, wrapping, pointer units, host visibility and final storage use for
both transports. Those rules surround the indirect-body protocol described
here. The [Windows submission chapter](../wddm.md) separately describes WDDM
command addresses, completion publication and native retirement; those native
contracts are not supplied by the Linux ring wrapper below.

[Windows native SDMA rings](windows.md) publish a persistent ring frontier and
append their own progress epilogue. That public producer supplies no Windows
IB context-save operands or external-command retirement protocol.

[Conditional execution](conditional.md) describes `COND_EXE` ranges and the
native submission condition surrounding an IB. A copy predicate and a
driver-owned preemption condition have different storage and completion owners.

## Native generation and transport

Linux selects the scheduled backend using native discovery or legacy ASIC
registration. The following table covers its SI, CIK and SDMA2.4–7.1 emitters.
The backend name is not a compiler GFX target, and a backend can serve several
physical IP revisions. The final column is the primary-ring commit alignment
in DWORDs, independently of the entry packet and external body.
[Native discovery][routing] [Legacy SI selection][si-route]
[Legacy CIK selection][cik-route] [Legacy VI selection][vi-route]

| Native selection | Scheduled backend | Entry DWORDs | Context operands | Ring commit alignment, DWORDs |
| --- | --- | ---: | --- | ---: |
| SI DMA1.0: Tahiti, Pitcairn, Verde, Oland, Hainan | `si_dma` | 3 | None. | 16 |
| CIK SDMA2.0: Bonaire, Hawaii, Kaveri, Kabini, Mullins | `cik_sdma` | 4 | None. | 16 |
| Topaz SDMA2.4 | `sdma_v2_4` | 6 | Zero. | 16 |
| SDMA3.0 and 3.1 | `sdma_v3_0` | 6 | Zero. | 16 |
| 4.0.0, 4.0.1, 4.1.0, 4.1.1, 4.1.2, 4.2.0, 4.2.2, 4.4.0 | `sdma_v4_0` | 6 | Zero. | 256 |
| 4.4.2, 4.4.4, 4.4.5 | `sdma_v4_4_2` | 6 | Zero. | 256 |
| 5.0.0, 5.0.1, 5.0.2, 5.0.5 | `sdma_v5_0` | 6 | Driver CSA owner. | 16 |
| 5.2.0, 5.2.1, 5.2.2, 5.2.3, 5.2.4, 5.2.5, 5.2.6, 5.2.7 | `sdma_v5_2` | 6 | Driver CSA owner. | 16 |
| 6.0.0, 6.0.1, 6.0.2, 6.0.3, 6.1.0, 6.1.1, 6.1.2, 6.1.3, 6.1.4, 6.4.0 | `sdma_v6_0` | 6 | Driver CSA owner. | 16 |
| 7.0.0, 7.0.1 | `sdma_v7_0` | 6 | Driver CSA owner. | 16 |
| 7.1.0 | `sdma_v7_1` | 6 | Driver CSA owner. | 16 |

The packet emitters supply the entry sizes and context operands:
[SI][ib-si], [CIK][ib-cik], [2.4][ib24], [3.0/3.1][ib3],
[4.0 backend][ib4], [4.4.2 backend][ib442], [5.0][ib5], [5.2][ib52],
[6.x][ib6], [7.0][ib7] and [7.1][ib71]. The ring callback tables independently
supply `align_mask`: `0xff` for both 4.x backends and `0xf` for the others.
[SI table][ring-si] [CIK table][ring-cik] [2.4 table][ring24]
[3.x table][ring3] [4.0 tables][ring4] [4.4.2 tables][ring442]
[5.0 table][ring5] [5.2 table][ring52] [6.x table][ring6]
[7.0 table][ring7] [7.1 table][ring71]

SDMA3.1 shares the 3.0 implementation. Native 4.4.0 routes through
`sdma_v4_0`; the separate `sdma_v4_4.c` file supplies its RAS helpers.
[3.x registration][versions3] [4.4 helpers][ras44]
[Native discovery][routing]

## Packet representation

All listed emitters select a 32-byte-aligned external body and arrange for
the entry packet to end at an eight-DWORD primary-ring boundary. The legacy
encodings differ in both packet length and field placement.

### SI: three DWORDs

`DMA_IB_PACKET` places opcode `0x4` in header bits 31:28 and the job VMID in
bits 23:20. The emitter passes zero for the macro's low 20-bit operand.
Word 1 contains body address bits 31:5 with low bits zero. Word 2 combines
address bits 39:32 in bits 7:0 with the direct body DWORD count in bits
31:12; bits 11:8 remain zero. The emitted address is therefore 40 bits,
unlike the later separate full address words. This packed count is not the
header macro's low operand. [SI builder][ib-si] [SI definitions][fields-si]

### CIK: four DWORDs

`SDMA_PACKET` places opcode `4` in bits 7:0, suboperation zero in bits 15:8
and VMID in bits 19:16. Words 1–2 contain the low/high body byte address,
with the low five address bits cleared. Word 3 receives `ib->length_dw`
directly. This emitter has no CSA words. Its direct count assignment alone
does not establish a field-width limit from a later generated layout.
[CIK builder][ib-cik] [CIK definitions][fields-cik]

### SDMA2.4 and later: six DWORDs

The Iceland, Tonga, Vega, Navi, SDMA6 and SDMA7.1 packet definitions retain
these word offsets and direct 20-bit count. SDMA7.0 uses the SDMA6 packet
header. The scheduled builders emit only the job VMID in addition to the
opcode; the newer named PRIV field stays zero.
[Iceland fields][fields24] [Tonga fields][fields3]
[Vega fields][fields442] [Navi fields][fields5]
[SDMA6 fields][fields6] [SDMA7.0 header selection][header7]
[SDMA7.1 fields][fields71]

| Word | Encoding and units |
| --- | --- |
| 0 | [INDIRECT opcode 4][opcode] in bits 7:0; suboperation bits 15:8 zero; job VMID in bits 19:16; other bits zero. Navi, SDMA6 and SDMA7.1 define bit 31 PRIV; the Iceland, Tonga and Vega layouts do not name it. |
| 1–2 | Absolute GPU byte address, low then high DWORD. The base is 32-byte aligned; the emitter clears the low five bits. |
| 3 | Direct command-body DWORD count in bits 19:0. This is neither bytes nor count-minus-one. The encoded count is distinct from allocator and submission limits. |
| 4–5 | Context-save-area (CSA) address, low then high DWORD. The 2.4, 3.x and 4.x emitters write zero; 5.x, 6.x and 7.x obtain it from the driver's CSA owner. |

The count field can represent `0xfffff` DWORDs. If a producer pads its body
to eight DWORDs, its largest representable padded count is `0xffff8`.
Neither quantity overrides that producer's allocation or submission limit.
The inspected INDIRECT layouts have no PM4 VALID or CHAIN fields.

## Entry, body and submission framing

For primary-ring DWORD position `W`, packet length `L` and NOP count `P`,
the scheduled entry emitters satisfy `(W + P + L) mod 8 = 0`:

| Entry form | Padding before the entry |
| --- | --- |
| SI, three DWORDs | Append single NOPs until `W mod 8 == 5`. |
| CIK, four DWORDs | `(4 - W) & 7` DWORDs. |
| Six DWORDs | `(2 - W) & 7` DWORDs. |

The alignment applies to the **end** of the entry packet. It is independent
of the body's base-address alignment. The kernel ring position here is in
DWORDs; ROCr's direct KFD SDMA publication instead uses byte frontiers.
[SI entry][ib-si] [CIK entry][ib-cik] [Six-word entry][ib5]
[Native USER publication][rocr-publish]

Every inspected `pad_ib` helper extends the external body to a multiple of
eight DWORDs. SI appends its single-word NOP. The later helpers use a counted
first NOP when the instance's `burst_nop` flag is true; its count is the
number of following padding DWORDs. The remaining words are ordinary NOPs.
CIK, 2.4 and 3.x set that flag for firmware **feature version** at least 20;
the common firmware loader used by the 4.x–7.x backends applies the same
predicate. Firmware feature version and firmware image version are separate
fields. [SI body padding][pad-si] [CIK body padding][pad-cik]
[2.4 body padding][pad24] [3.x body padding][pad3]
[4.0 body padding][pad4] [4.4.2 body padding][pad442]
[5.0 body padding][pad5] [5.2 body padding][pad52]
[6.x body padding][pad6] [7.0 body padding][pad7]
[7.1 body padding][pad71] [CIK feature predicate][burst-cik]
[2.4 predicate][burst24] [3.x predicate][burst3]
[Common firmware predicate][burst-common]

Primary-ring commit has another padding operation: `amdgpu_ring_commit`
rounds the complete submitted ring stream to `align_mask + 1` DWORDs, issues
a host memory barrier, then publishes WPTR. The table above gives the
backend-specific 16- or 256-DWORD alignment. This final padding includes the
wrapper's completion work and is not the external body's length.
[Ring commit][ring-commit]

The DRM `AMDGPU_INFO_HW_IP_INFO` query supplies a separate userspace
submission contract. Its DMA branch reports 256-byte IB-start alignment and
four-byte IB-size alignment. Those values differ from the emitter's masked
32-byte base, the optional eight-DWORD body-padding helper and the kernel
ring's fetch padding. They describe different actors' inputs; substituting
one alignment for another loses that distinction. [DMA query][query-align]
[Returned alignment fields][query-fields]

Mesa keeps allocation alignment and executable padding separate as well.
Its shared GPU information selects the maximum of the query's two byte
alignments and 256 bytes for allocation. A separate SDMA padding mask of
`0xf` makes RADV's scheduled command bodies multiples of 16 DWORDs, including
NOPs for an otherwise empty body. This is a userspace producer policy; it
differs from the kernel's eight-DWORD helper and four-byte query field.
[Mesa allocation alignment][mesa-alignment] [Mesa padding table][mesa-padding]
[RADV body padding][radv-padding]

## Context storage

`amdgpu_sdma_get_csa_mc_addr` returns zero for VMID zero, an SR-IOV VF or
disabled mid-command-buffer preemption. Otherwise it finds the SDMA instance
from the native ring and returns the reserved CSA virtual address plus
`8192 + 64 * instance_index` bytes. Ordinary and page rings of one instance
share that index. Lookup failure or an index above 31 produces zero. The
64-byte stride is the driver's storage partition, not a general caller-chosen
allocation size. [CSA addressing and selection][csa]

With mid-command-buffer preemption enabled, device initialization allocates
and clears a 128 KiB CSA object, and DRM file creation maps it into the file's
VM at the reserved address. File teardown unmaps that view; device teardown
owns the object. The packet emitter borrows the resulting address instead of
allocating storage per IB. Those owners and conditions do not supply a rule
for substituting VMID zero, CSA zero or a new allocation in a KFD process
queue. [CSA allocation][csa-allocate] [CSA storage initialization][csa-storage]
[Per-file mapping][csa-map] [Per-file unmapping][csa-unmap]
[Device release][csa-release] [CSA size][csa-size]

## Publication, completion and final use

The ordinary scheduler composes more than an INDIRECT packet:

```text
host: initialize IB → submit/commit ring program → wait fence → retire IB
ring: context/memory setup → HDP flush → IB → HDP invalidate → completion fence
```

IB-pool allocation sets `AMDGPU_IB_FLAG_EMIT_MEM_SYNC`. The scheduler emits
that synchronization only when the flag is set and the engine supplies the
callback; this is not one identical cache sequence across generations. It also
owns the conditional context work, HDP operations, optional user fence and
final completion fence. The [cache reference](cache.md) describes the separate
kernel-ring and USER cache transports. [Allocation][ib-owner] [Conditional
memory setup][memsync] [Scheduled completion][schedule]

All eleven backend `ring_test_ib` callers allocate 256 bytes, schedule a
WRITE body and wait for the returned `dma_fence`. SI submits four body
DWORDs, CIK five, and 2.4–7.1 eight with explicit trailing NOPs. The legacy
callers therefore differ from their available eight-DWORD padding helpers;
`amdgpu_ib_schedule` does not itself pad these bodies. Their successful paths
inspect the written value and then free the IB. This is an actual first-level
ring → IB → ring owner whose completion establishes safe storage release.
`amdgpu_ib_free` also accepts a last-use fence for suballocator retirement.
Seeing only the destination write would not supply the same ownership
evidence. [SI caller][owner-si] [CIK caller][owner-cik]
[2.4 caller][owner24] [3.x caller][owner3] [4.0 caller][owner4]
[4.4.2 caller][owner442] [5.0 caller][owner5] [5.2 caller][owner52]
[6.x caller][owner6] [7.0 caller][owner7] [7.1 caller][owner71]
[Fence-aware free][ib-owner] [Submission wrapper][schedule]

The 256-byte allocation is an observed test shape, not a documented maximum
prefetch extent. These successful paths also do not establish a failed-wait
retirement rule, recursive IB execution or direct USER command publication.

## Direct queues and secondary commands

ROCr's ordinary KFD SDMA submitter reserves primary-ring space, copies its
transfer commands inline, appends completion work and publishes byte WPTR and
doorbell. Its consumed frontier controls ring-byte reuse. The KFD test queue
likewise emits inline packets and appends FENCE/TRAP for event completion.
These callers establish useful ordinary USER behavior without supplying
INDIRECT's context operands or external-storage lifetime. [ROCr inline
commands][rocr-inline] [Publication and reuse][rocr-publish] [ROCr completion
and commit][rocr-completion] [KFD test queue][kfd-user]

Secondary command-buffer APIs do not imply SDMA IB nesting. PAL's DMA command
buffer calls its base `CmdStream::Call`, which copies child words inline.
RADV's secondary execution also copies command words outside its GFX-only IB2
branch. First-level RADV submission passes IB address, byte length and engine
to the DRM scheduler, which owns the primary-ring wrapper. [PAL nested
call][pal-nested] [PAL inline implementation][pal-inline] [RADV secondary
commands][radv-inline] [RADV scheduled submission][radv-submit]

KFD's v9 MQD builder sets `SWITCH_INSIDE_IB`, expressing native scheduling
policy for long IB work. That MQD field is not proof that every loading path
installs the corresponding register: the inspected GC9.4.3 manual loader does
not write IB_CNTL. Neither fact establishes the USER packet operands or proves
hardware incapability. [MQD policy][mqd] [Manual loader][mqd-load]

The separate DRM userq transport has another ownership surface. SDMA6.1.1 with
firmware version at least 17 and `disable_uq` clear enables that path. Its
construction accepts a caller's `csa_va`, validates a mapped 32-byte range and
passes the address into MQD properties. Mesa allocates that SDMA CSA, but its
inspected userq packet submitter handles only GFX/COMPUTE. Queue creation
alone does not establish an SDMA INDIRECT producer, the relationship between
queue CSA and packet CSA, or final command-storage use. These DRM facts do not
extend the KFD contract. [DRM admission][drm-gate] [DRM CSA][drm-csa] [Mesa
queue construction][mesa-create] [Mesa packet submission][mesa-submit]

[routing]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_discovery.c#L2785-L2848
[ib442]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v4_4_2.c#L371-L390
[ib6]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v6_0.c#L265-L291
[opcode]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v6_0_0_pkt_open.h#L29
[fields442]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/vega10_sdma_pkt_open.h#L2067-L2119
[fields6]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v6_0_0_pkt_open.h#L3882-L3940
[pad442]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v4_4_2.c#L1261-L1276
[pad6]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v6_0.c#L1126-L1141
[csa]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_sdma.c#L32-L92
[ib-owner]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_ib.c#L64-L100
[memsync]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_ib.c#L238-L252
[schedule]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_ib.c#L269-L349
[owner442]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v4_4_2.c#L1109-L1166
[owner6]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v6_0.c#L964-L1030
[rocr-inline]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L499-L577
[rocr-completion]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L614-L663
[rocr-publish]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L1954-L2094
[kfd-user]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/libhsakmt/tests/kfdtest/src/SDMAQueue.cpp#L69-L98
[pal-nested]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/dmaCmdBuffer.cpp#L1291-L1307
[pal-inline]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/cmdStream.cpp#L604-L623
[radv-inline]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/vulkan/winsys/amdgpu/radv_amdgpu_cs.c#L697-L740
[radv-submit]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/vulkan/winsys/amdgpu/radv_amdgpu_cs.c#L1752-L1772
[mqd]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_mqd_manager_v9.c#L545-L574
[mqd-load]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gc_9_4_3.c#L59-L126
[drm-gate]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v6_0.c#L1393-L1396
[drm-csa]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/mes_userqueue.c#L441-L483
[mesa-create]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/winsys/amdgpu/drm/amdgpu_userq.c#L214-L226
[mesa-submit]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/winsys/amdgpu/drm/amdgpu_cs.cpp#L1488-L1621

[ib-si]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/si_dma.c#L77-L92
[pad-si]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/si_dma.c#L415-L419
[owner-si]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/si_dma.c#L258-L309
[ib-cik]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/cik_sdma.c#L222-L238
[pad-cik]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/cik_sdma.c#L798-L813
[owner-cik]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/cik_sdma.c#L652-L705
[ib24]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v2_4.c#L246-L265
[pad24]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v2_4.c#L733-L748
[owner24]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v2_4.c#L583-L640
[ib3]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v3_0.c#L423-L442
[pad3]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v3_0.c#L1007-L1022
[owner3]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v3_0.c#L858-L914
[ib4]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v4_0.c#L806-L825
[pad4]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v4_0.c#L1667-L1682
[owner4]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v4_0.c#L1515-L1572
[ib5]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v5_0.c#L429-L455
[pad5]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v5_0.c#L1236-L1251
[owner5]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v5_0.c#L1073-L1140
[ib52]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v5_2.c#L277-L303
[pad52]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v5_2.c#L1136-L1151
[owner52]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v5_2.c#L973-L1039
[ib7]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v7_0.c#L267-L293
[pad7]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v7_0.c#L1145-L1160
[owner7]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v7_0.c#L979-L1045
[ib71]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v7_1.c#L261-L287
[pad71]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v7_1.c#L1151-L1166
[owner71]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v7_1.c#L969-L1035
[si-route]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/si.c#L2688-L2738
[cik-route]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/cik.c#L2185-L2253
[vi-route]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/vi.c#L2049-L2148
[versions3]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v3_0.c#L1623-L1639
[ras44]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v4_4.c#L24-L60
[fields-si]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sid.h#L563-L575
[fields-cik]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/cikd.h#L495-L511
[fields24]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/iceland_sdma_pkt_open.h#L1633-L1686
[fields3]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/tonga_sdma_pkt_open.h#L1633-L1686
[fields5]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/navi10_sdma_pkt_open.h#L3343-L3402
[fields71]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v7_1_0_pkt_open.h#L3881-L3940
[ring-si]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/si_dma.c#L735-L757
[ring-cik]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/cik_sdma.c#L1235-L1259
[ring24]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v2_4.c#L1122-L1147
[ring3]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v3_0.c#L1503-L1528
[ring4]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v4_0.c#L2425-L2485
[ring442]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v4_4_2.c#L2131-L2190
[ring5]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v5_0.c#L1929-L1959
[ring52]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v5_2.c#L1934-L1964
[ring6]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v6_0.c#L1740-L1770
[ring7]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v7_0.c#L1672-L1702
[ring71]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v7_1.c#L1637-L1665
[ring-commit]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_ring.c#L169-L189
[query-align]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_kms.c#L479-L494
[query-fields]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/include/uapi/drm/amdgpu_drm.h#L1543-L1557
[burst-cik]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/cik_sdma.c#L539-L548
[burst24]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v2_4.c#L155-L159
[burst3]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v3_0.c#L316-L320
[burst-common]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_sdma.c#L150-L185
[csa-allocate]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_device.c#L2436-L2447
[csa-storage]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_csa.c#L29-L51
[csa-map]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_kms.c#L1523-L1530
[csa-unmap]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_kms.c#L1596-L1602
[csa-release]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_device.c#L2964-L2972
[csa-size]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_csa.h#L28-L28
[header7]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v7_0.c#L40
[mesa-alignment]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_gpu_info.c#L521-L528
[mesa-padding]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_gpu_info.c#L464-L475
[radv-padding]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/vulkan/winsys/amdgpu/radv_amdgpu_cs.c#L451-L499
