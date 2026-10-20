# Command-processor atomic operations

`ATOMIC_MEM` asks the command processor to perform one atomic memory operation
using the TC atomic-operation encoding. The operation selects its data width and
arithmetic; packet controls select how the command processor issues it. A
return-form operation can also supply the previous value to a subsequent
`COPY_DATA`. Atomicity of that cell, ordering of other payload, and retirement
of the command and result storage are separate contracts.

## Applicability

PAL's ordinary compute `CmdMemoryAtomic` emits the operation into its main
compute command stream. Its `gfx9` backend serves GFX10 and GFX11 at the cited
revision; a separate GFX12 builder preserves the integer operation conversion
but changes the policy field. The backend directory and merged-header names
are not a claim that every listed generation uses the same native queue
protocol. [Backend predicate][pal-backend] [Compute caller][pal-caller]
[GFX12 caller][pal12-caller]

Three consumers illustrate different uses:

| Consumer | Atomic use | Surrounding contract |
| --- | --- | --- |
| PAL compute commands | Integer ALU operation, single pass, no old-value result exposed to the caller. | `PostPrefetch` stage and `CoherQueueAtomic` access class; barriers depend on the adjacent actors. |
| ROCr PC-sampling buffer exchange | Return-form SWAP64, followed by confirmed `COPY_DATA` of the previous value. | Fine-grained GPU target, host-accessible result, selected XCC, and an AQL-carried IB completion. |
| RADV performance-counter mutex | Non-return CMPSWAP32 in loop-until-compare mode. | GFX queue lock/unlock preambles and postambles with device-owned control storage. This caller is not a compute-queue loop recipe. |

[PAL contract][pal-contract] [ROCr exchange][rocr-swap]
[RADV lock][radv-lock] [RADV queue predicate][radv-queue]

## Packet representation

The GFX10/GFX11 MEC form occupies nine little-endian DWORDs. Word numbers below
start at zero, including the header. The target is a byte address, with no
address shift in the packet. PAL requires a nonzero address aligned to four
bytes for a 32-bit operation or eight bytes for a 64-bit operation.
[MEC layout][mec-layout] [Builder][pal-builder]

| Word and bits | Meaning |
| --- | --- |
| 0, 31:30 / 29:16 / 15:8 | Type 3 / count 7 / opcode `0x1e`. The ordinary zero-low-byte header is `0xc0071e00`. |
| 0, 7:0 | Reserved in the MEC header. PAL's shared builder uses the ordinary zero values here. |
| 1, 6:0 | TC atomic operation. The operation determines the width; there is no separate 32/64-bit selector. |
| 1, 7 | Reserved. |
| 1, 11:8 | Command mode. |
| 1, 24:12 | Reserved in this layout. |
| 1, 26:25 | Cache policy: LRU `0`, STREAM `1`, NOA `2`, BYPASS `3`. |
| 1, 31:27 | Reserved in the MEC layout. |
| 2–3 | Target byte address, low DWORD then high DWORD. |
| 4–5 | Source operand, low DWORD then high DWORD. |
| 6–7 | Comparison operand, low DWORD then high DWORD. |
| 8, 12:0 | Loop interval. |
| 8, 31:13 | Reserved. |

[Header][mec-header] [Header construction][pal-header]
[Ordinary header defaults][pal-header-defaults]

The ME layout instead gives control bits 31:30 an `engine_sel` field whose
defined value is micro-engine `0`; those bits remain reserved for MEC. PAL
checks the ME/MEC packet size and relevant command/policy enum agreement, then
uses a zero-initialized ME structure for both engines. The ordinary builder
selects single pass and LRU, fills the address and source, and leaves comparison,
loop, and reserved fields zero. For a 32-bit operation only the low source
DWORD is semantically used; the builder still copies the supplied high DWORD.
[ME layout][me-layout] [Builder][pal-builder] [Operand contract][pal-contract]

## Integer operation conversion

This is the complete 22-entry PAL `AtomicOp` conversion. Each TC symbol below
has the prefix `TC_OP_ATOMIC_`. **PAL emits the RTN code**, even though
`CmdMemoryAtomic` supplies no returned-value destination. The last column gives
the separately defined non-RTN encoding; it does not describe a second mode
chosen by that builder. Both PAL backends use the same conversion order and
values. [Public enum][pal-enum] [Conversion][pal-conversion]
[TC operation values][tc-ops] [GFX12 conversion][pal12-builder]

| PAL operation and enum value | Width | TC RTN symbol | RTN code | Non-RTN code |
| --- | --- | --- | --- | --- |
| `AddInt32`, `0x00` | 32 bits | `ADD_RTN_32` | `0x0f` | `0x4f` |
| `SubInt32`, `0x01` | 32 bits | `SUB_RTN_32` | `0x10` | `0x50` |
| `MinUint32`, `0x02` | 32 bits | `UMIN_RTN_32` | `0x12` | `0x52` |
| `MaxUint32`, `0x03` | 32 bits | `UMAX_RTN_32` | `0x14` | `0x54` |
| `MinSint32`, `0x04` | 32 bits | `SMIN_RTN_32` | `0x11` | `0x51` |
| `MaxSint32`, `0x05` | 32 bits | `SMAX_RTN_32` | `0x13` | `0x53` |
| `AndInt32`, `0x06` | 32 bits | `AND_RTN_32` | `0x15` | `0x55` |
| `OrInt32`, `0x07` | 32 bits | `OR_RTN_32` | `0x16` | `0x56` |
| `XorInt32`, `0x08` | 32 bits | `XOR_RTN_32` | `0x17` | `0x57` |
| `IncUint32`, `0x09` | 32 bits | `INC_RTN_32` | `0x18` | `0x58` |
| `DecUint32`, `0x0a` | 32 bits | `DEC_RTN_32` | `0x19` | `0x59` |
| `AddInt64`, `0x0b` | 64 bits | `ADD_RTN_64` | `0x2f` | `0x6f` |
| `SubInt64`, `0x0c` | 64 bits | `SUB_RTN_64` | `0x30` | `0x70` |
| `MinUint64`, `0x0d` | 64 bits | `UMIN_RTN_64` | `0x32` | `0x72` |
| `MaxUint64`, `0x0e` | 64 bits | `UMAX_RTN_64` | `0x34` | `0x74` |
| `MinSint64`, `0x0f` | 64 bits | `SMIN_RTN_64` | `0x31` | `0x71` |
| `MaxSint64`, `0x10` | 64 bits | `SMAX_RTN_64` | `0x33` | `0x73` |
| `AndInt64`, `0x11` | 64 bits | `AND_RTN_64` | `0x35` | `0x75` |
| `OrInt64`, `0x12` | 64 bits | `OR_RTN_64` | `0x36` | `0x76` |
| `XorInt64`, `0x13` | 64 bits | `XOR_RTN_64` | `0x37` | `0x77` |
| `IncUint64`, `0x14` | 64 bits | `INC_RTN_64` | `0x38` | `0x78` |
| `DecUint64`, `0x15` | 64 bits | `DEC_RTN_64` | `0x39` | `0x79` |

Signed and unsigned min/max have different operation numbers. INC/DEC also
have their own numbers; the builder passes the source operand without
rewriting the operation. PAL's unconditional command-retirement increment uses
ADD32 with source `1`, not INC32. [Tracker update][pal-postamble]

SWAP and CMPSWAP are present in the packet operation definitions but absent
from PAL's public `AtomicOp` enum:

| Operation | 32-bit RTN / non-RTN | 64-bit RTN / non-RTN | Corroborating caller |
| --- | --- | --- | --- |
| SWAP | `0x07` / `0x47` | `0x27` / `0x67` | ROCr emits `0x27` and retrieves its result. |
| CMPSWAP | `0x08` / `0x48` | `0x28` / `0x68` | RADV emits the non-RTN 32-bit operation for its GFX mutex. |

[TC definitions][tc-ops] [GFX12 packet enum][mec12-ops]
[ROCr swap][rocr-swap] [RADV compare-swap][radv-lock]

Ignoring a SWAP's previous value gives an atomic store of the source operand;
it does not require selecting a non-RTN opcode. Conversely, an `_RTN` suffix
does not itself place that previous value in host memory. The packet has no
returned-value address. The larger TC enum also contains floating-point and
other operations; their presence is not a complete command-processor caller
or mapping contract.

SWAP still issues a read-modify-write transaction. Support for an ordinary
aligned store does not establish that stronger route. PAL's CP-stage immediate
write uses confirmed [`COPY_DATA`](copy.md#execution-and-ordering); a
[`RELEASE_MEM`](release.md) update instead runs at the selected pipeline stage
and does not stop subsequent CP commands. Both supply writes without RMW;
their target mapping separately determines atomic width and observer reach.
WDDM makes this distinction explicit for CPU-visible fence updates:
`DXGK_VIDSCHCAPS::No64BitAtomics` distinguishes 32-bit updates from atomic
64-bit updates to coherent memory. The adapter's
`D3DKMT_WDDM_2_0_CAPS::Support64BitAtomics` reports the native width capability.
Neither fact grants PCIe AtomicOps support for SWAP, ADD, or compare-swap.
[CPU-visible update width][wddm-update-width]
[Monitored-fence mapping][wddm-monitored-fence]
[Native adapter capabilities][wddm-native-capabilities]

## Command modes and comparison

| Command value | Defined name | Source-backed use |
| --- | --- | --- |
| `0` | `single_pass_atomic` | PAL's ordinary integer builder and ROCr's SWAP64 issue one operation. |
| `1` | `loop_until_compare_satisfied` | RADV's GFX mutex repeatedly attempts CMPSWAP32, with source `1`, comparison `0`, and loop interval `10`. |
| `2` | `wait_for_write_confirmation` | Defined in the packet enum; the callers above do not establish an operation/operand/completion sequence for this mode. |
| `3` | `send_and_continue` | Defined in the packet enum; the callers above do not establish a complete asynchronous ownership protocol for this mode. |

[Command definitions][mec-layout] [PAL builder][pal-builder]
[Mesa emitter][mesa-atomic] [RADV caller][radv-lock]

RADV's lock attempts to replace an unlocked zero with one. The companion
unlock command uses confirmed `COPY_DATA` to restore zero after its pass-state
updates. The lock commands are attached to initial/continuation preambles and
the unlock to a postamble. The buffer is device-owned GTT and added to the
command stream's buffer list. Its CPU accessibility does not establish a
concurrent CPU-atomic participant. [Control allocation][radv-owner]
[Lock and unlock][radv-lock] [Submission composition][radv-queue]

This caller supplies an actual use of the comparison words and looping mode,
but it explicitly runs on `AMD_IP_GFX`; it supplies neither MEC scheduling
progress nor a cross-queue fairness guarantee. The emitter writes raw loop
interval `10`. Neither that caller nor the cited generated packet layout gives
a time unit or a bounded completion deadline for the interval. The mode names
`2` and `3` alone are insufficient to construct another runtime sequence.

## Retrieving the previous value

ROCr's non-large-BAR sampling path demonstrates return transport separately
from the atomic update. The target comes from its fine-grained GPU allocation;
each XCC has a host-accessible, GPU-accessible eight-byte result allocation and
a completion signal. The CPU command array is distinct from the queue's
executable IB. [Path selection][rocr-selection] [Result owners][rocr-owners]
[Target allocation][rocr-target]

The selected exchange sequence is:

```text
SWAP_RTN_64(target, replacement)
  → COPY_DATA(atomic_return_data, TC/L2 result, 64 bits, confirmed)
  → AQL carrier SYSTEM release and completion
  → host signal acquire
  → read result
```

`COPY_DATA` uses source selector `6`, destination selector `2`, count bit 16
set, and confirmation bit 20 set; its unused source-address words are zero.
The destination is a naturally aligned eight-byte result. For multiple XCCs,
ROCr wraps the **atomic and result-copy pair** in `PRED_EXEC` selecting virtual
XCC 0. The returned value is consumed from the same selected CP path; it is not
an independently addressed result attached to each ATOMIC_MEM packet.
[Selectors][rocr-fields] [Paired commands and routing][rocr-swap]
[Completion and result read][rocr-result]

An exchange selects a new buffer; it does not complete prior writers of the
old buffer. ROCr separately waits for the old buffer's written-count before
copying its payload. Its GFX12.0/gfx125 condition also emits GL2 writeback
before that copy. These later operations carry the payload-completion and
visibility obligations that the atomic exchange cannot supply.
[Payload join][rocr-payload]

The executable IB owner remains important. `ExecutePM4` copies the CPU array
into a shared IB under a mutex, publishes the AQL packet with a release header
store and doorbell, and waits internally only when it owns the completion
signal on GFX9+. With a caller-supplied signal it returns after publication;
the sampling caller then performs its signal acquire. Thus the packet pairing
and result observation are explicit, but the helper mutex alone does not
establish shared-IB retention through that external wait. A composed caller
must also serialize every reuse of that executable storage through completed
use. Per-XCC CPU arrays and result cells do not replace that queue-level owner.
[Shared IB and carrier][rocr-carrier] [Publication and wait branches][rocr-publication]
[AQL-carried command storage](../aql/transfers.md)

## Participants and native memory

PAL promises that its command-memory atomics are atomic with respect to shader
atomics. This is a positive CP/shader contract, with natural alignment and
the operation's width. It does not make ordinary non-atomic shader accesses
race-safe or establish CPU participation on every GPU mapping.
[Atomicity and barrier contract][pal-contract]

CPU participation additionally binds the operation and width to the backing,
page-table policy, native interconnect route, and participating processors.
The host must use a compatible atomic access to the same cell when concurrent
access is intended. A shared virtual address or a coherent mapping alone
does not supply the mutual atomicity premise.

Linux's route selection distinguishes these cases:

| Native predicate | Route evidence |
| --- | --- |
| SR-IOV virtual function | PF-to-VF support flags must equal the combined 32- and 64-bit support mask. |
| Non-VF APU with native GC strictly greater than 9.0.0 | Internal atomic route; no PCIe-root requirement. |
| Non-VF CPU-connected xGMI and native GC at least 12.1.0 | CPU-connected fabric route. |
| Other devices | Enable 32- and 64-bit PCIe AtomicOps to the root complex. |

[Native route selection][linux-route]

The code's strict `GC > 9.0.0` APU predicate is more specific than its adjacent
“gfx9 onwards” comment. KFD can also accept devices without the route when the
selected firmware acknowledges that absence: its GFX11 RS64 predicate uses
MEC firmware version at least 509, while F32 retains the route requirement.
Successful device/queue availability therefore does not prove host-atomic
reach, and a route flag is not a per-operation support table.
[Firmware predicate][linux-firmware] [KFD route check][linux-admission]

### GFX11 APU exchange composition

AMD's GFX11 memory-system description forwards write-uncached atomics to the
fabric. Its operation tables mark 32/64-bit exchange as native for fine-grained
pinned host DRAM at system scope on GFX11 APUs, both with and without PCIe
atomics. The table's backing, granularity, scope, and APU column are part of
that statement. [Memory-system rule][amd-memory] [Table dimensions][amd-context]
[Table construction][amd-renderer] [Without PCIe][amd-no-pcie]
[With PCIe][amd-pcie]

One Linux GMC11 mapping composition is owned GTT with KFD COHERENT, CPU cached
pages, and no CPU_GTT_USWC flag: COHERENT selects GPU MTYPE_UC, while the cached
GTT receives SYSTEM/SNOOPED PTEs. These are placement and cache facts, not
additional participants. [Flag translation][linux-flags]
[GPU policy][linux-pte] [CPU cache choice][linux-cpu]
[System snooping][linux-snoop]

Applying the documented memory-system rule to CP TC SWAP is an architectural
composition: the operation definitions and ROCr establish the CP client;
AMD's table establishes the operation-specific host-memory behavior; Linux
establishes this mapping and route. The AMD table does not name ATOMIC_MEM,
and ROCr's target is GPU memory. This composition therefore does not silently
extend to every RMW operation, dGPU PCIe route, imported mapping, or peer GPU.

### GFX12 cache policy and explicit scope

Coherent system memory does not have the same GPU cache policy across native
GC revisions or driver implementations. KFD's mapping flags start with
`AMDGPU_VM_MTYPE_DEFAULT`; the selected GMC callback translates that choice
and the BO flags:

| Driver and native mapping path | Owned GTT with COHERENT | Explicit UNCACHED |
| --- | --- | --- |
| Linux `50d05c7`, GMC11 | MTYPE_UC. | MTYPE_UC. |
| Linux `50d05c7`, GMC12.0 callback for GC12.0.0 and GC12.0.1 | MTYPE_NC; COHERENT alone does not select UC. | MTYPE_UC. |
| Linux `50d05c7`, GMC12.1 callback for GC12.1.0 | Revision-dependent MTYPE_NC or MTYPE_UC. | MTYPE_UC. |
| ROCm AMDGPU `8202127`, GMC12.0 callback for GC12.0.0 and GC12.0.1 | MTYPE_NC; COHERENT alone does not select UC. | MTYPE_UC. |
| ROCm AMDGPU `8202127`, GMC12.1 callback for GC12.1.0 | MTYPE_NC or MTYPE_UC according to the policy below. | MTYPE_UC. |

[KFD mapping flags][linux-mapping] [GMC11 translation][linux-pte]
[GMC12.0 translation][linux12-pte]
[Linux callback selection][linux12-selection]
[Linux GC12.1 policy][linux121-coherence]
[ROCm GMC12.0 translation][rocm12-pte]
[ROCm callback selection][rocm12-selection]

In both cited drivers, `gmc_v12_0_early_init` replaces the GMC callbacks with
`gmc_v12_1_gmc_funcs` for native GC12.1.0. The IP-block name alone therefore
does not identify the active PTE policy. For valid BO mappings,
`gmc_v12_1_get_vm_pte` applies `gmc_v12_1_get_coherence_flags` after the
requested MTYPE. [Callback selection][rocm12-selection]
[PTE construction][rocm121-pte]

That GC12.1 policy calls same-device VRAM local and treats GTT as remote. Its
MTYPE choices are:

| Predicate in ROCm AMDGPU `8202127` | Local MTYPE | Remote MTYPE |
| --- | --- | --- |
| `rev_id & 0x10` is set, before module overrides | RW | NC |
| `rev_id & 0x10` is clear, before module overrides | NC | UC |
| `amdgpu_mtype_local == 0` | RW | NC |
| `amdgpu_mtype_local == 1` | NC | Retains the revision choice. |
| Other `amdgpu_mtype_local` values | Retains the revision choice. | Retains the revision choice. |

UNCACHED selects UC for either placement. EXT_COHERENT selects UC for remote
memory; COHERENT alone retains the placement policy. A non-NC choice replaces
the requested MTYPE; an NC choice preserves it. The callback adds SNOOPED
and sets `AMDGPU_PTE_BUS_ATOMICS` when the BO is local or
`adev->have_atomics_support` is true. That page bit does not replace the CPU
route's operation-width contract or the command's scope.
[Coherence and bus-atomic policy][rocm121-coherence]

Linux `50d05c7` and the `releases/therock-7.14` revision `5081d70` apply the
same placement, coherence and BUS_ATOMICS rules, but their
`amdgpu_mtype_local == 0` override changes only the local choice. Remote memory
retains its revision-dependent NC/UC choice. The driver revision is part of
the mapping description. [Linux policy][linux121-coherence]
[ROCm release policy][rocm121-therock]

AMD's GFX12 memory-system description keeps fine-grained system memory
cacheable: device-scope atomics can complete in L2, while system-scope atomics
bypass L2 toward the fabric. A PCIe route that cannot perform the requested
operation can degrade it to a load/operation/store sequence that is not atomic
against the CPU. Cache policy and native operation support remain independent.
[GFX12 memory-system rules][amd12-memory]

PAL's GFX12 CP connects directly to MALL, and its barrier planner classifies
queue atomics among the clients that bypass GL2. Shader accesses still need
the appropriate [CP/shader cache handoff](cache.md#gfx12-cp-and-shader-handoffs).
This connection is distinct from the packet's scope representation: the PAL
GFX12 layout reserves bits 24:23, while Linux's native GC12.1 layout uses them
for `SCOPE`, with zero meaning CU and three meaning SYSTEM. Copying an earlier
zero-filled control word therefore does not request SYSTEM scope on GC12.1.
[CP connection and barriers][pal12-cache] [Earlier fields][mec12-layout]
[GC12.1 scope][linux12-layout]

## Ordering, completion, and last use

PAL's GFX10/GFX11 planner identifies queue atomics as GL2 clients and CPU/memory
accesses as clients that bypass GL2. It requests GL2 writeback for a
queue-atomic-to-CPU transition, and GL2 writeback/invalidation for the reverse
direction. Its GFX12 planner places queue atomics in the bypass-GL2 class;
transitions involving shader GL2 therefore follow the
[GFX12 CP and shader rules](cache.md#gfx12-cp-and-shader-handoffs).
Other shader consumers can require their own cache invalidation. The atomic
opcode and cache-policy field do not encode these surrounding memory
dependencies. [Access classes][pal-clients]
[Cache transition][pal-cache]

The operation's PAL stage is `PostPrefetch`. That stage alone does not request
a shader EOS/EOP event. In the non-PWS compute barrier path, any required
shader-idle wait occurs before cache work; remaining cache work is emitted
through `ACQUIRE_MEM`. A subsequent PostPrefetch event uses confirmed
`WRITE_DATA` to MEMORY. A CS/bottom-of-pipe event instead uses `RELEASE_MEM`.
Thus ordinary CP atomic publication has an explicit stage/cache/event path;
it does not acquire a shader join merely by being followed by a CP write.
[Release-stage selection][pal-stages] [Cache acquire point][pal-acquire-point]
[Barrier lowering][pal-barrier] [Event contract][pal-event-contract]
[Event implementation][pal-event] [Write confirmation][pal-confirm]

A no-result compute update can follow this complete flow:

1. Establish a valid read/write mapping for the naturally aligned target,
   plus command and completion storage. Initialize them before publication,
   using the native CPU-to-GPU visibility protocol.
2. Complete any prior payload producer and perform the transition to the
   atomic's CP/GL2 access. A prior asynchronous shader or DMA engine needs its
   actual execution join; command order alone is insufficient.
3. Issue the selected single-pass atomic. A no-result caller leaves the TC
   return value unused and has no host result buffer to reclaim.
4. Perform the transition from the atomic to the next observer, then signal
   completion at the required stage. On PAL's cited CPU transition this
   includes GL2 writeback before the confirmed event store.
5. The observer acquires that completion under the mapping's visibility
   contract before using the cell. If another queue consumes it, preserve the
   target and dependency signal through that queue's final use as well.
6. Retire all submitted command references before rebuilding command storage;
   release mappings only after every user has completed.

[Cross-queue ownership](handoff.md#owners-and-a-complete-sequence) and
[command-buffer ownership](command-buffers.md#publication-and-memory-ownership)
describe the independent payload and command-storage boundaries.

PAL's command allocator gives a concrete atomic-retirement consumer. Its
compute postamble first joins active CP DMA and shader users, then ADD32s one
into the root chunk's done counter. The postamble explicitly relies on a KMD
EOP cache flush to make the counter visible in memory. The tracker reserves
eight bytes but compares its low 32-bit count with the submission count under
a no-wrap assumption. Automatic reuse calls `IsIdleOnGpu` before returning
chunks to the free list. [Compute postamble][pal-postamble]
[Tracker owner][pal-tracker] [Automatic reuse][pal-reuse]

That is a scheduled-transport retirement protocol. A raw user ring does not
inherit its KMD trailer, and a returned atomic value is not itself a signal
that all shader, notification, or nested-command users have ended. The
GFX12 postamble retains the KMD premise but directly emits CS_PARTIAL_FLUSH
before its ADD32. [GFX12 postamble][pal12-postamble]

## Generation-specific controls

The nine-DWORD size and integer conversion remain stable across the cited
PAL builders, but policy bits change meaning:

| Definition | Word 1, bits 24:23 | Word 1, bits 26:25 | Ordinary emitted choice |
| --- | --- | --- | --- |
| PAL GFX10/GFX11 MEC | Reserved. | Cache policy: LRU `0`, STREAM `1`, NOA `2`, BYPASS `3`. | Single pass, LRU. |
| PAL GFX12 MEC | Reserved. | Temporal: RT `0`, NT `1`, HT `2`, LU `3`. | Single pass, RT. |
| Linux native GC12.1 | Scope: CU `0`, SE `1`, DEVICE `2`, SYSTEM `3`. | Temporal: RT `0`, NT `1`, FW `2`, UC `3`. | This header defines fields; it is not an atomic caller sequence. |

[Earlier MEC layout][mec-layout] [GFX12 MEC layout][mec12-layout]
[GFX12 builder][pal12-builder] [GC12.1 definition][linux12-layout]

The temporal names `HT/LU` and `FW/UC` and the addition of scope describe
different source/native revisions. Equal bit positions do not make their
policies interchangeable. In particular, importing Linux GC12.1 scope bits
into the earlier PAL layouts sets reserved bits. Conversely, an earlier
zero-reserved-field packet is not evidence of SYSTEM scope on GC12.1.

GFX12's packet enum additionally names 32-bit return-form CLAMP_SUB (`0x1a`)
and COND_SUB (`0x1b`), absent from PAL's 22-operation interface. Earlier TC
definitions assign those numbers to different non-atomic operations. The
opcode namespace itself must therefore be interpreted at its generation;
header enumeration alone supplies no portable arithmetic recipe.
[GFX12 operation enum][mec12-ops] [Earlier TC enum][tc-ops]

[pal-backend]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/device.h#L2561-L2568
[pal-caller]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9ComputeCmdBuffer.cpp#L384-L394
[pal12-caller]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx12/gfx12ComputeCmdBuffer.cpp#L185-L195
[pal-contract]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/inc/core/palCmdBuffer.h#L3936-L3957
[mec-layout]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/chip/gfx9_plus_merged_f32_mec_pm4_packets.h#L284-L378
[mec-header]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/chip/gfx9_plus_merged_f32_mec_pm4_packets.h#L43-L54
[pal-header]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9CmdUtil.cpp#L284-L301
[pal-header-defaults]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9CmdUtil.h#L805-L810
[me-layout]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/chip/gfx9_plus_merged_f32_me_pm4_packets.h#L384-L485
[pal-builder]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9CmdUtil.cpp#L798-L854
[pal-enum]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/inc/core/palCmdBuffer.h#L159-L185
[pal-conversion]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9CmdUtil.cpp#L178-L210
[tc-ops]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/chip/gfx9_plus_merged_enum.h#L13486-L13610
[pal12-builder]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx12/gfx12CmdUtil.cpp#L2638-L2709
[mec12-ops]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx12/chip/gfx12_merged_f32_mec_pm4_packets.h#L138-L199
[mec12-layout]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx12/chip/gfx12_merged_f32_mec_pm4_packets.h#L201-L295
[mesa-atomic]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_cmdbuf_cp.c#L543-L559
[radv-lock]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/vulkan/radv_queue.c#L1546-L1603
[radv-queue]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/vulkan/radv_queue.c#L1683-L1707
[radv-owner]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/vulkan/radv_device.c#L491-L528
[rocr-selection]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp#L3787-L3795
[rocr-owners]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp#L3854-L3879
[rocr-target]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp#L4128-L4144
[rocr-swap]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp#L4723-L4773
[rocr-fields]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/inc/amd_gpu_pm4.h#L96-L113
[rocr-result]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp#L4781-L4804
[rocr-payload]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp#L4840-L4869
[rocr-carrier]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp#L1613-L1646
[rocr-publication]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_aql_queue.cpp#L1729-L1759
[linux-route]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_device.c#L4006-L4028
[linux-firmware]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_device.c#L227-L237
[linux-admission]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdkfd/kfd_device.c#L772-L786
[amd-memory]: https://github.com/ROCm/legacy-rocm-build/blob/85a16825737e43a14ff431754b359380e78062a7/docs/reference/gpu-atomics-operation.rst#L157-L171
[amd-context]: https://github.com/ROCm/legacy-rocm-build/blob/85a16825737e43a14ff431754b359380e78062a7/docs/conf.py#L206-L210
[amd-renderer]: https://github.com/ROCm/legacy-rocm-build/blob/85a16825737e43a14ff431754b359380e78062a7/docs/reference/gpu-atomics-operation.rst#L541-L609
[amd-no-pcie]: https://github.com/ROCm/legacy-rocm-build/blob/85a16825737e43a14ff431754b359380e78062a7/docs/data/reference/gpu-atomics-operation/hw-atomics_nopcie_gfx.csv#L338-L343
[amd-pcie]: https://github.com/ROCm/legacy-rocm-build/blob/85a16825737e43a14ff431754b359380e78062a7/docs/data/reference/gpu-atomics-operation/hw-atomics_pcie_gfx.csv#L338-L343
[linux-flags]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gpuvm.c#L1748-L1779
[linux-pte]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/gmc_v11_0.c#L468-L513
[linux-mapping]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gpuvm.c#L513-L525
[linux12-pte]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/gmc_v12_0.c#L507-L549
[linux12-selection]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/gmc_v12_0.c#L635-L654
[linux121-coherence]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/gmc_v12_1.c#L574-L663
[rocm12-pte]: https://github.com/ROCm/amdgpu/blob/820212794d184710298efcecce42c0215bfc9c6a/drivers/gpu/drm/amd/amdgpu/gmc_v12_0.c#L510-L553
[rocm12-selection]: https://github.com/ROCm/amdgpu/blob/820212794d184710298efcecce42c0215bfc9c6a/drivers/gpu/drm/amd/amdgpu/gmc_v12_0.c#L638-L658
[rocm121-pte]: https://github.com/ROCm/amdgpu/blob/820212794d184710298efcecce42c0215bfc9c6a/drivers/gpu/drm/amd/amdgpu/gmc_v12_1.c#L665-L709
[rocm121-coherence]: https://github.com/ROCm/amdgpu/blob/820212794d184710298efcecce42c0215bfc9c6a/drivers/gpu/drm/amd/amdgpu/gmc_v12_1.c#L603-L663
[rocm121-therock]: https://github.com/ROCm/amdgpu/blob/5081d704e60e807e9541c355c53df57d88ea8cde/drivers/gpu/drm/amd/amdgpu/gmc_v12_1.c#L528-L587
[amd12-memory]: https://github.com/ROCm/legacy-rocm-build/blob/85a16825737e43a14ff431754b359380e78062a7/docs/reference/gpu-atomics-operation.rst#L173-L203
[pal12-cache]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx12/gfx12Barrier.cpp#L406-L435
[linux-cpu]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_ttm.c#L1190-L1214
[linux-snoop]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_ttm.c#L1432-L1476
[pal-clients]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9AcquireReleaseBarrier.cpp#L56-L68
[pal-cache]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9AcquireReleaseBarrier.cpp#L347-L365
[pal-stages]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9AcquireReleaseBarrier.cpp#L436-L488
[pal-acquire-point]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9AcquireReleaseBarrier.cpp#L1834-L1867
[pal-barrier]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9AcquireReleaseBarrier.cpp#L1998-L2042
[pal-event-contract]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/inc/core/palCmdBuffer.h#L3874-L3884
[pal-event]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9ComputeCmdBuffer.cpp#L1327-L1383
[pal-confirm]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9CmdUtil.cpp#L4674-L4687
[pal-postamble]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/gfx9ComputeCmdBuffer.cpp#L1227-L1267
[pal-tracker]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/cmdStreamAllocation.cpp#L404-L479
[pal-reuse]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/cmdAllocator.cpp#L703-L739
[pal12-postamble]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx12/gfx12ComputeCmdBuffer.cpp#L946-L980
[linux12-layout]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/gfx_v12_1_pkt.h#L60-L83

[wddm-update-width]: https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/d3dkmddi/ns-d3dkmddi-_dxgk_vidschcaps
[wddm-monitored-fence]: https://learn.microsoft.com/en-us/windows-hardware/drivers/display/context-monitoring
[wddm-native-capabilities]: https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/d3dkmdt/ns-d3dkmdt-_d3dkmt_wddm_2_0_caps
