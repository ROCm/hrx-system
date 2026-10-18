# SDMA completion stores

FENCE publishes a value after preceding transfer work. ROCr uses it for
completion because ordinary COPY and WRITE commands can overlap. Its classic
form writes one DWORD; two such packets, a native 64-bit fence, and an atomic
signal update have different representations and observation contracts. A
completion value also has an owner: a user event, a kernel ring-progress word
and an interrupt mailbox do not establish the same resource lifetime.
[Completion choice][choice]

## Representation and applicability

### Classic four-DWORD form

CIK and later SDMA use opcode 5 in header bits 7:0 and suboperation zero in
bits 15:8. The packet occupies 16 bytes and stores one 32-bit value at a
DWORD-aligned byte address. The destination mapping and cache controls belong
to the native transport; the address words alone do not establish CPU access,
coherence or physical-address permission.
[CIK encoding][cik-encoding] [ROCr layout][layouts]

| DWORD | Contents |
| --- | --- |
| 0 | Opcode/suboperation and generation-specific controls below. |
| 1 | Destination address bits 31:0. PAL GFX12 explicitly reserves bits 1:0 and names the remaining field `addr_31_2`. |
| 2 | Destination address bits 63:32. |
| 3 | Immediate `data`, 32 bits. |

PAL assigns `LowPart(address) >> 2` to the `addr_31_2` **bitfield**. Its
position restores the original aligned byte-address bits in the wire word;
the whole DWORD is not an address divided by four.
[GFX12 layout][pal-layout] [GFX12 builder][pal12-fence]

The Iceland, Tonga and Vega10 Linux headers name only opcode/suboperation and
the three body words. The Navi10 header adds the classic controls below;
Linux SDMA6/7.1 and PAL's `gfx103Plus` overlay additionally name LLC policy
and CPV. These are source layout names, not interchangeable ISA/engine-version
numbers. [Iceland][linux-iceland] [Tonga][linux-tonga]
[Vega10][linux-vega10] [Navi10][linux-navi10]
[SDMA6][linux-layout6] [SDMA7.1][linux-layout7] [PAL GFX10][pal10-layout]

| Header bits | Classic policy layout | PAL GFX12 layout | ROCr GFX12 layout |
| --- | --- | --- | --- |
| 7:0 | `op` | `op` | `op` |
| 15:8 | `sub_op` | `sub_op` | `sub_op` |
| 19:16 | `mtype` at 18:16; `gcc` at 19. | Reserved. | `mtype` at 17:16; 19:18 reserved. |
| 20 | `sys` | `sys` | `sys` |
| 21 | Reserved. | Reserved. | Reserved. |
| 22 | `snp` | `snp` | `snp` |
| 23 | `gpa` | `gpa` | `gpa` |
| 25:24 | `l2_policy` | Reserved. | `scope` |
| 28:26 | Base layout reserves these bits; the later overlay names `llc_policy` at 26 and `cpv` at 28, with 27 reserved. | `mall_policy` at 27:26; 28 reserved. | `temporal_hint` |
| 31:29 | Reserved. | Reserved. | Reserved. |

The PAL and ROCr GFX12 layouts disagree at MTYPE, scope and temporal/MALL
policy. Linux's SDMA7.1 header retains the older three-bit MTYPE/GCC layout.
The shared opcode does not identify which interpretation applies to a native
firmware interface. The following builders establish their emitted fields;
they do not resolve that disagreement by generation name alone.
[PAL GFX12][pal-layout] [ROCr layouts][layouts] [Linux SDMA7.1][linux-layout7]

### Source-selected controls

ROCr zeroes the packet and selects fields by the GPU ISA major and its
`scopeFields` template parameter. Its use of SYS is explicitly for signals
in system memory. SNP, GPA, temporal hint and other unselected controls stay
zero. [ROCr builder][builder]

| ROCr predicate | Fields set by the FENCE32 builder |
| --- | --- |
| ISA major < 10 | Opcode 5; other header fields zero. |
| ISA major 10 or 11 | Classic MTYPE bits 18:16 set to 3. |
| ISA major >= 12 | New two-bit MTYPE at bits 17:16 set to 3, SYS at bit 20 set to one because ROCr uses this fence for system-memory signals. |
| ISA major >= 12 and `scopeFields` | Additionally, SYS scope (3) at bits 25:24. |

The builder's ISA-major branch precedes its scope decision. Selecting the V6
scoped template for gfx11.5 therefore does not add a scope field to its
FENCE32, even though the template sets scope fields on other operations. The
[factory predicates](atomics.md#architecture-and-transport-differences)
identify which template ROCr selects for each native transport.

PAL's builders have distinct policies even within one backend:

| PAL path | Emitted controls |
| --- | --- |
| GFX10 `WriteFenceCmd` and `CmdWriteImmediate` | MTYPE=3 (`MTYPE_UC`). When `supportsMall`, fill LLC policy and CPV from the write-policy helpers; L2 policy remains zero. |
| GFX10 `WriteEventCmd` | MTYPE=3; no LLC/CPV override. |
| GFX12 equivalents | Destination MALL policy when `supportsMall`, otherwise zero. SYS/SNP/GPA remain zero; the layout has no MTYPE or scope field. |

The GFX10 LLC helper requires Navi2x and the write-bypass setting; CPV requires
a non-default bypass setting and valid KMD SDMA L2 policy. GFX12 uses the
two-bit destination setting: 0 regular temporal, 1 non-temporal, 2
high-priority temporal, 3 last-use. These are cache-policy selections, not a
payload release or an atomicity guarantee.
[GFX10 fence][pal10-fence] [Immediate write][pal10-immediate]
[Event write][pal10-event] [MTYPE value][pal-mtype]
[GFX10 policy][pal10-policy] [GFX12 fence][pal12-fence]
[Immediate write][pal12-immediate] [Event write][pal12-event]
[GFX12 policy][pal12-policy]

### SI legacy DMA form

SI uses a different four-DWORD representation: `DMA_PACKET_FENCE` is 6 in
header bits 31:28. Linux emits the remaining header bits as zero, masks the
low address with `0xfffffffc`, emits only eight high address bits, then the
32-bit value. That emitter therefore carries a 40-bit byte address; it does
not use the later opcode-5 layout. Its optional 64-bit value is still two
packets, low word first. [SI encoding][si-encoding] [SI emitter][si-emitter]

### Linux native emission

The scheduled-ring emitters all implement `AMDGPU_FENCE_FLAG_64BIT` by
writing the low value at A and then the high value at A+4. The flag selects
two FENCE32 packets, not a native 64-bit operation. `AMDGPU_FENCE_FLAG_INT`
controls notification only in the newer emitters. None of these SDMA
emitters consumes the separate `AMDGPU_FENCE_FLAG_TC_WB_ONLY` flag as a
packet cache operation. [Native flags][linux-flags]

| Native backend | FENCE header | Following TRAP |
| --- | --- | --- |
| SI | Legacy opcode 6 at bits 31:28. | Unconditional, one DWORD, legacy opcode 7. [Emitter][si-emitter] |
| CIK | Opcode 5 only. | Unconditional, one DWORD, opcode 6. [Emitter][linux-cik] |
| SDMA2.4 / 3.0 | Opcode 5 only. | Unconditional, two DWORDs, context zero. [2.4][linux-24] [3.0][linux-30] |
| SDMA4.0 / 4.4.2 | Opcode 5 only. | Unconditional, two DWORDs, context zero. [4.0][linux-40] [4.4.2][linux-fence] |
| SDMA5.0 / 5.2 | Opcode 5 and MTYPE=3, described as UC. | Only with INT; two DWORDs, context zero. [5.0][linux-50] [5.2][linux-52] |
| SDMA6.0 / 7.0 | Opcode 5 and MTYPE=3. | Only with INT; two DWORDs, context zero. [6.0][linux-60] [7.0][linux-70] |
| SDMA7.1 | Opcode 5 and MTYPE=3. | Only with INT; two DWORDs, context zero. [7.1][linux-71] |

The native polling emitter passes flags zero, but the older backends still
append their unconditional TRAP. A polling API therefore does not by itself
imply an interrupt-free packet sequence. [Polling emitter][linux-poll-emit]

## Execution and lifetime

A completion sequence consists of payload work, the payload's required cache
release, and the completion update. A consumer acquires the control word under
its mapping's visibility contract before accessing the payload. Cache actions
for the payload and visibility of the control word are independent; a later
payload acquire cannot repair an unreadable completion predicate. [Native
submission sequence](atomics.md#copy-completion-through-add64) [Cache
ownership](cache.md)

### Linux ring completion and user fences

The ordinary scheduled submission has two completion-value domains:

```text
IB execution → selected HDP invalidation → optional user fence low/high
  → native ring-progress FENCE32 → native TRAP → remaining submission framing
```

The older emitters also append a TRAP immediately after the optional user-fence
pair, before the native ring-progress FENCE32.

The optional user fence receives the 64-bit submission-context sequence also
returned as the command-submission handle. The parser resolves and retains
its backing BO. The native ring-progress word instead receives a 32-bit
`sync_seq` belonging to the native ring; `amdgpu_fence_emit` associates that
value with a CPU `dma_fence` and requests INT. These addresses, values and
software objects are distinct. [Submission order][linux-submission]
[Context sequence][linux-context-sequence] [User backing][linux-user-backing]
[User address][linux-user-address] [Handle assignment][linux-user-sequence]
[Native producer][linux-native-producer]

`amdgpu_fence_read` reads one little-endian DWORD from the ring's writeback
mapping. The interrupt handler selects the native ring and calls
`amdgpu_fence_process`, which observes that word and signals the corresponding
CPU fence objects; a timer can invoke the same reader. TRAP is a reason to
check completion, not the completion value itself.
[CPU reader][linux-native-reader] [SDMA7.1 interrupt][linux-interrupt]
[Completion processing][linux-native-process]

The ring's software table has twice the power-of-two outstanding-submission
bound. Indexing is masked, an occupied old entry is waited before replacement,
and polling compares the signed 32-bit difference between requested and
observed values. That bounded wrap protocol does not make an arbitrary
32-bit counter globally ordered, or make the split user-fence write atomic.
[Table initialization][linux-native-table] [Entry reuse][linux-native-producer]
[Polling comparison][linux-native-poll]

On the ordinary successful scheduler path, native completion signals the
job's finished fence; IB suballocation release carries that fence until it
signals. The primary ring is a separately owned persistent allocation whose
commit can append padding after the completion packets. Observing an early
user word does not transfer ownership of all that command storage.
[Scheduler completion][linux-scheduler-completion]
[Native job][linux-job-run] [Callback registration][linux-scheduler-parent]
[Finished fence][linux-scheduler-fence] [IB release owner][linux-job-free]
[Suballocator retirement][linux-suballoc]
[Command framing and storage](command-buffers.md#publication-completion-and-final-use)

### ROCr signal stores

One FENCE32 is not an indivisible 64-bit signal operation. ROCr's non-atomic
completion path computes `LoadRelaxed() - 1` on the CPU, interpreted as an
unsigned 64-bit value. It emits the high-word fence only when that new value
exceeds `UINT32_MAX`, then emits the low-word fence. This is neither an
unconditional high/low assignment nor a concurrent read-modify-write. The
directed atomic-link predicate selecting this path belongs to the
[atomic completion protocol](atomics.md#copy-completion-through-add64).
[Completion sizing][choice] [Emission][completion]

The other FENCE32 roles use their own small-value protocols. A gang leader
acknowledges a peer signal from one to zero after its last poll. Classic
fan-out uses a zero store to open its private prologue gate and, without
platform atomics, one private zero-valued body completion per engine group;
the epilogue joins those bodies before publishing the user result. Those
low-word operations rely on their initialized control values, not on a
general full-width signal assignment.
[Gang acknowledgement][rocr-gang] [Classic gate][rocr-prologue]
[Body completion][rocr-bodies] [Epilogue join][rocr-epilogue]

ROCr may append a mailbox FENCE and TRAP after its output signal update. The
mailbox store writes a 32-bit event ID. This is separate from the ABI's signed
64-bit signal value at byte offset 8 in a 64-byte-aligned structure. The
output value can therefore be visible while notification commands still use
the mailbox. Payload completion, notification completion, primary-ring
consumption, and the final use of each allocation are distinct observations.
[Signal ABI][rocr-signal-abi] [Notification tail][completion]
[Notification owner](atomics.md#memory-and-lifetime)

The [native notification protocol](../notifications.md) follows that mailbox
through KFD interrupt delivery, host wait registration and event reuse. The
decoder's mailbox requirements differ between CP, SDMA and shader interrupts.

ROCr's busy and interrupt host waits compare the complete 64-bit value.
`WaitAcquire` calls the relaxed value loop and then performs an acquire
fence. A blocked event wakeup returns to that value loop; it does not replace
the completion predicate. Host-wait retention protects the signal object
during the wait, not later device mailbox accesses.
[Busy wait][rocr-busy-wait] [Interrupt wait][rocr-interrupt-wait]

CLR provides a concrete payload consumer: its staged device-to-host copy
calls `WaitCurrent` before copying staging bytes into the caller's buffer.
Its `CpuWaitForSignal` and `WaitForSignal` nevertheless have relaxed
already-complete fast paths that skip the acquire-wait call. That source
policy is narrower than an unconditional acquire on every return; it does
not establish a visibility failure or a guarantee for another mapping.
[Payload consumer][clr-staging] [Tracker wait][clr-tracker-wait]
[Wait policy][clr-wait]

### PAL immediate writes and events

`CmdWriteImmediate` emits one FENCE for a 32-bit value. Its 64-bit case emits
the low DWORD at address A, then a second FENCE with the high DWORD at A+4,
for eight command DWORDs total. Both GFX10 and GFX12 require only four-byte
address alignment for this pair. This is not the five-DWORD native 64-bit
form. The builders do not inspect `stageMask`; PAL's public contract limits
SDMA immediate writes to bottom-of-pipe. Its bus-addressable marker update
is one concrete caller selecting bottom-of-pipe and a 32-bit value.
[GFX10 immediate][pal10-immediate] [GFX12 immediate][pal12-immediate]
[Public stage contract][pal-immediate-api] [Marker caller][pal-marker]

PAL GPU events use one four-byte cell with an eight-byte **allocation**
alignment. The set and reset values are `0xdeadbeef` and `0xcafebabe`.
`CmdSetEvent`/`CmdResetEvent` resolve the bound memory and invoke the
generation's FENCE32 event builder. An SDMA waiter uses a memory
`POLL_REGMEM`, equality comparison and a full 32-bit mask against the set
value. The [polling chapter](poll.md) owns the poll's cache and retry fields.
[Event forwarding][pal-event-forward] [Bound-memory lookup][pal-event-write]
[Values][pal-event-values] [Storage requirements][pal-event-memory]
[GFX10 waiter][pal10-event-wait] [GFX12 waiter][pal12-event-wait]

For CPU-visible events, binding maps the caller's allocation and initializes
it to reset. `GetStatus` makes one volatile 32-bit load and recognizes only
the set value as set; this implementation contains no explicit C++ acquire
fence. A `gpuAccessOnly` event instead forbids CPU status/set/reset and leaves
initialization to the client. Neither rebinding nor event destruction waits
for GPU users. Consequently, seeing a set value does not retire another
queue's pending poll or a later reset. The backing remains live through all
users, and resetting the value requires retirement of its prior waiters.
[Binding][pal-event-bind] [CPU observer][pal-event-observer]
[GPU-only contract][pal-event-api] [Destruction][pal-event-destroy]

The SDMA release-event wrapper performs its ordinary barrier work before
setting the event; acquire-event polls the events before its barrier work.
Those wrappers do not turn a control-cell FENCE into an unconditional cache
flush. Command storage has its separate [postamble and allocator
retirement](atomics.md#mem_incr-and-command-allocator-retirement), which can
follow a user event embedded earlier in the stream.
[Event barrier order][pal-event-barriers] [Barrier implementation][pal-barrier]

### Mesa markers, gang handoff and host completion

Mesa's `ac_emit_sdma_fence` emits the four-DWORD form with header
`0x00030005`: opcode 5 and MTYPE=3. It has no generation or stage argument,
adds no SYS/scope field, and emits no TRAP. RADV uses it for transfer-queue
markers and private gang publication. `CmdWriteBufferMarker2AMD` registers
the caller's buffer BO; its transfer branch emits FENCE before returning,
without using the requested pipeline stage. The address-only marker entry
does not itself register a BO. The packet helper supplies no independent
host-wait or allocation-retirement object.
[Common emitter][mesa-emitter] [Encoding][mesa-encoding]
[Marker callers][mesa-markers]

RADV's transfer image-copy path can hand unsupported SDMA image work to an
ACE compute follower. Its command-buffer gang storage contains separate
leader-to-follower and follower-to-leader DWORDs. SDMA publishes with FENCE32;
ACE publishes through a CP release with selected cache writeback and write
confirmation. Each consumer waits for its expected value with a full-mask
greater-or-equal comparison, using the operation for its own engine.
[Concrete transfer caller][mesa-gang-caller] [Publisher and directions][mesa-gang]
[Wait dispatch][mesa-wait]

Those words are zero-initialized in upload storage or a separate eight-byte,
four-byte-aligned VRAM allocation. The separate allocation enforces placement or
noncoherent pairing; GL2_BYPASS is selected for the latter. RADV's actual
SDMA/ACE coherence predicate is `gfx_level <= GFX8 ||
cp_sdma_ge_use_system_memory_scope`, with the flag set for GFX12 in the cited
source. This also selects the noncoherent path for GFX9, despite the nearby
comment naming GFX10–11. It changes allocation policy, not the FENCE header.
[Storage and predicate][mesa-gang-storage] [Device flag][mesa-scope-flag]

Command-buffer finalization clears the private words for resubmission.
Separately, the queue's submitted gang postambles make the leader wait for
the whole gang before its kernel user fence can permit reuse. The intermediate
FENCE32 publications alone do not establish that final use. Transfer barriers
use the separate [pending-transfer drain](ordering.md#pending-transfer-drains).
[Private reset][mesa-gang-reset] [Queue retirement join][mesa-gang-retirement]
[Postamble submission][mesa-gang-submission]

Radeonsi's SDMA image-copy path instead obtains completion from winsys
submission. For kernel queues, winsys requests a user fence and syncobj,
publishes the assigned sequence and mapped word to its CPU fence object,
then permits completion observation. After submission assignment, the host
can compare the mapped 64-bit word against the requested sequence. An
incomplete zero-timeout query returns false; a waiting call can fall back to
the syncobj. This source has no high/low retry protocol proving atomic
observation of the kernel's split write. The fence retains the context owning
the mapping until final context release; winsys' separate 32-bit BO-use
sequence is not the kernel submission sequence.
[Image-copy submission][radeonsi-copy] [Submission request][mesa-host-request]
[Sequence publication][mesa-host-publish] [CPU wait][mesa-host-wait]
[Context retention][mesa-host-owner] [Mapping lifetime][mesa-host-mapping]
[BO sequence type][mesa-bo-sequence] [BO sequence owner][mesa-bo-sequence-owner]

## Native 64-bit form

ROCr's `SDMA_PKT_FENCE_64B_GFX1250` is opcode 5/suboperation 2 and occupies
five DWORDs, or 20 bytes. It carries a 64-bit value and an eight-byte-aligned
destination. Its header uses the ROCr GFX12 positions above: two-bit MTYPE,
SYS, SNP, GPA, scope and temporal hint. The builder zeroes the packet, then
sets MTYPE=3, SYS=1 and SYS scope=3 when `scopeFields` is selected.
[Constants][rocr-constants] [64-bit layout][wide-layout]
[64-bit builder][wide-builder]

| DWORD | Contents |
| --- | --- |
| 0 | Header; emitted as `0x00130205` without scope or `0x03130205` with SYS scope. |
| 1 | Address bits 31:3 in the same bit positions; bits 2:0 reserved zero. |
| 2 | Address bits 63:32. |
| 3 | Immediate value bits 31:0. |
| 4 | Immediate value bits 63:32. |

The actual fused fan-out selector is ISA **major == 12 and minor >= 5**.
Its selected native-wide stores write zero: a private gate when profiling
requires a prologue, and final output after a retained epilogue has polled
the full output value to one. The selected templates omit that epilogue
when neither profiling nor a mailbox needs it; the final body's inline
SIGNAL then performs the completion decrement instead.
[ISA predicate][rocr-wide-predicate] [Owner selection][rocr-wide-owner]
[Gate emission][rocr-wide-gate] [Epilogue predicate][rocr-wide-tail-select]
[Count adjustment][rocr-wide-count] [Joined output][rocr-wide-tail]

The separate prologue/epilogue helpers also contain wide-store branches, but
their fan-out owner calls those helpers only on its non-fused path. Their
presence is not another selected wide protocol. The native packet represents
a 64-bit store, but these runtime callers only write zero. Their use does not
establish indivisible observation by every agent or read-modify-write semantics.
The [fused copy/signaling protocol](atomics.md#gfx125-fused-waitcopysignal)
uses a separate inline arithmetic operation.
[Separate helper selection][rocr-classic-owner]

### Conditional-interrupt declaration

PAL's GFX12 header separately declares `FENCE_CONDITIONAL_INTERRUPT`, opcode
5/suboperation 1, as eight DWORDs. Its policy header matches PAL's ordinary
FENCE but adds `ddw` at bit 31; DWORDs 1–2 carry the fence address, 3–4 the
data, 5–6 a reference address, and 7 `int_context_data`. The declaration and
opcode table establish those fields, but not the comparison rule, DDW
meaning, interrupt-delivery protocol or final-use contract. PAL's ordinary
builders above do not select it. It is distinct from ROCr's suboperation-2
FENCE_64B and the explicit FENCE32/TRAP sequences.
[Suboperation][pal-conditional-op] [Complete layout][pal-conditional-layout]

ROCr's [Windows native SDMA epilogue](windows.md#progress-epilogue-and-notification)
provides a concrete consumer: a 64-bit queue progress write, comparison against
the same address, and the queue's native interrupt context. That transport
uses the packet with an accompanying WDDM progress ID; it does not generalize
the packet to arbitrary event objects.

## Complete transfer handoff

A CPU-observed transfer composes the packet with its ownership protocol:

1. The owner maps payload, commands and completion storage for their actual
   actors, initializes the expected signal state, and publishes input data.
2. SDMA waits for the producer dependency, performs the transfer and the
   selected payload release, then writes the completion value using the
   generation and transport's control-word policy.
3. The consumer checks the value with its protocol's width, comparison and
   wrap rules, and performs the required acquire before reading the payload.
   An interrupt only assists that observation.
4. The owner retains payload through its final consumer, completion storage
   through every writer, waiter and notification tail, and command storage
   through its native retirement mechanism. Resetting a word or seeing one
   intermediate publication cannot retire another engine's pending use.

The [directed cache recipes](cache.md), [native notifications](../notifications.md)
and [command publication](publication.md) supply those separate contracts.

[choice]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L469-L479
[builder]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L2097-L2137
[layouts]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/inc/sdma_registers.h#L612-L690
[linux-fence]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v4_4_2.c#L450-L476
[pal-layout]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/sdma/gfx12/gfx12_merged_sdma_packets.h#L2274-L2322
[completion]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L614-L663
[wide-layout]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/inc/sdma_registers.h#L692-L742
[wide-builder]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L2689-L2711
[cik-encoding]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/cikd.h#L495-L513
[si-encoding]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sid.h#L557-L581
[si-emitter]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/si_dma.c#L106-L126
[linux-iceland]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/iceland_sdma_pkt_open.h#L1745-L1775
[linux-tonga]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/tonga_sdma_pkt_open.h#L1745-L1775
[linux-vega10]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/vega10_sdma_pkt_open.h#L2178-L2208
[linux-navi10]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/navi10_sdma_pkt_open.h#L3461-L3527
[linux-layout6]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v6_0_0_pkt_open.h#L4112-L4190
[linux-layout7]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v7_1_0_pkt_open.h#L4112-L4190
[pal10-layout]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/sdma/gfx10/gfx10_merged_sdma_packets.h#L2202-L2257
[pal10-fence]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/sdma/gfx10/gfx10DmaCmdBuffer.cpp#L325-L352
[pal10-immediate]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/sdma/gfx10/gfx10DmaCmdBuffer.cpp#L1121-L1165
[pal10-event]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/sdma/gfx10/gfx10DmaCmdBuffer.cpp#L1231-L1258
[pal10-policy]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/sdma/gfx10/gfx10DmaCmdBuffer.cpp#L372-L413
[pal12-fence]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/sdma/gfx12/gfx12DmaCmdBuffer.cpp#L323-L343
[pal12-immediate]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/sdma/gfx12/gfx12DmaCmdBuffer.cpp#L906-L943
[pal12-event]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/sdma/gfx12/gfx12DmaCmdBuffer.cpp#L987-L1013
[pal12-policy]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/sdma/gfx12/gfx12DmaCmdBuffer.cpp#L346-L385
[pal10-event-wait]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/sdma/gfx10/gfx10DmaCmdBuffer.cpp#L70-L105
[pal12-event-wait]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/sdma/gfx12/gfx12DmaCmdBuffer.cpp#L88-L120
[pal-mtype]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/gfx9/chip/gfx9_plus_merged_enum.h#L6166-L6175
[pal-immediate-api]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/inc/core/palCmdBuffer.h#L4048-L4067
[pal-marker]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/dmaCmdBuffer.cpp#L1842-L1851
[pal-event-forward]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/cmdBuffer.h#L527-L531
[pal-event-write]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/cmdBuffer.cpp#L1000-L1020
[pal-event-values]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/gpuEvent.h#L54-L56
[pal-event-memory]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/gpuEvent.cpp#L35-L37
[pal-event-bind]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/gpuEvent.cpp#L156-L245
[pal-event-observer]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/gpuEvent.cpp#L107-L150
[pal-event-api]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/inc/core/palGpuEvent.h#L45-L80
[pal-event-destroy]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/gpuEvent.cpp#L60-L82
[pal-event-barriers]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/dmaCmdBuffer.cpp#L347-L380
[pal-barrier]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/dmaCmdBuffer.cpp#L389-L449
[linux-flags]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_ring.h#L62-L64
[linux-cik]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/cik_sdma.c#L278-L299
[linux-24]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v2_4.c#L306-L328
[linux-30]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v3_0.c#L483-L505
[linux-40]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v4_0.c#L885-L911
[linux-50]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v5_0.c#L522-L552
[linux-52]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v5_2.c#L372-L402
[linux-60]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v6_0.c#L356-L386
[linux-70]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v7_0.c#L358-L388
[linux-71]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v7_1.c#L326-L356
[linux-poll-emit]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_fence.c#L159-L181
[linux-submission]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_ib.c#L288-L349
[linux-context-sequence]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_ctx.c#L738-L763
[linux-user-backing]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_cs.c#L134-L156
[linux-user-address]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_cs.c#L992-L998
[linux-user-sequence]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_cs.c#L1339-L1376
[linux-native-producer]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_fence.c#L101-L146
[linux-native-reader]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_fence.c#L63-L90
[linux-interrupt]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/sdma_v7_1.c#L1508-L1548
[linux-native-process]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_fence.c#L207-L267
[linux-native-table]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_fence.c#L443-L469
[linux-native-poll]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_fence.c#L310-L340
[linux-job-free]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_job.c#L296-L312
[linux-job-run]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/amd/amdgpu/amdgpu_job.c#L438-L472
[linux-scheduler-completion]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/scheduler/sched_main.c#L163-L188
[linux-scheduler-parent]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/scheduler/sched_main.c#L1042-L1068
[linux-scheduler-fence]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/scheduler/sched_fence.c#L65-L85
[linux-suballoc]: https://github.com/torvalds/linux/blob/50d05c7c76c96b90462f24debacca971d2e86713/drivers/gpu/drm/drm_suballoc.c#L455-L489
[mesa-emitter]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_cmdbuf_sdma.c#L35-L44
[mesa-encoding]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/sid.h#L325-L343
[mesa-markers]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/vulkan/radv_cmd_buffer.c#L17182-L17231
[mesa-gang-caller]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/vulkan/meta/radv_meta_copy.c#L106-L135
[mesa-gang]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/vulkan/radv_cmd_buffer.c#L1917-L2037
[mesa-wait]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/vulkan/radv_cs.h#L172-L185
[mesa-gang-storage]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/vulkan/radv_cmd_buffer.c#L1832-L1914
[mesa-scope-flag]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/common/ac_gpu_info.c#L1214-L1218
[mesa-gang-reset]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/vulkan/radv_cmd_buffer.c#L2060-L2085
[mesa-gang-retirement]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/vulkan/radv_queue.c#L1441-L1457
[mesa-gang-submission]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/amd/vulkan/radv_queue.c#L1718-L1729
[radeonsi-copy]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/drivers/radeonsi/si_sdma_copy_image.c#L388-L455
[mesa-host-request]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/winsys/amdgpu/drm/amdgpu_cs.cpp#L1398-L1418
[mesa-host-publish]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/winsys/amdgpu/drm/amdgpu_cs.cpp#L2137-L2153
[mesa-host-wait]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/winsys/amdgpu/drm/amdgpu_cs.cpp#L182-L243
[mesa-host-owner]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/winsys/amdgpu/drm/amdgpu_cs.cpp#L30-L60
[mesa-host-mapping]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/winsys/amdgpu/drm/amdgpu_cs.h#L202-L215
[mesa-bo-sequence]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/winsys/amdgpu/drm/amdgpu_winsys.h#L131-L139
[mesa-bo-sequence-owner]: https://gitlab.freedesktop.org/mesa/mesa/-/blob/0ba4b08edc65075e9346d20d5310261939aaaf48/src/gallium/winsys/amdgpu/drm/amdgpu_cs.cpp#L1776-L1815
[rocr-gang]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L580-L612
[rocr-prologue]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L872-L929
[rocr-bodies]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L1526-L1557
[rocr-epilogue]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L998-L1063
[rocr-signal-abi]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/inc/amd_hsa_signal.h#L58-L77
[rocr-busy-wait]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/default_signal.cpp#L77-L121
[rocr-interrupt-wait]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/interrupt_signal.cpp#L142-L212
[clr-staging]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/clr/rocclr/device/rocm/rocblit.cpp#L1137-L1169
[clr-tracker-wait]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/clr/rocclr/device/rocm/rocvirtual.cpp#L955-L964
[clr-wait]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/clr/rocclr/device/rocm/rocvirtual.hpp#L55-L97
[rocr-constants]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/inc/sdma_registers.h#L55-L81
[rocr-wide-predicate]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L187-L205
[rocr-wide-owner]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp#L1706-L1719
[rocr-wide-gate]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L1199-L1242
[rocr-wide-tail-select]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L1088-L1101
[rocr-wide-count]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L1194-L1197
[rocr-wide-tail]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_blit_sdma.cpp#L1291-L1335
[rocr-classic-owner]: https://github.com/ROCm/rocm-systems/blob/8d57824901ffa7d961c00a37d055a108723b93ca/projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp#L1753-L1865
[pal-conditional-op]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/sdma/gfx12/gfx12_merged_sdma_packets.h#L44-L70
[pal-conditional-layout]: https://github.com/GPUOpen-Drivers/pal/blob/c5e800072a32f68b6ccc4422936d96167c6e0728/src/core/hw/gfxip/sdma/gfx12/gfx12_merged_sdma_packets.h#L2324-L2408
