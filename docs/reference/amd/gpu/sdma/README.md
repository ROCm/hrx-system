# SDMA

SDMA transfers data and executes control operations independently of shader
dispatch. Its packet representation, memory routing, and completion protocol
depend on the engine generation and native transport.

| Topic | Mechanisms |
| --- | --- |
| [Engine selection](engine-selection.md) | Ordinary/xGMI queue pools, directed engine masks, copy executors, runtime blit/native engine identities, read-only queue observations, and topology routing. |
| [Queue context](context.md) | KFD MQD families, `RB_CNTL`, ring and pointer units, doorbell offsets, manual/HWS/MES installation, WPTR polling and retained storage. |
| [Queue publication](publication.md) | Byte frontiers, reservation and ordered commit, wrap/padding, native visibility and storage ownership. |
| [Windows ring submission](windows.md) | Native gfx12+ SDMA admission, `AmdQueueT`, kernel-published byte frontiers, progress-fence epilogues, and dependency-polling policy. |
| [Device-generated commands](device-publication.md) | GPUVM doorbells, shader-authored packets, pre-WPTR visibility, lane progress, and separate command/payload credits. |
| [Linear copy](copy.md) | `COPY_LINEAR` / legacy `DMA_PACKET_COPY`: complete source layouts, metadata and policy selection, native count families, exact byte extents, chunking and storage ownership. |
| [Buffer exchange](swap.md) | `COPY_LINEAR_SWAP`, `COPY_LINEAR_SWAP_WAITSIGNAL_GFX1250`: equal-sized read/write operands, aligned chunks, engine joins and residency ownership. |
| [Indirect source and destination](indirect-copy.md) | `COPY_LINEAR_WAITSIGNAL_INDIRECT_GFX1250`, `LINEAR_INDIRECT_SRC` / `DST` / `SRCDST`: execution-time payload addresses, fixed lengths, slot publication and retirement. |
| [Broadcast and multicast](fanout.md) | `COPY_LINEAR_BROADCAST` / `COPY_BROADCAST_LINEAR`, `COPY_MULTICAST`, fused wait/signal blocks, destination pairing, and joins across copy engines. |
| [Batch composition](fanout.md#batch-composition-and-descriptor-ownership) | `hsa_amd_memory_async_batch_copy`: operation/entry/packet counts, agent and engine grouping, descriptor-control propagation and per-operation completion ownership. |
| [Rectangular copy](rectangular-copy.md) | `COPY_LINEAR_SUBWIN` / `COPY_LINEAR_RECT`: element units, row/slice pitches, subwindow layouts, tiling and geometry-specific cache controls. |
| [Image transfers](images.md) | `COPY_TILED_SUBWIN` / `COPY_T2T_SUBWIN`: native surface and mip operands, metadata layouts, compression admission, staged transfers and temporary-storage reuse. |
| [Constant fill](fill.md) | `CONST_FILL` / `CONSTANT_FILL`: legacy and DWORD count forms, chunk limits, cache/compression fields, caller selection and completion. |
| [Inline data writes](write.md) | `WRITE_LINEAR` / `WRITE_UNTILED` / legacy `DMA_PACKET_WRITE`: direct versus minus-one counts, policy layouts, metadata/control callers, inline versus uploaded input and final storage use. |
| [Page-table updates](page-tables.md) | `PTEPDE_GEN` / `GENERATE_PTE_PDE`, ordinary COPY/WRITE, dedicated PTEPDE COPY/RMW layouts, native entry construction, IB data ownership and translation readiness. |
| [Ordering](ordering.md) | `NOP` / legacy `DMA_PACKET_NOP`: counted bodies, firmware burst selection, framing versus transfer drains, temporary-buffer reuse, overlap and NPD. |
| [Completion stores](fence.md) | `FENCE` / `FENCE_64B` / `FENCE_CONDITIONAL_INTERRUPT`: generation layouts, split stores, native ring versus user completion, event/marker observers and final storage use. |
| [Interrupts](trap.md) | `TRAP` / `DUMMY_TRAP`: context widths, native callback forms, mailbox notification, ring-fence observation and resource lifetime. |
| [Memory dependencies](poll.md) | `POLL_REGMEM` / `POLL_MEM_64B`: legacy/classic/wide layouts, policy and retry fields, dependency callers, signal lifetime, and progress versus terminal joins. |
| [Conditional execution](conditional.md) | `COND_EXE`: guarded DWORD ranges, Boolean64 sampling, per-execution predicate state and driver-owned submission conditions. |
| [Atomic operations and signaling](atomics.md) | `ATOMIC` (`ADD64`), `MEM_INCR`, `SEMAPHORE` / legacy `DMA_PACKET_SEMAPHORE`: complete source layouts, directed atomic reach, caller selection, tracker modes and fused signal ownership. |
| [Cache maintenance](cache.md) | `USER_GCR` / `GCR_REQ` / `GCR_USER`: complete layouts, control-word differences, caller selection, kernel `EMIT_MEM_SYNC`, HDP predicates and release/acquire ownership. |
| [Command buffers](command-buffers.md) | Generation-specific IB entries, body and submission alignment, context storage, direct rings and scheduled retirement. |
| [Timestamps](timing.md) | `TIMESTAMP_GET_GLOBAL`: policy layouts, transfer/gang/fanout intervals, sample readiness and lifetime, and native versus translated clock units. |

Payload visibility, a control-word update, notification, and storage
retirement are separate edges. The programming sequences identify which
operation establishes each edge and which actor still owns the referenced
memory.
