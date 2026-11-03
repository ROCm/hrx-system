// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_SHORT_WAIT_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_SHORT_WAIT_H_

#include "loom/target/arch/amdgpu/planning/structural_packet.h"
#include "loom/target/arch/amdgpu/planning/wait_states.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_short_wait_t loom_amdgpu_short_wait_t;

// Fixed ALU reasons occupy their wait-state reason bit. Store retention uses
// two private channels to distinguish VALU and other physical overwrites. DOT
// identities have independent channels for opcode-specific forwarding.
typedef uint32_t loom_amdgpu_short_wait_reasons_t;

enum {
  // VMEM scalar reads have a longer lifetime than ordinary VALU scalar reads.
  LOOM_AMDGPU_SHORT_WAIT_VMEM_SGPR_READ = 1u << 8,
  // DIV_FMAS reads the VCC condition through its four-slot forwarding path.
  LOOM_AMDGPU_SHORT_WAIT_DIV_FMAS_CONDITION = 1u << 9,
  // Address and message consumers read M0 after scalar ALU writes.
  LOOM_AMDGPU_SHORT_WAIT_M0_READ = 1u << 10,
  // PERMLANE reads need separation from VALU writers, unlike DPP's any-writer
  // dependency.
  LOOM_AMDGPU_SHORT_WAIT_PERMLANE_READ = 1u << 11,
  // Six native DOT identities retain separate three-slot release channels.
  LOOM_AMDGPU_SHORT_WAIT_DOT_SHIFT = 12,
  LOOM_AMDGPU_SHORT_WAIT_DOT_MASK = 63u << LOOM_AMDGPU_SHORT_WAIT_DOT_SHIFT,
  // Independent FMA64 instructions wait for the DGEMM pipeline to release.
  LOOM_AMDGPU_SHORT_WAIT_DGEMM_PIPELINE = 1u << 18,
  // DGEMM activates still-fresh VALU writes needing memory-read separation.
  LOOM_AMDGPU_SHORT_WAIT_DGEMM_MEMORY_SHIFT = 19,
  LOOM_AMDGPU_SHORT_WAIT_DGEMM_MEMORY =
      1u << LOOM_AMDGPU_SHORT_WAIT_DGEMM_MEMORY_SHIFT,
};

typedef struct loom_amdgpu_short_wait_match_t {
  // Reason requiring the largest unsatisfied delay.
  loom_amdgpu_wait_state_reason_t reason;
  // Local producer node, or NONE for a joined incoming obligation.
  uint32_t producer_node;
  // Original instruction separation for the matched reason.
  uint16_t required_cycles;
  // Instruction separation supplied before the consumer.
  uint16_t observed_cycles;
  // Remaining instruction slots to insert before the packet.
  uint16_t cycles;
} loom_amdgpu_short_wait_match_t;

typedef iree_status_t (*loom_amdgpu_short_wait_emit_fn_t)(
    void* user_data, const loom_amdgpu_wait_state_t* wait_state);

// Selects bounded physical dependency channels from retained classification
// and processor facts. Returns NULL when no mechanism is enabled. CFG summaries
// are allocated only when the function has edges.
iree_status_t loom_amdgpu_short_wait_allocate(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    loom_amdgpu_processor_scheduling_bits_t scheduling,
    loom_amdgpu_descriptor_traits_t traits, uint8_t dot_identity_mask,
    iree_arena_allocator_t* arena, loom_amdgpu_short_wait_t** out_state);

// Starts collection for a block in the owning final-packet traversal.
void loom_amdgpu_short_wait_begin_block(loom_amdgpu_short_wait_t* state,
                                        uint16_t block_index);

// Begins a native packet and matches retained inputs and pipeline obligations.
// Further physical reads are queried before publishing any packet results.
loom_amdgpu_short_wait_match_t loom_amdgpu_short_wait_begin_packet(
    loom_amdgpu_short_wait_t* state, const loom_low_packet_view_t* packet,
    const loom_amdgpu_structural_packet_info_t* structural,
    loom_amdgpu_descriptor_traits_t traits);

// Queries one physical range at its native instruction offset within a packet.
// Retains its prefix footprint for predecessor resolution. Assignments for
// non-register locations and other physical register classes are ignored.
loom_amdgpu_short_wait_match_t loom_amdgpu_short_wait_query(
    loom_amdgpu_short_wait_t* state, uint16_t register_class, uint32_t base,
    uint16_t count, uint32_t instruction_offset,
    loom_amdgpu_short_wait_reasons_t reasons);

// Publishes one physical result after its producing native instruction. The
// offset names that instruction's end relative to the current packet start.
// Multiple reasons for the same result share one retained physical span.
void loom_amdgpu_short_wait_record(loom_amdgpu_short_wait_t* state,
                                   uint32_t producer_node,
                                   uint16_t register_class, uint32_t base,
                                   uint16_t count,
                                   uint32_t instruction_end_offset,
                                   loom_amdgpu_short_wait_reasons_t reasons,
                                   uint16_t cycles);

// Accounts for actual inserted native instruction progress.
void loom_amdgpu_short_wait_advance(loom_amdgpu_short_wait_t* state,
                                    uint64_t cycles);

// Retires every active short dependency at an opaque native boundary. The
// returned local match covers producers in the current block; incoming CFG
// obligations are retained for resolution at the same packet boundary.
loom_amdgpu_short_wait_match_t loom_amdgpu_short_wait_drain(
    loom_amdgpu_short_wait_t* state, const loom_low_packet_view_t* packet,
    uint32_t instruction_offset);

// Completes store payload reads without advancing independent ALU windows.
void loom_amdgpu_short_wait_drain_stores(loom_amdgpu_short_wait_t* state);

// Publishes retained prefix queries and native packet progress after local
// inserted delays. Retained inputs, DOT results, and pipeline transitions are
// published by this commit.
void loom_amdgpu_short_wait_commit(loom_amdgpu_short_wait_t* state,
                                   const loom_low_packet_view_t* packet);

// Retains outgoing physical obligations and guaranteed native issue progress.
iree_status_t loom_amdgpu_short_wait_end_block(loom_amdgpu_short_wait_t* state);

// Solves the existing CFG over retained summaries and emits boundary waits in
// scheduled order. The resolver never revisits IR or allocation operands.
iree_status_t loom_amdgpu_short_wait_resolve(
    loom_amdgpu_short_wait_t* state, loom_amdgpu_short_wait_emit_fn_t emit,
    void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_SHORT_WAIT_H_
