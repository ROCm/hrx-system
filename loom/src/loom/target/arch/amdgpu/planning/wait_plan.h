// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU wait-counter planning over scheduled target-low functions.
//
// The shared low scheduler records target-neutral descriptor facts in scheduled
// order. This layer owns the AMDGPU interpretation of those facts: asynchronous
// packets create outstanding counter work, and explicit waits bound or drain
// counters. The immutable plan retains missing waits as insertions and
// redundant authored full waits as elisions. Packetization, fixed-delay
// planning, and native emission consume these decisions without changing the
// source IR.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_PLAN_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_PLAN_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/packet_hazard_plan.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/target/arch/amdgpu/planning/address_state.h"
#include "loom/target/arch/amdgpu/planning/wait_counters.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

struct loom_amdgpu_matrix_result_wait_t;

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_amdgpu_wait_plan_action_kind_e {
  // Unknown or uninitialized action kind.
  LOOM_AMDGPU_WAIT_PLAN_ACTION_UNKNOWN = 0,
  // Wait already exists in the scheduled low stream.
  LOOM_AMDGPU_WAIT_PLAN_ACTION_EXPLICIT = 1,
  // Wait must be inserted before final emission.
  LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED = 2,
} loom_amdgpu_wait_plan_action_kind_t;

typedef enum loom_amdgpu_wait_plan_action_flag_bits_e {
  // Action was derived from an allocation storage-release record.
  LOOM_AMDGPU_WAIT_PLAN_ACTION_FLAG_STORAGE_RELEASE = 1u << 0,
} loom_amdgpu_wait_plan_action_flag_bits_t;
typedef uint8_t loom_amdgpu_wait_plan_action_flags_t;

typedef enum loom_amdgpu_wait_plan_reason_e {
  // Unknown or uninitialized wait reason.
  LOOM_AMDGPU_WAIT_PLAN_REASON_UNKNOWN = 0,
  // Explicit wait packet in the low stream bounds or drains this counter.
  LOOM_AMDGPU_WAIT_PLAN_REASON_EXPLICIT_PACKET = 1,
  // A consumer uses an outstanding asynchronous result.
  LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE = 2,
  // A barrier observes memory that may still have outstanding packets.
  LOOM_AMDGPU_WAIT_PLAN_REASON_BARRIER = 3,
  // Reason identifier 4 is reserved to keep report identifiers stable.
  // A packet overwrites physical registers that still receive an outstanding
  // asynchronous result.
  LOOM_AMDGPU_WAIT_PLAN_REASON_READ_RESULT_REUSE = 5,
  // An RDNA VALU consumes a nearby TRANS result before va_vdst is drained.
  LOOM_AMDGPU_WAIT_PLAN_REASON_TRANS_RESULT_USE = 6,
  // An ALU consumes a rewritten scalar pair previously read by VALU. The
  // native wait-state plan supplies these actions to compile reporting.
  LOOM_AMDGPU_WAIT_PLAN_REASON_SGPR_READ_DEPENDENCY = 7,
  // A memory-effect dependency observes an outstanding memory packet.
  LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_EFFECT = 8,
  // Program exit observes outstanding memory stores.
  LOOM_AMDGPU_WAIT_PLAN_REASON_PROGRAM_EXIT = 9,
  // A packet reuses scalar source registers still consumed by an outstanding
  // memory packet.
  LOOM_AMDGPU_WAIT_PLAN_REASON_MEMORY_SOURCE_REUSE = 10,
  // A packet writes EXEC while outstanding gfx125x VMEM translations still
  // retain the execution mask used when they were issued.
  LOOM_AMDGPU_WAIT_PLAN_REASON_XCNT_EXEC_REUSE = 11,
  // A loop-entry SSA dependency uses a target count derived from its producer
  // position in the preheader counter epoch.
  LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_ENTRY_DERIVED_SSA_USE = 12,
  // A loop-entry SSA dependency requires a conservative full counter drain.
  LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_ENTRY_CONSERVATIVE_SSA_USE = 13,
  // A loop-carried SSA dependency uses a target count derived from the stable
  // producer frontier or a known younger suffix issued in the current block.
  LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_CARRIED_DERIVED_SSA_USE = 14,
  // A loop-carried SSA dependency requires a conservative full counter drain.
  LOOM_AMDGPU_WAIT_PLAN_REASON_LOOP_CARRIED_CONSERVATIVE_SSA_USE = 15,
  // A tensor issue must bound outstanding tensor work independently of
  // aliasing.
  LOOM_AMDGPU_WAIT_PLAN_REASON_TENSOR_ISSUE_DRAIN = 16,
  // A GFX12.0 ordinary SYSTEM store must follow prior memory completion.
  LOOM_AMDGPU_WAIT_PLAN_REASON_SYSTEM_SCOPE_STORE = 17,
  // Total number of wait-plan reason values.
  LOOM_AMDGPU_WAIT_PLAN_REASON_COUNT,
} loom_amdgpu_wait_plan_reason_t;

typedef enum loom_amdgpu_wait_plan_residual_action_e {
  // Unknown or uninitialized residual action id.
  LOOM_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_UNKNOWN = 0,
  // Insert an AMDGPU wait packet before the affected packet.
  LOOM_AMDGPU_WAIT_PLAN_RESIDUAL_ACTION_WAIT_PACKET = 1,
} loom_amdgpu_wait_plan_residual_action_t;

// One AMDGPU wait-counter action in scheduled packet order.
typedef struct loom_amdgpu_wait_plan_action_t {
  // Whether the action is present in the IR or must be inserted.
  loom_amdgpu_wait_plan_action_kind_t kind;
  // Internal origin and materialization flags.
  loom_amdgpu_wait_plan_action_flags_t flags;
  // Why this wait action exists.
  loom_amdgpu_wait_plan_reason_t reason;
  // AMDGPU wait counter affected by the action.
  uint16_t counter_id;
  // Wait target value. Zero drains all outstanding packets for the counter;
  // nonzero bounds the remaining work. Authored bounds need an ordering proof
  // before they can establish completion of an individual producer.
  // ALU waits instead pack the ISA's independent dependency-field bounds.
  uint16_t target_count;
  // Region block containing the insertion point or explicit wait.
  uint32_t block_index;
  // Node before which a planned wait is inserted, or explicit wait node.
  uint32_t node_index;
  // Scheduled ordinal before which a planned wait is inserted, or the explicit
  // wait node's scheduled ordinal.
  uint32_t scheduled_ordinal;
  // Producer node that forced the wait, or LOOM_LOW_SCHEDULE_NODE_NONE.
  uint32_t producer_node;
  // Consumer node that needs the wait, or LOOM_LOW_SCHEDULE_NODE_NONE.
  uint32_t consumer_node;
  // Locally tracked outstanding packet count before the wait action. An
  // authored bound may exceed this count without proving a full drain.
  uint32_t outstanding_before;
} loom_amdgpu_wait_plan_action_t;

// AMDGPU wait-counter table for one scheduled low function.
typedef struct loom_amdgpu_wait_plan_t {
  // Schedule table this plan was built from.
  const loom_low_schedule_table_t* schedule;
  // Optional allocation table used for physical-assignment hazards.
  const loom_low_allocation_table_t* allocation;
  // Canonical counter-progress sidecar for explicit waits and producers.
  loom_low_packet_progress_table_t progress;
  // Canonical residual hazard sidecar for planned wait actions.
  loom_low_packet_hazard_plan_t hazard_plan;
  // AMDGPU implementation action rows in scheduled packet order.
  const loom_amdgpu_wait_plan_action_t* actions;
  // Number of action records.
  iree_host_size_t action_count;
  // Node-indexed bitset of redundant authored full memory waits. NULL when no
  // wait is elided. These nodes emit neither an instruction nor counter
  // progress; packetization, fixed-delay planning, and emission share this
  // decision.
  const uint64_t* elided_wait_nodes;
  // Union of native descriptor traits retained by node classification.
  loom_amdgpu_descriptor_traits_t descriptor_traits;
  // Native DOT identities requiring independent fixed-delay channels.
  uint8_t dot_identity_mask;
} loom_amdgpu_wait_plan_t;

// Returns whether the canonical counter walk proved an authored wait redundant.
// A NULL plan preserves every authored packet.
static inline bool loom_amdgpu_wait_plan_elides_node(
    const loom_amdgpu_wait_plan_t* plan, uint32_t node_index) {
  return plan != NULL && plan->elided_wait_nodes != NULL &&
         (plan->elided_wait_nodes[node_index / 64] &
          (UINT64_C(1) << (node_index % 64))) != 0;
}

// Returns the stable diagnostic spelling for an AMDGPU wait counter id.
iree_string_view_t loom_amdgpu_wait_counter_name(uint16_t counter_id);

// Returns the stable progress-class spelling for an AMDGPU wait counter id.
iree_string_view_t loom_amdgpu_wait_counter_progress_class_name(
    uint16_t counter_id);

// Returns the stable diagnostic spelling for an AMDGPU wait-plan reason.
iree_string_view_t loom_amdgpu_wait_plan_reason_name(
    loom_amdgpu_wait_plan_reason_t reason);

// Returns the stable diagnostic spelling for an AMDGPU residual action id.
iree_string_view_t loom_amdgpu_wait_plan_residual_action_name(
    uint16_t action_id);

// Builds an AMDGPU wait-counter plan from a successful schedule and allocation.
// Coalesced edge segments forward pending completion dependencies; materialized
// segments consume their sources before establishing ready destination values.
// The plan also materializes target storage-release actions requested by
// allocation, including pending result writes and issued memory source reads.
// Planned waits share insertion points across zero-instruction packets, but
// never across block entries or transitions retained in |address_state|.
// The address-state overlay is borrowed only for this call.
// When |out_matrix_results| is present, this traversal retains static MFMA
// exits and source facts in |matrix_arena| for the subsequent fixed-wait
// planner. Both may be NULL when only the counter-wait plan is needed. Builder
// state is allocated from |transient_arena| and may be discarded after this
// function returns. The caller must keep |schedule| and |allocation| immutable
// and |arena| alive for as long as |out_plan| is used.
iree_status_t loom_amdgpu_wait_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_address_state_plan_t* address_state,
    iree_arena_allocator_t* matrix_arena,
    struct loom_amdgpu_matrix_result_wait_t** out_matrix_results,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* transient_arena,
    loom_amdgpu_wait_plan_t* out_plan);

// Appends the common packet hazard JSON representation of |plan| to |builder|.
iree_status_t loom_amdgpu_wait_plan_format_json(
    const loom_amdgpu_wait_plan_t* plan, iree_string_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_PLAN_H_
