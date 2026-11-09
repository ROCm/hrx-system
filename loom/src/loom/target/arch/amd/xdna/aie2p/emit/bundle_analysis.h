// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_ANALYSIS_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_ANALYSIS_H_

#include "iree/base/internal/arena.h"
#include "loom/codegen/low/frame.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/leaf_program.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_aie2p_block_terminator_e {
  LOOM_AIE2P_BLOCK_TERMINATOR_NONE = 0,
  LOOM_AIE2P_BLOCK_TERMINATOR_BRANCH = 1,
  LOOM_AIE2P_BLOCK_TERMINATOR_CONDITIONAL_BRANCH = 2,
  LOOM_AIE2P_BLOCK_TERMINATOR_RETURN = 3,
} loom_aie2p_block_terminator_t;

typedef struct loom_aie2p_block_branch_t {
  // Native branch descriptor selected for the block terminator.
  uint32_t descriptor_ordinal;
  // Source-order destination block used by the final branch fixup.
  uint32_t target_block_index;
} loom_aie2p_block_branch_t;

typedef enum loom_aie2p_block_alignment_e {
  // Fallthrough-only entry with no address constraint.
  LOOM_AIE2P_BLOCK_ALIGNMENT_NONE = 0,
  // Forward-only branch destination that may enter a preceding NOP tail.
  LOOM_AIE2P_BLOCK_ALIGNMENT_NOP_PREFIX = 1,
  // Function entry or backward destination: align without an added NOP prefix.
  LOOM_AIE2P_BLOCK_ALIGNMENT_DIRECT = 2,
} loom_aie2p_block_alignment_t;

typedef struct loom_aie2p_hardware_loop_analysis_t {
  // Dedicated predecessor that executes the loop setup once.
  uint16_t preheader_block_index;
  // Comparison-only natural-loop header omitted by hardware execution.
  uint16_t header_block_index;
  // Single loop body and latch executed by LS/LE.
  uint16_t body_block_index;
  // Unique fallthrough after the final hardware iteration.
  uint16_t continuation_block_index;
  // Exact positive architectural loop count.
  uint16_t trip_count;
  // Whether structural analysis admitted this candidate.
  bool is_candidate;
} loom_aie2p_hardware_loop_analysis_t;

typedef struct loom_aie2p_block_analysis_t {
  // Structural terminator packet ending this block.
  uint32_t terminator_packet_index;
  // Logical issue cycle at which the structural terminator completes.
  uint32_t terminator_issue_cycle;
  // Structural control kind selected for this block.
  loom_aie2p_block_terminator_t terminator;
  // Allocation-owned parallel result transport before a native return.
  loom_low_move_range_t return_moves;
  // Selected physical branches, shared by destination alignment and emission.
  struct {
    // Native branches in emission order; fallthrough contributes no entry.
    loom_aie2p_block_branch_t values[2];
    // Number of selected native branches, from zero through two.
    uint8_t count;
  } branches;
  // Entry alignment required by the selected native incoming branches.
  loom_aie2p_block_alignment_t alignment;
  // Hardware-loop candidate whose comparison header is this block.
  loom_aie2p_hardware_loop_analysis_t hardware_loop;
  // True after hardware-loop selection removes this block's native control.
  bool elide_control;
} loom_aie2p_block_analysis_t;

typedef struct loom_aie2p_bundle_plan_analysis_t {
  // Per-block control and cycle summaries in source block order.
  loom_aie2p_block_analysis_t* blocks;
  // Resolved route for each allocation move, owned by construction scratch.
  loom_aie2p_move_route_id_t* move_route_ids_by_move;
  // Native instruction slots required by packet and edge allocation moves.
  iree_host_size_t move_slot_count;
  // Conservative branch, return, and delay-bundle capacity.
  iree_host_size_t control_bundle_capacity;
  // Direct module-local calls requiring final symbol fixups.
  iree_host_size_t call_fixup_count;
  // Accepted local-storage address fixups in source packet order.
  loom_aie2p_planned_storage_fixup_t* storage_fixups;
  // Address fixup index for each scheduled packet, or UINT32_MAX when absent.
  uint32_t* address_fixup_indices_by_packet;
  // Number of accepted records in |storage_fixups|.
  iree_host_size_t storage_fixup_count;
  // Symbolic read-only data addresses requiring placement fixups.
  iree_host_size_t read_only_data_fixup_count;
  // Structurally admitted hardware-loop candidates.
  iree_host_size_t hardware_loop_candidate_count;
} loom_aie2p_bundle_plan_analysis_t;

// Classifies the scheduled CFG, admits final-frame semantics, and computes
// exact plan storage capacities. A diagnosed rejection returns OK with
// |out_accepted| false and no plan may consume the partial analysis.
iree_status_t loom_aie2p_bundle_plan_analyze(
    const loom_low_emission_frame_t* frame,
    const loom_low_storage_layout_t* storage_layout,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_allocator_t* plan_arena, iree_arena_allocator_t* scratch_arena,
    loom_aie2p_bundle_plan_analysis_t* out_analysis, bool* out_accepted);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_ANALYSIS_H_
