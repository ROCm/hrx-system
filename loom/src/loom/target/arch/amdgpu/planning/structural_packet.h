// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_STRUCTURAL_PACKET_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_STRUCTURAL_PACKET_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/schedule/types.h"
#include "loom/ops/low/ops.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum loom_amdgpu_structural_packet_analysis_flag_bits_e {
  // Treat structural movement as opaque materialization when no allocation
  // table is available.
  LOOM_AMDGPU_STRUCTURAL_PACKET_ANALYSIS_FLAG_REQUIRE_ALLOCATION = 1u << 0,
} loom_amdgpu_structural_packet_analysis_flag_bits_t;
typedef uint32_t loom_amdgpu_structural_packet_analysis_flags_t;

typedef enum loom_amdgpu_structural_packet_flag_bits_e {
  // The op is a structural movement packet such as low.copy/move/slice/concat.
  LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MOVEMENT = 1u << 0,
  // The packet emits concrete target instructions.
  LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES = 1u << 1,
  // The packet forwards dependency-producing values without target work.
  LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_FORWARDS_DEPENDENCIES = 1u << 2,
  // The materialized packet issues at least one vector-ALU instruction.
  LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_USES_VECTOR_ALU = 1u << 3,
  // The materialized packet reads the target-visible SCC state.
  LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_READS_SCC = 1u << 4,
  // Taking the conditional branch skips the packet's final native branch.
  LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_CONDITIONAL_TAIL = 1u << 6,
} loom_amdgpu_structural_packet_flag_bits_t;
typedef uint32_t loom_amdgpu_structural_packet_flags_t;

typedef struct loom_amdgpu_structural_packet_info_t {
  // Classification flags for the structural packet.
  loom_amdgpu_structural_packet_flags_t flags;
  // Scheduled native instructions represented by the structural packet,
  // excluding target insertion overlays.
  uint64_t instruction_count;
  // Final sequential physical moves emitted by the structural packet.
  loom_low_move_range_t moves;
  // Native vector-ALU instructions outside of |moves|.
  uint64_t vector_alu_instruction_count;
  // Native scalar-ALU instructions outside of |moves|.
  uint64_t scalar_alu_instruction_count;
} loom_amdgpu_structural_packet_info_t;

// Returns native issue progress guaranteed along every outgoing path.
static inline uint64_t loom_amdgpu_structural_packet_minimum_instruction_count(
    const loom_amdgpu_structural_packet_info_t* info) {
  return info->instruction_count -
         (iree_any_bit_set(info->flags,
                           LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_CONDITIONAL_TAIL)
              ? 1u
              : 0u);
}

// Returns the number of native control-transfer instructions represented by
// |node|. This is independent of allocation and may be queried before physical
// locations have been assigned.
static inline uint64_t loom_amdgpu_structural_packet_control_transfer_count(
    const loom_low_schedule_table_t* schedule,
    const loom_low_schedule_node_t* node) {
  const loom_op_t* op = node->op;
  if (op == NULL) {
    return 0;
  }
  if (loom_low_return_isa(op)) {
    return 1;
  }
  if (loom_low_br_isa(op)) {
    const uint32_t destination_block_index =
        loom_low_packet_block_index(schedule, loom_low_br_dest(op));
    return destination_block_index == node->block_index + 1 ? 0 : 1;
  }
  if (!loom_low_cond_br_isa(op)) {
    return 0;
  }

  const loom_block_t* true_destination = loom_low_cond_br_true_dest(op);
  const loom_block_t* false_destination = loom_low_cond_br_false_dest(op);
  const uint32_t true_block_index =
      loom_low_packet_block_index(schedule, true_destination);
  const bool true_fallthrough = true_block_index == node->block_index + 1;
  if (true_destination == false_destination) {
    return true_fallthrough ? 0 : 1;
  }
  const uint32_t false_block_index =
      loom_low_packet_block_index(schedule, false_destination);
  const bool false_fallthrough = false_block_index == node->block_index + 1;
  return true_fallthrough || false_fallthrough ? 1 : 2;
}

// Returns AMDGPU native scheduling facts for one structural low packet.
loom_amdgpu_structural_packet_info_t loom_amdgpu_structural_packet_analyze(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_schedule_node_t* node,
    loom_amdgpu_structural_packet_analysis_flags_t analysis_flags);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_STRUCTURAL_PACKET_H_
