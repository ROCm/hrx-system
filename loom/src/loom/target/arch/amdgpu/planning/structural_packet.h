// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_STRUCTURAL_PACKET_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_STRUCTURAL_PACKET_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/schedule/types.h"

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
  // The materialized packet reads the target-visible SCC state.
  LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_READS_SCC = 1u << 4,
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
