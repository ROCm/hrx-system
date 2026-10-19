// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_SGPR_READ_WAIT_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_SGPR_READ_WAIT_H_

#include "loom/target/arch/amdgpu/planning/structural_packet.h"
#include "loom/target/arch/amdgpu/planning/wait_states.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // The native instruction issues on SALU.
  LOOM_AMDGPU_SGPR_READ_WAIT_SCALAR = 1,
  // The native instruction issues on VALU.
  LOOM_AMDGPU_SGPR_READ_WAIT_VECTOR = 2,
  // A native interlock or dependency wait completes selected destinations.
  LOOM_AMDGPU_SGPR_READ_WAIT_DRAIN = 3,
};

enum {
  // Complete scalar-written SGPR and VCC destinations.
  LOOM_AMDGPU_SGPR_READ_WAIT_SA_SDST = 1u << 0,
  // Complete vector-written ordinary SGPR destinations.
  LOOM_AMDGPU_SGPR_READ_WAIT_VA_SDST = 1u << 1,
  // Complete vector-written VCC destinations.
  LOOM_AMDGPU_SGPR_READ_WAIT_VA_VCC = 1u << 2,
};

// Final native instruction facts. Bits name hardware scalar DWORDs, including
// VCC at 106:107 and ABI-fixed TTMP at 108:123; EXEC, M0, and null are
// excluded.
typedef struct loom_amdgpu_sgpr_read_wait_event_t {
  // Scalar DWORDs read by this native instruction.
  uint64_t reads[2];
  // Scalar DWORDs written after the instruction reads its inputs.
  uint64_t writes[2];
  // Owning scheduled packet, including structural movement packets.
  uint32_t packet_index;
  // Native instruction boundary before this instruction within its packet.
  uint32_t instruction_offset;
  // SALU, VALU, or an explicit destination-completion event.
  uint8_t kind;
  // Destination fields completed by a DRAIN event.
  uint8_t drain_fields;
} loom_amdgpu_sgpr_read_wait_event_t;

typedef struct loom_amdgpu_sgpr_read_wait_t loom_amdgpu_sgpr_read_wait_t;

typedef iree_status_t (*loom_amdgpu_sgpr_read_wait_emit_fn_t)(
    void* user_data, const loom_amdgpu_wait_state_t* wait_state);

// Creates invocation-scoped state only for a processor with this hazard.
iree_status_t loom_amdgpu_sgpr_read_wait_allocate(
    const loom_low_schedule_table_t* schedule, iree_arena_allocator_t* arena,
    loom_amdgpu_sgpr_read_wait_t** out_state);

// Retains relevant physical instruction facts in final native order. The
// owning packet traversal supplies moves individually, not as SSA bundles.
iree_status_t loom_amdgpu_sgpr_read_wait_collect(
    loom_amdgpu_sgpr_read_wait_t* state, uint16_t block_index,
    const loom_amdgpu_sgpr_read_wait_event_t* event);

// Collects final descriptor operands or sequential structural moves. A VOPD
// second packet shares the first packet's single native instruction boundary.
iree_status_t loom_amdgpu_sgpr_read_wait_collect_packet(
    loom_amdgpu_sgpr_read_wait_t* state,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_view_t* packet,
    const loom_low_packet_view_t* second_packet,
    const loom_amdgpu_structural_packet_info_t* structural,
    loom_amdgpu_descriptor_traits_t traits);

// Consumes a retained ALU-counter action; the packed bound is decoded once by
// counter planning, including explicit partial destination-field waits.
iree_status_t loom_amdgpu_sgpr_read_wait_collect_wait(
    loom_amdgpu_sgpr_read_wait_t* state, const loom_low_packet_view_t* packet,
    uint16_t bound);

// Propagates fixed-size block summaries, then emits residual dependency waits.
// Neither CFG solve visits instructions, IR, or allocation records.
iree_status_t loom_amdgpu_sgpr_read_wait_resolve(
    loom_amdgpu_sgpr_read_wait_t* state,
    loom_amdgpu_sgpr_read_wait_emit_fn_t emit, void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_SGPR_READ_WAIT_H_
