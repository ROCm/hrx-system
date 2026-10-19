// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU hardware wait-state planning over scheduled target-low functions.
//
// AMDGPU has hazards that are not modeled by wait counters. CDNA MFMA/SMFMAC
// packets need fixed scalar no-op cycles both after matrix results before
// ordinary VGPR/AGPR consumers and after legacy VALU writes before matrix
// source, DPP, or fixed-lane VGPR-to-SGPR reads. GFX940-family transcendental
// VALU results and destination-selected sub-DWORD writes also need fixed waits
// before dependent VALU consumers, and nearby VALU or VMEM reads of
// VALU-written SGPRs need fixed waits because the hardware does not interlock
// those dependencies.
// GFX11 wave64 mask storage needs destination depctr waits after ALU reuse.
// RDNA3+ processors can use `s_delay_alu` for short ALU dependency windows.
// This table records target-owned insertion points after scheduling and
// allocation, where physical register identity is known.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_STATES_H_
#define LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_STATES_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/allocation.h"
#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/packet_hazard_plan.h"
#include "loom/target/arch/amdgpu/planning/matrix_wait_states.h"

#ifdef __cplusplus
extern "C" {
#endif

struct loom_amdgpu_vopd_plan_t;
struct loom_amdgpu_wait_packet_plan_t;
struct loom_amdgpu_matrix_coexecution_t;
struct loom_amdgpu_matrix_result_wait_t;
struct loom_amdgpu_processor_info_t;
struct loom_amdgpu_address_state_plan_t;

// Maximum number of cycles a single `s_nop` packet can wait. The SOPP SIMM16
// operand is encoded as wait_cycles - 1, and current AMDGPU targets use a
// 4-bit no-op payload.
#define LOOM_AMDGPU_WAIT_STATE_MAX_S_NOP_CYCLES 16u

typedef enum loom_amdgpu_wait_state_reason_e {
  // Unknown or uninitialized wait-state reason.
  LOOM_AMDGPU_WAIT_STATE_REASON_UNKNOWN = 0,
  // A packet consumes VGPR/AGPR storage produced by an outstanding MFMA/SMFMAC.
  LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_RESULT_USE = 1,
  // An MFMA/SMFMAC consumes VGPR storage written by a legacy VALU packet.
  LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE = 2,
  // A non-transcendental VALU consumes a GFX940-family transcendental result.
  LOOM_AMDGPU_WAIT_STATE_REASON_TRANS_RESULT_USE = 3,
  // A VALU or VMEM packet consumes SGPR storage written by a recent VALU
  // packet.
  LOOM_AMDGPU_WAIT_STATE_REASON_VALU_SGPR_READ = 4,
  // A lane-permute packet consumes VGPR storage written by a recent VALU
  // packet.
  LOOM_AMDGPU_WAIT_STATE_REASON_LANE_PERMUTE_VGPR_READ = 5,
  // A fixed-lane VGPR-to-SGPR read consumes VGPR storage written by a recent
  // VALU packet.
  LOOM_AMDGPU_WAIT_STATE_REASON_READFIRSTLANE_VGPR_READ = 6,
  // A VALU packet consumes VGPR storage written by a recent destination
  // selector forwarding producer.
  LOOM_AMDGPU_WAIT_STATE_REASON_DST_SEL_FORWARDING_USE = 7,
  // An RDNA3+ ALU packet consumes storage written by a recent ALU packet.
  LOOM_AMDGPU_WAIT_STATE_REASON_DELAY_ALU_DEPENDENCY = 8,
  // A matrix packet reads a result retained by matrix/vector coexecution.
  LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_COEXECUTION_MATRIX_USE = 9,
  // An ordinary VALU packet reads or overwrites storage retained by
  // matrix/vector coexecution.
  LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_COEXECUTION_VALU_USE = 10,
  // A writer reuses VGPR payload storage still read by a wide VMEM store.
  LOOM_AMDGPU_WAIT_STATE_REASON_STORE_DATA_REUSE = 11,
  // An ALU instruction overwrites storage previously read as a lane mask.
  LOOM_AMDGPU_WAIT_STATE_REASON_MASK_WRITE = 12,
  // DIV_FMAS consumes a condition written to VCC by a recent VALU packet.
  LOOM_AMDGPU_WAIT_STATE_REASON_DIV_FMAS_CONDITION = 13,
  // An address or message consumer reads M0 after a scalar ALU write.
  LOOM_AMDGPU_WAIT_STATE_REASON_M0_READ = 14,
  // A DOT result is read or overwritten without same-opcode forwarding.
  LOOM_AMDGPU_WAIT_STATE_REASON_DOT_RESULT_USE = 15,
  // F64 FMA/FMAC issues before an independent DGEMM pipeline releases.
  LOOM_AMDGPU_WAIT_STATE_REASON_DGEMM_PIPELINE = 16,
  // An intervening DGEMM disabled the VALU-to-memory hardware interlock.
  LOOM_AMDGPU_WAIT_STATE_REASON_DGEMM_MEMORY = 17,
  // A replayable memory clause would overwrite an address or mix load/store.
  LOOM_AMDGPU_WAIT_STATE_REASON_XNACK_REPLAY_CLAUSE = 18,
  // An ALU reads a rewritten scalar pair previously consumed by VALU.
  LOOM_AMDGPU_WAIT_STATE_REASON_SGPR_READ_DEPENDENCY = 19,
  // Number of wait-state reasons, including UNKNOWN.
  LOOM_AMDGPU_WAIT_STATE_REASON_COUNT_ = 20,
} loom_amdgpu_wait_state_reason_t;

typedef enum loom_amdgpu_wait_state_action_e {
  // Unknown or uninitialized wait-state action.
  LOOM_AMDGPU_WAIT_STATE_ACTION_UNKNOWN = 0,
  // Scalar no-op packet that waits one or more cycles.
  LOOM_AMDGPU_WAIT_STATE_ACTION_S_NOP = 1,
  // Scalar delay packet with a packed S_DELAY_ALU dependency immediate.
  LOOM_AMDGPU_WAIT_STATE_ACTION_S_DELAY_ALU = 2,
  // Vector no-op packet contributing one vector issue slot.
  LOOM_AMDGPU_WAIT_STATE_ACTION_V_NOP = 3,
  // Dependency-counter wait for selected ALU destination fields.
  LOOM_AMDGPU_WAIT_STATE_ACTION_S_WAITCNT_DEPCTR = 4,
  // GFX12 dependency-counter wait for selected scalar destinations.
  LOOM_AMDGPU_WAIT_STATE_ACTION_S_WAIT_ALU = 5,
} loom_amdgpu_wait_state_action_t;

// One hardware wait action in scheduled packet and instruction order.
typedef struct loom_amdgpu_wait_state_t {
  // Why this wait state exists.
  loom_amdgpu_wait_state_reason_t reason;
  // Concrete residual packet kind used to satisfy the wait.
  loom_amdgpu_wait_state_action_t action;
  // Region block containing the insertion point.
  uint32_t block_index;
  // Schedule node containing the residual insertion boundary.
  uint32_t node_index;
  // Scheduled ordinal containing the residual insertion boundary.
  uint32_t scheduled_ordinal;
  // Native instruction boundary within the packet: zero precedes it, one
  // follows a descriptor instruction, and N follows the Nth resolved move.
  // Target insertion overlays do not contribute to this offset.
  uint32_t instruction_offset;
  // Producer node that forced the wait.
  uint32_t producer_node;
  // Consumer node that needs the wait, or NONE for a post-write barrier.
  uint32_t consumer_node;
  // Required target progress before |consumer_node| may read the hazard.
  // Dependency-counter barriers do not claim a fixed cycle count.
  uint16_t required_cycle_count;
  // Target progress already supplied before the residual wait.
  uint16_t observed_cycle_count;
  // Residual cycles to wait before |consumer_node|.
  uint16_t cycle_count;
  // Packed SIMM16 operand for S_DELAY_ALU or S_WAITCNT_DEPCTR actions.
  uint16_t immediate;
  // Matrix result wait table profile, or UNKNOWN for non-matrix reasons.
  loom_amdgpu_matrix_wait_profile_t matrix_wait_profile;
  // Matrix result use table key, or UNKNOWN for non-matrix reasons.
  loom_amdgpu_matrix_wait_result_use_t matrix_result_use;
  // Matrix result pass count used for the wait table lookup.
  uint16_t matrix_pass_count;
} loom_amdgpu_wait_state_t;

// AMDGPU hardware wait-state table for one scheduled and allocated low
// function.
typedef struct loom_amdgpu_wait_state_plan_t {
  // Schedule table this plan was built from.
  const loom_low_schedule_table_t* schedule;
  // Allocation table this plan was built from.
  const loom_low_allocation_table_t* allocation;
  // Target progress facts used to explain independent intervening instructions.
  loom_low_packet_progress_table_t progress;
  // Common residual hazard sidecar for fixed wait-state actions.
  loom_low_packet_hazard_plan_t hazard_plan;
  // Wait states in scheduled packet order.
  const loom_amdgpu_wait_state_t* states;
  // Number of wait-state records.
  iree_host_size_t state_count;
  // Hardware allocation floor needed by final physical operand hazards.
  uint32_t minimum_vgpr_count;
} loom_amdgpu_wait_state_plan_t;

struct loom_amdgpu_processor_properties_t;

// Returns the stable spelling for a wait-state reason.
iree_string_view_t loom_amdgpu_wait_state_reason_name(
    loom_amdgpu_wait_state_reason_t reason);

// Returns the stable spelling for a wait-state residual action.
iree_string_view_t loom_amdgpu_wait_state_action_name(
    loom_amdgpu_wait_state_action_t action);

// Builds AMDGPU hardware wait-state insertions from the final scheduled,
// allocated, and VOPD-packetized low function. Elided authored waits in
// |wait_packets->wait_plan| provide no instruction-slot progress. Concrete
// wait insertions supply native progress. |vopd_plan| may be NULL
// when the target has no native packetization. Processor properties come from
// the resolved target record and must be present. Concrete |address_state|
// transitions also separate replayable memory clauses. The caller must keep the
// input plans immutable and |arena| alive for as long as |out_plan| is used.
iree_status_t loom_amdgpu_wait_state_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const struct loom_amdgpu_processor_properties_t* processor_properties,
    const struct loom_amdgpu_address_state_plan_t* address_state,
    const struct loom_amdgpu_wait_packet_plan_t* wait_packets,
    const struct loom_amdgpu_vopd_plan_t* vopd_plan,
    struct loom_amdgpu_matrix_coexecution_t* matrix_coexecution,
    struct loom_amdgpu_matrix_result_wait_t* matrix_results,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* transient_arena,
    loom_amdgpu_wait_state_plan_t* out_plan);

// Returns the number of concrete wait-state instructions needed by |plan|.
uint64_t loom_amdgpu_wait_state_plan_instruction_count(
    const loom_amdgpu_wait_state_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PLANNING_WAIT_STATES_H_
