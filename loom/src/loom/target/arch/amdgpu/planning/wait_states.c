// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_states.h"

#include <inttypes.h>
#include <string.h>

#include "loom/codegen/low/packet.h"
#include "loom/codegen/low/packet_hazard_plan_json.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/encoding/encoding.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/target/arch/amdgpu/planning/descriptor_semantics.h"
#include "loom/target/arch/amdgpu/planning/matrix_coexecution.h"
#include "loom/target/arch/amdgpu/planning/matrix_wait_states.h"
#include "loom/target/arch/amdgpu/planning/structural_packet.h"
#include "loom/target/arch/amdgpu/planning/vopd_plan.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

#define LOOM_AMDGPU_WAIT_STATE_VALU_TO_MATRIX_CYCLES 2u
#define LOOM_AMDGPU_WAIT_STATE_TRANS_RESULT_USE_CYCLES 1u
#define LOOM_AMDGPU_WAIT_STATE_VALU_SGPR_READ_CYCLES 2u
#define LOOM_AMDGPU_WAIT_STATE_DPP_VGPR_READ_CYCLES 2u
#define LOOM_AMDGPU_WAIT_STATE_READFIRSTLANE_VGPR_READ_CYCLES 1u
#define LOOM_AMDGPU_WAIT_STATE_DST_SEL_FORWARDING_CYCLES 1u

#define LOOM_AMDGPU_DELAY_ALU_VALU_MAX 5u
#define LOOM_AMDGPU_DELAY_ALU_VALU_CYCLES 4u
#define LOOM_AMDGPU_DELAY_ALU_TRANS_MAX 4u
#define LOOM_AMDGPU_DELAY_ALU_SALU_CYCLES_MAX 4u
#define LOOM_AMDGPU_DELAY_ALU_SALU_BASE 8u
#define LOOM_AMDGPU_DELAY_ALU_SELECTOR_CAPACITY 2u

enum {
  LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_INSTRUCTION_SLOT = 1,
  LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_DELAY_ALU_DEPENDENCY = 2,
  LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_VECTOR_ISSUE = 3,
  // Reasons through DELAY_ALU_DEPENDENCY use the dense per-register fixed-wait
  // hazard arrays. Matrix coexecution has its own bounded physical frontier.
  LOOM_AMDGPU_WAIT_STATE_TRACKED_REASON_COUNT =
      LOOM_AMDGPU_WAIT_STATE_REASON_DELAY_ALU_DEPENDENCY + 1,
};

typedef enum loom_amdgpu_delay_alu_type_e {
  LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER = 0,
  LOOM_AMDGPU_DELAY_ALU_TYPE_VALU = 1,
  LOOM_AMDGPU_DELAY_ALU_TYPE_TRANS = 2,
  LOOM_AMDGPU_DELAY_ALU_TYPE_SALU = 3,
} loom_amdgpu_delay_alu_type_t;

typedef enum loom_amdgpu_wait_state_vgpr_flag_bits_e {
  LOOM_AMDGPU_WAIT_STATE_VGPR_FLAG_VALID = 1u << 0,
} loom_amdgpu_wait_state_vgpr_flag_bits_t;
typedef uint8_t loom_amdgpu_wait_state_vgpr_flags_t;

typedef enum loom_amdgpu_wait_state_sgpr_flag_bits_e {
  LOOM_AMDGPU_WAIT_STATE_SGPR_FLAG_VALID = 1u << 0,
} loom_amdgpu_wait_state_sgpr_flag_bits_t;
typedef uint8_t loom_amdgpu_wait_state_sgpr_flags_t;

typedef enum loom_amdgpu_wait_state_reason_flag_bits_e {
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_MATRIX_RESULT_USE =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_RESULT_USE,
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_TO_MATRIX_USE =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE,
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_TRANS_RESULT_USE =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_TRANS_RESULT_USE,
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_SGPR_READ =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_VALU_SGPR_READ,
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_DPP_VGPR_READ =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_DPP_VGPR_READ,
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_READFIRSTLANE_VGPR_READ =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_READFIRSTLANE_VGPR_READ,
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_DST_SEL_FORWARDING_USE =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_DST_SEL_FORWARDING_USE,
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_DELAY_ALU_DEPENDENCY =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_DELAY_ALU_DEPENDENCY,
} loom_amdgpu_wait_state_reason_flag_bits_t;
typedef uint32_t loom_amdgpu_wait_state_reason_flags_t;

typedef enum loom_amdgpu_wait_state_matrix_result_use_e {
  LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_UNKNOWN = 0,
  LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_NON_MATRIX = 1,
  LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_MATRIX_SRCC = 2,
  LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_MATRIX_SRCC_EXACT = 3,
  LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_MATRIX_SRCC_OVERLAP = 4,
  LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_MATRIX_SRC_AB = 5,
} loom_amdgpu_wait_state_matrix_result_use_t;

enum {
  LOOM_AMDGPU_WAIT_STATE_MATRIX_FAMILY_USE_COUNT =
      LOOM_AMDGPU_MATRIX_FAMILY_SWMMAC + 1,
  LOOM_AMDGPU_WAIT_STATE_MATRIX_OPERAND_USE_CAPACITY = 4,
};

typedef struct loom_amdgpu_wait_state_matrix_family_use_row_t {
  // Non-zero when this matrix family reads VALU results as matrix inputs.
  uint8_t reads_valu_results;
  // Non-zero when this matrix family participates in result wait-state timing.
  uint8_t tracks_result_waits;
  // Matrix result use kind for each packet operand index.
  loom_amdgpu_wait_state_matrix_result_use_t
      operand_uses[LOOM_AMDGPU_WAIT_STATE_MATRIX_OPERAND_USE_CAPACITY];
} loom_amdgpu_wait_state_matrix_family_use_row_t;

static const loom_amdgpu_wait_state_matrix_family_use_row_t
    kMatrixFamilyUseRows[LOOM_AMDGPU_WAIT_STATE_MATRIX_FAMILY_USE_COUNT] = {
        [LOOM_AMDGPU_MATRIX_FAMILY_MFMA] =
            {
                .reads_valu_results = 1,
                .tracks_result_waits = 1,
                .operand_uses =
                    {
                        LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_MATRIX_SRC_AB,
                        LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_MATRIX_SRC_AB,
                        LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_MATRIX_SRCC,
                    },
            },
        [LOOM_AMDGPU_MATRIX_FAMILY_SMFMAC] =
            {
                .reads_valu_results = 1,
                .tracks_result_waits = 1,
                .operand_uses =
                    {
                        LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_MATRIX_SRCC,
                        LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_MATRIX_SRC_AB,
                        LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_MATRIX_SRC_AB,
                        LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_MATRIX_SRC_AB,
                    },
            },
};

typedef struct loom_amdgpu_wait_state_hazard_t {
  // Active-state flags for this physical VGPR.
  loom_amdgpu_wait_state_vgpr_flags_t flags;
  // Hazard reason associated with this outstanding write.
  loom_amdgpu_wait_state_reason_t reason;
  // Schedule node that produced the outstanding hazard.
  uint32_t producer_node;
  // First physical VGPR covered by the producer result.
  uint32_t producer_location_base;
  // Number of physical VGPRs covered by the producer result.
  uint32_t producer_location_count;
  // Instruction-position immediately after the producer packet.
  uint64_t producer_end_position;
  // Matrix execution pass count for matrix-result hazards, or zero.
  uint16_t matrix_pass_count;
  // Required wait cycles before the matching consumer reads the VGPR.
  uint16_t required_cycle_count;
} loom_amdgpu_wait_state_hazard_t;

typedef struct loom_amdgpu_delay_alu_info_t {
  // Builder epoch in which this producer state was recorded.
  uint32_t epoch;
  // Original modeled latency for the most recent VALU write.
  uint8_t valu_required_cycles;
  // Schedule node that produced the outstanding VALU write.
  uint32_t valu_producer_node;
  // VALU issue count immediately before the VALU producer.
  uint64_t valu_number_base;
  // Original modeled latency for the most recent TRANS write.
  uint8_t trans_required_cycles;
  // Schedule node that produced the outstanding TRANS write.
  uint32_t trans_producer_node;
  // TRANS issue count immediately before the TRANS producer.
  uint64_t trans_number_base;
  // VALU issue count immediately before the TRANS producer.
  uint64_t trans_valu_number_base;
  // Original modeled latency for the most recent SALU write.
  uint8_t salu_required_cycles;
  // Schedule node that produced the outstanding SALU write.
  uint32_t salu_producer_node;
  // Instruction position immediately before the SALU producer.
  uint64_t salu_producer_position;
} loom_amdgpu_delay_alu_info_t;

typedef struct loom_amdgpu_wait_state_vgpr_t {
  // Per-reason outstanding fixed-wait hazard state for this physical VGPR.
  loom_amdgpu_wait_state_hazard_t
      hazards[LOOM_AMDGPU_WAIT_STATE_TRACKED_REASON_COUNT];
  // Recent ALU producer state for S_DELAY_ALU dependency insertion.
  loom_amdgpu_delay_alu_info_t delay_alu;
} loom_amdgpu_wait_state_vgpr_t;

typedef struct loom_amdgpu_wait_state_sgpr_t {
  // Per-reason outstanding fixed-wait hazard state for this physical SGPR.
  loom_amdgpu_wait_state_hazard_t
      hazards[LOOM_AMDGPU_WAIT_STATE_TRACKED_REASON_COUNT];
  // Recent ALU producer state for S_DELAY_ALU dependency insertion.
  loom_amdgpu_delay_alu_info_t delay_alu;
} loom_amdgpu_wait_state_sgpr_t;

typedef struct loom_amdgpu_wait_state_match_t {
  // Reason responsible for the largest unsatisfied wait.
  loom_amdgpu_wait_state_reason_t reason;
  // Producer node responsible for the largest unsatisfied wait.
  uint32_t producer_node;
  // Required target progress before the current consumer.
  uint16_t required_cycle_count;
  // Target progress already supplied before the current consumer.
  uint16_t observed_cycle_count;
  // Additional cycles required before the current consumer.
  uint16_t cycle_count;
  // Packed S_DELAY_ALU SIMM16 operand for delay-ALU actions.
  uint16_t delay_alu_immediate;
  // Matrix result wait table profile, or UNKNOWN for non-matrix reasons.
  loom_amdgpu_matrix_wait_profile_t matrix_wait_profile;
  // Matrix result use table key, or UNKNOWN for non-matrix reasons.
  loom_amdgpu_matrix_wait_result_use_t matrix_result_use;
  // Matrix result pass count used for the wait table lookup.
  uint16_t matrix_pass_count;
} loom_amdgpu_wait_state_match_t;

typedef enum loom_amdgpu_wait_state_packet_flag_bits_e {
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DESCRIPTOR = 1u << 0,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_MATRIX = 1u << 1,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_MATRIX_READS_VALU = 1u << 2,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_TRANS_PRODUCER = 1u << 3,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_TRANS_FORWARDING_CONSUMER = 1u << 4,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_USES_VECTOR_ALU = 1u << 5,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_USES_VECTOR_MEMORY = 1u << 6,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DPP_CONSUMER = 1u << 7,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_READFIRSTLANE_CONSUMER = 1u << 8,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DST_SEL_FORWARDING_PRODUCER = 1u << 9,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DST_SEL_FORWARDING_CONSUMER = 1u << 10,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_VALU_SGPR_READ_CONSUMER = 1u << 11,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_VALU_SGPR_READ_PRODUCER = 1u << 12,
} loom_amdgpu_wait_state_packet_flag_bits_t;
typedef uint32_t loom_amdgpu_wait_state_packet_flags_t;

typedef struct loom_amdgpu_wait_state_packet_info_t {
  // Compact packet classification flags.
  loom_amdgpu_wait_state_packet_flags_t flags;
  // Matrix contract descriptor for target-native matrix packets.
  const loom_amdgpu_matrix_contract_descriptor_t* matrix_contract;
  // Structural packet classification for non-descriptor low packets.
  loom_amdgpu_structural_packet_info_t structural;
  // Number of target instructions represented by this packet.
  uint64_t instruction_count;
  // Delay-ALU producer class for this packet.
  loom_amdgpu_delay_alu_type_t delay_alu_type;
  // Modeled latency used when recording delay-ALU producers.
  uint16_t delay_alu_latency_cycles;
  // Matrix result pass count used by fixed matrix wait tables.
  uint16_t matrix_pass_count;
  // Matrix wait cycles needed before ordinary consumers.
  uint16_t matrix_wait_cycles;
} loom_amdgpu_wait_state_packet_info_t;

typedef enum loom_amdgpu_delay_alu_accumulator_flag_bits_e {
  LOOM_AMDGPU_DELAY_ALU_ACCUMULATOR_FLAG_UNENCODED_CANDIDATES = 1u << 0,
} loom_amdgpu_delay_alu_accumulator_flag_bits_t;
typedef uint8_t loom_amdgpu_delay_alu_accumulator_flags_t;

typedef struct loom_amdgpu_delay_alu_candidate_t {
  // Target-format S_DELAY_ALU INSTID selector for the producer.
  uint16_t dependency_code;
  // Schedule node that produced the dependency.
  uint32_t producer_node;
  // Required target progress before the current consumer.
  uint16_t required_cycle_count;
  // Target progress already supplied before the current consumer.
  uint16_t observed_cycle_count;
  // Additional cycles required before the current consumer.
  uint16_t residual_cycle_count;
} loom_amdgpu_delay_alu_candidate_t;

typedef struct loom_amdgpu_delay_alu_accumulator_t {
  // Best candidates that fit in one S_DELAY_ALU immediate.
  loom_amdgpu_delay_alu_candidate_t
      candidates[LOOM_AMDGPU_DELAY_ALU_SELECTOR_CAPACITY];
  // Number of populated candidates.
  uint8_t candidate_count;
  // Accumulation flags.
  loom_amdgpu_delay_alu_accumulator_flags_t flags;
  // Strongest dependency if the candidate set must fall back to S_NOP.
  loom_amdgpu_wait_state_match_t fallback_match;
} loom_amdgpu_delay_alu_accumulator_t;

typedef struct loom_amdgpu_wait_state_builder_t {
  // Schedule table being analyzed.
  const loom_low_schedule_table_t* schedule;
  // Allocation table being analyzed.
  const loom_low_allocation_table_t* allocation;
  // Final VOPD memberships indexed by scheduled packet.
  const loom_amdgpu_vopd_packet_t* vopd_packets;
  // Final VOPD pairs referenced by |vopd_packets|.
  const loom_amdgpu_vopd_pair_t* vopd_pairs;
  // Descriptor set selected by the scheduled target.
  const loom_low_descriptor_set_t* descriptor_set;
  // Arena that owns output tables retained by the completed plan.
  iree_arena_allocator_t* arena;
  // Arena that owns builder state discarded after plan construction.
  iree_arena_allocator_t* transient_arena;
  // Processor properties selected by the low target, or NULL if unavailable.
  const loom_amdgpu_processor_properties_t* processor_properties;
  // Cached processor scheduling and fixed-wait hazard feature bits.
  loom_amdgpu_processor_scheduling_bits_t processor_scheduling;
  // True when the selected processor and descriptor set can emit S_DELAY_ALU.
  bool has_delay_alu;
  // Transient matrix/vector coexecution frontier, or NULL when not selected.
  loom_amdgpu_matrix_coexecution_t* matrix_coexecution;
  // Per-physical-VGPR outstanding fixed-wait hazard state.
  loom_amdgpu_wait_state_vgpr_t* vgprs;
  // Number of entries in |vgprs|.
  iree_host_size_t vgpr_count;
  // Per-physical-SGPR outstanding fixed-wait hazard state.
  loom_amdgpu_wait_state_sgpr_t* sgprs;
  // Number of entries in |sgprs|.
  iree_host_size_t sgpr_count;
  // Recent ALU producer state for the physical SCC condition register.
  loom_amdgpu_delay_alu_info_t scc_delay_alu;
  // Output wait-state rows.
  loom_amdgpu_wait_state_t* states;
  // Number of populated output wait-state rows.
  iree_host_size_t state_count;
  // Number of packets that emit native instruction-slot progress.
  iree_host_size_t progress_event_count;
  // Allocated output wait-state capacity.
  iree_host_size_t state_capacity;
  // Native instruction count indexed by scheduled packet.
  uint32_t* packet_instruction_counts;
  // Target progress facts for scheduled native instruction slots.
  loom_low_packet_progress_table_t progress;
  // Common residual hazard records for emitted wait states.
  loom_low_packet_hazard_plan_t hazard_plan;
  // Current ordinary instruction position in the active block.
  uint64_t current_position;
  // Active epoch for delay-ALU producer state.
  uint32_t delay_alu_epoch;
  // Number of delay-tracked VALU packets issued in the active epoch.
  uint64_t delay_alu_valu_count;
  // Number of delay-tracked TRANS packets issued in the active epoch.
  uint64_t delay_alu_trans_count;
} loom_amdgpu_wait_state_builder_t;

static const iree_string_view_t kAmdgpuWaitStateReasonNames[] = {
    [LOOM_AMDGPU_WAIT_STATE_REASON_UNKNOWN] = IREE_SVL("unknown"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_RESULT_USE] =
        IREE_SVL("matrix_result_use"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE] =
        IREE_SVL("valu_to_matrix_use"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_TRANS_RESULT_USE] =
        IREE_SVL("trans_result_use"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_VALU_SGPR_READ] = IREE_SVL("valu_sgpr_read"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_DPP_VGPR_READ] = IREE_SVL("dpp_vgpr_read"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_READFIRSTLANE_VGPR_READ] =
        IREE_SVL("readfirstlane_vgpr_read"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_DST_SEL_FORWARDING_USE] =
        IREE_SVL("dst_sel_forwarding_use"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_DELAY_ALU_DEPENDENCY] =
        IREE_SVL("delay_alu_dependency"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_COEXECUTION_MATRIX_USE] =
        IREE_SVL("matrix_coexecution_matrix_use"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_COEXECUTION_VALU_USE] =
        IREE_SVL("matrix_coexecution_valu_use"),
};

static const iree_string_view_t kAmdgpuWaitStateActionNames[] = {
    [LOOM_AMDGPU_WAIT_STATE_ACTION_UNKNOWN] = IREE_SVL("unknown"),
    [LOOM_AMDGPU_WAIT_STATE_ACTION_S_NOP] = IREE_SVL("amdgpu.s_nop"),
    [LOOM_AMDGPU_WAIT_STATE_ACTION_S_DELAY_ALU] =
        IREE_SVL("amdgpu.s_delay_alu"),
    [LOOM_AMDGPU_WAIT_STATE_ACTION_V_NOP] = IREE_SVL("amdgpu.v_nop"),
};

static iree_string_view_t loom_amdgpu_wait_state_progress_class_name(
    uint32_t progress_class_id) {
  switch (progress_class_id) {
    case LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_INSTRUCTION_SLOT:
      return IREE_SV("amdgpu.instruction_slot");
    case LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_DELAY_ALU_DEPENDENCY:
      return IREE_SV("amdgpu.delay_alu_dependency");
    case LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_VECTOR_ISSUE:
      return IREE_SV("amdgpu.vector_issue");
    default:
      return IREE_SV("unknown");
  }
}

static uint32_t loom_amdgpu_wait_state_progress_class_id(
    const loom_amdgpu_wait_state_t* wait_state) {
  if (wait_state->action == LOOM_AMDGPU_WAIT_STATE_ACTION_S_DELAY_ALU) {
    return LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_DELAY_ALU_DEPENDENCY;
  }
  if (wait_state->action == LOOM_AMDGPU_WAIT_STATE_ACTION_V_NOP) {
    return LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_VECTOR_ISSUE;
  }
  return LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_INSTRUCTION_SLOT;
}

static const loom_amdgpu_wait_state_reason_flags_t
    kAmdgpuWaitStateReasonFlags[] = {
        [LOOM_AMDGPU_WAIT_STATE_REASON_UNKNOWN] = 0,
        [LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_RESULT_USE] =
            LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_MATRIX_RESULT_USE,
        [LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE] =
            LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_TO_MATRIX_USE,
        [LOOM_AMDGPU_WAIT_STATE_REASON_TRANS_RESULT_USE] =
            LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_TRANS_RESULT_USE,
        [LOOM_AMDGPU_WAIT_STATE_REASON_VALU_SGPR_READ] =
            LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_SGPR_READ,
        [LOOM_AMDGPU_WAIT_STATE_REASON_DPP_VGPR_READ] =
            LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_DPP_VGPR_READ,
        [LOOM_AMDGPU_WAIT_STATE_REASON_READFIRSTLANE_VGPR_READ] =
            LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_READFIRSTLANE_VGPR_READ,
        [LOOM_AMDGPU_WAIT_STATE_REASON_DST_SEL_FORWARDING_USE] =
            LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_DST_SEL_FORWARDING_USE,
        [LOOM_AMDGPU_WAIT_STATE_REASON_DELAY_ALU_DEPENDENCY] =
            LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_DELAY_ALU_DEPENDENCY,
};

iree_string_view_t loom_amdgpu_wait_state_reason_name(
    loom_amdgpu_wait_state_reason_t reason) {
  if ((iree_host_size_t)reason >= IREE_ARRAYSIZE(kAmdgpuWaitStateReasonNames)) {
    return kAmdgpuWaitStateReasonNames[LOOM_AMDGPU_WAIT_STATE_REASON_UNKNOWN];
  }
  return kAmdgpuWaitStateReasonNames[reason];
}

iree_string_view_t loom_amdgpu_wait_state_action_name(
    loom_amdgpu_wait_state_action_t action) {
  if ((iree_host_size_t)action >= IREE_ARRAYSIZE(kAmdgpuWaitStateActionNames)) {
    return kAmdgpuWaitStateActionNames[LOOM_AMDGPU_WAIT_STATE_ACTION_UNKNOWN];
  }
  return kAmdgpuWaitStateActionNames[action];
}

static loom_amdgpu_wait_state_reason_flags_t loom_amdgpu_wait_state_reason_flag(
    loom_amdgpu_wait_state_reason_t reason) {
  if ((iree_host_size_t)reason >= IREE_ARRAYSIZE(kAmdgpuWaitStateReasonFlags)) {
    return 0;
  }
  return kAmdgpuWaitStateReasonFlags[reason];
}

static bool loom_amdgpu_wait_state_reason_is_valid(
    loom_amdgpu_wait_state_reason_t reason) {
  return reason > LOOM_AMDGPU_WAIT_STATE_REASON_UNKNOWN &&
         reason < LOOM_AMDGPU_WAIT_STATE_REASON_COUNT_;
}

static bool loom_amdgpu_wait_state_reason_is_tracked(
    loom_amdgpu_wait_state_reason_t reason) {
  return reason > LOOM_AMDGPU_WAIT_STATE_REASON_UNKNOWN &&
         reason <= LOOM_AMDGPU_WAIT_STATE_REASON_DELAY_ALU_DEPENDENCY;
}

static bool loom_amdgpu_wait_state_assignment_is_physical_vgpr(
    const loom_low_allocation_assignment_t* assignment) {
  if (assignment == NULL) {
    return false;
  }
  return loom_low_allocation_assignment_is_physical_register_class(
      assignment, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
}

static bool loom_amdgpu_wait_state_assignment_is_physical_sgpr(
    const loom_low_allocation_assignment_t* assignment) {
  if (assignment == NULL) {
    return false;
  }
  return loom_low_allocation_assignment_is_physical_register_class(
      assignment, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
}

static bool loom_amdgpu_wait_state_assignment_is_physical_scc(
    const loom_low_allocation_assignment_t* assignment) {
  if (assignment == NULL) {
    return false;
  }
  return loom_low_allocation_assignment_is_physical_register_class(
      assignment, LOOM_AMDGPU_REG_CLASS_ID_SCC);
}

static const loom_low_allocation_assignment_t*
loom_amdgpu_wait_state_assignment(const loom_low_allocation_table_t* allocation,
                                  loom_value_id_t value_id) {
  return loom_low_allocation_try_map_active_value_assignment(allocation,
                                                             value_id, NULL);
}

static iree_status_t loom_amdgpu_wait_state_allocate(
    loom_amdgpu_wait_state_builder_t* builder) {
  const loom_low_allocation_table_t* allocation = builder->allocation;
  const iree_host_size_t packet_count =
      loom_low_packet_count(builder->schedule);
  if (packet_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(builder->transient_arena, packet_count,
                                  sizeof(*builder->packet_instruction_counts),
                                  (void**)&builder->packet_instruction_counts));
  }
  const iree_host_size_t vgpr_count =
      allocation->physical_extents
          .ends_by_reg_class[LOOM_AMDGPU_REG_CLASS_ID_VGPR];
  const iree_host_size_t sgpr_count =
      allocation->physical_extents
          .ends_by_reg_class[LOOM_AMDGPU_REG_CLASS_ID_SGPR];
  if (vgpr_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->transient_arena, vgpr_count, sizeof(*builder->vgprs),
        (void**)&builder->vgprs));
    builder->vgpr_count = vgpr_count;
  }
  if (sgpr_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->transient_arena, sgpr_count, sizeof(*builder->sgprs),
        (void**)&builder->sgprs));
    builder->sgpr_count = sgpr_count;
  }
  const iree_host_size_t states_per_packet =
      builder->matrix_coexecution != NULL ? 3 : 2;
  if (!iree_host_size_checked_mul(builder->schedule->scheduled_node_count,
                                  states_per_packet,
                                  &builder->state_capacity)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU wait-state plan capacity overflows");
  }
  if (builder->state_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, builder->state_capacity, sizeof(*builder->states),
        (void**)&builder->states));
  }
  return iree_ok_status();
}

static const loom_amdgpu_matrix_contract_descriptor_t*
loom_amdgpu_wait_state_contract_for_descriptor(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_descriptor_t* descriptor) {
  const loom_amdgpu_descriptor_ref_t descriptor_ref =
      loom_amdgpu_descriptor_ref_for_descriptor(builder->descriptor_set,
                                                descriptor);
  return loom_amdgpu_matrix_contract_wait_state_descriptor_for_low_descriptor_ref(
      descriptor_ref);
}

static bool loom_amdgpu_wait_state_matrix_tracks_result_waits(
    const loom_amdgpu_matrix_contract_descriptor_t* contract) {
  if ((uint32_t)contract->family >=
      LOOM_AMDGPU_WAIT_STATE_MATRIX_FAMILY_USE_COUNT) {
    return false;
  }
  return kMatrixFamilyUseRows[contract->family].tracks_result_waits != 0;
}

static bool loom_amdgpu_wait_state_matrix_result_pass_count(
    const loom_amdgpu_matrix_contract_descriptor_t* contract,
    uint16_t* out_pass_count) {
  *out_pass_count = 0;
  if (!loom_amdgpu_wait_state_matrix_tracks_result_waits(contract)) {
    return false;
  }
  switch (contract->result_payload.register_count) {
    case 2:
    case 4:
    case 8:
    case 16:
    case 32:
      *out_pass_count = contract->result_payload.register_count;
      return true;
    default:
      IREE_ASSERT_UNREACHABLE(
          "generated AMDGPU matrix contract has unsupported wait-state result "
          "payload register count");
      return false;
  }
}

static bool loom_amdgpu_wait_state_matrix_result_table_use(
    loom_amdgpu_wait_state_matrix_result_use_t use,
    loom_amdgpu_matrix_wait_result_use_t* out_table_use) {
  *out_table_use = LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_UNKNOWN;
  switch (use) {
    case LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_NON_MATRIX:
      *out_table_use = LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX;
      return true;
    case LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_MATRIX_SRCC_EXACT:
      *out_table_use = LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_EXACT;
      return true;
    case LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_MATRIX_SRCC_OVERLAP:
      *out_table_use = LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRCC_OVERLAP;
      return true;
    case LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_MATRIX_SRC_AB:
      *out_table_use = LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_MATRIX_SRC_AB;
      return true;
    case LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_MATRIX_SRCC:
    case LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_UNKNOWN:
    default:
      return false;
  }
}

static loom_amdgpu_matrix_wait_profile_t
loom_amdgpu_wait_state_matrix_wait_profile(
    const loom_amdgpu_wait_state_builder_t* builder) {
  loom_amdgpu_matrix_wait_profile_t wait_profile =
      LOOM_AMDGPU_MATRIX_WAIT_PROFILE_MFMA_PRE_GFX950;
  if (builder->processor_properties != NULL &&
      !loom_amdgpu_matrix_wait_profile_from_feature_profile(
          builder->processor_properties->features.matrix, &wait_profile)) {
    return LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN;
  }
  return wait_profile;
}

static bool loom_amdgpu_wait_state_matrix_result_wait_cycles(
    const loom_amdgpu_wait_state_builder_t* builder, uint16_t pass_count,
    loom_amdgpu_wait_state_matrix_result_use_t use, uint16_t* out_cycle_count) {
  *out_cycle_count = 0;
  loom_amdgpu_matrix_wait_result_use_t table_use =
      LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_UNKNOWN;
  if (!loom_amdgpu_wait_state_matrix_result_table_use(use, &table_use)) {
    return false;
  }
  const loom_amdgpu_matrix_wait_profile_t wait_profile =
      loom_amdgpu_wait_state_matrix_wait_profile(builder);
  return loom_amdgpu_matrix_wait_result_cycle_count(wait_profile, pass_count,
                                                    table_use, out_cycle_count);
}

static bool loom_amdgpu_wait_state_matrix_reads_valu_results(
    const loom_amdgpu_matrix_contract_descriptor_t* contract) {
  if ((uint32_t)contract->family >=
      LOOM_AMDGPU_WAIT_STATE_MATRIX_FAMILY_USE_COUNT) {
    return false;
  }
  return kMatrixFamilyUseRows[contract->family].reads_valu_results != 0;
}

static loom_amdgpu_wait_state_matrix_result_use_t
loom_amdgpu_wait_state_matrix_operand_result_use(
    const loom_amdgpu_matrix_contract_descriptor_t* contract,
    uint16_t packet_operand_index) {
  if ((uint32_t)contract->family >=
          LOOM_AMDGPU_WAIT_STATE_MATRIX_FAMILY_USE_COUNT ||
      packet_operand_index >=
          LOOM_AMDGPU_WAIT_STATE_MATRIX_OPERAND_USE_CAPACITY) {
    return LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_UNKNOWN;
  }
  return kMatrixFamilyUseRows[contract->family]
      .operand_uses[packet_operand_index];
}

static bool loom_amdgpu_wait_state_has_scheduling(
    const loom_amdgpu_wait_state_builder_t* builder,
    loom_amdgpu_processor_scheduling_bits_t bits) {
  return iree_any_bit_set(builder->processor_scheduling, bits);
}

static bool loom_amdgpu_wait_state_target_has_delay_alu(
    const loom_amdgpu_wait_state_builder_t* builder) {
  if (builder->processor_properties == NULL ||
      builder->descriptor_set == NULL ||
      !loom_amdgpu_wait_state_has_scheduling(
          builder, LOOM_AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU)) {
    return false;
  }
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(
          builder->descriptor_set->descriptor_set_ordinal);
  return descriptor_set_info != NULL &&
         descriptor_set_info->sopp.delay_alu != 0;
}

static loom_amdgpu_delay_alu_type_t loom_amdgpu_wait_state_delay_alu_type(
    const loom_amdgpu_wait_state_builder_t* builder,
    loom_amdgpu_descriptor_traits_t descriptor_traits) {
  if (!builder->has_delay_alu) {
    return LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER;
  }
  if (iree_any_bit_set(descriptor_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_TRANSCENDENTAL)) {
    if (loom_amdgpu_wait_state_has_scheduling(
            builder, LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR)) {
      return LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER;
    }
    return LOOM_AMDGPU_DELAY_ALU_TYPE_TRANS;
  }
  if (iree_any_bit_set(descriptor_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU |
                           LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX)) {
    return LOOM_AMDGPU_DELAY_ALU_TYPE_VALU;
  }
  if (iree_any_bit_set(descriptor_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_ALU)) {
    return LOOM_AMDGPU_DELAY_ALU_TYPE_SALU;
  }
  return LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER;
}

static loom_amdgpu_delay_alu_type_t
loom_amdgpu_wait_state_structural_delay_alu_type(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_amdgpu_structural_packet_info_t* info) {
  if (!builder->has_delay_alu) {
    return LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER;
  }
  if (info->vector_alu_instruction_count != 0) {
    return LOOM_AMDGPU_DELAY_ALU_TYPE_VALU;
  }
  return info->scalar_alu_instruction_count != 0 ||
                 iree_any_bit_set(info->flags,
                                  LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_READS_SCC)
             ? LOOM_AMDGPU_DELAY_ALU_TYPE_SALU
             : LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER;
}

static uint16_t loom_amdgpu_wait_state_descriptor_latency_cycles(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_descriptor_t* descriptor) {
  const loom_low_descriptor_set_t* descriptor_set = builder->descriptor_set;
  if (descriptor_set == NULL || descriptor == NULL ||
      descriptor->schedule_class_id >= descriptor_set->schedule_class_count) {
    return 0;
  }
  const loom_low_schedule_class_t* schedule_class =
      &descriptor_set->schedule_classes[descriptor->schedule_class_id];
  return schedule_class->latency_cycles;
}

static uint16_t loom_amdgpu_wait_state_delay_alu_latency_cycles(
    loom_amdgpu_delay_alu_type_t type, uint16_t schedule_latency_cycles) {
  switch (type) {
    case LOOM_AMDGPU_DELAY_ALU_TYPE_VALU:
      return schedule_latency_cycles > LOOM_AMDGPU_DELAY_ALU_VALU_CYCLES
                 ? schedule_latency_cycles
                 : LOOM_AMDGPU_DELAY_ALU_VALU_CYCLES;
    case LOOM_AMDGPU_DELAY_ALU_TYPE_TRANS:
    case LOOM_AMDGPU_DELAY_ALU_TYPE_SALU:
      return schedule_latency_cycles;
    case LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER:
    default:
      return 0;
  }
}

static const loom_named_attr_t* loom_amdgpu_wait_state_find_packet_attr(
    loom_named_attr_slice_t attrs, loom_string_id_t name_id) {
  if (name_id == LOOM_STRING_ID_INVALID) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < attrs.count; ++i) {
    const loom_named_attr_t* attr = &attrs.entries[i];
    if (attr->name_id == name_id) {
      return attr;
    }
  }
  return NULL;
}

static iree_status_t loom_amdgpu_wait_state_read_dst_sel_immediate(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet, bool* out_has_value,
    int64_t* out_value) {
  *out_has_value = false;
  *out_value = 0;
  const loom_low_descriptor_set_t* descriptor_set = builder->descriptor_set;
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  if (descriptor == NULL) {
    return iree_ok_status();
  }
  const loom_amdgpu_descriptor_immediate_slots_t immediate_slots =
      loom_amdgpu_descriptor_immediate_slots(descriptor_set, descriptor);
  if (immediate_slots.sdwa_dst_sel == LOOM_LOW_ID_NONE) {
    return iree_ok_status();
  }
  const uint16_t descriptor_immediate_index = immediate_slots.sdwa_dst_sel;
  IREE_ASSERT_LT(descriptor_immediate_index, descriptor->immediate_count);
  const uint32_t immediate_index =
      descriptor->immediate_start + descriptor_immediate_index;
  IREE_ASSERT_LT(immediate_index, descriptor_set->immediate_count);
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[immediate_index];
  const iree_string_view_t field_name = loom_low_descriptor_set_string(
      descriptor_set, immediate->field_name_string_offset);
  const loom_string_id_t field_name_id =
      loom_module_lookup_string(builder->schedule->module, field_name);
  const loom_named_attr_t* attr = loom_amdgpu_wait_state_find_packet_attr(
      loom_low_packet_attrs(packet), field_name_id);
  if (attr == NULL) {
    if (iree_all_bits_set(immediate->flags,
                          LOOM_LOW_IMMEDIATE_FLAG_DEFAULT_VALUE)) {
      *out_has_value = true;
      *out_value = immediate->default_value;
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU wait-state SDWA descriptor requires "
                            "dst_sel immediate");
  }
  if (attr->value.kind != LOOM_ATTR_I64) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU wait-state dst_sel immediate must be "
                            "i64");
  }
  *out_has_value = true;
  *out_value = loom_attr_as_i64(attr->value);
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_wait_state_packet_has_destination_selection_forwarding_hazard(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_descriptor_traits_t descriptor_traits, bool* out_has_hazard) {
  *out_has_hazard = false;
  if (iree_any_bit_set(
          descriptor_traits,
          LOOM_AMDGPU_DESCRIPTOR_TRAIT_DESTINATION_SELECTION_FORWARDING)) {
    *out_has_hazard = true;
    return iree_ok_status();
  }
  if (!iree_any_bit_set(descriptor_traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_SDWA)) {
    return iree_ok_status();
  }
  bool has_dst_sel = false;
  int64_t dst_sel = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_read_dst_sel_immediate(
      builder, packet, &has_dst_sel, &dst_sel));
  if (!has_dst_sel) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AMDGPU wait-state SDWA descriptor has no dst_sel "
                            "immediate");
  }
  if (dst_sel < 0 || dst_sel > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU wait-state dst_sel immediate %" PRId64
                            " is out of range",
                            dst_sel);
  }
  *out_has_hazard =
      loom_amdgpu_sdwa_dst_selector_writes_subdword((uint32_t)dst_sel);
  return iree_ok_status();
}

static void loom_amdgpu_wait_state_clear_assignment(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment) {
  if (assignment == NULL) {
    return;
  }
  uint64_t end = 0;
  if (!loom_low_allocation_assignment_location_exclusive_end(assignment,
                                                             &end)) {
    return;
  }
  if (loom_amdgpu_wait_state_assignment_is_physical_vgpr(assignment)) {
    if (end > builder->vgpr_count) {
      return;
    }
    for (uint32_t i = 0; i < assignment->location_count; ++i) {
      builder->vgprs[assignment->location_base + i] =
          (loom_amdgpu_wait_state_vgpr_t){0};
    }
    return;
  }
  if (loom_amdgpu_wait_state_assignment_is_physical_sgpr(assignment)) {
    if (end > builder->sgpr_count) {
      return;
    }
    for (uint32_t i = 0; i < assignment->location_count; ++i) {
      builder->sgprs[assignment->location_base + i] =
          (loom_amdgpu_wait_state_sgpr_t){0};
    }
  }
}

static void loom_amdgpu_wait_state_record_assignment(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment,
    loom_amdgpu_wait_state_reason_t reason, uint32_t producer_node,
    uint16_t cycle_count, uint16_t matrix_pass_count,
    uint64_t producer_end_position) {
  if (!loom_amdgpu_wait_state_reason_is_tracked(reason)) {
    return;
  }
  if (!loom_amdgpu_wait_state_assignment_is_physical_vgpr(assignment)) {
    return;
  }
  uint64_t end = 0;
  if (!loom_low_allocation_assignment_location_exclusive_end(assignment,
                                                             &end)) {
    return;
  }
  if (end > builder->vgpr_count) {
    return;
  }
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    loom_amdgpu_wait_state_hazard_t* hazard =
        &builder->vgprs[assignment->location_base + i].hazards[reason];
    *hazard = (loom_amdgpu_wait_state_hazard_t){
        .flags = LOOM_AMDGPU_WAIT_STATE_VGPR_FLAG_VALID,
        .reason = reason,
        .producer_node = producer_node,
        .producer_location_base = assignment->location_base,
        .producer_location_count = assignment->location_count,
        .producer_end_position = producer_end_position,
        .matrix_pass_count = matrix_pass_count,
        .required_cycle_count = cycle_count,
    };
  }
}

static void loom_amdgpu_wait_state_record_sgpr_assignment(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment,
    loom_amdgpu_wait_state_reason_t reason, uint32_t producer_node,
    uint16_t cycle_count, uint64_t producer_end_position) {
  if (!loom_amdgpu_wait_state_reason_is_tracked(reason)) {
    return;
  }
  if (!loom_amdgpu_wait_state_assignment_is_physical_sgpr(assignment)) {
    return;
  }
  uint64_t end = 0;
  if (!loom_low_allocation_assignment_location_exclusive_end(assignment,
                                                             &end)) {
    return;
  }
  if (end > builder->sgpr_count) {
    return;
  }
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    loom_amdgpu_wait_state_hazard_t* hazard =
        &builder->sgprs[assignment->location_base + i].hazards[reason];
    *hazard = (loom_amdgpu_wait_state_hazard_t){
        .flags = LOOM_AMDGPU_WAIT_STATE_SGPR_FLAG_VALID,
        .reason = reason,
        .producer_node = producer_node,
        .producer_location_base = assignment->location_base,
        .producer_location_count = assignment->location_count,
        .producer_end_position = producer_end_position,
        .required_cycle_count = cycle_count,
    };
  }
}

static void loom_amdgpu_wait_state_match_active_hazard(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_amdgpu_wait_state_hazard_t* hazard,
    uint16_t required_cycle_count,
    loom_amdgpu_matrix_wait_result_use_t matrix_result_use,
    loom_amdgpu_wait_state_match_t* match) {
  const uint64_t elapsed =
      builder->current_position >= hazard->producer_end_position
          ? builder->current_position - hazard->producer_end_position
          : 0;
  if (elapsed >= required_cycle_count) {
    return;
  }
  const uint16_t remaining = (uint16_t)(required_cycle_count - elapsed);
  if (remaining > match->cycle_count) {
    loom_amdgpu_matrix_wait_profile_t matrix_wait_profile =
        LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN;
    uint16_t matrix_pass_count = 0;
    if (hazard->reason == LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_RESULT_USE &&
        matrix_result_use != LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_UNKNOWN) {
      matrix_wait_profile = loom_amdgpu_wait_state_matrix_wait_profile(builder);
      matrix_pass_count = hazard->matrix_pass_count;
    }
    *match = (loom_amdgpu_wait_state_match_t){
        .reason = hazard->reason,
        .producer_node = hazard->producer_node,
        .required_cycle_count = required_cycle_count,
        .observed_cycle_count = (uint16_t)elapsed,
        .cycle_count = remaining,
        .matrix_wait_profile = matrix_wait_profile,
        .matrix_result_use = matrix_result_use,
        .matrix_pass_count = matrix_pass_count,
    };
  }
}

static void loom_amdgpu_wait_state_match_assignment(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment,
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons,
    loom_amdgpu_wait_state_match_t* match) {
  if (!loom_amdgpu_wait_state_assignment_is_physical_vgpr(assignment)) {
    return;
  }
  uint64_t end = 0;
  if (!loom_low_allocation_assignment_location_exclusive_end(assignment,
                                                             &end)) {
    return;
  }
  if (end > builder->vgpr_count) {
    return;
  }
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    const loom_amdgpu_wait_state_vgpr_t* vgpr_state =
        &builder->vgprs[assignment->location_base + i];
    for (uint32_t reason = LOOM_AMDGPU_WAIT_STATE_REASON_UNKNOWN + 1;
         reason < LOOM_AMDGPU_WAIT_STATE_TRACKED_REASON_COUNT; ++reason) {
      const loom_amdgpu_wait_state_hazard_t* hazard =
          &vgpr_state->hazards[reason];
      if (!iree_any_bit_set(hazard->flags,
                            LOOM_AMDGPU_WAIT_STATE_VGPR_FLAG_VALID)) {
        continue;
      }
      if (!iree_any_bit_set(allowed_reasons, loom_amdgpu_wait_state_reason_flag(
                                                 hazard->reason))) {
        continue;
      }
      const loom_amdgpu_matrix_wait_result_use_t matrix_result_use =
          hazard->reason == LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_RESULT_USE
              ? LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_NON_MATRIX
              : LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_UNKNOWN;
      loom_amdgpu_wait_state_match_active_hazard(builder, hazard,
                                                 hazard->required_cycle_count,
                                                 matrix_result_use, match);
    }
  }
}

static void loom_amdgpu_wait_state_match_sgpr_assignment(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment,
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons,
    loom_amdgpu_wait_state_match_t* match) {
  if (!loom_amdgpu_wait_state_assignment_is_physical_sgpr(assignment)) {
    return;
  }
  uint64_t end = 0;
  if (!loom_low_allocation_assignment_location_exclusive_end(assignment,
                                                             &end)) {
    return;
  }
  if (end > builder->sgpr_count) {
    return;
  }
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    const loom_amdgpu_wait_state_sgpr_t* sgpr_state =
        &builder->sgprs[assignment->location_base + i];
    for (uint32_t reason = LOOM_AMDGPU_WAIT_STATE_REASON_UNKNOWN + 1;
         reason < LOOM_AMDGPU_WAIT_STATE_TRACKED_REASON_COUNT; ++reason) {
      const loom_amdgpu_wait_state_hazard_t* hazard =
          &sgpr_state->hazards[reason];
      if (!iree_any_bit_set(hazard->flags,
                            LOOM_AMDGPU_WAIT_STATE_SGPR_FLAG_VALID)) {
        continue;
      }
      if (!iree_any_bit_set(allowed_reasons, loom_amdgpu_wait_state_reason_flag(
                                                 hazard->reason))) {
        continue;
      }
      loom_amdgpu_wait_state_match_active_hazard(
          builder, hazard, hazard->required_cycle_count,
          LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_UNKNOWN, match);
    }
  }
}

static void loom_amdgpu_wait_state_match_value(
    const loom_amdgpu_wait_state_builder_t* builder, loom_value_id_t value_id,
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons,
    loom_amdgpu_wait_state_match_t* match) {
  const loom_low_allocation_assignment_t* assignment =
      loom_amdgpu_wait_state_assignment(builder->allocation, value_id);
  loom_amdgpu_wait_state_match_assignment(builder, assignment, allowed_reasons,
                                          match);
}

static bool loom_amdgpu_wait_state_matrix_result_hazard_is_exact(
    const loom_amdgpu_wait_state_hazard_t* hazard,
    const loom_low_allocation_assignment_t* assignment) {
  return hazard->producer_location_base == assignment->location_base &&
         hazard->producer_location_count == assignment->location_count;
}

static void loom_amdgpu_wait_state_match_matrix_result_assignment(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment,
    loom_amdgpu_wait_state_matrix_result_use_t use,
    loom_amdgpu_wait_state_match_t* match) {
  if (!loom_amdgpu_wait_state_assignment_is_physical_vgpr(assignment)) {
    return;
  }
  uint64_t end = 0;
  if (!loom_low_allocation_assignment_location_exclusive_end(assignment,
                                                             &end)) {
    return;
  }
  if (end > builder->vgpr_count) {
    return;
  }
  for (uint32_t i = 0; i < assignment->location_count; ++i) {
    const loom_amdgpu_wait_state_hazard_t* hazard =
        &builder->vgprs[assignment->location_base + i]
             .hazards[LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_RESULT_USE];
    if (!iree_any_bit_set(hazard->flags,
                          LOOM_AMDGPU_WAIT_STATE_VGPR_FLAG_VALID)) {
      continue;
    }
    loom_amdgpu_wait_state_matrix_result_use_t actual_use = use;
    if (use == LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_MATRIX_SRCC) {
      actual_use =
          loom_amdgpu_wait_state_matrix_result_hazard_is_exact(hazard,
                                                               assignment)
              ? LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_MATRIX_SRCC_EXACT
              : LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_MATRIX_SRCC_OVERLAP;
    }
    uint16_t required_cycle_count = 0;
    if (!loom_amdgpu_wait_state_matrix_result_wait_cycles(
            builder, hazard->matrix_pass_count, actual_use,
            &required_cycle_count)) {
      continue;
    }
    loom_amdgpu_matrix_wait_result_use_t table_use =
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_UNKNOWN;
    if (!loom_amdgpu_wait_state_matrix_result_table_use(actual_use,
                                                        &table_use)) {
      continue;
    }
    loom_amdgpu_wait_state_match_active_hazard(
        builder, hazard, required_cycle_count, table_use, match);
  }
}

static void loom_amdgpu_wait_state_match_matrix_result_value(
    const loom_amdgpu_wait_state_builder_t* builder, loom_value_id_t value_id,
    loom_amdgpu_wait_state_matrix_result_use_t use,
    loom_amdgpu_wait_state_match_t* match) {
  const loom_low_allocation_assignment_t* assignment =
      loom_amdgpu_wait_state_assignment(builder->allocation, value_id);
  loom_amdgpu_wait_state_match_matrix_result_assignment(builder, assignment,
                                                        use, match);
}

static void loom_amdgpu_wait_state_match_sgpr_value(
    const loom_amdgpu_wait_state_builder_t* builder, loom_value_id_t value_id,
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons,
    loom_amdgpu_wait_state_match_t* match) {
  const loom_low_allocation_assignment_t* assignment =
      loom_amdgpu_wait_state_assignment(builder->allocation, value_id);
  loom_amdgpu_wait_state_match_sgpr_assignment(builder, assignment,
                                               allowed_reasons, match);
}

static uint8_t loom_amdgpu_wait_state_delay_alu_clamp_cycles(
    uint16_t cycle_count) {
  return cycle_count > UINT8_MAX ? UINT8_MAX : (uint8_t)cycle_count;
}

static loom_amdgpu_delay_alu_info_t loom_amdgpu_wait_state_delay_alu_make_info(
    const loom_amdgpu_wait_state_builder_t* builder,
    loom_amdgpu_delay_alu_type_t type, uint16_t latency_cycles,
    uint32_t producer_node) {
  const uint8_t cycles =
      loom_amdgpu_wait_state_delay_alu_clamp_cycles(latency_cycles);
  loom_amdgpu_delay_alu_info_t info = {
      .epoch = builder->delay_alu_epoch,
  };
  switch (type) {
    case LOOM_AMDGPU_DELAY_ALU_TYPE_VALU:
      info.valu_required_cycles = cycles;
      info.valu_producer_node = producer_node;
      info.valu_number_base = builder->delay_alu_valu_count;
      break;
    case LOOM_AMDGPU_DELAY_ALU_TYPE_TRANS:
      info.trans_required_cycles = cycles;
      info.trans_producer_node = producer_node;
      info.trans_number_base = builder->delay_alu_trans_count;
      info.trans_valu_number_base = builder->delay_alu_valu_count;
      break;
    case LOOM_AMDGPU_DELAY_ALU_TYPE_SALU: {
      const uint8_t salu_cycles = cycles > LOOM_AMDGPU_DELAY_ALU_SALU_CYCLES_MAX
                                      ? LOOM_AMDGPU_DELAY_ALU_SALU_CYCLES_MAX
                                      : cycles;
      info.salu_required_cycles = salu_cycles;
      info.salu_producer_node = producer_node;
      info.salu_producer_position = builder->current_position;
      break;
    }
    case LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER:
    default:
      break;
  }
  return info;
}

static void loom_amdgpu_wait_state_delay_alu_update_match(
    uint16_t required_cycle_count, uint16_t observed_cycle_count,
    uint16_t residual_cycle_count, uint32_t producer_node,
    loom_amdgpu_wait_state_match_t* match) {
  if (residual_cycle_count == 0 || residual_cycle_count <= match->cycle_count) {
    return;
  }
  *match = (loom_amdgpu_wait_state_match_t){
      .reason = LOOM_AMDGPU_WAIT_STATE_REASON_DELAY_ALU_DEPENDENCY,
      .producer_node = producer_node,
      .required_cycle_count = required_cycle_count,
      .observed_cycle_count = observed_cycle_count,
      .cycle_count = residual_cycle_count,
      .delay_alu_immediate = match->delay_alu_immediate,
      .matrix_wait_profile = LOOM_AMDGPU_MATRIX_WAIT_PROFILE_UNKNOWN,
      .matrix_result_use = LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_UNKNOWN,
  };
}

static bool loom_amdgpu_wait_state_delay_alu_info_is_current(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_amdgpu_delay_alu_info_t* info) {
  return info->epoch == builder->delay_alu_epoch;
}

static bool loom_amdgpu_wait_state_delay_alu_cycle_delta(
    const loom_amdgpu_wait_state_builder_t* builder, uint8_t required_cycles,
    uint64_t producer_position, uint16_t* out_observed_cycles,
    uint16_t* out_residual_cycles) {
  *out_observed_cycles = 0;
  *out_residual_cycles = 0;
  if (required_cycles == 0) {
    return false;
  }
  const uint64_t elapsed = builder->current_position >= producer_position
                               ? builder->current_position - producer_position
                               : 0;
  if (elapsed >= required_cycles) {
    return false;
  }
  *out_observed_cycles = (uint16_t)elapsed;
  *out_residual_cycles = (uint16_t)(required_cycles - elapsed);
  return true;
}

static bool loom_amdgpu_wait_state_delay_alu_counter_delta(
    uint64_t counter, uint64_t base, uint8_t maximum_delta,
    uint8_t* out_delta) {
  *out_delta = 0;
  if (counter < base) {
    return false;
  }
  const uint64_t delta = counter - base;
  if (delta >= maximum_delta) {
    return false;
  }
  *out_delta = (uint8_t)delta;
  return true;
}

// VALU/TRANS delay selectors identify recent producer packets by their ALU
// issue class. Ordinary scalar packets between producer and consumer do not
// advance that selector, so observed progress must come from the class counter
// rather than the generic instruction position.
static bool loom_amdgpu_wait_state_delay_alu_class_delta(
    uint8_t required_cycles, uint64_t counter, uint64_t base,
    uint8_t maximum_delta, uint16_t* out_observed_cycles,
    uint16_t* out_residual_cycles, uint8_t* out_number) {
  *out_observed_cycles = 0;
  *out_residual_cycles = 0;
  *out_number = 0;
  if (required_cycles == 0 ||
      !loom_amdgpu_wait_state_delay_alu_counter_delta(
          counter, base, maximum_delta, out_number) ||
      *out_number > required_cycles) {
    return false;
  }
  if (*out_number == required_cycles) {
    *out_observed_cycles = (uint16_t)(required_cycles - 1);
    *out_residual_cycles = 1;
    return true;
  }
  *out_observed_cycles = *out_number;
  *out_residual_cycles = (uint16_t)(required_cycles - *out_number);
  return true;
}

static bool loom_amdgpu_wait_state_delay_alu_valu_delta(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_amdgpu_delay_alu_info_t* info, uint16_t* out_observed_cycles,
    uint16_t* out_residual_cycles, uint8_t* out_valu_number) {
  return loom_amdgpu_wait_state_delay_alu_info_is_current(builder, info) &&
         loom_amdgpu_wait_state_delay_alu_class_delta(
             info->valu_required_cycles, builder->delay_alu_valu_count,
             info->valu_number_base, LOOM_AMDGPU_DELAY_ALU_VALU_MAX,
             out_observed_cycles, out_residual_cycles, out_valu_number);
}

static bool loom_amdgpu_wait_state_delay_alu_trans_delta(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_amdgpu_delay_alu_info_t* info, uint16_t* out_observed_cycles,
    uint16_t* out_residual_cycles, uint8_t* out_trans_number,
    uint8_t* out_trans_valu_number) {
  return loom_amdgpu_wait_state_delay_alu_info_is_current(builder, info) &&
         loom_amdgpu_wait_state_delay_alu_class_delta(
             info->trans_required_cycles, builder->delay_alu_trans_count,
             info->trans_number_base, LOOM_AMDGPU_DELAY_ALU_TRANS_MAX,
             out_observed_cycles, out_residual_cycles, out_trans_number) &&
         loom_amdgpu_wait_state_delay_alu_counter_delta(
             builder->delay_alu_valu_count, info->trans_valu_number_base,
             UINT8_MAX, out_trans_valu_number);
}

static bool loom_amdgpu_wait_state_delay_alu_salu_delta(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_amdgpu_delay_alu_info_t* info, uint16_t* out_observed_cycles,
    uint16_t* out_residual_cycles) {
  return loom_amdgpu_wait_state_delay_alu_info_is_current(builder, info) &&
         loom_amdgpu_wait_state_delay_alu_cycle_delta(
             builder, info->salu_required_cycles, info->salu_producer_position,
             out_observed_cycles, out_residual_cycles) &&
         *out_residual_cycles < LOOM_AMDGPU_DELAY_ALU_SALU_CYCLES_MAX;
}

static bool loom_amdgpu_wait_state_delay_alu_candidate_is_better(
    const loom_amdgpu_delay_alu_candidate_t* source,
    const loom_amdgpu_delay_alu_candidate_t* target) {
  if (source->residual_cycle_count != target->residual_cycle_count) {
    return source->residual_cycle_count > target->residual_cycle_count;
  }
  if (source->required_cycle_count != target->required_cycle_count) {
    return source->required_cycle_count > target->required_cycle_count;
  }
  if (source->observed_cycle_count != target->observed_cycle_count) {
    return source->observed_cycle_count < target->observed_cycle_count;
  }
  return source->dependency_code < target->dependency_code;
}

static loom_amdgpu_delay_alu_type_t
loom_amdgpu_wait_state_delay_alu_dependency_class(uint16_t dependency_code) {
  if (dependency_code <= 4) {
    return LOOM_AMDGPU_DELAY_ALU_TYPE_VALU;
  }
  if (dependency_code <= 7) {
    return LOOM_AMDGPU_DELAY_ALU_TYPE_TRANS;
  }
  return LOOM_AMDGPU_DELAY_ALU_TYPE_SALU;
}

static bool loom_amdgpu_wait_state_delay_alu_candidate_matches(
    const loom_amdgpu_delay_alu_candidate_t* lhs,
    const loom_amdgpu_delay_alu_candidate_t* rhs) {
  // Moves in one structural packet share an issue class and latency, so its
  // strongest residual subsumes earlier moves. VOPD component results have
  // distinct producer nodes but the same dependency selector because they
  // issue in one native packet; one selector waits for both.
  const bool same_dependency_class =
      loom_amdgpu_wait_state_delay_alu_dependency_class(lhs->dependency_code) ==
      loom_amdgpu_wait_state_delay_alu_dependency_class(rhs->dependency_code);
  return same_dependency_class &&
         (lhs->producer_node == rhs->producer_node ||
          lhs->dependency_code == rhs->dependency_code);
}

static void loom_amdgpu_wait_state_delay_alu_sort_candidates(
    loom_amdgpu_delay_alu_accumulator_t* accumulator) {
  if (accumulator->candidate_count < 2) {
    return;
  }
  if (loom_amdgpu_wait_state_delay_alu_candidate_is_better(
          &accumulator->candidates[1], &accumulator->candidates[0])) {
    const loom_amdgpu_delay_alu_candidate_t temporary =
        accumulator->candidates[0];
    accumulator->candidates[0] = accumulator->candidates[1];
    accumulator->candidates[1] = temporary;
  }
}

static void loom_amdgpu_wait_state_delay_alu_add_candidate(
    loom_amdgpu_delay_alu_accumulator_t* accumulator, uint16_t dependency_code,
    uint16_t required_cycle_count, uint16_t observed_cycle_count,
    uint16_t residual_cycle_count, uint32_t producer_node) {
  if (residual_cycle_count == 0) {
    return;
  }
  loom_amdgpu_wait_state_delay_alu_update_match(
      required_cycle_count, observed_cycle_count, residual_cycle_count,
      producer_node, &accumulator->fallback_match);
  const loom_amdgpu_delay_alu_candidate_t candidate = {
      .dependency_code = dependency_code,
      .producer_node = producer_node,
      .required_cycle_count = required_cycle_count,
      .observed_cycle_count = observed_cycle_count,
      .residual_cycle_count = residual_cycle_count,
  };
  for (uint8_t i = 0; i < accumulator->candidate_count; ++i) {
    if (!loom_amdgpu_wait_state_delay_alu_candidate_matches(
            &candidate, &accumulator->candidates[i])) {
      continue;
    }
    if (loom_amdgpu_wait_state_delay_alu_candidate_is_better(
            &candidate, &accumulator->candidates[i])) {
      accumulator->candidates[i] = candidate;
      loom_amdgpu_wait_state_delay_alu_sort_candidates(accumulator);
    }
    return;
  }
  if (accumulator->candidate_count < LOOM_AMDGPU_DELAY_ALU_SELECTOR_CAPACITY) {
    accumulator->candidates[accumulator->candidate_count++] = candidate;
    loom_amdgpu_wait_state_delay_alu_sort_candidates(accumulator);
    return;
  }
  accumulator->flags |=
      LOOM_AMDGPU_DELAY_ALU_ACCUMULATOR_FLAG_UNENCODED_CANDIDATES;
  const uint8_t worst_index = accumulator->candidate_count - 1;
  if (loom_amdgpu_wait_state_delay_alu_candidate_is_better(
          &candidate, &accumulator->candidates[worst_index])) {
    accumulator->candidates[worst_index] = candidate;
    loom_amdgpu_wait_state_delay_alu_sort_candidates(accumulator);
  }
}

static void loom_amdgpu_wait_state_delay_alu_accumulate_info(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_amdgpu_delay_alu_info_t* info,
    loom_amdgpu_delay_alu_accumulator_t* accumulator) {
  uint16_t observed_cycles = 0;
  uint16_t residual_cycles = 0;
  uint8_t trans_number = 0;
  uint8_t trans_valu_number = 0;
  const bool has_trans = loom_amdgpu_wait_state_delay_alu_trans_delta(
      builder, info, &observed_cycles, &residual_cycles, &trans_number,
      &trans_valu_number);
  if (has_trans) {
    loom_amdgpu_wait_state_delay_alu_add_candidate(
        accumulator, (uint16_t)(4u + trans_number), info->trans_required_cycles,
        observed_cycles, residual_cycles, info->trans_producer_node);
  }
  uint8_t valu_number = 0;
  if (loom_amdgpu_wait_state_delay_alu_valu_delta(
          builder, info, &observed_cycles, &residual_cycles, &valu_number) &&
      (!has_trans || valu_number <= trans_valu_number)) {
    loom_amdgpu_wait_state_delay_alu_add_candidate(
        accumulator, valu_number, info->valu_required_cycles, observed_cycles,
        residual_cycles, info->valu_producer_node);
  }
  if (loom_amdgpu_wait_state_delay_alu_salu_delta(
          builder, info, &observed_cycles, &residual_cycles)) {
    const uint16_t salu_code =
        (uint16_t)(residual_cycles + LOOM_AMDGPU_DELAY_ALU_SALU_BASE);
    loom_amdgpu_wait_state_delay_alu_add_candidate(
        accumulator, salu_code, info->salu_required_cycles, observed_cycles,
        residual_cycles, info->salu_producer_node);
  }
}

static loom_amdgpu_wait_state_action_t
loom_amdgpu_wait_state_delay_alu_accumulator_action(
    const loom_amdgpu_delay_alu_accumulator_t* accumulator,
    loom_amdgpu_wait_state_match_t* match) {
  if (accumulator->fallback_match.cycle_count == 0) {
    return LOOM_AMDGPU_WAIT_STATE_ACTION_UNKNOWN;
  }
  *match = accumulator->fallback_match;
  if (iree_any_bit_set(
          accumulator->flags,
          LOOM_AMDGPU_DELAY_ALU_ACCUMULATOR_FLAG_UNENCODED_CANDIDATES)) {
    match->delay_alu_immediate = 0;
    return LOOM_AMDGPU_WAIT_STATE_ACTION_S_NOP;
  }
  uint16_t immediate = 0;
  if (accumulator->candidate_count >= 1) {
    immediate |= accumulator->candidates[0].dependency_code;
  }
  if (accumulator->candidate_count >= 2) {
    immediate |= (uint16_t)(accumulator->candidates[1].dependency_code << 7);
  }
  match->delay_alu_immediate = immediate;
  return LOOM_AMDGPU_WAIT_STATE_ACTION_S_DELAY_ALU;
}

static void loom_amdgpu_wait_state_delay_alu_clear_all(
    loom_amdgpu_wait_state_builder_t* builder) {
  ++builder->delay_alu_epoch;
  builder->delay_alu_valu_count = 0;
  builder->delay_alu_trans_count = 0;
  if (builder->delay_alu_epoch != 0) {
    return;
  }
  builder->delay_alu_epoch = 1;
  for (iree_host_size_t i = 0; i < builder->vgpr_count; ++i) {
    builder->vgprs[i].delay_alu = (loom_amdgpu_delay_alu_info_t){0};
  }
  for (iree_host_size_t i = 0; i < builder->sgpr_count; ++i) {
    builder->sgprs[i].delay_alu = (loom_amdgpu_delay_alu_info_t){0};
  }
  builder->scc_delay_alu = (loom_amdgpu_delay_alu_info_t){0};
}

static void loom_amdgpu_wait_state_delay_alu_advance_counters(
    loom_amdgpu_wait_state_builder_t* builder,
    loom_amdgpu_delay_alu_type_t type, uint64_t instruction_count) {
  if (instruction_count == 0) {
    return;
  }
  switch (type) {
    case LOOM_AMDGPU_DELAY_ALU_TYPE_VALU:
      builder->delay_alu_valu_count += instruction_count;
      break;
    case LOOM_AMDGPU_DELAY_ALU_TYPE_TRANS:
      builder->delay_alu_trans_count += instruction_count;
      break;
    case LOOM_AMDGPU_DELAY_ALU_TYPE_SALU:
    case LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER:
    default:
      break;
  }
}

static void loom_amdgpu_wait_state_delay_alu_match_assignment(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment,
    loom_amdgpu_delay_alu_accumulator_t* accumulator) {
  if (assignment == NULL) {
    return;
  }
  if (loom_amdgpu_wait_state_assignment_is_physical_scc(assignment)) {
    loom_amdgpu_wait_state_delay_alu_accumulate_info(
        builder, &builder->scc_delay_alu, accumulator);
    builder->scc_delay_alu = (loom_amdgpu_delay_alu_info_t){0};
    return;
  }
  uint64_t end = 0;
  if (!loom_low_allocation_assignment_location_exclusive_end(assignment,
                                                             &end)) {
    return;
  }
  if (loom_amdgpu_wait_state_assignment_is_physical_vgpr(assignment)) {
    if (end > builder->vgpr_count) {
      return;
    }
    for (uint32_t i = 0; i < assignment->location_count; ++i) {
      loom_amdgpu_delay_alu_info_t* slot =
          &builder->vgprs[assignment->location_base + i].delay_alu;
      loom_amdgpu_wait_state_delay_alu_accumulate_info(builder, slot,
                                                       accumulator);
      *slot = (loom_amdgpu_delay_alu_info_t){0};
    }
    return;
  }
  if (loom_amdgpu_wait_state_assignment_is_physical_sgpr(assignment)) {
    if (end > builder->sgpr_count) {
      return;
    }
    for (uint32_t i = 0; i < assignment->location_count; ++i) {
      loom_amdgpu_delay_alu_info_t* slot =
          &builder->sgprs[assignment->location_base + i].delay_alu;
      loom_amdgpu_wait_state_delay_alu_accumulate_info(builder, slot,
                                                       accumulator);
      *slot = (loom_amdgpu_delay_alu_info_t){0};
    }
  }
}

static void loom_amdgpu_wait_state_delay_alu_accumulate_operands(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_delay_alu_accumulator_t* accumulator) {
  const loom_op_t* op = packet->node->op;
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_amdgpu_wait_state_assignment(builder->allocation,
                                          loom_op_const_operands(op)[i]);
    loom_amdgpu_wait_state_delay_alu_match_assignment(builder, assignment,
                                                      accumulator);
  }
}

static loom_amdgpu_wait_state_action_t
loom_amdgpu_wait_state_delay_alu_match_operands(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_wait_state_match_t* match) {
  loom_amdgpu_delay_alu_accumulator_t accumulator = {0};
  loom_amdgpu_wait_state_delay_alu_accumulate_operands(builder, packet,
                                                       &accumulator);
  return loom_amdgpu_wait_state_delay_alu_accumulator_action(&accumulator,
                                                             match);
}

static void loom_amdgpu_wait_state_delay_alu_record_assignment(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment,
    loom_amdgpu_delay_alu_type_t type, uint16_t latency_cycles,
    uint32_t producer_node) {
  if (type == LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER || latency_cycles == 0 ||
      assignment == NULL) {
    return;
  }
  const loom_amdgpu_delay_alu_info_t info =
      loom_amdgpu_wait_state_delay_alu_make_info(builder, type, latency_cycles,
                                                 producer_node);
  if (loom_amdgpu_wait_state_assignment_is_physical_scc(assignment)) {
    builder->scc_delay_alu = info;
    return;
  }
  uint64_t end = 0;
  if (!loom_low_allocation_assignment_location_exclusive_end(assignment,
                                                             &end)) {
    return;
  }
  if (loom_amdgpu_wait_state_assignment_is_physical_vgpr(assignment)) {
    if (end > builder->vgpr_count) {
      return;
    }
    for (uint32_t i = 0; i < assignment->location_count; ++i) {
      builder->vgprs[assignment->location_base + i].delay_alu = info;
    }
    return;
  }
  if (loom_amdgpu_wait_state_assignment_is_physical_sgpr(assignment)) {
    if (end > builder->sgpr_count) {
      return;
    }
    for (uint32_t i = 0; i < assignment->location_count; ++i) {
      builder->sgprs[assignment->location_base + i].delay_alu = info;
    }
  }
}

static void loom_amdgpu_wait_state_delay_alu_record_results(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet, loom_amdgpu_delay_alu_type_t type,
    uint16_t latency_cycles) {
  if (type == LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER || latency_cycles == 0) {
    return;
  }
  const loom_op_t* op = packet->node->op;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_amdgpu_wait_state_assignment(builder->allocation,
                                          loom_op_const_results(op)[i]);
    loom_amdgpu_wait_state_delay_alu_record_assignment(
        builder, assignment, type, latency_cycles, packet->node_index);
  }
}

static void loom_amdgpu_wait_state_delay_alu_clear_assignment(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment) {
  if (assignment == NULL) {
    return;
  }
  if (loom_amdgpu_wait_state_assignment_is_physical_scc(assignment)) {
    builder->scc_delay_alu = (loom_amdgpu_delay_alu_info_t){0};
    return;
  }
  uint64_t end = 0;
  if (!loom_low_allocation_assignment_location_exclusive_end(assignment,
                                                             &end)) {
    return;
  }
  if (loom_amdgpu_wait_state_assignment_is_physical_vgpr(assignment)) {
    if (end > builder->vgpr_count) {
      return;
    }
    for (uint32_t i = 0; i < assignment->location_count; ++i) {
      builder->vgprs[assignment->location_base + i].delay_alu =
          (loom_amdgpu_delay_alu_info_t){0};
    }
    return;
  }
  if (loom_amdgpu_wait_state_assignment_is_physical_sgpr(assignment)) {
    if (end > builder->sgpr_count) {
      return;
    }
    for (uint32_t i = 0; i < assignment->location_count; ++i) {
      builder->sgprs[assignment->location_base + i].delay_alu =
          (loom_amdgpu_delay_alu_info_t){0};
    }
  }
}

static void loom_amdgpu_wait_state_delay_alu_clear_results(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet) {
  const loom_op_t* op = packet->node->op;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_amdgpu_wait_state_assignment(builder->allocation,
                                          loom_op_const_results(op)[i]);
    loom_amdgpu_wait_state_delay_alu_clear_assignment(builder, assignment);
  }
}

static void loom_amdgpu_wait_state_record_move_vgpr_hazard(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet, uint32_t location,
    loom_amdgpu_wait_state_reason_t reason, uint16_t cycle_count) {
  loom_amdgpu_wait_state_hazard_t* hazard =
      &builder->vgprs[location].hazards[reason];
  *hazard = (loom_amdgpu_wait_state_hazard_t){
      .flags = LOOM_AMDGPU_WAIT_STATE_VGPR_FLAG_VALID,
      .reason = reason,
      .producer_node = packet->node_index,
      .producer_location_base = location,
      .producer_location_count = 1,
      .producer_end_position = builder->current_position + 1,
      .required_cycle_count = cycle_count,
  };
}

static void loom_amdgpu_wait_state_apply_move(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet, const loom_low_move_t* move,
    bool processor_has_valu_sgpr_read_hazard) {
  const uint16_t register_class_id = move->destination.descriptor_reg_class_id;
  const uint32_t location = move->destination.location;
  loom_amdgpu_delay_alu_type_t delay_alu_type =
      LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER;
  if (register_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    loom_amdgpu_wait_state_vgpr_t* state = &builder->vgprs[location];
    *state = (loom_amdgpu_wait_state_vgpr_t){0};
    loom_amdgpu_wait_state_record_move_vgpr_hazard(
        builder, packet, location,
        LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE,
        LOOM_AMDGPU_WAIT_STATE_VALU_TO_MATRIX_CYCLES);
    if (!builder->has_delay_alu) {
      loom_amdgpu_wait_state_record_move_vgpr_hazard(
          builder, packet, location,
          LOOM_AMDGPU_WAIT_STATE_REASON_DPP_VGPR_READ,
          LOOM_AMDGPU_WAIT_STATE_DPP_VGPR_READ_CYCLES);
    }
    if (processor_has_valu_sgpr_read_hazard) {
      loom_amdgpu_wait_state_record_move_vgpr_hazard(
          builder, packet, location,
          LOOM_AMDGPU_WAIT_STATE_REASON_READFIRSTLANE_VGPR_READ,
          LOOM_AMDGPU_WAIT_STATE_READFIRSTLANE_VGPR_READ_CYCLES);
    }
    if (builder->has_delay_alu) {
      delay_alu_type = LOOM_AMDGPU_DELAY_ALU_TYPE_VALU;
      const uint16_t delay_alu_latency_cycles =
          loom_amdgpu_wait_state_delay_alu_latency_cycles(delay_alu_type, 0);
      state->delay_alu = loom_amdgpu_wait_state_delay_alu_make_info(
          builder, delay_alu_type, delay_alu_latency_cycles,
          packet->node_index);
    }
  } else if (register_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    loom_amdgpu_wait_state_sgpr_t* state = &builder->sgprs[location];
    *state = (loom_amdgpu_wait_state_sgpr_t){0};
    if (builder->has_delay_alu) {
      delay_alu_type = LOOM_AMDGPU_DELAY_ALU_TYPE_SALU;
      const uint16_t delay_alu_latency_cycles =
          loom_amdgpu_wait_state_delay_alu_latency_cycles(delay_alu_type, 1);
      state->delay_alu = loom_amdgpu_wait_state_delay_alu_make_info(
          builder, delay_alu_type, delay_alu_latency_cycles,
          packet->node_index);
    }
  }
  loom_amdgpu_wait_state_delay_alu_advance_counters(builder, delay_alu_type, 1);
  ++builder->current_position;
}

static void loom_amdgpu_wait_state_apply_moves(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet, loom_low_move_range_t moves,
    bool processor_has_valu_sgpr_read_hazard) {
  // Consume the allocator's final native order once. Each destination must
  // retain its own producer position because the last move has no same-class
  // progress before a following consumer.
  for (iree_host_size_t i = 0; i < moves.count; ++i) {
    const loom_low_move_t* move = &builder->allocation->moves[moves.start + i];
    loom_amdgpu_wait_state_apply_move(builder, packet, move,
                                      processor_has_valu_sgpr_read_hazard);
  }
}

static iree_status_t loom_amdgpu_wait_state_append(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* insertion_packet,
    const loom_low_packet_view_t* consumer_packet,
    const loom_amdgpu_wait_state_match_t* match,
    loom_amdgpu_wait_state_action_t action) {
  if (match->cycle_count == 0) {
    return iree_ok_status();
  }
  if (builder->state_count >= builder->state_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AMDGPU wait-state plan capacity exhausted");
  }
  builder->states[builder->state_count++] = (loom_amdgpu_wait_state_t){
      .reason = match->reason,
      .action = action,
      .block_index = insertion_packet->node->block_index,
      .node_index = insertion_packet->node_index,
      .scheduled_ordinal = insertion_packet->node->scheduled_ordinal,
      .producer_node = match->producer_node,
      .consumer_node = consumer_packet->node->source_ordinal,
      .required_cycle_count = match->required_cycle_count,
      .observed_cycle_count = match->observed_cycle_count,
      .cycle_count = match->cycle_count,
      .delay_alu_immediate = match->delay_alu_immediate,
      .matrix_wait_profile = match->matrix_wait_profile,
      .matrix_result_use = match->matrix_result_use,
      .matrix_pass_count = match->matrix_pass_count,
  };
  if (action == LOOM_AMDGPU_WAIT_STATE_ACTION_V_NOP) {
    IREE_ASSERT(builder->matrix_coexecution != NULL);
    loom_amdgpu_matrix_coexecution_advance(builder->matrix_coexecution,
                                           match->cycle_count);
    loom_amdgpu_wait_state_delay_alu_advance_counters(
        builder, LOOM_AMDGPU_DELAY_ALU_TYPE_VALU, match->cycle_count);
    builder->current_position += match->cycle_count;
  } else if (action == LOOM_AMDGPU_WAIT_STATE_ACTION_S_NOP) {
    builder->current_position += match->cycle_count;
  } else {
    builder->current_position += 1;
  }
  return iree_ok_status();
}

static void loom_amdgpu_wait_state_match_descriptor_operands(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons,
    loom_amdgpu_wait_state_match_t* match) {
  const loom_op_t* op = packet->node->op;
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_amdgpu_wait_state_match_value(builder, loom_op_const_operands(op)[i],
                                       allowed_reasons, match);
  }
}

static void loom_amdgpu_wait_state_match_result_writes(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons,
    loom_amdgpu_wait_state_match_t* match) {
  const loom_op_t* op = packet->node->op;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_amdgpu_wait_state_assignment(builder->allocation,
                                          loom_op_const_results(op)[i]);
    loom_amdgpu_wait_state_match_assignment(builder, assignment,
                                            allowed_reasons, match);
  }
}

static void loom_amdgpu_wait_state_match_descriptor_sgpr_operands(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons,
    loom_amdgpu_wait_state_match_t* match) {
  const loom_op_t* op = packet->node->op;
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_amdgpu_wait_state_match_sgpr_value(
        builder, loom_op_const_operands(op)[i], allowed_reasons, match);
  }
}

static void loom_amdgpu_wait_state_match_matrix_descriptor_operands(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_matrix_contract_descriptor_t* contract,
    loom_amdgpu_wait_state_match_t* match) {
  const loom_op_t* op = packet->node->op;
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_amdgpu_wait_state_matrix_result_use_t use =
        loom_amdgpu_wait_state_matrix_operand_result_use(contract, i);
    if (use == LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_UNKNOWN) {
      continue;
    }
    loom_amdgpu_wait_state_match_matrix_result_value(
        builder, loom_op_const_operands(op)[i], use, match);
  }
}

static void loom_amdgpu_wait_state_clear_results(
    loom_amdgpu_wait_state_builder_t* builder, const loom_op_t* op) {
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_amdgpu_wait_state_assignment(builder->allocation,
                                          loom_op_const_results(op)[i]);
    loom_amdgpu_wait_state_clear_assignment(builder, assignment);
  }
}

static void loom_amdgpu_wait_state_record_results(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_wait_state_reason_t reason, uint16_t cycle_count,
    uint16_t matrix_pass_count, uint64_t instruction_count) {
  const loom_op_t* op = packet->node->op;
  const uint64_t producer_end_position =
      builder->current_position + instruction_count;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_amdgpu_wait_state_assignment(builder->allocation,
                                          loom_op_const_results(op)[i]);
    loom_amdgpu_wait_state_record_assignment(
        builder, assignment, reason, packet->node_index, cycle_count,
        matrix_pass_count, producer_end_position);
  }
}

static void loom_amdgpu_wait_state_record_sgpr_results(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_wait_state_reason_t reason, uint16_t cycle_count,
    uint64_t instruction_count) {
  const loom_op_t* op = packet->node->op;
  const uint64_t producer_end_position =
      builder->current_position + instruction_count;
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_amdgpu_wait_state_assignment(builder->allocation,
                                          loom_op_const_results(op)[i]);
    loom_amdgpu_wait_state_record_sgpr_assignment(
        builder, assignment, reason, packet->node_index, cycle_count,
        producer_end_position);
  }
}

static void loom_amdgpu_wait_state_record_vector_alu_results(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_wait_state_packet_info_t* info) {
  const bool processor_has_valu_sgpr_read_hazard =
      loom_amdgpu_wait_state_has_scheduling(
          builder, LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES);
  loom_amdgpu_wait_state_clear_results(builder, packet->node->op);
  loom_amdgpu_wait_state_record_results(
      builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE,
      LOOM_AMDGPU_WAIT_STATE_VALU_TO_MATRIX_CYCLES, 0, info->instruction_count);
  if (!builder->has_delay_alu) {
    loom_amdgpu_wait_state_record_results(
        builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_DPP_VGPR_READ,
        LOOM_AMDGPU_WAIT_STATE_DPP_VGPR_READ_CYCLES, 0,
        info->instruction_count);
  }
  if (processor_has_valu_sgpr_read_hazard) {
    loom_amdgpu_wait_state_record_results(
        builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_READFIRSTLANE_VGPR_READ,
        LOOM_AMDGPU_WAIT_STATE_READFIRSTLANE_VGPR_READ_CYCLES, 0,
        info->instruction_count);
  }
  if (iree_any_bit_set(info->flags,
                       LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_TRANS_PRODUCER)) {
    loom_amdgpu_wait_state_record_results(
        builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_TRANS_RESULT_USE,
        LOOM_AMDGPU_WAIT_STATE_TRANS_RESULT_USE_CYCLES, 0,
        info->instruction_count);
  }
  if (iree_any_bit_set(
          info->flags,
          LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DST_SEL_FORWARDING_PRODUCER)) {
    loom_amdgpu_wait_state_record_results(
        builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_DST_SEL_FORWARDING_USE,
        LOOM_AMDGPU_WAIT_STATE_DST_SEL_FORWARDING_CYCLES, 0,
        info->instruction_count);
  }
  if (iree_any_bit_set(
          info->flags,
          LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_VALU_SGPR_READ_PRODUCER)) {
    loom_amdgpu_wait_state_record_sgpr_results(
        builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_VALU_SGPR_READ,
        LOOM_AMDGPU_WAIT_STATE_VALU_SGPR_READ_CYCLES, info->instruction_count);
  }
}

static iree_status_t loom_amdgpu_wait_state_packet_analyze(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_wait_state_packet_info_t* out_info) {
  *out_info = (loom_amdgpu_wait_state_packet_info_t){0};
  if (packet->descriptor == NULL) {
    out_info->structural = loom_amdgpu_structural_packet_analyze(
        builder->schedule, builder->allocation, packet->node, 0);
    out_info->instruction_count = out_info->structural.instruction_count;
    out_info->delay_alu_type = loom_amdgpu_wait_state_structural_delay_alu_type(
        builder, &out_info->structural);
    if (out_info->delay_alu_type != LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER) {
      out_info->delay_alu_latency_cycles =
          loom_amdgpu_wait_state_delay_alu_latency_cycles(
              out_info->delay_alu_type, 0);
    }
    return iree_ok_status();
  }

  out_info->flags |= LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DESCRIPTOR;
  out_info->instruction_count = 1;
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  const loom_low_descriptor_set_t* descriptor_set = builder->descriptor_set;
  const loom_amdgpu_descriptor_traits_t descriptor_traits =
      loom_amdgpu_descriptor_traits(descriptor_set, descriptor);
  if (iree_any_bit_set(descriptor_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU)) {
    out_info->flags |= LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_USES_VECTOR_ALU;
  }
  if (iree_any_bit_set(descriptor_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY)) {
    out_info->flags |= LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_USES_VECTOR_MEMORY;
  }

  out_info->matrix_contract =
      loom_amdgpu_wait_state_contract_for_descriptor(builder, descriptor);
  if (out_info->matrix_contract != NULL) {
    if (loom_amdgpu_wait_state_matrix_result_pass_count(
            out_info->matrix_contract, &out_info->matrix_pass_count)) {
      out_info->flags |= LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_MATRIX;
      if (!loom_amdgpu_wait_state_matrix_result_wait_cycles(
              builder, out_info->matrix_pass_count,
              LOOM_AMDGPU_WAIT_STATE_MATRIX_RESULT_USE_NON_MATRIX,
              &out_info->matrix_wait_cycles)) {
        IREE_ASSERT_UNREACHABLE(
            "generated AMDGPU matrix wait-state table is missing a result-use "
            "row");
      }
    }
    if (loom_amdgpu_wait_state_matrix_reads_valu_results(
            out_info->matrix_contract)) {
      out_info->flags |= LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_MATRIX_READS_VALU;
    }
  }

  const bool processor_has_trans_waits = loom_amdgpu_wait_state_has_scheduling(
      builder, LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_WAIT_STATES);
  if (processor_has_trans_waits &&
      iree_any_bit_set(descriptor_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_TRANSCENDENTAL)) {
    out_info->flags |= LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_TRANS_PRODUCER;
  }
  if (processor_has_trans_waits &&
      iree_any_bit_set(descriptor_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU) &&
      !iree_any_bit_set(descriptor_traits,
                        LOOM_AMDGPU_DESCRIPTOR_TRAIT_TRANSCENDENTAL)) {
    out_info->flags |=
        LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_TRANS_FORWARDING_CONSUMER;
  }

  const bool processor_has_valu_sgpr_read_hazard =
      loom_amdgpu_wait_state_has_scheduling(
          builder, LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES);
  if (iree_any_bit_set(descriptor_traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_DPP)) {
    out_info->flags |= LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DPP_CONSUMER;
  }
  if (iree_any_bit_set(descriptor_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_READFIRSTLANE)) {
    out_info->flags |=
        LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_READFIRSTLANE_CONSUMER;
  }
  if (processor_has_valu_sgpr_read_hazard &&
      iree_any_bit_set(descriptor_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU |
                           LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY)) {
    out_info->flags |=
        LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_VALU_SGPR_READ_CONSUMER;
  }
  if (processor_has_valu_sgpr_read_hazard &&
      iree_any_bit_set(descriptor_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU)) {
    out_info->flags |=
        LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_VALU_SGPR_READ_PRODUCER;
  }

  if (loom_amdgpu_wait_state_has_scheduling(
          builder,
          LOOM_AMDGPU_PROCESSOR_SCHEDULING_DESTINATION_SELECTION_WAIT_STATES)) {
    bool has_forwarding_hazard = false;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_wait_state_packet_has_destination_selection_forwarding_hazard(
            builder, packet, descriptor_traits, &has_forwarding_hazard));
    if (has_forwarding_hazard) {
      out_info->flags |=
          LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DST_SEL_FORWARDING_PRODUCER;
    }
    if (iree_any_bit_set(descriptor_traits,
                         LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU)) {
      out_info->flags |=
          LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DST_SEL_FORWARDING_CONSUMER;
    }
  }

  // RDNA matrix instructions use the vector ALU dependency mechanism even
  // though their schedule resource is classified separately from ordinary
  // VALU instructions. Targets without S_DELAY_ALU reject the classification
  // inside loom_amdgpu_wait_state_delay_alu_type.
  out_info->delay_alu_type =
      loom_amdgpu_wait_state_delay_alu_type(builder, descriptor_traits);
  if (out_info->delay_alu_type != LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER) {
    out_info->delay_alu_latency_cycles =
        loom_amdgpu_wait_state_delay_alu_latency_cycles(
            out_info->delay_alu_type,
            loom_amdgpu_wait_state_descriptor_latency_cycles(builder,
                                                             descriptor));
  }
  return iree_ok_status();
}

static void loom_amdgpu_wait_state_match_structural_operands(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_structural_packet_info_t* info,
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons,
    loom_amdgpu_wait_state_match_t* match) {
  const loom_op_t* op = packet->node->op;
  if (loom_low_br_isa(op)) {
    loom_value_slice_t args = loom_low_br_args(op);
    for (uint16_t i = 0; i < args.count; ++i) {
      loom_amdgpu_wait_state_match_value(builder, args.values[i],
                                         allowed_reasons, match);
    }
    return;
  }
  if (!iree_any_bit_set(info->flags,
                        LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES)) {
    return;
  }
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_amdgpu_wait_state_match_value(builder, loom_op_const_operands(op)[i],
                                       allowed_reasons, match);
  }
}

static void loom_amdgpu_wait_state_match_packet_hazards(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_wait_state_packet_info_t* info,
    loom_amdgpu_wait_state_match_t* match) {
  const bool processor_has_valu_sgpr_read_hazard =
      loom_amdgpu_wait_state_has_scheduling(
          builder, LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES);
  const bool processor_has_delay_alu = builder->has_delay_alu;

  if (iree_any_bit_set(info->flags,
                       LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_MATRIX_READS_VALU)) {
    loom_amdgpu_wait_state_match_matrix_descriptor_operands(
        builder, packet, info->matrix_contract, match);
    loom_amdgpu_wait_state_match_descriptor_operands(
        builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_TO_MATRIX_USE,
        match);
  } else if (iree_any_bit_set(info->flags,
                              LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DESCRIPTOR)) {
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons =
        LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_MATRIX_RESULT_USE;
    if (iree_any_bit_set(
            info->flags,
            LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_TRANS_FORWARDING_CONSUMER)) {
      allowed_reasons |= LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_TRANS_RESULT_USE;
    }
    if (iree_any_bit_set(info->flags,
                         LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DPP_CONSUMER) &&
        !processor_has_delay_alu) {
      allowed_reasons |= LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_DPP_VGPR_READ;
    }
    if (iree_any_bit_set(
            info->flags,
            LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_READFIRSTLANE_CONSUMER) &&
        processor_has_valu_sgpr_read_hazard) {
      allowed_reasons |=
          LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_READFIRSTLANE_VGPR_READ;
    }
    if (iree_any_bit_set(
            info->flags,
            LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DST_SEL_FORWARDING_CONSUMER)) {
      allowed_reasons |=
          LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_DST_SEL_FORWARDING_USE;
    }
    loom_amdgpu_wait_state_match_descriptor_operands(builder, packet,
                                                     allowed_reasons, match);
    if (iree_any_bit_set(
            info->flags,
            LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_VALU_SGPR_READ_CONSUMER)) {
      loom_amdgpu_wait_state_match_descriptor_sgpr_operands(
          builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_SGPR_READ,
          match);
    }
  } else {
    loom_amdgpu_wait_state_match_structural_operands(
        builder, packet, &info->structural,
        LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_MATRIX_RESULT_USE |
            LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_TRANS_RESULT_USE,
        match);
  }
  if (!iree_any_bit_set(info->flags,
                        LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_MATRIX) &&
      (iree_any_bit_set(info->flags,
                        LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DESCRIPTOR) ||
       (packet->node->op->result_count != 0 &&
        iree_any_bit_set(info->structural.flags,
                         LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES)))) {
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons =
        LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_MATRIX_RESULT_USE;
    if (iree_any_bit_set(
            info->flags,
            LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DST_SEL_FORWARDING_CONSUMER)) {
      allowed_reasons |=
          LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_DST_SEL_FORWARDING_USE;
    }
    loom_amdgpu_wait_state_match_result_writes(builder, packet, allowed_reasons,
                                               match);
  }
}

static iree_status_t loom_amdgpu_wait_state_apply_matrix_coexecution(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* insertion_packet,
    const loom_low_packet_view_t* consumer_packet,
    const loom_amdgpu_matrix_coexecution_match_t* coexecution_match) {
  if (coexecution_match->residual_issue_count == 0) {
    return iree_ok_status();
  }
  const loom_amdgpu_wait_state_match_t match = {
      .reason =
          coexecution_match->matrix_consumer
              ? LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_COEXECUTION_MATRIX_USE
              : LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_COEXECUTION_VALU_USE,
      .producer_node = coexecution_match->producer_node,
      .required_cycle_count = coexecution_match->required_issue_count,
      .observed_cycle_count = coexecution_match->observed_issue_count,
      .cycle_count = coexecution_match->residual_issue_count,
  };
  return loom_amdgpu_wait_state_append(builder, insertion_packet,
                                       consumer_packet, &match,
                                       LOOM_AMDGPU_WAIT_STATE_ACTION_V_NOP);
}

static iree_status_t loom_amdgpu_wait_state_apply_packet(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet) {
  loom_amdgpu_wait_state_packet_info_t info = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_wait_state_packet_analyze(builder, packet, &info));
  const bool processor_has_valu_sgpr_read_hazard =
      loom_amdgpu_wait_state_has_scheduling(
          builder, LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES);
  const bool processor_has_delay_alu = builder->has_delay_alu;

  uint16_t vector_issue_count = 0;
  if (builder->matrix_coexecution != NULL) {
    loom_amdgpu_matrix_coexecution_match_t coexecution_match = {0};
    loom_amdgpu_matrix_coexecution_inspect_packet(
        builder->matrix_coexecution, packet,
        packet->descriptor == NULL ? &info.structural : NULL,
        &coexecution_match, &vector_issue_count);
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_apply_matrix_coexecution(
        builder, packet, packet, &coexecution_match));
  }

  loom_amdgpu_wait_state_match_t match = {0};
  loom_amdgpu_wait_state_match_packet_hazards(builder, packet, &info, &match);
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_append(
      builder, packet, packet, &match, LOOM_AMDGPU_WAIT_STATE_ACTION_S_NOP));

  if (iree_any_bit_set(info.flags,
                       LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_USES_VECTOR_MEMORY)) {
    // VMEM packets are not VALU consumers for fixed ALU dependency windows and
    // naturally separate later VALU consumers from currently tracked producers.
    loom_amdgpu_wait_state_delay_alu_clear_all(builder);
  } else if ((iree_any_bit_set(info.flags,
                               LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DESCRIPTOR) &&
              info.delay_alu_type != LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER) ||
             info.structural.moves.count != 0 ||
             iree_any_bit_set(info.structural.flags,
                              LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_READS_SCC)) {
    loom_amdgpu_wait_state_match_t delay_alu_match = {0};
    const loom_amdgpu_wait_state_action_t delay_alu_action =
        loom_amdgpu_wait_state_delay_alu_match_operands(builder, packet,
                                                        &delay_alu_match);
    if (delay_alu_action != LOOM_AMDGPU_WAIT_STATE_ACTION_UNKNOWN) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_append(
          builder, packet, packet, &delay_alu_match, delay_alu_action));
    }
  }

  if (iree_any_bit_set(info.flags, LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_MATRIX)) {
    loom_amdgpu_wait_state_clear_results(builder, packet->node->op);
    loom_amdgpu_wait_state_record_results(
        builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_RESULT_USE,
        info.matrix_wait_cycles, info.matrix_pass_count,
        info.instruction_count);
  } else if (iree_any_bit_set(
                 info.flags,
                 LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_USES_VECTOR_ALU)) {
    loom_amdgpu_wait_state_record_vector_alu_results(builder, packet, &info);
  } else if (iree_any_bit_set(info.flags,
                              LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DESCRIPTOR)) {
    loom_amdgpu_wait_state_clear_results(builder, packet->node->op);
  } else if (info.structural.moves.count != 0) {
    loom_amdgpu_wait_state_apply_moves(builder, packet, info.structural.moves,
                                       processor_has_valu_sgpr_read_hazard);
  } else {
    if (info.structural.vector_alu_instruction_count != 0) {
      loom_amdgpu_wait_state_clear_results(builder, packet->node->op);
      loom_amdgpu_wait_state_record_results(
          builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE,
          LOOM_AMDGPU_WAIT_STATE_VALU_TO_MATRIX_CYCLES, 0,
          info.instruction_count);
      if (!processor_has_delay_alu) {
        loom_amdgpu_wait_state_record_results(
            builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_DPP_VGPR_READ,
            LOOM_AMDGPU_WAIT_STATE_DPP_VGPR_READ_CYCLES, 0,
            info.instruction_count);
      }
      if (processor_has_valu_sgpr_read_hazard) {
        loom_amdgpu_wait_state_record_results(
            builder, packet,
            LOOM_AMDGPU_WAIT_STATE_REASON_READFIRSTLANE_VGPR_READ,
            LOOM_AMDGPU_WAIT_STATE_READFIRSTLANE_VGPR_READ_CYCLES, 0,
            info.instruction_count);
      }
    } else if (info.instruction_count != 0) {
      loom_amdgpu_wait_state_clear_results(builder, packet->node->op);
    }
  }

  if (info.structural.moves.count == 0) {
    if (info.delay_alu_type != LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER &&
        info.delay_alu_latency_cycles != 0) {
      loom_amdgpu_wait_state_delay_alu_record_results(
          builder, packet, info.delay_alu_type, info.delay_alu_latency_cycles);
    } else if (info.instruction_count != 0) {
      loom_amdgpu_wait_state_delay_alu_clear_results(builder, packet);
    }
    loom_amdgpu_wait_state_delay_alu_advance_counters(
        builder, info.delay_alu_type, info.instruction_count);
    builder->current_position += info.instruction_count;
  } else {
    loom_amdgpu_wait_state_delay_alu_advance_counters(
        builder, LOOM_AMDGPU_DELAY_ALU_TYPE_VALU,
        info.structural.vector_alu_instruction_count);
    loom_amdgpu_wait_state_delay_alu_advance_counters(
        builder, LOOM_AMDGPU_DELAY_ALU_TYPE_SALU,
        info.structural.scalar_alu_instruction_count);
    builder->current_position +=
        info.instruction_count - info.structural.moves.count;
  }
  if (info.instruction_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU wait-state progress exceeds uint32_t units");
  }
  builder->packet_instruction_counts[packet->packet_index] =
      (uint32_t)info.instruction_count;
  if (builder->matrix_coexecution != NULL) {
    IREE_ASSERT_LE(vector_issue_count, info.instruction_count);
    loom_amdgpu_matrix_coexecution_commit_packet(builder->matrix_coexecution,
                                                 packet, vector_issue_count);
  }
  if (info.instruction_count != 0) {
    ++builder->progress_event_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_state_apply_vopd_pair(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* first_packet,
    const loom_low_packet_view_t* second_packet) {
  loom_amdgpu_wait_state_packet_info_t first_info = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_packet_analyze(
      builder, first_packet, &first_info));
  loom_amdgpu_wait_state_packet_info_t second_info = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_packet_analyze(
      builder, second_packet, &second_info));

  if (builder->matrix_coexecution != NULL) {
    loom_amdgpu_matrix_coexecution_match_t coexecution_match = {0};
    uint16_t first_vector_issue_count = 0;
    loom_amdgpu_matrix_coexecution_inspect_packet(
        builder->matrix_coexecution, first_packet, NULL, &coexecution_match,
        &first_vector_issue_count);
    IREE_ASSERT_EQ(first_vector_issue_count, 1);
    const loom_low_packet_view_t* coexecution_consumer = first_packet;
    const uint16_t first_residual = coexecution_match.residual_issue_count;
    uint16_t second_vector_issue_count = 0;
    loom_amdgpu_matrix_coexecution_inspect_packet(
        builder->matrix_coexecution, second_packet, NULL, &coexecution_match,
        &second_vector_issue_count);
    IREE_ASSERT_EQ(second_vector_issue_count, 1);
    if (coexecution_match.residual_issue_count > first_residual) {
      coexecution_consumer = second_packet;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_apply_matrix_coexecution(
        builder, first_packet, coexecution_consumer, &coexecution_match));
  }

  loom_amdgpu_wait_state_match_t match = {0};
  loom_amdgpu_wait_state_match_packet_hazards(builder, first_packet,
                                              &first_info, &match);
  const loom_low_packet_view_t* match_consumer = first_packet;
  const uint16_t first_match_cycles = match.cycle_count;
  loom_amdgpu_wait_state_match_packet_hazards(builder, second_packet,
                                              &second_info, &match);
  if (match.cycle_count > first_match_cycles) {
    match_consumer = second_packet;
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_append(
      builder, first_packet, match_consumer, &match,
      LOOM_AMDGPU_WAIT_STATE_ACTION_S_NOP));

  loom_amdgpu_delay_alu_accumulator_t accumulator = {0};
  loom_amdgpu_wait_state_delay_alu_accumulate_operands(builder, first_packet,
                                                       &accumulator);
  const loom_low_packet_view_t* delay_consumer = first_packet;
  const uint16_t first_delay_cycles = accumulator.fallback_match.cycle_count;
  loom_amdgpu_wait_state_delay_alu_accumulate_operands(builder, second_packet,
                                                       &accumulator);
  if (accumulator.fallback_match.cycle_count > first_delay_cycles) {
    delay_consumer = second_packet;
  }
  loom_amdgpu_wait_state_match_t delay_match = {0};
  const loom_amdgpu_wait_state_action_t delay_action =
      loom_amdgpu_wait_state_delay_alu_accumulator_action(&accumulator,
                                                          &delay_match);
  if (delay_action != LOOM_AMDGPU_WAIT_STATE_ACTION_UNKNOWN) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_append(
        builder, first_packet, delay_consumer, &delay_match, delay_action));
  }

  loom_amdgpu_wait_state_record_vector_alu_results(builder, first_packet,
                                                   &first_info);
  loom_amdgpu_wait_state_record_vector_alu_results(builder, second_packet,
                                                   &second_info);
  loom_amdgpu_wait_state_delay_alu_record_results(
      builder, first_packet, first_info.delay_alu_type,
      first_info.delay_alu_latency_cycles);
  loom_amdgpu_wait_state_delay_alu_record_results(
      builder, second_packet, second_info.delay_alu_type,
      second_info.delay_alu_latency_cycles);
  loom_amdgpu_wait_state_delay_alu_advance_counters(
      builder, LOOM_AMDGPU_DELAY_ALU_TYPE_VALU, 1);
  ++builder->current_position;

  builder->packet_instruction_counts[first_packet->packet_index] = 1;
  builder->packet_instruction_counts[second_packet->packet_index] = 0;
  if (builder->matrix_coexecution != NULL) {
    loom_amdgpu_matrix_coexecution_commit_vopd_pair(
        builder->matrix_coexecution);
  }
  ++builder->progress_event_count;
  return iree_ok_status();
}

static void loom_amdgpu_wait_state_progress_query(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_view_t* packet,
    loom_low_packet_progress_emit_fn_t emit, void* emit_user_data) {
  (void)schedule;
  (void)allocation;
  const loom_amdgpu_wait_state_builder_t* builder =
      (const loom_amdgpu_wait_state_builder_t*)user_data;
  const uint32_t instruction_count =
      builder->packet_instruction_counts[packet->packet_index];
  if (instruction_count != 0) {
    const loom_low_packet_progress_event_t event = {
        .progress_class_id =
            LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_INSTRUCTION_SLOT,
        .progress_class_name = loom_amdgpu_wait_state_progress_class_name(
            LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_INSTRUCTION_SLOT),
        .action = LOOM_LOW_PACKET_PROGRESS_ACTION_ADVANCE,
        .units = instruction_count,
    };
    emit(emit_user_data, &event);
  }
}

static iree_status_t loom_amdgpu_wait_state_build_progress(
    loom_amdgpu_wait_state_builder_t* builder) {
  const loom_low_packet_progress_provider_t provider = {
      .user_data = builder,
      .event_count = builder->progress_event_count,
      .query = loom_amdgpu_wait_state_progress_query,
  };
  return loom_low_packet_progress_build(builder->schedule, builder->allocation,
                                        &provider, builder->arena,
                                        &builder->progress);
}

static bool loom_amdgpu_wait_state_matches_packet(
    const loom_amdgpu_wait_state_t* wait_state,
    const loom_low_packet_view_t* packet) {
  return wait_state->block_index == packet->node->block_index &&
         wait_state->scheduled_ordinal == packet->node->scheduled_ordinal &&
         wait_state->node_index == packet->node_index;
}

static void loom_amdgpu_wait_state_hazard_query(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_table_t* progress,
    const loom_low_packet_view_t* packet,
    loom_low_packet_hazard_plan_emit_fn_t emit, void* emit_user_data) {
  (void)schedule;
  (void)allocation;
  (void)progress;
  const loom_amdgpu_wait_state_builder_t* builder =
      (const loom_amdgpu_wait_state_builder_t*)user_data;
  for (iree_host_size_t i = 0; i < builder->state_count; ++i) {
    const loom_amdgpu_wait_state_t* wait_state = &builder->states[i];
    if (!loom_amdgpu_wait_state_matches_packet(wait_state, packet)) {
      continue;
    }
    const uint32_t progress_class_id =
        loom_amdgpu_wait_state_progress_class_id(wait_state);
    const loom_low_packet_hazard_plan_event_t event = {
        .kind = LOOM_LOW_PACKET_HAZARD_PLAN_RECORD_ACTION,
        .action_id = (uint16_t)wait_state->action,
        .action_name = loom_amdgpu_wait_state_action_name(wait_state->action),
        .reason_id = (uint16_t)wait_state->reason,
        .reason_name = loom_amdgpu_wait_state_reason_name(wait_state->reason),
        .producer_node_index = wait_state->producer_node,
        .progress_class_id = progress_class_id,
        .progress_class_name =
            loom_amdgpu_wait_state_progress_class_name(progress_class_id),
        .required_progress = wait_state->required_cycle_count,
        .observed_progress = wait_state->observed_cycle_count,
        .residual_progress = wait_state->cycle_count,
    };
    emit(emit_user_data, &event);
  }
}

static iree_status_t loom_amdgpu_wait_state_build_hazard_plan(
    loom_amdgpu_wait_state_builder_t* builder) {
  const loom_low_packet_hazard_plan_provider_t provider = {
      .user_data = builder,
      .event_count = builder->state_count,
      .query = loom_amdgpu_wait_state_hazard_query,
  };
  return loom_low_packet_hazard_plan_build(
      builder->schedule, builder->allocation, &builder->progress, &provider,
      builder->arena, &builder->hazard_plan);
}

static iree_status_t loom_amdgpu_wait_state_plan_build_with_scratch(
    loom_amdgpu_wait_state_builder_t* builder) {
  builder->processor_scheduling =
      builder->processor_properties != NULL
          ? builder->processor_properties->features.scheduling
          : 0;
  builder->has_delay_alu = loom_amdgpu_wait_state_target_has_delay_alu(builder);
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_allocate(builder));
  builder->delay_alu_epoch = 1;
  for (iree_host_size_t block_index = 0;
       block_index < builder->schedule->block_count; ++block_index) {
    if (block_index > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU wait-state block index exceeds uint32_t");
    }
    if (builder->vgpr_count != 0) {
      memset(builder->vgprs, 0, builder->vgpr_count * sizeof(*builder->vgprs));
    }
    if (builder->sgpr_count != 0) {
      memset(builder->sgprs, 0, builder->sgpr_count * sizeof(*builder->sgprs));
    }
    builder->current_position = 0;
    builder->delay_alu_valu_count = 0;
    builder->delay_alu_trans_count = 0;
    if (builder->matrix_coexecution != NULL) {
      loom_amdgpu_matrix_coexecution_begin_block(builder->matrix_coexecution,
                                                 (uint16_t)block_index);
    }
    const loom_low_schedule_block_t* block =
        &builder->schedule->blocks[block_index];
    for (uint32_t scheduled_ordinal = 0;
         scheduled_ordinal < block->scheduled_node_count; ++scheduled_ordinal) {
      const loom_low_packet_view_t packet = loom_low_packet_at_block_ordinal(
          builder->schedule, (uint32_t)block_index, scheduled_ordinal);
      const loom_amdgpu_vopd_packet_role_t vopd_role =
          builder->vopd_packets != NULL
              ? builder->vopd_packets[packet.packet_index].role
              : LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE;
      if (vopd_role == LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND) {
        builder->packet_instruction_counts[packet.packet_index] = 0;
        continue;
      }
      if (vopd_role == LOOM_AMDGPU_VOPD_PACKET_ROLE_FIRST) {
        const uint32_t pair_index =
            builder->vopd_packets[packet.packet_index].pair_index;
        const loom_amdgpu_vopd_pair_t* pair = &builder->vopd_pairs[pair_index];
        const loom_low_packet_view_t second_packet =
            loom_low_packet_at(builder->schedule, pair->second_packet_index);
        IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_apply_vopd_pair(
            builder, &packet, &second_packet));
        continue;
      }
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_wait_state_apply_packet(builder, &packet));
    }
    if (builder->matrix_coexecution != NULL) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_coexecution_end_block(
          builder->matrix_coexecution));
    }
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_build_progress(builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_build_hazard_plan(builder));
  return iree_ok_status();
}

iree_status_t loom_amdgpu_wait_state_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_processor_properties_t* processor_properties,
    const struct loom_amdgpu_vopd_plan_t* vopd_plan,
    loom_amdgpu_matrix_coexecution_t* matrix_coexecution,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* transient_arena,
    loom_amdgpu_wait_state_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_wait_state_plan_t){0};

  loom_low_allocation_value_scratch_t scratch = {0};
  iree_status_t status =
      loom_low_allocation_acquire_value_scratch(allocation, &scratch);
  const bool has_vopd_pairs = vopd_plan != NULL && vopd_plan->pair_count != 0;
  loom_amdgpu_wait_state_builder_t builder = {
      .schedule = schedule,
      .allocation = allocation,
      .processor_properties = processor_properties,
      .vopd_packets = has_vopd_pairs ? vopd_plan->packets : NULL,
      .vopd_pairs = has_vopd_pairs ? vopd_plan->pairs : NULL,
      .descriptor_set = schedule->target.descriptor_set,
      .arena = arena,
      .transient_arena = transient_arena,
      .matrix_coexecution = matrix_coexecution,
  };
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_wait_state_plan_build_with_scratch(&builder);
  }
  if (iree_status_is_ok(status)) {
    *out_plan = (loom_amdgpu_wait_state_plan_t){
        .schedule = schedule,
        .allocation = allocation,
        .progress = builder.progress,
        .hazard_plan = builder.hazard_plan,
        .states = builder.states,
        .state_count = builder.state_count,
    };
    out_plan->hazard_plan.progress = &out_plan->progress;
  }
  loom_low_allocation_release_value_scratch(&scratch);
  return status;
}

static iree_status_t loom_amdgpu_wait_state_write_states_json(
    const loom_amdgpu_wait_state_plan_t* plan, loom_output_stream_t* stream) {
  loom_json_array_writer_t states;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &states));
  for (iree_host_size_t i = 0; i < plan->state_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&states));
    const loom_amdgpu_wait_state_t* state = &plan->states[i];
    loom_json_object_writer_t state_object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &state_object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &state_object, IREE_SV("index"), i));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("reason"), (uint32_t)state->reason));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &state_object, IREE_SV("reason_name"),
        loom_amdgpu_wait_state_reason_name(state->reason)));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("action"), (uint32_t)state->action));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &state_object, IREE_SV("action_name"),
        loom_amdgpu_wait_state_action_name(state->action)));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("block"), state->block_index));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("node"), state->node_index));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("scheduled_ordinal"), state->scheduled_ordinal));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("producer_node"), state->producer_node));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("consumer_node"), state->consumer_node));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("required"), state->required_cycle_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("observed"), state->observed_cycle_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("residual"), state->cycle_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("delay_alu_immediate"),
        state->delay_alu_immediate));
    if (state->matrix_result_use !=
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_UNKNOWN) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &state_object, IREE_SV("matrix_wait_profile"),
          (uint32_t)state->matrix_wait_profile));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &state_object, IREE_SV("matrix_wait_profile_name"),
          loom_amdgpu_matrix_wait_profile_name(state->matrix_wait_profile)));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &state_object, IREE_SV("matrix_result_use"),
          (uint32_t)state->matrix_result_use));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &state_object, IREE_SV("matrix_result_use_name"),
          loom_amdgpu_matrix_wait_result_use_name(state->matrix_result_use)));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &state_object, IREE_SV("matrix_pass_count"),
          state->matrix_pass_count));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&state_object));
  }
  return loom_json_array_end(&states);
}

iree_status_t loom_amdgpu_wait_state_plan_format_text(
    const loom_amdgpu_wait_state_plan_t* plan, iree_string_builder_t* builder) {
  if (plan == NULL || builder == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU wait-state plan and builder are required");
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "amdgpu.wait_state_plan states=%" PRIhsz " progress=%" PRIhsz
      " hazards=%" PRIhsz "\n",
      plan->state_count, plan->progress.record_count,
      plan->hazard_plan.record_count));
  for (iree_host_size_t i = 0; i < plan->state_count; ++i) {
    const loom_amdgpu_wait_state_t* state = &plan->states[i];
    const iree_string_view_t reason_name =
        loom_amdgpu_wait_state_reason_name(state->reason);
    const iree_string_view_t action_name =
        loom_amdgpu_wait_state_action_name(state->action);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "state[%" PRIhsz "] reason=%.*s action=%.*s at=b%" PRIu32 ":n%" PRIu32
        "/o%" PRIu32 " producer=n%" PRIu32 " consumer=n%" PRIu32
        " required=%" PRIu16 " observed=%" PRIu16 " residual=%" PRIu16,
        i, (int)reason_name.size, reason_name.data, (int)action_name.size,
        action_name.data, state->block_index, state->node_index,
        state->scheduled_ordinal, state->producer_node, state->consumer_node,
        state->required_cycle_count, state->observed_cycle_count,
        state->cycle_count));
    if (state->action == LOOM_AMDGPU_WAIT_STATE_ACTION_S_DELAY_ALU) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " delay_alu=0x%04" PRIx16, state->delay_alu_immediate));
    }
    if (state->matrix_result_use !=
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_UNKNOWN) {
      const iree_string_view_t profile_name =
          loom_amdgpu_matrix_wait_profile_name(state->matrix_wait_profile);
      const iree_string_view_t result_use_name =
          loom_amdgpu_matrix_wait_result_use_name(state->matrix_result_use);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " matrix=%.*s/%.*s/pass%" PRIu16, (int)profile_name.size,
          profile_name.data, (int)result_use_name.size, result_use_name.data,
          state->matrix_pass_count));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_wait_state_plan_format_json(
    const loom_amdgpu_wait_state_plan_t* plan, iree_string_builder_t* builder) {
  if (plan == NULL || builder == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU wait-state plan and builder are required");
  }
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("format"), IREE_SV("loom.amdgpu.wait_state_plan.v1")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("state_count"), plan->state_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("progress_count"), plan->progress.record_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("hazard_count"), plan->hazard_plan.record_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("states")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_write_states_json(plan, &stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("progress")));
  IREE_RETURN_IF_ERROR(
      loom_low_packet_progress_write_json_array(&plan->progress, &stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("hazards")));
  IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_write_json_array(
      &plan->hazard_plan, &stream));
  return loom_json_object_end(&object);
}

uint64_t loom_amdgpu_wait_state_plan_instruction_count(
    const loom_amdgpu_wait_state_plan_t* plan) {
  if (plan == NULL) {
    return 0;
  }
  uint64_t instruction_count = 0;
  for (iree_host_size_t i = 0; i < plan->state_count; ++i) {
    const loom_amdgpu_wait_state_t* state = &plan->states[i];
    switch (state->action) {
      case LOOM_AMDGPU_WAIT_STATE_ACTION_S_NOP:
        instruction_count += (state->cycle_count +
                              LOOM_AMDGPU_WAIT_STATE_MAX_S_NOP_CYCLES - 1u) /
                             LOOM_AMDGPU_WAIT_STATE_MAX_S_NOP_CYCLES;
        break;
      case LOOM_AMDGPU_WAIT_STATE_ACTION_S_DELAY_ALU:
        instruction_count += 1;
        break;
      case LOOM_AMDGPU_WAIT_STATE_ACTION_V_NOP:
        instruction_count += state->cycle_count;
        break;
      case LOOM_AMDGPU_WAIT_STATE_ACTION_UNKNOWN:
      default:
        IREE_ASSERT(false && "unsupported AMDGPU wait-state action");
        break;
    }
  }
  return instruction_count;
}
