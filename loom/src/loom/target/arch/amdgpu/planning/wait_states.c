// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_states.h"

#include <string.h>

#include "loom/codegen/low/packet.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/target/arch/amdgpu/encoding/encoding.h"
#include "loom/target/arch/amdgpu/planning/address_state.h"
#include "loom/target/arch/amdgpu/planning/delay_alu.h"
#include "loom/target/arch/amdgpu/planning/descriptor_semantics.h"
#include "loom/target/arch/amdgpu/planning/mask_write_wait.h"
#include "loom/target/arch/amdgpu/planning/matrix_coexecution.h"
#include "loom/target/arch/amdgpu/planning/matrix_result_wait.h"
#include "loom/target/arch/amdgpu/planning/matrix_wait_states.h"
#include "loom/target/arch/amdgpu/planning/replay_clause.h"
#include "loom/target/arch/amdgpu/planning/sgpr_read_wait.h"
#include "loom/target/arch/amdgpu/planning/short_wait.h"
#include "loom/target/arch/amdgpu/planning/structural_packet.h"
#include "loom/target/arch/amdgpu/planning/vopd_plan.h"
#include "loom/target/arch/amdgpu/planning/wait_packets.h"
#include "loom/target/arch/amdgpu/planning/wait_plan.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info.h"
#include "loom/util/segmented_storage.h"

// Target payload size for lazily appended wait-state segments. Segments stay
// small enough to share normal compiler workspace blocks while avoiding a
// scheduled-node-sized allocation when only a sparse set of waits is emitted.
#define LOOM_AMDGPU_WAIT_STATE_SEGMENT_BYTE_LENGTH (4u * 1024u)

// Number of wait-state rows stored in each append segment.
#define LOOM_AMDGPU_WAIT_STATES_PER_SEGMENT     \
  (LOOM_AMDGPU_WAIT_STATE_SEGMENT_BYTE_LENGTH / \
   sizeof(loom_amdgpu_wait_state_t))

static_assert(LOOM_AMDGPU_WAIT_STATES_PER_SEGMENT > 0,
              "wait state must fit in one append segment");

// One stable segment of wait states populated before exact finalization.
typedef struct loom_amdgpu_wait_state_segment_t {
  // Wait states in append order.
  loom_amdgpu_wait_state_t states[LOOM_AMDGPU_WAIT_STATES_PER_SEGMENT];
} loom_amdgpu_wait_state_segment_t;

static_assert(sizeof(loom_amdgpu_wait_state_segment_t) <=
                  LOOM_AMDGPU_WAIT_STATE_SEGMENT_BYTE_LENGTH,
              "wait-state segment exceeds its byte budget");

// Sparse append stream for one ordered class of wait states.
typedef struct loom_amdgpu_wait_state_stream_t {
  // Stable segments populated in wait-state order.
  loom_segmented_storage_t segments;
  // Current append segment, or NULL before the first wait state.
  loom_amdgpu_wait_state_segment_t* tail;
  // Number of populated wait states in |tail|.
  iree_host_size_t tail_count;
  // Total number of populated wait states.
  iree_host_size_t state_count;
} loom_amdgpu_wait_state_stream_t;

enum {
  LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_INSTRUCTION_SLOT = 1,
  LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_DELAY_ALU_DEPENDENCY = 2,
  LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_VECTOR_ISSUE = 3,
  LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_DEPCTR_DESTINATION = 4,
};

typedef enum loom_amdgpu_wait_state_reason_flag_bits_e {
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_TO_MATRIX_USE =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE,
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_TRANS_RESULT_USE =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_TRANS_RESULT_USE,
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_SGPR_READ =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_VALU_SGPR_READ,
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_LANE_PERMUTE_VGPR_READ =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_LANE_PERMUTE_VGPR_READ,
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_READFIRSTLANE_VGPR_READ =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_READFIRSTLANE_VGPR_READ,
  LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_DST_SEL_FORWARDING_USE =
      1u << LOOM_AMDGPU_WAIT_STATE_REASON_DST_SEL_FORWARDING_USE,
} loom_amdgpu_wait_state_reason_flag_bits_t;
typedef uint32_t loom_amdgpu_wait_state_reason_flags_t;

typedef struct loom_amdgpu_wait_state_vgpr_t {
  // Recent ALU producer state for S_DELAY_ALU dependency insertion.
  loom_amdgpu_delay_alu_info_t delay_alu;
} loom_amdgpu_wait_state_vgpr_t;

typedef struct loom_amdgpu_wait_state_sgpr_t {
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
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_TRANS_PRODUCER = 1u << 3,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_TRANS_FORWARDING_CONSUMER = 1u << 4,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_USES_VECTOR_ALU = 1u << 5,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_USES_VECTOR_MEMORY = 1u << 6,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_LANE_PERMUTE_CONSUMER = 1u << 7,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_READFIRSTLANE_CONSUMER = 1u << 8,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DST_SEL_FORWARDING_PRODUCER = 1u << 9,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DST_SEL_FORWARDING_CONSUMER = 1u << 10,
  LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_VALU_SGPR_READ_CONSUMER = 1u << 11,
} loom_amdgpu_wait_state_packet_flag_bits_t;
typedef uint32_t loom_amdgpu_wait_state_packet_flags_t;

typedef struct loom_amdgpu_wait_state_packet_info_t {
  // Compact packet classification flags.
  loom_amdgpu_wait_state_packet_flags_t flags;
  // Generated descriptor traits retained for final physical hazard queries.
  loom_amdgpu_descriptor_traits_t descriptor_traits;
  // Matrix source facts retained by the counter-wait traversal.
  const loom_amdgpu_matrix_result_source_t* matrix_source;
  // Structural packet classification for non-descriptor low packets.
  loom_amdgpu_structural_packet_info_t structural;
  // Number of target instructions represented by this packet.
  uint64_t instruction_count;
  // Delay-ALU producer class for this packet.
  loom_amdgpu_delay_alu_type_t delay_alu_type;
  // Modeled latency used when recording delay-ALU producers.
  uint16_t delay_alu_latency_cycles;
} loom_amdgpu_wait_state_packet_info_t;

typedef struct loom_amdgpu_wait_state_builder_t {
  // Schedule table being analyzed.
  const loom_low_schedule_table_t* schedule;
  // Allocation table being analyzed.
  const loom_low_allocation_table_t* allocation;
  // Canonical authored-wait elisions, which supply no instruction progress.
  const loom_amdgpu_wait_plan_t* wait_plan;
  // Concrete counter waits supplying store-source issue progress.
  const loom_amdgpu_wait_packet_plan_t* wait_packets;
  // Concrete address transitions separating replayable memory clauses.
  const loom_amdgpu_address_state_plan_t* address_state;
  // Physical register retention for the current native memory clause.
  loom_amdgpu_replay_clause_t* replay_clause;
  // Mandatory short physical dependencies, or NULL on unaffected targets.
  loom_amdgpu_short_wait_t* short_wait;
  // Final VOPD memberships indexed by scheduled packet.
  const loom_amdgpu_vopd_packet_t* vopd_packets;
  // Final VOPD pairs referenced by |vopd_packets|.
  const loom_amdgpu_vopd_pair_t* vopd_pairs;
  // Descriptor set selected by the scheduled target.
  const loom_low_descriptor_set_t* descriptor_set;
  // Machine schedule model selected by the scheduled target.
  const loom_low_schedule_model_t* schedule_model;
  // Arena that owns output tables retained by the completed plan.
  iree_arena_allocator_t* arena;
  // Arena that owns builder state discarded after plan construction.
  iree_arena_allocator_t* transient_arena;
  // Processor properties established by target-record resolution.
  const loom_amdgpu_processor_properties_t* processor_properties;
  // Pending result windows surviving physical overwrites and block edges.
  loom_amdgpu_matrix_result_wait_t* matrix_results;
  // Native matrix wait contract selected by the resolved processor.
  loom_amdgpu_matrix_wait_profile_t matrix_wait_profile;
  // Cached processor scheduling and fixed-wait hazard feature bits.
  loom_amdgpu_processor_scheduling_bits_t processor_scheduling;
  // True when the selected processor and descriptor set can emit S_DELAY_ALU.
  bool has_delay_alu;
  // Transient matrix/vector coexecution frontier, or NULL when not selected.
  loom_amdgpu_matrix_coexecution_t* matrix_coexecution;
  // GFX11 wave64 mask reuse summaries, or NULL on unaffected targets.
  loom_amdgpu_mask_write_wait_t* mask_writes;
  // GFX12 scalar-pair read history and destination dependencies.
  loom_amdgpu_sgpr_read_wait_t* sgpr_reads;
  // Optional per-physical-VGPR delay-ALU producer state.
  loom_amdgpu_wait_state_vgpr_t* vgprs;
  // Number of entries in |vgprs|.
  iree_host_size_t vgpr_count;
  // Hardware allocation floor required by physical shift-count operands.
  uint32_t minimum_vgpr_count;
  // Optional delay-ALU producers across allocated and ABI-fixed SGPRs.
  struct {
    // Dense allocated prefix followed by the descriptor's fixed-location range.
    loom_amdgpu_wait_state_sgpr_t* states;
    // Total number of state rows, including ABI-fixed locations.
    iree_host_size_t count;
    // Number of state rows in the dense allocated prefix.
    uint32_t allocated_count;
    // First physical location represented by the compact fixed-location suffix.
    uint32_t fixed_location_base;
  } sgprs;
  // Recent ALU producer state for the physical SCC condition register.
  loom_amdgpu_delay_alu_info_t scc_delay_alu;
  // Ordinary pre-packet waits discovered by the final-packet traversal.
  loom_amdgpu_wait_state_stream_t ordinary_states;
  // Residual waits discovered after physical-hazard CFG propagation.
  loom_amdgpu_wait_state_stream_t resolved_states;
  // Final contiguous output wait-state rows.
  loom_amdgpu_wait_state_t* states;
  // Number of populated output wait-state rows.
  iree_host_size_t state_count;
  // Number of packets that emit native instruction-slot progress.
  iree_host_size_t progress_event_count;
  // Native instruction count indexed by scheduled packet.
  uint32_t* packet_instruction_counts;
  // Target progress facts for scheduled native instruction slots.
  loom_low_packet_progress_table_t progress;
  // Common residual hazard records for emitted wait states.
  loom_low_packet_hazard_plan_t hazard_plan;
  // Current ordinary instruction position in the active block.
  uint64_t current_position;
  // Recent native ALU producer identities and completed dependency selectors.
  loom_amdgpu_delay_alu_state_t delay_alu;
} loom_amdgpu_wait_state_builder_t;

static const iree_string_view_t kAmdgpuWaitStateReasonNames[] = {
    [LOOM_AMDGPU_WAIT_STATE_REASON_UNKNOWN] = IREE_SVL("unknown"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_MASK_WRITE] = IREE_SVL("mask_write"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_SGPR_READ_DEPENDENCY] =
        IREE_SVL("sgpr_read_dependency"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_DIV_FMAS_CONDITION] =
        IREE_SVL("div_fmas_condition"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_M0_READ] = IREE_SVL("m0_read"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_DOT_RESULT_USE] = IREE_SVL("dot_result_use"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_DGEMM_PIPELINE] = IREE_SVL("dgemm_pipeline"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_DGEMM_MEMORY] = IREE_SVL("dgemm_memory"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_XNACK_REPLAY_CLAUSE] =
        IREE_SVL("xnack_replay_clause"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_STORE_DATA_REUSE] =
        IREE_SVL("store_data_reuse"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_RESULT_USE] =
        IREE_SVL("matrix_result_use"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_VALU_TO_MATRIX_USE] =
        IREE_SVL("valu_to_matrix_use"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_TRANS_RESULT_USE] =
        IREE_SVL("trans_result_use"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_VALU_SGPR_READ] = IREE_SVL("valu_sgpr_read"),
    [LOOM_AMDGPU_WAIT_STATE_REASON_LANE_PERMUTE_VGPR_READ] =
        IREE_SVL("lane_permute_vgpr_read"),
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
    [LOOM_AMDGPU_WAIT_STATE_ACTION_S_WAITCNT_DEPCTR] =
        IREE_SVL("amdgpu.s_waitcnt_depctr"),
    [LOOM_AMDGPU_WAIT_STATE_ACTION_S_WAIT_ALU] = IREE_SVL("amdgpu.s_wait_alu"),
};

static iree_string_view_t loom_amdgpu_wait_state_progress_class_name(
    uint16_t progress_class_id) {
  switch (progress_class_id) {
    case LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_INSTRUCTION_SLOT:
      return IREE_SV("amdgpu.instruction_slot");
    case LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_DELAY_ALU_DEPENDENCY:
      return IREE_SV("amdgpu.delay_alu_dependency");
    case LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_VECTOR_ISSUE:
      return IREE_SV("amdgpu.vector_issue");
    case LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_DEPCTR_DESTINATION:
      return IREE_SV("amdgpu.depctr_destination");
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
  if (wait_state->action == LOOM_AMDGPU_WAIT_STATE_ACTION_S_WAITCNT_DEPCTR ||
      wait_state->action == LOOM_AMDGPU_WAIT_STATE_ACTION_S_WAIT_ALU) {
    return LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_DEPCTR_DESTINATION;
  }
  return LOOM_AMDGPU_WAIT_STATE_PROGRESS_CLASS_INSTRUCTION_SLOT;
}

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

static bool loom_amdgpu_wait_state_reason_is_valid(
    loom_amdgpu_wait_state_reason_t reason) {
  return reason > LOOM_AMDGPU_WAIT_STATE_REASON_UNKNOWN &&
         reason < LOOM_AMDGPU_WAIT_STATE_REASON_COUNT_;
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

// Assignments and finalized moves stay wholly within the allocated prefix or
// the descriptor's fixed range. ABI-fixed locations do not consume allocatable
// register resources, but still participate in physical hazards.
static loom_amdgpu_wait_state_sgpr_t* loom_amdgpu_wait_state_sgpr_at(
    const loom_amdgpu_wait_state_builder_t* builder, uint32_t location) {
  const uint32_t index = location < builder->sgprs.allocated_count
                             ? location
                             : builder->sgprs.allocated_count + location -
                                   builder->sgprs.fixed_location_base;
  return &builder->sgprs.states[index];
}

static loom_amdgpu_delay_alu_info_t* loom_amdgpu_wait_state_delay_alu_move_slot(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_move_location_t* location) {
  if (location->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
    return &builder->vgprs[location->location].delay_alu;
  }
  if (location->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
    return &loom_amdgpu_wait_state_sgpr_at(builder, location->location)
                ->delay_alu;
  }
  return NULL;
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
  if (!builder->has_delay_alu) {
    return iree_ok_status();
  }
  const iree_host_size_t vgpr_count =
      allocation->physical_extents
          .ends_by_reg_class[LOOM_AMDGPU_REG_CLASS_ID_VGPR];
  if (vgpr_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->transient_arena, vgpr_count, sizeof(*builder->vgprs),
        (void**)&builder->vgprs));
    builder->vgpr_count = vgpr_count;
  }
  const loom_low_reg_class_t* sgpr_class =
      &builder->descriptor_set->reg_classes[LOOM_AMDGPU_REG_CLASS_ID_SGPR];
  builder->sgprs.allocated_count =
      allocation->physical_extents
          .ends_by_reg_class[LOOM_AMDGPU_REG_CLASS_ID_SGPR];
  builder->sgprs.fixed_location_base =
      loom_low_reg_class_fixed_location_base(sgpr_class);
  builder->sgprs.count = builder->sgprs.allocated_count +
                         loom_low_reg_class_fixed_location_count(sgpr_class);
  if (builder->sgprs.count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->transient_arena, builder->sgprs.count,
        sizeof(*builder->sgprs.states), (void**)&builder->sgprs.states));
  }
  return iree_ok_status();
}

static bool loom_amdgpu_wait_state_has_scheduling(
    const loom_amdgpu_wait_state_builder_t* builder,
    loom_amdgpu_processor_scheduling_bits_t bits) {
  return iree_any_bit_set(builder->processor_scheduling, bits);
}

static bool loom_amdgpu_wait_state_target_has_delay_alu(
    const loom_amdgpu_wait_state_builder_t* builder) {
  if (!loom_amdgpu_wait_state_has_scheduling(
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
  if (descriptor_set == NULL || descriptor == NULL) {
    return 0;
  }
  const loom_low_descriptor_view_t* descriptor_view =
      loom_low_descriptor_set_descriptor_view(descriptor_set, descriptor);
  if (descriptor_view->schedule_class_id == LOOM_LOW_SCHEDULE_CLASS_NONE) {
    return 0;
  }
  const uint16_t schedule_class_id = descriptor_view->schedule_class_id;
  if (schedule_class_id >= descriptor_set->schedule_class_count) {
    return 0;
  }
  const loom_low_schedule_class_t* schedule_class =
      loom_low_schedule_model_schedule_class(builder->schedule_model,
                                             descriptor_view);
  return schedule_class->latency_cycles;
}

static bool
loom_amdgpu_wait_state_packet_has_destination_selection_forwarding_hazard(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_descriptor_traits_t descriptor_traits) {
  if (iree_any_bit_set(
          descriptor_traits,
          LOOM_AMDGPU_DESCRIPTOR_TRAIT_DESTINATION_SELECTION_FORWARDING)) {
    return true;
  }
  if (!iree_any_bit_set(descriptor_traits, LOOM_AMDGPU_DESCRIPTOR_TRAIT_SDWA)) {
    return false;
  }
  const loom_low_descriptor_set_t* descriptor_set = builder->descriptor_set;
  const loom_amdgpu_descriptor_immediate_slots_t slots =
      loom_amdgpu_descriptor_immediate_slots(descriptor_set,
                                             packet->descriptor);
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[loom_low_descriptor_layout(descriptor_set,
                                                             packet->descriptor)
                                      ->immediate_start +
                                  slots.sdwa_dst_sel];
  const loom_attribute_t attr =
      loom_low_packet_immediate_attr(packet, immediate);
  const int64_t selection =
      attr.kind == LOOM_ATTR_ABSENT ? immediate->default_value : attr.i64;
  return loom_amdgpu_sdwa_dst_selector_writes_subdword((uint32_t)selection);
}

static void loom_amdgpu_wait_state_merge_short_match(
    const loom_amdgpu_short_wait_match_t* candidate,
    loom_amdgpu_wait_state_match_t* match) {
  if (candidate->cycles > match->cycle_count) {
    *match = (loom_amdgpu_wait_state_match_t){
        .reason = candidate->reason,
        .producer_node = candidate->producer_node,
        .required_cycle_count = candidate->required_cycles,
        .observed_cycle_count = candidate->observed_cycles,
        .cycle_count = candidate->cycles,
    };
  }
}

static void loom_amdgpu_wait_state_match_assignment(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment,
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons,
    loom_amdgpu_wait_state_match_t* match) {
  if (builder->short_wait == NULL || assignment == NULL ||
      assignment->location_kind !=
          LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
    return;
  }
  const loom_amdgpu_short_wait_match_t candidate = loom_amdgpu_short_wait_query(
      builder->short_wait, assignment->descriptor_reg_class_id,
      assignment->location_base, assignment->location_count, 0,
      (loom_amdgpu_short_wait_reasons_t)allowed_reasons);
  loom_amdgpu_wait_state_merge_short_match(&candidate, match);
}

static loom_amdgpu_wait_state_action_t
loom_amdgpu_wait_state_delay_alu_accumulator_action(
    const loom_amdgpu_delay_alu_accumulator_t* accumulator,
    loom_amdgpu_wait_state_match_t* match) {
  const loom_amdgpu_delay_alu_match_t delay =
      loom_amdgpu_delay_alu_select(accumulator);
  if (delay.cycle_count == 0) {
    return LOOM_AMDGPU_WAIT_STATE_ACTION_UNKNOWN;
  }
  *match = (loom_amdgpu_wait_state_match_t){
      .reason = LOOM_AMDGPU_WAIT_STATE_REASON_DELAY_ALU_DEPENDENCY,
      .producer_node = delay.producer_node,
      .required_cycle_count = delay.required_cycle_count,
      .observed_cycle_count = delay.observed_cycle_count,
      .cycle_count = delay.cycle_count,
      .delay_alu_immediate = delay.delay_alu_immediate,
  };
  return delay.delay_alu_immediate != 0
             ? LOOM_AMDGPU_WAIT_STATE_ACTION_S_DELAY_ALU
             : LOOM_AMDGPU_WAIT_STATE_ACTION_S_NOP;
}

static void loom_amdgpu_wait_state_delay_alu_clear_all(
    loom_amdgpu_wait_state_builder_t* builder) {
  if (!loom_amdgpu_delay_alu_reset(&builder->delay_alu)) {
    return;
  }
  for (iree_host_size_t i = 0; i < builder->vgpr_count; ++i) {
    builder->vgprs[i].delay_alu = (loom_amdgpu_delay_alu_info_t){0};
  }
  for (iree_host_size_t i = 0; i < builder->sgprs.count; ++i) {
    builder->sgprs.states[i].delay_alu = (loom_amdgpu_delay_alu_info_t){0};
  }
  builder->scc_delay_alu = (loom_amdgpu_delay_alu_info_t){0};
}

static void loom_amdgpu_wait_state_delay_alu_match_assignment(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_allocation_assignment_t* assignment,
    loom_amdgpu_delay_alu_accumulator_t* accumulator) {
  if (assignment == NULL) {
    return;
  }
  if (loom_amdgpu_wait_state_assignment_is_physical_scc(assignment)) {
    loom_amdgpu_delay_alu_accumulate_info(&builder->delay_alu,
                                          builder->current_position,
                                          &builder->scc_delay_alu, accumulator);
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
      loom_amdgpu_delay_alu_accumulate_info(
          &builder->delay_alu, builder->current_position, slot, accumulator);
      *slot = (loom_amdgpu_delay_alu_info_t){0};
    }
    return;
  }
  if (loom_amdgpu_wait_state_assignment_is_physical_sgpr(assignment)) {
    loom_amdgpu_wait_state_sgpr_t* states =
        loom_amdgpu_wait_state_sgpr_at(builder, assignment->location_base);
    for (uint32_t i = 0; i < assignment->location_count; ++i) {
      loom_amdgpu_delay_alu_info_t* slot = &states[i].delay_alu;
      loom_amdgpu_delay_alu_accumulate_info(
          &builder->delay_alu, builder->current_position, slot, accumulator);
      *slot = (loom_amdgpu_delay_alu_info_t){0};
    }
  }
}

static void loom_amdgpu_wait_state_delay_alu_accumulate_operands(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_delay_alu_accumulator_t* accumulator) {
  for (uint16_t i = 0; i < packet->node->operand_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_packet_operand_assignment(builder->allocation, packet, i);
    loom_amdgpu_wait_state_delay_alu_match_assignment(builder, assignment,
                                                      accumulator);
  }
}

static loom_amdgpu_wait_state_action_t
loom_amdgpu_wait_state_delay_alu_match_packet(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_wait_state_packet_info_t* info,
    loom_amdgpu_wait_state_match_t* match) {
  if (!builder->has_delay_alu) {
    return LOOM_AMDGPU_WAIT_STATE_ACTION_UNKNOWN;
  }
  loom_amdgpu_delay_alu_accumulator_t accumulator = {0};
  if (iree_any_bit_set(info->flags,
                       LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DESCRIPTOR)) {
    loom_amdgpu_wait_state_delay_alu_accumulate_operands(builder, packet,
                                                         &accumulator);
  } else {
    for (iree_host_size_t i = 0; i < info->structural.moves.count; ++i) {
      const loom_low_move_t* move =
          &builder->allocation->moves[info->structural.moves.start + i];
      const uint32_t access_count = loom_low_move_access_count(move);
      for (uint32_t j = 0; j < access_count; ++j) {
        loom_amdgpu_delay_alu_info_t* source =
            loom_amdgpu_wait_state_delay_alu_move_slot(
                builder, loom_low_move_read(move, j));
        if (source != NULL) {
          loom_amdgpu_delay_alu_accumulate_info(&builder->delay_alu,
                                                builder->current_position,
                                                source, &accumulator);
          *source = (loom_amdgpu_delay_alu_info_t){0};
        }
      }
      // Only actual reads consume pre-group producers. An overwritten cycle
      // temporary is not an input when read later in this native move order.
      // apply_moves records the new producers after the group's input wait.
      for (uint32_t j = 0; j < access_count; ++j) {
        loom_amdgpu_delay_alu_info_t* destination =
            loom_amdgpu_wait_state_delay_alu_move_slot(
                builder, loom_low_move_write(move, j));
        if (destination != NULL) {
          *destination = (loom_amdgpu_delay_alu_info_t){0};
        }
      }
    }
    if (iree_any_bit_set(info->structural.flags,
                         LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_READS_SCC)) {
      loom_amdgpu_delay_alu_accumulate_info(
          &builder->delay_alu, builder->current_position,
          &builder->scc_delay_alu, &accumulator);
      builder->scc_delay_alu = (loom_amdgpu_delay_alu_info_t){0};
    }
  }
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
  const loom_amdgpu_delay_alu_info_t info = loom_amdgpu_delay_alu_make_info(
      &builder->delay_alu, builder->current_position, type, latency_cycles,
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
    loom_amdgpu_wait_state_sgpr_t* states =
        loom_amdgpu_wait_state_sgpr_at(builder, assignment->location_base);
    for (uint32_t i = 0; i < assignment->location_count; ++i) {
      states[i].delay_alu = info;
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
  for (uint16_t i = 0; i < packet->node->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_packet_result_assignment(builder->allocation, packet, i);
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
    loom_amdgpu_wait_state_sgpr_t* states =
        loom_amdgpu_wait_state_sgpr_at(builder, assignment->location_base);
    for (uint32_t i = 0; i < assignment->location_count; ++i) {
      states[i].delay_alu = (loom_amdgpu_delay_alu_info_t){0};
    }
  }
}

static void loom_amdgpu_wait_state_delay_alu_clear_results(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet) {
  if (!builder->has_delay_alu) {
    return;
  }
  for (uint16_t i = 0; i < packet->node->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_packet_result_assignment(builder->allocation, packet, i);
    loom_amdgpu_wait_state_delay_alu_clear_assignment(builder, assignment);
  }
}

static void loom_amdgpu_wait_state_apply_moves(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet, loom_low_move_range_t moves,
    bool processor_has_valu_sgpr_read_hazard) {
  // Native moves keep their own issue positions, including cycle scratch.
  for (iree_host_size_t i = 0; i < moves.count; ++i) {
    const loom_low_move_t* move = &builder->allocation->moves[moves.start + i];
    loom_amdgpu_delay_alu_type_t type = LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER;
    for (uint32_t j = 0; j < loom_low_move_access_count(move); ++j) {
      const loom_low_move_location_t* write = loom_low_move_write(move, j);
      const uint16_t register_class = write->descriptor_reg_class_id;
      const uint32_t location = write->location;
      if (builder->short_wait != NULL &&
          (register_class == LOOM_AMDGPU_REG_CLASS_ID_VGPR ||
           register_class ==
               loom_amdgpu_agpr_reg_class_id(builder->descriptor_set))) {
        const loom_amdgpu_short_wait_reasons_t two_slot_reasons =
            LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_TO_MATRIX_USE |
            (loom_amdgpu_short_wait_reasons_t)
                LOOM_AMDGPU_SHORT_WAIT_PERMLANE_READ |
            (register_class == LOOM_AMDGPU_REG_CLASS_ID_VGPR
                 ? LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_LANE_PERMUTE_VGPR_READ
                 : 0);
        loom_amdgpu_short_wait_record(builder->short_wait, packet->node_index,
                                      register_class, location, 1,
                                      (uint32_t)i + 1, two_slot_reasons, 2);
        if (processor_has_valu_sgpr_read_hazard) {
          loom_amdgpu_short_wait_record(
              builder->short_wait, packet->node_index, register_class, location,
              1, (uint32_t)i + 1,
              LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_READFIRSTLANE_VGPR_READ, 1);
        }
      }
      if (builder->has_delay_alu) {
        loom_amdgpu_delay_alu_info_t* destination = NULL;
        if (register_class == LOOM_AMDGPU_REG_CLASS_ID_VGPR) {
          type = LOOM_AMDGPU_DELAY_ALU_TYPE_VALU;
          destination = &builder->vgprs[location].delay_alu;
        } else if (register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR) {
          type = LOOM_AMDGPU_DELAY_ALU_TYPE_SALU;
          destination =
              &loom_amdgpu_wait_state_sgpr_at(builder, location)->delay_alu;
        }
        if (destination != NULL) {
          *destination = loom_amdgpu_delay_alu_make_info(
              &builder->delay_alu, builder->current_position, type,
              loom_amdgpu_delay_alu_latency_cycles(
                  type, type == LOOM_AMDGPU_DELAY_ALU_TYPE_SALU ? 1 : 0),
              packet->node_index);
        }
      }
    }
    loom_amdgpu_delay_alu_advance(&builder->delay_alu, type, 1);
    ++builder->current_position;
  }
}

static iree_status_t loom_amdgpu_wait_state_stream_append(
    loom_amdgpu_wait_state_stream_t* stream,
    iree_arena_allocator_t* transient_arena,
    loom_amdgpu_wait_state_t wait_state) {
  if (stream->tail_count == LOOM_AMDGPU_WAIT_STATES_PER_SEGMENT) {
    stream->tail = NULL;
    stream->tail_count = 0;
  }
  if (stream->tail == NULL) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        &stream->segments, transient_arena, &segment));
    stream->tail = (loom_amdgpu_wait_state_segment_t*)segment;
  }
  stream->tail->states[stream->tail_count++] = wait_state;
  ++stream->state_count;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_wait_state_append_resolved(
    void* user_data, const loom_amdgpu_wait_state_t* wait_state) {
  loom_amdgpu_wait_state_builder_t* builder =
      (loom_amdgpu_wait_state_builder_t*)user_data;
  return loom_amdgpu_wait_state_stream_append(
      &builder->resolved_states, builder->transient_arena, *wait_state);
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
  const loom_amdgpu_wait_state_t wait_state = {
      .reason = match->reason,
      .action = action,
      .block_index = insertion_packet->node->block_index,
      .node_index = insertion_packet->node_index,
      .scheduled_ordinal = insertion_packet->node->scheduled_ordinal,
      .producer_node = match->producer_node,
      .consumer_node = consumer_packet->node_index,
      .required_cycle_count = match->required_cycle_count,
      .observed_cycle_count = match->observed_cycle_count,
      .cycle_count = match->cycle_count,
      .immediate = match->delay_alu_immediate,
      .matrix_wait_profile = match->matrix_wait_profile,
      .matrix_result_use = match->matrix_result_use,
      .matrix_pass_count = match->matrix_pass_count,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_stream_append(
      &builder->ordinary_states, builder->transient_arena, wait_state));
  if (builder->replay_clause != NULL &&
      match->reason != LOOM_AMDGPU_WAIT_STATE_REASON_XNACK_REPLAY_CLAUSE) {
    // A replay separator already retains the consumer as its new clause.
    // Every other concrete prefix wait ends the preceding memory clause.
    loom_amdgpu_replay_clause_break(builder->replay_clause);
  }
  if (builder->short_wait != NULL) {
    loom_amdgpu_short_wait_advance(
        builder->short_wait, action == LOOM_AMDGPU_WAIT_STATE_ACTION_S_DELAY_ALU
                                 ? 1
                                 : match->cycle_count);
  }
  if (builder->matrix_results != NULL) {
    loom_amdgpu_matrix_result_wait_advance(
        builder->matrix_results,
        action == LOOM_AMDGPU_WAIT_STATE_ACTION_S_DELAY_ALU
            ? 1
            : match->cycle_count);
  }
  if (action == LOOM_AMDGPU_WAIT_STATE_ACTION_V_NOP) {
    IREE_ASSERT(builder->matrix_coexecution != NULL);
    loom_amdgpu_matrix_coexecution_advance(builder->matrix_coexecution,
                                           match->cycle_count);
    loom_amdgpu_delay_alu_advance(&builder->delay_alu,
                                  LOOM_AMDGPU_DELAY_ALU_TYPE_VALU,
                                  match->cycle_count);
    builder->current_position += match->cycle_count;
  } else if (action == LOOM_AMDGPU_WAIT_STATE_ACTION_S_NOP) {
    builder->current_position += match->cycle_count;
  } else {
    loom_amdgpu_delay_alu_complete(&builder->delay_alu,
                                   match->delay_alu_immediate);
    builder->current_position += 1;
  }
  return iree_ok_status();
}

static void loom_amdgpu_wait_state_match_descriptor_operands(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons,
    loom_amdgpu_wait_state_match_t* match) {
  if (allowed_reasons == 0) {
    return;
  }
  for (uint16_t i = 0; i < packet->node->operand_count; ++i) {
    loom_amdgpu_wait_state_match_assignment(
        builder,
        loom_low_packet_operand_assignment(builder->allocation, packet, i),
        allowed_reasons, match);
  }
}

static void loom_amdgpu_wait_state_match_result_writes(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons,
    loom_amdgpu_wait_state_match_t* match) {
  if (allowed_reasons == 0) {
    return;
  }
  for (uint16_t i = 0; i < packet->node->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_packet_result_assignment(builder->allocation, packet, i);
    loom_amdgpu_wait_state_match_assignment(builder, assignment,
                                            allowed_reasons, match);
  }
}

static void loom_amdgpu_wait_state_record_results(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_wait_state_packet_info_t* info) {
  if (builder->short_wait == NULL || info->instruction_count == 0) {
    return;
  }
  const bool is_valu =
      iree_any_bit_set(info->flags,
                       LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_USES_VECTOR_ALU) ||
      info->structural.vector_alu_instruction_count != 0;
  const loom_amdgpu_short_wait_reasons_t vector_two_slot_reasons =
      is_valu ? (loom_amdgpu_short_wait_reasons_t)
                        LOOM_AMDGPU_SHORT_WAIT_PERMLANE_READ |
                    (loom_amdgpu_descriptor_dot_identity_mask(
                         info->descriptor_traits) == 0
                         ? LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_TO_MATRIX_USE
                         : 0)
              : 0;
  loom_amdgpu_short_wait_reasons_t vector_one_slot_reasons =
      is_valu ? LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_READFIRSTLANE_VGPR_READ : 0;
  if (iree_any_bit_set(info->flags,
                       LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_TRANS_PRODUCER)) {
    vector_one_slot_reasons |=
        LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_TRANS_RESULT_USE;
  }
  if (iree_any_bit_set(
          info->flags,
          LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DST_SEL_FORWARDING_PRODUCER)) {
    vector_one_slot_reasons |=
        LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_DST_SEL_FORWARDING_USE;
  }
  for (uint16_t i = 0; i < packet->node->result_count; ++i) {
    const loom_low_allocation_assignment_t* assignment =
        loom_low_packet_result_assignment(builder->allocation, packet, i);
    if (assignment == NULL ||
        assignment->location_kind !=
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
      continue;
    }
    const uint16_t register_class = assignment->descriptor_reg_class_id;
    if (register_class == LOOM_AMDGPU_REG_CLASS_ID_VGPR ||
        register_class ==
            loom_amdgpu_agpr_reg_class_id(builder->descriptor_set)) {
      const loom_amdgpu_short_wait_reasons_t two_slot_reasons =
          vector_two_slot_reasons |
          (register_class == LOOM_AMDGPU_REG_CLASS_ID_VGPR
               ? LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_LANE_PERMUTE_VGPR_READ
               : 0);
      loom_amdgpu_short_wait_record(
          builder->short_wait, packet->node_index, register_class,
          assignment->location_base, assignment->location_count,
          (uint32_t)info->instruction_count, two_slot_reasons, 2);
      loom_amdgpu_short_wait_record(
          builder->short_wait, packet->node_index, register_class,
          assignment->location_base, assignment->location_count,
          (uint32_t)info->instruction_count, vector_one_slot_reasons, 1);
    } else if (is_valu && (register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR ||
                           register_class == LOOM_AMDGPU_REG_CLASS_ID_VCC)) {
      loom_amdgpu_short_wait_record(
          builder->short_wait, packet->node_index, register_class,
          assignment->location_base, assignment->location_count,
          (uint32_t)info->instruction_count,
          LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_SGPR_READ, 2);
      loom_amdgpu_short_wait_record(
          builder->short_wait, packet->node_index, register_class,
          assignment->location_base, assignment->location_count,
          (uint32_t)info->instruction_count,
          register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR
              ? LOOM_AMDGPU_SHORT_WAIT_VMEM_SGPR_READ
              : LOOM_AMDGPU_SHORT_WAIT_DIV_FMAS_CONDITION,
          register_class == LOOM_AMDGPU_REG_CLASS_ID_SGPR ? 5 : 4);
    } else if (iree_any_bit_set(info->descriptor_traits,
                                LOOM_AMDGPU_DESCRIPTOR_TRAIT_SCALAR_ALU) &&
               iree_any_bit_set(loom_amdgpu_reg_class_traits(
                                    builder->descriptor_set, register_class),
                                LOOM_AMDGPU_REG_CLASS_TRAIT_M0)) {
      loom_amdgpu_short_wait_record(
          builder->short_wait, packet->node_index, register_class,
          assignment->location_base, assignment->location_count,
          (uint32_t)info->instruction_count, LOOM_AMDGPU_SHORT_WAIT_M0_READ, 1);
    }
  }
}

static iree_status_t loom_amdgpu_wait_state_packet_analyze(
    loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    loom_amdgpu_wait_state_packet_info_t* out_info) {
  *out_info = (loom_amdgpu_wait_state_packet_info_t){0};
  if (packet->descriptor == NULL) {
    out_info->structural = loom_amdgpu_structural_packet_analyze(
        builder->schedule, builder->allocation, packet->node_index, 0);
    out_info->instruction_count = out_info->structural.instruction_count;
    out_info->delay_alu_type = loom_amdgpu_wait_state_structural_delay_alu_type(
        builder, &out_info->structural);
    if (out_info->delay_alu_type != LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER) {
      out_info->delay_alu_latency_cycles =
          loom_amdgpu_delay_alu_latency_cycles(out_info->delay_alu_type, 0);
    }
    return iree_ok_status();
  }

  out_info->flags |= LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DESCRIPTOR;
  out_info->instruction_count = 1;
  const loom_low_descriptor_t* descriptor = packet->descriptor;
  const loom_low_descriptor_set_t* descriptor_set = builder->descriptor_set;
  const loom_amdgpu_descriptor_traits_t descriptor_traits =
      loom_amdgpu_descriptor_traits(descriptor_set, descriptor);
  out_info->descriptor_traits = descriptor_traits;
  if (iree_any_bit_set(descriptor_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_SHIFT64_ALLOCATION)) {
    const loom_low_allocation_assignment_t* count =
        loom_low_packet_operand_assignment(builder->allocation, packet, 0);
    if (count->descriptor_reg_class_id == LOOM_AMDGPU_REG_CLASS_ID_VGPR &&
        (count->location_base & 7u) == 7u) {
      const uint32_t minimum_vgpr_count = count->location_base + 2;
      if (minimum_vgpr_count > builder->minimum_vgpr_count) {
        builder->minimum_vgpr_count = minimum_vgpr_count;
      }
    }
  }
  if (iree_any_bit_set(descriptor_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU)) {
    out_info->flags |= LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_USES_VECTOR_ALU;
  }
  if (iree_any_bit_set(descriptor_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_MEMORY)) {
    out_info->flags |= LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_USES_VECTOR_MEMORY;
  }

  if (builder->matrix_results != NULL) {
    out_info->matrix_source = loom_amdgpu_matrix_result_wait_source_at(
        builder->matrix_results, (uint32_t)packet->packet_index);
    if (out_info->matrix_source != NULL) {
      out_info->flags |= LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_MATRIX;
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
  if (iree_any_bit_set(descriptor_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_DPP |
                           LOOM_AMDGPU_DESCRIPTOR_TRAIT_PERMLANE)) {
    out_info->flags |= LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_LANE_PERMUTE_CONSUMER;
  }
  if (iree_any_bit_set(descriptor_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_READFIRSTLANE)) {
    out_info->flags |=
        LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_READFIRSTLANE_CONSUMER;
  }
  if (processor_has_valu_sgpr_read_hazard &&
      iree_any_bit_set(descriptor_traits,
                       LOOM_AMDGPU_DESCRIPTOR_TRAIT_VECTOR_ALU)) {
    out_info->flags |=
        LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_VALU_SGPR_READ_CONSUMER;
  }

  if (loom_amdgpu_wait_state_has_scheduling(
          builder,
          LOOM_AMDGPU_PROCESSOR_SCHEDULING_DESTINATION_SELECTION_WAIT_STATES)) {
    if (loom_amdgpu_wait_state_packet_has_destination_selection_forwarding_hazard(
            builder, packet, descriptor_traits)) {
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
    out_info->delay_alu_latency_cycles = loom_amdgpu_delay_alu_latency_cycles(
        out_info->delay_alu_type,
        loom_amdgpu_wait_state_descriptor_latency_cycles(builder, descriptor));
  }
  return iree_ok_status();
}

static void loom_amdgpu_wait_state_match_structural_operands(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_structural_packet_info_t* info,
    loom_amdgpu_wait_state_match_t* match) {
  if (builder->short_wait == NULL || info->instruction_count == 0) {
    return;
  }
  const loom_amdgpu_short_wait_reasons_t read_reasons =
      LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_TRANS_RESULT_USE |
      LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_DST_SEL_FORWARDING_USE |
      LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_SGPR_READ;
  if (info->moves.count != 0) {
    for (iree_host_size_t i = 0; i < info->moves.count; ++i) {
      const loom_low_move_t* move =
          &builder->allocation->moves[info->moves.start + i];
      if (move->destination.descriptor_reg_class_id !=
              LOOM_AMDGPU_REG_CLASS_ID_VGPR &&
          move->destination.descriptor_reg_class_id !=
              loom_amdgpu_agpr_reg_class_id(builder->descriptor_set)) {
        continue;
      }
      for (uint32_t j = 0; j < loom_low_move_access_count(move); ++j) {
        const loom_low_move_location_t* read = loom_low_move_read(move, j);
        const loom_low_move_location_t* write = loom_low_move_write(move, j);
        if (read->location_kind ==
            LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER) {
          const loom_amdgpu_short_wait_match_t candidate =
              loom_amdgpu_short_wait_query(
                  builder->short_wait, read->descriptor_reg_class_id,
                  read->location, 1, (uint32_t)i, read_reasons);
          loom_amdgpu_wait_state_merge_short_match(&candidate, match);
        }
        const loom_amdgpu_short_wait_match_t candidate =
            loom_amdgpu_short_wait_query(
                builder->short_wait, write->descriptor_reg_class_id,
                write->location, 1, (uint32_t)i,
                LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_DST_SEL_FORWARDING_USE);
        loom_amdgpu_wait_state_merge_short_match(&candidate, match);
      }
    }
    return;
  }
  if (info->vector_alu_instruction_count == 0) {
    return;
  }
  loom_amdgpu_wait_state_match_descriptor_operands(builder, packet,
                                                   read_reasons, match);
  loom_amdgpu_wait_state_match_result_writes(
      builder, packet,
      LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_DST_SEL_FORWARDING_USE, match);
}

static void loom_amdgpu_wait_state_match_packet_hazards(
    const loom_amdgpu_wait_state_builder_t* builder,
    const loom_low_packet_view_t* packet,
    const loom_amdgpu_wait_state_packet_info_t* info,
    loom_amdgpu_wait_state_match_t* match) {
  const bool processor_has_valu_sgpr_read_hazard =
      loom_amdgpu_wait_state_has_scheduling(
          builder, LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES);

  if (iree_any_bit_set(info->flags,
                       LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_MATRIX)) {
    loom_amdgpu_wait_state_match_descriptor_operands(
        builder, packet, LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_TO_MATRIX_USE,
        match);
  } else if (iree_any_bit_set(info->flags,
                              LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DESCRIPTOR)) {
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons = 0;
    if (iree_any_bit_set(
            info->flags,
            LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_TRANS_FORWARDING_CONSUMER)) {
      allowed_reasons |= LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_TRANS_RESULT_USE;
    }
    if (iree_any_bit_set(
            info->flags,
            LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_LANE_PERMUTE_CONSUMER)) {
      allowed_reasons |=
          iree_any_bit_set(info->descriptor_traits,
                           LOOM_AMDGPU_DESCRIPTOR_TRAIT_DPP)
              ? LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_LANE_PERMUTE_VGPR_READ
              : (loom_amdgpu_short_wait_reasons_t)
                    LOOM_AMDGPU_SHORT_WAIT_PERMLANE_READ;
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
    if (iree_any_bit_set(
            info->flags,
            LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_VALU_SGPR_READ_CONSUMER)) {
      allowed_reasons |= LOOM_AMDGPU_WAIT_STATE_REASON_FLAG_VALU_SGPR_READ;
    }
    if (iree_any_bit_set(
            info->flags,
            LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_USES_VECTOR_MEMORY)) {
      allowed_reasons |= LOOM_AMDGPU_SHORT_WAIT_VMEM_SGPR_READ;
    }
    if (iree_any_bit_set(info->descriptor_traits,
                         LOOM_AMDGPU_DESCRIPTOR_TRAIT_DIV_FMAS)) {
      allowed_reasons |= LOOM_AMDGPU_SHORT_WAIT_DIV_FMAS_CONDITION;
    }
    allowed_reasons |= LOOM_AMDGPU_SHORT_WAIT_M0_READ;
    loom_amdgpu_wait_state_match_descriptor_operands(builder, packet,
                                                     allowed_reasons, match);
  } else {
    loom_amdgpu_wait_state_match_structural_operands(builder, packet,
                                                     &info->structural, match);
  }
  if (!iree_any_bit_set(info->flags,
                        LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_MATRIX) &&
      (iree_any_bit_set(info->flags,
                        LOOM_AMDGPU_WAIT_STATE_PACKET_FLAG_DESCRIPTOR) ||
       (packet->node->op->result_count != 0 &&
        iree_any_bit_set(info->structural.flags,
                         LOOM_AMDGPU_STRUCTURAL_PACKET_FLAG_MATERIALIZES)))) {
    loom_amdgpu_wait_state_reason_flags_t allowed_reasons = 0;
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
  if (builder->mask_writes != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_mask_write_wait_collect(
        builder->mask_writes, packet, &info.structural,
        info.descriptor_traits));
  }
  if (builder->sgpr_reads != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sgpr_read_wait_collect_packet(
        builder->sgpr_reads, builder->allocation, packet, NULL,
        &info.structural, info.descriptor_traits));
  }
  const bool processor_has_valu_sgpr_read_hazard =
      loom_amdgpu_wait_state_has_scheduling(
          builder, LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES);
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

  loom_amdgpu_short_wait_match_t short_match = {0};
  if (builder->short_wait != NULL) {
    short_match = loom_amdgpu_short_wait_begin_packet(
        builder->short_wait, packet, &info.structural, info.descriptor_traits);
  }
  loom_amdgpu_wait_state_match_t match = {0};
  loom_amdgpu_wait_state_match_packet_hazards(builder, packet, &info, &match);
  if (builder->matrix_results != NULL) {
    const loom_amdgpu_matrix_result_match_t matrix_match =
        loom_amdgpu_matrix_result_wait_inspect(
            builder->matrix_results, packet, info.matrix_source,
            &info.structural,
            iree_any_bit_set(
                info.descriptor_traits,
                LOOM_AMDGPU_DESCRIPTOR_TRAIT_MATRIX_MEMORY_CONSUMER));
    if (matrix_match.remaining > match.cycle_count) {
      match = (loom_amdgpu_wait_state_match_t){
          .reason = LOOM_AMDGPU_WAIT_STATE_REASON_MATRIX_RESULT_USE,
          .producer_node = matrix_match.source->node_index,
          .required_cycle_count = matrix_match.required,
          .observed_cycle_count =
              (uint16_t)(matrix_match.required - matrix_match.remaining),
          .cycle_count = matrix_match.remaining,
          .matrix_wait_profile = builder->matrix_wait_profile,
          .matrix_result_use = matrix_match.use,
          .matrix_pass_count = matrix_match.source->instruction.pass_count,
      };
    }
  }
  loom_amdgpu_wait_state_merge_short_match(&short_match, &match);
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
        loom_amdgpu_wait_state_delay_alu_match_packet(builder, packet, &info,
                                                      &delay_alu_match);
    if (delay_alu_action != LOOM_AMDGPU_WAIT_STATE_ACTION_UNKNOWN) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_append(
          builder, packet, packet, &delay_alu_match, delay_alu_action));
    }
  }

  if (builder->replay_clause != NULL) {
    const uint32_t producer_node = loom_amdgpu_replay_clause_apply(
        builder->replay_clause, packet, info.descriptor_traits,
        info.instruction_count);
    if (producer_node != LOOM_LOW_PACKET_INDEX_NONE) {
      const loom_amdgpu_wait_state_match_t replay_match = {
          .reason = LOOM_AMDGPU_WAIT_STATE_REASON_XNACK_REPLAY_CLAUSE,
          .producer_node = producer_node,
          .required_cycle_count = 1,
          .cycle_count = 1,
      };
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_wait_state_append(builder, packet, packet, &replay_match,
                                        LOOM_AMDGPU_WAIT_STATE_ACTION_S_NOP));
    }
  }

  if (info.structural.moves.count == 0) {
    loom_amdgpu_wait_state_record_results(builder, packet, &info);
  } else {
    loom_amdgpu_wait_state_apply_moves(builder, packet, info.structural.moves,
                                       processor_has_valu_sgpr_read_hazard);
  }

  if (info.structural.moves.count == 0) {
    if (info.delay_alu_type != LOOM_AMDGPU_DELAY_ALU_TYPE_OTHER &&
        info.delay_alu_latency_cycles != 0) {
      loom_amdgpu_wait_state_delay_alu_record_results(
          builder, packet, info.delay_alu_type, info.delay_alu_latency_cycles);
    } else if (info.instruction_count != 0) {
      loom_amdgpu_wait_state_delay_alu_clear_results(builder, packet);
    }
    loom_amdgpu_delay_alu_advance(&builder->delay_alu, info.delay_alu_type,
                                  info.instruction_count);
    builder->current_position += info.instruction_count;
  } else {
    loom_amdgpu_delay_alu_advance(&builder->delay_alu,
                                  LOOM_AMDGPU_DELAY_ALU_TYPE_VALU,
                                  info.structural.vector_alu_instruction_count);
    loom_amdgpu_delay_alu_advance(&builder->delay_alu,
                                  LOOM_AMDGPU_DELAY_ALU_TYPE_SALU,
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
  if (builder->short_wait != NULL) {
    loom_amdgpu_short_wait_commit(builder->short_wait, packet);
  }
  if (builder->matrix_results != NULL) {
    loom_amdgpu_matrix_result_wait_advance(
        builder->matrix_results,
        packet->descriptor != NULL
            ? 1
            : loom_amdgpu_structural_packet_minimum_instruction_count(
                  &info.structural));
    loom_amdgpu_matrix_result_wait_commit(builder->matrix_results,
                                          info.matrix_source);
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

  if (builder->sgpr_reads != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sgpr_read_wait_collect_packet(
        builder->sgpr_reads, builder->allocation, first_packet, second_packet,
        &first_info.structural, first_info.descriptor_traits));
  }

  if (builder->short_wait != NULL) {
    loom_amdgpu_short_wait_begin_packet(builder->short_wait, first_packet,
                                        &first_info.structural,
                                        first_info.descriptor_traits);
  }

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

  loom_amdgpu_wait_state_record_results(builder, first_packet, &first_info);
  loom_amdgpu_wait_state_record_results(builder, second_packet, &second_info);
  loom_amdgpu_wait_state_delay_alu_record_results(
      builder, first_packet, first_info.delay_alu_type,
      first_info.delay_alu_latency_cycles);
  loom_amdgpu_wait_state_delay_alu_record_results(
      builder, second_packet, second_info.delay_alu_type,
      second_info.delay_alu_latency_cycles);
  loom_amdgpu_delay_alu_advance(&builder->delay_alu,
                                LOOM_AMDGPU_DELAY_ALU_TYPE_VALU, 1);
  ++builder->current_position;

  builder->packet_instruction_counts[first_packet->packet_index] = 1;
  builder->packet_instruction_counts[second_packet->packet_index] = 0;
  if (builder->short_wait != NULL) {
    loom_amdgpu_short_wait_commit(builder->short_wait, first_packet);
  }
  if (builder->matrix_coexecution != NULL) {
    loom_amdgpu_matrix_coexecution_commit_vopd_pair(
        builder->matrix_coexecution);
  }
  ++builder->progress_event_count;
  return iree_ok_status();
}

static const loom_amdgpu_wait_state_t* loom_amdgpu_wait_state_stream_at(
    const loom_amdgpu_wait_state_stream_t* stream, iree_host_size_t index) {
  IREE_ASSERT(index < stream->state_count);
  const uint32_t segment_index =
      (uint32_t)(index / LOOM_AMDGPU_WAIT_STATES_PER_SEGMENT);
  const iree_host_size_t segment_offset =
      index % LOOM_AMDGPU_WAIT_STATES_PER_SEGMENT;
  const loom_amdgpu_wait_state_segment_t* segment =
      (const loom_amdgpu_wait_state_segment_t*)
          loom_segmented_storage_const_segment(&stream->segments,
                                               segment_index);
  return &segment->states[segment_offset];
}

// Returns true when |ordinary| precedes |resolved| in the final wait table.
// Ordinary waits retain the original first position when both streams target
// the same packet.
static bool loom_amdgpu_wait_state_ordinary_precedes(
    const loom_amdgpu_wait_state_t* ordinary,
    const loom_amdgpu_wait_state_t* resolved) {
  return ordinary->block_index < resolved->block_index ||
         (ordinary->block_index == resolved->block_index &&
          ordinary->scheduled_ordinal <= resolved->scheduled_ordinal);
}

static iree_status_t loom_amdgpu_wait_state_finalize(
    loom_amdgpu_wait_state_builder_t* builder) {
  const iree_host_size_t ordinary_count = builder->ordinary_states.state_count;
  const iree_host_size_t resolved_count = builder->resolved_states.state_count;
  IREE_ASSERT(ordinary_count <= IREE_HOST_SIZE_MAX - resolved_count);
  builder->state_count = ordinary_count + resolved_count;
  if (builder->state_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, builder->state_count, sizeof(*builder->states),
      (void**)&builder->states));

  iree_host_size_t ordinary_index = 0;
  iree_host_size_t resolved_index = 0;
  iree_host_size_t output_index = 0;
  while (output_index < builder->state_count) {
    const loom_amdgpu_wait_state_t* next = NULL;
    if (resolved_index == resolved_count) {
      next = loom_amdgpu_wait_state_stream_at(&builder->ordinary_states,
                                              ordinary_index++);
    } else if (ordinary_index == ordinary_count) {
      next = loom_amdgpu_wait_state_stream_at(&builder->resolved_states,
                                              resolved_index++);
    } else {
      const loom_amdgpu_wait_state_t* ordinary =
          loom_amdgpu_wait_state_stream_at(&builder->ordinary_states,
                                           ordinary_index);
      const loom_amdgpu_wait_state_t* resolved =
          loom_amdgpu_wait_state_stream_at(&builder->resolved_states,
                                           resolved_index);
      if (loom_amdgpu_wait_state_ordinary_precedes(ordinary, resolved)) {
        next = ordinary;
        ++ordinary_index;
      } else {
        next = resolved;
        ++resolved_index;
      }
    }
    builder->states[output_index++] = *next;
  }
  IREE_ASSERT_EQ(ordinary_index, ordinary_count);
  IREE_ASSERT_EQ(resolved_index, resolved_count);
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
      .class_name = loom_amdgpu_wait_state_progress_class_name,
  };
  return loom_low_packet_progress_build(builder->schedule, builder->allocation,
                                        &provider, builder->arena,
                                        &builder->progress);
}

typedef struct loom_amdgpu_wait_state_hazard_query_t {
  // Borrowed states not yet projected, in scheduled packet order.
  const loom_amdgpu_wait_state_t* states;
  // Number of remaining states, including all waits at the next insertion
  // point.
  iree_host_size_t remaining_count;
} loom_amdgpu_wait_state_hazard_query_t;

static void loom_amdgpu_wait_state_hazard_query(
    void* user_data, const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_table_t* progress,
    const loom_low_packet_view_t* packet,
    loom_low_packet_hazard_plan_emit_fn_t emit, void* emit_user_data) {
  (void)schedule;
  (void)allocation;
  (void)progress;
  loom_amdgpu_wait_state_hazard_query_t* query =
      (loom_amdgpu_wait_state_hazard_query_t*)user_data;
  // The common builder visits packets once in order. States were appended in
  // that same order, including VOPD waits attached to the first paired packet.
  while (query->remaining_count != 0) {
    const loom_amdgpu_wait_state_t* wait_state = query->states;
    if (wait_state->node_index != packet->node_index) {
      break;
    }
    ++query->states;
    --query->remaining_count;
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
  loom_amdgpu_wait_state_hazard_query_t query = {
      .states = builder->states,
      .remaining_count = builder->state_count,
  };
  const loom_low_packet_hazard_plan_provider_t provider = {
      .user_data = &query,
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
      builder->processor_properties->features.scheduling;
  builder->has_delay_alu = loom_amdgpu_wait_state_target_has_delay_alu(builder);
  if (loom_amdgpu_wait_state_has_scheduling(
          builder, LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_DEPCTR)) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sgpr_read_wait_allocate(
        builder->schedule, builder->transient_arena, &builder->sgpr_reads));
  }
  if (loom_amdgpu_wait_state_has_scheduling(
          builder, LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_MASK_WRITE_DEPCTR) &&
      loom_low_resolved_target_bundle(&builder->schedule->target)
              ->snapshot->subgroup_size == 64) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_mask_write_wait_allocate(
        builder->schedule, builder->allocation, builder->transient_arena,
        &builder->mask_writes));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_short_wait_allocate(
      builder->schedule, builder->allocation, builder->processor_scheduling,
      builder->wait_plan->descriptor_traits,
      builder->wait_plan->dot_identity_mask, builder->transient_arena,
      &builder->short_wait));
  IREE_RETURN_IF_ERROR(loom_amdgpu_replay_clause_allocate(
      builder->schedule, builder->allocation,
      builder->wait_plan->descriptor_traits, builder->transient_arena,
      &builder->replay_clause));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_allocate(builder));
  iree_host_size_t wait_packet_cursor = 0;
  iree_host_size_t wait_action_cursor = 0;
  iree_host_size_t address_state_cursor = 0;
  builder->delay_alu.epoch = 1;
  for (iree_host_size_t block_index = 0;
       block_index < builder->schedule->block_count; ++block_index) {
    if (block_index > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AMDGPU wait-state block index exceeds uint32_t");
    }
    if (builder->vgpr_count != 0) {
      memset(builder->vgprs, 0, builder->vgpr_count * sizeof(*builder->vgprs));
    }
    if (builder->sgprs.count != 0) {
      memset(builder->sgprs.states, 0,
             builder->sgprs.count * sizeof(*builder->sgprs.states));
    }
    if (builder->short_wait != NULL) {
      loom_amdgpu_short_wait_begin_block(builder->short_wait,
                                         (uint16_t)block_index);
    }
    builder->current_position = 0;
    builder->delay_alu.valu_count = 0;
    builder->delay_alu.trans_count = 0;
    if (builder->matrix_coexecution != NULL) {
      loom_amdgpu_matrix_coexecution_begin_block(builder->matrix_coexecution,
                                                 (uint16_t)block_index);
    }
    if (builder->matrix_results != NULL) {
      loom_amdgpu_matrix_result_wait_begin_block(builder->matrix_results,
                                                 (uint16_t)block_index);
    }
    const loom_low_schedule_block_t* block =
        &builder->schedule->blocks[block_index];
    for (uint32_t scheduled_ordinal = 0;
         scheduled_ordinal < block->scheduled_node_count; ++scheduled_ordinal) {
      const loom_low_packet_view_t packet = loom_low_packet_at_block_ordinal(
          builder->schedule, (uint32_t)block_index, scheduled_ordinal);
      if (builder->replay_clause != NULL &&
          address_state_cursor < builder->address_state->transition_count &&
          builder->address_state->transitions[address_state_cursor]
                  .node_index == packet.node_index) {
        loom_amdgpu_replay_clause_break(builder->replay_clause);
        ++address_state_cursor;
      }
      if (builder->short_wait != NULL || builder->sgpr_reads != NULL) {
        // CDNA's shared VM counter drains store payloads as well as load
        // results. A full drain retires the entire two-slot source window,
        // including obligations arriving through a predecessor.
        while (wait_action_cursor < builder->wait_plan->action_count &&
               builder->wait_plan->actions[wait_action_cursor].node_index ==
                   packet.node_index) {
          const loom_amdgpu_wait_plan_action_t* action =
              &builder->wait_plan->actions[wait_action_cursor++];
          if (builder->short_wait != NULL && action->target_count == 0 &&
              (action->counter_id == LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD ||
               action->counter_id == LOOM_AMDGPU_WAIT_COUNTER_VMEM_STORE)) {
            loom_amdgpu_short_wait_drain_stores(builder->short_wait);
          }
          if (builder->sgpr_reads != NULL &&
              action->counter_id == LOOM_AMDGPU_WAIT_COUNTER_ALU) {
            IREE_RETURN_IF_ERROR(loom_amdgpu_sgpr_read_wait_collect_wait(
                builder->sgpr_reads, &packet, action->target_count));
          }
        }
      }
      while (wait_packet_cursor < builder->wait_packets->packet_count &&
             builder->wait_packets->packets[wait_packet_cursor].node_index ==
                 packet.node_index) {
        if (builder->replay_clause != NULL) {
          loom_amdgpu_replay_clause_break(builder->replay_clause);
        }
        if (builder->short_wait != NULL) {
          loom_amdgpu_short_wait_advance(builder->short_wait, 1);
        }
        if (builder->matrix_results != NULL) {
          loom_amdgpu_matrix_result_wait_advance(builder->matrix_results, 1);
        }
        ++wait_packet_cursor;
      }
      if (loom_amdgpu_wait_plan_elides_node(builder->wait_plan,
                                            packet.node_index)) {
        builder->packet_instruction_counts[packet.packet_index] = 0;
        continue;
      }
      const loom_amdgpu_vopd_packet_role_t vopd_role =
          builder->vopd_packets != NULL
              ? builder->vopd_packets[packet.packet_index].role
              : LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE;
      if (vopd_role == LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND) {
        builder->packet_instruction_counts[packet.packet_index] = 0;
        continue;
      }
      if (vopd_role == LOOM_AMDGPU_VOPD_PACKET_ROLE_FIRST) {
        if (builder->replay_clause != NULL) {
          loom_amdgpu_replay_clause_break(builder->replay_clause);
        }
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
    if (builder->short_wait != NULL) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_short_wait_end_block(builder->short_wait));
    }
    if (builder->matrix_coexecution != NULL) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_coexecution_end_block(
          builder->matrix_coexecution));
    }
    if (builder->matrix_results != NULL) {
      loom_amdgpu_matrix_result_wait_end_block(builder->matrix_results);
    }
  }
  if (builder->short_wait != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_short_wait_resolve(
        builder->short_wait, loom_amdgpu_wait_state_append_resolved, builder));
  }
  if (builder->mask_writes != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_mask_write_wait_resolve(
        builder->mask_writes, loom_amdgpu_wait_state_append_resolved, builder));
  }
  if (builder->sgpr_reads != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_sgpr_read_wait_resolve(
        builder->sgpr_reads, loom_amdgpu_wait_state_append_resolved, builder));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_finalize(builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_build_progress(builder));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_build_hazard_plan(builder));
  return iree_ok_status();
}

static void loom_amdgpu_wait_state_plan_publish(
    const loom_amdgpu_wait_state_builder_t* builder,
    loom_amdgpu_wait_state_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_wait_state_plan_t){
      .schedule = builder->schedule,
      .allocation = builder->allocation,
      .progress = builder->progress,
      .hazard_plan = builder->hazard_plan,
      .states = builder->states,
      .state_count = builder->state_count,
      .minimum_vgpr_count = builder->minimum_vgpr_count,
  };
  out_plan->hazard_plan.progress = &out_plan->progress;
}

iree_status_t loom_amdgpu_wait_state_plan_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_amdgpu_processor_properties_t* processor_properties,
    const loom_amdgpu_address_state_plan_t* address_state,
    const loom_amdgpu_wait_packet_plan_t* wait_packets,
    const struct loom_amdgpu_vopd_plan_t* vopd_plan,
    loom_amdgpu_matrix_coexecution_t* matrix_coexecution,
    loom_amdgpu_matrix_result_wait_t* matrix_results,
    iree_arena_allocator_t* arena, iree_arena_allocator_t* transient_arena,
    loom_amdgpu_wait_state_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_wait_state_plan_t){0};

  const bool has_vopd_pairs = vopd_plan != NULL && vopd_plan->pair_count != 0;
  loom_amdgpu_wait_state_builder_t builder = {
      .schedule = schedule,
      .allocation = allocation,
      .processor_properties = processor_properties,
      .matrix_wait_profile =
          loom_amdgpu_matrix_wait_profile_from_feature_profile(
              processor_properties->features.matrix),
      .wait_plan = wait_packets->wait_plan,
      .wait_packets = wait_packets,
      .address_state = address_state,
      .vopd_packets = has_vopd_pairs ? vopd_plan->packets : NULL,
      .vopd_pairs = has_vopd_pairs ? vopd_plan->pairs : NULL,
      .descriptor_set = schedule->target.descriptor_set,
      .schedule_model = schedule->target.schedule_model,
      .arena = arena,
      .transient_arena = transient_arena,
      .matrix_coexecution = matrix_coexecution,
      .matrix_results = matrix_results,
  };
  loom_segmented_storage_initialize(
      sizeof(loom_amdgpu_wait_state_segment_t),
      iree_alignof(loom_amdgpu_wait_state_segment_t),
      &builder.ordinary_states.segments);
  loom_segmented_storage_initialize(
      sizeof(loom_amdgpu_wait_state_segment_t),
      iree_alignof(loom_amdgpu_wait_state_segment_t),
      &builder.resolved_states.segments);
  iree_status_t status =
      loom_amdgpu_wait_state_plan_build_with_scratch(&builder);
  if (iree_status_is_ok(status)) {
    loom_amdgpu_wait_state_plan_publish(&builder, out_plan);
  }
  return status;
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
      case LOOM_AMDGPU_WAIT_STATE_ACTION_S_WAITCNT_DEPCTR:
      case LOOM_AMDGPU_WAIT_STATE_ACTION_S_WAIT_ALU:
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
