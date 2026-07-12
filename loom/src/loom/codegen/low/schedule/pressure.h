// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Mutable register and target-resource pressure simulation for low scheduling.

#ifndef LOOM_CODEGEN_LOW_SCHEDULE_PRESSURE_H_
#define LOOM_CODEGEN_LOW_SCHEDULE_PRESSURE_H_

#include "loom/codegen/low/schedule/context.h"
#include "loom/codegen/low/schedule/dependency_index.h"
#include "loom/codegen/low/schedule/pressure_alias.h"
#include "loom/codegen/low/schedule/ready_policy.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_schedule_alias_pressure_record_t
    loom_low_schedule_alias_pressure_record_t;
typedef struct loom_low_schedule_resource_pressure_record_t
    loom_low_schedule_resource_pressure_record_t;
typedef struct loom_low_schedule_unlock_record_t
    loom_low_schedule_unlock_record_t;

// Mutable pressure simulation state for one scheduling run.
struct loom_low_schedule_pressure_state_t {
  // Current live register units by descriptor register-class ID.
  uint64_t* current_live_units_by_reg_class;
  // Shared pressure state for overlapping register classes.
  struct {
    // Mutable records indexed by one-based alias-set ID.
    loom_low_schedule_alias_pressure_record_t* records;
    // Alias-set IDs seen with live pressure in the current block.
    uint16_t* block_ids;
    // Alias-set IDs touched by the current candidate.
    uint16_t* candidate_delta_touched_ids;
    // Number of populated entries in block_ids.
    iree_host_size_t block_count;
    // Number of populated entries in candidate_delta_touched_ids.
    iree_host_size_t candidate_delta_touched_count;
  } alias_sets;
  // Incremental high-water state for derived target pressure resources.
  struct {
    // Per-class high-water marks, dense by descriptor register-class ID.
    uint64_t* peak_live_units_by_reg_class;
    // Mutable records indexed by pressure-resource ID.
    loom_low_schedule_resource_pressure_record_t* records;
    // Resource IDs touched by the current candidate.
    uint16_t* candidate_touched_ids;
    // Number of populated entries in candidate_touched_ids.
    uint16_t candidate_touched_count;
    // Current aggregate penalty across all derived resources.
    uint32_t pressure_cliff_penalty;
  } resources;
  // First target pressure cliff above the source-order pressure ceiling,
  // indexed by descriptor register-class ID.
  uint32_t* first_actionable_pressure_cliff_indices;
  // Register-class IDs seen with live pressure in the current block.
  uint16_t* block_reg_class_ids;
  // True when a register class is present in block_reg_class_ids.
  uint8_t* block_reg_class_touched_flags;
  // Value ordinals with per-block pressure state to reset before reuse.
  loom_value_ordinal_t* block_value_ordinals;
  // Candidate operand multiplicity by local value ordinal.
  uint16_t* candidate_operand_use_counts;
  // Per-candidate counters reused for alias units and unlocked operand uses.
  uint32_t* candidate_scratch_counts;
  // Value ordinals touched in candidate_operand_use_counts.
  loom_value_ordinal_t* candidate_operand_ordinals;
  // Scratch live-unit delta by descriptor register-class ID.
  int64_t* candidate_delta_units_by_reg_class;
  // True when a register class has candidate delta state to reset.
  uint8_t* candidate_delta_touched_flags;
  // Register-class IDs touched in candidate_delta_units_by_reg_class.
  uint16_t* candidate_delta_touched_reg_class_ids;
  // Remaining distinct consumer count, dense by local value ordinal.
  uint32_t* remaining_consumer_counts;
  // XOR of remaining consumer node indices, dense by local value ordinal.
  uint32_t* remaining_consumer_node_xors;
  // Incrementally maintained consumers unlocked by each producer.
  struct {
    // Remaining distinct dependency producers by consumer.
    loom_low_schedule_dependency_frontier_t frontier;
    // Pressure summaries indexed by producer node.
    loom_low_schedule_unlock_record_t* records;
    // Descriptor-consumer list heads indexed by producer node.
    uint32_t* descriptor_heads;
    // Next descriptor consumer indexed by consumer node.
    uint32_t* descriptor_next_nodes;
  } unlocks;
  // Storage-alias ownership shared by authored and scheduled pressure models.
  loom_low_schedule_pressure_alias_state_t storage_aliases;
  // Number of touched candidate register classes.
  iree_host_size_t candidate_delta_touched_count;
  // Current aggregate live register units in the simulated schedule.
  uint64_t current_live_units;
  // Pressure-cliff penalty for the current simulated schedule state.
  uint32_t current_pressure_cliff_penalty;
  // Number of populated entries in block_reg_class_ids.
  iree_host_size_t block_reg_class_count;
  // Number of populated entries in block_value_ordinals.
  iree_host_size_t block_value_count;
  // Number of populated entries in candidate_operand_ordinals.
  iree_host_size_t candidate_operand_count;
};

enum loom_low_schedule_pressure_source_kind_e {
  LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_NONE = 0,
  LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS = 1,
  LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_RESOURCE = 2,
};
typedef uint8_t loom_low_schedule_pressure_source_kind_t;

// Complete bounded score for one ready scheduling candidate.
typedef struct loom_low_schedule_candidate_score_t {
  // Aggregate live register units after scheduling the candidate.
  uint64_t projected_live_units;
  // Live register units whose last use is the candidate.
  uint64_t killed_live_units;
  // Register result units made live by the candidate.
  uint64_t produced_live_units;
  // Projected pressure-cliff penalty relative to current state.
  int64_t pressure_cliff_penalty_delta;
  // Live register values whose last use is the candidate.
  uint32_t killed_live_value_count;
  // Register result values made live by the candidate.
  uint32_t produced_live_value_count;
  // Longest same-block latency path starting at the candidate.
  uint32_t critical_path_cycles;
  // Downstream visible register demand reached through structural nodes.
  uint32_t pressure_demand_units;
  // Register headroom reserved for results opened by the candidate.
  uint32_t pressure_reserve_units;
  // Cycles until all same-block SSA producers are ready.
  uint32_t data_ready_stall_cycles;
  // Cycles until descriptor resources can accept this candidate.
  uint32_t resource_stall_cycles;
  // Cycles until target hazard distance rows are satisfied.
  uint32_t hazard_stall_cycles;
  // Maximum stall across data, resources, and target hazards.
  uint32_t effective_stall_cycles;
  // Target pressure-cliff penalty from projected live units.
  uint32_t pressure_cliff_penalty;
  // Pressure-cliff penalty excluding speculative reserve units.
  uint32_t actual_pressure_cliff_penalty;
  // Crossed pressure cliff, or LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE.
  uint32_t pressure_cliff_units;
  // Live units before the next cliff, or PRESSURE_CLIFF_NONE.
  uint32_t units_until_pressure_cliff;
  // Source-order tie breaker.
  uint32_t source_ordinal;
  // Maximum latency of same-block SSA producers consumed by the candidate.
  uint16_t dependency_latency_cycles;
  // Descriptor latency for the candidate itself.
  uint16_t latency_cycles;
  // Resource causing resource_stall_cycles, or LOOM_LOW_RESOURCE_NONE.
  uint16_t bottleneck_resource_id;
  // Target pair-affinity reward. Larger scores are better.
  uint16_t pair_affinity_score;
  // Structurally possible placement alternatives for an equal-benefit pair.
  uint16_t pair_placement_option_count;
  // Number of storage-relation rows owned by the candidate.
  uint16_t storage_relation_count;
  // Register-class or resource ID for the closest pressure cliff.
  uint16_t pressure_cliff_source_id;
  // Interpretation of pressure_cliff_source_id.
  loom_low_schedule_pressure_source_kind_t pressure_cliff_source_kind;
  // Candidate properties used by pressure-aware comparison.
  uint8_t flags;
} loom_low_schedule_candidate_score_t;

enum loom_low_schedule_candidate_flag_bits_e {
  LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_DESCRIPTOR = 1u << 0,
  LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_NON_GROWING_DESCRIPTOR = 1u << 1,
  LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_PRESERVES_ZERO_PRESSURE_PENALTY = 1u << 2,
  LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_HAS_MULTI_USE_RESULT = 1u << 3,
  LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_MAKES_PRESSURE_PROGRESS = 1u << 4,
  LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_PRESSURE_SENSITIVE = 1u << 5,
};

// Returns true when |strategy| simulates register and target pressure.
static inline bool loom_low_schedule_strategy_uses_pressure(
    loom_low_schedule_strategy_t strategy) {
  return strategy >= LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE &&
         strategy <= LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL;
}

iree_status_t loom_low_schedule_pressure_initialize(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count,
    loom_low_schedule_pressure_state_t* out_pressure_state);

loom_low_schedule_ready_keys_t loom_low_schedule_pressure_ready_keys(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index);

void loom_low_schedule_pressure_initialize_block(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_block_t* block_record,
    loom_low_schedule_pressure_state_t* pressure_state);

void loom_low_schedule_pressure_initialize_current_cliff_penalty(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state);

iree_status_t loom_low_schedule_pressure_initialize_unlock_summaries(
    loom_low_schedule_build_state_t* state, uint32_t node_count,
    loom_low_schedule_pressure_state_t* pressure_state);

// Publishes one consumer to its final unscheduled dependency producer.
void loom_low_schedule_pressure_publish_unlock_consumer(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t producer_node,
    uint32_t consumer_node);

void loom_low_schedule_pressure_score_candidate(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_ready_policy_t* ready_policy,
    const uint32_t* indegrees, uint32_t node_index,
    loom_low_schedule_candidate_score_t* out_score);

// Resolves a pressure source to its target descriptor-authored name.
iree_string_view_t loom_low_schedule_pressure_source_name(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_source_kind_t source_kind, uint16_t source_id);

void loom_low_schedule_pressure_note_node_scheduled(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index,
    const loom_low_schedule_candidate_score_t* score);

void loom_low_schedule_pressure_update_ready_consumers(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_ready_policy_t* ready_policy, uint32_t scheduled_node);

void loom_low_schedule_pressure_compute_node_priorities(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count,
    const loom_low_schedule_dependency_detail_index_t* dependency_details,
    loom_low_schedule_pressure_state_t* pressure_state);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SCHEDULE_PRESSURE_H_
