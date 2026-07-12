// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/allocation/target_constraints.h"
#include "loom/codegen/low/function_model.h"
#include "loom/codegen/low/schedule/context.h"
#include "loom/codegen/low/schedule/descriptor_rows.h"
#include "loom/codegen/low/schedule/diagnostics.h"
#include "loom/codegen/low/schedule/graph.h"
#include "loom/codegen/low/schedule/ready_frontier.h"
#include "loom/codegen/low/storage_relation.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/registers.h"

#define LOOM_LOW_SCHEDULE_READY_SOURCE_NOMINEE_COUNT 16
#define LOOM_LOW_SCHEDULE_READY_VIEW_SEARCH_COUNT 16
#define LOOM_LOW_SCHEDULE_READY_VIEW_NOMINEE_COUNT 2
#define LOOM_LOW_SCHEDULE_READY_NOMINEE_CAPACITY 24
#define LOOM_LOW_SCHEDULE_DESCRIPTOR_FRONTIER_CAPACITY 16
#define LOOM_LOW_SCHEDULE_PAIR_LOOKAHEAD_CAPACITY 16

enum loom_low_schedule_state_access_bits_e {
  LOOM_LOW_SCHEDULE_STATE_ACCESS_READ = 1u << 0,
  LOOM_LOW_SCHEDULE_STATE_ACCESS_WRITE = 1u << 1,
};

enum loom_low_schedule_alias_pressure_flag_bits_e {
  LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_BLOCK_TOUCHED = 1u << 0,
  LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_CANDIDATE_TOUCHED = 1u << 1,
};

enum loom_low_schedule_resource_pressure_flag_bits_e {
  LOOM_LOW_SCHEDULE_RESOURCE_PRESSURE_FLAG_CANDIDATE_TOUCHED = 1u << 0,
};

enum loom_low_schedule_unlock_flag_bits_e {
  // At least one descriptor node becomes ready with this producer.
  LOOM_LOW_SCHEDULE_UNLOCK_FLAG_DESCRIPTOR = 1u << 0,
};
typedef uint8_t loom_low_schedule_unlock_flags_t;

// Incremental pressure summary for consumers unlocked by one producer.
typedef struct loom_low_schedule_unlock_record_t {
  // Sum of downstream pressure demand across unlocked consumers.
  uint32_t demand_units;
  // Maximum downstream activation width across unlocked consumers.
  uint32_t activation_units;
  // Number of retained descriptor consumers, capped at capacity + 1.
  uint8_t descriptor_count;
  // Summary flags from loom_low_schedule_unlock_flag_bits_e.
  loom_low_schedule_unlock_flags_t flags;
} loom_low_schedule_unlock_record_t;

typedef enum loom_low_schedule_resource_high_water_mode_e {
  LOOM_LOW_SCHEDULE_RESOURCE_HIGH_WATER_SOURCE_BASELINE = 0,
  LOOM_LOW_SCHEDULE_RESOURCE_HIGH_WATER_SCHEDULED = 1,
} loom_low_schedule_resource_high_water_mode_t;

typedef struct loom_low_schedule_alias_pressure_record_t {
  // Current live units in the shared alias namespace.
  uint64_t current_live_units;
  // Live-unit delta projected for the candidate being scored.
  int64_t candidate_delta_units;
  // Mutable loom_low_schedule_alias_pressure_flag_bits_e bits.
  uint8_t flags;
} loom_low_schedule_alias_pressure_record_t;

typedef struct loom_low_schedule_resource_pressure_record_t {
  // Current sum of rounded member-class high-water marks.
  uint64_t current_peak_units;
  // Additional resource units projected by the candidate being scored.
  uint64_t candidate_added_units;
  // Penalty accumulated above the authored source-order baseline.
  uint32_t pressure_cliff_penalty;
  // First target cliff not yet crossed by the scheduled high-water mark.
  uint16_t next_cliff_index;
  // Mutable loom_low_schedule_resource_pressure_flag_bits_e bits.
  uint8_t flags;
} loom_low_schedule_resource_pressure_record_t;

typedef struct loom_low_schedule_pressure_state_t {
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
    // Number of populated entries in |candidate_touched_ids|.
    uint16_t candidate_touched_count;
    // Current aggregate penalty across all derived resources.
    uint32_t pressure_cliff_penalty;
  } resources;
  // First target pressure cliff above the source-order pressure ceiling,
  // indexed by descriptor register-class ID.
  uint32_t* first_actionable_pressure_cliff_indices;
  // Register-class IDs seen with live pressure in the current block.
  uint16_t* block_reg_class_ids;
  // True when a register class is present in |block_reg_class_ids|.
  uint8_t* block_reg_class_touched_flags;
  // Value ordinals with per-block pressure state to reset before reuse.
  loom_value_ordinal_t* block_value_ordinals;
  // Candidate operand multiplicity by local value ordinal.
  uint16_t* candidate_operand_use_counts;
  // Per-candidate counters reused sequentially for storage-relation alias
  // units and unlocked-frontier operand uses.
  uint32_t* candidate_scratch_counts;
  // Value ordinals touched in |candidate_operand_use_counts|.
  loom_value_ordinal_t* candidate_operand_ordinals;
  // Scratch live-unit delta by descriptor register-class ID for one candidate.
  int64_t* candidate_delta_units_by_reg_class;
  // True when a register class has a nonzero or previously nonzero candidate
  // delta that must be reset after scoring.
  uint8_t* candidate_delta_touched_flags;
  // Register-class IDs touched in |candidate_delta_units_by_reg_class|.
  uint16_t* candidate_delta_touched_reg_class_ids;
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
  // Dependencies from ready structural setup nodes, indexed by consumer node.
  uint32_t* ready_setup_dependency_counts;
  // Active alias lists keyed by source value ordinal.
  uint32_t* alias_heads;
  // Source value ordinals touched in |alias_heads|.
  loom_value_ordinal_t* alias_source_ordinals;
  // Active alias units claimed from each source value ordinal.
  uint32_t* alias_source_unit_counts;
  // Alias records introduced by scheduled storage-relation results.
  loom_low_schedule_pressure_alias_t* aliases;
  // Number of touched register-class IDs for the current candidate.
  iree_host_size_t candidate_delta_touched_count;
  // Current aggregate live register units in the simulated schedule.
  uint64_t current_live_units;
  // Pressure-cliff penalty for the current simulated schedule state.
  uint32_t current_pressure_cliff_penalty;
  // Number of populated entries in |block_reg_class_ids|.
  iree_host_size_t block_reg_class_count;
  // Number of populated entries in |block_value_ordinals|.
  iree_host_size_t block_value_count;
  // Number of populated entries in |candidate_operand_ordinals|.
  iree_host_size_t candidate_operand_count;
  // Number of populated entries in |aliases|.
  iree_host_size_t alias_count;
  // Number of populated entries in |alias_source_ordinals|.
  iree_host_size_t alias_source_count;
  // Allocated alias record capacity.
  iree_host_size_t alias_capacity;
} loom_low_schedule_pressure_state_t;

typedef struct loom_low_schedule_ready_policy_t {
  // Shared ready membership and nomination heaps.
  loom_low_schedule_ready_frontier_t frontier;
  // Remaining distinct consumer count, dense by local value ordinal.
  uint32_t* remaining_consumer_counts;
  // XOR of remaining consumer node indices, dense by local value ordinal.
  uint32_t* remaining_consumer_node_xors;
} loom_low_schedule_ready_policy_t;

typedef enum loom_low_schedule_ready_membership_change_e {
  LOOM_LOW_SCHEDULE_READY_MEMBERSHIP_REMOVE = 0,
  LOOM_LOW_SCHEDULE_READY_MEMBERSHIP_INSERT = 1,
} loom_low_schedule_ready_membership_change_t;

enum loom_low_schedule_pressure_source_kind_e {
  LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_NONE = 0,
  LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS = 1,
  LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_RESOURCE = 2,
};
typedef uint8_t loom_low_schedule_pressure_source_kind_t;

typedef struct loom_low_schedule_candidate_score_t {
  // Aggregate live register units after scheduling the candidate.
  uint64_t projected_live_units;
  // Live register units whose last use is the candidate.
  uint64_t killed_live_units;
  // Register result units made live by the candidate.
  uint64_t produced_live_units;
  // Projected pressure-cliff penalty relative to the current schedule state.
  int64_t pressure_cliff_penalty_delta;
  // Live register values whose last use is the candidate.
  uint32_t killed_live_value_count;
  // Register result values made live by the candidate.
  uint32_t produced_live_value_count;
  // Longest same-block latency path starting at the candidate.
  uint32_t critical_path_cycles;
  // Downstream visible register demand reached through structural nodes.
  uint32_t pressure_demand_units;
  // Register headroom reserved to consume results opened by the candidate.
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
  // Live units remaining before the next pressure cliff, or
  // LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE.
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

typedef struct loom_low_schedule_pressure_demand_t {
  // Downstream visible register demand reached through structural nodes.
  uint32_t demand_units;
  // Maximum register activation of a descriptor made ready by the candidate.
  uint32_t activation_units;
  // Pair-affinity reward made available by the candidate.
  uint16_t pair_affinity_score;
  // Candidate properties discovered while traversing the ready frontier.
  uint8_t candidate_flags;
} loom_low_schedule_pressure_demand_t;

typedef enum loom_low_schedule_candidate_compare_mode_e {
  LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT = 0,
  LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF = 1,
} loom_low_schedule_candidate_compare_mode_t;

static iree_status_t loom_low_schedule_verify_memory_access_table(
    loom_low_memory_access_table_t table, const loom_op_t* low_func_op,
    const loom_region_t* body) {
  IREE_ASSERT(body != NULL);
  if (table.count == 0) {
    return iree_ok_status();
  }
  if (!table.values || table.function_op != low_func_op ||
      table.count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "low schedule memory access table must match the scheduled function");
  }
  for (iree_host_size_t i = 0; i < table.count; ++i) {
    const loom_low_memory_access_record_t* record = &table.values[i];
    IREE_ASSERT(record->op != NULL);
    IREE_ASSERT(record->position.block_index !=
                LOOM_BLOCK_REGION_INDEX_INVALID);
    IREE_ASSERT(record->position.block_index < body->block_count);
    IREE_ASSERT(record->position.block_ordinal != 0);
    if (i != 0) {
      IREE_ASSERT(loom_low_memory_access_position_compare_order(
                      &table.values[i - 1].position, &record->position) < 0);
    }
  }
  return iree_ok_status();
}

static uint32_t loom_low_schedule_positive_delta_u32(uint32_t lhs,
                                                     uint32_t rhs) {
  return lhs > rhs ? lhs - rhs : 0;
}

static uint32_t loom_low_schedule_max_u32(uint32_t lhs, uint32_t rhs) {
  return lhs > rhs ? lhs : rhs;
}

static bool loom_low_schedule_uses_pressure_strategy(
    const loom_low_schedule_build_state_t* state) {
  return state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE ||
         state->options->strategy ==
             LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING ||
         state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL;
}

static iree_status_t loom_low_schedule_initialize_value_records(
    loom_low_schedule_build_state_t* state) {
  const loom_local_value_domain_t* value_domain = state->value_domain;
  if (value_domain->value_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, value_domain->value_count, sizeof(*state->values),
      (void**)&state->values));
  for (loom_value_ordinal_t ordinal = 0; ordinal < value_domain->value_count;
       ++ordinal) {
    const loom_value_id_t value_id = value_domain->value_ids[ordinal];
    loom_low_schedule_value_record_t* value = &state->values[ordinal];
    *value = (loom_low_schedule_value_record_t){
        .value_id = value_id,
        .producer_node = LOOM_LOW_SCHEDULE_NODE_NONE,
        .register_class_id = LOOM_LOW_REG_CLASS_NONE,
    };
    const loom_type_t type = loom_module_value_type(state->module, value_id);
    if (!loom_low_type_is_register(type)) {
      continue;
    }
    if (loom_low_register_type_resolver_try_resolve(
            &state->register_type_resolver, type, &value->register_class_id,
            NULL)) {
      value->unit_count = loom_low_register_type_unit_count(type);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_initialize_storage(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_initialize_value_records(state));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, state->body->block_count, sizeof(*state->blocks),
      (void**)&state->blocks));
  memset(state->blocks, 0, state->body->block_count * sizeof(*state->blocks));
  if (node_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->arena, node_count,
                                                   sizeof(*state->nodes),
                                                   (void**)&state->nodes));
    memset(state->nodes, 0, node_count * sizeof(*state->nodes));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count, sizeof(*state->scheduled_node_indices),
        (void**)&state->scheduled_node_indices));
    const iree_host_size_t placement_pair_use_capacity = node_count / 2;
    if (state->options->pair_affinities.placement_recipe_count != 0 &&
        placement_pair_use_capacity != 0) {
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate_array(state->arena, placement_pair_use_capacity,
                                    sizeof(*state->placement_pair_uses),
                                    (void**)&state->placement_pair_uses));
    }
    if (state->options->preferred_pair_uses.count != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, node_count, sizeof(*state->preferred_pair_nodes),
          (void**)&state->preferred_pair_nodes));
      memset(state->preferred_pair_nodes, 0xFF,
             node_count * sizeof(*state->preferred_pair_nodes));
    }
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count, sizeof(*state->state_chain_read_heads),
        (void**)&state->state_chain_read_heads));
    memset(state->state_chain_read_heads, 0xFF,
           node_count * sizeof(*state->state_chain_read_heads));
    if (loom_low_schedule_uses_pressure_strategy(state) &&
        iree_any_bit_set(state->options->flags,
                         LOOM_LOW_SCHEDULE_FLAG_RETAIN_PRESSURE_STEPS)) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, node_count, sizeof(*state->pressure_steps),
          (void**)&state->pressure_steps));
    }
    if ((state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE ||
         state->options->strategy ==
             LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING ||
         state->options->strategy ==
             LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) &&
        iree_any_bit_set(state->options->diagnostic_flags,
                         LOOM_LOW_SCHEDULE_DIAGNOSTIC_CANDIDATE_DECISIONS)) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, node_count, sizeof(*state->candidate_decisions),
          (void**)&state->candidate_decisions));
    }
    if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, node_count, sizeof(*state->node_ready_issue_cycles),
          (void**)&state->node_ready_issue_cycles));
      memset(state->node_ready_issue_cycles, 0,
             node_count * sizeof(*state->node_ready_issue_cycles));
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, node_count, sizeof(*state->node_critical_path_cycles),
          (void**)&state->node_critical_path_cycles));
      memset(state->node_critical_path_cycles, 0,
             node_count * sizeof(*state->node_critical_path_cycles));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_initialize_storage_read_tables(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count) {
  bool needs_storage_read_tracking = false;
  bool needs_edge_source_worklist = false;
  iree_host_size_t max_operand_count = 0;
  for (iree_host_size_t node_index = 0; node_index < node_count; ++node_index) {
    const loom_low_schedule_node_t* node = &state->nodes[node_index];
    max_operand_count = iree_max(max_operand_count, node->operand_count);
    const loom_op_t* op = node->op;
    const loom_tied_result_t* tied_results = loom_op_tied_results(op);
    const loom_value_ordinal_t* operand_ordinals =
        loom_low_schedule_node_const_operand_ordinals(node);
    for (uint16_t i = 0; i < op->tied_result_count; ++i) {
      const loom_tied_result_t tied = tied_results[i];
      IREE_ASSERT_LT(tied.operand_index, node->operand_count);
      state->values[operand_ordinals[tied.operand_index]].flags |=
          LOOM_LOW_SCHEDULE_VALUE_FLAG_STORAGE_READ_TRACKED;
      needs_storage_read_tracking = true;
    }
    const uint16_t relation_count =
        loom_low_storage_relation_count(state->module, op);
    state->storage_reads.relation_count += relation_count;
    loom_low_storage_relation_iterator_t iterator;
    loom_low_storage_relation_iterator_initialize(state->module, op, &iterator);
    loom_low_storage_relation_t relation;
    while (loom_low_storage_relation_iterator_next(&iterator, &relation)) {
      if (relation.cause == LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_BRANCH ||
          relation.cause == LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_YIELD) {
        const loom_value_ordinal_t destination_ordinal =
            loom_local_value_domain_ordinal(state->value_domain,
                                            relation.destination_value_id);
        state->values[destination_ordinal].flags |=
            LOOM_LOW_SCHEDULE_VALUE_FLAG_STORAGE_READ_TRACKED;
        needs_storage_read_tracking = true;
        needs_edge_source_worklist = true;
      }
    }
  }
  if (!needs_storage_read_tracking || state->value_domain->value_count == 0) {
    return iree_ok_status();
  }
  const loom_value_ordinal_t value_count = state->value_domain->value_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, value_count, sizeof(*state->storage_reads.heads),
      (void**)&state->storage_reads.heads));
  memset(state->storage_reads.heads, 0xFF,
         value_count * sizeof(*state->storage_reads.heads));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, value_count, sizeof(*state->storage_reads.touched_ordinals),
      (void**)&state->storage_reads.touched_ordinals));
  if (max_operand_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, max_operand_count,
        sizeof(*state->storage_reads.operand_relation_flags),
        (void**)&state->storage_reads.operand_relation_flags));
    state->storage_reads.operand_relation_flag_capacity = max_operand_count;
  }
  if (needs_edge_source_worklist) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, value_count,
        sizeof(*state->storage_reads.edge_source_worklist),
        (void**)&state->storage_reads.edge_source_worklist));
    state->storage_reads.edge_source_worklist_capacity = value_count;
  }
  return iree_ok_status();
}

static uint32_t loom_low_schedule_descriptor_ordinal(
    const loom_low_schedule_build_state_t* state,
    const loom_low_descriptor_t* descriptor) {
  return loom_low_descriptor_set_descriptor_ordinal(
      state->target.descriptor_set, descriptor);
}

static iree_status_t loom_low_schedule_initialize_pair_affinity_index(
    loom_low_schedule_build_state_t* state) {
  const loom_low_schedule_pair_affinity_list_t affinities =
      state->options->pair_affinities;
  if (loom_low_schedule_pair_affinity_list_is_empty(affinities)) {
    return iree_ok_status();
  }
  IREE_ASSERT(affinities.placement_recipe_count == 0 ||
              affinities.placement_recipes != NULL);
  IREE_ASSERT_LE(affinities.count, UINT32_MAX);
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  if (descriptor_set->descriptor_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->arena, descriptor_set->descriptor_count,
                                sizeof(*state->pair_affinity_heads),
                                (void**)&state->pair_affinity_heads));
  memset(
      state->pair_affinity_heads, 0xFF,
      descriptor_set->descriptor_count * sizeof(*state->pair_affinity_heads));

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, affinities.count, sizeof(*state->pair_affinity_records),
      (void**)&state->pair_affinity_records));
  for (iree_host_size_t i = 0; i < affinities.count; ++i) {
    const loom_low_schedule_pair_affinity_t* affinity = &affinities.values[i];
    if (affinity->priority == 0) {
      continue;
    }
    const uint32_t first_ordinal =
        loom_low_schedule_descriptor_ordinal(state, affinity->first_descriptor);
    const uint32_t second_ordinal = loom_low_schedule_descriptor_ordinal(
        state, affinity->second_descriptor);
    if (first_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE ||
        second_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
      continue;
    }
    if (affinity->placement_recipe_index !=
        LOOM_LOW_PLACEMENT_PAIR_RECIPE_NONE) {
      const uint16_t recipe_index =
          (uint16_t)(affinity->placement_recipe_index - 1u);
      IREE_ASSERT_LT(recipe_index, affinities.placement_recipe_count);
    }
    const uint32_t record_index = (uint32_t)state->pair_affinity_record_count++;
    state->pair_affinity_records[record_index] =
        (loom_low_schedule_pair_affinity_record_t){
            .first_descriptor_ordinal = first_ordinal,
            .second_descriptor_ordinal = second_ordinal,
            .next_record = state->pair_affinity_heads[first_ordinal],
            .reverse_next_record = LOOM_LOW_SCHEDULE_PAIR_AFFINITY_RECORD_NONE,
            .priority = affinity->priority,
            .placement_recipe_index = affinity->placement_recipe_index,
        };
    state->pair_affinity_heads[first_ordinal] = record_index;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_initialize_pair_setup_index(
    loom_low_schedule_build_state_t* state) {
  if (state->detached_copy_node_count == 0 ||
      state->pair_affinity_record_count == 0) {
    return iree_ok_status();
  }
  const uint32_t descriptor_count =
      state->target.descriptor_set->descriptor_count;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->arena, descriptor_count,
                                sizeof(*state->pair_affinity_reverse_heads),
                                (void**)&state->pair_affinity_reverse_heads));
  memset(state->pair_affinity_reverse_heads, 0xFF,
         descriptor_count * sizeof(*state->pair_affinity_reverse_heads));
  for (uint32_t record_index = 0;
       record_index < state->pair_affinity_record_count; ++record_index) {
    loom_low_schedule_pair_affinity_record_t* record =
        &state->pair_affinity_records[record_index];
    record->reverse_next_record =
        state->pair_affinity_reverse_heads[record->second_descriptor_ordinal];
    state->pair_affinity_reverse_heads[record->second_descriptor_ordinal] =
        record_index;
  }
  return iree_ok_status();
}

static void loom_low_schedule_record_pressure_limit(uint32_t* existing_limit,
                                                    uint32_t limit_units) {
  if (*existing_limit == UINT32_MAX || limit_units < *existing_limit) {
    *existing_limit = limit_units;
  }
}

static iree_status_t loom_low_schedule_initialize_pressure_limits(
    loom_low_schedule_build_state_t* state) {
  const bool uses_pressure_strategy =
      state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE ||
      state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING ||
      state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL;
  const bool emits_pressure_diagnostics =
      iree_any_bit_set(state->options->diagnostic_flags,
                       LOOM_LOW_SCHEDULE_DIAGNOSTIC_PRESSURE_PEAKS);
  if (!uses_pressure_strategy && !emits_pressure_diagnostics) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  if (descriptor_set->reg_class_count == 0) {
    return iree_ok_status();
  }
  bool has_pressure_limit = state->options->allocation_budget_count != 0;
  uint16_t alias_set_count = 0;
  for (uint32_t reg_class_id = 0;
       reg_class_id < descriptor_set->reg_class_count; ++reg_class_id) {
    const loom_low_reg_class_t* reg_class =
        &descriptor_set->reg_classes[reg_class_id];
    has_pressure_limit |= reg_class->allocatable_count != 0;
    alias_set_count = iree_max(alias_set_count, reg_class->alias_set_id);
  }
  if (!has_pressure_limit) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->arena, descriptor_set->reg_class_count,
                                sizeof(*state->pressure_limits.by_reg_class),
                                (void**)&state->pressure_limits.by_reg_class));
  memset(state->pressure_limits.by_reg_class, 0xFF,
         descriptor_set->reg_class_count *
             sizeof(*state->pressure_limits.by_reg_class));

  state->pressure_limits.alias_set_count = alias_set_count;
  if (alias_set_count != 0) {
    const iree_host_size_t alias_set_slot_count =
        (iree_host_size_t)alias_set_count + 1;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->arena, alias_set_slot_count,
                                  sizeof(*state->pressure_limits.alias_sets),
                                  (void**)&state->pressure_limits.alias_sets));
    memset(state->pressure_limits.alias_sets, 0xFF,
           alias_set_slot_count * sizeof(*state->pressure_limits.alias_sets));
  }

  for (uint32_t reg_class_id = 0;
       reg_class_id < descriptor_set->reg_class_count; ++reg_class_id) {
    const loom_low_reg_class_t* reg_class =
        &descriptor_set->reg_classes[reg_class_id];
    const uint16_t alias_set_id = reg_class->alias_set_id;
    if (alias_set_id != 0 &&
        state->pressure_limits.alias_sets[alias_set_id]
                .representative_reg_class_id == LOOM_LOW_REG_CLASS_NONE) {
      state->pressure_limits.alias_sets[alias_set_id]
          .representative_reg_class_id = (uint16_t)reg_class_id;
    }
    const uint32_t allocatable_count = reg_class->allocatable_count;
    if (allocatable_count == 0) {
      continue;
    }
    if (alias_set_id != 0) {
      loom_low_schedule_record_pressure_limit(
          &state->pressure_limits.alias_sets[alias_set_id].live_unit_limit,
          allocatable_count);
    } else {
      loom_low_schedule_record_pressure_limit(
          &state->pressure_limits.by_reg_class[reg_class_id],
          allocatable_count);
    }
  }

  for (iree_host_size_t i = 0; i < state->options->allocation_budget_count;
       ++i) {
    const loom_low_allocation_budget_t* budget =
        &state->options->allocation_budgets[i];
    uint16_t budget_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
    if (!loom_low_descriptor_set_lookup_register_class(
            descriptor_set, budget->register_class, &budget_reg_class_id,
            NULL)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "low schedule allocation budget references unknown register class "
          "'%.*s'",
          (int)budget->register_class.size, budget->register_class.data);
    }
    const uint16_t alias_set_id =
        descriptor_set->reg_classes[budget_reg_class_id].alias_set_id;
    if (alias_set_id != 0) {
      loom_low_schedule_record_pressure_limit(
          &state->pressure_limits.alias_sets[alias_set_id].live_unit_limit,
          budget->max_units);
    } else {
      loom_low_schedule_record_pressure_limit(
          &state->pressure_limits.by_reg_class[budget_reg_class_id],
          budget->max_units);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_verify_structural_state_reads(
    loom_low_schedule_build_state_t* state) {
  if (loom_low_schedule_structural_state_read_list_is_empty(
          state->options->structural_state_reads)) {
    return iree_ok_status();
  }
  if (state->options->structural_state_reads.values == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low schedule structural state reads require table rows");
  }
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  for (iree_host_size_t i = 0; i < state->options->structural_state_reads.count;
       ++i) {
    const loom_low_schedule_structural_state_read_t* row =
        &state->options->structural_state_reads.values[i];
    if (row->result_reg_class_id >= descriptor_set->reg_class_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low schedule structural state read references invalid result "
          "register class %" PRIu16,
          row->result_reg_class_id);
    }
    if (row->state_reg_class_id >= descriptor_set->reg_class_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low schedule structural state read references invalid state "
          "register class %" PRIu16,
          row->state_reg_class_id);
    }
    if (state->reg_class_state_flags == NULL ||
        state->reg_class_state_flags[row->state_reg_class_id] == 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low schedule structural state read references non-state register "
          "class %" PRIu16,
          row->state_reg_class_id);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_initialize_descriptor_tables(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count) {
  iree_host_size_t effect_use_capacity = 0;
  iree_host_size_t hazard_use_capacity = 0;
  bool has_state_reg_class = false;
  bool has_resource_uses = false;
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  const iree_host_size_t reg_class_count = descriptor_set->reg_class_count;
  const iree_host_size_t resource_count = descriptor_set->resource_count;
  if (reg_class_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count, sizeof(*state->reg_class_state_flags),
        (void**)&state->reg_class_state_flags));
    memset(state->reg_class_state_flags, 0,
           reg_class_count * sizeof(*state->reg_class_state_flags));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count, sizeof(*state->state_last_write_nodes),
        (void**)&state->state_last_write_nodes));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count,
        sizeof(*state->state_ordering_frontier_nodes),
        (void**)&state->state_ordering_frontier_nodes));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count, sizeof(*state->state_read_heads),
        (void**)&state->state_read_heads));
  }
  IREE_RETURN_IF_ERROR(loom_low_schedule_initialize_pressure_limits(state));
  for (uint32_t operand_index = 0;
       operand_index < descriptor_set->operand_count; ++operand_index) {
    const loom_low_operand_t* operand =
        &descriptor_set->operands[operand_index];
    uint8_t access_flags = 0;
    if (iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_STATE_READ)) {
      access_flags |= LOOM_LOW_SCHEDULE_STATE_ACCESS_READ;
    }
    if (iree_any_bit_set(operand->flags, LOOM_LOW_OPERAND_FLAG_STATE_WRITE)) {
      access_flags |= LOOM_LOW_SCHEDULE_STATE_ACCESS_WRITE;
    }
    if (access_flags == 0) {
      continue;
    }
    const uint32_t alt_index = operand->reg_class_alt_start;
    if (alt_index >= descriptor_set->reg_class_alt_count) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low schedule state operand register-class alternative is out of "
          "range");
    }
    const loom_low_reg_class_alt_t* alt =
        &descriptor_set->reg_class_alts[alt_index];
    if (alt->reg_class_id >= descriptor_set->reg_class_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low schedule state operand register class is "
                              "out of range");
    }
    state->reg_class_state_flags[alt->reg_class_id] |= access_flags;
    has_state_reg_class = true;
  }
  if (has_state_reg_class && node_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count,
        sizeof(*state->state_last_dependency_consumer_nodes),
        (void**)&state->state_last_dependency_consumer_nodes));
    memset(state->state_last_dependency_consumer_nodes, 0xFF,
           node_count * sizeof(*state->state_last_dependency_consumer_nodes));
  }
  IREE_RETURN_IF_ERROR(loom_low_schedule_verify_structural_state_reads(state));
  for (iree_host_size_t node_index = 0; node_index < node_count; ++node_index) {
    const loom_low_schedule_node_t* node = &state->nodes[node_index];
    const loom_low_schedule_class_t* schedule_class = node->schedule_class;
    has_resource_uses |=
        schedule_class != NULL && schedule_class->issue_use_count != 0;
    if (!iree_host_size_checked_add(
            effect_use_capacity,
            node->descriptor ? node->descriptor->effect_count : 0,
            &effect_use_capacity)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low schedule effect-use capacity exceeds host size");
    }
    if (!iree_host_size_checked_add(
            hazard_use_capacity,
            schedule_class ? schedule_class->hazard_count : 0,
            &hazard_use_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "low schedule hazard-use capacity exceeds host "
                              "size");
    }
  }
  if (node_count != 0 && descriptor_set->schedule_class_count != 0) {
    const uint32_t schedule_class_count = descriptor_set->schedule_class_count;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, schedule_class_count, sizeof(*state->model_summaries),
        (void**)&state->model_summaries));
    memset(state->model_summaries, 0,
           schedule_class_count * sizeof(*state->model_summaries));
  }
  if (has_resource_uses) {
    IREE_ASSERT(resource_count <= UINT16_MAX);
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, resource_count, sizeof(*state->resource_summaries),
        (void**)&state->resource_summaries));
    memset(state->resource_summaries, 0,
           resource_count * sizeof(*state->resource_summaries));
    for (iree_host_size_t i = 0; i < resource_count; ++i) {
      const loom_low_resource_t* resource = &descriptor_set->resources[i];
      IREE_ASSERT(resource->capacity_per_cycle != 0);
      iree_string_view_t resource_name = loom_low_descriptor_set_string(
          descriptor_set, resource->name_string_offset);
      state->resource_summaries[i] = (loom_low_schedule_resource_summary_t){
          .resource_id = (uint16_t)i,
          .resource_name = resource_name,
          .resource_kind = resource->kind,
          .resource_flags = resource->flags,
          .capacity_per_cycle = resource->capacity_per_cycle,
          .contention_group_id = resource->contention_group_id,
      };
    }
  }
  if (resource_count != 0 &&
      state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->arena, resource_count,
                                  sizeof(*state->resource_ready_issue_cycles),
                                  (void**)&state->resource_ready_issue_cycles));
    memset(state->resource_ready_issue_cycles, 0,
           resource_count * sizeof(*state->resource_ready_issue_cycles));
  }
  if (effect_use_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, effect_use_capacity, sizeof(*state->effect_uses),
        (void**)&state->effect_uses));
    state->effect_use_capacity = effect_use_capacity;
  }
  iree_host_size_t effect_read_capacity = 0;
  if (!iree_host_size_checked_add(effect_use_capacity, node_count,
                                  &effect_read_capacity)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "low schedule effect-frontier read capacity exceeds host size");
  }
  if (effect_read_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, effect_read_capacity, sizeof(*state->effect_read_nodes),
        (void**)&state->effect_read_nodes));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->arena, effect_read_capacity,
                                  sizeof(*state->effect_read_summaries),
                                  (void**)&state->effect_read_summaries));
    state->effect_read_capacity = effect_read_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, effect_read_capacity, sizeof(*state->effect_write_nodes),
        (void**)&state->effect_write_nodes));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->arena, effect_read_capacity,
                                  sizeof(*state->effect_write_summaries),
                                  (void**)&state->effect_write_summaries));
    state->effect_write_capacity = effect_read_capacity;
  }
  if (hazard_use_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, hazard_use_capacity, sizeof(*state->hazard_uses),
        (void**)&state->hazard_uses));
    state->hazard_use_capacity = hazard_use_capacity;
  }
  return iree_ok_status();
}

static bool loom_low_schedule_strategy_is_valid(
    loom_low_schedule_strategy_t strategy) {
  switch (strategy) {
    case LOOM_LOW_SCHEDULE_STRATEGY_SOURCE_PRIORITY:
    case LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE:
    case LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING:
    case LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL:
      return true;
    default:
      return false;
  }
}

static iree_status_t loom_low_schedule_allocate_pressure_state(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count,
    loom_low_schedule_pressure_state_t* out_pressure_state) {
  *out_pressure_state = (loom_low_schedule_pressure_state_t){0};
  if (!loom_low_schedule_uses_pressure_strategy(state)) {
    return iree_ok_status();
  }
  const loom_value_ordinal_t value_count = state->value_domain->value_count;
  if (value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, value_count,
        sizeof(*out_pressure_state->block_value_ordinals),
        (void**)&out_pressure_state->block_value_ordinals));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, value_count,
        sizeof(*out_pressure_state->candidate_operand_use_counts),
        (void**)&out_pressure_state->candidate_operand_use_counts));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, value_count,
        sizeof(*out_pressure_state->candidate_operand_ordinals),
        (void**)&out_pressure_state->candidate_operand_ordinals));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, value_count,
        sizeof(*out_pressure_state->candidate_scratch_counts),
        (void**)&out_pressure_state->candidate_scratch_counts));
    memset(out_pressure_state->candidate_operand_use_counts, 0,
           value_count *
               sizeof(*out_pressure_state->candidate_operand_use_counts));
    memset(out_pressure_state->candidate_scratch_counts, 0,
           value_count * sizeof(*out_pressure_state->candidate_scratch_counts));
    const iree_host_size_t alias_capacity = state->storage_reads.relation_count;
    if (alias_capacity != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, value_count, sizeof(*out_pressure_state->alias_heads),
          (void**)&out_pressure_state->alias_heads));
      memset(out_pressure_state->alias_heads, 0xFF,
             value_count * sizeof(*out_pressure_state->alias_heads));
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, value_count,
          sizeof(*out_pressure_state->alias_source_ordinals),
          (void**)&out_pressure_state->alias_source_ordinals));
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, value_count,
          sizeof(*out_pressure_state->alias_source_unit_counts),
          (void**)&out_pressure_state->alias_source_unit_counts));
      memset(
          out_pressure_state->alias_source_unit_counts, 0,
          value_count * sizeof(*out_pressure_state->alias_source_unit_counts));
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, alias_capacity, sizeof(*out_pressure_state->aliases),
          (void**)&out_pressure_state->aliases));
      out_pressure_state->alias_capacity = alias_capacity;
    }
  }
  const uint32_t reg_class_count =
      state->target.descriptor_set->reg_class_count;
  if (reg_class_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count,
        sizeof(*out_pressure_state->current_live_units_by_reg_class),
        (void**)&out_pressure_state->current_live_units_by_reg_class));
    memset(out_pressure_state->current_live_units_by_reg_class, 0,
           reg_class_count *
               sizeof(*out_pressure_state->current_live_units_by_reg_class));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count,
        sizeof(*out_pressure_state->block_reg_class_ids),
        (void**)&out_pressure_state->block_reg_class_ids));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count,
        sizeof(*out_pressure_state->block_reg_class_touched_flags),
        (void**)&out_pressure_state->block_reg_class_touched_flags));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count,
        sizeof(*out_pressure_state->candidate_delta_units_by_reg_class),
        (void**)&out_pressure_state->candidate_delta_units_by_reg_class));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count,
        sizeof(*out_pressure_state->candidate_delta_touched_flags),
        (void**)&out_pressure_state->candidate_delta_touched_flags));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count,
        sizeof(*out_pressure_state->candidate_delta_touched_reg_class_ids),
        (void**)&out_pressure_state->candidate_delta_touched_reg_class_ids));
    if (state->pressure_cliffs != NULL &&
        !loom_low_pressure_cliff_table_is_empty(*state->pressure_cliffs)) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, reg_class_count,
          sizeof(*out_pressure_state->first_actionable_pressure_cliff_indices),
          (void**)&out_pressure_state
              ->first_actionable_pressure_cliff_indices));
      for (uint16_t reg_class_id = 0; reg_class_id < reg_class_count;
           ++reg_class_id) {
        const loom_low_pressure_cliff_range_t range =
            loom_low_pressure_cliff_table_range(state->pressure_cliffs,
                                                reg_class_id);
        out_pressure_state
            ->first_actionable_pressure_cliff_indices[reg_class_id] =
            range.start;
      }
    }
    memset(out_pressure_state->candidate_delta_units_by_reg_class, 0,
           reg_class_count *
               sizeof(*out_pressure_state->candidate_delta_units_by_reg_class));
    memset(out_pressure_state->block_reg_class_touched_flags, 0,
           reg_class_count *
               sizeof(*out_pressure_state->block_reg_class_touched_flags));
    memset(out_pressure_state->candidate_delta_touched_flags, 0,
           reg_class_count *
               sizeof(*out_pressure_state->candidate_delta_touched_flags));
  }
  const uint16_t alias_set_count = state->pressure_limits.alias_set_count;
  if (alias_set_count != 0) {
    const iree_host_size_t alias_set_slot_count =
        (iree_host_size_t)alias_set_count + 1;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, alias_set_slot_count,
        sizeof(*out_pressure_state->alias_sets.records),
        (void**)&out_pressure_state->alias_sets.records));
    memset(
        out_pressure_state->alias_sets.records, 0,
        alias_set_slot_count * sizeof(*out_pressure_state->alias_sets.records));
    uint16_t* touched_ids = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, (iree_host_size_t)alias_set_count * 2,
        sizeof(*touched_ids), (void**)&touched_ids));
    out_pressure_state->alias_sets.block_ids = touched_ids;
    out_pressure_state->alias_sets.candidate_delta_touched_ids =
        touched_ids + alias_set_count;
  }
  if (state->pressure_resources != NULL) {
    const uint16_t resource_count = state->pressure_resources->resource_count;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count,
        sizeof(*out_pressure_state->resources.peak_live_units_by_reg_class),
        (void**)&out_pressure_state->resources.peak_live_units_by_reg_class));
    memset(
        out_pressure_state->resources.peak_live_units_by_reg_class, 0,
        reg_class_count *
            sizeof(
                *out_pressure_state->resources.peak_live_units_by_reg_class));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, resource_count,
        sizeof(*out_pressure_state->resources.records),
        (void**)&out_pressure_state->resources.records));
    memset(out_pressure_state->resources.records, 0,
           resource_count * sizeof(*out_pressure_state->resources.records));
    for (uint16_t resource_id = 0; resource_id < resource_count;
         ++resource_id) {
      out_pressure_state->resources.records[resource_id].next_cliff_index =
          state->pressure_resources->resources[resource_id].cliff_start;
    }
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, resource_count,
        sizeof(*out_pressure_state->resources.candidate_touched_ids),
        (void**)&out_pressure_state->resources.candidate_touched_ids));
  }
  const uint32_t descriptor_count =
      state->target.descriptor_set->descriptor_count;
  if (state->pair_affinity_reverse_heads != NULL &&
      state->detached_copy_node_count != 0 && descriptor_count != 0) {
    if (node_count != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, node_count,
          sizeof(*out_pressure_state->ready_setup_dependency_counts),
          (void**)&out_pressure_state->ready_setup_dependency_counts));
      memset(out_pressure_state->ready_setup_dependency_counts, 0,
             node_count *
                 sizeof(*out_pressure_state->ready_setup_dependency_counts));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_ready_policy_initialize(
    loom_low_schedule_build_state_t* state, uint32_t node_count,
    loom_low_schedule_ready_policy_t* out_policy) {
  *out_policy = (loom_low_schedule_ready_policy_t){0};
  const uint8_t view_count = loom_low_schedule_uses_pressure_strategy(state)
                                 ? LOOM_LOW_SCHEDULE_READY_VIEW_COUNT
                                 : 1;
  IREE_RETURN_IF_ERROR(loom_low_schedule_ready_frontier_initialize(
      node_count, state->target.descriptor_set->descriptor_count, view_count,
      state->arena, &out_policy->frontier));
  if (loom_low_schedule_uses_pressure_strategy(state) &&
      state->value_domain->value_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, state->value_domain->value_count,
        sizeof(*out_policy->remaining_consumer_counts),
        (void**)&out_policy->remaining_consumer_counts));
    memset(out_policy->remaining_consumer_counts, 0,
           state->value_domain->value_count *
               sizeof(*out_policy->remaining_consumer_counts));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, state->value_domain->value_count,
        sizeof(*out_policy->remaining_consumer_node_xors),
        (void**)&out_policy->remaining_consumer_node_xors));
    memset(out_policy->remaining_consumer_node_xors, 0,
           state->value_domain->value_count *
               sizeof(*out_policy->remaining_consumer_node_xors));
  }
  return iree_ok_status();
}

static void loom_low_schedule_note_block_pressure_reg_class(
    loom_low_schedule_pressure_state_t* pressure_state, uint16_t reg_class_id) {
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE ||
      pressure_state->block_reg_class_touched_flags == NULL ||
      pressure_state->block_reg_class_touched_flags[reg_class_id]) {
    return;
  }
  pressure_state->block_reg_class_touched_flags[reg_class_id] = 1;
  pressure_state->block_reg_class_ids[pressure_state->block_reg_class_count++] =
      reg_class_id;
}

static inline uint16_t loom_low_schedule_alias_set_id(
    const loom_low_schedule_build_state_t* state, uint16_t reg_class_id) {
  return reg_class_id == LOOM_LOW_REG_CLASS_NONE
             ? 0
             : state->target.descriptor_set->reg_classes[reg_class_id]
                   .alias_set_id;
}

static void loom_low_schedule_note_block_pressure_alias_set(
    loom_low_schedule_pressure_state_t* pressure_state, uint16_t alias_set_id) {
  if (alias_set_id == 0 || pressure_state->alias_sets.records == NULL) {
    return;
  }
  loom_low_schedule_alias_pressure_record_t* record =
      &pressure_state->alias_sets.records[alias_set_id];
  if (iree_any_bit_set(record->flags,
                       LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_BLOCK_TOUCHED)) {
    return;
  }
  record->flags |= LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_BLOCK_TOUCHED;
  pressure_state->alias_sets
      .block_ids[pressure_state->alias_sets.block_count++] = alias_set_id;
}

static inline void loom_low_schedule_adjust_alias_pressure(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint16_t reg_class_id,
    int64_t delta_units) {
  if (pressure_state->alias_sets.records == NULL || delta_units == 0) {
    return;
  }
  const uint16_t alias_set_id =
      loom_low_schedule_alias_set_id(state, reg_class_id);
  if (alias_set_id == 0) {
    return;
  }
  loom_low_schedule_alias_pressure_record_t* record =
      &pressure_state->alias_sets.records[alias_set_id];
  if (delta_units < 0) {
    const uint64_t removed_units = (uint64_t)-delta_units;
    IREE_ASSERT_LE(removed_units, record->current_live_units);
    record->current_live_units -= removed_units;
  } else {
    loom_low_schedule_note_block_pressure_alias_set(pressure_state,
                                                    alias_set_id);
    record->current_live_units += (uint64_t)delta_units;
  }
}

static void loom_low_schedule_reset_block_alias_pressure(
    loom_low_schedule_pressure_state_t* pressure_state) {
  for (iree_host_size_t i = 0; i < pressure_state->alias_sets.block_count;
       ++i) {
    loom_low_schedule_alias_pressure_record_t* record =
        &pressure_state->alias_sets
             .records[pressure_state->alias_sets.block_ids[i]];
    record->current_live_units = 0;
    record->flags &= ~LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_BLOCK_TOUCHED;
  }
  pressure_state->alias_sets.block_count = 0;
}

static void loom_low_schedule_advance_resource_cliffs(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint16_t resource_id,
    loom_low_schedule_resource_high_water_mode_t mode) {
  const loom_low_pressure_resource_t* resource =
      &state->pressure_resources->resources[resource_id];
  loom_low_schedule_resource_pressure_record_t* record =
      &pressure_state->resources.records[resource_id];
  const uint16_t cliff_end = resource->cliff_start + resource->cliff_count;
  while (record->next_cliff_index < cliff_end) {
    const loom_low_pressure_cliff_t* cliff =
        &state->pressure_resources->cliffs[record->next_cliff_index];
    if (cliff->cliff_units > record->current_peak_units) break;
    if (mode == LOOM_LOW_SCHEDULE_RESOURCE_HIGH_WATER_SCHEDULED) {
      const uint32_t penalty = cliff->tier_before - cliff->tier_after;
      record->pressure_cliff_penalty =
          iree_math_saturating_add_u32(record->pressure_cliff_penalty, penalty);
      pressure_state->resources.pressure_cliff_penalty =
          iree_math_saturating_add_u32(
              pressure_state->resources.pressure_cliff_penalty, penalty);
    }
    ++record->next_cliff_index;
  }
}

static void loom_low_schedule_note_resource_high_water(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint16_t reg_class_id,
    loom_low_schedule_resource_high_water_mode_t mode) {
  if (state->pressure_resources == NULL) return;
  const uint64_t current_live_units =
      pressure_state->current_live_units_by_reg_class[reg_class_id];
  uint64_t* peak_live_units =
      &pressure_state->resources.peak_live_units_by_reg_class[reg_class_id];
  if (current_live_units <= *peak_live_units) return;
  const uint64_t previous_peak_live_units = *peak_live_units;
  *peak_live_units = current_live_units;
  const loom_low_pressure_resource_member_range_t range =
      loom_low_pressure_resource_table_member_range(state->pressure_resources,
                                                    reg_class_id);
  for (uint16_t i = 0; i < range.count; ++i) {
    const uint16_t member_index =
        state->pressure_resources->member_indices_by_reg_class[range.start + i];
    const loom_low_pressure_resource_member_t* member =
        &state->pressure_resources->members[member_index];
    IREE_ASSERT_EQ(member->descriptor_reg_class_id, reg_class_id);
    const uint64_t previous_contribution =
        loom_low_pressure_round_resource_units(
            previous_peak_live_units, member->contribution_granularity);
    const uint64_t current_contribution =
        loom_low_pressure_round_resource_units(
            current_live_units, member->contribution_granularity);
    loom_low_schedule_resource_pressure_record_t* record =
        &pressure_state->resources.records[member->resource_id];
    record->current_peak_units = iree_math_saturating_add_u64(
        record->current_peak_units,
        current_contribution - previous_contribution);
    loom_low_schedule_advance_resource_cliffs(state, pressure_state,
                                              member->resource_id, mode);
  }
}

static void loom_low_schedule_reset_source_resource_pressure(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state) {
  if (state->pressure_resources == NULL) return;
  memset(pressure_state->resources.peak_live_units_by_reg_class, 0,
         state->target.descriptor_set->reg_class_count *
             sizeof(*pressure_state->resources.peak_live_units_by_reg_class));
  for (uint16_t resource_id = 0;
       resource_id < state->pressure_resources->resource_count; ++resource_id) {
    loom_low_schedule_resource_pressure_record_t* record =
        &pressure_state->resources.records[resource_id];
    record->current_peak_units = 0;
    record->candidate_added_units = 0;
    record->pressure_cliff_penalty = 0;
    record->flags = 0;
  }
  pressure_state->resources.candidate_touched_count = 0;
  pressure_state->resources.pressure_cliff_penalty = 0;
}

static void loom_low_schedule_advance_source_pressure_cliff_floor(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint16_t reg_class_id) {
  if (pressure_state->first_actionable_pressure_cliff_indices == NULL) {
    return;
  }
  const loom_low_pressure_cliff_range_t range =
      loom_low_pressure_cliff_table_range(state->pressure_cliffs, reg_class_id);
  const uint32_t range_end = range.start + range.count;
  uint32_t* first_actionable_cliff =
      &pressure_state->first_actionable_pressure_cliff_indices[reg_class_id];
  const uint64_t source_live_units =
      pressure_state->current_live_units_by_reg_class[reg_class_id];
  while (*first_actionable_cliff < range_end &&
         state->pressure_cliffs->values[*first_actionable_cliff].cliff_units <=
             source_live_units) {
    ++*first_actionable_cliff;
  }
}

static void loom_low_schedule_remove_source_pressure_value(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t value_ordinal) {
  loom_low_schedule_value_record_t* value = &state->values[value_ordinal];
  if (!iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
    return;
  }
  value->flags &= ~LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
  const uint32_t unit_count = value->live_unit_count;
  value->live_unit_count = 0;
  IREE_ASSERT_LE(unit_count, pressure_state->current_live_units);
  pressure_state->current_live_units -= unit_count;
  const uint16_t reg_class_id = value->register_class_id;
  if (reg_class_id != LOOM_LOW_REG_CLASS_NONE) {
    IREE_ASSERT_LE(
        unit_count,
        pressure_state->current_live_units_by_reg_class[reg_class_id]);
    pressure_state->current_live_units_by_reg_class[reg_class_id] -= unit_count;
    loom_low_schedule_adjust_alias_pressure(state, pressure_state, reg_class_id,
                                            -(int64_t)unit_count);
  }
}

static void loom_low_schedule_add_source_pressure_value(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t value_ordinal) {
  loom_low_schedule_value_record_t* value = &state->values[value_ordinal];
  if (iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE) ||
      value->unit_count == 0) {
    return;
  }
  value->flags |= LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
  value->live_unit_count = value->unit_count;
  pressure_state->block_value_ordinals[pressure_state->block_value_count++] =
      value_ordinal;
  pressure_state->current_live_units += value->unit_count;
  const uint16_t reg_class_id = value->register_class_id;
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE) {
    return;
  }
  loom_low_schedule_note_block_pressure_reg_class(pressure_state, reg_class_id);
  pressure_state->current_live_units_by_reg_class[reg_class_id] +=
      value->unit_count;
  loom_low_schedule_adjust_alias_pressure(state, pressure_state, reg_class_id,
                                          (int64_t)value->unit_count);
  loom_low_schedule_advance_source_pressure_cliff_floor(state, pressure_state,
                                                        reg_class_id);
  loom_low_schedule_note_resource_high_water(
      state, pressure_state, reg_class_id,
      LOOM_LOW_SCHEDULE_RESOURCE_HIGH_WATER_SOURCE_BASELINE);
}

static void loom_low_schedule_reverse_source_pressure_node(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_node_t* node) {
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t result_index = 0; result_index < node->result_count;
       ++result_index) {
    loom_low_schedule_remove_source_pressure_value(
        state, pressure_state, result_ordinals[result_index]);
  }
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  for (uint16_t operand_index = 0; operand_index < node->operand_count;
       ++operand_index) {
    loom_low_schedule_add_source_pressure_value(
        state, pressure_state, operand_ordinals[operand_index]);
  }
}

static void loom_low_schedule_remove_source_pressure_block_arguments(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_block_t* block) {
  for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
    const loom_value_ordinal_t value_ordinal = loom_local_value_domain_ordinal(
        state->value_domain, loom_block_arg_id(block, arg_index));
    loom_low_schedule_remove_source_pressure_value(state, pressure_state,
                                                   value_ordinal);
  }
}

static void loom_low_schedule_reset_source_pressure_sweep(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state) {
  for (iree_host_size_t i = 0; i < pressure_state->block_value_count; ++i) {
    loom_low_schedule_value_record_t* value =
        &state->values[pressure_state->block_value_ordinals[i]];
    value->flags &= ~LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
    value->live_unit_count = 0;
  }
  pressure_state->block_value_count = 0;
  for (iree_host_size_t i = 0; i < pressure_state->block_reg_class_count; ++i) {
    const uint16_t reg_class_id = pressure_state->block_reg_class_ids[i];
    pressure_state->block_reg_class_touched_flags[reg_class_id] = 0;
    pressure_state->current_live_units_by_reg_class[reg_class_id] = 0;
  }
  pressure_state->block_reg_class_count = 0;
  loom_low_schedule_reset_block_alias_pressure(pressure_state);
  loom_low_schedule_reset_source_resource_pressure(state, pressure_state);
  pressure_state->current_live_units = 0;
}

static void loom_low_schedule_reset_candidate_operand_uses(
    loom_low_schedule_pressure_state_t* pressure_state) {
  for (iree_host_size_t i = 0; i < pressure_state->candidate_operand_count;
       ++i) {
    const loom_value_ordinal_t value_ordinal =
        pressure_state->candidate_operand_ordinals[i];
    pressure_state->candidate_operand_use_counts[value_ordinal] = 0;
    pressure_state->candidate_scratch_counts[value_ordinal] = 0;
  }
  pressure_state->candidate_operand_count = 0;
}

static void loom_low_schedule_note_candidate_operand_use(
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t value_ordinal) {
  uint16_t* use_count =
      &pressure_state->candidate_operand_use_counts[value_ordinal];
  if (*use_count == 0) {
    pressure_state->candidate_operand_ordinals
        [pressure_state->candidate_operand_count++] = value_ordinal;
  }
  ++*use_count;
}

static uint32_t loom_low_schedule_saturate_u64_to_u32(uint64_t value) {
  return value > UINT32_MAX ? UINT32_MAX : (uint32_t)value;
}

static uint64_t loom_low_schedule_ready_pressure_key(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  uint64_t killed_units = 0;
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  for (uint16_t i = 0; i < node->operand_count; ++i) {
    loom_low_schedule_note_candidate_operand_use(pressure_state,
                                                 operand_ordinals[i]);
  }
  for (iree_host_size_t i = 0; i < pressure_state->candidate_operand_count;
       ++i) {
    const loom_value_ordinal_t value_ordinal =
        pressure_state->candidate_operand_ordinals[i];
    const loom_low_schedule_value_record_t* value =
        &state->values[value_ordinal];
    if (iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE) &&
        value->remaining_use_count ==
            pressure_state->candidate_operand_use_counts[value_ordinal]) {
      killed_units += value->live_unit_count;
    }
  }
  loom_low_schedule_reset_candidate_operand_uses(pressure_state);

  uint64_t produced_units = 0;
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t i = 0; i < node->result_count; ++i) {
    const loom_low_schedule_value_record_t* value =
        &state->values[result_ordinals[i]];
    if (value->remaining_use_count != 0 &&
        !iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
      produced_units += value->unit_count;
    }
  }
  const uint32_t growth = loom_low_schedule_saturate_u64_to_u32(
      produced_units > killed_units ? produced_units - killed_units : 0);
  const uint32_t relief = loom_low_schedule_saturate_u64_to_u32(
      killed_units > produced_units ? killed_units - produced_units : 0);
  return ((uint64_t)growth << 32) | (uint64_t)(UINT32_MAX - relief);
}

static uint64_t loom_low_schedule_ready_schedule_key(
    const loom_low_schedule_build_state_t* state, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  switch (state->options->strategy) {
    case LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL: {
      const uint32_t critical_path =
          state->node_critical_path_cycles != NULL
              ? state->node_critical_path_cycles[node_index]
              : 0;
      return UINT32_MAX - critical_path;
    }
    case LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING: {
      const uint16_t dependency_latency =
          state->node_dependency_latency_cycles != NULL
              ? state->node_dependency_latency_cycles[node_index]
              : 0;
      const uint16_t latency = node->schedule_class != NULL
                                   ? node->schedule_class->latency_cycles
                                   : 0;
      return ((uint64_t)dependency_latency << 32) |
             (uint64_t)(UINT16_MAX - latency);
    }
    default:
      return node->source_ordinal;
  }
}

static uint64_t loom_low_schedule_ready_storage_key(
    const loom_low_schedule_build_state_t* state, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  const uint16_t relation_count =
      node->op != NULL
          ? loom_low_storage_relation_count(state->module, node->op)
          : 0;
  return relation_count == 0 ? UINT64_MAX
                             : (uint64_t)(UINT16_MAX - relation_count);
}

static loom_low_schedule_ready_keys_t loom_low_schedule_ready_keys(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  return (loom_low_schedule_ready_keys_t){
      .values =
          {
              [LOOM_LOW_SCHEDULE_READY_VIEW_SOURCE] = node->source_ordinal,
              [LOOM_LOW_SCHEDULE_READY_VIEW_PRESSURE] =
                  loom_low_schedule_ready_pressure_key(state, pressure_state,
                                                       node_index),
              [LOOM_LOW_SCHEDULE_READY_VIEW_SCHEDULE] =
                  loom_low_schedule_ready_schedule_key(state, node_index),
              [LOOM_LOW_SCHEDULE_READY_VIEW_STORAGE] =
                  loom_low_schedule_ready_storage_key(state, node_index),
          },
  };
}

static void loom_low_schedule_note_block_pressure_use(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t value_ordinal, uint16_t use_count) {
  loom_low_schedule_value_record_t* value = &state->values[value_ordinal];
  IREE_ASSERT_LE(use_count, UINT32_MAX - value->remaining_use_count);
  if (value->remaining_use_count == 0) {
    pressure_state->block_value_ordinals[pressure_state->block_value_count++] =
        value_ordinal;
  }
  value->remaining_use_count += use_count;
}

static void loom_low_schedule_initialize_block_pressure(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_block_t* block_record,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_ready_policy_t* ready_policy) {
  pressure_state->current_live_units = 0;
  for (iree_host_size_t i = 0; i < pressure_state->block_reg_class_count; ++i) {
    const uint16_t reg_class_id = pressure_state->block_reg_class_ids[i];
    pressure_state->block_reg_class_touched_flags[reg_class_id] = 0;
  }
  pressure_state->block_reg_class_count = 0;
  loom_low_schedule_reset_block_alias_pressure(pressure_state);
  for (iree_host_size_t i = 0; i < pressure_state->block_value_count; ++i) {
    loom_value_ordinal_t ordinal = pressure_state->block_value_ordinals[i];
    ready_policy->remaining_consumer_counts[ordinal] = 0;
    ready_policy->remaining_consumer_node_xors[ordinal] = 0;
    state->values[ordinal].remaining_use_count = 0;
    state->values[ordinal].live_unit_count = 0;
    state->values[ordinal].flags &= ~LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
  }
  pressure_state->block_value_count = 0;
  for (iree_host_size_t i = 0; i < pressure_state->alias_source_count; ++i) {
    const loom_value_ordinal_t source_ordinal =
        pressure_state->alias_source_ordinals[i];
    pressure_state->alias_heads[source_ordinal] = LOOM_LOW_SCHEDULE_NODE_NONE;
    pressure_state->alias_source_unit_counts[source_ordinal] = 0;
  }
  pressure_state->alias_count = 0;
  pressure_state->alias_source_count = 0;
  if (pressure_state->current_live_units_by_reg_class) {
    memset(pressure_state->current_live_units_by_reg_class, 0,
           state->target.descriptor_set->reg_class_count *
               sizeof(*pressure_state->current_live_units_by_reg_class));
  }

  const uint32_t block_node_end =
      block_record->node_start + block_record->node_count;
  for (uint32_t node_index = block_record->node_start;
       node_index < block_node_end; ++node_index) {
    const loom_low_schedule_node_t* node = &state->nodes[node_index];
    const loom_value_ordinal_t* operand_ordinals =
        loom_low_schedule_node_const_operand_ordinals(node);
    for (uint16_t operand_index = 0; operand_index < node->operand_count;
         ++operand_index) {
      loom_low_schedule_note_candidate_operand_use(
          pressure_state, operand_ordinals[operand_index]);
    }
    for (iree_host_size_t i = 0; i < pressure_state->candidate_operand_count;
         ++i) {
      const loom_value_ordinal_t value_ordinal =
          pressure_state->candidate_operand_ordinals[i];
      IREE_ASSERT_NE(ready_policy->remaining_consumer_counts[value_ordinal],
                     UINT32_MAX);
      ++ready_policy->remaining_consumer_counts[value_ordinal];
      ready_policy->remaining_consumer_node_xors[value_ordinal] ^= node_index;
      loom_low_schedule_note_block_pressure_use(
          state, pressure_state, value_ordinal,
          pressure_state->candidate_operand_use_counts[value_ordinal]);
    }
    loom_low_schedule_reset_candidate_operand_uses(pressure_state);
  }

  for (iree_host_size_t i = 0; i < pressure_state->block_value_count; ++i) {
    loom_low_schedule_value_record_t* value =
        &state->values[pressure_state->block_value_ordinals[i]];
    if (value->remaining_use_count == 0) {
      continue;
    }
    const uint32_t producer_node = value->producer_node;
    if (producer_node != LOOM_LOW_SCHEDULE_NODE_NONE &&
        state->nodes[producer_node].block == block_record->block) {
      continue;
    }
    const uint32_t unit_count = value->unit_count;
    if (unit_count == 0) {
      continue;
    }
    value->flags |= LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
    pressure_state->current_live_units += unit_count;
    value->live_unit_count = unit_count;
    const uint16_t reg_class_id = value->register_class_id;
    if (reg_class_id != LOOM_LOW_REG_CLASS_NONE &&
        pressure_state->current_live_units_by_reg_class) {
      loom_low_schedule_note_block_pressure_reg_class(pressure_state,
                                                      reg_class_id);
      pressure_state->current_live_units_by_reg_class[reg_class_id] +=
          unit_count;
      loom_low_schedule_adjust_alias_pressure(
          state, pressure_state, reg_class_id, (int64_t)unit_count);
      loom_low_schedule_note_resource_high_water(
          state, pressure_state, reg_class_id,
          LOOM_LOW_SCHEDULE_RESOURCE_HIGH_WATER_SCHEDULED);
    }
  }
}

static void loom_low_schedule_score_candidate_resources(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node,
    loom_low_schedule_candidate_score_t* score) {
  score->resource_stall_cycles = 0;
  score->bottleneck_resource_id = LOOM_LOW_RESOURCE_NONE;
  const loom_low_schedule_class_t* schedule_class = node->schedule_class;
  if (state->options->strategy != LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL ||
      state->resource_ready_issue_cycles == NULL || schedule_class == NULL) {
    return;
  }
  for (uint16_t i = 0; i < schedule_class->issue_use_count; ++i) {
    const loom_low_issue_use_t* issue_use =
        &state->target.descriptor_set
             ->issue_uses[schedule_class->issue_use_start + i];
    IREE_ASSERT(issue_use->resource_id <
                state->target.descriptor_set->resource_count);
    const loom_low_resource_t* resource =
        &state->target.descriptor_set->resources[issue_use->resource_id];
    IREE_ASSERT(resource->capacity_per_cycle != 0);
    IREE_ASSERT(issue_use->units <= resource->capacity_per_cycle);
    const uint32_t use_start = iree_math_saturating_add_u32(
        state->current_issue_cycle, issue_use->stage);
    const uint32_t stall_cycles = loom_low_schedule_positive_delta_u32(
        state->resource_ready_issue_cycles[issue_use->resource_id], use_start);
    if (stall_cycles > score->resource_stall_cycles) {
      score->resource_stall_cycles = stall_cycles;
      score->bottleneck_resource_id = issue_use->resource_id;
    }
  }
}

static uint32_t loom_low_schedule_min_distance_hazard_stall(
    const loom_low_schedule_build_state_t* state,
    const loom_low_hazard_t* hazard) {
  uint32_t stall_cycles = 0;
  for (iree_host_size_t i = 0; i < state->hazard_state_count; ++i) {
    const loom_low_schedule_hazard_state_t* hazard_state =
        &state->hazard_states[i];
    if (hazard_state->kind != hazard->kind ||
        hazard_state->reference_kind != hazard->reference_kind ||
        hazard_state->reference_id != hazard->reference_id ||
        hazard_state->block_index != state->current_block_index ||
        hazard_state->producer_stage != hazard->consumer_stage) {
      continue;
    }
    const uint16_t required_distance = hazard_state->distance > hazard->distance
                                           ? hazard_state->distance
                                           : hazard->distance;
    const uint32_t actual_distance =
        state->current_issue_cycle >= hazard_state->issue_cycle
            ? state->current_issue_cycle - hazard_state->issue_cycle
            : 0;
    if (actual_distance < required_distance) {
      const uint32_t required_stall = required_distance - actual_distance;
      if (required_stall > stall_cycles) {
        stall_cycles = required_stall;
      }
    }
  }
  return stall_cycles;
}

static void loom_low_schedule_score_candidate_hazards(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node,
    loom_low_schedule_candidate_score_t* score) {
  score->hazard_stall_cycles = 0;
  const loom_low_schedule_class_t* schedule_class = node->schedule_class;
  if (state->options->strategy != LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL ||
      schedule_class == NULL) {
    return;
  }
  for (uint16_t i = 0; i < schedule_class->hazard_count; ++i) {
    const loom_low_hazard_t* hazard =
        &state->target.descriptor_set
             ->hazards[schedule_class->hazard_start + i];
    if (hazard->kind != LOOM_LOW_HAZARD_KIND_MIN_DISTANCE) {
      continue;
    }
    score->hazard_stall_cycles = loom_low_schedule_max_u32(
        score->hazard_stall_cycles,
        loom_low_schedule_min_distance_hazard_stall(state, hazard));
  }
}

static void loom_low_schedule_reset_candidate_pressure_deltas(
    loom_low_schedule_pressure_state_t* pressure_state) {
  for (iree_host_size_t i = 0;
       i < pressure_state->candidate_delta_touched_count; ++i) {
    const uint16_t reg_class_id =
        pressure_state->candidate_delta_touched_reg_class_ids[i];
    pressure_state->candidate_delta_units_by_reg_class[reg_class_id] = 0;
    pressure_state->candidate_delta_touched_flags[reg_class_id] = 0;
  }
  pressure_state->candidate_delta_touched_count = 0;
  for (iree_host_size_t i = 0;
       i < pressure_state->alias_sets.candidate_delta_touched_count; ++i) {
    loom_low_schedule_alias_pressure_record_t* record =
        &pressure_state->alias_sets.records
             [pressure_state->alias_sets.candidate_delta_touched_ids[i]];
    record->candidate_delta_units = 0;
    record->flags &= ~LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_CANDIDATE_TOUCHED;
  }
  pressure_state->alias_sets.candidate_delta_touched_count = 0;
  for (uint16_t i = 0; i < pressure_state->resources.candidate_touched_count;
       ++i) {
    loom_low_schedule_resource_pressure_record_t* record =
        &pressure_state->resources
             .records[pressure_state->resources.candidate_touched_ids[i]];
    record->candidate_added_units = 0;
    record->flags &=
        ~LOOM_LOW_SCHEDULE_RESOURCE_PRESSURE_FLAG_CANDIDATE_TOUCHED;
  }
  pressure_state->resources.candidate_touched_count = 0;
}

static void loom_low_schedule_note_candidate_pressure_delta(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint16_t reg_class_id,
    int64_t delta_units) {
  if (reg_class_id == LOOM_LOW_REG_CLASS_NONE || delta_units == 0 ||
      pressure_state->candidate_delta_units_by_reg_class == NULL) {
    return;
  }
  if (!pressure_state->candidate_delta_touched_flags[reg_class_id]) {
    pressure_state->candidate_delta_touched_flags[reg_class_id] = 1;
    pressure_state->candidate_delta_touched_reg_class_ids
        [pressure_state->candidate_delta_touched_count++] = reg_class_id;
  }
  pressure_state->candidate_delta_units_by_reg_class[reg_class_id] +=
      delta_units;
  if (pressure_state->alias_sets.records == NULL) {
    return;
  }
  const uint16_t alias_set_id =
      loom_low_schedule_alias_set_id(state, reg_class_id);
  if (alias_set_id == 0) {
    return;
  }
  loom_low_schedule_alias_pressure_record_t* record =
      &pressure_state->alias_sets.records[alias_set_id];
  if (!iree_any_bit_set(
          record->flags,
          LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_CANDIDATE_TOUCHED)) {
    record->flags |= LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_CANDIDATE_TOUCHED;
    pressure_state->alias_sets.candidate_delta_touched_ids
        [pressure_state->alias_sets.candidate_delta_touched_count++] =
        alias_set_id;
  }
  record->candidate_delta_units += delta_units;
}

static bool loom_low_schedule_storage_relation_can_alias_pressure(
    const loom_low_schedule_build_state_t* state,
    const loom_low_storage_relation_t* relation,
    loom_value_ordinal_t result_ordinal, loom_value_ordinal_t source_ordinal) {
  if (relation->kind != LOOM_LOW_STORAGE_RELATION_SAME_STORAGE &&
      relation->kind != LOOM_LOW_STORAGE_RELATION_SUBRANGE &&
      relation->kind != LOOM_LOW_STORAGE_RELATION_CONTIGUOUS_PART) {
    return false;
  }
  const loom_low_schedule_value_record_t* result =
      &state->values[result_ordinal];
  const loom_low_schedule_value_record_t* source =
      &state->values[source_ordinal];
  if (result->register_class_id == LOOM_LOW_REG_CLASS_NONE ||
      result->register_class_id != source->register_class_id) {
    return false;
  }
  IREE_ASSERT(relation->destination_unit_offset <= result->unit_count &&
                  relation->unit_count <=
                      result->unit_count - relation->destination_unit_offset,
              "verified storage relation destination units must fit result");
  IREE_ASSERT(relation->source_unit_offset <= source->unit_count &&
                  relation->unit_count <=
                      source->unit_count - relation->source_unit_offset,
              "verified storage relation source units must fit source");
  return relation->unit_count != 0;
}

static bool loom_low_schedule_value_lives_after_scored_candidate(
    const loom_low_schedule_value_record_t* value,
    const loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t value_ordinal) {
  if (!iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
    return false;
  }
  const uint32_t candidate_use_count =
      pressure_state->candidate_operand_use_counts[value_ordinal];
  return value->remaining_use_count > candidate_use_count;
}

static bool loom_low_schedule_relation_source_is_available_after_candidate(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t source_ordinal) {
  const loom_low_schedule_value_record_t* source =
      &state->values[source_ordinal];
  return source->live_unit_count == source->unit_count &&
         loom_low_schedule_value_lives_after_scored_candidate(
             source, pressure_state, source_ordinal);
}

static uint32_t loom_low_schedule_candidate_claimed_source_units(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t source_ordinal) {
  const uint32_t active_alias_units =
      pressure_state->alias_source_unit_counts
          ? pressure_state->alias_source_unit_counts[source_ordinal]
          : 0;
  const loom_low_schedule_value_record_t* source =
      &state->values[source_ordinal];
  if (active_alias_units >= source->unit_count) {
    return source->unit_count;
  }
  return iree_min(
      source->unit_count,
      iree_math_saturating_add_u32(
          active_alias_units,
          pressure_state->candidate_scratch_counts[source_ordinal]));
}

static uint32_t loom_low_schedule_result_alias_units_after_candidate(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_node_t* node, loom_value_ordinal_t result_ordinal) {
  uint32_t alias_units = 0;
  loom_low_storage_relation_iterator_t iterator;
  loom_low_storage_relation_iterator_initialize(state->module, node->op,
                                                &iterator);
  loom_low_storage_relation_t relation;
  while (loom_low_storage_relation_iterator_next(&iterator, &relation)) {
    const loom_value_ordinal_t destination_ordinal =
        loom_local_value_domain_ordinal(state->value_domain,
                                        relation.destination_value_id);
    if (destination_ordinal != result_ordinal) {
      continue;
    }
    const loom_value_ordinal_t source_ordinal = loom_local_value_domain_ordinal(
        state->value_domain, relation.source_value_id);
    if (!loom_low_schedule_storage_relation_can_alias_pressure(
            state, &relation, result_ordinal, source_ordinal)) {
      continue;
    }
    if (!loom_low_schedule_relation_source_is_available_after_candidate(
            state, pressure_state, source_ordinal)) {
      continue;
    }
    const loom_low_schedule_value_record_t* source =
        &state->values[source_ordinal];
    const uint32_t claimed_units =
        loom_low_schedule_candidate_claimed_source_units(state, pressure_state,
                                                         source_ordinal);
    if (claimed_units >= source->unit_count) {
      continue;
    }
    const uint32_t relation_alias_units =
        iree_min(relation.unit_count, source->unit_count - claimed_units);
    pressure_state->candidate_scratch_counts[source_ordinal] +=
        relation_alias_units;
    alias_units += relation_alias_units;
  }
  const loom_low_schedule_value_record_t* result =
      &state->values[result_ordinal];
  IREE_ASSERT(alias_units <= result->unit_count,
              "verified storage relations must not over-cover result units");
  return alias_units;
}

static uint32_t loom_low_schedule_candidate_alias_transfer_units(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t source_ordinal) {
  if (pressure_state->alias_heads == NULL) {
    return 0;
  }
  uint32_t transfer_units = 0;
  for (uint32_t alias_index = pressure_state->alias_heads[source_ordinal];
       alias_index != LOOM_LOW_SCHEDULE_NODE_NONE;
       alias_index = pressure_state->aliases[alias_index].next_alias) {
    const loom_low_schedule_pressure_alias_t* alias =
        &pressure_state->aliases[alias_index];
    if (alias->transferred) {
      continue;
    }
    const loom_low_schedule_value_record_t* result =
        &state->values[alias->result_ordinal];
    if (!loom_low_schedule_value_lives_after_scored_candidate(
            result, pressure_state, alias->result_ordinal)) {
      continue;
    }
    IREE_ASSERT(
        alias->unit_count <= result->unit_count &&
            result->live_unit_count <= result->unit_count - alias->unit_count,
        "transferred alias units must fit result pressure units");
    transfer_units += alias->unit_count;
  }
  return transfer_units;
}

static uint64_t loom_low_schedule_project_live_units(
    uint64_t current_live_units, int64_t delta_units) {
  if (delta_units < 0) {
    const uint64_t removed_units = (uint64_t)(-delta_units);
    IREE_ASSERT_LE(removed_units, current_live_units);
    return current_live_units - removed_units;
  }
  const uint64_t added_units = (uint64_t)delta_units;
  IREE_ASSERT_LE(added_units, UINT64_MAX - current_live_units);
  return current_live_units + added_units;
}

static iree_string_view_t loom_low_schedule_reg_class_name(
    const loom_low_schedule_build_state_t* state, uint16_t reg_class_id) {
  IREE_ASSERT_LT(reg_class_id, state->target.descriptor_set->reg_class_count);
  return loom_low_descriptor_set_string(
      state->target.descriptor_set,
      state->target.descriptor_set->reg_classes[reg_class_id]
          .name_string_offset);
}

static iree_string_view_t loom_low_schedule_pressure_source_name(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_source_kind_t source_kind, uint16_t source_id) {
  switch (source_kind) {
    case LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS:
      return loom_low_schedule_reg_class_name(state, source_id);
    case LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_RESOURCE:
      IREE_ASSERT(state->pressure_resources != NULL);
      IREE_ASSERT_LT(source_id, state->pressure_resources->resource_count);
      return state->pressure_resources->resources[source_id].name;
    default:
      return iree_string_view_empty();
  }
}

static void loom_low_schedule_project_candidate_resource_pressure(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state) {
  for (iree_host_size_t i = 0;
       i < pressure_state->candidate_delta_touched_count; ++i) {
    const uint16_t reg_class_id =
        pressure_state->candidate_delta_touched_reg_class_ids[i];
    const uint64_t projected_live_units = loom_low_schedule_project_live_units(
        pressure_state->current_live_units_by_reg_class[reg_class_id],
        pressure_state->candidate_delta_units_by_reg_class[reg_class_id]);
    const uint64_t peak_live_units =
        pressure_state->resources.peak_live_units_by_reg_class[reg_class_id];
    if (projected_live_units <= peak_live_units) continue;
    const loom_low_pressure_resource_member_range_t range =
        loom_low_pressure_resource_table_member_range(state->pressure_resources,
                                                      reg_class_id);
    for (uint16_t j = 0; j < range.count; ++j) {
      const uint16_t member_index =
          state->pressure_resources
              ->member_indices_by_reg_class[range.start + j];
      const loom_low_pressure_resource_member_t* member =
          &state->pressure_resources->members[member_index];
      IREE_ASSERT_EQ(member->descriptor_reg_class_id, reg_class_id);
      const uint64_t peak_contribution = loom_low_pressure_round_resource_units(
          peak_live_units, member->contribution_granularity);
      const uint64_t projected_contribution =
          loom_low_pressure_round_resource_units(
              projected_live_units, member->contribution_granularity);
      loom_low_schedule_resource_pressure_record_t* record =
          &pressure_state->resources.records[member->resource_id];
      if (!iree_any_bit_set(
              record->flags,
              LOOM_LOW_SCHEDULE_RESOURCE_PRESSURE_FLAG_CANDIDATE_TOUCHED)) {
        record->flags |=
            LOOM_LOW_SCHEDULE_RESOURCE_PRESSURE_FLAG_CANDIDATE_TOUCHED;
        pressure_state->resources.candidate_touched_ids
            [pressure_state->resources.candidate_touched_count++] =
            member->resource_id;
      }
      record->candidate_added_units = iree_math_saturating_add_u64(
          record->candidate_added_units,
          projected_contribution - peak_contribution);
    }
  }
}

static void loom_low_schedule_score_candidate_resource_pressure(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_candidate_score_t* score) {
  uint32_t resource_penalty = pressure_state->resources.pressure_cliff_penalty;
  for (uint16_t i = 0; i < pressure_state->resources.candidate_touched_count;
       ++i) {
    const uint16_t resource_id =
        pressure_state->resources.candidate_touched_ids[i];
    const loom_low_schedule_resource_pressure_record_t* record =
        &pressure_state->resources.records[resource_id];
    const uint64_t projected_peak_units = iree_math_saturating_add_u64(
        record->current_peak_units, record->candidate_added_units);
    const loom_low_pressure_resource_t* resource =
        &state->pressure_resources->resources[resource_id];
    const uint16_t cliff_end = resource->cliff_start + resource->cliff_count;
    uint16_t cliff_index = record->next_cliff_index;
    while (cliff_index < cliff_end) {
      const loom_low_pressure_cliff_t* cliff =
          &state->pressure_resources->cliffs[cliff_index];
      if (cliff->cliff_units > projected_peak_units) {
        const uint64_t units_until_cliff =
            cliff->cliff_units - projected_peak_units;
        if (units_until_cliff < score->units_until_pressure_cliff) {
          score->pressure_cliff_source_kind =
              LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_RESOURCE;
          score->pressure_cliff_source_id = resource_id;
          score->units_until_pressure_cliff = (uint32_t)units_until_cliff;
        }
        break;
      }
      const uint32_t penalty = cliff->tier_before - cliff->tier_after;
      resource_penalty =
          iree_math_saturating_add_u32(resource_penalty, penalty);
      if (score->pressure_cliff_units ==
          LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE) {
        score->pressure_cliff_source_kind =
            LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_RESOURCE;
        score->pressure_cliff_source_id = resource_id;
        score->pressure_cliff_units = cliff->cliff_units;
      }
      ++cliff_index;
    }
  }
  score->pressure_cliff_penalty = iree_math_saturating_add_u32(
      score->pressure_cliff_penalty, resource_penalty);
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
    score->actual_pressure_cliff_penalty = iree_math_saturating_add_u32(
        score->actual_pressure_cliff_penalty, resource_penalty);
  }
}

static void loom_low_schedule_score_candidate_pressure_cliffs_for_class(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_candidate_score_t* score, uint16_t reg_class_id) {
  const uint64_t current_live_units =
      pressure_state->current_live_units_by_reg_class[reg_class_id];
  const int64_t delta_units =
      pressure_state->candidate_delta_touched_flags[reg_class_id]
          ? pressure_state->candidate_delta_units_by_reg_class[reg_class_id]
          : 0;
  if (current_live_units == 0 && delta_units == 0) {
    return;
  }
  const uint64_t projected_live_units =
      loom_low_schedule_project_live_units(current_live_units, delta_units);
  const loom_low_pressure_cliff_range_t range =
      loom_low_pressure_cliff_table_range(state->pressure_cliffs, reg_class_id);
  IREE_ASSERT(pressure_state->first_actionable_pressure_cliff_indices != NULL);
  for (uint32_t cliff_index =
           pressure_state
               ->first_actionable_pressure_cliff_indices[reg_class_id];
       cliff_index < range.start + range.count; ++cliff_index) {
    const loom_low_pressure_cliff_t* cliff =
        &state->pressure_cliffs->values[cliff_index];
    // Protect target tiers that the source order preserves. Cliffs already
    // crossed by the authored function are excluded so greedy local decisions
    // do not attempt a global occupancy recovery.
    if (projected_live_units >= cliff->cliff_units) {
      const uint32_t penalty = cliff->tier_before - cliff->tier_after;
      score->pressure_cliff_penalty += penalty;
      if (state->options->strategy ==
          LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
        score->actual_pressure_cliff_penalty += penalty;
      }
      if (score->pressure_cliff_units ==
          LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE) {
        score->pressure_cliff_source_kind =
            LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS;
        score->pressure_cliff_source_id = reg_class_id;
        score->pressure_cliff_units = cliff->cliff_units;
      }
      continue;
    }
    const uint64_t units_until_cliff =
        cliff->cliff_units - projected_live_units;
    if (units_until_cliff < score->units_until_pressure_cliff) {
      score->pressure_cliff_source_kind =
          LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS;
      score->pressure_cliff_source_id = reg_class_id;
      score->units_until_pressure_cliff = (uint32_t)units_until_cliff;
    }
    break;
  }
}

static void loom_low_schedule_score_candidate_pressure_cliffs(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_candidate_score_t* score) {
  if (state->pressure_cliffs == NULL ||
      loom_low_pressure_cliff_table_is_empty(*state->pressure_cliffs) ||
      pressure_state->current_live_units_by_reg_class == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < pressure_state->block_reg_class_count; ++i) {
    loom_low_schedule_score_candidate_pressure_cliffs_for_class(
        state, pressure_state, score, pressure_state->block_reg_class_ids[i]);
  }
  for (iree_host_size_t i = 0;
       i < pressure_state->candidate_delta_touched_count; ++i) {
    const uint16_t reg_class_id =
        pressure_state->candidate_delta_touched_reg_class_ids[i];
    if (pressure_state->block_reg_class_touched_flags[reg_class_id]) {
      continue;
    }
    loom_low_schedule_score_candidate_pressure_cliffs_for_class(
        state, pressure_state, score, reg_class_id);
  }
}

static void loom_low_schedule_score_candidate_pressure_limit(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_candidate_score_t* score, uint16_t reg_class_id,
    uint32_t limit_units, uint64_t current_live_units, int64_t delta_units) {
  if (limit_units == UINT32_MAX) {
    return;
  }
  if (current_live_units == 0 && delta_units == 0) {
    return;
  }
  const uint64_t projected_live_units =
      loom_low_schedule_project_live_units(current_live_units, delta_units);
  uint64_t reserved_live_units = projected_live_units;
  if (delta_units > 0) {
    reserved_live_units = iree_math_saturating_add_u64(
        reserved_live_units, score->pressure_reserve_units);
  }
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL &&
      projected_live_units >= limit_units) {
    const uint64_t actual_limit_debt = projected_live_units - limit_units + 1;
    const uint32_t actual_penalty = actual_limit_debt > UINT32_MAX
                                        ? UINT32_MAX
                                        : (uint32_t)actual_limit_debt;
    score->actual_pressure_cliff_penalty = iree_math_saturating_add_u32(
        score->actual_pressure_cliff_penalty, actual_penalty);
  }
  if (reserved_live_units >= limit_units) {
    const uint64_t reserved_limit_debt = reserved_live_units - limit_units + 1;
    const uint32_t penalty = reserved_limit_debt > UINT32_MAX
                                 ? UINT32_MAX
                                 : (uint32_t)reserved_limit_debt;
    score->pressure_cliff_penalty =
        iree_math_saturating_add_u32(score->pressure_cliff_penalty, penalty);
    if (score->pressure_cliff_units == LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE) {
      score->pressure_cliff_source_kind =
          LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS;
      score->pressure_cliff_source_id = reg_class_id;
      score->pressure_cliff_units = limit_units;
    }
    return;
  }
  const uint64_t units_until_limit = limit_units - reserved_live_units;
  if (units_until_limit < score->units_until_pressure_cliff) {
    score->pressure_cliff_source_kind =
        LOOM_LOW_SCHEDULE_PRESSURE_SOURCE_REGISTER_CLASS;
    score->pressure_cliff_source_id = reg_class_id;
    score->units_until_pressure_cliff = (uint32_t)units_until_limit;
  }
}

static void loom_low_schedule_score_candidate_pressure_limit_for_class(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_candidate_score_t* score, uint16_t reg_class_id) {
  if (state->pressure_limits.alias_sets != NULL &&
      state->target.descriptor_set->reg_classes[reg_class_id].alias_set_id !=
          0) {
    return;
  }
  const int64_t delta_units =
      pressure_state->candidate_delta_touched_flags[reg_class_id]
          ? pressure_state->candidate_delta_units_by_reg_class[reg_class_id]
          : 0;
  loom_low_schedule_score_candidate_pressure_limit(
      state, score, reg_class_id,
      state->pressure_limits.by_reg_class[reg_class_id],
      pressure_state->current_live_units_by_reg_class[reg_class_id],
      delta_units);
}

static void loom_low_schedule_score_candidate_pressure_limit_for_alias_set(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_candidate_score_t* score, uint16_t alias_set_id) {
  const loom_low_schedule_alias_pressure_record_t* record =
      &pressure_state->alias_sets.records[alias_set_id];
  loom_low_schedule_score_candidate_pressure_limit(
      state, score,
      state->pressure_limits.alias_sets[alias_set_id]
          .representative_reg_class_id,
      state->pressure_limits.alias_sets[alias_set_id].live_unit_limit,
      record->current_live_units, record->candidate_delta_units);
}

static void loom_low_schedule_score_candidate_pressure_limits(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_candidate_score_t* score) {
  if (state->pressure_limits.by_reg_class == NULL ||
      pressure_state->current_live_units_by_reg_class == NULL) {
    return;
  }
  for (iree_host_size_t i = 0; i < pressure_state->block_reg_class_count; ++i) {
    loom_low_schedule_score_candidate_pressure_limit_for_class(
        state, pressure_state, score, pressure_state->block_reg_class_ids[i]);
  }
  for (iree_host_size_t i = 0;
       i < pressure_state->candidate_delta_touched_count; ++i) {
    const uint16_t reg_class_id =
        pressure_state->candidate_delta_touched_reg_class_ids[i];
    if (pressure_state->block_reg_class_touched_flags[reg_class_id]) {
      continue;
    }
    loom_low_schedule_score_candidate_pressure_limit_for_class(
        state, pressure_state, score, reg_class_id);
  }
  for (iree_host_size_t i = 0; i < pressure_state->alias_sets.block_count;
       ++i) {
    loom_low_schedule_score_candidate_pressure_limit_for_alias_set(
        state, pressure_state, score, pressure_state->alias_sets.block_ids[i]);
  }
  for (iree_host_size_t i = 0;
       i < pressure_state->alias_sets.candidate_delta_touched_count; ++i) {
    const uint16_t alias_set_id =
        pressure_state->alias_sets.candidate_delta_touched_ids[i];
    if (iree_any_bit_set(pressure_state->alias_sets.records[alias_set_id].flags,
                         LOOM_LOW_SCHEDULE_ALIAS_PRESSURE_FLAG_BLOCK_TOUCHED)) {
      continue;
    }
    loom_low_schedule_score_candidate_pressure_limit_for_alias_set(
        state, pressure_state, score, alias_set_id);
  }
}

static void loom_low_schedule_initialize_current_pressure_cliff_penalty(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state) {
  loom_low_schedule_reset_candidate_pressure_deltas(pressure_state);
  loom_low_schedule_candidate_score_t score = {
      .pressure_cliff_units = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
      .units_until_pressure_cliff = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
  };
  loom_low_schedule_score_candidate_pressure_cliffs(state, pressure_state,
                                                    &score);
  if (state->pressure_resources != NULL) {
    loom_low_schedule_score_candidate_resource_pressure(state, pressure_state,
                                                        &score);
  }
  loom_low_schedule_score_candidate_pressure_limits(state, pressure_state,
                                                    &score);
  pressure_state->current_pressure_cliff_penalty =
      score.actual_pressure_cliff_penalty;
}

static bool loom_low_schedule_node_result_used_by(
    const loom_low_schedule_node_t* producer,
    const loom_low_schedule_node_t* consumer) {
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(producer);
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(consumer);
  for (uint16_t result_index = 0; result_index < producer->result_count;
       ++result_index) {
    const loom_value_ordinal_t result_ordinal = result_ordinals[result_index];
    for (uint16_t operand_index = 0; operand_index < consumer->operand_count;
         ++operand_index) {
      if (result_ordinal == operand_ordinals[operand_index]) {
        return true;
      }
    }
  }
  return false;
}

static bool loom_low_schedule_node_is_pair_transparent(
    const loom_low_schedule_node_t* node) {
  if (node->kind != LOOM_LOW_SCHEDULE_NODE_STRUCTURAL || node->op == NULL) {
    return false;
  }
  switch (node->op->kind) {
    case LOOM_OP_LOW_SLICE:
    case LOOM_OP_LOW_CONCAT:
      return true;
    default:
      return false;
  }
}

static const loom_low_schedule_pair_affinity_record_t*
loom_low_schedule_find_pair_affinity(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* first,
    const loom_low_schedule_node_t* second) {
  if (first == NULL || second == NULL || first->descriptor == NULL ||
      second->descriptor == NULL || state->pair_affinity_heads == NULL) {
    return NULL;
  }
  const uint32_t first_ordinal =
      loom_low_schedule_descriptor_ordinal(state, first->descriptor);
  const uint32_t second_ordinal =
      loom_low_schedule_descriptor_ordinal(state, second->descriptor);
  if (first_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE ||
      second_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return NULL;
  }
  for (uint32_t record_index = state->pair_affinity_heads[first_ordinal];
       record_index != LOOM_LOW_SCHEDULE_PAIR_AFFINITY_RECORD_NONE;
       record_index = state->pair_affinity_records[record_index].next_record) {
    const loom_low_schedule_pair_affinity_record_t* record =
        &state->pair_affinity_records[record_index];
    if (record->second_descriptor_ordinal == second_ordinal) {
      return record;
    }
  }
  return NULL;
}

static uint16_t loom_low_schedule_ready_pair_affinity_priority(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_ready_policy_t* ready_policy,
    uint32_t descriptor_ordinal) {
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) return 0;
  uint16_t priority = 0;
  for (uint32_t record_index = state->pair_affinity_heads[descriptor_ordinal];
       record_index != LOOM_LOW_SCHEDULE_PAIR_AFFINITY_RECORD_NONE;
       record_index = state->pair_affinity_records[record_index].next_record) {
    const loom_low_schedule_pair_affinity_record_t* record =
        &state->pair_affinity_records[record_index];
    if (loom_low_schedule_ready_frontier_descriptor_count(
            &ready_policy->frontier, record->second_descriptor_ordinal) != 0) {
      priority = iree_max(priority, record->priority);
    }
  }
  for (uint32_t record_index =
           state->pair_affinity_reverse_heads[descriptor_ordinal];
       record_index != LOOM_LOW_SCHEDULE_PAIR_AFFINITY_RECORD_NONE;
       record_index =
           state->pair_affinity_records[record_index].reverse_next_record) {
    const loom_low_schedule_pair_affinity_record_t* record =
        &state->pair_affinity_records[record_index];
    if (loom_low_schedule_ready_frontier_descriptor_count(
            &ready_policy->frontier, record->first_descriptor_ordinal) != 0) {
      priority = iree_max(priority, record->priority);
    }
  }
  return priority;
}

static bool loom_low_schedule_node_is_pair_setup(
    const loom_low_schedule_node_t* node) {
  return iree_any_bit_set(node->flags, LOOM_LOW_SCHEDULE_NODE_FLAG_PAIR_SETUP);
}

static void loom_low_schedule_ready_policy_update_setup_dependencies(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index,
    loom_low_schedule_ready_membership_change_t change) {
  if (pressure_state->ready_setup_dependency_counts == NULL ||
      !loom_low_schedule_node_is_pair_setup(&state->nodes[node_index])) {
    return;
  }
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  const uint32_t group_begin = loom_low_schedule_dependency_index_group_begin(
      &state->dependency_index, node_index);
  const uint32_t group_end = loom_low_schedule_dependency_index_group_end(
      &state->dependency_index, node_index);
  for (uint32_t group_index = group_begin; group_index < group_end;
       ++group_index) {
    const loom_low_schedule_dependency_group_t* group =
        loom_low_schedule_dependency_index_group_at(&state->dependency_index,
                                                    group_index);
    const uint32_t consumer_node = group->consumer_node;
    if (state->nodes[consumer_node].block_index != node->block_index ||
        state->nodes[consumer_node].scheduled_ordinal !=
            LOOM_LOW_SCHEDULE_NODE_NONE) {
      continue;
    }
    uint32_t* count =
        &pressure_state->ready_setup_dependency_counts[consumer_node];
    if (change == LOOM_LOW_SCHEDULE_READY_MEMBERSHIP_INSERT) {
      IREE_ASSERT_LE(group->dependency_count, UINT32_MAX - *count);
      *count += group->dependency_count;
    } else {
      IREE_ASSERT_LE(group->dependency_count, *count);
      *count -= group->dependency_count;
    }
  }
}

static void loom_low_schedule_ready_policy_insert(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_ready_policy_t* ready_policy, uint32_t node_index) {
  loom_low_schedule_ready_keys_t keys = {0};
  if (loom_low_schedule_uses_pressure_strategy(state)) {
    keys = loom_low_schedule_ready_keys(state, pressure_state, node_index);
  } else {
    keys.values[LOOM_LOW_SCHEDULE_READY_VIEW_SOURCE] =
        state->nodes[node_index].source_ordinal;
  }
  const loom_low_descriptor_t* descriptor = state->nodes[node_index].descriptor;
  const uint32_t descriptor_ordinal =
      descriptor != NULL
          ? loom_low_schedule_descriptor_ordinal(state, descriptor)
          : LOOM_LOW_SCHEDULE_READY_NODE_NONE;
  loom_low_schedule_ready_frontier_insert(&ready_policy->frontier, node_index,
                                          &keys, descriptor_ordinal);
  loom_low_schedule_ready_policy_update_setup_dependencies(
      state, pressure_state, node_index,
      LOOM_LOW_SCHEDULE_READY_MEMBERSHIP_INSERT);
}

static void loom_low_schedule_ready_policy_remove(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_ready_policy_t* ready_policy, uint32_t node_index) {
  loom_low_schedule_ready_policy_update_setup_dependencies(
      state, pressure_state, node_index,
      LOOM_LOW_SCHEDULE_READY_MEMBERSHIP_REMOVE);
  loom_low_schedule_ready_frontier_remove(&ready_policy->frontier, node_index);
}

static bool loom_low_schedule_pair_is_preferred(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* first,
    const loom_low_schedule_node_t* second) {
  if (state->preferred_pair_nodes == NULL) {
    return false;
  }
  return state->preferred_pair_nodes[first->source_ordinal].successor_node ==
         second->source_ordinal;
}

static uint16_t loom_low_schedule_pair_affinity_priority(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* first,
    const loom_low_schedule_node_t* second,
    uint16_t* out_placement_option_count) {
  *out_placement_option_count = 0;
  const loom_low_schedule_pair_affinity_record_t* record =
      loom_low_schedule_find_pair_affinity(state, first, second);
  if (record == NULL || loom_low_schedule_node_result_used_by(first, second)) {
    return 0;
  }
  if (record->placement_recipe_index == LOOM_LOW_PLACEMENT_PAIR_RECIPE_NONE) {
    return loom_low_schedule_pair_is_preferred(state, first, second)
               ? UINT16_MAX
               : record->priority;
  }
  const uint16_t recipe_index = (uint16_t)(record->placement_recipe_index - 1u);
  const loom_low_placement_pair_recipe_t* recipe =
      &state->options->pair_affinities.placement_recipes[recipe_index];
  const loom_low_placement_pair_use_t use = {
      .first_op = first->op,
      .second_op = second->op,
      .placement_recipe_index = record->placement_recipe_index,
      .priority = record->priority,
  };
  *out_placement_option_count =
      loom_low_placement_pair_possible_alternative_count(&use, recipe);
  if (*out_placement_option_count == 0) {
    return 0;
  }
  return loom_low_schedule_pair_is_preferred(state, first, second)
             ? UINT16_MAX
             : record->priority;
}

static bool loom_low_schedule_node_can_start_pair_affinity(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node) {
  if (node == NULL || node->descriptor == NULL ||
      state->pair_affinity_heads == NULL) {
    return false;
  }
  const uint32_t descriptor_ordinal =
      loom_low_schedule_descriptor_ordinal(state, node->descriptor);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return false;
  }
  return state->pair_affinity_heads[descriptor_ordinal] !=
         LOOM_LOW_SCHEDULE_PAIR_AFFINITY_RECORD_NONE;
}

static bool loom_low_schedule_node_can_unlock_pair_affinity(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node) {
  return state->pending_pair_affinity_node == LOOM_LOW_SCHEDULE_NODE_NONE &&
         loom_low_schedule_node_is_pair_setup(node);
}

static uint16_t loom_low_schedule_scale_direct_pair_affinity(
    uint16_t priority) {
  return priority > UINT16_MAX / 2u ? UINT16_MAX : (uint16_t)(priority * 2u);
}

static uint16_t loom_low_schedule_score_candidate_pair_affinity(
    const loom_low_schedule_build_state_t* state, uint32_t node_index,
    uint16_t* out_placement_option_count) {
  *out_placement_option_count = 0;
  if (loom_low_schedule_pair_affinity_list_is_empty(
          state->options->pair_affinities) ||
      state->pending_pair_affinity_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return 0;
  }
  const loom_low_schedule_node_t* anchor =
      &state->nodes[state->pending_pair_affinity_node];
  const loom_low_schedule_node_t* candidate = &state->nodes[node_index];
  uint16_t priority = loom_low_schedule_pair_affinity_priority(
      state, anchor, candidate, out_placement_option_count);
  if (priority != 0) {
    return loom_low_schedule_scale_direct_pair_affinity(priority);
  }

  if (!loom_low_schedule_node_is_pair_transparent(candidate)) {
    return 0;
  }
  const uint32_t group_begin = loom_low_schedule_dependency_index_group_begin(
      &state->dependency_index, node_index);
  const uint32_t group_end = loom_low_schedule_dependency_index_group_end(
      &state->dependency_index, node_index);
  const uint32_t lookahead_end =
      group_begin + iree_min(group_end - group_begin,
                             LOOM_LOW_SCHEDULE_PAIR_LOOKAHEAD_CAPACITY);
  for (uint32_t group_index = group_begin; group_index < lookahead_end;
       ++group_index) {
    if (!loom_low_schedule_dependency_index_group_has_ssa(
            &state->dependency_index, group_index)) {
      continue;
    }
    const loom_low_schedule_node_t* consumer =
        &state->nodes[loom_low_schedule_dependency_index_group_at(
                          &state->dependency_index, group_index)
                          ->consumer_node];
    uint16_t placement_option_count = 0;
    priority = loom_low_schedule_pair_affinity_priority(
        state, anchor, consumer, &placement_option_count);
    if (priority != 0) {
      *out_placement_option_count = placement_option_count;
      return priority;
    }
  }
  return 0;
}

static uint16_t loom_low_schedule_preferred_pair_member_priority(
    const loom_low_schedule_build_state_t* state, uint32_t node_index) {
  if (state->pending_pair_affinity_node != LOOM_LOW_SCHEDULE_NODE_NONE ||
      state->preferred_pair_nodes == NULL ||
      state->nodes[node_index].descriptor == NULL) {
    return 0;
  }
  const loom_low_schedule_preferred_pair_node_t* pair_node =
      &state->preferred_pair_nodes[node_index];
  return pair_node->predecessor_node != LOOM_LOW_SCHEDULE_NODE_NONE ||
                 pair_node->successor_node != LOOM_LOW_SCHEDULE_NODE_NONE
             ? UINT16_MAX
             : 0;
}

static uint16_t loom_low_schedule_preferred_pair_anchor_priority(
    const loom_low_schedule_build_state_t* state, const uint32_t* indegrees,
    uint32_t node_index) {
  if (loom_low_schedule_preferred_pair_member_priority(state, node_index) ==
          0 ||
      indegrees == NULL) {
    return 0;
  }
  const uint32_t successor_node =
      state->preferred_pair_nodes[node_index].successor_node;
  if (successor_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
      state->nodes[successor_node].scheduled_ordinal !=
          LOOM_LOW_SCHEDULE_NODE_NONE ||
      indegrees[successor_node] != 0) {
    return 0;
  }
  return UINT16_MAX;
}

static uint32_t loom_low_schedule_ready_pair_nominee(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_ready_policy_t* ready_policy) {
  const uint32_t anchor_node = state->pending_pair_affinity_node;
  if (anchor_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return LOOM_LOW_SCHEDULE_NODE_NONE;
  }
  if (state->preferred_pair_nodes != NULL) {
    const uint32_t preferred_node =
        state->preferred_pair_nodes[anchor_node].successor_node;
    if (preferred_node != LOOM_LOW_SCHEDULE_NODE_NONE &&
        loom_low_schedule_ready_frontier_contains(&ready_policy->frontier,
                                                  preferred_node)) {
      return preferred_node;
    }
  }

  const loom_low_schedule_node_t* anchor = &state->nodes[anchor_node];
  if (anchor->descriptor == NULL || state->pair_affinity_heads == NULL) {
    return LOOM_LOW_SCHEDULE_NODE_NONE;
  }
  const uint32_t anchor_descriptor_ordinal =
      loom_low_schedule_descriptor_ordinal(state, anchor->descriptor);
  if (anchor_descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return LOOM_LOW_SCHEDULE_NODE_NONE;
  }

  uint32_t best_node = LOOM_LOW_SCHEDULE_NODE_NONE;
  uint16_t best_priority = 0;
  for (uint32_t record_index =
           state->pair_affinity_heads[anchor_descriptor_ordinal];
       record_index != LOOM_LOW_SCHEDULE_PAIR_AFFINITY_RECORD_NONE;
       record_index = state->pair_affinity_records[record_index].next_record) {
    const loom_low_schedule_pair_affinity_record_t* record =
        &state->pair_affinity_records[record_index];
    const uint32_t candidate_node =
        loom_low_schedule_ready_frontier_descriptor_head(
            &ready_policy->frontier, record->second_descriptor_ordinal);
    if (candidate_node == LOOM_LOW_SCHEDULE_READY_NODE_NONE) continue;
    if (record->priority > best_priority ||
        (record->priority == best_priority &&
         (best_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
          state->nodes[candidate_node].source_ordinal <
              state->nodes[best_node].source_ordinal))) {
      best_node = candidate_node;
      best_priority = record->priority;
    }
  }
  return best_node;
}

static bool loom_low_schedule_descriptor_frontier_is_non_growing(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    uint32_t candidate_node_index, uint16_t candidate_storage_relation_count,
    uint64_t candidate_killed_units, uint64_t candidate_produced_units,
    const uint32_t* consumer_nodes, iree_host_size_t consumer_count) {
  const iree_host_size_t candidate_operand_count =
      pressure_state->candidate_operand_count;
  for (iree_host_size_t i = 0; i < candidate_operand_count; ++i) {
    const loom_value_ordinal_t value_ordinal =
        pressure_state->candidate_operand_ordinals[i];
    pressure_state->candidate_scratch_counts[value_ordinal] = 0;
  }

  for (iree_host_size_t i = 0; i < consumer_count; ++i) {
    const loom_low_schedule_node_t* consumer = &state->nodes[consumer_nodes[i]];
    const loom_value_ordinal_t* operand_ordinals =
        loom_low_schedule_node_const_operand_ordinals(consumer);
    for (uint16_t operand_index = 0; operand_index < consumer->operand_count;
         ++operand_index) {
      const loom_value_ordinal_t value_ordinal =
          operand_ordinals[operand_index];
      uint32_t* consumer_use_count =
          &pressure_state->candidate_scratch_counts[value_ordinal];
      if ((*consumer_use_count)++ == 0 &&
          pressure_state->candidate_operand_use_counts[value_ordinal] == 0) {
        pressure_state->candidate_operand_ordinals
            [pressure_state->candidate_operand_count++] = value_ordinal;
      }
    }
  }

  uint64_t killed_units = 0;
  for (iree_host_size_t i = 0; i < pressure_state->candidate_operand_count;
       ++i) {
    const loom_value_ordinal_t value_ordinal =
        pressure_state->candidate_operand_ordinals[i];
    const uint32_t consumer_use_count =
        pressure_state->candidate_scratch_counts[value_ordinal];
    if (consumer_use_count == 0) {
      continue;
    }
    const loom_low_schedule_value_record_t* value =
        &state->values[value_ordinal];
    const uint32_t candidate_use_count =
        pressure_state->candidate_operand_use_counts[value_ordinal];
    IREE_ASSERT_GE(value->remaining_use_count, candidate_use_count);
    const uint32_t remaining_use_count =
        value->remaining_use_count - candidate_use_count;
    IREE_ASSERT_GE(remaining_use_count, consumer_use_count);
    if (remaining_use_count != consumer_use_count) {
      continue;
    }
    if (candidate_storage_relation_count != 0) {
      continue;
    }
    if (iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
      killed_units += value->live_unit_count;
    } else if (value->producer_node == candidate_node_index) {
      killed_units += value->unit_count;
    }
  }

  uint64_t produced_units = 0;
  for (iree_host_size_t i = 0; i < consumer_count; ++i) {
    const loom_low_schedule_node_t* consumer = &state->nodes[consumer_nodes[i]];
    const loom_value_ordinal_t* result_ordinals =
        loom_low_schedule_node_const_result_ordinals(consumer);
    for (uint16_t result_index = 0; result_index < consumer->result_count;
         ++result_index) {
      const loom_low_schedule_value_record_t* value =
          &state->values[result_ordinals[result_index]];
      if (value->remaining_use_count != 0) {
        produced_units += value->unit_count;
      }
    }
  }

  for (iree_host_size_t i = 0; i < pressure_state->candidate_operand_count;
       ++i) {
    const loom_value_ordinal_t value_ordinal =
        pressure_state->candidate_operand_ordinals[i];
    pressure_state->candidate_scratch_counts[value_ordinal] = 0;
  }
  pressure_state->candidate_operand_count = candidate_operand_count;
  const uint64_t total_killed_units =
      iree_math_saturating_add_u64(candidate_killed_units, killed_units);
  const uint64_t total_produced_units =
      iree_math_saturating_add_u64(candidate_produced_units, produced_units);
  return total_produced_units <= total_killed_units;
}

static void loom_low_schedule_publish_unlock_consumer(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t producer_node,
    uint32_t consumer_node) {
  if (state->nodes[producer_node].block_index !=
      state->nodes[consumer_node].block_index) {
    return;
  }
  loom_low_schedule_unlock_record_t* record =
      &pressure_state->unlocks.records[producer_node];
  record->demand_units = iree_math_saturating_add_u32(
      record->demand_units, state->node_pressure_demand_units[consumer_node]);
  record->activation_units = loom_low_schedule_max_u32(
      record->activation_units,
      state->node_pressure_activation_units[consumer_node]);
  if (state->nodes[consumer_node].descriptor != NULL) {
    record->flags |= LOOM_LOW_SCHEDULE_UNLOCK_FLAG_DESCRIPTOR;
    if (pressure_state->unlocks.descriptor_heads != NULL &&
        record->descriptor_count <
            LOOM_LOW_SCHEDULE_DESCRIPTOR_FRONTIER_CAPACITY) {
      pressure_state->unlocks.descriptor_next_nodes[consumer_node] =
          pressure_state->unlocks.descriptor_heads[producer_node];
      pressure_state->unlocks.descriptor_heads[producer_node] = consumer_node;
    }
    if (record->descriptor_count <=
        LOOM_LOW_SCHEDULE_DESCRIPTOR_FRONTIER_CAPACITY) {
      ++record->descriptor_count;
    }
  }
  ++state->unlock_summary_publication_count;
}

static iree_status_t loom_low_schedule_initialize_unlock_summaries(
    loom_low_schedule_build_state_t* state, uint32_t node_count,
    loom_low_schedule_pressure_state_t* pressure_state) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_dependency_frontier_initialize(
      &state->dependency_index, state->arena,
      &pressure_state->unlocks.frontier));
  if (node_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, node_count, sizeof(*pressure_state->unlocks.records),
      (void**)&pressure_state->unlocks.records));
  memset(pressure_state->unlocks.records, 0,
         node_count * sizeof(*pressure_state->unlocks.records));
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count,
        sizeof(*pressure_state->unlocks.descriptor_heads),
        (void**)&pressure_state->unlocks.descriptor_heads));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count,
        sizeof(*pressure_state->unlocks.descriptor_next_nodes),
        (void**)&pressure_state->unlocks.descriptor_next_nodes));
    memset(pressure_state->unlocks.descriptor_heads, 0xFF,
           node_count * sizeof(*pressure_state->unlocks.descriptor_heads));
  }
  for (uint32_t consumer_node = 0; consumer_node < node_count;
       ++consumer_node) {
    const uint32_t producer_node =
        loom_low_schedule_dependency_frontier_remaining_producer(
            &pressure_state->unlocks.frontier, consumer_node);
    if (producer_node != LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_NONE) {
      loom_low_schedule_publish_unlock_consumer(state, pressure_state,
                                                producer_node, consumer_node);
    }
  }
  return iree_ok_status();
}

static uint16_t loom_low_schedule_score_pair_setup_unlocks(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_ready_policy_t* ready_policy,
    const uint32_t* indegrees, uint32_t node_index) {
  if (pressure_state->ready_setup_dependency_counts == NULL ||
      !loom_low_schedule_node_can_unlock_pair_affinity(
          state, &state->nodes[node_index])) {
    return 0;
  }
  uint16_t priority = 0;
  const uint32_t group_begin = loom_low_schedule_dependency_index_group_begin(
      &state->dependency_index, node_index);
  const uint32_t group_end = loom_low_schedule_dependency_index_group_end(
      &state->dependency_index, node_index);
  const uint32_t lookahead_end =
      group_begin + iree_min(group_end - group_begin,
                             LOOM_LOW_SCHEDULE_PAIR_LOOKAHEAD_CAPACITY);
  for (uint32_t group_index = group_begin; group_index < lookahead_end;
       ++group_index) {
    const uint32_t consumer_node = loom_low_schedule_dependency_index_group_at(
                                       &state->dependency_index, group_index)
                                       ->consumer_node;
    const loom_low_descriptor_t* consumer_descriptor =
        state->nodes[consumer_node].descriptor;
    if (consumer_descriptor == NULL ||
        pressure_state->ready_setup_dependency_counts[consumer_node] !=
            indegrees[consumer_node]) {
      continue;
    }
    priority = iree_max(
        priority,
        loom_low_schedule_preferred_pair_member_priority(state, consumer_node));
    priority = iree_max(
        priority,
        loom_low_schedule_ready_pair_affinity_priority(
            state, ready_policy,
            loom_low_schedule_descriptor_ordinal(state, consumer_descriptor)));
  }
  return priority;
}

static loom_low_schedule_pressure_demand_t
loom_low_schedule_score_candidate_pressure_demand(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_ready_policy_t* ready_policy,
    const uint32_t* indegrees, uint32_t node_index,
    uint16_t storage_relation_count, uint64_t killed_live_units,
    uint64_t produced_live_units) {
  loom_low_schedule_pressure_demand_t demand = {
      .demand_units = state->node_pressure_demand_units != NULL
                          ? state->node_pressure_demand_units[node_index]
                          : 1,
      .activation_units =
          state->node_pressure_activation_units != NULL
              ? state->node_pressure_activation_units[node_index]
              : 1,
  };
  const loom_low_schedule_unlock_record_t* unlock_record =
      &pressure_state->unlocks.records[node_index];
  demand.demand_units = loom_low_schedule_max_u32(demand.demand_units,
                                                  unlock_record->demand_units);
  demand.activation_units = loom_low_schedule_max_u32(
      demand.activation_units, unlock_record->activation_units);
  if (iree_any_bit_set(unlock_record->flags,
                       LOOM_LOW_SCHEDULE_UNLOCK_FLAG_DESCRIPTOR)) {
    demand.candidate_flags |=
        LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_DESCRIPTOR;
  }
  if (pressure_state->unlocks.descriptor_heads != NULL &&
      unlock_record->descriptor_count != 0 &&
      unlock_record->descriptor_count <=
          LOOM_LOW_SCHEDULE_DESCRIPTOR_FRONTIER_CAPACITY) {
    uint32_t
        descriptor_consumers[LOOM_LOW_SCHEDULE_DESCRIPTOR_FRONTIER_CAPACITY];
    uint32_t consumer_node =
        pressure_state->unlocks.descriptor_heads[node_index];
    for (uint8_t i = 0; i < unlock_record->descriptor_count; ++i) {
      IREE_ASSERT_NE(consumer_node, LOOM_LOW_SCHEDULE_NODE_NONE);
      descriptor_consumers[i] = consumer_node;
      consumer_node =
          pressure_state->unlocks.descriptor_next_nodes[consumer_node];
    }
    if (loom_low_schedule_descriptor_frontier_is_non_growing(
            state, pressure_state, node_index, storage_relation_count,
            killed_live_units, produced_live_units, descriptor_consumers,
            unlock_record->descriptor_count)) {
      demand.candidate_flags |=
          LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_NON_GROWING_DESCRIPTOR;
    }
  }
  demand.pair_affinity_score = loom_low_schedule_score_pair_setup_unlocks(
      state, pressure_state, ready_policy, indegrees, node_index);
  return demand;
}

static uint8_t loom_low_schedule_classify_candidate_pressure(
    const loom_low_schedule_candidate_score_t* score) {
  const bool unlocks_descriptor = iree_any_bit_set(
      score->flags, LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_DESCRIPTOR);
  const bool unlocks_non_growing_descriptor = iree_any_bit_set(
      score->flags,
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_NON_GROWING_DESCRIPTOR);
  const bool has_register_activity = score->killed_live_value_count != 0 ||
                                     score->produced_live_value_count != 0;
  const bool preserves_zero_pressure = iree_any_bit_set(
      score->flags,
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_PRESERVES_ZERO_PRESSURE_PENALTY);
  const bool has_multi_use_result = iree_any_bit_set(
      score->flags, LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_HAS_MULTI_USE_RESULT);
  const bool makes_pressure_progress =
      (unlocks_descriptor && unlocks_non_growing_descriptor) ||
      (score->storage_relation_count != 0 &&
       (score->killed_live_units == 0 && score->produced_live_units == 0)) ||
      (!has_multi_use_result &&
       ((score->storage_relation_count != 0) ||
        (!unlocks_descriptor && score->pressure_cliff_penalty_delta < 0) ||
        (!unlocks_descriptor && has_register_activity &&
         preserves_zero_pressure) ||
        (!unlocks_descriptor && has_register_activity &&
         score->killed_live_units >= score->produced_live_units)));
  uint8_t flags = score->flags;
  if (makes_pressure_progress) {
    flags |= LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_MAKES_PRESSURE_PROGRESS;
  }
  if (score->pressure_cliff_penalty_delta != 0 ||
      (score->units_until_pressure_cliff !=
           LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE &&
       score->units_until_pressure_cliff <= score->produced_live_units)) {
    flags |= LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_PRESSURE_SENSITIVE;
  }
  return flags;
}

static void loom_low_schedule_score_candidate(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_ready_policy_t* ready_policy,
    const uint32_t* indegrees, uint32_t node_index,
    loom_low_schedule_candidate_score_t* out_score) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  loom_low_schedule_reset_candidate_pressure_deltas(pressure_state);
  uint64_t killed_live_units = 0;
  uint32_t killed_live_value_count = 0;
  uint64_t produced_live_units = 0;
  uint32_t produced_live_value_count = 0;
  bool has_multi_use_result = false;
  const uint16_t storage_relation_count =
      node->op != NULL
          ? loom_low_storage_relation_count(state->module, node->op)
          : 0;

  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  for (uint16_t operand_index = 0; operand_index < node->operand_count;
       ++operand_index) {
    const loom_value_ordinal_t value_ordinal = operand_ordinals[operand_index];
    loom_low_schedule_note_candidate_operand_use(pressure_state, value_ordinal);
  }
  for (iree_host_size_t i = 0; i < pressure_state->candidate_operand_count;
       ++i) {
    const loom_value_ordinal_t value_ordinal =
        pressure_state->candidate_operand_ordinals[i];
    const loom_low_schedule_value_record_t* value =
        &state->values[value_ordinal];
    if (!iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
      continue;
    }
    const uint32_t candidate_use_count =
        pressure_state->candidate_operand_use_counts[value_ordinal];
    if (value->remaining_use_count != candidate_use_count) {
      continue;
    }
    const uint32_t transfer_units =
        loom_low_schedule_candidate_alias_transfer_units(state, pressure_state,
                                                         value_ordinal);
    IREE_ASSERT(transfer_units <= value->live_unit_count,
                "transferred alias units must be covered by source pressure");
    const uint32_t unit_count = value->live_unit_count - transfer_units;
    killed_live_units += unit_count;
    ++killed_live_value_count;
    loom_low_schedule_note_candidate_pressure_delta(
        state, pressure_state, value->register_class_id, -(int64_t)unit_count);
  }
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t result_index = 0; result_index < node->result_count;
       ++result_index) {
    const loom_low_schedule_value_record_t* value =
        &state->values[result_ordinals[result_index]];
    has_multi_use_result =
        has_multi_use_result || value->remaining_use_count > 1;
    if (value->remaining_use_count == 0) {
      continue;
    }
    if (iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
      continue;
    }
    const uint32_t alias_units =
        loom_low_schedule_result_alias_units_after_candidate(
            state, pressure_state, node, result_ordinals[result_index]);
    const uint32_t unit_count = value->unit_count - alias_units;
    produced_live_units += unit_count;
    if (unit_count != 0) {
      ++produced_live_value_count;
    }
    loom_low_schedule_note_candidate_pressure_delta(
        state, pressure_state, value->register_class_id, (int64_t)unit_count);
  }
  IREE_ASSERT_LE(killed_live_units, pressure_state->current_live_units);
  uint64_t projected_live_units =
      pressure_state->current_live_units - killed_live_units;
  projected_live_units += produced_live_units;
  const loom_low_schedule_pressure_demand_t pressure_demand =
      loom_low_schedule_score_candidate_pressure_demand(
          state, pressure_state, ready_policy, indegrees, node_index,
          storage_relation_count, killed_live_units, produced_live_units);
  loom_low_schedule_reset_candidate_operand_uses(pressure_state);
  const uint16_t dependency_latency_cycles =
      state->node_dependency_latency_cycles != NULL
          ? state->node_dependency_latency_cycles[node_index]
          : 0;
  uint32_t data_ready_stall_cycles = 0;
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL &&
      state->node_ready_issue_cycles != NULL) {
    data_ready_stall_cycles = loom_low_schedule_positive_delta_u32(
        state->node_ready_issue_cycles[node_index], state->current_issue_cycle);
  }
  uint16_t pair_placement_option_count = 0;
  const uint16_t direct_pair_affinity_score =
      loom_low_schedule_score_candidate_pair_affinity(
          state, node_index, &pair_placement_option_count);
  const uint16_t preferred_anchor_score =
      loom_low_schedule_preferred_pair_anchor_priority(state, indegrees,
                                                       node_index);
  const uint16_t latency_cycles =
      node->schedule_class ? node->schedule_class->latency_cycles : 0;
  *out_score = (loom_low_schedule_candidate_score_t){
      .projected_live_units = projected_live_units,
      .killed_live_units = killed_live_units,
      .killed_live_value_count = killed_live_value_count,
      .produced_live_units = produced_live_units,
      .produced_live_value_count = produced_live_value_count,
      .dependency_latency_cycles = dependency_latency_cycles,
      .latency_cycles = latency_cycles,
      .critical_path_cycles = state->node_critical_path_cycles != NULL
                                  ? state->node_critical_path_cycles[node_index]
                                  : latency_cycles,
      .pressure_demand_units = pressure_demand.demand_units,
      .pressure_reserve_units =
          produced_live_units > killed_live_units &&
                  !iree_any_bit_set(
                      pressure_demand.candidate_flags,
                      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_UNLOCKS_DESCRIPTOR)
              ? pressure_demand.activation_units
              : 0,
      .data_ready_stall_cycles = data_ready_stall_cycles,
      .pair_affinity_score =
          iree_max(preferred_anchor_score,
                   iree_max(direct_pair_affinity_score,
                            pressure_demand.pair_affinity_score)),
      .pair_placement_option_count = pair_placement_option_count,
      .storage_relation_count = storage_relation_count,
      .bottleneck_resource_id = LOOM_LOW_RESOURCE_NONE,
      .pressure_cliff_units = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
      .units_until_pressure_cliff = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
      .source_ordinal = node->source_ordinal,
      .flags = pressure_demand.candidate_flags,
  };
  if (has_multi_use_result) {
    out_score->flags |= LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_HAS_MULTI_USE_RESULT;
  }
  loom_low_schedule_score_candidate_pressure_cliffs(state, pressure_state,
                                                    out_score);
  if (state->pressure_resources != NULL) {
    loom_low_schedule_project_candidate_resource_pressure(state,
                                                          pressure_state);
    loom_low_schedule_score_candidate_resource_pressure(state, pressure_state,
                                                        out_score);
  }
  loom_low_schedule_score_candidate_pressure_limits(state, pressure_state,
                                                    out_score);
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
    out_score->pressure_cliff_penalty_delta =
        (int64_t)out_score->pressure_cliff_penalty -
        (int64_t)pressure_state->current_pressure_cliff_penalty;
    if (pressure_state->current_pressure_cliff_penalty == 0 &&
        out_score->actual_pressure_cliff_penalty == 0) {
      out_score->flags |=
          LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_PRESERVES_ZERO_PRESSURE_PENALTY;
    }
    out_score->flags = loom_low_schedule_classify_candidate_pressure(out_score);
  }
  loom_low_schedule_score_candidate_resources(state, node, out_score);
  loom_low_schedule_score_candidate_hazards(state, node, out_score);
  out_score->effective_stall_cycles = loom_low_schedule_max_u32(
      out_score->data_ready_stall_cycles,
      loom_low_schedule_max_u32(out_score->resource_stall_cycles,
                                out_score->hazard_stall_cycles));
}

static int loom_low_schedule_compare_candidate_pressure(
    const loom_low_schedule_candidate_score_t* lhs,
    const loom_low_schedule_candidate_score_t* rhs) {
  if (lhs->projected_live_units != rhs->projected_live_units) {
    return lhs->projected_live_units < rhs->projected_live_units ? -1 : 1;
  }
  if (lhs->killed_live_units != rhs->killed_live_units) {
    return lhs->killed_live_units > rhs->killed_live_units ? -1 : 1;
  }
  if (lhs->produced_live_units != rhs->produced_live_units) {
    return lhs->produced_live_units < rhs->produced_live_units ? -1 : 1;
  }
  return 0;
}

static uint32_t loom_low_schedule_candidate_positive_pressure_growth(
    const loom_low_schedule_candidate_score_t* score) {
  if (score->produced_live_units <= score->killed_live_units) {
    return 0;
  }
  const uint64_t growth = score->produced_live_units - score->killed_live_units;
  return growth > UINT32_MAX ? UINT32_MAX : (uint32_t)growth;
}

static int loom_low_schedule_compare_candidate_pressure_efficiency(
    const loom_low_schedule_candidate_score_t* lhs,
    const loom_low_schedule_candidate_score_t* rhs) {
  const uint32_t lhs_growth =
      loom_low_schedule_candidate_positive_pressure_growth(lhs);
  const uint32_t rhs_growth =
      loom_low_schedule_candidate_positive_pressure_growth(rhs);
  const uint32_t lhs_demand =
      lhs->pressure_demand_units != 0 ? lhs->pressure_demand_units : 1;
  const uint32_t rhs_demand =
      rhs->pressure_demand_units != 0 ? rhs->pressure_demand_units : 1;
  const uint64_t lhs_cost = (uint64_t)lhs_growth * rhs_demand;
  const uint64_t rhs_cost = (uint64_t)rhs_growth * lhs_demand;
  if (lhs_cost == rhs_cost) {
    return 0;
  }
  return lhs_cost < rhs_cost ? -1 : 1;
}

static int loom_low_schedule_compare_candidate_live_values(
    const loom_low_schedule_candidate_score_t* lhs,
    const loom_low_schedule_candidate_score_t* rhs) {
  const int64_t lhs_delta = (int64_t)lhs->produced_live_value_count -
                            (int64_t)lhs->killed_live_value_count;
  const int64_t rhs_delta = (int64_t)rhs->produced_live_value_count -
                            (int64_t)rhs->killed_live_value_count;
  if (lhs_delta != rhs_delta) {
    return lhs_delta < rhs_delta ? -1 : 1;
  }
  if (lhs->killed_live_value_count != rhs->killed_live_value_count) {
    return lhs->killed_live_value_count > rhs->killed_live_value_count ? -1 : 1;
  }
  if (lhs->produced_live_value_count != rhs->produced_live_value_count) {
    return lhs->produced_live_value_count < rhs->produced_live_value_count ? -1
                                                                           : 1;
  }
  return 0;
}

static bool loom_low_schedule_candidate_shortens_producer_live_range(
    const loom_low_schedule_candidate_score_t* score) {
  return score->dependency_latency_cycles != 0 &&
         score->killed_live_units > score->produced_live_units;
}

static bool loom_low_schedule_candidate_compacts_live_values(
    const loom_low_schedule_candidate_score_t* score) {
  return score->storage_relation_count != 0 &&
         score->killed_live_value_count > score->produced_live_value_count;
}

static bool loom_low_schedule_candidate_has_better_pair_affinity(
    const loom_low_schedule_candidate_score_t* lhs,
    const loom_low_schedule_candidate_score_t* rhs) {
  if (lhs->pair_affinity_score != rhs->pair_affinity_score) {
    return lhs->pair_affinity_score > rhs->pair_affinity_score;
  }
  return lhs->pair_placement_option_count > rhs->pair_placement_option_count;
}

static bool loom_low_schedule_candidate_pair_affinity_differs(
    const loom_low_schedule_candidate_score_t* lhs,
    const loom_low_schedule_candidate_score_t* rhs) {
  return lhs->pair_affinity_score != rhs->pair_affinity_score ||
         (lhs->pair_affinity_score != 0 &&
          lhs->pair_placement_option_count != rhs->pair_placement_option_count);
}

static bool loom_low_schedule_candidate_threatens_pressure_cliff(
    const loom_low_schedule_candidate_score_t* score) {
  return score->pressure_cliff_penalty_delta > 0;
}

static loom_low_schedule_candidate_compare_mode_t
loom_low_schedule_choose_candidate_compare_mode(
    const loom_low_schedule_candidate_score_t* scores,
    iree_host_size_t score_count) {
  uint8_t combined_flags = 0;
  for (iree_host_size_t i = 0; i < score_count; ++i) {
    combined_flags |= scores[i].flags;
  }
  const uint8_t pressure_relief_flags =
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_PRESSURE_SENSITIVE |
      LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_MAKES_PRESSURE_PROGRESS;
  return iree_all_bits_set(combined_flags, pressure_relief_flags)
             ? LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF
             : LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT;
}

static bool loom_low_schedule_candidate_score_less(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_candidate_compare_mode_t compare_mode,
    const loom_low_schedule_candidate_score_t* lhs,
    const loom_low_schedule_candidate_score_t* rhs) {
  const int pressure_order =
      loom_low_schedule_compare_candidate_pressure(lhs, rhs);
  const int pressure_efficiency_order =
      loom_low_schedule_compare_candidate_pressure_efficiency(lhs, rhs);
  const int live_value_order =
      loom_low_schedule_compare_candidate_live_values(lhs, rhs);
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
    if (compare_mode == LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF) {
      const bool lhs_makes_pressure_progress = iree_any_bit_set(
          lhs->flags, LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_MAKES_PRESSURE_PROGRESS);
      const bool rhs_makes_pressure_progress = iree_any_bit_set(
          rhs->flags, LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_MAKES_PRESSURE_PROGRESS);
      if (lhs_makes_pressure_progress != rhs_makes_pressure_progress) {
        return lhs_makes_pressure_progress;
      }
      if (lhs_makes_pressure_progress) {
        if (lhs->pressure_cliff_penalty != rhs->pressure_cliff_penalty) {
          return lhs->pressure_cliff_penalty < rhs->pressure_cliff_penalty;
        }
        if (pressure_efficiency_order != 0) {
          return pressure_efficiency_order < 0;
        }
        if (pressure_order != 0) {
          return pressure_order < 0;
        }
        if (lhs->critical_path_cycles != rhs->critical_path_cycles) {
          return lhs->critical_path_cycles > rhs->critical_path_cycles;
        }
      }
    }
    if (lhs->effective_stall_cycles != rhs->effective_stall_cycles) {
      return lhs->effective_stall_cycles < rhs->effective_stall_cycles;
    }
    if (lhs->hazard_stall_cycles != rhs->hazard_stall_cycles) {
      return lhs->hazard_stall_cycles < rhs->hazard_stall_cycles;
    }
    if (lhs->resource_stall_cycles != rhs->resource_stall_cycles) {
      return lhs->resource_stall_cycles < rhs->resource_stall_cycles;
    }
    if (lhs->data_ready_stall_cycles != rhs->data_ready_stall_cycles) {
      return lhs->data_ready_stall_cycles < rhs->data_ready_stall_cycles;
    }
    if (loom_low_schedule_candidate_pair_affinity_differs(lhs, rhs)) {
      return loom_low_schedule_candidate_has_better_pair_affinity(lhs, rhs);
    }
    if (live_value_order != 0 &&
        (loom_low_schedule_candidate_compacts_live_values(lhs) ||
         loom_low_schedule_candidate_compacts_live_values(rhs)) &&
        (pressure_order == 0 || pressure_order == live_value_order)) {
      return live_value_order < 0;
    }
    if (pressure_order != 0) {
      const bool lhs_shortens =
          loom_low_schedule_candidate_shortens_producer_live_range(lhs);
      const bool rhs_shortens =
          loom_low_schedule_candidate_shortens_producer_live_range(rhs);
      if (lhs_shortens && !rhs_shortens && pressure_order < 0) {
        return true;
      }
      if (rhs_shortens && !lhs_shortens && pressure_order > 0) {
        return false;
      }
      if (lhs_shortens && rhs_shortens) {
        return pressure_order < 0;
      }
    }
    if (lhs->critical_path_cycles != rhs->critical_path_cycles) {
      return lhs->critical_path_cycles > rhs->critical_path_cycles;
    }
  }
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING) {
    if (lhs->dependency_latency_cycles != rhs->dependency_latency_cycles) {
      return lhs->dependency_latency_cycles < rhs->dependency_latency_cycles;
    }
    if (lhs->latency_cycles != rhs->latency_cycles) {
      return lhs->latency_cycles > rhs->latency_cycles;
    }
  }
  if (state->options->strategy != LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL &&
      lhs->pressure_cliff_penalty != rhs->pressure_cliff_penalty) {
    return lhs->pressure_cliff_penalty < rhs->pressure_cliff_penalty;
  }
  if (loom_low_schedule_candidate_pair_affinity_differs(lhs, rhs)) {
    return loom_low_schedule_candidate_has_better_pair_affinity(lhs, rhs);
  }
  if (pressure_efficiency_order != 0) {
    return pressure_efficiency_order < 0;
  }
  if (pressure_order != 0) {
    return pressure_order < 0;
  }
  return lhs->source_ordinal < rhs->source_ordinal;
}

static void loom_low_schedule_record_candidate_decision(
    loom_low_schedule_build_state_t* state, uint32_t block_index,
    uint32_t scheduled_ordinal, uint32_t ready_candidate_count,
    uint32_t scored_candidate_count, uint32_t chosen_node,
    const loom_low_schedule_candidate_score_t* chosen_score,
    uint32_t rejected_node,
    const loom_low_schedule_candidate_score_t* rejected_score) {
  if (!state->candidate_decisions) {
    return;
  }
  if (rejected_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return;
  }
  state->candidate_decisions[state->candidate_decision_count++] =
      (loom_low_schedule_candidate_decision_t){
          .block_index = block_index,
          .scheduled_ordinal = scheduled_ordinal,
          .ready_candidate_count = ready_candidate_count,
          .scored_candidate_count = scored_candidate_count,
          .chosen_node = chosen_node,
          .rejected_node = rejected_node,
          .chosen_dependency_latency_cycles =
              chosen_score->dependency_latency_cycles,
          .chosen_latency_cycles = chosen_score->latency_cycles,
          .chosen_pair_affinity_score = chosen_score->pair_affinity_score,
          .rejected_dependency_latency_cycles =
              rejected_score->dependency_latency_cycles,
          .rejected_latency_cycles = rejected_score->latency_cycles,
          .rejected_pair_affinity_score = rejected_score->pair_affinity_score,
          .chosen_projected_live_units = chosen_score->projected_live_units,
          .chosen_killed_live_units = chosen_score->killed_live_units,
          .chosen_produced_live_units = chosen_score->produced_live_units,
          .rejected_projected_live_units = rejected_score->projected_live_units,
          .rejected_killed_live_units = rejected_score->killed_live_units,
          .rejected_produced_live_units = rejected_score->produced_live_units,
          .chosen_data_ready_stall_cycles =
              chosen_score->data_ready_stall_cycles,
          .chosen_resource_stall_cycles = chosen_score->resource_stall_cycles,
          .chosen_hazard_stall_cycles = chosen_score->hazard_stall_cycles,
          .chosen_effective_stall_cycles = chosen_score->effective_stall_cycles,
          .chosen_bottleneck_resource_id = chosen_score->bottleneck_resource_id,
          .chosen_pressure_cliff_penalty = chosen_score->pressure_cliff_penalty,
          .chosen_pressure_cliff_source =
              loom_low_schedule_pressure_source_name(
                  state, chosen_score->pressure_cliff_source_kind,
                  chosen_score->pressure_cliff_source_id),
          .chosen_pressure_cliff_units = chosen_score->pressure_cliff_units,
          .chosen_units_until_pressure_cliff =
              chosen_score->units_until_pressure_cliff,
          .rejected_data_ready_stall_cycles =
              rejected_score->data_ready_stall_cycles,
          .rejected_resource_stall_cycles =
              rejected_score->resource_stall_cycles,
          .rejected_hazard_stall_cycles = rejected_score->hazard_stall_cycles,
          .rejected_effective_stall_cycles =
              rejected_score->effective_stall_cycles,
          .rejected_bottleneck_resource_id =
              rejected_score->bottleneck_resource_id,
          .rejected_pressure_cliff_penalty =
              rejected_score->pressure_cliff_penalty,
          .rejected_pressure_cliff_source =
              loom_low_schedule_pressure_source_name(
                  state, rejected_score->pressure_cliff_source_kind,
                  rejected_score->pressure_cliff_source_id),
          .rejected_pressure_cliff_units = rejected_score->pressure_cliff_units,
          .rejected_units_until_pressure_cliff =
              rejected_score->units_until_pressure_cliff,
      };
}

static void loom_low_schedule_note_pair_affinity_node_scheduled(
    loom_low_schedule_build_state_t* state, uint32_t node_index) {
  if (loom_low_schedule_pair_affinity_list_is_empty(
          state->options->pair_affinities)) {
    state->pending_pair_affinity_node = LOOM_LOW_SCHEDULE_NODE_NONE;
    return;
  }

  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  if (node->descriptor != NULL) {
    if (state->pending_pair_affinity_node != LOOM_LOW_SCHEDULE_NODE_NONE) {
      const loom_low_schedule_node_t* anchor =
          &state->nodes[state->pending_pair_affinity_node];
      const loom_low_schedule_pair_affinity_record_t* affinity =
          loom_low_schedule_find_pair_affinity(state, anchor, node);
      if (affinity != NULL &&
          !loom_low_schedule_node_result_used_by(anchor, node)) {
        const uint16_t placement_recipe_index =
            affinity->placement_recipe_index;
        if (placement_recipe_index != LOOM_LOW_PLACEMENT_PAIR_RECIPE_NONE) {
          const uint16_t recipe_index = (uint16_t)(placement_recipe_index - 1u);
          IREE_ASSERT_LT(
              recipe_index,
              state->options->pair_affinities.placement_recipe_count);
          const loom_low_placement_pair_recipe_t* recipe =
              &state->options->pair_affinities.placement_recipes[recipe_index];
          IREE_ASSERT_NE(recipe->relation_count, 0);
          IREE_ASSERT_NE(recipe->alternative_count, 0);
          IREE_ASSERT(state->placement_pair_uses != NULL);
          IREE_ASSERT_LT(state->placement_pair_use_count,
                         state->scheduled_node_count / 2);
          state->placement_pair_uses[state->placement_pair_use_count++] =
              (loom_low_placement_pair_use_t){
                  .first_op = anchor->op,
                  .second_op = node->op,
                  .placement_recipe_index = placement_recipe_index,
                  .priority = affinity->priority,
              };
        }
        state->pending_pair_affinity_node = LOOM_LOW_SCHEDULE_NODE_NONE;
        return;
      }
    }
    state->pending_pair_affinity_node =
        loom_low_schedule_node_can_start_pair_affinity(state, node)
            ? node_index
            : LOOM_LOW_SCHEDULE_NODE_NONE;
    return;
  }

  if (loom_low_schedule_node_is_pair_transparent(node)) {
    return;
  }
  state->pending_pair_affinity_node = LOOM_LOW_SCHEDULE_NODE_NONE;
}

static uint32_t loom_low_schedule_transfer_alias_units_from_source(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t source_ordinal) {
  if (pressure_state->alias_heads == NULL) {
    return 0;
  }
  uint32_t transfer_units = 0;
  for (uint32_t alias_index = pressure_state->alias_heads[source_ordinal];
       alias_index != LOOM_LOW_SCHEDULE_NODE_NONE;
       alias_index = pressure_state->aliases[alias_index].next_alias) {
    loom_low_schedule_pressure_alias_t* alias =
        &pressure_state->aliases[alias_index];
    if (alias->transferred) {
      continue;
    }
    loom_low_schedule_value_record_t* result =
        &state->values[alias->result_ordinal];
    if (!iree_any_bit_set(result->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
      continue;
    }
    IREE_ASSERT(
        alias->unit_count <= result->unit_count &&
            result->live_unit_count <= result->unit_count - alias->unit_count,
        "transferred alias units must fit result pressure units");
    result->live_unit_count += alias->unit_count;
    alias->transferred = true;
    transfer_units += alias->unit_count;
  }
  return transfer_units;
}

static bool loom_low_schedule_value_is_live_with_full_pressure(
    const loom_low_schedule_value_record_t* value, uint32_t unit_count) {
  return iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE) &&
         value->live_unit_count == value->unit_count &&
         value->unit_count >= unit_count;
}

static uint32_t loom_low_schedule_append_result_aliases(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_node_t* node, loom_value_ordinal_t result_ordinal) {
  if (pressure_state->alias_heads == NULL) {
    return 0;
  }
  uint32_t alias_units = 0;
  loom_low_storage_relation_iterator_t iterator;
  loom_low_storage_relation_iterator_initialize(state->module, node->op,
                                                &iterator);
  loom_low_storage_relation_t relation;
  while (loom_low_storage_relation_iterator_next(&iterator, &relation)) {
    const loom_value_ordinal_t destination_ordinal =
        loom_local_value_domain_ordinal(state->value_domain,
                                        relation.destination_value_id);
    if (destination_ordinal != result_ordinal) {
      continue;
    }
    const loom_value_ordinal_t source_ordinal = loom_local_value_domain_ordinal(
        state->value_domain, relation.source_value_id);
    if (!loom_low_schedule_storage_relation_can_alias_pressure(
            state, &relation, result_ordinal, source_ordinal)) {
      continue;
    }
    const loom_low_schedule_value_record_t* source =
        &state->values[source_ordinal];
    if (!loom_low_schedule_value_is_live_with_full_pressure(
            source, relation.unit_count)) {
      continue;
    }
    const uint32_t claimed_source_units =
        pressure_state->alias_source_unit_counts[source_ordinal];
    IREE_ASSERT(claimed_source_units <= source->unit_count,
                "active alias units must fit source pressure units");
    if (claimed_source_units == source->unit_count) {
      continue;
    }
    const uint32_t aliasable_units = iree_min(
        relation.unit_count, source->unit_count - claimed_source_units);
    IREE_ASSERT(pressure_state->alias_count < pressure_state->alias_capacity,
                "storage relation alias count must fit precomputed capacity");
    if (pressure_state->alias_heads[source_ordinal] ==
        LOOM_LOW_SCHEDULE_NODE_NONE) {
      pressure_state
          ->alias_source_ordinals[pressure_state->alias_source_count++] =
          source_ordinal;
    }
    const uint32_t alias_index = (uint32_t)pressure_state->alias_count++;
    pressure_state->aliases[alias_index] = (loom_low_schedule_pressure_alias_t){
        .source_ordinal = source_ordinal,
        .result_ordinal = result_ordinal,
        .unit_count = aliasable_units,
        .next_alias = pressure_state->alias_heads[source_ordinal],
        .transferred = false,
    };
    pressure_state->alias_heads[source_ordinal] = alias_index;
    pressure_state->alias_source_unit_counts[source_ordinal] += aliasable_units;
    alias_units += aliasable_units;
  }
  const loom_low_schedule_value_record_t* result =
      &state->values[result_ordinal];
  IREE_ASSERT(alias_units <= result->unit_count,
              "verified storage relations must not over-cover result units");
  return alias_units;
}

static void loom_low_schedule_note_pressure_node_scheduled(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index,
    const loom_low_schedule_candidate_score_t* score) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  uint64_t live_units_before = pressure_state->current_live_units;
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  for (uint16_t operand_index = 0; operand_index < node->operand_count;
       ++operand_index) {
    loom_low_schedule_value_record_t* value =
        &state->values[operand_ordinals[operand_index]];
    if (value->remaining_use_count == 0) {
      continue;
    }
    --value->remaining_use_count;
    if (value->remaining_use_count == 0 &&
        iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
      value->flags &= ~LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
      const uint32_t transfer_units =
          loom_low_schedule_transfer_alias_units_from_source(
              state, pressure_state, operand_ordinals[operand_index]);
      IREE_ASSERT(transfer_units <= value->live_unit_count,
                  "transferred alias units must be covered by source "
                  "pressure");
      const uint32_t unit_count = value->live_unit_count - transfer_units;
      value->live_unit_count = 0;
      IREE_ASSERT_LE(unit_count, pressure_state->current_live_units);
      pressure_state->current_live_units -= unit_count;
      const uint16_t reg_class_id = value->register_class_id;
      if (reg_class_id != LOOM_LOW_REG_CLASS_NONE &&
          pressure_state->current_live_units_by_reg_class) {
        IREE_ASSERT_LE(
            unit_count,
            pressure_state->current_live_units_by_reg_class[reg_class_id]);
        pressure_state->current_live_units_by_reg_class[reg_class_id] -=
            unit_count;
        loom_low_schedule_adjust_alias_pressure(
            state, pressure_state, reg_class_id, -(int64_t)unit_count);
      }
    }
  }

  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t result_index = 0; result_index < node->result_count;
       ++result_index) {
    loom_low_schedule_value_record_t* value =
        &state->values[result_ordinals[result_index]];
    if (value->remaining_use_count == 0) {
      continue;
    }
    if (iree_any_bit_set(value->flags, LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE)) {
      continue;
    }
    value->flags |= LOOM_LOW_SCHEDULE_VALUE_FLAG_LIVE;
    const uint32_t alias_units = loom_low_schedule_append_result_aliases(
        state, pressure_state, node, result_ordinals[result_index]);
    const uint32_t unit_count = value->unit_count - alias_units;
    value->live_unit_count = unit_count;
    pressure_state->current_live_units += unit_count;
    const uint16_t reg_class_id = value->register_class_id;
    if (reg_class_id != LOOM_LOW_REG_CLASS_NONE &&
        pressure_state->current_live_units_by_reg_class) {
      loom_low_schedule_note_block_pressure_reg_class(pressure_state,
                                                      reg_class_id);
      pressure_state->current_live_units_by_reg_class[reg_class_id] +=
          unit_count;
      loom_low_schedule_adjust_alias_pressure(
          state, pressure_state, reg_class_id, (int64_t)unit_count);
      loom_low_schedule_note_resource_high_water(
          state, pressure_state, reg_class_id,
          LOOM_LOW_SCHEDULE_RESOURCE_HIGH_WATER_SCHEDULED);
    }
  }
  IREE_ASSERT_EQ(pressure_state->current_live_units,
                 score->projected_live_units);
  // The realized pressure state exactly matches the candidate projection
  // checked above. Preserve its actual penalty while discarding any
  // speculative reserve used only to rank the candidate.
  pressure_state->current_pressure_cliff_penalty =
      score->actual_pressure_cliff_penalty;
  if (state->pressure_steps) {
    state->pressure_steps[state->pressure_step_count++] =
        (loom_low_schedule_pressure_step_t){
            .node_index = node_index,
            .block_index = node->block_index,
            .scheduled_ordinal = node->scheduled_ordinal,
            .live_units_before = live_units_before,
            .killed_live_units = score->killed_live_units,
            .produced_live_units = score->produced_live_units,
            .live_units_after = pressure_state->current_live_units,
        };
  }
}

static void loom_low_schedule_ready_policy_update_pressure_consumers(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_ready_policy_t* ready_policy, uint32_t scheduled_node) {
  if (ready_policy->remaining_consumer_counts == NULL) return;
  const loom_low_schedule_node_t* node = &state->nodes[scheduled_node];
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(node);
  for (uint16_t operand_index = 0; operand_index < node->operand_count;
       ++operand_index) {
    loom_low_schedule_note_candidate_operand_use(
        pressure_state, operand_ordinals[operand_index]);
  }
  for (iree_host_size_t i = 0; i < pressure_state->candidate_operand_count;
       ++i) {
    const loom_value_ordinal_t value_ordinal =
        pressure_state->candidate_operand_ordinals[i];
    uint32_t* remaining_consumer_count =
        &ready_policy->remaining_consumer_counts[value_ordinal];
    uint32_t* remaining_consumer_node_xor =
        &ready_policy->remaining_consumer_node_xors[value_ordinal];
    IREE_ASSERT_NE(*remaining_consumer_count, 0u);
    --*remaining_consumer_count;
    *remaining_consumer_node_xor ^= scheduled_node;
    // A candidate can kill the value only when it is the sole distinct
    // consumer. The XOR identifies that consumer without walking the fan-out.
    if (*remaining_consumer_count == 1) {
      const uint32_t consumer_node = *remaining_consumer_node_xor;
      if (!loom_low_schedule_ready_frontier_contains(&ready_policy->frontier,
                                                     consumer_node)) {
        continue;
      }
      loom_low_schedule_ready_frontier_update_key(
          &ready_policy->frontier, LOOM_LOW_SCHEDULE_READY_VIEW_PRESSURE,
          consumer_node,
          loom_low_schedule_ready_pressure_key(state, pressure_state,
                                               consumer_node));
    }
  }
  loom_low_schedule_reset_candidate_operand_uses(pressure_state);
}

static bool loom_low_schedule_node_is_unscheduled_in_block(
    const loom_low_schedule_build_state_t* state, uint32_t node_index,
    iree_host_size_t node_count, iree_host_size_t block_index) {
  return node_index < node_count &&
         state->nodes[node_index].block_index == block_index &&
         state->nodes[node_index].scheduled_ordinal ==
             LOOM_LOW_SCHEDULE_NODE_NONE;
}

static bool loom_low_schedule_node_is_source_order_boundary(
    const loom_low_schedule_node_t* node) {
  return iree_any_bit_set(node->flags,
                          LOOM_LOW_SCHEDULE_NODE_FLAG_SOURCE_ORDER_BOUNDARY);
}

// Returns the exclusive end of the next independently reorderable source
// range. Boundary nodes occupy one-node ranges so every earlier range is fully
// scheduled before the boundary and every later range begins after it.
static uint32_t loom_low_schedule_source_range_end(
    const loom_low_schedule_build_state_t* state, uint32_t range_start,
    uint32_t block_node_end) {
  if (loom_low_schedule_node_is_source_order_boundary(
          &state->nodes[range_start])) {
    return range_start + 1;
  }
  uint32_t range_end = range_start + 1;
  while (range_end < block_node_end &&
         !loom_low_schedule_node_is_source_order_boundary(
             &state->nodes[range_end])) {
    ++range_end;
  }
  return range_end;
}

static void loom_low_schedule_initialize_source_range_ready_frontier(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_ready_policy_t* ready_policy, const uint32_t* indegrees,
    uint32_t range_start, uint32_t range_end) {
  IREE_ASSERT_EQ(
      loom_low_schedule_ready_frontier_count(&ready_policy->frontier), 0u);
  for (uint32_t node_index = range_start; node_index < range_end;
       ++node_index) {
    if (indegrees[node_index] == 0) {
      loom_low_schedule_ready_policy_insert(state, pressure_state, ready_policy,
                                            node_index);
    }
  }
}

static uint32_t loom_low_schedule_count_unscheduled_nodes_in_block(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_block_t* block_record) {
  uint32_t unscheduled_count = 0;
  const uint32_t block_node_end =
      block_record->node_start + block_record->node_count;
  for (uint32_t node_index = block_record->node_start;
       node_index < block_node_end; ++node_index) {
    if (state->nodes[node_index].scheduled_ordinal ==
        LOOM_LOW_SCHEDULE_NODE_NONE) {
      ++unscheduled_count;
    }
  }
  return unscheduled_count;
}

static void loom_low_schedule_record_dependency_cycle_path(
    loom_low_schedule_failure_t* failure, const uint32_t* parent_nodes,
    uint32_t producer_node, uint32_t consumer_node) {
  uint32_t reverse_nodes[LOOM_LOW_SCHEDULE_FAILURE_CYCLE_NODE_CAPACITY];
  uint32_t reverse_count = 0;
  bool truncated = false;
  uint32_t cursor = producer_node;
  while (cursor != LOOM_LOW_SCHEDULE_NODE_NONE) {
    if (reverse_count < IREE_ARRAYSIZE(reverse_nodes)) {
      reverse_nodes[reverse_count++] = cursor;
    } else {
      truncated = true;
    }
    if (cursor == consumer_node) {
      break;
    }
    cursor = parent_nodes[cursor];
  }
  if (cursor != consumer_node) {
    failure->flags |= LOOM_LOW_SCHEDULE_FAILURE_FLAG_WITNESS_EDGE_ONLY;
    return;
  }
  failure->cycle_node_count = 0;
  for (uint32_t i = reverse_count; i > 0; --i) {
    failure->cycle_nodes[failure->cycle_node_count++] = reverse_nodes[i - 1];
  }
  if (truncated) {
    failure->flags |= LOOM_LOW_SCHEDULE_FAILURE_FLAG_CYCLE_PATH_TRUNCATED;
  }
}

static void loom_low_schedule_record_dependency_cycle_failure(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_block_t* block_record, uint32_t scheduled_in_block,
    uint32_t producer_node, uint32_t consumer_node,
    const loom_low_schedule_dependency_t* dependency) {
  state->failure = (loom_low_schedule_failure_t){
      .kind = LOOM_LOW_SCHEDULE_FAILURE_DEPENDENCY_CYCLE,
      .flags = 0,
      .block_index = state->current_block_index,
      .block_node_count = block_record->node_count,
      .scheduled_node_count = scheduled_in_block,
      .unscheduled_node_count =
          loom_low_schedule_count_unscheduled_nodes_in_block(state,
                                                             block_record),
      .producer_node = producer_node,
      .consumer_node = consumer_node,
      .dependency_kind = dependency->kind,
      .operand_index = dependency->operand_index,
      .cycle_node_count = 0,
  };
}

static const loom_low_schedule_dependency_t*
loom_low_schedule_find_dependency_witness(
    const loom_low_schedule_build_state_t* state, uint32_t producer_node,
    uint32_t consumer_node) {
  for (uint32_t dependency_index = 0;
       dependency_index < state->dependencies.count; ++dependency_index) {
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_dependency_graph_at(&state->dependencies,
                                              dependency_index);
    if (dependency->producer_node == producer_node &&
        dependency->consumer_node == consumer_node) {
      return dependency;
    }
  }
  IREE_ASSERT_UNREACHABLE("dependency group must retain a raw witness");
  return NULL;
}

static iree_status_t loom_low_schedule_record_first_unresolved_dependency(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_block_t* block_record, iree_host_size_t node_count,
    uint32_t scheduled_in_block) {
  for (uint32_t i = 0; i < state->dependencies.count; ++i) {
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_dependency_graph_at(&state->dependencies, i);
    if (!loom_low_schedule_node_is_unscheduled_in_block(
            state, dependency->consumer_node, node_count,
            state->current_block_index)) {
      continue;
    }
    if (dependency->producer_node >= node_count ||
        state->nodes[dependency->producer_node].scheduled_ordinal !=
            LOOM_LOW_SCHEDULE_NODE_NONE) {
      continue;
    }
    loom_low_schedule_record_dependency_cycle_failure(
        state, block_record, scheduled_in_block, dependency->producer_node,
        dependency->consumer_node, dependency);
    state->failure.flags |= LOOM_LOW_SCHEDULE_FAILURE_FLAG_WITNESS_EDGE_ONLY;
    state->failure.cycle_nodes[0] = dependency->producer_node;
    state->failure.cycle_nodes[1] = dependency->consumer_node;
    state->failure.cycle_node_count = 2;
    return iree_ok_status();
  }
  state->failure = (loom_low_schedule_failure_t){
      .kind = LOOM_LOW_SCHEDULE_FAILURE_DEPENDENCY_CYCLE,
      .flags = LOOM_LOW_SCHEDULE_FAILURE_FLAG_WITNESS_EDGE_ONLY,
      .block_index = state->current_block_index,
      .block_node_count = block_record->node_count,
      .scheduled_node_count = scheduled_in_block,
      .unscheduled_node_count =
          loom_low_schedule_count_unscheduled_nodes_in_block(state,
                                                             block_record),
      .producer_node = LOOM_LOW_SCHEDULE_NODE_NONE,
      .consumer_node = LOOM_LOW_SCHEDULE_NODE_NONE,
      .dependency_kind = LOOM_LOW_SCHEDULE_DEPENDENCY_UNKNOWN,
      .operand_index = UINT32_MAX,
      .cycle_node_count = 0,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_record_dependency_cycle(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_block_t* block_record, iree_host_size_t node_count,
    uint32_t scheduled_in_block) {
  uint8_t* visit_states = NULL;
  uint32_t* parent_nodes = NULL;
  uint32_t* stack_nodes = NULL;
  uint32_t* stack_next_groups = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, node_count, sizeof(*visit_states), (void**)&visit_states));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, node_count, sizeof(*parent_nodes), (void**)&parent_nodes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, node_count, sizeof(*stack_nodes), (void**)&stack_nodes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->arena, node_count,
                                                 sizeof(*stack_next_groups),
                                                 (void**)&stack_next_groups));
  memset(visit_states, 0, node_count * sizeof(*visit_states));
  memset(parent_nodes, 0xFF, node_count * sizeof(*parent_nodes));

  const uint32_t block_node_end =
      block_record->node_start + block_record->node_count;
  for (uint32_t start_node = block_record->node_start;
       start_node < block_node_end; ++start_node) {
    if (!loom_low_schedule_node_is_unscheduled_in_block(
            state, start_node, node_count, state->current_block_index) ||
        visit_states[start_node] != 0) {
      continue;
    }
    iree_host_size_t stack_count = 1;
    stack_nodes[0] = start_node;
    stack_next_groups[0] = loom_low_schedule_dependency_index_group_begin(
        &state->dependency_index, start_node);
    visit_states[start_node] = 1;
    while (stack_count != 0) {
      const uint32_t producer_node = stack_nodes[stack_count - 1];
      bool advanced = false;
      const uint32_t group_end = loom_low_schedule_dependency_index_group_end(
          &state->dependency_index, producer_node);
      for (uint32_t group_index = stack_next_groups[stack_count - 1];
           group_index < group_end; ++group_index) {
        stack_next_groups[stack_count - 1] = group_index + 1;
        const uint32_t consumer_node =
            loom_low_schedule_dependency_index_group_at(
                &state->dependency_index, group_index)
                ->consumer_node;
        if (!loom_low_schedule_node_is_unscheduled_in_block(
                state, consumer_node, node_count, state->current_block_index)) {
          continue;
        }
        if (visit_states[consumer_node] == 0) {
          parent_nodes[consumer_node] = producer_node;
          stack_nodes[stack_count] = consumer_node;
          stack_next_groups[stack_count] =
              loom_low_schedule_dependency_index_group_begin(
                  &state->dependency_index, consumer_node);
          visit_states[consumer_node] = 1;
          ++stack_count;
          advanced = true;
          break;
        }
        if (visit_states[consumer_node] == 1) {
          const loom_low_schedule_dependency_t* dependency =
              loom_low_schedule_find_dependency_witness(state, producer_node,
                                                        consumer_node);
          loom_low_schedule_record_dependency_cycle_failure(
              state, block_record, scheduled_in_block, producer_node,
              consumer_node, dependency);
          loom_low_schedule_record_dependency_cycle_path(
              &state->failure, parent_nodes, producer_node, consumer_node);
          return iree_ok_status();
        }
      }
      if (advanced) {
        continue;
      }
      visit_states[producer_node] = 2;
      --stack_count;
    }
  }

  return loom_low_schedule_record_first_unresolved_dependency(
      state, block_record, node_count, scheduled_in_block);
}

static iree_status_t loom_low_schedule_handle_dependency_cycle(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_block_t* block_record, iree_host_size_t node_count,
    uint32_t scheduled_in_block) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_record_dependency_cycle(
      state, block_record, node_count, scheduled_in_block));
  if (state->options->emitter.fn == NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low schedule dependency cycle in block %" PRIu32,
                            state->current_block_index);
  }
  IREE_RETURN_IF_ERROR(
      loom_low_schedule_emit_dependency_cycle(state, &state->failure));
  ++state->error_count;
  return iree_ok_status();
}

static void loom_low_schedule_compute_node_priorities(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count,
    const loom_low_schedule_dependency_detail_index_t* dependency_details,
    loom_low_schedule_pressure_state_t* pressure_state) {
  if (state->node_critical_path_cycles == NULL &&
      state->node_dependency_latency_cycles == NULL &&
      state->node_pressure_demand_units == NULL &&
      state->node_pressure_activation_units == NULL &&
      pressure_state->first_actionable_pressure_cliff_indices == NULL) {
    return;
  }
  for (iree_host_size_t i = node_count; i > 0; --i) {
    const uint32_t node_index = (uint32_t)(i - 1);
    const loom_low_schedule_node_t* node = &state->nodes[node_index];
    uint16_t dependency_latency_cycles = 0;
    if (state->node_dependency_latency_cycles != NULL) {
      const loom_value_ordinal_t* operand_ordinals =
          loom_low_schedule_node_const_operand_ordinals(node);
      for (uint16_t operand_index = 0; operand_index < node->operand_count;
           ++operand_index) {
        const uint32_t producer_node =
            state->values[operand_ordinals[operand_index]].producer_node;
        if (producer_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
            state->nodes[producer_node].block != node->block) {
          continue;
        }
        const loom_low_schedule_class_t* producer_schedule_class =
            state->nodes[producer_node].schedule_class;
        const uint16_t producer_latency =
            producer_schedule_class != NULL
                ? producer_schedule_class->latency_cycles
                : 0;
        dependency_latency_cycles =
            iree_max(dependency_latency_cycles, producer_latency);
      }
      state->node_dependency_latency_cycles[node_index] =
          dependency_latency_cycles;
    }
    uint32_t successor_path_cycles = 0;
    uint32_t pressure_demand_units = 0;
    uint32_t pressure_activation_units = 0;
    if (dependency_details->dependency_count != 0) {
      const uint32_t dependency_begin =
          dependency_details->producer_dependency_starts[node_index];
      const uint32_t dependency_end =
          dependency_details->producer_dependency_starts[node_index + 1];
      for (uint32_t i = dependency_begin; i < dependency_end; ++i) {
        const uint32_t dependency_index =
            loom_low_schedule_dependency_detail_index_at(dependency_details, i);
        const loom_low_schedule_dependency_t* dependency =
            loom_low_schedule_dependency_graph_at(&state->dependencies,
                                                  dependency_index);
        if (dependency->producer_node != node_index ||
            dependency->consumer_node >= node_count) {
          continue;
        }
        const loom_low_schedule_node_t* consumer =
            &state->nodes[dependency->consumer_node];
        if (consumer->block_index != node->block_index) {
          continue;
        }
        if (state->node_critical_path_cycles != NULL) {
          successor_path_cycles = loom_low_schedule_max_u32(
              successor_path_cycles,
              state->node_critical_path_cycles[dependency->consumer_node]);
        }
        if (state->node_pressure_demand_units != NULL &&
            state->node_pressure_activation_units != NULL &&
            dependency->kind == LOOM_LOW_SCHEDULE_DEPENDENCY_SSA) {
          uint32_t consumer_demand =
              consumer->kind == LOOM_LOW_SCHEDULE_NODE_STRUCTURAL
                  ? state->node_pressure_demand_units[dependency->consumer_node]
                  : 0;
          if (consumer_demand == 0 &&
              dependency->operand_index < consumer->operand_count) {
            const loom_value_ordinal_t operand_ordinal =
                loom_low_schedule_node_const_operand_ordinals(
                    consumer)[dependency->operand_index];
            consumer_demand = state->values[operand_ordinal].unit_count;
          }
          if (consumer_demand == 0) {
            consumer_demand = 1;
          }
          pressure_demand_units = iree_math_saturating_add_u32(
              pressure_demand_units, consumer_demand);
          const uint32_t consumer_activation =
              consumer->kind == LOOM_LOW_SCHEDULE_NODE_STRUCTURAL
                  ? state->node_pressure_activation_units[dependency
                                                              ->consumer_node]
                  : consumer_demand;
          pressure_activation_units = loom_low_schedule_max_u32(
              pressure_activation_units, consumer_activation);
        }
      }
    }
    if (state->node_critical_path_cycles != NULL) {
      const uint16_t latency_cycles =
          node->schedule_class ? node->schedule_class->latency_cycles : 0;
      state->node_critical_path_cycles[node_index] =
          iree_math_saturating_add_u32(latency_cycles, successor_path_cycles);
    }
    if (state->node_pressure_demand_units != NULL) {
      state->node_pressure_demand_units[node_index] =
          pressure_demand_units != 0 ? pressure_demand_units : 1;
    }
    if (state->node_pressure_activation_units != NULL) {
      state->node_pressure_activation_units[node_index] =
          pressure_activation_units != 0 ? pressure_activation_units : 1;
    }
    if (pressure_state->first_actionable_pressure_cliff_indices != NULL ||
        state->pressure_resources != NULL) {
      loom_low_schedule_reverse_source_pressure_node(state, pressure_state,
                                                     node);
      const loom_low_schedule_block_t* block_record =
          &state->blocks[node->block_index];
      if (node_index == block_record->node_start) {
        loom_low_schedule_remove_source_pressure_block_arguments(
            state, pressure_state, node->block);
      }
    }
  }
  if (pressure_state->first_actionable_pressure_cliff_indices != NULL ||
      state->pressure_resources != NULL) {
    // The reverse source sweep shares the existing priority traversal and
    // leaves only its immutable per-class cliff floors behind.
    loom_low_schedule_reset_source_pressure_sweep(state, pressure_state);
  }
}

static bool loom_low_schedule_add_ready_nominee(uint32_t node_index,
                                                uint32_t* nominees,
                                                uint8_t* nominee_count) {
  if (node_index == LOOM_LOW_SCHEDULE_NODE_NONE) return false;
  for (uint8_t i = 0; i < *nominee_count; ++i) {
    if (nominees[i] == node_index) return false;
  }
  IREE_ASSERT_LT(*nominee_count, LOOM_LOW_SCHEDULE_READY_NOMINEE_CAPACITY);
  nominees[(*nominee_count)++] = node_index;
  return true;
}

static void loom_low_schedule_add_ready_view_nominees(
    const loom_low_schedule_ready_policy_t* ready_policy,
    loom_low_schedule_ready_view_t view, uint8_t search_count,
    uint8_t nominee_limit, uint32_t* nominees, uint8_t* nominee_count) {
  uint32_t view_nominees[LOOM_LOW_SCHEDULE_READY_COPY_CAPACITY];
  const uint8_t view_nominee_count = loom_low_schedule_ready_frontier_copy_best(
      &ready_policy->frontier, view, search_count, view_nominees);
  uint8_t added_nominee_count = 0;
  for (uint8_t i = 0; i < view_nominee_count; ++i) {
    if (loom_low_schedule_add_ready_nominee(view_nominees[i], nominees,
                                            nominee_count) &&
        ++added_nominee_count == nominee_limit) {
      break;
    }
  }
}

static uint8_t loom_low_schedule_collect_source_nominees(
    const loom_low_schedule_ready_policy_t* ready_policy,
    uint32_t* out_nominees) {
  uint8_t nominee_count = 0;
  loom_low_schedule_add_ready_view_nominees(
      ready_policy, LOOM_LOW_SCHEDULE_READY_VIEW_SOURCE,
      LOOM_LOW_SCHEDULE_READY_SOURCE_NOMINEE_COUNT,
      LOOM_LOW_SCHEDULE_READY_SOURCE_NOMINEE_COUNT, out_nominees,
      &nominee_count);
  return nominee_count;
}

static void loom_low_schedule_collect_recovery_nominees(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_ready_policy_t* ready_policy, bool pressure_relief,
    uint32_t* nominees, uint8_t* nominee_count) {
  if (pressure_relief) {
    loom_low_schedule_add_ready_view_nominees(
        ready_policy, LOOM_LOW_SCHEDULE_READY_VIEW_PRESSURE,
        LOOM_LOW_SCHEDULE_READY_VIEW_SEARCH_COUNT,
        LOOM_LOW_SCHEDULE_READY_VIEW_NOMINEE_COUNT, nominees, nominee_count);
    loom_low_schedule_add_ready_view_nominees(
        ready_policy, LOOM_LOW_SCHEDULE_READY_VIEW_STORAGE,
        LOOM_LOW_SCHEDULE_READY_VIEW_SEARCH_COUNT,
        LOOM_LOW_SCHEDULE_READY_VIEW_NOMINEE_COUNT, nominees, nominee_count);
  } else {
    loom_low_schedule_add_ready_view_nominees(
        ready_policy, LOOM_LOW_SCHEDULE_READY_VIEW_SCHEDULE,
        LOOM_LOW_SCHEDULE_READY_VIEW_SEARCH_COUNT,
        LOOM_LOW_SCHEDULE_READY_VIEW_NOMINEE_COUNT, nominees, nominee_count);
  }
  (void)loom_low_schedule_add_ready_nominee(
      loom_low_schedule_ready_pair_nominee(state, ready_policy), nominees,
      nominee_count);
}

static iree_status_t loom_low_schedule_run_list_scheduler(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count) {
  uint32_t* indegrees = NULL;
  loom_low_schedule_pressure_state_t pressure_state = {0};
  loom_low_schedule_ready_policy_t ready_policy = {0};
  uint32_t nominees[LOOM_LOW_SCHEDULE_READY_NOMINEE_CAPACITY];
  loom_low_schedule_candidate_score_t
      nominee_scores[LOOM_LOW_SCHEDULE_READY_NOMINEE_CAPACITY];
  if (state->dependencies.count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "low schedule dependency count exceeds uint32_t adjacency capacity");
  }
  IREE_ASSERT_LE(node_count, UINT32_MAX);
  if (node_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count, sizeof(*indegrees), (void**)&indegrees));
  }
  if (loom_low_schedule_uses_pressure_strategy(state)) {
    if (node_count != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, node_count,
          sizeof(*state->node_dependency_latency_cycles),
          (void**)&state->node_dependency_latency_cycles));
      memset(state->node_dependency_latency_cycles, 0,
             node_count * sizeof(*state->node_dependency_latency_cycles));
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, node_count, sizeof(*state->node_pressure_demand_units),
          (void**)&state->node_pressure_demand_units));
      memset(state->node_pressure_demand_units, 0,
             node_count * sizeof(*state->node_pressure_demand_units));
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, node_count,
          sizeof(*state->node_pressure_activation_units),
          (void**)&state->node_pressure_activation_units));
      memset(state->node_pressure_activation_units, 0,
             node_count * sizeof(*state->node_pressure_activation_units));
    }
  }
  IREE_RETURN_IF_ERROR(loom_low_schedule_allocate_pressure_state(
      state, node_count, &pressure_state));
  iree_arena_allocator_t dependency_scratch_arena;
  iree_arena_initialize(state->arena->block_pool, &dependency_scratch_arena);
  loom_low_schedule_dependency_detail_index_t dependency_details;
  iree_status_t dependency_index_status =
      loom_low_schedule_dependency_index_initialize(
          &state->dependencies, (uint32_t)node_count, &dependency_scratch_arena,
          state->arena, indegrees, &state->dependency_index,
          &dependency_details);
  if (iree_status_is_ok(dependency_index_status)) {
    loom_low_schedule_compute_node_priorities(
        state, node_count, &dependency_details, &pressure_state);
  }
  iree_arena_deinitialize(&dependency_scratch_arena);
  IREE_RETURN_IF_ERROR(dependency_index_status);
  if (loom_low_schedule_uses_pressure_strategy(state)) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_initialize_unlock_summaries(
        state, (uint32_t)node_count, &pressure_state));
  }
  IREE_RETURN_IF_ERROR(loom_low_schedule_ready_policy_initialize(
      state, (uint32_t)node_count, &ready_policy));

  for (iree_host_size_t block_index = 0; block_index < state->body->block_count;
       ++block_index) {
    loom_low_schedule_block_t* block_record = &state->blocks[block_index];
    const uint32_t block_node_end =
        block_record->node_start + block_record->node_count;
    block_record->scheduled_node_start = (uint32_t)state->scheduled_node_count;
    block_record->scheduled_node_count = 0;
    state->current_block_index = block_index;
    state->current_issue_cycle = 0;
    state->pending_pair_affinity_node = LOOM_LOW_SCHEDULE_NODE_NONE;
    if (state->resource_ready_issue_cycles != NULL) {
      memset(state->resource_ready_issue_cycles, 0,
             state->target.descriptor_set->resource_count *
                 sizeof(*state->resource_ready_issue_cycles));
    }
    if (loom_low_schedule_uses_pressure_strategy(state)) {
      loom_low_schedule_initialize_block_pressure(
          state, block_record, &pressure_state, &ready_policy);
      if (state->options->strategy ==
          LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
        loom_low_schedule_initialize_current_pressure_cliff_penalty(
            state, &pressure_state);
      }
    }
    uint32_t scheduled_in_block = 0;
    uint32_t range_start = block_record->node_start;
    uint32_t range_end = range_start < block_node_end
                             ? loom_low_schedule_source_range_end(
                                   state, range_start, block_node_end)
                             : range_start;
    loom_low_schedule_initialize_source_range_ready_frontier(
        state, &pressure_state, &ready_policy, indegrees, range_start,
        range_end);
    uint32_t scheduled_in_range = 0;
    while (scheduled_in_block < block_record->node_count) {
      state->current_issue_cycle = scheduled_in_block;
      uint32_t chosen_node = LOOM_LOW_SCHEDULE_NODE_NONE;
      loom_low_schedule_candidate_score_t chosen_score = {0};
      uint32_t rejected_node = LOOM_LOW_SCHEDULE_NODE_NONE;
      loom_low_schedule_candidate_score_t rejected_score = {0};
      const uint32_t ready_candidate_count =
          loom_low_schedule_ready_frontier_count(&ready_policy.frontier);
      uint32_t scored_candidate_count = 0;
      if (ready_candidate_count == 0) {
        return loom_low_schedule_handle_dependency_cycle(
            state, block_record, node_count, scheduled_in_block);
      }
      if (!loom_low_schedule_uses_pressure_strategy(state)) {
        (void)loom_low_schedule_ready_frontier_copy_best(
            &ready_policy.frontier, LOOM_LOW_SCHEDULE_READY_VIEW_SOURCE,
            /*capacity=*/1, &chosen_node);
      } else {
        uint8_t nominee_count =
            loom_low_schedule_collect_source_nominees(&ready_policy, nominees);
        IREE_ASSERT_NE(nominee_count, 0);
        for (uint8_t i = 0; i < nominee_count; ++i) {
          loom_low_schedule_score_candidate(state, &pressure_state,
                                            &ready_policy, indegrees,
                                            nominees[i], &nominee_scores[i]);
        }
        scored_candidate_count = nominee_count;
        const loom_low_schedule_candidate_compare_mode_t compare_mode =
            loom_low_schedule_choose_candidate_compare_mode(nominee_scores,
                                                            nominee_count);
        for (uint8_t i = 0; i < nominee_count; ++i) {
          const uint32_t node_index = nominees[i];
          if (chosen_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
              loom_low_schedule_candidate_score_less(
                  state, compare_mode, &nominee_scores[i], &chosen_score)) {
            if (chosen_node != LOOM_LOW_SCHEDULE_NODE_NONE) {
              rejected_node = chosen_node;
              rejected_score = chosen_score;
            }
            chosen_node = node_index;
            chosen_score = nominee_scores[i];
          } else if (rejected_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
                     loom_low_schedule_candidate_score_less(state, compare_mode,
                                                            &nominee_scores[i],
                                                            &rejected_score)) {
            rejected_node = node_index;
            rejected_score = nominee_scores[i];
          }
        }
        const bool recover_pressure =
            loom_low_schedule_candidate_threatens_pressure_cliff(&chosen_score);
        if (recover_pressure || chosen_score.effective_stall_cycles != 0) {
          const uint8_t source_nominee_count = nominee_count;
          loom_low_schedule_collect_recovery_nominees(
              state, &ready_policy, recover_pressure, nominees, &nominee_count);
          const loom_low_schedule_candidate_compare_mode_t recovery_mode =
              recover_pressure
                  ? LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF
                  : LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_DEFAULT;
          for (uint8_t i = source_nominee_count; i < nominee_count; ++i) {
            loom_low_schedule_score_candidate(state, &pressure_state,
                                              &ready_policy, indegrees,
                                              nominees[i], &nominee_scores[i]);
            ++scored_candidate_count;
            if (recover_pressure &&
                !iree_any_bit_set(
                    nominee_scores[i].flags,
                    LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_MAKES_PRESSURE_PROGRESS)) {
              continue;
            }
            if (loom_low_schedule_candidate_score_less(
                    state, recovery_mode, &nominee_scores[i], &chosen_score)) {
              rejected_node = chosen_node;
              rejected_score = chosen_score;
              chosen_node = nominees[i];
              chosen_score = nominee_scores[i];
            } else if (rejected_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
                       loom_low_schedule_candidate_score_less(
                           state, recovery_mode, &nominee_scores[i],
                           &rejected_score)) {
              rejected_node = nominees[i];
              rejected_score = nominee_scores[i];
            }
          }
        }
      }
      loom_low_schedule_ready_policy_remove(state, &pressure_state,
                                            &ready_policy, chosen_node);

      state->nodes[chosen_node].scheduled_ordinal = scheduled_in_block++;
      ++scheduled_in_range;
      state->current_issue_cycle = state->nodes[chosen_node].scheduled_ordinal;
      state->scheduled_node_indices[state->scheduled_node_count++] =
          chosen_node;
      ++block_record->scheduled_node_count;
      loom_low_schedule_note_pair_affinity_node_scheduled(state, chosen_node);
      if (loom_low_schedule_uses_pressure_strategy(state)) {
        loom_low_schedule_record_candidate_decision(
            state, block_index, state->nodes[chosen_node].scheduled_ordinal,
            ready_candidate_count, scored_candidate_count, chosen_node,
            &chosen_score, rejected_node, &rejected_score);
        loom_low_schedule_note_pressure_node_scheduled(
            state, &pressure_state, chosen_node, &chosen_score);
        loom_low_schedule_ready_policy_update_pressure_consumers(
            state, &pressure_state, &ready_policy, chosen_node);
      }
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_note_descriptor_rows_for_node(state, chosen_node));

      const uint32_t group_begin =
          loom_low_schedule_dependency_index_group_begin(
              &state->dependency_index, chosen_node);
      const uint32_t group_end = loom_low_schedule_dependency_index_group_end(
          &state->dependency_index, chosen_node);
      for (uint32_t group_index = group_begin; group_index < group_end;
           ++group_index) {
        const loom_low_schedule_dependency_group_t* group =
            loom_low_schedule_dependency_index_group_at(
                &state->dependency_index, group_index);
        const uint32_t consumer_node = group->consumer_node;
        IREE_ASSERT_LE(group->dependency_count, indegrees[consumer_node]);
        indegrees[consumer_node] -= group->dependency_count;
        if (loom_low_schedule_uses_pressure_strategy(state)) {
          const uint32_t remaining_producer =
              loom_low_schedule_dependency_frontier_consume_group(
                  &pressure_state.unlocks.frontier, chosen_node, group);
          if (remaining_producer != LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_NONE) {
            loom_low_schedule_publish_unlock_consumer(
                state, &pressure_state, remaining_producer, consumer_node);
          }
        }
        if (state->node_ready_issue_cycles != NULL &&
            loom_low_schedule_dependency_index_group_has_ssa(
                &state->dependency_index, group_index)) {
          const loom_low_schedule_class_t* schedule_class =
              state->nodes[chosen_node].schedule_class;
          const uint32_t ready_cycle = iree_math_saturating_add_u32(
              state->current_issue_cycle,
              schedule_class ? schedule_class->latency_cycles : 0);
          if (ready_cycle > state->node_ready_issue_cycles[consumer_node]) {
            state->node_ready_issue_cycles[consumer_node] = ready_cycle;
          }
        }
        if (indegrees[consumer_node] == 0 && consumer_node >= range_start &&
            consumer_node < range_end) {
          loom_low_schedule_ready_policy_insert(state, &pressure_state,
                                                &ready_policy, consumer_node);
        }
      }
      if (scheduled_in_range == range_end - range_start) {
        range_start = range_end;
        if (range_start < block_node_end) {
          range_end = loom_low_schedule_source_range_end(state, range_start,
                                                         block_node_end);
          scheduled_in_range = 0;
          loom_low_schedule_initialize_source_range_ready_frontier(
              state, &pressure_state, &ready_policy, indegrees, range_start,
              range_end);
        }
      }
    }
  }
  return iree_ok_status();
}

static uint32_t loom_low_schedule_find_node_for_preferred_op(
    const loom_low_schedule_build_state_t* state, const loom_op_t* op) {
  if (op == NULL || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    return LOOM_LOW_SCHEDULE_NODE_NONE;
  }
  uint16_t block_index = 0;
  if (!loom_region_try_block_index(state->body, op->parent_block,
                                   &block_index)) {
    return LOOM_LOW_SCHEDULE_NODE_NONE;
  }
  const loom_low_schedule_block_t* block_record = &state->blocks[block_index];
  uint32_t begin = block_record->node_start;
  uint32_t end = begin + block_record->node_count;
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2;
    const loom_op_t* candidate_op = state->nodes[middle].op;
    if (candidate_op->block_ordinal < op->block_ordinal) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  const uint32_t block_end =
      block_record->node_start + block_record->node_count;
  return begin < block_end && state->nodes[begin].op == op
             ? begin
             : LOOM_LOW_SCHEDULE_NODE_NONE;
}

static void loom_low_schedule_initialize_preferred_pair_index(
    loom_low_schedule_build_state_t* state) {
  if (state->preferred_pair_nodes == NULL) {
    return;
  }
  const loom_low_placement_pair_use_list_t preferred_pairs =
      state->options->preferred_pair_uses;
  for (iree_host_size_t i = 0; i < preferred_pairs.count; ++i) {
    const loom_low_placement_pair_use_t* pair = &preferred_pairs.values[i];
    const uint32_t first_node =
        loom_low_schedule_find_node_for_preferred_op(state, pair->first_op);
    const uint32_t second_node =
        loom_low_schedule_find_node_for_preferred_op(state, pair->second_op);
    if (first_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
        second_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
      continue;
    }
    IREE_ASSERT(state->preferred_pair_nodes[first_node].successor_node ==
                    LOOM_LOW_SCHEDULE_NODE_NONE ||
                state->preferred_pair_nodes[first_node].successor_node ==
                    second_node);
    IREE_ASSERT(state->preferred_pair_nodes[second_node].predecessor_node ==
                    LOOM_LOW_SCHEDULE_NODE_NONE ||
                state->preferred_pair_nodes[second_node].predecessor_node ==
                    first_node);
    state->preferred_pair_nodes[first_node].successor_node = second_node;
    state->preferred_pair_nodes[second_node].predecessor_node = first_node;
  }
}

iree_status_t loom_low_schedule_function(
    const loom_low_function_model_t* model,
    const loom_low_schedule_options_t* options, iree_arena_allocator_t* arena,
    loom_low_schedule_table_t* out_table) {
  if (!loom_low_schedule_strategy_is_valid(options->strategy)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown low schedule strategy %d",
                            (int)options->strategy);
  }
  *out_table = (loom_low_schedule_table_t){
      .module = model->module,
      .function_op = model->function_op,
      .target = model->target,
      .error_count = model->error_count,
  };
  loom_target_bundle_storage_rebind(&out_table->target.bundle_storage);
  if (model->error_count != 0) return iree_ok_status();
  IREE_ASSERT(loom_local_value_domain_is_acquired(&model->value_domain));

  loom_low_schedule_build_state_t state = {
      .module = model->module,
      .options = options,
      .pressure_cliffs = options->pressure_model != NULL
                             ? &options->pressure_model->register_class_cliffs
                             : NULL,
      .pressure_resources = options->pressure_model != NULL &&
                                    !loom_low_pressure_resource_table_is_empty(
                                        &options->pressure_model->resources)
                                ? &options->pressure_model->resources
                                : NULL,
      .arena = arena,
      .function_op = model->function_op,
      .body = model->body,
      .target = model->target,
      .value_domain = &model->value_domain,
      .cfg_graph = &model->cfg_graph,
  };
  loom_target_bundle_storage_rebind(&state.target.bundle_storage);
  loom_low_schedule_dependency_graph_initialize(&state.dependencies);
  IREE_ASSERT(state.body != NULL);
  IREE_RETURN_IF_ERROR(loom_low_schedule_verify_memory_access_table(
      options->memory_access_table, model->function_op, state.body));
  if (options->memory_access_table.function_op == model->function_op) {
    state.memory_access_records = options->memory_access_table.values;
    state.memory_access_record_count = options->memory_access_table.count;
  }
  state.register_type_resolver =
      loom_low_register_type_resolver_for_descriptor_set(
          state.target.descriptor_set);
  IREE_RETURN_IF_ERROR(
      loom_low_schedule_initialize_pair_affinity_index(&state));

  const iree_host_size_t node_count = model->node_count;
  const bool needs_liveness =
      iree_any_bit_set(options->flags,
                       LOOM_LOW_SCHEDULE_FLAG_RETAIN_LIVENESS) ||
      iree_any_bit_set(options->diagnostic_flags,
                       LOOM_LOW_SCHEDULE_DIAGNOSTIC_PRESSURE_PEAKS);
  loom_liveness_analysis_t liveness = {0};
  iree_status_t status =
      loom_low_schedule_initialize_storage(&state, node_count);
  if (iree_status_is_ok(status)) {
    status = loom_low_schedule_fill_nodes(&state);
  }
  if (iree_status_is_ok(status)) {
    loom_low_schedule_initialize_preferred_pair_index(&state);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_schedule_initialize_pair_setup_index(&state);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_low_schedule_initialize_storage_read_tables(&state, node_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_schedule_initialize_descriptor_tables(&state, node_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_schedule_build_dependencies(&state);
  }
  if (iree_status_is_ok(status) && needs_liveness) {
    status = loom_liveness_analyze_local_value_domain_with_cfg_graph(
        &model->value_domain, &model->cfg_graph, loom_liveness_order_empty(),
        arena, &liveness);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_schedule_run_list_scheduler(&state, node_count);
  }
  if (iree_status_is_ok(status) && state.error_count == 0) {
    loom_low_schedule_compact_model_summaries(&state);
    loom_low_schedule_compact_resource_summaries(&state);
  }
  if (iree_status_is_ok(status) && state.error_count == 0 &&
      iree_any_bit_set(options->diagnostic_flags,
                       LOOM_LOW_SCHEDULE_DIAGNOSTIC_PRESSURE_PEAKS)) {
    status = loom_low_schedule_emit_pressure_diagnostics(&state, &liveness);
  }
  if (iree_status_is_ok(status) && state.error_count == 0 &&
      iree_any_bit_set(options->diagnostic_flags,
                       LOOM_LOW_SCHEDULE_DIAGNOSTIC_CANDIDATE_DECISIONS)) {
    status = loom_low_schedule_emit_candidate_decision_diagnostics(&state);
  }
  if (iree_status_is_ok(status) && state.error_count == 0 &&
      iree_any_bit_set(options->diagnostic_flags,
                       LOOM_LOW_SCHEDULE_DIAGNOSTIC_MODEL_QUALITY)) {
    status = loom_low_schedule_emit_model_diagnostics(&state);
  }
  if (iree_status_is_ok(status) && state.error_count == 0 &&
      iree_any_bit_set(options->diagnostic_flags,
                       LOOM_LOW_SCHEDULE_DIAGNOSTIC_RESOURCE_BOTTLENECKS)) {
    status = loom_low_schedule_emit_resource_diagnostics(&state);
  }
  if (iree_status_is_ok(status) && state.error_count == 0 &&
      iree_any_bit_set(options->diagnostic_flags,
                       LOOM_LOW_SCHEDULE_DIAGNOSTIC_HAZARD_GAPS)) {
    status = loom_low_schedule_emit_hazard_gap_diagnostics(&state);
  }

  if (iree_status_is_ok(status)) {
    *out_table = (loom_low_schedule_table_t){
        .module = model->module,
        .function_op = model->function_op,
        .target = state.target,
        .memory_access_table = options->memory_access_table,
        .value_ids = model->value_domain.value_ids,
        .value_count = model->value_domain.value_count,
        .liveness = liveness,
        .blocks = state.blocks,
        .block_count = state.body->block_count,
        .cfg_graph = model->cfg_graph,
        .nodes = state.nodes,
        .node_count = node_count,
        .dependency_group_count = state.dependency_index.group_count,
        .unlock_summary_publication_count =
            state.unlock_summary_publication_count,
        .scheduled_node_indices = state.scheduled_node_indices,
        .scheduled_node_count = state.scheduled_node_count,
        .placement_pair_uses =
            {
                .values = state.placement_pair_uses,
                .count = state.placement_pair_use_count,
                .placement_recipes = options->pair_affinities.placement_recipes,
                .placement_recipe_count =
                    options->pair_affinities.placement_recipe_count,
            },
        .error_count = state.error_count,
        .failure = state.failure,
        .pressure_steps = state.pressure_steps,
        .pressure_step_count = state.pressure_step_count,
        .candidate_decisions = state.candidate_decisions,
        .candidate_decision_count = state.candidate_decision_count,
        .resource_use_count = state.resource_use_count,
        .effect_uses = state.effect_uses,
        .effect_use_count = state.effect_use_count,
        .hazard_uses = state.hazard_uses,
        .hazard_use_count = state.hazard_use_count,
        .hazard_gaps = state.hazard_gaps,
        .hazard_gap_count = state.hazard_gap_count,
        .model_summaries = state.model_summaries,
        .model_summary_count = state.model_summary_count,
        .resource_summaries = state.resource_summaries,
        .resource_summary_count = state.resource_summary_count,
    };
    loom_low_schedule_dependency_graph_move(&state.dependencies,
                                            &out_table->dependencies);
    loom_target_bundle_storage_rebind(&out_table->target.bundle_storage);
  }
  return status;
}
