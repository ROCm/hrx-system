// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <inttypes.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/schedule/context.h"
#include "loom/codegen/low/schedule/descriptor_rows.h"
#include "loom/codegen/low/schedule/diagnostics.h"
#include "loom/codegen/low/schedule/graph.h"
#include "loom/codegen/low/storage_relation.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/registers.h"

#define LOOM_LOW_SCHEDULE_READY_WINDOW 16
#define LOOM_LOW_SCHEDULE_READY_STALL_SCAN_WINDOW 64

enum loom_low_schedule_state_access_bits_e {
  LOOM_LOW_SCHEDULE_STATE_ACCESS_READ = 1u << 0,
  LOOM_LOW_SCHEDULE_STATE_ACCESS_WRITE = 1u << 1,
};

typedef struct loom_low_schedule_pressure_state_t {
  // Current live register units by descriptor register-class ID.
  uint64_t* current_live_units_by_reg_class;
  // Register-class IDs seen with live pressure in the current block.
  uint16_t* block_reg_class_ids;
  // True when a register class is present in |block_reg_class_ids|.
  uint8_t* block_reg_class_touched_flags;
  // Value ordinals with per-block pressure state to reset before reuse.
  loom_value_ordinal_t* block_value_ordinals;
  // Candidate operand multiplicity by local value ordinal.
  uint16_t* candidate_operand_use_counts;
  // Value ordinals touched in |candidate_operand_use_counts|.
  loom_value_ordinal_t* candidate_operand_ordinals;
  // Scratch live-unit delta by descriptor register-class ID for one candidate.
  int64_t* candidate_delta_units_by_reg_class;
  // True when a register class has a nonzero or previously nonzero candidate
  // delta that must be reset after scoring.
  uint8_t* candidate_delta_touched_flags;
  // Register-class IDs touched in |candidate_delta_units_by_reg_class|.
  uint16_t* candidate_delta_touched_reg_class_ids;
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

typedef struct loom_low_schedule_candidate_score_t {
  // Aggregate live register units after scheduling the candidate.
  uint64_t projected_live_units;
  // Live register units whose last use is the candidate.
  uint64_t killed_live_units;
  // Live register values whose last use is the candidate.
  uint32_t killed_live_value_count;
  // Register result units made live by the candidate.
  uint64_t produced_live_units;
  // Register result values made live by the candidate.
  uint32_t produced_live_value_count;
  // Maximum latency of same-block SSA producers consumed by the candidate.
  uint16_t dependency_latency_cycles;
  // Descriptor latency for the candidate itself.
  uint16_t latency_cycles;
  // Longest same-block latency path starting at the candidate.
  uint32_t critical_path_cycles;
  // Cycles until all same-block SSA producers are ready.
  uint32_t data_ready_stall_cycles;
  // Cycles until descriptor resources can accept this candidate.
  uint32_t resource_stall_cycles;
  // Cycles until target hazard distance rows are satisfied.
  uint32_t hazard_stall_cycles;
  // Maximum stall across data, resources, and target hazards.
  uint32_t effective_stall_cycles;
  // Resource causing resource_stall_cycles, or LOOM_LOW_RESOURCE_NONE.
  uint16_t bottleneck_resource_id;
  // Target pressure-cliff penalty from projected live units.
  uint32_t pressure_cliff_penalty;
  // Target pair-affinity reward. Larger scores are better.
  uint16_t pair_affinity_score;
  // Number of storage-relation rows owned by the candidate.
  uint16_t storage_relation_count;
  // Register class for the closest crossed or future pressure cliff.
  uint16_t pressure_cliff_reg_class_id;
  // Crossed pressure cliff, or LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE.
  uint32_t pressure_cliff_units;
  // Live units remaining before the next pressure cliff, or
  // LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE.
  uint32_t units_until_pressure_cliff;
  // Source-order tie breaker.
  uint32_t source_ordinal;
} loom_low_schedule_candidate_score_t;

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

typedef struct loom_low_schedule_ready_heap_t {
  // Ready node indices in a min-heap ordered by source ordinal.
  uint32_t* node_indices;
  // Number of ready nodes currently in |node_indices|.
  iree_host_size_t count;
} loom_low_schedule_ready_heap_t;

static uint32_t loom_low_schedule_positive_delta_u32(uint32_t lhs,
                                                     uint32_t rhs) {
  return lhs > rhs ? lhs - rhs : 0;
}

static uint32_t loom_low_schedule_max_u32(uint32_t lhs, uint32_t rhs) {
  return lhs > rhs ? lhs : rhs;
}

static void loom_low_schedule_count_nodes(const loom_region_t* body,
                                          iree_host_size_t* out_node_count) {
  iree_host_size_t node_count = 0;
  const loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) { node_count += block->op_count; }
  *out_node_count = node_count;
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
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count, sizeof(*state->state_chain_read_heads),
        (void**)&state->state_chain_read_heads));
    memset(state->state_chain_read_heads, 0xFF,
           node_count * sizeof(*state->state_chain_read_heads));
    if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE ||
        state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING ||
        state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
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
  bool has_tied_results = false;
  for (iree_host_size_t node_index = 0; node_index < node_count; ++node_index) {
    if (state->nodes[node_index].op->tied_result_count != 0) {
      has_tied_results = true;
      break;
    }
  }
  if (!has_tied_results || state->value_domain->value_count == 0) {
    return iree_ok_status();
  }
  const loom_value_ordinal_t value_count = state->value_domain->value_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, value_count, sizeof(*state->storage_reads.heads),
      (void**)&state->storage_reads.heads));
  memset(state->storage_reads.heads, 0xFF,
         value_count * sizeof(*state->storage_reads.heads));
  return iree_arena_allocate_array(
      state->arena, value_count, sizeof(*state->storage_reads.touched_ordinals),
      (void**)&state->storage_reads.touched_ordinals);
}

static iree_status_t loom_low_schedule_initialize_pressure_cliff_ranges(
    loom_low_schedule_build_state_t* state) {
  if (loom_low_schedule_pressure_cliff_list_is_empty(
          state->options->pressure_cliffs)) {
    return iree_ok_status();
  }
  const loom_low_descriptor_set_t* descriptor_set =
      state->target.descriptor_set;
  if (descriptor_set->reg_class_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low schedule pressure cliffs require descriptor register classes");
  }
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->arena, descriptor_set->reg_class_count,
                                sizeof(*state->pressure_cliff_ranges),
                                (void**)&state->pressure_cliff_ranges));
  memset(
      state->pressure_cliff_ranges, 0,
      descriptor_set->reg_class_count * sizeof(*state->pressure_cliff_ranges));

  uint16_t previous_reg_class_id = LOOM_LOW_REG_CLASS_NONE;
  uint32_t previous_cliff_units = 0;
  for (iree_host_size_t i = 0; i < state->options->pressure_cliffs.count; ++i) {
    if (i > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low schedule pressure cliff index exceeds uint32_t");
    }
    const loom_low_schedule_pressure_cliff_t* cliff =
        &state->options->pressure_cliffs.values[i];
    if (cliff->descriptor_reg_class_id >= descriptor_set->reg_class_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low schedule pressure cliff references invalid register class "
          "%" PRIu16,
          cliff->descriptor_reg_class_id);
    }
    if (cliff->cliff_units == 0) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "low schedule pressure cliff cannot be zero");
    }
    if (cliff->tier_after >= cliff->tier_before) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low schedule pressure cliff must lower the target tier");
    }
    if (i != 0 && (cliff->descriptor_reg_class_id < previous_reg_class_id ||
                   (cliff->descriptor_reg_class_id == previous_reg_class_id &&
                    cliff->cliff_units <= previous_cliff_units))) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low schedule pressure cliffs must be sorted by register class and "
          "cliff units");
    }
    loom_low_schedule_pressure_cliff_range_t* range =
        &state->pressure_cliff_ranges[cliff->descriptor_reg_class_id];
    if (range->count == 0) {
      range->start = (uint32_t)i;
    }
    ++range->count;
    previous_reg_class_id = cliff->descriptor_reg_class_id;
    previous_cliff_units = cliff->cliff_units;
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
  iree_host_size_t resource_use_capacity = 0;
  iree_host_size_t effect_use_capacity = 0;
  iree_host_size_t hazard_use_capacity = 0;
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
        state->arena, reg_class_count, sizeof(*state->state_read_heads),
        (void**)&state->state_read_heads));
  }
  IREE_RETURN_IF_ERROR(
      loom_low_schedule_initialize_pressure_cliff_ranges(state));
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
  }
  IREE_RETURN_IF_ERROR(loom_low_schedule_verify_structural_state_reads(state));
  for (iree_host_size_t node_index = 0; node_index < node_count; ++node_index) {
    if (!iree_host_size_checked_add(resource_use_capacity,
                                    state->nodes[node_index].issue_use_count,
                                    &resource_use_capacity)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low schedule resource-use capacity exceeds host size");
    }
    if (!iree_host_size_checked_add(effect_use_capacity,
                                    state->nodes[node_index].effect_count,
                                    &effect_use_capacity)) {
      return iree_make_status(
          IREE_STATUS_OUT_OF_RANGE,
          "low schedule effect-use capacity exceeds host size");
    }
    if (!iree_host_size_checked_add(hazard_use_capacity,
                                    state->nodes[node_index].hazard_count,
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
  if (resource_use_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, resource_use_capacity, sizeof(*state->resource_uses),
        (void**)&state->resource_uses));
    state->resource_use_capacity = resource_use_capacity;
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

static bool loom_low_schedule_uses_pressure_strategy(
    const loom_low_schedule_build_state_t* state) {
  return state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_PRESSURE ||
         state->options->strategy ==
             LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING ||
         state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL;
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

static iree_host_size_t loom_low_schedule_count_storage_relations(
    const loom_low_schedule_build_state_t* state) {
  iree_host_size_t relation_count = 0;
  for (uint32_t block_index = 0; block_index < state->body->block_count;
       ++block_index) {
    const loom_low_schedule_block_t* block_record = &state->blocks[block_index];
    const uint32_t block_node_end =
        block_record->node_start + block_record->node_count;
    for (uint32_t node_index = block_record->node_start;
         node_index < block_node_end; ++node_index) {
      relation_count += loom_low_storage_relation_count(
          state->module, state->nodes[node_index].op);
    }
  }
  return relation_count;
}

static iree_status_t loom_low_schedule_allocate_pressure_state(
    loom_low_schedule_build_state_t* state,
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
    memset(out_pressure_state->candidate_operand_use_counts, 0,
           value_count *
               sizeof(*out_pressure_state->candidate_operand_use_counts));
    const iree_host_size_t alias_capacity =
        loom_low_schedule_count_storage_relations(state);
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

static void loom_low_schedule_reset_candidate_operand_uses(
    loom_low_schedule_pressure_state_t* pressure_state) {
  for (iree_host_size_t i = 0; i < pressure_state->candidate_operand_count;
       ++i) {
    const loom_value_ordinal_t value_ordinal =
        pressure_state->candidate_operand_ordinals[i];
    pressure_state->candidate_operand_use_counts[value_ordinal] = 0;
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

static iree_status_t loom_low_schedule_note_block_pressure_use(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_value_ordinal_t value_ordinal) {
  loom_low_schedule_value_record_t* value = &state->values[value_ordinal];
  if (value->remaining_use_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "low schedule pressure use count exceeds uint32_t");
  }
  if (value->remaining_use_count == 0) {
    pressure_state->block_value_ordinals[pressure_state->block_value_count++] =
        value_ordinal;
  }
  ++value->remaining_use_count;
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_initialize_block_pressure(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_block_t* block_record,
    loom_low_schedule_pressure_state_t* pressure_state) {
  pressure_state->current_live_units = 0;
  for (iree_host_size_t i = 0; i < pressure_state->block_reg_class_count; ++i) {
    const uint16_t reg_class_id = pressure_state->block_reg_class_ids[i];
    pressure_state->block_reg_class_touched_flags[reg_class_id] = 0;
  }
  pressure_state->block_reg_class_count = 0;
  for (iree_host_size_t i = 0; i < pressure_state->block_value_count; ++i) {
    loom_value_ordinal_t ordinal = pressure_state->block_value_ordinals[i];
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
      IREE_RETURN_IF_ERROR(loom_low_schedule_note_block_pressure_use(
          state, pressure_state, operand_ordinals[operand_index]));
    }
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
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_score_candidate_resources(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node,
    loom_low_schedule_candidate_score_t* score) {
  score->resource_stall_cycles = 0;
  score->bottleneck_resource_id = LOOM_LOW_RESOURCE_NONE;
  if (state->options->strategy != LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL ||
      state->resource_ready_issue_cycles == NULL ||
      node->schedule_class_id == LOOM_LOW_SCHEDULE_CLASS_NONE) {
    return iree_ok_status();
  }
  IREE_ASSERT(node->schedule_class_id <
              state->target.descriptor_set->schedule_class_count);

  const loom_low_schedule_class_t* schedule_class =
      &state->target.descriptor_set->schedule_classes[node->schedule_class_id];
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
  return iree_ok_status();
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

static iree_status_t loom_low_schedule_score_candidate_hazards(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node,
    loom_low_schedule_candidate_score_t* score) {
  score->hazard_stall_cycles = 0;
  if (state->options->strategy != LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL ||
      node->schedule_class_id == LOOM_LOW_SCHEDULE_CLASS_NONE) {
    return iree_ok_status();
  }
  IREE_ASSERT(node->schedule_class_id <
              state->target.descriptor_set->schedule_class_count);

  const loom_low_schedule_class_t* schedule_class =
      &state->target.descriptor_set->schedule_classes[node->schedule_class_id];
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
  return iree_ok_status();
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
}

static void loom_low_schedule_note_candidate_pressure_delta(
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

static uint32_t loom_low_schedule_candidate_relation_alias_units_before(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_node_t* node, uint16_t relation_end,
    loom_value_ordinal_t result_ordinal, loom_value_ordinal_t source_ordinal) {
  uint32_t claimed_units =
      pressure_state->alias_source_unit_counts
          ? pressure_state->alias_source_unit_counts[source_ordinal]
          : 0;
  const loom_low_schedule_value_record_t* source =
      &state->values[source_ordinal];
  if (claimed_units >= source->unit_count) {
    return source->unit_count;
  }
  for (uint16_t i = 0; i < relation_end; ++i) {
    loom_low_storage_relation_t relation = {0};
    loom_low_storage_relation_get(state->module, node->op, i, &relation);
    const loom_value_ordinal_t destination_ordinal =
        loom_local_value_domain_ordinal(state->value_domain,
                                        relation.destination_value_id);
    if (destination_ordinal != result_ordinal) {
      continue;
    }
    const loom_value_ordinal_t relation_source_ordinal =
        loom_local_value_domain_ordinal(state->value_domain,
                                        relation.source_value_id);
    if (relation_source_ordinal != source_ordinal) {
      continue;
    }
    if (!loom_low_schedule_storage_relation_can_alias_pressure(
            state, &relation, result_ordinal, source_ordinal)) {
      continue;
    }
    if (!loom_low_schedule_relation_source_is_available_after_candidate(
            state, pressure_state, source_ordinal)) {
      continue;
    }
    const uint32_t available_units = source->unit_count - claimed_units;
    claimed_units += iree_min(relation.unit_count, available_units);
    if (claimed_units == source->unit_count) {
      break;
    }
  }
  return claimed_units;
}

static uint32_t loom_low_schedule_result_alias_units_after_candidate(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_pressure_state_t* pressure_state,
    const loom_low_schedule_node_t* node, loom_value_ordinal_t result_ordinal) {
  uint32_t alias_units = 0;
  const uint16_t relation_count =
      loom_low_storage_relation_count(state->module, node->op);
  for (uint16_t i = 0; i < relation_count; ++i) {
    loom_low_storage_relation_t relation = {0};
    loom_low_storage_relation_get(state->module, node->op, i, &relation);
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
        loom_low_schedule_candidate_relation_alias_units_before(
            state, pressure_state, node, i, result_ordinal, source_ordinal);
    if (claimed_units >= source->unit_count) {
      continue;
    }
    alias_units +=
        iree_min(relation.unit_count, source->unit_count - claimed_units);
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

static iree_status_t loom_low_schedule_project_reg_class_live_units(
    uint64_t current_live_units, int64_t delta_units,
    uint64_t* out_projected_live_units) {
  if (delta_units < 0) {
    const uint64_t removed_units = (uint64_t)(-delta_units);
    if (removed_units > current_live_units) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low schedule pressure accounting underflow for register class");
    }
    *out_projected_live_units = current_live_units - removed_units;
    return iree_ok_status();
  }
  const uint64_t added_units = (uint64_t)delta_units;
  if (current_live_units > UINT64_MAX - added_units) {
    *out_projected_live_units = UINT64_MAX;
    return iree_ok_status();
  }
  *out_projected_live_units = current_live_units + added_units;
  return iree_ok_status();
}

static iree_status_t
loom_low_schedule_score_candidate_pressure_cliffs_for_class(
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
    return iree_ok_status();
  }
  uint64_t projected_live_units = 0;
  IREE_RETURN_IF_ERROR(loom_low_schedule_project_reg_class_live_units(
      current_live_units, delta_units, &projected_live_units));
  const loom_low_schedule_pressure_cliff_range_t range =
      state->pressure_cliff_ranges[reg_class_id];
  for (uint32_t cliff_index = range.start;
       cliff_index < range.start + range.count; ++cliff_index) {
    const loom_low_schedule_pressure_cliff_t* cliff =
        &state->options->pressure_cliffs.values[cliff_index];
    // Score the projected pressure state, not only a below-to-above
    // transition. Once a schedule is already over a cliff, remaining over it
    // is still pressure debt that should lose to candidates that pay it down.
    if (projected_live_units >= cliff->cliff_units) {
      score->pressure_cliff_penalty += cliff->tier_before - cliff->tier_after;
      if (score->pressure_cliff_units ==
          LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE) {
        score->pressure_cliff_reg_class_id = reg_class_id;
        score->pressure_cliff_units = cliff->cliff_units;
      }
      continue;
    }
    const uint64_t units_until_cliff =
        cliff->cliff_units - projected_live_units;
    if (units_until_cliff < score->units_until_pressure_cliff) {
      score->pressure_cliff_reg_class_id = reg_class_id;
      score->units_until_pressure_cliff = (uint32_t)units_until_cliff;
    }
    break;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_schedule_score_candidate_pressure_cliffs(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_candidate_score_t* score) {
  if (state->pressure_cliff_ranges == NULL ||
      pressure_state->current_live_units_by_reg_class == NULL) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < pressure_state->block_reg_class_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_score_candidate_pressure_cliffs_for_class(
            state, pressure_state, score,
            pressure_state->block_reg_class_ids[i]));
  }
  for (iree_host_size_t i = 0;
       i < pressure_state->candidate_delta_touched_count; ++i) {
    const uint16_t reg_class_id =
        pressure_state->candidate_delta_touched_reg_class_ids[i];
    if (pressure_state->block_reg_class_touched_flags[reg_class_id]) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_score_candidate_pressure_cliffs_for_class(
            state, pressure_state, score, reg_class_id));
  }
  return iree_ok_status();
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

static uint16_t loom_low_schedule_pair_affinity_priority(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* first,
    const loom_low_schedule_node_t* second) {
  if (first == NULL || second == NULL || first->descriptor == NULL ||
      second->descriptor == NULL) {
    return 0;
  }
  const loom_low_schedule_pair_affinity_list_t affinities =
      state->options->pair_affinities;
  for (iree_host_size_t i = 0; i < affinities.count; ++i) {
    const loom_low_schedule_pair_affinity_t* affinity = &affinities.values[i];
    if (affinity->priority == 0) {
      continue;
    }
    if (affinity->first_descriptor == first->descriptor &&
        affinity->second_descriptor == second->descriptor) {
      return affinity->priority;
    }
  }
  return 0;
}

static bool loom_low_schedule_node_can_start_pair_affinity(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_node_t* node) {
  if (node == NULL || node->descriptor == NULL) {
    return false;
  }
  const loom_low_schedule_pair_affinity_list_t affinities =
      state->options->pair_affinities;
  for (iree_host_size_t i = 0; i < affinities.count; ++i) {
    const loom_low_schedule_pair_affinity_t* affinity = &affinities.values[i];
    if (affinity->priority != 0 &&
        affinity->first_descriptor == node->descriptor) {
      return true;
    }
  }
  return false;
}

static uint16_t loom_low_schedule_scale_direct_pair_affinity(
    uint16_t priority) {
  return priority > UINT16_MAX / 2u ? UINT16_MAX : (uint16_t)(priority * 2u);
}

static uint16_t loom_low_schedule_score_candidate_pair_affinity(
    const loom_low_schedule_build_state_t* state, uint32_t node_index) {
  if (loom_low_schedule_pair_affinity_list_is_empty(
          state->options->pair_affinities) ||
      state->pending_pair_affinity_node == LOOM_LOW_SCHEDULE_NODE_NONE) {
    return 0;
  }
  const loom_low_schedule_node_t* anchor =
      &state->nodes[state->pending_pair_affinity_node];
  const loom_low_schedule_node_t* candidate = &state->nodes[node_index];
  uint16_t priority =
      loom_low_schedule_pair_affinity_priority(state, anchor, candidate);
  if (priority != 0 &&
      !loom_low_schedule_node_result_used_by(anchor, candidate)) {
    return loom_low_schedule_scale_direct_pair_affinity(priority);
  }

  if (!loom_low_schedule_node_is_pair_transparent(candidate) ||
      state->outgoing_heads == NULL || state->outgoing_next_indices == NULL) {
    return 0;
  }
  for (uint32_t dependency_index = state->outgoing_heads[node_index];
       dependency_index != LOOM_LOW_SCHEDULE_NODE_NONE;
       dependency_index = state->outgoing_next_indices[dependency_index]) {
    const loom_low_schedule_dependency_t* dependency =
        &state->dependencies[dependency_index];
    if (dependency->kind != LOOM_LOW_SCHEDULE_DEPENDENCY_SSA ||
        dependency->producer_node != node_index) {
      continue;
    }
    const loom_low_schedule_node_t* consumer =
        &state->nodes[dependency->consumer_node];
    priority =
        loom_low_schedule_pair_affinity_priority(state, anchor, consumer);
    if (priority != 0 &&
        !loom_low_schedule_node_result_used_by(anchor, consumer)) {
      return priority;
    }
  }
  return 0;
}

static iree_status_t loom_low_schedule_score_candidate(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index,
    loom_low_schedule_candidate_score_t* out_score) {
  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  loom_low_schedule_reset_candidate_pressure_deltas(pressure_state);
  uint64_t killed_live_units = 0;
  uint32_t killed_live_value_count = 0;
  uint64_t produced_live_units = 0;
  uint32_t produced_live_value_count = 0;
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
        pressure_state, value->register_class_id, -(int64_t)unit_count);
  }
  const loom_value_ordinal_t* result_ordinals =
      loom_low_schedule_node_const_result_ordinals(node);
  for (uint16_t result_index = 0; result_index < node->result_count;
       ++result_index) {
    const loom_low_schedule_value_record_t* value =
        &state->values[result_ordinals[result_index]];
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
        pressure_state, value->register_class_id, (int64_t)unit_count);
  }
  loom_low_schedule_reset_candidate_operand_uses(pressure_state);

  if (killed_live_units > pressure_state->current_live_units) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low schedule pressure candidate killed more units than are live");
  }
  uint64_t projected_live_units =
      pressure_state->current_live_units - killed_live_units;
  projected_live_units += produced_live_units;
  uint16_t dependency_latency_cycles = 0;
  for (uint16_t operand_index = 0; operand_index < node->operand_count;
       ++operand_index) {
    const uint32_t producer_node =
        state->values[operand_ordinals[operand_index]].producer_node;
    if (producer_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
        state->nodes[producer_node].block != node->block) {
      continue;
    }
    const uint16_t producer_latency =
        state->nodes[producer_node].latency_cycles;
    if (producer_latency > dependency_latency_cycles) {
      dependency_latency_cycles = producer_latency;
    }
  }
  uint32_t data_ready_stall_cycles = 0;
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL &&
      state->node_ready_issue_cycles != NULL) {
    data_ready_stall_cycles = loom_low_schedule_positive_delta_u32(
        state->node_ready_issue_cycles[node_index], state->current_issue_cycle);
  }
  *out_score = (loom_low_schedule_candidate_score_t){
      .projected_live_units = projected_live_units,
      .killed_live_units = killed_live_units,
      .killed_live_value_count = killed_live_value_count,
      .produced_live_units = produced_live_units,
      .produced_live_value_count = produced_live_value_count,
      .dependency_latency_cycles = dependency_latency_cycles,
      .latency_cycles = node->latency_cycles,
      .critical_path_cycles = state->node_critical_path_cycles != NULL
                                  ? state->node_critical_path_cycles[node_index]
                                  : node->latency_cycles,
      .data_ready_stall_cycles = data_ready_stall_cycles,
      .pair_affinity_score =
          loom_low_schedule_score_candidate_pair_affinity(state, node_index),
      .storage_relation_count = storage_relation_count,
      .bottleneck_resource_id = LOOM_LOW_RESOURCE_NONE,
      .pressure_cliff_reg_class_id = LOOM_LOW_REG_CLASS_NONE,
      .pressure_cliff_units = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
      .units_until_pressure_cliff = LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE,
      .source_ordinal = node->source_ordinal,
  };
  IREE_RETURN_IF_ERROR(loom_low_schedule_score_candidate_pressure_cliffs(
      state, pressure_state, out_score));
  IREE_RETURN_IF_ERROR(
      loom_low_schedule_score_candidate_resources(state, node, out_score));
  IREE_RETURN_IF_ERROR(
      loom_low_schedule_score_candidate_hazards(state, node, out_score));
  out_score->effective_stall_cycles = loom_low_schedule_max_u32(
      out_score->data_ready_stall_cycles,
      loom_low_schedule_max_u32(out_score->resource_stall_cycles,
                                out_score->hazard_stall_cycles));
  return iree_ok_status();
}

static int loom_low_schedule_compare_candidate_pressure(
    loom_low_schedule_candidate_score_t lhs,
    loom_low_schedule_candidate_score_t rhs) {
  if (lhs.projected_live_units != rhs.projected_live_units) {
    return lhs.projected_live_units < rhs.projected_live_units ? -1 : 1;
  }
  if (lhs.killed_live_units != rhs.killed_live_units) {
    return lhs.killed_live_units > rhs.killed_live_units ? -1 : 1;
  }
  if (lhs.produced_live_units != rhs.produced_live_units) {
    return lhs.produced_live_units < rhs.produced_live_units ? -1 : 1;
  }
  if (lhs.units_until_pressure_cliff != rhs.units_until_pressure_cliff) {
    if (lhs.units_until_pressure_cliff > rhs.units_until_pressure_cliff) {
      return -1;
    }
    return 1;
  }
  return 0;
}

static int loom_low_schedule_compare_candidate_live_values(
    loom_low_schedule_candidate_score_t lhs,
    loom_low_schedule_candidate_score_t rhs) {
  const int64_t lhs_delta = (int64_t)lhs.produced_live_value_count -
                            (int64_t)lhs.killed_live_value_count;
  const int64_t rhs_delta = (int64_t)rhs.produced_live_value_count -
                            (int64_t)rhs.killed_live_value_count;
  if (lhs_delta != rhs_delta) {
    return lhs_delta < rhs_delta ? -1 : 1;
  }
  if (lhs.killed_live_value_count != rhs.killed_live_value_count) {
    return lhs.killed_live_value_count > rhs.killed_live_value_count ? -1 : 1;
  }
  if (lhs.produced_live_value_count != rhs.produced_live_value_count) {
    return lhs.produced_live_value_count < rhs.produced_live_value_count ? -1
                                                                         : 1;
  }
  return 0;
}

static bool loom_low_schedule_candidate_shortens_producer_live_range(
    loom_low_schedule_candidate_score_t score) {
  return score.dependency_latency_cycles != 0 &&
         score.killed_live_units > score.produced_live_units;
}

static bool loom_low_schedule_candidate_compacts_live_values(
    loom_low_schedule_candidate_score_t score) {
  return score.storage_relation_count != 0 &&
         score.killed_live_value_count > score.produced_live_value_count;
}

static bool loom_low_schedule_candidate_has_better_pair_affinity(
    loom_low_schedule_candidate_score_t lhs,
    loom_low_schedule_candidate_score_t rhs) {
  return lhs.pair_affinity_score > rhs.pair_affinity_score;
}

static bool loom_low_schedule_candidate_threatens_pressure_cliff(
    loom_low_schedule_candidate_score_t score) {
  if (score.pressure_cliff_penalty != 0) {
    return true;
  }
  return score.units_until_pressure_cliff !=
             LOOM_LOW_SCHEDULE_PRESSURE_CLIFF_NONE &&
         score.units_until_pressure_cliff <= score.produced_live_units;
}

static bool loom_low_schedule_candidate_score_less(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_candidate_score_t lhs,
    loom_low_schedule_candidate_score_t rhs) {
  const int pressure_order =
      loom_low_schedule_compare_candidate_pressure(lhs, rhs);
  const int live_value_order =
      loom_low_schedule_compare_candidate_live_values(lhs, rhs);
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
    if (lhs.pressure_cliff_penalty != rhs.pressure_cliff_penalty) {
      return lhs.pressure_cliff_penalty < rhs.pressure_cliff_penalty;
    }
    if (pressure_order != 0 &&
        (loom_low_schedule_candidate_threatens_pressure_cliff(lhs) ||
         loom_low_schedule_candidate_threatens_pressure_cliff(rhs))) {
      return pressure_order < 0;
    }
    if (lhs.effective_stall_cycles != rhs.effective_stall_cycles) {
      return lhs.effective_stall_cycles < rhs.effective_stall_cycles;
    }
    if (lhs.hazard_stall_cycles != rhs.hazard_stall_cycles) {
      return lhs.hazard_stall_cycles < rhs.hazard_stall_cycles;
    }
    if (lhs.resource_stall_cycles != rhs.resource_stall_cycles) {
      return lhs.resource_stall_cycles < rhs.resource_stall_cycles;
    }
    if (lhs.data_ready_stall_cycles != rhs.data_ready_stall_cycles) {
      return lhs.data_ready_stall_cycles < rhs.data_ready_stall_cycles;
    }
    if (lhs.pair_affinity_score != rhs.pair_affinity_score) {
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
    if (lhs.critical_path_cycles != rhs.critical_path_cycles) {
      return lhs.critical_path_cycles > rhs.critical_path_cycles;
    }
  }
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_LATENCY_HIDING) {
    if (lhs.dependency_latency_cycles != rhs.dependency_latency_cycles) {
      return lhs.dependency_latency_cycles < rhs.dependency_latency_cycles;
    }
    if (lhs.latency_cycles != rhs.latency_cycles) {
      return lhs.latency_cycles > rhs.latency_cycles;
    }
  }
  if (lhs.pressure_cliff_penalty != rhs.pressure_cliff_penalty) {
    return lhs.pressure_cliff_penalty < rhs.pressure_cliff_penalty;
  }
  if (lhs.pair_affinity_score != rhs.pair_affinity_score) {
    return loom_low_schedule_candidate_has_better_pair_affinity(lhs, rhs);
  }
  if (pressure_order != 0) {
    return pressure_order < 0;
  }
  return lhs.source_ordinal < rhs.source_ordinal;
}

static void loom_low_schedule_record_candidate_decision(
    loom_low_schedule_build_state_t* state, uint32_t block_index,
    uint32_t scheduled_ordinal, uint32_t ready_candidate_count,
    uint32_t chosen_node, loom_low_schedule_candidate_score_t chosen_score,
    uint32_t rejected_node,
    loom_low_schedule_candidate_score_t rejected_score) {
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
          .chosen_node = chosen_node,
          .rejected_node = rejected_node,
          .chosen_dependency_latency_cycles =
              chosen_score.dependency_latency_cycles,
          .chosen_latency_cycles = chosen_score.latency_cycles,
          .chosen_pair_affinity_score = chosen_score.pair_affinity_score,
          .rejected_dependency_latency_cycles =
              rejected_score.dependency_latency_cycles,
          .rejected_latency_cycles = rejected_score.latency_cycles,
          .rejected_pair_affinity_score = rejected_score.pair_affinity_score,
          .chosen_projected_live_units = chosen_score.projected_live_units,
          .chosen_killed_live_units = chosen_score.killed_live_units,
          .chosen_produced_live_units = chosen_score.produced_live_units,
          .rejected_projected_live_units = rejected_score.projected_live_units,
          .rejected_killed_live_units = rejected_score.killed_live_units,
          .rejected_produced_live_units = rejected_score.produced_live_units,
          .chosen_data_ready_stall_cycles =
              chosen_score.data_ready_stall_cycles,
          .chosen_resource_stall_cycles = chosen_score.resource_stall_cycles,
          .chosen_hazard_stall_cycles = chosen_score.hazard_stall_cycles,
          .chosen_effective_stall_cycles = chosen_score.effective_stall_cycles,
          .chosen_bottleneck_resource_id = chosen_score.bottleneck_resource_id,
          .chosen_pressure_cliff_penalty = chosen_score.pressure_cliff_penalty,
          .chosen_pressure_cliff_reg_class_id =
              chosen_score.pressure_cliff_reg_class_id,
          .chosen_pressure_cliff_units = chosen_score.pressure_cliff_units,
          .chosen_units_until_pressure_cliff =
              chosen_score.units_until_pressure_cliff,
          .rejected_data_ready_stall_cycles =
              rejected_score.data_ready_stall_cycles,
          .rejected_resource_stall_cycles =
              rejected_score.resource_stall_cycles,
          .rejected_hazard_stall_cycles = rejected_score.hazard_stall_cycles,
          .rejected_effective_stall_cycles =
              rejected_score.effective_stall_cycles,
          .rejected_bottleneck_resource_id =
              rejected_score.bottleneck_resource_id,
          .rejected_pressure_cliff_penalty =
              rejected_score.pressure_cliff_penalty,
          .rejected_pressure_cliff_reg_class_id =
              rejected_score.pressure_cliff_reg_class_id,
          .rejected_pressure_cliff_units = rejected_score.pressure_cliff_units,
          .rejected_units_until_pressure_cliff =
              rejected_score.units_until_pressure_cliff,
      };
}

static void loom_low_schedule_note_pair_affinity_node_scheduled(
    loom_low_schedule_build_state_t* state, uint32_t node_index,
    loom_low_schedule_candidate_score_t score) {
  if (loom_low_schedule_pair_affinity_list_is_empty(
          state->options->pair_affinities)) {
    state->pending_pair_affinity_node = LOOM_LOW_SCHEDULE_NODE_NONE;
    return;
  }

  const loom_low_schedule_node_t* node = &state->nodes[node_index];
  if (node->descriptor != NULL) {
    if (score.pair_affinity_score != 0 &&
        state->pending_pair_affinity_node != LOOM_LOW_SCHEDULE_NODE_NONE) {
      state->pending_pair_affinity_node = LOOM_LOW_SCHEDULE_NODE_NONE;
      return;
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
  const uint16_t relation_count =
      loom_low_storage_relation_count(state->module, node->op);
  for (uint16_t i = 0; i < relation_count; ++i) {
    loom_low_storage_relation_t relation = {0};
    loom_low_storage_relation_get(state->module, node->op, i, &relation);
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

static iree_status_t loom_low_schedule_note_pressure_node_scheduled(
    loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state, uint32_t node_index,
    loom_low_schedule_candidate_score_t score) {
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
      if (unit_count > pressure_state->current_live_units) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "low schedule pressure accounting underflow for value %u",
            value->value_id);
      }
      pressure_state->current_live_units -= unit_count;
      const uint16_t reg_class_id = value->register_class_id;
      if (reg_class_id != LOOM_LOW_REG_CLASS_NONE &&
          pressure_state->current_live_units_by_reg_class) {
        if (unit_count >
            pressure_state->current_live_units_by_reg_class[reg_class_id]) {
          return iree_make_status(
              IREE_STATUS_FAILED_PRECONDITION,
              "low schedule pressure accounting underflow for register class");
        }
        pressure_state->current_live_units_by_reg_class[reg_class_id] -=
            unit_count;
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
    }
  }
  if (pressure_state->current_live_units != score.projected_live_units) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low schedule pressure accounting did not match chosen candidate: "
        "node=%" PRIu32 " current=%" PRIu64 " projected=%" PRIu64
        " killed=%" PRIu64 " produced=%" PRIu64,
        node_index, pressure_state->current_live_units,
        score.projected_live_units, score.killed_live_units,
        score.produced_live_units);
  }
  if (state->pressure_steps) {
    state->pressure_steps[state->pressure_step_count++] =
        (loom_low_schedule_pressure_step_t){
            .node_index = node_index,
            .block_index = node->block_index,
            .scheduled_ordinal = node->scheduled_ordinal,
            .live_units_before = live_units_before,
            .killed_live_units = score.killed_live_units,
            .produced_live_units = score.produced_live_units,
            .live_units_after = pressure_state->current_live_units,
        };
  }
  return iree_ok_status();
}

static bool loom_low_schedule_ready_node_less(
    const loom_low_schedule_build_state_t* state, uint32_t lhs_node,
    uint32_t rhs_node) {
  const loom_low_schedule_node_t* lhs = &state->nodes[lhs_node];
  const loom_low_schedule_node_t* rhs = &state->nodes[rhs_node];
  if (lhs->source_ordinal != rhs->source_ordinal) {
    return lhs->source_ordinal < rhs->source_ordinal;
  }
  return lhs_node < rhs_node;
}

static void loom_low_schedule_ready_heap_push(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_ready_heap_t* heap, uint32_t node_index) {
  iree_host_size_t index = heap->count++;
  while (index != 0) {
    const iree_host_size_t parent_index = (index - 1) / 2;
    const uint32_t parent_node = heap->node_indices[parent_index];
    if (loom_low_schedule_ready_node_less(state, parent_node, node_index)) {
      break;
    }
    heap->node_indices[index] = parent_node;
    index = parent_index;
  }
  heap->node_indices[index] = node_index;
}

static void loom_low_schedule_ready_heap_sift_down(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_ready_heap_t* heap, iree_host_size_t index,
    uint32_t node_index) {
  while (true) {
    const iree_host_size_t left_index = index * 2 + 1;
    if (left_index >= heap->count) {
      break;
    }
    const iree_host_size_t right_index = left_index + 1;
    iree_host_size_t child_index = left_index;
    if (right_index < heap->count && loom_low_schedule_ready_node_less(
                                         state, heap->node_indices[right_index],
                                         heap->node_indices[left_index])) {
      child_index = right_index;
    }
    const uint32_t child_node = heap->node_indices[child_index];
    if (loom_low_schedule_ready_node_less(state, node_index, child_node)) {
      break;
    }
    heap->node_indices[index] = child_node;
    index = child_index;
  }
  if (heap->count != 0) {
    heap->node_indices[index] = node_index;
  }
}

static uint32_t loom_low_schedule_ready_heap_remove_at(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_ready_heap_t* heap, iree_host_size_t index) {
  IREE_ASSERT(index < heap->count);
  const uint32_t result = heap->node_indices[index];
  const uint32_t node_index = heap->node_indices[--heap->count];
  if (index == heap->count) {
    return result;
  }
  if (index != 0) {
    iree_host_size_t parent_index = (index - 1) / 2;
    uint32_t parent_node = heap->node_indices[parent_index];
    if (loom_low_schedule_ready_node_less(state, node_index, parent_node)) {
      do {
        heap->node_indices[index] = parent_node;
        index = parent_index;
        if (index == 0) {
          break;
        }
        parent_index = (index - 1) / 2;
        parent_node = heap->node_indices[parent_index];
      } while (
          loom_low_schedule_ready_node_less(state, node_index, parent_node));
      heap->node_indices[index] = node_index;
      return result;
    }
  }
  loom_low_schedule_ready_heap_sift_down(state, heap, index, node_index);
  return result;
}

static uint32_t loom_low_schedule_ready_heap_pop(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_ready_heap_t* heap) {
  return loom_low_schedule_ready_heap_remove_at(state, heap, 0);
}

static bool loom_low_schedule_node_is_unscheduled_in_block(
    const loom_low_schedule_build_state_t* state, uint32_t node_index,
    iree_host_size_t node_count, iree_host_size_t block_index) {
  return node_index < node_count &&
         state->nodes[node_index].block_index == block_index &&
         state->nodes[node_index].scheduled_ordinal ==
             LOOM_LOW_SCHEDULE_NODE_NONE;
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

static iree_status_t loom_low_schedule_record_first_unresolved_dependency(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_block_t* block_record, iree_host_size_t node_count,
    uint32_t scheduled_in_block) {
  for (iree_host_size_t i = 0; i < state->dependency_count; ++i) {
    const loom_low_schedule_dependency_t* dependency = &state->dependencies[i];
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
  uint32_t* stack_next_dependencies = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, node_count, sizeof(*visit_states), (void**)&visit_states));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, node_count, sizeof(*parent_nodes), (void**)&parent_nodes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, node_count, sizeof(*stack_nodes), (void**)&stack_nodes));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, node_count, sizeof(*stack_next_dependencies),
      (void**)&stack_next_dependencies));
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
    stack_next_dependencies[0] = state->outgoing_heads[start_node];
    visit_states[start_node] = 1;
    while (stack_count != 0) {
      const uint32_t producer_node = stack_nodes[stack_count - 1];
      bool advanced = false;
      for (uint32_t dependency_index = stack_next_dependencies[stack_count - 1];
           dependency_index != LOOM_LOW_SCHEDULE_NODE_NONE;
           dependency_index = state->outgoing_next_indices[dependency_index]) {
        stack_next_dependencies[stack_count - 1] =
            state->outgoing_next_indices[dependency_index];
        const loom_low_schedule_dependency_t* dependency =
            &state->dependencies[dependency_index];
        if (!loom_low_schedule_node_is_unscheduled_in_block(
                state, dependency->consumer_node, node_count,
                state->current_block_index)) {
          continue;
        }
        const uint32_t consumer_node = dependency->consumer_node;
        if (visit_states[consumer_node] == 0) {
          parent_nodes[consumer_node] = producer_node;
          stack_nodes[stack_count] = consumer_node;
          stack_next_dependencies[stack_count] =
              state->outgoing_heads[consumer_node];
          visit_states[consumer_node] = 1;
          ++stack_count;
          advanced = true;
          break;
        }
        if (visit_states[consumer_node] == 1) {
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

static void loom_low_schedule_compute_node_critical_paths(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count,
    const uint32_t* outgoing_heads, const uint32_t* outgoing_next_indices) {
  if (state->node_critical_path_cycles == NULL) {
    return;
  }
  for (iree_host_size_t i = node_count; i > 0; --i) {
    const uint32_t node_index = (uint32_t)(i - 1);
    const loom_low_schedule_node_t* node = &state->nodes[node_index];
    uint32_t successor_path_cycles = 0;
    if (outgoing_next_indices != NULL) {
      for (uint32_t dependency_index = outgoing_heads[node_index];
           dependency_index != LOOM_LOW_SCHEDULE_NODE_NONE;
           dependency_index = outgoing_next_indices[dependency_index]) {
        const loom_low_schedule_dependency_t* dependency =
            &state->dependencies[dependency_index];
        if (dependency->producer_node != node_index ||
            dependency->consumer_node >= node_count) {
          continue;
        }
        const loom_low_schedule_node_t* consumer =
            &state->nodes[dependency->consumer_node];
        if (consumer->block_index != node->block_index) {
          continue;
        }
        successor_path_cycles = loom_low_schedule_max_u32(
            successor_path_cycles,
            state->node_critical_path_cycles[dependency->consumer_node]);
      }
    }
    state->node_critical_path_cycles[node_index] = iree_math_saturating_add_u32(
        node->latency_cycles, successor_path_cycles);
  }
}

static iree_status_t loom_low_schedule_run_list_scheduler(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count) {
  uint32_t* indegrees = NULL;
  uint32_t* outgoing_heads = NULL;
  uint32_t* outgoing_next_indices = NULL;
  loom_low_schedule_ready_heap_t ready_heap = {0};
  uint32_t* inspected_nodes = NULL;
  loom_low_schedule_candidate_score_t* inspected_scores = NULL;
  loom_low_schedule_pressure_state_t pressure_state = {0};
  if (state->dependency_count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "low schedule dependency count exceeds uint32_t adjacency capacity");
  }
  if (node_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count, sizeof(*indegrees), (void**)&indegrees));
    memset(indegrees, 0, node_count * sizeof(*indegrees));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->arena, node_count,
                                                   sizeof(*outgoing_heads),
                                                   (void**)&outgoing_heads));
    memset(outgoing_heads, 0xFF, node_count * sizeof(*outgoing_heads));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count, sizeof(*ready_heap.node_indices),
        (void**)&ready_heap.node_indices));
  }
  if (state->dependency_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, state->dependency_count, sizeof(*outgoing_next_indices),
        (void**)&outgoing_next_indices));
    memset(outgoing_next_indices, 0xFF,
           state->dependency_count * sizeof(*outgoing_next_indices));
  }
  if (loom_low_schedule_uses_pressure_strategy(state)) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, LOOM_LOW_SCHEDULE_READY_WINDOW, sizeof(*inspected_nodes),
        (void**)&inspected_nodes));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, LOOM_LOW_SCHEDULE_READY_WINDOW, sizeof(*inspected_scores),
        (void**)&inspected_scores));
  }
  IREE_RETURN_IF_ERROR(
      loom_low_schedule_allocate_pressure_state(state, &pressure_state));
  for (iree_host_size_t i = 0; i < state->dependency_count; ++i) {
    const loom_low_schedule_dependency_t* dependency = &state->dependencies[i];
    if (dependency->consumer_node < node_count) {
      ++indegrees[dependency->consumer_node];
    }
    if (dependency->producer_node < node_count) {
      outgoing_next_indices[i] = outgoing_heads[dependency->producer_node];
      outgoing_heads[dependency->producer_node] = (uint32_t)i;
    }
  }
  state->outgoing_heads = outgoing_heads;
  state->outgoing_next_indices = outgoing_next_indices;
  loom_low_schedule_compute_node_critical_paths(
      state, node_count, outgoing_heads, outgoing_next_indices);

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
      IREE_RETURN_IF_ERROR(loom_low_schedule_initialize_block_pressure(
          state, block_record, &pressure_state));
    }
    ready_heap.count = 0;
    for (uint32_t node_index = block_record->node_start;
         node_index < block_node_end; ++node_index) {
      if (indegrees[node_index] == 0) {
        loom_low_schedule_ready_heap_push(state, &ready_heap, node_index);
      }
    }
    uint32_t scheduled_in_block = 0;
    while (scheduled_in_block < block_record->node_count) {
      state->current_issue_cycle = scheduled_in_block;
      uint32_t chosen_node = LOOM_LOW_SCHEDULE_NODE_NONE;
      loom_low_schedule_candidate_score_t chosen_score = {0};
      uint32_t rejected_node = LOOM_LOW_SCHEDULE_NODE_NONE;
      loom_low_schedule_candidate_score_t rejected_score = {0};
      uint32_t ready_candidate_count = 0;
      if (ready_heap.count == 0) {
        return loom_low_schedule_handle_dependency_cycle(
            state, block_record, node_count, scheduled_in_block);
      }
      if (!loom_low_schedule_uses_pressure_strategy(state)) {
        chosen_node = loom_low_schedule_ready_heap_pop(state, &ready_heap);
      } else {
        const iree_host_size_t inspected_count =
            ready_heap.count < LOOM_LOW_SCHEDULE_READY_WINDOW
                ? ready_heap.count
                : LOOM_LOW_SCHEDULE_READY_WINDOW;
        ready_candidate_count = (uint32_t)inspected_count;
        for (iree_host_size_t i = 0; i < inspected_count; ++i) {
          const uint32_t node_index =
              loom_low_schedule_ready_heap_pop(state, &ready_heap);
          inspected_nodes[i] = node_index;
          IREE_RETURN_IF_ERROR(loom_low_schedule_score_candidate(
              state, &pressure_state, node_index, &inspected_scores[i]));
          if (chosen_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
              loom_low_schedule_candidate_score_less(state, inspected_scores[i],
                                                     chosen_score)) {
            if (chosen_node != LOOM_LOW_SCHEDULE_NODE_NONE) {
              rejected_node = chosen_node;
              rejected_score = chosen_score;
            }
            chosen_node = node_index;
            chosen_score = inspected_scores[i];
          } else if (rejected_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
                     loom_low_schedule_candidate_score_less(
                         state, inspected_scores[i], rejected_score)) {
            rejected_node = node_index;
            rejected_score = inspected_scores[i];
          }
        }
        iree_host_size_t chosen_ready_heap_index = IREE_HOST_SIZE_MAX;
        const bool chosen_threatens_pressure_cliff =
            loom_low_schedule_candidate_threatens_pressure_cliff(chosen_score);
        iree_host_size_t scan_count = 0;
        if (chosen_threatens_pressure_cliff) {
          scan_count = ready_heap.count;
        } else if (chosen_score.effective_stall_cycles != 0 &&
                   ready_heap.count != 0) {
          const iree_host_size_t extra_scan_capacity =
              LOOM_LOW_SCHEDULE_READY_STALL_SCAN_WINDOW >
                      LOOM_LOW_SCHEDULE_READY_WINDOW
                  ? LOOM_LOW_SCHEDULE_READY_STALL_SCAN_WINDOW -
                        LOOM_LOW_SCHEDULE_READY_WINDOW
                  : 0;
          scan_count = ready_heap.count < extra_scan_capacity
                           ? ready_heap.count
                           : extra_scan_capacity;
        }
        if (scan_count != 0) {
          for (iree_host_size_t i = 0; i < scan_count; ++i) {
            loom_low_schedule_candidate_score_t candidate_score = {0};
            const uint32_t node_index = ready_heap.node_indices[i];
            IREE_RETURN_IF_ERROR(loom_low_schedule_score_candidate(
                state, &pressure_state, node_index, &candidate_score));
            ++ready_candidate_count;
            if (loom_low_schedule_candidate_score_less(state, candidate_score,
                                                       chosen_score)) {
              if (chosen_node != LOOM_LOW_SCHEDULE_NODE_NONE) {
                rejected_node = chosen_node;
                rejected_score = chosen_score;
              }
              chosen_node = node_index;
              chosen_score = candidate_score;
              chosen_ready_heap_index = i;
            } else if (rejected_node == LOOM_LOW_SCHEDULE_NODE_NONE ||
                       loom_low_schedule_candidate_score_less(
                           state, candidate_score, rejected_score)) {
              rejected_node = node_index;
              rejected_score = candidate_score;
            }
          }
        }
        if (chosen_ready_heap_index != IREE_HOST_SIZE_MAX) {
          (void)loom_low_schedule_ready_heap_remove_at(state, &ready_heap,
                                                       chosen_ready_heap_index);
        }
        for (iree_host_size_t i = 0; i < inspected_count; ++i) {
          if (inspected_nodes[i] == chosen_node) {
            continue;
          }
          loom_low_schedule_ready_heap_push(state, &ready_heap,
                                            inspected_nodes[i]);
        }
      }

      state->nodes[chosen_node].scheduled_ordinal = scheduled_in_block++;
      state->current_issue_cycle = state->nodes[chosen_node].scheduled_ordinal;
      state->scheduled_node_indices[state->scheduled_node_count++] =
          chosen_node;
      ++block_record->scheduled_node_count;
      loom_low_schedule_note_pair_affinity_node_scheduled(state, chosen_node,
                                                          chosen_score);
      if (loom_low_schedule_uses_pressure_strategy(state)) {
        loom_low_schedule_record_candidate_decision(
            state, block_index, state->nodes[chosen_node].scheduled_ordinal,
            ready_candidate_count, chosen_node, chosen_score, rejected_node,
            rejected_score);
        IREE_RETURN_IF_ERROR(loom_low_schedule_note_pressure_node_scheduled(
            state, &pressure_state, chosen_node, chosen_score));
      }
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_note_descriptor_rows_for_node(state, chosen_node));

      for (uint32_t dependency_index = outgoing_heads[chosen_node];
           dependency_index != LOOM_LOW_SCHEDULE_NODE_NONE;
           dependency_index = outgoing_next_indices[dependency_index]) {
        const loom_low_schedule_dependency_t* dependency =
            &state->dependencies[dependency_index];
        if (dependency->producer_node != chosen_node) {
          continue;
        }
        if (dependency->consumer_node < node_count) {
          --indegrees[dependency->consumer_node];
          if (state->node_ready_issue_cycles != NULL &&
              dependency->kind == LOOM_LOW_SCHEDULE_DEPENDENCY_SSA) {
            const uint32_t ready_cycle = iree_math_saturating_add_u32(
                state->current_issue_cycle,
                state->nodes[chosen_node].latency_cycles);
            if (ready_cycle >
                state->node_ready_issue_cycles[dependency->consumer_node]) {
              state->node_ready_issue_cycles[dependency->consumer_node] =
                  ready_cycle;
            }
          }
          if (indegrees[dependency->consumer_node] == 0 &&
              state->nodes[dependency->consumer_node].block_index ==
                  block_index) {
            loom_low_schedule_ready_heap_push(state, &ready_heap,
                                              dependency->consumer_node);
          }
        }
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_schedule_function(
    loom_module_t* module, const loom_op_t* low_func_op,
    const loom_low_schedule_options_t* options, iree_arena_allocator_t* arena,
    loom_low_schedule_table_t* out_table) {
  if (!loom_low_function_def_isa(low_func_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "expected low.func.def or low.kernel.def");
  }
  if (!loom_low_schedule_strategy_is_valid(options->strategy)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown low schedule strategy %d",
                            (int)options->strategy);
  }
  *out_table = (loom_low_schedule_table_t){0};

  loom_low_schedule_build_state_t state = {
      .module = module,
      .options = options,
      .arena = arena,
      .function_op = low_func_op,
      .body = loom_low_function_body((loom_op_t*)low_func_op),
  };
  IREE_ASSERT(state.body != NULL);
  IREE_RETURN_IF_ERROR(loom_low_schedule_verify_memory_access_table(
      options->memory_access_table, low_func_op, state.body));
  if (options->memory_access_table.function_op == low_func_op) {
    state.memory_access_records = options->memory_access_table.values;
    state.memory_access_record_count = options->memory_access_table.count;
  }
  IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
      module, low_func_op, options->descriptor_registry,
      options->target_selection, options->emitter, &state.target));
  if (!state.target.descriptor_set) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low function target did not resolve");
  }
  state.register_type_resolver =
      loom_low_register_type_resolver_for_descriptor_set(
          state.target.descriptor_set);

  iree_host_size_t node_count = 0;
  loom_low_schedule_count_nodes(state.body, &node_count);
  const bool needs_liveness =
      iree_any_bit_set(options->flags,
                       LOOM_LOW_SCHEDULE_FLAG_RETAIN_LIVENESS) ||
      iree_any_bit_set(options->diagnostic_flags,
                       LOOM_LOW_SCHEDULE_DIAGNOSTIC_PRESSURE_PEAKS);
  loom_local_value_domain_t value_domain = {0};
  loom_liveness_analysis_t liveness = {0};
  iree_status_t status = loom_local_value_domain_acquire_for_region(
      module, state.body, arena, &value_domain);
  if (iree_status_is_ok(status)) {
    state.value_domain = &value_domain;
    status = loom_low_schedule_initialize_storage(&state, node_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_low_schedule_fill_nodes(&state);
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
    status = loom_liveness_analyze_local_value_domain(
        &value_domain, loom_liveness_order_empty(), arena, &liveness);
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
        .module = module,
        .function_op = low_func_op,
        .target = state.target,
        .memory_access_table = options->memory_access_table,
        .value_ids = value_domain.value_ids,
        .value_count = value_domain.value_count,
        .liveness = liveness,
        .blocks = state.blocks,
        .block_count = state.body->block_count,
        .nodes = state.nodes,
        .node_count = node_count,
        .dependencies = state.dependencies,
        .dependency_count = state.dependency_count,
        .visibility_dependencies = state.visibility_dependencies,
        .visibility_dependency_count = state.visibility_dependency_count,
        .scheduled_node_indices = state.scheduled_node_indices,
        .scheduled_node_count = state.scheduled_node_count,
        .error_count = state.error_count,
        .failure = state.failure,
        .pressure_steps = state.pressure_steps,
        .pressure_step_count = state.pressure_step_count,
        .candidate_decisions = state.candidate_decisions,
        .candidate_decision_count = state.candidate_decision_count,
        .resource_uses = state.resource_uses,
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
    loom_target_bundle_storage_rebind(&out_table->target.bundle_storage);
  }
  loom_local_value_domain_release(&value_domain);
  return status;
}
