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
#include "loom/codegen/low/schedule/candidate_policy.h"
#include "loom/codegen/low/schedule/completion_wait.h"
#include "loom/codegen/low/schedule/context.h"
#include "loom/codegen/low/schedule/descriptor_rows.h"
#include "loom/codegen/low/schedule/diagnostics.h"
#include "loom/codegen/low/schedule/graph.h"
#include "loom/codegen/low/schedule/pressure.h"
#include "loom/codegen/low/schedule/ready_frontier.h"
#include "loom/codegen/low/schedule/ready_policy.h"
#include "loom/codegen/low/storage_relation.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/target/registers.h"

enum loom_low_schedule_state_access_bits_e {
  LOOM_LOW_SCHEDULE_STATE_ACCESS_READ = 1u << 0,
  LOOM_LOW_SCHEDULE_STATE_ACCESS_WRITE = 1u << 1,
};

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
        .state_next_write =
            {
                .node_index = LOOM_LOW_SCHEDULE_NODE_NONE,
                .endpoint =
                    {
                        .attachment_index = LOOM_LOW_ID_NONE,
                        .timing_event_id = LOOM_LOW_TIMING_EVENT_NONE,
                        .attachment_kind =
                            LOOM_LOW_SCHEDULE_DEPENDENCY_ATTACHMENT_NONE,
                    },
            },
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
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->arena, state->body->block_count,
                                sizeof(*state->liveness_block_orders),
                                (void**)&state->liveness_block_orders));
  if (node_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(state->arena, node_count,
                                                   sizeof(*state->nodes),
                                                   (void**)&state->nodes));
    memset(state->nodes, 0, node_count * sizeof(*state->nodes));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count, sizeof(*state->scheduled_node_indices),
        (void**)&state->scheduled_node_indices));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count, sizeof(*state->issue_groups),
        (void**)&state->issue_groups));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count, sizeof(*state->scheduled_ops),
        (void**)&state->scheduled_ops));
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
    if (loom_low_schedule_strategy_uses_pressure(state->options->strategy) &&
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
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, node_count, sizeof(*state->node_ready_issue_cycles),
        (void**)&state->node_ready_issue_cycles));
    memset(state->node_ready_issue_cycles, 0,
           node_count * sizeof(*state->node_ready_issue_cycles));
    if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, node_count, sizeof(*state->node_completion_wait_cycles),
          (void**)&state->node_completion_wait_cycles));
      memset(state->node_completion_wait_cycles, 0,
             node_count * sizeof(*state->node_completion_wait_cycles));
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          state->arena, node_count,
          sizeof(*state->node_opened_completion_latency_cycles),
          (void**)&state->node_opened_completion_latency_cycles));
      memset(
          state->node_opened_completion_latency_cycles, 0,
          node_count * sizeof(*state->node_opened_completion_latency_cycles));
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
  IREE_ASSERT_LE(node_count, UINT32_MAX);
  IREE_RETURN_IF_ERROR(loom_low_schedule_storage_relation_index_initialize(
      state->module, state->value_domain, state->nodes, (uint32_t)node_count,
      state->storage_relation_count, state->arena, &state->storage_relations));
  bool needs_storage_read_tracking = false;
  bool needs_edge_source_worklist = false;
  iree_host_size_t max_operand_count = 0;
  for (iree_host_size_t node_index = 0; node_index < node_count; ++node_index) {
    const loom_low_schedule_node_t* node = &state->nodes[node_index];
    max_operand_count = iree_max(max_operand_count, node->operand_count);
    const uint32_t relation_begin =
        loom_low_schedule_storage_relation_index_begin(
            &state->storage_relations, (uint32_t)node_index);
    const uint32_t relation_end = loom_low_schedule_storage_relation_index_end(
        &state->storage_relations, (uint32_t)node_index);
    for (uint32_t relation_index = relation_begin;
         relation_index < relation_end; ++relation_index) {
      const loom_low_schedule_storage_relation_t* relation =
          loom_low_schedule_storage_relation_index_at(&state->storage_relations,
                                                      relation_index);
      if (relation->cause == LOOM_LOW_STORAGE_RELATION_CAUSE_TIED_RESULT) {
        state->values[relation->source_ordinal].flags |=
            LOOM_LOW_SCHEDULE_VALUE_FLAG_STORAGE_READ_TRACKED;
        needs_storage_read_tracking = true;
      }
      if (relation->cause == LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_BRANCH ||
          relation->cause == LOOM_LOW_STORAGE_RELATION_CAUSE_LOW_SCF_YIELD) {
        state->values[relation->destination_ordinal].flags |=
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
    const uint32_t first_ordinal = loom_low_descriptor_set_descriptor_ordinal(
        descriptor_set, affinity->first_descriptor);
    const uint32_t second_ordinal = loom_low_descriptor_set_descriptor_ordinal(
        descriptor_set, affinity->second_descriptor);
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
  if (state->detached_transfer_node_count == 0 ||
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
  bool has_resource_uses = false;
  iree_host_size_t max_descriptor_operand_count = 0;
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
        state->arena, reg_class_count, sizeof(*state->state_last_writes),
        (void**)&state->state_last_writes));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count, sizeof(*state->state_first_writes),
        (void**)&state->state_first_writes));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, reg_class_count, sizeof(*state->state_ordering_frontiers),
        (void**)&state->state_ordering_frontiers));
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
  }
  IREE_RETURN_IF_ERROR(loom_low_schedule_verify_structural_state_reads(state));
  for (iree_host_size_t node_index = 0; node_index < node_count; ++node_index) {
    const loom_low_schedule_node_t* node = &state->nodes[node_index];
    if (node->descriptor != NULL) {
      max_descriptor_operand_count =
          iree_max(max_descriptor_operand_count, node->operand_count);
    }
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
  if (max_descriptor_operand_count != 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->arena, max_descriptor_operand_count,
                                  sizeof(*state->descriptor_operands.indices),
                                  (void**)&state->descriptor_operands.indices));
    state->descriptor_operands.capacity = max_descriptor_operand_count;
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
  if (state->options->strategy == LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_resource_calendar_initialize(
        descriptor_set, state->arena, &state->resource_calendar));
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
        state->arena, effect_read_capacity, sizeof(*state->effect_read_entries),
        (void**)&state->effect_read_entries));
    state->effect_read_capacity = effect_read_capacity;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->arena, effect_read_capacity,
                                  sizeof(*state->effect_write_entries),
                                  (void**)&state->effect_write_entries));
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

static void loom_low_schedule_insert_ready_node(
    const loom_low_schedule_build_state_t* state,
    loom_low_schedule_pressure_state_t* pressure_state,
    loom_low_schedule_ready_policy_t* ready_policy, uint32_t node_index) {
  loom_low_schedule_ready_keys_t keys = {0};
  if (loom_low_schedule_strategy_uses_pressure(state->options->strategy)) {
    keys = loom_low_schedule_pressure_ready_keys(state, pressure_state,
                                                 node_index);
  } else {
    keys.values[LOOM_LOW_SCHEDULE_READY_VIEW_SOURCE] =
        state->nodes[node_index].source_ordinal;
  }
  loom_low_schedule_ready_policy_insert(state, ready_policy, node_index, &keys);
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

static bool loom_low_schedule_node_has_zero_issue_width(
    const loom_low_schedule_node_t* node) {
  return iree_any_bit_set(node->flags,
                          LOOM_LOW_SCHEDULE_NODE_FLAG_ZERO_ISSUE_WIDTH);
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
      loom_low_schedule_insert_ready_node(state, pressure_state, ready_policy,
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

// Finds the explicit architectural-state value responsible for a
// read-before-clobber dependency. Other state dependency forms have no SSA
// value that can be rematerialized.
static loom_value_id_t loom_low_schedule_state_value_for_dependency(
    const loom_low_schedule_build_state_t* state,
    const loom_low_schedule_dependency_t* dependency) {
  if (dependency->kind != LOOM_LOW_SCHEDULE_DEPENDENCY_STATE) {
    return LOOM_VALUE_ID_INVALID;
  }
  const loom_low_schedule_node_t* reader =
      &state->nodes[dependency->producer_node];
  const loom_value_ordinal_t* operand_ordinals =
      loom_low_schedule_node_const_operand_ordinals(reader);
  if (dependency->value_operand_index != LOOM_LOW_ID_NONE) {
    IREE_ASSERT_LT(dependency->value_operand_index, reader->operand_count);
    return state->values[operand_ordinals[dependency->value_operand_index]]
        .value_id;
  }
  loom_value_id_t state_value_id = LOOM_VALUE_ID_INVALID;
  for (uint16_t i = 0; i < reader->operand_count; ++i) {
    const loom_low_schedule_value_record_t* value =
        &state->values[operand_ordinals[i]];
    if (value->register_class_id == LOOM_LOW_REG_CLASS_NONE ||
        state->reg_class_state_flags[value->register_class_id] == 0 ||
        value->state_next_write.node_index != dependency->consumer_node) {
      continue;
    }
    if (state_value_id != LOOM_VALUE_ID_INVALID &&
        state_value_id != value->value_id) {
      return LOOM_VALUE_ID_INVALID;
    }
    state_value_id = value->value_id;
  }
  return state_value_id;
}

static void loom_low_schedule_record_cycle_state_value(
    loom_low_schedule_build_state_t* state) {
  if (state->failure.state_value_id != LOOM_VALUE_ID_INVALID ||
      state->failure.cycle_node_count < 2 ||
      iree_any_bit_set(state->failure.flags,
                       LOOM_LOW_SCHEDULE_FAILURE_FLAG_CYCLE_PATH_TRUNCATED |
                           LOOM_LOW_SCHEDULE_FAILURE_FLAG_WITNESS_EDGE_ONLY)) {
    return;
  }
  for (uint32_t i = 0; i < state->failure.cycle_node_count; ++i) {
    const uint32_t producer_node = state->failure.cycle_nodes[i];
    const uint32_t consumer_node =
        state->failure.cycle_nodes[(i + 1) % state->failure.cycle_node_count];
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_find_dependency_witness(state, producer_node,
                                                  consumer_node);
    const loom_value_id_t state_value_id =
        loom_low_schedule_state_value_for_dependency(state, dependency);
    if (state_value_id != LOOM_VALUE_ID_INVALID) {
      state->failure.state_value_id = state_value_id;
      return;
    }
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
      .operand_index = dependency->value_operand_index == LOOM_LOW_ID_NONE
                           ? UINT32_MAX
                           : dependency->value_operand_index,
      .state_value_id =
          loom_low_schedule_state_value_for_dependency(state, dependency),
      .cycle_node_count = 0,
  };
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
      .state_value_id = LOOM_VALUE_ID_INVALID,
      .cycle_node_count = 0,
  };
  return iree_ok_status();
}

// Records a dependency that points backward across the current source range.
// Source-order boundaries impose an implicit edge from the current range to
// every later range. A dependency in the opposite direction therefore closes
// a cycle even though that implicit edge is not stored in the dependency graph.
static bool loom_low_schedule_record_source_range_cycle(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_block_t* block_record, iree_host_size_t node_count,
    uint32_t scheduled_in_block, uint32_t range_start, uint32_t range_end) {
  const loom_low_schedule_dependency_t* selected_dependency = NULL;
  for (uint32_t i = 0; i < state->dependencies.count; ++i) {
    const loom_low_schedule_dependency_t* dependency =
        loom_low_schedule_dependency_graph_at(&state->dependencies, i);
    if (dependency->consumer_node < range_start ||
        dependency->consumer_node >= range_end ||
        dependency->producer_node < range_end ||
        !loom_low_schedule_node_is_unscheduled_in_block(
            state, dependency->producer_node, node_count,
            state->current_block_index)) {
      continue;
    }
    selected_dependency = dependency;
    if (dependency->kind == LOOM_LOW_SCHEDULE_DEPENDENCY_STATE &&
        dependency->value_operand_index != LOOM_LOW_ID_NONE) {
      break;
    }
  }
  if (selected_dependency == NULL) {
    return false;
  }

  loom_low_schedule_record_dependency_cycle_failure(
      state, block_record, scheduled_in_block,
      selected_dependency->producer_node, selected_dependency->consumer_node,
      selected_dependency);
  state->failure.flags |= LOOM_LOW_SCHEDULE_FAILURE_FLAG_WITNESS_EDGE_ONLY;
  state->failure.cycle_nodes[0] = selected_dependency->producer_node;
  state->failure.cycle_nodes[1] = selected_dependency->consumer_node;
  state->failure.cycle_node_count = 2;
  return true;
}

static iree_status_t loom_low_schedule_record_dependency_cycle(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_block_t* block_record, iree_host_size_t node_count,
    uint32_t scheduled_in_block, uint32_t range_start, uint32_t range_end) {
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
          loom_low_schedule_record_cycle_state_value(state);
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

  if (loom_low_schedule_record_source_range_cycle(
          state, block_record, node_count, scheduled_in_block, range_start,
          range_end)) {
    return iree_ok_status();
  }
  return loom_low_schedule_record_first_unresolved_dependency(
      state, block_record, node_count, scheduled_in_block);
}

static iree_status_t loom_low_schedule_handle_dependency_cycle(
    loom_low_schedule_build_state_t* state,
    const loom_low_schedule_block_t* block_record, iree_host_size_t node_count,
    uint32_t scheduled_in_block, uint32_t range_start, uint32_t range_end) {
  IREE_RETURN_IF_ERROR(loom_low_schedule_record_dependency_cycle(
      state, block_record, node_count, scheduled_in_block, range_start,
      range_end));
  if (state->options->emitter.fn != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_low_schedule_emit_dependency_cycle(state, &state->failure));
  }
  ++state->error_count;
  return iree_ok_status();
}

static uint32_t loom_low_schedule_add_signed_issue_separation(
    uint32_t producer_issue_cycle, int32_t minimum_separation_cycles) {
  if (minimum_separation_cycles >= 0) {
    return iree_math_saturating_add_u32(producer_issue_cycle,
                                        (uint32_t)minimum_separation_cycles);
  }
  const uint32_t magnitude = minimum_separation_cycles == INT32_MIN
                                 ? (uint32_t)INT32_MAX + 1u
                                 : (uint32_t)-minimum_separation_cycles;
  return producer_issue_cycle > magnitude ? producer_issue_cycle - magnitude
                                          : 0;
}

static void loom_low_schedule_note_issue_group(
    loom_low_schedule_build_state_t* state, uint32_t node_index) {
  loom_low_schedule_node_t* node = &state->nodes[node_index];
  if (state->issue_group_count != 0) {
    loom_low_schedule_issue_group_t* group =
        &state->issue_groups[state->issue_group_count - 1];
    if (group->block_index == node->block_index) {
      IREE_ASSERT_LE(group->issue_cycle, node->issue_cycle);
    }
    if (group->block_index == node->block_index &&
        group->issue_cycle == node->issue_cycle) {
      IREE_ASSERT_NE(group->scheduled_node_count, UINT32_MAX);
      ++group->scheduled_node_count;
      node->issue_group_ordinal = (uint32_t)state->issue_group_count - 1;
      return;
    }
  }
  IREE_ASSERT_LT(state->issue_group_count, UINT32_MAX);
  node->issue_group_ordinal = (uint32_t)state->issue_group_count;
  state->issue_groups[state->issue_group_count++] =
      (loom_low_schedule_issue_group_t){
          .block_index = node->block_index,
          .issue_cycle = node->issue_cycle,
          .scheduled_node_start = (uint32_t)state->scheduled_node_count,
          .scheduled_node_count = 1,
      };
}

static void loom_low_schedule_apply_candidate_descriptor(
    loom_low_schedule_build_state_t* state, loom_low_schedule_node_t* node,
    const loom_low_schedule_candidate_score_t* score) {
  if (state->options->strategy != LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL ||
      node->source_descriptor == NULL ||
      score->selected_descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return;
  }
  node->descriptor = &state->target.descriptor_set
                          ->descriptors[score->selected_descriptor_ordinal];
  const loom_low_descriptor_view_t* descriptor_view =
      loom_low_descriptor_set_descriptor_view_at(
          state->target.descriptor_set, score->selected_descriptor_ordinal);
  node->schedule_class =
      &state->target.descriptor_set
           ->schedule_classes[descriptor_view->schedule_class_id];
}

static iree_status_t loom_low_schedule_run_list_scheduler(
    loom_low_schedule_build_state_t* state, iree_host_size_t node_count) {
  uint32_t* indegrees = NULL;
  loom_low_schedule_pressure_state_t pressure_state = {0};
  loom_low_schedule_ready_policy_t ready_policy = {0};
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
  if (loom_low_schedule_strategy_uses_pressure(state->options->strategy)) {
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
  IREE_RETURN_IF_ERROR(loom_low_schedule_pressure_initialize(state, node_count,
                                                             &pressure_state));
  iree_arena_allocator_t dependency_scratch_arena;
  iree_arena_initialize(state->arena->block_pool, &dependency_scratch_arena);
  loom_low_schedule_dependency_detail_index_t dependency_details;
  iree_status_t dependency_index_status =
      loom_low_schedule_dependency_index_initialize(
          &state->dependencies, (uint32_t)node_count, &dependency_scratch_arena,
          state->arena, indegrees, &state->dependency_index,
          &dependency_details);
  if (iree_status_is_ok(dependency_index_status)) {
    loom_low_schedule_pressure_compute_node_priorities(
        state, node_count, &dependency_details, &pressure_state);
  }
  iree_arena_deinitialize(&dependency_scratch_arena);
  IREE_RETURN_IF_ERROR(dependency_index_status);
  if (loom_low_schedule_strategy_uses_pressure(state->options->strategy)) {
    IREE_RETURN_IF_ERROR(loom_low_schedule_pressure_initialize_unlock_summaries(
        state, (uint32_t)node_count, &pressure_state));
  }
  const uint8_t ready_view_count =
      loom_low_schedule_strategy_uses_pressure(state->options->strategy)
          ? LOOM_LOW_SCHEDULE_READY_VIEW_COUNT
          : 1;
  IREE_RETURN_IF_ERROR(loom_low_schedule_ready_policy_initialize(
      state, (uint32_t)node_count, ready_view_count, &ready_policy));

  for (iree_host_size_t block_index = 0; block_index < state->body->block_count;
       ++block_index) {
    loom_low_schedule_block_t* block_record = &state->blocks[block_index];
    const uint32_t block_node_end =
        block_record->node_start + block_record->node_count;
    block_record->scheduled_node_start = (uint32_t)state->scheduled_node_count;
    block_record->scheduled_node_count = 0;
    block_record->issue_group_start = (uint32_t)state->issue_group_count;
    block_record->issue_group_count = 0;
    state->liveness_block_orders[block_index] = (loom_liveness_block_order_t){
        .block = block_record->block,
        .ops = block_record->node_count != 0
                   ? &state->scheduled_ops[state->scheduled_node_count]
                   : NULL,
        .op_count = block_record->node_count,
    };
    state->current_block_index = block_index;
    state->current_issue_cycle = 0;
    state->pending_pair_affinity_node = LOOM_LOW_SCHEDULE_NODE_NONE;
    loom_low_schedule_resource_calendar_reset(&state->resource_calendar);
    if (loom_low_schedule_strategy_uses_pressure(state->options->strategy)) {
      loom_low_schedule_pressure_initialize_block(state, block_record,
                                                  &pressure_state);
      if (state->options->strategy ==
          LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
        loom_low_schedule_pressure_initialize_current_cliff_penalty(
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
      const uint32_t ready_candidate_count =
          loom_low_schedule_ready_frontier_count(&ready_policy.frontier);
      if (ready_candidate_count == 0) {
        return loom_low_schedule_handle_dependency_cycle(
            state, block_record, node_count, scheduled_in_block, range_start,
            range_end);
      }
      loom_low_schedule_candidate_selection_t selection;
      loom_low_schedule_candidate_policy_select(
          state, &pressure_state, &ready_policy, indegrees,
          ready_candidate_count, &selection);
      const uint32_t chosen_node = selection.chosen_node;
      loom_low_schedule_ready_policy_remove(state, &ready_policy, chosen_node);

      loom_low_schedule_node_t* chosen = &state->nodes[chosen_node];
      loom_low_schedule_apply_candidate_descriptor(state, chosen,
                                                   &selection.chosen_score);
      const bool is_storage_setup = iree_any_bit_set(
          chosen->flags, LOOM_LOW_SCHEDULE_NODE_FLAG_STORAGE_SETUP);
      uint32_t issue_cycle = state->current_issue_cycle;
      if (state->options->strategy ==
          LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL) {
        issue_cycle = iree_math_saturating_add_u32(
            issue_cycle, selection.chosen_score.effective_stall_cycles);
      } else if (!is_storage_setup) {
        issue_cycle =
            iree_max(issue_cycle, state->node_ready_issue_cycles[chosen_node]);
      }
      state->current_issue_cycle = issue_cycle;
      chosen->scheduled_ordinal = scheduled_in_block++;
      chosen->issue_cycle = issue_cycle;
      loom_low_schedule_note_issue_group(state, chosen_node);
      block_record->issue_group_count =
          (uint32_t)state->issue_group_count - block_record->issue_group_start;
      ++scheduled_in_range;
      state->scheduled_node_indices[state->scheduled_node_count] = chosen_node;
      state->scheduled_ops[state->scheduled_node_count] = chosen->op;
      ++state->scheduled_node_count;
      ++block_record->scheduled_node_count;
      loom_low_schedule_ready_policy_note_node_scheduled(state, chosen_node);
      if (loom_low_schedule_strategy_uses_pressure(state->options->strategy)) {
        loom_low_schedule_candidate_policy_record_decision(
            state, block_index, state->nodes[chosen_node].scheduled_ordinal,
            &selection);
        loom_low_schedule_pressure_note_node_scheduled(
            state, &pressure_state, chosen_node, &selection.chosen_score);
        loom_low_schedule_pressure_update_ready_consumers(
            state, &pressure_state, &ready_policy, chosen_node);
      }
      IREE_RETURN_IF_ERROR(
          loom_low_schedule_note_descriptor_rows_for_node(state, chosen_node));

      const uint32_t producer_available_issue_cycle =
          iree_max(issue_cycle, state->node_ready_issue_cycles[chosen_node]);
      uint32_t completion_ready_issue_cycle = producer_available_issue_cycle;
      bool has_wait_counter_hazard = false;
      uint16_t completion_wait_cycles = 0;
      if (state->node_completion_wait_cycles != NULL) {
        const loom_low_schedule_class_t* schedule_class =
            state->nodes[chosen_node].schedule_class;
        has_wait_counter_hazard = loom_low_schedule_class_query_completion_wait(
            state->target.descriptor_set, schedule_class,
            &completion_wait_cycles);
        completion_ready_issue_cycle = iree_math_saturating_add_u32(
            producer_available_issue_cycle,
            schedule_class != NULL ? schedule_class->latency_cycles : 0);
      }
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
        if (loom_low_schedule_strategy_uses_pressure(
                state->options->strategy)) {
          const uint32_t remaining_producer =
              loom_low_schedule_dependency_frontier_consume_group(
                  &pressure_state.unlocks.frontier, chosen_node, group);
          if (remaining_producer != LOOM_LOW_SCHEDULE_DEPENDENCY_GROUP_NONE) {
            loom_low_schedule_pressure_publish_unlock_consumer(
                state, &pressure_state, remaining_producer, consumer_node);
          }
        }
        uint32_t consumer_ready_issue_cycle =
            loom_low_schedule_add_signed_issue_separation(
                producer_available_issue_cycle,
                group->minimum_issue_separation_cycles);
        if (state->node_completion_wait_cycles != NULL) {
          // Memory effects only require completion latency when the target
          // identifies the producer as counter-tracked. Other effect edges
          // order issue but do not imply a completion wait.
          if (has_wait_counter_hazard &&
              loom_low_schedule_dependency_index_group_has_effect(
                  &state->dependency_index, group_index)) {
            consumer_ready_issue_cycle = iree_max(
                consumer_ready_issue_cycle, completion_ready_issue_cycle);
            state->node_completion_wait_cycles[consumer_node] =
                iree_max(state->node_completion_wait_cycles[consumer_node],
                         completion_wait_cycles);
          }
        }
        if (consumer_ready_issue_cycle >
            state->node_ready_issue_cycles[consumer_node]) {
          state->node_ready_issue_cycles[consumer_node] =
              consumer_ready_issue_cycle;
        }
        if (indegrees[consumer_node] == 0 && consumer_node >= range_start &&
            consumer_node < range_end) {
          loom_low_schedule_insert_ready_node(state, &pressure_state,
                                              &ready_policy, consumer_node);
        }
      }
      if (scheduled_in_block < block_record->node_count &&
          state->options->strategy !=
              LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL &&
          !loom_low_schedule_node_has_zero_issue_width(chosen)) {
        if (state->current_issue_cycle == UINT32_MAX) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "low schedule issue cycle overflows");
        }
        ++state->current_issue_cycle;
      }
      if (scheduled_in_range == range_end - range_start) {
        range_start = range_end;
        if (range_start < block_node_end) {
          if (state->options->strategy ==
                  LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL &&
              !loom_low_schedule_node_has_zero_issue_width(chosen)) {
            if (state->current_issue_cycle == UINT32_MAX) {
              return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                      "low schedule issue cycle overflows");
            }
            ++state->current_issue_cycle;
          }
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
  if (model->error_count != 0) return iree_ok_status();
  IREE_ASSERT(loom_local_value_domain_is_acquired(&model->value_domain));

  loom_low_schedule_build_state_t state = {
      .module = model->module,
      .options = options,
      .pressure_cliffs = options->residency_model != NULL
                             ? &options->residency_model->direct_resources
                             : NULL,
      .pressure_resources =
          options->residency_model != NULL &&
                  !loom_target_residency_derived_resource_table_is_empty(
                      &options->residency_model->derived_resources)
              ? &options->residency_model->derived_resources
              : NULL,
      .arena = arena,
      .function_op = model->function_op,
      .body = model->body,
      .target = model->target,
      .value_domain = &model->value_domain,
      .cfg_graph = &model->cfg_graph,
  };
  loom_low_schedule_dependency_graph_initialize(&state.dependencies);
  loom_low_storage_layout_builder_initialize(&state.storage_layout_builder);
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
    loom_low_storage_layout_t storage_layout;
    loom_low_storage_layout_builder_finish(&state.storage_layout_builder,
                                           &storage_layout);
    *out_table = (loom_low_schedule_table_t){
        .module = model->module,
        .function_op = model->function_op,
        .target = state.target,
        .memory_access_table = options->memory_access_table,
        .storage_layout = storage_layout,
        .value_ids = model->value_domain.value_ids,
        .value_count = model->value_domain.value_count,
        .liveness = liveness,
        .blocks = state.blocks,
        .block_count = state.body->block_count,
        .operation_order =
            {
                .blocks = state.liveness_block_orders,
                .block_count = state.body->block_count,
            },
        .cfg_graph = model->cfg_graph,
        .loop_forest = model->loop_forest,
        .nodes = state.nodes,
        .node_count = node_count,
        .dependency_group_count = state.dependency_index.group_count,
        .unlock_summary_publication_count =
            state.unlock_summary_publication_count,
        .scheduled_node_indices = state.scheduled_node_indices,
        .scheduled_node_count = state.scheduled_node_count,
        .issue_groups = state.issue_groups,
        .issue_group_count = state.issue_group_count,
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
        .matrix_coexecution_source_use_count =
            state.matrix_coexecution_source_use_count,
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
  }
  return status;
}
