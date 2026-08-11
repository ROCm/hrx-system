// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation_live_range_splitting.h"

#include "loom/codegen/low/allocation/live_range.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/error/error_catalog.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/registers.h"

static loom_low_allocation_live_range_split_result_t
loom_low_allocation_live_range_split_result_empty(void) {
  return (loom_low_allocation_live_range_split_result_t){
      .source_value_id = LOOM_VALUE_ID_INVALID,
      .split_value_id = LOOM_VALUE_ID_INVALID,
      .source_assignment_index = UINT32_MAX,
  };
}

typedef struct loom_low_allocation_pair_replication_repair_t {
  // Operation operand rewritten when this repair is committed.
  loom_op_t* user_op;
  // Next repair index + 1 for the same source value, or zero.
  uint32_t next_repair_index_plus_one;
  // Operand index within |user_op|.
  uint16_t operand_index;
} loom_low_allocation_pair_replication_repair_t;

typedef struct loom_low_allocation_pair_replication_candidate_t {
  // Source value requiring an independently placed replica.
  loom_value_id_t source_value_id;
  // First repair index + 1 in the candidate repair list, or zero.
  uint32_t first_repair_index_plus_one;
  // Conservative native packet cost of copying the source register value.
  uint32_t copy_packet_count;
  // Sum of target-declared packet savings across repaired pairs.
  uint64_t gross_packet_savings;
} loom_low_allocation_pair_replication_candidate_t;

static bool loom_low_allocation_value_is_reference_register(
    const loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_value_id_t value_id) {
  const loom_low_register_type_resolver_t resolver =
      loom_low_register_type_resolver_for_descriptor_set(
          table->target.descriptor_set);
  return loom_low_register_type_resolver_has_class_flags(
      &resolver, loom_module_value_type(module, value_id),
      LOOM_LOW_REG_CLASS_FLAG_REFERENCE);
}

static bool loom_low_allocation_fixed_value_overlaps_spill_assignment(
    const loom_low_allocation_resolved_fixed_value_t* fixed_value,
    const loom_low_allocation_assignment_t* spill_assignment) {
  return loom_liveness_value_class_equal(fixed_value->interval->value_class,
                                         spill_assignment->value_class) &&
         loom_low_allocation_live_range_assignment_overlaps_interval(
             spill_assignment, fixed_value->interval);
}

static bool loom_low_allocation_fixed_value_has_only_split_transfer_use(
    const loom_value_t* value) {
  if (value->use_count != 1) {
    return false;
  }
  const loom_use_t use = loom_value_uses(value)[0];
  const loom_op_t* user_op = loom_use_user_op(use);
  return (loom_low_copy_isa(user_op) || loom_low_move_isa(user_op)) &&
         loom_use_operand_index(use) == 0;
}

static bool loom_low_allocation_split_use_is_eligible(
    loom_value_id_t value_id, const loom_block_t* insertion_block,
    const loom_op_t* insertion_anchor, loom_use_t use) {
  loom_op_t* user_op = loom_use_user_op(use);
  const uint16_t operand_index = loom_use_operand_index(use);
  if (user_op == NULL || iree_any_bit_set(user_op->flags, LOOM_OP_FLAG_DEAD) ||
      user_op->parent_block == NULL ||
      operand_index >= user_op->operand_count) {
    return false;
  }
  if (loom_op_operands(user_op)[operand_index] != value_id) {
    return false;
  }
  if (user_op->parent_block == insertion_block && insertion_anchor != NULL &&
      user_op->block_ordinal <= insertion_anchor->block_ordinal) {
    return false;
  }
  return true;
}

static bool loom_low_allocation_value_can_split_after_definition(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_block_t** out_insertion_block, loom_op_t** out_insertion_anchor) {
  *out_insertion_block = NULL;
  *out_insertion_anchor = NULL;
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    loom_block_t* defining_block = loom_value_def_block(value);
    if (defining_block == NULL) {
      return false;
    }
    *out_insertion_block = defining_block;
    return true;
  }
  loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL ||
      iree_any_bit_set(defining_op->flags, LOOM_OP_FLAG_DEAD) ||
      defining_op->parent_block == NULL) {
    return false;
  }
  *out_insertion_block = defining_op->parent_block;
  *out_insertion_anchor = defining_op;
  return true;
}

static bool loom_low_allocation_pair_value_ref_equal(
    const loom_low_placement_pair_value_ref_t* lhs,
    const loom_low_placement_pair_value_ref_t* rhs) {
  return lhs->component == rhs->component && lhs->kind == rhs->kind &&
         lhs->index == rhs->index;
}

static loom_op_t* loom_low_allocation_pair_component_op(
    const loom_low_placement_pair_use_t* use,
    loom_low_placement_pair_component_t component) {
  switch (component) {
    case LOOM_LOW_PLACEMENT_PAIR_COMPONENT_FIRST:
      return (loom_op_t*)use->first_op;
    case LOOM_LOW_PLACEMENT_PAIR_COMPONENT_SECOND:
      return (loom_op_t*)use->second_op;
    default:
      IREE_ASSERT_UNREACHABLE("unknown low placement pair component");
      return NULL;
  }
}

static bool loom_low_allocation_pair_relation_is_satisfied(
    const loom_low_allocation_table_t* table,
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_pair_relation_t* relation) {
  const loom_value_id_t result_value_id =
      loom_low_placement_pair_value_id(use, &relation->result);
  const loom_value_id_t source_value_id =
      loom_low_placement_pair_value_id(use, &relation->source);
  const loom_low_allocation_assignment_t* result_assignment =
      loom_low_allocation_try_map_active_value_assignment(
          table, result_value_id, /*out_assignment_index=*/NULL);
  const loom_low_allocation_assignment_t* source_assignment =
      loom_low_allocation_try_map_active_value_assignment(
          table, source_value_id, /*out_assignment_index=*/NULL);
  if (result_assignment == NULL || source_assignment == NULL) {
    return false;
  }
  IREE_ASSERT_LE(relation->result.unit_offset, result_assignment->unit_count);
  IREE_ASSERT_LE(relation->unit_count,
                 result_assignment->unit_count - relation->result.unit_offset);
  IREE_ASSERT_LE(relation->source.unit_offset, source_assignment->unit_count);
  IREE_ASSERT_LE(relation->unit_count,
                 source_assignment->unit_count - relation->source.unit_offset);
  const loom_low_placement_relation_t concrete_relation = {
      .result_unit_offset = relation->result.unit_offset,
      .source_unit_offset = relation->source.unit_offset,
      .unit_count = relation->unit_count,
      .location_mask = relation->location_mask,
      .kind = relation->kind,
  };
  return loom_low_allocation_storage_placement_relation_satisfied(
      table->target.descriptor_set, &concrete_relation, result_assignment,
      source_assignment);
}

static bool loom_low_allocation_pair_alternative_is_satisfied(
    const loom_low_allocation_table_t* table,
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_pair_relation_t* relations,
    uint16_t relation_count) {
  for (uint16_t i = 0; i < relation_count; ++i) {
    if (!loom_low_allocation_pair_relation_is_satisfied(table, use,
                                                        &relations[i])) {
      return false;
    }
  }
  return true;
}

static bool loom_low_allocation_pair_use_is_satisfied(
    const loom_low_allocation_table_t* table,
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_pair_recipe_t* recipe) {
  for (uint16_t alternative_index = 0;
       alternative_index < recipe->alternative_count; ++alternative_index) {
    const loom_low_placement_pair_relation_t* relations =
        &recipe->relations[alternative_index * recipe->relation_count];
    if (loom_low_allocation_pair_alternative_is_satisfied(
            table, use, relations, recipe->relation_count)) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_low_allocation_satisfied_pair_packet_savings(
    const loom_low_allocation_table_t* table,
    loom_low_placement_pair_use_list_t pair_uses,
    uint64_t* out_packet_savings) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(out_packet_savings);
  *out_packet_savings = 0;
  if (pair_uses.count == 0) {
    return iree_ok_status();
  }
  loom_low_allocation_value_scratch_t value_scratch = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_acquire_value_scratch(table, &value_scratch));
  uint64_t packet_savings = 0;
  for (iree_host_size_t i = 0; i < pair_uses.count; ++i) {
    const loom_low_placement_pair_use_t* use = &pair_uses.values[i];
    if (use->placement_recipe_index == LOOM_LOW_PLACEMENT_PAIR_RECIPE_NONE) {
      continue;
    }
    const uint16_t recipe_index = (uint16_t)(use->placement_recipe_index - 1u);
    IREE_ASSERT_LT(recipe_index, pair_uses.placement_recipe_count);
    const loom_low_placement_pair_recipe_t* recipe =
        &pair_uses.placement_recipes[recipe_index];
    if (loom_low_allocation_pair_use_is_satisfied(table, use, recipe)) {
      packet_savings += recipe->packet_savings;
    }
  }
  loom_low_allocation_release_value_scratch(&value_scratch);
  *out_packet_savings = packet_savings;
  return iree_ok_status();
}

static bool loom_low_allocation_pair_try_select_replica_operand(
    const loom_low_allocation_table_t* table,
    const loom_low_placement_pair_use_t* use,
    const loom_low_placement_pair_recipe_t* recipe, loom_op_t** out_user_op,
    uint16_t* out_operand_index, loom_value_id_t* out_source_value_id) {
  *out_user_op = NULL;
  *out_operand_index = 0;
  *out_source_value_id = LOOM_VALUE_ID_INVALID;

  const loom_low_placement_pair_component_t preferred_components[] = {
      LOOM_LOW_PLACEMENT_PAIR_COMPONENT_SECOND,
      LOOM_LOW_PLACEMENT_PAIR_COMPONENT_FIRST,
  };
  for (uint16_t alternative_index = 0;
       alternative_index < recipe->alternative_count; ++alternative_index) {
    const loom_low_placement_pair_relation_t* relations =
        &recipe->relations[alternative_index * recipe->relation_count];
    for (iree_host_size_t component_index = 0;
         component_index < IREE_ARRAYSIZE(preferred_components);
         ++component_index) {
      const loom_low_placement_pair_component_t preferred_component =
          preferred_components[component_index];
      for (uint16_t relation_index = 0; relation_index < recipe->relation_count;
           ++relation_index) {
        const loom_low_placement_pair_relation_t* relation =
            &relations[relation_index];
        if (relation->kind !=
                LOOM_LOW_PLACEMENT_RELATION_DIFFERENT_MASKED_LOCATION &&
            relation->kind != LOOM_LOW_PLACEMENT_RELATION_DISJOINT_STORAGE) {
          continue;
        }
        const loom_value_id_t result_value_id =
            loom_low_placement_pair_value_id(use, &relation->result);
        const loom_value_id_t source_value_id =
            loom_low_placement_pair_value_id(use, &relation->source);
        if (result_value_id != source_value_id ||
            loom_low_allocation_pair_relation_is_satisfied(table, use,
                                                           relation)) {
          continue;
        }
        const loom_low_placement_pair_value_ref_t* refs[] = {
            &relation->result,
            &relation->source,
        };
        for (iree_host_size_t ref_index = 0; ref_index < IREE_ARRAYSIZE(refs);
             ++ref_index) {
          const loom_low_placement_pair_value_ref_t* repair_ref =
              refs[ref_index];
          if (repair_ref->component != preferred_component ||
              repair_ref->kind != LOOM_LOW_PLACEMENT_PAIR_VALUE_OPERAND ||
              loom_low_allocation_pair_value_ref_equal(&relation->result,
                                                       &relation->source) ||
              !loom_low_placement_pair_alternative_can_separate_ref(
                  use, relations, recipe->relation_count, repair_ref)) {
            continue;
          }
          loom_op_t* user_op =
              loom_low_allocation_pair_component_op(use, repair_ref->component);
          IREE_ASSERT(user_op != NULL);
          IREE_ASSERT_LT(repair_ref->index, user_op->operand_count);
          *out_user_op = user_op;
          *out_operand_index = repair_ref->index;
          *out_source_value_id = result_value_id;
          return true;
        }
      }
    }
  }
  return false;
}

static bool loom_low_allocation_pair_replication_source_is_eligible(
    const loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_value_id_t source_value_id) {
  const loom_value_t* source_value = loom_module_value(module, source_value_id);
  IREE_ASSERT_NE(source_value->use_count, 0);
  return !loom_value_is_consumed(source_value) &&
         !loom_low_allocation_value_is_reference_register(module, table,
                                                          source_value_id);
}

static iree_status_t loom_low_allocation_pair_replication_rollback_edits(
    loom_rewriter_t* rewriter,
    const loom_low_allocation_pair_replication_result_t* result) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = result->edit_count;
       i > 0 && iree_status_is_ok(status); --i) {
    const loom_low_allocation_pair_replication_edit_t* edit =
        &result->edits[i - 1];
    if (edit->copy_op == NULL ||
        iree_any_bit_set(edit->copy_op->flags, LOOM_OP_FLAG_DEAD)) {
      continue;
    }
    status = loom_rewriter_replace_all_uses_with(
        rewriter, edit->replica_value_id, edit->source_value_id);
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_erase(rewriter, edit->copy_op);
    }
  }
  return status;
}

iree_status_t loom_low_allocation_replicate_pair_sources(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_low_placement_pair_use_list_t pair_uses, iree_arena_allocator_t* arena,
    loom_low_allocation_pair_replication_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_result);
  *out_result = (loom_low_allocation_pair_replication_result_t){0};
  if (pair_uses.count == 0 || pair_uses.placement_recipe_count == 0 ||
      table->error_count != 0 || table->spill_plan_count != 0 ||
      table->spill_count != 0) {
    return iree_ok_status();
  }
  IREE_ASSERT_LE(pair_uses.count, UINT32_MAX);

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(arena->block_pool, &scratch_arena);
  loom_low_allocation_pair_replication_candidate_t* candidates = NULL;
  loom_low_allocation_pair_replication_repair_t* repairs = NULL;
  uint32_t* candidate_indices_by_value_ordinal = NULL;
  iree_status_t status = iree_ok_status();
  loom_low_allocation_value_scratch_t value_scratch = {0};
  status = loom_low_allocation_acquire_value_scratch(table, &value_scratch);
  uint32_t candidate_count = 0;
  uint32_t repair_count = 0;
  for (iree_host_size_t i = 0; i < pair_uses.count && iree_status_is_ok(status);
       ++i) {
    const loom_low_placement_pair_use_t* use = &pair_uses.values[i];
    if (use->placement_recipe_index == LOOM_LOW_PLACEMENT_PAIR_RECIPE_NONE) {
      continue;
    }
    const uint16_t recipe_index = (uint16_t)(use->placement_recipe_index - 1u);
    IREE_ASSERT_LT(recipe_index, pair_uses.placement_recipe_count);
    const loom_low_placement_pair_recipe_t* recipe =
        &pair_uses.placement_recipes[recipe_index];
    if (loom_low_allocation_pair_use_is_satisfied(table, use, recipe)) {
      out_result->baseline_satisfied_packet_savings += recipe->packet_savings;
      continue;
    }
    loom_op_t* user_op = NULL;
    uint16_t operand_index = 0;
    loom_value_id_t source_value_id = LOOM_VALUE_ID_INVALID;
    if (!loom_low_allocation_pair_try_select_replica_operand(
            table, use, recipe, &user_op, &operand_index, &source_value_id) ||
        !loom_low_allocation_pair_replication_source_is_eligible(
            module, table, source_value_id)) {
      continue;
    }
    if (candidates == NULL) {
      status =
          iree_arena_allocate_array(&scratch_arena, pair_uses.count,
                                    sizeof(*candidates), (void**)&candidates);
      if (iree_status_is_ok(status)) {
        memset(candidates, 0, pair_uses.count * sizeof(*candidates));
        status = iree_arena_allocate_array(&scratch_arena, pair_uses.count,
                                           sizeof(*repairs), (void**)&repairs);
      }
      if (iree_status_is_ok(status)) {
        IREE_ASSERT_NE(table->liveness.value_count, 0);
        status = iree_arena_allocate_array(
            &scratch_arena, table->liveness.value_count,
            sizeof(*candidate_indices_by_value_ordinal),
            (void**)&candidate_indices_by_value_ordinal);
      }
      if (iree_status_is_ok(status)) {
        memset(candidate_indices_by_value_ordinal, 0,
               table->liveness.value_count *
                   sizeof(*candidate_indices_by_value_ordinal));
      } else {
        break;
      }
    }
    const loom_value_ordinal_t value_ordinal =
        loom_module_value_ordinal_scratch_lookup(module, source_value_id);
    IREE_ASSERT_NE(value_ordinal, LOOM_VALUE_ORDINAL_INVALID);
    uint32_t candidate_index_plus_one =
        candidate_indices_by_value_ordinal[value_ordinal];
    if (candidate_index_plus_one == 0) {
      IREE_ASSERT_LT(candidate_count, pair_uses.count);
      const loom_type_t source_type =
          loom_module_value_type(module, source_value_id);
      IREE_ASSERT(loom_low_type_is_register(source_type));
      const uint32_t copy_packet_count =
          loom_low_register_type_unit_count(source_type);
      IREE_ASSERT_NE(copy_packet_count, 0);
      candidate_index_plus_one = ++candidate_count;
      candidate_indices_by_value_ordinal[value_ordinal] =
          candidate_index_plus_one;
      candidates[candidate_index_plus_one - 1] =
          (loom_low_allocation_pair_replication_candidate_t){
              .source_value_id = source_value_id,
              .copy_packet_count = copy_packet_count,
          };
    }
    loom_low_allocation_pair_replication_candidate_t* candidate =
        &candidates[candidate_index_plus_one - 1];
    candidate->gross_packet_savings += recipe->packet_savings;
    IREE_ASSERT_LT(repair_count, pair_uses.count);
    repairs[repair_count] = (loom_low_allocation_pair_replication_repair_t){
        .user_op = user_op,
        .next_repair_index_plus_one = candidate->first_repair_index_plus_one,
        .operand_index = operand_index,
    };
    candidate->first_repair_index_plus_one = ++repair_count;
  }
  loom_low_allocation_release_value_scratch(&value_scratch);

  iree_host_size_t profitable_candidate_count = 0;
  if (iree_status_is_ok(status)) {
    for (uint32_t i = 0; i < candidate_count; ++i) {
      const loom_low_allocation_pair_replication_candidate_t* candidate =
          &candidates[i];
      if (candidate->gross_packet_savings > candidate->copy_packet_count) {
        ++profitable_candidate_count;
      }
    }
  }
  if (iree_status_is_ok(status) && profitable_candidate_count != 0) {
    status = iree_arena_allocate_array(arena, profitable_candidate_count,
                                       sizeof(*out_result->edits),
                                       (void**)&out_result->edits);
  }

  loom_rewriter_t rewriter = {0};
  if (iree_status_is_ok(status) && profitable_candidate_count != 0) {
    status = loom_rewriter_initialize(&rewriter, module, &scratch_arena);
  }
  for (uint32_t i = 0; i < candidate_count && iree_status_is_ok(status); ++i) {
    const loom_low_allocation_pair_replication_candidate_t* candidate =
        &candidates[i];
    if (candidate->gross_packet_savings <= candidate->copy_packet_count) {
      continue;
    }
    loom_block_t* insertion_block = NULL;
    loom_op_t* insertion_anchor = NULL;
    if (!loom_low_allocation_value_can_split_after_definition(
            module, candidate->source_value_id, &insertion_block,
            &insertion_anchor)) {
      continue;
    }
    const loom_builder_ip_t saved_ip = loom_builder_save(&rewriter.builder);
    if (insertion_anchor != NULL) {
      loom_builder_set_after(&rewriter.builder, insertion_anchor);
    } else if (insertion_block->first_op != NULL) {
      loom_builder_set_before(&rewriter.builder, insertion_block->first_op);
    } else {
      loom_builder_set_block(&rewriter.builder, insertion_block);
    }
    loom_op_t* copy_op = NULL;
    status = loom_low_copy_build(
        &rewriter.builder, candidate->source_value_id, true,
        loom_module_value_type(module, candidate->source_value_id),
        LOOM_LOCATION_NONE, &copy_op);
    loom_builder_restore(&rewriter.builder, saved_ip);
    if (!iree_status_is_ok(status)) {
      break;
    }
    const loom_value_id_t replica_value_id = loom_low_copy_result(copy_op);
    out_result->edits[out_result->edit_count++] =
        (loom_low_allocation_pair_replication_edit_t){
            .copy_op = copy_op,
            .source_value_id = candidate->source_value_id,
            .replica_value_id = replica_value_id,
        };
    status = loom_rewriter_try_set_derived_value_name(
        &rewriter, candidate->source_value_id, replica_value_id,
        IREE_SV("replica"));
    for (uint32_t repair_index_plus_one =
             candidate->first_repair_index_plus_one;
         repair_index_plus_one != 0 && iree_status_is_ok(status);) {
      const loom_low_allocation_pair_replication_repair_t* repair =
          &repairs[repair_index_plus_one - 1];
      IREE_ASSERT(repair->user_op != NULL);
      IREE_ASSERT_LT(repair->operand_index, repair->user_op->operand_count);
      IREE_ASSERT_EQ(loom_op_operands(repair->user_op)[repair->operand_index],
                     candidate->source_value_id);
      status = loom_rewriter_set_operand(
          &rewriter, repair->user_op, repair->operand_index, replica_value_id);
      repair_index_plus_one = repair->next_repair_index_plus_one;
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
  }
  if (!iree_status_is_ok(status) && out_result->edit_count != 0) {
    status = iree_status_join(
        status, loom_low_allocation_pair_replication_rollback_edits(
                    &rewriter, out_result));
    *out_result = (loom_low_allocation_pair_replication_result_t){0};
  }
  loom_rewriter_deinitialize(&rewriter);
  iree_arena_deinitialize(&scratch_arena);
  return status;
}

iree_status_t loom_low_allocation_rollback_pair_replication(
    loom_module_t* module,
    const loom_low_allocation_pair_replication_result_t* result,
    iree_arena_allocator_t* arena) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(result);
  IREE_ASSERT_ARGUMENT(arena);
  if (result->edit_count == 0) {
    return iree_ok_status();
  }
  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(loom_rewriter_initialize(&rewriter, module, arena));
  const iree_status_t status =
      loom_low_allocation_pair_replication_rollback_edits(&rewriter, result);
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

static iree_status_t loom_low_allocation_try_split_fixed_value(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_allocation_resolved_fixed_value_t* fixed_value,
    uint32_t source_assignment_index, iree_arena_allocator_t* arena,
    loom_low_allocation_live_range_split_result_t* result) {
  if (fixed_value->value_id == LOOM_VALUE_ID_INVALID ||
      fixed_value->value_id >= module->values.count) {
    return iree_ok_status();
  }

  const loom_value_t* value = loom_module_value(module, fixed_value->value_id);
  if (loom_value_is_consumed(value) ||
      loom_module_value_has_predicate_attribute_uses(module,
                                                     fixed_value->value_id) ||
      loom_module_value_has_type_uses(module, fixed_value->value_id) ||
      value->use_count == 0 ||
      loom_low_allocation_fixed_value_has_only_split_transfer_use(value)) {
    return iree_ok_status();
  }

  loom_block_t* insertion_block = NULL;
  loom_op_t* insertion_anchor = NULL;
  if (!loom_low_allocation_value_can_split_after_definition(
          module, fixed_value->value_id, &insertion_block, &insertion_anchor)) {
    return iree_ok_status();
  }

  const uint32_t original_use_count = value->use_count;
  const loom_use_t* original_uses = loom_value_uses(value);
  for (uint32_t i = 0; i < original_use_count; ++i) {
    if (!loom_low_allocation_split_use_is_eligible(
            fixed_value->value_id, insertion_block, insertion_anchor,
            original_uses[i])) {
      return iree_ok_status();
    }
  }

  loom_rewriter_t rewriter = {0};
  IREE_RETURN_IF_ERROR(loom_rewriter_initialize(&rewriter, module, arena));
  loom_builder_ip_t saved_ip = loom_builder_save(&rewriter.builder);
  if (insertion_anchor != NULL) {
    loom_builder_set_after(&rewriter.builder, insertion_anchor);
  } else if (insertion_block->first_op != NULL) {
    loom_builder_set_before(&rewriter.builder, insertion_block->first_op);
  } else {
    loom_builder_set_block(&rewriter.builder, insertion_block);
  }

  loom_op_t* transfer_op = NULL;
  const loom_type_t value_type =
      loom_module_value_type(module, fixed_value->value_id);
  iree_status_t status =
      loom_low_allocation_value_is_reference_register(module, table,
                                                      fixed_value->value_id)
          ? loom_low_move_build(&rewriter.builder, fixed_value->value_id, true,
                                value_type, LOOM_LOCATION_NONE, &transfer_op)
          : loom_low_copy_build(&rewriter.builder, fixed_value->value_id, true,
                                value_type, LOOM_LOCATION_NONE, &transfer_op);
  loom_builder_restore(&rewriter.builder, saved_ip);
  if (iree_status_is_ok(status)) {
    const loom_value_id_t split_value_id = loom_op_results(transfer_op)[0];
    status = loom_rewriter_try_set_derived_value_name(
        &rewriter, fixed_value->value_id, split_value_id, IREE_SV("split"));
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_replace_all_uses_except(
          &rewriter, fixed_value->value_id, split_value_id, transfer_op);
    }
    if (iree_status_is_ok(status)) {
      result->source_value_id = fixed_value->value_id;
      result->split_value_id = split_value_id;
      result->source_assignment_index = source_assignment_index;
      result->transfer_packet_count = 1;
      result->rewritten_operand_count = original_use_count;
    }
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

iree_status_t loom_low_allocation_split_fixed_value_spill_plan(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    iree_arena_allocator_t* arena,
    loom_low_allocation_live_range_split_result_t* out_result) {
  *out_result = loom_low_allocation_live_range_split_result_empty();
  if (table->spill_plan_count == 0 || table->fixed_value_count == 0) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < table->spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* spill_plan = &table->spill_plans[i];
    if (spill_plan->assignment_index >= table->assignment_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low allocation spill plan references an invalid assignment");
    }
    const loom_low_allocation_assignment_t* spill_assignment =
        &table->assignments[spill_plan->assignment_index];
    for (iree_host_size_t j = 0; j < table->fixed_value_count; ++j) {
      const loom_low_allocation_resolved_fixed_value_t* fixed_value =
          &table->fixed_values[j];
      if (!loom_low_allocation_fixed_value_overlaps_spill_assignment(
              fixed_value, spill_assignment)) {
        continue;
      }
      uint32_t source_assignment_index = UINT32_MAX;
      loom_low_allocation_assignment_for_value_ordinal(
          table, fixed_value->value_ordinal, &source_assignment_index);
      IREE_RETURN_IF_ERROR(loom_low_allocation_try_split_fixed_value(
          module, table, fixed_value, source_assignment_index, arena,
          out_result));
      if (out_result->rewritten_operand_count != 0) {
        return iree_ok_status();
      }
    }
  }
  return iree_ok_status();
}

static iree_string_view_t loom_low_allocation_live_range_split_trigger_name(
    loom_low_allocation_live_range_split_trigger_t trigger) {
  switch (trigger) {
    case LOOM_LOW_ALLOCATION_LIVE_RANGE_SPLIT_TRIGGER_SPILL_PLAN:
      return IREE_SV("spill-plan");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_low_allocation_live_range_split_value_class_name(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_live_range_split_result_t* result) {
  if (result->source_assignment_index < table->assignment_count) {
    const loom_low_allocation_assignment_t* assignment =
        &table->assignments[result->source_assignment_index];
    return loom_low_diagnostic_value_class_name(table->target.descriptor_set,
                                                assignment->value_class);
  }
  for (iree_host_size_t i = 0; i < table->fixed_value_count; ++i) {
    const loom_low_allocation_resolved_fixed_value_t* fixed_value =
        &table->fixed_values[i];
    if (fixed_value->value_id == result->source_value_id) {
      return loom_low_diagnostic_value_class_name(
          table->target.descriptor_set, fixed_value->interval->value_class);
    }
  }
  return IREE_SV("<unknown>");
}

iree_status_t loom_low_allocation_live_range_split_emit_decision(
    const loom_low_allocation_table_t* table,
    loom_low_allocation_live_range_split_trigger_t trigger,
    const loom_low_allocation_live_range_split_result_t* result,
    iree_diagnostic_emitter_t emitter) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(result);
  if (emitter.fn == NULL || result->rewritten_operand_count == 0) {
    return iree_ok_status();
  }
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&table->target)),
      loom_param_string(loom_low_diagnostic_export_name(&table->target)),
      loom_param_string(loom_low_diagnostic_config_key(&table->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(table->module, table->function_op)),
      loom_param_string(loom_low_diagnostic_value_name(
          table->module, result->source_value_id)),
      loom_param_string(loom_low_diagnostic_value_name(table->module,
                                                       result->split_value_id)),
      loom_param_string(
          loom_low_allocation_live_range_split_value_class_name(table, result)),
      loom_param_string(
          loom_low_allocation_live_range_split_trigger_name(trigger)),
      loom_param_u32(result->transfer_packet_count),
      loom_param_u32(result->rewritten_operand_count),
      loom_param_string(IREE_SV("fixed-value-spill-plan")),
  };
  const loom_diagnostic_emission_t emission = {
      .op = table->function_op,
      .error = LOOM_ERR_BACKEND_046,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}
