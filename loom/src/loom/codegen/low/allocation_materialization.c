// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation_materialization.h"

#include <stdio.h>
#include <string.h>

#include "loom/codegen/low/allocation/spill_plan.h"
#include "loom/codegen/low/allocation/storage.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/low/storage.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/type_builders.h"
#include "loom/ops/type_registry.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/registers.h"

typedef struct loom_low_materialized_spill_slot_t {
  // SSA value ID produced by the generated low.storage.reserve op.
  loom_value_id_t storage_value_id;
} loom_low_materialized_spill_slot_t;

typedef struct loom_low_materialized_traffic_t {
  // Number of logical spill transfers materialized.
  uint32_t count;
  // Byte traffic represented by the materialized transfers.
  uint64_t bytes;
} loom_low_materialized_traffic_t;

typedef struct loom_low_materialized_value_list_builder_t {
  // Materialized values in insertion order.
  loom_value_id_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
  // Capacity of |values|.
  iree_host_size_t capacity;
} loom_low_materialized_value_list_builder_t;

typedef struct loom_low_materialized_value_facts_t {
  // Values that generated spill traffic requires in a physical register.
  loom_low_materialized_value_list_builder_t required_register_values;
  // Reload results private to one consumer setup chain.
  loom_low_materialized_value_list_builder_t per_user_placement_values;
  // Values defined outside their surrounding semantic execution domain.
  loom_low_materialized_value_list_builder_t expanded_execution_values;
} loom_low_materialized_value_facts_t;

typedef struct loom_low_materialized_action_state_t {
  // Reload result or a replacement source for a later store action.
  loom_value_id_t value_id;
} loom_low_materialized_action_state_t;

static iree_status_t loom_low_allocation_append_materialized_value(
    loom_value_id_t value_id, iree_arena_allocator_t* arena,
    loom_low_materialized_value_list_builder_t* list) {
  if (list->count == list->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, list->count, list->count + 1, sizeof(*list->values),
        &list->capacity, (void**)&list->values));
  }
  list->values[list->count++] = value_id;
  return iree_ok_status();
}

static iree_status_t
loom_low_allocation_record_materialized_required_register_value(
    loom_value_id_t value_id, iree_arena_allocator_t* arena,
    loom_low_materialized_value_facts_t* facts) {
  return loom_low_allocation_append_materialized_value(
      value_id, arena, &facts->required_register_values);
}

static bool loom_low_allocation_materialized_value_has_private_consumer(
    const loom_module_t* module, loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(module, value_id);
  const loom_op_t* defining_op = loom_value_def_op(value);
  if (defining_op == NULL || defining_op->result_count != 1 ||
      value->use_count == 0 || loom_value_has_attribute_uses(value) ||
      loom_module_value_has_type_uses(module, value_id)) {
    return false;
  }
  const loom_use_t* uses = loom_value_uses(value);
  const loom_op_t* user = loom_use_user_op(uses[0]);
  if (user == NULL) {
    return false;
  }
  for (uint32_t i = 1; i < value->use_count; ++i) {
    if (loom_use_user_op(uses[i]) != user) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_low_allocation_record_materialized_reload_value(
    const loom_module_t* module, loom_value_id_t value_id,
    iree_arena_allocator_t* arena, loom_low_materialized_value_facts_t* facts) {
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_record_materialized_required_register_value(
          value_id, arena, facts));
  if (!loom_low_allocation_materialized_value_has_private_consumer(module,
                                                                   value_id)) {
    return iree_ok_status();
  }
  return loom_low_allocation_append_materialized_value(
      value_id, arena, &facts->per_user_placement_values);
}

static iree_status_t loom_low_allocation_record_materialized_action_values(
    const loom_low_spill_materialized_action_t* action,
    iree_arena_allocator_t* arena, loom_low_materialized_value_facts_t* facts) {
  for (iree_host_size_t i = 0; i < action->emitted_value_count; ++i) {
    const loom_low_spill_materialized_value_t value = action->emitted_values[i];
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_record_materialized_required_register_value(
            value.value_id, arena, facts));
    if (iree_any_bit_set(
            value.flags,
            LOOM_LOW_SPILL_MATERIALIZED_VALUE_FLAG_EXPANDED_EXECUTION)) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_append_materialized_value(
          value.value_id, arena, &facts->expanded_execution_values));
    }
    if (iree_any_bit_set(
            value.flags,
            LOOM_LOW_SPILL_MATERIALIZED_VALUE_FLAG_PRIVATE_TO_CONSUMER)) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_append_materialized_value(
          value.value_id, arena, &facts->per_user_placement_values));
    }
  }
  return iree_ok_status();
}

typedef struct loom_low_allocation_storage_prefix_t {
  // Function body containing the allocation table's IR snapshot.
  loom_region_t* body;
  // Function entry block receiving generated storage reservations.
  loom_block_t* entry_block;
  // Last storage reservation or ABI import in the normalized entry prefix.
  loom_op_t* storage_insertion_anchor;
  // Last contiguous ABI, storage, or spill op before executable entry work.
  loom_op_t* entry_traffic_insertion_anchor;
  // Number of existing storage reservations used to name generated slots.
  iree_host_size_t storage_reserve_count;
  // Whether generated reservations extend the contiguous entry traffic prefix.
  bool generated_storage_extends_entry_prefix;
} loom_low_allocation_storage_prefix_t;

static const loom_low_allocation_assignment_t*
loom_low_allocation_spill_plan_assignment(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plan) {
  IREE_ASSERT_LT(plan->assignment_index, table->assignment_count);
  return &table->assignments[plan->assignment_index];
}

static void loom_low_allocation_record_materialized_spill(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plan,
    loom_low_materialized_traffic_t store_traffic,
    loom_low_materialized_traffic_t reload_traffic,
    loom_low_allocation_materialized_spill_flags_t flags,
    loom_low_allocation_materialized_spill_t* record) {
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_spill_plan_assignment(table, plan);
  *record = (loom_low_allocation_materialized_spill_t){
      .value_id = plan->value_id,
      .descriptor_reg_class_id = assignment->descriptor_reg_class_id,
      .flags = flags,
      .assignment_index = plan->assignment_index,
      .slot_index = plan->slot_index,
      .slot_space = plan->slot_space,
      .byte_size = plan->byte_size,
      .byte_alignment = plan->byte_alignment,
      .store_count = store_traffic.count,
      .store_bytes = store_traffic.bytes,
      .reload_count = reload_traffic.count,
      .reload_bytes = reload_traffic.bytes,
  };
}

static iree_status_t loom_low_allocation_emit_materialized_spill(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id,
    loom_low_materialized_traffic_t store_traffic,
    loom_low_materialized_traffic_t reload_traffic,
    iree_diagnostic_emitter_t emitter) {
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_spill_plan_assignment(table, plan);
  const loom_op_t* origin_op = loom_low_diagnostic_value_origin_op(
      table->module, plan->value_id, table->function_op);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&table->target)),
      loom_param_string(loom_low_diagnostic_export_name(&table->target)),
      loom_param_string(loom_low_diagnostic_config_key(&table->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(table->module, table->function_op)),
      loom_param_string(
          loom_low_diagnostic_value_name(table->module, plan->value_id)),
      loom_param_string(loom_low_diagnostic_value_origin_operation_name(
          table->module, plan->value_id, table->function_op)),
      loom_param_string(loom_low_diagnostic_reg_class_name(
          table->target.descriptor_set, assignment->descriptor_reg_class_id)),
      loom_param_string(
          loom_low_diagnostic_value_name(table->module, storage_value_id)),
      loom_param_u64(plan->byte_size),
      loom_param_u32(store_traffic.count),
      loom_param_u64(store_traffic.bytes),
      loom_param_u32(reload_traffic.count),
      loom_param_u64(reload_traffic.bytes),
  };
  loom_diagnostic_emission_t emission = {
      .op = origin_op,
      .error = LOOM_ERR_BACKEND_009,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_low_allocation_emit_unsupported_spill_storage_space(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plan,
    loom_storage_space_t storage_space,
    loom_low_storage_space_set_t supported_storage_spaces,
    iree_diagnostic_emitter_t emitter) {
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_spill_plan_assignment(table, plan);
  iree_string_view_t supported_storage_space_names[LOOM_STORAGE_SPACE_COUNT_];
  const iree_host_size_t supported_storage_space_count =
      loom_low_storage_space_set_names(
          supported_storage_spaces,
          IREE_ARRAYSIZE(supported_storage_space_names),
          supported_storage_space_names);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&table->target)),
      loom_param_string(loom_low_diagnostic_export_name(&table->target)),
      loom_param_string(loom_low_diagnostic_config_key(&table->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(table->module, table->function_op)),
      loom_param_string(
          loom_low_diagnostic_value_name(table->module, plan->value_id)),
      loom_param_string(loom_low_diagnostic_reg_class_name(
          table->target.descriptor_set, assignment->descriptor_reg_class_id)),
      loom_param_string(loom_low_spill_slot_space_name(plan->slot_space)),
      loom_param_string(loom_low_storage_type_space_name(storage_space)),
      loom_param_string_list(supported_storage_space_names,
                             supported_storage_space_count),
  };
  loom_diagnostic_emission_t emission = {
      .module = table->module,
      .op = loom_low_diagnostic_value_origin_op(table->module, plan->value_id,
                                                table->function_op),
      .error = LOOM_ERR_BACKEND_019,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t
loom_low_allocation_validate_supported_spill_storage_spaces(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* spill_plans,
    const loom_low_spill_materialization_plan_t* materialization_plan,
    const loom_low_allocation_materialization_options_t* options,
    loom_low_allocation_materialization_result_t* result) {
  if (!options || !options->has_supported_storage_spaces) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < materialization_plan->record_count; ++i) {
    if (!loom_low_spill_materialization_record_has_traffic(
            &materialization_plan->records[i])) {
      continue;
    }
    const loom_low_allocation_spill_plan_t* plan = &spill_plans[i];
    const loom_storage_space_t storage_space =
        loom_low_allocation_storage_space_for_spill_slot(plan->slot_space);
    if (loom_low_storage_space_set_contains(options->supported_storage_spaces,
                                            storage_space)) {
      continue;
    }
    ++result->error_count;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_emit_unsupported_spill_storage_space(
            table, plan, storage_space, options->supported_storage_spaces,
            options->emitter));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_set_storage_value_name(
    loom_module_t* module, iree_string_view_t function_name,
    iree_host_size_t storage_index, loom_value_id_t storage_value_id,
    iree_arena_allocator_t* arena) {
  char suffix[64] = {0};
  const int suffix_length =
      snprintf(suffix, sizeof(suffix), "_spill_storage_%zu", storage_index);
  const iree_host_size_t name_length =
      function_name.size + (iree_host_size_t)suffix_length;

  char* name_storage = NULL;
  if (name_length > 0) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(arena, name_length, (void**)&name_storage));
    memcpy(name_storage, function_name.data, function_name.size);
    memcpy(name_storage + function_name.size, suffix,
           (iree_host_size_t)suffix_length);
  }

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      module, iree_make_string_view(name_storage, name_length), &name_id));
  return loom_module_set_value_name(module, storage_value_id, name_id);
}

static iree_status_t loom_low_allocation_prepare_storage_prefix(
    loom_module_t* module, const loom_op_t* function_op,
    iree_arena_allocator_t* arena,
    loom_low_allocation_storage_prefix_t* out_prefix) {
  loom_region_t* body = loom_low_function_body((loom_op_t*)function_op);
  loom_block_t* entry_block = loom_region_entry_block(body);
  loom_op_t* insertion_op = entry_block->first_op;
  loom_op_t* preamble_anchor = NULL;
  loom_op_t* last_storage_reserve = NULL;
  loom_op_t* entry_traffic_insertion_anchor = NULL;
  iree_host_size_t storage_reserve_count = 0;
  bool entry_prefix_open = true;
  loom_rewriter_t rewriter = {0};
  iree_status_t status = iree_ok_status();
  loom_op_t* op = entry_block->first_op;
  while (op != NULL && iree_status_is_ok(status)) {
    loom_op_t* next_op = op->next_op;
    if (loom_low_storage_reserve_isa(op)) {
      // Repair may insert entry copies before earlier reservations. Restore
      // their prefix in the same walk that retains insertion anchors. Moving
      // declarations in order preserves every existing storage byte offset.
      if (op == insertion_op) {
        insertion_op = next_op;
      } else {
        if (rewriter.module == NULL) {
          loom_rewriter_initialize(&rewriter, module, arena);
        }
        status = loom_rewriter_move_before(&rewriter, op, insertion_op);
      }
      ++storage_reserve_count;
      last_storage_reserve = op;
    } else {
      if (op == insertion_op && loom_low_function_entry_preamble_op(op)) {
        preamble_anchor = op;
        insertion_op = next_op;
      }
      if (entry_prefix_open) {
        if (loom_low_function_entry_traffic_op(op)) {
          entry_traffic_insertion_anchor = op;
        } else {
          entry_prefix_open = false;
        }
      }
    }
    op = next_op;
  }
  loom_rewriter_deinitialize(&rewriter);
  if (iree_status_is_ok(status)) {
    loom_op_t* storage_insertion_anchor =
        last_storage_reserve ? last_storage_reserve : preamble_anchor;
    const bool extends_entry_prefix =
        entry_traffic_insertion_anchor == NULL ||
        (storage_insertion_anchor != NULL &&
         storage_insertion_anchor->block_ordinal >=
             entry_traffic_insertion_anchor->block_ordinal);
    *out_prefix = (loom_low_allocation_storage_prefix_t){
        .body = body,
        .entry_block = entry_block,
        .storage_insertion_anchor = storage_insertion_anchor,
        .entry_traffic_insertion_anchor = extends_entry_prefix
                                              ? storage_insertion_anchor
                                              : entry_traffic_insertion_anchor,
        .storage_reserve_count = storage_reserve_count,
        .generated_storage_extends_entry_prefix = extends_entry_prefix,
    };
  }
  return status;
}

static void loom_low_allocation_set_storage_insertion_point(
    loom_builder_t* builder, loom_op_t* function_op,
    const loom_low_allocation_storage_prefix_t* prefix) {
  loom_builder_enter_region(builder, function_op, prefix->body);
  if (prefix->storage_insertion_anchor) {
    loom_builder_set_after(builder, prefix->storage_insertion_anchor);
  } else if (prefix->entry_block->first_op) {
    loom_builder_set_before(builder, prefix->entry_block->first_op);
  }
}

static void loom_low_allocation_set_spill_insertion_point(
    loom_builder_t* builder,
    const loom_low_allocation_storage_prefix_t* storage_prefix,
    loom_block_t* block) {
  if (block == storage_prefix->entry_block) {
    if (storage_prefix->entry_traffic_insertion_anchor) {
      loom_builder_set_after(builder,
                             storage_prefix->entry_traffic_insertion_anchor);
      return;
    }
  }
  if (block->first_op) {
    loom_builder_set_before(builder, block->first_op);
  }
}

static bool loom_low_allocation_defines_entry_preamble_value(
    const loom_op_t* function_op, const loom_op_t* op) {
  if (!loom_low_function_entry_preamble_op(op)) {
    return false;
  }
  const loom_region_t* body = loom_low_function_const_body(function_op);
  return op->parent_block == loom_region_const_entry_block(body);
}

static iree_status_t loom_low_allocation_insert_storage_reserves(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_low_allocation_storage_prefix_t* storage_prefix,
    const loom_low_allocation_spill_plan_t* spill_plans,
    const loom_low_spill_materialization_plan_t* materialization_plan,
    iree_host_size_t spill_plan_count, iree_diagnostic_emitter_t emitter,
    iree_arena_allocator_t* arena, loom_low_materialized_spill_slot_t* slots,
    loom_low_allocation_materialization_result_t* result) {
  const uint32_t* representatives = materialization_plan->slot_representatives;
  const iree_string_view_t function_name =
      loom_low_diagnostic_function_name(module, table->function_op);
  const iree_host_size_t storage_name_start =
      storage_prefix->storage_reserve_count;
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, NULL, &builder);
  loom_low_allocation_set_storage_insertion_point(
      &builder, (loom_op_t*)table->function_op, storage_prefix);
  loom_low_storage_layout_space_sizes_t sizes =
      table->storage_layout.space_sizes;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < spill_plan_count && iree_status_is_ok(status) &&
       result->error_count == 0;
       ++i) {
    if (!loom_low_spill_materialization_record_has_traffic(
            &materialization_plan->records[i])) {
      continue;
    }
    if (representatives && representatives[i] != i) {
      slots[i] = slots[representatives[i]];
      continue;
    }
    const loom_low_allocation_spill_plan_t* spill_plan = &spill_plans[i];
    const loom_storage_space_t storage_space =
        loom_low_allocation_storage_space_for_spill_slot(
            spill_plan->slot_space);

    loom_op_t* reserve_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
        &builder, (int64_t)spill_plan->byte_size,
        (int64_t)spill_plan->byte_alignment, loom_type_storage(storage_space),
        table->function_op->location, &reserve_op));
    if (!loom_low_storage_layout_accumulate_reservation(module, reserve_op,
                                                        &sizes)) {
      const loom_diagnostic_param_t params[] = {
          loom_param_string(function_name),
          loom_param_string(loom_low_storage_type_space_name(storage_space)),
      };
      ++result->error_count;
      status = iree_diagnostic_emit(emitter,
                                    &(loom_diagnostic_emission_t){
                                        .op = reserve_op,
                                        .error = LOOM_ERR_TARGET_096,
                                        .params = params,
                                        .param_count = IREE_ARRAYSIZE(params),
                                    });
      continue;
    }
    loom_value_id_t storage_value_id =
        loom_low_storage_reserve_storage(reserve_op);
    if (storage_prefix->generated_storage_extends_entry_prefix) {
      storage_prefix->entry_traffic_insertion_anchor = reserve_op;
    }
    const iree_host_size_t storage_name_index =
        storage_name_start + result->storage_count;
    IREE_RETURN_IF_ERROR(loom_low_allocation_set_storage_value_name(
        module, function_name, storage_name_index, storage_value_id, arena));
    loom_builder_set_after(&builder, reserve_op);
    slots[i] = (loom_low_materialized_spill_slot_t){
        .storage_value_id = storage_value_id,
    };
    ++result->storage_count;
    result->storage_bytes += spill_plan->byte_size;
    result->storage_minimum_alignment =
        iree_max(result->storage_minimum_alignment, spill_plan->byte_alignment);
  }

  return status;
}

static iree_status_t loom_low_allocation_rebuild_direct_br_without_arg(
    loom_module_t* module, loom_op_t* branch_op, uint16_t arg_index,
    iree_arena_allocator_t* arena) {
  const uint16_t old_count = branch_op->operand_count;
  const uint16_t new_count = (uint16_t)(old_count - 1);
  loom_value_id_t* new_args = NULL;
  if (new_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, new_count, sizeof(*new_args), (void**)&new_args));
    const loom_value_id_t* old_args = loom_op_const_operands(branch_op);
    uint16_t new_index = 0;
    for (uint16_t i = 0; i < old_count; ++i) {
      if (i == arg_index) {
        continue;
      }
      new_args[new_index++] = old_args[i];
    }
  }

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, branch_op->parent_block,
                          &builder);
  loom_builder_set_before(&builder, branch_op);
  loom_op_t* replacement_op = NULL;
  loom_block_t* destination = loom_direct_branch_dest(branch_op);
  if (loom_low_br_identity_isa(branch_op)) {
    uint16_t identity_start = (uint16_t)loom_low_br_identity_start(branch_op);
    uint16_t identity_count = (uint16_t)loom_low_br_identity_count(branch_op);
    if (arg_index < identity_start) {
      --identity_start;
    } else if (arg_index < identity_start + identity_count) {
      --identity_count;
    }
    if (identity_count != 0) {
      IREE_RETURN_IF_ERROR(loom_low_br_identity_build(
          &builder, identity_start, identity_count, destination, new_args,
          new_count, branch_op->location, &replacement_op));
    } else {
      IREE_RETURN_IF_ERROR(loom_low_br_build(&builder, destination, new_args,
                                             new_count, branch_op->location,
                                             &replacement_op));
    }
  } else {
    IREE_RETURN_IF_ERROR(loom_low_br_build(&builder, destination, new_args,
                                           new_count, branch_op->location,
                                           &replacement_op));
  }
  return loom_op_erase(module, branch_op);
}

static loom_value_id_t loom_low_allocation_materialization_resolve_value(
    loom_low_spill_materialization_value_ref_t value,
    const loom_low_materialized_action_state_t* action_states,
    iree_host_size_t action_count) {
  if (value.action_index == UINT32_MAX) {
    IREE_ASSERT_NE(value.value_id, LOOM_VALUE_ID_INVALID);
    return value.value_id;
  }
  IREE_ASSERT_LT(value.action_index, action_count);
  const loom_value_id_t value_id = action_states[value.action_index].value_id;
  IREE_ASSERT_NE(value_id, LOOM_VALUE_ID_INVALID);
  return value_id;
}

static loom_op_t* loom_low_allocation_materialization_target_op(
    loom_module_t* module,
    const loom_low_spill_materialization_target_t* target,
    uint16_t* out_operand_index) {
  switch (target->kind) {
    case LOOM_LOW_SPILL_MATERIALIZATION_TARGET_OPERATION_OPERAND:
      *out_operand_index = target->operand_index;
      return target->owner.operation;
    case LOOM_LOW_SPILL_MATERIALIZATION_TARGET_BRANCH_ARGUMENT: {
      const loom_value_t* block_argument =
          loom_module_value(module, target->block_argument_value_id);
      IREE_ASSERT(loom_value_is_block_arg(block_argument));
      loom_block_t* destination_block = loom_value_def_block(block_argument);
      *out_operand_index = loom_value_def_index(block_argument);
      loom_op_t* branch_op =
          (loom_op_t*)loom_block_const_last_op(target->owner.predecessor_block);
      IREE_ASSERT(loom_direct_branch_isa(branch_op));
      IREE_ASSERT_EQ(loom_direct_branch_dest(branch_op), destination_block);
      return branch_op;
    }
    case LOOM_LOW_SPILL_MATERIALIZATION_TARGET_ACTION_SOURCE:
      IREE_ASSERT_UNREACHABLE("action targets have no emitted operation yet");
      return NULL;
  }
  IREE_ASSERT_UNREACHABLE("unknown spill materialization target kind");
  return NULL;
}

typedef struct loom_low_materialization_insertion_t {
  // Original operation anchoring the emitted traffic, or NULL at entry.
  loom_op_t* anchor_op;
  // Source location assigned to the generated traffic.
  loom_location_id_t location;
  // Whether traffic belongs after |anchor_op|.
  bool after;
  // Whether traffic extends the normalized entry prefix.
  bool entry_prefix;
} loom_low_materialization_insertion_t;

static loom_low_materialization_insertion_t
loom_low_allocation_materialization_insertion(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* spill_plans,
    const loom_low_spill_materialization_action_t* action,
    const loom_low_spill_materialization_instruction_t* instruction) {
  const loom_liveness_operation_point_t* operation = loom_liveness_operation_at(
      &table->liveness, instruction->operation_index);
  loom_low_materialization_insertion_t insertion = {
      .anchor_op = (loom_op_t*)operation->op,
      .location = operation->op->location,
      .after = loom_low_spill_materialization_instruction_is_after(instruction),
  };
  switch (action->kind) {
    case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_DEFINITION_STORE: {
      const loom_low_allocation_spill_plan_t* spill_plan =
          &spill_plans[action->spill_plan_index];
      const loom_value_t* value =
          loom_module_value(module, spill_plan->value_id);
      const bool extends_entry_prefix =
          loom_value_is_block_arg(value)
              ? loom_value_def_block(value) ==
                    loom_region_const_entry_block(
                        loom_low_function_const_body(table->function_op))
              : loom_low_allocation_defines_entry_preamble_value(
                    table->function_op, loom_value_def_op(value));
      if (extends_entry_prefix) {
        insertion.anchor_op = NULL;
        insertion.location = table->function_op->location;
        insertion.after = false;
        insertion.entry_prefix = true;
      }
      break;
    }
    case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_SHARED_SLICE_RELOAD:
      // Capacity follows accepted schedule order, while mutation must dominate
      // every source-order slice until the generated reload is rescheduled.
      insertion.anchor_op = action->value.shared_slice_reload.first_slice_op;
      insertion.location = insertion.anchor_op->location;
      insertion.after = false;
      break;
    case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_EDGE_STORE:
    case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_OPERAND_RELOAD:
    case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_NARROW_SLICE_RELOAD:
      break;
    case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_REMOVE_BRANCH_ARGUMENT:
    case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_REMOVE_BLOCK_ARGUMENT:
      IREE_ASSERT_UNREACHABLE("structural action has no traffic insertion");
      break;
  }
  return insertion;
}

static void loom_low_allocation_materialization_set_insertion_point(
    loom_module_t* module, loom_builder_t* builder,
    loom_low_allocation_storage_prefix_t* storage_prefix,
    const loom_low_materialization_insertion_t* insertion,
    loom_op_t** inout_after_source_anchor,
    loom_op_t** inout_after_insertion_anchor) {
  if (insertion->entry_prefix) {
    loom_builder_initialize(module, &module->arena, storage_prefix->entry_block,
                            builder);
    loom_low_allocation_set_spill_insertion_point(builder, storage_prefix,
                                                  storage_prefix->entry_block);
    return;
  }
  IREE_ASSERT_ARGUMENT(insertion->anchor_op);
  loom_builder_initialize(module, &module->arena,
                          insertion->anchor_op->parent_block, builder);
  if (!insertion->after) {
    loom_builder_set_before(builder, insertion->anchor_op);
    return;
  }
  if (*inout_after_source_anchor != insertion->anchor_op) {
    *inout_after_source_anchor = insertion->anchor_op;
    *inout_after_insertion_anchor = insertion->anchor_op;
  }
  loom_builder_set_after(builder, *inout_after_insertion_anchor);
}

typedef struct loom_low_materialized_op_capture_t {
  // Builder callback restored after the traffic recipe finishes.
  loom_builder_callback_t previous_callback;
  // Last operation finalized by the traffic recipe.
  loom_op_t* last_op;
} loom_low_materialized_op_capture_t;

static iree_status_t loom_low_allocation_capture_materialized_op(
    void* user_data, loom_op_t* op) {
  loom_low_materialized_op_capture_t* capture =
      (loom_low_materialized_op_capture_t*)user_data;
  if (capture->previous_callback.fn != NULL) {
    IREE_RETURN_IF_ERROR(capture->previous_callback.fn(
        capture->previous_callback.user_data, op));
  }
  capture->last_op = op;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_emit_spill_traffic(
    const loom_low_spill_materialization_traffic_plan_t* traffic_plan,
    iree_host_size_t instruction_index,
    const loom_low_spill_materialization_traffic_t* traffic,
    uint32_t slot_byte_size, loom_builder_t* builder,
    loom_value_id_t storage_value_id, loom_value_id_t source_value_id,
    loom_location_id_t location, iree_arena_allocator_t* arena,
    loom_op_t** out_last_op, loom_low_spill_materialized_action_t* out_action) {
  *out_last_op = NULL;
  *out_action = (loom_low_spill_materialized_action_t){
      .value_id = LOOM_VALUE_ID_INVALID,
  };
  if (traffic_plan->emit == NULL) {
    const int64_t byte_offset =
        loom_low_spill_materialization_traffic_byte_offset(traffic,
                                                           slot_byte_size);
    if (traffic->kind == LOOM_LOW_SPILL_MATERIALIZATION_TRAFFIC_STORE) {
      IREE_ASSERT_NE(source_value_id, LOOM_VALUE_ID_INVALID);
      IREE_RETURN_IF_ERROR((traffic->moves_source ? loom_low_spill_move_build
                                                  : loom_low_spill_build)(
          builder, source_value_id, storage_value_id, byte_offset, location,
          out_last_op));
    } else {
      IREE_ASSERT_EQ(traffic->kind,
                     LOOM_LOW_SPILL_MATERIALIZATION_TRAFFIC_RELOAD);
      IREE_ASSERT_EQ(source_value_id, LOOM_VALUE_ID_INVALID);
      IREE_RETURN_IF_ERROR((traffic->moves_source ? loom_low_reload_move_build
                                                  : loom_low_reload_build)(
          builder, storage_value_id, byte_offset, traffic->value_type, location,
          out_last_op));
      out_action->value_id = loom_low_storage_load_result(*out_last_op);
    }
    return iree_ok_status();
  }

  IREE_ASSERT_ARGUMENT(arena);
  iree_arena_reset(arena);
  loom_low_materialized_op_capture_t capture = {
      .previous_callback = builder->on_op_finalized,
  };
  builder->on_op_finalized = (loom_builder_callback_t){
      .fn = loom_low_allocation_capture_materialized_op,
      .user_data = &capture,
  };
  iree_status_t status = traffic_plan->emit(
      traffic_plan->user_data, instruction_index, builder, storage_value_id,
      source_value_id, location, arena, out_action);
  builder->on_op_finalized = capture.previous_callback;
  IREE_RETURN_IF_ERROR(status);
  IREE_ASSERT(capture.last_op != NULL);
  IREE_ASSERT_EQ(traffic->kind == LOOM_LOW_SPILL_MATERIALIZATION_TRAFFIC_RELOAD,
                 out_action->value_id != LOOM_VALUE_ID_INVALID);
  *out_last_op = capture.last_op;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_remove_branch_argument(
    loom_module_t* module,
    const loom_low_spill_materialization_action_t* action,
    iree_bitmap_t per_user_placement_values, iree_arena_allocator_t* arena) {
  const loom_value_t* block_argument = loom_module_value(
      module, action->value.remove_branch_argument.block_argument_value_id);
  IREE_ASSERT(loom_value_is_block_arg(block_argument));
  const uint16_t argument_index = loom_value_def_index(block_argument);
  loom_op_t* branch_op = (loom_op_t*)loom_block_const_last_op(
      action->value.remove_branch_argument.predecessor_block);
  const loom_value_id_t payload =
      loom_op_const_operands(branch_op)[argument_index];
  if (payload < per_user_placement_values.bit_count) {
    iree_bitmap_reset(per_user_placement_values, payload);
  }
  return loom_low_allocation_rebuild_direct_br_without_arg(
      module, branch_op, argument_index, arena);
}

static iree_status_t loom_low_allocation_apply_spill_materialization_plan(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* spill_plans,
    const loom_low_materialized_spill_slot_t* slots,
    const loom_low_spill_materialization_plan_t* materialization_plan,
    const loom_low_spill_materialization_traffic_plan_t* traffic_plan,
    loom_low_allocation_storage_prefix_t* storage_prefix,
    iree_bitmap_t per_user_placement_values,
    iree_arena_allocator_t* traffic_arena, iree_arena_allocator_t* arena,
    loom_low_materialized_value_facts_t* facts) {
  loom_low_materialized_action_state_t* action_states = NULL;
  if (materialization_plan->action_count != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, materialization_plan->action_count, sizeof(*action_states),
        (void**)&action_states));
    for (iree_host_size_t i = 0; i < materialization_plan->action_count; ++i) {
      action_states[i].value_id = LOOM_VALUE_ID_INVALID;
    }
  }

  loom_op_t* after_source_anchor = NULL;
  loom_op_t* after_insertion_anchor = NULL;
  for (iree_host_size_t i = 0; i < materialization_plan->instruction_count;
       ++i) {
    const loom_low_spill_materialization_instruction_t* instruction =
        &materialization_plan->instructions[i];
    const uint32_t action_index =
        loom_low_spill_materialization_instruction_action_index(instruction);
    IREE_ASSERT_LT(action_index, materialization_plan->action_count);
    const loom_low_spill_materialization_action_t* action =
        &materialization_plan->actions[action_index];
    IREE_ASSERT_LT(action->spill_plan_index,
                   materialization_plan->record_count);
    const loom_low_allocation_spill_plan_t* spill_plan =
        &spill_plans[action->spill_plan_index];
    const loom_value_id_t storage_value_id =
        slots[action->spill_plan_index].storage_value_id;
    const loom_low_materialization_insertion_t insertion =
        loom_low_allocation_materialization_insertion(
            module, table, spill_plans, action, instruction);
    loom_builder_t builder = {0};
    loom_low_allocation_materialization_set_insertion_point(
        module, &builder, storage_prefix, &insertion, &after_source_anchor,
        &after_insertion_anchor);
    loom_low_spill_materialization_traffic_t traffic = {0};
    const bool has_traffic = loom_low_spill_materialization_plan_query_traffic(
        module, spill_plans, materialization_plan, action_index, &traffic);
    loom_op_t* last_traffic_op = NULL;
    loom_low_spill_materialized_action_t materialized_action = {
        .value_id = LOOM_VALUE_ID_INVALID,
    };
    switch (action->kind) {
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_DEFINITION_STORE: {
        IREE_ASSERT(has_traffic);
        IREE_RETURN_IF_ERROR(loom_low_allocation_emit_spill_traffic(
            traffic_plan, i, &traffic, spill_plan->byte_size, &builder,
            storage_value_id, spill_plan->value_id, insertion.location,
            traffic_arena, &last_traffic_op, &materialized_action));
        break;
      }
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_EDGE_STORE: {
        IREE_ASSERT(has_traffic);
        const loom_value_id_t source_value_id =
            action_states[action_index].value_id != LOOM_VALUE_ID_INVALID
                ? action_states[action_index].value_id
                : loom_low_allocation_materialization_resolve_value(
                      action->value.edge_store.source, action_states,
                      materialization_plan->action_count);
        IREE_RETURN_IF_ERROR(loom_low_allocation_emit_spill_traffic(
            traffic_plan, i, &traffic, spill_plan->byte_size, &builder,
            storage_value_id, source_value_id, insertion.location,
            traffic_arena, &last_traffic_op, &materialized_action));
        break;
      }
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_OPERAND_RELOAD: {
        IREE_ASSERT(has_traffic);
        IREE_RETURN_IF_ERROR(loom_low_allocation_emit_spill_traffic(
            traffic_plan, i, &traffic, spill_plan->byte_size, &builder,
            storage_value_id, LOOM_VALUE_ID_INVALID, insertion.location,
            traffic_arena, &last_traffic_op, &materialized_action));
        const loom_value_id_t reload_value_id = materialized_action.value_id;
        action_states[action_index].value_id = reload_value_id;
        const loom_low_spill_materialization_target_t* target =
            &action->value.operand_reload.target;
        if (target->kind ==
            LOOM_LOW_SPILL_MATERIALIZATION_TARGET_ACTION_SOURCE) {
          IREE_ASSERT_LT(target->owner.action_index,
                         materialization_plan->action_count);
          IREE_ASSERT_EQ(action_states[target->owner.action_index].value_id,
                         LOOM_VALUE_ID_INVALID);
          action_states[target->owner.action_index].value_id = reload_value_id;
        } else {
          uint16_t operand_index = 0;
          loom_op_t* user_op = loom_low_allocation_materialization_target_op(
              module, target, &operand_index);
          IREE_RETURN_IF_ERROR(loom_op_set_operand(
              module, user_op, operand_index, reload_value_id));
        }
        break;
      }
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_NARROW_SLICE_RELOAD: {
        IREE_ASSERT(has_traffic);
        loom_op_t* slice_op = action->value.narrow_slice_reload.slice_op;
        const loom_value_id_t slice_result = loom_low_slice_result(slice_op);
        IREE_RETURN_IF_ERROR(loom_low_allocation_emit_spill_traffic(
            traffic_plan, i, &traffic, spill_plan->byte_size, &builder,
            storage_value_id, LOOM_VALUE_ID_INVALID, insertion.location,
            traffic_arena, &last_traffic_op, &materialized_action));
        const loom_value_id_t reload_value_id = materialized_action.value_id;
        action_states[action_index].value_id = reload_value_id;
        IREE_RETURN_IF_ERROR(loom_value_replace_all_uses_with(
            module, slice_result, reload_value_id));
        IREE_RETURN_IF_ERROR(loom_op_erase(module, slice_op));
        break;
      }
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_SHARED_SLICE_RELOAD: {
        IREE_ASSERT(has_traffic);
        IREE_RETURN_IF_ERROR(loom_low_allocation_emit_spill_traffic(
            traffic_plan, i, &traffic, spill_plan->byte_size, &builder,
            storage_value_id, LOOM_VALUE_ID_INVALID, insertion.location,
            traffic_arena, &last_traffic_op, &materialized_action));
        const loom_value_id_t reload_value_id = materialized_action.value_id;
        action_states[action_index].value_id = reload_value_id;
        const uint32_t target_start =
            action->value.shared_slice_reload.target_start;
        const uint32_t target_count =
            action->value.shared_slice_reload.target_count;
        IREE_ASSERT_LE((iree_host_size_t)target_start + target_count,
                       materialization_plan->target_count);
        for (uint32_t j = 0; j < target_count; ++j) {
          uint16_t operand_index = 0;
          loom_op_t* slice_op = loom_low_allocation_materialization_target_op(
              module, &materialization_plan->targets[target_start + j],
              &operand_index);
          IREE_RETURN_IF_ERROR(loom_op_set_operand(
              module, slice_op, operand_index, reload_value_id));
        }
        break;
      }
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_REMOVE_BRANCH_ARGUMENT:
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_REMOVE_BLOCK_ARGUMENT:
        IREE_ASSERT_UNREACHABLE("structural action entered traffic stream");
        break;
    }
    IREE_ASSERT(has_traffic);
    IREE_ASSERT_ARGUMENT(last_traffic_op);
    if (traffic_plan->emit != NULL) {
      IREE_RETURN_IF_ERROR(
          loom_low_allocation_record_materialized_action_values(
              &materialized_action, arena, facts));
    }
    if (insertion.entry_prefix) {
      storage_prefix->entry_traffic_insertion_anchor = last_traffic_op;
    } else if (insertion.after) {
      after_insertion_anchor = last_traffic_op;
    }
  }

  // Value facts retain transformation order and observe the complete emitted
  // use graph. In particular, an action-target reload has its store use now.
  for (uint32_t action_index = 0;
       action_index < materialization_plan->action_count; ++action_index) {
    const loom_low_spill_materialization_action_t* action =
        &materialization_plan->actions[action_index];
    if (traffic_plan->emit != NULL) {
      switch (action->kind) {
        case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_OPERAND_RELOAD:
        case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_NARROW_SLICE_RELOAD:
        case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_SHARED_SLICE_RELOAD: {
          const loom_value_id_t value_id = action_states[action_index].value_id;
          if (loom_low_allocation_materialized_value_has_private_consumer(
                  module, value_id)) {
            IREE_RETURN_IF_ERROR(loom_low_allocation_append_materialized_value(
                value_id, arena, &facts->per_user_placement_values));
          }
          break;
        }
        default:
          break;
      }
      continue;
    }
    switch (action->kind) {
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_DEFINITION_STORE: {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_record_materialized_required_register_value(
                spill_plans[action->spill_plan_index].value_id, arena, facts));
        break;
      }
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_EDGE_STORE: {
        const loom_value_id_t source_value_id =
            action_states[action_index].value_id != LOOM_VALUE_ID_INVALID
                ? action_states[action_index].value_id
                : loom_low_allocation_materialization_resolve_value(
                      action->value.edge_store.source, action_states,
                      materialization_plan->action_count);
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_record_materialized_required_register_value(
                source_value_id, arena, facts));
        break;
      }
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_OPERAND_RELOAD:
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_NARROW_SLICE_RELOAD: {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_record_materialized_reload_value(
                module, action_states[action_index].value_id, arena, facts));
        break;
      }
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_SHARED_SLICE_RELOAD: {
        IREE_RETURN_IF_ERROR(
            loom_low_allocation_record_materialized_required_register_value(
                action_states[action_index].value_id, arena, facts));
        break;
      }
      default:
        break;
    }
  }

  // CFG signature mutation remains in transformation order after every
  // traffic action has consumed its original anchor and logical payload.
  for (iree_host_size_t i = 0; i < materialization_plan->action_count; ++i) {
    const loom_low_spill_materialization_action_t* action =
        &materialization_plan->actions[i];
    switch (action->kind) {
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_REMOVE_BRANCH_ARGUMENT: {
        IREE_RETURN_IF_ERROR(loom_low_allocation_remove_branch_argument(
            module, action, per_user_placement_values, arena));
        break;
      }
      case LOOM_LOW_SPILL_MATERIALIZATION_ACTION_REMOVE_BLOCK_ARGUMENT: {
        loom_value_t* block_argument = loom_module_value(
            module, action->value.remove_block_argument.value_id);
        IREE_ASSERT(loom_value_is_block_arg(block_argument));
        loom_block_t* block = loom_value_def_block(block_argument);
        const uint16_t argument_index = loom_value_def_index(block_argument);
        IREE_RETURN_IF_ERROR(
            loom_block_remove_arg(module, block, argument_index));
        break;
      }
      default:
        break;
    }
  }
  return iree_ok_status();
}

#if IREE_HAVE_ATTRIBUTE(minsize)
__attribute__((minsize))
#endif
iree_status_t loom_low_allocation_materialize_spills(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* spill_plans,
    iree_host_size_t spill_plan_count,
    const loom_low_allocation_materialization_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_allocation_materialization_result_t* out_result) {
  loom_module_t* module = table->module;
  loom_low_allocation_materialization_result_t result = {0};
  loom_low_materialized_value_facts_t value_facts = {0};
  if (out_result) {
    *out_result = result;
  }

  if (spill_plan_count == 0) {
    return iree_ok_status();
  }

  const bool emit_spill_diagnostics =
      options && options->emit_spill_diagnostics;
  const bool record_materialized_spills =
      options && options->record_materialized_spills;
  const iree_diagnostic_emitter_t emitter =
      options ? options->emitter : (iree_diagnostic_emitter_t){0};
  const iree_bitmap_t per_user_placement_values =
      options ? options->per_user_placement_values : (iree_bitmap_t){0};
  if (spill_plan_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "materialized storage count overflow");
  }

  loom_low_spill_materialization_plan_t materialization_plan = {0};
  IREE_RETURN_IF_ERROR(loom_low_spill_materialization_plan_build(
      table, spill_plans, spill_plan_count, arena, &materialization_plan));
  IREE_ASSERT_EQ(materialization_plan.record_count, spill_plan_count);

  IREE_RETURN_IF_ERROR(
      loom_low_allocation_validate_supported_spill_storage_spaces(
          table, spill_plans, &materialization_plan, options, &result));
  if (result.error_count != 0) {
    if (out_result) {
      *out_result = result;
    }
    return iree_ok_status();
  }

  loom_low_spill_materialization_traffic_plan_t traffic_plan = {0};
  if (options && options->plan_traffic != NULL) {
    bool accepted = false;
    IREE_RETURN_IF_ERROR(options->plan_traffic(
        options->plan_traffic_user_data, table, spill_plans, spill_plan_count,
        &materialization_plan, arena, emitter, &accepted, &traffic_plan));
    if (!accepted) {
      ++result.error_count;
      if (out_result) {
        *out_result = result;
      }
      return iree_ok_status();
    }
    IREE_ASSERT(traffic_plan.emit != NULL);
  }
  if (traffic_plan.emit == NULL) {
    for (iree_host_size_t i = 0; i < spill_plan_count; ++i) {
      if (spill_plans[i].defined_parts != 0 &&
          loom_low_spill_materialization_record_has_traffic(
              &materialization_plan.records[i])) {
        const loom_low_allocation_spill_plan_t* plan = &spill_plans[i];
        const loom_low_allocation_assignment_t* assignment =
            loom_low_allocation_spill_plan_assignment(table, plan);
        const loom_diagnostic_param_t params[] = {
            loom_param_string(
                loom_low_diagnostic_function_name(module, table->function_op)),
            loom_param_string(
                loom_low_diagnostic_value_name(module, plan->value_id)),
            loom_param_string(loom_low_diagnostic_reg_class_name(
                table->target.descriptor_set,
                assignment->descriptor_reg_class_id)),
            loom_param_u32(plan->defined_parts),
        };
        const loom_diagnostic_emission_t emission = {
            .module = module,
            .op = loom_low_diagnostic_value_origin_op(module, plan->value_id,
                                                      table->function_op),
            .error = LOOM_ERR_BACKEND_058,
            .params = params,
            .param_count = IREE_ARRAYSIZE(params),
        };
        IREE_RETURN_IF_ERROR(iree_diagnostic_emit(emitter, &emission));
        ++result.error_count;
      }
    }
    if (result.error_count != 0) {
      if (out_result) {
        *out_result = result;
      }
      return iree_ok_status();
    }
  }

  loom_low_allocation_storage_prefix_t storage_prefix;
  IREE_RETURN_IF_ERROR(loom_low_allocation_prepare_storage_prefix(
      module, table->function_op, arena, &storage_prefix));

  loom_low_materialized_spill_slot_t* slots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, spill_plan_count, sizeof(*slots), (void**)&slots));
  loom_low_allocation_materialized_spill_t* materialized_spills = NULL;
  if (record_materialized_spills) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, spill_plan_count, sizeof(*materialized_spills),
        (void**)&materialized_spills));
  }

  IREE_RETURN_IF_ERROR(loom_low_allocation_insert_storage_reserves(
      module, table, &storage_prefix, spill_plans, &materialization_plan,
      spill_plan_count, emitter, arena, slots, &result));
  if (result.error_count != 0) {
    if (out_result) {
      *out_result = result;
    }
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < spill_plan_count; ++i) {
    if (spill_plans[i].value_id < per_user_placement_values.bit_count) {
      iree_bitmap_reset(per_user_placement_values, spill_plans[i].value_id);
    }
    const loom_low_spill_materialization_record_t* plan_record =
        &materialization_plan.records[i];
    if (plan_record->store_count > UINT32_MAX - result.spill_count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized spill count overflow");
    }
    if (plan_record->store_bytes > UINT64_MAX - result.spill_bytes) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized spill byte count overflow");
    }
    if (plan_record->reload_count > UINT32_MAX - result.reload_count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized reload count overflow");
    }
    if (plan_record->reload_bytes > UINT64_MAX - result.reload_bytes) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized reload byte count overflow");
    }
    result.spill_count += plan_record->store_count;
    result.spill_bytes += plan_record->store_bytes;
    result.reload_count += plan_record->reload_count;
    result.reload_bytes += plan_record->reload_bytes;
  }

  iree_arena_allocator_t traffic_arena;
  iree_arena_allocator_t* traffic_arena_ptr = NULL;
  if (traffic_plan.emit != NULL) {
    iree_arena_initialize(arena->block_pool, &traffic_arena);
    traffic_arena_ptr = &traffic_arena;
  }
  const iree_status_t traffic_status =
      loom_low_allocation_apply_spill_materialization_plan(
          module, table, spill_plans, slots, &materialization_plan,
          &traffic_plan, &storage_prefix, per_user_placement_values,
          traffic_arena_ptr, arena, &value_facts);
  if (traffic_arena_ptr != NULL) {
    iree_arena_deinitialize(&traffic_arena);
  }
  IREE_RETURN_IF_ERROR(traffic_status);

  for (iree_host_size_t i = 0; i < spill_plan_count; ++i) {
    const loom_low_spill_materialization_record_t* plan_record =
        &materialization_plan.records[i];
    const loom_low_materialized_traffic_t materialized_store_traffic = {
        .count = plan_record->store_count,
        .bytes = plan_record->store_bytes,
    };
    const loom_low_materialized_traffic_t materialized_reload_traffic = {
        .count = plan_record->reload_count,
        .bytes = plan_record->reload_bytes,
    };
    const loom_low_allocation_materialized_spill_flags_t record_flags =
        plan_record->value_is_block_argument
            ? LOOM_LOW_ALLOCATION_MATERIALIZED_SPILL_FLAG_VALUE_WAS_BLOCK_ARGUMENT
            : 0;
    if (record_materialized_spills) {
      loom_low_allocation_record_materialized_spill(
          table, &spill_plans[i], materialized_store_traffic,
          materialized_reload_traffic, record_flags, &materialized_spills[i]);
    }
    if (emit_spill_diagnostics &&
        loom_low_spill_materialization_record_has_traffic(plan_record)) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_emit_materialized_spill(
          table, &spill_plans[i], slots[i].storage_value_id,
          materialized_store_traffic, materialized_reload_traffic, emitter));
    }
  }

  result.materialized_spills = materialized_spills;
  result.materialized_spill_count =
      record_materialized_spills ? spill_plan_count : 0;
  result.required_register_values =
      (loom_low_allocation_materialized_value_list_t){
          .values = value_facts.required_register_values.values,
          .count = value_facts.required_register_values.count,
      };
  result.per_user_placement_values =
      (loom_low_allocation_materialized_value_list_t){
          .values = value_facts.per_user_placement_values.values,
          .count = value_facts.per_user_placement_values.count,
      };
  result.expanded_execution_values =
      (loom_low_allocation_materialized_value_list_t){
          .values = value_facts.expanded_execution_values.values,
          .count = value_facts.expanded_execution_values.count,
      };
  if (out_result) {
    *out_result = result;
  }
  return iree_ok_status();
}
