// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation_materialization.h"

#include <stdio.h>
#include <string.h>

#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"

typedef struct loom_low_materialized_spill_slot_t {
  // Table spill slot ordinal represented by this generated storage value.
  uint32_t slot_index;
  // SSA value ID produced by the generated low.storage.reserve op.
  loom_value_id_t storage_value_id;
} loom_low_materialized_spill_slot_t;

static iree_status_t loom_low_allocation_emit_materialized_spill(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id, iree_diagnostic_emitter_t emitter) {
  if (plan->assignment_index >= table->assignment_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation spill plan references an out-of-range assignment");
  }
  const loom_low_allocation_assignment_t* assignment =
      &table->assignments[plan->assignment_index];
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&table->target)),
      loom_param_string(loom_low_diagnostic_export_name(&table->target)),
      loom_param_string(loom_low_diagnostic_config_key(&table->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(table->module, table->function_op)),
      loom_param_string(
          loom_low_diagnostic_value_name(table->module, plan->value_id)),
      loom_param_string(loom_low_diagnostic_value_class_name(
          table->target.descriptor_set, assignment->value_class)),
      loom_param_string(
          loom_low_diagnostic_value_name(table->module, storage_value_id)),
      loom_param_u32(plan->byte_size),
      loom_param_u32(plan->store_count),
      loom_param_u32(plan->reload_count),
  };
  loom_diagnostic_emission_t emission = {
      .op = loom_low_diagnostic_value_origin_op(table->module, plan->value_id,
                                                table->function_op),
      .error = LOOM_ERR_BACKEND_009,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_low_allocation_map_slot_space(
    loom_low_spill_slot_space_t slot_space, loom_storage_space_t* out_space) {
  switch (slot_space) {
    case LOOM_LOW_SPILL_SLOT_SPACE_STACK:
      *out_space = LOOM_STORAGE_SPACE_STACK;
      return iree_ok_status();
    case LOOM_LOW_SPILL_SLOT_SPACE_SCRATCH:
      *out_space = LOOM_STORAGE_SPACE_SCRATCH;
      return iree_ok_status();
    case LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE:
      *out_space = LOOM_STORAGE_SPACE_PRIVATE;
      return iree_ok_status();
    case LOOM_LOW_SPILL_SLOT_SPACE_LDS:
      *out_space = LOOM_STORAGE_SPACE_WORKGROUP;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown allocation spill slot space %u",
                              (unsigned)slot_space);
  }
}

static loom_diagnostic_string_list_t
loom_low_allocation_supported_storage_space_names(
    loom_low_storage_space_set_t supported_storage_spaces,
    iree_string_view_t* storage_space_names) {
  static const loom_storage_space_t kStorageSpaceOrder[] = {
      LOOM_STORAGE_SPACE_STACK,
      LOOM_STORAGE_SPACE_SCRATCH,
      LOOM_STORAGE_SPACE_PRIVATE,
      LOOM_STORAGE_SPACE_WORKGROUP,
  };
  iree_host_size_t count = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kStorageSpaceOrder); ++i) {
    const loom_storage_space_t storage_space = kStorageSpaceOrder[i];
    if (!loom_low_storage_space_set_contains(supported_storage_spaces,
                                             storage_space)) {
      continue;
    }
    storage_space_names[count++] =
        iree_make_cstring_view(loom_storage_space_name(storage_space));
  }
  return (loom_diagnostic_string_list_t){
      .values = storage_space_names,
      .count = count,
  };
}

static iree_status_t loom_low_allocation_emit_unsupported_spill_storage_space(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plan,
    loom_storage_space_t storage_space,
    loom_low_storage_space_set_t supported_storage_spaces,
    iree_diagnostic_emitter_t emitter) {
  if (plan->assignment_index >= table->assignment_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation spill plan references an out-of-range assignment");
  }
  const loom_low_allocation_assignment_t* assignment =
      &table->assignments[plan->assignment_index];
  iree_string_view_t supported_storage_space_names[LOOM_STORAGE_SPACE_COUNT_];
  const loom_diagnostic_string_list_t supported_storage_space_list =
      loom_low_allocation_supported_storage_space_names(
          supported_storage_spaces, supported_storage_space_names);
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_target_key(&table->target)),
      loom_param_string(loom_low_diagnostic_export_name(&table->target)),
      loom_param_string(loom_low_diagnostic_config_key(&table->target)),
      loom_param_string(
          loom_low_diagnostic_function_name(table->module, table->function_op)),
      loom_param_string(
          loom_low_diagnostic_value_name(table->module, plan->value_id)),
      loom_param_string(loom_low_diagnostic_value_class_name(
          table->target.descriptor_set, assignment->value_class)),
      loom_param_string(loom_low_spill_slot_space_name(plan->slot_space)),
      loom_param_string(
          iree_make_cstring_view(loom_storage_space_name(storage_space))),
      loom_param_string_list(supported_storage_space_list.values,
                             supported_storage_space_list.count),
  };
  loom_diagnostic_emission_t emission = {
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
    const loom_low_allocation_table_t* table, iree_host_size_t spill_plan_count,
    const loom_low_allocation_materialization_options_t* options,
    loom_low_allocation_materialization_result_t* result) {
  if (!options || !options->has_supported_storage_spaces) {
    return iree_ok_status();
  }
  for (iree_host_size_t i = 0; i < spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* plan = &table->spill_plans[i];
    loom_storage_space_t storage_space = LOOM_STORAGE_SPACE_COUNT_;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_map_slot_space(plan->slot_space, &storage_space));
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

static iree_status_t loom_low_allocation_verify_no_existing_storage_traffic(
    const loom_op_t* function_op) {
  loom_region_t* body = loom_low_function_body((loom_op_t*)function_op);
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (loom_low_spill_isa(op) || loom_low_reload_isa(op)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "low allocation materialization requires a low function without "
            "existing low.spill or low.reload ops");
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_set_storage_value_name(
    loom_module_t* module, loom_symbol_ref_t function_ref,
    iree_host_size_t storage_index, loom_value_id_t storage_value_id,
    iree_arena_allocator_t* arena) {
  if (function_ref.module_id != 0 ||
      function_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "spill storage function symbol is not defined");
  }
  loom_string_id_t function_name_id =
      module->symbols.entries[function_ref.symbol_id].name_id;
  if (function_name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "spill storage function symbol has no name");
  }
  iree_string_view_t function_name = module->strings.entries[function_name_id];

  char suffix[64] = {0};
  int suffix_length =
      snprintf(suffix, sizeof(suffix), "_spill_storage_%zu", storage_index);
  if (suffix_length < 0 || (iree_host_size_t)suffix_length >= sizeof(suffix)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "spill storage value name overflow");
  }

  iree_host_size_t name_length = 0;
  if (!iree_host_size_checked_add(
          function_name.size, (iree_host_size_t)suffix_length, &name_length)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "spill storage value name overflow");
  }

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

static iree_status_t loom_low_allocation_count_storage_reserves(
    const loom_op_t* function_op, iree_host_size_t* out_count) {
  *out_count = 0;
  const loom_region_t* body = loom_low_function_const_body(function_op);
  if (body == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low allocation materialization requires a low function body");
  }
  iree_host_size_t count = 0;
  const loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (!loom_low_storage_reserve_isa(op)) {
        continue;
      }
      if (count == IREE_HOST_SIZE_MAX) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "spill storage count overflows host size");
      }
      ++count;
    }
  }
  *out_count = count;
  return iree_ok_status();
}

static bool loom_low_allocation_entry_preamble_op(const loom_op_t* op) {
  return loom_low_live_in_isa(op) || loom_low_resource_isa(op);
}

static bool loom_low_allocation_entry_storage_declaration_prefix_op(
    const loom_op_t* op) {
  return loom_low_allocation_entry_preamble_op(op) ||
         loom_low_storage_reserve_isa(op);
}

static bool loom_low_allocation_entry_storage_prefix_op(const loom_op_t* op) {
  return loom_low_allocation_entry_preamble_op(op) ||
         loom_low_storage_reserve_isa(op) || loom_low_spill_isa(op);
}

static iree_status_t loom_low_allocation_set_storage_insertion_point(
    loom_builder_t* builder, loom_op_t* function_op) {
  loom_region_t* body = loom_low_function_body(function_op);
  if (!body || body->block_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low function has no entry block");
  }
  loom_builder_enter_region(builder, function_op, body);
  loom_block_t* entry_block = loom_region_entry_block(body);

  const loom_op_t* insertion_anchor = NULL;
  const loom_op_t* scan_op = entry_block->first_op;
  while (scan_op &&
         loom_low_allocation_entry_storage_declaration_prefix_op(scan_op)) {
    insertion_anchor = scan_op;
    scan_op = scan_op->next_op;
  }
  if (insertion_anchor) {
    loom_builder_set_after(builder, insertion_anchor);
  } else if (entry_block->first_op) {
    loom_builder_set_before(builder, entry_block->first_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_set_block_arg_spill_insertion_point(
    loom_builder_t* builder, const loom_op_t* function_op,
    loom_block_t* block) {
  const loom_region_t* body = loom_low_function_const_body(function_op);
  const loom_block_t* entry_block =
      body ? loom_region_const_entry_block(body) : NULL;
  if (block == entry_block) {
    const loom_op_t* insertion_anchor = NULL;
    const loom_op_t* scan_op = block->first_op;
    while (scan_op && loom_low_allocation_entry_storage_prefix_op(scan_op)) {
      insertion_anchor = scan_op;
      scan_op = scan_op->next_op;
    }
    if (insertion_anchor) {
      loom_builder_set_after(builder, insertion_anchor);
      return iree_ok_status();
    }
  }
  if (block->first_op) {
    loom_builder_set_before(builder, block->first_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_insert_storage_reserves(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    iree_host_size_t spill_plan_count, iree_arena_allocator_t* arena,
    loom_low_materialized_spill_slot_t* slots) {
  loom_symbol_ref_t function_ref = loom_low_function_callee(table->function_op);
  iree_host_size_t storage_name_start = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_count_storage_reserves(
      table->function_op, &storage_name_start));
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, NULL, &builder);
  IREE_RETURN_IF_ERROR(loom_low_allocation_set_storage_insertion_point(
      &builder, (loom_op_t*)table->function_op));
  for (iree_host_size_t i = 0; i < spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* plan = &table->spill_plans[i];
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (slots[j].slot_index == plan->slot_index) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "low allocation materialization requires unique spill slot "
            "ordinals");
      }
    }

    loom_storage_space_t storage_space = LOOM_STORAGE_SPACE_COUNT_;
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_map_slot_space(plan->slot_space, &storage_space));

    loom_op_t* reserve_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
        &builder, (int64_t)plan->byte_size, (int64_t)plan->byte_alignment,
        loom_type_storage(storage_space), table->function_op->location,
        &reserve_op));
    loom_value_id_t storage_value_id =
        loom_low_storage_reserve_storage(reserve_op);
    iree_host_size_t storage_name_index = 0;
    if (!iree_host_size_checked_add(storage_name_start, i,
                                    &storage_name_index)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "spill storage name index overflows host size");
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_set_storage_value_name(
        module, function_ref, storage_name_index, storage_value_id, arena));
    loom_builder_set_after(&builder, reserve_op);
    slots[i] = (loom_low_materialized_spill_slot_t){
        .slot_index = plan->slot_index,
        .storage_value_id = storage_value_id,
    };
  }

  return iree_ok_status();
}

static iree_status_t loom_low_allocation_snapshot_value_uses(
    loom_module_t* module, loom_value_id_t value_id,
    iree_arena_allocator_t* arena, loom_use_t** out_uses,
    uint32_t* out_use_count) {
  loom_value_t* value = loom_module_value(module, value_id);
  *out_use_count = value->use_count;
  *out_uses = NULL;
  if (value->use_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, value->use_count, sizeof(**out_uses), (void**)out_uses));
  memcpy(*out_uses, loom_value_uses(value),
         (iree_host_size_t)value->use_count * sizeof(**out_uses));
  return iree_ok_status();
}

static bool loom_low_allocation_use_is_removed_block_arg_edge(
    loom_use_t use, const loom_block_t* block, uint16_t arg_index) {
  const loom_op_t* user_op = loom_use_user_op(use);
  return user_op && loom_low_br_isa(user_op) &&
         loom_low_br_dest(user_op) == block &&
         loom_use_operand_index(use) == arg_index;
}

static uint32_t loom_low_allocation_count_materialized_reloads(
    const loom_use_t* uses, uint32_t use_count, const loom_block_t* block,
    uint16_t arg_index) {
  uint32_t reload_count = 0;
  for (uint32_t i = 0; i < use_count; ++i) {
    if (!loom_low_allocation_use_is_removed_block_arg_edge(uses[i], block,
                                                           arg_index)) {
      ++reload_count;
    }
  }
  return reload_count;
}

static bool loom_low_allocation_block_arg_plan(
    loom_module_t* module, const loom_op_t* function_op,
    const loom_low_allocation_spill_plan_t* plan, loom_block_t** out_block,
    uint16_t* out_arg_index, bool* out_is_entry_arg) {
  *out_block = NULL;
  *out_arg_index = 0;
  *out_is_entry_arg = false;

  if (plan->value_id >= module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, plan->value_id);
  if (!loom_value_is_block_arg(value)) {
    return false;
  }

  loom_block_t* block = loom_value_def_block(value);
  if (!block) {
    return false;
  }
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    if (loom_block_arg_id(block, i) == plan->value_id) {
      *out_block = block;
      *out_arg_index = i;
      break;
    }
  }
  if (!*out_block) {
    return false;
  }

  const loom_region_t* body = loom_low_function_const_body(function_op);
  *out_is_entry_arg = body && block == loom_region_const_entry_block(body);
  return true;
}

static iree_status_t loom_low_allocation_rebuild_br_without_arg(
    loom_module_t* module, loom_op_t* branch_op, uint16_t arg_index,
    iree_arena_allocator_t* arena) {
  if (!loom_low_br_isa(branch_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "expected low.br while removing block argument");
  }
  if (arg_index >= branch_op->operand_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "low.br payload index %u out of range for %u operand(s)",
        (unsigned)arg_index, (unsigned)branch_op->operand_count);
  }
  if (!branch_op->parent_block) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low.br is detached while removing payload");
  }

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
  IREE_RETURN_IF_ERROR(loom_low_br_build(&builder, loom_low_br_dest(branch_op),
                                         new_args, new_count,
                                         branch_op->location, &replacement_op));
  return loom_op_erase(module, branch_op);
}

static iree_status_t loom_low_allocation_insert_spill_store(
    loom_module_t* module, const loom_op_t* function_op,
    const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id) {
  loom_value_t* value = loom_module_value(module, plan->value_id);
  loom_builder_t builder;
  loom_op_t* spill_op = NULL;
  if (loom_value_is_block_arg(value)) {
    loom_block_t* block = loom_def_block(value->def);
    if (!block) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "spilled block argument has no defining block");
    }
    loom_builder_initialize(module, &module->arena, block, &builder);
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_set_block_arg_spill_insertion_point(
            &builder, function_op, block));
    return loom_low_spill_build(&builder, plan->value_id, storage_value_id, 0,
                                function_op->location, &spill_op);
  }

  loom_op_t* defining_op = loom_def_op(value->def);
  if (!defining_op) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "spilled op result has no defining op");
  }
  if (!defining_op->parent_block) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "spilled op result defining op is detached");
  }
  loom_builder_initialize(module, &module->arena, defining_op->parent_block,
                          &builder);
  loom_builder_set_after(&builder, defining_op);
  return loom_low_spill_build(&builder, plan->value_id, storage_value_id, 0,
                              defining_op->location, &spill_op);
}

static iree_status_t loom_low_allocation_insert_reload_for_use(
    loom_module_t* module, const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id, loom_use_t use) {
  loom_op_t* user_op = loom_use_user_op(use);
  uint16_t operand_index = loom_use_operand_index(use);
  if (!user_op->parent_block) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "spilled value user op is detached");
  }
  if (operand_index >= user_op->operand_count ||
      loom_op_const_operands(user_op)[operand_index] != plan->value_id) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "allocation spill plan is stale for value %u",
                            (unsigned)plan->value_id);
  }

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, user_op->parent_block,
                          &builder);
  loom_builder_set_before(&builder, user_op);
  loom_op_t* reload_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_reload_build(&builder, storage_value_id, 0,
                            loom_module_value_type(module, plan->value_id),
                            user_op->location, &reload_op));
  return loom_op_set_operand(module, user_op, operand_index,
                             loom_low_reload_result(reload_op));
}

static iree_status_t loom_low_allocation_materialize_block_arg_edges(
    loom_module_t* module, const loom_op_t* function_op,
    const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id, loom_block_t* block, uint16_t arg_index,
    uint32_t reload_count, iree_arena_allocator_t* arena,
    uint32_t* out_store_count) {
  *out_store_count = 0;
  loom_region_t* body = loom_low_function_body((loom_op_t*)function_op);
  if (!body) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low function has no body");
  }

  uint32_t store_count = 0;
  loom_block_t* predecessor_block = NULL;
  loom_region_for_each_block(body, predecessor_block) {
    loom_op_t* branch_op =
        (loom_op_t*)loom_block_const_last_op(predecessor_block);
    if (!branch_op || !loom_low_br_isa(branch_op) ||
        loom_low_br_dest(branch_op) != block) {
      continue;
    }
    if (arg_index >= branch_op->operand_count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low.br predecessor payload count is stale for spilled block "
          "argument");
    }

    const loom_value_id_t payload =
        loom_op_const_operands(branch_op)[arg_index];
    if (payload == LOOM_VALUE_ID_INVALID || payload >= module->values.count) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low.br predecessor payload is invalid for spilled block argument");
    }
    if (reload_count != 0 && payload != plan->value_id) {
      loom_builder_t builder;
      loom_builder_initialize(module, &module->arena, branch_op->parent_block,
                              &builder);
      loom_builder_set_before(&builder, branch_op);
      loom_op_t* spill_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_low_spill_build(&builder, payload, storage_value_id, 0,
                               branch_op->location, &spill_op));
      if (store_count == UINT32_MAX) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "materialized spill count overflow");
      }
      ++store_count;
    }

    IREE_RETURN_IF_ERROR(loom_low_allocation_rebuild_br_without_arg(
        module, branch_op, arg_index, arena));
  }

  if (reload_count != 0 && store_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "spilled non-entry block argument has reloads but no incoming value "
        "to store");
  }
  if (plan->store_count != store_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation spill plan store count is stale for value %u",
        (unsigned)plan->value_id);
  }

  IREE_RETURN_IF_ERROR(loom_block_remove_arg(module, block, arg_index));
  *out_store_count = store_count;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_materialize_one_spill_plan(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id, iree_diagnostic_emitter_t emitter,
    iree_arena_allocator_t* arena,
    loom_low_allocation_materialization_result_t* result) {
  loom_use_t* uses = NULL;
  uint32_t use_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_snapshot_value_uses(
      module, plan->value_id, arena, &uses, &use_count));

  loom_block_t* block_arg_block = NULL;
  uint16_t block_arg_index = 0;
  bool block_arg_is_entry = false;
  const bool is_block_arg = loom_low_allocation_block_arg_plan(
      module, table->function_op, plan, &block_arg_block, &block_arg_index,
      &block_arg_is_entry);
  uint32_t reload_count = use_count;
  if (is_block_arg && !block_arg_is_entry) {
    reload_count = loom_low_allocation_count_materialized_reloads(
        uses, use_count, block_arg_block, block_arg_index);
  }
  if (reload_count != plan->reload_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation spill plan reload count is stale for value %u",
        (unsigned)plan->value_id);
  }
  uint32_t expected_store_count = reload_count > 0 ? 1u : 0u;
  if (is_block_arg && !block_arg_is_entry) {
    expected_store_count = plan->store_count;
  }
  if (plan->store_count != expected_store_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation spill plan store count is stale for value %u",
        (unsigned)plan->value_id);
  }

  if (plan->store_count > 0 && (!is_block_arg || block_arg_is_entry)) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_insert_spill_store(
        module, table->function_op, plan, storage_value_id));
    if (result->spill_count == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized spill count overflow");
    }
    ++result->spill_count;
  }

  for (uint32_t i = 0; i < use_count; ++i) {
    if (is_block_arg && !block_arg_is_entry &&
        loom_low_allocation_use_is_removed_block_arg_edge(
            uses[i], block_arg_block, block_arg_index)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_allocation_insert_reload_for_use(
        module, plan, storage_value_id, uses[i]));
    if (result->reload_count == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized reload count overflow");
    }
    ++result->reload_count;
  }
  if (is_block_arg && !block_arg_is_entry) {
    uint32_t store_count = 0;
    IREE_RETURN_IF_ERROR(loom_low_allocation_materialize_block_arg_edges(
        module, table->function_op, plan, storage_value_id, block_arg_block,
        block_arg_index, reload_count, arena, &store_count));
    if (store_count > UINT32_MAX - result->spill_count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized spill count overflow");
    }
    result->spill_count += store_count;
  }
  return loom_low_allocation_emit_materialized_spill(table, plan,
                                                     storage_value_id, emitter);
}

iree_status_t loom_low_allocation_materialize_spills(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_allocation_materialization_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_allocation_materialization_result_t* out_result) {
  loom_low_allocation_materialization_result_t result = {0};
  if (out_result) *out_result = result;

  if (table->module != module) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "allocation table belongs to a different module");
  }
  if (table->spill_plan_count == 0) return iree_ok_status();

  const bool allow_existing_storage_traffic =
      options && options->allow_existing_storage_traffic;
  const bool emit_spill_diagnostics =
      options && options->emit_spill_diagnostics;
  const iree_diagnostic_emitter_t emitter =
      options ? options->emitter : (iree_diagnostic_emitter_t){0};
  if (!allow_existing_storage_traffic) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_verify_no_existing_storage_traffic(
        table->function_op));
  }
  iree_host_size_t spill_plan_count = table->spill_plan_count;
  if (options && options->max_spill_plan_count > 0 &&
      options->max_spill_plan_count < spill_plan_count) {
    spill_plan_count = options->max_spill_plan_count;
  }
  if (spill_plan_count == 0) return iree_ok_status();
  if (spill_plan_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "materialized storage count overflow");
  }

  IREE_RETURN_IF_ERROR(
      loom_low_allocation_validate_supported_spill_storage_spaces(
          table, spill_plan_count, options, &result));
  if (result.error_count != 0) {
    if (out_result) *out_result = result;
    return iree_ok_status();
  }

  loom_low_materialized_spill_slot_t* slots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, spill_plan_count, sizeof(*slots), (void**)&slots));
  memset(slots, 0, spill_plan_count * sizeof(*slots));

  IREE_RETURN_IF_ERROR(loom_low_allocation_insert_storage_reserves(
      module, table, spill_plan_count, arena, slots));
  result.storage_count = (uint32_t)spill_plan_count;

  for (iree_host_size_t i = 0; i < spill_plan_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_materialize_one_spill_plan(
        module, table, &table->spill_plans[i], slots[i].storage_value_id,
        emit_spill_diagnostics ? emitter : (iree_diagnostic_emitter_t){0},
        arena, &result));
  }

  if (out_result) *out_result = result;
  return iree_ok_status();
}
