// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation_materialization.h"

#include <stdio.h>
#include <string.h>

#include "loom/codegen/low/allocation/spill_plan.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/type_registry.h"

typedef struct loom_low_materialized_spill_slot_t {
  // SSA value ID produced by the generated low.storage.reserve op.
  loom_value_id_t storage_value_id;
} loom_low_materialized_spill_slot_t;

typedef struct loom_low_materialized_traffic_t {
  // Number of materialized traffic ops inserted.
  uint32_t count;
  // Byte traffic represented by the inserted ops.
  uint64_t bytes;
} loom_low_materialized_traffic_t;

typedef struct loom_low_slice_reload_group_t {
  // Block containing slice uses of the spilled value.
  loom_block_t* block;
  // First slice use in |block| receiving a full-width reload.
  loom_op_t* first_slice_op;
  // Full-width reload shared by slice uses in |block|.
  loom_value_id_t full_reload_value_id;
  // Number of slice uses in |block|.
  uint32_t slice_count;
  // Byte traffic if each slice use reloads only its projected unit.
  uint64_t narrow_reload_bytes;
  // Whether slice uses in |block| share one full-width reload.
  bool use_full_reload;
} loom_low_slice_reload_group_t;

typedef struct loom_low_slice_reload_plan_t {
  // Slice-use groups indexed by entries in |group_indices_by_use|.
  loom_low_slice_reload_group_t* groups;
  // Number of records in |groups|.
  uint32_t group_count;
  // Group index for each snapshotted use, or UINT32_MAX for other uses.
  uint32_t* group_indices_by_use;
  // Byte width of each classified slice reload.
  uint32_t unit_byte_size;
} loom_low_slice_reload_plan_t;

typedef struct loom_low_allocation_storage_prefix_t {
  // Function body containing the allocation table's IR snapshot.
  loom_region_t* body;
  // Function entry block receiving generated storage reservations.
  loom_block_t* entry_block;
  // Last existing storage reservation or ABI preamble op in the entry block.
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
      .value_class = assignment->value_class,
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
      loom_param_string(loom_low_diagnostic_value_class_name(
          table->target.descriptor_set, assignment->value_class)),
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

static loom_storage_space_t loom_low_allocation_map_slot_space(
    loom_low_spill_slot_space_t slot_space) {
  switch (slot_space) {
    case LOOM_LOW_SPILL_SLOT_SPACE_STACK:
      return LOOM_STORAGE_SPACE_STACK;
    case LOOM_LOW_SPILL_SLOT_SPACE_SCRATCH:
      return LOOM_STORAGE_SPACE_SCRATCH;
    case LOOM_LOW_SPILL_SLOT_SPACE_PRIVATE:
      return LOOM_STORAGE_SPACE_PRIVATE;
    case LOOM_LOW_SPILL_SLOT_SPACE_LDS:
      return LOOM_STORAGE_SPACE_WORKGROUP;
    default:
      IREE_CHECK_UNREACHABLE("unknown generated spill slot space");
      return LOOM_STORAGE_SPACE_COUNT_;
  }
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
      loom_param_string(loom_low_diagnostic_value_class_name(
          table->target.descriptor_set, assignment->value_class)),
      loom_param_string(loom_low_spill_slot_space_name(plan->slot_space)),
      loom_param_string(loom_low_storage_type_space_name(storage_space)),
      loom_param_string_list(supported_storage_space_names,
                             supported_storage_space_count),
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
    const loom_storage_space_t storage_space =
        loom_low_allocation_map_slot_space(plan->slot_space);
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

static bool loom_low_allocation_entry_preamble_op(const loom_op_t* op) {
  return loom_low_live_in_isa(op) || loom_low_resource_isa(op);
}

static bool loom_low_allocation_entry_storage_prefix_op(const loom_op_t* op) {
  return loom_low_allocation_entry_preamble_op(op) ||
         loom_low_storage_reserve_isa(op) || loom_low_spill_isa(op);
}

static void loom_low_allocation_analyze_storage_prefix(
    const loom_op_t* function_op,
    loom_low_allocation_storage_prefix_t* out_prefix) {
  loom_region_t* body = loom_low_function_body((loom_op_t*)function_op);
  loom_block_t* entry_block = loom_region_entry_block(body);
  loom_op_t* preamble_anchor = NULL;
  loom_op_t* last_storage_reserve = NULL;
  loom_op_t* entry_traffic_insertion_anchor = NULL;
  iree_host_size_t storage_reserve_count = 0;
  bool entry_prefix_open = true;
  loom_op_t* op = NULL;
  loom_block_for_each_op(entry_block, op) {
    if (entry_prefix_open) {
      if (loom_low_allocation_entry_storage_prefix_op(op)) {
        entry_traffic_insertion_anchor = op;
      } else {
        entry_prefix_open = false;
      }
    }
    if (loom_low_storage_reserve_isa(op)) {
      ++storage_reserve_count;
      last_storage_reserve = op;
    } else if (loom_low_allocation_entry_preamble_op(op)) {
      preamble_anchor = op;
    }
  }
  loom_op_t* storage_insertion_anchor =
      last_storage_reserve ? last_storage_reserve : preamble_anchor;
  *out_prefix = (loom_low_allocation_storage_prefix_t){
      .body = body,
      .entry_block = entry_block,
      .storage_insertion_anchor = storage_insertion_anchor,
      .entry_traffic_insertion_anchor = entry_traffic_insertion_anchor,
      .storage_reserve_count = storage_reserve_count,
      .generated_storage_extends_entry_prefix =
          storage_insertion_anchor == entry_traffic_insertion_anchor,
  };
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
  if (!loom_low_allocation_entry_preamble_op(op)) {
    return false;
  }
  const loom_region_t* body = loom_low_function_const_body(function_op);
  return op->parent_block == loom_region_const_entry_block(body);
}

static iree_status_t loom_low_allocation_insert_storage_reserves(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_low_allocation_storage_prefix_t* storage_prefix,
    iree_host_size_t spill_plan_count, iree_arena_allocator_t* arena,
    loom_low_materialized_spill_slot_t* slots) {
  const iree_string_view_t function_name =
      loom_low_diagnostic_function_name(module, table->function_op);
  const iree_host_size_t storage_name_start =
      storage_prefix->storage_reserve_count;
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, NULL, &builder);
  loom_low_allocation_set_storage_insertion_point(
      &builder, (loom_op_t*)table->function_op, storage_prefix);
  for (iree_host_size_t i = 0; i < spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* plan = &table->spill_plans[i];
    const loom_storage_space_t storage_space =
        loom_low_allocation_map_slot_space(plan->slot_space);

    loom_op_t* reserve_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_storage_reserve_build(
        &builder, (int64_t)plan->byte_size, (int64_t)plan->byte_alignment,
        loom_type_storage(storage_space), table->function_op->location,
        &reserve_op));
    loom_value_id_t storage_value_id =
        loom_low_storage_reserve_storage(reserve_op);
    if (storage_prefix->generated_storage_extends_entry_prefix) {
      storage_prefix->entry_traffic_insertion_anchor = reserve_op;
    }
    const iree_host_size_t storage_name_index = storage_name_start + i;
    IREE_RETURN_IF_ERROR(loom_low_allocation_set_storage_value_name(
        module, function_name, storage_name_index, storage_value_id, arena));
    loom_builder_set_after(&builder, reserve_op);
    slots[i] = (loom_low_materialized_spill_slot_t){
        .storage_value_id = storage_value_id,
    };
  }

  return iree_ok_status();
}

static bool loom_low_allocation_use_is_removed_block_arg_edge(
    loom_use_t use, const loom_block_t* block, uint16_t arg_index) {
  const loom_op_t* user_op = loom_use_user_op(use);
  const uint16_t operand_index = loom_use_operand_index(use);
  return loom_low_br_isa(user_op) && loom_low_br_dest(user_op) == block &&
         operand_index == arg_index;
}

static iree_status_t loom_low_allocation_snapshot_reload_uses(
    loom_module_t* module, loom_value_id_t value_id,
    const loom_block_t* removed_block_arg_block, uint16_t removed_arg_index,
    iree_arena_allocator_t* arena, loom_use_t** out_uses,
    uint32_t* out_use_count) {
  loom_value_t* value = loom_module_value(module, value_id);
  *out_uses = NULL;
  *out_use_count = 0;
  if (value->use_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, value->use_count, sizeof(**out_uses), (void**)out_uses));
  const loom_use_t* value_uses = loom_value_uses(value);
  if (!removed_block_arg_block) {
    memcpy(*out_uses, value_uses,
           (iree_host_size_t)value->use_count * sizeof(**out_uses));
    *out_use_count = value->use_count;
    return iree_ok_status();
  }

  uint32_t use_count = 0;
  for (uint32_t i = 0; i < value->use_count; ++i) {
    if (loom_low_allocation_use_is_removed_block_arg_edge(
            value_uses[i], removed_block_arg_block, removed_arg_index)) {
      continue;
    }
    (*out_uses)[use_count++] = value_uses[i];
  }
  *out_use_count = use_count;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_rebuild_br_without_arg(
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
  IREE_RETURN_IF_ERROR(loom_low_rebuild_br(&builder, branch_op, new_args,
                                           new_count, &replacement_op));
  return loom_op_erase(module, branch_op);
}

static iree_status_t loom_low_allocation_insert_spill_store(
    loom_module_t* module, const loom_op_t* function_op,
    loom_low_allocation_storage_prefix_t* storage_prefix,
    const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id) {
  loom_value_t* value = loom_module_value(module, plan->value_id);
  loom_builder_t builder;
  loom_op_t* spill_op = NULL;
  bool extends_entry_prefix = false;
  loom_location_id_t location = function_op->location;
  if (loom_value_is_block_arg(value)) {
    loom_block_t* block = loom_value_def_block(value);
    loom_builder_initialize(module, &module->arena, block, &builder);
    loom_low_allocation_set_spill_insertion_point(&builder, storage_prefix,
                                                  block);
    extends_entry_prefix = block == storage_prefix->entry_block;
  } else {
    loom_op_t* defining_op = loom_value_def_op(value);
    loom_builder_initialize(module, &module->arena, defining_op->parent_block,
                            &builder);
    if (loom_low_allocation_defines_entry_preamble_value(function_op,
                                                         defining_op)) {
      loom_low_allocation_set_spill_insertion_point(&builder, storage_prefix,
                                                    defining_op->parent_block);
      extends_entry_prefix = true;
    } else {
      loom_builder_set_after(&builder, defining_op);
    }
    location = defining_op->location;
  }
  IREE_RETURN_IF_ERROR(loom_low_spill_build(
      &builder, plan->value_id, storage_value_id, 0, location, &spill_op));
  if (extends_entry_prefix) {
    storage_prefix->entry_traffic_insertion_anchor = spill_op;
  }
  return iree_ok_status();
}

static bool loom_low_allocation_slice_reload_use(
    const loom_low_allocation_assignment_t* assignment,
    const loom_low_allocation_spill_plan_t* plan, const loom_region_t* body,
    loom_use_t use, loom_op_t** out_slice_op, uint16_t* out_block_index,
    uint32_t* out_unit_byte_size) {
  *out_slice_op = NULL;
  *out_block_index = 0;
  *out_unit_byte_size = 0;
  loom_op_t* user_op = loom_use_user_op(use);
  int64_t reload_offset = 0;
  if (!loom_low_allocation_spill_plan_slice_reload_byte_offset(
          assignment, plan->byte_size, user_op, loom_use_operand_index(use),
          out_unit_byte_size, &reload_offset)) {
    return false;
  }
  (void)reload_offset;
  uint16_t block_index = 0;
  if (!loom_region_try_block_index(body, user_op->parent_block, &block_index)) {
    return false;
  }
  *out_slice_op = user_op;
  *out_block_index = block_index;
  return true;
}

static iree_status_t loom_low_allocation_insert_slice_reload(
    loom_module_t* module, const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id, loom_op_t* slice_op,
    uint32_t unit_byte_size) {
  const int64_t reload_offset =
      loom_low_slice_offset(slice_op) * (int64_t)unit_byte_size;
  const loom_value_id_t slice_result = loom_low_slice_result(slice_op);
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, slice_op->parent_block,
                          &builder);
  loom_builder_set_before(&builder, slice_op);
  loom_op_t* reload_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_reload_build(&builder, storage_value_id, reload_offset,
                            loom_module_value_type(module, slice_result),
                            slice_op->location, &reload_op));
  IREE_RETURN_IF_ERROR(loom_value_replace_all_uses_with(
      module, slice_result, loom_low_reload_result(reload_op)));
  return loom_op_erase(module, slice_op);
}

static iree_status_t loom_low_allocation_insert_full_slice_reload(
    loom_module_t* module, const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id, loom_low_slice_reload_group_t* group) {
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, group->block, &builder);
  loom_builder_set_before(&builder, group->first_slice_op);
  loom_op_t* reload_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_reload_build(&builder, storage_value_id, 0,
                            loom_module_value_type(module, plan->value_id),
                            group->first_slice_op->location, &reload_op));
  group->full_reload_value_id = loom_low_reload_result(reload_op);
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_prepare_slice_reloads(
    loom_module_t* module, const loom_low_allocation_assignment_t* assignment,
    const loom_low_allocation_spill_plan_t* plan, const loom_region_t* body,
    loom_value_id_t storage_value_id, const loom_use_t* uses,
    uint32_t use_count, iree_arena_allocator_t* arena,
    loom_low_slice_reload_plan_t* out_reload_plan,
    loom_low_materialized_traffic_t* out_reload_traffic) {
  *out_reload_plan = (loom_low_slice_reload_plan_t){0};
  *out_reload_traffic = (loom_low_materialized_traffic_t){0};
  if (use_count == 0) return iree_ok_status();
  if (assignment->unit_count <= 1 ||
      plan->byte_size % assignment->unit_count != 0) {
    return iree_ok_status();
  }

  uint32_t* group_indices_by_block = NULL;
  loom_low_slice_reload_group_t* groups = NULL;
  uint32_t* group_indices_by_use = NULL;
  uint32_t group_count = 0;
  for (uint32_t i = 0; i < use_count; ++i) {
    loom_op_t* slice_op = NULL;
    uint16_t block_index = 0;
    uint32_t unit_byte_size = 0;
    if (!loom_low_allocation_slice_reload_use(assignment, plan, body, uses[i],
                                              &slice_op, &block_index,
                                              &unit_byte_size)) {
      continue;
    }
    if (!groups) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, body->block_count, sizeof(*group_indices_by_block),
          (void**)&group_indices_by_block));
      memset(group_indices_by_block, 0xFF,
             body->block_count * sizeof(*group_indices_by_block));
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, body->block_count, sizeof(*groups), (void**)&groups));
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, use_count, sizeof(*group_indices_by_use),
          (void**)&group_indices_by_use));
      memset(group_indices_by_use, 0xFF,
             use_count * sizeof(*group_indices_by_use));
    }
    uint32_t group_index = group_indices_by_block[block_index];
    if (group_index == UINT32_MAX) {
      group_index = group_count++;
      group_indices_by_block[block_index] = group_index;
      groups[group_index] = (loom_low_slice_reload_group_t){
          .block = body->blocks[block_index],
          .first_slice_op = slice_op,
          .full_reload_value_id = LOOM_VALUE_ID_INVALID,
      };
    }
    group_indices_by_use[i] = group_index;
    ++groups[group_index].slice_count;
    groups[group_index].narrow_reload_bytes += unit_byte_size;
  }
  if (!groups) return iree_ok_status();

  loom_low_materialized_traffic_t reload_traffic = {0};
  for (uint32_t i = 0; i < group_count; ++i) {
    loom_low_slice_reload_group_t* group = &groups[i];
    group->use_full_reload =
        loom_low_allocation_spill_plan_use_full_slice_reload(
            group->slice_count, group->narrow_reload_bytes, plan->byte_size);
    if (!group->use_full_reload) continue;
    IREE_RETURN_IF_ERROR(loom_low_allocation_insert_full_slice_reload(
        module, plan, storage_value_id, group));
    ++reload_traffic.count;
    reload_traffic.bytes += plan->byte_size;
  }

  *out_reload_plan = (loom_low_slice_reload_plan_t){
      .groups = groups,
      .group_count = group_count,
      .group_indices_by_use = group_indices_by_use,
      .unit_byte_size = plan->byte_size / assignment->unit_count,
  };
  *out_reload_traffic = reload_traffic;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_insert_full_reload_for_use(
    loom_module_t* module, const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id, loom_use_t use) {
  loom_op_t* user_op = loom_use_user_op(use);
  const uint16_t operand_index = loom_use_operand_index(use);
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

static iree_status_t loom_low_allocation_insert_reloads_for_uses(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id, const loom_use_t* uses,
    uint32_t use_count, iree_arena_allocator_t* arena,
    loom_low_materialized_traffic_t* out_reload_traffic) {
  *out_reload_traffic = (loom_low_materialized_traffic_t){0};
  loom_low_materialized_traffic_t reload_traffic = {0};
  loom_low_slice_reload_plan_t slice_reload_plan = {0};
  const loom_low_allocation_assignment_t* assignment =
      loom_low_allocation_spill_plan_assignment(table, plan);
  const loom_region_t* body = loom_low_function_const_body(table->function_op);
  if (use_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_prepare_slice_reloads(
        module, assignment, plan, body, storage_value_id, uses, use_count,
        arena, &slice_reload_plan, &reload_traffic));
  }
  if (!slice_reload_plan.group_indices_by_use) {
    for (uint32_t i = 0; i < use_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_insert_full_reload_for_use(
          module, plan, storage_value_id, uses[i]));
    }
    reload_traffic.count = use_count;
    reload_traffic.bytes = (uint64_t)use_count * plan->byte_size;
    *out_reload_traffic = reload_traffic;
    return iree_ok_status();
  }

  for (uint32_t i = 0; i < use_count; ++i) {
    const uint32_t group_index = slice_reload_plan.group_indices_by_use[i];
    if (group_index != UINT32_MAX) {
      IREE_ASSERT_LT(group_index, slice_reload_plan.group_count);
      loom_op_t* slice_op = loom_use_user_op(uses[i]);
      if (slice_reload_plan.groups[group_index].use_full_reload) {
        IREE_RETURN_IF_ERROR(loom_op_set_operand(
            module, slice_op, /*operand_index=*/0,
            slice_reload_plan.groups[group_index].full_reload_value_id));
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_low_allocation_insert_slice_reload(
          module, plan, storage_value_id, slice_op,
          slice_reload_plan.unit_byte_size));
      ++reload_traffic.count;
      reload_traffic.bytes += slice_reload_plan.unit_byte_size;
      continue;
    }

    IREE_RETURN_IF_ERROR(loom_low_allocation_insert_full_reload_for_use(
        module, plan, storage_value_id, uses[i]));
    ++reload_traffic.count;
    reload_traffic.bytes += plan->byte_size;
  }
  *out_reload_traffic = reload_traffic;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_materialize_block_arg_edges(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id, loom_block_t* block, uint16_t arg_index,
    uint32_t reload_count, iree_arena_allocator_t* arena,
    loom_low_materialized_traffic_t* out_store_traffic) {
  *out_store_traffic = (loom_low_materialized_traffic_t){0};

  loom_low_materialized_traffic_t store_traffic = {0};
  const loom_cfg_block_index_span_t predecessors = loom_cfg_graph_predecessors(
      &table->cfg_graph, loom_block_region_index(block));
  for (iree_host_size_t i = 0; i < predecessors.count; ++i) {
    loom_block_t* predecessor_block =
        (loom_block_t*)table->cfg_graph.blocks[predecessors.values[i]].block;
    loom_op_t* branch_op =
        (loom_op_t*)loom_block_const_last_op(predecessor_block);

    const loom_value_id_t payload =
        loom_op_const_operands(branch_op)[arg_index];
    if (reload_count != 0 && payload != plan->value_id) {
      loom_builder_t builder;
      loom_builder_initialize(module, &module->arena, branch_op->parent_block,
                              &builder);
      loom_builder_set_before(&builder, branch_op);
      loom_op_t* spill_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_low_spill_build(&builder, payload, storage_value_id, 0,
                               branch_op->location, &spill_op));
      ++store_traffic.count;
      store_traffic.bytes += plan->byte_size;
    }

    IREE_RETURN_IF_ERROR(loom_low_allocation_rebuild_br_without_arg(
        module, branch_op, arg_index, arena));
  }

  if (reload_count != 0 && store_traffic.count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "spilled non-entry block argument has reloads but no incoming value "
        "to store");
  }
  IREE_RETURN_IF_ERROR(loom_block_remove_arg(module, block, arg_index));
  *out_store_traffic = store_traffic;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_materialize_one_spill_plan(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_low_allocation_storage_prefix_t* storage_prefix,
    const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id, iree_arena_allocator_t* arena,
    loom_low_allocation_materialization_result_t* result,
    loom_low_allocation_materialized_spill_flags_t* out_record_flags) {
  *out_record_flags = 0;
  loom_block_t* block_arg_block = NULL;
  uint16_t block_arg_index = 0;
  const loom_value_t* spill_value = loom_module_value(module, plan->value_id);
  const bool is_block_arg = loom_value_is_block_arg(spill_value);
  if (is_block_arg) {
    block_arg_block = loom_value_def_block(spill_value);
    block_arg_index = loom_value_def_index(spill_value);
    *out_record_flags |=
        LOOM_LOW_ALLOCATION_MATERIALIZED_SPILL_FLAG_VALUE_WAS_BLOCK_ARGUMENT;
  }
  const bool block_arg_is_entry =
      block_arg_block == storage_prefix->entry_block;
  const loom_block_t* removed_block_arg_block =
      is_block_arg && !block_arg_is_entry ? block_arg_block : NULL;
  loom_use_t* uses = NULL;
  uint32_t use_count = 0;
  IREE_RETURN_IF_ERROR(loom_low_allocation_snapshot_reload_uses(
      module, plan->value_id, removed_block_arg_block, block_arg_index, arena,
      &uses, &use_count));
  const bool needs_definition_store =
      use_count > 0 && (!is_block_arg || block_arg_is_entry);

  if (needs_definition_store) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_insert_spill_store(
        module, table->function_op, storage_prefix, plan, storage_value_id));
    if (result->spill_count == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized spill count overflow");
    }
    if (plan->byte_size > UINT64_MAX - result->spill_bytes) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized spill byte count overflow");
    }
    ++result->spill_count;
    result->spill_bytes += plan->byte_size;
  }

  loom_low_materialized_traffic_t inserted_reload_traffic = {0};
  IREE_RETURN_IF_ERROR(loom_low_allocation_insert_reloads_for_uses(
      module, table, plan, storage_value_id, uses, use_count, arena,
      &inserted_reload_traffic));
  if (inserted_reload_traffic.count > UINT32_MAX - result->reload_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "materialized reload count overflow");
  }
  if (inserted_reload_traffic.bytes > UINT64_MAX - result->reload_bytes) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "materialized reload byte count overflow");
  }
  result->reload_count += inserted_reload_traffic.count;
  result->reload_bytes += inserted_reload_traffic.bytes;
  if (is_block_arg && !block_arg_is_entry) {
    loom_low_materialized_traffic_t inserted_store_traffic = {0};
    IREE_RETURN_IF_ERROR(loom_low_allocation_materialize_block_arg_edges(
        module, table, plan, storage_value_id, block_arg_block, block_arg_index,
        inserted_reload_traffic.count, arena, &inserted_store_traffic));
    if (inserted_store_traffic.count > UINT32_MAX - result->spill_count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized spill count overflow");
    }
    if (inserted_store_traffic.bytes > UINT64_MAX - result->spill_bytes) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized spill byte count overflow");
    }
    result->spill_count += inserted_store_traffic.count;
    result->spill_bytes += inserted_store_traffic.bytes;
  }
  return iree_ok_status();
}

iree_status_t loom_low_allocation_materialize_spills(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_materialization_options_t* options,
    iree_arena_allocator_t* arena,
    loom_low_allocation_materialization_result_t* out_result) {
  loom_module_t* module = table->module;
  loom_low_allocation_materialization_result_t result = {0};
  if (out_result) *out_result = result;

  if (table->spill_plan_count == 0) return iree_ok_status();

  const bool emit_spill_diagnostics =
      options && options->emit_spill_diagnostics;
  const bool record_materialized_spills =
      options && options->record_materialized_spills;
  const iree_diagnostic_emitter_t emitter =
      options ? options->emitter : (iree_diagnostic_emitter_t){0};
  loom_low_allocation_storage_prefix_t storage_prefix;
  loom_low_allocation_analyze_storage_prefix(table->function_op,
                                             &storage_prefix);
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
  loom_low_allocation_materialized_spill_t* materialized_spills = NULL;
  if (record_materialized_spills) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, spill_plan_count, sizeof(*materialized_spills),
        (void**)&materialized_spills));
  }

  IREE_RETURN_IF_ERROR(loom_low_allocation_insert_storage_reserves(
      module, table, &storage_prefix, spill_plan_count, arena, slots));
  result.storage_count = (uint32_t)spill_plan_count;

  for (iree_host_size_t i = 0; i < spill_plan_count; ++i) {
    const uint32_t prior_spill_count = result.spill_count;
    const uint32_t prior_reload_count = result.reload_count;
    const uint64_t prior_spill_bytes = result.spill_bytes;
    const uint64_t prior_reload_bytes = result.reload_bytes;
    loom_low_allocation_materialized_spill_flags_t record_flags = 0;
    IREE_RETURN_IF_ERROR(loom_low_allocation_materialize_one_spill_plan(
        module, table, &storage_prefix, &table->spill_plans[i],
        slots[i].storage_value_id, arena, &result, &record_flags));
    const loom_low_materialized_traffic_t materialized_store_traffic = {
        .count = result.spill_count - prior_spill_count,
        .bytes = result.spill_bytes - prior_spill_bytes,
    };
    const loom_low_materialized_traffic_t materialized_reload_traffic = {
        .count = result.reload_count - prior_reload_count,
        .bytes = result.reload_bytes - prior_reload_bytes,
    };
    const uint64_t byte_size = table->spill_plans[i].byte_size;
    result.storage_bytes += byte_size;
    if (record_materialized_spills) {
      loom_low_allocation_record_materialized_spill(
          table, &table->spill_plans[i], materialized_store_traffic,
          materialized_reload_traffic, record_flags, &materialized_spills[i]);
    }
    if (emit_spill_diagnostics) {
      IREE_RETURN_IF_ERROR(loom_low_allocation_emit_materialized_spill(
          table, &table->spill_plans[i], slots[i].storage_value_id,
          materialized_store_traffic, materialized_reload_traffic, emitter));
    }
  }

  result.materialized_spills = materialized_spills;
  result.materialized_spill_count =
      record_materialized_spills ? spill_plan_count : 0;
  if (out_result) *out_result = result;
  return iree_ok_status();
}
