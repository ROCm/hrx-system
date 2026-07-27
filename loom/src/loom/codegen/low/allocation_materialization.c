// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/allocation_materialization.h"

#include <stdio.h>
#include <string.h>

#include "loom/codegen/low/allocation/spill_plan.h"
#include "loom/codegen/low/diagnostics.h"
#include "loom/codegen/low/function.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"

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
  // Number of slice uses in |block|.
  uint32_t slice_count;
  // Byte traffic if each slice use reloads only its projected unit.
  uint64_t narrow_reload_bytes;
} loom_low_slice_reload_group_t;

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

static void loom_low_allocation_record_materialized_spill(
    const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plan,
    loom_low_materialized_traffic_t store_traffic,
    loom_low_materialized_traffic_t reload_traffic,
    loom_low_allocation_materialized_spill_flags_t flags,
    loom_low_allocation_materialized_spill_t* record) {
  IREE_ASSERT_LT(plan->assignment_index, table->assignment_count);
  const loom_low_allocation_assignment_t* assignment =
      &table->assignments[plan->assignment_index];
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
  if (plan->assignment_index >= table->assignment_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation spill plan references an out-of-range assignment");
  }
  const loom_low_allocation_assignment_t* assignment =
      &table->assignments[plan->assignment_index];
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
      loom_param_string(
          iree_make_cstring_view(loom_storage_space_name(storage_space))),
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
  if (!op || !loom_low_allocation_entry_preamble_op(op)) {
    return false;
  }
  const loom_region_t* body = loom_low_function_const_body(function_op);
  const loom_block_t* entry_block =
      body ? loom_region_const_entry_block(body) : NULL;
  return op->parent_block == entry_block;
}

static iree_status_t loom_low_allocation_insert_storage_reserves(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    loom_low_allocation_storage_prefix_t* storage_prefix,
    iree_host_size_t spill_plan_count, iree_arena_allocator_t* arena,
    loom_low_materialized_spill_slot_t* slots) {
  loom_symbol_ref_t function_ref = loom_low_function_callee(table->function_op);
  const iree_host_size_t storage_name_start =
      storage_prefix->storage_reserve_count;
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, NULL, &builder);
  loom_low_allocation_set_storage_insertion_point(
      &builder, (loom_op_t*)table->function_op, storage_prefix);
  for (iree_host_size_t i = 0; i < spill_plan_count; ++i) {
    const loom_low_allocation_spill_plan_t* plan = &table->spill_plans[i];
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
    if (storage_prefix->generated_storage_extends_entry_prefix) {
      storage_prefix->entry_traffic_insertion_anchor = reserve_op;
    }
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

static bool loom_low_allocation_operand_is_removed_block_arg_edge(
    const loom_op_t* user_op, uint16_t operand_index, const loom_block_t* block,
    uint16_t arg_index) {
  return block && user_op && loom_low_br_isa(user_op) &&
         loom_low_br_dest(user_op) == block && operand_index == arg_index;
}

static bool loom_low_allocation_use_is_removed_block_arg_edge(
    loom_use_t use, const loom_block_t* block, uint16_t arg_index) {
  const loom_op_t* user_op = loom_use_user_op(use);
  const uint16_t operand_index = loom_use_operand_index(use);
  return loom_low_allocation_operand_is_removed_block_arg_edge(
      user_op, operand_index, block, arg_index);
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
    loom_low_allocation_storage_prefix_t* storage_prefix,
    const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id) {
  loom_value_t* value = loom_module_value(module, plan->value_id);
  loom_builder_t builder;
  loom_op_t* spill_op = NULL;
  bool extends_entry_prefix = false;
  loom_location_id_t location = function_op->location;
  if (loom_value_is_block_arg(value)) {
    loom_block_t* block = loom_def_block(value->def);
    if (!block) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "spilled block argument has no defining block");
    }
    loom_builder_initialize(module, &module->arena, block, &builder);
    loom_low_allocation_set_spill_insertion_point(&builder, storage_prefix,
                                                  block);
    extends_entry_prefix = block == storage_prefix->entry_block;
  } else {
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

static iree_status_t loom_low_allocation_validate_spill_use(
    const loom_op_t* user_op, uint16_t operand_index,
    const loom_low_allocation_spill_plan_t* plan) {
  if (!user_op) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "spilled value use has no user op");
  }
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
  if (user_op == NULL || user_op->parent_block == NULL ||
      loom_use_operand_index(use) >= user_op->operand_count ||
      loom_op_const_operands(user_op)[loom_use_operand_index(use)] !=
          plan->value_id) {
    return false;
  }
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

static iree_status_t loom_low_allocation_try_insert_reload_for_slice_use(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id, loom_op_t* slice_op,
    uint16_t operand_index,
    loom_low_materialized_traffic_t* out_reload_traffic) {
  *out_reload_traffic = (loom_low_materialized_traffic_t){0};
  if (!loom_low_slice_isa(slice_op) || operand_index != 0) {
    return iree_ok_status();
  }
  if (plan->assignment_index >= table->assignment_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation spill plan references an out-of-range assignment");
  }
  const loom_low_allocation_assignment_t* assignment =
      &table->assignments[plan->assignment_index];
  int64_t reload_offset = 0;
  uint32_t unit_byte_size = 0;
  if (!loom_low_allocation_spill_plan_slice_reload_byte_offset(
          assignment, plan->byte_size, slice_op, operand_index, &unit_byte_size,
          &reload_offset)) {
    return iree_ok_status();
  }

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
  IREE_RETURN_IF_ERROR(loom_op_erase(module, slice_op));
  *out_reload_traffic = (loom_low_materialized_traffic_t){
      .count = 1,
      .bytes = unit_byte_size,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_insert_dense_slice_reload_for_group(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plan, const loom_region_t* body,
    loom_value_id_t storage_value_id, const loom_use_t* uses,
    uint32_t use_count, const loom_low_slice_reload_group_t* group,
    uint8_t* handled_use_flags,
    loom_low_materialized_traffic_t* out_reload_traffic) {
  *out_reload_traffic = (loom_low_materialized_traffic_t){0};

  const loom_low_allocation_assignment_t* assignment =
      &table->assignments[plan->assignment_index];
  loom_op_t* first_slice_op = NULL;
  loom_op_t* op = NULL;
  loom_block_for_each_op(group->block, op) {
    uint32_t unit_byte_size = 0;
    int64_t reload_offset = 0;
    if (loom_low_slice_isa(op) && loom_low_slice_source(op) == plan->value_id &&
        loom_low_allocation_spill_plan_slice_reload_byte_offset(
            assignment, plan->byte_size, op, /*operand_index=*/0,
            &unit_byte_size, &reload_offset)) {
      first_slice_op = op;
      break;
    }
    (void)unit_byte_size;
    (void)reload_offset;
  }
  if (first_slice_op == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "dense slice reload group has no remaining slice use");
  }

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, group->block, &builder);
  loom_builder_set_before(&builder, first_slice_op);
  loom_op_t* reload_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_reload_build(&builder, storage_value_id, 0,
                            loom_module_value_type(module, plan->value_id),
                            first_slice_op->location, &reload_op));
  const loom_value_id_t reload_result = loom_low_reload_result(reload_op);

  uint32_t rewritten_slice_count = 0;
  for (uint32_t i = 0; i < use_count; ++i) {
    if (handled_use_flags[i] != 0) {
      continue;
    }
    loom_op_t* slice_op = NULL;
    uint16_t block_index = 0;
    uint32_t unit_byte_size = 0;
    if (!loom_low_allocation_slice_reload_use(assignment, plan, body, uses[i],
                                              &slice_op, &block_index,
                                              &unit_byte_size) ||
        body->blocks[block_index] != group->block) {
      continue;
    }
    (void)unit_byte_size;
    IREE_RETURN_IF_ERROR(loom_low_allocation_validate_spill_use(
        slice_op, loom_use_operand_index(uses[i]), plan));
    IREE_RETURN_IF_ERROR(loom_op_set_operand(
        module, slice_op, /*operand_index=*/0, reload_result));
    handled_use_flags[i] = 1;
    ++rewritten_slice_count;
  }
  if (rewritten_slice_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "dense slice reload group rewrote no slice uses");
  }

  *out_reload_traffic = (loom_low_materialized_traffic_t){
      .count = 1,
      .bytes = plan->byte_size,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_insert_dense_slice_reloads_for_uses(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id, const loom_use_t* uses,
    uint32_t use_count, const loom_block_t* removed_block_arg_block,
    uint16_t removed_arg_index, iree_arena_allocator_t* arena,
    uint8_t** out_handled_use_flags,
    loom_low_materialized_traffic_t* out_reload_traffic) {
  *out_handled_use_flags = NULL;
  *out_reload_traffic = (loom_low_materialized_traffic_t){0};
  if (use_count == 0 || plan->assignment_index >= table->assignment_count) {
    return iree_ok_status();
  }
  const loom_low_allocation_assignment_t* assignment =
      &table->assignments[plan->assignment_index];
  if (assignment->unit_count <= 1 ||
      plan->byte_size % assignment->unit_count != 0) {
    return iree_ok_status();
  }

  const loom_region_t* body = loom_low_function_const_body(table->function_op);
  if (body == NULL || body->block_count == 0) {
    return iree_ok_status();
  }

  bool has_slice_reload_use = false;
  for (uint32_t i = 0; i < use_count; ++i) {
    if (loom_low_allocation_use_is_removed_block_arg_edge(
            uses[i], removed_block_arg_block, removed_arg_index)) {
      continue;
    }
    loom_op_t* slice_op = NULL;
    uint16_t block_index = 0;
    uint32_t unit_byte_size = 0;
    if (loom_low_allocation_slice_reload_use(assignment, plan, body, uses[i],
                                             &slice_op, &block_index,
                                             &unit_byte_size)) {
      has_slice_reload_use = true;
      break;
    }
    (void)slice_op;
    (void)block_index;
    (void)unit_byte_size;
  }
  if (!has_slice_reload_use) {
    return iree_ok_status();
  }

  uint32_t* group_indices_by_block = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, body->block_count, sizeof(*group_indices_by_block),
      (void**)&group_indices_by_block));
  for (uint16_t i = 0; i < body->block_count; ++i) {
    group_indices_by_block[i] = UINT32_MAX;
  }

  loom_low_slice_reload_group_t* groups = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, use_count, sizeof(*groups), (void**)&groups));
  uint32_t group_count = 0;
  for (uint32_t i = 0; i < use_count; ++i) {
    if (loom_low_allocation_use_is_removed_block_arg_edge(
            uses[i], removed_block_arg_block, removed_arg_index)) {
      continue;
    }
    loom_op_t* slice_op = NULL;
    uint16_t block_index = 0;
    uint32_t unit_byte_size = 0;
    if (!loom_low_allocation_slice_reload_use(assignment, plan, body, uses[i],
                                              &slice_op, &block_index,
                                              &unit_byte_size)) {
      continue;
    }
    (void)slice_op;
    uint32_t group_index = group_indices_by_block[block_index];
    if (group_index == UINT32_MAX) {
      group_index = group_count++;
      group_indices_by_block[block_index] = group_index;
      groups[group_index] = (loom_low_slice_reload_group_t){
          .block = body->blocks[block_index],
      };
    }
    if (groups[group_index].slice_count == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "dense slice reload count overflow");
    }
    if (unit_byte_size > UINT64_MAX - groups[group_index].narrow_reload_bytes) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "dense slice reload byte count overflow");
    }
    ++groups[group_index].slice_count;
    groups[group_index].narrow_reload_bytes += unit_byte_size;
  }

  loom_low_materialized_traffic_t reload_traffic = {0};
  uint32_t full_reload_group_count = 0;
  for (uint32_t i = 0; i < group_count; ++i) {
    const bool use_full_reload =
        loom_low_allocation_spill_plan_use_full_slice_reload(
            groups[i].slice_count, groups[i].narrow_reload_bytes,
            plan->byte_size);
    full_reload_group_count += use_full_reload ? 1 : 0;
  }
  if (full_reload_group_count == 0) {
    return iree_ok_status();
  }

  uint8_t* handled_use_flags = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, use_count,
                                                 sizeof(*handled_use_flags),
                                                 (void**)&handled_use_flags));
  memset(handled_use_flags, 0, use_count * sizeof(*handled_use_flags));

  for (uint32_t i = 0; i < group_count; ++i) {
    const bool use_full_reload =
        loom_low_allocation_spill_plan_use_full_slice_reload(
            groups[i].slice_count, groups[i].narrow_reload_bytes,
            plan->byte_size);
    if (!use_full_reload) {
      continue;
    }
    loom_low_materialized_traffic_t inserted_reload_traffic = {0};
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_insert_dense_slice_reload_for_group(
            module, table, plan, body, storage_value_id, uses, use_count,
            &groups[i], handled_use_flags, &inserted_reload_traffic));
    if (inserted_reload_traffic.count > UINT32_MAX - reload_traffic.count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized reload count overflow");
    }
    if (inserted_reload_traffic.bytes > UINT64_MAX - reload_traffic.bytes) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized reload byte count overflow");
    }
    reload_traffic.count += inserted_reload_traffic.count;
    reload_traffic.bytes += inserted_reload_traffic.bytes;
  }
  *out_handled_use_flags = handled_use_flags;
  *out_reload_traffic = reload_traffic;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_insert_reload_for_use(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id, loom_use_t use,
    loom_low_materialized_traffic_t* out_reload_traffic) {
  *out_reload_traffic = (loom_low_materialized_traffic_t){0};
  loom_op_t* user_op = loom_use_user_op(use);
  const uint16_t operand_index = loom_use_operand_index(use);
  IREE_RETURN_IF_ERROR(
      loom_low_allocation_validate_spill_use(user_op, operand_index, plan));

  IREE_RETURN_IF_ERROR(loom_low_allocation_try_insert_reload_for_slice_use(
      module, table, plan, storage_value_id, user_op, operand_index,
      out_reload_traffic));
  if (out_reload_traffic->count != 0) {
    return iree_ok_status();
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
  IREE_RETURN_IF_ERROR(loom_op_set_operand(module, user_op, operand_index,
                                           loom_low_reload_result(reload_op)));
  *out_reload_traffic = (loom_low_materialized_traffic_t){
      .count = 1,
      .bytes = plan->byte_size,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_insert_reloads_for_uses(
    loom_module_t* module, const loom_low_allocation_table_t* table,
    const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id, const loom_use_t* uses,
    uint32_t use_count, const loom_block_t* removed_block_arg_block,
    uint16_t removed_arg_index, iree_arena_allocator_t* arena,
    loom_low_materialized_traffic_t* out_reload_traffic) {
  *out_reload_traffic = (loom_low_materialized_traffic_t){0};
  loom_low_materialized_traffic_t reload_traffic = {0};
  uint8_t* handled_use_flags = NULL;
  if (use_count != 0) {
    IREE_RETURN_IF_ERROR(
        loom_low_allocation_insert_dense_slice_reloads_for_uses(
            module, table, plan, storage_value_id, uses, use_count,
            removed_block_arg_block, removed_arg_index, arena,
            &handled_use_flags, &reload_traffic));
  }
  for (uint32_t i = 0; i < use_count; ++i) {
    if (handled_use_flags != NULL && handled_use_flags[i] != 0) {
      continue;
    }
    if (loom_low_allocation_use_is_removed_block_arg_edge(
            uses[i], removed_block_arg_block, removed_arg_index)) {
      continue;
    }

    loom_low_materialized_traffic_t inserted_reload_traffic = {0};
    IREE_RETURN_IF_ERROR(loom_low_allocation_insert_reload_for_use(
        module, table, plan, storage_value_id, uses[i],
        &inserted_reload_traffic));
    if (inserted_reload_traffic.count > UINT32_MAX - reload_traffic.count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized reload count overflow");
    }
    if (inserted_reload_traffic.bytes > UINT64_MAX - reload_traffic.bytes) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "materialized reload byte count overflow");
    }
    reload_traffic.count += inserted_reload_traffic.count;
    reload_traffic.bytes += inserted_reload_traffic.bytes;
  }
  *out_reload_traffic = reload_traffic;
  return iree_ok_status();
}

static iree_status_t loom_low_allocation_materialize_block_arg_edges(
    loom_module_t* module, const loom_op_t* function_op,
    const loom_low_allocation_spill_plan_t* plan,
    loom_value_id_t storage_value_id, loom_block_t* block, uint16_t arg_index,
    uint32_t reload_count, iree_arena_allocator_t* arena,
    loom_low_materialized_traffic_t* out_store_traffic) {
  *out_store_traffic = (loom_low_materialized_traffic_t){0};
  loom_region_t* body = loom_low_function_body((loom_op_t*)function_op);
  if (!body) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "low function has no body");
  }

  loom_low_materialized_traffic_t store_traffic = {0};
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
      if (store_traffic.count == UINT32_MAX) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "materialized spill count overflow");
      }
      if (plan->byte_size > UINT64_MAX - store_traffic.bytes) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "materialized spill byte count overflow");
      }
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
  if (is_block_arg) {
    *out_record_flags |=
        LOOM_LOW_ALLOCATION_MATERIALIZED_SPILL_FLAG_VALUE_WAS_BLOCK_ARGUMENT;
  }
  uint32_t reload_count = use_count;
  if (is_block_arg && !block_arg_is_entry) {
    reload_count = loom_low_allocation_count_materialized_reloads(
        uses, use_count, block_arg_block, block_arg_index);
  }
  const bool needs_definition_store =
      reload_count > 0 && (!is_block_arg || block_arg_is_entry);

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
  if (is_block_arg && !block_arg_is_entry) {
    IREE_RETURN_IF_ERROR(loom_low_allocation_insert_reloads_for_uses(
        module, table, plan, storage_value_id, uses, use_count, block_arg_block,
        block_arg_index, arena, &inserted_reload_traffic));
  } else {
    IREE_RETURN_IF_ERROR(loom_low_allocation_insert_reloads_for_uses(
        module, table, plan, storage_value_id, uses, use_count, NULL, 0, arena,
        &inserted_reload_traffic));
  }
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
        module, table->function_op, plan, storage_value_id, block_arg_block,
        block_arg_index, reload_count, arena, &inserted_store_traffic));
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
  memset(slots, 0, spill_plan_count * sizeof(*slots));
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
