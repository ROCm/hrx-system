// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/config/config.h"

#include <inttypes.h>

#include "iree/base/api.h"
#include "loom/analysis/symbol_value_constraints.h"
#include "loom/ops/config/ops.h"
#include "loom/rewrite/remap.h"
#include "loom/tooling/config/config_application.h"

loom_value_id_t loom_tooling_config_symbol_result_value(const loom_op_t* op) {
  if (loom_config_decl_isa(op)) return loom_config_decl_type(op);
  if (loom_config_def_isa(op)) return loom_config_def_type(op);
  return LOOM_VALUE_ID_INVALID;
}

iree_string_view_t loom_tooling_config_symbol_name(
    const loom_module_t* module, const loom_symbol_t* symbol) {
  if (!module || !symbol || symbol->name_id == LOOM_STRING_ID_INVALID ||
      symbol->name_id >= module->strings.count) {
    return IREE_SV("<invalid>");
  }
  return module->strings.entries[symbol->name_id];
}

uint16_t loom_tooling_config_find_symbol(const loom_module_t* module,
                                         iree_string_view_t key) {
  const loom_string_id_t name_id = loom_module_lookup_string(module, key);
  return name_id == LOOM_STRING_ID_INVALID
             ? LOOM_SYMBOL_ID_INVALID
             : loom_module_find_symbol(module, name_id);
}

bool loom_tooling_config_symbol_is_config(const loom_symbol_t* symbol) {
  return symbol && symbol->defining_op &&
         loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_CONFIG);
}

iree_status_t loom_tooling_config_remap_type_and_value(
    const loom_module_t* source_module, loom_module_t* target_module,
    loom_type_t source_type, loom_attribute_t source_value,
    iree_arena_block_pool_t* block_pool, loom_type_t* out_target_type,
    loom_attribute_t* out_target_value) {
  *out_target_type = (loom_type_t){0};
  *out_target_value = loom_attr_absent();

  iree_arena_allocator_t remap_arena;
  iree_arena_initialize(block_pool, &remap_arena);
  loom_ir_remap_t remap = {0};
  const loom_ir_remap_options_t remap_options = {0};
  iree_status_t status = loom_ir_remap_initialize(
      source_module, target_module, &remap_arena, &remap_options, &remap);
  if (iree_status_is_ok(status)) {
    status = loom_ir_remap_type(&remap, source_type, out_target_type);
  }
  if (iree_status_is_ok(status)) {
    status = loom_ir_remap_attribute(&remap, source_value, out_target_value);
  }
  iree_arena_deinitialize(&remap_arena);
  return status;
}

static iree_status_t loom_tooling_config_replace_with_def(
    loom_module_t* module, loom_op_t* old_op, loom_symbol_ref_t symbol,
    loom_attribute_t value, loom_type_t type) {
  loom_block_t* block = old_op->parent_block;
  loom_op_t* before_op = old_op->next_op;
  loom_op_t* parent_op = old_op->parent_op;
  const loom_location_id_t location = old_op->location;
  const loom_value_id_t old_result =
      loom_tooling_config_symbol_result_value(old_op);

  IREE_RETURN_IF_ERROR(loom_op_erase(module, old_op));

  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, block, &builder);
  builder.ip.parent_op = parent_op;
  builder.ip.before_op = before_op;
  loom_op_t* new_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_config_def_build(&builder, symbol, value, type, location, &new_op));
  if (old_result != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_module_copy_value_name(
        module, old_result, loom_config_def_type(new_op)));
  }
  return iree_ok_status();
}

iree_status_t loom_tooling_config_apply_exact_value(loom_module_t* module,
                                                    iree_string_view_t key,
                                                    loom_op_t* old_op,
                                                    loom_type_t type,
                                                    loom_attribute_t value) {
  if (loom_config_decl_isa(old_op)) {
    IREE_RETURN_IF_ERROR(loom_symbol_value_constraints_check_exact(
        key, type, loom_config_decl_type(old_op), value,
        loom_config_decl_predicates(old_op)));
  }
  const loom_symbol_ref_t symbol = loom_config_decl_isa(old_op)
                                       ? loom_config_decl_symbol(old_op)
                                       : loom_config_def_symbol(old_op);
  return loom_tooling_config_replace_with_def(module, old_op, symbol, value,
                                              type);
}

iree_status_t loom_tooling_config_overlay_module(
    loom_module_t* module, const loom_module_t* config_module,
    iree_arena_block_pool_t* block_pool,
    loom_tooling_config_materialize_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(config_module);
  IREE_ASSERT_ARGUMENT(block_pool);
  if (module == config_module) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "config module must be distinct from target module");
  }
  if (module->context != config_module->context) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "config and target modules use different contexts");
  }

  loom_tooling_config_materialize_result_t result = {0};
  const loom_block_t* config_block =
      loom_region_const_entry_block(config_module->body);
  for (const loom_op_t* config_op = config_block->first_op; config_op != NULL;
       config_op = config_op->next_op) {
    if (!loom_config_def_isa(config_op)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "config module contains non-config.def operation kind %u",
          (unsigned)config_op->kind);
    }
    const loom_symbol_ref_t source_ref = loom_config_def_symbol(config_op);
    if (!loom_symbol_ref_is_valid(source_ref) || source_ref.module_id != 0 ||
        source_ref.symbol_id >= config_module->symbols.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "config.def has an invalid symbol reference");
    }
    const loom_symbol_t* source_symbol =
        &config_module->symbols.entries[source_ref.symbol_id];
    const iree_string_view_t key =
        loom_tooling_config_symbol_name(config_module, source_symbol);

    const uint16_t target_symbol_id =
        loom_tooling_config_find_symbol(module, key);
    if (target_symbol_id == LOOM_SYMBOL_ID_INVALID) {
      ++result.ignored_count;
      continue;
    }
    loom_symbol_t* target_symbol = &module->symbols.entries[target_symbol_id];
    if (!loom_tooling_config_symbol_is_config(target_symbol)) {
      ++result.ignored_count;
      continue;
    }
    loom_op_t* target_op = target_symbol->defining_op;
    const loom_value_id_t target_value =
        loom_tooling_config_symbol_result_value(target_op);
    if (target_value == LOOM_VALUE_ID_INVALID ||
        target_value >= module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "config '%.*s' has no result value",
                              (int)key.size, key.data);
    }
    const loom_type_t target_type =
        loom_module_value_type(module, target_value);
    const loom_value_id_t source_value = loom_config_def_type(config_op);
    if (source_value >= config_module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "config definition '%.*s' has no result value",
                              (int)key.size, key.data);
    }

    loom_type_t remapped_type = {0};
    loom_attribute_t remapped_value = loom_attr_absent();
    IREE_RETURN_IF_ERROR(loom_tooling_config_remap_type_and_value(
        config_module, module,
        loom_module_value_type(config_module, source_value),
        loom_config_def_value(config_op), block_pool, &remapped_type,
        &remapped_value));
    if (!loom_type_equal(remapped_type, target_type)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "config definition '%.*s' type does not match the target declaration",
          (int)key.size, key.data);
    }
    IREE_RETURN_IF_ERROR(loom_tooling_config_apply_exact_value(
        module, key, target_op, target_type, remapped_value));
    ++result.materialized_count;
  }

  if (out_result) *out_result = result;
  return iree_ok_status();
}

iree_status_t loom_tooling_config_require_resolved_module(
    const loom_module_t* module,
    loom_tooling_config_resolution_result_t* out_result) {
  IREE_ASSERT_ARGUMENT(module);
  loom_tooling_config_resolution_result_t result = {0};
  iree_string_view_t first_unresolved_name = iree_string_view_empty();

  const loom_symbol_t* symbol = NULL;
  loom_module_for_each_symbol(module, symbol) {
    if (!symbol->defining_op || !loom_config_decl_isa(symbol->defining_op)) {
      continue;
    }
    if (result.unresolved_count == 0) {
      first_unresolved_name = loom_tooling_config_symbol_name(module, symbol);
    }
    ++result.unresolved_count;
  }

  if (out_result) *out_result = result;
  if (result.unresolved_count == 0) {
    return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "unresolved config '@%.*s' remains for final compilation (%" PRIhsz
      " unresolved config%s total)",
      (int)first_unresolved_name.size, first_unresolved_name.data,
      result.unresolved_count, result.unresolved_count == 1 ? "" : "s");
}
