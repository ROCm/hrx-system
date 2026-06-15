// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/diagnostics.h"

#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"

iree_string_view_t loom_low_diagnostic_string_or_placeholder(
    iree_string_view_t value, iree_string_view_t placeholder) {
  return iree_string_view_is_empty(value) ? placeholder : value;
}

iree_string_view_t loom_low_diagnostic_symbol_name(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!module || !loom_symbol_ref_is_valid(symbol_ref) ||
      symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id >= module->strings.count) return IREE_SV("<unnamed>");
  return module->strings.entries[symbol->name_id];
}

iree_string_view_t loom_low_diagnostic_target_key(
    const loom_low_resolved_target_t* target) {
  if (!target) return IREE_SV("<empty>");
  return loom_low_diagnostic_string_or_placeholder(target->target_name,
                                                   IREE_SV("<empty>"));
}

iree_string_view_t loom_low_diagnostic_export_name(
    const loom_low_resolved_target_t* target) {
  if (!target) return IREE_SV("<empty>");
  return loom_low_diagnostic_string_or_placeholder(
      target->bundle_storage.export_plan.name, IREE_SV("<empty>"));
}

iree_string_view_t loom_low_diagnostic_config_key(
    const loom_low_resolved_target_t* target) {
  if (!target) return IREE_SV("<empty>");
  return loom_low_diagnostic_string_or_placeholder(
      target->bundle_storage.config.name, IREE_SV("<empty>"));
}

iree_string_view_t loom_low_diagnostic_function_name(
    const loom_module_t* module, const loom_op_t* function_op) {
  if (loom_low_func_def_isa(function_op)) {
    return loom_low_diagnostic_symbol_name(
        module, loom_low_func_def_callee(function_op));
  }
  if (loom_low_kernel_def_isa(function_op)) {
    return loom_low_diagnostic_symbol_name(
        module, loom_low_kernel_def_callee(function_op));
  }
  if (loom_low_func_decl_isa(function_op)) {
    return loom_low_diagnostic_symbol_name(
        module, loom_low_func_decl_callee(function_op));
  }
  return IREE_SV("<unnamed>");
}

iree_string_view_t loom_low_diagnostic_value_name(const loom_module_t* module,
                                                  loom_value_id_t value_id) {
  if (!module || value_id >= module->values.count) return IREE_SV("<unknown>");
  const loom_value_t* value = loom_module_value(module, value_id);
  if (value->name_id >= module->strings.count) return IREE_SV("<unnamed>");
  return module->strings.entries[value->name_id];
}

iree_string_view_t loom_low_diagnostic_value_class_name(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_liveness_value_class_t value_class) {
  if (value_class.type_kind != LOOM_TYPE_REGISTER || descriptor_set == NULL ||
      value_class.register_descriptor_set_stable_id !=
          descriptor_set->stable_id ||
      value_class.register_class_id >= descriptor_set->reg_class_count) {
    return IREE_SV("<unknown>");
  }
  const loom_low_reg_class_t* reg_class =
      &descriptor_set->reg_classes[value_class.register_class_id];
  return loom_low_descriptor_set_string(descriptor_set,
                                        reg_class->name_string_offset);
}

iree_string_view_t loom_low_diagnostic_block_name(const loom_module_t* module,
                                                  const loom_block_t* block) {
  if (!module || !block || block->label_id == LOOM_STRING_ID_INVALID ||
      block->label_id >= module->strings.count) {
    return IREE_SV("<anonymous>");
  }
  return module->strings.entries[block->label_id];
}

const loom_op_t* loom_low_diagnostic_value_origin_op(
    const loom_module_t* module, loom_value_id_t value_id,
    const loom_op_t* fallback_op) {
  if (!module || value_id >= module->values.count) return fallback_op;
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) return fallback_op;
  const loom_op_t* defining_op = loom_def_op(value->def);
  return defining_op ? defining_op : fallback_op;
}
