// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/diagnostics.h"

#include "loom/codegen/low/function.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
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
  if (symbol->name_id >= module->strings.count) {
    return IREE_SV("<unnamed>");
  }
  return loom_string_table_get(&module->strings, symbol->name_id);
}

iree_string_view_t loom_low_diagnostic_target_key(
    const loom_low_resolved_target_t* target) {
  if (!target) {
    return IREE_SV("<empty>");
  }
  return loom_low_diagnostic_string_or_placeholder(target->target_name,
                                                   IREE_SV("<empty>"));
}

iree_string_view_t loom_low_diagnostic_export_name(
    const loom_low_resolved_target_t* target) {
  const loom_target_bundle_t* bundle = loom_low_resolved_target_bundle(target);
  if (!bundle) {
    return IREE_SV("<empty>");
  }
  return loom_low_diagnostic_string_or_placeholder(bundle->export_plan->name,
                                                   IREE_SV("<empty>"));
}

iree_string_view_t loom_low_diagnostic_config_key(
    const loom_low_resolved_target_t* target) {
  const loom_target_bundle_t* bundle = loom_low_resolved_target_bundle(target);
  if (!bundle) {
    return IREE_SV("<empty>");
  }
  return loom_low_diagnostic_string_or_placeholder(bundle->config->name,
                                                   IREE_SV("<empty>"));
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

static iree_string_view_t loom_low_diagnostic_allocation_mode_name(
    uint8_t allocation_mode) {
  switch (allocation_mode) {
    case 0:
    case LOOM_LOW_ALLOCATION_VIRTUAL:
      return IREE_SV("virtual");
    case LOOM_LOW_ALLOCATION_ASSIGNED:
      return IREE_SV("assigned");
    case LOOM_LOW_ALLOCATION_FIXED:
      return IREE_SV("fixed");
    default:
      return IREE_SV("<unknown>");
  }
}

iree_status_t loom_low_diagnostic_admit_allocation_synthesis(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_diagnostic_emitter_t emitter, bool* out_admitted) {
  *out_admitted = false;
  const uint8_t allocation_mode = loom_low_function_allocation(function_op);
  if (allocation_mode == 0 || allocation_mode == LOOM_LOW_ALLOCATION_VIRTUAL) {
    *out_admitted = true;
    return iree_ok_status();
  }

  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_function_name(module, function_op)),
      loom_param_string(
          loom_low_diagnostic_allocation_mode_name(allocation_mode)),
  };
  const loom_diagnostic_emission_t emission = {
      .op = function_op,
      .error = LOOM_ERR_BACKEND_051,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

iree_status_t loom_low_diagnostic_emit_missing_target(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_diagnostic_emitter_t emitter) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_function_name(module, function_op)),
  };
  const loom_diagnostic_emission_t emission = {
      .op = function_op,
      .error = LOOM_ERR_TARGET_026,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

iree_status_t loom_low_diagnostic_validate_workgroup_storage_limit(
    const loom_module_t* module, const loom_op_t* function_op,
    const loom_low_resolved_target_t* target, uint64_t workgroup_storage_bytes,
    iree_diagnostic_emitter_t emitter, bool* out_valid) {
  if (out_valid != NULL) {
    *out_valid = true;
  }
  const loom_target_bundle_t* bundle = loom_low_resolved_target_bundle(target);
  if (bundle == NULL) {
    return iree_ok_status();
  }
  const uint64_t limit = bundle->snapshot->max_workgroup_storage_bytes;
  if (limit == 0 || workgroup_storage_bytes <= limit) {
    return iree_ok_status();
  }

  if (out_valid != NULL) {
    *out_valid = false;
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_diagnostic_function_name(module, function_op)),
      loom_param_string(loom_low_diagnostic_target_key(target)),
      loom_param_u64(workgroup_storage_bytes),
      loom_param_u64(limit),
  };
  const loom_diagnostic_emission_t emission = {
      .op = function_op,
      .error = LOOM_ERR_TARGET_051,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

iree_string_view_t loom_low_diagnostic_operation_name(
    const loom_module_t* module, const loom_op_t* op) {
  if (!module || !op) {
    return IREE_SV("<unknown>");
  }
  return loom_op_name(module, op);
}

iree_string_view_t loom_low_diagnostic_value_name(const loom_module_t* module,
                                                  loom_value_id_t value_id) {
  if (!module || value_id >= module->values.count) {
    return IREE_SV("<unknown>");
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (value->name_id >= module->strings.count) {
    return IREE_SV("<unnamed>");
  }
  return loom_string_table_get(&module->strings, value->name_id);
}

iree_string_view_t loom_low_diagnostic_reg_class_name(
    const loom_low_descriptor_set_t* descriptor_set,
    uint16_t descriptor_reg_class_id) {
  const loom_low_reg_class_t* reg_class =
      &descriptor_set->reg_classes[descriptor_reg_class_id];
  return loom_low_descriptor_set_string(descriptor_set,
                                        reg_class->name_string_ref);
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
  return loom_low_diagnostic_reg_class_name(descriptor_set,
                                            value_class.register_class_id);
}

iree_string_view_t loom_low_diagnostic_block_name(const loom_module_t* module,
                                                  const loom_block_t* block) {
  if (!module || !block || block->label_id == LOOM_STRING_ID_INVALID ||
      block->label_id >= module->strings.count) {
    return IREE_SV("<anonymous>");
  }
  return loom_string_table_get(&module->strings, block->label_id);
}

const loom_op_t* loom_low_diagnostic_value_origin_op(
    const loom_module_t* module, loom_value_id_t value_id,
    const loom_op_t* fallback_op) {
  if (!module || value_id >= module->values.count) {
    return fallback_op;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return fallback_op;
  }
  const loom_op_t* defining_op = loom_def_op(value->def);
  return defining_op ? defining_op : fallback_op;
}

iree_string_view_t loom_low_diagnostic_value_origin_operation_name(
    const loom_module_t* module, loom_value_id_t value_id,
    const loom_op_t* fallback_op) {
  if (!module || value_id >= module->values.count) {
    return loom_low_diagnostic_operation_name(module, fallback_op);
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return IREE_SV("<block-argument>");
  }
  const loom_op_t* defining_op = loom_def_op(value->def);
  return loom_low_diagnostic_operation_name(
      module, defining_op ? defining_op : fallback_op);
}
