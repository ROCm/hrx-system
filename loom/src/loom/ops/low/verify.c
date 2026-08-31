// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/reference.h"
#include "loom/ops/function_contract_verify.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/successor_verify.h"
#include "loom/ops/type_registry.h"
#include "loom/target/registers.h"
#include "loom/util/stable_id.h"

typedef struct loom_low_callee_signature_t {
  // Defining function-like op for related diagnostic locations.
  const loom_op_t* definition_op;
  // Callee argument value IDs in signature order.
  const loom_value_id_t* argument_ids;
  // Number of callee arguments.
  uint16_t argument_count;
  // Callee result value IDs in signature order.
  const loom_value_id_t* result_ids;
  // Number of callee results.
  uint16_t result_count;
} loom_low_callee_signature_t;

static iree_status_t loom_low_emit_related(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    const loom_error_def_t* error, const loom_diagnostic_param_t* params,
    iree_host_size_t param_count, const loom_diagnostic_related_op_t* related,
    iree_host_size_t related_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
      .related_ops = related,
      .related_op_count = related_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_low_emit(iree_diagnostic_emitter_t emitter,
                                   const loom_op_t* op,
                                   const loom_error_def_t* error,
                                   const loom_diagnostic_param_t* params,
                                   iree_host_size_t param_count) {
  return loom_low_emit_related(emitter, op, error, params, param_count, NULL,
                               0);
}

static iree_status_t loom_low_emit_attr_value_error(
    const loom_op_t* op, uint16_t attr_index, iree_string_view_t attr_name,
    int64_t actual_value, iree_string_view_t expected_constraint,
    iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(attr_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attr_index)),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_low_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_emit_string_attr_value_error(
    const loom_op_t* op, uint16_t attr_index, iree_string_view_t attr_name,
    iree_string_view_t actual_value, iree_string_view_t expected_constraint,
    iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(attr_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attr_index)),
      loom_param_string(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_low_emit(emitter, op, LOOM_ERR_STRUCTURE_027, params,
                       IREE_ARRAYSIZE(params));
}

static bool loom_low_qualified_key_segment_start(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static bool loom_low_qualified_key_segment_continue(char c) {
  return loom_low_qualified_key_segment_start(c) || (c >= '0' && c <= '9');
}

static bool loom_low_qualified_key_is_valid(iree_string_view_t key) {
  if (iree_string_view_is_empty(key)) {
    return false;
  }
  bool saw_separator = false;
  bool expect_segment_start = true;
  for (iree_host_size_t i = 0; i < key.size; ++i) {
    char c = key.data[i];
    if (c == '.') {
      if (expect_segment_start) {
        return false;
      }
      saw_separator = true;
      expect_segment_start = true;
      continue;
    }
    if (expect_segment_start) {
      if (!loom_low_qualified_key_segment_start(c)) {
        return false;
      }
      expect_segment_start = false;
      continue;
    }
    if (!loom_low_qualified_key_segment_continue(c)) {
      return false;
    }
  }
  return saw_separator && !expect_segment_start;
}

static iree_string_view_t loom_low_string_or_empty(const loom_module_t* module,
                                                   loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_string_view_t loom_low_op_name(const loom_module_t* module,
                                           const loom_op_t* op) {
  if (!op) {
    return IREE_SV("<null>");
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable) {
    return IREE_SV("<unknown>");
  }
  return loom_op_vtable_name(vtable);
}

static iree_string_view_t loom_low_symbol_name(const loom_module_t* module,
                                               loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  iree_string_view_t name = loom_low_string_or_empty(module, symbol->name_id);
  return iree_string_view_is_empty(name) ? IREE_SV("<unnamed>") : name;
}

static iree_string_view_t loom_low_symbol_definition_name(
    const loom_symbol_t* symbol) {
  if (!symbol || !symbol->definition) {
    return IREE_SV("unresolved");
  }
  return loom_symbol_definition_descriptor_name(symbol->definition);
}

static const loom_symbol_t* loom_low_lookup_defined_symbol(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return NULL;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (!symbol->definition || !symbol->defining_op) {
    return NULL;
  }
  return symbol;
}

static bool loom_low_function_isa(const loom_op_t* op) {
  return loom_low_func_def_isa(op) || loom_low_func_decl_isa(op);
}

static bool loom_low_executable_def_isa(const loom_op_t* op) {
  return loom_low_func_def_isa(op) || loom_low_kernel_def_isa(op);
}

static loom_symbol_ref_t loom_low_function_symbol(const loom_op_t* op) {
  if (loom_low_func_def_isa(op)) {
    return loom_low_func_def_callee(op);
  }
  if (loom_low_kernel_def_isa(op)) {
    return loom_low_kernel_def_callee(op);
  }
  if (loom_low_func_decl_isa(op)) {
    return loom_low_func_decl_callee(op);
  }
  return loom_symbol_ref_null();
}

static loom_symbol_ref_t loom_low_function_target(const loom_op_t* op) {
  if (loom_low_func_def_isa(op)) {
    return loom_low_func_def_target(op);
  }
  if (loom_low_kernel_def_isa(op)) {
    return loom_low_kernel_def_target(op);
  }
  if (loom_low_func_decl_isa(op)) {
    return loom_low_func_decl_target(op);
  }
  return loom_symbol_ref_null();
}

static bool loom_low_symbol_ref_equal(loom_symbol_ref_t lhs,
                                      loom_symbol_ref_t rhs) {
  return lhs.module_id == rhs.module_id && lhs.symbol_id == rhs.symbol_id;
}

static bool loom_low_optional_attr_is_present(const loom_op_t* op,
                                              uint16_t attr_index) {
  return attr_index < op->attribute_count &&
         !loom_attr_is_absent(loom_op_attrs(op)[attr_index]);
}

static bool loom_low_function_explicit_abi(const loom_op_t* op,
                                           loom_target_abi_kind_t* out_abi,
                                           uint16_t* out_abi_attr_index) {
  if (loom_low_func_def_isa(op) &&
      loom_low_optional_attr_is_present(op, loom_low_func_def_abi_ATTR_INDEX)) {
    *out_abi = loom_low_func_def_abi(op);
    *out_abi_attr_index = loom_low_func_def_abi_ATTR_INDEX;
    return true;
  }
  if (loom_low_func_decl_isa(op) &&
      loom_low_optional_attr_is_present(op,
                                        loom_low_func_decl_abi_ATTR_INDEX)) {
    *out_abi = loom_low_func_decl_abi(op);
    *out_abi_attr_index = loom_low_func_decl_abi_ATTR_INDEX;
    return true;
  }
  return false;
}

static loom_type_t loom_low_type_attr(const loom_module_t* module,
                                      loom_type_id_t type_id) {
  if (type_id == LOOM_TYPE_ID_INVALID || type_id >= module->types.count) {
    return loom_type_none();
  }
  return module->types.entries[type_id];
}

static iree_status_t loom_low_emit_symbol_kind_mismatch(
    const loom_module_t* module, const loom_op_t* op, loom_symbol_ref_t ref,
    uint16_t attr_index, const loom_symbol_t* symbol,
    iree_string_view_t expected_kind, iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(loom_low_symbol_name(module, ref)),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attr_index)),
      loom_param_string(loom_low_symbol_definition_name(symbol)),
      loom_param_string(expected_kind),
  };
  loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("defined here"),
      .op = symbol ? symbol->defining_op : NULL,
  }};
  return loom_low_emit_related(
      emitter, op, LOOM_ERR_SYMBOL_003, params, IREE_ARRAYSIZE(params), related,
      symbol && symbol->defining_op ? IREE_ARRAYSIZE(related) : 0);
}

static iree_status_t loom_low_verify_optional_exactness_mode_is_named(
    const loom_op_t* op, uint16_t attr_index, uint8_t value,
    iree_string_view_t attribute_name, iree_diagnostic_emitter_t emitter) {
  if (!loom_low_optional_attr_is_present(op, attr_index) || value != 0) {
    return iree_ok_status();
  }
  return loom_low_emit_attr_value_error(op, attr_index, attribute_name, value,
                                        IREE_SV("named exactness mode"),
                                        emitter);
}

static iree_status_t loom_low_verify_function_attr_present(
    const loom_op_t* op, uint16_t attr_index, iree_string_view_t attribute_name,
    iree_string_view_t expected_constraint, iree_diagnostic_emitter_t emitter) {
  if (loom_low_optional_attr_is_present(op, attr_index)) {
    return iree_ok_status();
  }
  return loom_low_emit_attr_value_error(op, attr_index, attribute_name, 0,
                                        expected_constraint, emitter);
}

static iree_status_t loom_low_verify_positive_u32_attr(
    const loom_op_t* op, uint16_t attr_index, int64_t value,
    iree_string_view_t attribute_name, iree_diagnostic_emitter_t emitter) {
  if (!loom_low_optional_attr_is_present(op, attr_index)) {
    return iree_ok_status();
  }
  if (value > 0 && value <= UINT32_MAX) {
    return iree_ok_status();
  }
  return loom_low_emit_attr_value_error(op, attr_index, attribute_name, value,
                                        IREE_SV("positive u32"), emitter);
}

static iree_status_t loom_low_verify_optional_positive_u32_triple(
    const loom_op_t* op, uint16_t x_attr_index, iree_string_view_t x_attr_name,
    iree_string_view_t x_expected_constraint, uint16_t y_attr_index,
    iree_string_view_t y_attr_name, iree_string_view_t y_expected_constraint,
    uint16_t z_attr_index, iree_string_view_t z_attr_name,
    iree_string_view_t z_expected_constraint,
    iree_diagnostic_emitter_t emitter) {
  const bool has_x = loom_low_optional_attr_is_present(op, x_attr_index);
  const bool has_y = loom_low_optional_attr_is_present(op, y_attr_index);
  const bool has_z = loom_low_optional_attr_is_present(op, z_attr_index);
  if (!has_x && !has_y && !has_z) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_function_attr_present(
      op, x_attr_index, x_attr_name, x_expected_constraint, emitter));
  IREE_RETURN_IF_ERROR(loom_low_verify_function_attr_present(
      op, y_attr_index, y_attr_name, y_expected_constraint, emitter));
  IREE_RETURN_IF_ERROR(loom_low_verify_function_attr_present(
      op, z_attr_index, z_attr_name, z_expected_constraint, emitter));
  IREE_RETURN_IF_ERROR(loom_low_verify_positive_u32_attr(
      op, x_attr_index, loom_attr_as_i64(loom_op_attrs(op)[x_attr_index]),
      x_attr_name, emitter));
  IREE_RETURN_IF_ERROR(loom_low_verify_positive_u32_attr(
      op, y_attr_index, loom_attr_as_i64(loom_op_attrs(op)[y_attr_index]),
      y_attr_name, emitter));
  return loom_low_verify_positive_u32_attr(
      op, z_attr_index, loom_attr_as_i64(loom_op_attrs(op)[z_attr_index]),
      z_attr_name, emitter);
}

static iree_status_t loom_low_verify_kernel_contract(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter) {
  const bool has_export_symbol = loom_low_optional_attr_is_present(
      op, loom_low_kernel_def_export_symbol_ATTR_INDEX);
  const bool has_export_linkage = loom_low_optional_attr_is_present(
      op, loom_low_kernel_def_export_linkage_ATTR_INDEX);
  if (!has_export_symbol && has_export_linkage) {
    return loom_low_verify_function_attr_present(
        op, loom_low_kernel_def_export_symbol_ATTR_INDEX, IREE_SV("export"),
        IREE_SV("present when linkage is present"), emitter);
  }

  IREE_RETURN_IF_ERROR(loom_low_verify_optional_positive_u32_triple(
      op, loom_low_kernel_def_workgroup_size_x_ATTR_INDEX,
      IREE_SV("workgroup_size_x"),
      IREE_SV("present when workgroup_size_y or workgroup_size_z is present"),
      loom_low_kernel_def_workgroup_size_y_ATTR_INDEX,
      IREE_SV("workgroup_size_y"),
      IREE_SV("present when workgroup_size_x or workgroup_size_z is present"),
      loom_low_kernel_def_workgroup_size_z_ATTR_INDEX,
      IREE_SV("workgroup_size_z"),
      IREE_SV("present when workgroup_size_x or workgroup_size_y is present"),
      emitter));
  IREE_RETURN_IF_ERROR(loom_low_verify_optional_positive_u32_triple(
      op, loom_low_kernel_def_workgroup_count_x_ATTR_INDEX,
      IREE_SV("workgroup_count_x"),
      IREE_SV("present when workgroup_count_y or workgroup_count_z is present"),
      loom_low_kernel_def_workgroup_count_y_ATTR_INDEX,
      IREE_SV("workgroup_count_y"),
      IREE_SV("present when workgroup_count_x or workgroup_count_z is present"),
      loom_low_kernel_def_workgroup_count_z_ATTR_INDEX,
      IREE_SV("workgroup_count_z"),
      IREE_SV("present when workgroup_count_x or workgroup_count_y is present"),
      emitter));
  IREE_RETURN_IF_ERROR(loom_low_verify_optional_positive_u32_triple(
      op, loom_low_kernel_def_workgroup_cluster_size_x_ATTR_INDEX,
      IREE_SV("workgroup_cluster_size_x"),
      IREE_SV("present when another cluster dimension is present"),
      loom_low_kernel_def_workgroup_cluster_size_y_ATTR_INDEX,
      IREE_SV("workgroup_cluster_size_y"),
      IREE_SV("present when another cluster dimension is present"),
      loom_low_kernel_def_workgroup_cluster_size_z_ATTR_INDEX,
      IREE_SV("workgroup_cluster_size_z"),
      IREE_SV("present when another cluster dimension is present"), emitter));

  const bool has_cluster_size_x = loom_low_optional_attr_is_present(
      op, loom_low_kernel_def_workgroup_cluster_size_x_ATTR_INDEX);
  const bool has_cluster_size_y = loom_low_optional_attr_is_present(
      op, loom_low_kernel_def_workgroup_cluster_size_y_ATTR_INDEX);
  const bool has_cluster_size_z = loom_low_optional_attr_is_present(
      op, loom_low_kernel_def_workgroup_cluster_size_z_ATTR_INDEX);
  if (!has_cluster_size_x && !has_cluster_size_y && !has_cluster_size_z) {
    return iree_ok_status();
  }
  if (!has_cluster_size_x || !has_cluster_size_y || !has_cluster_size_z) {
    return iree_ok_status();
  }

  const int64_t cluster_size_x_i64 =
      loom_low_kernel_def_workgroup_cluster_size_x(op);
  const int64_t cluster_size_y_i64 =
      loom_low_kernel_def_workgroup_cluster_size_y(op);
  const int64_t cluster_size_z_i64 =
      loom_low_kernel_def_workgroup_cluster_size_z(op);
  if (cluster_size_x_i64 <= 0 || cluster_size_x_i64 > UINT32_MAX ||
      cluster_size_y_i64 <= 0 || cluster_size_y_i64 > UINT32_MAX ||
      cluster_size_z_i64 <= 0 || cluster_size_z_i64 > UINT32_MAX) {
    return iree_ok_status();
  }
  const uint32_t cluster_size_x = (uint32_t)cluster_size_x_i64;
  const uint32_t cluster_size_y = (uint32_t)cluster_size_y_i64;
  const uint32_t cluster_size_z = (uint32_t)cluster_size_z_i64;
  if (cluster_size_x == 1 && cluster_size_y == 1 && cluster_size_z == 1) {
    return loom_low_emit_attr_value_error(
        op, loom_low_kernel_def_workgroup_cluster_size_x_ATTR_INDEX,
        IREE_SV("workgroup_cluster_size_x"), cluster_size_x,
        IREE_SV("omitted when the complete cluster shape is 1x1x1"), emitter);
  }
  const uint64_t cluster_size_xy =
      (uint64_t)cluster_size_x * (uint64_t)cluster_size_y;
  if (cluster_size_xy > UINT32_MAX ||
      cluster_size_z > UINT32_MAX / cluster_size_xy) {
    return loom_low_emit_attr_value_error(
        op, loom_low_kernel_def_workgroup_cluster_size_z_ATTR_INDEX,
        IREE_SV("workgroup_cluster_size_z"), cluster_size_z,
        IREE_SV("a value whose product with the x and y dimensions fits u32"),
        emitter);
  }

  const bool has_workgroup_count_x = loom_low_optional_attr_is_present(
      op, loom_low_kernel_def_workgroup_count_x_ATTR_INDEX);
  const bool has_workgroup_count_y = loom_low_optional_attr_is_present(
      op, loom_low_kernel_def_workgroup_count_y_ATTR_INDEX);
  const bool has_workgroup_count_z = loom_low_optional_attr_is_present(
      op, loom_low_kernel_def_workgroup_count_z_ATTR_INDEX);
  if (!has_workgroup_count_x || !has_workgroup_count_y ||
      !has_workgroup_count_z) {
    return iree_ok_status();
  }
  const int64_t workgroup_count_x_i64 =
      loom_low_kernel_def_workgroup_count_x(op);
  const int64_t workgroup_count_y_i64 =
      loom_low_kernel_def_workgroup_count_y(op);
  const int64_t workgroup_count_z_i64 =
      loom_low_kernel_def_workgroup_count_z(op);
  if (workgroup_count_x_i64 <= 0 || workgroup_count_x_i64 > UINT32_MAX ||
      workgroup_count_y_i64 <= 0 || workgroup_count_y_i64 > UINT32_MAX ||
      workgroup_count_z_i64 <= 0 || workgroup_count_z_i64 > UINT32_MAX) {
    return iree_ok_status();
  }
  const uint32_t workgroup_counts[] = {
      (uint32_t)workgroup_count_x_i64,
      (uint32_t)workgroup_count_y_i64,
      (uint32_t)workgroup_count_z_i64,
  };
  const uint32_t cluster_sizes[] = {
      cluster_size_x,
      cluster_size_y,
      cluster_size_z,
  };
  const uint16_t workgroup_count_attr_indices[] = {
      loom_low_kernel_def_workgroup_count_x_ATTR_INDEX,
      loom_low_kernel_def_workgroup_count_y_ATTR_INDEX,
      loom_low_kernel_def_workgroup_count_z_ATTR_INDEX,
  };
  const iree_string_view_t workgroup_count_attr_names[] = {
      IREE_SV("workgroup_count_x"),
      IREE_SV("workgroup_count_y"),
      IREE_SV("workgroup_count_z"),
  };
  const iree_string_view_t expected_constraints[] = {
      IREE_SV("a positive multiple of workgroup_cluster_size_x"),
      IREE_SV("a positive multiple of workgroup_cluster_size_y"),
      IREE_SV("a positive multiple of workgroup_cluster_size_z"),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(workgroup_counts); ++i) {
    if (workgroup_counts[i] % cluster_sizes[i] != 0) {
      return loom_low_emit_attr_value_error(
          op, workgroup_count_attr_indices[i], workgroup_count_attr_names[i],
          workgroup_counts[i], expected_constraints[i], emitter);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_exactness_modes(
    const loom_op_t* op, uint16_t allocation_attr_index, uint8_t allocation,
    uint16_t schedule_attr_index, uint8_t schedule,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_optional_exactness_mode_is_named(
      op, allocation_attr_index, allocation, IREE_SV("allocation"), emitter));
  return loom_low_verify_optional_exactness_mode_is_named(
      op, schedule_attr_index, schedule, IREE_SV("schedule"), emitter);
}

static iree_status_t loom_low_verify_function_exactness_modes(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter) {
  if (loom_low_func_def_isa(op)) {
    return loom_low_verify_exactness_modes(
        op, loom_low_func_def_allocation_ATTR_INDEX,
        loom_low_func_def_allocation(op), loom_low_func_def_schedule_ATTR_INDEX,
        loom_low_func_def_schedule(op), emitter);
  }
  return loom_low_verify_exactness_modes(
      op, loom_low_func_decl_allocation_ATTR_INDEX,
      loom_low_func_decl_allocation(op), loom_low_func_decl_schedule_ATTR_INDEX,
      loom_low_func_decl_schedule(op), emitter);
}

static iree_status_t loom_low_verify_kernel_exactness_modes(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_exactness_modes(
      op, loom_low_kernel_def_allocation_ATTR_INDEX,
      loom_low_kernel_def_allocation(op),
      loom_low_kernel_def_schedule_ATTR_INDEX, loom_low_kernel_def_schedule(op),
      emitter);
}

static iree_status_t loom_low_verify_decl_code_import(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const bool import_kind_present = loom_low_optional_attr_is_present(
      op, loom_low_func_decl_import_kind_ATTR_INDEX);
  const bool code_symbol_present = loom_low_optional_attr_is_present(
      op, loom_low_func_decl_code_symbol_ATTR_INDEX);
  if (import_kind_present && loom_low_func_decl_import_kind(op) == 0) {
    IREE_RETURN_IF_ERROR(loom_low_emit_attr_value_error(
        op, loom_low_func_decl_import_kind_ATTR_INDEX, IREE_SV("import"), 0,
        IREE_SV("named import kind"), emitter));
  }
  if (import_kind_present && code_symbol_present) {
    iree_string_view_t code_symbol =
        loom_low_string_or_empty(module, loom_low_func_decl_code_symbol(op));
    if (!iree_string_view_is_empty(code_symbol)) {
      return iree_ok_status();
    }
    return loom_low_emit_string_attr_value_error(
        op, loom_low_func_decl_code_symbol_ATTR_INDEX, IREE_SV("code_symbol"),
        code_symbol, IREE_SV("non-empty imported code symbol"), emitter);
  }
  if (import_kind_present) {
    return loom_low_verify_function_attr_present(
        op, loom_low_func_decl_code_symbol_ATTR_INDEX, IREE_SV("code_symbol"),
        IREE_SV("present when import kind is present"), emitter);
  }
  if (code_symbol_present) {
    return loom_low_verify_function_attr_present(
        op, loom_low_func_decl_import_kind_ATTR_INDEX, IREE_SV("import"),
        IREE_SV("present when code_symbol is present"), emitter);
  }
  return iree_ok_status();
}

static bool loom_low_load_func_like_signature(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_low_callee_signature_t* out_signature) {
  loom_func_like_t func =
      loom_func_like_cast(module, symbol ? symbol->defining_op : NULL);
  if (!loom_func_like_isa(func)) {
    return false;
  }
  out_signature->definition_op = symbol->defining_op;
  out_signature->argument_ids =
      loom_func_like_arg_ids(func, &out_signature->argument_count);
  out_signature->result_ids = loom_op_const_results(func.op);
  out_signature->result_count = func.op->result_count;
  return true;
}

static bool loom_low_load_low_signature(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_low_callee_signature_t* out_signature) {
  if (!symbol || !loom_low_function_isa(symbol->defining_op)) {
    return false;
  }
  return loom_low_load_func_like_signature(module, symbol, out_signature);
}

static iree_status_t loom_low_emit_descriptor_key_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, uint16_t attr_index,
    iree_string_view_t field_name, iree_string_view_t key,
    iree_string_view_t expected) {
  return loom_low_emit_string_attr_value_error(op, attr_index, field_name, key,
                                               expected, emitter);
}

static iree_status_t loom_low_verify_qualified_key_attr(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_string_id_t string_id,
    uint16_t attr_index, iree_string_view_t field_name,
    iree_string_view_t expected) {
  iree_string_view_t key = loom_low_string_or_empty(module, string_id);
  if (loom_low_qualified_key_is_valid(key)) {
    return iree_ok_status();
  }
  return loom_low_emit_descriptor_key_error(emitter, op, attr_index, field_name,
                                            key, expected);
}

static bool loom_low_is_power_of_two_i64(int64_t value) {
  return value > 0 && (((uint64_t)value & ((uint64_t)value - 1)) == 0);
}

static iree_status_t loom_low_emit_type_constraint_error(
    const loom_op_t* op, loom_diagnostic_field_kind_t field_kind,
    uint16_t field_index, iree_string_view_t field_name,
    loom_type_t actual_type, iree_string_view_t expected_constraint,
    iree_diagnostic_emitter_t emitter) {
  const loom_error_def_t* error = field_kind == LOOM_DIAGNOSTIC_FIELD_RESULT
                                      ? LOOM_ERR_TYPE_004
                                      : LOOM_ERR_TYPE_003;
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(field_name),
          loom_diagnostic_field_ref(field_kind, field_index)),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_low_emit(emitter, op, error, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_emit_subrange_error(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, int64_t offset,
    int64_t size, int64_t bound) {
  loom_diagnostic_param_t params[] = {
      loom_param_i64(0),     loom_param_i64(offset),
      loom_param_i64(size),  loom_param_i64(offset + size),
      loom_param_i64(bound),
  };
  return loom_low_emit(emitter, op, LOOM_ERR_SUBRANGE_004, params,
                       IREE_ARRAYSIZE(params));
}

static uint32_t loom_low_saturating_u32(iree_host_size_t value) {
  if (value > UINT32_MAX) return UINT32_MAX;
  return (uint32_t)value;
}

static iree_status_t loom_low_emit_count_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t field_name, iree_host_size_t actual_count,
    iree_string_view_t expected_field_name, iree_host_size_t expected_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_u32(loom_low_saturating_u32(actual_count)),
      loom_param_string(expected_field_name),
      loom_param_u32(loom_low_saturating_u32(expected_count)),
  };
  return loom_low_emit(emitter, op, LOOM_ERR_STRUCTURE_013, params,
                       IREE_ARRAYSIZE(params));
}

static void loom_low_format_indexed_field_name(char* buffer,
                                               iree_host_size_t buffer_capacity,
                                               const char* prefix,
                                               uint16_t index) {
  iree_snprintf(buffer, buffer_capacity, "%s[%u]", prefix, index);
}

static iree_status_t loom_low_emit_value_type_mismatch(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t value_field_name,
    loom_value_id_t value_id, iree_string_view_t expected_field_name,
    loom_value_id_t expected_value_id) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(value_field_name),
      loom_param_type(loom_module_value_type(module, value_id)),
      loom_param_string(expected_field_name),
      loom_param_type(loom_module_value_type(module, expected_value_id)),
  };
  return loom_low_emit(emitter, op, LOOM_ERR_TYPE_001, params,
                       IREE_ARRAYSIZE(params));
}

static const loom_block_t* loom_low_region_entry_block_or_null(
    const loom_region_t* region) {
  if (!region || region->block_count == 0) return NULL;
  return loom_region_const_entry_block(region);
}

static iree_status_t loom_low_verify_region_has_no_entry_args(
    const loom_region_t* region, iree_string_view_t region_name,
    const loom_op_t* op, iree_diagnostic_emitter_t emitter) {
  const loom_block_t* entry = loom_low_region_entry_block_or_null(region);
  iree_host_size_t actual_count = entry ? entry->arg_count : 0;
  if (actual_count == 0) {
    return iree_ok_status();
  }
  return loom_low_emit_count_mismatch(emitter, op, region_name, actual_count,
                                      IREE_SV("0 branch arguments"), 0);
}

static iree_string_view_t loom_low_actual_ancestor_name(
    const loom_module_t* module, const loom_op_t* nested_op) {
  const loom_op_t* parent = nested_op->parent_op;
  while (parent) {
    if (loom_low_executable_def_isa(parent)) {
      return loom_low_op_name(module, parent);
    }
    loom_func_like_t func = loom_func_like_const_cast(module, parent);
    if (loom_func_like_isa(func) && loom_func_like_body(func)) {
      return loom_low_op_name(module, parent);
    }
    parent = parent->parent_op;
  }
  return IREE_SV("none");
}

static iree_status_t loom_low_emit_low_entry_placement_error(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_view_t subject_name, iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_op_name(module, op)),
      loom_param_string(IREE_SV("required")),
      loom_param_string(subject_name),
      loom_param_string(loom_low_actual_ancestor_name(module, op)),
  };
  return loom_low_emit(emitter, op, LOOM_ERR_STRUCTURE_029, params,
                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_emit_order_error(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_view_t placement, iree_string_view_t reference_name,
    iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_op_name(module, op)),
      loom_param_string(placement),
      loom_param_string(reference_name),
  };
  return loom_low_emit(emitter, op, LOOM_ERR_STRUCTURE_030, params,
                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_emit_block_placement_error(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_view_t expected_block, iree_string_view_t actual_block,
    iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_op_name(module, op)),
      loom_param_string(expected_block),
      loom_param_string(actual_block),
  };
  return loom_low_emit(emitter, op, LOOM_ERR_STRUCTURE_031, params,
                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_emit_structural_origin_error(
    const loom_module_t* module, const loom_op_t* op,
    loom_diagnostic_field_kind_t field_kind, uint16_t field_index,
    iree_string_view_t field_name, iree_string_view_t required_origin,
    iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_op_name(module, op)),
      loom_param_with_field_ref(
          loom_param_string(field_name),
          loom_diagnostic_field_ref(field_kind, field_index)),
      loom_param_string(required_origin),
  };
  return loom_low_emit(emitter, op, LOOM_ERR_STRUCTURE_032, params,
                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_emit_owner_mismatch_error(
    const loom_module_t* module, const loom_op_t* op,
    loom_diagnostic_field_kind_t field_kind, uint16_t field_index,
    iree_string_view_t field_name, iree_string_view_t owner_kind,
    const loom_diagnostic_related_op_t* related, iree_host_size_t related_count,
    iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_op_name(module, op)),
      loom_param_with_field_ref(
          loom_param_string(field_name),
          loom_diagnostic_field_ref(field_kind, field_index)),
      loom_param_string(owner_kind),
  };
  return loom_low_emit_related(emitter, op, LOOM_ERR_STRUCTURE_033, params,
                               IREE_ARRAYSIZE(params), related, related_count);
}

static iree_status_t loom_low_verify_stable_id_attr(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_string_id_t key_id,
    int64_t stable_id, uint16_t stable_id_attr_index,
    iree_string_view_t stable_id_field_name) {
  iree_string_view_t key = loom_low_string_or_empty(module, key_id);
  uint64_t expected_id = loom_stable_id_from_string(key);
  if (stable_id == (int64_t)expected_id) {
    return iree_ok_status();
  }
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(stable_id_field_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    stable_id_attr_index)),
      loom_param_i64(stable_id),
      loom_param_string(IREE_SV("stable ID derived from key")),
  };
  return loom_low_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_verify_same_register_unit_count(
    const loom_module_t* module, const loom_op_t* op, loom_value_id_t source_id,
    loom_value_id_t result_id, iree_diagnostic_emitter_t emitter) {
  const loom_type_t source_type = loom_module_value_type(module, source_id);
  const loom_type_t result_type = loom_module_value_type(module, result_id);
  if (!loom_type_is_register(source_type) ||
      !loom_type_is_register(result_type)) {
    return iree_ok_status();
  }
  if (loom_low_register_type_unit_count(source_type) ==
      loom_low_register_type_unit_count(result_type)) {
    return iree_ok_status();
  }
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(IREE_SV("source")),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, 0)),
      loom_param_type(source_type),
      loom_param_with_field_ref(
          loom_param_string(IREE_SV("result")),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, 0)),
      loom_param_type(result_type),
  };
  return loom_low_emit(emitter, op, LOOM_ERR_TYPE_001, params,
                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_emit_register_value_relation_error(
    const loom_module_t* module, const loom_op_t* op,
    uint16_t source_operand_index, loom_type_t source_type,
    loom_type_t result_type, iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_op_name(module, op)),
      loom_param_with_field_ref(
          loom_param_type(source_type),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND,
                                    source_operand_index)),
      loom_param_with_field_ref(
          loom_param_type(result_type),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, 0)),
  };
  return loom_low_emit(emitter, op, LOOM_ERR_TYPE_017, params,
                       IREE_ARRAYSIZE(params));
}

// Returns true when a slice preserves a static vector's element sequence.
//
// Static vectors use one register unit per element. A structural slice may
// project either a scalar element or a contiguous vector fragment, independent
// of the source and fragment ranks.
static bool loom_low_slice_has_exact_vector_value_relation(
    loom_type_t source_type, loom_type_t result_type) {
  if (!loom_type_register_has_value_type(source_type) ||
      !loom_type_register_has_value_type(result_type)) {
    return false;
  }
  const loom_type_t* source_value_type =
      loom_type_register_value_type(source_type);
  const loom_type_t* result_value_type =
      loom_type_register_value_type(result_type);
  if (!loom_type_is_vector(*source_value_type) ||
      !loom_type_is_all_static(*source_value_type) ||
      !loom_type_element_type_equals(*source_value_type, *result_value_type)) {
    return false;
  }
  uint64_t source_element_count = 0;
  if (!loom_type_static_element_count(*source_value_type,
                                      &source_element_count) ||
      source_element_count != loom_low_register_type_unit_count(source_type)) {
    return false;
  }
  const uint32_t result_unit_count =
      loom_low_register_type_unit_count(result_type);
  if (loom_type_is_scalar(*result_value_type)) return result_unit_count == 1;
  uint64_t result_element_count = 0;
  return loom_type_is_vector(*result_value_type) &&
         loom_type_is_all_static(*result_value_type) &&
         loom_type_static_element_count(*result_value_type,
                                        &result_element_count) &&
         result_element_count == result_unit_count;
}

// Returns true when scalar and static-vector fragments exactly construct a
// static vector represented with one register unit per element.
static bool loom_low_concat_has_exact_vector_value_relation(
    const loom_module_t* module, loom_value_slice_t sources,
    loom_type_t result_type) {
  if (!loom_type_register_has_value_type(result_type)) return false;
  const loom_type_t* result_value_type =
      loom_type_register_value_type(result_type);
  uint64_t result_element_count = 0;
  if (!loom_type_is_vector(*result_value_type) ||
      !loom_type_is_all_static(*result_value_type) ||
      !loom_type_static_element_count(*result_value_type,
                                      &result_element_count) ||
      result_element_count != loom_low_register_type_unit_count(result_type)) {
    return false;
  }
  for (iree_host_size_t i = 0; i < sources.count; ++i) {
    const loom_type_t source_type =
        loom_module_value_type(module, sources.values[i]);
    if (!loom_type_register_has_value_type(source_type)) {
      return false;
    }
    const loom_type_t* source_value_type =
        loom_type_register_value_type(source_type);
    if (!loom_type_element_type_equals(*source_value_type,
                                       *result_value_type)) {
      return false;
    }
    const uint32_t source_unit_count =
        loom_low_register_type_unit_count(source_type);
    if (loom_type_is_scalar(*source_value_type)) {
      if (source_unit_count != 1) return false;
      continue;
    }
    uint64_t source_element_count = 0;
    if (!loom_type_is_vector(*source_value_type) ||
        !loom_type_is_all_static(*source_value_type) ||
        !loom_type_static_element_count(*source_value_type,
                                        &source_element_count) ||
        source_element_count != source_unit_count) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_low_verify_transfer_register_value_relation(
    const loom_module_t* module, const loom_op_t* op, loom_value_id_t source_id,
    loom_value_id_t result_id, iree_diagnostic_emitter_t emitter) {
  const loom_type_t source_type = loom_module_value_type(module, source_id);
  const loom_type_t result_type = loom_module_value_type(module, result_id);
  if (!loom_low_register_type_same_class(source_type, result_type) ||
      !loom_low_register_type_same_unit_count(source_type, result_type) ||
      (!loom_type_register_has_value_type(source_type) &&
       !loom_type_register_has_value_type(result_type)) ||
      loom_type_equal(source_type, result_type)) {
    return iree_ok_status();
  }
  return loom_low_emit_register_value_relation_error(
      module, op, /*source_operand_index=*/0, source_type, result_type,
      emitter);
}

static iree_status_t loom_low_verify_slice_register_range(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const int64_t offset = loom_low_slice_offset(op);
  if (offset < 0) {
    return loom_low_emit_attr_value_error(
        op, loom_low_slice_offset_ATTR_INDEX, IREE_SV("offset"), offset,
        IREE_SV("non-negative register-unit offset"), emitter);
  }

  const loom_type_t source_type =
      loom_module_value_type(module, loom_low_slice_source(op));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_low_slice_result(op));
  if (!loom_type_is_register(source_type) ||
      !loom_type_is_register(result_type)) {
    return iree_ok_status();
  }

  const uint64_t source_unit_count =
      loom_low_register_type_unit_count(source_type);
  const uint64_t result_unit_count =
      loom_low_register_type_unit_count(result_type);
  const uint64_t offset_unit_count = (uint64_t)offset;
  if (offset_unit_count > source_unit_count ||
      result_unit_count > source_unit_count - offset_unit_count) {
    return loom_low_emit_subrange_error(emitter, op, offset,
                                        (int64_t)result_unit_count,
                                        (int64_t)source_unit_count);
  }
  if (!loom_low_register_type_same_class(source_type, result_type) ||
      (!loom_type_register_has_value_type(source_type) &&
       !loom_type_register_has_value_type(result_type)) ||
      (offset == 0 && loom_type_equal(source_type, result_type)) ||
      loom_low_slice_has_exact_vector_value_relation(source_type,
                                                     result_type)) {
    return iree_ok_status();
  }
  return loom_low_emit_register_value_relation_error(
      module, op, /*source_operand_index=*/0, source_type, result_type,
      emitter);
}

static iree_status_t loom_low_verify_concat_register_value_relation(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const loom_value_slice_t sources = loom_low_concat_sources(op);
  if (sources.count == 0) return iree_ok_status();

  const loom_type_t result_type =
      loom_module_value_type(module, loom_low_concat_result(op));
  if (!loom_low_type_is_register(result_type)) return iree_ok_status();

  uint64_t source_unit_count = 0;
  uint16_t diagnostic_source_index = 0;
  bool has_typed_register = loom_type_register_has_value_type(result_type);
  for (uint16_t i = 0; i < sources.count; ++i) {
    const loom_type_t source_type =
        loom_module_value_type(module, sources.values[i]);
    if (!loom_low_register_type_same_class(source_type, result_type)) {
      return iree_ok_status();
    }
    source_unit_count += loom_low_register_type_unit_count(source_type);
    if (loom_type_register_has_value_type(source_type)) {
      if (!has_typed_register) diagnostic_source_index = i;
      has_typed_register = true;
    }
  }
  if (source_unit_count != loom_low_register_type_unit_count(result_type) ||
      !has_typed_register) {
    return iree_ok_status();
  }

  const loom_type_t diagnostic_source_type =
      loom_module_value_type(module, sources.values[diagnostic_source_index]);
  if (sources.count == 1 &&
      loom_type_equal(diagnostic_source_type, result_type)) {
    return iree_ok_status();
  }
  if (loom_low_concat_has_exact_vector_value_relation(module, sources,
                                                      result_type)) {
    return iree_ok_status();
  }
  return loom_low_emit_register_value_relation_error(
      module, op, diagnostic_source_index, diagnostic_source_type, result_type,
      emitter);
}

static const loom_op_t* loom_low_find_enclosing_low_executable_def(
    const loom_module_t* module, const loom_op_t* nested_op) {
  const loom_op_t* parent = nested_op->parent_op;
  while (parent) {
    if (loom_low_executable_def_isa(parent)) {
      return parent;
    }
    loom_func_like_t func = loom_func_like_const_cast(module, parent);
    if (loom_func_like_isa(func) && loom_func_like_body(func)) {
      return NULL;
    }
    parent = parent->parent_op;
  }
  return NULL;
}

static iree_status_t loom_low_verify_nested_under_low_entry(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_view_t subject_name, iree_diagnostic_emitter_t emitter,
    const loom_op_t** out_enclosing_entry) {
  const loom_op_t* enclosing_func =
      loom_low_find_enclosing_low_executable_def(module, op);
  if (!enclosing_func) {
    return loom_low_emit_low_entry_placement_error(module, op, subject_name,
                                                   emitter);
  }
  if (out_enclosing_entry) {
    *out_enclosing_entry = enclosing_func;
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_storage_type(
    const loom_module_t* module, const loom_op_t* op, loom_value_id_t value_id,
    loom_diagnostic_field_kind_t field_kind, uint16_t field_index,
    iree_string_view_t field_name, iree_diagnostic_emitter_t emitter) {
  const loom_type_t storage_type = loom_module_value_type(module, value_id);
  if (loom_type_is_storage(storage_type) &&
      loom_storage_space_is_valid(loom_type_storage_space(storage_type))) {
    return iree_ok_status();
  }
  return loom_low_emit_type_constraint_error(op, field_kind, field_index,
                                             field_name, storage_type,
                                             IREE_SV("storage"), emitter);
}

typedef struct loom_low_static_storage_reference_t {
  // Reserving op that owns the backing storage identity.
  const loom_op_t* reserve_op;

  // Static byte length available through this storage handle.
  int64_t byte_length;
} loom_low_static_storage_reference_t;

static bool loom_low_try_resolve_storage_reference(
    const loom_module_t* module, loom_value_id_t storage_id,
    loom_low_static_storage_reference_t* out_reference) {
  int64_t resolved_byte_length = 0;
  int64_t projection_offset = 0;
  int64_t projection_byte_length = 0;
  while (true) {
    const loom_value_t* storage_value = loom_module_value(module, storage_id);
    if (!storage_value || loom_value_is_block_arg(storage_value)) return false;
    const loom_op_t* defining_op = loom_value_def_op(storage_value);
    if (!defining_op) return false;

    const bool is_reserve = loom_low_storage_reserve_isa(defining_op);
    int64_t available_byte_length = 0;
    if (is_reserve) {
      available_byte_length = loom_low_storage_reserve_byte_length(defining_op);
    } else if (loom_low_storage_view_isa(defining_op)) {
      available_byte_length = loom_low_storage_view_byte_length(defining_op);
    } else {
      return false;
    }
    if (available_byte_length <= 0) return false;

    if (resolved_byte_length == 0) {
      resolved_byte_length = available_byte_length;
    } else if (projection_offset < 0 ||
               projection_offset >= available_byte_length ||
               projection_byte_length >
                   available_byte_length - projection_offset) {
      return false;
    }

    if (is_reserve) {
      *out_reference = (loom_low_static_storage_reference_t){
          .reserve_op = defining_op,
          .byte_length = resolved_byte_length,
      };
      return true;
    }

    projection_offset = loom_low_storage_view_offset(defining_op);
    projection_byte_length = available_byte_length;
    storage_id = loom_low_storage_view_source(defining_op);
  }
}

static iree_status_t loom_low_verify_storage_use(
    const loom_module_t* module, const loom_op_t* op,
    loom_value_id_t storage_id, uint16_t storage_operand_index, int64_t offset,
    uint16_t offset_attr_index, iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_storage_type(
      module, op, storage_id, LOOM_DIAGNOSTIC_FIELD_OPERAND,
      storage_operand_index, IREE_SV("storage"), emitter));

  const loom_op_t* enclosing_func = NULL;
  IREE_RETURN_IF_ERROR(loom_low_verify_nested_under_low_entry(
      module, op, IREE_SV("low executable"), emitter, &enclosing_func));

  loom_low_static_storage_reference_t reference = {0};
  if (!loom_low_try_resolve_storage_reference(module, storage_id, &reference)) {
    return loom_low_emit_structural_origin_error(
        module, op, LOOM_DIAGNOSTIC_FIELD_OPERAND, storage_operand_index,
        IREE_SV("storage"),
        IREE_SV("a low.storage.reserve result or low.storage.view projection"),
        emitter);
  }

  const loom_op_t* reserve_func =
      loom_low_find_enclosing_low_executable_def(module, reference.reserve_op);
  if (reserve_func != enclosing_func) {
    loom_diagnostic_related_op_t related[] = {{
        .label = IREE_SV("storage reserved here"),
        .op = reference.reserve_op,
    }};
    return loom_low_emit_owner_mismatch_error(
        module, op, LOOM_DIAGNOSTIC_FIELD_OPERAND, storage_operand_index,
        IREE_SV("storage"), IREE_SV("low entry"), related,
        IREE_ARRAYSIZE(related), emitter);
  }

  if (offset < 0) {
    return loom_low_emit_attr_value_error(
        op, offset_attr_index, IREE_SV("offset"), offset,
        IREE_SV("non-negative byte offset"), emitter);
  }

  if (offset >= reference.byte_length) {
    return loom_low_emit_subrange_error(emitter, op, offset, 1,
                                        reference.byte_length);
  }

  return iree_ok_status();
}

static bool loom_low_resource_kind_is_known(uint8_t kind) {
  switch (kind) {
    case LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER:
    case LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING:
    case LOOM_LOW_RESOURCE_IMPORT_KIND_COMMAND_INPUT:
      return true;
    default:
      return false;
  }
}

static iree_string_view_t loom_low_resource_import_kind_name(uint8_t kind) {
  switch (kind) {
    case LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER:
      return IREE_SV("native_pointer");
    case LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING:
      return IREE_SV("hal_binding");
    case LOOM_LOW_RESOURCE_IMPORT_KIND_COMMAND_INPUT:
      return IREE_SV("command_input");
    default:
      return IREE_SV("unknown");
  }
}

static bool loom_low_resource_matches_export_abi(uint8_t kind,
                                                 loom_target_abi_kind_t abi) {
  switch (kind) {
    case LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER:
      return abi == LOOM_TARGET_ABI_OBJECT_FUNCTION;
    case LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING:
      return abi == LOOM_TARGET_ABI_HAL_KERNEL;
    case LOOM_LOW_RESOURCE_IMPORT_KIND_COMMAND_INPUT:
      return abi == LOOM_TARGET_ABI_COMMAND_PROGRAM;
    default:
      return false;
  }
}

static loom_target_abi_kind_t loom_low_resource_expected_abi(uint8_t kind) {
  switch (kind) {
    case LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER:
      return LOOM_TARGET_ABI_OBJECT_FUNCTION;
    case LOOM_LOW_RESOURCE_IMPORT_KIND_HAL_BINDING:
      return LOOM_TARGET_ABI_HAL_KERNEL;
    case LOOM_LOW_RESOURCE_IMPORT_KIND_COMMAND_INPUT:
      return LOOM_TARGET_ABI_COMMAND_PROGRAM;
    default:
      return LOOM_TARGET_ABI_UNKNOWN;
  }
}

static iree_status_t loom_low_verify_resource_function_abi(
    const loom_module_t* module, const loom_op_t* resource_op,
    const loom_op_t* function_op, uint16_t function_abi_attr_index,
    loom_target_abi_kind_t abi, uint8_t kind,
    iree_diagnostic_emitter_t emitter) {
  if (loom_low_resource_matches_export_abi(kind, abi)) {
    return iree_ok_status();
  }
  loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("function ABI defined here"),
      .op = function_op,
      .field_ref = loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                             function_abi_attr_index),
  }};
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_op_name(module, resource_op)),
      loom_param_with_field_ref(
          loom_param_string(IREE_SV("import_kind")),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    loom_low_resource_import_kind_ATTR_INDEX)),
      loom_param_string(loom_low_resource_import_kind_name(kind)),
      loom_param_string(
          loom_target_abi_kind_name(loom_low_resource_expected_abi(kind))),
      loom_param_string(loom_target_abi_kind_name(abi)),
  };
  return loom_low_emit_related(emitter, resource_op, LOOM_ERR_TARGET_041,
                               params, IREE_ARRAYSIZE(params), related,
                               IREE_ARRAYSIZE(related));
}

static iree_status_t loom_low_verify_resource_op(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const uint8_t import_kind = loom_low_resource_import_kind(op);
  if (!loom_low_resource_kind_is_known(import_kind)) {
    return loom_low_emit_attr_value_error(
        op, loom_low_resource_import_kind_ATTR_INDEX, IREE_SV("import_kind"),
        import_kind, IREE_SV("supported target resource import kind"), emitter);
  }

  if (loom_low_resource_index(op) < 0) {
    return loom_low_emit_attr_value_error(
        op, loom_low_resource_index_ATTR_INDEX, IREE_SV("index"),
        loom_low_resource_index(op), IREE_SV("non-negative resource index"),
        emitter);
  }

  loom_attribute_t extent =
      loom_op_attrs(op)[loom_low_resource_extent_ATTR_INDEX];
  if (!loom_attr_is_absent(extent) && loom_low_resource_extent(op) < 0) {
    return loom_low_emit_attr_value_error(
        op, loom_low_resource_extent_ATTR_INDEX, IREE_SV("extent"),
        loom_low_resource_extent(op), IREE_SV(">= 0"), emitter);
  }

  if (!loom_attr_is_absent(extent) &&
      loom_low_resource_extent_value_is_present(op)) {
    return loom_low_emit_attr_value_error(
        op, loom_low_resource_extent_ATTR_INDEX, IREE_SV("extent"),
        loom_low_resource_extent(op),
        IREE_SV("absent when extent operand is present"), emitter);
  }

  loom_attribute_t cache_swizzle_stride =
      loom_op_attrs(op)[loom_low_resource_cache_swizzle_stride_ATTR_INDEX];
  if (!loom_attr_is_absent(cache_swizzle_stride)) {
    const int64_t stride = loom_low_resource_cache_swizzle_stride(op);
    if (stride < 0 || stride > 0x3FFF) {
      return loom_low_emit_attr_value_error(
          op, loom_low_resource_cache_swizzle_stride_ATTR_INDEX,
          IREE_SV("cache_swizzle_stride"), stride,
          IREE_SV("14-bit non-negative byte stride"), emitter);
    }
  }

  const loom_type_t source_type =
      loom_low_type_attr(module, loom_low_resource_source_type(op));
  if (loom_type_kind(source_type) == LOOM_TYPE_NONE) {
    return loom_low_emit_attr_value_error(
        op, loom_low_resource_source_type_ATTR_INDEX, IREE_SV("source_type"),
        (int64_t)loom_low_resource_source_type(op), IREE_SV("valid Loom type"),
        emitter);
  }

  const loom_op_t* enclosing_func =
      loom_low_find_enclosing_low_executable_def(module, op);
  if (!enclosing_func) {
    return loom_low_emit_low_entry_placement_error(
        module, op, IREE_SV("low executable"), emitter);
  }

  loom_target_abi_kind_t function_abi = LOOM_TARGET_ABI_UNKNOWN;
  uint16_t function_abi_attr_index = 0;
  if (loom_low_function_explicit_abi(enclosing_func, &function_abi,
                                     &function_abi_attr_index)) {
    return loom_low_verify_resource_function_abi(
        module, op, enclosing_func, function_abi_attr_index, function_abi,
        import_kind, emitter);
  }

  return iree_ok_status();
}

static iree_status_t loom_low_verify_function_preamble(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_diagnostic_emitter_t emitter) {
  loom_region_t* body = NULL;
  if (loom_low_func_def_isa(function_op)) {
    body = loom_low_func_def_body(function_op);
  } else if (loom_low_kernel_def_isa(function_op)) {
    body = loom_low_kernel_def_body(function_op);
  }
  if (body == NULL || body->block_count == 0) {
    return iree_ok_status();
  }

  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  bool preamble_open = true;
  const loom_op_t* nested_op = NULL;
  loom_block_for_each_op(entry_block, nested_op) {
    if (loom_low_live_in_isa(nested_op) || loom_low_resource_isa(nested_op)) {
      if (!preamble_open) {
        return loom_low_emit_order_error(module, nested_op, IREE_SV("before"),
                                         IREE_SV("ordinary low packets"),
                                         emitter);
      }
      continue;
    }
    preamble_open = false;
  }

  for (uint16_t block_index = 1; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    loom_block_for_each_op(block, nested_op) {
      if (!loom_low_live_in_isa(nested_op) &&
          !loom_low_resource_isa(nested_op)) {
        continue;
      }
      return loom_low_emit_block_placement_error(
          module, nested_op, IREE_SV("the low entry block"),
          IREE_SV("a non-entry block"), emitter);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_kernel_returns(
    const loom_module_t* module, const loom_op_t* kernel_op,
    iree_diagnostic_emitter_t emitter) {
  loom_region_t* body = loom_low_kernel_def_body(kernel_op);
  if (body == NULL) {
    return iree_ok_status();
  }
  loom_block_t* block = NULL;
  loom_region_for_each_block(body, block) {
    loom_op_t* nested_op = NULL;
    loom_block_for_each_op(block, nested_op) {
      if (!loom_low_return_isa(nested_op) || nested_op->operand_count == 0) {
        continue;
      }
      loom_diagnostic_param_t params[] = {
          loom_param_string(loom_low_op_name(module, nested_op)),
          loom_param_u32(nested_op->operand_count),
          loom_param_u32(0),
      };
      return loom_low_emit(emitter, nested_op, LOOM_ERR_STRUCTURE_001, params,
                           IREE_ARRAYSIZE(params));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_low_emit_callee_related(
    iree_diagnostic_emitter_t emitter, const loom_op_t* call_op,
    const loom_op_t* definition_op, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("defined here"),
      .op = definition_op,
  }};
  return loom_low_emit_related(emitter, call_op, error, params, param_count,
                               related,
                               definition_op ? IREE_ARRAYSIZE(related) : 0);
}

static iree_status_t loom_low_emit_call_callee_kind_mismatch(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_symbol_ref_t callee, uint16_t callee_attr_index,
    const loom_symbol_t* symbol, iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(loom_low_symbol_name(module, callee)),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    callee_attr_index)),
      loom_param_string(loom_low_symbol_definition_name(symbol)),
      loom_param_string(IREE_SV("low function")),
  };
  return loom_low_emit_callee_related(emitter, call_op, symbol->defining_op,
                                      LOOM_ERR_SYMBOL_003, params,
                                      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_emit_call_count_mismatch(
    const loom_module_t* module, const loom_op_t* call_op,
    const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter, const loom_error_def_t* error,
    uint16_t actual_count, uint16_t expected_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_op_name(module, call_op)),
      loom_param_u32(actual_count),
      loom_param_u32(expected_count),
  };
  return loom_low_emit_callee_related(emitter, call_op,
                                      signature->definition_op, error, params,
                                      IREE_ARRAYSIZE(params));
}

static void loom_low_format_call_field_name(char* buffer,
                                            iree_host_size_t buffer_capacity,
                                            const char* prefix,
                                            uint16_t field_index) {
  iree_snprintf(buffer, buffer_capacity, "%s %u", prefix, field_index);
}

static iree_status_t loom_low_emit_call_type_mismatch(
    const loom_op_t* call_op, const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter,
    loom_diagnostic_field_kind_t field_ref_kind, const char* field_prefix,
    uint16_t field_index, loom_type_t actual_type,
    const char* callee_field_prefix, loom_type_t expected_type) {
  char field_name[32];
  char callee_field_name[32];
  loom_low_format_call_field_name(field_name, sizeof(field_name), field_prefix,
                                  field_index);
  loom_low_format_call_field_name(callee_field_name, sizeof(callee_field_name),
                                  callee_field_prefix, field_index);
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(iree_make_cstring_view(field_name)),
          loom_diagnostic_field_ref(field_ref_kind, field_index)),
      loom_param_type(actual_type),
      loom_param_string(iree_make_cstring_view(callee_field_name)),
      loom_param_type(expected_type),
  };
  return loom_low_emit_callee_related(
      emitter, call_op, signature->definition_op, LOOM_ERR_TYPE_001, params,
      IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_emit_call_impure_callee_error(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_symbol_ref_t callee, iree_string_view_t boundary_name,
    const loom_op_t* related_op, iree_string_view_t related_label,
    iree_diagnostic_emitter_t emitter) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_symbol_name(module, callee)),
      loom_param_string(boundary_name),
  };
  loom_diagnostic_related_op_t related[] = {{
      .label = related_label,
      .op = related_op,
  }};
  return loom_low_emit_related(emitter, call_op, LOOM_ERR_STRUCTURE_034, params,
                               IREE_ARRAYSIZE(params), related,
                               related_op ? IREE_ARRAYSIZE(related) : 0);
}

static bool loom_low_func_like_is_pure(const loom_module_t* module,
                                       const loom_op_t* op) {
  loom_func_like_t func = loom_func_like_const_cast(module, op);
  if (!loom_func_like_isa(func)) {
    return false;
  }
  loom_region_t* body = loom_func_like_body(func);
  if (body) {
    return !loom_region_has_read_effects(body) &&
           !loom_region_has_write_effects(body) &&
           !loom_region_has_convergent_effects(body);
  }
  return loom_func_like_purity(func) != 0;
}

static iree_status_t loom_low_verify_call_argument_count(
    const loom_module_t* module, const loom_op_t* call_op,
    const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  if (call_op->operand_count == signature->argument_count) {
    return iree_ok_status();
  }
  return loom_low_emit_call_count_mismatch(
      module, call_op, signature, emitter, LOOM_ERR_STRUCTURE_001,
      call_op->operand_count, signature->argument_count);
}

static iree_status_t loom_low_verify_call_result_count(
    const loom_module_t* module, const loom_op_t* call_op,
    const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  if (call_op->result_count == signature->result_count) {
    return iree_ok_status();
  }
  return loom_low_emit_call_count_mismatch(
      module, call_op, signature, emitter, LOOM_ERR_STRUCTURE_002,
      call_op->result_count, signature->result_count);
}

static iree_status_t loom_low_verify_call_argument_types(
    const loom_module_t* module, const loom_op_t* call_op,
    const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  uint16_t compare_count = call_op->operand_count;
  if (compare_count > signature->argument_count) {
    compare_count = signature->argument_count;
  }
  const loom_value_id_t* call_operands = loom_op_const_operands(call_op);
  for (uint16_t i = 0; i < compare_count; ++i) {
    loom_type_t actual_type = loom_module_value_type(module, call_operands[i]);
    loom_type_t expected_type =
        loom_module_value_type(module, signature->argument_ids[i]);
    if (loom_type_equal(actual_type, expected_type)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_emit_call_type_mismatch(
        call_op, signature, emitter, LOOM_DIAGNOSTIC_FIELD_OPERAND, "operand",
        i, actual_type, "callee argument", expected_type));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_call_result_types(
    const loom_module_t* module, const loom_op_t* call_op,
    const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  uint16_t compare_count = call_op->result_count;
  if (compare_count > signature->result_count) {
    compare_count = signature->result_count;
  }
  const loom_value_id_t* call_results = loom_op_const_results(call_op);
  for (uint16_t i = 0; i < compare_count; ++i) {
    loom_type_t actual_type = loom_module_value_type(module, call_results[i]);
    loom_type_t expected_type =
        loom_module_value_type(module, signature->result_ids[i]);
    if (loom_type_equal(actual_type, expected_type)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_low_emit_call_type_mismatch(
        call_op, signature, emitter, LOOM_DIAGNOSTIC_FIELD_RESULT, "result", i,
        actual_type, "callee result", expected_type));
  }
  return iree_ok_status();
}

static iree_status_t loom_low_verify_call_purity(
    const loom_module_t* module, const loom_op_t* call_op,
    loom_symbol_ref_t callee, uint8_t purity,
    const loom_low_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  if (purity == 0) {
    return iree_ok_status();
  }

  const loom_op_t* purity_op = signature->definition_op;
  if (loom_low_func_like_is_pure(module, purity_op)) {
    return iree_ok_status();
  }

  return loom_low_emit_call_impure_callee_error(
      module, call_op, callee, loom_low_symbol_name(module, callee), purity_op,
      IREE_SV("contract defined here"), emitter);
}

static iree_status_t loom_low_verify_function_symbol_context(
    const loom_module_t* module, const loom_op_t* op,
    uint16_t callee_attr_index,
    const loom_low_callee_signature_t* callee_signature,
    iree_diagnostic_emitter_t emitter) {
  const loom_op_t* caller_op =
      loom_low_find_enclosing_low_executable_def(module, op);
  if (!caller_op) {
    return loom_low_emit_low_entry_placement_error(
        module, op, IREE_SV("low executable"), emitter);
  }

  loom_symbol_ref_t caller_target = loom_low_function_target(caller_op);
  loom_symbol_ref_t callee_target =
      loom_low_function_target(callee_signature->definition_op);
  if (loom_low_symbol_ref_equal(caller_target, callee_target)) {
    return iree_ok_status();
  }
  loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("callee defined here"),
      .op = callee_signature->definition_op,
  }};
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_op_name(module, op)),
      loom_param_with_field_ref(
          loom_param_string(IREE_SV("callee")),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    callee_attr_index)),
      loom_param_string(loom_low_symbol_name(module, callee_target)),
      loom_param_string(loom_low_symbol_name(module, caller_target)),
  };
  return loom_low_emit_related(emitter, op, LOOM_ERR_TARGET_040, params,
                               IREE_ARRAYSIZE(params), related,
                               IREE_ARRAYSIZE(related));
}

iree_status_t loom_low_copy_verify(const loom_module_t* module,
                                   const loom_op_t* op,
                                   iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_same_register_unit_count(
      module, op, loom_low_copy_source(op), loom_low_copy_result(op), emitter));
  return loom_low_verify_transfer_register_value_relation(
      module, op, loom_low_copy_source(op), loom_low_copy_result(op), emitter);
}

iree_status_t loom_low_move_verify(const loom_module_t* module,
                                   const loom_op_t* op,
                                   iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_same_register_unit_count(
      module, op, loom_low_move_source(op), loom_low_move_result(op), emitter));
  return loom_low_verify_transfer_register_value_relation(
      module, op, loom_low_move_source(op), loom_low_move_result(op), emitter);
}

iree_status_t loom_low_slice_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_slice_register_range(module, op, emitter);
}

iree_status_t loom_low_concat_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_concat_register_value_relation(module, op, emitter);
}

iree_status_t loom_low_live_in_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_qualified_key_attr(
      module, op, emitter, loom_low_live_in_source(op),
      loom_low_live_in_source_ATTR_INDEX, IREE_SV("source"),
      IREE_SV("qualified target live-in key")));
  IREE_RETURN_IF_ERROR(loom_low_verify_stable_id_attr(
      module, op, emitter, loom_low_live_in_source(op),
      loom_low_live_in_source_id(op), loom_low_live_in_source_id_ATTR_INDEX,
      IREE_SV("source_id")));

  const loom_op_t* enclosing_func =
      loom_low_find_enclosing_low_executable_def(module, op);
  if (!enclosing_func) {
    return loom_low_emit_low_entry_placement_error(
        module, op, IREE_SV("low executable"), emitter);
  }
  return iree_ok_status();
}

iree_status_t loom_low_br_verify(const loom_module_t* module,
                                 const loom_op_t* op,
                                 iree_diagnostic_emitter_t emitter) {
  loom_value_slice_t args = loom_low_br_args(op);
  return loom_ops_verify_successor_args(module, emitter, op, IREE_SV("low.br"),
                                        0, loom_low_br_dest(op), args.values,
                                        args.count);
}

iree_status_t loom_low_cond_br_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_ops_verify_successor_args(
      module, emitter, op, IREE_SV("low.cond_br"), 0,
      loom_low_cond_br_true_dest(op), NULL, 0));
  return loom_ops_verify_successor_args(
      module, emitter, op, IREE_SV("low.cond_br"), 1,
      loom_low_cond_br_false_dest(op), NULL, 0);
}

iree_status_t loom_low_switch_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  const loom_successor_slice_t targets = loom_low_switch_target_dests(op);
  if (targets.count == 0) {
    return loom_low_emit_count_mismatch(
        emitter, op, IREE_SV("target_dests"), 0,
        IREE_SV("at least one target destination"), 1);
  }
  IREE_RETURN_IF_ERROR(loom_ops_verify_successor_args(
      module, emitter, op, IREE_SV("low.switch"), 0,
      loom_low_switch_default_dest(op), NULL, 0));
  for (uint16_t i = 0; i < targets.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ops_verify_successor_args(
        module, emitter, op, IREE_SV("low.switch"), (uint16_t)(i + 1),
        targets.blocks[i], NULL, 0));
  }
  return iree_ok_status();
}

iree_status_t loom_low_scf_if_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_nested_under_low_entry(
      module, op, IREE_SV("low executable"), emitter, NULL));
  IREE_RETURN_IF_ERROR(loom_low_verify_region_has_no_entry_args(
      loom_low_scf_if_then_region(op), IREE_SV("then_region"), op, emitter));
  if (loom_low_scf_if_else_region(op)) {
    IREE_RETURN_IF_ERROR(loom_low_verify_region_has_no_entry_args(
        loom_low_scf_if_else_region(op), IREE_SV("else_region"), op, emitter));
  }
  if (op->result_count == 0 || loom_low_scf_if_else_region(op)) {
    return iree_ok_status();
  }
  return loom_low_emit_count_mismatch(
      emitter, op, IREE_SV("else_region"), 0,
      IREE_SV("present when low.scf.if has results"), op->result_count);
}

iree_status_t loom_low_scf_for_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_nested_under_low_entry(
      module, op, IREE_SV("low executable"), emitter, NULL));

  const bool has_unroll_factor = loom_low_scf_for_unroll_factor_is_present(op);
  const bool has_unroll_policy = loom_low_optional_attr_is_present(
      op, loom_low_scf_for_unroll_policy_ATTR_INDEX);
  if (has_unroll_factor && has_unroll_policy) {
    return loom_low_emit_attr_value_error(
        op, loom_low_scf_for_unroll_policy_ATTR_INDEX, IREE_SV("unroll"), 2,
        IREE_SV("either bare unroll or unroll factor, not both"), emitter);
  }

  const loom_block_t* body_entry =
      loom_low_region_entry_block_or_null(loom_low_scf_for_body(op));
  if (!body_entry) {
    return iree_ok_status();
  }

  loom_value_slice_t iter_args = loom_low_scf_for_iter_args(op);
  const iree_host_size_t expected_arg_count =
      (iree_host_size_t)iter_args.count + 1;
  if (body_entry->arg_count != expected_arg_count) {
    return loom_low_emit_count_mismatch(
        emitter, op, IREE_SV("body"), body_entry->arg_count,
        IREE_SV("1 + iter_args"), expected_arg_count);
  }

  loom_value_id_t iv_arg = loom_block_arg_id(body_entry, 0);
  if (!loom_type_equal(
          loom_module_value_type(module, iv_arg),
          loom_module_value_type(module, loom_low_scf_for_lower_bound(op)))) {
    return loom_low_emit_value_type_mismatch(
        module, emitter, op, IREE_SV("body[0]"), iv_arg, IREE_SV("lower_bound"),
        loom_low_scf_for_lower_bound(op));
  }

  for (uint16_t i = 0; i < iter_args.count; ++i) {
    loom_value_id_t body_arg = loom_block_arg_id(body_entry, (uint16_t)(i + 1));
    loom_value_id_t iter_arg = iter_args.values[i];
    if (loom_type_equal(loom_module_value_type(module, body_arg),
                        loom_module_value_type(module, iter_arg))) {
      continue;
    }
    char body_arg_name[32];
    char iter_arg_name[32];
    loom_low_format_indexed_field_name(body_arg_name, sizeof(body_arg_name),
                                       "body", (uint16_t)(i + 1));
    loom_low_format_indexed_field_name(iter_arg_name, sizeof(iter_arg_name),
                                       "iter_args", i);
    return loom_low_emit_value_type_mismatch(
        module, emitter, op, iree_make_cstring_view(body_arg_name), body_arg,
        iree_make_cstring_view(iter_arg_name), iter_arg);
  }

  return iree_ok_status();
}

iree_status_t loom_low_scf_while_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_nested_under_low_entry(
      module, op, IREE_SV("low executable"), emitter, NULL);
}

iree_status_t loom_low_func_def_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_function_contract_verify(module, op, emitter));
  IREE_RETURN_IF_ERROR(loom_low_verify_function_exactness_modes(op, emitter));
  return loom_low_verify_function_preamble(module, op, emitter);
}

iree_status_t loom_low_kernel_def_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_kernel_contract(op, emitter));
  IREE_RETURN_IF_ERROR(loom_low_verify_kernel_exactness_modes(op, emitter));
  IREE_RETURN_IF_ERROR(loom_low_verify_kernel_returns(module, op, emitter));
  return loom_low_verify_function_preamble(module, op, emitter);
}

iree_status_t loom_low_func_decl_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_function_contract_verify(module, op, emitter));
  IREE_RETURN_IF_ERROR(loom_low_verify_function_exactness_modes(op, emitter));
  IREE_RETURN_IF_ERROR(
      loom_function_import_contract_verify(module, op, emitter));
  return loom_low_verify_decl_code_import(module, op, emitter);
}

iree_status_t loom_low_storage_reserve_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_nested_under_low_entry(
      module, op, IREE_SV("low executable"), emitter, NULL));

  IREE_RETURN_IF_ERROR(loom_low_verify_storage_type(
      module, op, loom_low_storage_reserve_storage(op),
      LOOM_DIAGNOSTIC_FIELD_RESULT, 0, IREE_SV("storage"), emitter));

  if (loom_low_storage_reserve_byte_length(op) <= 0) {
    IREE_RETURN_IF_ERROR(loom_low_emit_attr_value_error(
        op, loom_low_storage_reserve_byte_length_ATTR_INDEX,
        IREE_SV("byte_length"), loom_low_storage_reserve_byte_length(op),
        IREE_SV("positive byte length"), emitter));
  }

  if (!loom_low_is_power_of_two_i64(
          loom_low_storage_reserve_byte_alignment(op))) {
    IREE_RETURN_IF_ERROR(loom_low_emit_attr_value_error(
        op, loom_low_storage_reserve_byte_alignment_ATTR_INDEX,
        IREE_SV("byte_alignment"), loom_low_storage_reserve_byte_alignment(op),
        IREE_SV("positive power-of-two byte alignment"), emitter));
  }

  return iree_ok_status();
}

iree_status_t loom_low_storage_view_verify(const loom_module_t* module,
                                           const loom_op_t* op,
                                           iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_storage_use(
      module, op, loom_low_storage_view_source(op), 0,
      loom_low_storage_view_offset(op), loom_low_storage_view_offset_ATTR_INDEX,
      emitter));
  IREE_RETURN_IF_ERROR(loom_low_verify_storage_type(
      module, op, loom_low_storage_view_result(op),
      LOOM_DIAGNOSTIC_FIELD_RESULT, 0, IREE_SV("result"), emitter));
  if (loom_low_storage_view_byte_length(op) <= 0) {
    return loom_low_emit_attr_value_error(
        op, loom_low_storage_view_byte_length_ATTR_INDEX,
        IREE_SV("byte_length"), loom_low_storage_view_byte_length(op),
        IREE_SV("positive byte length"), emitter);
  }

  loom_low_static_storage_reference_t source_reference = {0};
  if (loom_low_try_resolve_storage_reference(
          module, loom_low_storage_view_source(op), &source_reference) &&
      loom_low_storage_view_byte_length(op) >
          source_reference.byte_length - loom_low_storage_view_offset(op)) {
    return loom_low_emit_subrange_error(
        emitter, op, loom_low_storage_view_offset(op),
        loom_low_storage_view_byte_length(op), source_reference.byte_length);
  }
  return iree_ok_status();
}

iree_status_t loom_low_spill_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_storage_use(module, op, loom_low_spill_storage(op), 1,
                                     loom_low_spill_offset(op),
                                     loom_low_spill_offset_ATTR_INDEX, emitter);
}

iree_status_t loom_low_reload_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_storage_use(
      module, op, loom_low_reload_storage(op), 0, loom_low_reload_offset(op),
      loom_low_reload_offset_ATTR_INDEX, emitter);
}

iree_status_t loom_low_storage_address_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_storage_use(
      module, op, loom_low_storage_address_storage(op), 0,
      loom_low_storage_address_offset(op),
      loom_low_storage_address_offset_ATTR_INDEX, emitter));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_low_storage_address_result(op));
  if (!loom_type_is_register(result_type)) {
    return loom_low_emit_type_constraint_error(
        op, LOOM_DIAGNOSTIC_FIELD_RESULT, 0, IREE_SV("result"), result_type,
        IREE_SV("register"), emitter);
  }
  return iree_ok_status();
}

iree_status_t loom_low_resource_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_resource_op(module, op, emitter);
}

iree_status_t loom_low_func_call_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  loom_symbol_ref_t callee = loom_low_func_call_callee(op);
  const loom_symbol_t* symbol = loom_low_lookup_defined_symbol(module, callee);
  if (!symbol) {
    return iree_ok_status();
  }

  loom_low_callee_signature_t low_signature = {0};
  if (!loom_low_load_low_signature(module, symbol, &low_signature)) {
    return loom_low_emit_call_callee_kind_mismatch(
        module, op, callee, loom_low_func_call_callee_ATTR_INDEX, symbol,
        emitter);
  }

  IREE_RETURN_IF_ERROR(loom_low_verify_function_symbol_context(
      module, op, loom_low_func_call_callee_ATTR_INDEX, &low_signature,
      emitter));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_call_argument_count(module, op, &low_signature, emitter));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_call_result_count(module, op, &low_signature, emitter));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_call_argument_types(module, op, &low_signature, emitter));
  IREE_RETURN_IF_ERROR(
      loom_low_verify_call_result_types(module, op, &low_signature, emitter));
  return loom_low_verify_call_purity(module, op, callee,
                                     loom_low_func_call_purity(op),
                                     &low_signature, emitter);
}

static iree_status_t loom_low_emit_indirect_call_count_mismatch(
    const loom_module_t* module, const loom_op_t* op,
    const loom_error_def_t* error, uint16_t actual_count,
    uint16_t expected_count, iree_diagnostic_emitter_t emitter) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_op_name(module, op)),
      loom_param_u32(actual_count),
      loom_param_u32(expected_count),
  };
  return loom_low_emit(emitter, op, error, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_emit_indirect_call_type_mismatch(
    const loom_op_t* op, loom_diagnostic_field_kind_t field_kind,
    uint16_t field_index, const char* field_prefix, loom_type_t actual_type,
    const char* expected_prefix, loom_type_t expected_type,
    iree_diagnostic_emitter_t emitter) {
  char field_name[32];
  char expected_name[32];
  iree_snprintf(field_name, sizeof(field_name), "%s %u", field_prefix,
                field_index);
  iree_snprintf(expected_name, sizeof(expected_name), "%s %u", expected_prefix,
                field_index);
  const uint16_t diagnostic_field_index =
      field_kind == LOOM_DIAGNOSTIC_FIELD_OPERAND ? (uint16_t)(field_index + 1)
                                                  : field_index;
  const loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(iree_make_cstring_view(field_name)),
          loom_diagnostic_field_ref(field_kind, diagnostic_field_index)),
      loom_param_type(actual_type),
      loom_param_string(iree_make_cstring_view(expected_name)),
      loom_param_type(expected_type),
  };
  return loom_low_emit(emitter, op, LOOM_ERR_TYPE_001, params,
                       IREE_ARRAYSIZE(params));
}

static iree_status_t loom_low_verify_indirect_call_value_type(
    const loom_module_t* module, const loom_op_t* op, loom_value_id_t value_id,
    loom_diagnostic_field_kind_t field_kind, uint16_t field_index,
    const char* field_prefix, const char* expected_prefix,
    loom_type_t expected_type, iree_diagnostic_emitter_t emitter) {
  const loom_type_t register_type = loom_module_value_type(module, value_id);
  const loom_type_t* actual_type = loom_type_register_value_type(register_type);
  if (actual_type && loom_type_equal(*actual_type, expected_type)) {
    return iree_ok_status();
  }
  return loom_low_emit_indirect_call_type_mismatch(
      op, field_kind, field_index, field_prefix,
      actual_type ? *actual_type : register_type, expected_prefix,
      expected_type, emitter);
}

static iree_status_t loom_low_verify_function_ref_register(
    const loom_module_t* module, const loom_op_t* op, loom_value_id_t value_id,
    loom_diagnostic_field_kind_t field_kind, uint16_t field_index,
    iree_string_view_t field_name, loom_type_t* out_signature_type,
    iree_diagnostic_emitter_t emitter) {
  if (out_signature_type != NULL) {
    *out_signature_type = loom_type_none();
  }
  const loom_type_t register_type = loom_module_value_type(module, value_id);
  const loom_type_t* reference_type =
      loom_type_register_value_type(register_type);
  if (reference_type != NULL) {
    const loom_type_t signature_type =
        loom_func_ref_resolve_signature(module, *reference_type);
    if (loom_type_is_function(signature_type)) {
      if (out_signature_type != NULL) {
        *out_signature_type = signature_type;
      }
      return iree_ok_status();
    }
  }
  return loom_low_emit_type_constraint_error(
      op, field_kind, field_index, field_name, register_type,
      IREE_SV("register carrying func.ref with a function signature"), emitter);
}

static bool loom_low_function_type_matches_low_signature(
    const loom_module_t* module, loom_type_t function_type,
    const loom_low_callee_signature_t* signature) {
  if (loom_type_func_arg_count(function_type) != signature->argument_count ||
      loom_type_func_result_count(function_type) != signature->result_count) {
    return false;
  }
  const loom_type_t* argument_types = loom_type_func_arg_types(function_type);
  for (uint16_t i = 0; i < signature->argument_count; ++i) {
    const loom_type_t register_type =
        loom_module_value_type(module, signature->argument_ids[i]);
    const loom_type_t* logical_type =
        loom_type_register_value_type(register_type);
    if (logical_type == NULL ||
        !loom_type_equal(*logical_type, argument_types[i])) {
      return false;
    }
  }
  const loom_type_t* result_types = loom_type_func_result_types(function_type);
  for (uint16_t i = 0; i < signature->result_count; ++i) {
    const loom_type_t register_type =
        loom_module_value_type(module, signature->result_ids[i]);
    const loom_type_t* logical_type =
        loom_type_register_value_type(register_type);
    if (logical_type == NULL ||
        !loom_type_equal(*logical_type, result_types[i])) {
      return false;
    }
  }
  return true;
}

iree_status_t loom_low_func_null_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  return loom_low_verify_function_ref_register(
      module, op, loom_low_func_null_result(op), LOOM_DIAGNOSTIC_FIELD_RESULT,
      0, IREE_SV("result"), NULL, emitter);
}

iree_status_t loom_low_func_compare_null_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_function_ref_register(
      module, op, loom_low_func_compare_null_function(op),
      LOOM_DIAGNOSTIC_FIELD_OPERAND, 0, IREE_SV("function"), NULL, emitter));
  return loom_low_verify_indirect_call_value_type(
      module, op, loom_low_func_compare_null_result(op),
      LOOM_DIAGNOSTIC_FIELD_RESULT, 0, "result", "expected result",
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), emitter);
}

iree_status_t loom_low_func_address_verify(const loom_module_t* module,
                                           const loom_op_t* op,
                                           iree_diagnostic_emitter_t emitter) {
  loom_type_t signature_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_verify_function_ref_register(
      module, op, loom_low_func_address_result(op),
      LOOM_DIAGNOSTIC_FIELD_RESULT, 0, IREE_SV("result"), &signature_type,
      emitter));

  const loom_symbol_ref_t callee = loom_low_func_address_callee(op);
  const loom_symbol_t* symbol = loom_low_lookup_defined_symbol(module, callee);
  if (symbol == NULL) return iree_ok_status();
  loom_low_callee_signature_t low_signature = {0};
  if (!loom_low_load_low_signature(module, symbol, &low_signature)) {
    return loom_low_emit_call_callee_kind_mismatch(
        module, op, callee, loom_low_func_address_callee_ATTR_INDEX, symbol,
        emitter);
  }
  IREE_RETURN_IF_ERROR(loom_low_verify_function_symbol_context(
      module, op, loom_low_func_address_callee_ATTR_INDEX, &low_signature,
      emitter));
  if (loom_low_function_type_matches_low_signature(module, signature_type,
                                                   &low_signature)) {
    return iree_ok_status();
  }
  return loom_low_emit_type_constraint_error(
      op, LOOM_DIAGNOSTIC_FIELD_RESULT, 0, IREE_SV("result"),
      loom_module_value_type(module, loom_low_func_address_result(op)),
      IREE_SV("register carrying func.ref matching the callee signature"),
      emitter);
}

iree_status_t loom_low_func_import_resolved_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_indirect_call_value_type(
      module, op, loom_low_func_import_resolved_result(op),
      LOOM_DIAGNOSTIC_FIELD_RESULT, 0, "result", "expected result",
      loom_type_scalar(LOOM_SCALAR_TYPE_I1), emitter));
  IREE_RETURN_IF_ERROR(loom_function_optional_import_contract_verify(
      module, op, loom_low_func_import_resolved_callee(op),
      loom_low_func_import_resolved_callee_ATTR_INDEX, emitter));

  const loom_symbol_ref_t callee = loom_low_func_import_resolved_callee(op);
  const loom_symbol_t* symbol = loom_low_lookup_defined_symbol(module, callee);
  if (symbol == NULL) return iree_ok_status();
  loom_low_callee_signature_t low_signature = {0};
  if (!loom_low_load_low_signature(module, symbol, &low_signature)) {
    return loom_low_emit_call_callee_kind_mismatch(
        module, op, callee, loom_low_func_import_resolved_callee_ATTR_INDEX,
        symbol, emitter);
  }
  return loom_low_verify_function_symbol_context(
      module, op, loom_low_func_import_resolved_callee_ATTR_INDEX,
      &low_signature, emitter);
}

iree_status_t loom_low_func_call_indirect_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_low_verify_nested_under_low_entry(
      module, op, IREE_SV("low executable"), emitter, NULL));

  const loom_value_id_t target_id = loom_low_func_call_indirect_target(op);
  const loom_type_t target_register_type =
      loom_module_value_type(module, target_id);
  const loom_type_t* target_value_type =
      loom_type_register_value_type(target_register_type);
  if (!target_value_type || !loom_func_ref_type_isa(*target_value_type)) {
    return loom_low_emit_type_constraint_error(
        op, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0, IREE_SV("target"),
        target_register_type, IREE_SV("register carrying func.ref"), emitter);
  }
  const loom_type_t target_signature =
      loom_func_ref_resolve_signature(module, *target_value_type);
  if (!loom_type_is_function(target_signature)) {
    return loom_low_emit_type_constraint_error(
        op, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0, IREE_SV("target"),
        target_register_type,
        IREE_SV("register carrying func.ref with a function signature"),
        emitter);
  }

  const loom_value_slice_t operands = loom_low_func_call_indirect_operands(op);
  const loom_value_slice_t results = loom_low_func_call_indirect_results(op);
  const uint16_t expected_argument_count =
      loom_type_func_arg_count(target_signature);
  const uint16_t expected_result_count =
      loom_type_func_result_count(target_signature);
  if (operands.count != expected_argument_count) {
    IREE_RETURN_IF_ERROR(loom_low_emit_indirect_call_count_mismatch(
        module, op, LOOM_ERR_STRUCTURE_001, operands.count,
        expected_argument_count, emitter));
  }
  if (results.count != expected_result_count) {
    IREE_RETURN_IF_ERROR(loom_low_emit_indirect_call_count_mismatch(
        module, op, LOOM_ERR_STRUCTURE_002, results.count,
        expected_result_count, emitter));
  }

  const loom_type_t* expected_argument_types =
      loom_type_func_arg_types(target_signature);
  const uint16_t argument_count = operands.count < expected_argument_count
                                      ? operands.count
                                      : expected_argument_count;
  for (uint16_t i = 0; i < argument_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_indirect_call_value_type(
        module, op, operands.values[i], LOOM_DIAGNOSTIC_FIELD_OPERAND, i,
        "operand", "function argument", expected_argument_types[i], emitter));
  }

  const loom_type_t* expected_result_types =
      loom_type_func_result_types(target_signature);
  const uint16_t result_count = results.count < expected_result_count
                                    ? results.count
                                    : expected_result_count;
  for (uint16_t i = 0; i < result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_low_verify_indirect_call_value_type(
        module, op, results.values[i], LOOM_DIAGNOSTIC_FIELD_RESULT, i,
        "result", "function result", expected_result_types[i], emitter));
  }
  return iree_ok_status();
}

iree_status_t loom_low_func_ref_cast_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  const loom_value_id_t source_id = loom_low_func_ref_cast_source(op);
  const loom_value_id_t result_id = loom_low_func_ref_cast_result(op);
  IREE_RETURN_IF_ERROR(loom_low_verify_same_register_unit_count(
      module, op, source_id, result_id, emitter));

  const loom_type_t source_register_type =
      loom_module_value_type(module, source_id);
  const loom_type_t result_register_type =
      loom_module_value_type(module, result_id);
  const loom_type_t* source_type =
      loom_type_register_value_type(source_register_type);
  if (source_type == NULL || !loom_func_ref_type_isa(*source_type)) {
    return loom_low_emit_type_constraint_error(
        op, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0, IREE_SV("source"),
        source_register_type, IREE_SV("register carrying synchronous func.ref"),
        emitter);
  }
  const loom_type_t* result_type =
      loom_type_register_value_type(result_register_type);
  if (result_type == NULL || !loom_func_ref_type_isa(*result_type)) {
    return loom_low_emit_type_constraint_error(
        op, LOOM_DIAGNOSTIC_FIELD_RESULT, 0, IREE_SV("result"),
        result_register_type, IREE_SV("register carrying yieldable func.ref"),
        emitter);
  }

  const loom_type_t source_signature =
      loom_func_ref_resolve_signature(module, *source_type);
  const loom_type_t result_signature =
      loom_func_ref_resolve_signature(module, *result_type);
  if (loom_func_ref_type_has_yieldability(*source_type) ||
      !loom_func_ref_type_has_yieldability(*result_type) ||
      !loom_type_is_function(source_signature) ||
      !loom_type_equal(source_signature, result_signature)) {
    return loom_low_emit_register_value_relation_error(
        module, op, /*source_operand_index=*/0, source_register_type,
        result_register_type, emitter);
  }
  return iree_ok_status();
}

iree_status_t loom_low_invoke_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  loom_symbol_ref_t callee = loom_low_invoke_callee(op);
  const loom_symbol_t* symbol = loom_low_lookup_defined_symbol(module, callee);
  if (!symbol) {
    return iree_ok_status();
  }

  loom_low_callee_signature_t low_signature = {0};
  if (!loom_low_load_low_signature(module, symbol, &low_signature)) {
    return loom_low_emit_call_callee_kind_mismatch(
        module, op, callee, loom_low_invoke_callee_ATTR_INDEX, symbol, emitter);
  }

  return loom_low_verify_call_purity(
      module, op, callee, loom_low_invoke_purity(op), &low_signature, emitter);
}

loom_trait_flags_t loom_low_func_call_effective_traits(const loom_op_t* op) {
  if (loom_low_func_call_purity(op) != 0) {
    return LOOM_TRAIT_PURE;
  }
  return LOOM_TRAIT_UNKNOWN_EFFECTS;
}

loom_trait_flags_t loom_low_invoke_effective_traits(const loom_op_t* op) {
  if (loom_low_invoke_purity(op) != 0) {
    return LOOM_TRAIT_PURE;
  }
  return LOOM_TRAIT_UNKNOWN_EFFECTS;
}
