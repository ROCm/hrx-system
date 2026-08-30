// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/control_uniformity.h"
#include "loom/analysis/symbol_facts.h"
#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cache.h"
#include "loom/ops/combining.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"
#include "loom/ops/template/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/util/fact_table.h"
#include "loom/util/walk.h"

static iree_status_t loom_kernel_emit(iree_diagnostic_emitter_t emitter,
                                      const loom_op_t* op,
                                      const loom_error_def_t* error,
                                      const loom_diagnostic_param_t* params,
                                      iree_host_size_t param_count) {
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_kernel_emit_integer_field_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t field_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(field_name),
      loom_param_i64(actual_value),
      loom_param_string(expected_constraint),
  };
  return loom_kernel_emit(emitter, op, LOOM_ERR_STRUCTURE_014, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_kernel_emit_attribute_value_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, int64_t actual_value,
    iree_string_view_t expected_constraint) {
  return loom_kernel_emit_integer_field_constraint(
      emitter, op, attr_name, actual_value, expected_constraint);
}

static bool loom_kernel_optional_attr_is_present(const loom_op_t* op,
                                                 uint16_t attr_index) {
  return attr_index < op->attribute_count &&
         !loom_attr_is_absent(loom_op_attrs(op)[attr_index]);
}

static iree_status_t loom_kernel_verify_contract_attr_present(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, uint16_t attr_index,
    iree_string_view_t attr_name, iree_string_view_t expected_constraint) {
  if (loom_kernel_optional_attr_is_present(op, attr_index)) {
    return iree_ok_status();
  }
  return loom_kernel_emit_attribute_value_constraint(
      emitter, op, attr_name, /*actual_value=*/0, expected_constraint);
}

static iree_status_t loom_kernel_verify_positive_u32_attr(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, uint16_t attr_index,
    int64_t value, iree_string_view_t attr_name) {
  if (!loom_kernel_optional_attr_is_present(op, attr_index)) {
    return iree_ok_status();
  }
  if (value > 0 && value <= UINT32_MAX) {
    return iree_ok_status();
  }
  return loom_kernel_emit_attribute_value_constraint(
      emitter, op, attr_name, value, IREE_SV("positive u32"));
}

static iree_status_t loom_kernel_verify_export_contract(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    uint8_t export_symbol_attr_index, uint8_t export_linkage_attr_index) {
  const bool has_export_symbol =
      loom_kernel_optional_attr_is_present(op, export_symbol_attr_index);
  const bool has_export_linkage =
      loom_kernel_optional_attr_is_present(op, export_linkage_attr_index);
  if (!has_export_symbol && has_export_linkage) {
    return loom_kernel_verify_contract_attr_present(
        emitter, op, export_symbol_attr_index, IREE_SV("export"),
        IREE_SV("present when linkage is present"));
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_verify_launch_config_purity(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const loom_region_t* config = loom_kernel_def_config(op);
  if (!loom_region_has_read_effects(config) &&
      !loom_region_has_write_effects(config) &&
      !loom_region_has_convergent_effects(config)) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
      loom_param_u32(config->read_effect_count),
      loom_param_u32(config->write_effect_count),
      loom_param_u32(config->convergent_effect_count),
  };
  return loom_kernel_emit(emitter, op, LOOM_ERR_STRUCTURE_052, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_kernel_emit_entry_related(
    iree_diagnostic_emitter_t emitter, const loom_op_t* use_op,
    const loom_op_t* definition_op, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("defined here"),
      .op = definition_op,
  }};
  loom_diagnostic_emission_t emission = {
      .op = use_op,
      .error = error,
      .params = params,
      .param_count = param_count,
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static bool loom_kernel_entry_type_matches(
    const loom_module_t* module, loom_type_t actual_type,
    loom_type_t expected_type, const loom_type_value_remap_t* value_remap) {
  const loom_type_kind_t actual_kind = loom_type_kind(actual_type);
  if ((actual_kind == LOOM_TYPE_TENSOR || actual_kind == LOOM_TYPE_VIEW) &&
      loom_type_kind(expected_type) == LOOM_TYPE_BUFFER) {
    return true;
  }
  return loom_type_equal_after_value_remap(module, expected_type, actual_type,
                                           value_remap);
}

static iree_status_t loom_kernel_verify_entry_operand_group(
    const loom_module_t* module, const loom_op_t* launch_op,
    const loom_op_t* definition_op, iree_diagnostic_emitter_t emitter,
    iree_string_view_t actual_field_name,
    iree_string_view_t expected_field_name, loom_value_slice_t actual_values,
    const loom_value_id_t* expected_values, uint16_t expected_value_count,
    uint16_t flat_operand_offset) {
  if (actual_values.count != expected_value_count) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(actual_field_name),
        loom_param_u32(actual_values.count),
        loom_param_string(expected_field_name),
        loom_param_u32(expected_value_count),
    };
    return loom_kernel_emit_entry_related(emitter, launch_op, definition_op,
                                          LOOM_ERR_STRUCTURE_013, params,
                                          IREE_ARRAYSIZE(params));
  }

  loom_type_value_remap_t value_remap = {
      .source_values = expected_values,
      .target_values = actual_values.values,
      .count = actual_values.count,
  };
  for (uint16_t i = 0; i < actual_values.count; ++i) {
    loom_type_t actual_type =
        loom_module_value_type(module, actual_values.values[i]);
    loom_type_t expected_type =
        loom_module_value_type(module, expected_values[i]);
    if (loom_kernel_entry_type_matches(module, actual_type, expected_type,
                                       &value_remap)) {
      continue;
    }

    char actual_name[32];
    char expected_name[32];
    iree_snprintf(actual_name, sizeof(actual_name), "%.*s %u",
                  (int)actual_field_name.size, actual_field_name.data, i);
    iree_snprintf(expected_name, sizeof(expected_name), "%.*s %u",
                  (int)expected_field_name.size, expected_field_name.data, i);
    loom_diagnostic_param_t params[] = {
        loom_param_with_field_ref(
            loom_param_string(iree_make_cstring_view(actual_name)),
            loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND,
                                      flat_operand_offset + i)),
        loom_param_type(actual_type),
        loom_param_string(iree_make_cstring_view(expected_name)),
        loom_param_type(expected_type),
    };
    return loom_kernel_emit_entry_related(emitter, launch_op, definition_op,
                                          LOOM_ERR_TYPE_001, params,
                                          IREE_ARRAYSIZE(params));
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_emit_launch_placement_error(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, iree_string_view_t expected_parent) {
  iree_string_view_t actual_parent =
      op->parent_op ? loom_op_name(module, op->parent_op) : IREE_SV("none");
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
      loom_param_string(IREE_SV("direct")),
      loom_param_string(expected_parent),
      loom_param_string(actual_parent),
  };
  return loom_kernel_emit(emitter, op, LOOM_ERR_STRUCTURE_029, params,
                          IREE_ARRAYSIZE(params));
}

iree_status_t loom_kernel_launch_config_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  (void)module;
  const uint32_t cluster_dimension_count =
      (uint32_t)loom_kernel_launch_config_workgroup_cluster_size_x_is_present(
          op) +
      (uint32_t)loom_kernel_launch_config_workgroup_cluster_size_y_is_present(
          op) +
      (uint32_t)loom_kernel_launch_config_workgroup_cluster_size_z_is_present(
          op);
  if (cluster_dimension_count == 0 || cluster_dimension_count == 3) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("kernel.launch.config")),
      loom_param_u32(op->operand_count),
      loom_param_u32(9),
  };
  return loom_kernel_emit(emitter, op, LOOM_ERR_STRUCTURE_001, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_kernel_emit_operand_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t operand_name, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(operand_name),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_kernel_emit(emitter, op, LOOM_ERR_TYPE_003, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_kernel_emit_result_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t result_name, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(result_name),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_kernel_emit(emitter, op, LOOM_ERR_TYPE_004, params,
                          IREE_ARRAYSIZE(params));
}

static iree_string_view_t loom_kernel_value_name(const loom_module_t* module,
                                                 loom_value_id_t value_id) {
  if (value_id >= module->values.count) return IREE_SV("<invalid>");
  loom_string_id_t name_id = loom_module_value(module, value_id)->name_id;
  if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
    return IREE_SV("<unnamed>");
  }
  return module->strings.entries[name_id];
}

static iree_string_view_t loom_kernel_op_name(const loom_module_t* module,
                                              const loom_op_t* op) {
  if (!op) return IREE_SV("<null>");
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  if (!vtable) return IREE_SV("<unknown>");
  return loom_op_vtable_name(vtable);
}

static iree_status_t loom_kernel_emit_value_use_count_constraint(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t value_id, uint32_t actual_count,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_kernel_value_name(module, value_id)),
      loom_param_u32(actual_count),
      loom_param_string(expected_constraint),
  };
  return loom_kernel_emit(emitter, op, LOOM_ERR_DOMINANCE_009, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_kernel_emit_value_user_constraint(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t value_id, const loom_op_t* user_op,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_kernel_value_name(module, value_id)),
      loom_param_string(loom_kernel_op_name(module, user_op)),
      loom_param_string(expected_constraint),
  };
  return loom_kernel_emit(emitter, op, LOOM_ERR_DOMINANCE_010, params,
                          IREE_ARRAYSIZE(params));
}

static bool loom_kernel_type_is_opaque_dialect(const loom_module_t* module,
                                               loom_type_t type,
                                               iree_string_view_t name) {
  if (!loom_type_is_dialect(type) || loom_type_dialect_param_count(type) != 0) {
    return false;
  }
  loom_string_id_t name_id = loom_type_dialect_name_id(type);
  if (name_id == LOOM_STRING_ID_INVALID || name_id >= module->strings.count) {
    return false;
  }
  return iree_string_view_equal(module->strings.entries[name_id], name);
}

static bool loom_kernel_type_is_async_token(const loom_module_t* module,
                                            loom_type_t type) {
  return loom_kernel_type_is_opaque_dialect(module, type,
                                            IREE_SV("kernel.async.token"));
}

static bool loom_kernel_type_is_async_group(const loom_module_t* module,
                                            loom_type_t type) {
  return loom_kernel_type_is_opaque_dialect(module, type,
                                            IREE_SV("kernel.async.group"));
}

static bool loom_kernel_type_is_tensor_lds_descriptor(
    const loom_module_t* module, loom_type_t type) {
  return loom_kernel_type_is_opaque_dialect(
      module, type, IREE_SV("kernel.tensor.lds.descriptor"));
}

static iree_status_t loom_kernel_verify_result_async_token(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t result_name,
    loom_value_id_t result_id) {
  loom_type_t result_type = loom_module_value_type(module, result_id);
  if (loom_kernel_type_is_async_token(module, result_type)) {
    return iree_ok_status();
  }
  return loom_kernel_emit_result_constraint(
      emitter, op, result_name, result_type, IREE_SV("kernel.async.token"));
}

static iree_status_t loom_kernel_verify_result_async_group(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t result_name,
    loom_value_id_t result_id) {
  loom_type_t result_type = loom_module_value_type(module, result_id);
  if (loom_kernel_type_is_async_group(module, result_type)) {
    return iree_ok_status();
  }
  return loom_kernel_emit_result_constraint(
      emitter, op, result_name, result_type, IREE_SV("kernel.async.group"));
}

static iree_status_t loom_kernel_verify_result_tensor_lds_descriptor(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t result_name,
    loom_value_id_t result_id) {
  loom_type_t result_type = loom_module_value_type(module, result_id);
  if (loom_kernel_type_is_tensor_lds_descriptor(module, result_type)) {
    return iree_ok_status();
  }
  return loom_kernel_emit_result_constraint(
      emitter, op, result_name, result_type,
      IREE_SV("kernel.tensor.lds.descriptor"));
}

static iree_status_t loom_kernel_verify_operand_async_group(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t operand_name,
    loom_value_id_t operand_id) {
  loom_type_t operand_type = loom_module_value_type(module, operand_id);
  if (loom_kernel_type_is_async_group(module, operand_type)) {
    return iree_ok_status();
  }
  return loom_kernel_emit_operand_constraint(
      emitter, op, operand_name, operand_type, IREE_SV("kernel.async.group"));
}

static iree_status_t loom_kernel_verify_operand_tensor_lds_descriptor(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t operand_name,
    loom_value_id_t operand_id) {
  loom_type_t operand_type = loom_module_value_type(module, operand_id);
  if (loom_kernel_type_is_tensor_lds_descriptor(module, operand_type)) {
    return iree_ok_status();
  }
  return loom_kernel_emit_operand_constraint(
      emitter, op, operand_name, operand_type,
      IREE_SV("kernel.tensor.lds.descriptor"));
}

static iree_status_t loom_kernel_verify_operand_i32(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t operand_name,
    loom_value_id_t operand_id) {
  loom_type_t operand_type = loom_module_value_type(module, operand_id);
  if (loom_type_is_scalar(operand_type) &&
      loom_type_element_type(operand_type) == LOOM_SCALAR_TYPE_I32) {
    return iree_ok_status();
  }
  return loom_kernel_emit_operand_constraint(emitter, op, operand_name,
                                             operand_type, IREE_SV("i32"));
}

static bool loom_kernel_type_is_collective_value(loom_type_t type) {
  return loom_type_is_scalar(type) ||
         (loom_type_is_vector(type) && loom_type_rank(type) == 1);
}

static iree_status_t loom_kernel_verify_collective_operand(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t operand_name,
    loom_value_id_t operand_id) {
  loom_type_t operand_type = loom_module_value_type(module, operand_id);
  if (loom_kernel_type_is_collective_value(operand_type)) {
    return iree_ok_status();
  }
  return loom_kernel_emit_operand_constraint(
      emitter, op, operand_name, operand_type,
      IREE_SV("scalar or rank-1 vector"));
}

static iree_status_t loom_kernel_verify_first_result_i32_or_i64(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t result_name) {
  if (op->result_count == 0) {
    return iree_ok_status();
  }
  loom_value_id_t result_id = loom_op_const_results(op)[0];
  loom_type_t result_type = loom_module_value_type(module, result_id);
  if (loom_type_is_scalar(result_type)) {
    loom_scalar_type_t scalar_type = loom_type_element_type(result_type);
    if (scalar_type == LOOM_SCALAR_TYPE_I32 ||
        scalar_type == LOOM_SCALAR_TYPE_I64) {
      return iree_ok_status();
    }
  }
  return loom_kernel_emit_result_constraint(emitter, op, result_name,
                                            result_type, IREE_SV("i32 or i64"));
}

static iree_status_t loom_kernel_verify_result_i1(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, uint16_t result_index,
    iree_string_view_t result_name) {
  if (result_index >= op->result_count) {
    return iree_ok_status();
  }
  loom_value_id_t result_id = loom_op_const_results(op)[result_index];
  loom_type_t result_type = loom_module_value_type(module, result_id);
  if (loom_type_is_scalar(result_type) &&
      loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_I1) {
    return iree_ok_status();
  }
  return loom_kernel_emit_result_constraint(emitter, op, result_name,
                                            result_type, IREE_SV("i1"));
}

static iree_status_t loom_kernel_verify_combining_kind_for_value(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t value_id, loom_combining_kind_t kind) {
  if (!loom_combining_kind_is_valid(kind)) {
    return iree_ok_status();
  }
  loom_type_t value_type = loom_module_value_type(module, value_id);
  if (!loom_kernel_type_is_collective_value(value_type)) {
    return iree_ok_status();
  }
  loom_scalar_type_t element_type = loom_type_element_type(value_type);
  if (loom_scalar_type_is_integer(element_type) &&
      loom_combining_kind_accepts_integer(kind)) {
    return iree_ok_status();
  }
  if (loom_scalar_type_is_float(element_type) &&
      loom_combining_kind_accepts_float(kind)) {
    return iree_ok_status();
  }
  iree_string_view_t expected_constraint =
      loom_combining_kind_accepts_integer(kind)
          ? IREE_SV("integer element type for combining kind")
          : IREE_SV("floating-point element type for combining kind");
  return loom_kernel_emit_operand_constraint(emitter, op, IREE_SV("value"),
                                             value_type, expected_constraint);
}

static iree_status_t loom_kernel_verify_cluster_attrs(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    uint16_t cluster_size_attr_index, int64_t cluster_size,
    uint16_t cluster_stride_attr_index, int64_t cluster_stride) {
  const bool has_cluster_size =
      loom_kernel_optional_attr_is_present(op, cluster_size_attr_index);
  const bool has_cluster_stride =
      loom_kernel_optional_attr_is_present(op, cluster_stride_attr_index);
  if (has_cluster_stride && !has_cluster_size) {
    return loom_kernel_verify_contract_attr_present(
        emitter, op, cluster_size_attr_index, IREE_SV("cluster_size"),
        IREE_SV("present when cluster_stride is present"));
  }
  IREE_RETURN_IF_ERROR(loom_kernel_verify_positive_u32_attr(
      emitter, op, cluster_size_attr_index, cluster_size,
      IREE_SV("cluster_size")));
  return loom_kernel_verify_positive_u32_attr(
      emitter, op, cluster_stride_attr_index, cluster_stride,
      IREE_SV("cluster_stride"));
}

static bool loom_kernel_try_get_block_arg_memory_space(
    const loom_module_t* module, const loom_op_t* use_op,
    const loom_value_t* value,
    loom_value_fact_memory_space_t* out_memory_space) {
  const loom_block_t* block = loom_value_def_block(value);
  const loom_region_t* region = block ? block->parent_region : NULL;
  if (!region) return false;
  for (const loom_op_t* parent_op = use_op ? use_op->parent_op : NULL;
       parent_op != NULL; parent_op = parent_op->parent_op) {
    loom_region_t* const* regions = loom_op_regions(parent_op);
    for (uint8_t i = 0; i < parent_op->region_count; ++i) {
      if (regions[i] != region) continue;
      const loom_op_vtable_t* vtable = loom_op_vtable(module, parent_op);
      const loom_region_descriptor_t* descriptor =
          loom_op_vtable_region_descriptor(vtable, i);
      if (descriptor &&
          iree_any_bit_set(descriptor->flags, LOOM_REGION_GLOBAL_BUFFER_ARGS)) {
        *out_memory_space = LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL;
        return true;
      }
      return false;
    }
  }
  return false;
}

static bool loom_kernel_try_get_local_buffer_memory_space(
    const loom_module_t* module, const loom_op_t* use_op,
    loom_value_id_t buffer_id,
    loom_value_fact_memory_space_t* out_memory_space) {
  for (uint8_t depth = 0; depth < 32; ++depth) {
    if (buffer_id >= module->values.count) return false;
    const loom_value_t* value = loom_module_value(module, buffer_id);
    if (loom_value_is_block_arg(value)) {
      return loom_kernel_try_get_block_arg_memory_space(module, use_op, value,
                                                        out_memory_space);
    }
    const loom_op_t* defining_op = loom_value_def_op(value);
    if (!defining_op) return false;
    if (loom_buffer_alloca_isa(defining_op)) {
      *out_memory_space = loom_buffer_alloca_memory_space(defining_op);
      return true;
    }
    if (loom_buffer_assume_memory_space_isa(defining_op)) {
      *out_memory_space =
          loom_buffer_assume_memory_space_memory_space(defining_op);
      return true;
    }
    const loom_trait_flags_t traits =
        loom_op_effective_traits(module, defining_op);
    const uint16_t result_index = loom_value_def_index(value);
    if (!loom_traits_are_fact_identity(traits) ||
        result_index >= defining_op->operand_count) {
      return false;
    }
    buffer_id = loom_op_const_operands(defining_op)[result_index];
  }
  return false;
}

static bool loom_kernel_try_get_local_view_memory_space(
    const loom_module_t* module, const loom_op_t* use_op,
    loom_value_id_t view_id, loom_value_fact_memory_space_t* out_memory_space) {
  for (uint8_t depth = 0; depth < 32; ++depth) {
    if (view_id >= module->values.count) return false;
    const loom_value_t* value = loom_module_value(module, view_id);
    if (loom_value_is_block_arg(value)) return false;
    const loom_op_t* defining_op = loom_value_def_op(value);
    if (!defining_op) return false;

    if (loom_buffer_view_isa(defining_op)) {
      return loom_kernel_try_get_local_buffer_memory_space(
          module, use_op, loom_buffer_view_buffer(defining_op),
          out_memory_space);
    }
    if (loom_view_subview_isa(defining_op)) {
      view_id = loom_view_subview_source(defining_op);
      continue;
    }
    if (loom_view_refine_isa(defining_op)) {
      view_id = loom_view_refine_source(defining_op);
      continue;
    }
    return false;
  }
  return false;
}

static bool loom_kernel_memory_space_is_global_source(
    loom_value_fact_memory_space_t memory_space) {
  return memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL ||
         memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT ||
         memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR;
}

static bool loom_kernel_memory_space_is_global_dest(
    loom_value_fact_memory_space_t memory_space) {
  return memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL ||
         memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR;
}

static bool loom_kernel_view_memory_space_is(
    const loom_module_t* module, const loom_op_t* use_op,
    loom_value_id_t view_id,
    bool (*predicate)(loom_value_fact_memory_space_t)) {
  loom_value_fact_memory_space_t memory_space =
      LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
  if (!loom_kernel_try_get_local_view_memory_space(module, use_op, view_id,
                                                   &memory_space)) {
    return false;
  }
  return predicate(memory_space);
}

static bool loom_kernel_memory_space_is_workgroup(
    loom_value_fact_memory_space_t memory_space) {
  return memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP;
}

static bool loom_kernel_type_static_element_byte_count(
    loom_type_t type, int64_t* out_byte_count) {
  int32_t bit_count = loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (bit_count <= 0 || (bit_count % 8) != 0) return false;
  *out_byte_count = bit_count / 8;
  return true;
}

static bool loom_kernel_type_static_byte_count_from_axis(
    loom_type_t type, uint8_t first_axis, int64_t* out_byte_count) {
  if (!out_byte_count || !loom_type_is_view(type) ||
      first_axis > loom_type_rank(type)) {
    return false;
  }

  int64_t byte_count = 0;
  if (!loom_kernel_type_static_element_byte_count(type, &byte_count)) {
    return false;
  }
  for (uint8_t axis = first_axis; axis < loom_type_rank(type); ++axis) {
    if (loom_type_dim_is_dynamic_at(type, axis)) return false;
    int64_t dimension_size = loom_type_dim_static_size_at(type, axis);
    if (dimension_size < 0 ||
        !iree_checked_mul_i64(byte_count, dimension_size, &byte_count)) {
      return false;
    }
  }

  *out_byte_count = byte_count;
  return true;
}

static bool loom_kernel_type_static_byte_count(loom_type_t type,
                                               int64_t* out_byte_count) {
  return loom_kernel_type_static_byte_count_from_axis(type, /*first_axis=*/0,
                                                      out_byte_count);
}

static iree_status_t loom_kernel_verify_static_byte_count(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, iree_string_view_t operand_name,
    loom_value_id_t view_id, int64_t* out_byte_count) {
  loom_type_t view_type = loom_module_value_type(module, view_id);
  if (loom_kernel_type_static_byte_count(view_type, out_byte_count)) {
    return iree_ok_status();
  }
  return loom_kernel_emit_operand_constraint(
      emitter, op, operand_name, view_type,
      IREE_SV("view with a static byte-addressable footprint"));
}

static iree_status_t loom_kernel_verify_same_static_byte_count(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t source_id, loom_value_id_t dest_id) {
  int64_t source_byte_count = 0;
  IREE_RETURN_IF_ERROR(loom_kernel_verify_static_byte_count(
      module, emitter, op, IREE_SV("source"), source_id, &source_byte_count));

  int64_t dest_byte_count = 0;
  IREE_RETURN_IF_ERROR(loom_kernel_verify_static_byte_count(
      module, emitter, op, IREE_SV("dest"), dest_id, &dest_byte_count));

  if (source_byte_count == dest_byte_count) return iree_ok_status();
  loom_type_t dest_type = loom_module_value_type(module, dest_id);
  return loom_kernel_emit_operand_constraint(
      emitter, op, IREE_SV("dest"), dest_type,
      IREE_SV("same static byte footprint as source"));
}

static bool loom_kernel_byte_count_is_cluster_load_supported(
    int64_t byte_count) {
  return byte_count == 1 || byte_count == 4 || byte_count == 8 ||
         byte_count == 16;
}

static iree_status_t loom_kernel_verify_cluster_static_byte_count(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t source_id, loom_value_id_t dest_id) {
  int64_t source_byte_count = 0;
  IREE_RETURN_IF_ERROR(loom_kernel_verify_static_byte_count(
      module, emitter, op, IREE_SV("source"), source_id, &source_byte_count));

  int64_t dest_byte_count = 0;
  IREE_RETURN_IF_ERROR(loom_kernel_verify_static_byte_count(
      module, emitter, op, IREE_SV("dest"), dest_id, &dest_byte_count));

  if (source_byte_count != dest_byte_count) {
    loom_type_t dest_type = loom_module_value_type(module, dest_id);
    return loom_kernel_emit_operand_constraint(
        emitter, op, IREE_SV("dest"), dest_type,
        IREE_SV("same static byte footprint as source"));
  }

  if (loom_kernel_byte_count_is_cluster_load_supported(source_byte_count)) {
    return iree_ok_status();
  }
  loom_type_t source_type = loom_module_value_type(module, source_id);
  return loom_kernel_emit_operand_constraint(
      emitter, op, IREE_SV("source"), source_type,
      IREE_SV("view with static 1, 4, 8, or 16 byte footprint"));
}

static iree_status_t loom_kernel_verify_async_copy_memory_spaces(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t source_id, loom_value_id_t dest_id,
    uint8_t direction) {
  switch ((loom_kernel_direction_t)direction) {
    case LOOM_KERNEL_DIRECTION_GLOBAL_TO_WORKGROUP:
      if (!loom_kernel_view_memory_space_is(
              module, op, source_id,
              loom_kernel_memory_space_is_global_source)) {
        loom_type_t source_type = loom_module_value_type(module, source_id);
        return loom_kernel_emit_operand_constraint(
            emitter, op, IREE_SV("source"), source_type,
            IREE_SV("global, constant, or descriptor memory-space fact"));
      }
      if (!loom_kernel_view_memory_space_is(
              module, op, dest_id, loom_kernel_memory_space_is_workgroup)) {
        loom_type_t dest_type = loom_module_value_type(module, dest_id);
        return loom_kernel_emit_operand_constraint(
            emitter, op, IREE_SV("dest"), dest_type,
            IREE_SV("workgroup memory-space fact"));
      }
      return iree_ok_status();
    case LOOM_KERNEL_DIRECTION_WORKGROUP_TO_GLOBAL:
      if (!loom_kernel_view_memory_space_is(
              module, op, source_id, loom_kernel_memory_space_is_workgroup)) {
        loom_type_t source_type = loom_module_value_type(module, source_id);
        return loom_kernel_emit_operand_constraint(
            emitter, op, IREE_SV("source"), source_type,
            IREE_SV("workgroup memory-space fact"));
      }
      if (!loom_kernel_view_memory_space_is(
              module, op, dest_id, loom_kernel_memory_space_is_global_dest)) {
        loom_type_t dest_type = loom_module_value_type(module, dest_id);
        return loom_kernel_emit_operand_constraint(
            emitter, op, IREE_SV("dest"), dest_type,
            IREE_SV("global or descriptor memory-space fact"));
      }
      return iree_ok_status();
    case LOOM_KERNEL_DIRECTION_COUNT_:
      break;
  }
  return loom_kernel_emit_attribute_value_constraint(
      emitter, op, IREE_SV("direction"), direction,
      IREE_SV("global_to_workgroup or workgroup_to_global"));
}

static iree_status_t loom_kernel_verify_async_cache_policy(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op, uint8_t cache_scope,
    uint8_t cache_temporal, bool is_store) {
  loom_cache_policy_error_t error =
      loom_cache_policy_validate(cache_scope, cache_temporal,
                                 is_store ? LOOM_CACHE_POLICY_ACCESS_STORE
                                          : LOOM_CACHE_POLICY_ACCESS_LOAD);
  if (error == LOOM_CACHE_POLICY_ERROR_NONE) return iree_ok_status();
  iree_string_view_t attr_name = loom_cache_policy_error_attr_name(error);
  int64_t actual_value =
      iree_string_view_equal(attr_name, IREE_SV("cache_scope"))
          ? cache_scope
          : cache_temporal;
  return loom_kernel_emit_attribute_value_constraint(
      emitter, op, attr_name, actual_value,
      loom_cache_policy_error_expected_constraint(error));
}

static iree_status_t loom_kernel_verify_copy_token_group_use(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t token_id) {
  if (token_id >= module->values.count) return iree_ok_status();
  const loom_value_t* token = loom_module_value(module, token_id);
  if (token->use_count != 1) {
    return loom_kernel_emit_value_use_count_constraint(
        module, emitter, op, token_id, token->use_count,
        IREE_SV("exactly one kernel.async.group use"));
  }

  const loom_use_t use = loom_value_uses(token)[0];
  const loom_op_t* user_op = loom_use_user_op(use);
  if (loom_kernel_async_group_isa(user_op)) return iree_ok_status();
  return loom_kernel_emit_value_user_constraint(
      module, emitter, op, token_id, user_op,
      IREE_SV("kernel.async.group token operand"));
}

static iree_status_t loom_kernel_verify_token_defined_by_copy(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t token_id) {
  if (token_id >= module->values.count) return iree_ok_status();
  const loom_value_t* token = loom_module_value(module, token_id);
  const loom_op_t* defining_op =
      loom_value_is_block_arg(token) ? NULL : loom_value_def_op(token);
  if (defining_op &&
      (loom_kernel_async_copy_isa(defining_op) ||
       loom_kernel_async_copy_mask_isa(defining_op) ||
       loom_kernel_async_gather_isa(defining_op) ||
       loom_kernel_async_gather_mask_isa(defining_op) ||
       loom_kernel_async_cluster_gather_isa(defining_op) ||
       loom_kernel_async_cluster_gather_mask_isa(defining_op) ||
       loom_kernel_async_tensor_load_to_lds_isa(defining_op) ||
       loom_kernel_async_tensor_store_from_lds_isa(defining_op))) {
    return iree_ok_status();
  }
  return loom_kernel_emit_value_user_constraint(
      module, emitter, op, token_id, defining_op,
      IREE_SV("kernel async transfer result"));
}

static iree_status_t loom_kernel_verify_group_has_uses(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t group_id) {
  if (group_id >= module->values.count) return iree_ok_status();
  const loom_value_t* group = loom_module_value(module, group_id);
  if (group->use_count != 0) return iree_ok_status();
  return loom_kernel_emit_value_use_count_constraint(
      module, emitter, op, group_id, 0,
      IREE_SV("at least one wait or carried group use"));
}

static iree_status_t loom_kernel_verify_group_origin_if_local(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t group_id) {
  if (group_id >= module->values.count) return iree_ok_status();
  const loom_value_t* group = loom_module_value(module, group_id);
  if (loom_value_is_block_arg(group)) return iree_ok_status();
  const loom_op_t* defining_op = loom_value_def_op(group);
  // Template applications are selected and inlined before whole-function async
  // lifetime verification. Runtime calls remain unsupported ownership
  // boundaries and are deliberately rejected here.
  if (defining_op && (loom_kernel_async_group_isa(defining_op) ||
                      loom_template_apply_isa(defining_op))) {
    return iree_ok_status();
  }
  return loom_kernel_emit_value_user_constraint(
      module, emitter, op, group_id, defining_op,
      IREE_SV("kernel.async.group result"));
}

static iree_status_t loom_kernel_verify_gather_destination(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t source_id, loom_value_id_t dest_id) {
  loom_type_t source_type = loom_module_value_type(module, source_id);
  loom_type_t dest_type = loom_module_value_type(module, dest_id);
  if (!loom_type_is_view(source_type) || !loom_type_is_view(dest_type)) {
    return iree_ok_status();
  }

  uint8_t source_rank = loom_type_rank(source_type);
  uint8_t dest_rank = loom_type_rank(dest_type);
  if (dest_rank != source_rank + 1) {
    return loom_kernel_emit_operand_constraint(
        emitter, op, IREE_SV("dest"), dest_type,
        IREE_SV("view with one leading subgroup-lane axis"));
  }

  int64_t source_byte_count = 0;
  IREE_RETURN_IF_ERROR(loom_kernel_verify_static_byte_count(
      module, emitter, op, IREE_SV("source"), source_id, &source_byte_count));

  int64_t dest_lane_byte_count = 0;
  if (!loom_kernel_type_static_byte_count_from_axis(dest_type, /*first_axis=*/1,
                                                    &dest_lane_byte_count)) {
    return loom_kernel_emit_operand_constraint(
        emitter, op, IREE_SV("dest"), dest_type,
        IREE_SV("view with static byte-addressable trailing lane footprint"));
  }
  if (source_byte_count <= dest_lane_byte_count) return iree_ok_status();
  return loom_kernel_emit_operand_constraint(
      emitter, op, IREE_SV("dest"), dest_type,
      IREE_SV("trailing lane byte footprint at least source footprint"));
}

static iree_status_t loom_kernel_verify_gather_memory_spaces(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t source_id, loom_value_id_t dest_id) {
  if (!loom_kernel_view_memory_space_is(
          module, op, source_id, loom_kernel_memory_space_is_global_source)) {
    loom_type_t source_type = loom_module_value_type(module, source_id);
    return loom_kernel_emit_operand_constraint(
        emitter, op, IREE_SV("source"), source_type,
        IREE_SV("global, constant, or descriptor memory-space fact"));
  }
  if (!loom_kernel_view_memory_space_is(
          module, op, dest_id, loom_kernel_memory_space_is_workgroup)) {
    loom_type_t dest_type = loom_module_value_type(module, dest_id);
    return loom_kernel_emit_operand_constraint(
        emitter, op, IREE_SV("dest"), dest_type,
        IREE_SV("workgroup memory-space fact"));
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_verify_async_copy_like(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t source_id, loom_value_id_t dest_id,
    uint8_t direction, uint8_t cache_scope, uint8_t cache_temporal,
    loom_value_id_t token_id) {
  IREE_RETURN_IF_ERROR(loom_kernel_verify_result_async_token(
      module, emitter, op, IREE_SV("token"), token_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_same_static_byte_count(
      module, emitter, op, source_id, dest_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_async_copy_memory_spaces(
      module, emitter, op, source_id, dest_id, direction));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_async_cache_policy(
      emitter, op, cache_scope, cache_temporal,
      direction == LOOM_KERNEL_DIRECTION_WORKGROUP_TO_GLOBAL));
  return loom_kernel_verify_copy_token_group_use(module, emitter, op, token_id);
}

static iree_status_t loom_kernel_verify_async_gather_like(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t source_id, loom_value_id_t dest_id,
    uint8_t cache_scope, uint8_t cache_temporal, loom_value_id_t token_id) {
  IREE_RETURN_IF_ERROR(loom_kernel_verify_result_async_token(
      module, emitter, op, IREE_SV("token"), token_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_gather_destination(
      module, emitter, op, source_id, dest_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_gather_memory_spaces(
      module, emitter, op, source_id, dest_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_async_cache_policy(
      emitter, op, cache_scope, cache_temporal, /*is_store=*/false));
  return loom_kernel_verify_copy_token_group_use(module, emitter, op, token_id);
}

static iree_status_t loom_kernel_verify_async_cluster_gather_like(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t source_id, loom_value_id_t dest_id,
    loom_value_id_t cluster_mask_id, uint8_t cache_scope,
    uint8_t cache_temporal, loom_value_id_t token_id) {
  IREE_RETURN_IF_ERROR(loom_kernel_verify_result_async_token(
      module, emitter, op, IREE_SV("token"), token_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_cluster_static_byte_count(
      module, emitter, op, source_id, dest_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_gather_memory_spaces(
      module, emitter, op, source_id, dest_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_operand_i32(
      module, emitter, op, IREE_SV("cluster_mask"), cluster_mask_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_async_cache_policy(
      emitter, op, cache_scope, cache_temporal, /*is_store=*/false));
  return loom_kernel_verify_copy_token_group_use(module, emitter, op, token_id);
}

static bool loom_kernel_type_is_static_i32_vector(loom_type_t type,
                                                  int64_t lane_count) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         !loom_type_dim_is_dynamic_at(type, 0) &&
         loom_type_dim_static_size_at(type, 0) == lane_count &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

static iree_status_t loom_kernel_verify_tensor_endpoint_types(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t source_id, loom_value_id_t dest_id) {
  loom_type_t source_type = loom_module_value_type(module, source_id);
  loom_type_t dest_type = loom_module_value_type(module, dest_id);
  if (!loom_type_is_view(source_type)) {
    return loom_kernel_emit_operand_constraint(emitter, op, IREE_SV("source"),
                                               source_type, IREE_SV("view"));
  }
  if (!loom_type_is_view(dest_type)) {
    return loom_kernel_emit_operand_constraint(emitter, op, IREE_SV("dest"),
                                               dest_type, IREE_SV("view"));
  }

  uint8_t source_rank = loom_type_rank(source_type);
  if (source_rank == 0 || source_rank > 5) {
    return loom_kernel_emit_operand_constraint(emitter, op, IREE_SV("source"),
                                               source_type,
                                               IREE_SV("view rank in [1, 5]"));
  }
  if (source_rank != loom_type_rank(dest_type)) {
    return loom_kernel_emit_operand_constraint(emitter, op, IREE_SV("dest"),
                                               dest_type,
                                               IREE_SV("same rank as source"));
  }

  loom_scalar_type_t source_element_type = loom_type_element_type(source_type);
  if (source_element_type != loom_type_element_type(dest_type)) {
    return loom_kernel_emit_operand_constraint(
        emitter, op, IREE_SV("dest"), dest_type,
        IREE_SV("same element type as source"));
  }

  int32_t element_bit_count = loom_scalar_type_bitwidth(source_element_type);
  if (element_bit_count != 8 && element_bit_count != 16 &&
      element_bit_count != 32 && element_bit_count != 64) {
    return loom_kernel_emit_operand_constraint(
        emitter, op, IREE_SV("source"), source_type,
        IREE_SV("view with 1, 2, 4, or 8 byte element type"));
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_verify_async_tensor_like(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* op, loom_value_id_t source_id, loom_value_id_t dest_id,
    loom_value_id_t descriptor_id, uint8_t direction, uint8_t cache_scope,
    uint8_t cache_temporal, loom_value_id_t token_id) {
  IREE_RETURN_IF_ERROR(loom_kernel_verify_result_async_token(
      module, emitter, op, IREE_SV("token"), token_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_operand_tensor_lds_descriptor(
      module, emitter, op, IREE_SV("descriptor"), descriptor_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_tensor_endpoint_types(
      module, emitter, op, source_id, dest_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_async_copy_memory_spaces(
      module, emitter, op, source_id, dest_id, direction));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_async_cache_policy(
      emitter, op, cache_scope, cache_temporal,
      direction == LOOM_KERNEL_DIRECTION_WORKGROUP_TO_GLOBAL));
  return loom_kernel_verify_copy_token_group_use(module, emitter, op, token_id);
}

static iree_status_t loom_kernel_emit_barrier_control_constraint(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    const loom_op_t* barrier_op, loom_value_fact_uniform_scope_t required_scope,
    const loom_control_uniformity_failure_t* failure) {
  const iree_string_view_t scope_name =
      loom_control_uniformity_scope_name(required_scope);
  const iree_string_view_t control_value_name =
      failure->control_value == LOOM_VALUE_ID_INVALID
          ? IREE_SV("<unexposed>")
          : loom_kernel_value_name(module, failure->control_value);
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("kernel.barrier")),
      loom_param_string(scope_name),
      loom_param_string(loom_control_uniformity_source_name(failure->source)),
      loom_param_string(control_value_name),
      loom_param_string(loom_kernel_op_name(module, failure->control_op)),
      loom_param_string(loom_control_uniformity_fact_distribution_name(
          failure->control_facts)),
  };
  return loom_kernel_emit(emitter, barrier_op, LOOM_ERR_STRUCTURE_038, params,
                          IREE_ARRAYSIZE(params));
}

static loom_value_fact_uniform_scope_t
loom_kernel_barrier_required_uniform_scope(const loom_op_t* barrier_op) {
  if (!loom_kernel_barrier_isa(barrier_op)) {
    return LOOM_VALUE_FACT_UNIFORM_SCOPE_NONE;
  }
  const loom_atomic_scope_t barrier_scope =
      loom_kernel_barrier_scope(barrier_op);
  if (barrier_scope == LOOM_ATOMIC_SCOPE_SUBGROUP) {
    return LOOM_VALUE_FACT_UNIFORM_SCOPE_SUBGROUP;
  }
  if (barrier_scope == LOOM_ATOMIC_SCOPE_WORKGROUP) {
    return LOOM_VALUE_FACT_UNIFORM_SCOPE_WORKGROUP;
  }
  return LOOM_VALUE_FACT_UNIFORM_SCOPE_NONE;
}

static iree_status_t loom_kernel_verify_barrier_control(
    const loom_module_t* module, iree_diagnostic_emitter_t emitter,
    loom_control_uniformity_info_t* control_uniformity,
    const loom_op_t* barrier_op,
    loom_value_fact_uniform_scope_t required_scope) {
  loom_control_uniformity_failure_t failure = {0};
  bool proven = false;
  IREE_RETURN_IF_ERROR(loom_control_uniformity_prove_execution(
      control_uniformity, barrier_op, required_scope, &failure, &proven));
  if (!proven) {
    return loom_kernel_emit_barrier_control_constraint(
        module, emitter, barrier_op, required_scope, &failure);
  }
  return iree_ok_status();
}

typedef struct loom_kernel_barrier_control_verifier_t {
  // Module containing the kernel and authored target record.
  const loom_module_t* module;
  // Structured diagnostic sink.
  iree_diagnostic_emitter_t emitter;
  // Function whose barriers are being checked.
  loom_func_like_t function;
  // Function-scoped scratch storage.
  iree_arena_allocator_t* arena;
  // Value facts populated when the first relevant barrier is encountered.
  loom_value_fact_table_t fact_table;
  // Reusable control summary over fact_table.
  loom_control_uniformity_info_t control_uniformity;
} loom_kernel_barrier_control_verifier_t;

static iree_status_t loom_kernel_barrier_control_target_facts(
    loom_kernel_barrier_control_verifier_t* verifier,
    const loom_target_facts_t** out_target_facts) {
  *out_target_facts = NULL;
  const loom_symbol_ref_t target_ref =
      loom_func_like_target(verifier->function);
  if (!loom_symbol_ref_is_valid(target_ref) || target_ref.module_id != 0 ||
      target_ref.symbol_id >= verifier->module->symbols.count) {
    return iree_ok_status();
  }

  loom_symbol_fact_table_t symbol_facts = {0};
  loom_symbol_fact_table_initialize(&symbol_facts, verifier->arena);
  const loom_symbol_facts_base_t* base_facts = NULL;
  IREE_RETURN_IF_ERROR(loom_symbol_fact_table_lookup_ref(
      &symbol_facts, verifier->module, target_ref, &base_facts));
  const loom_target_symbol_facts_t* target_facts =
      loom_target_symbol_facts_cast(base_facts);
  if (target_facts) {
    *out_target_facts = target_facts->projection;
  }
  return iree_ok_status();
}

static iree_status_t loom_kernel_barrier_control_verifier_initialize(
    loom_kernel_barrier_control_verifier_t* verifier) {
  if (verifier->fact_table.arena) return iree_ok_status();

  const loom_target_facts_t* target_facts = NULL;
  IREE_RETURN_IF_ERROR(
      loom_kernel_barrier_control_target_facts(verifier, &target_facts));
  IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(
      &verifier->fact_table, verifier->arena, verifier->module->values.count));
  verifier->fact_table.context.target_facts = target_facts;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_compute(
      &verifier->fact_table, verifier->module, verifier->function));
  loom_control_uniformity_info_initialize(
      verifier->module, &verifier->fact_table, verifier->arena,
      &verifier->control_uniformity);
  return iree_ok_status();
}

static iree_status_t loom_kernel_verify_barrier_control_walk(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  (void)context;
  *out_result = LOOM_WALK_CONTINUE;
  loom_kernel_barrier_control_verifier_t* verifier = user_data;
  const loom_value_fact_uniform_scope_t required_scope =
      loom_kernel_barrier_required_uniform_scope(op);
  if (required_scope == LOOM_VALUE_FACT_UNIFORM_SCOPE_NONE) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_kernel_barrier_control_verifier_initialize(verifier));
  return loom_kernel_verify_barrier_control(verifier->module, verifier->emitter,
                                            &verifier->control_uniformity, op,
                                            required_scope);
}

static iree_status_t loom_kernel_verify_barrier_controls(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_func_like_t function = loom_func_like_cast(module, (loom_op_t*)op);
  if (!loom_func_like_isa(function)) return iree_ok_status();

  iree_arena_allocator_t arena;
  iree_arena_initialize(module->arena.block_pool, &arena);

  loom_kernel_barrier_control_verifier_t verifier = {
      .module = module,
      .emitter = emitter,
      .function = function,
      .arena = &arena,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  iree_status_t status =
      loom_walk_region(module, loom_kernel_def_body(op), LOOM_WALK_PRE_ORDER,
                       (loom_walk_callback_t){
                           .fn = loom_kernel_verify_barrier_control_walk,
                           .user_data = &verifier,
                       },
                       &arena, &walk_result);
  iree_arena_deinitialize(&arena);
  return status;
}

iree_status_t loom_kernel_def_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_kernel_verify_export_contract(
      emitter, op, loom_kernel_def_export_symbol_ATTR_INDEX,
      loom_kernel_def_export_linkage_ATTR_INDEX));
  IREE_RETURN_IF_ERROR(
      loom_kernel_verify_launch_config_purity(module, op, emitter));
  return loom_kernel_verify_barrier_controls(module, op, emitter);
}

iree_status_t loom_kernel_decl_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  (void)module;
  return loom_kernel_verify_export_contract(
      emitter, op, loom_kernel_decl_export_symbol_ATTR_INDEX,
      loom_kernel_decl_export_linkage_ATTR_INDEX);
}

static bool loom_kernel_is_indirect_workgroup_count_type(loom_type_t type) {
  return loom_type_is_view(type) && loom_type_rank(type) == 1 &&
         !loom_type_dim_is_dynamic_at(type, 0) &&
         loom_type_dim_static_size_at(type, 0) == 3 &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         !loom_type_has_encoding(type);
}

static iree_status_t loom_kernel_verify_dispatch_workgroup_counts(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const loom_value_slice_t counts = loom_kernel_dispatch_workgroup_counts(op);
  if (counts.count == 0 || counts.count > 3) {
    return loom_kernel_emit_integer_field_constraint(
        emitter, op, IREE_SV("workgroup count operand count"), counts.count,
        IREE_SV("one to three"));
  }

  const loom_type_t first_type =
      loom_module_value_type(module, counts.values[0]);
  if (counts.count == 1 &&
      loom_kernel_is_indirect_workgroup_count_type(first_type)) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < counts.count; ++i) {
    const loom_type_t type = loom_module_value_type(module, counts.values[i]);
    if (loom_type_is_scalar(type) &&
        loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX) {
      continue;
    }
    return loom_kernel_emit_operand_constraint(
        emitter, op, IREE_SV("workgroup_counts"), type,
        counts.count == 1 ? IREE_SV("index or dense view<3xi32>")
                          : IREE_SV("index"));
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_launch_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  loom_symbol_ref_t callee = loom_kernel_launch_callee(op);
  const loom_symbol_t* symbol = &module->symbols.entries[callee.symbol_id];

  loom_value_slice_t workloads = loom_kernel_launch_workloads(op);
  loom_value_slice_t workload_args =
      loom_kernel_workload_arg_ids(module, symbol->defining_op);
  IREE_RETURN_IF_ERROR(loom_kernel_verify_entry_operand_group(
      module, op, symbol->defining_op, emitter, IREE_SV("workload"),
      IREE_SV("kernel workload"), workloads, workload_args.values,
      workload_args.count,
      /*flat_operand_offset=*/0));

  loom_func_like_t kernel = loom_func_like_cast(module, symbol->defining_op);
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(kernel, &argument_count);
  loom_value_slice_t arguments = loom_kernel_launch_arguments(op);
  return loom_kernel_verify_entry_operand_group(
      module, op, symbol->defining_op, emitter, IREE_SV("argument"),
      IREE_SV("kernel ABI argument"), arguments, argument_ids, argument_count,
      workloads.count);
}

iree_status_t loom_kernel_dispatch_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(
      loom_kernel_verify_dispatch_workgroup_counts(module, op, emitter));
  const loom_value_slice_t workgroup_size =
      loom_kernel_dispatch_workgroup_size(op);
  if (workgroup_size.count != 0 && workgroup_size.count != 3) {
    return loom_kernel_emit_integer_field_constraint(
        emitter, op, IREE_SV("workgroup size operand count"),
        workgroup_size.count, IREE_SV("zero or three"));
  }

  const loom_symbol_ref_t callee = loom_kernel_dispatch_callee(op);
  const loom_symbol_t* symbol = &module->symbols.entries[callee.symbol_id];
  const loom_func_like_t entry =
      loom_func_like_cast(module, symbol->defining_op);
  uint16_t argument_count = 0;
  const loom_value_id_t* argument_ids =
      loom_func_like_arg_ids(entry, &argument_count);
  const loom_value_slice_t arguments = loom_kernel_dispatch_arguments(op);
  const loom_value_slice_t workgroup_counts =
      loom_kernel_dispatch_workgroup_counts(op);
  return loom_kernel_verify_entry_operand_group(
      module, op, symbol->defining_op, emitter, IREE_SV("argument"),
      IREE_SV("kernel entry ABI argument"), arguments, argument_ids,
      argument_count, workgroup_counts.count);
}

iree_status_t loom_kernel_launch_yield_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  if (op->parent_op && (loom_kernel_launch_serial_isa(op->parent_op) ||
                        loom_kernel_launch_concurrent_isa(op->parent_op))) {
    return iree_ok_status();
  }
  return loom_kernel_emit_launch_placement_error(
      module, op, emitter,
      IREE_SV("kernel.launch.serial or kernel.launch.concurrent"));
}

iree_status_t loom_kernel_launch_schedule_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_region_t* body = loom_kernel_launch_serial_isa(op)
                            ? loom_kernel_launch_serial_body(op)
                            : loom_kernel_launch_concurrent_body(op);
  const loom_block_t* block = loom_region_const_block(body, 0);
  loom_op_t* child_op = NULL;
  loom_block_for_each_op(block, child_op) {
    if (loom_kernel_launch_isa(child_op) ||
        loom_kernel_dispatch_isa(child_op) ||
        loom_kernel_launch_serial_isa(child_op) ||
        loom_kernel_launch_concurrent_isa(child_op) ||
        loom_kernel_launch_yield_isa(child_op)) {
      continue;
    }
    return loom_kernel_emit_launch_placement_error(
        module, child_op, emitter, IREE_SV("kernel launch schedule"));
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_barrier_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  (void)module;

  loom_value_fact_memory_space_t memory_space =
      loom_kernel_barrier_memory_space(op);
  if (memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP &&
      memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL) {
    return loom_kernel_emit_attribute_value_constraint(
        emitter, op, IREE_SV("memory_space"), memory_space,
        IREE_SV("workgroup or global memory space"));
  }

  loom_atomic_ordering_t ordering = loom_kernel_barrier_ordering(op);
  const bool ordering_supported =
      memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP
          ? ordering == LOOM_ATOMIC_ORDERING_ACQ_REL
          : ordering == LOOM_ATOMIC_ORDERING_ACQUIRE ||
                ordering == LOOM_ATOMIC_ORDERING_RELEASE ||
                ordering == LOOM_ATOMIC_ORDERING_ACQ_REL;
  if (!ordering_supported) {
    return loom_kernel_emit_attribute_value_constraint(
        emitter, op, IREE_SV("ordering"), ordering,
        memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP
            ? IREE_SV("acq_rel ordering for workgroup memory")
            : IREE_SV(
                  "acquire, release, or acq_rel ordering for global memory"));
  }

  loom_atomic_scope_t scope = loom_kernel_barrier_scope(op);
  const bool scope_supported =
      memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP
          ? scope == LOOM_ATOMIC_SCOPE_SUBGROUP ||
                scope == LOOM_ATOMIC_SCOPE_WORKGROUP
          : scope == LOOM_ATOMIC_SCOPE_WORKGROUP;
  if (!scope_supported) {
    return loom_kernel_emit_attribute_value_constraint(
        emitter, op, IREE_SV("scope"), scope,
        memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP
            ? IREE_SV("subgroup or workgroup scope for workgroup memory")
            : IREE_SV("workgroup scope for global memory"));
  }
  return iree_ok_status();
}

iree_status_t loom_kernel_tensor_lds_descriptor_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_value_slice_t dgroups = loom_kernel_tensor_lds_descriptor_dgroups(op);
  if (dgroups.count != 2 && dgroups.count != 4) {
    return loom_kernel_emit_integer_field_constraint(
        emitter, op, IREE_SV("dgroups"), dgroups.count,
        IREE_SV("two or four AMDGPU tensor dgroups"));
  }

  static const int64_t expected_lanes[] = {4, 8, 4, 4};
  for (uint16_t i = 0; i < dgroups.count; ++i) {
    loom_value_id_t dgroup_id = loom_value_slice_get(dgroups, i);
    loom_type_t dgroup_type = loom_module_value_type(module, dgroup_id);
    if (!loom_kernel_type_is_static_i32_vector(dgroup_type,
                                               expected_lanes[i])) {
      return loom_kernel_emit_operand_constraint(
          emitter, op, IREE_SV("dgroups"), dgroup_type,
          i == 1 ? IREE_SV("D1 vector<8xi32>")
                 : IREE_SV("D0/D2/D3 vector<4xi32>"));
    }
  }

  loom_value_id_t descriptor_id =
      loom_kernel_tensor_lds_descriptor_descriptor(op);
  return loom_kernel_verify_result_tensor_lds_descriptor(
      module, emitter, op, IREE_SV("descriptor"), descriptor_id);
}

iree_status_t loom_kernel_subgroup_shuffle_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_kernel_verify_collective_operand(
      module, emitter, op, IREE_SV("value"),
      loom_kernel_subgroup_shuffle_value(op)));
  IREE_RETURN_IF_ERROR(
      loom_kernel_verify_operand_i32(module, emitter, op, IREE_SV("offset"),
                                     loom_kernel_subgroup_shuffle_offset(op)));
  return loom_kernel_verify_operand_i32(module, emitter, op, IREE_SV("width"),
                                        loom_kernel_subgroup_shuffle_width(op));
}

iree_status_t loom_kernel_subgroup_broadcast_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_kernel_verify_collective_operand(
      module, emitter, op, IREE_SV("value"),
      loom_kernel_subgroup_broadcast_value(op)));
  return loom_kernel_verify_operand_i32(
      module, emitter, op, IREE_SV("lane"),
      loom_kernel_subgroup_broadcast_lane(op));
}

iree_status_t loom_kernel_subgroup_value_result_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  if (op->operand_count == 0) {
    return iree_ok_status();
  }
  return loom_kernel_verify_collective_operand(
      module, emitter, op, IREE_SV("value"), loom_op_const_operands(op)[0]);
}

iree_status_t loom_kernel_subgroup_reduce_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_value_id_t value_id = loom_kernel_subgroup_reduce_value(op);
  IREE_RETURN_IF_ERROR(loom_kernel_verify_collective_operand(
      module, emitter, op, IREE_SV("value"), value_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_combining_kind_for_value(
      module, emitter, op, value_id, loom_kernel_subgroup_reduce_kind(op)));
  return loom_kernel_verify_cluster_attrs(
      emitter, op, loom_kernel_subgroup_reduce_cluster_size_ATTR_INDEX,
      loom_kernel_subgroup_reduce_cluster_size(op),
      loom_kernel_subgroup_reduce_cluster_stride_ATTR_INDEX,
      loom_kernel_subgroup_reduce_cluster_stride(op));
}

iree_status_t loom_kernel_subgroup_scan_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_value_id_t value_id = loom_kernel_subgroup_scan_value(op);
  IREE_RETURN_IF_ERROR(loom_kernel_verify_collective_operand(
      module, emitter, op, IREE_SV("value"), value_id));
  IREE_RETURN_IF_ERROR(loom_kernel_verify_combining_kind_for_value(
      module, emitter, op, value_id, loom_kernel_subgroup_scan_kind(op)));
  return loom_kernel_verify_cluster_attrs(
      emitter, op, loom_kernel_subgroup_scan_cluster_size_ATTR_INDEX,
      loom_kernel_subgroup_scan_cluster_size(op),
      loom_kernel_subgroup_scan_cluster_stride_ATTR_INDEX,
      loom_kernel_subgroup_scan_cluster_stride(op));
}

iree_status_t loom_kernel_subgroup_mask_result_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_first_result_i32_or_i64(module, emitter, op,
                                                    IREE_SV("mask"));
}

iree_status_t loom_kernel_subgroup_match_all_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_kernel_verify_first_result_i32_or_i64(
      module, emitter, op, IREE_SV("mask")));
  return loom_kernel_verify_result_i1(module, emitter, op, /*result_index=*/1,
                                      IREE_SV("all_equal"));
}

iree_status_t loom_kernel_workgroup_reduce_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_value_id_t value_id = loom_kernel_workgroup_reduce_value(op);
  IREE_RETURN_IF_ERROR(loom_kernel_verify_collective_operand(
      module, emitter, op, IREE_SV("value"), value_id));
  return loom_kernel_verify_combining_kind_for_value(
      module, emitter, op, value_id, loom_kernel_workgroup_reduce_kind(op));
}

iree_status_t loom_kernel_workgroup_scan_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_value_id_t value_id = loom_kernel_workgroup_scan_value(op);
  IREE_RETURN_IF_ERROR(loom_kernel_verify_collective_operand(
      module, emitter, op, IREE_SV("value"), value_id));
  return loom_kernel_verify_combining_kind_for_value(
      module, emitter, op, value_id, loom_kernel_workgroup_scan_kind(op));
}

iree_status_t loom_kernel_workgroup_vote_count_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_first_result_i32_or_i64(module, emitter, op,
                                                    IREE_SV("result"));
}

iree_status_t loom_kernel_async_copy_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_async_copy_like(
      module, emitter, op, loom_kernel_async_copy_source(op),
      loom_kernel_async_copy_dest(op), loom_kernel_async_copy_direction(op),
      loom_kernel_async_copy_cache_scope(op),
      loom_kernel_async_copy_cache_temporal(op),
      loom_kernel_async_copy_token(op));
}

iree_status_t loom_kernel_async_copy_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_async_copy_like(
      module, emitter, op, loom_kernel_async_copy_mask_source(op),
      loom_kernel_async_copy_mask_dest(op),
      loom_kernel_async_copy_mask_direction(op),
      loom_kernel_async_copy_mask_cache_scope(op),
      loom_kernel_async_copy_mask_cache_temporal(op),
      loom_kernel_async_copy_mask_token(op));
}

iree_status_t loom_kernel_async_gather_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_async_gather_like(
      module, emitter, op, loom_kernel_async_gather_source(op),
      loom_kernel_async_gather_dest(op),
      loom_kernel_async_gather_cache_scope(op),
      loom_kernel_async_gather_cache_temporal(op),
      loom_kernel_async_gather_token(op));
}

iree_status_t loom_kernel_async_gather_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_async_gather_like(
      module, emitter, op, loom_kernel_async_gather_mask_source(op),
      loom_kernel_async_gather_mask_dest(op),
      loom_kernel_async_gather_mask_cache_scope(op),
      loom_kernel_async_gather_mask_cache_temporal(op),
      loom_kernel_async_gather_mask_token(op));
}

iree_status_t loom_kernel_async_cluster_gather_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_async_cluster_gather_like(
      module, emitter, op, loom_kernel_async_cluster_gather_source(op),
      loom_kernel_async_cluster_gather_dest(op),
      loom_kernel_async_cluster_gather_cluster_mask(op),
      loom_kernel_async_cluster_gather_cache_scope(op),
      loom_kernel_async_cluster_gather_cache_temporal(op),
      loom_kernel_async_cluster_gather_token(op));
}

iree_status_t loom_kernel_async_cluster_gather_mask_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_async_cluster_gather_like(
      module, emitter, op, loom_kernel_async_cluster_gather_mask_source(op),
      loom_kernel_async_cluster_gather_mask_dest(op),
      loom_kernel_async_cluster_gather_mask_cluster_mask(op),
      loom_kernel_async_cluster_gather_mask_cache_scope(op),
      loom_kernel_async_cluster_gather_mask_cache_temporal(op),
      loom_kernel_async_cluster_gather_mask_token(op));
}

iree_status_t loom_kernel_async_tensor_load_to_lds_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_async_tensor_like(
      module, emitter, op, loom_kernel_async_tensor_load_to_lds_source(op),
      loom_kernel_async_tensor_load_to_lds_dest(op),
      loom_kernel_async_tensor_load_to_lds_descriptor(op),
      LOOM_KERNEL_DIRECTION_GLOBAL_TO_WORKGROUP,
      loom_kernel_async_tensor_load_to_lds_cache_scope(op),
      loom_kernel_async_tensor_load_to_lds_cache_temporal(op),
      loom_kernel_async_tensor_load_to_lds_token(op));
}

iree_status_t loom_kernel_async_tensor_store_from_lds_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_kernel_verify_async_tensor_like(
      module, emitter, op, loom_kernel_async_tensor_store_from_lds_source(op),
      loom_kernel_async_tensor_store_from_lds_dest(op),
      loom_kernel_async_tensor_store_from_lds_descriptor(op),
      LOOM_KERNEL_DIRECTION_WORKGROUP_TO_GLOBAL,
      loom_kernel_async_tensor_store_from_lds_cache_scope(op),
      loom_kernel_async_tensor_store_from_lds_cache_temporal(op),
      loom_kernel_async_tensor_store_from_lds_token(op));
}

iree_status_t loom_kernel_async_group_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  loom_value_slice_t tokens = loom_kernel_async_group_tokens(op);
  for (uint16_t i = 0; i < tokens.count; ++i) {
    loom_value_id_t token_id = loom_value_slice_get(tokens, i);
    loom_type_t token_type = loom_module_value_type(module, token_id);
    if (!loom_kernel_type_is_async_token(module, token_type)) {
      IREE_RETURN_IF_ERROR(loom_kernel_emit_operand_constraint(
          emitter, op, IREE_SV("tokens"), token_type,
          IREE_SV("kernel.async.token")));
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_kernel_verify_token_defined_by_copy(
        module, emitter, op, token_id));
  }

  loom_value_id_t group_id = loom_kernel_async_group_group(op);
  IREE_RETURN_IF_ERROR(loom_kernel_verify_result_async_group(
      module, emitter, op, IREE_SV("group"), group_id));
  return loom_kernel_verify_group_has_uses(module, emitter, op, group_id);
}

iree_status_t loom_kernel_async_wait_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  loom_value_id_t group_id = loom_kernel_async_wait_group(op);
  IREE_RETURN_IF_ERROR(loom_kernel_verify_operand_async_group(
      module, emitter, op, IREE_SV("group"), group_id));
  int64_t newer_groups = loom_kernel_async_wait_newer_groups(op);
  if (newer_groups < 0 || newer_groups > UINT16_MAX) {
    return loom_kernel_emit_attribute_value_constraint(
        emitter, op, IREE_SV("newer_groups"), newer_groups,
        IREE_SV("nonnegative i16 wait count"));
  }
  return loom_kernel_verify_group_origin_if_local(module, emitter, op,
                                                  group_id);
}
