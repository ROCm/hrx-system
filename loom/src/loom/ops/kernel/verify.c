// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cache.h"
#include "loom/ops/combining.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/view/ops.h"

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

static iree_status_t loom_kernel_verify_def_contract(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op) {
  const bool has_export_symbol = loom_kernel_optional_attr_is_present(
      op, loom_kernel_def_export_symbol_ATTR_INDEX);
  const bool has_export_linkage = loom_kernel_optional_attr_is_present(
      op, loom_kernel_def_export_linkage_ATTR_INDEX);
  if (!has_export_symbol && has_export_linkage) {
    return loom_kernel_verify_contract_attr_present(
        emitter, op, loom_kernel_def_export_symbol_ATTR_INDEX,
        IREE_SV("export"), IREE_SV("present when linkage is present"));
  }
  return iree_ok_status();
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
  loom_string_id_t name_id = module->values.entries[value_id].name_id;
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

static bool loom_kernel_try_get_local_buffer_memory_space(
    const loom_module_t* module, loom_value_id_t buffer_id,
    loom_value_fact_memory_space_t* out_memory_space) {
  if (buffer_id >= module->values.count) return false;
  const loom_value_t* value = loom_module_value(module, buffer_id);
  if (loom_value_is_block_arg(value)) return false;
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
  return false;
}

static bool loom_kernel_try_get_local_view_memory_space(
    const loom_module_t* module, loom_value_id_t view_id,
    loom_value_fact_memory_space_t* out_memory_space) {
  for (uint8_t depth = 0; depth < 32; ++depth) {
    if (view_id >= module->values.count) return false;
    const loom_value_t* value = loom_module_value(module, view_id);
    if (loom_value_is_block_arg(value)) return false;
    const loom_op_t* defining_op = loom_value_def_op(value);
    if (!defining_op) return false;

    if (loom_buffer_view_isa(defining_op)) {
      return loom_kernel_try_get_local_buffer_memory_space(
          module, loom_buffer_view_buffer(defining_op), out_memory_space);
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
    const loom_module_t* module, loom_value_id_t view_id,
    bool (*predicate)(loom_value_fact_memory_space_t)) {
  loom_value_fact_memory_space_t memory_space =
      LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN;
  if (!loom_kernel_try_get_local_view_memory_space(module, view_id,
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
              module, source_id, loom_kernel_memory_space_is_global_source)) {
        loom_type_t source_type = loom_module_value_type(module, source_id);
        return loom_kernel_emit_operand_constraint(
            emitter, op, IREE_SV("source"), source_type,
            IREE_SV("global, constant, or descriptor memory-space fact"));
      }
      if (!loom_kernel_view_memory_space_is(
              module, dest_id, loom_kernel_memory_space_is_workgroup)) {
        loom_type_t dest_type = loom_module_value_type(module, dest_id);
        return loom_kernel_emit_operand_constraint(
            emitter, op, IREE_SV("dest"), dest_type,
            IREE_SV("workgroup memory-space fact"));
      }
      return iree_ok_status();
    case LOOM_KERNEL_DIRECTION_WORKGROUP_TO_GLOBAL:
      if (!loom_kernel_view_memory_space_is(
              module, source_id, loom_kernel_memory_space_is_workgroup)) {
        loom_type_t source_type = loom_module_value_type(module, source_id);
        return loom_kernel_emit_operand_constraint(
            emitter, op, IREE_SV("source"), source_type,
            IREE_SV("workgroup memory-space fact"));
      }
      if (!loom_kernel_view_memory_space_is(
              module, dest_id, loom_kernel_memory_space_is_global_dest)) {
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
  if (defining_op && loom_kernel_async_group_isa(defining_op)) {
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
          module, source_id, loom_kernel_memory_space_is_global_source)) {
    loom_type_t source_type = loom_module_value_type(module, source_id);
    return loom_kernel_emit_operand_constraint(
        emitter, op, IREE_SV("source"), source_type,
        IREE_SV("global, constant, or descriptor memory-space fact"));
  }
  if (!loom_kernel_view_memory_space_is(
          module, dest_id, loom_kernel_memory_space_is_workgroup)) {
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

iree_status_t loom_kernel_def_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  (void)module;
  return loom_kernel_verify_def_contract(emitter, op);
}

iree_status_t loom_kernel_barrier_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  (void)module;

  loom_value_fact_memory_space_t memory_space =
      loom_kernel_barrier_memory_space(op);
  if (memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return loom_kernel_emit_attribute_value_constraint(
        emitter, op, IREE_SV("memory_space"), memory_space,
        IREE_SV("workgroup memory space"));
  }

  loom_atomic_ordering_t ordering = loom_kernel_barrier_ordering(op);
  if (ordering != LOOM_ATOMIC_ORDERING_ACQ_REL) {
    return loom_kernel_emit_attribute_value_constraint(
        emitter, op, IREE_SV("ordering"), ordering,
        IREE_SV("acq_rel ordering"));
  }

  loom_atomic_scope_t scope = loom_kernel_barrier_scope(op);
  if (scope != LOOM_ATOMIC_SCOPE_SUBGROUP &&
      scope != LOOM_ATOMIC_SCOPE_WORKGROUP) {
    return loom_kernel_emit_attribute_value_constraint(
        emitter, op, IREE_SV("scope"), scope,
        IREE_SV("subgroup or workgroup scope"));
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
