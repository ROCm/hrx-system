// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"

static iree_status_t loom_index_emit(iree_diagnostic_emitter_t emitter,
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

static bool loom_index_type_is_address(loom_type_t type) {
  if (!loom_type_is_scalar(type)) return false;
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET;
}

static bool loom_index_type_is_integer_or_address(loom_type_t type) {
  if (!loom_type_is_scalar(type)) return false;
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return loom_scalar_type_is_integer(scalar_type) ||
         scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET;
}

static iree_status_t loom_index_emit_operand_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t operand_name, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(operand_name),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_index_emit(emitter, op, LOOM_ERR_TYPE_003, params,
                         IREE_ARRAYSIZE(params));
}

static iree_status_t loom_index_emit_result_constraint(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t result_name, loom_type_t actual_type,
    iree_string_view_t expected_constraint) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(result_name),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_index_emit(emitter, op, LOOM_ERR_TYPE_004, params,
                         IREE_ARRAYSIZE(params));
}

static iree_status_t loom_index_emit_assume_count_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("values")),
      loom_param_u32(op->operand_count),
      loom_param_string(IREE_SV("results")),
      loom_param_u32(op->result_count),
  };
  return loom_index_emit(emitter, op, LOOM_ERR_STRUCTURE_013, params,
                         IREE_ARRAYSIZE(params));
}

static void loom_index_format_assume_field_name(
    char* buffer, iree_host_size_t buffer_capacity, const char* prefix,
    uint16_t field_index) {
  iree_snprintf(buffer, buffer_capacity, "%s[%u]", prefix, field_index);
}

static iree_status_t loom_index_emit_assume_type_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    uint16_t field_index, loom_type_t value_type, loom_type_t result_type) {
  char value_name[32];
  char result_name[32];
  loom_index_format_assume_field_name(value_name, sizeof(value_name), "values",
                                      field_index);
  loom_index_format_assume_field_name(result_name, sizeof(result_name),
                                      "results", field_index);
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(iree_make_cstring_view(value_name)),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND,
                                    field_index)),
      loom_param_type(value_type),
      loom_param_with_field_ref(
          loom_param_string(iree_make_cstring_view(result_name)),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, field_index)),
      loom_param_type(result_type),
  };
  return loom_index_emit(emitter, op, LOOM_ERR_TYPE_001, params,
                         IREE_ARRAYSIZE(params));
}

iree_status_t loom_index_cast_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  loom_type_t input_type =
      loom_module_value_type(module, loom_index_cast_input(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_index_cast_result(op));

  if (!loom_index_type_is_integer_or_address(input_type)) {
    return loom_index_emit_operand_constraint(
        emitter, op, IREE_SV("input"), input_type,
        IREE_SV("integer, index, or offset"));
  }
  if (!loom_index_type_is_integer_or_address(result_type)) {
    return loom_index_emit_result_constraint(
        emitter, op, IREE_SV("result"), result_type,
        IREE_SV("integer, index, or offset"));
  }
  if (!loom_index_type_is_address(input_type) &&
      !loom_index_type_is_address(result_type)) {
    return loom_index_emit_result_constraint(
        emitter, op, IREE_SV("result"), result_type,
        IREE_SV("index or offset boundary"));
  }
  return iree_ok_status();
}

iree_status_t loom_index_assume_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  if (op->operand_count != op->result_count) {
    return loom_index_emit_assume_count_mismatch(emitter, op);
  }

  const loom_value_id_t* values = loom_op_const_operands(op);
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_type_t value_type = loom_module_value_type(module, values[i]);
    loom_type_t result_type = loom_module_value_type(module, results[i]);
    if (!loom_index_type_is_address(value_type) ||
        !loom_index_type_is_address(result_type)) {
      continue;
    }
    if (!loom_type_equal(value_type, result_type)) {
      return loom_index_emit_assume_type_mismatch(emitter, op, i, value_type,
                                                  result_type);
    }
  }
  return iree_ok_status();
}
