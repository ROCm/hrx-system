// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ops/scalar/ops.h"

static iree_status_t loom_scalar_emit(iree_diagnostic_emitter_t emitter,
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

static iree_status_t loom_scalar_emit_assume_count_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("values")),
      loom_param_u32(op->operand_count),
      loom_param_string(IREE_SV("results")),
      loom_param_u32(op->result_count),
  };
  return loom_scalar_emit(emitter, op, LOOM_ERR_STRUCTURE_013, params,
                          IREE_ARRAYSIZE(params));
}

static void loom_scalar_format_assume_field_name(
    char* buffer, iree_host_size_t buffer_capacity, const char* prefix,
    uint16_t field_index) {
  iree_snprintf(buffer, buffer_capacity, "%s[%u]", prefix, field_index);
}

static iree_status_t loom_scalar_emit_assume_type_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    uint16_t field_index, loom_type_t value_type, loom_type_t result_type) {
  char value_name[32];
  char result_name[32];
  loom_scalar_format_assume_field_name(value_name, sizeof(value_name), "values",
                                       field_index);
  loom_scalar_format_assume_field_name(result_name, sizeof(result_name),
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
  return loom_scalar_emit(emitter, op, LOOM_ERR_TYPE_001, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_scalar_emit_attribute_kind_mismatch(
    iree_diagnostic_emitter_t emitter, const loom_op_t* op,
    iree_string_view_t attr_name, uint16_t attr_index,
    loom_attr_kind_t actual_kind, loom_attr_kind_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(attr_name),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE,
                                    attr_index)),
      loom_param_u32(actual_kind),
      loom_param_u32(expected_kind),
  };
  return loom_scalar_emit(emitter, op, LOOM_ERR_TYPE_005, params,
                          IREE_ARRAYSIZE(params));
}

static bool loom_scalar_type_is_integer_payload(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_scalar_type_is_integer(loom_type_element_type(type));
}

iree_status_t loom_scalar_constant_verify(const loom_module_t* module,
                                          const loom_op_t* op,
                                          iree_diagnostic_emitter_t emitter) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_scalar_constant_result(op));
  if (!loom_type_is_scalar(result_type)) {
    return iree_ok_status();
  }
  loom_scalar_type_t result_scalar_type = loom_type_element_type(result_type);
  if (result_scalar_type == LOOM_SCALAR_TYPE_INDEX ||
      result_scalar_type == LOOM_SCALAR_TYPE_OFFSET) {
    loom_diagnostic_param_t params[] = {
        loom_param_string(IREE_SV("result")),
        loom_param_type(result_type),
        loom_param_string(
            IREE_SV("integer or floating-point scalar; use index.constant")),
    };
    loom_diagnostic_emission_t emission = {
        .op = op,
        .error = LOOM_ERR_TYPE_004,
        .params = params,
        .param_count = IREE_ARRAYSIZE(params),
    };
    return iree_diagnostic_emit(emitter, &emission);
  }

  loom_attribute_t value = loom_scalar_constant_value(op);
  loom_attr_kind_t expected_kind = LOOM_ATTR_ANY;
  if (loom_attr_matches_scalar_type(value, result_scalar_type,
                                    &expected_kind)) {
    return iree_ok_status();
  }

  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("value")),
      loom_param_u32(value.kind),
      loom_param_u32(expected_kind),
  };
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TYPE_005,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

iree_status_t loom_scalar_assume_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  if (op->operand_count != op->result_count) {
    return loom_scalar_emit_assume_count_mismatch(emitter, op);
  }

  const loom_value_id_t* values = loom_op_const_operands(op);
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    loom_type_t value_type = loom_module_value_type(module, values[i]);
    loom_type_t result_type = loom_module_value_type(module, results[i]);
    if (!loom_scalar_type_is_integer_payload(value_type) ||
        !loom_scalar_type_is_integer_payload(result_type)) {
      continue;
    }
    if (!loom_type_equal(value_type, result_type)) {
      return loom_scalar_emit_assume_type_mismatch(emitter, op, i, value_type,
                                                   result_type);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_scalar_geluf_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  loom_attribute_t scale_attr = loom_op_attrs(op)[1];
  bool has_scale = !loom_attr_is_absent(scale_attr);
  if (loom_scalar_geluf_variant(op) == LOOM_SCALAR_GELUF_VARIANT_LOGISTIC) {
    if (has_scale) return iree_ok_status();
    return loom_scalar_emit_attribute_kind_mismatch(
        emitter, op, IREE_SV("scale"), /*attr_index=*/1, LOOM_ATTR_ABSENT,
        LOOM_ATTR_F64);
  }
  if (!has_scale) return iree_ok_status();
  return loom_scalar_emit_attribute_kind_mismatch(
      emitter, op, IREE_SV("scale"), /*attr_index=*/1, scale_attr.kind,
      LOOM_ATTR_ABSENT);
}
