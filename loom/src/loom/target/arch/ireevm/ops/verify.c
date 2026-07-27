// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/function_contract_verify.h"
#include "loom/target/arch/ireevm/ops/ops.h"

static iree_string_view_t loom_ireevm_module_string(
    const loom_module_t* module, loom_string_id_t string_id) {
  if (string_id == LOOM_STRING_ID_INVALID ||
      string_id >= module->strings.count) {
    return iree_string_view_empty();
  }
  return module->strings.entries[string_id];
}

static iree_status_t loom_ireevm_emit_type_constraint_error(
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
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_ireevm_emit_string_attr_value_error(
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
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_STRUCTURE_027,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_ireevm_verify_non_empty_string_attr(
    const loom_module_t* module, const loom_op_t* op, uint16_t attr_index,
    loom_string_id_t string_id, iree_string_view_t attr_name,
    iree_string_view_t expected_constraint, iree_diagnostic_emitter_t emitter) {
  if (string_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ireevm import attribute %.*s string id %u is "
                            "invalid",
                            (int)attr_name.size, attr_name.data,
                            (uint32_t)string_id);
  }

  iree_string_view_t value = module->strings.entries[string_id];
  if (!iree_string_view_is_empty(value)) {
    return iree_ok_status();
  }
  return loom_ireevm_emit_string_attr_value_error(
      op, attr_index, attr_name, value, expected_constraint, emitter);
}

iree_status_t loom_ireevm_import_decl_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_function_contract_verify(module, op, emitter));
  return loom_ireevm_verify_non_empty_string_attr(
      module, op, loom_ireevm_import_decl_import_symbol_ATTR_INDEX,
      loom_ireevm_import_decl_import_symbol(op), IREE_SV("symbol"),
      IREE_SV("non-empty import symbol"), emitter);
}

static bool loom_ireevm_type_is_dialect_named(const loom_module_t* module,
                                              loom_type_t type,
                                              iree_string_view_t name,
                                              uint16_t param_count) {
  if (!loom_type_is_dialect(type) ||
      loom_type_dialect_param_count(type) != param_count) {
    return false;
  }
  return iree_string_view_equal(
      loom_ireevm_module_string(module, loom_type_dialect_name_id(type)), name);
}

static bool loom_ireevm_type_is_buffer(const loom_module_t* module,
                                       loom_type_t type) {
  return loom_ireevm_type_is_dialect_named(module, type,
                                           IREE_SV("ireevm.buffer"), 0);
}

static bool loom_ireevm_type_is_buffer_ref(const loom_module_t* module,
                                           loom_type_t type) {
  if (!loom_ireevm_type_is_dialect_named(module, type, IREE_SV("ireevm.ref"),
                                         1)) {
    return false;
  }
  const loom_type_t* params = loom_type_dialect_params(type);
  return params && loom_ireevm_type_is_buffer(module, params[0]);
}

static bool loom_ireevm_type_is_scalar(loom_type_t type,
                                       loom_scalar_type_t scalar_type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == scalar_type;
}

static iree_status_t loom_ireevm_verify_buffer_ref_field(
    const loom_module_t* module, const loom_op_t* op,
    loom_diagnostic_field_kind_t field_kind, uint16_t field_index,
    iree_string_view_t field_name, loom_value_id_t value_id,
    iree_diagnostic_emitter_t emitter) {
  loom_type_t actual_type = loom_module_value_type(module, value_id);
  if (loom_ireevm_type_is_buffer_ref(module, actual_type)) {
    return iree_ok_status();
  }
  return loom_ireevm_emit_type_constraint_error(
      op, field_kind, field_index, field_name, actual_type,
      IREE_SV("ireevm.ref<ireevm.buffer>"), emitter);
}

static iree_status_t loom_ireevm_verify_scalar_field(
    const loom_module_t* module, const loom_op_t* op,
    loom_diagnostic_field_kind_t field_kind, uint16_t field_index,
    iree_string_view_t field_name, loom_value_id_t value_id,
    loom_scalar_type_t scalar_type, iree_diagnostic_emitter_t emitter) {
  loom_type_t actual_type = loom_module_value_type(module, value_id);
  if (loom_ireevm_type_is_scalar(actual_type, scalar_type)) {
    return iree_ok_status();
  }
  const char* expected_type_name = loom_scalar_type_name(scalar_type);
  return loom_ireevm_emit_type_constraint_error(
      op, field_kind, field_index, field_name, actual_type,
      iree_make_cstring_view(expected_type_name), emitter);
}

static loom_scalar_type_t loom_ireevm_buffer_load_result_type(
    loom_op_kind_t kind) {
  switch (kind) {
    case LOOM_OP_IREEVM_BUFFER_LOAD_I8_U:
    case LOOM_OP_IREEVM_BUFFER_LOAD_I8_S:
    case LOOM_OP_IREEVM_BUFFER_LOAD_I16_U:
    case LOOM_OP_IREEVM_BUFFER_LOAD_I16_S:
    case LOOM_OP_IREEVM_BUFFER_LOAD_I32:
      return LOOM_SCALAR_TYPE_I32;
    case LOOM_OP_IREEVM_BUFFER_LOAD_I64:
      return LOOM_SCALAR_TYPE_I64;
    case LOOM_OP_IREEVM_BUFFER_LOAD_F32:
      return LOOM_SCALAR_TYPE_F32;
    case LOOM_OP_IREEVM_BUFFER_LOAD_F64:
      return LOOM_SCALAR_TYPE_F64;
    default:
      return LOOM_SCALAR_TYPE_COUNT_;
  }
}

static loom_scalar_type_t loom_ireevm_buffer_store_value_type(
    loom_op_kind_t kind) {
  switch (kind) {
    case LOOM_OP_IREEVM_BUFFER_STORE_I8:
    case LOOM_OP_IREEVM_BUFFER_STORE_I16:
    case LOOM_OP_IREEVM_BUFFER_STORE_I32:
      return LOOM_SCALAR_TYPE_I32;
    case LOOM_OP_IREEVM_BUFFER_STORE_I64:
      return LOOM_SCALAR_TYPE_I64;
    case LOOM_OP_IREEVM_BUFFER_STORE_F32:
      return LOOM_SCALAR_TYPE_F32;
    case LOOM_OP_IREEVM_BUFFER_STORE_F64:
      return LOOM_SCALAR_TYPE_F64;
    default:
      return LOOM_SCALAR_TYPE_COUNT_;
  }
}

iree_status_t loom_ireevm_buffer_op_verify(const loom_module_t* module,
                                           const loom_op_t* op,
                                           iree_diagnostic_emitter_t emitter) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_value_id_t* results = loom_op_const_results(op);
  IREE_RETURN_IF_ERROR(loom_ireevm_verify_buffer_ref_field(
      module, op, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0, IREE_SV("buffer"),
      operands[0], emitter));

  if (op->kind == LOOM_OP_IREEVM_BUFFER_LENGTH) {
    return loom_ireevm_verify_scalar_field(
        module, op, LOOM_DIAGNOSTIC_FIELD_RESULT, 0, IREE_SV("result"),
        results[0], LOOM_SCALAR_TYPE_I64, emitter);
  }

  IREE_RETURN_IF_ERROR(loom_ireevm_verify_scalar_field(
      module, op, LOOM_DIAGNOSTIC_FIELD_OPERAND, 1, IREE_SV("element_offset"),
      operands[1], LOOM_SCALAR_TYPE_I64, emitter));

  loom_scalar_type_t load_result_type =
      loom_ireevm_buffer_load_result_type(op->kind);
  if (load_result_type != LOOM_SCALAR_TYPE_COUNT_) {
    return loom_ireevm_verify_scalar_field(
        module, op, LOOM_DIAGNOSTIC_FIELD_RESULT, 0, IREE_SV("result"),
        results[0], load_result_type, emitter);
  }

  loom_scalar_type_t store_value_type =
      loom_ireevm_buffer_store_value_type(op->kind);
  if (store_value_type != LOOM_SCALAR_TYPE_COUNT_) {
    return loom_ireevm_verify_scalar_field(
        module, op, LOOM_DIAGNOSTIC_FIELD_OPERAND, 2, IREE_SV("value"),
        operands[2], store_value_type, emitter);
  }

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown IREE VM buffer op kind %u",
                          (unsigned)op->kind);
}
