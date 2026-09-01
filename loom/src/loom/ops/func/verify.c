// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/error_catalog.h"
#include "loom/ops/callable_effects.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/func/reference.h"
#include "loom/ops/function_contract_verify.h"
#include "loom/ops/type_registry.h"

static iree_status_t loom_func_emit(iree_diagnostic_emitter_t emitter,
                                    const loom_op_t* op,
                                    const loom_error_def_t* error,
                                    const loom_diagnostic_param_t* params,
                                    iree_host_size_t param_count) {
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = error,
      .params = params,
      .param_count = param_count,
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_func_verify_call_purity(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  if (loom_func_call_purity(op) == 0) return iree_ok_status();
  const loom_symbol_ref_t callee = loom_func_call_callee(op);
  if (!loom_symbol_ref_is_valid(callee) || callee.module_id != 0 ||
      callee.symbol_id >= module->symbols.count) {
    return iree_ok_status();
  }
  const loom_symbol_t* symbol = &module->symbols.entries[callee.symbol_id];
  if (!symbol->definition || !symbol->defining_op) return iree_ok_status();
  const loom_func_like_t function =
      loom_func_like_const_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(function) ||
      loom_callable_effects_is_pure(function)) {
    return iree_ok_status();
  }

  const iree_string_view_t callee_name =
      module->strings.entries[symbol->name_id];
  const loom_diagnostic_param_t params[] = {
      loom_param_string(callee_name),
      loom_param_string(callee_name),
  };
  const loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("contract defined here"),
      .op = symbol->defining_op,
  }};
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_STRUCTURE_034,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_func_verify_ref_result(
    const loom_module_t* module, const loom_op_t* op, loom_value_id_t result_id,
    iree_diagnostic_emitter_t emitter) {
  const loom_type_t type = loom_module_value_type(module, result_id);
  if (loom_type_kind(loom_func_ref_resolve_signature(module, type)) !=
      LOOM_TYPE_NONE) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(IREE_SV("result")),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, 0)),
      loom_param_type(type),
      loom_param_string(IREE_SV("func.ref with a function signature")),
  };
  return loom_func_emit(emitter, op, LOOM_ERR_TYPE_004, params,
                        IREE_ARRAYSIZE(params));
}

static iree_status_t loom_func_verify_ref_operand(
    const loom_module_t* module, const loom_op_t* op,
    loom_value_id_t operand_id, iree_diagnostic_emitter_t emitter) {
  const loom_type_t type = loom_module_value_type(module, operand_id);
  if (loom_type_kind(loom_func_ref_resolve_signature(module, type)) !=
      LOOM_TYPE_NONE) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(IREE_SV("function reference")),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, 0)),
      loom_param_type(type),
      loom_param_string(IREE_SV("func.ref with a function signature")),
  };
  return loom_func_emit(emitter, op, LOOM_ERR_TYPE_003, params,
                        IREE_ARRAYSIZE(params));
}

static iree_status_t loom_func_emit_ref_type_mismatch(
    const loom_op_t* op, loom_type_t source_type, loom_type_t result_type,
    iree_diagnostic_emitter_t emitter) {
  const loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(IREE_SV("source")),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, 0)),
      loom_param_type(source_type),
      loom_param_with_field_ref(
          loom_param_string(IREE_SV("result")),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, 0)),
      loom_param_type(result_type),
  };
  return loom_func_emit(emitter, op, LOOM_ERR_TYPE_001, params,
                        IREE_ARRAYSIZE(params));
}

static iree_status_t loom_func_emit_ref_constraint(
    const loom_op_t* op, loom_diagnostic_field_kind_t field_kind,
    iree_string_view_t field_name, loom_type_t actual_type,
    iree_string_view_t expected_constraint, iree_diagnostic_emitter_t emitter) {
  const loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(loom_param_string(field_name),
                                loom_diagnostic_field_ref(field_kind, 0)),
      loom_param_type(actual_type),
      loom_param_string(expected_constraint),
  };
  return loom_func_emit(emitter, op,
                        field_kind == LOOM_DIAGNOSTIC_FIELD_OPERAND
                            ? LOOM_ERR_TYPE_003
                            : LOOM_ERR_TYPE_004,
                        params, IREE_ARRAYSIZE(params));
}

iree_status_t loom_func_def_verify(const loom_module_t* module,
                                   const loom_op_t* op,
                                   iree_diagnostic_emitter_t emitter) {
  return loom_function_contract_verify(module, op, emitter);
}

iree_status_t loom_func_decl_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_function_contract_verify(module, op, emitter));
  return loom_function_import_contract_verify(module, op, emitter);
}

iree_status_t loom_func_null_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  return loom_func_verify_ref_result(module, op, loom_func_null_result(op),
                                     emitter);
}

iree_status_t loom_func_compare_null_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  return loom_func_verify_ref_operand(
      module, op, loom_func_compare_null_function(op), emitter);
}

iree_status_t loom_func_address_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  const loom_value_id_t result_id = loom_func_address_result(op);
  const loom_type_t result_type = loom_module_value_type(module, result_id);
  const loom_type_t signature_type =
      loom_func_ref_resolve_signature(module, result_type);
  if (loom_type_kind(signature_type) == LOOM_TYPE_NONE) {
    return loom_func_verify_ref_result(module, op, result_id, emitter);
  }
  return loom_function_type_contract_verify(
      module, op, loom_func_address_callee(op), signature_type, emitter);
}

iree_status_t loom_func_ref_cast_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  const loom_type_t source_type =
      loom_module_value_type(module, loom_func_ref_cast_source(op));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_func_ref_cast_result(op));
  const loom_type_t source_signature =
      loom_func_ref_resolve_signature(module, source_type);
  if (loom_type_kind(source_signature) == LOOM_TYPE_NONE) {
    return loom_func_verify_ref_operand(module, op,
                                        loom_func_ref_cast_source(op), emitter);
  }
  const loom_type_t result_signature =
      loom_func_ref_resolve_signature(module, result_type);
  if (loom_type_kind(result_signature) == LOOM_TYPE_NONE) {
    return loom_func_verify_ref_result(module, op,
                                       loom_func_ref_cast_result(op), emitter);
  }
  if (loom_func_ref_type_has_yieldability(source_type)) {
    return loom_func_emit_ref_constraint(
        op, LOOM_DIAGNOSTIC_FIELD_OPERAND, IREE_SV("source"), source_type,
        IREE_SV("synchronous func.ref"), emitter);
  }
  if (!loom_func_ref_type_has_yieldability(result_type)) {
    return loom_func_emit_ref_constraint(
        op, LOOM_DIAGNOSTIC_FIELD_RESULT, IREE_SV("result"), result_type,
        IREE_SV("yieldable func.ref"), emitter);
  }
  if (!loom_type_equal(source_signature, result_signature)) {
    return loom_func_emit_ref_type_mismatch(op, source_type, result_type,
                                            emitter);
  }
  return iree_ok_status();
}

iree_status_t loom_func_import_resolved_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  return loom_function_optional_import_contract_verify(
      module, op, loom_func_import_resolved_callee(op),
      loom_func_import_resolved_callee_ATTR_INDEX, emitter);
}

iree_status_t loom_func_call_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(loom_function_call_contract_verify(
      module, op, loom_func_call_callee(op), loom_func_call_operands(op),
      loom_func_call_results(op), emitter));
  return loom_func_verify_call_purity(module, op, emitter);
}

static iree_status_t loom_func_emit_indirect_call_count_mismatch(
    const loom_op_t* op, const loom_error_def_t* error, uint16_t actual_count,
    uint16_t expected_count, iree_diagnostic_emitter_t emitter) {
  const loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("func.call.indirect")),
      loom_param_u32(actual_count),
      loom_param_u32(expected_count),
  };
  return loom_func_emit(emitter, op, error, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_func_emit_indirect_call_type_mismatch(
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
  return loom_func_emit(emitter, op, LOOM_ERR_TYPE_001, params,
                        IREE_ARRAYSIZE(params));
}

iree_status_t loom_func_call_indirect_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const loom_value_id_t target_id = loom_func_call_indirect_target(op);
  const loom_type_t target_type = loom_module_value_type(module, target_id);
  const loom_type_t target_signature =
      loom_func_ref_resolve_signature(module, target_type);
  if (loom_type_kind(target_signature) == LOOM_TYPE_NONE) {
    return loom_func_verify_ref_operand(module, op, target_id, emitter);
  }

  const loom_value_slice_t operands = loom_func_call_indirect_operands(op);
  const loom_value_slice_t results = loom_func_call_indirect_results(op);
  const uint16_t expected_argument_count =
      loom_type_func_arg_count(target_signature);
  const uint16_t expected_result_count =
      loom_type_func_result_count(target_signature);
  if (operands.count != expected_argument_count) {
    IREE_RETURN_IF_ERROR(loom_func_emit_indirect_call_count_mismatch(
        op, LOOM_ERR_STRUCTURE_001, operands.count, expected_argument_count,
        emitter));
  }
  if (results.count != expected_result_count) {
    IREE_RETURN_IF_ERROR(loom_func_emit_indirect_call_count_mismatch(
        op, LOOM_ERR_STRUCTURE_002, results.count, expected_result_count,
        emitter));
  }

  const loom_type_t* expected_argument_types =
      loom_type_func_arg_types(target_signature);
  const uint16_t argument_count = operands.count < expected_argument_count
                                      ? operands.count
                                      : expected_argument_count;
  for (uint16_t i = 0; i < argument_count; ++i) {
    const loom_type_t actual_type =
        loom_module_value_type(module, operands.values[i]);
    if (loom_type_equal(actual_type, expected_argument_types[i])) continue;
    IREE_RETURN_IF_ERROR(loom_func_emit_indirect_call_type_mismatch(
        op, LOOM_DIAGNOSTIC_FIELD_OPERAND, i, "operand", actual_type,
        "function argument", expected_argument_types[i], emitter));
  }

  const loom_type_t* expected_result_types =
      loom_type_func_result_types(target_signature);
  const uint16_t result_count = results.count < expected_result_count
                                    ? results.count
                                    : expected_result_count;
  for (uint16_t i = 0; i < result_count; ++i) {
    const loom_type_t actual_type =
        loom_module_value_type(module, results.values[i]);
    if (loom_type_equal(actual_type, expected_result_types[i])) continue;
    IREE_RETURN_IF_ERROR(loom_func_emit_indirect_call_type_mismatch(
        op, LOOM_DIAGNOSTIC_FIELD_RESULT, i, "result", actual_type,
        "function result", expected_result_types[i], emitter));
  }
  return iree_ok_status();
}
