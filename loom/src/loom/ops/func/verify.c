// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/error_catalog.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/function_contract_verify.h"

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

static iree_status_t loom_func_verify_function_result(
    const loom_module_t* module, const loom_op_t* op, loom_value_id_t result_id,
    iree_diagnostic_emitter_t emitter) {
  const loom_type_t type = loom_module_value_type(module, result_id);
  if (loom_type_is_function(type)) return iree_ok_status();
  const loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(IREE_SV("result")),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_RESULT, 0)),
      loom_param_type(type),
      loom_param_string(IREE_SV("function type")),
  };
  return loom_func_emit(emitter, op, LOOM_ERR_TYPE_004, params,
                        IREE_ARRAYSIZE(params));
}

static iree_status_t loom_func_verify_function_operand(
    const loom_module_t* module, const loom_op_t* op,
    loom_value_id_t operand_id, iree_diagnostic_emitter_t emitter) {
  const loom_type_t type = loom_module_value_type(module, operand_id);
  if (loom_type_is_function(type)) return iree_ok_status();
  const loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(IREE_SV("function")),
          loom_diagnostic_field_ref(LOOM_DIAGNOSTIC_FIELD_OPERAND, 0)),
      loom_param_type(type),
      loom_param_string(IREE_SV("function type")),
  };
  return loom_func_emit(emitter, op, LOOM_ERR_TYPE_003, params,
                        IREE_ARRAYSIZE(params));
}

static const loom_symbol_t* loom_func_lookup_defined_symbol(
    const loom_module_t* module, loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return NULL;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  return symbol->definition && symbol->defining_op ? symbol : NULL;
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
  return loom_func_verify_function_result(module, op, loom_func_null_result(op),
                                          emitter);
}

iree_status_t loom_func_compare_null_verify(const loom_module_t* module,
                                            const loom_op_t* op,
                                            iree_diagnostic_emitter_t emitter) {
  return loom_func_verify_function_operand(
      module, op, loom_func_compare_null_function(op), emitter);
}

iree_status_t loom_func_address_verify(const loom_module_t* module,
                                       const loom_op_t* op,
                                       iree_diagnostic_emitter_t emitter) {
  const loom_value_id_t result_id = loom_func_address_result(op);
  const loom_type_t result_type = loom_module_value_type(module, result_id);
  if (!loom_type_is_function(result_type)) {
    return loom_func_verify_function_result(module, op, result_id, emitter);
  }
  return loom_function_type_contract_verify(
      module, op, loom_func_address_callee(op), result_type, emitter);
}

iree_status_t loom_func_import_resolved_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter) {
  const loom_symbol_ref_t callee = loom_func_import_resolved_callee(op);
  const loom_symbol_t* symbol = loom_func_lookup_defined_symbol(module, callee);
  if (!symbol) return iree_ok_status();
  const loom_func_like_t function =
      loom_func_like_cast(module, symbol->defining_op);
  if (loom_func_like_isa(function) &&
      loom_func_like_import_policy(function) ==
          LOOM_FUNC_DECL_IMPORT_POLICY_OPTIONAL) {
    return iree_ok_status();
  }
  const loom_diagnostic_param_t params[] = {
      loom_param_string(module->strings.entries[symbol->name_id]),
      loom_param_string(
          loom_symbol_definition_descriptor_name(symbol->definition)),
      loom_param_string(IREE_SV("optional function import")),
  };
  const loom_diagnostic_related_op_t related[] = {{
      .label = IREE_SV("defined here"),
      .op = symbol->defining_op,
  }};
  const loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_SYMBOL_003,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related,
      .related_op_count = IREE_ARRAYSIZE(related),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

iree_status_t loom_func_call_verify(const loom_module_t* module,
                                    const loom_op_t* op,
                                    iree_diagnostic_emitter_t emitter) {
  return loom_function_call_contract_verify(
      module, op, loom_func_call_callee(op), loom_func_call_operands(op),
      loom_func_call_results(op), emitter);
}
