// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"

typedef struct loom_test_callee_signature_t {
  const loom_op_t* definition_op;
  const loom_value_id_t* argument_ids;
  uint16_t argument_count;
  const loom_value_id_t* result_ids;
  uint16_t result_count;
} loom_test_callee_signature_t;

static iree_string_view_t loom_test_symbol_name(const loom_module_t* module,
                                                loom_symbol_ref_t symbol_ref) {
  if (!loom_symbol_ref_is_valid(symbol_ref) || symbol_ref.module_id != 0 ||
      symbol_ref.symbol_id >= module->symbols.count) {
    return IREE_SV("<unnamed>");
  }
  const loom_symbol_t* symbol = &module->symbols.entries[symbol_ref.symbol_id];
  if (symbol->name_id != LOOM_STRING_ID_INVALID &&
      symbol->name_id < module->strings.count) {
    return module->strings.entries[symbol->name_id];
  }
  return IREE_SV("<unnamed>");
}

static iree_string_view_t loom_test_symbol_definition_name(
    const loom_symbol_t* symbol) {
  if (!symbol || !symbol->definition) return IREE_SV("unresolved");
  return loom_symbol_definition_descriptor_name(symbol->definition);
}

static loom_symbol_ref_t loom_test_call_like_callee(const loom_op_t* op) {
  return loom_attr_as_symbol(loom_op_const_attrs(op)[0]);
}

static iree_status_t loom_test_emit_callee_diagnostic(
    iree_diagnostic_emitter_t emitter, const loom_op_t* call_op,
    const loom_op_t* definition_op, const loom_error_def_t* error,
    const loom_diagnostic_param_t* params, iree_host_size_t param_count) {
  loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("defined here"),
      .op = definition_op,
  }};
  loom_diagnostic_emission_t emission = {
      .op = call_op,
      .error = error,
      .params = params,
      .param_count = param_count,
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_test_emit_callee_kind_mismatch(
    const loom_module_t* module, const loom_op_t* call_op,
    iree_diagnostic_emitter_t emitter, loom_symbol_ref_t callee,
    const loom_symbol_t* symbol) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_test_symbol_name(module, callee)),
      loom_param_string(loom_test_symbol_definition_name(symbol)),
      loom_param_string(IREE_SV("function")),
  };
  return loom_test_emit_callee_diagnostic(emitter, call_op, symbol->defining_op,
                                          LOOM_ERR_SYMBOL_003, params,
                                          IREE_ARRAYSIZE(params));
}

static const loom_symbol_t* loom_test_lookup_callee_symbol(
    const loom_module_t* module, const loom_op_t* call_op) {
  loom_symbol_ref_t callee = loom_test_call_like_callee(call_op);
  if (!loom_symbol_ref_is_valid(callee) || callee.module_id != 0 ||
      callee.symbol_id >= module->symbols.count) {
    return NULL;
  }

  const loom_symbol_t* symbol = &module->symbols.entries[callee.symbol_id];
  if (symbol->definition == NULL || symbol->defining_op == NULL) {
    return NULL;
  }
  return symbol;
}

static bool loom_test_load_callee_signature(
    const loom_module_t* module, const loom_symbol_t* symbol,
    loom_test_callee_signature_t* out_signature) {
  loom_func_like_t callee = loom_func_like_cast(module, symbol->defining_op);
  if (!loom_func_like_isa(callee)) {
    return false;
  }

  out_signature->definition_op = symbol->defining_op;
  out_signature->argument_ids =
      loom_func_like_arg_ids(callee, &out_signature->argument_count);
  out_signature->result_ids = loom_op_const_results(callee.op);
  out_signature->result_count = callee.op->result_count;
  return true;
}

static loom_value_id_t loom_test_map_callee_value_id(
    const loom_test_callee_signature_t* signature, const loom_op_t* call_op,
    loom_value_id_t callee_value_id) {
  const loom_value_id_t* call_operands = loom_op_const_operands(call_op);
  for (uint16_t i = 0;
       i < signature->argument_count && i < call_op->operand_count; ++i) {
    if (signature->argument_ids[i] == callee_value_id) {
      return call_operands[i];
    }
  }

  const loom_value_id_t* call_results = loom_op_const_results(call_op);
  for (uint16_t i = 0; i < signature->result_count && i < call_op->result_count;
       ++i) {
    if (signature->result_ids[i] == callee_value_id) {
      return call_results[i];
    }
  }

  return callee_value_id;
}

static bool loom_test_signature_encoding_matches(
    const loom_test_callee_signature_t* signature, const loom_op_t* call_op,
    loom_type_t call_type, loom_type_t callee_type) {
  if (call_type.encoding_flags != callee_type.encoding_flags) {
    return false;
  }
  if (!iree_all_bits_set(callee_type.encoding_flags, LOOM_ENCODING_FLAG_SSA)) {
    return call_type.encoding_id == callee_type.encoding_id;
  }
  return call_type.encoding_id ==
         (uint16_t)loom_test_map_callee_value_id(signature, call_op,
                                                 callee_type.encoding_id);
}

static bool loom_test_signature_dim_matches(
    const loom_test_callee_signature_t* signature, const loom_op_t* call_op,
    uint64_t call_dim, uint64_t callee_dim) {
  if (loom_dim_is_dynamic(call_dim) != loom_dim_is_dynamic(callee_dim)) {
    return false;
  }
  if (!loom_dim_is_dynamic(callee_dim)) {
    return loom_dim_static_size(call_dim) == loom_dim_static_size(callee_dim);
  }
  return loom_dim_value_id(call_dim) ==
         loom_test_map_callee_value_id(signature, call_op,
                                       loom_dim_value_id(callee_dim));
}

static bool loom_test_signature_type_matches(
    const loom_test_callee_signature_t* signature, const loom_op_t* call_op,
    loom_type_t call_type, loom_type_t callee_type) {
  if (loom_type_kind(call_type) != loom_type_kind(callee_type) ||
      loom_type_element_type(call_type) !=
          loom_type_element_type(callee_type) ||
      loom_type_rank(call_type) != loom_type_rank(callee_type) ||
      !loom_test_signature_encoding_matches(signature, call_op, call_type,
                                            callee_type)) {
    return false;
  }

  switch (loom_type_kind(callee_type)) {
    case LOOM_TYPE_FUNCTION: {
      const loom_func_type_data_t* call_data = loom_type_func_data(call_type);
      const loom_func_type_data_t* callee_data =
          loom_type_func_data(callee_type);
      if (!call_data || !callee_data) {
        return call_data == callee_data;
      }
      if (call_data->arg_count != callee_data->arg_count ||
          call_data->result_count != callee_data->result_count) {
        return false;
      }
      uint16_t nested_type_count =
          (uint16_t)(callee_data->arg_count + callee_data->result_count);
      for (uint16_t i = 0; i < nested_type_count; ++i) {
        if (!loom_test_signature_type_matches(signature, call_op,
                                              call_data->types[i],
                                              callee_data->types[i])) {
          return false;
        }
      }
      return true;
    }
    case LOOM_TYPE_DIALECT: {
      uint16_t param_count = loom_type_dialect_param_count(callee_type);
      if (loom_type_dialect_name_id(call_type) !=
              loom_type_dialect_name_id(callee_type) ||
          loom_type_dialect_param_count(call_type) != param_count) {
        return false;
      }
      const loom_type_t* call_params = loom_type_dialect_params(call_type);
      const loom_type_t* callee_params = loom_type_dialect_params(callee_type);
      if (!call_params || !callee_params) {
        return call_params == callee_params;
      }
      for (uint16_t i = 0; i < param_count; ++i) {
        if (!loom_test_signature_type_matches(
                signature, call_op, call_params[i], callee_params[i])) {
          return false;
        }
      }
      return true;
    }
    case LOOM_TYPE_TILE:
    case LOOM_TYPE_TENSOR:
    case LOOM_TYPE_VECTOR:
    case LOOM_TYPE_VIEW:
    case LOOM_TYPE_POOL: {
      uint8_t rank = loom_type_rank(callee_type);
      for (uint8_t i = 0; i < rank; ++i) {
        if (!loom_test_signature_dim_matches(signature, call_op,
                                             loom_type_dim(call_type, i),
                                             loom_type_dim(callee_type, i))) {
          return false;
        }
      }
      return true;
    }
    case LOOM_TYPE_NONE:
    case LOOM_TYPE_SCALAR:
    case LOOM_TYPE_GROUP:
    case LOOM_TYPE_ENCODING:
    case LOOM_TYPE_BUFFER:
    case LOOM_TYPE_REGISTER:
    case LOOM_TYPE_STORAGE:
    case LOOM_TYPE_COUNT_:
      return true;
  }
  return false;
}

static iree_status_t loom_test_emit_call_like_count_mismatch(
    const loom_module_t* module, const loom_op_t* call_op,
    const loom_op_t* definition_op, iree_diagnostic_emitter_t emitter,
    const loom_error_def_t* error, uint16_t actual_count,
    uint16_t expected_count) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, call_op)),
      loom_param_u32(actual_count),
      loom_param_u32(expected_count),
  };
  return loom_test_emit_callee_diagnostic(
      emitter, call_op, definition_op, error, params, IREE_ARRAYSIZE(params));
}

static iree_status_t loom_test_emit_constant_kind_mismatch(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    loom_attr_kind_t actual_kind, loom_attr_kind_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("value")),
      loom_param_u32(actual_kind),
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

static iree_status_t loom_test_emit_constant_result_type_mismatch(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    loom_type_t result_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("result 0")),
      loom_param_type(result_type),
      loom_param_string(IREE_SV("scalar/tile/tensor/vector")),
  };
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TYPE_004,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_test_verify_call_like_argument_count(
    const loom_module_t* module, const loom_op_t* call_op,
    const loom_test_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  if (call_op->operand_count == signature->argument_count) {
    return iree_ok_status();
  }
  return loom_test_emit_call_like_count_mismatch(
      module, call_op, signature->definition_op, emitter,
      LOOM_ERR_STRUCTURE_001, call_op->operand_count,
      signature->argument_count);
}

static iree_status_t loom_test_verify_call_like_result_count(
    const loom_module_t* module, const loom_op_t* call_op,
    const loom_test_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  if (call_op->result_count == signature->result_count) {
    return iree_ok_status();
  }
  return loom_test_emit_call_like_count_mismatch(
      module, call_op, signature->definition_op, emitter,
      LOOM_ERR_STRUCTURE_002, call_op->result_count, signature->result_count);
}

static void loom_test_format_field_name(char* buffer,
                                        iree_host_size_t buffer_capacity,
                                        const char* prefix,
                                        uint16_t field_index) {
  iree_snprintf(buffer, buffer_capacity, "%s %u", prefix, field_index);
}

static iree_status_t loom_test_emit_call_like_type_mismatch(
    const loom_op_t* call_op, const loom_op_t* definition_op,
    iree_diagnostic_emitter_t emitter,
    loom_diagnostic_field_kind_t call_field_kind, uint16_t call_field_index,
    const char* call_field_prefix, loom_type_t call_type,
    const char* callee_field_prefix, loom_type_t callee_type) {
  char call_field_name[32];
  char callee_field_name[32];
  loom_test_format_field_name(call_field_name, sizeof(call_field_name),
                              call_field_prefix, call_field_index);
  loom_test_format_field_name(callee_field_name, sizeof(callee_field_name),
                              callee_field_prefix, call_field_index);
  loom_diagnostic_param_t params[] = {
      loom_param_with_field_ref(
          loom_param_string(iree_make_cstring_view(call_field_name)),
          loom_diagnostic_field_ref(call_field_kind, call_field_index)),
      loom_param_type(call_type),
      loom_param_string(iree_make_cstring_view(callee_field_name)),
      loom_param_type(callee_type),
  };
  return loom_test_emit_callee_diagnostic(emitter, call_op, definition_op,
                                          LOOM_ERR_TYPE_001, params,
                                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_test_verify_call_like_argument_types(
    const loom_module_t* module, const loom_op_t* call_op,
    const loom_test_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  uint16_t compare_count = call_op->operand_count;
  if (compare_count > signature->argument_count) {
    compare_count = signature->argument_count;
  }

  const loom_value_id_t* call_operands = loom_op_const_operands(call_op);
  for (uint16_t i = 0; i < compare_count; ++i) {
    loom_type_t call_type = loom_module_value_type(module, call_operands[i]);
    loom_type_t callee_type =
        loom_module_value_type(module, signature->argument_ids[i]);
    if (loom_test_signature_type_matches(signature, call_op, call_type,
                                         callee_type)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_test_emit_call_like_type_mismatch(
        call_op, signature->definition_op, emitter,
        LOOM_DIAGNOSTIC_FIELD_OPERAND, i, "operand", call_type,
        "callee argument", callee_type));
  }
  return iree_ok_status();
}

static iree_status_t loom_test_verify_call_like_result_types(
    const loom_module_t* module, const loom_op_t* call_op,
    const loom_test_callee_signature_t* signature,
    iree_diagnostic_emitter_t emitter) {
  uint16_t compare_count = call_op->result_count;
  if (compare_count > signature->result_count) {
    compare_count = signature->result_count;
  }

  const loom_value_id_t* call_results = loom_op_const_results(call_op);
  for (uint16_t i = 0; i < compare_count; ++i) {
    loom_type_t call_type = loom_module_value_type(module, call_results[i]);
    loom_type_t callee_type =
        loom_module_value_type(module, signature->result_ids[i]);
    if (loom_test_signature_type_matches(signature, call_op, call_type,
                                         callee_type)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_test_emit_call_like_type_mismatch(
        call_op, signature->definition_op, emitter,
        LOOM_DIAGNOSTIC_FIELD_RESULT, i, "result", call_type, "callee result",
        callee_type));
  }
  return iree_ok_status();
}

iree_status_t loom_test_call_like_verify(const loom_module_t* module,
                                         const loom_op_t* op,
                                         iree_diagnostic_emitter_t emitter) {
  const loom_symbol_t* symbol = loom_test_lookup_callee_symbol(module, op);
  if (!symbol) {
    return iree_ok_status();
  }

  loom_test_callee_signature_t signature = {0};
  if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_FUNC_LIKE)) {
    return iree_ok_status();
  }
  if (!loom_test_load_callee_signature(module, symbol, &signature)) {
    return loom_test_emit_callee_kind_mismatch(
        module, op, emitter, loom_test_call_like_callee(op), symbol);
  }

  IREE_RETURN_IF_ERROR(loom_test_verify_call_like_argument_count(
      module, op, &signature, emitter));
  IREE_RETURN_IF_ERROR(
      loom_test_verify_call_like_result_count(module, op, &signature, emitter));
  IREE_RETURN_IF_ERROR(loom_test_verify_call_like_argument_types(
      module, op, &signature, emitter));
  IREE_RETURN_IF_ERROR(
      loom_test_verify_call_like_result_types(module, op, &signature, emitter));
  return iree_ok_status();
}

iree_status_t loom_test_constant_verify(const loom_module_t* module,
                                        const loom_op_t* op,
                                        iree_diagnostic_emitter_t emitter) {
  loom_type_t result_type =
      loom_module_value_type(module, loom_test_constant_result(op));
  loom_type_kind_t result_kind = loom_type_kind(result_type);
  if (result_kind != LOOM_TYPE_SCALAR && result_kind != LOOM_TYPE_TILE &&
      result_kind != LOOM_TYPE_TENSOR && result_kind != LOOM_TYPE_VECTOR) {
    return loom_test_emit_constant_result_type_mismatch(op, emitter,
                                                        result_type);
  }

  loom_attribute_t value = loom_test_constant_value(op);
  loom_attr_kind_t expected_kind = LOOM_ATTR_ANY;
  if (loom_attr_matches_scalar_type(value, loom_type_element_type(result_type),
                                    &expected_kind)) {
    return iree_ok_status();
  }
  return loom_test_emit_constant_kind_mismatch(
      op, emitter, (loom_attr_kind_t)value.kind, expected_kind);
}
