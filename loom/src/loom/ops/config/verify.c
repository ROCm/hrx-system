// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbol_value_constraints.h"
#include "loom/error/emitter.h"
#include "loom/error/error_catalog.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/config/ops.h"
#include "loom/ops/encoding/roles.h"

static loom_type_t loom_config_symbol_type(const loom_module_t* module,
                                           const loom_symbol_t* symbol) {
  loom_type_t none = {0};
  if (!symbol || !symbol->defining_op) {
    return none;
  }
  if (loom_config_decl_isa(symbol->defining_op)) {
    return loom_module_value_type(module,
                                  loom_config_decl_type(symbol->defining_op));
  }
  if (loom_config_def_isa(symbol->defining_op)) {
    return loom_module_value_type(module,
                                  loom_config_def_type(symbol->defining_op));
  }
  return none;
}

static bool loom_config_type_is_supported(loom_type_t type) {
  return loom_type_is_scalar(type) || loom_type_is_encoding(type);
}

static iree_status_t loom_config_emit(iree_diagnostic_emitter_t emitter,
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

static iree_status_t loom_config_emit_unsupported_type(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    loom_type_t config_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("config value")),
      loom_param_type(config_type),
      loom_param_string(IREE_SV("scalar or encoding")),
  };
  return loom_config_emit(emitter, op, LOOM_ERR_TYPE_004, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_config_emit_value_kind_mismatch(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    loom_attr_kind_t actual_kind, loom_attr_kind_t expected_kind) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("value")),
      loom_param_u32(actual_kind),
      loom_param_u32(expected_kind),
  };
  return loom_config_emit(emitter, op, LOOM_ERR_TYPE_005, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_config_emit_type_mismatch(
    const loom_op_t* op, const loom_op_t* definition_op,
    iree_diagnostic_emitter_t emitter, loom_type_t result_type,
    loom_type_t config_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("result")),
      loom_param_type(result_type),
      loom_param_string(IREE_SV("referenced config")),
      loom_param_type(config_type),
  };
  loom_diagnostic_related_op_t related_ops[] = {{
      .label = IREE_SV("defined here"),
      .op = definition_op,
  }};
  loom_diagnostic_emission_t emission = {
      .op = op,
      .error = LOOM_ERR_TYPE_001,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
      .related_ops = related_ops,
      .related_op_count = IREE_ARRAYSIZE(related_ops),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

static iree_status_t loom_config_emit_value_type_mismatch(
    const loom_op_t* op, iree_diagnostic_emitter_t emitter,
    loom_type_t value_type, loom_type_t config_type) {
  loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("value")),
      loom_param_type(value_type),
      loom_param_string(IREE_SV("config")),
      loom_param_type(config_type),
  };
  return loom_config_emit(emitter, op, LOOM_ERR_TYPE_001, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_config_emit_predicate_origin(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, uint16_t predicate_index,
    uint8_t argument_index) {
  char field_name[40];
  iree_snprintf(field_name, sizeof(field_name), "predicates[%u].arg[%u]",
                predicate_index, argument_index);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_op_name(module, op)),
      loom_param_string(iree_make_cstring_view(field_name)),
      loom_param_string(IREE_SV("the declared config value")),
  };
  return loom_config_emit(emitter, op, LOOM_ERR_STRUCTURE_032, params,
                          IREE_ARRAYSIZE(params));
}

static iree_status_t loom_config_verify_predicates(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter, loom_attribute_t predicates,
    loom_value_id_t config_value) {
  for (uint16_t predicate_index = 0; predicate_index < predicates.count;
       ++predicate_index) {
    const loom_predicate_t* predicate =
        &predicates.predicate_list[predicate_index];
    if (predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE) {
      return loom_config_emit_predicate_origin(module, op, emitter,
                                               predicate_index, 0);
    }
    for (uint8_t argument_index = 0; argument_index < predicate->arg_count;
         ++argument_index) {
      if (predicate->arg_tags[argument_index] != LOOM_PRED_ARG_VALUE) {
        continue;
      }
      if (predicate->args[argument_index] < 0 ||
          predicate->args[argument_index] > UINT32_MAX ||
          (loom_value_id_t)predicate->args[argument_index] != config_value) {
        return loom_config_emit_predicate_origin(
            module, op, emitter, predicate_index, argument_index);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_config_verify_type(const loom_module_t* module,
                                             const loom_op_t* op,
                                             iree_diagnostic_emitter_t emitter,
                                             loom_value_id_t config_value_id) {
  loom_type_t config_type = loom_module_value_type(module, config_value_id);
  if (loom_config_type_is_supported(config_type)) {
    return iree_ok_status();
  }
  return loom_config_emit_unsupported_type(op, emitter, config_type);
}

static iree_status_t loom_config_verify_value(const loom_module_t* module,
                                              const loom_op_t* op,
                                              iree_diagnostic_emitter_t emitter,
                                              loom_value_id_t config_value_id,
                                              loom_attribute_t value) {
  loom_type_t config_type = loom_module_value_type(module, config_value_id);
  IREE_RETURN_IF_ERROR(
      loom_config_verify_type(module, op, emitter, config_value_id));

  if (loom_type_is_scalar(config_type)) {
    loom_attr_kind_t expected_kind = LOOM_ATTR_ANY;
    if (!loom_attr_matches_scalar_type(
            value, loom_type_element_type(config_type), &expected_kind)) {
      return loom_config_emit_value_kind_mismatch(
          op, emitter, (loom_attr_kind_t)value.kind, expected_kind);
    }
    return iree_ok_status();
  }

  if (loom_type_is_encoding(config_type)) {
    if (value.kind != LOOM_ATTR_ENCODING) {
      return loom_config_emit_value_kind_mismatch(
          op, emitter, (loom_attr_kind_t)value.kind, LOOM_ATTR_ENCODING);
    }
    const loom_encoding_t* encoding =
        loom_module_encoding(module, loom_attr_as_encoding_id(value));
    if (!encoding) {
      return iree_ok_status();
    }
    const loom_encoding_role_t config_role =
        loom_type_encoding_role(config_type);
    const loom_encoding_role_t value_role =
        loom_encoding_static_role(module, encoding);
    if (config_role != LOOM_ENCODING_ROLE_UNKNOWN &&
        value_role != LOOM_ENCODING_ROLE_UNKNOWN && config_role != value_role) {
      return loom_config_emit_value_type_mismatch(
          op, emitter, loom_type_encoding_with_role(value_role), config_type);
    }
  }
  return iree_ok_status();
}

iree_status_t loom_config_decl_verify(const loom_module_t* module,
                                      const loom_op_t* op,
                                      iree_diagnostic_emitter_t emitter) {
  IREE_RETURN_IF_ERROR(
      loom_config_verify_type(module, op, emitter, loom_config_decl_type(op)));
  return loom_config_verify_predicates(module, op, emitter,
                                       loom_config_decl_predicates(op),
                                       loom_config_decl_type(op));
}

iree_status_t loom_config_def_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  const loom_value_id_t config_value = loom_config_def_type(op);
  const loom_attribute_t value = loom_config_def_value(op);
  const loom_attribute_t predicates = loom_config_def_predicates(op);
  IREE_RETURN_IF_ERROR(
      loom_config_verify_value(module, op, emitter, config_value, value));
  IREE_RETURN_IF_ERROR(loom_config_verify_predicates(module, op, emitter,
                                                     predicates, config_value));
  const loom_symbol_ref_t symbol = loom_config_def_symbol(op);
  const iree_string_view_t name = loom_string_table_get(
      &module->strings, module->symbols.entries[symbol.symbol_id].name_id);
  iree_status_t status = loom_symbol_value_constraints_check_exact(
      name, loom_module_value_type(module, config_value), config_value, value,
      predicates);
  if (iree_status_is_ok(status)) {
    return status;
  }
  // Translate the value-contract diagnostic at the owning source operation.
  const loom_diagnostic_param_t params[] = {
      loom_param_string(IREE_SV("value")),
      loom_param_i64(value.kind == LOOM_ATTR_BOOL ? loom_attr_as_bool(value)
                                                  : loom_attr_as_i64(value)),
      loom_param_string(iree_status_message(status)),
  };
  iree_status_t diagnostic_status = loom_config_emit(
      emitter, op, LOOM_ERR_STRUCTURE_014, params, IREE_ARRAYSIZE(params));
  iree_status_free(status);
  return diagnostic_status;
}

iree_status_t loom_config_get_verify(const loom_module_t* module,
                                     const loom_op_t* op,
                                     iree_diagnostic_emitter_t emitter) {
  loom_symbol_ref_t ref = loom_config_get_config(op);
  if (!loom_symbol_ref_is_valid(ref) ||
      ref.symbol_id >= module->symbols.count) {
    return iree_ok_status();
  }

  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (symbol->definition == NULL || symbol->defining_op == NULL) {
    return iree_ok_status();
  }
  if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_CONFIG)) {
    return iree_ok_status();
  }

  loom_type_t config_type = loom_config_symbol_type(module, symbol);
  loom_type_t result_type =
      loom_module_value_type(module, loom_config_get_result(op));
  if (!loom_type_equal(result_type, config_type)) {
    IREE_RETURN_IF_ERROR(loom_config_emit_type_mismatch(
        op, symbol->defining_op, emitter, result_type, config_type));
  }
  return iree_ok_status();
}
