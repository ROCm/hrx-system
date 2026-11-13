// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbol_value_constraints.h"

#include <inttypes.h>

#include "loom/decision/predicate.h"
#include "loom/error/error_catalog.h"
#include "loom/error/parameter.h"
#include "loom/ir/attribute.h"
#include "loom/ir/float_facts.h"

static loom_value_facts_t loom_symbol_value_exact_facts(
    loom_type_t type, loom_attribute_t value) {
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (loom_scalar_type_is_float(scalar_type)) {
    return loom_value_facts_exact_float(scalar_type, loom_attr_as_f64(value));
  }
  return loom_value_facts_exact_i64(value.kind == LOOM_ATTR_BOOL
                                        ? loom_attr_as_bool(value)
                                        : loom_attr_as_i64(value));
}

static const loom_predicate_t* loom_symbol_value_find_violation(
    loom_type_t type, loom_value_id_t contract_value,
    loom_attribute_t exact_value, loom_attribute_t predicates) {
  if (predicates.count == 0) {
    return NULL;
  }
  const loom_value_facts_t facts =
      loom_symbol_value_exact_facts(type, exact_value);
  for (uint16_t i = 0; i < predicates.count; ++i) {
    const loom_predicate_t* predicate = &predicates.predicate_list[i];
    if (predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE ||
        (loom_value_id_t)predicate->args[0] != contract_value) {
      continue;
    }
    loom_decision_predicate_operand_t operands[3] = {0};
    for (uint8_t j = 0; j < predicate->arg_count; ++j) {
      if (predicate->arg_tags[j] == LOOM_PRED_ARG_CONST) {
        operands[j].facts = loom_value_facts_exact_i64(predicate->args[j]);
        operands[j].identity = LOOM_DECISION_OPERAND_IDENTITY_NONE;
      } else {
        const loom_value_id_t value = (loom_value_id_t)predicate->args[j];
        operands[j].facts =
            value == contract_value ? facts : loom_value_facts_unknown();
        operands[j].identity = value;
      }
    }
    if (loom_decision_predicate_evaluate(predicate->kind, operands) !=
        LOOM_DECISION_TRUTH_TRUE) {
      return predicate;
    }
  }
  return NULL;
}

static iree_string_view_t loom_symbol_value_format_exact(
    loom_type_t type, loom_attribute_t value, char* buffer,
    iree_host_size_t capacity) {
  if (loom_scalar_type_is_float(loom_type_element_type(type))) {
    double rounded_value = 0.0;
    loom_value_facts_as_exact_float(loom_type_element_type(type),
                                    loom_symbol_value_exact_facts(type, value),
                                    &rounded_value);
    iree_snprintf(buffer, capacity, "%.17g", rounded_value);
  } else if (value.kind == LOOM_ATTR_BOOL) {
    return loom_attr_as_bool(value) ? IREE_SV("true") : IREE_SV("false");
  } else {
    iree_snprintf(buffer, capacity, "%" PRId64, loom_attr_as_i64(value));
  }
  return iree_make_cstring_view(buffer);
}

iree_status_t loom_symbol_value_constraints_verify_exact(
    const loom_module_t* module, const loom_op_t* op,
    iree_string_view_t symbol_name, loom_type_t type,
    loom_value_id_t contract_value, loom_attribute_t exact_value,
    loom_attribute_t predicates, iree_diagnostic_emitter_t emitter,
    bool* out_accepted) {
  *out_accepted = true;
  const loom_predicate_t* violation = loom_symbol_value_find_violation(
      type, contract_value, exact_value, predicates);
  if (violation == NULL) {
    return iree_ok_status();
  }
  *out_accepted = false;
  char value_buffer[64];
  const loom_diagnostic_param_t params[] = {
      loom_param_string(symbol_name),
      loom_param_string(loom_symbol_value_format_exact(
          type, exact_value, value_buffer, sizeof(value_buffer))),
      loom_param_type(type),
      loom_param_string(
          iree_make_cstring_view(loom_predicate_kind_name(violation->kind))),
  };
  const loom_diagnostic_emission_t emission = {
      .module = module,
      .op = op,
      .error = LOOM_ERR_SYMBOL_008,
      .params = params,
      .param_count = IREE_ARRAYSIZE(params),
  };
  return iree_diagnostic_emit(emitter, &emission);
}

iree_status_t loom_symbol_value_constraints_check_exact(
    iree_string_view_t symbol_name, loom_type_t type,
    loom_value_id_t contract_value, loom_attribute_t exact_value,
    loom_attribute_t predicates) {
  if (predicates.kind == LOOM_ATTR_ABSENT || predicates.count == 0) {
    return iree_ok_status();
  }
  if (predicates.kind != LOOM_ATTR_PREDICATE_LIST ||
      !predicates.predicate_list) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol '@%.*s' has invalid value constraints",
                            (int)symbol_name.size, symbol_name.data);
  }

  if (!loom_type_is_scalar(type)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "symbol '@%.*s' has value constraints on a non-scalar type",
        (int)symbol_name.size, symbol_name.data);
  }
  const loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (!loom_attr_matches_scalar_type(exact_value, scalar_type, NULL)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "symbol '@%.*s' exact value does not match its "
                            "scalar type",
                            (int)symbol_name.size, symbol_name.data);
  }

  const loom_predicate_t* violation = loom_symbol_value_find_violation(
      type, contract_value, exact_value, predicates);
  if (violation == NULL) {
    return iree_ok_status();
  }
  char value_buffer[64];
  const iree_string_view_t value = loom_symbol_value_format_exact(
      type, exact_value, value_buffer, sizeof(value_buffer));
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "symbol '@%.*s' value %.*s violates constraint '%s'",
                          (int)symbol_name.size, symbol_name.data,
                          (int)value.size, value.data,
                          loom_predicate_kind_name(violation->kind));
}
