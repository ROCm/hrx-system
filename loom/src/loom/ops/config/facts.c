// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the config dialect.

#include "loom/ir/facts.h"

#include "loom/ir/module.h"
#include "loom/ops/config/ops.h"

#define LOOM_CONFIG_DECL_PREDICATES_ATTR_INDEX 1

static const loom_op_t* loom_config_get_definition(const loom_module_t* module,
                                                   const loom_op_t* op) {
  loom_symbol_ref_t ref = loom_config_get_config(op);
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return NULL;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_CONFIG)) {
    return NULL;
  }
  return symbol->defining_op;
}

static bool loom_config_scalar_value_facts(loom_type_t type,
                                           loom_attribute_t value,
                                           loom_value_facts_t* out_facts) {
  if (!loom_type_is_scalar(type)) {
    return false;
  }

  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (!loom_attr_matches_scalar_type(value, scalar_type, NULL)) {
    return false;
  }

  if (scalar_type == LOOM_SCALAR_TYPE_I1) {
    if (value.kind == LOOM_ATTR_BOOL) {
      *out_facts = loom_value_facts_exact_i64(loom_attr_as_bool(value));
      return true;
    }
    *out_facts = loom_value_facts_exact_i64(loom_attr_as_i64(value));
    return true;
  }
  if (scalar_type == LOOM_SCALAR_TYPE_INDEX ||
      scalar_type == LOOM_SCALAR_TYPE_OFFSET ||
      loom_scalar_type_is_integer(scalar_type)) {
    *out_facts = loom_value_facts_exact_i64(loom_attr_as_i64(value));
    return true;
  }
  if (loom_scalar_type_is_float(scalar_type)) {
    *out_facts = loom_value_facts_exact_f64(loom_attr_as_f64(value));
    return true;
  }
  return false;
}

static bool loom_config_predicate_facts_support_type(loom_type_t type) {
  if (!loom_type_is_scalar(type)) return false;
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET ||
         loom_scalar_type_is_integer(scalar_type);
}

static loom_attribute_t loom_config_decl_predicate_list(const loom_op_t* op) {
  if (!op || op->attribute_count <= LOOM_CONFIG_DECL_PREDICATES_ATTR_INDEX) {
    return loom_attr_absent();
  }
  return loom_op_attrs(op)[LOOM_CONFIG_DECL_PREDICATES_ATTR_INDEX];
}

static void loom_config_get_apply_decl_predicates(
    const loom_module_t* module, const loom_op_t* op,
    const loom_op_t* definition_op, loom_value_facts_t* result_facts) {
  loom_attribute_t predicates = loom_config_decl_predicate_list(definition_op);
  if (predicates.kind != LOOM_ATTR_PREDICATE_LIST || predicates.count == 0 ||
      !predicates.predicate_list) {
    return;
  }

  loom_value_id_t definition_value = loom_config_decl_type(definition_op);
  loom_value_id_t result_value = loom_config_get_result(op);
  if (definition_value >= module->values.count ||
      result_value >= module->values.count) {
    return;
  }

  loom_type_t result_type = loom_module_value_type(module, result_value);
  if (!loom_config_predicate_facts_support_type(result_type)) {
    return;
  }

  for (uint16_t i = 0; i < predicates.count; ++i) {
    const loom_predicate_t* predicate = &predicates.predicate_list[i];
    if (predicate->arg_count == 0 ||
        predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE ||
        predicate->args[0] < 0 ||
        (loom_value_id_t)predicate->args[0] != definition_value) {
      continue;
    }
    loom_value_facts_apply_predicate(result_facts, predicate);
  }
}

iree_status_t loom_config_def_facts(loom_fact_context_t* context,
                                    const loom_module_t* module,
                                    const loom_op_t* op,
                                    const loom_value_facts_t* operand_facts,
                                    loom_value_facts_t* result_facts) {
  (void)context;
  (void)operand_facts;

  loom_value_id_t result_value = loom_config_def_type(op);
  result_facts[0] = loom_value_facts_unknown();
  if (result_value >= module->values.count) {
    return iree_ok_status();
  }

  loom_value_facts_t value_facts = loom_value_facts_unknown();
  if (loom_config_scalar_value_facts(
          loom_module_value_type(module, result_value),
          loom_config_def_value(op), &value_facts)) {
    result_facts[0] = value_facts;
  }
  return iree_ok_status();
}

iree_status_t loom_config_get_facts(loom_fact_context_t* context,
                                    const loom_module_t* module,
                                    const loom_op_t* op,
                                    const loom_value_facts_t* operand_facts,
                                    loom_value_facts_t* result_facts) {
  (void)context;
  (void)operand_facts;

  result_facts[0] = loom_value_facts_unknown();
  const loom_op_t* definition_op = loom_config_get_definition(module, op);
  if (!definition_op) return iree_ok_status();

  if (loom_config_def_isa(definition_op)) {
    loom_value_id_t result_value = loom_config_get_result(op);
    if (result_value >= module->values.count) {
      return iree_ok_status();
    }
    loom_value_facts_t value_facts = loom_value_facts_unknown();
    if (loom_config_scalar_value_facts(
            loom_module_value_type(module, result_value),
            loom_config_def_value(definition_op), &value_facts)) {
      result_facts[0] = value_facts;
    }
    return iree_ok_status();
  }

  if (loom_config_decl_isa(definition_op)) {
    loom_config_get_apply_decl_predicates(module, op, definition_op,
                                          &result_facts[0]);
  }
  return iree_ok_status();
}
