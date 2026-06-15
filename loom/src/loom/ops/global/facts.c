// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the global dialect.

#include "loom/ir/facts.h"

#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"

#define LOOM_GLOBAL_PREDICATES_ATTR_INDEX 1

static const loom_op_t* loom_global_load_definition(const loom_module_t* module,
                                                    const loom_op_t* op) {
  loom_symbol_ref_t ref = loom_global_load_global(op);
  if (!loom_symbol_ref_is_valid(ref) || ref.module_id != 0 ||
      ref.symbol_id >= module->symbols.count) {
    return NULL;
  }
  const loom_symbol_t* symbol = &module->symbols.entries[ref.symbol_id];
  if (!loom_symbol_implements(symbol, LOOM_SYMBOL_INTERFACE_GLOBAL)) {
    return NULL;
  }
  return symbol->defining_op;
}

static loom_value_id_t loom_global_definition_value(const loom_op_t* op) {
  if (loom_global_constant_isa(op)) return loom_global_constant_type(op);
  if (loom_global_variable_isa(op)) return loom_global_variable_type(op);
  return LOOM_VALUE_ID_INVALID;
}

static loom_attribute_t loom_global_definition_predicates(const loom_op_t* op) {
  if (!op || op->attribute_count <= LOOM_GLOBAL_PREDICATES_ATTR_INDEX) {
    return loom_attr_absent();
  }
  return loom_op_attrs(op)[LOOM_GLOBAL_PREDICATES_ATTR_INDEX];
}

static bool loom_global_scalar_initializer_facts(
    loom_type_t type, loom_attribute_t initializer,
    loom_value_facts_t* out_facts) {
  if (loom_attr_is_absent(initializer) || !loom_type_is_scalar(type)) {
    return false;
  }

  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  if (!loom_attr_matches_scalar_type(initializer, scalar_type, NULL)) {
    return false;
  }

  if (scalar_type == LOOM_SCALAR_TYPE_I1) {
    if (initializer.kind == LOOM_ATTR_BOOL) {
      *out_facts = loom_value_facts_exact_i64(loom_attr_as_bool(initializer));
      return true;
    }
    *out_facts = loom_value_facts_exact_i64(loom_attr_as_i64(initializer));
    return true;
  }
  if (scalar_type == LOOM_SCALAR_TYPE_INDEX ||
      scalar_type == LOOM_SCALAR_TYPE_OFFSET ||
      loom_scalar_type_is_integer(scalar_type)) {
    *out_facts = loom_value_facts_exact_i64(loom_attr_as_i64(initializer));
    return true;
  }
  if (loom_scalar_type_is_float(scalar_type)) {
    *out_facts = loom_value_facts_exact_f64(loom_attr_as_f64(initializer));
    return true;
  }
  return false;
}

static bool loom_global_predicate_facts_support_type(loom_type_t type) {
  if (!loom_type_is_scalar(type)) return false;
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET ||
         loom_scalar_type_is_integer(scalar_type);
}

static bool loom_global_map_definition_type_value(
    loom_type_t definition_type, loom_type_t load_type,
    loom_value_id_t definition_value, loom_value_id_t* out_load_value) {
  *out_load_value = LOOM_VALUE_ID_INVALID;
  if (!loom_type_is_shaped(definition_type) ||
      !loom_type_is_shaped(load_type)) {
    return false;
  }
  if (loom_type_rank(definition_type) != loom_type_rank(load_type)) {
    return false;
  }

  bool found = false;
  uint8_t rank = loom_type_rank(definition_type);
  for (uint8_t i = 0; i < rank; ++i) {
    if (!loom_type_dim_is_dynamic_at(definition_type, i)) continue;
    if (loom_type_dim_value_id_at(definition_type, i) != definition_value) {
      continue;
    }
    if (!loom_type_dim_is_dynamic_at(load_type, i)) return false;

    loom_value_id_t load_value = loom_type_dim_value_id_at(load_type, i);
    if (found && load_value != *out_load_value) return false;
    *out_load_value = load_value;
    found = true;
  }

  if (loom_type_has_ssa_encoding(definition_type) &&
      loom_type_encoding_value_id(definition_type) == definition_value) {
    if (!loom_type_has_ssa_encoding(load_type)) return false;
    loom_value_id_t load_value = loom_type_encoding_value_id(load_type);
    if (found && load_value != *out_load_value) return false;
    *out_load_value = load_value;
    found = true;
  }

  return found;
}

static bool loom_global_load_result_index(loom_value_slice_t results,
                                          loom_value_id_t value_id,
                                          iree_host_size_t* out_index) {
  for (iree_host_size_t i = 0; i < results.count; ++i) {
    if (results.values[i] != value_id) continue;
    *out_index = i;
    return true;
  }
  return false;
}

static void loom_global_load_apply_definition_predicates(
    const loom_module_t* module, const loom_op_t* op,
    const loom_op_t* definition_op, loom_value_facts_t* result_facts) {
  loom_attribute_t predicates =
      loom_global_definition_predicates(definition_op);
  if (predicates.kind != LOOM_ATTR_PREDICATE_LIST || predicates.count == 0 ||
      !predicates.predicate_list) {
    return;
  }

  loom_value_id_t definition_value =
      loom_global_definition_value(definition_op);
  if (definition_value == LOOM_VALUE_ID_INVALID ||
      definition_value >= module->values.count) {
    return;
  }

  loom_value_slice_t results = loom_global_load_result(op);
  if (results.count == 0 || results.values[0] >= module->values.count) return;

  loom_type_t definition_type =
      loom_module_value_type(module, definition_value);
  loom_type_t load_type = loom_module_value_type(module, results.values[0]);
  for (uint16_t i = 0; i < predicates.count; ++i) {
    const loom_predicate_t* predicate = &predicates.predicate_list[i];
    if (predicate->arg_count == 0 ||
        predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE ||
        predicate->args[0] < 0) {
      continue;
    }

    loom_value_id_t load_value = LOOM_VALUE_ID_INVALID;
    if (!loom_global_map_definition_type_value(
            definition_type, load_type, (loom_value_id_t)predicate->args[0],
            &load_value)) {
      continue;
    }

    iree_host_size_t result_index = 0;
    if (!loom_global_load_result_index(results, load_value, &result_index)) {
      continue;
    }
    if (results.values[result_index] >= module->values.count) continue;
    loom_type_t result_type =
        loom_module_value_type(module, results.values[result_index]);
    if (!loom_global_predicate_facts_support_type(result_type)) continue;
    loom_value_facts_apply_predicate(&result_facts[result_index], predicate);
  }
}

iree_status_t loom_global_load_facts(loom_fact_context_t* context,
                                     const loom_module_t* module,
                                     const loom_op_t* op,
                                     const loom_value_facts_t* operand_facts,
                                     loom_value_facts_t* result_facts) {
  (void)context;
  (void)operand_facts;

  loom_value_slice_t results = loom_global_load_result(op);
  for (iree_host_size_t i = 0; i < results.count; ++i) {
    result_facts[i] = loom_value_facts_unknown();
  }

  const loom_op_t* definition_op = loom_global_load_definition(module, op);
  if (!definition_op) return iree_ok_status();

  loom_global_load_apply_definition_predicates(module, op, definition_op,
                                               result_facts);
  if (!loom_global_constant_isa(definition_op) || results.count == 0 ||
      results.values[0] >= module->values.count) {
    return iree_ok_status();
  }

  loom_value_facts_t initializer_facts = loom_value_facts_unknown();
  if (loom_global_scalar_initializer_facts(
          loom_module_value_type(module, results.values[0]),
          loom_global_constant_initializer(definition_op),
          &initializer_facts)) {
    result_facts[0] = initializer_facts;
  }
  return iree_ok_status();
}
