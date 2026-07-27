// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/facts.h"

#include "loom/ir/module.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/ops/view/reference.h"

static bool loom_sanitizer_type_accepts_integer_predicates(loom_type_t type) {
  if (!loom_type_is_scalar(type)) return false;
  loom_scalar_type_t scalar_type = loom_type_element_type(type);
  return loom_scalar_type_is_integer(scalar_type) ||
         scalar_type == LOOM_SCALAR_TYPE_INDEX ||
         scalar_type == LOOM_SCALAR_TYPE_OFFSET;
}

static bool loom_sanitizer_type_accepts_float_predicates(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_scalar_type_is_float(loom_type_element_type(type));
}

static bool loom_sanitizer_type_accepts_predicate(loom_type_t type,
                                                  uint8_t predicate_kind) {
  switch ((loom_predicate_kind_t)predicate_kind) {
    case LOOM_PREDICATE_EQ:
    case LOOM_PREDICATE_NE:
    case LOOM_PREDICATE_LT:
    case LOOM_PREDICATE_LE:
    case LOOM_PREDICATE_GT:
    case LOOM_PREDICATE_GE:
    case LOOM_PREDICATE_MUL:
    case LOOM_PREDICATE_MIN:
    case LOOM_PREDICATE_MAX:
    case LOOM_PREDICATE_POW2:
    case LOOM_PREDICATE_RANGE:
      return loom_sanitizer_type_accepts_integer_predicates(type);
    case LOOM_PREDICATE_NOT_NAN:
    case LOOM_PREDICATE_NOT_INF:
    case LOOM_PREDICATE_FINITE:
      return loom_sanitizer_type_accepts_float_predicates(type);
    case LOOM_PREDICATE_COUNT_:
      return false;
  }
  return false;
}

static bool loom_sanitizer_find_value(loom_value_slice_t values,
                                      loom_value_id_t value_id,
                                      uint16_t* out_ordinal) {
  for (uint16_t i = 0; i < values.count; ++i) {
    if (values.values[i] != value_id) continue;
    *out_ordinal = i;
    return true;
  }
  return false;
}

static bool loom_sanitizer_resolve_predicate_constants(
    const loom_predicate_t* source, loom_value_slice_t values,
    const loom_value_facts_t* operand_facts, loom_predicate_t* out_predicate) {
  *out_predicate = *source;
  for (uint8_t i = 1; i < source->arg_count; ++i) {
    if (source->arg_tags[i] != LOOM_PRED_ARG_VALUE) continue;
    if (source->args[i] < 0) return false;
    uint16_t operand_ordinal = 0;
    if (!loom_sanitizer_find_value(values, (loom_value_id_t)source->args[i],
                                   &operand_ordinal)) {
      return false;
    }
    int64_t exact_value = 0;
    if (!loom_value_facts_as_exact_i64(operand_facts[operand_ordinal],
                                       &exact_value)) {
      continue;
    }
    out_predicate->arg_tags[i] = LOOM_PRED_ARG_CONST;
    out_predicate->args[i] = exact_value;
  }
  return true;
}

iree_status_t loom_sanitizer_assert_value_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_value_slice_t values = loom_sanitizer_assert_value_values(op);
  loom_value_slice_t results = loom_sanitizer_assert_value_results(op);
  uint16_t fact_count =
      values.count < results.count ? values.count : results.count;
  for (uint16_t i = 0; i < fact_count; ++i) {
    result_facts[i] = operand_facts[i];
  }
  for (uint16_t i = fact_count; i < results.count; ++i) {
    result_facts[i] = loom_value_facts_unknown();
  }

  loom_attribute_t predicates = loom_sanitizer_assert_value_predicates(op);
  for (uint16_t i = 0; i < predicates.count; ++i) {
    const loom_predicate_t* predicate = &predicates.predicate_list[i];
    if (predicate->arg_count == 0 ||
        predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE ||
        predicate->args[0] < 0) {
      continue;
    }
    uint16_t target_ordinal = 0;
    if (!loom_sanitizer_find_value(values, (loom_value_id_t)predicate->args[0],
                                   &target_ordinal)) {
      continue;
    }
    if (target_ordinal >= fact_count) continue;
    loom_type_t result_type =
        loom_module_value_type(module, results.values[target_ordinal]);
    if (!loom_sanitizer_type_accepts_predicate(result_type, predicate->kind)) {
      continue;
    }

    loom_predicate_t resolved_predicate = {0};
    if (!loom_sanitizer_resolve_predicate_constants(
            predicate, values, operand_facts, &resolved_predicate)) {
      continue;
    }
    loom_value_facts_apply_predicate(&result_facts[target_ordinal],
                                     &resolved_predicate);
  }
  return iree_ok_status();
}

iree_status_t loom_sanitizer_assert_layout_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  loom_type_t source_type =
      loom_module_value_type(module, loom_sanitizer_assert_layout_view(op));
  loom_type_t result_type =
      loom_module_value_type(module, loom_sanitizer_assert_layout_result(op));
  return loom_view_reference_make_refine(
      context, module, loom_sanitizer_assert_layout_view(op), operand_facts[0],
      source_type, result_type, &result_facts[0]);
}
