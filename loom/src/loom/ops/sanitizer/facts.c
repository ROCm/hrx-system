// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/facts.h"

#include "loom/ir/module.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/ops/view/reference.h"
#include "loom/util/predicate_facts.h"

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
  return loom_value_fact_table_apply_alias_predicates(
      context->table, values.values, fact_count, predicates.predicate_list,
      predicates.count, result_facts);
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
