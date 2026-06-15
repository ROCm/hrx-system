// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fact implementations for the scf dialect.

#include "loom/ir/facts.h"

#include "loom/ir/attribute.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/scf/ops.h"
#include "loom/util/fact_table.h"
#include "loom/util/math.h"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

static bool loom_scf_lookup_key_matches_selector_facts(
    loom_value_facts_t selector_facts, int64_t key) {
  if (loom_value_facts_is_float(selector_facts)) return true;
  if (key < selector_facts.range_lo || key > selector_facts.range_hi) {
    return false;
  }
  int64_t divisor = selector_facts.known_divisor;
  return divisor <= 1 || key % divisor == 0;
}

static bool loom_scf_lookup_key_is_explicit(loom_attribute_t case_keys,
                                            int64_t key) {
  for (uint16_t i = 0; i < case_keys.count; ++i) {
    if (case_keys.i64_array[i] == key) return true;
    if (case_keys.i64_array[i] > key) return false;
  }
  return false;
}

static bool loom_scf_fact_type_is_i1(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1;
}

static void loom_scf_mark_lane_distribution_for_result(
    const loom_module_t* module, const loom_op_t* op, uint16_t result_index,
    loom_value_facts_t* facts) {
  const loom_value_id_t result_id = loom_op_const_results(op)[result_index];
  const loom_type_t result_type = loom_module_value_type(module, result_id);
  if (loom_scf_fact_type_is_i1(result_type)) {
    loom_value_facts_mark_lane_predicate(facts);
  } else {
    loom_value_facts_mark_lane_varying(facts);
  }
}

static bool loom_scf_select_arms_are_proven_equal(
    const loom_op_t* op, const loom_value_facts_t* operand_facts) {
  if (loom_scf_select_true_value(op) == loom_scf_select_false_value(op)) {
    return true;
  }
  return loom_value_facts_is_exact(operand_facts[1]) &&
         loom_value_facts_equal(operand_facts[1], operand_facts[2]);
}

static bool loom_scf_lookup_default_row_may_match(
    loom_value_facts_t selector_facts, loom_attribute_t case_keys) {
  if (loom_value_facts_is_float(selector_facts)) return true;
  if (loom_value_facts_is_exact(selector_facts)) {
    return !loom_scf_lookup_key_is_explicit(case_keys, selector_facts.range_lo);
  }

  int64_t span = 0;
  if (!loom_checked_sub_i64(selector_facts.range_hi, selector_facts.range_lo,
                            &span) ||
      span < 0 || span > 4096) {
    return true;
  }

  for (int64_t offset = 0; offset <= span; ++offset) {
    int64_t candidate = selector_facts.range_lo + offset;
    if (!loom_scf_lookup_key_matches_selector_facts(selector_facts,
                                                    candidate)) {
      continue;
    }
    if (!loom_scf_lookup_key_is_explicit(case_keys, candidate)) return true;
  }
  return false;
}

static iree_status_t loom_scf_meet_result_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, uint16_t result_index, loom_value_facts_t lhs,
    loom_value_facts_t rhs, loom_value_facts_t* out_facts) {
  const loom_value_id_t* results = loom_op_const_results(op);
  loom_value_id_t result_id = results[result_index];
  return loom_value_fact_table_meet_for_type(
      context->table, module, loom_module_value_type(module, result_id),
      context->table, lhs, context->table, rhs, out_facts);
}

static iree_status_t loom_scf_lookup_meet_row_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, uint16_t result_count, iree_host_size_t row_index,
    const loom_value_facts_t* operand_facts, bool initialized_results,
    loom_value_facts_t* result_facts) {
  iree_host_size_t row_offset = 1 + row_index * result_count;
  for (uint16_t i = 0; i < result_count; ++i) {
    const loom_value_facts_t* candidate = &operand_facts[row_offset + i];
    if (!initialized_results) {
      result_facts[i] = *candidate;
    } else {
      IREE_RETURN_IF_ERROR(
          loom_scf_meet_result_facts(context, module, op, i, result_facts[i],
                                     *candidate, &result_facts[i]));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_scf_select_facts(loom_fact_context_t* context,
                                    const loom_module_t* module,
                                    const loom_op_t* op,
                                    const loom_value_facts_t* operand_facts,
                                    loom_value_facts_t* result_facts) {
  if (loom_value_facts_is_exact(operand_facts[0])) {
    result_facts[0] =
        operand_facts[0].range_lo ? operand_facts[1] : operand_facts[2];
    return iree_ok_status();
  }
  if (loom_scf_select_arms_are_proven_equal(op, operand_facts)) {
    result_facts[0] = operand_facts[1];
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_scf_meet_result_facts(context, module, op, 0, operand_facts[1],
                                 operand_facts[2], &result_facts[0]));
  if (loom_value_facts_is_lane_varying(operand_facts[0]) ||
      loom_value_facts_is_lane_predicate(operand_facts[0])) {
    loom_scf_mark_lane_distribution_for_result(module, op, 0, &result_facts[0]);
  }
  return iree_ok_status();
}

iree_status_t loom_scf_lookup_facts(loom_fact_context_t* context,
                                    const loom_module_t* module,
                                    const loom_op_t* op,
                                    const loom_value_facts_t* operand_facts,
                                    loom_value_facts_t* result_facts) {
  loom_attribute_t case_keys = loom_scf_lookup_case_keys(op);
  loom_value_slice_t values = loom_scf_lookup_values(op);
  if (op->result_count == 0 || case_keys.kind != LOOM_ATTR_I64_ARRAY ||
      (case_keys.count > 0 && !case_keys.i64_array)) {
    for (uint16_t i = 0; i < op->result_count; ++i) {
      result_facts[i] = loom_value_facts_unknown();
    }
    return iree_ok_status();
  }

  iree_host_size_t row_count = (iree_host_size_t)case_keys.count + 1;
  iree_host_size_t expected_value_count =
      row_count * (iree_host_size_t)op->result_count;
  if (values.count != expected_value_count) {
    for (uint16_t i = 0; i < op->result_count; ++i) {
      result_facts[i] = loom_value_facts_unknown();
    }
    return iree_ok_status();
  }

  loom_value_facts_t selector_facts = operand_facts[0];
  if (!loom_value_facts_is_float(selector_facts) &&
      loom_value_facts_is_exact(selector_facts)) {
    iree_host_size_t row_index = case_keys.count;
    for (uint16_t i = 0; i < case_keys.count; ++i) {
      if (case_keys.i64_array[i] == selector_facts.range_lo) {
        row_index = i;
        break;
      }
    }
    for (uint16_t i = 0; i < op->result_count; ++i) {
      result_facts[i] = operand_facts[1 + row_index * op->result_count + i];
    }
    return iree_ok_status();
  }

  bool initialized_results = false;
  for (uint16_t i = 0; i < case_keys.count; ++i) {
    if (!loom_scf_lookup_key_matches_selector_facts(selector_facts,
                                                    case_keys.i64_array[i])) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_scf_lookup_meet_row_facts(
        context, module, op, op->result_count, i, operand_facts,
        initialized_results, result_facts));
    initialized_results = true;
  }
  if (loom_scf_lookup_default_row_may_match(selector_facts, case_keys)) {
    IREE_RETURN_IF_ERROR(loom_scf_lookup_meet_row_facts(
        context, module, op, op->result_count, case_keys.count, operand_facts,
        initialized_results, result_facts));
    initialized_results = true;
  }
  if (!initialized_results) {
    for (uint16_t i = 0; i < op->result_count; ++i) {
      result_facts[i] = loom_value_facts_unknown();
    }
  }
  return iree_ok_status();
}
