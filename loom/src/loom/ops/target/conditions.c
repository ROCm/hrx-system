// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/target/ops.h"
#include "loom/target/condition.h"
#include "loom/util/fact_table.h"

iree_status_t loom_target_subgroup_size_facts(
    loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts) {
  (void)module;
  (void)operand_facts;
  const loom_target_facts_t* target_facts = context->target_facts;
  const uint32_t subgroup_size =
      target_facts != NULL
          ? target_facts->storage.bundle.snapshot->subgroup_size
          : 0;
  result_facts[0] = subgroup_size != 0
                        ? loom_value_facts_exact_i64((int64_t)subgroup_size)
                        : loom_value_facts_make(1, UINT32_MAX, 1);
  loom_value_facts_mark_cluster_uniform(&result_facts[0]);

  return loom_value_fact_table_define_contextual_query_origin(
      context->table, loom_target_subgroup_size_result(op),
      (loom_value_fact_contextual_query_origin_t){
          .family_kind = LOOM_PARAMETERIZED_ATTR_TARGET_SUBGROUP_SIZE,
          .key = loom_attr_absent(),
      });
}

static iree_string_view_t loom_target_subgroup_size_condition_validate(
    loom_attribute_t condition) {
  const int64_t subgroup_size = loom_target_subgroup_size_attr_size(condition);
  if (subgroup_size > 0 && subgroup_size <= UINT32_MAX) {
    return iree_string_view_empty();
  }
  return IREE_SV(
      "target.subgroup.size requires a nonzero unsigned 32-bit size");
}

static loom_target_condition_outcome_t
loom_target_subgroup_size_condition_evaluate(const loom_target_facts_t* facts,
                                             loom_attribute_t condition) {
  const uint32_t actual_size = facts->storage.bundle.snapshot->subgroup_size;
  if (actual_size == 0) return LOOM_TARGET_CONDITION_UNKNOWN;
  const uint32_t required_size =
      (uint32_t)loom_target_subgroup_size_attr_size(condition);
  return actual_size == required_size ? LOOM_TARGET_CONDITION_MATCH
                                      : LOOM_TARGET_CONDITION_REJECT;
}

static bool loom_target_subgroup_size_condition_project_query_predicate(
    loom_attribute_t condition, loom_attribute_t query_key,
    loom_value_id_t query_value_id, loom_predicate_t* out_predicate) {
  if (!loom_attr_is_absent(query_key)) return false;
  *out_predicate = (loom_predicate_t){
      .kind = LOOM_PREDICATE_EQ,
      .arg_count = 2,
      .arg_tags = {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST,
                   LOOM_PRED_ARG_NONE},
      .args = {query_value_id, loom_target_subgroup_size_attr_size(condition),
               0},
  };
  return true;
}

const loom_target_condition_descriptor_t loom_target_subgroup_size_condition = {
    .validate = loom_target_subgroup_size_condition_validate,
    .evaluate = loom_target_subgroup_size_condition_evaluate,
    .project_query_predicate =
        loom_target_subgroup_size_condition_project_query_predicate,
};
