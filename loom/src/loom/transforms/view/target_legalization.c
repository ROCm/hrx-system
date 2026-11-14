// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/view/target_legalization.h"

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/view/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/view/atomic.h"

static iree_status_t loom_view_legalize_atomic(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  // The generic decomposition inherits the target's scalar arithmetic mode.
  // A target provider may use the reference implementation after establishing
  // that its execution environment preserves subnormals.
  if (iree_any_bit_set(op->instance_flags, LOOM_MEMORY_ACCESS_FLAG_NOFTZ)) {
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
    };
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_view_atomic_rewrite_cmpxchg(context->rewriter, op));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

static iree_status_t loom_view_legalize_atomic_private(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (iree_any_bit_set(op->instance_flags, LOOM_MEMORY_ACCESS_FLAG_NOFTZ)) {
    return iree_ok_status();
  }
  const loom_memory_access_t access =
      loom_memory_access_cast(context->module, op);
  loom_value_fact_view_reference_t reference = {0};
  if (!loom_value_facts_query_view_reference(
          &context->fact_table->context,
          loom_value_fact_table_lookup(context->fact_table,
                                       loom_memory_access_view(access)),
          &reference) ||
      reference.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_view_atomic_rewrite_private(context->rewriter, op));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static const loom_target_legalizer_rule_t kViewLegalizerRules[] = {
    {.root_kind = LOOM_OP_VIEW_ATOMIC_LOAD,
     .legalize = loom_view_legalize_atomic_private},
    {.root_kind = LOOM_OP_VIEW_ATOMIC_STORE,
     .legalize = loom_view_legalize_atomic_private},
    {.root_kind = LOOM_OP_VIEW_ATOMIC_REDUCE,
     .legalize = loom_view_legalize_atomic_private},
    {.root_kind = LOOM_OP_VIEW_ATOMIC_RMW,
     .legalize = loom_view_legalize_atomic_private},
    {.root_kind = LOOM_OP_VIEW_ATOMIC_CMPXCHG,
     .legalize = loom_view_legalize_atomic_private},
    {
        .root_kind = LOOM_OP_VIEW_ATOMIC_REDUCE,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD | LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_view_legalize_atomic,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
    },
    {
        .root_kind = LOOM_OP_VIEW_ATOMIC_RMW,
        .first_operand_element_types =
            LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD | LOOM_SCALAR_TYPE_SET_FLOAT,
        .legalize = loom_view_legalize_atomic,
        .flags = LOOM_TARGET_LEGALIZER_ENTRY_FLAG_REQUIRE_CONTRACT_REJECTION,
    },
};

static const loom_target_legalizer_provider_t kViewLegalizerProvider = {
    .name = IREE_SVL("view"),
    .scope = LOOM_LOW_DESCRIPTOR_SCOPE_SHARED,
    .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE,
    .rules = kViewLegalizerRules,
    .rule_count = IREE_ARRAYSIZE(kViewLegalizerRules),
};

const loom_target_legalizer_provider_t* loom_view_target_legalizer_provider(
    void) {
  return &kViewLegalizerProvider;
}
