// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scf/cleanup_patterns.h"

#include "loom/ir/module.h"
#include "loom/ops/scf/ops.h"
#include "loom/rewrite/rewriter.h"

typedef struct loom_scf_prepared_operand_replacement_t {
  // Root-local application owned by the shared prepared batch.
  loom_rewrite_action_t action;
  // Live root operand selected as the replacement.
  uint8_t operand;
} loom_scf_prepared_operand_replacement_t;

static iree_status_t loom_scf_apply_prepared_operand_replacement(
    const loom_rewrite_action_t* base_action, loom_rewriter_t* rewriter) {
  loom_op_t* op = base_action->root;
  if (loom_rewriter_is_trivially_dead(rewriter, op)) {
    return loom_rewriter_erase(rewriter, op);
  }
  const loom_scf_prepared_operand_replacement_t* action =
      (const loom_scf_prepared_operand_replacement_t*)base_action;
  const loom_value_id_t replacement =
      loom_op_const_operands(op)[action->operand];
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static iree_status_t loom_scf_prepare_select_pattern(
    const loom_rewrite_pattern_t* pattern, const loom_module_t* module,
    const loom_value_fact_table_t* facts, const loom_op_t* op,
    iree_arena_allocator_t* arena, loom_rewrite_action_t** out_action) {
  (void)pattern;
  (void)module;
  *out_action = NULL;
  if (facts && loom_value_facts_is_exact(loom_value_fact_table_lookup(
                   facts, loom_scf_select_result(op)))) {
    return iree_ok_status();
  }

  uint8_t replacement_operand = 0;
  if (loom_scf_select_true_value(op) == loom_scf_select_false_value(op)) {
    replacement_operand = 1;
  } else if (facts) {
    bool condition = false;
    if (!loom_value_facts_as_exact_bool(
            loom_value_fact_table_lookup(facts, loom_scf_select_condition(op)),
            &condition)) {
      return iree_ok_status();
    }
    replacement_operand = condition ? 1 : 2;
  } else {
    return iree_ok_status();
  }

  loom_scf_prepared_operand_replacement_t* action = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*action), (void**)&action));
  action->action.apply = loom_scf_apply_prepared_operand_replacement;
  action->operand = replacement_operand;
  *out_action = &action->action;
  return iree_ok_status();
}

static const loom_rewrite_pattern_t kScfPreparedCanonicalizationPatterns[] = {
    {
        .root_kind = LOOM_OP_SCF_SELECT,
        .prepare = loom_scf_prepare_select_pattern,
    },
};

const loom_rewrite_pattern_provider_t
    loom_scf_prepared_canonicalization_pattern_provider = {
        .name = IREE_SVL("scf-prepared-canonicalization"),
        .patterns = kScfPreparedCanonicalizationPatterns,
        .pattern_count = IREE_ARRAYSIZE(kScfPreparedCanonicalizationPatterns),
};
