// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/index/cleanup_patterns.h"

#include "loom/analysis/condition_facts.h"
#include "loom/analysis/symbolic_expr.h"
#include "loom/analysis/symbolic_expr_proof.h"
#include "loom/ir/module.h"
#include "loom/ops/index/carrier.h"
#include "loom/ops/index/compare.h"
#include "loom/ops/index/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/cleanup/patterns.h"

static iree_status_t loom_index_replace_single_result_with_value(
    loom_rewriter_t* rewriter, loom_op_t* op, loom_value_id_t replacement) {
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static iree_status_t loom_index_replace_single_result_with_exact_i64(
    loom_rewriter_t* rewriter, loom_op_t* op, int64_t value) {
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t result = loom_op_const_results(op)[0];
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, result);

  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_build_constant(rewriter, loom_value_facts_exact_i64(value),
                                   result_type, op->location, &replacement));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_index_replace_single_result_with_value(rewriter, op, replacement);
}

typedef enum loom_index_prepared_canonicalization_kind_e {
  LOOM_INDEX_PREPARED_CANONICALIZATION_REPLACE_OPERAND = 0,
  LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_ADD = 1,
  LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_MUL = 2,
  LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_SHRUI = 3,
  LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_SHLI = 4,
  LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_ANDI = 5,
} loom_index_prepared_canonicalization_kind_t;

typedef struct loom_index_prepared_canonicalization_t {
  // Root-local application owned by the shared prepared batch.
  loom_rewrite_action_t action;
  // Replacement operation selected from coherent facts.
  loom_index_prepared_canonicalization_kind_t kind;
  // Live root operand used as the replacement or first binary operand.
  uint8_t first_operand;
  // Live root operand used as the second binary operand.
  uint8_t second_operand;
  // Integer immediate materialized as the second binary operand.
  int64_t immediate;
} loom_index_prepared_canonicalization_t;

static bool loom_index_preparation_query_exact_i64(
    const loom_value_fact_table_t* facts, loom_value_id_t value_id,
    int64_t* out_value) {
  const loom_value_facts_t value_facts =
      loom_value_fact_table_lookup(facts, value_id);
  if (!loom_value_facts_is_exact(value_facts) ||
      loom_value_facts_is_float(value_facts)) {
    return false;
  }
  *out_value = value_facts.range_lo;
  return true;
}

static bool loom_index_preparation_exact_i64(
    const loom_value_fact_table_t* facts, loom_value_id_t value_id,
    int64_t expected) {
  int64_t actual = 0;
  return loom_index_preparation_query_exact_i64(facts, value_id, &actual) &&
         actual == expected;
}

static bool loom_index_preparation_result_is_exact(
    const loom_value_fact_table_t* facts, const loom_op_t* op) {
  return loom_value_facts_is_exact(
      loom_value_fact_table_lookup(facts, loom_op_const_results(op)[0]));
}

static iree_status_t loom_index_apply_prepared_canonicalization(
    const loom_rewrite_action_t* base_action, loom_rewriter_t* rewriter) {
  loom_op_t* op = base_action->root;
  if (loom_rewriter_is_trivially_dead(rewriter, op)) {
    return loom_rewriter_erase(rewriter, op);
  }
  const loom_index_prepared_canonicalization_t* action =
      (const loom_index_prepared_canonicalization_t*)base_action;
  const loom_value_id_t* operands = loom_op_const_operands(op);
  if (action->kind == LOOM_INDEX_PREPARED_CANONICALIZATION_REPLACE_OPERAND) {
    const loom_value_id_t replacement = operands[action->first_operand];
    return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                    1);
  }

  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t result_type =
      loom_module_value_type(rewriter->module, loom_op_const_results(op)[0]);
  loom_op_t* replacement_op = NULL;
  switch (action->kind) {
    case LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_ADD: {
      IREE_RETURN_IF_ERROR(loom_index_add_build(
          &rewriter->builder, operands[action->first_operand],
          operands[action->second_operand], result_type, op->location,
          &replacement_op));
      break;
    }
    case LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_MUL: {
      IREE_RETURN_IF_ERROR(loom_index_mul_build(
          &rewriter->builder, operands[action->first_operand],
          operands[action->second_operand], op->location, &replacement_op));
      break;
    }
    case LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_SHRUI:
    case LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_SHLI:
    case LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_ANDI: {
      loom_op_t* immediate_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_constant_build(
          &rewriter->builder, loom_attr_i64(action->immediate), result_type,
          op->location, &immediate_op));
      const loom_value_id_t immediate_value =
          loom_index_constant_result(immediate_op);
      if (action->kind == LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_SHRUI) {
        IREE_RETURN_IF_ERROR(loom_index_shrui_build(
            &rewriter->builder, operands[action->first_operand],
            immediate_value, op->location, &replacement_op));
      } else if (action->kind ==
                 LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_SHLI) {
        IREE_RETURN_IF_ERROR(loom_index_shli_build(
            &rewriter->builder, operands[action->first_operand],
            immediate_value, op->location, &replacement_op));
      } else {
        IREE_RETURN_IF_ERROR(loom_index_andi_build(
            &rewriter->builder, operands[action->first_operand],
            immediate_value, op->location, &replacement_op));
      }
      break;
    }
    case LOOM_INDEX_PREPARED_CANONICALIZATION_REPLACE_OPERAND:
    default:
      IREE_ASSERT_UNREACHABLE("invalid prepared index canonicalization");
      IREE_BUILTIN_UNREACHABLE();
  }
  const loom_value_id_t replacement = loom_op_results(replacement_op)[0];
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

static bool loom_index_preparation_shift_amount_is_valid(
    const loom_value_fact_table_t* facts, int64_t amount) {
  const int32_t bitwidth = loom_index_target_carrier_bitwidth(
      &facts->context, LOOM_SCALAR_TYPE_INDEX);
  return amount >= 0 && bitwidth > 0 && amount < bitwidth;
}

static const loom_op_t* loom_index_preparation_defining_op(
    const loom_module_t* module, loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(module, value_id);
  return loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
}

static iree_status_t loom_index_prepare_canonicalization_pattern(
    const loom_rewrite_pattern_t* pattern, const loom_module_t* module,
    const loom_value_fact_table_t* facts, const loom_op_t* op,
    iree_arena_allocator_t* arena, loom_rewrite_action_t** out_action) {
  (void)pattern;
  *out_action = NULL;
  if (!facts || loom_index_preparation_result_is_exact(facts, op)) {
    return iree_ok_status();
  }

  loom_index_prepared_canonicalization_kind_t kind =
      LOOM_INDEX_PREPARED_CANONICALIZATION_REPLACE_OPERAND;
  uint8_t first_operand = 0;
  uint8_t second_operand = 0;
  int64_t immediate = 0;
  const loom_value_id_t* operands = loom_op_const_operands(op);
  switch (op->kind) {
    case LOOM_OP_INDEX_ADD: {
      if (loom_index_preparation_exact_i64(facts, operands[0], 0)) {
        first_operand = 1;
      } else if (loom_index_preparation_exact_i64(facts, operands[1], 0)) {
        first_operand = 0;
      } else {
        return iree_ok_status();
      }
      break;
    }
    case LOOM_OP_INDEX_MADD: {
      if (loom_index_preparation_exact_i64(facts, operands[0], 0) ||
          loom_index_preparation_exact_i64(facts, operands[1], 0)) {
        first_operand = 2;
      } else if (loom_index_preparation_exact_i64(facts, operands[2], 0)) {
        kind = LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_MUL;
        first_operand = 0;
        second_operand = 1;
      } else if (loom_index_preparation_exact_i64(facts, operands[0], 1)) {
        kind = LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_ADD;
        first_operand = 2;
        second_operand = 1;
      } else if (loom_index_preparation_exact_i64(facts, operands[1], 1)) {
        kind = LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_ADD;
        first_operand = 2;
        second_operand = 0;
      } else {
        return iree_ok_status();
      }
      break;
    }
    case LOOM_OP_INDEX_MUL: {
      int64_t factor = 0;
      if (loom_index_preparation_exact_i64(facts, operands[0], 1)) {
        first_operand = 1;
      } else if (loom_index_preparation_exact_i64(facts, operands[1], 1)) {
        first_operand = 0;
      } else if (loom_index_preparation_query_exact_i64(facts, operands[0],
                                                        &factor) &&
                 factor > 1 && iree_math_is_power_of_two_i64(factor)) {
        immediate = iree_math_floor_log2_u64((uint64_t)factor);
        if (!loom_index_preparation_shift_amount_is_valid(facts, immediate)) {
          return iree_ok_status();
        }
        kind = LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_SHLI;
        first_operand = 1;
      } else if (loom_index_preparation_query_exact_i64(facts, operands[1],
                                                        &factor) &&
                 factor > 1 && iree_math_is_power_of_two_i64(factor)) {
        immediate = iree_math_floor_log2_u64((uint64_t)factor);
        if (!loom_index_preparation_shift_amount_is_valid(facts, immediate)) {
          return iree_ok_status();
        }
        kind = LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_SHLI;
        first_operand = 0;
      } else {
        return iree_ok_status();
      }
      break;
    }
    case LOOM_OP_INDEX_DIV: {
      if (operands[0] == operands[1]) {
        return iree_ok_status();
      }
      if (loom_index_preparation_exact_i64(facts, operands[1], 1)) {
        first_operand = 0;
        break;
      }

      int64_t divisor = 0;
      const loom_value_facts_t lhs_facts =
          loom_value_fact_table_lookup(facts, operands[0]);
      const loom_type_t lhs_type = loom_module_value_type(module, operands[0]);
      const loom_op_t* lhs_def =
          loom_index_preparation_defining_op(module, operands[0]);
      if (!loom_index_preparation_query_exact_i64(facts, operands[1],
                                                  &divisor) ||
          divisor <= 0 || !iree_math_is_power_of_two_i64(divisor) ||
          (lhs_def &&
           (loom_index_mul_isa(lhs_def) || loom_index_madd_isa(lhs_def))) ||
          !loom_type_is_scalar(lhs_type) ||
          !loom_value_facts_is_non_negative(lhs_facts) ||
          !loom_index_value_facts_fit_unsigned_target_carrier(
              &facts->context, loom_type_element_type(lhs_type), lhs_facts)) {
        return iree_ok_status();
      }
      kind = LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_SHRUI;
      first_operand = 0;
      immediate = iree_math_floor_log2_u64((uint64_t)divisor);
      break;
    }
    case LOOM_OP_INDEX_REM: {
      const loom_value_facts_t lhs_facts =
          loom_value_fact_table_lookup(facts, operands[0]);
      const loom_value_facts_t rhs_facts =
          loom_value_fact_table_lookup(facts, operands[1]);
      if (loom_value_facts_is_non_negative(lhs_facts) &&
          loom_value_facts_is_non_negative(rhs_facts) &&
          loom_value_facts_remui_is_identity(lhs_facts, rhs_facts, 64)) {
        first_operand = 0;
        break;
      }

      int64_t divisor = 0;
      const loom_op_t* lhs_def =
          loom_index_preparation_defining_op(module, operands[0]);
      if (!loom_index_preparation_query_exact_i64(facts, operands[1],
                                                  &divisor) ||
          divisor <= 0 || !iree_math_is_power_of_two_i64(divisor) ||
          !loom_value_facts_is_non_negative(lhs_facts) ||
          loom_value_facts_divisible_by(lhs_facts, divisor) ||
          (lhs_def && loom_index_add_isa(lhs_def))) {
        return iree_ok_status();
      }
      kind = LOOM_INDEX_PREPARED_CANONICALIZATION_BUILD_ANDI;
      first_operand = 0;
      immediate = divisor - 1;
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("unsupported prepared index operation");
      IREE_BUILTIN_UNREACHABLE();
  }

  loom_index_prepared_canonicalization_t* action = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*action), (void**)&action));
  action->action.apply = loom_index_apply_prepared_canonicalization;
  action->kind = kind;
  action->first_operand = first_operand;
  action->second_operand = second_operand;
  action->immediate = immediate;
  *out_action = &action->action;
  return iree_ok_status();
}

static const loom_rewrite_pattern_t kIndexPreparedCanonicalizationPatterns[] = {
    {
        .root_kind = LOOM_OP_INDEX_ADD,
        .prepare = loom_index_prepare_canonicalization_pattern,
    },
    {
        .root_kind = LOOM_OP_INDEX_MADD,
        .prepare = loom_index_prepare_canonicalization_pattern,
    },
    {
        .root_kind = LOOM_OP_INDEX_MUL,
        .prepare = loom_index_prepare_canonicalization_pattern,
    },
    {
        .root_kind = LOOM_OP_INDEX_DIV,
        .prepare = loom_index_prepare_canonicalization_pattern,
    },
    {
        .root_kind = LOOM_OP_INDEX_REM,
        .prepare = loom_index_prepare_canonicalization_pattern,
    },
};

const loom_rewrite_pattern_provider_t
    loom_index_prepared_canonicalization_pattern_provider = {
        .name = IREE_SVL("index-prepared-canonicalization"),
        .patterns = kIndexPreparedCanonicalizationPatterns,
        .pattern_count = IREE_ARRAYSIZE(kIndexPreparedCanonicalizationPatterns),
};
static iree_status_t loom_index_symbolic_sub_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  *out_changed = false;
  loom_cleanup_pattern_context_t* cleanup_context =
      (loom_cleanup_pattern_context_t*)context;

  loom_symbolic_value_difference_t difference = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_simplify_value_difference(
      cleanup_context->symbolic_expression_context, loom_index_sub_lhs(op),
      loom_index_sub_rhs(op), &difference));
  switch (difference.kind) {
    case LOOM_SYMBOLIC_VALUE_DIFFERENCE_CONSTANT: {
      loom_value_facts_t constant_facts =
          loom_value_facts_exact_i64(difference.constant);
      const loom_type_t result_type = loom_module_value_type(
          rewriter->module, loom_op_const_results(op)[0]);
      if (loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_OFFSET) {
        const loom_fact_context_t* fact_context =
            rewriter->fact_table ? &rewriter->fact_table->context : NULL;
        constant_facts =
            loom_index_wrap_offset_facts(fact_context, constant_facts);
        if (!loom_value_facts_is_exact(constant_facts)) {
          return iree_ok_status();
        }
      }
      IREE_RETURN_IF_ERROR(loom_index_replace_single_result_with_exact_i64(
          rewriter, op, constant_facts.range_lo));
      *out_changed = true;
      return iree_ok_status();
    }
    case LOOM_SYMBOLIC_VALUE_DIFFERENCE_VALUE: {
      const loom_type_t result_type = loom_module_value_type(
          rewriter->module, loom_op_const_results(op)[0]);
      const loom_type_t replacement_type =
          loom_module_value_type(rewriter->module, difference.value_id);
      if (!loom_type_equal(result_type, replacement_type)) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_index_replace_single_result_with_value(
          rewriter, op, difference.value_id));
      *out_changed = true;
      return iree_ok_status();
    }
    case LOOM_SYMBOLIC_VALUE_DIFFERENCE_UNKNOWN:
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_index_symbolic_cmp_pattern(
    const loom_rewrite_pattern_t* pattern, void* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_changed) {
  (void)pattern;
  *out_changed = false;
  loom_cleanup_pattern_context_t* cleanup_context =
      (loom_cleanup_pattern_context_t*)context;

  loom_condition_integer_comparison_t comparison;
  if (!loom_condition_integer_comparison_describe(rewriter->module, op,
                                                  &comparison)) {
    return iree_ok_status();
  }
  const loom_type_t operand_type =
      loom_module_value_type(rewriter->module, comparison.lhs);
  if (!loom_type_is_scalar(operand_type)) {
    return iree_ok_status();
  }
  const loom_value_facts_t lhs_facts =
      loom_rewriter_value_facts(rewriter, comparison.lhs);
  const loom_value_facts_t rhs_facts =
      loom_rewriter_value_facts(rewriter, comparison.rhs);
  loom_condition_integer_relation_t relation = {
      .left = {.kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
               .value_id = comparison.lhs},
      .right = {.kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
                .value_id = comparison.rhs},
  };
  if (!loom_condition_integer_comparison_equivalent_relation(
          &comparison, rewriter->module,
          rewriter->fact_table ? &rewriter->fact_table->context : NULL,
          /*assumed_truth=*/true, &lhs_facts, &rhs_facts, &relation.relation)) {
    return iree_ok_status();
  }

  loom_symbolic_proof_result_t proof = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_prove_value_relation(
      cleanup_context->symbolic_expression_context, relation.relation,
      relation.left.value_id, relation.right.value_id, &proof));
  if (proof == LOOM_SYMBOLIC_PROOF_UNKNOWN) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_index_replace_single_result_with_exact_i64(
      rewriter, op, proof == LOOM_SYMBOLIC_PROOF_TRUE ? 1 : 0));
  *out_changed = true;
  return iree_ok_status();
}

static const loom_rewrite_pattern_t kIndexUniversalPostTypePatterns[] = {
    {
        .root_kind = LOOM_OP_INDEX_SUB,
        .match_and_rewrite = loom_index_symbolic_sub_pattern,
    },
    {
        .root_kind = LOOM_OP_INDEX_CMP,
        .match_and_rewrite = loom_index_symbolic_cmp_pattern,
    },
};

const loom_rewrite_pattern_provider_t
    loom_index_universal_post_type_pattern_provider = {
        .name = IREE_SVL("index"),
        .patterns = kIndexUniversalPostTypePatterns,
        .pattern_count = IREE_ARRAYSIZE(kIndexUniversalPostTypePatterns),
};
