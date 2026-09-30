// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/symbolic_quotient.h"

#include <stdint.h>

#include "loom/analysis/symbolic_value.h"
#include "loom/ir/module.h"
#include "loom/ops/index/carrier.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"

typedef enum loom_symbolic_quotient_scale_kind_e {
  LOOM_SYMBOLIC_QUOTIENT_SCALE_DIVISOR = 0,
  LOOM_SYMBOLIC_QUOTIENT_SCALE_SHIFT = 1,
} loom_symbolic_quotient_scale_kind_t;

typedef struct loom_symbolic_quotient_def_t {
  loom_value_id_t dividend;
  loom_value_id_t scale_value;
  loom_symbolic_quotient_scale_kind_t scale_kind;
} loom_symbolic_quotient_def_t;

//===----------------------------------------------------------------------===//
// Quotient and scale recognition
//===----------------------------------------------------------------------===//

static const loom_op_t* loom_symbolic_quotient_value_defining_op(
    const loom_symbolic_expr_context_t* context, loom_value_id_t value_id) {
  if (!context->module || value_id >= context->module->values.count) {
    return NULL;
  }
  const loom_value_t* value = loom_module_value(context->module, value_id);
  return loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
}

static bool loom_symbolic_quotient_def(
    const loom_op_t* op, loom_symbolic_quotient_def_t* out_quotient) {
  switch (op->kind) {
    case LOOM_OP_INDEX_DIV:
    case LOOM_OP_SCALAR_DIVUI:
      out_quotient->scale_kind = LOOM_SYMBOLIC_QUOTIENT_SCALE_DIVISOR;
      break;
    case LOOM_OP_INDEX_SHRUI:
    case LOOM_OP_SCALAR_SHRUI:
      out_quotient->scale_kind = LOOM_SYMBOLIC_QUOTIENT_SCALE_SHIFT;
      break;
    default:
      return false;
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  out_quotient->dividend = operands[0];
  out_quotient->scale_value = operands[1];
  return true;
}

static bool loom_symbolic_quotient_left_shift_def(const loom_op_t* op,
                                                  loom_value_id_t* out_value,
                                                  loom_value_id_t* out_amount) {
  if (op->kind != LOOM_OP_INDEX_SHLI && op->kind != LOOM_OP_SCALAR_SHLI) {
    return false;
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  *out_value = operands[0];
  *out_amount = operands[1];
  return true;
}

static iree_status_t loom_symbolic_quotient_value_is_positive(
    loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    bool* out_is_positive) {
  loom_value_facts_t facts = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_context_lookup_facts(context, value_id, &facts));
  *out_is_positive = loom_value_facts_is_positive(facts);
  return iree_ok_status();
}

static iree_status_t loom_symbolic_quotient_shift_is_supported(
    loom_symbolic_expr_context_t* context, loom_value_id_t shifted_value,
    loom_value_id_t amount_value, bool* out_supported) {
  *out_supported = false;
  loom_value_facts_t amount_facts = {0};
  IREE_RETURN_IF_ERROR(loom_symbolic_expr_context_lookup_facts(
      context, amount_value, &amount_facts));
  int64_t amount = 0;
  if (!loom_value_facts_as_exact_i64(amount_facts, &amount) || amount < 0) {
    return iree_ok_status();
  }

  const loom_scalar_type_t scalar_type = loom_type_element_type(
      loom_module_value_type(context->module, shifted_value));
  int32_t bit_count = loom_scalar_type_bitwidth(scalar_type);
  if (scalar_type == LOOM_SCALAR_TYPE_INDEX) {
    bit_count = loom_index_target_carrier_bitwidth(
        context->fact_table ? &context->fact_table->context : NULL,
        scalar_type);
    if (bit_count == 0) {
      bit_count = 64;
    }
  }
  *out_supported = bit_count > 0 && amount < bit_count;
  return iree_ok_status();
}

static iree_status_t loom_symbolic_quotient_value_is_exact_i64(
    loom_symbolic_expr_context_t* context, loom_value_id_t value_id,
    int64_t* out_value, bool* out_is_exact) {
  loom_value_facts_t facts = {0};
  IREE_RETURN_IF_ERROR(
      loom_symbolic_expr_context_lookup_facts(context, value_id, &facts));
  *out_is_exact = loom_value_facts_as_exact_i64(facts, out_value);
  return iree_ok_status();
}

static iree_status_t loom_symbolic_quotient_scale_matches_multiplier(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_quotient_def_t* quotient,
    loom_value_id_t multiplier_value, bool* out_match) {
  if (quotient->scale_kind == LOOM_SYMBOLIC_QUOTIENT_SCALE_DIVISOR) {
    return loom_symbolic_values_match(context, quotient->scale_value,
                                      multiplier_value, out_match);
  }

  *out_match = false;
  int64_t shift_amount = 0;
  bool shift_is_exact = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_quotient_value_is_exact_i64(
      context, quotient->scale_value, &shift_amount, &shift_is_exact));
  if (!shift_is_exact || shift_amount < 0 || shift_amount > 62) {
    return iree_ok_status();
  }
  int64_t multiplier = 0;
  bool multiplier_is_exact = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_quotient_value_is_exact_i64(
      context, multiplier_value, &multiplier, &multiplier_is_exact));
  *out_match = multiplier_is_exact &&
               multiplier == (INT64_C(1) << (uint32_t)shift_amount);
  return iree_ok_status();
}

static iree_status_t loom_symbolic_quotient_scale_matches_shift(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_quotient_def_t* quotient,
    loom_value_id_t shift_amount_value, bool* out_match) {
  if (quotient->scale_kind == LOOM_SYMBOLIC_QUOTIENT_SCALE_SHIFT) {
    return loom_symbolic_values_match(context, quotient->scale_value,
                                      shift_amount_value, out_match);
  }

  *out_match = false;
  int64_t divisor = 0;
  bool divisor_is_exact = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_quotient_value_is_exact_i64(
      context, quotient->scale_value, &divisor, &divisor_is_exact));
  int64_t shift_amount = 0;
  bool shift_is_exact = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_quotient_value_is_exact_i64(
      context, shift_amount_value, &shift_amount, &shift_is_exact));
  *out_match = divisor_is_exact && shift_is_exact && shift_amount >= 0 &&
               shift_amount <= 62 &&
               divisor == (INT64_C(1) << (uint32_t)shift_amount);
  return iree_ok_status();
}

static iree_status_t loom_symbolic_quotient_matches_product(
    loom_symbolic_expr_context_t* context, loom_value_id_t product_value,
    const loom_symbolic_quotient_def_t* quotient, loom_value_id_t bound_value,
    bool* out_match) {
  *out_match = false;
  loom_value_id_t product_lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t product_rhs = LOOM_VALUE_ID_INVALID;
  if (!loom_symbolic_value_product_factors(context, product_value, &product_lhs,
                                           &product_rhs)) {
    return iree_ok_status();
  }

  bool lhs_matches_bound = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_values_match(
      context, product_lhs, bound_value, &lhs_matches_bound));
  if (lhs_matches_bound) {
    IREE_RETURN_IF_ERROR(loom_symbolic_quotient_scale_matches_multiplier(
        context, quotient, product_rhs, out_match));
    if (*out_match) {
      return iree_ok_status();
    }
  }

  bool rhs_matches_bound = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_values_match(
      context, product_rhs, bound_value, &rhs_matches_bound));
  return rhs_matches_bound ? loom_symbolic_quotient_scale_matches_multiplier(
                                 context, quotient, product_lhs, out_match)
                           : iree_ok_status();
}

static iree_status_t loom_symbolic_quotient_matches_scaled_bound(
    loom_symbolic_expr_context_t* context, loom_value_id_t scaled_bound_value,
    const loom_symbolic_quotient_def_t* quotient, loom_value_id_t bound_value,
    bool* out_match) {
  IREE_RETURN_IF_ERROR(loom_symbolic_quotient_matches_product(
      context, scaled_bound_value, quotient, bound_value, out_match));
  if (*out_match) {
    return iree_ok_status();
  }

  const loom_value_id_t scaled_bound_source =
      loom_symbolic_expr_assumption_source_value(context, scaled_bound_value);
  const loom_op_t* scaled_bound_op =
      loom_symbolic_quotient_value_defining_op(context, scaled_bound_source);
  loom_value_id_t shifted_bound = LOOM_VALUE_ID_INVALID;
  loom_value_id_t shift_amount = LOOM_VALUE_ID_INVALID;
  if (!scaled_bound_op || !loom_symbolic_quotient_left_shift_def(
                              scaled_bound_op, &shifted_bound, &shift_amount)) {
    return iree_ok_status();
  }

  bool bound_matches = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_values_match(context, shifted_bound,
                                                  bound_value, &bound_matches));
  return bound_matches ? loom_symbolic_quotient_scale_matches_shift(
                             context, quotient, shift_amount, out_match)
                       : iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Bound proofs
//===----------------------------------------------------------------------===//

static bool loom_symbolic_quotient_bound_relation(
    loom_symbolic_integer_relation_t relation,
    loom_symbolic_proof_result_t* out_result) {
  bool result = false;
  if (!loom_symbolic_integer_relation_implies(LOOM_SYMBOLIC_INTEGER_RELATION_LT,
                                              relation, &result)) {
    return false;
  }
  *out_result = result ? LOOM_SYMBOLIC_PROOF_TRUE : LOOM_SYMBOLIC_PROOF_FALSE;
  return true;
}

static iree_status_t loom_symbolic_quotient_inputs_are_supported(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_quotient_def_t* quotient, loom_value_id_t bound_value,
    bool* out_supported) {
  bool dividend_non_negative = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_value_is_non_negative(
      context, quotient->dividend, &dividend_non_negative));
  bool bound_non_negative = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_value_is_non_negative(
      context, bound_value, &bound_non_negative));
  if (!dividend_non_negative || !bound_non_negative) {
    *out_supported = false;
    return iree_ok_status();
  }

  if (quotient->scale_kind == LOOM_SYMBOLIC_QUOTIENT_SCALE_SHIFT) {
    return loom_symbolic_quotient_shift_is_supported(
        context, quotient->dividend, quotient->scale_value, out_supported);
  }
  return loom_symbolic_quotient_value_is_positive(
      context, quotient->scale_value, out_supported);
}

typedef struct loom_symbolic_quotient_condition_proof_t {
  loom_symbolic_expr_context_t* context;
  loom_symbolic_quotient_def_t quotient;
  loom_value_id_t bound_value;
  iree_status_t status;
  bool matched;
} loom_symbolic_quotient_condition_proof_t;

static iree_status_t loom_symbolic_quotient_matches_relation_operands(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_quotient_def_t* quotient, loom_value_id_t bound_value,
    loom_value_id_t left_value, loom_value_id_t right_value, bool* out_match,
    bool* out_dividend_is_left) {
  *out_match = false;
  *out_dividend_is_left = false;
  bool left_matches_dividend = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_values_match(
      context, left_value, quotient->dividend, &left_matches_dividend));
  bool right_matches_dividend = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_values_match(
      context, right_value, quotient->dividend, &right_matches_dividend));
  if (left_matches_dividend == right_matches_dividend) {
    return iree_ok_status();
  }

  const loom_value_id_t scaled_bound =
      left_matches_dividend ? right_value : left_value;
  IREE_RETURN_IF_ERROR(loom_symbolic_quotient_matches_scaled_bound(
      context, scaled_bound, quotient, bound_value, out_match));
  *out_dividend_is_left = left_matches_dividend;
  return iree_ok_status();
}

static bool loom_symbolic_quotient_visit_condition_relation(
    void* user_data, const loom_condition_integer_relation_t* relation) {
  loom_symbolic_quotient_condition_proof_t* proof =
      (loom_symbolic_quotient_condition_proof_t*)user_data;
  if (relation->left.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE ||
      relation->right.kind != LOOM_CONDITION_INTEGER_OPERAND_VALUE) {
    return true;
  }

  bool operands_match = false;
  bool dividend_is_left = false;
  proof->status = loom_symbolic_quotient_matches_relation_operands(
      proof->context, &proof->quotient, proof->bound_value,
      relation->left.value_id, relation->right.value_id, &operands_match,
      &dividend_is_left);
  if (!iree_status_is_ok(proof->status) || !operands_match) {
    return iree_status_is_ok(proof->status);
  }

  const loom_value_id_t relation_dividend =
      dividend_is_left ? relation->left.value_id : relation->right.value_id;
  const loom_value_id_t scaled_bound =
      dividend_is_left ? relation->right.value_id : relation->left.value_id;
  const loom_condition_integer_relation_t product_query = {
      .relation = LOOM_SYMBOLIC_INTEGER_RELATION_LT,
      .left = {.kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
               .value_id = relation_dividend},
      .right = {.kind = LOOM_CONDITION_INTEGER_OPERAND_VALUE,
                .value_id = scaled_bound},
  };
  bool product_relation = false;
  if (!loom_condition_fact_scope_proves_integer_relation(
          proof->context->condition_scope, proof->context->fact_table,
          &product_query, &product_relation) ||
      !product_relation) {
    return true;
  }

  // The comparison is against the materialized scaled bound, which may have
  // wrapped. Its nonnegative residue cannot exceed the mathematical product,
  // so the strict quotient bound remains valid. The converse needs no-wrap
  // proof.
  proof->matched = true;
  return false;
}

static iree_status_t loom_symbolic_quotient_condition_proves_bound(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_quotient_def_t* quotient, loom_value_id_t bound_value,
    bool* out_matched) {
  *out_matched = false;
  if (!context->condition_scope || !context->fact_table ||
      !loom_condition_fact_scope_has_integer_relations(
          context->condition_scope)) {
    return iree_ok_status();
  }

  loom_value_id_t anchors[2] = {quotient->dividend, LOOM_VALUE_ID_INVALID};
  iree_host_size_t anchor_count = 1;
  const loom_value_id_t source =
      loom_symbolic_expr_assumption_source_value(context, quotient->dividend);
  if (source != quotient->dividend) {
    anchors[anchor_count++] = source;
  }
  loom_symbolic_quotient_condition_proof_t proof = {
      .context = context,
      .quotient = *quotient,
      .bound_value = bound_value,
      .status = iree_ok_status(),
  };
  (void)loom_condition_fact_scope_for_each_value_anchored_while(
      context->condition_scope, context->fact_table, anchors, anchor_count,
      loom_symbolic_quotient_visit_condition_relation, &proof);
  if (iree_status_is_ok(proof.status)) {
    *out_matched = proof.matched;
  }
  return proof.status;
}

typedef struct loom_symbolic_quotient_predicate_proof_t {
  const loom_symbolic_quotient_def_t* quotient;
  loom_value_id_t bound_value;
  bool matched;
} loom_symbolic_quotient_predicate_proof_t;

static iree_status_t loom_symbolic_quotient_visit_identity_predicate(
    loom_symbolic_expr_context_t* context, loom_value_id_t identity_value,
    const loom_predicate_t* predicate, void* user_data, bool* out_continue) {
  (void)identity_value;
  *out_continue = true;
  loom_symbolic_quotient_predicate_proof_t* proof =
      (loom_symbolic_quotient_predicate_proof_t*)user_data;
  if (predicate->arg_count != 2 ||
      predicate->arg_tags[0] != LOOM_PRED_ARG_VALUE ||
      predicate->arg_tags[1] != LOOM_PRED_ARG_VALUE || predicate->args[0] < 0 ||
      predicate->args[1] < 0) {
    return iree_ok_status();
  }

  loom_symbolic_integer_relation_t predicate_relation =
      LOOM_SYMBOLIC_INTEGER_RELATION_EQ;
  if (!loom_symbolic_value_predicate_relation(predicate, &predicate_relation)) {
    return iree_ok_status();
  }

  bool operands_match = false;
  bool dividend_is_left = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_quotient_matches_relation_operands(
      context, proof->quotient, proof->bound_value,
      (loom_value_id_t)predicate->args[0], (loom_value_id_t)predicate->args[1],
      &operands_match, &dividend_is_left));
  if (!operands_match) {
    return iree_ok_status();
  }

  if (!dividend_is_left) {
    predicate_relation =
        loom_symbolic_integer_relation_swap(predicate_relation);
  }
  bool proves_strict_bound = false;
  if (loom_symbolic_integer_relation_implies(predicate_relation,
                                             LOOM_SYMBOLIC_INTEGER_RELATION_LT,
                                             &proves_strict_bound) &&
      proves_strict_bound) {
    proof->matched = true;
    *out_continue = false;
  }
  return iree_ok_status();
}

static iree_status_t loom_symbolic_quotient_predicate_proves_bound(
    loom_symbolic_expr_context_t* context,
    const loom_symbolic_quotient_def_t* quotient, loom_value_id_t bound_value,
    bool* out_matched) {
  loom_symbolic_quotient_predicate_proof_t proof = {
      .quotient = quotient,
      .bound_value = bound_value,
  };
  IREE_RETURN_IF_ERROR(loom_symbolic_value_for_each_identity_predicate(
      context, quotient->dividend,
      loom_symbolic_quotient_visit_identity_predicate, &proof));
  *out_matched = proof.matched;
  return iree_ok_status();
}

iree_status_t loom_symbolic_expr_quotient_bound_proves_relation(
    loom_symbolic_expr_context_t* context,
    loom_symbolic_integer_relation_t relation, loom_value_id_t quotient_value,
    loom_value_id_t bound_value, bool* out_matched,
    loom_symbolic_proof_result_t* out_result) {
  *out_matched = false;
  const loom_value_id_t quotient_source =
      loom_symbolic_expr_assumption_source_value(context, quotient_value);
  const loom_op_t* quotient_op =
      loom_symbolic_quotient_value_defining_op(context, quotient_source);
  loom_symbolic_quotient_def_t quotient = {0};
  if (!quotient_op || !loom_symbolic_quotient_def(quotient_op, &quotient)) {
    return iree_ok_status();
  }

  bool inputs_supported = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_quotient_inputs_are_supported(
      context, &quotient, bound_value, &inputs_supported));
  loom_symbolic_proof_result_t quotient_result = LOOM_SYMBOLIC_PROOF_UNKNOWN;
  if (!inputs_supported ||
      !loom_symbolic_quotient_bound_relation(relation, &quotient_result)) {
    return iree_ok_status();
  }

  loom_value_id_t launch_bound = LOOM_VALUE_ID_INVALID;
  if (loom_symbolic_expr_kernel_coordinate_launch_bound_value(
          context, quotient.dividend, &launch_bound)) {
    bool launch_bound_matches = false;
    IREE_RETURN_IF_ERROR(loom_symbolic_quotient_matches_scaled_bound(
        context, launch_bound, &quotient, bound_value, &launch_bound_matches));
    if (launch_bound_matches) {
      *out_matched = true;
      *out_result = quotient_result;
      return iree_ok_status();
    }
  }

  bool predicate_proves_bound = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_quotient_predicate_proves_bound(
      context, &quotient, bound_value, &predicate_proves_bound));
  if (predicate_proves_bound) {
    *out_matched = true;
    *out_result = quotient_result;
    return iree_ok_status();
  }

  bool condition_proves_bound = false;
  IREE_RETURN_IF_ERROR(loom_symbolic_quotient_condition_proves_bound(
      context, &quotient, bound_value, &condition_proves_bound));
  *out_matched = condition_proves_bound;
  if (condition_proves_bound) {
    *out_result = quotient_result;
  }
  return iree_ok_status();
}
