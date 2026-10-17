// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/index/target_legalization.h"

#include "loom/ops/index/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/predicate_facts.h"

// Native extrema contracts take precedence over this comparison/selection
// decomposition. Targets without a native form still receive the exact signed
// index semantics through their ordinary comparison and selection contracts.
static iree_status_t loom_index_legalize_extrema(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  const bool is_minimum = loom_index_min_isa(op);
  const loom_value_id_t lhs = loom_op_operands(op)[0];
  const loom_value_id_t rhs = loom_op_operands(op)[1];
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_op_t* comparison = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      builder,
      is_minimum ? LOOM_INDEX_CMP_PREDICATE_SLT : LOOM_INDEX_CMP_PREDICATE_SGT,
      lhs, rhs, op->location, &comparison));
  loom_op_t* selection = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_select_build(
      builder, loom_index_cmp_result(comparison), lhs, rhs,
      loom_module_value_type(context->module, lhs), op->location, &selection));
  loom_value_id_t replacement = loom_scf_select_result(selection);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static iree_status_t loom_index_legalize_divrem(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t lhs = loom_index_divrem_lhs(op);
  const loom_value_id_t rhs = loom_index_divrem_rhs(op);
  loom_op_t* quotient = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_div_build(builder, lhs, rhs, op->location, &quotient));
  loom_op_t* product = NULL;
  IREE_RETURN_IF_ERROR(loom_index_mul_build(
      builder, loom_index_div_result(quotient), rhs, op->location, &product));
  loom_op_t* remainder = NULL;
  IREE_RETURN_IF_ERROR(loom_index_sub_build(
      builder, lhs, loom_index_mul_result(product),
      loom_module_value_type(context->module, loom_op_const_results(op)[1]),
      op->location, &remainder));
  loom_value_id_t replacements[2] = {loom_index_div_result(quotient),
                                     loom_index_sub_result(remainder)};
  if (context->fact_table) {
    loom_predicate_t predicates[2];
    const uint16_t predicate_count =
        loom_value_facts_export_integer_refinements(
            replacements[1],
            loom_value_fact_table_lookup(context->fact_table, replacements[1]),
            loom_value_fact_table_lookup(context->fact_table,
                                         loom_op_const_results(op)[1]),
            predicates, IREE_ARRAYSIZE(predicates));
    if (predicate_count != 0) {
      const loom_type_t result_type =
          loom_module_value_type(context->module, replacements[1]);
      loom_op_t* assumption = NULL;
      IREE_RETURN_IF_ERROR(loom_index_assume_build(
          builder, &replacements[1], 1, predicates, predicate_count,
          &result_type, 1, op->location, &assumption));
      replacements[1] = loom_index_assume_results(assumption).values[0];
    }
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacements, IREE_ARRAYSIZE(replacements), checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacements, IREE_ARRAYSIZE(replacements)));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

// Decomposes a rotation in the target-selected index domain. Both shift counts
// are masked independently so a zero rotation never materializes an undefined
// full-width shift. Target-native index contracts retain precedence over this
// reference fallback.
static iree_status_t loom_index_legalize_rotation(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };

  const uint32_t bit_width = loom_target_legalization_context_bundle(context)
                                 ->snapshot->index_bitwidth;
  if (bit_width == 0 || bit_width > 64 || (bit_width & (bit_width - 1)) != 0) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const loom_value_id_t value = loom_op_operands(op)[0];
  const loom_value_id_t count = loom_op_operands(op)[1];
  loom_value_id_t replacement = value;
  if (bit_width != 1) {
    loom_op_t* mask_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_index_constant_build(builder, loom_attr_i64(bit_width - 1),
                                  index_type, op->location, &mask_op));
    loom_op_t* amount_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_andi_build(
        builder, count, loom_index_constant_result(mask_op), op->location,
        &amount_op));

    loom_op_t* zero_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        builder, loom_attr_i64(0), index_type, op->location, &zero_op));
    loom_op_t* negative_amount_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_index_sub_build(builder, loom_index_constant_result(zero_op),
                             loom_index_andi_result(amount_op), index_type,
                             op->location, &negative_amount_op));
    loom_op_t* reverse_amount_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_andi_build(
        builder, loom_index_sub_result(negative_amount_op),
        loom_index_constant_result(mask_op), op->location, &reverse_amount_op));

    const loom_value_id_t amount = loom_index_andi_result(amount_op);
    const loom_value_id_t reverse_amount =
        loom_index_andi_result(reverse_amount_op);
    const bool rotate_left = loom_index_rotli_isa(op);
    loom_op_t* left_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_shli_build(
        builder, value, rotate_left ? amount : reverse_amount, op->location,
        &left_op));
    loom_op_t* right_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_shrui_build(
        builder, value, rotate_left ? reverse_amount : amount, op->location,
        &right_op));
    loom_op_t* combine_op = NULL;
    IREE_RETURN_IF_ERROR(loom_index_ori_build(
        builder, loom_index_shli_result(left_op),
        loom_index_shrui_result(right_op), op->location, &combine_op));
    replacement = loom_index_ori_result(combine_op);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  out_result->action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN;
  return iree_ok_status();
}

static const loom_target_legalizer_rule_t kIndexLegalizerRules[] = {
    {
        .root_kind = LOOM_OP_INDEX_MIN,
        .legalize = loom_index_legalize_extrema,
    },
    {
        .root_kind = LOOM_OP_INDEX_MAX,
        .legalize = loom_index_legalize_extrema,
    },
    {
        .root_kind = LOOM_OP_INDEX_DIVREM,
        .legalize = loom_index_legalize_divrem,
    },
    {
        .root_kind = LOOM_OP_INDEX_ROTLI,
        .legalize = loom_index_legalize_rotation,
    },
    {
        .root_kind = LOOM_OP_INDEX_ROTRI,
        .legalize = loom_index_legalize_rotation,
    },
};

static const loom_target_legalizer_provider_t kIndexLegalizerProvider = {
    .name = IREE_SVL("index"),
    .strategy = LOOM_TARGET_LEGALIZER_STRATEGY_REFERENCE,
    .rules = kIndexLegalizerRules,
    .rule_count = IREE_ARRAYSIZE(kIndexLegalizerRules),
};

const loom_target_legalizer_provider_t* loom_index_target_legalizer_provider(
    void) {
  return &kIndexLegalizerProvider;
}
