// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/signed_unary_legalization.h"

#include "loom/ir/module.h"
#include "loom/ops/vector/ops.h"

iree_status_t loom_vector_legalize_signed_unary(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t input = loom_op_const_operands(op)[0];
  const loom_type_t type = loom_module_value_type(context->module, input);

  loom_value_id_t replacement = input;
  if (loom_type_element_type(type) != LOOM_SCALAR_TYPE_I1) {
    loom_op_t* zero_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_splati_build(builder, 0, type, op->location, &zero_op));
    const loom_value_id_t zero = loom_vector_splati_result(zero_op);

    if (!loom_vector_signi_isa(op)) {
      loom_op_t* negation_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_subi_build(builder, /*instance_flags=*/0,
                                                  zero, input, type,
                                                  op->location, &negation_op));
      replacement = loom_vector_subi_result(negation_op);
    }

    if (!loom_vector_negi_isa(op)) {
      loom_type_t predicate_type = type;
      predicate_type.header =
          loom_type_make_header(loom_type_kind(type), LOOM_SCALAR_TYPE_I1,
                                loom_type_rank(type), loom_type_flags(type));
      loom_op_t* negative_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_cmpi_build(
          builder, LOOM_VECTOR_CMPI_PREDICATE_SLT, input, zero, type,
          predicate_type, op->location, &negative_op));

      loom_value_id_t nonnegative = input;
      if (loom_vector_signi_isa(op)) {
        loom_op_t* one_op = NULL;
        IREE_RETURN_IF_ERROR(
            loom_vector_splati_build(builder, 1, type, op->location, &one_op));
        loom_op_t* minus_one_op = NULL;
        IREE_RETURN_IF_ERROR(loom_vector_splati_build(
            builder, -1, type, op->location, &minus_one_op));
        loom_op_t* nonzero_op = NULL;
        IREE_RETURN_IF_ERROR(loom_vector_cmpi_build(
            builder, LOOM_VECTOR_CMPI_PREDICATE_NE, input, zero, type,
            predicate_type, op->location, &nonzero_op));
        loom_op_t* positive_op = NULL;
        IREE_RETURN_IF_ERROR(loom_vector_select_build(
            builder, loom_vector_cmpi_result(nonzero_op),
            loom_vector_splati_result(one_op), zero, type, op->location,
            &positive_op));
        nonnegative = loom_vector_select_result(positive_op);
        replacement = loom_vector_splati_result(minus_one_op);
      }

      // Comparisons and selection preserve byte lanes and qword lanes even on
      // targets without packed arithmetic byte shifts or signed qword min/max.
      loom_op_t* select_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_select_build(
          builder, loom_vector_cmpi_result(negative_op), replacement,
          nonnegative, type, op->location, &select_op));
      replacement = loom_vector_select_result(select_op);
    }
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}
