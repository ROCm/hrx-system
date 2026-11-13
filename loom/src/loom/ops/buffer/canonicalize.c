// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/rewrite/rewriter.h"

iree_status_t loom_buffer_compare_address_canonicalize(
    loom_op_t* op, loom_rewriter_t* rewriter) {
  // Exact value identity establishes the same byte origin. Sharing an
  // allocation or alias scope alone does not establish that origin.
  if (loom_rewriter_value_identity(rewriter,
                                   loom_buffer_compare_address_lhs(op)) !=
      loom_rewriter_value_identity(rewriter,
                                   loom_buffer_compare_address_rhs(op))) {
    const loom_value_id_t left_offset =
        loom_buffer_compare_address_lhs_offset(op);
    const loom_value_id_t right_offset =
        loom_buffer_compare_address_rhs_offset(op);
    int64_t constant = 0;
    if (loom_rewriter_value_identity(rewriter, left_offset) !=
            loom_rewriter_value_identity(rewriter, right_offset) ||
        (loom_value_facts_as_exact_i64(
             loom_rewriter_value_facts(rewriter, left_offset), &constant) &&
         constant == 0)) {
      return iree_ok_status();
    }
    // Equal displacements cancel without establishing whether the two origins
    // alias. Valid positions cannot overflow the target's address space.
    loom_builder_set_before(&rewriter->builder, op);
    loom_op_t* zero = NULL;
    IREE_RETURN_IF_ERROR(loom_index_constant_build(
        &rewriter->builder, loom_attr_i64(0),
        loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET), op->location, &zero));
    const loom_value_id_t origin = loom_index_constant_result(zero);
    const loom_value_id_t operands[] = {
        loom_buffer_compare_address_lhs(op), origin,
        loom_buffer_compare_address_rhs(op), origin};
    return loom_rewriter_set_operands(rewriter, op, operands);
  }
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_index_cmp_predicate_t predicate =
      loom_buffer_compare_address_predicate(op) ==
              LOOM_BUFFER_COMPARE_ADDRESS_PREDICATE_EQ
          ? LOOM_INDEX_CMP_PREDICATE_EQ
          : LOOM_INDEX_CMP_PREDICATE_NE;
  loom_op_t* comparison = NULL;
  IREE_RETURN_IF_ERROR(loom_index_cmp_build(
      &rewriter->builder, predicate, loom_buffer_compare_address_lhs_offset(op),
      loom_buffer_compare_address_rhs_offset(op), op->location, &comparison));
  const loom_value_id_t replacement = loom_index_cmp_result(comparison);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}
