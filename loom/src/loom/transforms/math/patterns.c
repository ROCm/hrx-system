// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/math/patterns.h"

#include "loom/ir/module.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"

static iree_status_t loom_math_legalize_mulf_scalar_loop(
    const loom_math_legalize_recipe_context_t* context, loom_op_t* op,
    loom_rewriter_t* rewriter) {
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_type_t vector_type = context->query.value_type;
  const loom_type_t scalar_type =
      loom_type_scalar(loom_type_element_type(vector_type));
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* begin = NULL;
  loom_op_t* end = NULL;
  loom_op_t* step = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(0), index_type, op->location, &begin));
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(loom_type_dim_static_size_at(vector_type, 0)),
      index_type, op->location, &end));
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(1), index_type, op->location, &step));

  const loom_value_id_t left = loom_vector_mulf_lhs(op);
  const loom_value_id_t right = loom_vector_mulf_rhs(op);
  loom_op_t* loop = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      builder, /*build_flags=*/0, loom_index_constant_result(begin),
      loom_index_constant_result(end), loom_index_constant_result(step), &left,
      1, /*result_types=*/NULL, /*tied_results=*/NULL, /*tied_result_count=*/0,
      /*pipeline_depth=*/LOOM_VALUE_ID_INVALID,
      /*unroll_factor=*/LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, op->location, &loop));
  loom_builder_ip_t saved =
      loom_builder_enter_region(builder, loop, loom_scf_for_body(loop));
  const loom_value_id_t lane =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 0);
  const loom_value_id_t aggregate =
      loom_region_entry_arg_id(loom_scf_for_body(loop), 1);
  const int64_t dynamic_index = INT64_MIN;
  loom_op_t* left_lane = NULL;
  loom_op_t* right_lane = NULL;
  loom_op_t* product = NULL;
  loom_op_t* inserted = NULL;
  loom_op_t* yield = NULL;
  // Each lane is read before its only update, so the carried aggregate still
  // contains that lane of the left input without keeping another vector live.
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(builder, aggregate, &lane, 1,
                                                 &dynamic_index, 1, scalar_type,
                                                 op->location, &left_lane));
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(builder, right, &lane, 1,
                                                 &dynamic_index, 1, scalar_type,
                                                 op->location, &right_lane));
  IREE_RETURN_IF_ERROR(
      loom_scalar_mulf_build(builder, loom_vector_mulf_fastmath(op),
                             loom_vector_extract_result(left_lane),
                             loom_vector_extract_result(right_lane),
                             scalar_type, op->location, &product));
  IREE_RETURN_IF_ERROR(loom_vector_insert_build(
      builder, loom_scalar_mulf_result(product), aggregate, &lane, 1,
      &dynamic_index, 1, vector_type, op->location, &inserted));
  const loom_value_id_t yielded = loom_vector_insert_result(inserted);
  IREE_RETURN_IF_ERROR(
      loom_scf_yield_build(builder, &yielded, 1, op->location, &yield));
  loom_builder_restore(builder, saved);
  const loom_value_id_t replacement = loom_scf_for_results(loop).values[0];
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement,
                                                  1);
}

iree_status_t loom_math_legalize_rewrite_recipe(
    const loom_math_legalize_recipe_context_t* context, loom_op_t* op,
    loom_rewriter_t* rewriter, bool* out_rewritten) {
  *out_rewritten = false;

  if (context->decision.recipe == LOOM_TARGET_MATH_RECIPE_MULF_SCALAR_LOOP) {
    *out_rewritten = true;
    return loom_math_legalize_mulf_scalar_loop(context, op, rewriter);
  }

  IREE_RETURN_IF_ERROR(loom_math_legalize_rewrite_elementwise_recipe(
      context, op, rewriter, out_rewritten));
  if (*out_rewritten) {
    return iree_ok_status();
  }

  return iree_ok_status();
}
